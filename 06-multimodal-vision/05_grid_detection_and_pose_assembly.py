"""Turn dense output maps into objects: a grid detector's boxes, and people assembled from part maps.

Demonstrates how single-shot detection and bottom-up pose estimation read structure out of tensors:
    1. Render a detection dataset whose every box is recorded, and cluster anchor shapes by IoU.
    2. Encode each box onto its grid cell and best anchor, and decode it back.
    3. Build a one-scale YOLO detector and the loss terms that train it.
    4. Train the detector and follow each loss term.
    5. Decode predictions, suppress overlaps per class by hand, and check against torchvision.
    6. Score mean average precision on unseen images and draw detections.
    7. Prepare COCO annotations as grid targets and count what one scale and three scales can hold.
    8. Render part confidence maps and part affinity fields from keypoints.
    9. Pair keypoints into limbs by line integrals over the affinity fields, and by distance alone.

Module 06: Multimodal Vision - Grid Detection and Pose Assembly.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "outputs" / "grid_detection_and_pose"
SEED = 3407

IMG = 160
STRIDE = 16
GRID = IMG // STRIDE
CLASSES = ("box", "disk", "triangle")
ANCHORS = 3
TRAIN_IMAGES, TEST_IMAGES = 4000, 500
EPOCHS, BATCH, LEARNING_RATE = 30, 32, 1e-3
LAMBDA_COORD, LAMBDA_NOOBJ, IGNORE_IOU = 5.0, 0.5, 0.5
NMS_IOU, SCORE_FOR_METRIC, SCORE_FOR_DRAWING = 0.45, 0.001, 0.3

COCO_INPUT = 416
KEYPOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
             "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle")
SKELETON = ((16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13), (6, 7), (6, 8), (7, 9),
            (8, 10), (9, 11), (2, 3), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7))   # 1-based, as COCO
MAP_STRIDE = 4
PCM_SIGMA, PAF_WIDTH, PEAK_THRESHOLD = 1.5, 1.0, 0.3
POSE_SCENES = 200


# ---------------------------------------------------------------------------
# 1. Detection data and anchors
# ---------------------------------------------------------------------------

def box_iou(a, b):
    """IoU between every box in a (N, 4) and every box in b (M, 4), as (x1, y1, x2, y2)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda boxes: (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def render_detection(rng):
    """One image with 1-5 non-overlapping shapes, each box measured from the pixels it painted."""
    noise = cv2.GaussianBlur(rng.normal(0, 1, (IMG, IMG, 3)).astype(np.float32), (0, 0), 3)
    img = np.clip(110 + 35 * noise / noise.std(), 0, 255)
    boxes, labels = [], []
    for _ in range(rng.integers(1, 6)):
        for _attempt in range(20):
            size = int(rng.integers(12, 64))
            aspect = float(rng.uniform(0.6, 1.6))
            w, h = size, max(8, int(size * aspect))
            cx, cy = int(rng.integers(w // 2 + 1, IMG - w // 2 - 1)), int(rng.integers(h // 2 + 1, IMG - h // 2 - 1))
            label = int(rng.integers(len(CLASSES)))
            mask = np.zeros((IMG, IMG), np.uint8)
            if label == 0:
                cv2.rectangle(mask, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), 1, -1)
            elif label == 1:
                cv2.ellipse(mask, (cx, cy), (w // 2, h // 2), 0, 0, 360, 1, -1)
            else:
                cv2.fillPoly(mask, [np.array([(cx, cy - h // 2), (cx - w // 2, cy + h // 2),
                                              (cx + w // 2, cy + h // 2)], np.int32)], 1)
            ys, xs = np.nonzero(mask)
            box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
            if boxes and box_iou(box[None], np.array(boxes)).max() > 0.1:
                continue
            img[mask > 0] = rng.integers(30, 256, 3)
            boxes.append(box)
            labels.append(label)
            break
    return img.astype(np.uint8), np.array(boxes, np.float32).reshape(-1, 4), np.array(labels, np.int64)


def kmeans_anchors(wh, k, rng, iterations=100):
    """Cluster box shapes with 1 - IoU as the distance, as if every box were centred at one point.

    This is the anchor clustering used for YOLO: k-means in its alternation of
    assigning and updating, but not k-means in the strict sense, which uses squared
    Euclidean distance and the arithmetic mean. Euclidean distance on (w, h) would
    let large boxes dominate the clusters; IoU between two shapes aligned at a corner
    measures what an anchor is for, how much of a box it already covers. Each centre
    moves to the per-dimension median shape of its members, and the result is sorted
    by area.
    """
    centres = wh[rng.choice(len(wh), k, replace=False)].astype(np.float64)
    for _ in range(iterations):
        inter = np.minimum(wh[:, None, 0], centres[None, :, 0]) * np.minimum(wh[:, None, 1], centres[None, :, 1])
        iou = inter / (wh.prod(1)[:, None] + centres.prod(1)[None, :] - inter)
        assign = iou.argmax(1)
        moved = np.array([np.median(wh[assign == j], axis=0) if (assign == j).any() else centres[j]
                          for j in range(k)])
        if np.allclose(moved, centres):
            break
        centres = moved
    centres = centres[np.argsort(centres.prod(1))]
    inter = np.minimum(wh[:, None, 0], centres[None, :, 0]) * np.minimum(wh[:, None, 1], centres[None, :, 1])
    best = (inter / (wh.prod(1)[:, None] + centres.prod(1)[None, :] - inter)).max(1)
    return centres, float(best.mean())


def shape_iou(wh, anchors):
    """IoU of one (w, h) against each anchor, aligned at a common centre."""
    inter = np.minimum(wh[0], anchors[:, 0]) * np.minimum(wh[1], anchors[:, 1])
    return inter / (wh[0] * wh[1] + anchors[:, 0] * anchors[:, 1] - inter)


# ---------------------------------------------------------------------------
# 2. Encoding and decoding
# ---------------------------------------------------------------------------

def encode(boxes, labels, anchors, stride=STRIDE, grid=GRID):
    """Write each box into the (anchor, row, column) slot that is responsible for it.

    The responsible cell is the one containing the box centre, and the responsible
    anchor is the one whose shape overlaps the box best. The slot stores the centre
    as an offset inside the cell, (cx / stride - column, cy / stride - row), which
    the network reaches through a sigmoid, and the size as log(w / anchor_w),
    log(h / anchor_h), which it reaches through an exponential. A second box that
    lands in an occupied slot cannot be represented and is counted as a collision.
    """
    target = np.zeros((len(anchors), grid, grid, 5 + len(CLASSES)), np.float32)
    collisions = 0
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        col, row = min(int(cx // stride), grid - 1), min(int(cy // stride), grid - 1)
        anchor = int(np.argmax(shape_iou((w, h), anchors)))
        if target[anchor, row, col, 4] == 1:
            collisions += 1
            continue
        target[anchor, row, col, :5] = (cx / stride - col, cy / stride - row,
                                        np.log(w / anchors[anchor, 0]), np.log(h / anchors[anchor, 1]), 1)
        target[anchor, row, col, 5 + label] = 1
    return target, collisions


def decode_targets(target, anchors, stride=STRIDE):
    """Read boxes back out of a target tensor."""
    a, row, col = np.nonzero(target[..., 4])
    t = target[a, row, col]
    cx, cy = (t[:, 0] + col) * stride, (t[:, 1] + row) * stride
    w, h = anchors[a, 0] * np.exp(t[:, 2]), anchors[a, 1] * np.exp(t[:, 3])
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1), t[:, 5:].argmax(1)


# ---------------------------------------------------------------------------
# 3-6. The detector
# ---------------------------------------------------------------------------

def build_detector():
    """Five 3x3 conv-BN-LeakyReLU blocks with four 2x2 pools (160 -> 10), then a 1x1 prediction head."""
    import torch.nn as nn

    def block(cin, cout):
        return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.1)]

    layers = []
    for cin, cout in ((3, 16), (16, 32), (32, 64), (64, 128)):
        layers += block(cin, cout) + [nn.MaxPool2d(2)]
    layers += block(128, 128) + [nn.Conv2d(128, ANCHORS * (5 + len(CLASSES)), 1)]
    return nn.Sequential(*layers)


def reshape_head(raw):
    """(B, A * (5 + C), S, S) -> (B, A, S, S, 5 + C)."""
    b, _, s, _ = raw.shape
    return raw.view(b, ANCHORS, 5 + len(CLASSES), s, s).permute(0, 1, 3, 4, 2)


def decode_head(p, anchors_t, stride=STRIDE):
    """Boxes (x1, y1, x2, y2) in pixels, objectness and class probabilities for every slot."""
    import torch

    s = p.shape[2]
    gy, gx = torch.meshgrid(torch.arange(s, device=p.device), torch.arange(s, device=p.device), indexing="ij")
    cx = (torch.sigmoid(p[..., 0]) + gx) * stride
    cy = (torch.sigmoid(p[..., 1]) + gy) * stride
    w = anchors_t[:, 0].view(1, -1, 1, 1) * torch.exp(p[..., 2].clamp(-6, 6))
    h = anchors_t[:, 1].view(1, -1, 1, 1) * torch.exp(p[..., 3].clamp(-6, 6))
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
    return boxes, torch.sigmoid(p[..., 4]), torch.softmax(p[..., 5:], -1)


def yolo_loss(p, target, gt_boxes, anchors_t):
    """Coordinate, objectness and class terms, each summed over its slots and averaged over the batch.

    Coordinates are regressed only in slots that hold a box: the sigmoid of the
    first two outputs against the stored cell offset, the raw next two against the
    stored log ratio. Objectness is a binary cross-entropy, pushed to one in those
    slots and to zero in every other slot, except slots whose predicted box already
    overlaps some real box by more than IGNORE_IOU: those are not asked to say no to
    an object they have nearly found. Classes are a cross-entropy in the box slots.
    """
    import torch
    import torch.nn.functional as F
    from torchvision.ops import box_iou as tv_box_iou

    obj = target[..., 4] == 1
    with torch.no_grad():
        boxes, _, _ = decode_head(p, anchors_t)
        ignore = torch.zeros_like(obj)
        for i, gt in enumerate(gt_boxes):
            if len(gt):
                ignore[i] = tv_box_iou(boxes[i].reshape(-1, 4), gt).max(1).values.view(obj.shape[1:]) > IGNORE_IOU
    noobj = ~obj & ~ignore
    batch = p.shape[0]
    terms = {
        "xy": LAMBDA_COORD * F.mse_loss(torch.sigmoid(p[..., :2][obj]), target[..., :2][obj], reduction="sum"),
        "wh": LAMBDA_COORD * F.mse_loss(p[..., 2:4][obj], target[..., 2:4][obj], reduction="sum"),
        "obj": F.binary_cross_entropy_with_logits(p[..., 4][obj], target[..., 4][obj], reduction="sum"),
        "noobj": LAMBDA_NOOBJ * F.binary_cross_entropy_with_logits(p[..., 4][noobj], target[..., 4][noobj],
                                                                   reduction="sum"),
        "class": F.cross_entropy(p[..., 5:][obj], target[..., 5:][obj].argmax(-1), reduction="sum"),
    }
    return {name: value / batch for name, value in terms.items()}


def nms_by_hand(boxes, scores, classes, iou_threshold=NMS_IOU):
    """Greedy suppression within each class: keep the best, drop what overlaps it, repeat."""
    keep = []
    for cls in np.unique(classes):
        idx = np.flatnonzero(classes == cls)
        order = idx[np.argsort(-scores[idx], kind="stable")]
        while order.size:
            best = order[0]
            keep.append(best)
            if order.size == 1:
                break
            overlap = box_iou(boxes[best][None], boxes[order[1:]])[0]
            order = order[1:][overlap <= iou_threshold]
    return np.array(sorted(keep), np.int64)


def detections(model, images, anchors_t, score_threshold, device):
    """Per image: boxes, scores and classes after score filtering and per-class NMS."""
    import torch

    results, agreements = [], 0
    from torchvision.ops import batched_nms
    with torch.no_grad():
        for start in range(0, len(images), 100):
            batch = torch.from_numpy(images[start:start + 100]).permute(0, 3, 1, 2).float().div(255).to(device)
            boxes, objectness, class_prob = decode_head(reshape_head(model(batch)), anchors_t)
            score, cls = (objectness[..., None] * class_prob).max(-1)
            for i in range(len(batch)):
                keep = score[i] > score_threshold
                b, s, c = boxes[i][keep].cpu().numpy(), score[i][keep].cpu().numpy(), cls[i][keep].cpu().numpy()
                ours = nms_by_hand(b, s, c)
                theirs = np.sort(batched_nms(torch.from_numpy(b), torch.from_numpy(s),
                                             torch.from_numpy(c), NMS_IOU).numpy())
                agreements += np.array_equal(ours, theirs)
                results.append((b[ours], s[ours], c[ours]))
    return results, agreements


def mean_average_precision(results, truth, iou_threshold=0.5):
    """Per-class average precision with 101-point interpolation, and their mean.

    Detections are taken in descending score, and each is matched to the highest-IoU
    ground-truth box of its class that no earlier detection has claimed, as COCO's
    evaluation does. (The VOC devkit instead takes the highest-IoU box outright and
    counts a false positive if it is already claimed.)
    """
    per_class = {}
    for cls, name in enumerate(CLASSES):
        total = sum(int((labels == cls).sum()) for _, labels in truth)
        ranked = sorted(((s, i, b) for i, (boxes, scores, classes) in enumerate(results)
                         for b, s, c in zip(boxes, scores, classes) if c == cls), key=lambda r: -r[0])
        used = defaultdict(set)
        tp = np.zeros(len(ranked))
        for k, (_, i, b) in enumerate(ranked):
            gt_boxes, gt_labels = truth[i]
            candidates = np.flatnonzero(gt_labels == cls)
            if len(candidates):
                overlap = box_iou(b[None], gt_boxes[candidates])[0]
                overlap[[c in used[i] for c in candidates]] = -1.0   # claimed boxes are out of the running
                best = int(np.argmax(overlap))
                if overlap[best] >= iou_threshold:
                    used[i].add(candidates[best])
                    tp[k] = 1
        recall = np.cumsum(tp) / max(total, 1)
        precision = np.cumsum(tp) / np.arange(1, len(tp) + 1)
        per_class[name] = float(np.mean([precision[recall >= level].max() if (recall >= level).any() else 0.0
                                         for level in np.linspace(0, 1, 101)]))
    return per_class


# ---------------------------------------------------------------------------
# 7. COCO as grid targets
# ---------------------------------------------------------------------------

def coco_boxes(annotation_file):
    """Non-crowd boxes per image, letterboxed to COCO_INPUT, with category names."""
    data = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    images = {img["id"]: img for img in data["images"]}
    names = {cat["id"]: cat["name"] for cat in data["categories"]}
    per_image = defaultdict(list)
    for ann in data["annotations"]:
        if not ann["iscrowd"] and ann["bbox"][2] > 1 and ann["bbox"][3] > 1:
            per_image[ann["image_id"]].append((ann["bbox"], ann["category_id"]))
    return images, names, per_image, len(data["annotations"])


def letterbox_boxes(image, boxes):
    """Scale an image's boxes so its long side is COCO_INPUT and centre the short side."""
    scale = COCO_INPUT / max(image["width"], image["height"])
    pad_x, pad_y = (COCO_INPUT - image["width"] * scale) / 2, (COCO_INPUT - image["height"] * scale) / 2
    out = np.array([[x * scale + pad_x, y * scale + pad_y, (x + w) * scale + pad_x, (y + h) * scale + pad_y]
                    for (x, y, w, h), _ in boxes], np.float32)
    return out, scale, (pad_x, pad_y)


# ---------------------------------------------------------------------------
# 8-9. Part confidence maps, part affinity fields and assembly
# ---------------------------------------------------------------------------

def synthetic_person(rng, x, y, height):
    """17 COCO-ordered keypoints of a standing figure with random arm and leg angles."""
    body = {"nose": (0, -0.9), "left_eye": (0.03, -0.93), "right_eye": (-0.03, -0.93),
            "left_ear": (0.07, -0.9), "right_ear": (-0.07, -0.9), "left_shoulder": (0.15, -0.7),
            "right_shoulder": (-0.15, -0.7), "left_hip": (0.1, -0.2), "right_hip": (-0.1, -0.2)}
    points = {name: np.array(offset, np.float64) for name, offset in body.items()}
    for side, sign in (("left", 1), ("right", -1)):
        upper = rng.uniform(-0.3, 2.4) * sign
        lower = upper + rng.uniform(-1.2, 1.2)
        points[f"{side}_elbow"] = points[f"{side}_shoulder"] + 0.28 * np.array([np.sin(upper), np.cos(upper)])
        points[f"{side}_wrist"] = points[f"{side}_elbow"] + 0.25 * np.array([np.sin(lower), np.cos(lower)])
        thigh = rng.uniform(-0.1, 0.4) * sign
        points[f"{side}_knee"] = points[f"{side}_hip"] + 0.4 * np.array([np.sin(thigh), np.cos(thigh)])
        points[f"{side}_ankle"] = points[f"{side}_knee"] + 0.4 * np.array([np.sin(thigh * 0.3), 1.0])
    return np.array([np.array([x, y]) + height * points[name] for name in KEYPOINTS])


def render_maps(people, visible, shape):
    """Part confidence maps (17 + background) and part affinity fields (2 per limb) at map scale.

    A confidence map puts a Gaussian at every person's keypoint of that type and
    keeps the maximum where two overlap, so nearby people stay separate peaks. An
    affinity field paints, inside a thin band along each limb, the unit vector that
    points from the limb's first keypoint to its second, and averages where two
    people's bands overlap. The field is what records which keypoint belongs to
    which: a band carries a direction, and only the right pairing runs along it.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    pcm = np.zeros((len(KEYPOINTS) + 1, h, w))
    for joints, seen in zip(people, visible):
        for j, (px, py) in enumerate(joints):
            if seen[j]:
                pcm[j] = np.maximum(pcm[j], np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * PCM_SIGMA ** 2)))
    pcm[-1] = 1 - pcm[:-1].max(0)
    paf = np.zeros((2 * len(SKELETON), h, w))
    for limb, (a, b) in enumerate(SKELETON):
        count = np.zeros((h, w))
        for joints, seen in zip(people, visible):
            if not (seen[a - 1] and seen[b - 1]):
                continue
            start, end = joints[a - 1], joints[b - 1]
            length = np.linalg.norm(end - start)
            if length < 1e-6:
                continue
            u = (end - start) / length
            along = (xx - start[0]) * u[0] + (yy - start[1]) * u[1]
            across = np.abs((xx - start[0]) * u[1] - (yy - start[1]) * u[0])
            band = (along >= 0) & (along <= length) & (across <= PAF_WIDTH)
            paf[2 * limb][band] += u[0]
            paf[2 * limb + 1][band] += u[1]
            count[band] += 1
        paf[2 * limb:2 * limb + 2] /= np.maximum(count, 1)
    return pcm, paf


def find_peaks(pcm, merge_within=1.5):
    """Local maxima above PEAK_THRESHOLD in each keypoint channel, as (x, y) arrays.

    A keypoint that falls exactly between two pixels gives them the same value, and
    both pass the local-maximum test, so candidates are taken strongest first and a
    candidate within merge_within pixels of one already kept is dropped as part of
    the same peak.
    """
    peaks = []
    for channel in pcm[:-1]:
        heat = channel.astype(np.float32)
        local = cv2.dilate(heat, np.ones((3, 3), np.uint8))
        ys, xs = np.nonzero((heat == local) & (heat > PEAK_THRESHOLD))
        kept = []
        for index in np.argsort(-heat[ys, xs]):
            point = np.array([xs[index], ys[index]], np.float64)
            if all(np.linalg.norm(point - other) > merge_within for other in kept):
                kept.append(point)
        peaks.append(np.array(kept).reshape(-1, 2))
    return peaks


def paf_score(paf_x, paf_y, a, b, map_height, samples=10):
    """Mean of the field projected onto the candidate limb, sampled along the segment.

    The line integral of the affinity field along a candidate connection is large
    only when the field runs parallel to it all the way, which a true limb does and
    a connection between two different people's joints mostly does not. A candidate
    also needs most samples positive, and is penalised when it is longer than half
    the map height.
    """
    d = b - a
    length = np.linalg.norm(d)
    if length < 1e-6:
        return -np.inf
    u = d / length
    t = np.linspace(0, 1, samples)
    xs = np.clip(np.rint(a[0] + t * d[0]).astype(int), 0, paf_x.shape[1] - 1)
    ys = np.clip(np.rint(a[1] + t * d[1]).astype(int), 0, paf_x.shape[0] - 1)
    dots = paf_x[ys, xs] * u[0] + paf_y[ys, xs] * u[1]
    if (dots > 0.05).mean() < 0.8:
        return -np.inf
    return float(dots.mean() + min(0.0, 0.5 * map_height / length - 1))


def connect(peaks, paf, map_height, use_field):
    """Greedy bipartite matching per limb: best-scoring pairs first, each keypoint used once per limb."""
    connections = []
    for limb, (a, b) in enumerate(SKELETON):
        candidates = []
        for i, pa in enumerate(peaks[a - 1]):
            for k, pb in enumerate(peaks[b - 1]):
                score = paf_score(paf[2 * limb], paf[2 * limb + 1], pa, pb, map_height) if use_field \
                    else -float(np.linalg.norm(pb - pa))
                if np.isfinite(score):
                    candidates.append((score, i, k))
        used_a, used_b = set(), set()
        for score, i, k in sorted(candidates, reverse=True):
            if i not in used_a and k not in used_b:
                used_a.add(i)
                used_b.add(k)
                connections.append((limb, i, k))
    return connections


def assembly_scores(people, visible, peaks, connections, radius=2.0):
    """Limb precision and recall against the people the maps were drawn from.

    Every true keypoint claims the nearest detected peak of its type within radius
    map pixels, which gives each peak an owner. A connection is correct when both of
    its peaks have the same owner. Recall is taken over every true limb whose two
    keypoints are labelled visible, whether or not a peak was found for them, so it
    counts a missed peak and a missed pairing alike.
    """
    owner = [dict() for _ in KEYPOINTS]
    for person, (joints, seen) in enumerate(zip(people, visible)):
        for j, point in enumerate(joints):
            if seen[j] and len(peaks[j]):
                gaps = np.linalg.norm(peaks[j] - point, axis=1)
                nearest = int(np.argmin(gaps))
                if gaps[nearest] <= radius:
                    owner[j][nearest] = person
    visible_limbs = sum(1 for joints, seen in zip(people, visible) for a, b in SKELETON
                        if seen[a - 1] and seen[b - 1])
    correct = sum(1 for limb, i, k in connections
                  if i in owner[SKELETON[limb][0] - 1] and k in owner[SKELETON[limb][1] - 1]
                  and owner[SKELETON[limb][0] - 1][i] == owner[SKELETON[limb][1] - 1][k])
    return correct, len(connections), visible_limbs


def draw_assembly(canvas, peaks, connections, scale):
    """Draw every connection in a colour per limb."""
    for limb, i, k in connections:
        a, b = SKELETON[limb]
        colour = tuple(int(c) for c in cv2.applyColorMap(np.uint8([[limb * 13]]), cv2.COLORMAP_HSV)[0, 0])
        pa, pb = peaks[a - 1][i] * scale, peaks[b - 1][k] * scale
        cv2.line(canvas, tuple(np.rint(pa).astype(int)), tuple(np.rint(pb).astype(int)), colour, 2)
    return canvas


def maps_overlay(image, pcm, paf):
    """The image beside its strongest part confidence and its affinity field magnitude."""
    h, w = image.shape[:2]
    confidence = cv2.resize((pcm[:-1].max(0) * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    magnitude = np.hypot(paf[0::2], paf[1::2]).max(0)
    field = cv2.resize((np.clip(magnitude, 0, 1) * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    blend = lambda heat: cv2.addWeighted(image, 0.45, cv2.applyColorMap(heat, cv2.COLORMAP_JET), 0.55, 0)
    return np.hstack([image, blend(confidence), blend(field)])


def keypoint_extent(person):
    """Bounding box of a person's labelled keypoints, as (x1, y1, x2, y2)."""
    seen = person[person[:, 2] > 0]
    return np.array([seen[:, 0].min(), seen[:, 1].min(), seen[:, 0].max(), seen[:, 1].max()])


def evaluate_scenes(scenes):
    """Run both assembly rules over a list of (people, visible, map shape) and total the scores."""
    totals = {True: [0, 0, 0], False: [0, 0, 0]}
    for people, visible, shape in scenes:
        pcm, paf = render_maps(people, visible, shape)
        peaks = find_peaks(pcm)
        for use_field in (True, False):
            scores = assembly_scores(people, visible, peaks, connect(peaks, paf, shape[0], use_field))
            totals[use_field] = [t + s for t, s in zip(totals[use_field], scores)]
    return totals


def print_assembly(totals):
    """Precision and recall of the two assembly rules."""
    print(f"  {'rule':<28}{'connections':>12}{'correct':>9}{'precision':>11}{'recall':>9}")
    for use_field, name in ((True, "affinity field line integral"), (False, "shortest distance")):
        correct, made, visible_limbs = totals[use_field]
        print(f"  {name:<28}{made:>12}{correct:>9}{correct / max(made, 1):>11.1%}{correct / max(visible_limbs, 1):>9.1%}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--coco-root", help="COCO folder with annotations/ and, optionally, val2017/")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # 1. Dataset and anchors
    print("--- 1. A detection dataset and anchors fitted to it ---")
    train = [render_detection(rng) for _ in range(TRAIN_IMAGES)]
    test = [render_detection(rng) for _ in range(TEST_IMAGES)]
    all_boxes = np.concatenate([b for _, b, _ in train])
    wh = np.stack([all_boxes[:, 2] - all_boxes[:, 0], all_boxes[:, 3] - all_boxes[:, 1]], 1)
    print(f"  {TRAIN_IMAGES} training and {TEST_IMAGES} test images of {IMG}x{IMG}, {len(all_boxes)} training boxes, "
          f"widths {wh[:, 0].min():.0f}-{wh[:, 0].max():.0f} px, heights {wh[:, 1].min():.0f}-{wh[:, 1].max():.0f} px")
    print(f"  {'k':>3}  {'anchors (w, h)':<44}{'mean best IoU':>14}")
    for k in (1, 3, 5):
        centres, fit = kmeans_anchors(wh, k, np.random.default_rng(SEED))
        print(f"  {k:>3}  {', '.join(f'({w:.0f}, {h:.0f})' for w, h in centres):<44}{fit:>14.3f}")
    anchors, _ = kmeans_anchors(wh, ANCHORS, np.random.default_rng(SEED))
    print(f"  using k = {ANCHORS}; each anchor becomes one predictor in every cell")

    # 2. Encoding
    print("\n--- 2. Encoding boxes onto the grid and back ---")
    targets, collisions, worst = [], 0, 0.0
    for img, boxes, labels in train:
        target, lost = encode(boxes, labels, anchors)
        targets.append(target)
        collisions += lost
        decoded, classes = decode_targets(target, anchors)
        if len(decoded):
            overlap = box_iou(decoded, boxes)
            worst = max(worst, float(np.abs(boxes[overlap.argmax(1)] - decoded).max()))
    print(f"  grid {GRID}x{GRID} at stride {STRIDE}, {ANCHORS} anchors: {GRID * GRID * ANCHORS} slots per image, "
          f"each holding (tx, ty, tw, th, objectness, {len(CLASSES)} classes)")
    print(f"  boxes that found their slot taken: {collisions} of {len(all_boxes)}")
    print(f"  largest coordinate error after decoding what was encoded: {worst:.2e} px")
    x1, y1, x2, y2 = train[0][1][0]
    t, _ = encode(train[0][1][:1], train[0][2][:1], anchors)
    a, row, col = (int(v[0]) for v in np.nonzero(t[..., 4]))
    print(f"  example: box ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}) -> anchor {a}, row {row}, column {col}, "
          f"stored {np.round(t[a, row, col, :4], 3).tolist()}")

    # 3. Model and loss
    import torch
    torch.manual_seed(SEED)
    # cuDNN chooses a convolution algorithm per shape and some of them accumulate in a
    # non-deterministic order, which is enough to move a trained metric between runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n--- 3. A one-scale detector and its loss ---")
    model = build_detector().to(device)
    anchors_t = torch.tensor(anchors, dtype=torch.float32, device=device)
    with torch.no_grad():
        out = model(torch.zeros(1, 3, IMG, IMG, device=device))
    print(f"  output {tuple(out.shape)} -> reshaped to (batch, anchors, rows, columns, {5 + len(CLASSES)}); "
          f"{sum(p.numel() for p in model.parameters())} parameters")
    print(f"  loss = xy + wh + obj + noobj + class, where xy and wh already carry the weight {LAMBDA_COORD} "
          f"and noobj the weight {LAMBDA_NOOBJ}")
    print(f"  (the table below prints the weighted terms); slots whose prediction overlaps a real box by "
          f"more than {IGNORE_IOU} are left out of noobj")

    # 4. Training
    print("\n--- 4. Training, term by term ---")
    images_t = torch.from_numpy(np.stack([img for img, _, _ in train])).permute(0, 3, 1, 2).contiguous()
    targets_t = torch.from_numpy(np.stack(targets))
    gt_t = [torch.from_numpy(boxes) for _, boxes, _ in train]
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(SEED)
    print(f"  Adam {LEARNING_RATE}, batch {BATCH}, {EPOCHS} epochs on {device}")
    print(f"  {'epoch':>5}" + "".join(f"{name:>9}" for name in ("xy", "wh", "obj", "noobj", "class", "total")) + f"{'time':>8}")
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        sums = defaultdict(float)
        order = torch.randperm(len(train), generator=generator)
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            x = images_t[idx].to(device).float().div(255)
            terms = yolo_loss(reshape_head(model(x)), targets_t[idx].to(device), [gt_t[i].to(device) for i in idx],
                              anchors_t)
            loss = sum(terms.values())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            for name, value in terms.items():
                sums[name] += value.item() * len(idx)
        if epoch in (1, 2, 5, 10, 20, EPOCHS):
            row = [sums[name] / len(train) for name in ("xy", "wh", "obj", "noobj", "class")]
            print(f"  {epoch:>5}" + "".join(f"{v:>9.3f}" for v in row) + f"{sum(row):>9.3f}"
                  f"{time.perf_counter() - started:>7.0f}s")
    model.eval()

    # 5. Decoding and NMS
    print("\n--- 5. Decoding and per-class non-maximum suppression ---")
    test_images = np.stack([img for img, _, _ in test])
    truth = [(boxes, labels) for _, boxes, labels in test]
    results, agreements = detections(model, test_images, anchors_t, SCORE_FOR_METRIC, device)
    kept = sum(len(r[0]) for r in results)
    print(f"  score = objectness x class probability; boxes above {SCORE_FOR_METRIC} go into suppression "
          f"at IoU {NMS_IOU}")
    print(f"  hand-written NMS keeps the same boxes as torchvision.ops.batched_nms in "
          f"{agreements} of {len(test)} images; {kept} boxes kept in total")

    # 6. mAP and drawings
    print("\n--- 6. Mean average precision on unseen images ---")
    ap = mean_average_precision(results, truth)
    for name, value in ap.items():
        print(f"  AP@0.5 {name:<10}{value:.3f}")
    print(f"  mAP@0.5 {np.mean(list(ap.values())):.3f} over {sum(len(b) for b, _ in truth)} test boxes")
    drawn = []
    shown, _ = detections(model, test_images[:4], anchors_t, SCORE_FOR_DRAWING, device)
    for (img, gt_boxes, _), (boxes, scores, classes) in zip(test[:4], shown):
        canvas = cv2.resize(img, (IMG * 2, IMG * 2), interpolation=cv2.INTER_NEAREST)
        for box in gt_boxes:
            cv2.rectangle(canvas, tuple(int(v * 2) for v in box[:2]), tuple(int(v * 2) for v in box[2:]), (0, 255, 0), 1)
        for box, score, cls in zip(boxes, scores, classes):
            cv2.rectangle(canvas, tuple(int(v * 2) for v in box[:2]), tuple(int(v * 2) for v in box[2:]), (0, 0, 255), 2)
            cv2.putText(canvas, f"{CLASSES[cls]} {score:.2f}", (int(box[0] * 2), max(int(box[1] * 2) - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        drawn.append(canvas)
    cv2.imwrite(str(OUT_DIR / "detections.png"), np.hstack(drawn))
    print(f"  four unseen images drawn with true boxes in green and detections above {SCORE_FOR_DRAWING} in red")

    # 7. COCO targets
    print("\n--- 7. COCO boxes as grid targets ---")
    if args.coco_root:
        root = Path(args.coco_root)
        images, names, per_image, total = coco_boxes(root / "annotations" / "instances_val2017.json")
        letterboxed = {i: letterbox_boxes(images[i], per_image[i]) for i in per_image}
        coco_wh = np.concatenate([np.stack([b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1) for b, _, _ in letterboxed.values()])
        print(f"  val2017: {len(images)} images, {total} annotations, {len(coco_wh)} non-crowd boxes, "
              f"letterboxed to {COCO_INPUT}x{COCO_INPUT}")
        nine, fit9 = kmeans_anchors(coco_wh, 9, np.random.default_rng(SEED))
        three, fit3 = kmeans_anchors(coco_wh, 3, np.random.default_rng(SEED))
        print(f"  IoU-clustered anchors: k=3 mean best IoU {fit3:.3f}; k=9 mean best IoU {fit9:.3f}")
        print(f"  k=9 anchors by area: {', '.join(f'({w:.0f},{h:.0f})' for w, h in nine)}")
        one_scale = three_scales = 0
        roundtrip = 0.0
        for boxes, _, _ in letterboxed.values():
            labels = np.zeros(len(boxes), np.int64)
            target, lost = encode(boxes, labels, three, stride=32, grid=COCO_INPUT // 32)
            one_scale += lost
            decoded, _ = decode_targets(target, three, stride=32)
            if len(decoded):
                roundtrip = max(roundtrip, float(np.abs(boxes[box_iou(decoded, boxes).argmax(1)] - decoded).max()))
            groups = [[], [], []]
            for box in boxes:
                shape = (box[2] - box[0], box[3] - box[1])
                groups[int(np.argmax(shape_iou(shape, nine))) // 3].append(box)
            for level, stride in enumerate((8, 16, 32)):
                if groups[level]:
                    level_boxes = np.array(groups[level])
                    _, lost = encode(level_boxes, np.zeros(len(level_boxes), np.int64), nine[3 * level:3 * level + 3],
                                     stride=stride, grid=COCO_INPUT // stride)
                    three_scales += lost
        print(f"  one scale, 13x13 grid, 3 anchors: {one_scale} boxes find their slot taken "
              f"({one_scale / len(coco_wh):.1%}); round-trip error of the rest {roundtrip:.2e} px")
        print(f"  three scales, 52/26/13 grids, 3 anchors each: {three_scales} ({three_scales / len(coco_wh):.1%})")
        print("  Small objects crowd together: a stride-32 cell covers 32x32 input pixels, and every")
        print("  person in a crowd within it competes for the same few slots. A stride-8 level gives")
        print("  small anchors sixteen times as many cells.")
        first = next(i for i in images if len(per_image[i]) >= 4)
        boxes, scale, (pad_x, pad_y) = letterboxed[first]
        print(f"  image {images[first]['file_name']} ({images[first]['width']}x{images[first]['height']}), "
              f"first three targets as class cx cy w h, normalised:")
        for box, (_, category) in list(zip(boxes, per_image[first]))[:3]:
            cx, cy = (box[0] + box[2]) / 2 / COCO_INPUT, (box[1] + box[3]) / 2 / COCO_INPUT
            w, h = (box[2] - box[0]) / COCO_INPUT, (box[3] - box[1]) / COCO_INPUT
            print(f"    {names[category]:<12} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}")
        path = root / "val2017" / images[first]["file_name"]
        if path.exists():
            picture = cv2.imread(str(path))
            for (x, y, w, h), category in per_image[first]:
                cv2.rectangle(picture, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
                cv2.putText(picture, names[category], (int(x), max(int(y) - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 1)
            cv2.imwrite(str(OUT_DIR / "coco_boxes.png"), picture)
    else:
        print("  pass --coco-root to encode the COCO val2017 boxes and compare one scale with three")

    # 8. Maps
    print("\n--- 8. Part confidence maps and part affinity fields ---")
    scene_rng = np.random.default_rng(SEED + 1)
    scenes = []
    for _ in range(POSE_SCENES):
        height = scene_rng.uniform(22, 32)
        first_x = scene_rng.uniform(20, 30)
        people = [synthetic_person(scene_rng, first_x, 38, height),
                  synthetic_person(scene_rng, first_x + height * scene_rng.uniform(0.2, 0.55), 38, height)]
        scenes.append((people, [np.ones(len(KEYPOINTS), bool)] * 2, (64, 64)))
    people, visible, shape = scenes[0]
    pcm, paf = render_maps(people, visible, shape)
    print(f"  {len(KEYPOINTS)} keypoint channels + 1 background, {len(SKELETON)} limbs x 2 = {paf.shape[0]} field "
          f"channels, at 1/{MAP_STRIDE} of the image")
    peaks = find_peaks(pcm)
    print(f"  first scene: two people, peaks found per keypoint type {[len(p) for p in peaks]}")
    limb = SKELETON.index((6, 8))
    field = np.hypot(paf[2 * limb], paf[2 * limb + 1])
    painted = shortened = 0
    for scene in scenes[:20]:
        magnitude = np.hypot(*render_maps(*scene)[1].reshape(len(SKELETON), 2, *scene[2]).transpose(1, 0, 2, 3))
        painted += int((magnitude > 0).sum())
        shortened += int((magnitude > 0).sum() - (magnitude > 0.99).sum())
    print(f"  first scene, left upper arm: {int((field > 0).sum())} pixels carry a direction, "
          f"magnitude {field[field > 0].min():.2f}-{field.max():.2f}")
    print(f"  over 20 scenes, {shortened / painted:.2%} of the painted pixels have a magnitude below 1, "
          f"which is where two people's bands cross and two directions are averaged")
    canvas = np.zeros((shape[0] * MAP_STRIDE, shape[1] * MAP_STRIDE, 3), np.uint8)
    for joints in people:
        for a, b in SKELETON:
            cv2.line(canvas, tuple(np.rint(joints[a - 1] * MAP_STRIDE).astype(int)),
                     tuple(np.rint(joints[b - 1] * MAP_STRIDE).astype(int)), (200, 200, 200), 3)
    cv2.imwrite(str(OUT_DIR / "pose_maps_synthetic.png"), maps_overlay(canvas, pcm, paf))

    # 9. Assembly
    print("\n--- 9. Pairing keypoints into limbs from the maps ---")
    print("  Each limb type is matched on its own; grouping the limbs into whole skeletons, one")
    print("  per person, is the step after this and is not done here. Recall counts every limb")
    print("  whose two keypoints are labelled visible, so it includes peaks that were missed.")
    print(f"  {POSE_SCENES} synthetic scenes, two people each, the second 0.2-0.55 body heights to the right:")
    print_assembly(evaluate_scenes(scenes))
    peaks = find_peaks(pcm)
    by_field = draw_assembly(canvas.copy(), peaks, connect(peaks, paf, shape[0], True), MAP_STRIDE)
    by_distance = draw_assembly(canvas.copy(), peaks, connect(peaks, paf, shape[0], False), MAP_STRIDE)
    cv2.imwrite(str(OUT_DIR / "pose_assembly.png"), np.hstack([by_field, by_distance]))
    print("  Distance alone joins each wrist to whichever elbow is closest, which is often the other")
    print("  person's once two people stand within an arm's length; it never rejects a pairing, so it")
    print("  reaches a higher recall while one connection in six joins two different people.")
    print("  The field is a direction painted along each true limb, so a pairing scores well only")
    print("  when the limb is actually there, and a candidate with too few positive samples is")
    print("  dropped rather than guessed at.")

    if args.coco_root:
        data = json.loads((Path(args.coco_root) / "annotations" / "person_keypoints_val2017.json").read_text(encoding="utf-8"))
        images = {img["id"]: img for img in data["images"]}
        per_image = defaultdict(list)
        for ann in data["annotations"]:
            if ann["num_keypoints"] >= 8 and not ann["iscrowd"]:
                per_image[ann["image_id"]].append(np.array(ann["keypoints"], np.float64).reshape(-1, 3))
        separated, overlapping = [], []
        for image_id, people in sorted(per_image.items()):
            if len(people) < 2 or min(keypoint_extent(p)[3] - keypoint_extent(p)[1] for p in people) < 80:
                continue
            extents = np.stack([keypoint_extent(p) for p in people])
            pairs = box_iou(extents, extents)
            np.fill_diagonal(pairs, 0)
            (overlapping if pairs.max() >= 0.2 else separated).append(image_id)

        def scenes_for(chosen):
            built = []
            for image_id in chosen:
                img = images[image_id]
                built.append(([p[:, :2] / MAP_STRIDE for p in per_image[image_id]],
                              [p[:, 2] > 0 for p in per_image[image_id]],
                              (int(np.ceil(img["height"] / MAP_STRIDE)), int(np.ceil(img["width"] / MAP_STRIDE)))))
            return built

        print(f"\n  COCO val2017, maps drawn from the labels, people at least 80 px tall and carrying 8 or")
        print(f"  more labelled keypoints. {len(separated[:150])} images where no two people's keypoint boxes")
        print(f"  overlap by more than 0.2 IoU:")
        print_assembly(evaluate_scenes(scenes_for(separated[:150])))
        print(f"  {len(overlapping[:150])} images where two of them do:")
        print_assembly(evaluate_scenes(scenes_for(overlapping[:150])))
        print("  Where people stand apart, the nearest candidate is usually the right one and distance")
        print("  is nearly as good. The field earns its cost exactly where the two rules disagree.")
        coco_scenes = scenes_for(overlapping[:1])
        example = overlapping[0]
        path = Path(args.coco_root) / "val2017" / images[example]["file_name"]
        if path.exists():
            people, visible, shape = coco_scenes[0]
            pcm, paf = render_maps(people, visible, shape)
            picture = cv2.imread(str(path))
            peaks = find_peaks(pcm)
            overlay = maps_overlay(picture, pcm, paf)
            assembled = draw_assembly(picture.copy(), peaks, connect(peaks, paf, shape[0], True), MAP_STRIDE)
            cv2.imwrite(str(OUT_DIR / "pose_maps_coco.png"), np.hstack([overlay, assembled]))
    else:
        print("  pass --coco-root to repeat the assembly on COCO val2017 keypoint labels")
    print(f"\n  images written to {OUT_DIR}")


if __name__ == "__main__":
    main()
