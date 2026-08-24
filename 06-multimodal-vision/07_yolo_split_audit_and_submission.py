"""Audit a detection split before training it, then price two changes that touch no boxes at all.

Demonstrates where a detection number comes from, and how little of it is the model:
    1. Synthesise a defect dataset whose every instance is recorded as it is drawn.
    2. Write the labels as VOC XML, then convert them to the YOLO text layout.
    3. Split the images the way a downloaded dataset usually arrives.
    4. Audit the split by class before a single epoch runs.
    5. Train a small detector and read its metric on the tiny split and on the full one.
    6. Rebuild the submission twice without moving a box, and score all three.

Module 06: Multimodal Vision - Detection Split Audit.
"""

import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent / "outputs" / "defect_detection"
CLASSES = ("scratch", "patch", "hole", "crack")
IMAGE_SIZE = 128
TOTAL_IMAGES = 160
SEED = 3407

# The split shape a downloaded dataset often arrives in: almost everything in
# train, a validation set small enough to fit on one screen, and a test set that
# nobody looks at until the end.
AS_ARRIVED = {"train": 128, "val": 2, "test": 30}

EPOCHS = 60
BATCH = 16
IOU_THRESHOLD = 0.5

# Average precision is defined over the full ranked list of detections, so the
# scoring pass has to ask for boxes the detector is barely confident about. The
# prediction helper defaults to a much higher bar, which quietly truncates that
# list before the metric ever sees it.
METRIC_CONFIDENCE = 0.001
VIEWING_CONFIDENCE = 0.25


def synthesise(rng):
    """Draw one plate of textured metal with defects on it, returning the boxes drawn.

    The boxes are not detected here, they are recorded as they are painted. That is
    the whole point of synthesising the data: the ground truth cannot disagree with
    the image, so any disagreement later belongs to the model or to the split.
    """
    noise = np.array(rng.choices(range(96, 152), k=IMAGE_SIZE * IMAGE_SIZE), dtype=np.uint8)
    img = Image.fromarray(noise.reshape(IMAGE_SIZE, IMAGE_SIZE), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(1.1)).convert("L")
    draw = ImageDraw.Draw(img)
    boxes = []

    for _ in range(rng.randint(1, 3)):
        label = rng.choice(CLASSES)
        x = rng.randint(8, IMAGE_SIZE - 40)
        y = rng.randint(8, IMAGE_SIZE - 40)
        if label == "scratch":
            length = rng.randint(18, 30)
            draw.line([x, y, x + length, y + rng.randint(-4, 4)], fill=245, width=2)
            box = (x - 2, y - 5, x + length + 2, y + 6)
        elif label == "patch":
            side = rng.randint(14, 24)
            draw.ellipse([x, y, x + side, y + side], fill=rng.randint(180, 210))
            box = (x, y, x + side, y + side)
        elif label == "hole":
            side = rng.randint(6, 11)
            draw.ellipse([x, y, x + side, y + side], fill=25)
            box = (x - 1, y - 1, x + side + 1, y + side + 1)
        else:
            points = [(x, y)]
            for step in range(4):
                points.append((points[-1][0] + rng.randint(4, 9), points[-1][1] + rng.randint(-7, 7)))
            draw.line(points, fill=40, width=2)
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            box = (min(xs) - 2, min(ys) - 3, max(xs) + 2, max(ys) + 3)
        clipped = (
            max(0, box[0]), max(0, box[1]),
            min(IMAGE_SIZE - 1, box[2]), min(IMAGE_SIZE - 1, box[3]),
        )
        if clipped[2] - clipped[0] > 3 and clipped[3] - clipped[1] > 3:
            boxes.append((label, *clipped))
    return img, boxes


def write_voc(path, filename, boxes):
    """Write one VOC annotation file, the layout hand-labelling tools produce."""
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(IMAGE_SIZE)
    ET.SubElement(size, "height").text = str(IMAGE_SIZE)
    ET.SubElement(size, "depth").text = "1"
    for label, x1, y1, x2, y2 in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label
        box = ET.SubElement(obj, "bndbox")
        for tag, value in (("xmin", x1), ("ymin", y1), ("xmax", x2), ("ymax", y2)):
            ET.SubElement(box, tag).text = str(int(value))
    ET.ElementTree(root).write(path, encoding="utf-8")


def voc_to_yolo(xml_path, txt_path):
    """Convert one VOC file to YOLO lines: class index, centre x, centre y, width, height.

    VOC stores absolute corners and YOLO stores a normalised centre and extent, so
    the conversion divides by the image size. Getting this wrong produces boxes that
    are valid numbers in the wrong places, which trains without complaint.
    """
    root = ET.parse(xml_path).getroot()
    width = float(root.find("size/width").text)
    height = float(root.find("size/height").text)
    lines = []
    for obj in root.findall("object"):
        label = obj.find("name").text.strip()
        if label not in CLASSES:
            continue
        box = obj.find("bndbox")
        x1, y1, x2, y2 = (float(box.find(tag).text) for tag in ("xmin", "ymin", "xmax", "ymax"))
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        bw, bh = (x2 - x1) / width, (y2 - y1) / height
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, bw, bh)):
            continue
        lines.append(f"{CLASSES.index(label)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return len(lines)


def build_dataset(seed=SEED):
    """Generate every image once, write both label formats, and return the truth table."""
    if ROOT.exists():
        shutil.rmtree(ROOT)
    rng = random.Random(seed)
    raw = ROOT / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "annotations").mkdir(parents=True)
    (raw / "labels").mkdir(parents=True)

    truth = {}
    for index in range(TOTAL_IMAGES):
        name = f"{index:04d}"
        image, boxes = synthesise(rng)
        image.save(raw / "images" / f"{name}.png")
        write_voc(raw / "annotations" / f"{name}.xml", f"{name}.png", boxes)
        voc_to_yolo(raw / "annotations" / f"{name}.xml", raw / "labels" / f"{name}.txt")
        truth[name] = boxes
    return truth


def split_names(truth, shape, seed=SEED):
    """Deal image names into splits in a fixed order, without looking at their contents."""
    names = sorted(truth)
    random.Random(seed).shuffle(names)
    out, cursor = {}, 0
    for split, count in shape.items():
        out[split] = names[cursor : cursor + count]
        cursor += count
    return out


def materialise(splits, layout_name):
    """Copy images and YOLO labels into the folder layout the trainer expects."""
    base = ROOT / layout_name
    raw = ROOT / "raw"
    for split, names in splits.items():
        (base / split / "images").mkdir(parents=True, exist_ok=True)
        (base / split / "labels").mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy(raw / "images" / f"{name}.png", base / split / "images" / f"{name}.png")
            shutil.copy(raw / "labels" / f"{name}.txt", base / split / "labels" / f"{name}.txt")
    yaml_path = base / "dataset.yaml"
    body = [f"path: {base.as_posix()}"]
    for split in splits:
        body.append(f"{split}: {split}/images")
    body.append("names:")
    body.extend(f"  {index}: {name}" for index, name in enumerate(CLASSES))
    yaml_path.write_text("\n".join(body), encoding="utf-8")
    return yaml_path


def audit(splits, truth):
    """Count instances per class per split, which is what a metric is averaged over."""
    table = {}
    for split, names in splits.items():
        counts = Counter(label for name in names for label, *_ in truth[name])
        table[split] = {
            "images": len(names),
            "instances": sum(counts.values()),
            "per_class": {cls: counts.get(cls, 0) for cls in CLASSES},
            "missing": [cls for cls in CLASSES if counts.get(cls, 0) == 0],
        }
    return table


def iou(box_a, box_b):
    """Return intersection over union for two (x1, y1, x2, y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def average_precision(predictions, truth, names):
    """Compute mean average precision at one IoU threshold, over the given images.

    The predictions are ranked by confidence and walked from the top down, so the
    score depends on the order of the list as much as on the boxes in it. That is
    the property the last step of this script exploits.
    """
    per_class = {}
    for index, cls in enumerate(CLASSES):
        gt = defaultdict(list)
        total = 0
        for name in names:
            for label, x1, y1, x2, y2 in truth[name]:
                if label == cls:
                    gt[name].append([x1, y1, x2, y2, False])
                    total += 1
        if total == 0:
            continue
        ranked = sorted(
            [p for p in predictions if p["class"] == index and p["image"] in names],
            key=lambda p: -p["confidence"],
        )
        tp = np.zeros(len(ranked))
        fp = np.zeros(len(ranked))
        for position, prediction in enumerate(ranked):
            best, best_iou = None, 0.0
            for candidate in gt[prediction["image"]]:
                score = iou(prediction["box"], candidate[:4])
                if score > best_iou:
                    best, best_iou = candidate, score
            if best is not None and best_iou >= IOU_THRESHOLD and not best[4]:
                best[4] = True
                tp[position] = 1
            else:
                fp[position] = 1
        cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
        recall = cum_tp / total
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        ap = 0.0
        for level in np.linspace(0, 1, 101):
            reachable = precision[recall >= level]
            ap += (reachable.max() if reachable.size else 0.0) / 101
        per_class[cls] = ap
    return per_class


def predict(model, split_dir, names, confidence=METRIC_CONFIDENCE):
    """Run the trained detector over a folder and return flat prediction records."""
    records = []
    for name in names:
        result = model.predict(
            split_dir / "images" / f"{name}.png",
            imgsz=IMAGE_SIZE,
            conf=confidence,
            verbose=False,
        )[0]
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            records.append(
                {
                    "image": name,
                    "class": int(box.cls.item()),
                    "confidence": float(box.conf.item()),
                    "box": (x1, y1, x2, y2),
                }
            )
    return records


def write_submission(path, records, order):
    """Write the prediction table in a given row order, one box per row."""
    lines = ["image_id,x1,y1,x2,y2,category_id,confidence"]
    for record in order:
        x1, y1, x2, y2 = (int(round(v)) for v in record["box"])
        lines.append(
            f"{record['image']},{x1},{y1},{x2},{y2},{record['class']},{record['confidence']:.4f}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return len(records)


def main():
    print("=" * 78)
    print("--- 1. A dataset whose every instance was recorded as it was drawn ---")
    truth = build_dataset()
    totals = Counter(label for boxes in truth.values() for label, *_ in boxes)
    print(f"{TOTAL_IMAGES} images at {IMAGE_SIZE}x{IMAGE_SIZE}, "
          f"{sum(totals.values())} instances")
    for cls in CLASSES:
        print(f"  {cls:<10}{totals[cls]:>5} instances")

    print()
    print("--- 2. VOC XML converted to the YOLO text layout ---")
    sample = sorted(truth)[0]
    xml_text = (ROOT / "raw" / "annotations" / f"{sample}.xml").read_text(encoding="utf-8")
    txt_text = (ROOT / "raw" / "labels" / f"{sample}.txt").read_text(encoding="utf-8")
    first_object = xml_text.split("<object>")[1].split("</object>")[0]
    print(f"{sample}.xml first object: {first_object.strip()[:96]}")
    print(f"{sample}.txt            : {txt_text.splitlines()[0]}")
    print("  absolute corners became a normalised centre and extent")

    print()
    print("--- 3. The split as the folder arrives ---")
    splits = split_names(truth, AS_ARRIVED)
    yaml_path = materialise(splits, "as_arrived")
    for split, names in splits.items():
        share = len(names) / TOTAL_IMAGES
        print(f"  {split:<6}{len(names):>5} images  {share:>6.1%} of the set")
    print(f"wrote {yaml_path}")

    print()
    print("--- 4. Auditing the split before any epoch runs ---")
    table = audit(splits, truth)
    header = f"{'split':<8}{'images':>8}{'inst':>7}" + "".join(f"{c:>10}" for c in CLASSES)
    print(header)
    for split, entry in table.items():
        row = f"{split:<8}{entry['images']:>8}{entry['instances']:>7}"
        row += "".join(f"{entry['per_class'][c]:>10}" for c in CLASSES)
        print(row)
    val = table["val"]
    print(f"the validation split holds {val['instances']} instances across "
          f"{len(CLASSES) - len(val['missing'])} of {len(CLASSES)} classes")
    if val["missing"]:
        print(f"  {', '.join(val['missing'])} never appear in it, so any mAP measured there "
              f"is an average over the classes that do")
    print("  and the checkpoint the trainer keeps is the one that scored best on exactly "
          f"these {val['images']} images")

    print()
    print("--- 5. Training, and reading the metric twice ---")
    os.environ["YOLO_VERBOSE"] = "false"
    from ultralytics import YOLO  # imported here so the earlier steps need no detector

    model = YOLO("yolo11n.yaml")
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH,
        project=str(ROOT / "runs"),
        name="audit",
        exist_ok=True,
        pretrained=False,
        seed=SEED,
        verbose=False,
        plots=False,
    )
    base = ROOT / "as_arrived"
    scores = {}
    for split in ("val", "test"):
        records = predict(model, base / split, splits[split])
        per_class = average_precision(records, truth, splits[split])
        scores[split] = (per_class, records)
        mean = sum(per_class.values()) / len(per_class) if per_class else 0.0
        detail = "  ".join(f"{cls}={value:.3f}" for cls, value in per_class.items())
        print(f"  {split:<5} mAP@{IOU_THRESHOLD} = {mean:.3f} over {len(per_class)} classes"
              f"   {detail}")
    val_mean = sum(scores["val"][0].values()) / len(scores["val"][0])
    test_mean = sum(scores["test"][0].values()) / len(scores["test"][0])
    print(f"  the validation figure is an average over {table['val']['instances']} instances "
          f"and the test figure over {table['test']['instances']}, so {val_mean:.3f} against "
          f"{test_mean:.3f} is a difference in sample size before it is anything else")
    print(f"  both were scored over every detection down to confidence {METRIC_CONFIDENCE}, "
          f"which is what average precision is defined over")

    truncated = predict(model, base / "test", splits["test"], confidence=VIEWING_CONFIDENCE)
    truncated_ap = average_precision(truncated, truth, splits["test"])
    truncated_mean = sum(truncated_ap.values()) / len(truncated_ap) if truncated_ap else 0.0
    full_mean = sum(scores["test"][0].values()) / len(scores["test"][0])
    print(f"  scoring the same weights at the default confidence {VIEWING_CONFIDENCE} instead "
          f"keeps {len(truncated)} of {len(scores['test'][1])} detections and reports "
          f"{truncated_mean:.3f} against {full_mean:.3f}")
    print("  the threshold that makes a picture readable is not the threshold the metric "
          "is defined at, and nothing warns when the two are swapped")

    print()
    print("--- 6. Two changes that move no boxes ---")
    test_records = scores["test"][1]
    test_names = splits["test"]
    submissions = ROOT / "submissions"
    submissions.mkdir(exist_ok=True)

    arrival = list(test_records)
    grouped = sorted(test_records, key=lambda r: (r["image"], -r["confidence"]))
    flattened = [dict(record, confidence=1.0) for record in grouped]

    variants = (
        ("as predicted", arrival),
        ("grouped by image_id", grouped),
        ("confidence set to 1.0", flattened),
    )
    for label, rows in variants:
        path = submissions / f"{label.replace(' ', '_')}.csv"
        write_submission(path, rows, rows)
        per_class = average_precision(rows, truth, test_names)
        mean = sum(per_class.values()) / len(per_class) if per_class else 0.0
        print(f"  {label:<24} {len(rows):>4} rows   mAP@{IOU_THRESHOLD} = {mean:.4f}")

    print("  sorting changed the file and not the score: average precision ranks the rows "
          "itself before scoring them")
    print("  flattening the confidence column did change it, because every row now ties and "
          "the ranking falls back to the order they were written in")
    print("  neither edit touched a coordinate, which is the whole point: the metric is a "
          "property of the submitted list, not only of the detector")
    print("=" * 78)


if __name__ == "__main__":
    main()
