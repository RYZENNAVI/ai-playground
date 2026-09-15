"""Describe a window by its gradients or by rectangle contrasts, and detect with each description.

Demonstrates the two hand-built detectors that preceded learned features:
    1. Render person windows, clutter windows, a star, and small face and non-face windows.
    2. Take Sobel gradients and draw direction with opacity set by magnitude.
    3. Vote gradients into 8x8 cell histograms three ways and measure what a 5-degree turn changes.
    4. Normalise overlapping blocks into a HOG descriptor and compare people with unknown windows.
    5. Enumerate every two-rectangle Haar feature in a window and count them in closed form.
    6. Build the integral image and check rectangle sums against brute force and OpenCV.
    7. Train a 20-round AdaBoost classifier over every feature of a 16x16 window.
    8. Score the strong classifier at several thresholds and draw the features it chose first.

Module 06: Multimodal Vision - HOG and Haar Detectors.
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "outputs" / "hog_and_haar"
SEED = 3407

WINDOW_H, WINDOW_W = 128, 64
CELL = 8
BINS = 9
BLOCK = 2
HYS_CLIP = 0.2

FACE = 16
PAPER_WINDOW = 24
ROUNDS = 20
TRAIN_PER_CLASS, TEST_PER_CLASS = 2000, 1000
STRONG_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


# ---------------------------------------------------------------------------
# 1. Synthetic windows
# ---------------------------------------------------------------------------

def clutter(rng, h, w):
    """Blurred texture with a few random strokes, the background every window starts from."""
    noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2.5)
    img = 0.5 + 0.08 * noise / noise.std()
    for _ in range(rng.integers(3, 7)):
        p = rng.integers(0, [w, h], 2)
        q = rng.integers(0, [w, h], 2)
        cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), float(rng.uniform(0.3, 0.7)), 1)
    return img


def person_window(rng):
    """A standing figure: head, torso, two arms and two legs, in a random pose and polarity."""
    img = clutter(rng, WINDOW_H, WINDOW_W)
    tone = 0.15 if rng.random() < 0.5 else 0.85
    cx = 32 + int(rng.integers(-3, 4))
    cv2.circle(img, (cx, 20), 8, tone, -1)
    cv2.ellipse(img, (cx, 52), (10, 22), 0, 0, 360, tone, -1)
    for side in (-1, 1):
        cv2.line(img, (cx + side * 8, 36), (cx + side * int(rng.integers(12, 20)), 70), tone, 4)
        cv2.line(img, (cx + side * 4, 72), (cx + side * int(rng.integers(4, 13)), 120), tone, 5)
    return cv2.GaussianBlur(img, (0, 0), 0.8)


def car_window(rng):
    """A wide low body with two wheels, the kind of structure a person descriptor should reject."""
    img = clutter(rng, WINDOW_H, WINDOW_W)
    tone = 0.2 if rng.random() < 0.5 else 0.8
    cv2.rectangle(img, (4, 70), (60, 92), tone, -1)
    cv2.rectangle(img, (14, 58), (48, 72), tone, -1)
    for x in (16, 48):
        cv2.circle(img, (x, 94), 7, 1.0 - tone, -1)
    return cv2.GaussianBlur(img, (0, 0), 0.8)


def star_image():
    """A filled five-pointed star, rendered once and turned later."""
    img = np.full((128, 128), 0.2, np.float32)
    angles = np.radians(np.arange(10) * 36 - 90)
    radii = np.where(np.arange(10) % 2 == 0, 50, 20)
    points = np.stack([64 + radii * np.cos(angles), 64 + radii * np.sin(angles)], 1)
    cv2.fillPoly(img, [np.rint(points).astype(np.int32)], 0.9)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def rotate(img, degrees):
    """Turn an image about its centre, filling the corners by reflection."""
    h, w = img.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def face_window(rng):
    """A 16x16 face: a bright oval, a dark eye band with two eyes, a nose bridge and a mouth."""
    yy, xx = np.mgrid[0:FACE, 0:FACE].astype(np.float32)
    img = np.full((FACE, FACE), rng.uniform(0.2, 0.8), np.float32)
    oval = ((xx - 7.5) / 7.0) ** 2 + ((yy - 8.0) / 8.0) ** 2 <= 1
    img[oval] = 0.62
    img[5:7, 2:14] -= 0.12
    for ex in (4.5, 10.5):
        img[np.hypot(xx - ex, yy - 5.5) <= 1.6] -= 0.25
    img[4:9, 7:9] += 0.08
    img[11:13, 5:11] -= 0.18
    matrix = cv2.getRotationMatrix2D((7.5, 8), float(rng.uniform(-8, 8)), float(rng.uniform(0.9, 1.1)))
    matrix[:, 2] += rng.uniform(-1, 1, 2)
    img = cv2.warpAffine(img, matrix, (FACE, FACE), borderMode=cv2.BORDER_REPLICATE)
    img = rng.uniform(0.6, 1.3) * img + rng.uniform(-0.15, 0.15)
    return cv2.GaussianBlur(img + rng.normal(0, 0.04, img.shape).astype(np.float32), (0, 0), 0.5)


def non_face_window(rng):
    """Texture, a ramp, rectangles, or a plain oval with no features on it."""
    kind = rng.integers(0, 4)
    if kind == 0:
        img = cv2.GaussianBlur(rng.normal(0.5, 0.2, (FACE, FACE)).astype(np.float32), (0, 0), rng.uniform(0.5, 2))
    elif kind == 1:
        t = np.linspace(0, 1, FACE, dtype=np.float32)
        img = np.outer(t, np.ones(FACE, np.float32)) if rng.random() < 0.5 else np.outer(np.ones(FACE, np.float32), t)
        img = img * rng.uniform(-0.6, 0.6) + 0.5
    elif kind == 2:
        img = np.full((FACE, FACE), rng.uniform(0.2, 0.8), np.float32)
        for _ in range(rng.integers(1, 4)):
            x0, y0 = rng.integers(0, FACE - 3, 2)
            x1, y1 = x0 + rng.integers(2, FACE - x0 + 1), y0 + rng.integers(2, FACE - y0 + 1)
            img[y0:y1, x0:x1] = rng.uniform(0, 1)
    else:
        yy, xx = np.mgrid[0:FACE, 0:FACE]
        img = np.full((FACE, FACE), rng.uniform(0.2, 0.8), np.float32)
        img[((xx - 7.5) / 7.0) ** 2 + ((yy - 8.0) / 8.0) ** 2 <= 1] = 0.62
    return (img + rng.normal(0, 0.04, img.shape)).astype(np.float32)


# ---------------------------------------------------------------------------
# 2-4. Gradients, cell histograms and the HOG descriptor
# ---------------------------------------------------------------------------

def gradients(img):
    """Sobel magnitude and unsigned direction in degrees on [0, 180)."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy), np.degrees(np.arctan2(gy, gx)) % 180


def direction_overlay(img, magnitude, direction):
    """Colour each pixel by gradient direction, opaque where the gradient is strong."""
    hue = np.rint(direction).astype(np.uint8)
    hsv = np.stack([hue, np.full_like(hue, 255), np.full_like(hue, 255)], -1)
    colour = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255
    alpha = (magnitude / magnitude.max())[..., None]
    base = np.repeat(img[..., None], 3, axis=2)
    return np.clip(255 * (alpha * colour + (1 - alpha) * base), 0, 255).astype(np.uint8)


def cell_histograms(img, mode):
    """Vote every pixel's gradient magnitude into (cells_y, cells_x, BINS) histograms.

    nearest: the whole vote goes to the one bin the direction falls in.
    orientation: the vote is split between the two bins whose centres bracket the
    direction, in proportion to how close it is to each, so a direction near a bin
    boundary no longer jumps from one bin to the other when it moves slightly.
    orientation+spatial: the vote is additionally split between the four cells
    whose centres surround the pixel, so a gradient near a cell border does not
    jump between cells when the image shifts or turns slightly.
    """
    magnitude, direction = gradients(img)
    h, w = img.shape
    cells_y, cells_x = h // CELL, w // CELL
    hist = np.zeros((cells_y, cells_x, BINS), np.float64)
    width = 180 / BINS
    ys, xs = np.mgrid[0:cells_y * CELL, 0:cells_x * CELL]
    mag = magnitude[:cells_y * CELL, :cells_x * CELL].ravel()
    ang = direction[:cells_y * CELL, :cells_x * CELL].ravel()
    ys, xs = ys.ravel(), xs.ravel()

    if mode == "nearest":
        np.add.at(hist, (ys // CELL, xs // CELL, (ang // width).astype(int) % BINS), mag)
        return hist

    position = ang / width - 0.5
    low = np.floor(position)
    orientation_votes = [((low.astype(int)) % BINS, 1 - (position - low)),
                         ((low.astype(int) + 1) % BINS, position - low)]
    if mode == "orientation":
        spatial_votes = [(ys // CELL, xs // CELL, np.ones_like(mag))]
    else:
        cy, cx = (ys + 0.5) / CELL - 0.5, (xs + 0.5) / CELL - 0.5
        y0, x0 = np.floor(cy), np.floor(cx)
        fy, fx = cy - y0, cx - x0
        spatial_votes = []
        for dy, wy in ((0, 1 - fy), (1, fy)):
            for dx, wx in ((0, 1 - fx), (1, fx)):
                spatial_votes.append(((y0 + dy).astype(int), (x0 + dx).astype(int), wy * wx))
    for row, col, spatial in spatial_votes:
        inside = (row >= 0) & (row < cells_y) & (col >= 0) & (col < cells_x)
        for bin_index, share in orientation_votes:
            np.add.at(hist, (row[inside], col[inside], bin_index[inside]),
                      (mag * spatial * share)[inside])
    return hist


def hog_descriptor(img):
    """Concatenate L2-Hys normalised 2x2-cell blocks, stride one cell, then normalise the whole.

    Each block vector is divided by its length, clipped at HYS_CLIP so that no
    single strong edge dominates, and divided by its length again. Neighbouring
    blocks share cells, so every cell appears in up to four blocks, each time
    normalised against a different neighbourhood. The finished descriptor is
    normalised once more so that windows of different overall contrast compare.
    """
    hist = cell_histograms(img, "orientation+spatial")
    blocks = []
    for y in range(hist.shape[0] - BLOCK + 1):
        for x in range(hist.shape[1] - BLOCK + 1):
            v = hist[y:y + BLOCK, x:x + BLOCK].ravel()
            v = v / np.sqrt((v ** 2).sum() + 1e-6)
            v = np.minimum(v, HYS_CLIP)
            blocks.append(v / np.sqrt((v ** 2).sum() + 1e-6))
    descriptor = np.concatenate(blocks)
    return descriptor / np.linalg.norm(descriptor)


# ---------------------------------------------------------------------------
# 5-6. Haar features and the integral image
# ---------------------------------------------------------------------------

def enumerate_two_rect(size):
    """Every horizontal and vertical two-rectangle feature as (vertical, x, y, w, h).

    (x, y) is the top-left corner and (w, h) the size of one of the two equal
    rectangles. A horizontal pair spans 2w by h; a vertical pair spans w by 2h.
    """
    rows = []
    for vertical in (0, 1):
        for a in range(1, size // 2 + 1):
            for b in range(1, size + 1):
                w, h = (b, a) if vertical else (a, b)
                span_w, span_h = (w, 2 * h) if vertical else (2 * w, h)
                ys, xs = np.mgrid[0:size - span_h + 1, 0:size - span_w + 1]
                block = np.stack([np.full(xs.size, vertical), xs.ravel(), ys.ravel(),
                                  np.full(xs.size, w), np.full(xs.size, h)], 1)
                rows.append(block)
    return np.concatenate(rows).astype(np.int32)


def closed_form_count(size):
    """Placements of a doubled side times placements of a plain side, for both orientations.

    A pair whose doubled side is 2a fits in size - 2a + 1 positions along that
    axis, for a = 1 .. size // 2; the other side b fits in size - b + 1 positions
    for b = 1 .. size. The two axes are independent, and vertical pairs mirror
    horizontal ones.
    """
    doubled = sum(size - 2 * a + 1 for a in range(1, size // 2 + 1))
    plain = sum(size - b + 1 for b in range(1, size + 1))
    return 2 * doubled * plain


def integral_image(img):
    """ii[y, x] = sum of img over rows < y and columns < x, with a leading row and column of zeros."""
    ii = np.zeros((img.shape[-2] + 1, img.shape[-1] + 1) if img.ndim == 2 else
                  (img.shape[0], img.shape[1] + 1, img.shape[2] + 1), np.float64)
    ii[..., 1:, 1:] = img.cumsum(axis=-2).cumsum(axis=-1)
    return ii


def rect_sum(ii, x, y, w, h):
    """Sum over a rectangle from four integral-image lookups, whatever its size."""
    return ii[..., y + h, x + w] - ii[..., y, x + w] - ii[..., y + h, x] + ii[..., y, x]


def feature_values(ii, features, chunk=4000):
    """Second rectangle minus first, for every window (rows) and feature (columns).

    The second rectangle sits to the right of the first for a horizontal pair and
    below it for a vertical one. Features are evaluated in chunks so that the
    four float64 lookups per feature never exist for the whole set at once.
    """
    values = np.empty((ii.shape[0], len(features)), np.float32)
    for start in range(0, len(features), chunk):
        block = features[start:start + chunk]
        vertical, x, y, w, h = block.T
        first = rect_sum(ii, x, y, w, h)
        second_x = np.where(vertical == 1, x, x + w)
        second_y = np.where(vertical == 1, y + h, y)
        values[:, start:start + chunk] = rect_sum(ii, second_x, second_y, w, h) - first
    return values


def normalise_windows(windows):
    """Subtract each window's mean and divide by its standard deviation, as before scanning."""
    flat = windows.reshape(len(windows), -1)
    mean, std = flat.mean(1), flat.std(1) + 1e-6
    return ((windows - mean[:, None, None]) / std[:, None, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# 7-8. AdaBoost
# ---------------------------------------------------------------------------

def best_stumps(values, order, sorted_values, weights, labels, chunk=2000):
    """For every feature, the threshold and polarity with the lowest weighted error.

    Sorting a feature's values once turns the search into two cumulative sums.
    Putting the threshold after the i-th smallest value and calling everything at
    or below it a face costs the weight of the non-faces at or below it plus the
    faces above it; the opposite polarity costs the complement. Both are read off
    for every i at once, and the order of values never changes between rounds.
    """
    positive = np.where(labels == 1, weights, 0).astype(np.float64)
    negative = np.where(labels == 0, weights, 0).astype(np.float64)
    total_pos, total_neg = positive.sum(), negative.sum()
    best = (np.inf, 0, 0.0, 1)
    for start in range(0, values.shape[1], chunk):
        idx = order[:, start:start + chunk]
        below_pos = np.cumsum(positive[idx], axis=0)
        below_neg = np.cumsum(negative[idx], axis=0)
        error_low = below_neg + (total_pos - below_pos)
        error_high = below_pos + (total_neg - below_neg)
        for error, polarity in ((error_low, 1), (error_high, -1)):
            flat = int(np.argmin(error))
            row, col = np.unravel_index(flat, error.shape)
            if error[row, col] < best[0]:
                best = (float(error[row, col]), start + col, float(sorted_values[row, start + col]), polarity)
    return best


def stump_predict(values, threshold, polarity):
    """1 where polarity * value <= polarity * threshold, else 0."""
    return (polarity * values <= polarity * threshold).astype(np.int8)


def train_adaboost(values, labels, rounds=ROUNDS):
    """Discrete AdaBoost with the initial weights and update of the face-detection formulation.

    Faces and non-faces each start with half the total weight, however many of
    each there are. Each round normalises the weights, takes the single feature
    stump with the lowest weighted error epsilon, and multiplies the weight of every
    example that stump got right by beta = epsilon / (1 - epsilon), so the next round
    concentrates on the examples still being misclassified. The stump's say in the
    final vote is alpha = log(1 / beta).
    """
    order = np.argsort(values, axis=0, kind="stable").astype(np.int32)
    sorted_values = np.take_along_axis(values, order, axis=0)
    pos = labels == 1
    weights = np.where(pos, 1 / (2 * pos.sum()), 1 / (2 * (~pos).sum()))
    stumps = []
    for _ in range(rounds):
        weights = weights / weights.sum()
        _, feature, threshold, polarity = best_stumps(values, order, sorted_values, weights, labels)
        predicted = stump_predict(values[:, feature], threshold, polarity)
        epsilon = float(weights[predicted != labels].sum())
        beta = max(epsilon, 1e-10) / (1 - epsilon)
        weights = weights * np.where(predicted == labels, beta, 1.0)
        stumps.append((feature, threshold, polarity, float(np.log(1 / beta)), epsilon))
    return stumps


def strong_scores(values_by_stump, stumps):
    """Weighted vote as a share of the total weight: 1 means every stump says face."""
    alphas = np.array([s[3] for s in stumps])
    votes = np.stack([stump_predict(values_by_stump[:, i], s[1], s[2]) for i, s in enumerate(stumps)], 1)
    return votes @ alphas / alphas.sum()


def report_thresholds(scores, labels):
    """Rates at each strong-classifier threshold, with the constant prediction for reference."""
    print(f"  {'threshold':>9}{'faces found':>13}{'false alarms':>14}{'accuracy':>10}")
    for theta in STRONG_THRESHOLDS:
        predicted = scores >= theta
        print(f"  {theta:>9.1f}{predicted[labels == 1].mean():>13.1%}"
              f"{predicted[labels == 0].mean():>14.1%}{(predicted == labels).mean():>10.1%}")
    majority = max(labels.mean(), 1 - labels.mean())
    print(f"  predicting the larger class for every window scores {majority:.1%} on this test set")


def run_adaboost(name, train_windows, train_labels, test_windows, test_labels, features):
    """Compute features, train, and report one dataset; return the stumps."""
    started = time.perf_counter()
    train_values = feature_values(integral_image(normalise_windows(train_windows)), features)
    stumps = train_adaboost(train_values, train_labels)
    elapsed = time.perf_counter() - started
    chosen = features[[s[0] for s in stumps]]
    test_values = feature_values(integral_image(normalise_windows(test_windows)), chosen)
    print(f"  {name}: {len(train_labels)} training windows x {len(features)} features, "
          f"{ROUNDS} rounds in {elapsed:.1f} s")
    print(f"  {'round':>5}{'feature (vertical, x, y, w, h)':>34}{'epsilon':>9}{'alpha':>8}")
    for index, (feature, _, _, alpha, epsilon) in enumerate(stumps[:5], 1):
        print(f"  {index:>5}{str(tuple(int(v) for v in features[feature])):>34}{epsilon:>9.3f}{alpha:>8.3f}")
    print(f"  ... round {ROUNDS}: epsilon {stumps[-1][4]:.3f}, alpha {stumps[-1][3]:.3f}")
    report_thresholds(strong_scores(test_values, stumps), test_labels)
    return stumps


def load_tinyface(root, count, rng):
    """Greyscale 16x16 crops from the TinyFace training images."""
    paths = sorted(Path(root).glob("Training_Set/**/*.jpg"))
    picked = rng.choice(len(paths), size=min(count, len(paths)), replace=False)
    windows = [cv2.resize(cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE), (FACE, FACE),
                          interpolation=cv2.INTER_AREA) for i in picked]
    return np.stack(windows).astype(np.float32) / 255, len(paths)


def load_cifar10(root, count, rng):
    """Greyscale 16x16 versions of CIFAR-10 training images."""
    with open(Path(root) / "data_batch_1", "rb") as handle:
        batch = pickle.load(handle, encoding="bytes")
    images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    picked = rng.choice(len(images), size=count, replace=False)
    windows = [cv2.resize(cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY), (FACE, FACE),
                          interpolation=cv2.INTER_AREA) for i in picked]
    return np.stack(windows).astype(np.float32) / 255


def split(faces, others, per_train, rng):
    """Shuffle two classes into training and test sets."""
    windows = np.concatenate([faces, others])
    labels = np.concatenate([np.ones(len(faces), np.int8), np.zeros(len(others), np.int8)])
    train = np.concatenate([rng.permutation(len(faces))[:per_train],
                            len(faces) + rng.permutation(len(others))[:per_train]])
    test = np.setdiff1d(np.arange(len(labels)), train)
    return windows[train], labels[train], windows[test], labels[test]


def draw_features(face, features, stumps, scale=15):
    """The mean face enlarged, with the first three chosen features outlined."""
    canvas = cv2.cvtColor(cv2.resize(np.clip(face * 255, 0, 255).astype(np.uint8), (FACE * scale, FACE * scale),
                                     interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    for index, (feature, _, _, _, _) in enumerate(stumps[:3]):
        vertical, x, y, w, h = (int(v) for v in features[feature])
        second = (x, y + h) if vertical else (x + w, y)
        colour = [(0, 0, 255), (0, 200, 0), (255, 0, 0)][index]
        cv2.rectangle(canvas, (x * scale, y * scale), ((x + w) * scale - 1, (y + h) * scale - 1), colour, 2)
        cv2.rectangle(canvas, (second[0] * scale, second[1] * scale),
                      ((second[0] + w) * scale - 1, (second[1] + h) * scale - 1), colour, 1)
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tinyface-root", help="TinyFace folder containing Training_Set/")
    parser.add_argument("--cifar10-root", help="folder containing the CIFAR-10 python batches")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # 1. Windows
    print("--- 1. Synthetic windows ---")
    people = [person_window(rng) for _ in range(3)]
    unknown_person, unknown_car = person_window(rng), car_window(rng)
    held_people = [person_window(rng) for _ in range(40)]
    held_cars = [car_window(rng) for _ in range(40)]
    held_clutter = [cv2.GaussianBlur(clutter(rng, WINDOW_H, WINDOW_W), (0, 0), 0.8) for _ in range(40)]
    star = star_image()
    faces = np.stack([face_window(rng) for _ in range(TRAIN_PER_CLASS + TEST_PER_CLASS)])
    others = np.stack([non_face_window(rng) for _ in range(TRAIN_PER_CLASS + TEST_PER_CLASS)])
    print(f"  person and car windows {WINDOW_W}x{WINDOW_H}; star 128x128; "
          f"{len(faces)} face and {len(others)} non-face windows {FACE}x{FACE}")
    cv2.imwrite(str(OUT_DIR / "windows.png"), np.clip(255 * np.hstack(
        people + [unknown_person, unknown_car]), 0, 255).astype(np.uint8))

    # 2. Gradients
    print("\n--- 2. Gradient direction, drawn with opacity proportional to magnitude ---")
    overlays = []
    for img in people:
        magnitude, direction = gradients(img)
        strong = magnitude > 0.25 * magnitude.max()
        overlays.append(direction_overlay(img, magnitude, direction))
        print(f"  person window: {strong.mean():.1%} of pixels above a quarter of the peak magnitude; "
              f"their directions, histogram of 4 bins: "
              f"{np.histogram(direction[strong], bins=4, range=(0, 180))[0].tolist()}")
    cv2.imwrite(str(OUT_DIR / "gradient_direction.png"), np.hstack(overlays))
    print("  Near-vertical limbs and torso sides have horizontal gradients, the 0-45 and 135-180")
    print("  bins; head and shoulders fill the middle. Weak gradients carry little shape, so they")
    print("  are drawn nearly transparent and, in the histograms below, vote with little weight.")

    # 3. Cell histograms under rotation
    print("\n--- 3. Cell histograms and a 5-degree turn ---")
    turned = rotate(star, -5)
    print(f"  {BINS} unsigned bins of {180 // BINS} deg, {CELL}x{CELL} cells")
    print(f"  {'voting':<22}{'distance, star vs turned':>26}{'relative to star':>18}")
    for mode in ("nearest", "orientation", "orientation+spatial"):
        a, b = cell_histograms(star, mode).ravel(), cell_histograms(turned, mode).ravel()
        distance = np.linalg.norm(a - b)
        print(f"  {mode:<22}{distance:>26.3f}{distance / np.linalg.norm(a):>18.3f}")
    print("  A 5-degree turn moves a direction a quarter of a bin. With the whole vote in one")
    print("  bin, every direction that crosses a boundary moves all of its weight; split")
    print("  between the two nearest bins, it moves a quarter. The same holds for pixels that")
    print("  the turn carries across a cell border.")

    # 4. HOG descriptor
    print("\n--- 4. Block-normalised HOG descriptor ---")
    reference = [hog_descriptor(p) for p in people]
    blocks_y, blocks_x = WINDOW_H // CELL - BLOCK + 1, WINDOW_W // CELL - BLOCK + 1
    print(f"  {WINDOW_W}x{WINDOW_H} window: {WINDOW_H // CELL}x{WINDOW_W // CELL} cells, "
          f"{blocks_y}x{blocks_x} blocks of {BLOCK}x{BLOCK}, length {len(reference[0])} "
          f"= {blocks_y * blocks_x} x {BLOCK * BLOCK * BINS}")
    print(f"  {'window':<16}" + "".join(f"{'person ' + str(i + 1):>10}" for i in range(3)) + f"{'mean':>9}")
    unknown_means = {}
    for name, img in (("unknown person", unknown_person), ("unknown car", unknown_car)):
        d = [np.linalg.norm(hog_descriptor(img) - r) for r in reference]
        unknown_means[name] = np.mean(d)
        print(f"  {name:<16}" + "".join(f"{v:>10.3f}" for v in d) + f"{np.mean(d):>9.3f}")
    closer = min(unknown_means, key=unknown_means.get)
    print(f"  smaller mean distance to the three people: {closer}")
    template = np.mean(reference, axis=0)
    print(f"  distance to the mean of the three people, {len(held_people)} new windows each:")
    for name, group in (("people", held_people), ("cars", held_cars), ("clutter", held_clutter)):
        d = np.array([np.linalg.norm(hog_descriptor(img) - template) for img in group])
        print(f"    {name:<8} mean {d.mean():.3f}  min {d.min():.3f}  max {d.max():.3f}")

    # 5. Haar features
    print("\n--- 5. Two-rectangle Haar features ---")
    for size in (PAPER_WINDOW, FACE):
        enumerated = enumerate_two_rect(size)
        print(f"  {size}x{size} window: enumerated {len(enumerated)}, closed form {closed_form_count(size)} "
              f"(horizontal {int((enumerated[:, 0] == 0).sum())}, vertical {int((enumerated[:, 0] == 1).sum())})")
    features = enumerate_two_rect(FACE)

    # 6. Integral image
    print("\n--- 6. The integral image ---")
    sample = np.clip(np.rint(faces[0] * 255), 0, 255).astype(np.uint8)
    ours = integral_image(sample.astype(np.float64))
    print(f"  largest difference from cv2.integral: {np.abs(ours - cv2.integral(sample)).max():.1f}")
    big = rng.random((480, 640))
    big_ii = integral_image(big)
    rects = np.stack([rng.integers(0, 400, 2000), rng.integers(0, 300, 2000),
                      rng.integers(1, 240, 2000), rng.integers(1, 180, 2000)], 1)
    started = time.perf_counter()
    brute = np.array([big[y:y + h, x:x + w].sum() for x, y, w, h in rects])
    brute_time = time.perf_counter() - started
    started = time.perf_counter()
    fast = rect_sum(big_ii, rects[:, 0], rects[:, 1], rects[:, 2], rects[:, 3])
    fast_time = time.perf_counter() - started
    print(f"  2000 random rectangles on a 640x480 image: largest difference {np.abs(brute - fast).max():.2e}; "
          f"slicing {brute_time * 1000:.1f} ms, four lookups each {fast_time * 1000:.2f} ms")
    print("  ii[y, x] holds the sum above and to the left, so any rectangle is D - B - C + A:")
    print("  the cost is four reads whether the rectangle is 2 pixels or 200,000.")
    cv2.imwrite(str(OUT_DIR / "integral_image.png"),
                np.hstack([cv2.resize(sample, (170, 170), interpolation=cv2.INTER_NEAREST),
                           cv2.resize(np.rint(255 * ours / ours.max()).astype(np.uint8), (170, 170),
                                      interpolation=cv2.INTER_NEAREST)]))

    # 7. AdaBoost
    print("\n--- 7. AdaBoost over every feature ---")
    train_w, train_l, test_w, test_l = split(faces, others, TRAIN_PER_CLASS, rng)
    stumps = run_adaboost("synthetic faces", train_w, train_l, test_w, test_l, features)

    # 8. Chosen features and real data
    print("\n--- 8. What the classifier looks at, and the same training on real images ---")
    cv2.imwrite(str(OUT_DIR / "haar_features.png"), draw_features(faces.mean(axis=0), features, stumps))
    for index, (feature, threshold, polarity, alpha, _) in enumerate(stumps[:3], 1):
        vertical, x, y, w, h = (int(v) for v in features[feature])
        kind = "top/bottom" if vertical else "left/right"
        print(f"  feature {index}: {kind} pair at ({x}, {y}), each {w}x{h}; face when second minus first "
              f"{'<=' if polarity == 1 else '>='} {threshold:+.2f} (alpha {alpha:.2f})")
    if args.tinyface_root and args.cifar10_root:
        real_faces, available = load_tinyface(args.tinyface_root, TRAIN_PER_CLASS + TEST_PER_CLASS, rng)
        real_others = load_cifar10(args.cifar10_root, TRAIN_PER_CLASS + TEST_PER_CLASS, rng)
        print(f"  TinyFace: {len(real_faces)} of {available} training crops; CIFAR-10: {len(real_others)} images")
        real = split(real_faces, real_others, TRAIN_PER_CLASS, rng)
        real_stumps = run_adaboost("TinyFace vs CIFAR-10", *real, features)
        cv2.imwrite(str(OUT_DIR / "haar_features_tinyface.png"),
                    draw_features(real_faces.mean(axis=0), features, real_stumps))
    else:
        print("  pass --tinyface-root and --cifar10-root to repeat the training on TinyFace crops")
        print("  against CIFAR-10 images")
    print(f"\n  images written to {OUT_DIR}")


if __name__ == "__main__":
    main()
