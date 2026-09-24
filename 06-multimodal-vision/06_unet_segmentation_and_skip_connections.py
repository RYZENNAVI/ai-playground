"""Label every pixel with a UNet, and measure what its skip connections and its resampling buy.

Demonstrates how a segmentation network is built and how it should be scored:
    1. Render a segmentation dataset with exact masks and measure how much of it is background.
    2. Score the constant prediction every model has to beat, and say what each metric counts.
    3. Build a UNet, the same encoder and decoder without skips, and a net that never resamples.
    4. Train all three and report pixel accuracy, mean IoU and per-class IoU.
    5. Score the three again only near class boundaries, where the skipped detail would show.
    6. Repeat the comparison on Pascal VOC 2012, with its ignore label excluded from every count.

Module 06: Multimodal Vision - Segmentation and Skip Connections.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "outputs" / "segmentation"
SEED = 3407

SIZE = 64
SHAPE_CLASSES = ("background", "box", "disk", "triangle", "bar")
TRAIN_IMAGES, TEST_IMAGES = 2400, 400
EPOCHS, BATCH, LEARNING_RATE = 12, 32, 2e-3
BOUNDARY_BAND = 3

VOC_SIZE, VOC_EPOCHS, VOC_IGNORE = 128, 16, 255
VOC_CLASSES = ("background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor")


# 1-2. Data and the constant baseline

def render_segmentation(rng):
    """One image and its exact label map: shapes painted on texture, each with its own class."""
    noise = cv2.GaussianBlur(rng.normal(0, 1, (SIZE, SIZE, 3)).astype(np.float32), (0, 0), 2.5)
    img = np.clip(110 + 40 * noise / noise.std(), 0, 255)
    label = np.zeros((SIZE, SIZE), np.uint8)
    for _ in range(rng.integers(1, 5)):
        kind = int(rng.integers(1, len(SHAPE_CLASSES)))
        colour = rng.integers(20, 256, 3).astype(np.float32)
        mask = np.zeros((SIZE, SIZE), np.uint8)
        cx, cy = rng.integers(10, SIZE - 10, 2)
        size = int(rng.integers(8, 22))
        if kind == 1:
            cv2.rectangle(mask, (cx - size // 2, cy - size // 2), (cx + size // 2, cy + size // 2), 1, -1)
        elif kind == 2:
            cv2.circle(mask, (int(cx), int(cy)), size // 2, 1, -1)
        elif kind == 3:
            cv2.fillPoly(mask, [np.array([(cx, cy - size // 2), (cx - size // 2, cy + size // 2),
                                          (cx + size // 2, cy + size // 2)], np.int32)], 1)
        else:
            angle = float(rng.uniform(0, 180))
            box = ((float(cx), float(cy)), (float(size * 2), 5.0), angle)
            cv2.fillPoly(mask, [np.rint(cv2.boxPoints(box)).astype(np.int32)], 1)
        img[mask > 0] = colour
        label[mask > 0] = kind
    return img.astype(np.uint8), label


def confusion(predicted, truth, classes, ignore=None):
    """Counts of (true class, predicted class) pairs, ignoring pixels labelled `ignore`."""
    keep = truth != ignore if ignore is not None else np.ones_like(truth, bool)
    pairs = truth[keep].astype(np.int64) * classes + predicted[keep].astype(np.int64)
    return np.bincount(pairs, minlength=classes * classes).reshape(classes, classes)


def metrics(matrix):
    """Pixel accuracy, mean IoU and per-class IoU from a confusion matrix.

    Pixel accuracy is the share of pixels on the diagonal, so a class covering most
    of the image sets its floor. IoU for a class divides its diagonal entry by every
    pixel that is either labelled it or predicted it, which counts a class that is
    never predicted as zero however small it is; mean IoU averages that over the
    classes that appear.
    """
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(0) + matrix.sum(1) - intersection
    present = matrix.sum(1) > 0
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    return intersection.sum() / matrix.sum(), float(iou[present].mean()), iou, present


def boundary_band(labels, width=BOUNDARY_BAND):
    """Pixels within `width` of a place where the label changes."""
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(labels, kernel) != cv2.erode(labels, kernel)
    return cv2.dilate(edges.astype(np.uint8), kernel, iterations=width - 1) > 0


# 3. Three networks

def make_models(classes, in_channels=3):
    """A UNet, the same shape without skips, and a network that stays at full resolution."""
    import torch
    import torch.nn as nn

    def double_conv(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                             nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        """Three halvings, a bottom, and three doublings; each doubling may see the matching encoder map."""

        def __init__(self, base=24, skips=True):
            super().__init__()
            self.skips = skips
            self.down1, self.down2, self.down3 = double_conv(in_channels, base), double_conv(base, base * 2), \
                double_conv(base * 2, base * 4)
            self.pool = nn.MaxPool2d(2)
            self.bottom = double_conv(base * 4, base * 8)
            self.up1 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.up3 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            factor = 2 if skips else 1
            self.conv1 = double_conv(base * 4 * factor, base * 4)
            self.conv2 = double_conv(base * 2 * factor, base * 2)
            self.conv3 = double_conv(base * factor, base)
            self.head = nn.Conv2d(base, classes, 1)

        def forward(self, x):
            x1 = self.down1(x)
            x2 = self.down2(self.pool(x1))
            x3 = self.down3(self.pool(x2))
            y = self.bottom(self.pool(x3))
            for up, conv, skip in ((self.up1, self.conv1, x3), (self.up2, self.conv2, x2),
                                   (self.up3, self.conv3, x1)):
                y = up(y)
                y = conv(torch.cat([y, skip], 1) if self.skips else y)
            return self.head(y)

    class FlatNet(nn.Module):
        """Convolutions at full resolution throughout: no pooling, no upsampling, no skips.

        Its channel counts are held down by what full resolution costs: every layer
        runs on sixty-four times the pixels the UNet's bottom layer does, so matching
        the UNet's parameter count here would cost far more than matching its time.
        """

        def __init__(self, base=48):
            super().__init__()
            self.body = nn.Sequential(double_conv(in_channels, base), double_conv(base, base * 2),
                                      double_conv(base * 2, base * 2))
            self.head = nn.Conv2d(base * 2, classes, 1)

        def forward(self, x):
            return self.head(self.body(x))

    return {"UNet with skips": UNet(skips=True), "UNet without skips": UNet(base=30, skips=False),
            "flat, no resampling": FlatNet()}


def receptive_field(halvings, blocks_per_level=2, kernel=3):
    """Approximately how far apart two input pixels can be and still reach the same output unit.

    Every 3x3 convolution adds two pixels to the field at the resolution it runs at,
    and every halving doubles what one pixel at that resolution covers in the input.
    Only the convolutions are counted: the 2x2 pooling windows and the transposed
    convolutions widen the field a little more, so the figure is an illustrative
    approximation of the real network's field, not an exact property of it.
    """
    field, jump = 1, 1
    for level in range(halvings + 1):
        for _ in range(blocks_per_level):
            field += (kernel - 1) * jump
        if level < halvings:
            jump *= 2
    for level in range(halvings, 0, -1):
        jump //= 2
        for _ in range(blocks_per_level):
            field += (kernel - 1) * jump
    return field


# 4-6. Training and scoring

def score_model(model, x_test, y_test, classes, device, ignore=None, with_boundary=True):
    """Confusion matrices over the test set, overall and within the boundary band."""
    import torch

    model.eval()
    matrix = np.zeros((classes, classes), np.int64)
    boundary = np.zeros((classes, classes), np.int64)
    with torch.no_grad():
        for start in range(0, len(x_test), 100):
            predicted = model(x_test[start:start + 100].to(device)).argmax(1).cpu().numpy()
            truth = y_test[start:start + 100].numpy()
            matrix += confusion(predicted, truth, classes, ignore)
            if with_boundary:
                band = np.stack([boundary_band(t.astype(np.uint8)) for t in truth])
                near = np.where(band, truth, ignore if ignore is not None else -1)
                boundary += confusion(predicted, near, classes, ignore if ignore is not None else -1)
    return matrix, boundary


def train_model(model, data, classes, epochs, device, ignore=None):
    """Train one model with Adam and cross-entropy, returning its confusion matrices and history.

    After every epoch the model is scored on the test set for the training curve.
    Scoring changes no weights and draws no random numbers, and its time is kept out
    of the reported training time.
    """
    import torch
    import torch.nn.functional as F

    x_train, y_train, x_test, y_test = data
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(SEED)
    started, scoring, history = time.perf_counter(), 0.0, []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        total, seen = 0.0, 0
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            images, labels = x_train[idx].to(device), y_train[idx].to(device)
            # Mirroring a labelled pair is free data; one coin decides for the whole batch.
            if torch.rand(1, generator=generator).item() < 0.5:
                images, labels = images.flip(-1), labels.flip(-1)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, ignore_index=ignore if ignore is not None else -100)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total, seen = total + loss.item() * len(idx), seen + len(idx)
        scoring_started = time.perf_counter()
        epoch_matrix, _ = score_model(model, x_test, y_test, classes, device, ignore, with_boundary=False)
        history.append((total / seen, metrics(epoch_matrix)[1]))
        scoring += time.perf_counter() - scoring_started
    elapsed = time.perf_counter() - started - scoring
    matrix, boundary = score_model(model, x_test, y_test, classes, device, ignore)
    return matrix, boundary, elapsed, history


def report(name, matrix, elapsed, parameters):
    """One line of headline numbers for a trained model."""
    accuracy, miou, _, _ = metrics(matrix)
    print(f"  {name:<22}{parameters:>11}{elapsed:>8.0f}s{accuracy:>16.2%}{miou:>11.3f}")


def load_voc(root, count=None):
    """Images and label maps from the VOC 2012 segmentation split, resized to VOC_SIZE."""
    root = Path(root)
    base = root / "VOC2012" if (root / "VOC2012").is_dir() else root
    splits = {}
    for split in ("train", "val"):
        names = (base / "ImageSets" / "Segmentation" / f"{split}.txt").read_text().split()
        if count:
            names = names[:count]
        images, labels = [], []
        from PIL import Image

        for name in names:
            img = cv2.imread(str(base / "JPEGImages" / f"{name}.jpg"))
            # The masks are palette PNGs whose pixel values are the class indices; reading
            # them through a decoder that resolves the palette would return colours instead.
            mask = np.array(Image.open(base / "SegmentationClass" / f"{name}.png"))
            images.append(cv2.resize(img, (VOC_SIZE, VOC_SIZE), interpolation=cv2.INTER_AREA))
            labels.append(cv2.resize(mask, (VOC_SIZE, VOC_SIZE), interpolation=cv2.INTER_NEAREST))
        splits[split] = (np.stack(images), np.stack(labels))
    return splits


def palette_image(labels, classes):
    """Colour a label map so it can be looked at, with the ignore label in grey."""
    colours = np.array([(0, 0, 0)] + [tuple(int(v) for v in cv2.applyColorMap(
        np.uint8([[i * 255 // max(classes - 1, 1)]]), cv2.COLORMAP_HSV)[0, 0]) for i in range(1, classes)], np.uint8)
    out = np.zeros(labels.shape + (3,), np.uint8)
    inside = labels < classes
    out[inside] = colours[labels[inside]]
    out[~inside] = (128, 128, 128)
    return out


def labelled(image, text):
    """A BGR uint8 image with its label on a black strip in the top-left corner."""
    image = image.copy()
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(image, (2, 2), (10 + text_w, 10 + text_h), (0, 0, 0), -1)
    cv2.putText(image, text, (6, 6 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    return image


def enlarge(image, scale):
    """Nearest-neighbour enlargement, so label maps keep hard edges."""
    return cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale), interpolation=cv2.INTER_NEAREST)


def tile_rows(rows):
    """Rows of equal-sized tiles, with grey dividers between tiles and between rows."""
    stacked = []
    for row in rows:
        cells = []
        for cell in row:
            cells += [cell, np.full((cell.shape[0], 4, 3), 128, np.uint8)]
        line = np.hstack(cells[:-1])
        stacked += [line, np.full((4, line.shape[1], 3), 128, np.uint8)]
    return np.vstack(stacked[:-1])


# Main

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--voc-root", help="folder holding VOC2012/ (or the VOC2012 folder itself)")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.manual_seed(SEED)
    # cuDNN chooses a convolution algorithm per shape and some of them accumulate in a
    # non-deterministic order, which moves mean IoU by a few points between runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data
    print("--- 1. A segmentation dataset whose masks are exact ---")
    train = [render_segmentation(rng) for _ in range(TRAIN_IMAGES)]
    test = [render_segmentation(rng) for _ in range(TEST_IMAGES)]
    train_labels = np.stack([label for _, label in train])
    test_labels = np.stack([label for _, label in test])
    shares = np.bincount(train_labels.ravel(), minlength=len(SHAPE_CLASSES)) / train_labels.size
    print(f"  {TRAIN_IMAGES} training and {TEST_IMAGES} test images of {SIZE}x{SIZE}, "
          f"{len(SHAPE_CLASSES)} classes")
    print("  " + "  ".join(f"{name} {share:.1%}" for name, share in zip(SHAPE_CLASSES, shares)))
    band = np.stack([boundary_band(label) for label in test_labels])
    print(f"  pixels within {BOUNDARY_BAND} of a class boundary: {band.mean():.1%} of the test set")
    legend = ", ".join(SHAPE_CLASSES[1:])
    cv2.imwrite(str(OUT_DIR / "shapes_dataset.png"), tile_rows([
        [labelled(enlarge(img, 3), "image") if i == 0 else enlarge(img, 3) for i, (img, _) in enumerate(train[:8])],
        [labelled(enlarge(palette_image(label, len(SHAPE_CLASSES)), 3), f"labels: {legend}") if i == 0
         else enlarge(palette_image(label, len(SHAPE_CLASSES)), 3) for i, (_, label) in enumerate(train[:8])]]))
    print("  shapes_dataset.png: eight training images above their label maps")

    # 2. Baseline
    print("\n--- 2. The constant prediction, and what each metric counts ---")
    constant = np.zeros_like(test_labels)
    matrix = confusion(constant, test_labels, len(SHAPE_CLASSES))
    accuracy, miou, iou, _ = metrics(matrix)
    print(f"  predicting background everywhere: pixel accuracy {accuracy:.2%}, mean IoU {miou:.3f}")
    print(f"  per-class IoU " + "  ".join(f"{name} {value:.3f}" for name, value in zip(SHAPE_CLASSES, iou)))
    print("  Pixel accuracy is dominated by the class with the most pixels, so it starts high")
    print("  whatever the model does; mean IoU gives every class the same weight, and a class")
    print("  the model never predicts scores zero.")

    # 3. Models
    print("\n--- 3. Three architectures for the same task ---")
    models = make_models(len(SHAPE_CLASSES))
    for name, model in models.items():
        total = sum(p.numel() for p in model.parameters())
        print(f"  {name:<22}{total:>10} parameters")
    print("  The counts are not matched. The UNet without skips is widened from 24 to 30 base")
    print("  channels and ends up with more parameters than the one with skips; the flat network")
    print("  is kept narrow because every one of its layers runs at full resolution.")
    print(f"  receptive field at the output, counting convolutions only: about {receptive_field(3)} px for the UNet, "
          f"{receptive_field(0, blocks_per_level=6)} px for the flat network")
    print("  Every 3x3 convolution adds two pixels to that field; halving the resolution first")
    print("  doubles what each of those pixels covers, which is how a stack of small kernels")
    print("  comes to see a large area without becoming a stack of large ones.")
    # The UNet's field is wider than the 64 px image, so the image sits on a black margin
    # large enough for the whole square to be drawn around it.
    zoom = 4
    margin = (receptive_field(3) - SIZE) // 2 * zoom + 20
    field_canvas = cv2.copyMakeBorder(enlarge(test[0][0], zoom), margin + 24, margin, margin, margin,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
    centre_x, centre_y = margin + SIZE // 2 * zoom, margin + 24 + SIZE // 2 * zoom
    for field, colour, name in ((receptive_field(3), (0, 255, 255), "UNet"),
                                (receptive_field(0, blocks_per_level=6), (255, 255, 0), "flat")):
        half = field * zoom // 2
        cv2.rectangle(field_canvas, (centre_x - half, centre_y - half), (centre_x + half, centre_y + half), colour, 3)
    cv2.circle(field_canvas, (centre_x, centre_y), 5, (0, 0, 255), -1)
    field_canvas = labelled(field_canvas, f"red pixel sees: UNet ~{receptive_field(3)} px (yellow), "
                                          f"flat ~{receptive_field(0, blocks_per_level=6)} px (cyan)")
    cv2.imwrite(str(OUT_DIR / "receptive_field.png"), field_canvas)
    print("  receptive_field.png: the approximate field of one output pixel for each design, on a test image")

    # 4. Training
    print("\n--- 4. Training all three ---")
    x_train = torch.from_numpy(np.stack([img for img, _ in train])).permute(0, 3, 1, 2).float().div(255)
    x_test = torch.from_numpy(np.stack([img for img, _ in test])).permute(0, 3, 1, 2).float().div(255)
    data = (x_train, torch.from_numpy(train_labels).long(), x_test, torch.from_numpy(test_labels).long())
    print(f"  Adam {LEARNING_RATE}, batch {BATCH}, {EPOCHS} epochs on {device}")
    print(f"  {'model':<22}{'parameters':>11}{'time':>9}{'pixel accuracy':>15}{'mean IoU':>11}")
    results = {}
    histories = {}
    for name, model in models.items():
        model = model.to(device)
        matrix, boundary, elapsed, histories[name] = train_model(model, data, len(SHAPE_CLASSES), EPOCHS, device)
        results[name] = (matrix, boundary, model)
        report(name, matrix, elapsed, sum(p.numel() for p in model.parameters()))
    print(f"  {'background everywhere':<22}{0:>11}{0:>8}s{accuracy:>16.2%}{miou:>11.3f}")
    print(f"  {'class':<22}" + "".join(f"{name.split(',')[0][:11]:>13}" for name in results))
    for index, name in enumerate(SHAPE_CLASSES):
        print(f"  IoU {name:<18}" + "".join(f"{metrics(m)[2][index]:>13.3f}" for m, _, _ in results.values()))

    def plot_curves(history_by_model, title, path):
        """Training loss and test mean IoU after every epoch, one line per model."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for name, history in history_by_model.items():
            epochs = np.arange(1, len(history) + 1)
            axes[0].plot(epochs, [h[0] for h in history], "o-", label=name)
            axes[1].plot(epochs, [h[1] for h in history], "o-", label=name)
        axes[0].set_title(f"{title}: training loss, mean over the epoch", fontsize=10)
        axes[1].set_title(f"{title}: test mean IoU after each epoch", fontsize=10)
        for ax in axes:
            ax.set_xlabel("epoch")
            ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)

    plot_curves(histories, "rendered shapes", OUT_DIR / "training_curves.png")
    names = ["background everywhere"] + list(results)
    scores = [(accuracy, miou, iou)] + [metrics(m)[:3] for m, _, _ in results.values()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    positions = np.arange(len(names))
    axes[0].bar(positions - 0.2, [100 * s[0] for s in scores], 0.4, label="pixel accuracy, %")
    axes[0].bar(positions + 0.2, [100 * s[1] for s in scores], 0.4, label="mean IoU x 100")
    for i, (acc, mean_iou, _) in enumerate(scores):
        axes[0].text(i - 0.2, 100 * acc + 1, f"{100 * acc:.1f}", ha="center", fontsize=8)
        axes[0].text(i + 0.2, 100 * mean_iou + 1, f"{mean_iou:.3f}", ha="center", fontsize=8)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(names, fontsize=9)
    axes[0].set_ylim(0, 110)
    axes[0].set_title("the constant prediction scores high on one metric and low on the other", fontsize=10)
    axes[0].legend(loc="lower right")
    width = 0.8 / len(names)
    for j, (name, (_, _, per_class)) in enumerate(zip(names, scores)):
        axes[1].bar(np.arange(len(SHAPE_CLASSES)) + (j - (len(names) - 1) / 2) * width, per_class, width, label=name)
    axes[1].set_xticks(np.arange(len(SHAPE_CLASSES)))
    axes[1].set_xticklabels(SHAPE_CLASSES)
    axes[1].set_title("IoU per class", fontsize=10)
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(names))   # below, clear of the bars
    fig.tight_layout()
    fig.savefig(OUT_DIR / "metrics_compare.png", dpi=110)
    plt.close(fig)
    print("  training_curves.png: loss and test mean IoU per epoch for the three models;")
    print("  metrics_compare.png: pixel accuracy against mean IoU, and IoU per class, for every model")
    print("  and the constant prediction")

    # 5. Boundaries
    print(f"\n--- 5. The same models within {BOUNDARY_BAND} pixels of a class boundary ---")
    print(f"  {'model':<22}{'boundary accuracy':>19}{'boundary mean IoU':>19}{'interior accuracy':>19}")
    for name, (matrix, boundary, _) in results.items():
        inside = matrix - boundary
        print(f"  {name:<22}{metrics(boundary)[0]:>19.2%}{metrics(boundary)[1]:>19.3f}"
              f"{metrics(inside)[0]:>19.2%}")
    print("  Away from the boundaries every model is close to perfect: the interior of a shape is")
    print("  decided by colour alone. The skips carry the encoder's full-resolution maps across to")
    print("  the decoder, and that is the information a boundary needs.")
    sample = x_test[:6].to(device)
    with torch.no_grad():
        model_predictions = {name: model(sample).argmax(1).cpu().numpy().astype(np.uint8)
                             for name, (_, _, model) in results.items()}
    row_of = lambda images, name: [labelled(enlarge(image, 3), name) if i == 0 else enlarge(image, 3)
                                   for i, image in enumerate(images)]
    prediction_rows = [row_of([img for img, _ in test[:6]], "image"),
                       row_of([palette_image(label, len(SHAPE_CLASSES)) for label in test_labels[:6]], "truth")]
    for name, predicted in model_predictions.items():
        prediction_rows.append(row_of([palette_image(p, len(SHAPE_CLASSES)) for p in predicted], name))
    cv2.imwrite(str(OUT_DIR / "shapes_predictions.png"), tile_rows(prediction_rows))

    error_rows, scale = [], 4
    for i in range(4):
        band = boundary_band(test_labels[i])
        band_view = test[i][0].copy()
        band_view[band] = (0.4 * band_view[band] + 0.6 * np.array([255, 255, 255])).astype(np.uint8)
        row = [labelled(enlarge(test[i][0], scale), "image") if i == 0 else enlarge(test[i][0], scale),
               labelled(enlarge(palette_image(test_labels[i], len(SHAPE_CLASSES)), scale), "truth") if i == 0
               else enlarge(palette_image(test_labels[i], len(SHAPE_CLASSES)), scale),
               labelled(enlarge(band_view, scale), f"{BOUNDARY_BAND} px band") if i == 0 else enlarge(band_view, scale)]
        for name, predicted in model_predictions.items():
            wrong = predicted[i] != test_labels[i]
            view = np.full((SIZE, SIZE, 3), 40, np.uint8)
            view[band] = (90, 90, 90)
            view[wrong & band] = (0, 0, 255)
            view[wrong & ~band] = (0, 255, 255)
            short = name.replace("UNet ", "").replace(", no resampling", "")
            row.append(labelled(enlarge(view, scale), f"{short}: {int(wrong.sum())} wrong"))
        error_rows.append(row)
    cv2.imwrite(str(OUT_DIR / "boundary_errors.png"), np.vstack([
        tile_rows(error_rows),
        cv2.putText(np.zeros((24, tile_rows(error_rows).shape[1], 3), np.uint8),
                    "error panels: grey = boundary band, red = wrong inside the band, yellow = wrong elsewhere",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)]))
    print("  shapes_predictions.png: six test images, their truth and each model's prediction;")
    print("  boundary_errors.png: four test images with the boundary band and where each model is wrong")

    # 6. Pascal VOC
    print("\n--- 6. Pascal VOC 2012 ---")
    if not args.voc_root:
        print("  pass --voc-root to repeat the comparison on the VOC 2012 segmentation split")
        print(f"\n  images written to {OUT_DIR}")
        return
    splits = load_voc(args.voc_root)
    (voc_train_x, voc_train_y), (voc_val_x, voc_val_y) = splits["train"], splits["val"]
    ignore_share = (voc_train_y == VOC_IGNORE).mean()
    counts = np.bincount(voc_train_y[voc_train_y != VOC_IGNORE].ravel(), minlength=len(VOC_CLASSES))
    print(f"  {len(voc_train_x)} training and {len(voc_val_x)} validation images, resized to "
          f"{VOC_SIZE}x{VOC_SIZE}, {len(VOC_CLASSES)} classes")
    print(f"  label {VOC_IGNORE} marks the band drawn around every object and covers {ignore_share:.2%} of the")
    print("  pixels; it is excluded from the loss and from every count below, because it is not a")
    print("  class the model is asked to predict")
    print(f"  background is {counts[0] / counts.sum():.1%} of the labelled pixels; the largest object class is "
          f"{VOC_CLASSES[int(counts[1:].argmax()) + 1]} at {counts[1:].max() / counts.sum():.1%}")
    voc_data = (torch.from_numpy(voc_train_x).permute(0, 3, 1, 2).float().div(255),
                torch.from_numpy(voc_train_y).long(),
                torch.from_numpy(voc_val_x).permute(0, 3, 1, 2).float().div(255),
                torch.from_numpy(voc_val_y).long())
    constant = np.zeros_like(voc_val_y)
    base_matrix = confusion(constant, voc_val_y, len(VOC_CLASSES), VOC_IGNORE)
    base_accuracy, base_miou, _, _ = metrics(base_matrix)
    print(f"  {'model':<22}{'parameters':>11}{'time':>9}{'pixel accuracy':>15}{'mean IoU':>11}")
    voc_results, voc_histories, voc_rows = {}, {}, []
    voc_row = lambda images, name: [labelled(enlarge(image, 2), name) if i == 0 else enlarge(image, 2)
                                    for i, image in enumerate(images)]
    voc_rows += [voc_row(list(voc_val_x[:6]), "image"),
                 voc_row([palette_image(label, len(VOC_CLASSES)) for label in voc_val_y[:6]], "truth, grey = ignore")]
    for name, model in list(make_models(len(VOC_CLASSES)).items())[:2]:
        model = model.to(device)
        matrix, boundary, elapsed, voc_histories[name] = train_model(model, voc_data, len(VOC_CLASSES), VOC_EPOCHS,
                                                                     device, VOC_IGNORE)
        voc_results[name] = (matrix, boundary)
        report(name, matrix, elapsed, sum(p.numel() for p in model.parameters()))
        with torch.no_grad():
            predicted = model(voc_data[2][:6].to(device)).argmax(1).cpu().numpy().astype(np.uint8)
        voc_rows.append(voc_row([palette_image(p, len(VOC_CLASSES)) for p in predicted], name))
    cv2.imwrite(str(OUT_DIR / "voc_predictions.png"), tile_rows(voc_rows))
    plot_curves(voc_histories, "VOC 2012", OUT_DIR / "voc_training_curves.png")
    print("  voc_predictions.png: six validation images, their truth and both UNets' predictions;")
    print("  voc_training_curves.png: loss and validation mean IoU per epoch")
    print(f"  {'background everywhere':<22}{0:>11}{0:>8}s{base_accuracy:>16.2%}{base_miou:>11.3f}")
    print(f"  {'model':<22}{'boundary accuracy':>19}{'interior accuracy':>19}{'classes ever predicted':>24}")
    for name, (matrix, boundary) in voc_results.items():
        predicted_classes = int((matrix.sum(0) > 0).sum())
        print(f"  {name:<22}{metrics(boundary)[0]:>19.2%}{metrics(matrix - boundary)[0]:>19.2%}"
              f"{predicted_classes:>24}")
    best = max(voc_results, key=lambda n: metrics(voc_results[n][0])[1])
    print(f"  highest mean IoU: {best}")
    print("  Twenty object classes over 1464 images is a small amount of data for this many")
    print("  parameters, and the numbers are far below what a pretrained encoder reaches; what is")
    print("  comparable here is the two architectures against each other and against the baseline.")
    print("  Pixel accuracy can even fall below the baseline while mean IoU rises: predicting an")
    print("  object class costs background pixels, and only one of the two numbers rewards it.")
    print(f"\n  images written to {OUT_DIR}")


if __name__ == "__main__":
    main()
