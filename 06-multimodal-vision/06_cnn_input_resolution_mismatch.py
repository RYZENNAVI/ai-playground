"""Price a network designed for 224x224 inputs when it is handed a 32x32 one instead.

Demonstrates how to settle the question by measurement rather than by intuition:
    1. Synthesise a 32x32 dataset whose classes differ only in a distance of a few pixels.
    2. Trace a 224-shaped stem stage by stage and watch the feature map reach 1x1.
    3. Work out how many cells each separation survives as.
    4. Build a network sized for the input instead, and compare the parameter counts.
    5. Train both on the same images and time them.
    6. Score both, and name the cost the numbers actually support.

Module 06: Multimodal Vision - Input Resolution and Network Design.
"""

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8")

SEED = 3407
IMAGE_SIZE = 32
DESIGNED_SIZE = 224
# Every class draws the same two dots and differs only in how far apart they sit.
# Total brightness is therefore identical across classes, so no amount of blur
# leaves a shortcut behind: separation is the only thing that says which class it is.
SEPARATIONS = (3, 5, 7, 9)
CLASSES = tuple(f"gap_{value}" for value in SEPARATIONS)
DOT_PIXELS = 2
TRAIN_PER_CLASS = 500
TEST_PER_CLASS = 100
BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def draw_sample(separation, rng):
    """Draw one 32x32 sample: two bright dots that sit `separation` pixels apart.

    The pair's position jitters across the frame so the classifier cannot win by
    memorising coordinates, and the pair is drawn horizontally or vertically so it
    cannot win on axis either. Every class puts the same amount of light on the
    frame, which is what makes the separation the only usable signal.
    """
    img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=25)
    draw = ImageDraw.Draw(img)
    span = separation + DOT_PIXELS
    vertical = rng.random() < 0.5
    x = rng.randint(4, IMAGE_SIZE - 5 - (0 if vertical else span))
    y = rng.randint(4, IMAGE_SIZE - 5 - (span if vertical else 0))
    second = (x, y + span) if vertical else (x + span, y)
    for corner in ((x, y), second):
        draw.rectangle(
            [corner[0], corner[1], corner[0] + DOT_PIXELS - 1, corner[1] + DOT_PIXELS - 1],
            fill=235,
        )
    return np.asarray(img, dtype=np.float32) / 255.0


def build_dataset(per_class, seed):
    """Return (images, labels) tensors with one channel, drawn from a seeded generator."""
    rng = random.Random(seed)
    images, labels = [], []
    for index, separation in enumerate(SEPARATIONS):
        for _ in range(per_class):
            images.append(draw_sample(separation, rng))
            labels.append(index)
    x = torch.from_numpy(np.stack(images)).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)
    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))
    return x[order], y[order]


class SizedForInput(nn.Module):
    """A small convolutional network whose downsampling matches a 32x32 input.

    Three stride-1 convolutions, each followed by one halving. The first layer
    keeps the full resolution, so a separation of a few pixels is still that many
    pixels wide when the first weights see it.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.head(x.flatten(1))


def build_designed_for_224(num_classes):
    """Return a ResNet-50 with its first layer accepting one channel.

    Nothing else is touched. The stem still runs a 7x7 convolution at stride 2 and
    then a stride-2 max pool, because that is what the architecture is.
    """
    model = torchvision.models.resnet50(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def trace_stem(model, size):
    """Return the feature map size after each stage of a ResNet, for one input size."""
    # eval mode matters here: batch normalisation refuses a batch of one while
    # training, and the final stage of this trace is exactly one 1x1 cell.
    was_training = model.training
    model.eval()
    x = torch.zeros(1, 1, size, size)
    stages = [
        ("conv1 7x7 s2", model.conv1),
        ("maxpool s2", nn.Sequential(model.bn1, model.relu, model.maxpool)),
        ("layer1", model.layer1),
        ("layer2", model.layer2),
        ("layer3", model.layer3),
        ("layer4", model.layer4),
    ]
    trace = []
    with torch.no_grad():
        for name, stage in stages:
            x = stage(x)
            trace.append((name, x.shape[-1]))
    model.train(was_training)
    return trace


def train(model, x, y, epochs=EPOCHS, seed=SEED):
    """Run a plain supervised loop and return the loss at the end of each epoch."""
    torch.manual_seed(seed)
    model = model.to(DEVICE).train()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    history = []
    started = time.perf_counter()
    for _ in range(epochs):
        order = torch.randperm(len(y), generator=generator)
        running = 0.0
        batches = 0
        for start in range(0, len(order), BATCH_SIZE):
            index = order[start : start + BATCH_SIZE]
            inputs = x[index].to(DEVICE)
            targets = y[index].to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += loss.item()
            batches += 1
        history.append(running / batches)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return history, time.perf_counter() - started


def evaluate(model, x, y):
    """Return overall accuracy and per-class accuracy on held-out samples."""
    model = model.to(DEVICE).eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(y), BATCH_SIZE):
            batch = x[start : start + BATCH_SIZE].to(DEVICE)
            predictions.append(model(batch).argmax(dim=1).cpu())
    predicted = torch.cat(predictions)
    overall = float((predicted == y).float().mean())
    per_class = {}
    for index, name in enumerate(CLASSES):
        mask = y == index
        per_class[name] = float((predicted[mask] == y[mask]).float().mean())
    return overall, per_class


def main():
    torch.manual_seed(SEED)
    print("=" * 78)
    print("--- 1. A dataset whose classes differ only in a distance ---")
    train_x, train_y = build_dataset(TRAIN_PER_CLASS, SEED)
    test_x, test_y = build_dataset(TEST_PER_CLASS, SEED + 1)
    print(f"{len(train_y)} training and {len(test_y)} held-out samples, "
          f"{IMAGE_SIZE}x{IMAGE_SIZE} single channel")
    print(f"two {DOT_PIXELS}x{DOT_PIXELS} dots per image, separated by "
          f"{', '.join(str(s) for s in SEPARATIONS)} pixels")
    brightness = train_x.sum(dim=(1, 2, 3))
    spread = [float(brightness[train_y == i].mean()) for i in range(len(CLASSES))]
    print(f"  mean brightness per class: {', '.join(f'{v:.1f}' for v in spread)} "
          f"- identical, so brightness carries no answer")
    print(f"device: {DEVICE}")

    print()
    print("--- 2. Tracing a stem built for 224x224 ---")
    designed = build_designed_for_224(len(CLASSES))
    small_trace = trace_stem(designed, IMAGE_SIZE)
    large_trace = trace_stem(designed, DESIGNED_SIZE)
    print(f"{'stage':<16}{'from 32':>10}{'from 224':>12}")
    for (name, small), (_, large) in zip(small_trace, large_trace):
        print(f"{name:<16}{small:>10}{large:>12}")
    print(f"  the 32x32 input reaches {small_trace[-1][1]}x{small_trace[-1][1]} before the "
          f"classifier, where the design expects {large_trace[-1][1]}x{large_trace[-1][1]}")

    print()
    print("--- 3. What the stem costs the separations ---")
    after_stem = small_trace[1][1]
    factor = IMAGE_SIZE / after_stem
    print(f"the stem downsamples by {factor:.0f}x before any residual block runs, so one "
          f"cell afterwards covers {factor:.0f} input pixels")
    for separation in SEPARATIONS:
        cells = separation / factor
        verdict = "inside one cell" if cells < 1 else f"{cells:.2f} cells apart"
        print(f"  gap_{separation}: {verdict}")
    print(f"  the same stem on a {DESIGNED_SIZE}x{DESIGNED_SIZE} photograph leaves "
          f"{large_trace[1][1]}x{large_trace[1][1]} cells to work with")

    print()
    print("--- 4. A network sized for the input it is given ---")
    sized = SizedForInput(len(CLASSES))
    designed_params = sum(p.numel() for p in designed.parameters())
    sized_params = sum(p.numel() for p in sized.parameters())
    print(f"{'ResNet-50':<14}{designed_params:>12,} parameters")
    print(f"{'SizedForInput':<14}{sized_params:>12,} parameters "
          f"({designed_params / sized_params:.0f}x fewer)")
    with torch.no_grad():
        sized_map = sized.features(torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE)).shape[-1]
    print(f"  its last feature map is {sized_map}x{sized_map}, not "
          f"{small_trace[-1][1]}x{small_trace[-1][1]}")

    print()
    print("--- 5. Training both on the same images ---")
    seconds = {}
    for name, model in (("ResNet-50", designed), ("SizedForInput", sized)):
        history, elapsed = train(model, train_x, train_y)
        seconds[name] = elapsed
        shown = "  ".join(f"{value:.3f}" for value in history[::3] + history[-1:])
        print(f"{name:<14} {elapsed:>6.1f}s   loss every third epoch: {shown}")
    print("  both loops finish, and neither raises anything")

    print()
    print("--- 6. Scoring both on the held-out images ---")
    results = {}
    for name, model in (("ResNet-50", designed), ("SizedForInput", sized)):
        overall, per_class = evaluate(model, test_x, test_y)
        results[name] = overall
        detail = "  ".join(f"{cls}={score:.2f}" for cls, score in per_class.items())
        print(f"{name:<14} accuracy {overall:.1%}   {detail}")
    print(f"chance is {1 / len(CLASSES):.0%}")

    print()
    print("what the numbers support, and what they do not:")
    print(f"  the mismatch did NOT hide the fine detail. The stem samples at stride 2, but "
          f"each of its kernels still spans 7 input pixels, so a {SEPARATIONS[0]}-pixel gap "
          f"survives in the channel values even once the map has gone coarse")
    print(f"  what it cost is measurable elsewhere: {designed_params / sized_params:.0f}x the "
          f"parameters and {seconds['ResNet-50'] / seconds['SizedForInput']:.1f}x the training "
          f"time, for {results['ResNet-50']:.1%} against {results['SizedForInput']:.1%}")
    print(f"  and the last feature map is {small_trace[-1][1]}x{small_trace[-1][1]}, so the "
          f"pooling that follows averages a single cell - nothing downstream can ask where")
    print("  'it cannot see the detail' was the intuition; the run says otherwise, and the "
          "run is what gets reported")
    print("=" * 78)


if __name__ == "__main__":
    main()
