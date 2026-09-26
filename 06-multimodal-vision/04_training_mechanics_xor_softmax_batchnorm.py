"""Train small networks by hand and account for what a hidden layer, a softmax and a mode switch do.

Demonstrates the mechanics underneath every image classifier:
    1. Show that no single linear boundary separates XOR, and print a network's outputs at random weights.
    2. Train a two-layer network on XOR by matrix backpropagation, across seeds, widths and initial scales.
    3. Derive the softmax cross-entropy gradient, check it numerically, and keep the softmax finite.
    4. Train a 784-256-10 network in numpy on digit images, one mini-batch at a time.
    5. Build a CNN with padding, stride, pooling, batch normalisation and dropout, and trace its shapes.
    6. Train and test the CNN.
    7. Take the trained CNN through training mode and evaluation mode and account for every difference.

Module 06: Multimodal Vision - Training Mechanics.
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")

OUT_BASE = Path(__file__).parent / "outputs" / "training_mechanics"
# A run on real digits and a run on the rendered ones write the same file names, so each
# gets its own folder and neither overwrites the other.
OUT_DIR = OUT_BASE / "synthetic"
SEED = 3407

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float64)
XOR_Y = np.array([[0], [1], [1], [0]], np.float64)
XOR_SEEDS = 100
XOR_EPOCHS = 10000
XOR_LEARNING_RATE = 2.0
XOR_TOLERANCE = 0.1

SYNTHETIC_TRAIN, SYNTHETIC_TEST = 12000, 2000
MLP_HIDDEN, MLP_EPOCHS, MLP_BATCH, MLP_LEARNING_RATE = 256, 5, 64, 0.1
CNN_EPOCHS, CNN_BATCH, CNN_LEARNING_RATE, DROPOUT = 3, 64, 1e-3, 0.3


# 1-2. XOR

def sigmoid(x):
    """Squash into (0, 1); its derivative in terms of its own output is s * (1 - s)."""
    return 1.0 / (1.0 + np.exp(-x))


def train_logistic(seeds, epochs=XOR_EPOCHS, learning_rate=XOR_LEARNING_RATE):
    """A single sigmoid unit on the two inputs, trained on XOR for many seeds at once."""
    rng = np.random.default_rng(SEED)
    w = rng.normal(0, 1, (seeds, 2, 1))
    b = np.zeros((seeds, 1, 1))
    for _ in range(epochs):
        out = sigmoid(np.einsum("nd,sdk->snk", XOR_X, w) + b)
        delta = (out - XOR_Y) * out * (1 - out) * (2 / len(XOR_X))
        w -= learning_rate * np.einsum("nd,snk->sdk", XOR_X, delta)
        b -= learning_rate * delta.sum(axis=1, keepdims=True)
    out = sigmoid(np.einsum("nd,sdk->snk", XOR_X, w) + b)
    return ((out > 0.5) == XOR_Y).mean(axis=(1, 2)), w, b


def init_xor(seeds, hidden, scale, rng):
    """Weights drawn from N(0, scale^2) for every seed, biases at zero."""
    return {"w1": rng.normal(0, scale, (seeds, 2, hidden)), "b1": np.zeros((seeds, 1, hidden)),
            "w2": rng.normal(0, scale, (seeds, hidden, 1)), "b2": np.zeros((seeds, 1, 1))}


def xor_forward(p):
    """Hidden activations and output for all four inputs, for every seed."""
    a1 = sigmoid(np.einsum("nd,sdh->snh", XOR_X, p["w1"]) + p["b1"])
    a2 = sigmoid(a1 @ p["w2"] + p["b2"])
    return a1, a2


def train_xor(seeds, hidden, scale, epochs=XOR_EPOCHS, learning_rate=XOR_LEARNING_RATE):
    """Full-batch gradient descent on mean squared error, backpropagated as matrix products.

    With E = mean((a2 - y)^2), the output layer's error signal is
    d2 = 2 (a2 - y) / N * a2 (1 - a2). The weights into it get a1^T d2. The signal
    carried back to the hidden layer is d1 = (d2 W2^T) * a1 (1 - a1), and the weights
    into the hidden layer get X^T d1. Every seed is a separate network; the leading
    axis of each array holds them side by side.
    """
    rng = np.random.default_rng(SEED)
    p = init_xor(seeds, hidden, scale, rng)
    solved_at = np.full(seeds, -1)
    history = np.empty((epochs, seeds))
    for epoch in range(epochs):
        a1, a2 = xor_forward(p)
        history[epoch] = ((a2 - XOR_Y) ** 2).mean(axis=(1, 2))
        solved = np.abs(a2 - XOR_Y).max(axis=(1, 2)) < XOR_TOLERANCE
        solved_at[(solved_at < 0) & solved] = epoch
        d2 = 2 * (a2 - XOR_Y) / len(XOR_X) * a2 * (1 - a2)
        d1 = (d2 @ p["w2"].transpose(0, 2, 1)) * a1 * (1 - a1)
        p["w2"] -= learning_rate * a1.transpose(0, 2, 1) @ d2
        p["b2"] -= learning_rate * d2.sum(axis=1, keepdims=True)
        p["w1"] -= learning_rate * np.einsum("nd,snh->sdh", XOR_X, d1)
        p["b1"] -= learning_rate * d1.sum(axis=1, keepdims=True)
    return solved_at, history, p


# 3. Softmax and cross-entropy

def softmax_naive(z):
    """exp(z) / sum(exp(z)), exactly as written."""
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def softmax(z):
    """The same function after subtracting the largest logit, which cancels in the ratio."""
    shifted = z - z.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(z, y_onehot):
    """Mean of -log p[true class], computed from logits as logsumexp(z) - z[true class].

    Working in log space needs no epsilon inside a log, so the function differentiated
    numerically below is exactly the one whose gradient is softmax minus one-hot.
    """
    shifted = z - z.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return float(-(y_onehot * log_probs).sum(axis=-1).mean())


# Digit images

FONTS = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
         cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)


def render_digit(digit, rng):
    """A 28x28 digit in a random font, size, stroke, offset, tilt and shear, with a stray stroke.

    The digit is drawn at twice the size and shrunk, so strokes are anti-aliased
    the way a scanned pen stroke is. A random affine map tilts, scales and shears
    it, a short line through a random part of the frame stands in for a stray mark,
    and noise is added last.
    """
    img = np.zeros((56, 56), np.uint8)
    font = FONTS[rng.integers(len(FONTS))]
    scale, thickness = rng.uniform(1.1, 1.9), int(rng.integers(2, 7))
    (tw, th), _ = cv2.getTextSize(str(digit), font, scale, thickness)
    origin = ((56 - tw) // 2 + int(rng.integers(-6, 7)), (56 + th) // 2 + int(rng.integers(-6, 7)))
    cv2.putText(img, str(digit), origin, font, scale, 255, thickness, cv2.LINE_AA)
    matrix = cv2.getRotationMatrix2D((28, 28), float(rng.uniform(-20, 20)), float(rng.uniform(0.8, 1.15)))
    matrix[0, 1] += rng.uniform(-0.2, 0.2)
    img = cv2.warpAffine(img, matrix, (56, 56))
    if rng.random() < 0.5:
        start, end = rng.integers(0, 56, 2), rng.integers(0, 56, 2)
        cv2.line(img, tuple(int(v) for v in start), tuple(int(v) for v in end), int(rng.integers(80, 256)),
                 int(rng.integers(1, 4)), cv2.LINE_AA)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA).astype(np.float32)
    return np.clip(img + rng.normal(0, 25, img.shape), 0, 255).astype(np.uint8)


def synthetic_digits(count, rng):
    """Balanced labels and their rendered images."""
    labels = np.arange(count) % 10
    rng.shuffle(labels)
    return np.stack([render_digit(d, rng) for d in labels]), labels


def read_idx(path):
    """Read an IDX file: a magic number whose last byte is the rank, the sizes, then raw bytes."""
    data = Path(path).read_bytes()
    rank = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(rank))
    return np.frombuffer(data, np.uint8, offset=4 + 4 * rank).reshape(shape)


def load_mnist(root):
    """Training and test images and labels from the four uncompressed IDX files."""
    root = Path(root)
    folder = root / "raw" if (root / "raw").is_dir() else root
    return (read_idx(folder / "train-images-idx3-ubyte"), read_idx(folder / "train-labels-idx1-ubyte"),
            read_idx(folder / "t10k-images-idx3-ubyte"), read_idx(folder / "t10k-labels-idx1-ubyte"))


# 4. A numpy multilayer perceptron

def mlp_forward(p, x):
    """784 -> ReLU(256) -> 10 logits."""
    z1 = x @ p["w1"] + p["b1"]
    a1 = np.maximum(z1, 0)
    return z1, a1, a1 @ p["w2"] + p["b2"]


def train_mlp(x_train, y_train, x_test, y_test, rng):
    """Mini-batch SGD on softmax cross-entropy; the output error is softmax minus one-hot."""
    p = {"w1": rng.normal(0, np.sqrt(2 / 784), (784, MLP_HIDDEN)).astype(np.float32),
         "b1": np.zeros(MLP_HIDDEN, np.float32),
         "w2": rng.normal(0, np.sqrt(2 / MLP_HIDDEN), (MLP_HIDDEN, 10)).astype(np.float32),
         "b2": np.zeros(10, np.float32)}
    eye = np.eye(10, dtype=np.float32)
    history = []
    print(f"  {'epoch':>5}{'train loss':>12}{'train acc':>11}{'test acc':>10}{'time':>8}")
    for epoch in range(1, MLP_EPOCHS + 1):
        started = time.perf_counter()
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), MLP_BATCH):   # the last, shorter batch is trained too
            idx = order[start:start + MLP_BATCH]
            x, y = x_train[idx], eye[y_train[idx]]
            z1, a1, logits = mlp_forward(p, x)
            dz2 = (softmax(logits) - y) / len(idx)
            dz1 = (dz2 @ p["w2"].T) * (z1 > 0)
            for name, grad in (("w2", a1.T @ dz2), ("b2", dz2.sum(0)), ("w1", x.T @ dz1), ("b1", dz1.sum(0))):
                p[name] -= MLP_LEARNING_RATE * grad
        train_logits = mlp_forward(p, x_train)[2]
        test_logits = mlp_forward(p, x_test)[2]
        history.append((cross_entropy(train_logits, eye[y_train]), (train_logits.argmax(1) == y_train).mean(),
                        (test_logits.argmax(1) == y_test).mean()))
        print(f"  {epoch:>5}{history[-1][0]:>12.4f}{history[-1][1]:>11.2%}{history[-1][2]:>10.2%}"
              f"{time.perf_counter() - started:>7.1f}s")
    return p, history


# 5-7. A convolutional network in PyTorch

def build_cnn():
    """Three convolution blocks and two fully connected layers."""
    import torch.nn as nn

    class DigitCNN(nn.Module):
        """conv-BN-ReLU-pool, conv-BN-ReLU-pool, stride-2 conv-BN-ReLU, then FC-ReLU-dropout-FC."""

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(128, 10))

        def forward(self, x):
            return self.classifier(self.features(x))

    return DigitCNN()


def describe(layer):
    """A short name for a layer: its type and the settings that change its output."""
    name = type(layer).__name__
    if name == "Conv2d":
        return (f"Conv2d {layer.in_channels}->{layer.out_channels} k{layer.kernel_size[0]} "
                f"s{layer.stride[0]} p{layer.padding[0]}")
    if name == "BatchNorm2d":
        return f"BatchNorm2d {layer.num_features}"
    if name == "MaxPool2d":
        return f"MaxPool2d k{layer.kernel_size} s{layer.stride}"
    if name == "Linear":
        return f"Linear {layer.in_features}->{layer.out_features}"
    if name == "Dropout":
        return f"Dropout p={layer.p}"
    return name


def output_size(n, kernel, stride, padding):
    """floor((n + 2 * padding - kernel) / stride) + 1."""
    return (n + 2 * padding - kernel) // stride + 1


def evaluate(model, x, y, batch=1000):
    """Accuracy with whatever mode the model is currently in."""
    import torch

    correct = 0
    with torch.no_grad():
        for start in range(0, len(x), batch):
            correct += (model(x[start:start + batch]).argmax(1) == y[start:start + batch]).sum().item()
    return correct / len(x)


def predictions(model, x, batch=1000):
    """Predicted classes with whatever mode the model is currently in."""
    import torch

    with torch.no_grad():
        return torch.cat([model(x[s:s + batch]).argmax(1) for s in range(0, len(x), batch)])


# Main

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mnist-root", help="MNIST folder holding the uncompressed IDX files (or raw/)")
    args = parser.parse_args()
    global OUT_DIR
    OUT_DIR = OUT_BASE / ("mnist" if args.mnist_root else "synthetic")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. XOR is not linearly separable
    print("--- 1. XOR, a straight line, and a network at random weights ---")
    linear_accuracy, logistic_w, logistic_b = train_logistic(XOR_SEEDS)
    print(f"  one sigmoid unit, {XOR_SEEDS} seeds, {XOR_EPOCHS} epochs: best accuracy {linear_accuracy.max():.0%}, "
          f"worst {linear_accuracy.min():.0%}")
    print("  A single unit outputs sigmoid(w1 x1 + w2 x2 + b), which is above 0.5 on one side")
    print("  of a straight line. (0,1) and (1,0) must be on the positive side and (0,0) and")
    print("  (1,1) on the other. A half-plane holds the midpoint of any two points it holds,")
    print("  and both pairs have the same midpoint, (0.5, 0.5), which would have to lie on both")
    print("  sides at once. So at most three of the four inputs can be right.")
    p = init_xor(1, 2, 1.0, np.random.default_rng(SEED))
    _, out = xor_forward(p)
    print("  two hidden units, weights drawn from N(0, 1), before any training:")
    for x, target, value in zip(XOR_X, XOR_Y[:, 0], out[0, :, 0]):
        print(f"    input {x.astype(int).tolist()}  output {value:.4f}  target {int(target)}")

    # 2. Backpropagation on XOR
    print("\n--- 2. Backpropagation on XOR, across seeds, widths and initial scales ---")
    print(f"  learning rate {XOR_LEARNING_RATE}, full batch, solved when every output is within "
          f"{XOR_TOLERANCE} of its target")
    print(f"  {'hidden':>6}{'init sd':>9}{'solved':>9}{'median epochs to solve':>24}")
    curves = {}
    for hidden in (2, 4, 8):
        for scale in (0.1, 1.0):
            solved_at, history, trained = train_xor(XOR_SEEDS, hidden, scale)
            done = solved_at[solved_at >= 0]
            median = f"{int(np.median(done))}" if len(done) else "-"
            print(f"  {hidden:>6}{scale:>9.1f}{len(done) / XOR_SEEDS:>9.0%}{median:>24}")
            if hidden == 2 and scale == 1.0:
                curves = {"solved": history[:, np.flatnonzero(solved_at >= 0)[:3]],
                          "unsolved": history[:, np.flatnonzero(solved_at < 0)[:3]]}
                stuck = history[-1, solved_at < 0]
                two_unit = (solved_at, trained)
    print("  Two hidden units are the fewest that can represent XOR, and some starting points")
    print("  lead gradient descent to a flat region where both units compute nearly the same")
    print(f"  thing; those runs end with a loss of about {np.median(stuck):.3f}. Extra units give")
    print("  more than one way to split the plane, and almost every start finds one. Weights")
    print("  that start near zero make the two hidden units almost identical, and the")
    print("  gradient needs many epochs to push them apart.")
    plt.figure(figsize=(8, 4.5))
    for label, block in curves.items():
        for column in range(block.shape[1]):
            plt.plot(block[:, column], color="tab:green" if label == "solved" else "tab:red",
                     label=label if column == 0 else None)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("mean squared error")
    plt.title("XOR, two hidden units, six seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "xor_loss.png", dpi=120)
    plt.close()

    grid_x, grid_y = np.meshgrid(np.linspace(-0.5, 1.5, 201), np.linspace(-0.5, 1.5, 201))
    grid = np.stack([grid_x.ravel(), grid_y.ravel()], 1)
    unit_output = lambda w, b: sigmoid(grid @ w + b).reshape(grid_x.shape)

    def network_output(params, seed):
        """The two-layer network's output over the plane, for one seed."""
        a1 = sigmoid(grid @ params["w1"][seed] + params["b1"][seed])
        return sigmoid(a1 @ params["w2"][seed] + params["b2"][seed]).reshape(grid_x.shape)

    solved_at, trained = two_unit
    xor_accuracy = ((xor_forward(trained)[1] > 0.5) == XOR_Y).mean(axis=(1, 2))
    good, bad = int(np.flatnonzero(solved_at >= 0)[0]), int(np.flatnonzero(solved_at < 0)[0])
    best_linear = int(np.argmax(linear_accuracy))
    panels = [(f"one unit, best of {XOR_SEEDS} seeds: {linear_accuracy[best_linear]:.0%}",
               unit_output(logistic_w[best_linear], logistic_b[best_linear])),
              (f"random weights: output {network_output(p, 0).min():.2f}-{network_output(p, 0).max():.2f} everywhere",
               network_output(p, 0)),
              (f"two hidden units, trained: {xor_accuracy[good]:.0%}", network_output(trained, good)),
              (f"two hidden units, stuck: {xor_accuracy[bad]:.0%}", network_output(trained, bad))]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (title, surface) in zip(axes, panels):
        ax.contourf(grid_x, grid_y, surface, levels=np.linspace(0, 1, 11), cmap="RdBu_r")
        ax.contour(grid_x, grid_y, surface, levels=[0.5], colors="k", linewidths=1.5)
        ax.scatter(XOR_X[:, 0], XOR_X[:, 1], c=["tab:blue" if t == 0 else "tab:red" for t in XOR_Y[:, 0]],
                   s=120, edgecolors="k", zorder=3)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
    fig.suptitle("output over the input plane (red high, blue low), black line at 0.5; red points want 1, blue 0")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "xor_boundaries.png", dpi=110)
    plt.close(fig)
    print("  xor_boundaries.png: the best single unit, the network at random weights, a trained seed")
    print("  that solves XOR and one that gets stuck; xor_loss.png: loss curves of three of each")

    # 3. Softmax and cross-entropy
    print("\n--- 3. Softmax cross-entropy: the gradient and the overflow ---")
    z = rng.normal(0, 2, (4, 10))
    y = np.eye(10)[rng.integers(0, 10, 4)]
    analytic = (softmax(z) - y) / len(z)
    numeric = np.zeros_like(z)
    for index in np.ndindex(z.shape):
        up, down = z.copy(), z.copy()
        up[index] += 1e-6
        down[index] -= 1e-6
        numeric[index] = (cross_entropy(up, y) - cross_entropy(down, y)) / 2e-6
    print(f"  4 samples x 10 classes: largest gap between (softmax - onehot) / N and a central "
          f"difference {np.abs(analytic - numeric).max():.2e}")
    print("  With p = softmax(z) and loss -log p_k, d(-log p_k)/dz_j = p_j - [j = k]: the exponent")
    print("  in the softmax and the log in the loss cancel, leaving no derivative of either.")
    big = np.array([[1000.0, 1001.0, 1002.0]])
    with np.errstate(over="ignore", invalid="ignore"):
        naive = softmax_naive(big)
    print(f"  logits {big[0].tolist()}: as written {naive[0].tolist()}, "
          f"after subtracting the maximum {np.round(softmax(big)[0], 4).tolist()}")
    print("  exp(1000) overflows a double. Subtracting any constant from every logit leaves the")
    print("  ratios unchanged, and subtracting the largest keeps every exponent at or below zero.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    classes = np.arange(10)
    axes[0].bar(classes - 0.2, analytic[0], 0.4, label="(softmax - one-hot) / N")
    axes[0].bar(classes + 0.2, numeric[0], 0.4, label="central difference")
    axes[0].axhline(0, color="k", linewidth=0.8)
    axes[0].set_xticks(classes)
    axes[0].set_xlabel(f"class (true class {int(y[0].argmax())})")
    axes[0].set_title(f"gradient for the first sample; largest gap over all {z.size} entries "
                      f"{np.abs(analytic - numeric).max():.1e}", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[1].bar(["1000", "1001", "1002"], softmax(big)[0], color="tab:green")
    for i, value in enumerate(softmax(big)[0]):
        axes[1].text(i, value + 0.01, f"{value:.4f}", ha="center")
    axes[1].set_title("logits 1000, 1001, 1002: as written every probability is nan;\n"
                      "after subtracting the largest logit", fontsize=10)
    axes[1].set_ylim(0, 0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "softmax_gradient.png", dpi=110)
    plt.close(fig)
    print("  softmax_gradient.png: the two gradients side by side, and the stable softmax of large logits")

    # 4. Numpy MLP on digits
    print("\n--- 4. A 784-256-10 network in numpy ---")
    if args.mnist_root:
        train_images, train_labels, test_images, test_labels = load_mnist(args.mnist_root)
        source = f"MNIST: {len(train_images)} training and {len(test_images)} test images"
    else:
        train_images, train_labels = synthetic_digits(SYNTHETIC_TRAIN, rng)
        test_images, test_labels = synthetic_digits(SYNTHETIC_TEST, rng)
        source = (f"rendered digits: {len(train_images)} training and {len(test_images)} test images "
                  f"(pass --mnist-root to use MNIST)")
    fig, axes = plt.subplots(2, 10, figsize=(12, 3.8))
    for ax, image, label in zip(axes.ravel(), train_images[:20], train_labels[:20]):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"label {int(label)}", fontsize=9)
        ax.axis("off")
    fig.suptitle("the first 20 training images: " + ("MNIST" if args.mnist_root else "rendered digits"))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "digits.png", dpi=110)
    plt.close(fig)
    mean, std = train_images.mean() / 255, train_images.std() / 255
    x_train = ((train_images.reshape(len(train_images), -1) / 255 - mean) / std).astype(np.float32)
    x_test = ((test_images.reshape(len(test_images), -1) / 255 - mean) / std).astype(np.float32)
    y_train, y_test = train_labels.astype(np.int64), test_labels.astype(np.int64)
    print(f"  {source}; pixels standardised with the training mean {mean:.4f} and sd {std:.4f}")
    print(f"  He initialisation, batch {MLP_BATCH}, learning rate {MLP_LEARNING_RATE}")
    mlp, mlp_history = train_mlp(x_train, y_train, x_test, y_test, rng)
    parameters = 784 * MLP_HIDDEN + MLP_HIDDEN + MLP_HIDDEN * 10 + 10
    test_error = 1 - (mlp_forward(mlp, x_test)[2].argmax(1) == y_test).mean()
    print(f"  {parameters} parameters, all updated from the gradient written out above; "
          f"test error rate {test_error:.2%}")

    # 5. CNN architecture
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    print("\n--- 5. A convolutional network and the shape at every layer ---")
    torch.manual_seed(SEED)
    # cuDNN picks a convolution algorithm per shape and some of them sum in a
    # non-deterministic order, so two runs of the same seed differ in the last digits
    # and then in the accuracy. These two lines are what make the numbers repeatable.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_cnn().to(device)
    shapes = []
    hooks = [layer.register_forward_hook(lambda m, i, o, name=name: shapes.append((name, tuple(o.shape[1:]))))
             for name, layer in list(model.features.named_children()) + list(model.classifier.named_children())]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, 28, 28, device=device))
    for hook in hooks:
        hook.remove()
    layers = list(model.features) + list(model.classifier)
    print(f"  {'layer':<28}{'output':>16}{'parameters':>12}")
    for (index, shape), layer in zip(shapes, layers):
        count = sum(p.numel() for p in layer.parameters())
        print(f"  {describe(layer):<28}{str(shape):>16}{count:>12}")
    print(f"  total {sum(p.numel() for p in model.parameters())} parameters")
    print(f"  output size = floor((n + 2p - k) / s) + 1: conv 3x3 p1 s1 keeps 28 -> "
          f"{output_size(28, 3, 1, 1)}, pool 2 s2 halves it to {output_size(28, 2, 2, 0)}, "
          f"and the stride-2 conv takes 7 -> {output_size(7, 3, 2, 1)}")

    # 6. Train and test
    print("\n--- 6. Training and testing the CNN ---")
    xt = torch.from_numpy(x_train.reshape(-1, 1, 28, 28)).to(device)
    yt = torch.from_numpy(y_train).to(device)
    xv = torch.from_numpy(x_test.reshape(-1, 1, 28, 28)).to(device)
    yv = torch.from_numpy(y_test).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=CNN_LEARNING_RATE)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    print(f"  Adam, learning rate {CNN_LEARNING_RATE}, batch {CNN_BATCH}, dropout {DROPOUT}, on {device}")
    print(f"  {'epoch':>5}{'running loss':>14}{'running acc':>13}{'test acc':>10}{'time':>8}")
    cnn_history = []
    for epoch in range(1, CNN_EPOCHS + 1):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(xt), generator=generator).to(device)
        total_loss, correct = 0.0, 0
        for start in range(0, len(order), CNN_BATCH):
            idx = order[start:start + CNN_BATCH]
            logits = model(xt[idx])
            loss = F.cross_entropy(logits, yt[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yt[idx]).sum().item()
        model.eval()
        cnn_history.append((total_loss / len(xt), correct / len(xt), evaluate(model, xv, yv)))
        print(f"  {epoch:>5}{cnn_history[-1][0]:>14.4f}{cnn_history[-1][1]:>13.2%}"
              f"{cnn_history[-1][2]:>10.2%}{time.perf_counter() - started:>7.1f}s")
    print("  The running figures are averaged over the epoch while the weights are still moving,")
    print("  in training mode; the test figure is measured once, at the end, in evaluation mode.")
    print("  Test accuracy is printed every epoch in both networks only to show the curve: no")
    print("  epoch, setting or stopping point is chosen from it. A project that tunes anything")
    print("  would watch a validation split carved from the training data and test once.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for history, name, colour in ((mlp_history, "numpy MLP", "tab:orange"), (cnn_history, "CNN", "tab:blue")):
        epochs = np.arange(1, len(history) + 1)
        axes[0].plot(epochs, [h[0] for h in history], "o-", color=colour, label=name)
        axes[1].plot(epochs, [100 * h[2] for h in history], "o-", color=colour, label=name)
    axes[0].set_title("training loss (MLP: whole set after the epoch; CNN: running mean)", fontsize=10)
    axes[1].set_title("test accuracy after each epoch, %", fontsize=10)
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.set_xticks(np.arange(1, max(MLP_EPOCHS, CNN_EPOCHS) + 1))
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_curves.png", dpi=110)
    plt.close(fig)

    mlp_predicted = mlp_forward(mlp, x_test)[2].argmax(1)
    cnn_predicted = predictions(model, xv).cpu().numpy()
    fig, axes = plt.subplots(2, 10, figsize=(12, 3.2))
    for row, (name, predicted) in enumerate((("MLP", mlp_predicted), ("CNN", cnn_predicted))):
        wrong = np.flatnonzero(predicted != y_test)
        for column, ax in enumerate(axes[row]):
            ax.axis("off")
            if column < len(wrong):
                i = wrong[column]
                ax.imshow(test_images[i], cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"{name}: {y_test[i]} as {predicted[i]}", fontsize=8)
    fig.suptitle(f"first wrong test images, true class as predicted class "
                 f"(MLP {int((mlp_predicted != y_test).sum())} wrong, CNN {int((cnn_predicted != y_test).sum())} "
                 f"wrong of {len(y_test)})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "misclassified.png", dpi=110)
    plt.close(fig)

    sample_image = xv[:1]
    stages = [("input", sample_image)]
    with torch.no_grad():
        for name, end in (("after block 1", 4), ("after block 2", 8), ("after block 3", 11)):
            stages.append((name, model.features[:end](sample_image)))
    fig, axes = plt.subplots(len(stages), 8, figsize=(12, 6.5))
    for row, (name, maps) in enumerate(stages):
        maps = maps[0].cpu().numpy()
        for column, ax in enumerate(axes[row]):
            ax.axis("off")
            if column < len(maps):
                ax.imshow(maps[column], cmap="gray" if row == 0 else "viridis")
                if column == 0:
                    ax.set_title(f"{name} {tuple(maps.shape)}", fontsize=9, loc="left")
    fig.suptitle(f"one test image (label {int(y_test[0])}) and the first 8 channels after each convolution block")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cnn_feature_maps.png", dpi=110)
    plt.close(fig)
    print("  training_curves.png: loss and test accuracy per epoch for both networks;")
    print("  misclassified.png: the first wrong test images of each; cnn_feature_maps.png: one image")
    print("  through the three convolution blocks, 28 -> 14 -> 7 -> 4 pixels per side")

    # 7. Modes
    print("\n--- 7. Training mode against evaluation mode, on the trained network ---")
    subset = slice(0, 5000)
    probe = copy.deepcopy(model)
    probe.eval()
    eval_train = evaluate(probe, xt[subset], yt[subset])
    eval_pred = predictions(probe, xv)
    probe.train()
    with torch.no_grad():
        train_train = evaluate(probe, xt[subset], yt[subset], batch=256)
        train_pred = predictions(probe, xv, batch=256)
    print(f"  same weights, first {subset.stop} training images: evaluation mode {eval_train:.2%}, "
          f"training mode {train_train:.2%}")
    print(f"  test predictions that change with the mode: {(eval_pred != train_pred).sum().item()} of {len(xv)}")
    print("  torch.no_grad() stops gradients being recorded, not the mode: inside it, training-mode")
    print("  dropout still drops and BatchNorm still updates its running statistics, which is why")
    print("  this probe runs on a copy.")

    dropout = nn.Dropout(DROPOUT).train()
    ones = torch.ones(100000, device=device)
    dropped = dropout(ones)
    print(f"  dropout p={DROPOUT} in training mode on a vector of ones: {(dropped == 0).float().mean().item():.3f} "
          f"zeroed, survivors scaled to {dropped[dropped > 0][0].item():.4f} = 1 / (1 - p), mean "
          f"{dropped.mean().item():.4f}; in evaluation mode it is the identity")

    bn = model.features[1]
    with torch.no_grad():
        conv_out = model.features[0](xt[:256])
        stats_eval = copy.deepcopy(bn).eval()(conv_out)
        by_hand_eval = ((conv_out - bn.running_mean[None, :, None, None])
                        / torch.sqrt(bn.running_var[None, :, None, None] + bn.eps)
                        * bn.weight[None, :, None, None] + bn.bias[None, :, None, None])
        batch_bn = copy.deepcopy(bn).train()
        stats_train = batch_bn(conv_out)
        mu = conv_out.mean(dim=(0, 2, 3))
        var = conv_out.var(dim=(0, 2, 3), unbiased=False)
        by_hand_train = ((conv_out - mu[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + bn.eps)
                         * bn.weight[None, :, None, None] + bn.bias[None, :, None, None])
        n = conv_out.numel() / conv_out.shape[1]
        expected_running = (1 - bn.momentum) * bn.running_var + bn.momentum * var * n / (n - 1)
    print(f"  first BatchNorm, evaluation mode: largest gap to (x - running mean) / sqrt(running var + eps) "
          f"* gamma + beta {(stats_eval - by_hand_eval).abs().max().item():.2e}")
    print(f"  first BatchNorm, training mode: largest gap to the same formula with this batch's mean and "
          f"variance {(stats_train - by_hand_train).abs().max().item():.2e}")
    print(f"  running variance after that one training-mode batch, against 0.9 * old + 0.1 * unbiased batch "
          f"variance: largest gap {(batch_bn.running_var - expected_running).abs().max().item():.2e}")
    print(f"  batch mean against running mean, per channel: largest gap "
          f"{(mu - bn.running_mean).abs().max().item():.4f}")

    def one_at_a_time(batchnorm_training, dropout_training):
        """Accuracy on the first 1000 test images fed singly, with each layer type in the chosen mode."""
        single = copy.deepcopy(model)
        for module in single.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train(batchnorm_training)
            elif isinstance(module, nn.Dropout):
                module.train(dropout_training)
        with torch.no_grad():
            return sum((single(xv[i:i + 1]).argmax(1) == yv[i:i + 1]).sum().item() for i in range(1000)) / 1000

    print("  one image at a time, first 1000 test images, each layer type set separately:")
    print(f"  {'BatchNorm':<11}{'Dropout':<11}{'accuracy':>9}")
    single_results = []
    for bn_mode, dropout_mode in ((True, True), (True, False), (False, True), (False, False)):
        single_results.append((bn_mode, dropout_mode, one_at_a_time(bn_mode, dropout_mode)))
        print(f"  {'training' if bn_mode else 'evaluation':<11}{'training' if dropout_mode else 'evaluation':<11}"
              f"{single_results[-1][2]:>9.2%}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    image = axes[0].imshow(dropped[:400].reshape(20, 20).cpu().numpy(), cmap="magma", vmin=0)
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    axes[0].set_title(f"dropout p={DROPOUT} on 400 ones: {(dropped[:400] == 0).float().mean().item():.0%} zero,\n"
                      f"the rest {1 / (1 - DROPOUT):.3f}", fontsize=10)
    axes[0].axis("off")
    axes[1].scatter(bn.running_mean.cpu().numpy(), mu.cpu().numpy(), color="tab:purple")
    low, high = float(min(bn.running_mean.min(), mu.min())), float(max(bn.running_mean.max(), mu.max()))
    axes[1].plot([low, high], [low, high], "k--", linewidth=1)
    axes[1].set_xlabel("running mean (used in evaluation mode)")
    axes[1].set_ylabel("mean of one training batch")
    axes[1].set_title("first BatchNorm, one point per channel", fontsize=10)
    names = [f"BN {'train' if b else 'eval'}\nDropout {'train' if d else 'eval'}" for b, d, _ in single_results]
    values = [100 * acc for _, _, acc in single_results]
    axes[2].bar(names, values, color=["tab:red", "tab:orange", "tab:olive", "tab:green"])
    for i, value in enumerate(values):
        axes[2].text(i, value + 0.3, f"{value:.1f}%", ha="center")
    axes[2].set_ylim(min(values) - 5, 100)
    axes[2].set_title("test images fed one at a time, first 1000", fontsize=10)
    fig.suptitle(f"same weights in both modes: {(eval_pred != train_pred).sum().item()} of {len(xv)} test "
                 f"predictions change")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mode_effects.png", dpi=110)
    plt.close(fig)
    print("  mode_effects.png: a dropout mask, batch against running means, and the four single-image")
    print("  mode combinations")
    print("  With one image, BatchNorm normalises each channel by that image's own spatial mean")
    print("  and variance, erasing how bright a feature map is overall, which is part of the")
    print("  evidence; running statistics collected over training carry that information instead.")
    print("  The rows separate the two layers: switching only BatchNorm or only Dropout shows how")
    print("  much of the drop each one accounts for.")
    print(f"\n  images written to {OUT_DIR}")


if __name__ == "__main__":
    main()
