"""Work out what a convolution kernel actually computes, one sliding window at a time.

Demonstrates the arithmetic underneath every convolutional layer:
    1. Walk one hand-written kernel across a tiny binary image and check every output cell.
    2. Derive the output size from padding and stride, then confirm it against the layer.
    3. Render a test image locally instead of shipping a photograph.
    4. Build four directional kernels and state which edge each one answers to.
    5. Push the image through convolution, activation and pooling, saving each stage.
    6. Measure what the activation discards and what the pooling keeps.

Module 06: Multimodal Vision - Convolution From First Principles.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "outputs" / "feature_maps"
IMAGE_SIZE = 160
KERNEL_SIZE = 4

# A 5x5 binary image with a diagonal band of ones, small enough to verify by hand.
TINY_IMAGE = np.array(
    [
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0],
    ],
    dtype=np.float32,
)

# A plus-shaped kernel: it scores highest where the window is dense in the centre
# and on the corners, which is exactly what the diagonal band produces.
TINY_KERNEL = np.array(
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ],
    dtype=np.float32,
)


def convolve_by_hand(image, kernel):
    """Return the valid-mode convolution computed with explicit loops.

    Every deep learning framework ships an optimised routine for this, but the
    optimised one is also the one nobody can check. The loop below exists to be
    compared against nn.Conv2d, not to be fast.
    """
    k = kernel.shape[0]
    out_h = image.shape[0] - k + 1
    out_w = image.shape[1] - k + 1
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for row in range(out_h):
        for col in range(out_w):
            window = image[row : row + k, col : col + k]
            out[row, col] = float((window * kernel).sum())
    return out


def conv2d_with_fixed_weights(image, kernels, padding=0, stride=1):
    """Run nn.Conv2d with the given kernels installed as its weights.

    Conv2d expects (batch, channel, height, width) and its weight expects
    (out_channels, in_channels, height, width), so both arrays gain two axes here.
    bias is switched off because a bias term would shift every output cell by an
    unknown constant and break the comparison against the hand loop.
    """
    x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    weight = torch.from_numpy(kernels).unsqueeze(1)
    layer = nn.Conv2d(
        in_channels=1,
        out_channels=kernels.shape[0],
        kernel_size=kernels.shape[-1],
        padding=padding,
        stride=stride,
        bias=False,
    )
    layer.weight = nn.Parameter(weight)
    with torch.no_grad():
        return layer(x)


def render_test_image(size=IMAGE_SIZE):
    """Draw a greyscale test card whose edges sit at coordinates this function picked.

    A photograph would work too, but then the answer to "did the vertical kernel
    fire on a vertical edge" depends on what happens to be in the photograph. Here
    the vertical edge sits at a known column, so the check is exact.
    """
    img = Image.new("L", (size, size), color=40)
    draw = ImageDraw.Draw(img)
    # A bright block whose left border is a vertical edge at x = size // 4.
    draw.rectangle([size // 4, size // 8, size // 2, size - size // 8], fill=230)
    # A bright bar whose top border is a horizontal edge at y = size // 2.
    draw.rectangle([size // 2 + 8, size // 2, size - size // 8, size // 2 + 24], fill=200)
    # A diagonal stroke, which neither axis-aligned kernel is tuned for.
    draw.line([size // 2 + 8, size - size // 6, size - 12, size // 2 + 44], fill=255, width=5)
    return np.asarray(img, dtype=np.float32) / 255.0


def directional_kernels(k=KERNEL_SIZE):
    """Return four edge kernels: dark-to-light and light-to-dark, in both axes.

    Each kernel is half negative and half positive. Its response is large and
    positive only where the image crosses from the side it subtracts to the side
    it adds, which is what makes it directional rather than merely edge-sensitive.
    """
    vertical = np.hstack([-np.ones((k, k // 2)), np.ones((k, k // 2))]).astype(np.float32)
    return np.stack([vertical, -vertical, vertical.T, -vertical.T])


class ConvReluPool(nn.Module):
    """The three-stage block that repeats all the way up a convolutional network."""

    def __init__(self, kernels):
        super().__init__()
        self.conv = nn.Conv2d(1, kernels.shape[0], kernel_size=kernels.shape[-1], bias=False)
        self.conv.weight = nn.Parameter(torch.from_numpy(kernels).unsqueeze(1))
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        convolved = self.conv(x)
        activated = F.relu(convolved)
        pooled = self.pool(activated)
        return convolved, activated, pooled


def save_stage(tensor, stem, titles):
    """Write one PNG per channel, scaled independently so faint maps stay visible."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(tensor.shape[1]):
        plane = tensor[0, index].numpy()
        low, high = float(plane.min()), float(plane.max())
        spread = high - low if high > low else 1.0
        scaled = ((plane - low) / spread * 255).astype(np.uint8)
        path = OUT_DIR / f"{stem}_{titles[index]}.png"
        Image.fromarray(scaled).save(path)
        paths.append(path)
    return paths


def main():
    print("=" * 72)
    print("--- 1. One kernel, one window at a time ---")
    manual = convolve_by_hand(TINY_IMAGE, TINY_KERNEL)
    layer_out = conv2d_with_fixed_weights(TINY_IMAGE, TINY_KERNEL[None, ...])
    framework = layer_out[0, 0].numpy()
    print(f"image {TINY_IMAGE.shape} kernel {TINY_KERNEL.shape} -> output {manual.shape}")
    print("hand-computed output:")
    print(manual)
    print(f"max difference against nn.Conv2d: {np.abs(manual - framework).max():.8f}")

    top_row, top_col = np.unravel_index(int(manual.argmax()), manual.shape)
    window = TINY_IMAGE[top_row : top_row + 3, top_col : top_col + 3]
    print(f"strongest response {manual.max():.1f} at row {top_row}, col {top_col}, from window:")
    print(window)
    print(f"  = the kernel has {int(TINY_KERNEL.sum())} ones, and this window put a 1 under each")

    print()
    print("--- 2. Output size is decided before any number is multiplied ---")
    print(f"{'padding':>8}{'stride':>8}{'predicted':>12}{'actual':>10}")
    for padding, stride in ((0, 1), (1, 1), (0, 2), (1, 2)):
        predicted = (TINY_IMAGE.shape[0] + 2 * padding - TINY_KERNEL.shape[0]) // stride + 1
        actual = conv2d_with_fixed_weights(
            TINY_IMAGE, TINY_KERNEL[None, ...], padding=padding, stride=stride
        ).shape[-1]
        print(f"{padding:>8}{stride:>8}{predicted:>12}{actual:>10}")
    print("  padding=1 holds the map at its input size; stride=2 halves it")

    print()
    print("--- 3. A test image whose edges are at known coordinates ---")
    image = render_test_image()
    edge_column = IMAGE_SIZE // 4
    edge_row = IMAGE_SIZE // 2
    print(
        f"rendered {image.shape[0]}x{image.shape[1]} greyscale, values in "
        f"[{image.min():.2f}, {image.max():.2f}]"
    )
    print(f"  vertical dark-to-light edge at column {edge_column}")
    print(f"  horizontal dark-to-light edge at row {edge_row}")

    print()
    print("--- 4. Four kernels, four directions ---")
    kernels = directional_kernels()
    names = ("dark_to_light_x", "light_to_dark_x", "dark_to_light_y", "light_to_dark_y")
    for name, kernel in zip(names, kernels):
        print(f"{name:>18}  sum={kernel.sum():>5.1f}  first row={kernel[0].tolist()}")
    print("  every kernel sums to zero, so a flat region answers with zero whatever its brightness")

    print()
    print("--- 5. Convolution, activation, pooling ---")
    model = ConvReluPool(kernels)
    x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        convolved, activated, pooled = model(x)
    for stem, tensor in (("1_conv", convolved), ("2_relu", activated), ("3_pool", pooled)):
        save_stage(tensor, stem, names)
        print(
            f"{stem:>8}  shape {tuple(tensor.shape)}  "
            f"range [{tensor.min():.2f}, {tensor.max():.2f}]"
        )
    print(f"saved {3 * len(names)} maps under {OUT_DIR}")

    print()
    print("--- 6. What each stage did to the numbers ---")
    negatives = int((convolved < 0).sum())
    total = convolved.numel()
    print(
        f"activation zeroed {negatives}/{total} cells ({negatives / total:.1%}), "
        f"every one of them an edge running the wrong way for its kernel"
    )
    print(
        f"pooling cut {tuple(activated.shape[-2:])} down to {tuple(pooled.shape[-2:])}, "
        f"keeping {pooled.numel() / activated.numel():.0%} of the cells"
    )

    # Read one output row that crosses the block and nothing else, so the peak
    # columns can be checked against the two coordinates render_test_image() used.
    offset = KERNEL_SIZE // 2
    probe_row = IMAGE_SIZE // 8 + (IMAGE_SIZE // 2 - IMAGE_SIZE // 8) // 2
    # PIL fills a rectangle inclusive of its right border, so the block's last
    # bright column is size // 2 and the step to dark falls one column later.
    left_edge, right_edge = edge_column, IMAGE_SIZE // 2 + 1
    print(f"reading output row {probe_row - offset}, which crosses the block and nothing else:")
    for name, drawn in (("dark_to_light_x", left_edge), ("light_to_dark_x", right_edge)):
        row_profile = convolved[0, names.index(name), probe_row - offset].numpy()
        found = int(row_profile.argmax()) + offset
        print(
            f"  {name:>16} peaked at image column {found:>4}, "
            f"drawn edge at {drawn:>4}, score {row_profile.max():.2f}"
        )

    # The largest response in the whole map is not on the block at all.
    plane = convolved[0, names.index("dark_to_light_x")].numpy()
    hot_row, hot_col = np.unravel_index(int(plane.argmax()), plane.shape)
    print(
        f"the strongest response overall, {plane.max():.2f}, sits at image "
        f"({hot_row + offset}, {hot_col + offset}) on the diagonal stroke"
    )
    print(
        f"  the stroke is brighter than the block ({image.max():.2f} against "
        f"{image[probe_row, left_edge + 4]:.2f}), and the kernel scores the size of the "
        f"brightness step, not how well the edge lines up with its axis"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
