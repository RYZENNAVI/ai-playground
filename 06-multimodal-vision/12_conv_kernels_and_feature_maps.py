"""This script walks hand-written convolution kernels across small rendered images and
checks every output cell, to work out what a convolutional layer computes.

The run goes through the arithmetic underneath every convolutional layer:
    1. A 3x3 X-shaped kernel convolved over a 5x5 binary image by hand, checked
       against nn.Conv2d, and the window behind the strongest cell.
    2. The output size for four padding and stride settings, predicted and measured.
    3. A 160x160 test image drawn with its edges at known coordinates, saved next to
       the feature maps. The horizontal edge is there for the saved _y maps; nothing
       later checks it by number.
    4. Four 4x4 edge kernels, one per direction, each summing to zero.
    5. The image through convolution, ReLU and 2x2 max pooling, one map saved per
       kernel and stage.
    6. What ReLU removed and pooling kept, where the two vertical kernels peak on a
       row that crosses only the block, and where the strongest response lies.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

# Print UTF-8 even when the output is piped or redirected on Windows.
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

# An X-shaped kernel. A diagonal band fills one diagonal of a window but not the
# other, so no window scores the full 5.
TINY_KERNEL = np.array(
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ],
    dtype=np.float32,
)


def convolve_by_hand(image, kernel):
    """Return the valid-mode convolution computed with explicit loops, to check nn.Conv2d.

    Like nn.Conv2d, it does not flip the kernel, so strictly it is a cross-correlation.
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

    Bias is off: its random initial value would shift every cell and break the comparison.
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
    """Draw a greyscale test card with its edges at known coordinates, so each check is exact."""
    img = Image.new("L", (size, size), color=40)
    draw = ImageDraw.Draw(img)
    # A bright block whose left border is a vertical edge at x = size // 4.
    draw.rectangle([size // 4, size // 8, size // 2, size - size // 8], fill=230)
    # A bright bar whose top border is a horizontal edge at y = size // 2.
    draw.rectangle([size // 2 + 8, size // 2, size - size // 8, size // 2 + 24], fill=200)
    # A bright stroke 5 pixels wide, nearly horizontal: it rises 10 pixels over 60.
    draw.line([size // 2 + 8, size - size // 6, size - 12, size // 2 + 44], fill=255, width=5)
    return np.asarray(img, dtype=np.float32) / 255.0


def directional_kernels(k=KERNEL_SIZE):
    """Return four edge kernels: dark-to-light and light-to-dark, in both axes.

    Each is half negative and half positive, so it answers to one direction of step only.
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
    hits = int((window * TINY_KERNEL).sum())
    ties = int((manual == manual.max()).sum())
    print(
        f"  = the kernel has {int(TINY_KERNEL.sum())} ones, "
        f"and this window put a 1 under {hits} of them"
    )
    print(f"  {ties} cells tie at {manual.max():.1f}; argmax reports the first")

    print()
    print("--- 2. Output size is decided before any number is multiplied ---")
    print(f"{'padding':>8}{'stride':>8}{'predicted':>12}{'actual':>10}")
    for padding, stride in ((0, 1), (1, 1), (0, 2), (1, 2)):
        predicted = (TINY_IMAGE.shape[0] + 2 * padding - TINY_KERNEL.shape[0]) // stride + 1
        actual = conv2d_with_fixed_weights(
            TINY_IMAGE, TINY_KERNEL[None, ...], padding=padding, stride=stride
        ).shape[-1]
        print(f"{padding:>8}{stride:>8}{predicted:>12}{actual:>10}")
    print("  padding=1 holds the map at its input size; stride=2 about halves it, rounding up")

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
    # Saved next to the feature maps so they can be compared with the input.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = OUT_DIR / "0_input.png"
    Image.fromarray((image * 255).astype(np.uint8)).save(input_path)
    print(f"  saved the input itself to {input_path.name}, next to the maps below")

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
    # A flat region answers with zero, but only to within floating point: mostly the
    # transposed kernels accumulate in a different order and leave a residue
    # around 1e-7. Counting every value below zero would score that residue as
    # signal, and most of the map is flat, so the tolerance is what makes the
    # sentence after the count true.
    tolerance = 1e-6
    negatives = int((convolved < -tolerance).sum())
    residue = int(((convolved < 0) & (convolved >= -tolerance)).sum())
    total = convolved.numel()
    print(
        f"activation zeroed {negatives}/{total} cells ({negatives / total:.1%}), "
        f"every one of them an edge running the wrong way for its kernel"
    )
    print(
        f"  a further {residue} cells sit below zero by less than {tolerance:g}, which is "
        f"rounding in a flat region rather than an edge, and is not counted above"
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
        f"({hot_row + offset}, {hot_col + offset}) at the left end of the stroke"
    )
    print(
        f"  the stroke's cut-off left end is as clean a vertical step as the block's edge, "
        f"only brighter ({image.max():.2f} against {image[probe_row, left_edge + 4]:.2f}); "
        f"the kernel scores the size of the step, so the brighter edge wins"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
