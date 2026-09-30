"""This script compresses a 512x512 grayscale test image with truncated SVD,
rebuilding it from its largest rank-1 terms, the low-rank shape LoRA assumes.
SVD writes a matrix as a sum of rank-1 terms, each weighted by a singular value.
Keeping the k largest terms gives a rank-k matrix. Its error depends only on the
weights left out, so the spectrum says how small k can be.

The run prints seven parts:
    1. A 3x2 matrix, decomposed. The two singular values are the square roots of
       the eigenvalues of A_T A, and the two rank-1 terms add back up to A.
    2. Signs flip in pairs. Negating column 1 of U alone changes the product.
       Negating row 1 of V_T as well restores A.
       The 3x2 matrix is only there to demonstrate these two steps.
    3. A test image. A 512x512 grayscale drawing: a diagonal gradient, a ring, a
       rectangle, stripes, a triangle, the word RANK and noise.
    4. Rebuilds at k = 1 to 200. The relative error and the stored numbers for
       each k, the same count for a 1000x1000 matrix, and the rebuilt images
       saved to outputs/.
    5. The spectrum of the image. The k that 90, 95 and 99 percent of the energy
       need, and the relative error at k = 1, 2, 3 and 8, which energy share
       understates.
    6. A floor and a ceiling. Gaussian noise (full rank) and the gradient
       background (rank 2), read the same way as the image.
    7. A stand-in weight update. A rank-12 product plus noise, read the same way.
       It shows the shape LoRA assumes and proves nothing about real updates.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.linalg import svd

# Print UTF-8 even when the output is piped or redirected on Windows.
sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path(__file__).parent / "outputs"
IMAGE_SIZE = 512
RANKS = (1, 2, 5, 10, 20, 50, 100, 200)
ENERGY_TARGETS = (0.90, 0.95, 0.99)
ADAPTER_SIZE = 512
# Not 8: part 5 measures k = 8 for the image, and sharing the number would read
# as if that result carried over to this matrix.
ADAPTER_RANK = 12


def hand_decomposition():
    """Part 1. Decompose a 3x2 matrix and add its rank-1 terms back up to A.
    It is small enough to check by hand: the eigenvalues of A_T A are 6.854 and 0.146."""
    matrix = np.array([[1.0, 2.0], [1.0, 1.0], [0.0, 0.0]])
    print("A =")
    print(matrix)

    left, values, right_t = svd(matrix, full_matrices=False)
    print("\nU (left singular vectors, one column per term):")
    print(left)
    print("\nsingular values:", values)
    print("\nV_T (right singular vectors, one row per term):")
    print(right_t)

    eigenvalues = np.linalg.eigvalsh(matrix.T @ matrix)[::-1]
    print("\neigenvalues of A_T A :", eigenvalues)
    print("singular values squared:", values**2)
    print("sqrt of eigenvalues    :", np.sqrt(eigenvalues))

    # Each singular value weights one rank-1 term. Print the terms one by one.
    print("\nA as a sum of rank-1 terms:")
    total = np.zeros_like(matrix)
    for index, value in enumerate(values):
        term = value * np.outer(left[:, index], right_t[index])
        total = total + term
        print(f"\n  term {index + 1}, weight {value:.8f}:")
        print(term)
    print("\nsum of both terms (should equal A):")
    print(total)
    print(f"max absolute deviation from A: {np.abs(total - matrix).max():.2e}")
    return matrix, left, values, right_t


def sign_pairing(matrix, left, values, right_t):
    """Part 2. Negate column 1 of U alone, then row 1 of V_T as well.
    After the first flip each vector is still a valid singular vector, but the product changes."""
    diagonal = np.diag(values)

    broken_left = left.copy()
    broken_left[:, 0] *= -1
    broken = broken_left @ diagonal @ right_t
    print("\nU with column 1 negated, V_T untouched:")
    print(broken)
    print(f"max absolute deviation from A: {np.abs(broken - matrix).max():.6f}")

    fixed_right_t = right_t.copy()
    fixed_right_t[0] *= -1
    fixed = broken_left @ diagonal @ fixed_right_t
    print("\nboth column 1 of U and row 1 of V_T negated:")
    print(fixed)
    print(f"max absolute deviation from A: {np.abs(fixed - matrix).max():.2e}")


def render_test_image(size):
    """Part 3. Draw a grayscale test image, so no photograph is needed."""
    # A diagonal gradient. It is an outer sum of two vectors, so its rank is
    # exactly 2, and part 6 uses it as the ceiling.
    ramp = np.linspace(0, 120, size, dtype=np.float64)
    background = ramp[:, None] + ramp[None, :] * 0.6
    image = Image.fromarray(np.clip(background, 0, 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(image)

    # Shapes along rows and columns come back at k = 2. The ring needs about 10
    # terms, RANK about 20, and the diagonal triangle is still blurred at k = 50.
    draw.ellipse([size * 0.08, size * 0.10, size * 0.46, size * 0.48], fill=235)
    draw.ellipse([size * 0.16, size * 0.18, size * 0.38, size * 0.40], fill=40)
    draw.rectangle([size * 0.55, size * 0.10, size * 0.92, size * 0.34], fill=200)
    draw.polygon(
        [(size * 0.60, size * 0.90), (size * 0.78, size * 0.52), (size * 0.95, size * 0.90)],
        fill=95,
    )
    for offset in range(0, int(size * 0.34), 12):
        draw.line(
            [(size * 0.06, size * 0.58 + offset), (size * 0.46, size * 0.58 + offset)],
            fill=250,
            width=3,
        )

    try:
        font = ImageFont.truetype("arial.ttf", int(size * 0.06))
    except OSError:
        font = ImageFont.load_default()
    draw.text((size * 0.56, size * 0.40), "RANK", fill=20, font=font)

    array = np.asarray(image, dtype=np.float64)
    rng = np.random.default_rng(3407)
    array = np.clip(array + rng.normal(0.0, 4.0, array.shape), 0, 255)
    print(f"Rendered a {array.shape[0]}x{array.shape[1]} grayscale image.")
    print(f"Pixel range: {array.min():.1f} to {array.max():.1f}")
    return array


def truncate(left, values, right_t, k):
    """Rebuild a matrix from its first k rank-1 terms."""
    return (left[:, :k] * values[:k]) @ right_t[:k]


def storage_numbers(rows, columns, k):
    """Count the numbers a rank-k factorisation stores (k columns of U, k singular
    values, k rows of V_T) against rows * columns for the dense matrix."""
    dense = rows * columns
    factored = k * (rows + columns + 1)
    return factored, dense, factored / dense


def reconstruction_table(array, ranks):
    """Part 4. Rebuild the image at several ranks and report error and storage."""
    rows, columns = array.shape
    left, values, right_t = svd(array, full_matrices=False)
    frobenius = np.linalg.norm(array)

    print(f"\nFull rank available: {len(values)}")
    print(f"Largest singular value: {values[0]:.2f}")
    print(f"Smallest singular value: {values[-1]:.6f}")
    print(f"\n{'k':>5} {'rel. error':>11} {'stored':>10} {'dense':>10} {'ratio':>9}")
    for k in ranks:
        if k > len(values):
            continue
        approximation = truncate(left, values, right_t, k)
        error = np.linalg.norm(array - approximation) / frobenius
        factored, dense, ratio = storage_numbers(rows, columns, k)
        print(f"{k:>5} {error:>10.2%} {factored:>10d} {dense:>10d} {ratio:>8.2%}")

    print("\nSame accounting on a 1000x1000 matrix:")
    for k in (3, 10, 50):
        factored, dense, ratio = storage_numbers(1000, 1000, k)
        print(f"  k={k:<3d} {factored:>8d} / {dense:<8d} = {ratio:.2%}")
    return left, values, right_t


def spectrum(values, targets):
    """Part 5. Report how many terms each share of the energy needs. Energy is the
    squared singular value, because the squares add up to the squared Frobenius norm."""
    energy = values**2
    cumulative = np.cumsum(energy) / energy.sum()
    print(f"\nFirst ten singular values: {np.round(values[:10], 2)}")
    print(f"Term 1 alone holds {cumulative[0]:.2%} of the total energy.")
    for target in targets:
        needed = int(np.searchsorted(cumulative, target) + 1)
        print(f"  {target:.0%} of the energy needs k = {needed} of {len(values)} terms.")

    # Energy share flatters the truncation, because energy is the square of the
    # error: relative error = sqrt(1 - cumulative energy). So k = 2 holds over 90
    # percent of the energy but leaves 27 percent of the matrix wrong: the stripes
    # are sharp, the ring is still a block. Pick k by error, not by energy share.
    print("\nEnergy share against relative error (error = sqrt(1 - energy)):")
    for k in (1, 2, 3, 8):
        if k <= len(values):
            share = cumulative[k - 1]
            print(f"  k={k:<3d} energy {share:.2%}  ->  relative error {np.sqrt(1 - share):.2%}")

    decade = min(len(values), 200)
    decay = values[0] / values[decade - 1]
    print(f"\nSingular value 1 is {decay:.1f}x larger than singular value {decade}.")


def baselines(image_values, size):
    """Part 6. Place the image spectrum between a full-rank floor and a rank-2 ceiling.
    The noise is full rank by construction, and part 3's gradient background is exactly rank 2."""
    rng = np.random.default_rng(11)
    noise_values = svd(rng.normal(0.0, 1.0, (size, size)), compute_uv=False)

    ramp = np.linspace(0, 120, size, dtype=np.float64)
    background_values = svd(ramp[:, None] + ramp[None, :] * 0.6, compute_uv=False)

    header = f"{'matrix':<24} {'rank':>6} {'sv1/sv200':>11} {'k=8 energy':>12} {'k=8 error':>11}"
    print("\n" + header)
    for label, values in (
        ("gaussian noise (floor)", noise_values),
        ("rendered test image", image_values),
        ("gradient background", background_values),
    ):
        energy = values**2
        share = energy[:8].sum() / energy.sum()
        floor = values[0] * 1e-12
        rank = int(np.count_nonzero(values > floor))
        tail = values[min(199, len(values) - 1)]
        # Past the true rank the tail is float residue, not a singular value, so
        # the ratio there is a division by rounding error rather than a number.
        ratio = f"{values[0] / tail:.1f}" if tail > floor else "exhausted"
        print(f"{label:<24} {rank:>6d} {ratio:>11} {share:>11.2%}"
              f" {np.sqrt(max(0.0, 1 - share)):>10.2%}")

    background_share = (background_values**2).sum() / (image_values**2).sum()
    print(f"\nThe gradient background alone holds {background_share:.0%} of the image's energy,")
    print("so the image sits close to the ceiling.")


def weight_update_spectrum(size, rank):
    """Part 7. Read a stand-in update, a rank-r product plus full-rank noise, the same way.
    Nothing measured on the image carries over: this matrix was built to have the low-rank shape."""
    rng = np.random.default_rng(23)
    product = rng.normal(0.0, 1.0, (size, rank)) @ rng.normal(0.0, 1.0, (rank, size))
    product = product / np.sqrt(rank)
    update = product + rng.normal(0.0, 0.05 * np.abs(product).mean(), (size, size))

    values = svd(update, compute_uv=False)
    cumulative = np.cumsum(values**2) / (values**2).sum()
    print(f"\nA {size}x{size} stand-in update, built as a rank-{rank} product plus noise:")
    print(f"  first {rank + 2} singular values: {np.round(values[:rank + 2], 2)}")
    for k in (rank, rank + 1, size // 2):
        share = cumulative[k - 1]
        print(f"  k={k:<4d} energy {share:.2%}  ->  relative error"
              f" {np.sqrt(max(0.0, 1 - share)):.2%}")
    factored, dense, ratio = storage_numbers(size, size, rank)
    print(f"  stored at rank {rank}: {factored} / {dense} = {ratio:.2%} of the dense update")
    print(f"  The drop after term {rank} is the shape LoRA assumes. This matrix was built")
    print("  with it, so it shows the assumption and does not test it.")


def save_reconstructions(array, left, values, right_t, ranks, directory):
    """Write the original and the truncated rebuilds to disk for visual comparison."""
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(directory / "svd_rank_full.png")
    written = ["svd_rank_full.png"]
    for k in ranks:
        if k > len(values):
            continue
        approximation = np.clip(truncate(left, values, right_t, k), 0, 255)
        name = f"svd_rank_{k:03d}.png"
        Image.fromarray(approximation.astype(np.uint8), mode="L").save(directory / name)
        written.append(name)
    print(f"\nWrote {len(written)} images to {directory}")
    print(f"  {', '.join(written)}")


def main():
    print("--- 1. A 3x2 matrix, decomposed ---")
    matrix, left, values, right_t = hand_decomposition()

    print("\n--- 2. Signs flip in pairs ---")
    sign_pairing(matrix, left, values, right_t)

    print("\n--- 3. A test image ---")
    array = render_test_image(IMAGE_SIZE)

    print("\n--- 4. Rebuilds at k = 1 to 200 ---")
    image_left, image_values, image_right_t = reconstruction_table(array, RANKS)
    save_reconstructions(array, image_left, image_values, image_right_t, RANKS, OUTPUT_DIR)

    print("\n--- 5. The spectrum of the image ---")
    spectrum(image_values, ENERGY_TARGETS)

    print("\n--- 6. A floor and a ceiling ---")
    baselines(image_values, IMAGE_SIZE)

    print("\n--- 7. A stand-in weight update ---")
    weight_update_spectrum(ADAPTER_SIZE, ADAPTER_RANK)


if __name__ == "__main__":
    main()
