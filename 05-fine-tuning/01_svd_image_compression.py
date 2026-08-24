"""Rebuild a matrix from a handful of rank-1 terms using the singular value decomposition.

Demonstrates the low-rank approximation that every adapter method rests on:
    1. Decompose a 3x2 matrix by hand and check the numbers against scipy.
    2. Show that singular values are the square roots of the eigenvalues of A_T A.
    3. Show that singular vector signs flip in pairs, so U and V must flip together.
    4. Render a test image locally instead of shipping a photograph.
    5. Rebuild the image from the top k singular values and measure the error.
    6. Count the numbers each rank costs, which is what "compression" really means.
    7. Read the singular value spectrum to decide how small k is allowed to be.

Module 05: Fine-Tuning - Low-Rank Reconstruction.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.linalg import svd

sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path(__file__).parent / "outputs"
IMAGE_SIZE = 512
RANKS = (1, 2, 5, 10, 20, 50, 100, 200)
ENERGY_TARGETS = (0.90, 0.95, 0.99)


def hand_decomposition():
    """Step 1-2. Decompose a 3x2 matrix and tie singular values back to eigenvalues.

    The matrix is small enough to follow term by term, which makes the identity
    sigma_i = sqrt(lambda_i) checkable rather than something to take on faith:
    the eigenvalues of A_T A come out as 6.854 and 0.146, and the singular
    values as their square roots, 2.618 and 0.382.
    """
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

    # The whole point of the decomposition: A is a weighted sum of rank-1 blocks,
    # and the singular value is the weight. Print each block on its own.
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
    """Step 3. Flip one column of U alone, then flip the matching row of V_T too.

    Singular vector signs are only fixed up to pairs. Negating column j of U and
    row j of V_T leaves the product untouched, but negating just one of them
    silently produces a different matrix. Writing down a decomposition with the
    signs "tidied up" on one side only is therefore wrong even though every
    individual number in it is right.
    """
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
    """Step 4. Draw a grayscale test image so no external photograph is needed.

    The drawing deliberately mixes three kinds of structure, because each one
    behaves differently under truncation: smooth gradients are captured by the
    first few terms, hard edges need mid-range terms, and fine text plus noise
    live in the long tail that truncation throws away first.
    """
    image = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(image)

    # A smooth diagonal gradient background: almost pure rank-1 content.
    ramp = np.linspace(0, 120, size, dtype=np.float64)
    background = ramp[:, None] + ramp[None, :] * 0.6
    image = Image.fromarray(np.clip(background, 0, 255).astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(image)

    # Hard-edged shapes: mid-range singular values.
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

    # Fine text: the first detail to disappear when k gets small.
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
    """Step 6. Count stored numbers for a rank-k factorisation and for the full matrix.

    A rank-k factorisation keeps k columns of U, k singular values and k rows of
    V_T, so it costs k * (rows + columns + 1) numbers against rows * columns for
    the dense matrix. For a 1000x1000 matrix at k=3 that is 6003 against
    1000000, which is 0.6 percent, not 6 percent - the ratio is easy to misplace
    by a factor of ten, so the script prints both sides of the division.
    """
    dense = rows * columns
    factored = k * (rows + columns + 1)
    return factored, dense, factored / dense


def reconstruction_table(array, ranks):
    """Step 5-6. Rebuild the image at several ranks and report error and cost."""
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

    print("\nSame accounting on a 1000x1000 matrix, where the ratio is easy to misstate:")
    for k in (3, 10, 50):
        factored, dense, ratio = storage_numbers(1000, 1000, k)
        print(f"  k={k:<3d} {factored:>8d} / {dense:<8d} = {ratio:.2%}")
    return left, values, right_t


def spectrum(values, targets):
    """Step 7. Report how many terms each share of the total energy needs.

    Energy is the squared singular value, because the squares add up to the
    squared Frobenius norm of the matrix. The rank needed for 90 percent is
    usually a small fraction of the full rank, and that gap is exactly what
    makes truncation worth doing.
    """
    energy = values**2
    cumulative = np.cumsum(energy) / energy.sum()
    print(f"\nFirst ten singular values: {np.round(values[:10], 2)}")
    print(f"Term 1 alone holds {cumulative[0]:.2%} of the total energy.")
    for target in targets:
        needed = int(np.searchsorted(cumulative, target) + 1)
        print(f"  {target:.0%} of the energy needs k = {needed} of {len(values)} terms.")

    # Energy share flatters the truncation, because energy is the square of the
    # error. Relative error equals sqrt(1 - cumulative energy), so a rank that
    # captures 90 percent of the energy still rebuilds the matrix with about a
    # third of its magnitude wrong, and the rebuilt image at that rank has lost
    # every hard edge. Pick k by reconstruction error, not by energy percentage.
    print("\nEnergy share against relative error (error = sqrt(1 - energy)):")
    for k in (1, 2, 3, 8):
        if k <= len(values):
            share = cumulative[k - 1]
            print(f"  k={k:<3d} energy {share:.2%}  ->  relative error {np.sqrt(1 - share):.2%}")

    decade = min(len(values), 200)
    decay = values[0] / values[decade - 1]
    print(f"\nSingular value 1 is {decay:.1f}x larger than singular value {decade}.")
    print("A spectrum that drops this fast is what makes a low-rank stand-in usable.")


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
    print("--- 1. Decompose a 3x2 matrix ---")
    matrix, left, values, right_t = hand_decomposition()

    print("\n--- 2. Singular values against eigenvalues: shown above ---")

    print("\n--- 3. Singular vector signs flip in pairs ---")
    sign_pairing(matrix, left, values, right_t)

    print("\n--- 4. Render a test image locally ---")
    array = render_test_image(IMAGE_SIZE)

    print("\n--- 5-6. Rebuild from the top k terms and count the cost ---")
    image_left, image_values, image_right_t = reconstruction_table(array, RANKS)

    print("\n--- 7. Read the singular value spectrum ---")
    spectrum(image_values, ENERGY_TARGETS)

    save_reconstructions(array, image_left, image_values, image_right_t, RANKS, OUTPUT_DIR)


if __name__ == "__main__":
    main()
