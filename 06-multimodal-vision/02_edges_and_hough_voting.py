"""Find edges with Canny, then let the edge pixels vote for lines, circles and a template.

Demonstrates the path from pixels to parametrised shapes, each stage scored against a drawn truth:
    1. Render a scene of lines and disks whose parameters are known, then add noise.
    2. Build Gaussian kernels of several sizes and widths and measure what each keeps.
    3. Take Sobel gradients and threshold them, with and without smoothing first.
    4. Quantise gradient directions and thin the edges with non-maximum suppression.
    5. Trace edges between two thresholds, for three threshold pairs, and compare with OpenCV.
    6. Vote for straight lines in Hesse normal form, over all angles and over a restricted range.
    7. Vote for circles, sampling every direction or only along the gradient.
    8. Build an R-table from a colour-segmented template and locate the template by voting.
    9. Extend the vote over scale and rotation and recover both.

Module 06: Multimodal Vision - Edges and Hough Voting.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "outputs" / "edges_and_hough"
SEED = 3407
HEIGHT, WIDTH = 240, 320
NOISE_SIGMA = 25.0

# Lines in Hesse normal form: rho = x cos(theta) + y sin(theta), theta in degrees.
TRUE_LINES = [(200.0, 45.0), (-60.0, 135.0), (200.0, 90.0), (50.0, 10.0)]
TRUE_CIRCLES = [(80, 70, 22), (230, 150, 35), (140, 120, 16)]   # (x, y, radius)

CANNY_SIZE, CANNY_SIGMA = 7, 1.4
SOBEL_THRESHOLD = 200.0
THRESHOLD_PAIRS = {"both low": (20, 50), "both high": (150, 250), "wide spacing": (40, 250)}
CANNY_PAIR = THRESHOLD_PAIRS["wide spacing"]

LANE_THETAS = ((30, 60), (120, 150))   # the prior used to restrict the line search
RADII = (12, 45)
CIRCLE_ANGLE_STEP = 6

TEMPLATE_POLYGON = [(-30, -40), (10, -45), (15, -10), (40, 5), (38, 30), (-5, 35), (-10, 10), (-35, 0)]
TEMPLATE_CENTRE = (60, 60)   # where the polygon origin sits in the 120x120 template image
TRUE_SCALE, TRUE_ROTATION, TRUE_POSITION = 0.7, 25.0, (225.0, 115.0)
SCALES = np.round(np.arange(0.5, 1.01, 0.1), 2)
ROTATIONS = np.arange(-45, 46, 5)
ANGLE_BIN = 5    # degrees per R-table bin, equal to the rotation step so a turn is a whole shift


# 1. The scene

def hesse_endpoints(rho, theta_deg, width=WIDTH, height=HEIGHT):
    """Two far-apart points on the line rho = x cos(theta) + y sin(theta).

    The foot of the perpendicular from the origin is rho * (cos, sin); the line runs
    through it along the direction (-sin, cos). Stepping far along that direction in
    both senses gives endpoints that cv2.line clips to the image.
    """
    theta = np.radians(theta_deg)
    foot = rho * np.array([np.cos(theta), np.sin(theta)])
    along = np.array([-np.sin(theta), np.cos(theta)])
    reach = 2 * (width + height)
    first, second = foot - reach * along, foot + reach * along
    return tuple(np.rint(first).astype(int)), tuple(np.rint(second).astype(int))


def render_scene(rng):
    """Draw the clean scene, its true edge map, and the noisy image the detectors see."""
    clean = np.full((HEIGHT, WIDTH), 70, np.uint8)
    for x, y, r in TRUE_CIRCLES:
        cv2.circle(clean, (x, y), r, 170, -1)
    for rho, theta in TRUE_LINES:
        cv2.line(clean, *hesse_endpoints(rho, theta), 215, 3)
    kernel = np.ones((3, 3), np.uint8)
    boundary = cv2.dilate(clean, kernel) != cv2.erode(clean, kernel)
    noisy = np.clip(clean + rng.normal(0, NOISE_SIGMA, clean.shape), 0, 255).astype(np.float32)
    return clean.astype(np.float32), boundary, noisy


def edge_scores(edges, boundary, tolerance=2.0):
    """Precision: share of edge pixels near a true boundary. Recall: the reverse."""
    to_truth = cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, 3)
    to_edges = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)
    precision = float((to_truth[edges] <= tolerance).mean()) if edges.any() else 0.0
    recall = float((to_edges[boundary] <= tolerance).mean())
    return precision, recall


# 2-5. Canny, stage by stage

def gaussian_kernel(size, sigma):
    """A size x size Gaussian sampled on integer offsets and normalised to sum to one."""
    offsets = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(offsets, offsets)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (kernel / kernel.sum()).astype(np.float32)


def captured_mass(size, sigma):
    """Share of a Gaussian's weight that falls inside a size x size window.

    The full Gaussian is approximated by a discrete kernel at least 8 sigma wide, which
    holds all but a negligible part of the weight, and the central window is summed.
    """
    wide = gaussian_kernel(8 * int(np.ceil(sigma)) + 1, sigma)
    centre, half = wide.shape[0] // 2, size // 2
    return float(wide[centre - half:centre + half + 1, centre - half:centre + half + 1].sum())


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
SOBEL_Y = SOBEL_X.T


def filter_image(image, kernel):
    """Correlate image with kernel, reflecting at the border.

    The sliding-window arithmetic itself is written out and checked against
    nn.Conv2d in script 12; here it is delegated to cv2.filter2D. filter2D does
    not flip the kernel, so SOBEL_X responds positively where brightness rises
    to the right, which is the sign the direction formulas below assume.
    """
    return cv2.filter2D(image, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT101)


def sobel_gradients(image):
    """Gradient components, magnitude, and direction in degrees on [0, 180)."""
    gx, gy = filter_image(image, SOBEL_X), filter_image(image, SOBEL_Y)
    magnitude = np.hypot(gx, gy)
    direction = np.degrees(np.arctan2(gy, gx)) % 180
    return gx, gy, magnitude, direction


def quantise_direction(direction):
    """Round each direction to the nearest of 0, 45, 90 and 135 degrees.

    A pixel has eight neighbours, lying along four lines through it. The gradient
    direction is snapped to whichever of those four lines is closest, so that the
    two neighbours it is compared against actually lie across the edge.
    """
    return (np.rint(direction / 45).astype(int) % 4) * 45


def non_maximum_suppression(magnitude, sector):
    """Keep a pixel only if it is at least as strong as both neighbours across the edge.

    An edge blurred by smoothing produces a ridge of large magnitude several pixels
    wide. Comparing each pixel with its two neighbours along the gradient keeps the
    crest of that ridge and drops its flanks, leaving a line one pixel thick. The
    comparison is strict on one side and not the other, so a flat-topped crest two
    pixels wide keeps exactly one of them.
    """
    h, w = magnitude.shape
    padded = np.pad(magnitude, 1)
    thin = np.zeros_like(magnitude)
    for angle, (dy, dx) in {0: (0, 1), 45: (1, 1), 90: (1, 0), 135: (1, -1)}.items():
        ahead = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        behind = padded[1 - dy:1 - dy + h, 1 - dx:1 - dx + w]
        keep = (sector == angle) & (magnitude > 0) & (magnitude >= ahead) & (magnitude > behind)
        thin[keep] = magnitude[keep]
    return thin


def hysteresis(thin, low, high):
    """Keep strong pixels, plus weak pixels connected to a strong one through other weak ones.

    Pixels at or above high are edges outright. Pixels between low and high are
    kept only if their 8-connected group of above-low pixels contains at least one
    strong pixel, so a faint stretch of a real contour survives while an isolated
    faint response from noise does not.
    """
    candidate = thin >= low
    count, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
    anchored = np.zeros(count, bool)
    anchored[np.unique(labels[thin >= high])] = True
    anchored[0] = False
    return anchored[labels]


def canny(image, low, high, size=CANNY_SIZE, sigma=CANNY_SIGMA):
    """Smooth, differentiate, thin and trace, returning every intermediate."""
    smoothed = filter_image(image, gaussian_kernel(size, sigma))
    _, _, magnitude, direction = sobel_gradients(smoothed)
    sector = quantise_direction(direction)
    thin = non_maximum_suppression(magnitude, sector)
    return {"smoothed": smoothed, "magnitude": magnitude, "sector": sector,
            "thin": thin, "edges": hysteresis(thin, low, high)}


def to_u8(array):
    """Stretch an array to 0-255 for saving."""
    array = array.astype(np.float32)
    span = array.max() - array.min()
    return np.zeros(array.shape, np.uint8) if span == 0 else \
        np.rint(255 * (array - array.min()) / span).astype(np.uint8)


def tile(image, label):
    """An image as a labelled BGR tile of the scene's size; non-uint8 arrays are stretched to 0-255."""
    image = image if image.dtype == np.uint8 else to_u8(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    if image.shape[:2] != (HEIGHT, WIDTH):
        image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(image, (2, 2), (10 + text_w, 10 + text_h), (0, 0, 0), -1)
    cv2.putText(image, label, (6, 6 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    return image


def save_grid(name, tiles, columns):
    """Tiles in rows of `columns`, with grey dividers, written to OUT_DIR."""
    tiles = tiles + [np.zeros_like(tiles[0])] * (-len(tiles) % columns)
    rows = []
    for start in range(0, len(tiles), columns):
        cells = []
        for cell in tiles[start:start + columns]:
            cells += [cell, np.full((HEIGHT, 4, 3), 128, np.uint8)]
        row = np.hstack(cells[:-1])
        rows += [row, np.full((4, row.shape[1], 3), 128, np.uint8)]
    cv2.imwrite(str(OUT_DIR / name), np.vstack(rows[:-1]))


def scene_u8(image):
    """A float scene clipped to 0-255 as a BGR image, without stretching."""
    return cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


# 6-7. Hough transforms for lines and circles

def hough_lines(edges, thetas_deg):
    """Accumulate votes in (rho, theta) space; every edge pixel votes once per angle.

    A pixel (x, y) lies on every line whose normal form satisfies
    rho = x cos(theta) + y sin(theta), which traces a sinusoid across the
    (theta, rho) plane. Pixels that are collinear trace sinusoids through one
    common point, and the accumulator cell at that point collects their votes.
    """
    diagonal = int(np.ceil(np.hypot(HEIGHT, WIDTH)))
    ys, xs = np.nonzero(edges)
    theta = np.radians(np.asarray(thetas_deg, np.float64))
    rho = np.rint(xs[:, None] * np.cos(theta) + ys[:, None] * np.sin(theta)).astype(int) + diagonal
    accumulator = np.zeros((2 * diagonal + 1, len(theta)), np.int32)
    columns = np.broadcast_to(np.arange(len(theta)), rho.shape)
    np.add.at(accumulator, (rho.ravel(), columns.ravel()), 1)
    return accumulator, diagonal, rho.size


def strongest_lines(accumulator, diagonal, thetas_deg, count=10, rho_gap=8, theta_gap=4):
    """The count highest cells, each suppressing its neighbourhood before the next is taken."""
    votes = accumulator.astype(np.float64).copy()
    found = []
    for _ in range(count):
        r, t = np.unravel_index(int(np.argmax(votes)), votes.shape)
        if votes[r, t] <= 0:
            break
        found.append((float(r - diagonal), float(thetas_deg[t]), int(accumulator[r, t])))
        votes[max(r - rho_gap, 0):r + rho_gap + 1, max(t - theta_gap, 0):t + theta_gap + 1] = -1
    return found


def match_line(truth, found, rho_tol=4.0, theta_tol=3.0):
    """Index of the first detection within tolerance of a true line, or None."""
    rho, theta = truth
    for index, (r, t, _) in enumerate(found):
        if abs(r - rho) <= rho_tol and abs(t - theta) <= theta_tol:
            return index
    return None


def hough_circles(edges, gx, gy, radii, along_gradient):
    """Accumulate votes in (y, x, radius) space.

    Without direction information, an edge pixel could lie on a circle of radius r
    centred anywhere on the circle of radius r around it, so it votes at every
    sampled angle. With the gradient known, the centre can only lie along the
    gradient line, at distance r in one sense or the other, so two votes suffice.
    """
    ys, xs = np.nonzero(edges)
    accumulator = np.zeros((HEIGHT, WIDTH, len(radii)), np.int32)
    if along_gradient:
        norm = np.hypot(gx[ys, xs], gy[ys, xs]) + 1e-9
        ux, uy = gx[ys, xs] / norm, gy[ys, xs] / norm
        directions = [(ux, uy), (-ux, -uy)]
    else:
        angles = np.radians(np.arange(0, 360, CIRCLE_ANGLE_STEP))
        directions = [(np.full(len(xs), np.cos(a)), np.full(len(xs), np.sin(a))) for a in angles]
    cast = 0
    for index, radius in enumerate(radii):
        for dx, dy in directions:
            cx = np.rint(xs + radius * dx).astype(int)
            cy = np.rint(ys + radius * dy).astype(int)
            valid = (cx >= 0) & (cx < WIDTH) & (cy >= 0) & (cy < HEIGHT)
            np.add.at(accumulator[:, :, index], (cy[valid], cx[valid]), 1)
            cast += int(valid.sum())
    return accumulator, cast


def strongest_circles(accumulator, radii, count=10, centre_gap=10):
    """The count strongest (x, y, r) after pooling neighbouring cells and normalising.

    An edge pixel sits about half a pixel outside or inside the drawn rim, and each
    vote is rounded to a whole cell, so the votes for one circle land on a small
    cluster of centres and radii rather than a single cell. Summing each radius with
    its two neighbours and blurring the centre plane gathers the cluster back.
    Dividing by the circumference then keeps a large circle from winning on length.
    """
    counts = accumulator.astype(np.float64)
    pooled = counts.copy()
    pooled[:, :, 1:] += counts[:, :, :-1]
    pooled[:, :, :-1] += counts[:, :, 1:]
    score = pooled / (2 * np.pi * np.asarray(radii, np.float64))[None, None, :]
    score = cv2.GaussianBlur(score, (0, 0), 1.0)
    found = []
    for _ in range(count):
        y, x, k = np.unravel_index(int(np.argmax(score)), score.shape)
        if score[y, x, k] <= 0:
            break
        found.append((int(x), int(y), int(radii[k]), float(score[y, x, k])))
        score[max(y - centre_gap, 0):y + centre_gap + 1, max(x - centre_gap, 0):x + centre_gap + 1, :] = -1
    return found


# 8-9. Generalised Hough transform

def match_circles(truths, found, centre_tol=3.0, radius_tol=2):
    """Pair each true circle with at most one detection, and each detection with at most one truth.

    Every (truth, detection) pair is costed by centre distance plus radius gap, and
    pairs are taken cheapest first, skipping any whose truth or detection is already
    used. A pair counts as a hit only within both tolerances.
    """
    pairs = sorted((np.hypot(c[0] - x, c[1] - y) + abs(c[2] - r), i, j)
                   for i, (x, y, r) in enumerate(truths) for j, c in enumerate(found))
    matched, used = [None] * len(truths), set()
    for _, i, j in pairs:
        if matched[i] is None and j not in used:
            matched[i] = j
            used.add(j)
    hits = sum(j is not None and np.hypot(found[j][0] - x, found[j][1] - y) <= centre_tol
               and abs(found[j][2] - r) <= radius_tol for j, (x, y, r) in zip(matched, truths))
    return matched, hits


def polygon_points(scale=1.0, rotation=0.0, position=(0.0, 0.0)):
    """Template polygon vertices after scaling, rotating (degrees) and translating."""
    turn = np.radians(rotation)
    rotate = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    points = scale * np.array(TEMPLATE_POLYGON, np.float64) @ rotate.T + np.array(position)
    return np.rint(points).astype(np.int32)


def render_template():
    """A yellow shape on a blue ground, as a colour image, the way a map region is marked."""
    template = np.full((120, 120, 3), (150, 60, 20), np.uint8)
    cv2.fillPoly(template, [polygon_points(position=TEMPLATE_CENTRE)], (0, 220, 240))
    return template


def render_ght_scene(rng, scale, rotation, position):
    """A grey scene holding the transformed template among distractor shapes."""
    scene = np.full((HEIGHT, WIDTH), 60, np.uint8)
    cv2.fillPoly(scene, [np.array([(30, 30), (110, 40), (95, 100), (40, 90)], np.int32)], 190)
    cv2.fillPoly(scene, [np.array([(40, 150), (120, 140), (130, 210), (60, 225), (30, 190)], np.int32)], 190)
    cv2.circle(scene, (160, 60), 25, 190, -1)
    cv2.fillPoly(scene, [polygon_points(scale, rotation, position)], 190)
    return np.clip(scene + rng.normal(0, 6, scene.shape), 0, 255).astype(np.float32)


def oriented_edges(image, low, high):
    """Canny edge coordinates with their gradient angle on [0, 360)."""
    stages = canny(image, low, high, size=5, sigma=1.0)
    gx, gy, _, _ = sobel_gradients(stages["smoothed"])
    ys, xs = np.nonzero(stages["edges"])
    angles = np.degrees(np.arctan2(gy[ys, xs], gx[ys, xs])) % 360
    return np.stack([xs, ys], 1).astype(np.float64), angles


def build_r_table(template_mask):
    """Index every template edge point's offset to the reference point by gradient angle.

    The reference point is the mask's centroid. For each edge point the table stores
    the vector from the point to the reference, filed under the point's gradient
    angle. At detection time an edge pixel with the same gradient angle looks up
    those vectors and votes at the positions they point to.
    """
    ys, xs = np.nonzero(template_mask)
    reference = np.array([xs.mean(), ys.mean()])
    points, angles = oriented_edges(template_mask.astype(np.float32) * 255, 40, 120)
    table = {}
    for point, angle in zip(points, angles):
        table.setdefault(int(angle // ANGLE_BIN), []).append(reference - point)
    offset = reference - np.array(TEMPLATE_CENTRE, np.float64)
    return {key: np.array(value) for key, value in table.items()}, len(points), offset


def place_offset(offset, scale, rotation):
    """Where the reference point sits relative to the polygon origin after a transform."""
    turn = np.radians(rotation)
    rotate = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    return scale * rotate @ offset


def ght_vote(table, points, angles, scale, rotation):
    """Accumulate reference-point votes for one (scale, rotation) hypothesis.

    Rotating a shape by phi rotates every gradient by phi as well, so a scene pixel
    with gradient angle alpha corresponds to template entries filed under
    alpha - phi. Their offset vectors are rotated by phi and multiplied by the scale
    before voting.
    """
    turn = np.radians(rotation)
    rotate = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    shift = int(round(rotation / ANGLE_BIN))
    bins = (angles // ANGLE_BIN).astype(int)
    accumulator = np.zeros((HEIGHT, WIDTH), np.float32)
    for key, vectors in table.items():
        scene = points[bins == (key + shift) % (360 // ANGLE_BIN)]
        if len(scene) == 0:
            continue
        offsets = scale * vectors @ rotate.T
        centres = np.rint(scene[:, None, :] + offsets[None, :, :]).reshape(-1, 2).astype(int)
        valid = (centres[:, 0] >= 0) & (centres[:, 0] < WIDTH) & (centres[:, 1] >= 0) & (centres[:, 1] < HEIGHT)
        np.add.at(accumulator, (centres[valid, 1], centres[valid, 0]), 1)
    return cv2.GaussianBlur(accumulator, (0, 0), 1.5)


# Main

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # 1. Scene
    print("--- 1. A scene whose edges are known before any detector runs ---")
    clean, boundary, noisy = render_scene(rng)
    print(f"  {WIDTH}x{HEIGHT}, {len(TRUE_LINES)} lines 3 px wide and {len(TRUE_CIRCLES)} filled disks, "
          f"Gaussian noise sigma {NOISE_SIGMA}")
    for rho, theta in TRUE_LINES:
        print(f"    line  rho {rho:>6.1f}, theta {theta:>5.1f} deg")
    for x, y, r in TRUE_CIRCLES:
        print(f"    disk  centre ({x}, {y}), radius {r}")
    print(f"  true boundary pixels: {int(boundary.sum())} (a 2 px band on each side of every step)")
    save_grid("scene.png", [tile(scene_u8(clean), "clean"), tile(scene_u8(noisy), f"noise sigma {NOISE_SIGMA:.0f}"),
                            tile(boundary, "true boundary")], 3)

    # 2. Gaussian kernels
    print("\n--- 2. Gaussian kernels: what each size and width keeps ---")
    flat = cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, 3) > 6
    grid = []
    print(f"  {'size':>4}{'sigma':>7}{'mass inside window':>20}{'noise left':>12}{'edge strength left':>20}")
    _, _, raw_edge, _ = sobel_gradients(clean)
    for size in (3, 5, 7):
        for sigma in (0.5, 1.0, 1.5):
            kernel = gaussian_kernel(size, sigma)
            smoothed = filter_image(noisy, kernel)
            noise_left = float((smoothed - filter_image(clean, kernel))[flat].std())
            edge_left = float(sobel_gradients(filter_image(clean, kernel))[2][boundary].mean()
                              / raw_edge[boundary].mean())
            print(f"  {size:>4}{sigma:>7.1f}{captured_mass(size, sigma):>20.3f}{noise_left:>12.2f}{edge_left:>20.3f}")
            grid.append(tile(scene_u8(smoothed), f"{size}x{size} sigma {sigma}: noise {noise_left:.1f}, "
                                                 f"edge {edge_left:.2f}"))
    save_grid("gaussian_grid.png", grid, 3)
    print("  gaussian_grid.png: rows are window sizes 3, 5, 7 and columns sigma 0.5, 1.0, 1.5")
    print(f"  noise before smoothing: {float((noisy - clean)[flat].std()):.2f}")
    print("  A wider sigma averages more pixels, so noise falls, and it also spreads each step")
    print("  over more pixels, so the gradient at the edge falls with it. A window too small")
    print("  for its sigma cuts off the tails: the kernel is renormalised, but it is no longer")
    print("  the Gaussian that sigma describes, and it smooths less than sigma promises.")

    # 3. Sobel and a single threshold
    print("\n--- 3. Sobel gradient and one threshold, with and without smoothing ---")
    print(f"  threshold {SOBEL_THRESHOLD:.0f} on the gradient magnitude (Sobel weights, intensities 0-255)")
    print(f"  {'input':<30}{'edge pixels':>12}{'precision':>11}{'recall':>9}")
    sobel_rows = []
    for name, image in (("noisy image", noisy),
                        (f"smoothed, {CANNY_SIZE}x{CANNY_SIZE} sigma {CANNY_SIGMA}",
                         filter_image(noisy, gaussian_kernel(CANNY_SIZE, CANNY_SIGMA)))):
        edges = sobel_gradients(image)[2] >= SOBEL_THRESHOLD
        precision, recall = edge_scores(edges, boundary)
        print(f"  {name:<30}{int(edges.sum()):>12}{precision:>11.3f}{recall:>9.3f}")
        sobel_rows.append(tile(edges, f"{name}: precision {precision:.2f}, recall {recall:.2f}"))
    save_grid("sobel_threshold.png", sobel_rows, 2)
    print("  Differentiation amplifies pixel-to-pixel noise. Smoothing first removes most of")
    print("  it, but a single threshold still leaves bands several pixels thick along each edge.")

    # 4. Direction quantisation and non-maximum suppression
    print("\n--- 4. Direction quantisation and non-maximum suppression ---")
    stages = canny(noisy, *CANNY_PAIR)
    sector, magnitude, thin = stages["sector"], stages["magnitude"], stages["thin"]
    above = magnitude >= CANNY_PAIR[0]
    print(f"  {'direction':>10}{'share of pixels above low':>28}")
    for angle in (0, 45, 90, 135):
        print(f"  {angle:>9}°{float((sector[above] == angle).mean()):>28.1%}")
    print(f"  pixels above {CANNY_PAIR[0]} before suppression {int(above.sum())}, after {int((thin >= CANNY_PAIR[0]).sum())}")
    print(f"  share of the gradient magnitude removed by suppression: "
          f"{float((magnitude - thin).sum() / magnitude.sum()):.1%}")
    direction_colour = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    for angle, colour in {0: (0, 0, 255), 45: (0, 255, 0), 90: (255, 0, 0), 135: (0, 255, 255)}.items():
        direction_colour[(sector == angle) & above] = colour
    raw_direction = np.zeros((HEIGHT, WIDTH), np.uint8)
    raw_direction[above] = np.rint(sobel_gradients(stages["smoothed"])[3][above] * 255 / 180).astype(np.uint8)
    save_grid("canny_stages.png", [
        tile(scene_u8(stages["smoothed"]), "smoothed"), tile(magnitude, "gradient magnitude"),
        tile(cv2.applyColorMap(raw_direction, cv2.COLORMAP_HSV), "direction, continuous"),
        tile(direction_colour, "quantised: red 0 green 45 blue 90 yellow 135"),
        tile(thin, "after non-maximum suppression"), tile(magnitude - thin, "what suppression removed")], 3)
    print("  canny_stages.png: smoothing, magnitude, raw and quantised direction, the thinned")
    print("  magnitude, and the part suppression removed")

    # 5. Hysteresis
    print("\n--- 5. Edge tracing between two thresholds ---")
    print(f"  {'pair':<14}{'low':>5}{'high':>6}{'edge pixels':>13}{'precision':>11}{'recall':>9}")
    traced = []
    for name, (low, high) in THRESHOLD_PAIRS.items():
        edges = hysteresis(thin, low, high)
        precision, recall = edge_scores(edges, boundary)
        print(f"  {name:<14}{low:>5}{high:>6}{int(edges.sum()):>13}{precision:>11.3f}{recall:>9.3f}")
        traced.append(tile(edges, f"{name} {low}/{high}: precision {precision:.2f}, recall {recall:.2f}"))
    strong_only = thin >= CANNY_PAIR[1]
    print(f"  {'high only':<14}{'-':>5}{CANNY_PAIR[1]:>6}{int(strong_only.sum()):>13}"
          f"{edge_scores(strong_only, boundary)[0]:>11.3f}{edge_scores(strong_only, boundary)[1]:>9.3f}")
    traced.append(tile(strong_only, f"high {CANNY_PAIR[1]} only: precision {edge_scores(strong_only, boundary)[0]:.2f}, "
                                    f"recall {edge_scores(strong_only, boundary)[1]:.2f}"))
    save_grid("hysteresis_pairs.png", traced, 2)
    print("  hysteresis_pairs.png: the three threshold pairs and the high threshold alone")
    smoothed_u8 = np.clip(np.rint(stages["smoothed"]), 0, 255).astype(np.uint8)
    ours = canny(smoothed_u8.astype(np.float32), *CANNY_PAIR, size=1, sigma=1.0)["edges"]
    reference = cv2.Canny(smoothed_u8, *CANNY_PAIR, apertureSize=3, L2gradient=True) > 0
    near = lambda a, b: float((cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)[a] <= 1).mean())
    print(f"  against cv2.Canny on the same smoothed image: {int(ours.sum())} vs {int(reference.sum())} pixels, "
          f"{near(ours, reference):.1%} of ours within 1 px of OpenCV's, {near(reference, ours):.1%} the other way")
    print("  Low pairs keep noise crests; high pairs drop faint stretches of real contours.")
    print("  A wide pair lets the high threshold decide what is an edge and the low one decide")
    print("  how far each edge is followed.")
    edges = stages["edges"]

    # 6. Hough lines
    print("\n--- 6. Hough transform for lines ---")
    all_thetas = np.arange(0, 180, 1.0)
    started = time.perf_counter()
    accumulator, diagonal, votes = hough_lines(edges, all_thetas)
    full_time = time.perf_counter() - started
    found = strongest_lines(accumulator, diagonal, all_thetas)
    print(f"  {int(edges.sum())} edge pixels, accumulator {accumulator.shape[0]} x {accumulator.shape[1]}, "
          f"{votes} votes, {full_time * 1000:.0f} ms")
    print(f"  {'rank':>4}{'rho':>8}{'theta':>8}{'votes':>7}  matches")
    for rank, (rho, theta, count) in enumerate(found, 1):
        hit = [f"line rho {r:.0f} theta {t:.0f}" for r, t in TRUE_LINES if match_line((r, t), [(rho, theta, 0)]) == 0]
        print(f"  {rank:>4}{rho:>8.0f}{theta:>8.0f}{count:>7}  {hit[0] if hit else '-'}")
    print(f"  true lines found in the top 10: {sum(match_line(t, found) is not None for t in TRUE_LINES)} of {len(TRUE_LINES)}")

    lane_thetas = np.concatenate([np.arange(a, b + 1, 1.0) for a, b in LANE_THETAS])
    started = time.perf_counter()
    lane_accumulator, _, lane_votes = hough_lines(edges, lane_thetas)
    lane_time = time.perf_counter() - started
    lane_found = strongest_lines(lane_accumulator, diagonal, lane_thetas, count=4)
    print(f"\n  restricted to theta in {LANE_THETAS} (the normals a pair of lane markings can have):")
    print(f"  accumulator {lane_accumulator.shape[0]} x {lane_accumulator.shape[1]}, {lane_votes} votes, "
          f"{lane_time * 1000:.0f} ms ({lane_votes / votes:.0%} of the full vote)")
    for rho, theta, count in lane_found:
        print(f"    rho {rho:>5.0f}  theta {theta:>4.0f}  votes {count}")
    lanes = [t for t in TRUE_LINES if any(a <= t[1] <= b for a, b in LANE_THETAS)]
    print(f"  lane lines among the top {len(lane_found)}: {sum(match_line(t, lane_found) is not None for t in lanes)} "
          f"of {len(lanes)}; the other two lines are outside the range and cannot be voted for")
    print("  The four true lines take the top four ranks. The peaks below them are weaker")
    print("  echoes: the far side of a 3 px stroke lies a few rho away at a slightly different")
    print("  theta, and short runs of disk rim are collinear enough to collect some votes.")
    def draw_lines(lines, label):
        """Detected lines on the noisy scene: green where one matches a true line, red otherwise."""
        canvas = scene_u8(noisy)
        for rho, theta, _ in lines:
            true = any(match_line(t, [(rho, theta, 0)]) == 0 for t in TRUE_LINES)
            cv2.line(canvas, *hesse_endpoints(rho, theta), (0, 255, 0) if true else (0, 0, 255), 1)
        return tile(canvas, label)

    save_grid("hough_lines.png", [
        tile(np.log1p(accumulator), "votes, theta 0-179 across, rho down"),
        draw_lines(found, "top 10, all angles: green true, red other"),
        tile(np.log1p(lane_accumulator), "votes, lane angles only"),
        draw_lines(lane_found, "top 4, lane angles only")], 2)
    print("  hough_lines.png: accumulator and top lines over all angles (top row) and over the")
    print("  lane angles only (bottom row); accumulators on a log scale")

    # 7. Hough circles
    print("\n--- 7. Hough transform for circles ---")
    gx, gy, _, _ = sobel_gradients(stages["smoothed"])
    radii = np.arange(RADII[0], RADII[1] + 1)
    print(f"  radii {RADII[0]}-{RADII[1]} px; scores are votes divided by circumference, "
          f"so a large circle does not win by length alone")
    print(f"  {'voting':<30}{'votes cast':>12}{'time':>9}{'true disks in top 10':>22}")
    circle_results = {}
    for name, along in ((f"every {CIRCLE_ANGLE_STEP} deg around each pixel", False), ("along the gradient only", True)):
        started = time.perf_counter()
        accumulator, cast = hough_circles(edges, gx, gy, radii, along)
        elapsed = time.perf_counter() - started
        top = strongest_circles(accumulator, radii)
        matched, hits = match_circles(TRUE_CIRCLES, top)
        print(f"  {name:<30}{cast:>12}{elapsed * 1000:>7.0f}ms{hits:>22}")
        circle_results[name] = (top, matched, accumulator, hits)
    top, matched, _, _ = circle_results["along the gradient only"]
    print("  gradient-directed vote, the ten strongest circles:")
    for rank, (x, y, r, score) in enumerate(top, 1):
        truth = [f"disk ({tx}, {ty}) r {tr}" for tx, ty, tr in TRUE_CIRCLES
                 if np.hypot(x - tx, y - ty) <= 3 and abs(r - tr) <= 2]
        print(f"    {rank:>2}  centre ({x:>3}, {y:>3})  r {r:>2}  score {score:.3f}  {truth[0] if truth else '-'}")
    print("  one-to-one match for each true disk (no detection is counted twice):")
    for (x, y, r), j in zip(TRUE_CIRCLES, matched):
        found_text = "none" if j is None else f"({top[j][0]}, {top[j][1]}) r {top[j][2]}, rank {j + 1}"
        print(f"    true ({x}, {y}) r {r}  ->  found {found_text}")
    print("  The gradient at a disk's rim points along the radius, so it names the centre's")
    print("  direction. Sampling every angle casts its votes around a whole ring instead,")
    print("  and only the ring through the true centre lines up across pixels. The saving")
    print("  rests on that gradient being reliable: on a blurred or textured rim a wrong")
    print("  direction sends both votes to the wrong place.")
    circle_tiles = []
    for name, (circles, pairs, votes_cube, hits) in circle_results.items():
        short = "every 6 deg" if name.startswith("every") else "along the gradient"
        canvas = scene_u8(noisy)
        true_ranks = {j for j, (x, y, r) in zip(pairs, TRUE_CIRCLES) if j is not None
                      and np.hypot(circles[j][0] - x, circles[j][1] - y) <= 3 and abs(circles[j][2] - r) <= 2}
        for j, (x, y, r, _) in enumerate(circles):
            cv2.circle(canvas, (x, y), r, (0, 255, 0) if j in true_ranks else (0, 0, 255), 1)
        circle_tiles += [tile(np.log1p(votes_cube.max(axis=2)), f"{short}: centre votes, best radius"),
                         tile(canvas, f"{short}: top 10, {hits} of 3 true (green)")]
    save_grid("hough_circles.png", circle_tiles, 2)
    print("  hough_circles.png: centre votes and top ten circles, sampling every angle (top) and")
    print("  voting along the gradient (bottom)")

    # 8. Generalised Hough, translation only
    print("\n--- 8. Generalised Hough transform: an arbitrary shape by its R-table ---")
    template = render_template()
    hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    template_mask = (hsv[..., 0] >= 20) & (hsv[..., 0] <= 35) & (hsv[..., 1] >= 100)
    table, template_points, offset = build_r_table(template_mask)
    print(f"  template: yellow polygon on blue, binarised by a hue lookup ({int(template_mask.sum())} pixels)")
    print(f"  R-table: {template_points} edge points in {len(table)} of {360 // ANGLE_BIN} "
          f"gradient-angle bins of {ANGLE_BIN} deg; reference point is the mask centroid")
    placed = (90.0, 125.0)
    upright = render_ght_scene(rng, 1.0, 0.0, placed)
    points, angles = oriented_edges(upright, 40, 120)
    votes = ght_vote(table, points, angles, 1.0, 0.0)
    y, x = np.unravel_index(int(np.argmax(votes)), votes.shape)
    expected = np.array(placed) + place_offset(offset, 1.0, 0.0)
    print(f"  scene with the shape unscaled and unturned among three distractors;")
    print(f"  its reference point is at ({expected[0]:.1f}, {expected[1]:.1f})")
    print(f"  peak at ({x}, {y}), error {np.hypot(x - expected[0], y - expected[1]):.2f} px, "
          f"peak height {votes[y, x]:.1f}")
    marked = scene_u8(upright)
    origin = np.array([x, y]) - place_offset(offset, 1.0, 0.0)
    cv2.polylines(marked, [polygon_points(position=origin)], True, (0, 0, 255), 1)
    upright_votes, upright_error = votes, np.hypot(x - expected[0], y - expected[1])

    # 9. Scale and rotation
    print("\n--- 9. The same vote over scale and rotation ---")
    scene = render_ght_scene(rng, TRUE_SCALE, TRUE_ROTATION, TRUE_POSITION)
    points, angles = oriented_edges(scene, 40, 120)
    started = time.perf_counter()
    results = []
    for scale in SCALES:
        for rotation in ROTATIONS:
            votes = ght_vote(table, points, angles, scale, rotation)
            y, x = np.unravel_index(int(np.argmax(votes)), votes.shape)
            results.append((float(votes[y, x]), float(scale), float(rotation), int(x), int(y)))
    elapsed = time.perf_counter() - started
    results.sort(key=lambda item: item[0], reverse=True)   # rank by peak height alone
    print(f"  {len(SCALES)} scales x {len(ROTATIONS)} rotations = {len(results)} hypotheses, "
          f"{elapsed:.1f} s")
    expected = np.array(TRUE_POSITION) + place_offset(offset, TRUE_SCALE, TRUE_ROTATION)
    print(f"  truth: scale {TRUE_SCALE}, rotation {TRUE_ROTATION:.0f} deg, "
          f"reference point ({expected[0]:.1f}, {expected[1]:.1f})")
    print(f"  {'rank':>4}{'peak':>8}{'scale':>7}{'rotation':>10}{'position':>14}")
    for rank, (peak, scale, rotation, x, y) in enumerate(results[:5], 1):
        print(f"  {rank:>4}{peak:>8.1f}{scale:>7.1f}{rotation:>10.0f}{f'({x}, {y})':>14}")
    best = results[0]
    print(f"  recovered scale {best[1]:.1f}, rotation {best[2]:.0f} deg, position error "
          f"{np.hypot(best[3] - expected[0], best[4] - expected[1]):.2f} px")
    print("  A turn by phi shifts every gradient angle by phi, which is a whole number of")
    print("  R-table bins when the rotation step equals the bin width, and scales every")
    print("  offset vector by the same factor as the shape. The true 25 deg lies on that")
    print("  grid; a turn between grid steps would be recovered only to the nearest step.")
    found = scene_u8(scene)
    origin = np.array([best[3], best[4]]) - place_offset(offset, best[1], best[2])
    cv2.polylines(found, [polygon_points(best[1], best[2], origin)], True, (0, 0, 255), 1)
    template_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    template_canvas[:, (WIDTH - HEIGHT) // 2:(WIDTH + HEIGHT) // 2] = cv2.resize(template, (HEIGHT, HEIGHT))
    peak_map = np.zeros((len(SCALES), len(ROTATIONS)))
    for peak, scale, rotation, _, _ in results:
        peak_map[int(np.argmin(np.abs(SCALES - scale))), int(np.argmin(np.abs(ROTATIONS - rotation)))] = peak
    peak_image = cv2.applyColorMap(to_u8(peak_map), cv2.COLORMAP_JET)
    save_grid("ght_match.png", [
        tile(template_canvas, "template"),
        tile(upright_votes, "votes, shape unscaled and unturned"),
        tile(marked, f"best peak: error {upright_error:.2f} px"),
        tile(peak_image, f"peak per hypothesis: scale {SCALES[0]}-{SCALES[-1]} down, turn across"),
        tile(ght_vote(table, points, angles, best[1], best[2]), f"votes at scale {best[1]:.1f}, {best[2]:.0f} deg"),
        tile(found, f"recovered scale {best[1]:.1f}, {best[2]:.0f} deg")], 3)
    print("  ght_match.png: the template, the translation-only vote and its match (top), and the")
    print("  peak for every scale and rotation, the winning vote and its match (bottom)")
    print(f"\n  images written to {OUT_DIR}")


if __name__ == "__main__":
    main()
