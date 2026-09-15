"""Follow one object through a clip by its colour, then measure motion from the pixels alone.

Demonstrates the classical tracking pipeline, each stage scored against a drawn ground truth:
    1. Render a clip of a coloured ellipse that travels a known path, grows, turns and dims.
    2. Threshold the object's colour in BGR and in HSV, before and after the light drops.
    3. Clean the mask with erosion and dilation.
    4. Label connected components two ways by hand and reconcile them with OpenCV.
    5. Track the largest component's centroid through the clip.
    6. Build a hue histogram of the object and back-project it onto every frame.
    7. Follow the back-projection with a fixed-size mean shift window.
    8. Let CAMSHIFT resize and orient the window from the image moments.
    9. Find corners from the structure tensor and read what its eigenvalues say.
    10. Match blocks between two frames whose true displacement is known.
    11. Solve Lucas-Kanade at the corners, on one level and on a pyramid.

Module 06: Multimodal Vision - Colour Tracking and Optical Flow.
"""

import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent / "outputs" / "colour_tracking"
SEED = 3407

HEIGHT, WIDTH = 240, 320
FRAMES = 90
DIM_FROM = 45          # frames from this index on are rendered at DIM_FACTOR of their light
DIM_FACTOR = 0.45
OBJECT_BGR = (200, 110, 30)   # a saturated blue, hue 106 on OpenCV's 0-179 scale
SPECKS = 40            # isolated object-coloured pixels scattered over the background
HOLES = 12             # background-coloured pixels punched into the object each frame

# Colour rules. OpenCV holds a colour image as B, G, R in that order, so the box below
# is a box in the BGR cube; it is fitted to the object as it looks at full light. The
# HSV rule constrains hue and saturation, and asks only that the pixel is not black.
BGR_BOX = {"b": (140, 255), "g": (60, 150), "r": (0, 90)}
HUE_RANGE = (96, 116)
MIN_SATURATION = 120
MIN_VALUE = 40

HIST_BINS = 16
GATE_SATURATION = 60   # below this saturation a pixel's hue is not trusted
GATE_VALUE = 32
CAMSHIFT_MARGIN = 1.25

HARRIS_K = 0.05
HARRIS_SIGMA = 1.5

BLOCK = 16
SEARCH_RADIUS = 8
SMALL_SHIFT = (3.6, -2.4)
LARGE_SHIFT = (11.3, 6.8)
LK_WINDOW = 21


# ---------------------------------------------------------------------------
# 1. The clip
# ---------------------------------------------------------------------------

def object_state(t):
    """Return centre, semi-axes and angle of the ellipse in frame t."""
    phase = t / (FRAMES - 1)
    cx = 160 + 95 * np.sin(2 * np.pi * phase)
    cy = 120 + 45 * np.sin(4 * np.pi * phase)
    a = 26 + 22 * phase
    b = 15 + 11 * phase
    angle = 120 * phase
    return cx, cy, a, b, angle


def draw_ellipse_mask(cx, cy, a, b, angle):
    """Rasterise one ellipse at sub-pixel precision, so the mask is the ground truth."""
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.ellipse(mask, (round(cx * 16), round(cy * 16)), (round(a * 16), round(b * 16)),
                angle, 0, 360, 255, -1, cv2.LINE_8, 4)
    return mask > 0


def render_clip(rng):
    """Render every frame together with the object's true mask and parameters.

    The background is grey texture with a little independent noise per channel,
    which gives it a low saturation and a hue that jumps about from pixel to pixel.
    The object is shaded towards its rim by scaling all three channels together,
    which changes brightness and leaves hue alone. Halfway through, the whole frame
    is multiplied by DIM_FACTOR, the same thing a camera sees when the light fades.
    """
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    texture = cv2.GaussianBlur(rng.normal(0, 1, (HEIGHT, WIDTH)).astype(np.float32), (0, 0), 3)
    grey = 105 + 40 * texture / np.abs(texture).max()
    background = np.clip(grey[..., None] + rng.normal(0, 5, (HEIGHT, WIDTH, 3)), 0, 255)
    speck_y = rng.integers(0, HEIGHT, SPECKS)
    speck_x = rng.integers(0, WIDTH, SPECKS)
    colour = np.array(OBJECT_BGR, np.float32)

    frames, masks, states = [], [], []
    for t in range(FRAMES):
        cx, cy, a, b, angle = object_state(t)
        inside = draw_ellipse_mask(cx, cy, a, b, angle)
        radius = np.hypot((xx - cx) / a, (yy - cy) / b)
        shade = 1.0 - 0.25 * np.clip(radius, 0, 1)

        frame = background.copy()
        frame[inside] = colour * shade[inside, None]
        frame[speck_y, speck_x] = colour
        rho = np.sqrt(rng.random(HOLES)) * 0.6
        phi = rng.random(HOLES) * 2 * np.pi
        hole_x = np.clip(np.rint(cx + rho * a * np.cos(phi)), 0, WIDTH - 1).astype(int)
        hole_y = np.clip(np.rint(cy + rho * b * np.sin(phi)), 0, HEIGHT - 1).astype(int)
        frame[hole_y, hole_x] = background[hole_y, hole_x]
        if t >= DIM_FROM:
            frame *= DIM_FACTOR

        frames.append(np.clip(np.rint(frame), 0, 255).astype(np.uint8))
        masks.append(inside)
        states.append((cx, cy, a, b, angle))
    return frames, masks, states


def mask_centroid(mask):
    """Centroid (x, y) of a boolean mask."""
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def iou(a, b):
    """Intersection over union of two boolean masks."""
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


# ---------------------------------------------------------------------------
# 2-3. Colour thresholds and morphology
# ---------------------------------------------------------------------------

def bgr_box_mask(frame):
    """Keep pixels whose B, G and R each fall inside a fixed box in the colour cube.

    frame[..., 0] is blue, frame[..., 1] green and frame[..., 2] red: OpenCV's order.
    Which corner of the cube the axes are named after does not change the rule, but
    reading the channels in the wrong order does.
    """
    b, g, r = (frame[..., i].astype(int) for i in range(3))
    return ((BGR_BOX["b"][0] <= b) & (b <= BGR_BOX["b"][1])
            & (BGR_BOX["g"][0] <= g) & (g <= BGR_BOX["g"][1])
            & (BGR_BOX["r"][0] <= r) & (r <= BGR_BOX["r"][1]))


def hsv_mask(frame):
    """Keep pixels whose hue is in range, that are saturated, and that are not black.

    Scaling B, G and R by the same factor leaves the ratios between them unchanged,
    and hue and saturation are functions of those ratios only. Value is the one
    channel that follows the light, so the rule puts only a floor under it.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (HUE_RANGE[0] <= h) & (h <= HUE_RANGE[1]) & (s >= MIN_SATURATION) & (v >= MIN_VALUE)


def clean(mask):
    """Opening removes specks smaller than the element; closing then fills pinholes.

    Erosion keeps a pixel only if the whole 3x3 neighbourhood is foreground, which
    deletes anything thinner than three pixels and shaves one pixel off every rim.
    Dilation afterwards grows the survivors back by the same pixel. Running the pair
    the other way round, dilation then erosion, fills holes instead of removing dots.
    """
    kernel = np.ones((3, 3), np.uint8)
    as_u8 = mask.astype(np.uint8)
    eroded = cv2.erode(as_u8, kernel)
    opened = cv2.dilate(eroded, kernel)
    closed = cv2.erode(cv2.dilate(opened, kernel), kernel)
    return eroded > 0, opened > 0, closed > 0


def component_count(mask, connectivity=8):
    """Number of foreground components, background excluded."""
    return cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)[0] - 1


# ---------------------------------------------------------------------------
# 4. Connected components by hand
# ---------------------------------------------------------------------------

def neighbour_offsets(connectivity, causal):
    """Neighbours already visited in raster order (causal) or all of them."""
    four = [(0, -1), (-1, 0)] if causal else [(0, -1), (-1, 0), (0, 1), (1, 0)]
    if connectivity == 4:
        return four
    diagonal = [(-1, -1), (-1, 1)] if causal else [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    return four + diagonal


def two_pass_label(mask, connectivity):
    """Label components with a provisional pass and an equivalence-resolving pass.

    Pass one walks the image in raster order. A foreground pixel with no labelled
    neighbour above or to the left opens a new provisional label; one with labelled
    neighbours takes the smallest, and every other label it touched is recorded as
    equivalent in a union-find table. A U shape is the case that needs this: its two
    arms get different labels until the scan reaches the row that joins them.
    Pass two replaces every provisional label by its root and renumbers the roots.
    """
    h, w = mask.shape
    grid = mask.tolist()
    labels = [[0] * w for _ in range(h)]
    parent = [0]

    def find(label):
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    offsets = neighbour_offsets(connectivity, causal=True)
    for y in range(h):
        row, out = grid[y], labels[y]
        for x in range(w):
            if not row[x]:
                continue
            touched = []
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if ny >= 0 and 0 <= nx < w and labels[ny][nx]:
                    touched.append(labels[ny][nx])
            if not touched:
                parent.append(len(parent))
                out[x] = len(parent) - 1
                continue
            roots = {find(label) for label in touched}
            smallest = min(roots)
            out[x] = smallest
            for root in roots:
                parent[root] = smallest

    compact, result = {}, np.zeros((h, w), np.int32)
    for y in range(h):
        for x in range(w):
            if labels[y][x]:
                root = find(labels[y][x])
                result[y, x] = compact.setdefault(root, len(compact) + 1)
    return result, len(compact), len(parent) - 1


def flood_fill_label(mask, connectivity):
    """Label components one at a time by growing each from its first pixel.

    Every foreground pixel that is still unlabelled seeds a breadth-first search
    that claims its whole component before the scan moves on, so no equivalence
    table is needed. Area and bounding box are accumulated while the search runs.
    """
    h, w = mask.shape
    grid = mask.tolist()
    labels = np.zeros((h, w), np.int32)
    stats = []
    offsets = neighbour_offsets(connectivity, causal=False)
    for sy in range(h):
        for sx in range(w):
            if not grid[sy][sx] or labels[sy, sx]:
                continue
            label = len(stats) + 1
            labels[sy, sx] = label
            queue = deque([(sy, sx)])
            area, x0, y0, x1, y1 = 0, sx, sy, sx, sy
            while queue:
                y, x = queue.popleft()
                area += 1
                x0, x1, y0, y1 = min(x0, x), max(x1, x), min(y0, y), max(y1, y)
                for dy, dx in offsets:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and grid[ny][nx] and not labels[ny, nx]:
                        labels[ny, nx] = label
                        queue.append((ny, nx))
            stats.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1, area))
    return labels, stats


def same_partition(first, second):
    """True when two label images group exactly the same pixels together."""
    foreground = first > 0
    if not np.array_equal(foreground, second > 0):
        return False
    pairs = np.unique(np.stack([first[foreground], second[foreground]]), axis=1).shape[1]
    return pairs == len(np.unique(first[foreground])) == len(np.unique(second[foreground]))


def labelling_test_mask(frame):
    """The first frame's raw HSV mask, plus a U shape and two diagonally touching squares."""
    mask = hsv_mask(frame).astype(np.uint8)
    cv2.rectangle(mask, (20, 20), (70, 70), 1, -1)
    cv2.rectangle(mask, (32, 20), (58, 58), 0, -1)       # hollow out the U from the top
    cv2.rectangle(mask, (250, 190), (269, 209), 1, -1)
    cv2.rectangle(mask, (270, 210), (289, 229), 1, -1)   # touches the first only at a corner
    return mask > 0


# ---------------------------------------------------------------------------
# 6-8. Histogram back-projection, mean shift, CAMSHIFT
# ---------------------------------------------------------------------------

def saturation_gate(hsv):
    """Pixels saturated and bright enough for their hue to mean something."""
    return (hsv[..., 1] >= GATE_SATURATION) & (hsv[..., 2] >= GATE_VALUE)


def hue_histogram(frame, roi, bins=HIST_BINS, gated=True):
    """Hue histogram of the pixels inside roi, scaled so its peak is 255."""
    x, y, w, h = roi
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[y:y + h, x:x + w]
    hues = hsv[..., 0][saturation_gate(hsv)] if gated else hsv[..., 0].ravel()
    hist = np.bincount(hues.astype(int) * bins // 180, minlength=bins).astype(np.float32)
    peak = hist.max()
    return hist if peak == 0 else hist * (255.0 / peak)   # an ROI can gate everything away


def back_project(frame, hist, gated=True):
    """Replace every pixel by the histogram value of its hue bin."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    prob = np.rint(hist[hsv[..., 0].astype(int) * len(hist) // 180])
    if gated:
        prob[~saturation_gate(hsv)] = 0
    return prob.astype(np.uint8)


def mean_shift(prob, window, max_iter=20):
    """Move a fixed-size window to the centroid of the probability inside it until it stops."""
    x, y, w, h = window
    rows, cols = np.arange(h), np.arange(w)
    for _ in range(max_iter):
        patch = prob[y:y + h, x:x + w].astype(np.float64)
        mass = patch.sum()
        if mass == 0:
            break
        dx = int(np.rint(patch.sum(axis=0) @ cols / mass - w * 0.5))
        dy = int(np.rint(patch.sum(axis=1) @ rows / mass - h * 0.5))
        nx = min(max(x + dx, 0), WIDTH - w)
        ny = min(max(y + dy, 0), HEIGHT - h)
        moved = (nx - x) ** 2 + (ny - y) ** 2
        x, y = nx, ny
        if moved < 1:
            break
    return x, y, w, h


def cam_shift(prob, window, max_iter=20):
    """Mean shift whose window is resized and oriented from second-order moments.

    Inside the window the probability is treated as a mass distribution. Its
    centroid gives the position, and its central second moments mu20, mu02, mu11
    form a 2x2 covariance matrix. The eigenvector of the larger eigenvalue is the
    long axis, at 0.5 * atan2(2 mu11, mu20 - mu02). For a uniformly filled ellipse
    the variance along an axis is a squared over four, so each semi-axis is twice
    the square root of its eigenvalue. The next window is the bounding box of that
    ellipse, enlarged by CAMSHIFT_MARGIN so a growing object still fits.
    """
    x, y, w, h = window
    estimate = None
    for _ in range(max_iter):
        patch = prob[y:y + h, x:x + w].astype(np.float64)
        mass = patch.sum()
        if mass == 0:
            break
        rows, cols = np.mgrid[0:h, 0:w]
        mx, my = (patch * cols).sum() / mass, (patch * rows).sum() / mass
        mu20 = (patch * (cols - mx) ** 2).sum() / mass
        mu02 = (patch * (rows - my) ** 2).sum() / mass
        mu11 = (patch * (cols - mx) * (rows - my)).sum() / mass
        theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        spread = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
        a = 2 * np.sqrt((mu20 + mu02) / 2 + spread)
        b = 2 * np.sqrt(max((mu20 + mu02) / 2 - spread, 0.0))
        cx, cy = x + mx, y + my
        estimate = (cx, cy, a, b, np.degrees(theta) % 180)

        half_w = np.hypot(a * np.cos(theta), b * np.sin(theta)) * CAMSHIFT_MARGIN
        half_h = np.hypot(a * np.sin(theta), b * np.cos(theta)) * CAMSHIFT_MARGIN
        nw = int(min(np.ceil(2 * half_w), WIDTH))
        nh = int(min(np.ceil(2 * half_h), HEIGHT))
        nx = int(min(max(np.rint(cx - nw / 2), 0), WIDTH - nw))
        ny = int(min(max(np.rint(cy - nh / 2), 0), HEIGHT - nh))
        if (nx, ny, nw, nh) == (x, y, w, h):
            break
        x, y, w, h = nx, ny, nw, nh
    return estimate, (x, y, w, h)


def angle_gap(a, b):
    """Difference between two axis angles in degrees, where 0 and 180 are the same axis."""
    gap = abs(a - b) % 180
    return min(gap, 180 - gap)


def coverage(mask, window):
    """Share of the object's pixels that fall inside a window."""
    x, y, w, h = window
    return float(mask[y:y + h, x:x + w].sum() / mask.sum())


# ---------------------------------------------------------------------------
# 9. Harris corners
# ---------------------------------------------------------------------------

def corner_scene():
    """A grey image with known polygon corners and a disk that has none."""
    img = np.full((HEIGHT, WIDTH), 60, np.uint8)
    corners = []
    for (x0, y0, x1, y1), value in (((40, 40, 100, 100), 220), ((140, 50, 190, 120), 180),
                                    ((230, 40, 290, 90), 200)):
        cv2.rectangle(img, (x0, y0), (x1, y1), value, -1)
        corners += [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    centre, radius = np.array([80.0, 180.0]), 38.0
    turn = np.radians([30, 120, 210, 300])
    diamond = np.rint(centre + radius * np.stack([np.cos(turn), np.sin(turn)], 1)).astype(np.int32)
    cv2.fillPoly(img, [diamond], 200)
    corners += [tuple(point) for point in diamond]
    cv2.circle(img, (220, 180), 35, 170, -1)
    return cv2.GaussianBlur(img, (0, 0), 1.0), np.array(corners, np.float32)


def structure_tensor(gray):
    """Gaussian-weighted sums of Ix*Ix, Iy*Iy and Ix*Iy at every pixel."""
    image = gray.astype(np.float32) / 255.0
    ix = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    blur = lambda array: cv2.GaussianBlur(array, (0, 0), HARRIS_SIGMA)
    return blur(ix * ix), blur(iy * iy), blur(ix * iy)


def harris_response(gray):
    """R = det(M) - k * trace(M)^2, which is large only where both eigenvalues are."""
    sxx, syy, sxy = structure_tensor(gray)
    return (sxx * syy - sxy ** 2) - HARRIS_K * (sxx + syy) ** 2


def peaks(response, relative=0.01, size=9, limit=None):
    """Local maxima above a share of the global maximum, as (x, y) points."""
    local_max = cv2.dilate(response, np.ones((size, size), np.uint8))
    keep = (response == local_max) & (response > relative * response.max())
    ys, xs = np.nonzero(keep)
    order = np.argsort(-response[ys, xs])
    points = np.stack([xs[order], ys[order]], 1).astype(np.float32)
    return points[:limit] if limit else points


def match_points(truth, found, tolerance=4.0):
    """Count true points with a detection within tolerance pixels."""
    if len(found) == 0:
        return 0
    gaps = np.linalg.norm(truth[:, None, :] - found[None, :, :], axis=2)
    return int((gaps.min(axis=1) <= tolerance).sum())


# ---------------------------------------------------------------------------
# 10-11. Block matching and Lucas-Kanade
# ---------------------------------------------------------------------------

def texture_scene(rng):
    """Blurred noise with a few bright squares: texture at every scale a block might see."""
    noise = cv2.GaussianBlur(rng.normal(0, 1, (HEIGHT, WIDTH)).astype(np.float32), (0, 0), 2.0)
    img = 128 + 45 * noise / noise.std()
    for x0, y0 in ((50, 60), (170, 40), (240, 150), (90, 170)):
        img[y0:y0 + 24, x0:x0 + 24] += 60
    return np.clip(img, 0, 255).astype(np.float32)


def shift_image(img, dx, dy):
    """Move the content of img by (dx, dy) pixels with bilinear resampling."""
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def block_matching(first, second, block=BLOCK, radius=SEARCH_RADIUS):
    """For each block of the first frame, the integer shift that minimises SSD in the second.

    Every candidate position within radius pixels is compared, so the answer is
    the best integer displacement inside a (2 * radius + 1) square and nothing else:
    a motion longer than radius is not in the set being searched.
    """
    margin = radius + block
    vectors, centres = [], []
    for y in range(margin, HEIGHT - margin - block + 1, block):
        for x in range(margin, WIDTH - margin - block + 1, block):
            reference = first[y:y + block, x:x + block]
            area = second[y - radius:y + radius + block, x - radius:x + radius + block]
            windows = np.lib.stride_tricks.sliding_window_view(area, (block, block))
            ssd = ((windows - reference) ** 2).sum(axis=(2, 3))
            iy, ix = np.unravel_index(int(np.argmin(ssd)), ssd.shape)
            vectors.append((ix - radius, iy - radius))
            centres.append((x + block / 2, y + block / 2))
    return np.array(vectors, np.float32), np.array(centres, np.float32)


def lucas_kanade(prev, nxt, points, levels=1, window=LK_WINDOW, iterations=20):
    """Estimate each point's displacement by linearising brightness constancy in a window.

    Assuming I(x, y) = J(x + u, y + v) and expanding J to first order gives one
    equation per pixel, Ix u + Iy v = -It. Stacking the window's pixels and taking
    least squares yields G d = b with G the structure tensor of the window, the same
    matrix Harris scores. The expansion is only valid for small d, so the step is
    iterated, re-sampling J at the current estimate each time. On a pyramid the
    estimate starts at the coarsest level, where every displacement is 2^(L-1)
    times shorter, and is doubled on the way down. G is inverted with a relative
    cut-off, so a window whose G has rank one gets the component it can observe.
    """
    prev_pyramid, next_pyramid = [prev], [nxt]
    for _ in range(levels - 1):
        prev_pyramid.append(cv2.pyrDown(prev_pyramid[-1]))
        next_pyramid.append(cv2.pyrDown(next_pyramid[-1]))
    half = window // 2
    oy, ox = np.mgrid[-half:half + 1, -half:half + 1]
    ox, oy = ox.ravel().astype(np.float32), oy.ravel().astype(np.float32)
    sample = lambda img, mx, my: cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REFLECT101)

    flow = np.zeros_like(points)
    smallest_eigen = None
    for level in reversed(range(levels)):
        image, target = prev_pyramid[level], next_pyramid[level]
        ix = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        iy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        base = points / (2 ** level)
        mx = base[:, :1] + ox
        my = base[:, 1:] + oy
        iw, ixw, iyw = sample(image, mx, my), sample(ix, mx, my), sample(iy, mx, my)
        gxx, gyy, gxy = (ixw * ixw).sum(1), (iyw * iyw).sum(1), (ixw * iyw).sum(1)
        g = np.stack([np.stack([gxx, gxy], 1), np.stack([gxy, gyy], 1)], 1)
        g_inverse = np.linalg.pinv(g, rtol=1e-2)
        smallest_eigen = np.linalg.eigvalsh(g)[:, 0]
        for _ in range(iterations):
            warped = sample(target, mx + flow[:, :1], my + flow[:, 1:])
            it = warped - iw
            b = -np.stack([(ixw * it).sum(1), (iyw * it).sum(1)], 1)
            flow = flow + (g_inverse @ b[..., None])[..., 0]
        if level:
            flow = flow * 2
    return flow, smallest_eigen


def flow_error(flow, truth):
    """Euclidean error of every vector against the true displacement."""
    return np.linalg.norm(flow - np.array(truth, np.float32), axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    bright, dim = range(0, DIM_FROM), range(DIM_FROM, FRAMES)

    # 1. The clip
    print("--- 1. A clip whose every frame has a known mask ---")
    frames, masks, states = render_clip(rng)
    first_state, last_state = states[0], states[-1]
    print(f"  {FRAMES} frames of {WIDTH}x{HEIGHT}, light scaled by {DIM_FACTOR} from frame {DIM_FROM}")
    print(f"  frame  0: centre ({first_state[0]:.1f}, {first_state[1]:.1f}), semi-axes "
          f"{first_state[2]:.1f} x {first_state[3]:.1f}, angle {first_state[4]:.0f} deg, "
          f"{masks[0].sum()} pixels")
    print(f"  frame {FRAMES - 1}: centre ({last_state[0]:.1f}, {last_state[1]:.1f}), semi-axes "
          f"{last_state[2]:.1f} x {last_state[3]:.1f}, angle {last_state[4]:.0f} deg, "
          f"{masks[-1].sum()} pixels")
    # The whole clip as one video, plus the frames either side of the light change,
    # rather than ninety separate images.
    video = cv2.VideoWriter(str(OUT_DIR / "synthetic_tracking.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                            15, (WIDTH, HEIGHT))
    if video.isOpened():
        for frame in frames:
            video.write(frame)
        video.release()
        print(f"  clip written to synthetic_tracking.mp4 at 15 fps")
    else:
        print("  this OpenCV build cannot write mp4, so only the key frames are saved")
    key_frames = (0, DIM_FROM - 1, DIM_FROM, FRAMES - 1)
    for t in key_frames:
        cv2.imwrite(str(OUT_DIR / f"frame_{t:03d}.png"), frames[t])
    print(f"  key frames {list(key_frames)} saved: first, last bright, first dimmed, last")
    sample_bgr = frames[0][masks[0]].mean(axis=0)
    sample_dim = frames[DIM_FROM][masks[DIM_FROM]].mean(axis=0)
    hsv_bright = cv2.cvtColor(np.uint8([[sample_bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    hsv_dim = cv2.cvtColor(np.uint8([[sample_dim]]), cv2.COLOR_BGR2HSV)[0, 0]
    print(f"  mean object colour, bright  BGR {np.rint(sample_bgr).astype(int)}  HSV {hsv_bright}")
    print(f"  mean object colour, dimmed  BGR {np.rint(sample_dim).astype(int)}  HSV {hsv_dim}")
    channels = cv2.split(cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV))
    cv2.imwrite(str(OUT_DIR / "hsv_channels.png"),
                np.hstack([cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)] + list(channels)))
    print("  first frame written as grey, H, S and V side by side: the background is uniformly")
    print("  low in S and a random scatter in H, the object is uniform in both")

    # 2. Colour thresholds
    print("\n--- 2. The same object thresholded in two colour spaces ---")
    bgr_iou = [iou(bgr_box_mask(f), m) for f, m in zip(frames, masks)]
    hsv_iou = [iou(hsv_mask(f), m) for f, m in zip(frames, masks)]
    print(f"  {'rule':<10}{'IoU, bright frames':>20}{'IoU, dimmed frames':>20}")
    for name, scores in (("BGR box", bgr_iou), ("HSV", hsv_iou)):
        print(f"  {name:<10}{np.mean([scores[t] for t in bright]):>20.3f}"
              f"{np.mean([scores[t] for t in dim]):>20.3f}")
    print("  Dimming multiplies B, G and R by one factor. The BGR box tests absolute")
    print("  levels, so the object walks out of it; hue and saturation depend on the")
    print("  ratios between channels, which the factor does not change. That is the one")
    print("  kind of lighting change this measures: a change that scales all three channels")
    print("  equally. A light that changes colour moves the ratios, and hue with them.")

    def panel(image, label):
        """A mask or frame as a labelled BGR tile."""
        tile = cv2.cvtColor(image.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        cv2.putText(tile, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return tile

    cv2.imwrite(str(OUT_DIR / "colour_thresholds.png"), np.vstack([
        np.hstack([panel(frames[t], f"frame {t}"), panel(bgr_box_mask(frames[t]), "BGR box"),
                   panel(hsv_mask(frames[t]), "HSV"), panel(masks[t], "true mask")])
        for t in (0, DIM_FROM)]))
    print(f"  colour_thresholds.png: frame 0 and the first dimmed frame {DIM_FROM}, each as")
    print("  frame, BGR box mask, HSV mask and true mask")

    # 3. Morphology
    print("\n--- 3. Erosion and dilation on the HSV mask ---")
    print(f"  {'frame':>5} {'stage':<16}{'components':>11}{'pixels':>8}{'IoU':>8}")
    stage_rows = []
    for t in (0, FRAMES - 1):
        raw = hsv_mask(frames[t])
        eroded, opened, closed = clean(raw)
        stages = (("raw", raw), ("eroded", eroded), ("opened", opened), ("opened+closed", closed))
        for name, stage in stages:
            print(f"  {t:>5} {name:<16}{component_count(stage):>11}{stage.sum():>8}"
                  f"{iou(stage, masks[t]):>8.3f}")
        print(f"  {t:>5} {'true mask':<16}{1:>11}{masks[t].sum():>8}{1.0:>8.3f}")
        stage_rows.append(np.hstack([panel(stage, f"{name}, frame {t}") for name, stage in stages]
                                    + [panel(masks[t], "true mask")]))
    cv2.imwrite(str(OUT_DIR / "morphology_stages.png"), np.vstack(stage_rows))
    print("  morphology_stages.png: the same stages and the true mask, one row per frame")

    # 4. Connected components
    print("\n--- 4. Connected components, labelled by hand ---")
    test = labelling_test_mask(frames[0])
    component_panels = []
    for connectivity in (4, 8):
        two_pass, count, provisional = two_pass_label(test, connectivity)
        flood, stats = flood_fill_label(test, connectivity)
        n_cv, cv_labels, cv_stats, _ = cv2.connectedComponentsWithStats(
            test.astype(np.uint8), connectivity=connectivity)
        largest = max(stats, key=lambda s: s[4])
        cv_largest = tuple(int(v) for v in cv_stats[1:][np.argmax(cv_stats[1:, 4])])
        print(f"  {connectivity}-connectivity: two-pass {count} components "
              f"(from {provisional} provisional labels), flood fill {len(stats)}, "
              f"OpenCV {n_cv - 1}")
        print(f"    two-pass and flood fill group the same pixels: {same_partition(two_pass, flood)}; "
              f"flood fill and OpenCV: {same_partition(flood, cv_labels)}")
        print(f"    largest component (x, y, w, h, area): flood fill {largest}, OpenCV {cv_largest}")
        boxes = panel(test, f"{connectivity}-connectivity: {len(stats)} components")
        cv2.putText(boxes, "boxes only where area >= 20", (6, HEIGHT - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 255), 1, cv2.LINE_AA)
        for x, y, w, h, area in stats:
            if area >= 20:
                cv2.rectangle(boxes, (x, y), (x + w - 1, y + h - 1), (0, 0, 255), 1)
        component_panels.append(boxes)
    divider = np.full((HEIGHT, 4, 3), 128, np.uint8)
    cv2.imwrite(str(OUT_DIR / "component_boxes.png"), np.hstack([component_panels[0], divider, component_panels[1]]))
    print("  component_boxes.png: 4-connectivity on the left, 8 on the right; the squares at the")
    print("  bottom right get two boxes on the left and one on the right")
    print("  The two squares touch only at a corner: two components under 4-connectivity,")
    print("  one under 8. The U needs the equivalence table, because its arms are")
    print("  labelled separately until the scan reaches the bar that joins them.")

    # 5. Centroid tracking
    print("\n--- 5. Tracking the centroid of the largest component ---")
    # One entry per frame, NaN where nothing was found, so index t is always frame t.
    centroid_track, errors = np.full((FRAMES, 2), np.nan), np.full(FRAMES, np.nan)
    for t, (frame, mask) in enumerate(zip(frames, masks)):
        cleaned = clean(hsv_mask(frame))[2].astype(np.uint8)
        _, _, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        if len(stats) < 2:                       # the colour rule kept nothing in this frame
            continue
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        centroid_track[t] = centroids[biggest]
        errors[t] = np.hypot(*(centroids[biggest] - np.array(mask_centroid(mask))))
    print(f"  frames with no component at all: {int(np.isnan(errors).sum())} of {FRAMES}")
    print(f"  centroid error against the true mask: bright mean {np.nanmean(errors[:DIM_FROM]):.2f} px, "
          f"dimmed mean {np.nanmean(errors[DIM_FROM:]):.2f} px, worst {np.nanmax(errors):.2f} px")
    print("  Each frame is segmented from scratch, so the track never drifts; its error is")
    print("  the handful of pinholes and rim pixels the cleaned mask still disagrees on.")
    print("  Detection and tracking are separate jobs, and this step does the first one every")
    print("  frame. The two trackers below are given a starting box instead and have to keep")
    print("  hold of it, which is a different question and is scored separately.")

    # 6. Histogram back-projection
    print("\n--- 6. A hue histogram of the object, back-projected ---")
    # The starting box is read straight off the first frame's true mask. Steps 7 and 8
    # therefore measure what a tracker does with a perfect initialisation, not whether
    # it can find the object in the first place.
    ys, xs = np.nonzero(masks[0])
    roi = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    hist = hue_histogram(frames[0], roi)
    hsv0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV)
    gate0 = np.zeros((HEIGHT, WIDTH), np.uint8)
    gate0[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = 255
    gate0[~saturation_gate(hsv0)] = 0
    cv_hist = cv2.calcHist([hsv0], [0], gate0, [HIST_BINS], [0, 180]).ravel()
    cv_hist *= 255.0 / cv_hist.max()
    cv_back = cv2.calcBackProject([hsv0], [0], cv_hist, [0, 180], 1)
    print(f"  ROI {roi}, {HIST_BINS} bins of 180/{HIST_BINS} = {180 / HIST_BINS:.2f} hue units each")
    print(f"  bins holding the object: {[i for i, v in enumerate(hist) if v > 0]}, "
          f"peak at bin {int(np.argmax(hist))}")
    print(f"  largest gap to cv2.calcHist {np.abs(hist - cv_hist).max():.4f}, "
          f"to cv2.calcBackProject {int(np.abs(back_project(frames[0], hist, gated=False).astype(int) - cv_back).max())}")
    print(f"  {'frame':>5}{'mass on object, no gate':>26}{'with saturation gate':>24}")
    for t in (0, FRAMES - 1):
        for_frame = []
        for gated in (False, True):
            prob = back_project(frames[t], hist, gated).astype(float)
            for_frame.append(prob[masks[t]].sum() / prob.sum())
        print(f"  {t:>5}{for_frame[0]:>26.3f}{for_frame[1]:>24.3f}")
    print("  A grey pixel has almost equal B, G and R, so its hue is set by noise and lands")
    print("  in every bin, including the object's. Requiring some saturation removes them.")

    whole = (0, 0, WIDTH, HEIGHT)
    print(f"\n  {'bins':>5}{'object bins in ROI':>20}{'mass on object':>16}"
          f"{'whole frame, ungated: mass on object':>38}")
    for bins in (4, 16, 64, 180):
        roi_hist = hue_histogram(frames[0], roi, bins)
        prob = back_project(frames[-1], roi_hist).astype(float)
        whole_prob = back_project(frames[-1], hue_histogram(frames[0], whole, bins, gated=False),
                                  gated=False).astype(float)
        print(f"  {bins:>5}{int((roi_hist > 0).sum()):>20}{prob[masks[-1]].sum() / prob.sum():>16.3f}"
              f"{whole_prob[masks[-1]].sum() / whole_prob.sum():>38.3f}")
    print("  Shading scales all three channels together, so the object's hue takes two values")
    print("  and stays in two bins at every bin count; bin width only starts to matter once")
    print("  an object's hue spreads across a bin boundary. The histogram taken over the whole")
    print("  frame without the gate is dominated by the background and misses the object.")
    probs = [back_project(frame, hist) for frame in frames]

    # 7. Mean shift
    print("\n--- 7. Mean shift with a window that keeps its size ---")
    window = roi
    cv_window = roi
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1)
    ms_error, ms_cover, ms_track, agree = [], [], [], 0
    for prob, mask in zip(probs, masks):
        window = mean_shift(prob, window)
        _, cv_window = cv2.meanShift(prob, cv_window, criteria)
        agree += tuple(window) == tuple(cv_window)
        centre = (window[0] + window[2] / 2, window[1] + window[3] / 2)
        ms_track.append(centre)
        ms_error.append(np.hypot(centre[0] - mask_centroid(mask)[0], centre[1] - mask_centroid(mask)[1]))
        ms_cover.append(coverage(mask, window))
    third = FRAMES // 3
    print(f"  window fixed at {roi[2]}x{roi[3]}; same window as cv2.meanShift in {agree} of {FRAMES} frames")
    print(f"  {'':<16}{'first third':>12}{'last third':>12}")
    print(f"  {'centre error':<16}{np.mean(ms_error[:third]):>11.2f}px{np.mean(ms_error[-third:]):>11.2f}px")
    print(f"  {'object covered':<16}{np.mean(ms_cover[:third]):>12.1%}{np.mean(ms_cover[-third:]):>12.1%}")
    print(f"  The object grows from {masks[0].sum()} to {masks[-1].sum()} pixels. A window sized to")
    print("  the first frame still finds the densest part, but sees less of it each frame.")

    # 8. CAMSHIFT
    print("\n--- 8. CAMSHIFT: size and orientation from the moments ---")
    window, cv_window = roi, roi
    # As in step 5, a lost frame keeps its row as NaN so the thirds stay aligned with the clip.
    cs_rows, cv_error, last = np.full((FRAMES, 5), np.nan), np.full(FRAMES, np.nan), None
    cs_track = np.full((FRAMES, 5), np.nan)     # centre x, y, semi-axes and angle, for drawing
    for t, (prob, mask, state) in enumerate(zip(probs, masks, states)):
        estimate, window = cam_shift(prob, window)
        rotated, cv_window = cv2.CamShift(prob, cv_window, criteria)
        truth_centre = mask_centroid(mask)
        cv_error[t] = np.hypot(rotated[0][0] - truth_centre[0], rotated[0][1] - truth_centre[1])
        last = estimate if t == FRAMES - 1 else last
        if estimate is None:                     # no probability mass left inside the window
            continue
        cs_track[t] = estimate
        cs_rows[t] = (np.hypot(estimate[0] - truth_centre[0], estimate[1] - truth_centre[1]),
                      abs(estimate[2] - state[2]), abs(estimate[3] - state[3]),
                      angle_gap(estimate[4], state[4]), coverage(mask, window))
    print(f"  window resized and turned every frame from the second moments; "
          f"frames where the window held no probability at all: {int(np.isnan(cs_rows[:, 0]).sum())} of {FRAMES}")
    print(f"  {'':<22}{'first third':>12}{'last third':>12}")
    for column, name, unit in ((0, "centre error", "px"), (1, "long semi-axis error", "px"),
                               (2, "short semi-axis error", "px"), (3, "angle error", "deg")):
        print(f"  {name:<22}{np.nanmean(cs_rows[:third, column]):>9.2f} {unit:<3}"
              f"{np.nanmean(cs_rows[-third:, column]):>8.2f} {unit}")
    print(f"  {'object covered':<22}{np.nanmean(cs_rows[:third, 4]):>12.1%}{np.nanmean(cs_rows[-third:, 4]):>12.1%}")
    print(f"  cv2.CamShift centre error over the clip: mean {np.mean(cv_error):.2f} px")
    if last is None:
        print(f"  last frame: window lost; truth {last_state[2]:.1f} x {last_state[3]:.1f}, {last_state[4]:.1f} deg")
    else:
        print(f"  last frame estimate: semi-axes {last[2]:.1f} x {last[3]:.1f}, angle {last[4]:.1f} deg; "
              f"truth {last_state[2]:.1f} x {last_state[3]:.1f}, {last_state[4]:.1f} deg")
    print("  Both trackers start from the box step 6 took off the first frame's true mask, so")
    print("  these errors are what each rule does with a perfect start, not what it does from")
    print("  nothing.")

    # Drawn at twice the clip's size so the four paths and the legend stay legible.
    zoom = 2
    trajectory = cv2.resize(frames[-1], (WIDTH * zoom, HEIGHT * zoom), interpolation=cv2.INTER_NEAREST)
    to_pixels = lambda points: np.rint(np.asarray(points, np.float64) * zoom).astype(np.int32)

    def draw_path(path, colour, thickness):
        """Draw only the runs of consecutive frames whose position is known."""
        path = np.asarray(path, np.float64)
        known = ~np.isnan(path[:, 0])
        for run in np.split(np.arange(len(path)), np.flatnonzero(np.diff(known)) + 1):
            if known[run[0]] and len(run) > 1:
                cv2.polylines(trajectory, [to_pixels(path[run])], False, colour, thickness, cv2.LINE_AA)

    truth_path = np.array([mask_centroid(m) for m in masks], np.float64)
    # Widest first: CAMSHIFT's centre runs almost on top of the centroid, so it is drawn
    # wider and beneath it, and the thin centroid line stays visible along its middle.
    draw_path(truth_path, (0, 200, 0), 9)
    draw_path(cs_track[:, :2], (255, 0, 255), 5)
    draw_path(np.array(ms_track), (0, 255, 255), 2)
    draw_path(centroid_track, (0, 0, 255), 2)
    legend = (("true centre", (0, 200, 0)), ("CAMSHIFT centre", (255, 0, 255)),
              ("mean shift window centre", (0, 255, 255)), ("centroid of largest component", (0, 0, 255)))
    cv2.rectangle(trajectory, (4, 4), (300, 12 + 18 * len(legend)), (0, 0, 0), -1)
    for row, (name, colour) in enumerate(legend):
        y = 20 + 18 * row
        cv2.line(trajectory, (10, y - 4), (34, y - 4), colour, 3)
        cv2.putText(trajectory, name, (42, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT_DIR / "trajectory.png"), trajectory)
    print("  trajectory.png: the four centre paths over the last frame, widest underneath")
    cv2.imwrite(str(OUT_DIR / "backprojection.png"),
                np.hstack([frames[-1], cv2.cvtColor(probs[-1], cv2.COLOR_GRAY2BGR)]))

    # 9. Harris
    print("\n--- 9. Corners from the structure tensor ---")
    scene, true_corners = corner_scene()
    response = harris_response(scene)
    found = peaks(response)
    cv_found = peaks(cv2.cornerHarris(scene.astype(np.float32) / 255.0, 5, 3, HARRIS_K))
    print(f"  {len(true_corners)} true corners (three rectangles and a rotated square), plus a disk with none")
    print(f"  hand-written Harris: {len(found)} detections, {match_points(true_corners, found)} true corners "
          f"within 4 px; cv2.cornerHarris: {len(cv_found)} detections, "
          f"{match_points(true_corners, cv_found)} matched")
    sxx, syy, sxy = structure_tensor(scene)
    print(f"  {'point':<22}{'lambda 1':>11}{'lambda 2':>11}{'R':>12}")
    for name, (px, py) in (("flat, inside a square", (70, 70)), ("edge, middle of a side", (70, 40)),
                           ("corner", (40, 40))):
        tensor = np.array([[sxx[py, px], sxy[py, px]], [sxy[py, px], syy[py, px]]])
        low, high = np.linalg.eigvalsh(tensor)
        print(f"  {name:<22}{high:>11.4f}{low:>11.4f}{response[py, px]:>12.5f}")
    print("  The eigenvalues are how strongly brightness changes along the two principal")
    print("  directions of the window. Flat: both near zero. Edge: one large, one near zero,")
    print("  and R turns negative. Corner: both large, which is the only case R rewards.")
    marked = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    for x, y in found:
        cv2.circle(marked, (int(x), int(y)), 4, (0, 0, 255), 1)
    cv2.imwrite(str(OUT_DIR / "harris_corners.png"), marked)

    # 10. Block matching
    print("\n--- 10. Block matching against a known displacement ---")
    texture = texture_scene(rng)
    print(f"  {BLOCK}x{BLOCK} blocks, sum of squared differences, search radius {SEARCH_RADIUS} px")
    print(f"  {'true shift':<16}{'blocks':>7}{'median vector':>16}{'mean error':>12}{'within 1 px':>13}")
    for shift in (SMALL_SHIFT, LARGE_SHIFT):
        vectors, centres = block_matching(texture, shift_image(texture, *shift))
        error = flow_error(vectors, shift)
        median = np.median(vectors, axis=0)
        print(f"  {str(shift):<16}{len(vectors):>7}{'(' + f'{median[0]:.0f}, {median[1]:.0f}' + ')':>16}"
              f"{error.mean():>12.2f}{(error <= 1).mean():>13.1%}")
        if shift == SMALL_SHIFT:
            arrows = cv2.cvtColor(texture.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            for (cx, cy), (u, v) in zip(centres, vectors):
                cv2.arrowedLine(arrows, (int(cx), int(cy)), (int(cx + 3 * u), int(cy + 3 * v)),
                                (0, 0, 255), 1, tipLength=0.3)
            cv2.imwrite(str(OUT_DIR / "block_flow.png"), arrows)
    print("  Candidates are whole pixels, so a sub-pixel shift is rounded: the floor on the")
    print(f"  error here is the distance from {SMALL_SHIFT} to (4, -2). A shift longer than the")
    print("  radius has no correct candidate at all, and SSD returns the best wrong one.")

    # 11. Lucas-Kanade
    print("\n--- 11. Lucas-Kanade at the strongest corners ---")
    corners = peaks(harris_response(texture.astype(np.uint8)), relative=0.05, limit=400)
    inside = ((corners[:, 0] > 40) & (corners[:, 0] < WIDTH - 40)
              & (corners[:, 1] > 40) & (corners[:, 1] < HEIGHT - 40))
    points = corners[inside][:120]
    small_next = shift_image(texture, *SMALL_SHIFT)
    large_next = shift_image(texture, *LARGE_SHIFT)
    rows = [("one level", SMALL_SHIFT, lucas_kanade(texture, small_next, points, levels=1)[0]),
            ("one level", LARGE_SHIFT, lucas_kanade(texture, large_next, points, levels=1)[0]),
            ("3-level pyramid", LARGE_SHIFT, lucas_kanade(texture, large_next, points, levels=3)[0])]
    cv_flow, status, _ = cv2.calcOpticalFlowPyrLK(
        texture.astype(np.uint8), large_next.astype(np.uint8), points.reshape(-1, 1, 2), None,
        winSize=(LK_WINDOW, LK_WINDOW), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    rows.append(("cv2 pyramid", LARGE_SHIFT, cv_flow.reshape(-1, 2) - points))
    print(f"  {len(points)} Harris corners, window {LK_WINDOW}x{LK_WINDOW}")
    print(f"  {'method':<18}{'true shift':<14}{'median error':>13}{'within 0.1 px':>15}")
    for name, shift, flow in rows:
        error = flow_error(flow, shift)
        print(f"  {name:<18}{str(shift):<14}{np.median(error):>13.3f}{(error <= 0.1).mean():>15.1%}")
    print(f"  cv2 reported {int(status.sum())} of {len(points)} points as tracked")
    arrows = cv2.cvtColor(texture.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for (x, y), (u, v) in zip(points, rows[2][2]):
        cv2.circle(arrows, (int(x), int(y)), 2, (0, 0, 255), -1)
        cv2.arrowedLine(arrows, (int(x), int(y)), (int(round(x + u)), int(round(y + v))),
                        (0, 255, 255), 1, tipLength=0.25)
    cv2.imwrite(str(OUT_DIR / "lucas_kanade_flow.png"), arrows)
    print("  The first-order expansion holds over a few pixels. A pyramid level halves the")
    print("  displacement, so three levels bring an 11 px shift within reach of the top one.")

    edge_shift = (2.6, 1.7)
    scene_f = scene.astype(np.float32)
    corner_points = true_corners[:12]
    edge_points = np.array([((x0 + x1) / 2, y0) for x0, y0, x1 in
                            ((40, 40, 100), (140, 50, 190), (230, 40, 290))]
                           + [(x0, (y0 + y1) / 2) for x0, y0, y1 in
                              ((40, 40, 100), (140, 50, 120), (230, 40, 90))], np.float32)
    moved = shift_image(scene_f, *edge_shift)
    print(f"\n  The same solver on the rectangle scene, shifted by {edge_shift}:")
    print(f"  {'points':<24}{'smallest eigenvalue of G':>26}{'median error':>14}")
    for name, pts in (("rectangle corners", corner_points), ("middles of sides", edge_points)):
        flow, eigen = lucas_kanade(scene_f, moved, pts, levels=1)
        print(f"  {name:<24}{np.median(eigen):>26.1f}{np.median(flow_error(flow, edge_shift)):>14.3f}")
    horizontal = lucas_kanade(scene_f, moved, edge_points[:3], levels=1)[0]
    listed = ", ".join(f"({u:.2f}, {v:.2f})" for u, v in horizontal)
    print(f"  on the three top sides the recovered vectors are {listed}")
    print("  Along a straight edge the window looks the same wherever it slides, so G has")
    print("  rank one and only the motion across the edge is observable. That is why the")
    print("  points handed to Lucas-Kanade are corners: G is well conditioned exactly there.")
    print(f"\n  images written to {OUT_DIR}")


if __name__ == "__main__":
    main()
