"""Turn a hand-marked quadrilateral into a card-shaped crop.

Catalog gallery images are tight card scans; eBay photographs are whole
slabs on a desk, angled, with the grading label and background in frame.
Embedding one against the other would measure domain mismatch rather
than degradation tolerance, and read as a catastrophic M2 result for
entirely the wrong reason. This module removes that error source.

Crops are quadrilaterals, not boxes, on purpose: Half B's perspective
estimator reads corner deviation directly off the recorded quad, and
storing an axis-aligned box would mean redoing the whole hand-crop pass.

In the live app M3's detector supplies this quad. Until M3 exists the
corpus is cropped by hand, which also gives M3 a ground-truth set to be
evaluated against later.
"""

import json
import os

import numpy as np
from PIL import Image

# Matches catalog images.small (~245x342). DINOv2 resizes both to 224x224
# anyway, and rendering queries at the gallery's own scale avoids handing
# the real corpus an unintended sharpness advantage over the images it is
# being matched against.
CARD_SIZE = (245, 342)

# Four clicks inside this area is a misclick, not a crop.
MIN_QUAD_AREA = 1000.0

# Four [x, y] pairs, in click order (card top-left first, then clockwise).
Quad = list[list[float]]


def quad_area(quad: Quad) -> float:
    """Shoelace area, orientation-independent."""
    points = np.asarray(quad, dtype=np.float64)
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Sign of the cross product (b - a) x (c - a): which way c turns off a->b."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_properly_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """Whether open segments a-b and c-d cross at an interior point of both."""
    d1 = _orientation(c, d, a)
    d2 = _orientation(c, d, b)
    d3 = _orientation(a, b, c)
    d4 = _orientation(a, b, d)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 != 0 and d2 != 0


def validate_quad(quad: Quad) -> None:
    points = np.asarray(quad, dtype=np.float64)
    if points.shape != (4, 2):
        raise ValueError(f"expected 4 [x, y] points, got shape {points.shape}")
    area = quad_area(points)
    if area < MIN_QUAD_AREA:
        raise ValueError(f"quad area {area:.1f} is below {MIN_QUAD_AREA} — degenerate")
    # A quad recorded in click order (walking its boundary) is simple, not
    # self-intersecting, exactly when its two diagonals cross. A bow-tie
    # misclick swaps two adjacent corners, which keeps the shoelace area
    # comfortably above MIN_QUAD_AREA but makes the diagonals miss each
    # other entirely -- this is a distinct failure from a small/degenerate
    # quad and needs its own check.
    if not _segments_properly_cross(points[0], points[2], points[1], points[3]):
        raise ValueError(f"quad {quad} is self-intersecting (bow-tie) — check click order")


def perspective_coeffs(size: tuple[int, int], quad: Quad) -> tuple[float, ...]:
    """Coefficients mapping the output rectangle back onto `quad`.

    PIL's PERSPECTIVE transform maps OUTPUT coordinates to SOURCE
    coordinates, so the system solved here sends the output rectangle's
    corners to the recorded quad's corners -- the opposite direction from
    degrade.perspective_warp, which warps a rectangle outward.
    """
    width, height = size
    destination = [(0, 0), (width, 0), (width, height), (0, height)]
    rows, rhs = [], []
    for (dx, dy), (sx, sy) in zip(destination, quad):
        rows.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        rhs.append(sx)
        rows.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
        rhs.append(sy)
    solved = np.linalg.solve(np.array(rows, dtype=np.float64), np.array(rhs, dtype=np.float64))
    return tuple(float(c) for c in solved)


def apply_quad(image: Image.Image, quad: Quad, size: tuple[int, int] = CARD_SIZE) -> Image.Image:
    """Unwarp the quad out of the photograph into a card-shaped crop.

    Corner order is the order they were clicked -- card top-left first,
    then clockwise -- and is never sorted geometrically. A card
    photographed at 40 degrees has no meaningful "topmost" corner, and
    sorting would silently rotate some crops.
    """
    validate_quad(quad)
    return image.convert("RGB").transform(
        size,
        Image.Transform.PERSPECTIVE,
        perspective_coeffs(size, quad),
        Image.Resampling.BICUBIC,
    )


def load_crops(path: str) -> dict[str, Quad]:
    """item_id -> quad. Missing file means nothing has been cropped yet."""
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def save_crops(crops: dict[str, Quad], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.part"
    with open(tmp, "w") as fh:
        json.dump(crops, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)  # atomic — hours of hand-cropping live in here
