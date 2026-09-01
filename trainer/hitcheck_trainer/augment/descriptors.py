"""Content-only image statistics used to invert `degrade.py`.

Nothing here knows what a "strength" is. Each function returns a raw
descriptor that is monotone in one degradation axis; turning a descriptor
into a strength is `curves.py` plus `measure.py`'s job. Keeping that split
means a descriptor can be tested for the only property that actually
matters -- that it moves in one direction as its axis worsens -- without
any calibration data existing yet.
"""

import numpy as np
from PIL import Image, ImageFilter

# Radius of the probe blur used by `reblur_ratio`. Large enough to
# collapse a sharp image's fine detail, small enough that an already
# blurred image still has somewhere left to fall.
PROBE_RADIUS = 2.0

# Luma above which a pixel counts as part of the specular tail. 200 sits
# above a card's white border (which prints around 235 on a scan but sits
# well below that once the image is embedded in a photograph) and below
# the +105 that `add_glare` adds at full strength.
GLARE_THRESHOLD = 200.0

# JPEG's DCT grid. Boundaries fall between columns 7|8, 15|16, ... so the
# first difference at index i spans columns i and i+1, and the boundary
# differences are exactly those with i % 8 == 7.
BLOCK = 8


def luma(image: Image.Image) -> np.ndarray:
    """Float grayscale, 0-255. PIL's own L conversion, not a hand-rolled one."""
    return np.asarray(image.convert("L"), dtype=np.float64)


def laplacian_energy(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian -- the standard sharpness proxy.

    Used only as the numerator and denominator of a ratio (see
    `reblur_ratio`); the raw value is far too content-dependent to compare
    across a corpus spanning plain commons and busy full-arts.
    """
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    return float(lap.var())


def reblur_ratio(image: Image.Image) -> float:
    """How little a known probe blur changes this image. Rises with blur.

    Raw Laplacian variance cannot be used directly: a busy full-art scores
    higher than a plain common at identical blur. Under a probe blur a
    sharp image collapses and an already-blurred one barely moves, and the
    content-dependent scale divides out of the ratio.

    A flat image (zero Laplacian energy either side) returns 1.0 -- "the
    probe changed nothing" -- rather than raising. A blank slab back is a
    real input.
    """
    base = laplacian_energy(luma(image))
    if base <= 0.0:
        return 1.0
    probed = laplacian_energy(luma(image.filter(ImageFilter.GaussianBlur(PROBE_RADIUS))))
    return float(probed / base)


def bright_tail_mass(image: Image.Image) -> float:
    """Mean luma above GLARE_THRESHOLD, scaled to 0-1. Rises with glare.

    Mass rather than a pixel count: `add_glare` lays down a wide, soft,
    Gaussian-blurred ellipse, so a count at any single threshold would
    move in steps as the ellipse's shoulder crossed it, while the mass
    moves smoothly with the fill value.

    Content-dependent, unavoidably -- a card photographed against a white
    desk starts with tail mass a card on black does not. The calibration
    curve averages that out across the sample; a single image's glare
    estimate carries that content term and should not be read alone.
    """
    gray = luma(image)
    return float(np.clip(gray - GLARE_THRESHOLD, 0.0, None).mean() / 255.0)


def blockiness(image: Image.Image) -> float:
    """Discontinuity across the 8x8 DCT grid, relative to within-block detail.

    The header-less JPEG descriptor: a decoded H.264 stream frame has no
    quantization table to read, so compression has to be measured off the
    pixels. Normalising by within-block variation is what makes the number
    comparable across content -- a busy card has large differences
    everywhere, and only the RATIO of grid-aligned to non-grid-aligned
    differences tracks quantization.
    """
    gray = luma(image)
    if gray.shape[0] < 2 * BLOCK or gray.shape[1] < 2 * BLOCK:
        raise ValueError(f"image {gray.shape} is smaller than two {BLOCK}x{BLOCK} blocks")

    ratios = []
    for axis in (0, 1):
        diffs = np.abs(np.diff(gray, axis=axis))
        length = diffs.shape[axis]
        on_grid = (np.arange(length) % BLOCK) == BLOCK - 1
        boundary = diffs.compress(on_grid, axis=axis)
        interior = diffs.compress(~on_grid, axis=axis)
        if interior.size == 0 or boundary.size == 0:
            continue
        ratios.append(float(boundary.mean() / (interior.mean() + 1e-6)))
    if not ratios:
        raise ValueError("image has no usable block boundaries")
    return float(np.mean(ratios))


def quad_corner_deviation(quad: list[list[float]]) -> float:
    """Max corner offset from the quad's best-fit rectangle, as a fraction of it.

    `quad` is four [x, y] pairs in `corpus.crops.Quad` order: the card's
    top-left first, then clockwise.

    The best-fit rectangle is the least-squares axis-aligned one, which
    for this corner ordering has the closed form below (each edge is the
    mean of the two corners that define it). Deviations are normalised by
    that rectangle's OWN width and height, not the image's: how much desk
    the seller photographed is not a degradation, and normalising by the
    image would make the same warp read differently at different framings.

    `degrade.perspective_warp` displaces all four corners of a full-frame
    rectangle, so on a warped full frame this rectangle is that frame and
    the deviation is a direct read of the jitter. It is one sample of a
    uniform draw, not the parameter -- see `measure.estimate_perspective`.
    """
    points = np.asarray(quad, dtype=np.float64)
    if points.shape != (4, 2):
        raise ValueError(f"expected 4 [x, y] points, got shape {points.shape}")

    left = (points[0, 0] + points[3, 0]) / 2.0
    right = (points[1, 0] + points[2, 0]) / 2.0
    top = (points[0, 1] + points[1, 1]) / 2.0
    bottom = (points[2, 1] + points[3, 1]) / 2.0

    width, height = abs(right - left), abs(bottom - top)
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"quad {quad} has a degenerate best-fit rectangle")

    rectangle = np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float64
    )
    return float((np.abs(points - rectangle) / np.array([width, height])).max())
