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


def laplacian_energy(gray: np.ndarray) -> float | None:
    """Variance of the 4-neighbour Laplacian -- the standard sharpness proxy.

    Used only as the numerator and denominator of a ratio (see
    `reblur_ratio`); the raw value is far too content-dependent to compare
    across a corpus spanning plain commons and busy full-arts.

    Returns None -- never NaN -- when the image has no interior pixels to
    take a Laplacian over (either dimension under 3px). `np.var` of the
    empty slice would give NaN with a RuntimeWarning, and NaN compares
    False against every threshold downstream, which is exactly how a
    degenerate input turns into a confident number.
    """
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return None
    lap = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    return float(lap.var())


def reblur_ratio(image: Image.Image) -> float | None:
    """How little a known probe blur changes this image. Rises with blur.

    Raw Laplacian variance cannot be used directly: a busy full-art scores
    higher than a plain common at identical blur. Under a probe blur a
    sharp image collapses and an already-blurred one barely moves, and the
    content-dependent scale divides out of the ratio.

    Returns None -- UNMEASURABLE, not a reading -- when the image carries
    no Laplacian energy to divide by: a blank slab back, a black crop, a
    blown-out frame, or an image too small to take a Laplacian over. Those
    are real inputs, so this must not raise; but it must not answer
    either. An earlier version returned 1.0 here ("the probe changed
    nothing"), which reads as MAXIMUM blur against a curve that is
    increasing in blur and tops out near 0.05 -- a flat crop profiled as
    blur 1.00. The scale has no slot for "nothing to measure", so the
    honest answer is outside it.
    """
    base = laplacian_energy(luma(image))
    if base is None or not base > 0.0:
        return None
    probed = laplacian_energy(luma(image.filter(ImageFilter.GaussianBlur(PROBE_RADIUS))))
    if probed is None:
        return None
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


def quad_orientation(points: np.ndarray) -> float:
    """The quad's dominant in-plane rotation, in radians.

    Both pairs of opposite edges vote. The two "horizontal" edges
    (top-left->top-right and bottom-left->bottom-right) vote directly; the
    two "vertical" edges are turned -90 degrees, (x, y) -> (y, -x), so
    they vote on the same axis instead of on one 90 degrees away. Summing
    the vectors rather than averaging their angles avoids the wrap at
    +/-pi entirely.

    A perspective warp perturbs the four edges in different directions, so
    they partly cancel and the recovered angle stays near the quad's true
    orientation. A pure rotation moves all four the same way and is
    recovered exactly.
    """
    horizontal = (points[1] - points[0]) + (points[2] - points[3])
    vertical = (points[3] - points[0]) + (points[2] - points[1])
    total = horizontal + np.array([vertical[1], -vertical[0]], dtype=np.float64)
    if not np.hypot(total[0], total[1]) > 0.0:
        return 0.0
    return float(np.arctan2(total[1], total[0]))


def quad_corner_deviation(quad: list[list[float]]) -> float:
    """Max corner offset from the quad's best-fit rectangle, as a fraction of it.

    `quad` is four [x, y] pairs in `corpus.crops.Quad` order: the card's
    top-left first, then clockwise.

    ROTATION-INVARIANT, and that is the load-bearing property. The
    best-fit rectangle below is AXIS-ALIGNED, so measured on the raw quad
    a perfectly flat card merely tilted on the desk reads as perspective:
    5 degrees of in-plane rotation measured perspective-equivalent 0.80
    against the calibrated curve, 10 degrees measured off the top of it.
    Calibration never saw that, because `sweep_perspective` only ever
    feeds it `warped_corners` output, which jitters an AXIS-ALIGNED
    rectangle -- but corpus quads are hand-clicked around slabs lying on a
    desk, and `crops.apply_quad`'s own docstring notes that a card
    photographed at an angle has no meaningful topmost corner. So the quad
    is de-rotated by its dominant orientation (`quad_orientation`) about
    its centroid first, and only the residual -- the part a rotation
    cannot explain -- is measured. In-plane rotation is not a degradation
    `degrade.py` applies, and must not be charged to the perspective axis.

    The best-fit rectangle is the least-squares axis-aligned one of the
    DE-ROTATED quad, which for this corner ordering has the closed form
    below (each edge is the mean of the two corners that define it).
    Deviations are normalised by that rectangle's OWN width and height,
    not the image's: how much desk the seller photographed is not a
    degradation, and normalising by the image would make the same warp
    read differently at different framings.

    `degrade.perspective_warp` displaces all four corners of a full-frame
    rectangle, so on a warped full frame this rectangle is that frame and
    the deviation is a direct read of the jitter. It is one sample of a
    uniform draw, not the parameter -- see `measure.estimate_perspective`.
    """
    points = np.asarray(quad, dtype=np.float64)
    if points.shape != (4, 2):
        raise ValueError(f"expected 4 [x, y] points, got shape {points.shape}")

    theta = quad_orientation(points)
    cos, sin = np.cos(-theta), np.sin(-theta)
    rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)
    centre = points.mean(axis=0)
    points = (points - centre) @ rotation.T + centre

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
