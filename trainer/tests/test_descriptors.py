import numpy as np
import pytest
from PIL import Image, ImageFilter

from hitcheck_trainer.augment.degrade import warped_corners
from hitcheck_trainer.augment.descriptors import (
    blockiness,
    bright_tail_mass,
    laplacian_energy,
    luma,
    quad_corner_deviation,
    reblur_ratio,
)


def card_like(size=(240, 336), seed=7):
    """A structured stand-in for a card scan.

    Deliberately NOT uniform noise: the blur descriptors key off edge
    structure, and white noise has edge energy at every scale, which
    would make a broken descriptor look monotone anyway.

    Deliberately NO thin ruled lines either, and this one is load-bearing.
    A 1px grid at a stride that divides 8 lands hard edges exactly on the
    JPEG block boundaries `blockiness()` measures, which inflates the
    boundary term at high quality and makes the descriptor NON-MONOTONE
    (measured: 4.23, 4.83, 4.43, 3.93 across qualities 90/60/30/15).
    Moving the stride off a multiple of 8 does not fix it -- thin lines
    also dominate the within-block denominator, flattening the ratio to
    0.88-0.95 across the whole range. The gradient, art box and grain
    below give every descriptor a clean monotone response (blockiness
    1.20 -> 16.16 over the same sweep). Do not add ruled lines back.

    The art box's `+ 3` offsets are load-bearing for the same reason.
    `h // 4`, `h // 2`, `w // 4` and `3 * w // 4` are all multiples of 4
    at every size this fixture is used at, which puts the box's edges on
    the 8x8 grid: at 160x224 and 480x640 ALL FOUR land there, the
    within-block denominator collapses at low quality, and blockiness
    reaches 1.6e6 instead of ~16. Offsetting by 3 moves every edge to
    index 2 or 6 mod 8 regardless of size. Do not "tidy" the offsets away.
    """
    rng = np.random.default_rng(seed)
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.float64)
    arr += np.linspace(40, 200, w)[None, :, None]          # background gradient
    arr[h // 4 + 3 : h // 2 + 3, w // 4 + 3 : 3 * w // 4 + 3] = 90  # art box, off-grid
    arr += rng.normal(0, 6, arr.shape)                      # a little grain
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def test_luma_is_float_grayscale_in_0_255():
    gray = luma(card_like())
    assert gray.ndim == 2
    assert gray.dtype == np.float64
    assert 0.0 <= gray.min() and gray.max() <= 255.0


def test_laplacian_energy_is_zero_on_a_flat_field():
    flat = np.full((32, 32), 128.0)
    assert laplacian_energy(flat) == pytest.approx(0.0, abs=1e-9)


def test_laplacian_energy_drops_when_an_image_is_blurred():
    img = card_like()
    sharp = laplacian_energy(luma(img))
    soft = laplacian_energy(luma(img.filter(ImageFilter.GaussianBlur(3))))
    assert soft < sharp


def test_reblur_ratio_rises_monotonically_with_existing_blur():
    img = card_like()
    ratios = [
        reblur_ratio(img.filter(ImageFilter.GaussianBlur(r)))
        for r in (0.0, 1.0, 2.0, 4.0, 8.0)
    ]
    assert ratios == sorted(ratios)
    assert ratios[0] < ratios[-1]


def test_reblur_ratio_declines_on_a_flat_image_rather_than_reading_1_0():
    # A flat image has zero Laplacian energy; the ratio is 0/0. It must
    # not raise -- a blank slab back, a black crop or a blown-out frame is
    # a real input -- but it must not answer either.
    #
    # This used to return 1.0 ("a probe blur changes nothing"). The blur
    # curve is INCREASING in blur and its top point is ~0.0496, so 1.0 is
    # twenty times the most-blurred calibrated reading: `nearest` mapped
    # it to kernel 15 and a blank slab back profiled as blur 1.00,
    # MAXIMUM blur. The scale has no slot for "nothing to measure".
    for flat in (
        Image.new("RGB", (64, 64), (128, 128, 128)),
        Image.new("RGB", (245, 342), (0, 0, 0)),
        Image.new("RGB", (245, 342), (255, 255, 255)),
    ):
        assert reblur_ratio(flat) is None


@pytest.mark.parametrize("size", [(1, 1), (2, 2), (1, 500), (500, 2)])
def test_reblur_ratio_declines_on_an_image_too_small_to_take_a_laplacian(size):
    # Under 3px the Laplacian slices to an empty array and np.var gives
    # NaN with a RuntimeWarning. NaN compares False against every
    # threshold downstream, so `nearest`'s min-over-NaN returned the FIRST
    # calibrated point -- kernel 1 -- and the estimator reported a
    # confident blur 0.00. Same defect as the flat-image case, opposite
    # end of the scale; same answer.
    assert reblur_ratio(Image.new("RGB", size, (30, 90, 200))) is None


def test_laplacian_energy_declines_rather_than_returning_nan_when_too_small():
    assert laplacian_energy(np.zeros((2, 40))) is None
    assert laplacian_energy(np.zeros((40, 2))) is None
    assert laplacian_energy(np.zeros((3, 3))) == 0.0


def test_bright_tail_mass_is_zero_on_a_dark_image():
    assert bright_tail_mass(Image.new("RGB", (64, 64), (30, 30, 30))) == 0.0


def test_bright_tail_mass_rises_with_a_brighter_highlight():
    base = np.full((64, 64, 3), 30, dtype=np.uint8)
    masses = []
    for value in (210, 230, 255):
        arr = base.copy()
        arr[20:40, 20:40] = value
        masses.append(bright_tail_mass(Image.fromarray(arr)))
    assert masses == sorted(masses)
    assert masses[0] < masses[-1]


def test_blockiness_rises_as_jpeg_quality_falls():
    import io

    img = card_like()
    scores = []
    for quality in (90, 60, 30, 15):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        scores.append(blockiness(Image.open(buf).convert("RGB")))
    assert scores == sorted(scores)
    assert scores[0] < scores[-1]


def test_quad_corner_deviation_is_zero_for_a_perfect_rectangle():
    quad = [[10.0, 20.0], [110.0, 20.0], [110.0, 160.0], [10.0, 160.0]]
    assert quad_corner_deviation(quad) == pytest.approx(0.0, abs=1e-12)


def test_quad_corner_deviation_is_scale_invariant():
    # Normalising by the quad's own rectangle, not the image, is what makes
    # this true: how much desk the seller photographed is not a degradation,
    # and the same card shot closer must not read as more warped.
    small = [[0.0, 0.0], [110.0, 0.0], [100.0, 140.0], [0.0, 140.0]]
    big = [[p[0] * 2, p[1] * 2] for p in small]
    assert quad_corner_deviation(small) == pytest.approx(quad_corner_deviation(big))


def test_quad_corner_deviation_grows_with_the_corner_push():
    def pushed(dx):
        return [[0.0, 0.0], [100.0 + dx, 0.0], [100.0, 140.0], [0.0, 140.0]]

    assert quad_corner_deviation(pushed(5.0)) < quad_corner_deviation(pushed(15.0))


def rotated_rectangle(degrees, size=(240.0, 336.0), centre=(500.0, 400.0)):
    """A perfectly flat rectangle, rotated in-plane by `degrees`."""
    w, h = size
    points = np.float64([[0, 0], [w, 0], [w, h], [0, h]])
    points -= points.mean(axis=0)
    theta = np.radians(degrees)
    cos, sin = np.cos(theta), np.sin(theta)
    points = points @ np.array([[cos, -sin], [sin, cos]]).T
    return (points + np.float64(centre)).tolist()


@pytest.mark.parametrize("degrees", [-30.0, -10.0, -5.0, -2.0, 2.0, 5.0, 10.0, 20.0, 45.0])
def test_quad_corner_deviation_is_blind_to_pure_in_plane_rotation(degrees):
    # THE reason this descriptor de-rotates. Corpus quads are hand-clicked
    # around slabs lying on a desk, and a slab is never laid down square.
    # Measured against the axis-aligned best-fit rectangle this used to
    # use, a perfectly flat card read: 2 deg -> perspective 0.32, 5 deg ->
    # 0.80, 10 deg and 20 deg -> off the top of the curve. In-plane
    # rotation is not something degrade.py applies and must not be charged
    # to the perspective axis.
    assert quad_corner_deviation(rotated_rectangle(degrees)) == pytest.approx(0.0, abs=1e-9)


def test_quad_corner_deviation_still_sees_a_warp_hiding_under_a_rotation():
    # De-rotating must not de-warp. A quad that is warped AND tilted has
    # to read the same as the same warp lying square.
    square = [[0.0, 0.0], [240.0, 0.0], [222.0, 336.0], [11.0, 322.0]]
    flat = quad_corner_deviation(square)
    assert flat > 0.01

    points = np.asarray(square, dtype=np.float64)
    centre = points.mean(axis=0)
    theta = np.radians(17.0)
    cos, sin = np.cos(theta), np.sin(theta)
    tilted = ((points - centre) @ np.array([[cos, -sin], [sin, cos]]).T + centre).tolist()
    assert quad_corner_deviation(tilted) == pytest.approx(flat, rel=1e-9)


def test_quad_corner_deviation_recovers_the_forward_warp_after_de_rotation():
    # The forward model still inverts: mean deviation over seeds has to
    # rise with `perspective_warp`'s strength, monotonically and from
    # exactly zero.
    means = [
        sum(
            quad_corner_deviation(warped_corners((240, 336), seed, s).tolist())
            for seed in range(200)
        )
        / 200
        for s in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    ]
    assert means[0] == pytest.approx(0.0, abs=1e-12)
    assert means == sorted(means)
    assert means[-1] > means[1] * 4


def test_quad_corner_deviation_rejects_a_degenerate_quad():
    with pytest.raises(ValueError, match="degenerate"):
        quad_corner_deviation([[0.0, 0.0], [0.0, 0.0], [0.0, 10.0], [0.0, 10.0]])
