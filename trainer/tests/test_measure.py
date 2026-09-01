import io

import numpy as np
import pytest
from PIL import Image

from hitcheck_trainer.augment.measure import (
    AxisEstimate,
    DegradationProfile,
    estimate_jpeg,
    jpeg_quality,
)


def card_like(size=(240, 336), seed=7):
    rng = np.random.default_rng(seed)
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.float64)
    arr += np.linspace(40, 200, w)[None, :, None]
    arr[h // 4 + 3 : h // 2 + 3, w // 4 + 3 : 3 * w // 4 + 3] = 90
    arr += rng.normal(0, 6, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def encoded_at(strength, image=None):
    """The exact bytes `degrade.jpeg_artifacts` produces, reopened as a file.

    `jpeg_artifacts` returns `Image.open(buffer).convert("RGB")`, and the
    .convert() drops the `quantization` attribute -- so the estimator can
    never be fed its return value directly. Real inputs are files on disk,
    which do carry the table; this mirrors that.
    """
    quality = int(max(8, 60 - 45 * min(strength, 1.0)))
    buf = io.BytesIO()
    (image or card_like()).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf)


@pytest.mark.parametrize("strength", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_jpeg_estimate_round_trips_a_known_strength(strength):
    estimate = estimate_jpeg(encoded_at(strength))
    assert estimate.saturated is False
    # `quality` is int()-truncated in degrade.py, biasing the inverse high
    # by up to 1/45 = 0.023. Nothing tighter than that is available.
    assert estimate.strength == pytest.approx(strength, abs=0.03)


def test_jpeg_estimate_reports_saturation_at_and_above_full_strength():
    for strength in (1.0, 1.5, 3.0):
        estimate = estimate_jpeg(encoded_at(strength))
        assert estimate.saturated is True
        assert estimate.strength == pytest.approx(1.0)


def test_jpeg_estimate_is_zero_for_a_lightly_compressed_image():
    buf = io.BytesIO()
    card_like().save(buf, format="JPEG", quality=95)
    buf.seek(0)
    estimate = estimate_jpeg(Image.open(buf))
    assert estimate.strength == pytest.approx(0.0)
    assert estimate.saturated is False


def test_jpeg_quality_is_unavailable_without_a_quantization_table():
    # A PNG, or a decoded H.264 frame. This is the case the blockiness
    # fallback exists for -- estimate_jpeg must say "unavailable" rather
    # than guess.
    assert jpeg_quality(card_like()) is None
    assert estimate_jpeg(card_like()).strength is None


def test_jpeg_quality_recovers_the_encoder_quality_exactly():
    for quality in (20, 35, 50):
        buf = io.BytesIO()
        card_like().save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        assert jpeg_quality(Image.open(buf)) == quality


def test_axis_estimate_renders_saturation_and_absence_distinctly():
    assert str(AxisEstimate(0.35)) == "0.35"
    assert str(AxisEstimate(1.0, saturated=True)) == ">= 1.00"
    assert str(AxisEstimate(None)) == "unavailable"


def test_profile_summary_names_every_axis_and_never_collapses_them():
    profile = DegradationProfile(
        jpeg=AxisEstimate(0.5),
        blur=AxisEstimate(0.17),
        perspective=AxisEstimate(0.3),
        glare=AxisEstimate(1.0, saturated=True),
    )
    line = profile.summary()
    for axis in ("jpeg", "blur", "perspective", "glare"):
        assert axis in line
    assert ">= 1.00" in line
    # The spec's central argument: strength is a diagonal through a
    # four-dimensional space, not an axis. A mean would re-create it.
    assert not hasattr(profile, "mean")
    assert not hasattr(profile, "overall")


from hitcheck_trainer.augment.curves import Curve
from hitcheck_trainer.augment.degrade import motion_blur, warped_corners
from hitcheck_trainer.augment.descriptors import quad_corner_deviation, reblur_ratio
from hitcheck_trainer.augment.measure import (
    estimate_blur,
    estimate_perspective,
    strength_for_kernel,
)

KERNELS = (1, 3, 5, 7, 9, 11, 13, 15)


def blur_curve(image=None, kernels=KERNELS, seeds=range(24)):
    """Calibrate the blur axis in-test, on this test's own image.

    Averaged over seeds because `motion_blur` randomises the blur ANGLE
    and `reblur_ratio` responds to angle as well as kernel size. One seed
    per point would make the curve itself a sample of that noise.

    Deliberately not the checked-in curves.json: that is calibrated on
    catalog scans, and a test that depended on it would be asserting
    something about `data/images` rather than about the estimator.
    """
    image = image or card_like()
    seeds = list(seeds)
    points = []
    for kernel in kernels:
        if kernel == 1:
            points.append((float(kernel), reblur_ratio(image)))
            continue
        strength = (kernel - 3) / 12.0 + 1e-6
        ratios = [reblur_ratio(motion_blur(image, seed, strength)) for seed in seeds]
        points.append((float(kernel), sum(ratios) / len(ratios)))
    return Curve(name="blur", parameter="kernel", points=points)


def perspective_curve(strengths=(0.0, 0.25, 0.5, 0.75, 1.0), seeds=range(256)):
    points = []
    for strength in strengths:
        deviations = [
            quad_corner_deviation(warped_corners((240, 336), seed, strength).tolist())
            for seed in seeds
        ]
        points.append((float(strength), float(sum(deviations) / len(deviations))))
    return Curve(name="perspective", parameter="strength", points=points)


@pytest.mark.parametrize(
    "kernel,expected",
    [(1, 0.0), (3, 1 / 24), (5, 1 / 6), (9, 0.5), (15, 1.0)],
)
def test_strength_for_kernel_reports_the_interval_midpoint(kernel, expected):
    assert strength_for_kernel(kernel) == pytest.approx(expected)


def test_strength_for_kernel_rejects_an_even_kernel():
    # `int(3 + 12 * s) | 1` can never produce one, so an even kernel means
    # the caller derived it some other way and the answer would be fiction.
    with pytest.raises(ValueError, match="odd"):
        strength_for_kernel(8)


@pytest.mark.parametrize("strength", [0.1, 0.35])
def test_blur_estimate_recovers_a_known_strength_on_average(strength):
    """Averaged over seeds, and capped at 0.35. Both for measured reasons.

    `motion_blur` picks a random ANGLE per seed, and `reblur_ratio`
    responds to angle as well as to kernel size, because real content is
    anisotropic. Measured on catalog scans: the within-image spread
    across angles at a FIXED kernel is 0.014-0.040, while the gap
    between adjacent kernels' calibration points is only 0.004-0.011.
    The noise is three to nine times the signal, so a single image
    cannot resolve its own kernel. That puts blur in the same class as
    perspective and glare -- a corpus-median measurement, not a
    per-image one -- and a per-image round-trip assertion here would be
    asserting something the descriptor cannot deliver.

    The cap at 0.35 is the descriptor saturating. The curve's adjacent
    gaps shrink from 0.010 to 0.004 across the kernel range, so above
    roughly 0.4 readings compress toward the middle: measured mean
    recovery on catalog scans is -0.11 at strength 0.6 and -0.39 at 0.9.
    That is a real limit of the re-blur ratio, stated in
    `estimate_blur`'s docstring rather than hidden under a loose
    tolerance here.
    """
    image = card_like()
    curve = blur_curve(image)
    recovered = [
        estimate_blur(motion_blur(image, seed, strength), curve).strength
        for seed in range(400, 480)
    ]
    assert sum(recovered) / len(recovered) == pytest.approx(strength, abs=0.12)


def test_blur_estimate_compresses_rather_than_inflates_at_high_strength():
    # The saturation above must be one-directional, and this test pins
    # which direction. UNDER-reporting heavy blur makes the corpus look
    # sharper than it is, which biases the M2 extrapolation toward
    # "training required" -- the conservative side, and the side that
    # costs work rather than a wrong decision. Over-reporting would bias
    # toward skipping training we actually needed.
    image = card_like()
    curve = blur_curve(image)
    recovered = [
        estimate_blur(motion_blur(image, seed, 0.9), curve).strength
        for seed in range(400, 440)
    ]
    assert sum(recovered) / len(recovered) < 0.9


def test_blur_estimate_is_zero_on_an_unblurred_image():
    image = card_like()
    assert estimate_blur(image, blur_curve(image)).strength == pytest.approx(0.0)


def test_blur_estimate_is_ordered_across_strengths():
    # Weaker than round-tripping and independent of it: even where
    # quantisation makes two nearby strengths land on one kernel, the
    # estimate must never go DOWN as the true blur goes up.
    image = card_like()
    curve = blur_curve(image)
    estimates = [
        sum(
            estimate_blur(motion_blur(image, seed, s), curve).strength
            for seed in range(600, 640)
        )
        / 40
        for s in (0.1, 0.3, 0.5, 0.7, 0.9)
    ]
    assert estimates == sorted(estimates)


@pytest.mark.parametrize("strength", [0.2, 0.4, 0.6])
def test_perspective_estimate_recovers_a_known_strength_on_average(strength):
    # `shift` bounds a UNIFORM DRAW, so one image's deviation is one
    # sample, not the parameter. Averaging over seeds is the only honest
    # way to assert this -- a per-image tolerance tight enough to be
    # meaningful would fail on the tail of the distribution.
    #
    # Capped at 0.6 rather than 0.9 for a specific reason: at strength 0.8
    # a large minority of individual draws land above the curve's
    # strength-1.0 point, where `interpolate` clamps and flags saturation.
    # Those clamped samples bias the MEAN upward, so a round-trip up there
    # would be measuring the clamp rather than the estimator. The
    # saturation case gets its own test below.
    curve = perspective_curve()
    recovered = [
        estimate_perspective(
            warped_corners((240, 336), seed, strength).tolist(), curve
        ).strength
        for seed in range(200, 300)
    ]
    assert sum(recovered) / len(recovered) == pytest.approx(strength, abs=0.08)


def test_perspective_estimate_is_zero_for_an_unwarped_rectangle():
    quad = [[0.0, 0.0], [240.0, 0.0], [240.0, 336.0], [0.0, 336.0]]
    assert estimate_perspective(quad, perspective_curve()).strength == pytest.approx(0.0)


def test_perspective_estimate_saturates_above_the_calibrated_range():
    curve = perspective_curve(strengths=(0.0, 0.5, 1.0))
    # A quad far more warped than anything calibrated: the honest answer
    # is ">= 1.0", not an extrapolation.
    quad = [[0.0, 0.0], [240.0, 0.0], [140.0, 336.0], [0.0, 236.0]]
    estimate = estimate_perspective(quad, curve)
    assert estimate.saturated is True
    assert estimate.strength == pytest.approx(1.0)


from hitcheck_trainer.augment.curves import CurveBundle
from hitcheck_trainer.augment.degrade import add_glare
from hitcheck_trainer.augment.descriptors import blockiness, bright_tail_mass
from hitcheck_trainer.augment.measure import (
    estimate_glare,
    estimate_jpeg_blockiness,
    profile_image,
)


def glare_curve(image=None, strengths=(0.0, 0.25, 0.5, 0.75, 1.0), seeds=range(48)):
    image = image or card_like()
    points = []
    for strength in strengths:
        masses = [
            bright_tail_mass(add_glare(image, seed, strength)) for seed in seeds
        ]
        points.append((float(strength), float(sum(masses) / len(masses))))
    return Curve(name="glare", parameter="strength", points=points)


def blockiness_curve(image=None, strengths=(0.0, 0.25, 0.5, 0.75, 1.0)):
    image = image or card_like()
    points = []
    for strength in strengths:
        if strength <= 0:
            compressed = image
        else:
            quality = int(max(8, 60 - 45 * min(strength, 1.0)))
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            compressed = Image.open(buf).convert("RGB")
        points.append((float(strength), blockiness(compressed)))
    return Curve(name="jpeg_blockiness", parameter="strength", points=points)


@pytest.mark.parametrize("strength", [0.25, 0.5])
def test_glare_estimate_recovers_a_known_strength_on_average(strength):
    # add_glare draws a random ellipse centre AND both radii, so a single
    # image's tail mass is one sample of a wide distribution.
    #
    # Stops at 0.5 for the same reason the perspective round-trip stops at
    # 0.6: nearer the top of the curve a growing share of draws clamp and
    # flag saturation, which biases the mean upward and would turn this
    # into a test of the clamp. Saturation and ordering are covered
    # separately below, and between them they pin the rest of the range.
    image = card_like()
    curve = glare_curve(image)
    recovered = [
        estimate_glare(add_glare(image, seed, strength), curve).strength
        for seed in range(500, 580)
    ]
    assert sum(recovered) / len(recovered) == pytest.approx(strength, abs=0.12)


def test_glare_estimate_saturates_above_full_strength():
    # add_glare clamps with min(strength, 1.0), so 1.0 and 2.5 produce
    # IDENTICAL pixels. Reporting 2.5 would be inventing information.
    image = card_like()
    curve = glare_curve(image)
    estimate = estimate_glare(add_glare(image, 5, 2.5), curve)
    assert estimate.saturated is True
    assert estimate.strength == pytest.approx(1.0)


def test_glare_estimate_is_ordered_across_strengths():
    image = card_like()
    curve = glare_curve(image)
    estimates = [
        sum(
            estimate_glare(add_glare(image, seed, s), curve).strength
            for seed in range(600, 620)
        )
        / 20
        for s in (0.2, 0.4, 0.6, 0.8)
    ]
    assert estimates == sorted(estimates)


@pytest.mark.parametrize("strength", [0.3, 0.6, 0.9])
def test_blockiness_fallback_recovers_a_known_strength(strength):
    image = card_like()
    curve = blockiness_curve(image)
    quality = int(max(8, 60 - 45 * min(strength, 1.0)))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    estimate = estimate_jpeg_blockiness(Image.open(buf).convert("RGB"), curve)
    assert estimate.strength == pytest.approx(strength, abs=0.15)


def test_profile_image_prefers_the_header_over_the_blockiness_fallback():
    # The header read is exact; blockiness is a calibrated approximation.
    # Where both are available the exact one must win.
    image = card_like()
    bundle = CurveBundle(
        generated_by="test", sample_images=1, seeds=1,
        curves={
            "blur": blur_curve(image),
            "glare": glare_curve(image),
            "perspective": perspective_curve(),
            "jpeg_blockiness": blockiness_curve(image),
        },
    )
    source = encoded_at(0.5, image)
    profile = profile_image(image, quad=None, source=source, bundle=bundle)
    assert profile.jpeg.strength == pytest.approx(0.5, abs=0.03)


def test_profile_image_falls_back_to_blockiness_without_a_header():
    image = card_like()
    bundle = CurveBundle(
        generated_by="test", sample_images=1, seeds=1,
        curves={
            "blur": blur_curve(image),
            "glare": glare_curve(image),
            "perspective": perspective_curve(),
            "jpeg_blockiness": blockiness_curve(image),
        },
    )
    quality = int(max(8, 60 - 45 * 0.6))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    decoded = Image.open(buf).convert("RGB")  # no quantization attribute
    profile = profile_image(decoded, quad=None, source=None, bundle=bundle)
    assert profile.jpeg.strength == pytest.approx(0.6, abs=0.15)


def test_profile_image_reports_perspective_unavailable_without_a_quad():
    image = card_like()
    bundle = CurveBundle(
        generated_by="test", sample_images=1, seeds=1,
        curves={
            "blur": blur_curve(image),
            "glare": glare_curve(image),
            "perspective": perspective_curve(),
            "jpeg_blockiness": blockiness_curve(image),
        },
    )
    profile = profile_image(image, quad=None, source=None, bundle=bundle)
    assert profile.perspective.strength is None
    assert "perspective=unavailable" in profile.summary()
