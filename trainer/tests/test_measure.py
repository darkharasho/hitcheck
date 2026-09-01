import io
import json

import numpy as np
import pytest
from PIL import Image

from hitcheck_trainer.augment.measure import (
    AxisEstimate,
    DegradationProfile,
    axis_medians,
    estimate_jpeg,
    jpeg_quality,
)
from hitcheck_trainer.augment.measure import main as measure_main


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


def glossy_card(size=(240, 336), seed=7):
    """Card-like, but with the luma distribution of a real scan.

    `card_like()`'s gradient tops out at exactly GLARE_THRESHOLD, so on the
    median seed a glare ellipse crosses the threshold nowhere and the
    descriptor reads pure baseline. Real scans measured from data/images/
    sit at mean luma 98-203 with p95 213-244; this fixture matches that, so
    glare registers wherever the ellipse lands.
    """
    rng = np.random.default_rng(seed)
    w, h = size
    arr = np.full((h, w, 3), 215.0)               # bright border/foil stock
    arr[h // 8 : 7 * h // 8, w // 10 : 9 * w // 10] = 150.0   # art box
    arr[h // 4 : h // 2, w // 4 : 3 * w // 4] = 105.0         # artwork
    arr += rng.normal(0, 14, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def glare_curve(image=None, strengths=(0.0, 0.25, 0.5, 0.75, 1.0), seeds=range(48)):
    image = image or glossy_card()
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


@pytest.mark.parametrize("strength", [0.25, 0.5, 0.75])
def test_glare_estimate_recovers_a_known_strength_on_average(strength):
    # add_glare draws a random ellipse centre AND both radii, so a single
    # image's tail mass is one sample of a wide distribution.
    #
    # glossy_card()'s luma sits well above card_like()'s (mean ~170 vs a
    # 40-200 gradient that tops out exactly at GLARE_THRESHOLD), so the
    # ellipse registers wherever it lands and the round-trip is
    # well-conditioned across the whole range, not just the lower half.
    image = glossy_card()
    curve = glare_curve(image)
    recovered = [
        estimate_glare(add_glare(image, seed, strength), curve).strength
        for seed in range(500, 580)
    ]
    assert sum(recovered) / len(recovered) == pytest.approx(strength, abs=0.12)


def test_glare_estimate_is_stable_at_and_above_full_strength():
    # (a) is the real spec claim: add_glare clamps its fill with
    # min(strength, 1.0), so strength 1.0 and 2.5 produce BYTE-IDENTICAL
    # pixels -- estimate_glare must return the same .strength and the same
    # .saturated for both, for any seed. Reporting 2.5 would be inventing
    # information the pixels do not contain.
    #
    # This has to be checked directly against a curve rather than a single
    # gambled seed: at strength 1.0, 54.7% of seeds already read at or
    # above a mean-built curve's top point and 45.3% do not, so asserting
    # "seed 5 saturates" against a seed-averaged curve is a coin flip by
    # construction, on any fixture. (b) below checks the real clamp logic
    # instead, against a reading built to exceed the curve's top point.
    image = glossy_card()
    curve = glare_curve(image)
    for seed in range(12):
        at_one = estimate_glare(add_glare(image, seed, 1.0), curve)
        at_2_5 = estimate_glare(add_glare(image, seed, 2.5), curve)
        assert at_2_5.strength == pytest.approx(at_one.strength)
        assert at_2_5.saturated == at_one.saturated

    # (b) saturation fires deterministically once the descriptor reaches or
    # exceeds the curve's top calibrated point -- checked directly against
    # a reading constructed to exceed it, not against a gambled seed's
    # draw (a solid-white image has the maximum possible bright_tail_mass,
    # far past any calibrated top point).
    top_descriptor = curve.points[-1][1]
    all_white = Image.new("RGB", image.size, (255, 255, 255))
    assert bright_tail_mass(all_white) >= top_descriptor
    estimate = estimate_glare(all_white, curve)
    assert estimate.saturated is True
    assert estimate.strength == pytest.approx(1.0)


def test_glare_estimate_is_ordered_across_strengths():
    image = glossy_card()
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


def test_axis_medians_reports_count_and_saturation_beside_the_median():
    profiles = [
        DegradationProfile(
            jpeg=AxisEstimate(0.2), blur=AxisEstimate(0.1),
            perspective=AxisEstimate(None), glare=AxisEstimate(0.4),
        ),
        DegradationProfile(
            jpeg=AxisEstimate(0.6), blur=AxisEstimate(0.3),
            perspective=AxisEstimate(None), glare=AxisEstimate(1.0, saturated=True),
        ),
    ]
    medians = axis_medians(profiles)
    assert medians["jpeg"] == (pytest.approx(0.4), 2, 0)
    assert medians["blur"] == (pytest.approx(0.2), 2, 0)
    # Saturated entries are counted, never averaged: their true value is
    # unknown and folding them in as 1.0 would drag the median to the clamp.
    assert medians["glare"] == (pytest.approx(0.4), 1, 1)
    # Unmeasured is not zero.
    assert medians["perspective"] == (None, 0, 0)


def test_axis_medians_handles_an_axis_with_nothing_measured():
    profiles = [
        DegradationProfile(
            jpeg=AxisEstimate(None), blur=AxisEstimate(0.1),
            perspective=AxisEstimate(None), glare=AxisEstimate(None),
        )
    ]
    assert axis_medians(profiles)["jpeg"] == (None, 0, 0)


def test_main_refuses_an_empty_corpus_instead_of_printing_zeros(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.json").write_text(
        json.dumps({"entries": [], "discards": {}, "queries": []})
    )
    (corpus / "crops.json").write_text("{}")
    assert measure_main(["--corpus", str(corpus)]) == 1
    assert "no cropped corpus entries" in capsys.readouterr().out.lower()


def test_main_profiles_a_cropped_corpus_and_labels_the_stated_limit(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    (corpus / "images").mkdir(parents=True)
    photo = card_like(size=(480, 640))
    photo.save(corpus / "images" / "v1_1_0.jpg", quality=40)
    (corpus / "manifest.json").write_text(json.dumps({
        "entries": [{
            "item_id": "v1|1|0", "card_id": "base1-4",
            "image": "images/v1_1_0.jpg", "image_url": "u",
            "listing_url": "l", "aspects": {},
        }],
        "discards": {}, "queries": [],
    }))
    (corpus / "crops.json").write_text(json.dumps({
        "v1|1|0": [[40.0, 60.0], [430.0, 55.0], [438.0, 580.0], [46.0, 590.0]]
    }))

    assert measure_main(["--corpus", str(corpus)]) == 0
    out = capsys.readouterr().out
    for axis in ("jpeg", "blur", "perspective", "glare"):
        assert axis in out
    # The two labels the spec requires on every quoted number.
    assert "degrade.py" in out
    assert "not a physical measurement" in out.lower()


def test_main_skips_entries_the_crop_tool_marked_unusable(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    (corpus / "images").mkdir(parents=True)
    card_like(size=(480, 640)).save(corpus / "images" / "v1_1_0.jpg", quality=40)
    (corpus / "manifest.json").write_text(json.dumps({
        "entries": [{
            "item_id": "v1|1|0", "card_id": "base1-4",
            "image": "images/v1_1_0.jpg", "image_url": "u",
            "listing_url": "l", "aspects": {},
        }],
        "discards": {}, "queries": [],
    }))
    (corpus / "crops.json").write_text(json.dumps({
        "v1|1|0": [[40.0, 60.0], [430.0, 55.0], [438.0, 580.0], [46.0, 590.0]]
    }))
    (corpus / "skipped.json").write_text(json.dumps(["v1|1|0"]))

    assert measure_main(["--corpus", str(corpus)]) == 1
    assert "no cropped corpus entries" in capsys.readouterr().out.lower()
