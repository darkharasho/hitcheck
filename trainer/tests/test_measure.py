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
