import numpy as np
import pytest
from PIL import Image

from hitcheck_trainer.augment.calibrate import (
    BLUR_KERNELS,
    SWEEP_STRENGTHS,
    build_bundle,
    sample_catalog_images,
    sweep_blur,
    sweep_glare,
    sweep_jpeg_blockiness,
    sweep_perspective,
)


def card_like(size=(160, 224), seed=7):
    """Card-like, but with the luma distribution of a real scan.

    A flat 40-200 gradient tops out at exactly `GLARE_THRESHOLD` (200), so a
    glare ellipse crosses the threshold nowhere on some seeds and
    `bright_tail_mass` reads pure baseline. That is survivable at the 12
    seeds the glare sweep test averages over, but not at the 2 seeds
    `build_bundle` uses, where it ties the curve. Independently, this
    particular gradient also produces a non-monotone `blockiness` reading
    at this image size on this machine's libjpeg (see
    `.superpowers/sdd/2026-08-31-m2-calibration-axis/task-7-report.md`).
    `tests/test_measure.py`'s `glossy_card()` fixture (real-scan luma:
    mean 98-203, p95 213-244) avoids both -- adapted here at this file's
    smaller default size.
    """
    rng = np.random.default_rng(seed)
    w, h = size
    arr = np.full((h, w, 3), 215.0)  # bright border/foil stock
    arr[h // 8 : 7 * h // 8, w // 10 : 9 * w // 10] = 150.0  # art box
    arr[h // 4 : h // 2, w // 4 : 3 * w // 4] = 105.0  # artwork
    arr += rng.normal(0, 14, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def sample_images(n=3):
    return [card_like(seed=s) for s in range(n)]


def test_sweep_strengths_span_zero_to_one_inclusive():
    assert SWEEP_STRENGTHS[0] == 0.0
    assert SWEEP_STRENGTHS[-1] == 1.0
    assert list(SWEEP_STRENGTHS) == sorted(SWEEP_STRENGTHS)


def test_blur_kernels_are_the_reachable_odd_sizes_plus_the_no_blur_sentinel():
    # motion_blur: k = int(3 + 12 * strength) | 1, so 3..15 odd, and 1 is
    # the "strength <= 0, image untouched" case.
    assert BLUR_KERNELS == (1, 3, 5, 7, 9, 11, 13, 15)


def test_sweep_perspective_is_monotone_and_starts_at_zero():
    curve = sweep_perspective((240, 336), seeds=range(32))
    assert curve.parameter == "strength"
    assert curve.points[0] == (0.0, pytest.approx(0.0, abs=1e-12))
    assert curve.descriptors() == sorted(curve.descriptors())


def test_sweep_blur_covers_every_kernel_and_is_monotone():
    curve = sweep_blur(sample_images(), seeds=range(2))
    assert curve.parameter == "kernel"
    assert [int(v) for v, _ in curve.points] == list(BLUR_KERNELS)
    assert curve.descriptors() == sorted(curve.descriptors())


def test_sweep_glare_is_monotone_across_the_strength_range():
    curve = sweep_glare(sample_images(), seeds=range(12))
    assert curve.parameter == "strength"
    assert [v for v, _ in curve.points] == list(SWEEP_STRENGTHS)
    assert curve.descriptors() == sorted(curve.descriptors())


def test_sweep_jpeg_blockiness_is_monotone_across_the_strength_range():
    curve = sweep_jpeg_blockiness(sample_images())
    assert curve.parameter == "strength"
    assert curve.descriptors() == sorted(curve.descriptors())


def test_build_bundle_produces_exactly_the_four_curves_measure_asks_for():
    bundle = build_bundle(sample_images(), size=(240, 336), seeds=range(2))
    assert set(bundle.curves) == {"perspective", "blur", "glare", "jpeg_blockiness"}
    assert bundle.sample_images == 3
    assert bundle.seeds == 2
    assert "calibrate" in bundle.generated_by


def test_sample_catalog_images_is_deterministic_and_bounded(tmp_path):
    root = tmp_path / "images"
    for shard in ("base1", "base2"):
        (root / shard).mkdir(parents=True)
        for n in range(6):
            (root / shard / f"{shard}-{n}.png").write_bytes(b"")
    first = sample_catalog_images(str(root), count=5, seed=0)
    second = sample_catalog_images(str(root), count=5, seed=0)
    assert first == second
    assert len(first) == 5
    assert len(set(first)) == 5


def test_sample_catalog_images_returns_everything_when_asked_for_too_many(tmp_path):
    root = tmp_path / "images"
    (root / "base1").mkdir(parents=True)
    for n in range(3):
        (root / "base1" / f"base1-{n}.png").write_bytes(b"")
    assert len(sample_catalog_images(str(root), count=50, seed=0)) == 3


def test_sample_catalog_images_names_the_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError, match="no catalog images"):
        sample_catalog_images(str(tmp_path / "absent"), count=5, seed=0)
