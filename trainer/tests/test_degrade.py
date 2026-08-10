import numpy as np
from PIL import Image

from hitcheck_trainer.augment.degrade import (
    add_glare,
    degrade,
    jpeg_artifacts,
    motion_blur,
    perspective_warp,
)


def sample(size=(120, 168)):
    rng = np.random.default_rng(7)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


def test_degrade_is_deterministic_for_a_given_seed():
    a = degrade(sample(), seed=42)
    b = degrade(sample(), seed=42)
    assert np.array_equal(np.array(a), np.array(b))


def test_different_seeds_give_different_output():
    a = degrade(sample(), seed=1)
    b = degrade(sample(), seed=2)
    assert not np.array_equal(np.array(a), np.array(b))


def test_degrade_preserves_size_and_mode():
    img = sample()
    out = degrade(img, seed=3)
    assert out.size == img.size
    assert out.mode == "RGB"


def test_zero_strength_is_close_to_identity():
    img = sample()
    out = degrade(img, seed=5, strength=0.0)
    diff = np.abs(np.array(out, dtype=int) - np.array(img, dtype=int)).mean()
    assert diff < 2.0


def test_perspective_warp_changes_pixels():
    img = sample()
    assert not np.array_equal(np.array(perspective_warp(img, seed=1, strength=1.0)), np.array(img))


def test_glare_brightens_somewhere():
    img = Image.new("RGB", (120, 168), (40, 40, 40))
    out = np.array(add_glare(img, seed=1, strength=1.0), dtype=int)
    assert out.max() > 40


def test_motion_blur_reduces_local_variance():
    img = sample()
    before = np.array(img, dtype=float).var()
    after = np.array(motion_blur(img, seed=1, strength=1.0), dtype=float).var()
    assert after < before


def test_jpeg_artifacts_change_pixels_but_keep_shape():
    img = sample()
    out = jpeg_artifacts(img, seed=1, strength=1.0)
    assert out.size == img.size
    assert not np.array_equal(np.array(out), np.array(img))
