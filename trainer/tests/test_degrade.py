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


def _hard_edge(vertical, size=(120, 168)):
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    if vertical:
        arr[:, w // 2 :] = 255
    else:
        arr[h // 2 :, :] = 255
    return Image.fromarray(arr)


def _max_gradient(img_arr, vertical):
    gray = img_arr.astype(float).mean(axis=2)
    axis = 1 if vertical else 0
    return np.abs(np.diff(gray, axis=axis)).max()


def test_motion_blur_is_directional_not_isotropic():
    # A symmetric (isotropic) Gaussian blur, or an identity transform,
    # softens a vertical edge and a horizontal edge by the same amount.
    # A true directional motion blur smears one orientation much more
    # than the other, since it streaks along a single axis of movement.
    v_img = _hard_edge(vertical=True)
    h_img = _hard_edge(vertical=False)

    grad_v = _max_gradient(np.array(motion_blur(v_img, seed=1, strength=1.0)), vertical=True)
    grad_h = _max_gradient(np.array(motion_blur(h_img, seed=1, strength=1.0)), vertical=False)

    assert abs(grad_v - grad_h) / max(grad_v, grad_h) > 0.3


def test_motion_blur_direction_is_deterministic_for_a_given_seed():
    img = sample()
    a = motion_blur(img, seed=9, strength=1.0)
    b = motion_blur(img, seed=9, strength=1.0)
    assert np.array_equal(np.array(a), np.array(b))


def test_motion_blur_direction_differs_across_seeds():
    img = sample()
    a = motion_blur(img, seed=9, strength=1.0)
    b = motion_blur(img, seed=10, strength=1.0)
    assert not np.array_equal(np.array(a), np.array(b))


def test_jpeg_artifacts_change_pixels_but_keep_shape():
    img = sample()
    out = jpeg_artifacts(img, seed=1, strength=1.0)
    assert out.size == img.size
    assert not np.array_equal(np.array(out), np.array(img))


def test_warped_corners_is_the_identity_rectangle_at_zero_strength():
    from hitcheck_trainer.augment.degrade import warped_corners

    corners = warped_corners((120, 168), seed=4, strength=0.0)
    assert np.allclose(corners, [[0, 0], [120, 0], [120, 168], [0, 168]])


def test_warped_corners_stay_within_the_documented_jitter_bound():
    from hitcheck_trainer.augment.degrade import warped_corners

    w, h = 120, 168
    strength = 0.5
    corners = warped_corners((w, h), seed=11, strength=strength)
    rest = np.float64([[0, 0], [w, 0], [w, h], [0, h]])
    offset = np.abs(corners - rest) / np.float64([w, h])
    assert offset.max() <= 0.12 * strength + 1e-9


def test_warped_corners_is_deterministic_and_seed_dependent():
    from hitcheck_trainer.augment.degrade import warped_corners

    a = warped_corners((120, 168), seed=11, strength=0.5)
    b = warped_corners((120, 168), seed=11, strength=0.5)
    c = warped_corners((120, 168), seed=12, strength=0.5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
