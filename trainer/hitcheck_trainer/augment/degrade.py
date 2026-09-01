"""Bridge the domain gap between clean catalog scans and live stream frames.

Catalog images are flatbed scans. Real input is a card at an angle, under
uneven light, possibly behind slab plastic, motion-blurred, and squeezed
through a streaming codec. Every transform here is seeded so evaluation
runs are reproducible.
"""

import io

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw, ImageFilter


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _line_kernel(size: int, angle: float) -> np.ndarray:
    """Rasterize a normalized 1-pixel-wide line kernel at ``angle`` radians.

    Anti-aliased via bilinear splatting so the kernel isn't just a jagged
    stair-step at non-45-degree angles.
    """
    half = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)
    n_samples = size * 4
    dx, dy = np.cos(angle), np.sin(angle)
    for t in np.linspace(-half, half, n_samples):
        px, py = t * dx + half, t * dy + half
        x0, y0 = int(np.floor(px)), int(np.floor(py))
        wx, wy = px - x0, py - y0
        for xi, yi, w in (
            (x0, y0, (1 - wx) * (1 - wy)),
            (x0 + 1, y0, wx * (1 - wy)),
            (x0, y0 + 1, (1 - wx) * wy),
            (x0 + 1, y0 + 1, wx * wy),
        ):
            if 0 <= xi < size and 0 <= yi < size:
                kernel[yi, xi] += w
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return kernel


def _convolve_same(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution per channel, edge-padded to preserve size.

    Pure numpy (no scipy in the venv). PIL's ``ImageFilter.Kernel`` only
    supports 3x3/5x5 on the installed Pillow (12.3.0; kernels >=7x7 raise
    "bad kernel size"), so a directional kernel of the size needed here
    can't go through PIL and is done by hand via a sliding-window view.
    """
    k = kernel.shape[0]
    pad = k // 2
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    windows = sliding_window_view(padded, (k, k), axis=(0, 1))
    return np.tensordot(windows, kernel, axes=([3, 4], [0, 1]))


def warped_corners(size: tuple[int, int], seed: int, strength: float) -> np.ndarray:
    """Where `perspective_warp` sends the source rectangle's four corners.

    Order is [top-left, top-right, bottom-right, bottom-left], matching
    `corpus.crops.Quad`. `shift` bounds a UNIFORM DRAW, so this is one
    sample of the jitter distribution and not the parameter itself --
    anything inverting it must average over seeds.

    Exposed so `augment.measure` and `augment.calibrate` can test and
    calibrate the perspective inverse against the forward model rather
    than against a second copy of this arithmetic.
    """
    w, h = size
    corners = np.float64([[0, 0], [w, 0], [w, h], [0, h]])
    if strength <= 0:
        return corners
    shift = 0.12 * strength
    jitter = _rng(seed).uniform(-shift, shift, (4, 2)) * np.float64([w, h])
    return corners + jitter


def perspective_warp(image: Image.Image, seed: int, strength: float) -> Image.Image:
    """Simulate the card being held at an angle to the camera."""
    if strength <= 0:
        return image.convert("RGB")
    w, h = image.size
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    target = warped_corners((w, h), seed, strength)

    # Solve the 8-DOF transform mapping target -> source, which is the
    # direction PIL's PERSPECTIVE transform expects.
    matrix = []
    for (tx, ty), (sx, sy) in zip(target, corners):
        matrix.append([tx, ty, 1, 0, 0, 0, -sx * tx, -sx * ty])
        matrix.append([0, 0, 0, tx, ty, 1, -sy * tx, -sy * ty])
    a = np.array(matrix, dtype=np.float64)
    b = corners.reshape(8).astype(np.float64)
    coeffs = np.linalg.lstsq(a, b, rcond=None)[0]

    return image.convert("RGB").transform(
        (w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC
    )


def add_glare(image: Image.Image, seed: int, strength: float) -> Image.Image:
    """Simulate a specular highlight off slab plastic or a card sleeve."""
    if strength <= 0:
        return image.convert("RGB")
    rng = _rng(seed + 1)
    base = image.convert("RGB")
    w, h = base.size

    layer = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(layer)
    cx, cy = rng.uniform(0.15, 0.85, 2) * np.array([w, h])
    rx, ry = rng.uniform(0.15, 0.4, 2) * np.array([w, h])
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=int(190 * min(strength, 1.0)))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=max(w, h) * 0.06))

    arr = np.array(base, dtype=np.float32)
    mask = np.array(layer, dtype=np.float32)[..., None] / 255.0
    return Image.fromarray(np.clip(arr + mask * 255.0 * 0.55, 0, 255).astype(np.uint8))


def motion_blur(image: Image.Image, seed: int, strength: float) -> Image.Image:
    """Simulate the seller's hand moving while the frame is exposed.

    Directional (not a symmetric Gaussian): real hand-shake smears detail
    along one axis of movement while leaving the perpendicular axis
    comparatively sharp. Angle and kernel size are seeded from
    ``seed + 2``, following the ``seed`` / ``seed + 1`` convention used by
    ``perspective_warp`` and ``add_glare``.
    """
    if strength <= 0:
        return image.convert("RGB")
    rng = _rng(seed + 2)
    angle = rng.uniform(0, np.pi)
    k = int(3 + 12 * strength) | 1
    kernel = _line_kernel(k, angle)
    arr = np.array(image.convert("RGB"), dtype=np.float64)
    blurred = _convolve_same(arr, kernel)
    return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))


def jpeg_artifacts(image: Image.Image, seed: int, strength: float) -> Image.Image:
    """Simulate streaming codec compression.

    ``seed`` is accepted for interface consistency with the other
    transforms but unused: JPEG re-encoding at a given quality is already
    deterministic, so there is nothing here to seed.
    """
    if strength <= 0:
        return image.convert("RGB")
    quality = int(max(8, 60 - 45 * min(strength, 1.0)))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def degrade(image: Image.Image, seed: int, strength: float = 1.0) -> Image.Image:
    """Apply the full degradation chain in capture order."""
    out = perspective_warp(image, seed, strength)
    out = add_glare(out, seed, strength)
    out = motion_blur(out, seed, strength)
    out = jpeg_artifacts(out, seed, strength)
    return out
