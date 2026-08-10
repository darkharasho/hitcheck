"""Bridge the domain gap between clean catalog scans and live stream frames.

Catalog images are flatbed scans. Real input is a card at an angle, under
uneven light, possibly behind slab plastic, motion-blurred, and squeezed
through a streaming codec. Every transform here is seeded so evaluation
runs are reproducible.
"""

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def perspective_warp(image: Image.Image, seed: int, strength: float) -> Image.Image:
    """Simulate the card being held at an angle to the camera."""
    if strength <= 0:
        return image.convert("RGB")
    rng = _rng(seed)
    w, h = image.size
    shift = 0.12 * strength
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-shift, shift, (4, 2)) * np.float32([w, h])
    target = corners + jitter

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
    """Simulate the seller's hand moving while the frame is exposed."""
    if strength <= 0:
        return image.convert("RGB")
    radius = 0.4 + 1.4 * strength
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))


def jpeg_artifacts(image: Image.Image, seed: int, strength: float) -> Image.Image:
    """Simulate streaming codec compression."""
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
