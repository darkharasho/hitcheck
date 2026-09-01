"""Generate `curves.json` by sweeping `degrade.py` over catalog images.

Calibration by synthesis: drive each forward transform across its range,
record the descriptor at each setting, and store the monotone table. The
curves are native to `degrade.py`'s units by construction -- there is no
conversion factor anyone has to keep correct -- and they cost nothing at
runtime because the output is checked in.

Run this only when a descriptor or a forward transform changes. The
result is an artifact, not a build step:

    uv run python -m hitcheck_trainer.augment.calibrate

See docs/runbooks/2026-08-31-m2-axis-measurement.md.
"""

import argparse
import io
import os
import random
import statistics
import sys

from PIL import Image

from ..corpus.crops import CARD_SIZE
from .curves import DEFAULT_CURVES_PATH, Curve, CurveBundle, save_bundle
from .degrade import add_glare, motion_blur, warped_corners
from .descriptors import blockiness, bright_tail_mass, quad_corner_deviation, reblur_ratio

DEFAULT_IMAGES = "data/images"
DEFAULT_SAMPLE = 40
DEFAULT_SEEDS = 16

# Both ends inclusive: 0.0 anchors "undegraded" and 1.0 is where glare
# and JPEG clamp, which is what `curves.interpolate` reports saturation
# against. Spacing is 0.2 rather than 0.1 deliberately -- `Curve` REJECTS a
# non-monotone table, and the glare descriptor's per-seed spread is wider
# than a 0.1 step, so a finer sweep would need several times the seeds to
# stay monotone while buying accuracy that linear interpolation already
# supplies between these points.
SWEEP_STRENGTHS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# The only kernels `int(3 + 12 * strength) | 1` can produce, plus 1 for
# the strength<=0 case where motion_blur returns the image untouched.
BLUR_KERNELS = (1, 3, 5, 7, 9, 11, 13, 15)


def _strength_for_kernel_sweep(kernel: int) -> float:
    """A strength that lands squarely inside `kernel`'s interval.

    Numerically the same as `measure.strength_for_kernel`, but they are
    separate functions on purpose: this one needs a strength that PRODUCES
    the kernel, that one reports what a kernel IMPLIES about strength. The
    midpoint happens to serve both, and it is never on an interval
    boundary, so no rounding nudge is needed.
    """
    low = max(0.0, (kernel - 4) / 12.0)
    high = (kernel - 2) / 12.0
    return (low + high) / 2.0


def sweep_perspective(size: tuple[int, int], seeds) -> Curve:
    """Mean corner deviation per strength. Needs no images -- pure geometry.

    Averaged over seeds because `shift` bounds a uniform draw: one seed's
    deviation is a sample, and the curve has to describe the expectation.
    """
    seeds = list(seeds)
    points = []
    for strength in SWEEP_STRENGTHS:
        deviations = [
            quad_corner_deviation(warped_corners(size, seed, strength).tolist())
            for seed in seeds
        ]
        points.append((float(strength), sum(deviations) / len(deviations)))
    return Curve(name="perspective", parameter="strength", points=points)


def sweep_blur(images, seeds) -> Curve:
    """Mean re-blur ratio per kernel size, across images and seeds.

    Keyed by kernel, not strength: the forward parameter is quantised, so
    a strength-keyed table would imply a resolution the transform does
    not have.

    `reblur_ratio` returns None for an image with no Laplacian energy to
    divide by. A calibration sample of those is not a curve, so it fails
    loudly here rather than silently calibrating on whatever remains.
    """
    seeds = list(seeds)
    points = []
    for kernel in BLUR_KERNELS:
        ratios = []
        for image in images:
            if kernel == 1:
                ratios.append(reblur_ratio(image))
                continue
            strength = _strength_for_kernel_sweep(kernel)
            for seed in seeds:
                ratios.append(reblur_ratio(motion_blur(image, seed, strength)))
        measured = [r for r in ratios if r is not None]
        if not measured:
            raise ValueError(
                f"no measurable re-blur ratio at kernel {kernel}: every sample "
                "image is flat, so there is nothing to calibrate against"
            )
        points.append((float(kernel), sum(measured) / len(measured)))
    return Curve(name="blur", parameter="kernel", points=points)


def sweep_glare(images, seeds) -> Curve:
    """MEDIAN bright-tail mass per strength, across images and seeds.

    Seeds matter here more than anywhere else: `add_glare` randomises the
    ellipse's centre and both radii, so a single seed's tail mass varies
    by more than a strength step.

    Median, not mean, and it is the only sweep here that differs: the
    tail-mass distribution is strongly right-skewed (a handful of bright
    cards dominate the sum), so a mean-built curve sits above the typical
    image at every strength and `measure.estimate_glare` inverts biased
    low against it -- while `measure.axis_medians` aggregates the
    estimates by MEDIAN. Calibrating and reporting with the same statistic
    is the cheap half of that fix; the content term the other half would
    need is out of scope (see `measure.GLARE_CAVEAT`).
    """
    seeds = list(seeds)
    points = []
    for strength in SWEEP_STRENGTHS:
        masses = [
            bright_tail_mass(add_glare(image, seed, strength))
            for image in images
            for seed in seeds
        ]
        points.append((float(strength), statistics.median(masses)))
    return Curve(name="glare", parameter="strength", points=points)


def sweep_jpeg_blockiness(images) -> Curve:
    """Mean 8x8 blockiness per strength. Deterministic -- no seeds.

    Encodes through the same expression `degrade.jpeg_artifacts` uses so
    the curve tracks that function rather than a parallel one, and reads
    the descriptor off the DECODED pixels, which is the only thing a
    header-less stream frame offers.
    """
    points = []
    for strength in SWEEP_STRENGTHS:
        scores = []
        for image in images:
            if strength <= 0:
                scores.append(blockiness(image.convert("RGB")))
                continue
            quality = int(max(8, 60 - 45 * min(strength, 1.0)))
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            with Image.open(buffer) as encoded:
                scores.append(blockiness(encoded.convert("RGB")))
        points.append((float(strength), sum(scores) / len(scores)))
    return Curve(name="jpeg_blockiness", parameter="strength", points=points)


def build_bundle(images, size: tuple[int, int], seeds) -> CurveBundle:
    seeds = list(seeds)
    return CurveBundle(
        generated_by="hitcheck_trainer.augment.calibrate",
        sample_images=len(images),
        seeds=len(seeds),
        curves={
            "perspective": sweep_perspective(size, seeds),
            "blur": sweep_blur(images, seeds),
            "glare": sweep_glare(images, seeds),
            "jpeg_blockiness": sweep_jpeg_blockiness(images),
        },
    )


def sample_catalog_images(images_root: str, count: int, seed: int) -> list[str]:
    """A fixed, reproducible sample of catalog image paths.

    Sorted before sampling so the result depends on `seed` and nothing
    else -- filesystem walk order varies between machines and would make
    the checked-in curves unreproducible.
    """
    paths = sorted(
        os.path.join(root, name)
        for root, _, names in os.walk(images_root)
        for name in names
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not paths:
        raise FileNotFoundError(
            f"no catalog images under {images_root}. Run the catalog sync CLI first "
            "(see docs/runbooks/2026-08-31-m2-corpus.md)."
        )
    if count >= len(paths):
        return paths
    return sorted(random.Random(seed).sample(paths, count))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hitcheck-calibrate")
    parser.add_argument("--images", default=DEFAULT_IMAGES)
    parser.add_argument("--out", default=DEFAULT_CURVES_PATH)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help="catalog images to sweep over")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help="random draws per setting; glare and perspective "
                             "randomise, so this bounds curve noise")
    parser.add_argument("--seed", type=int, default=0, help="image-sample seed")
    args = parser.parse_args(argv)

    paths = sample_catalog_images(args.images, args.sample, args.seed)
    print(f"calibrating over {len(paths)} catalog images x {args.seeds} seeds...")
    images = []
    for path in paths:
        with Image.open(path) as handle:
            images.append(handle.convert("RGB").copy())

    # The crop the estimator actually runs on. `quad_corner_deviation`
    # normalises by the quad's own rectangle, so the perspective sweep is
    # exactly scale-invariant and this argument is inert today -- wired to
    # CARD_SIZE anyway so it stops reading like a coupling that isn't one,
    # and so it stays right if the descriptor ever stops normalising.
    bundle = build_bundle(images, size=CARD_SIZE, seeds=range(args.seeds))
    save_bundle(bundle, args.out)
    print(f"wrote {args.out}")
    for name, curve in sorted(bundle.curves.items()):
        span = f"{curve.descriptors()[0]:.4f} -> {curve.descriptors()[-1]:.4f}"
        print(f"  {name:16s} {curve.parameter:10s} {len(curve.points)} points  {span}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
