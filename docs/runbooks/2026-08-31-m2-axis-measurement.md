# Runbook: measuring the M2 corpus on the degradation axes

Half B of `docs/superpowers/specs/2026-08-31-m2-calibration-design.md`.
Half A's runbook (`2026-08-31-m2-corpus.md`) produces the corpus and the
accuracy number; this one produces the sentence that says how degraded
that corpus actually was.

## What this gives you, and what it does not

You get four independent strength-equivalents per image, in `degrade.py`'s
own units, reduced to a corpus median per axis. Three of them (JPEG, blur,
perspective) are comparable across sources; glare is not, and is marked
indicative-only in the output — see section 3. That lets the M2 write-up
say "these photographs sit at blur-equivalent 0.1, JPEG-equivalent 0.15"
instead of "seller photographs", and it is what lets a later stream corpus
be compared against this one on the same scale.

You do **not** get a physical measurement. "Blur-equivalent 0.35" means
"as blurry as `motion_blur` at strength 0.35". It says nothing about the
real point-spread function, and it must never be quoted as if it did.

You also do not get a single number. There isn't one. `degrade`'s
`strength` scalar drives four transforms at once, which makes it a
diagonal through a four-dimensional space rather than an axis; a real
frame sits at some arbitrary point in that space and may lie nowhere near
the diagonal. That is the whole reason this module exists.

## 1. Calibration curves

`trainer/hitcheck_trainer/augment/curves.json` is checked in. You only
need to regenerate it if a descriptor in `descriptors.py` or a forward
transform in `degrade.py` changes.

Regenerating needs `trainer/data/images/` populated — that is the catalog
image set, gitignored, ~3.2 GB. If it is missing, rebuild it with the
catalog sync CLI first (step 1 of `2026-08-31-m2-corpus.md`).

```bash
cd trainer
uv run python -m hitcheck_trainer.augment.calibrate
```

Defaults are 40 catalog images x 16 seeds and take several minutes — the
cost is `motion_blur`'s hand-rolled convolution at kernel 15, which is
pure numpy because there is no scipy in the venv. `--sample` and `--seeds`
trade runtime against curve noise.

The script prints each curve's descriptor span. If any span is nearly
flat, the curve cannot invert anything and the run is bad — do not commit
it. `tests/test_curves.py` asserts this, so a flat curve fails the suite
rather than shipping quietly.

Commit the regenerated `curves.json` in the same commit as whatever
change forced it.

## 2. Profile the corpus

```bash
cd trainer
uv run python -m hitcheck_trainer.augment.measure --corpus data/corpus
```

Reads Half A's `manifest.json`, `crops.json` and `skipped.json`. Entries
with no quad, entries the crop tool marked unusable, and entries whose
image is missing are all skipped, and the counts are printed so a short
run is visible rather than quietly shrinking the sample.

`data/corpus/` is the manual collection step from Half A's runbook — it
is not populated by this repository, and this CLI refuses to run against
an empty one rather than printing zeros. Build the corpus first
(`2026-08-31-m2-corpus.md`), then the hand-crop tool, before running this.

Output is one line per axis: median, how many images it was measured on,
and how many saturated.

## 3. Reading the output

**`saturated`** means the forward transform clamped. `add_glare` and
`jpeg_artifacts` both use `min(strength, 1.0)`, so above 1.0 they produce
identical pixels and the true value is unrecoverable. Those entries are
counted and excluded from the median rather than folded in as 1.0, which
would drag the median toward the clamp. A large saturated count on the
glare axis means the corpus has genuinely blown highlights, and the median
describes only the rest of it.

**`unavailable`** is not zero. The perspective axis is unavailable without
a recorded quad. The JPEG axis is unavailable when no un-resampled
`source` is supplied at all — the blockiness fallback reads discontinuity
across the 8x8 DCT grid, and a crop is a bicubic resample that destroys
that grid, so `profile_image` measures the source or declines rather than
returning a confidently wrong number off the crop. It never *renders*
"unavailable" from this CLI, though, because `measure.main` always opens
the file: a header-less corpus image routes to the fallback, and if the
fallback's descriptor itself fails (image smaller than two 8x8 blocks) it
raises `ValueError`, `measure.main` treats the whole entry as unreadable,
and the entry is dropped and counted in `failed` rather than profiled with
one missing axis. The blur axis is unavailable for a crop with no
Laplacian energy — a blank slab back, a black or blown-out frame.

**The glare axis is INDICATIVE ONLY. Do not quote it in the M2 write-up.**
The CLI marks it on the line and prints the caveat in full underneath.
`bright_tail_mass` counts luma above a fixed threshold, so it carries a
content term larger than the glare signal itself: measured on catalog
scans, undegraded baseline tail mass spans 0.0018–0.0346 while the entire
calibrated glare curve spans 0.0285–0.0513, and two of twelve
*undegraded* images already estimate glare 0.09 and 0.26. A card on a
white desk reads glared; a dim card reads clean at any glare. That makes
the number comparable only between images from the same source under the
same lighting — never across sources, and never against a later stream
corpus. Separating the content term needs a per-image undegraded
reference, which a real stream frame does not have, so the axis stays
deliberately under-informative rather than quotably wrong.

**Perspective never saturates**, unlike glare and JPEG. `perspective_warp`
has no `min(strength, 1.0)` clamp — `shift = 0.12 * strength` is
unbounded — so above the last calibrated point the estimator extrapolates
the curve's last segment and can legitimately report a strength above 1.0.
A perspective reading above 1.0 is a real reading, not an overflow. The
descriptor is also blind to in-plane rotation: a slab laid crooked on a
desk is not a perspective degradation and is de-rotated out before
measurement.

**Perspective is noisy per image.** `perspective_warp` draws its corner
jitter from a uniform distribution (`degrade.py:71`), so one image's
deviation is one sample and not the parameter. Only the corpus median
means anything. Never quote a single photograph's perspective number.

**The JPEG axis is exact when the header survives** and approximate when
it does not. eBay serves JPEG, so the corpus takes the exact path; stream
frames arrive as decoded H.264 with no quantization table and take the
blockiness fallback, which is calibrated rather than exact and is
confounded by blur applied after compression.

## 4. Using the numbers

Read the medians against the synthetic curve in
`docs/verification/2026-08-10-m2-zeroshot.md`. If the corpus profiles at,
say, blur-equivalent 0.1 while the synthetic curve was swept at 0.3, the
real-corpus accuracy number is being produced under materially gentler
conditions than that point on the curve, and the gap between them is the
margin a stream frame has to eat into.

Record the JPEG, blur and perspective medians in the M2 verification
write-up next to the accuracy number, with the "not a physical
measurement" caveat attached. An unlabelled axis number will be mistaken
for one within a month. **Leave glare out**, or carry its
indicative-only caveat verbatim — see section 3.
