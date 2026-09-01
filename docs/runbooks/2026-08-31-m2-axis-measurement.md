# Runbook: measuring the M2 corpus on the degradation axes

Half B of `docs/superpowers/specs/2026-08-31-m2-calibration-design.md`.
Half A's runbook (`2026-08-31-m2-corpus.md`) produces the corpus and the
accuracy number; this one produces the sentence that says how degraded
that corpus actually was.

## What this gives you, and what it does not

You get four independent strength-equivalents per image, in `degrade.py`'s
own units, reduced to a corpus median per axis. That lets the M2 write-up
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
a recorded quad; the JPEG axis is unavailable only if the fallback also
fails, since a header-less image routes to the blockiness descriptor.

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

Record the medians in the M2 verification write-up next to the accuracy
number, with the "not a physical measurement" caveat attached. An
unlabelled axis number will be mistaken for one within a month.
