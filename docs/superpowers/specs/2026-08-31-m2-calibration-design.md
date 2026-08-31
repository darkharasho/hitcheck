# M2 calibration: measure retrieval on real photographs, not a synthetic axis

**Date:** 2026-08-31
**Status:** Approved, not implemented
**Unblocks:** the M2 train/don't-train decision recorded as BLOCKED in
`docs/verification/2026-08-10-m2-zeroshot.md`

## The problem

M2 measured zero-shot DINOv2 retrieval against synthetically degraded catalog
images and produced a curve:

| strength | 0.0 | 0.1 | 0.2 | 0.3 | 0.5 | 1.0 | 1.5 |
|---|---|---|---|---|---|---|---|
| top-1 | 1.000 | 0.870 | 0.750 | 0.540 | 0.293 | 0.052 | 0.003 |

The verdict threshold is 0.90, so everything depends on where a real stream
frame sits on that axis — and nothing has ever measured that. The gate record's
own addendum concluded that `TRAIN_REQUIRED` should not be acted on, and that
the next step is calibration rather than training.

Two things make that axis worse than merely uncalibrated.

**It is a diagonal, not an axis.** `degrade()` drives four independent
transforms — perspective warp, glare, motion blur, JPEG — off one `strength`
scalar. Strength 1.0 stacks JPEG quality 15, a 15px blur kernel, ±29px corner
jitter and +105 glare simultaneously. That is a destroyed image, not a stream
frame. A real frame does not sit at a single strength; it sits at some
arbitrary point in a four-dimensional space that this diagonal may never pass
near.

**It flatters itself at every point, including zero.** `synthetic.py` queries
the gallery using degraded copies of *the very images already in the gallery* —
identical source pixels, lighting and crop. That is why strength 0.0 scores a
perfect 1.000: the eval is matching each image against itself. Real
identification never gets that. It matches a *different photograph* of the
same card against a catalog scan, and no value of `strength` simulates that
gap.

## Decision

Stop trying to locate real inputs on the synthetic axis as the primary move.
The strength scalar was only ever a proxy for real-world conditions; with a
corpus of real labelled photographs in hand, run the retrieval eval on those
directly and read the accuracy off it. That answers the M2 question without
the proxy.

Keep the axis measurement as a **bridge**, not a blocker. eBay seller
photographs are well-lit, static and high-resolution — meaningfully easier than
a compressed handheld stream frame — so a real-corpus number is an upper bound,
and shipping a decision off an upper bound invites a production surprise. The
axis estimators are what let us extrapolate from photograph conditions to
stream conditions before paying to hand-label a stream corpus, and they are
what quantify the photograph/frame delta once real frames exist.

Two halves, in that order:

- **Half A — corpus and eval.** Build a labelled corpus of real seller
  photographs; run the existing retrieval harness against it; report accuracy
  with an interval.
- **Half B — axis measurement.** Given any image, estimate independently where
  it sits on each of the four degradation axes, in `degrade.py`'s own units.

## Half A — the corpus

### Acquisition

eBay Browse API, production keyset (unblocked 2026-08-31 by the Marketplace
Account Deletion endpoint at `workers/ebay-account-deletion/`). Search by
set/era, filter on aspects, fetch item detail for Item Specifics, resolve to a
catalog `card_id`, download the image, store image and label in a manifest.

Structured Item Specifics — not listing titles — are the label source. Titles
are seller free text; parsing them would put label noise directly underneath
the accuracy number this whole exercise exists to produce.

Images resolve to `s-l1600` by URL suffix swap (measured: 734×1200 for a
listing whose summary URL was `s-l225` at 138×225).

### English only

The catalog is English-only. The first listing sampled during design was
Japanese: set `Sv2a: Pokemon Card 151`, number `201/165`. No such catalog row
exists — the catalog has English `151` with `Charizard ex` at numbers `6` and
`199`. Japanese prints share artwork with English ones but share no numbering,
and resolving them needs a separate Japanese card database.

Filter to `Language: English` and discard the rest. One generic query already
reports 22,179 matches, so corpus size is not the binding constraint and we can
afford to be strict.

### Label resolution

eBay gives free text (`Set: 151`, `Card Number: 201/165`, `Card Name: Charizard
ex`); the catalog wants `sv3pt5-199`. Resolution must strip `/total` from card
numbers (18,858 of 20,479 catalog numbers are bare digits) and fuzzy-match set
names.

This step is the accuracy-contaminating one: any mistake here surfaces as a
retrieval "miss" that is not one.

- Accept only unambiguous agreement on name, set and number. Discard everything
  else — never guess.
- Hand-verify a random 50 resolved entries to bound residual label noise.
- Report accuracy **with that bound attached**. An unbounded label-error rate
  sitting underneath the M2 verdict would make the number unusable for the
  decision it exists to settle.

Measured aspect coverage over 20 sampled graded listings: `Card Name` 17/20,
`Set` 17/20, `Card Number` 12/20, `Language` 15/20. Requiring all of
name/set/number costs roughly 40% of candidates.

### Crop

Catalog gallery images are tight card scans. eBay photographs are whole slabs
on a desk — angled, PSA label in frame, background around them. Embedding one
against the other measures domain mismatch rather than degradation tolerance,
and would read as a catastrophically bad M2 result for entirely the wrong
reason.

In the live app, M3's detector supplies this crop. M3 does not exist yet, so
the eval needs a stand-in: **hand-crop the first corpus**. It is a one-time
cost on a few hundred images, it removes a whole error source from a
measurement whose entire purpose is to be trusted, and it doubles as ground
truth for evaluating M3's detector later.

**The crop tool records a quadrilateral, not an axis-aligned box.** Half B's
perspective estimator reads corner deviation directly off that quad; recovering
it later would mean redoing the whole hand-crop pass.

Consequence to state in the write-up: this eval measures retrieval *given a
good crop*. That is the right thing to isolate, but the number is conditional
on M3 working.

## Half A — the eval

### Modules

- `hitcheck_trainer/corpus/ebay.py` — OAuth and Browse client. The only
  network-touching file.
- `hitcheck_trainer/corpus/resolve.py` — aspects to `card_id`. Pure, heavily
  tested.
- `hitcheck_trainer/corpus/manifest.py` — JSON store: image path, resolved id,
  provenance (itemId, aspects, listing URL).
- `hitcheck_trainer/eval/real.py` — mirrors `synthetic.py`, drawing queries
  from the manifest instead of degrading catalog images.

### Reuse, with one targeted refactor

`Embedder`, `CardIndex`, `build_index` and `score` carry over untouched —
`score` already takes `(true_id, ranked)`, exactly what a resolved corpus
produces.

`embed_in_chunks` does not fit: it is hardwired to
`image_path(images_root, card_id)` (`synthetic.py:56-75`), and corpus images
are keyed by eBay itemId. Generalize it to take `(label, path)` pairs, which
both callers can use.

Keeping one chunked-embedding implementation is deliberate. The 256-image
chunking exists because materialising the whole catalog once caused a global
OOM on this machine; a second loader must not quietly reintroduce that.

### Same gallery, deliberately

Real queries search the full 20,427-image index via `--reuse-index`, identical
to the synthetic run. Any difference in the resulting number is then
attributable to the queries, which is the point.

### Sample size and the inconclusive verdict

The threshold is 0.90 (`report.py:31`), so the eval must distinguish roughly
0.88 from 0.92. At p≈0.9 the standard error is √(p(1−p)/N): about 1.7% at
N=300, 1.3% at N=500.

Target **N ≥ 500** resolved, cropped queries.

`AccuracyReport` gains a confidence interval, and `verdict()` gains a third
outcome — **`INCONCLUSIVE`** — returned when the interval straddles the
threshold. Today `verdict()` is forced to choose at a boundary it cannot
actually resolve, so sampling noise can flip the decision silently. A verdict
flipped by noise is worse than no verdict.

## Half B — the axis measurement

`hitcheck_trainer/augment/measure.py`, sitting next to `degrade.py` as its
inverse. Output is a `DegradationProfile`: four independent strength-equivalents,
never one scalar, for the reason given at the top of this document.

| Axis | Method | Inverts against |
|---|---|---|
| JPEG | Read the quantization table from the file header, match against the scaled IJG standard table | `quality = 60 − 45·strength` |
| Blur | Re-blur ratio: apply a known probe blur, measure how far a sharpness descriptor moves | `k = int(3 + 12·strength) \| 1` |
| Perspective | Corner deviation of the recorded quad from its best-fit rectangle, normalised by image dimension | `shift = 0.12·strength` |
| Glare | Excess mass in the bright tail of the histogram | `fill = 190·strength` |

**JPEG needs no estimation** for the corpus — the header read is exact and
free, and eBay serves JPEG. Stream frames arrive as decoded H.264 with no
quantization table, so those fall back to a blockiness descriptor (8×8 boundary
discontinuity relative to within-block variation), inverted through a synthesis
curve.

### Two details of `degrade.py` that the inverse must respect

**Glare and JPEG saturate.** Both clamp with `min(strength, 1.0)`
(`degrade.py:101,138`), so above strength 1.0 they are flat and *not
invertible*. Those two estimators report `>= 1.0` rather than a point value
when they land at the clamp. Blur and perspective have no such clamp.

**Perspective jitter is random, not fixed.** `shift` bounds a uniform draw —
`rng.uniform(-shift, shift, (4, 2))` (`degrade.py:69-71`) — so a single image's
corner deviation is one sample, not the parameter. Estimate from the **maximum
absolute deviation across the four corners**, which approaches `shift` in
expectation, and treat a single-image estimate as noisy; it is only meaningful
averaged over a corpus.

**Blur kernel size is forced odd** (`int(3 + 12·strength) | 1`), so the inverse
resolves to a strength *interval* per kernel size, not a point. Report the
interval midpoint.

**Blur uses the re-blur ratio** rather than raw Laplacian variance. Sharp
images drop hard under a probe blur; already-blurred ones barely shift. The
ratio is content-robust in a way the raw value is not, which matters across a
corpus spanning plain commons and busy full-arts.

### Calibration curves by synthesis

For each axis, sweep its parameter across `degrade`'s range over a fixed sample
of catalog images, record the descriptor, and store the monotone
descriptor→parameter table as checked-in JSON with its generating script beside
it. Reproducible, no runtime cost, and native to `degrade.py`'s units by
construction.

### Testing

Half B is verifiable with no ground-truth corpus at all. Take a clean catalog
image, run `degrade` at a known strength, feed the result to the estimator,
assert it recovers that strength within tolerance. Round-trip across the full
strength range, per axis, offline against still images.

This satisfies the project's standing constraint that vision work stay testable
against stills with no screen, stream or network dependency — and it means
Half B can be built and trusted before the corpus exists.

### Stated limit

These estimators recover *`degrade.py`'s* parameters, not physical ground
truth. "This frame is at blur-strength 0.35" means "as blurry as `motion_blur`
at 0.35", not a claim about the real point-spread function. That is exactly the
number needed to read the M2 curve, and it must be labelled that way so it is
not later mistaken for a physical measurement.

## Sequencing

1. `report.py`: confidence interval and `INCONCLUSIVE` verdict. Small,
   independent, and everything downstream reports through it.
2. `corpus/`: eBay client, resolution, manifest.
3. Hand-crop pass, recording quads.
4. `eval/real.py`, and the `embed_in_chunks` refactor it needs.
5. Run; record a verification doc alongside `2026-08-10-m2-zeroshot.md`.
6. `measure.py` and its calibration curves.

Half B lands after the number, since it is the bridge to stream frames rather
than a blocker on the verdict. The consequence to accept: the first write-up
describes the corpus qualitatively as "seller photographs". It only gets to say
"these sit at blur-equivalent 0.1, JPEG-equivalent 0.15" once step 6 exists.

## Failure modes

- **Listing expiry.** Listings vanish; the corpus must survive them. Images and
  manifest are written once and never re-fetched. The eval reads from disk
  only. Reproducibility depends on this being strict.
- **Rate limits.** Browse allows ~5,000 calls/day; at two calls per item
  (search page plus detail) a 500-item corpus costs ~550. Comfortable, but it
  reuses the existing `catalog/backoff.py` rather than growing a second retry
  policy.
- **Ambiguous resolution, missing aspects, 404 images.** All discard, never
  guess. Every discard is counted, so the manifest states its own yield — a
  corpus that silently dropped 80% of candidates would skew toward whichever
  listings happen to have tidy Item Specifics.
- **Repo hygiene.** The manifest (itemIds, resolved ids, aspects) is small and
  checked in for reproducibility. The images are gitignored: they are sellers'
  copyrighted photographs, used locally for evaluation and not redistributed.

## Out of scope

- **Grade OCR and slab classes (M5).** Measured coverage: `Grade` on 1/20
  listings, `Certification Number` on 0/20. Grade will have to come from title
  parsing, which is a separate design.
- **Japanese cards.** Needs a Japanese card database.
- **Sold-listing history.** Requires the restricted Marketplace Insights API.
- **Any training decision.** This produces the number; it does not act on it.
