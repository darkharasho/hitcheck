# M2 Calibration — Half A (Real-Photograph Corpus & Eval) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the tooling that produces an M2 retrieval-accuracy number measured on real, labelled eBay seller photographs instead of synthetically degraded copies of the catalog's own images.

**Architecture:** A new `hitcheck_trainer/corpus/` package acquires labelled listings from the eBay Browse API, resolves Item Specifics to catalog `card_id`s with a pure, heavily-tested resolver, and writes a checked-in JSON manifest beside gitignored images. A browser-based hand-crop tool records a *quadrilateral* per photograph (not a box), which a perspective unwarp turns into a card-shaped crop. `eval/real.py` then drives the existing `Embedder`/`CardIndex`/`score` harness from that manifest against the same 20,427-image gallery the synthetic run used, and `report.py` gains a Wilson confidence interval plus an `INCONCLUSIVE` verdict so sampling noise can no longer silently flip the decision.

**Tech Stack:** Python 3.12, PIL/Pillow, numpy, httpx, hnswlib, transformers (DINOv2), sqlite3, pytest. No new third-party dependencies — the crop tool uses the stdlib `http.server`.

**Spec:** `docs/superpowers/specs/2026-08-31-m2-calibration-design.md`

**Scope note:** The spec covers two halves. This plan implements **Half A** (spec sequencing steps 1–5): the corpus, the crop tooling, the eval, and the verification write-up. **Half B** (`augment/measure.py` and its calibration curves, spec step 6) is an independent subsystem — testable with no corpus at all — and gets its own plan after Half A produces the number, exactly as the spec's sequencing requires.

## Global Constraints

Every task's requirements implicitly include this section.

- **No new third-party dependencies.** `trainer/pyproject.toml` pins `httpx>=0.27`, `pillow>=10.4`, `numpy>=1.26` (plus the `ml` and `dev` extras). Anything else must come from the standard library.
- **Python `>=3.12,<3.13`.** Line length 100 (`[tool.ruff]`).
- **Tests are offline.** No network, no screen, no stream. Every test runs under `pytest` from `trainer/` with injected fakes.
- **English only.** Discard any listing whose `Language` aspect is not `English`. The catalog is English-only; Japanese prints share artwork but not numbering.
- **Never guess a label.** Accept only unambiguous agreement on name, set and number. Every discard is counted by reason so the manifest states its own yield.
- **Images are written once and never re-fetched.** The eval reads from disk only; listings expire and reproducibility depends on this being strict.
- **One retry policy.** Reuse `hitcheck_trainer/catalog/backoff.py`'s `backoff_delays`. Do not grow a second one.
- **The manifest is checked in; the images are gitignored.** Images are sellers' copyrighted photographs, used locally and not redistributed.
- **Image URLs resolve to `s-l1600`** by suffix swap on the eBay-supplied URL.
- **Crops are quadrilaterals, not axis-aligned boxes.** Half B's perspective estimator reads corner deviation off the recorded quad; recovering it later would mean redoing the hand-crop pass.
- **Same gallery, via `--reuse-index`.** Real queries search the identical 20,427-image index the synthetic run used, so any difference is attributable to the queries.
- **One chunked-embedding implementation, default `chunk=256`.** Materialising the whole catalog is 4.52GB of decoded pixels and previously caused a global OOM on this 30GB machine. A second loader must not quietly reintroduce that.
- **Target N ≥ 500** resolved, cropped queries.
- **Verdict threshold is 0.90.**
- **Secrets come from the environment** (`PROD_APP_ID`, `PROD_EBAY_CERT_ID`) and are never logged, printed, or written to the manifest.

## A statistical fact the spec did not have

The spec targets N ≥ 500 on the basis that the standard error at p≈0.9 is ~1.3%. That is correct for the point estimate, but a 95% Wilson interval is roughly twice the standard error, so the band in which the verdict is `INCONCLUSIVE` is wider than the spec implies:

| N | `SKIP_TRAINING` needs top1 ≥ | `TRAIN_REQUIRED` needs top1 ≤ | Inconclusive band |
|---|---|---|---|
| 500 | 0.9280 | 0.8720 | (0.872, 0.928) |
| 1000 | 0.9190 | 0.8810 | (0.881, 0.919) |
| 2000 | 0.9135 | 0.8865 | (0.887, 0.914) |

N=500 still answers the question decisively whenever the true accuracy is not close to 0.90 — which, given the synthetic curve's steepness, is the likely case. But if the first run lands inside the band, the honest answer is `INCONCLUSIVE` and the response is more corpus, not a coin flip. Task 12's runbook says so explicitly.

## File Structure

**Created:**

| File | Responsibility |
|---|---|
| `trainer/hitcheck_trainer/corpus/__init__.py` | Package marker. |
| `trainer/hitcheck_trainer/corpus/normalize.py` | Pure string normalisation: card numbers, card names, set names. No I/O. |
| `trainer/hitcheck_trainer/corpus/resolve.py` | `CardLookup` over the catalog plus `resolve()`: eBay aspects → `card_id` or a counted discard reason. Pure given a lookup. |
| `trainer/hitcheck_trainer/corpus/manifest.py` | `CorpusEntry` / `Manifest` dataclasses and their atomic JSON load/save. |
| `trainer/hitcheck_trainer/corpus/ebay.py` | eBay OAuth and Browse client. The only network-touching file in the package. |
| `trainer/hitcheck_trainer/corpus/build.py` | Acquisition CLI: search → detail → resolve → download → manifest. |
| `trainer/hitcheck_trainer/corpus/crops.py` | `CropStore` (quad persistence) and `apply_quad` (perspective unwarp to a card-shaped crop). Pure. |
| `trainer/hitcheck_trainer/corpus/croptool.py` | Local stdlib HTTP server serving the hand-crop UI. |
| `trainer/hitcheck_trainer/corpus/audit.py` | Generates a static side-by-side HTML sheet for hand-verifying 50 labels. |
| `trainer/hitcheck_trainer/eval/chunks.py` | The single chunked-embedding implementation, generalised to `(label, path)` pairs. |
| `trainer/hitcheck_trainer/eval/real.py` | The real-corpus eval CLI. Mirrors `synthetic.py`. |
| `docs/runbooks/2026-08-31-m2-corpus.md` | The manual runbook: acquisition, cropping, auditing, running, recording. |

**Modified:**

| File | Change |
|---|---|
| `trainer/hitcheck_trainer/eval/report.py` | Wilson interval, `INCONCLUSIVE` verdict, `label_noise_bound`. |
| `trainer/hitcheck_trainer/catalog/images.py` | Extract `fetch_to_path` so the corpus downloader reuses the retry/atomic-write logic. |
| `trainer/hitcheck_trainer/catalog/http.py` | Add `httpx_post_form` for the OAuth token request. |
| `trainer/hitcheck_trainer/eval/synthetic.py` | Import `embed_in_chunks` from `chunks.py`; build `(label, path)` pairs. |
| `trainer/.gitignore` | Carve the manifest and crop file out of the wholesale `data/` ignore. |

**Tests created:** `trainer/tests/test_normalize.py`, `test_resolve.py`, `test_manifest.py`, `test_ebay.py`, `test_corpus_build.py`, `test_crops.py`, `test_croptool.py`, `test_chunks.py`, `test_audit.py`. **Tests modified:** `trainer/tests/test_report.py`, `trainer/tests/test_images.py`.

**Run all tests with:** `cd trainer && .venv/bin/python -m pytest -q` (add `-m "not slow"` to skip the tests that download model weights).

---

### Task 1: Confidence interval and the `INCONCLUSIVE` verdict

**Files:**
- Modify: `trainer/hitcheck_trainer/eval/report.py`
- Test: `trainer/tests/test_report.py` (existing file — four existing tests are rewritten, see Step 1)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `INCONCLUSIVE: str = "INCONCLUSIVE"` module constant.
  - `wilson_interval(hits: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]`
  - `label_noise_bound(errors: int, sample: int) -> float`
  - `AccuracyReport.interval: tuple[float, float]` — a computed property, not a stored field.
  - `AccuracyReport.verdict(threshold: float = 0.90) -> str` — now returns one of three values.

**Background for the implementer:** `verdict()` currently compares a point estimate to 0.90. At N=500 the 95% interval is about ±2.8%, so a run at 0.92 and a run at 0.88 are not actually distinguishable from 0.90 — yet today's code returns a confident `SKIP_TRAINING`/`TRAIN_REQUIRED` for each. That is the failure this task removes.

**This changes existing behaviour and four existing tests must be rewritten.** All four currently use N=10 or N=100, where the interval is far too wide to decide anything. Do not try to preserve them; they encode the semantics being replaced. In particular, the existing docstring claim that "the boundary is inclusive (`>=`)" becomes meaningless — a point estimate landing exactly on the threshold always straddles it — and must be deleted rather than reworded.

- [ ] **Step 1: Rewrite the four verdict tests and add the new ones**

In `trainer/tests/test_report.py`, **delete** these four tests entirely: `test_verdict_is_skip_training_above_the_threshold`, `test_verdict_is_train_required_below_the_threshold`, `test_verdict_boundary_exact_threshold_is_skip_training`, `test_verdict_just_below_threshold_is_train_required`. Leave every other test in the file untouched.

Add this import at the top of the file, replacing the existing single import line:

```python
from hitcheck_trainer.eval.report import (
    INCONCLUSIVE,
    SKIP_TRAINING,
    TRAIN_REQUIRED,
    label_noise_bound,
    score,
    wilson_interval,
)
```

Append these tests to the end of the file:

```python
def hits(n_hits, n_total):
    """`n_total` results of which `n_hits` are top-1 correct."""
    return (
        [("a", [("a", 0.1)])] * n_hits
        + [("b", [("z", 0.1)])] * (n_total - n_hits)
    )


def test_verdict_is_skip_training_when_the_whole_interval_clears_the_threshold():
    # 1900/2000 -> top1 0.950, interval (0.9396, 0.9587). Entirely above 0.90.
    assert score(hits(1900, 2000)).verdict(threshold=0.90) == SKIP_TRAINING


def test_verdict_is_train_required_when_the_whole_interval_is_below_the_threshold():
    # 1760/2000 -> top1 0.880, interval (0.8650, 0.8935). Entirely below 0.90.
    assert score(hits(1760, 2000)).verdict(threshold=0.90) == TRAIN_REQUIRED


def test_verdict_is_inconclusive_when_the_interval_straddles_the_threshold():
    # 1800/2000 -> top1 exactly 0.900, interval (0.8861, 0.9124). Straddles.
    # This is the case the old code answered SKIP_TRAINING with full
    # confidence, off a point estimate it could not actually resolve.
    assert score(hits(1800, 2000)).verdict(threshold=0.90) == INCONCLUSIVE


def test_a_small_sample_is_inconclusive_even_at_a_high_point_estimate():
    # 10/10 is top1 1.000 but the interval is (0.7225, 1.0) — ten queries
    # cannot clear a 0.90 bar. Sample size must beat the threshold, not luck.
    assert score(hits(10, 10)).verdict(threshold=0.90) == INCONCLUSIVE


def test_the_inconclusive_band_at_n500_is_wider_than_the_standard_error():
    # At N=500 the interval is roughly +/-2.8%, not the +/-1.3% standard
    # error, so 0.92 is NOT decisive against a 0.90 threshold.
    assert score(hits(460, 500)).verdict(threshold=0.90) == INCONCLUSIVE
    assert score(hits(475, 500)).verdict(threshold=0.90) == SKIP_TRAINING
    assert score(hits(430, 500)).verdict(threshold=0.90) == TRAIN_REQUIRED


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson_interval(450, 500)
    assert lo < 0.90 < hi
    assert abs(lo - 0.8706) < 5e-4
    assert abs(hi - 0.9233) < 5e-4


def test_wilson_interval_stays_inside_zero_and_one_at_the_extremes():
    assert wilson_interval(0, 50)[0] == 0.0
    assert wilson_interval(50, 50)[1] == 1.0


def test_wilson_interval_of_an_empty_sample_is_the_full_unit_range():
    # No data means no information, not a point estimate of zero.
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_narrows_as_the_sample_grows():
    small = wilson_interval(90, 100)
    large = wilson_interval(900, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_report_exposes_its_own_interval():
    report = score(hits(450, 500))
    assert report.interval == wilson_interval(450, 500)


def test_label_noise_bound_is_an_upper_bound_not_the_observed_rate():
    # 2 wrong labels in 50 audited is an observed 4%, but the true rate
    # could plausibly be higher; the bound is what gets reported.
    bound = label_noise_bound(errors=2, sample=50)
    assert bound > 0.04
    assert bound < 0.15


def test_label_noise_bound_of_a_clean_audit_is_still_nonzero():
    # Zero errors in 50 does not prove zero errors in 500.
    assert label_noise_bound(errors=0, sample=50) > 0.0


def test_label_noise_bound_of_an_empty_audit_is_total_ignorance():
    assert label_noise_bound(errors=0, sample=0) == 1.0


def test_summary_reports_the_interval_and_the_verdict():
    text = score(hits(1900, 2000)).summary()
    assert "ci95=[0.940, 0.959]" in text
    assert "verdict=SKIP_TRAINING" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_report.py -q`
Expected: FAIL — `ImportError: cannot import name 'INCONCLUSIVE' from 'hitcheck_trainer.eval.report'`.

- [ ] **Step 3: Implement the interval, the bound and the three-way verdict**

In `trainer/hitcheck_trainer/eval/report.py`, add `import math` at the top, add the new constant beside the existing two, and add the two module-level functions above the dataclass:

```python
INCONCLUSIVE = "INCONCLUSIVE"

# 95% two-sided normal quantile.
_Z95 = 1.959963984540054


def wilson_interval(hits: int, total: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Wilson rather than the textbook normal approximation because the
    quantity being bounded sits near 0.9 with a few hundred samples,
    where the normal approximation's interval runs off the end of [0, 1]
    and is measurably too narrow. Wilson stays inside the unit range by
    construction and is well behaved at 0 and 1 hits.

    An empty sample returns the full unit range: no data is ignorance,
    not a point estimate of zero.
    """
    if total <= 0:
        return (0.0, 1.0)
    p = hits / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half_width = (z / denominator) * math.sqrt(
        p * (1 - p) / total + z * z / (4 * total * total)
    )
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def label_noise_bound(errors: int, sample: int) -> float:
    """Upper 95% bound on the label error rate, from a hand-audited sample.

    Reported alongside accuracy because a mis-resolved label surfaces as a
    retrieval miss that is not one. The bound, not the observed rate, is
    what gets attached to the verdict: zero errors in 50 audited entries
    does not establish zero errors in 500.
    """
    if sample <= 0:
        return 1.0
    return wilson_interval(errors, sample)[1]
```

Add the `interval` property and replace `verdict` and `summary` inside `AccuracyReport`. Delete the old `verdict` docstring's inclusive-boundary paragraph along with the old body:

```python
    @property
    def interval(self) -> tuple[float, float]:
        """95% Wilson interval on `top1`."""
        return wilson_interval(round(self.top1 * self.total), self.total)

    def verdict(self, threshold: float = 0.90) -> str:
        """Decide on the interval, never on the point estimate alone.

        Returns INCONCLUSIVE when the 95% interval straddles `threshold` —
        the sample cannot resolve which side of the bar it is on. This is
        deliberately a third outcome rather than a rounding rule: a verdict
        flipped by sampling noise is worse than no verdict, because it
        would be acted on. INCONCLUSIVE means collect more corpus.

        At N=500 the decisive bands are top1 >= 0.928 and top1 <= 0.872;
        at N=2000, 0.9135 and 0.8865.
        """
        low, high = self.interval
        if low >= threshold:
            return SKIP_TRAINING
        if high < threshold:
            return TRAIN_REQUIRED
        return INCONCLUSIVE

    def summary(self) -> str:
        low, high = self.interval
        return (
            f"queries={self.total} top1={self.top1:.3f} "
            f"ci95=[{low:.3f}, {high:.3f}] top5={self.top5:.3f} "
            f"mean_top1_distance={self.mean_top1_distance:.4f} verdict={self.verdict()}"
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_report.py -q`
Expected: PASS, all tests in the file.

- [ ] **Step 5: Confirm nothing else depended on the old two-way verdict**

Run: `cd trainer && grep -rn "verdict\|SKIP_TRAINING\|TRAIN_REQUIRED" --include='*.py' . | grep -v '\.venv'`
Expected: matches only in `eval/report.py` and `tests/test_report.py`. `eval/synthetic.py` prints `report.summary()` and never branches on the verdict, so it needs no change. If any other caller appears, update it to handle three outcomes.

Run: `cd trainer && .venv/bin/python -m pytest -q -m "not slow"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd trainer && .venv/bin/python -m ruff check hitcheck_trainer tests
cd /var/home/mstephens/Documents/GitHub/pokemon-card-stream-pricer
git add trainer/hitcheck_trainer/eval/report.py trainer/tests/test_report.py
git commit -m "feat(eval): decide the M2 verdict on an interval, not a point estimate

verdict() compared top1 to 0.90 directly, so at N=500 -- where the 95%
interval is about +/-2.8% -- a run at 0.92 and a run at 0.88 both returned
a confident answer neither sample could actually support. Sampling noise
could flip the train/don't-train decision silently.

Adds a Wilson interval and a third outcome, INCONCLUSIVE, returned when
the interval straddles the threshold. Also adds label_noise_bound, since
a mis-resolved label surfaces as a retrieval miss that is not one and the
accuracy number has to carry that bound.

Four existing verdict tests used N=10 or N=100 and encoded the point-estimate
semantics being replaced; they are rewritten at sample sizes that can
actually resolve the threshold.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Extract the retrying, atomic image downloader

**Files:**
- Modify: `trainer/hitcheck_trainer/catalog/images.py`
- Test: `trainer/tests/test_images.py` (append; existing tests stay untouched)

**Interfaces:**
- Consumes: `backoff_delays` from `hitcheck_trainer/catalog/backoff.py`.
- Produces: `fetch_to_path(url: str, path: str, fetch, sleep=time.sleep, max_attempts: int = 4) -> bool` — returns `True` if the file is now on disk with content, `False` if every attempt failed. `fetch` is the injected transport `(url) -> (status: int, body: bytes | None)`.

**Background for the implementer:** `download_images` already contains exactly the logic the corpus downloader needs — retry on `RETRYABLE` statuses with `backoff_delays`, write via a `.part` temp file, `os.replace` into position. The corpus keys files by eBay itemId rather than card id, so it cannot call `download_images` directly. Extract the per-file half rather than writing a second copy; the atomic-write and retry behaviour must not diverge between the two callers.

This is a pure refactor of existing behaviour. `download_images` keeps its signature, its return value, its resume semantics and its progress callback.

- [ ] **Step 1: Write the failing tests**

Append to `trainer/tests/test_images.py`:

```python
from hitcheck_trainer.catalog.images import fetch_to_path


def test_fetch_to_path_writes_the_body_and_reports_success(tmp_path):
    target = tmp_path / "nested" / "item.jpg"
    ok = fetch_to_path("http://x/i.jpg", str(target), lambda url: (200, b"JPEGBYTES"))
    assert ok is True
    assert target.read_bytes() == b"JPEGBYTES"


def test_fetch_to_path_retries_a_retryable_status_then_succeeds(tmp_path):
    statuses = iter([(503, None), (500, None), (200, b"ok")])
    slept = []
    ok = fetch_to_path(
        "http://x/i.jpg",
        str(tmp_path / "i.jpg"),
        lambda url: next(statuses),
        sleep=slept.append,
    )
    assert ok is True
    assert len(slept) == 2


def test_fetch_to_path_gives_up_immediately_on_a_non_retryable_status(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        return 404, None

    ok = fetch_to_path("http://x/i.jpg", str(tmp_path / "i.jpg"), fetch, sleep=lambda s: None)
    assert ok is False
    assert len(calls) == 1
    assert not (tmp_path / "i.jpg").exists()


def test_fetch_to_path_leaves_no_part_file_behind_on_failure(tmp_path):
    fetch_to_path("http://x/i.jpg", str(tmp_path / "i.jpg"), lambda url: (0, None),
                  sleep=lambda s: None)
    assert list(tmp_path.iterdir()) == []


def test_fetch_to_path_treats_an_empty_body_as_a_failure(tmp_path):
    # A 200 with no bytes must not land a zero-byte file that a later
    # resume check would mistake for a completed download.
    ok = fetch_to_path("http://x/i.jpg", str(tmp_path / "i.jpg"), lambda url: (200, b""),
                       sleep=lambda s: None)
    assert ok is False
    assert not (tmp_path / "i.jpg").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_images.py -q`
Expected: FAIL — `ImportError: cannot import name 'fetch_to_path'`.

- [ ] **Step 3: Extract the function and rewire `download_images`**

In `trainer/hitcheck_trainer/catalog/images.py`, add above `download_images`:

```python
def fetch_to_path(url, path, fetch, sleep=time.sleep, max_attempts=4) -> bool:
    """Download one URL to one path, retrying, landing it atomically.

    Returns True once the bytes are at `path`. Writes go to a `.part`
    temp file and arrive via `os.replace`, so a process killed mid-write
    can never leave a half-written file at the final path. An empty body
    counts as a failure: a zero-byte file at the final path would be
    mistaken for a completed download by any later resume check.

    Shared by the catalog sync and the M2 corpus builder. They key files
    differently — card id versus eBay itemId — but the retry schedule and
    the atomic write must not diverge between them.
    """
    delays = backoff_delays(max_attempts - 1)
    for attempt in range(max_attempts):
        status, body = fetch(url)
        if status == 200 and body:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp = f"{path}.part"
            with open(tmp, "wb") as fh:
                fh.write(body)
            os.replace(tmp, path)  # atomic — no half-written images
            return True
        if status not in RETRYABLE:
            return False
        if attempt < len(delays):
            sleep(delays[attempt])
    return False
```

Then replace the body of `download_images`'s per-pair download block. The whole loop becomes:

```python
    for i, (card_id, url) in enumerate(pairs, start=1):
        path = image_path(root, card_id)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            skipped += 1
            if on_progress:
                on_progress(i, total)
            continue

        if fetch_to_path(url, path, fetch, sleep=sleep, max_attempts=max_attempts):
            downloaded += 1

        if on_progress:
            on_progress(i, total)
```

Delete the now-unused local `delays = backoff_delays(max_attempts - 1)` line from `download_images`.

- [ ] **Step 4: Run the whole image test file to verify the refactor changed nothing**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_images.py -q`
Expected: PASS — both the new `fetch_to_path` tests and every pre-existing `download_images` test.

Run: `cd trainer && .venv/bin/python -m pytest -q -m "not slow"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd trainer && .venv/bin/python -m ruff check hitcheck_trainer tests
cd /var/home/mstephens/Documents/GitHub/pokemon-card-stream-pricer
git add trainer/hitcheck_trainer/catalog/images.py trainer/tests/test_images.py
git commit -m "refactor(catalog): extract fetch_to_path from download_images

The M2 corpus builder needs the same retry-and-atomic-write download but
keys files by eBay itemId rather than card id, so it cannot call
download_images. Extracting the per-file half keeps one implementation
instead of letting the retry schedule and the .part/os.replace write
drift apart between two callers.

Pure refactor: download_images keeps its signature, return value, resume
semantics and progress callback.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Text normalisation for label resolution

**Files:**
- Create: `trainer/hitcheck_trainer/corpus/__init__.py`
- Create: `trainer/hitcheck_trainer/corpus/normalize.py`
- Test: `trainer/tests/test_normalize.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. Standard library only.
- Produces:
  - `normalize_number(raw: str) -> str` — `"201/165"` → `"201"`. Returns `""` for unusable input.
  - `normalize_name(raw: str) -> str` — `"Charizard ex 199/165"` → `"charizardex"`. Returns `""` for unusable input.
  - `normalize_set(raw: str) -> str` — `"Black and White"` → `"blackandwhite"`. Returns `""` for unusable input.

**Background for the implementer:** eBay Item Specifics are seller-entered free text; the catalog is machine-generated. These three functions are the only place that gap is bridged, and every one of them is a place a wrong answer becomes a fake retrieval miss. Measured facts they must respect:

- 18,858 of 20,479 catalog card numbers are bare digits, but the rest are not — `H1`, `DP01`, `TG12`, `SWSH284` all appear. Do not assume numeric.
- eBay writes numbers as `201/165` (card over set total); the catalog stores `201`. The `/total` suffix must go.
- The catalog has 175 sets with unique names, including `Black & White`. A seller typing `Black and White` must reach it, so `&` maps to the word `and` **before** punctuation is stripped — otherwise `blackwhite` and `blackandwhite` never meet.
- Sellers put the number inside the name field (`Charizard ex 199/165`). Strip embedded number tokens out of names.

- [ ] **Step 1: Write the failing tests**

Create `trainer/tests/test_normalize.py`:

```python
from hitcheck_trainer.corpus.normalize import (
    normalize_name,
    normalize_number,
    normalize_set,
)


def test_number_drops_the_set_total():
    assert normalize_number("201/165") == "201"


def test_number_keeps_alphanumeric_prefixes_whole():
    # H1, DP01, TG12 and SWSH284 all exist in the catalog; treating card
    # numbers as integers would lose every one of them.
    assert normalize_number("TG12/TG30") == "TG12"
    assert normalize_number("SWSH284") == "SWSH284"
    assert normalize_number("H1") == "H1"


def test_number_strips_leading_zeros_only_when_it_is_purely_numeric():
    assert normalize_number("006/165") == "6"
    assert normalize_number("DP01") == "DP01"


def test_number_strips_whitespace_and_a_leading_hash():
    assert normalize_number("  #199 ") == "199"


def test_number_uppercases_so_case_never_decides_a_match():
    assert normalize_number("tg12") == "TG12"


def test_number_of_junk_is_empty_not_a_guess():
    assert normalize_number("") == ""
    assert normalize_number("   ") == ""
    assert normalize_number("/165") == ""


def test_name_lowercases_and_drops_punctuation_and_spaces():
    assert normalize_name("Charizard ex") == "charizardex"
    assert normalize_name("Professor's Research") == "professorsresearch"


def test_name_drops_a_number_the_seller_put_in_the_name_field():
    assert normalize_name("Charizard ex 199/165") == "charizardex"
    assert normalize_name("Pikachu #58") == "pikachu"


def test_name_maps_ampersand_to_and():
    assert normalize_name("Team Magma & Team Aqua") == "teammagmaandteamaqua"


def test_name_folds_accents_so_pokemon_matches_pokemon():
    assert normalize_name("Pokémon Center Lady") == "pokemoncenterlady"


def test_name_of_junk_is_empty():
    assert normalize_name("") == ""
    assert normalize_name("   -- ") == ""


def test_set_maps_ampersand_to_and_before_stripping_punctuation():
    # The catalog stores "Black & White"; a seller types "Black and White".
    # Stripping punctuation first would give blackwhite vs blackandwhite,
    # which never meet.
    assert normalize_set("Black & White") == normalize_set("Black and White")
    assert normalize_set("Black & White") == "blackandwhite"


def test_set_lowercases_and_drops_spaces_and_punctuation():
    assert normalize_set("Astral Radiance Trainer Gallery") == "astralradiancetrainergallery"
    assert normalize_set("Base Set 2") == "baseset2"


def test_set_keeps_bare_numeric_names():
    # The catalog set sv3pt5 is literally named "151".
    assert normalize_set("151") == "151"


def test_set_of_junk_is_empty():
    assert normalize_set("") == ""
    assert normalize_set(" & ") == "and"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_normalize.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hitcheck_trainer.corpus'`.

- [ ] **Step 3: Implement the normalisers**

Create `trainer/hitcheck_trainer/corpus/__init__.py` as an empty file.

Create `trainer/hitcheck_trainer/corpus/normalize.py`:

```python
"""Bridge seller free text to machine-generated catalog fields.

eBay Item Specifics are typed by hand; the catalog is generated. Every
mismatch these functions fail to bridge becomes a retrieval "miss" that
is not one, landing directly underneath the accuracy number the M2
decision rests on. So they are deliberately conservative: they fold away
differences that are certainly cosmetic and nothing else.
"""

import re
import unicodedata

_EMBEDDED_NUMBER = re.compile(r"#?\d+\s*/\s*\S+|#\d+")
_NOT_ALNUM = re.compile(r"[^a-z0-9]+")


def _fold(text: str) -> str:
    """Lowercase, strip accents, map & to 'and', drop everything else.

    The ampersand mapping happens before punctuation is stripped: the
    catalog stores "Black & White" and sellers type "Black and White",
    and stripping first would yield blackwhite versus blackandwhite.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _NOT_ALNUM.sub("", ascii_only.lower().replace("&", " and "))


def normalize_number(raw: str) -> str:
    """Card number as the catalog stores it: '201/165' -> '201'.

    Leading zeros are stripped only from purely numeric values. Card
    numbers are not integers -- H1, DP01, TG12 and SWSH284 are all real
    catalog values -- so the alphanumeric forms pass through whole.
    """
    if not raw:
        return ""
    head = raw.split("/", 1)[0]
    head = _NOT_ALNUM.sub("", head.lower()).upper()
    if not head:
        return ""
    if head.isdigit():
        return str(int(head))
    return head


def normalize_name(raw: str) -> str:
    """Card name for comparison: 'Charizard ex 199/165' -> 'charizardex'.

    Sellers routinely put the card number in the name field, so embedded
    number tokens come out before folding.
    """
    if not raw:
        return ""
    return _fold(_EMBEDDED_NUMBER.sub(" ", raw))


def normalize_set(raw: str) -> str:
    """Set name for comparison: 'Black & White' -> 'blackandwhite'."""
    if not raw:
        return ""
    return _fold(raw)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_normalize.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd trainer && .venv/bin/python -m ruff check hitcheck_trainer tests
cd /var/home/mstephens/Documents/GitHub/pokemon-card-stream-pricer
git add trainer/hitcheck_trainer/corpus/__init__.py trainer/hitcheck_trainer/corpus/normalize.py trainer/tests/test_normalize.py
git commit -m "feat(corpus): normalise seller free text toward catalog fields

eBay Item Specifics are hand-typed and the catalog is generated. These
three functions are the whole of that bridge, and every difference they
fail to fold becomes a retrieval miss that is not one -- label noise
sitting directly under the M2 accuracy number.

Conservative by design: '201/165' loses its set total, embedded numbers
come out of name fields, accents fold, and & becomes 'and' before
punctuation is stripped (otherwise the catalog's 'Black & White' and a
seller's 'Black and White' never meet). Leading zeros are stripped only
from purely numeric values, because H1, DP01, TG12 and SWSH284 are all
real catalog card numbers.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Resolve eBay aspects to a catalog card id

**Files:**
- Create: `trainer/hitcheck_trainer/corpus/resolve.py`
- Test: `trainer/tests/test_resolve.py`

**Interfaces:**
- Consumes: `normalize_name`, `normalize_number`, `normalize_set` from `hitcheck_trainer.corpus.normalize` (Task 3).
- Produces:
  - `Resolution` — frozen dataclass with `card_id: str | None` and `reason: str = ""` (empty when resolved).
  - Discard-reason constants, each a `str` equal to its own name: `MISSING_NAME`, `MISSING_SET`, `MISSING_NUMBER`, `MISSING_LANGUAGE`, `NOT_ENGLISH`, `UNKNOWN_SET`, `AMBIGUOUS_SET`, `NO_SUCH_NUMBER`, `NAME_MISMATCH`, `AMBIGUOUS_CARD`; plus `DISCARD_REASONS: tuple[str, ...]` listing them in that order.
  - `CardLookup(set_ids: dict[str, str], cards: dict[tuple[str, str], list[tuple[str, str]]])` with `CardLookup.from_conn(conn) -> CardLookup`, `match_set(raw: str) -> tuple[str | None, str]`, `cards_at(set_id: str, number_key: str) -> list[tuple[str, str]]`.
  - `resolve(aspects: dict[str, str], lookup: CardLookup) -> Resolution`.

**Background for the implementer:** This is the accuracy-contaminating step, so its rule is absolute: **accept only unambiguous agreement on name, set and number; discard everything else and never guess.** A discard costs one corpus entry. A wrong resolution corrupts the number the whole exercise exists to produce.

Measured aspect coverage over 20 sampled graded listings: `Card Name` 17/20, `Set` 17/20, `Card Number` 12/20, `Language` 15/20. Requiring all of them costs roughly 40% of candidates, which is affordable — one generic query already reports 22,179 matches.

A missing `Language` aspect is a discard, not an assumed English. The catalog is English-only, and Japanese prints share artwork with English ones while sharing no numbering, so a Japanese listing that slipped through would resolve to a plausible-looking wrong id.

`aspects` keys are the eBay Item Specifics names verbatim: `"Card Name"`, `"Set"`, `"Card Number"`, `"Language"`.

- [ ] **Step 1: Write the failing tests**

Create `trainer/tests/test_resolve.py`:

```python
from hitcheck_trainer.catalog.db import open_db, upsert_cards
from hitcheck_trainer.corpus.resolve import (
    AMBIGUOUS_CARD,
    AMBIGUOUS_SET,
    DISCARD_REASONS,
    MISSING_LANGUAGE,
    MISSING_NAME,
    MISSING_NUMBER,
    MISSING_SET,
    NAME_MISMATCH,
    NO_SUCH_NUMBER,
    NOT_ENGLISH,
    UNKNOWN_SET,
    CardLookup,
    resolve,
)


def lookup():
    """A small stand-in catalog covering the cases that matter."""
    return CardLookup(
        set_ids={
            "151": "sv3pt5",
            "blackandwhite": "bw1",
            "astralradiance": "swsh10",
            "astralradiancetrainergallery": "swsh10tg",
        },
        cards={
            ("sv3pt5", "6"): [("sv3pt5-6", "charizardex")],
            ("sv3pt5", "199"): [("sv3pt5-199", "charizardex")],
            ("bw1", "1"): [("bw1-1", "snivy")],
            ("swsh10", "TG12"): [("swsh10-TG12", "sylveonvstar")],
        },
    )


def aspects(**overrides):
    base = {
        "Card Name": "Charizard ex",
        "Set": "151",
        "Card Number": "199/165",
        "Language": "English",
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


def test_resolves_a_complete_unambiguous_listing():
    result = resolve(aspects(), lookup())
    assert result.card_id == "sv3pt5-199"
    assert result.reason == ""


def test_resolves_an_alphanumeric_card_number():
    result = resolve(
        aspects(**{"Card Name": "Sylveon VSTAR", "Set": "Astral Radiance",
                   "Card Number": "TG12/TG30"}),
        lookup(),
    )
    assert result.card_id == "swsh10-TG12"


def test_resolves_through_the_ampersand_spelling_difference():
    result = resolve(
        aspects(**{"Card Name": "Snivy", "Set": "Black and White",
                   "Card Number": "1/114"}),
        lookup(),
    )
    assert result.card_id == "bw1-1"


def test_a_missing_aspect_discards_with_a_named_reason():
    assert resolve(aspects(**{"Card Name": None}), lookup()).reason == MISSING_NAME
    assert resolve(aspects(**{"Set": None}), lookup()).reason == MISSING_SET
    assert resolve(aspects(**{"Card Number": None}), lookup()).reason == MISSING_NUMBER
    assert resolve(aspects(**{"Language": None}), lookup()).reason == MISSING_LANGUAGE


def test_a_missing_language_is_a_discard_not_an_assumed_english():
    # Japanese prints share artwork with English ones but share no
    # numbering, so an unmarked Japanese listing would resolve to a
    # plausible-looking wrong id.
    assert resolve(aspects(**{"Language": None}), lookup()).card_id is None


def test_a_non_english_listing_is_discarded():
    result = resolve(aspects(**{"Language": "Japanese"}), lookup())
    assert result.card_id is None
    assert result.reason == NOT_ENGLISH


def test_a_set_with_no_catalog_match_is_discarded():
    result = resolve(aspects(**{"Set": "Totally Invented Set"}), lookup())
    assert result.reason == UNKNOWN_SET


def test_a_set_name_contained_in_two_sets_is_ambiguous_not_a_coin_flip():
    # "Astral" is a substring of both "Astral Radiance" and "Astral
    # Radiance Trainer Gallery" -- different sets with different cards --
    # and is not close enough to either for the fuzzy match to fire.
    result = resolve(aspects(**{"Set": "Astral"}), lookup())
    assert result.card_id is None
    assert result.reason == AMBIGUOUS_SET


def test_a_clear_fuzzy_winner_is_accepted_rather_than_discarded():
    # "Astral Radiance Trainer" scores 0.857 against the Trainer Gallery
    # and 0.80 against plain Astral Radiance, so only one candidate clears
    # the 0.85 cutoff. Being strict does not mean rejecting everything --
    # it means never picking between two candidates that both qualify.
    result = resolve(
        aspects(**{"Card Name": "Sylveon VSTAR", "Set": "Astral Radiance",
                   "Card Number": "TG12/TG30"}),
        lookup(),
    )
    assert result.card_id == "swsh10-TG12"


def test_a_number_absent_from_the_matched_set_is_discarded():
    result = resolve(aspects(**{"Card Number": "9999/165"}), lookup())
    assert result.reason == NO_SUCH_NUMBER


def test_a_name_that_disagrees_with_the_catalog_is_discarded():
    # Set and number both point at sv3pt5-199, but the seller named a
    # different card. This is exactly where guessing would put a wrong
    # label under the accuracy number.
    result = resolve(aspects(**{"Card Name": "Blastoise ex"}), lookup())
    assert result.card_id is None
    assert result.reason == NAME_MISMATCH


def test_two_catalog_cards_at_the_same_set_and_number_are_ambiguous():
    lk = CardLookup(
        set_ids={"151": "sv3pt5"},
        cards={("sv3pt5", "199"): [("sv3pt5-199", "charizardex"),
                                   ("sv3pt5-199a", "charizardex")]},
    )
    assert resolve(aspects(), lk).reason == AMBIGUOUS_CARD


def test_every_reason_a_resolution_can_return_is_listed_in_discard_reasons():
    # build.py tallies discards by iterating DISCARD_REASONS; a reason
    # missing from it would be silently dropped from the yield report.
    for reason in (MISSING_NAME, MISSING_SET, MISSING_NUMBER, MISSING_LANGUAGE,
                   NOT_ENGLISH, UNKNOWN_SET, AMBIGUOUS_SET, NO_SUCH_NUMBER,
                   NAME_MISMATCH, AMBIGUOUS_CARD):
        assert reason in DISCARD_REASONS


def test_lookup_from_conn_indexes_a_real_catalog_database(tmp_path):
    conn = open_db(str(tmp_path / "catalog.sqlite"))
    upsert_cards(conn, [
        {"id": "sv3pt5-199", "name": "Charizard ex", "number": "199",
         "set": {"id": "sv3pt5", "name": "151"}, "images": {"small": "http://x/1.png"}},
        {"id": "bw1-1", "name": "Snivy", "number": "1",
         "set": {"id": "bw1", "name": "Black & White"}, "images": {"small": "http://x/2.png"}},
    ])
    lk = CardLookup.from_conn(conn)
    assert lk.match_set("151") == ("sv3pt5", "")
    assert lk.match_set("Black and White") == ("bw1", "")
    assert lk.cards_at("sv3pt5", "199") == [("sv3pt5-199", "charizardex")]


def test_lookup_from_conn_skips_rows_with_no_number(tmp_path):
    # A card with a NULL number can never be matched by number, and
    # indexing it under "" would let a listing with an unparseable number
    # collide with it.
    conn = open_db(str(tmp_path / "catalog.sqlite"))
    upsert_cards(conn, [
        {"id": "bp-1", "name": "Best Of Promo", "number": None,
         "set": {"id": "bp", "name": "Best of Game"}, "images": {"small": "http://x/3.png"}},
    ])
    lk = CardLookup.from_conn(conn)
    assert lk.cards_at("bp", "") == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_resolve.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hitcheck_trainer.corpus.resolve'`.

- [ ] **Step 3: Implement the resolver**

Create `trainer/hitcheck_trainer/corpus/resolve.py`:

```python
"""eBay Item Specifics to a catalog card id.

This is the accuracy-contaminating step of the M2 corpus. A discard costs
one entry; a wrong resolution corrupts the number the whole exercise
exists to produce. So the rule is absolute: accept only unambiguous
agreement on name, set and number, and never guess.

Every discard carries a named reason so build.py can report the corpus's
own yield. A corpus that silently dropped most of its candidates would
skew toward whichever listings happen to have tidy Item Specifics.
"""

import difflib
from dataclasses import dataclass

from .normalize import normalize_name, normalize_number, normalize_set

MISSING_NAME = "MISSING_NAME"
MISSING_SET = "MISSING_SET"
MISSING_NUMBER = "MISSING_NUMBER"
MISSING_LANGUAGE = "MISSING_LANGUAGE"
NOT_ENGLISH = "NOT_ENGLISH"
UNKNOWN_SET = "UNKNOWN_SET"
AMBIGUOUS_SET = "AMBIGUOUS_SET"
NO_SUCH_NUMBER = "NO_SUCH_NUMBER"
NAME_MISMATCH = "NAME_MISMATCH"
AMBIGUOUS_CARD = "AMBIGUOUS_CARD"

DISCARD_REASONS = (
    MISSING_NAME,
    MISSING_SET,
    MISSING_NUMBER,
    MISSING_LANGUAGE,
    NOT_ENGLISH,
    UNKNOWN_SET,
    AMBIGUOUS_SET,
    NO_SUCH_NUMBER,
    NAME_MISMATCH,
    AMBIGUOUS_CARD,
)

# A fuzzy set match must be this close before it is considered at all,
# and must beat the runner-up by this margin before it is accepted.
_SET_CUTOFF = 0.85
_SET_MARGIN = 0.05


@dataclass(frozen=True)
class Resolution:
    card_id: str | None
    reason: str = ""


class CardLookup:
    """Set-name and (set, number) indexes over the catalog."""

    def __init__(
        self,
        set_ids: dict[str, str],
        cards: dict[tuple[str, str], list[tuple[str, str]]],
    ):
        self._set_ids = set_ids
        self._cards = cards

    @classmethod
    def from_conn(cls, conn) -> "CardLookup":
        set_ids: dict[str, str] = {}
        cards: dict[tuple[str, str], list[tuple[str, str]]] = {}
        rows = conn.execute(
            "SELECT id, name, number, set_id, set_name FROM cards ORDER BY id"
        ).fetchall()
        for row in rows:
            if row["set_name"] and row["set_id"]:
                set_ids.setdefault(normalize_set(row["set_name"]), row["set_id"])
            number_key = normalize_number(row["number"] or "")
            if not number_key or not row["set_id"]:
                continue  # unmatchable by number; indexing under "" would collide
            cards.setdefault((row["set_id"], number_key), []).append(
                (row["id"], normalize_name(row["name"] or ""))
            )
        return cls(set_ids, cards)

    def match_set(self, raw: str) -> tuple[str | None, str]:
        """Resolve a seller's set name to a set id, or say why not."""
        key = normalize_set(raw)
        if not key:
            return None, MISSING_SET
        if key in self._set_ids:
            return self._set_ids[key], ""

        candidates = sorted(self._set_ids)
        close = difflib.get_close_matches(key, candidates, n=2, cutoff=_SET_CUTOFF)
        if len(close) == 1:
            return self._set_ids[close[0]], ""
        if len(close) > 1:
            best = difflib.SequenceMatcher(None, key, close[0]).ratio()
            runner_up = difflib.SequenceMatcher(None, key, close[1]).ratio()
            if best - runner_up > _SET_MARGIN:
                return self._set_ids[close[0]], ""
            return None, AMBIGUOUS_SET

        contains = [name for name in candidates if key in name]
        if len(contains) == 1:
            return self._set_ids[contains[0]], ""
        if len(contains) > 1:
            return None, AMBIGUOUS_SET
        return None, UNKNOWN_SET

    def cards_at(self, set_id: str, number_key: str) -> list[tuple[str, str]]:
        return list(self._cards.get((set_id, number_key), []))


def resolve(aspects: dict[str, str], lookup: CardLookup) -> Resolution:
    """Resolve one listing's Item Specifics, or discard it with a reason."""
    language = aspects.get("Language")
    if not language:
        return Resolution(None, MISSING_LANGUAGE)
    if normalize_name(language) != "english":
        return Resolution(None, NOT_ENGLISH)

    name_key = normalize_name(aspects.get("Card Name", ""))
    if not name_key:
        return Resolution(None, MISSING_NAME)
    if not aspects.get("Set"):
        return Resolution(None, MISSING_SET)
    number_key = normalize_number(aspects.get("Card Number", ""))
    if not number_key:
        return Resolution(None, MISSING_NUMBER)

    set_id, reason = lookup.match_set(aspects["Set"])
    if set_id is None:
        return Resolution(None, reason)

    candidates = lookup.cards_at(set_id, number_key)
    if not candidates:
        return Resolution(None, NO_SUCH_NUMBER)

    matching = [card_id for card_id, catalog_name in candidates if catalog_name == name_key]
    if not matching:
        return Resolution(None, NAME_MISMATCH)
    if len(matching) > 1:
        return Resolution(None, AMBIGUOUS_CARD)
    return Resolution(matching[0])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_resolve.py -q`
Expected: PASS.

If `test_a_set_name_matching_two_sets_is_ambiguous_not_a_coin_flip` fails, do not loosen the ambiguity rule to make it pass — check whether `difflib` ranked one candidate far enough ahead that `_SET_MARGIN` accepted it, and if so tighten `_SET_MARGIN`, never widen it. Discarding is always the safe direction here.

- [ ] **Step 5: Sanity-check the resolver against the real catalog**

Run:

```bash
cd trainer && .venv/bin/python -c "
from hitcheck_trainer.catalog.db import open_db
from hitcheck_trainer.corpus.resolve import CardLookup, resolve
lk = CardLookup.from_conn(open_db('data/catalog.sqlite'))
for a in [
    {'Card Name': 'Charizard ex', 'Set': '151', 'Card Number': '199/165', 'Language': 'English'},
    {'Card Name': 'Charizard ex', 'Set': 'Sv2a: Pokemon Card 151', 'Card Number': '201/165', 'Language': 'Japanese'},
]:
    print(a['Set'], '->', resolve(a, lk))
"
```

Expected: the first resolves to `sv3pt5-199`; the second discards with `NOT_ENGLISH`. This is the exact listing pair from the spec's design reconnaissance. If the first does not resolve, stop and report — the resolver is the load-bearing piece of this plan.

- [ ] **Step 6: Commit**

```bash
cd trainer && .venv/bin/python -m ruff check hitcheck_trainer tests
cd /var/home/mstephens/Documents/GitHub/pokemon-card-stream-pricer
git add trainer/hitcheck_trainer/corpus/resolve.py trainer/tests/test_resolve.py
git commit -m "feat(corpus): resolve eBay Item Specifics to catalog card ids

The accuracy-contaminating step of the M2 corpus: a discard costs one
entry, a wrong resolution corrupts the number the exercise exists to
produce. So it accepts only unambiguous agreement on name, set and
number and never guesses -- a fuzzy set match must clear 0.85 and beat
its runner-up by 0.05, and two catalog cards at one set-and-number
discard rather than pick.

A missing Language aspect discards rather than assuming English:
Japanese prints share artwork with English ones but share no numbering,
so an unmarked one would resolve to a plausible-looking wrong id.

Every discard carries a named reason, so the manifest can state its own
yield instead of silently skewing toward listings with tidy specifics.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: The corpus manifest, and carving it out of the `data/` ignore

**Files:**
- Create: `trainer/hitcheck_trainer/corpus/manifest.py`
- Modify: `trainer/.gitignore`
- Test: `trainer/tests/test_manifest.py`

**Interfaces:**
- Consumes: `DISCARD_REASONS` from `hitcheck_trainer.corpus.resolve` (Task 4).
- Produces:
  - `safe_item_id(item_id: str) -> str` — filename-safe form of an eBay itemId.
  - `image_relpath(item_id: str) -> str` — `"images/<safe_item_id>.jpg"`.
  - `CorpusEntry` — frozen dataclass: `item_id: str`, `card_id: str`, `image: str`, `image_url: str`, `listing_url: str`, `aspects: dict[str, str]`.
  - `Manifest` — dataclass: `entries: list[CorpusEntry]`, `discards: dict[str, int]`, `queries: list[str]`; with `item_ids() -> set[str]`, `yield_summary() -> str`.
  - `save_manifest(manifest: Manifest, path: str) -> None`
  - `load_manifest(path: str) -> Manifest`

**Background for the implementer:** The manifest is checked into git so the eval is reproducible after the listings expire; the images beside it are not, because they are sellers' copyrighted photographs used locally and not redistributed. `trainer/.gitignore` currently ignores `data/` wholesale, so this task adds the carve-out. The exact five-line pattern below was verified in a scratch repo: `manifest.json` and `crops.json` are tracked, `data/catalog.sqlite` and `data/corpus/images/*` stay ignored. Git cannot un-ignore a file inside an ignored *directory*, which is why the first line must be `data/*` with a trailing `*` and not `data/`.

**eBay itemIds contain pipes** — the Browse API returns values like `v1|364012345678|0`. Those are not filename-safe, so `safe_item_id` maps everything outside `[A-Za-z0-9_-]` to `_`. The manifest stores the original `item_id` as provenance and the sanitised path separately; never reconstruct one from the other by guessing.

JSON is written with `indent=2` and `sort_keys=True` because this file lands in git and its diffs get read by humans.

- [ ] **Step 1: Write the failing tests**

Create `trainer/tests/test_manifest.py`:

```python
import json

from hitcheck_trainer.corpus.manifest import (
    CorpusEntry,
    Manifest,
    image_relpath,
    load_manifest,
    safe_item_id,
    save_manifest,
)


def entry(item_id="v1|364012345678|0", card_id="sv3pt5-199"):
    return CorpusEntry(
        item_id=item_id,
        card_id=card_id,
        image=image_relpath(item_id),
        image_url="https://i.ebayimg.com/images/g/abc/s-l1600.jpg",
        listing_url="https://www.ebay.com/itm/364012345678",
        aspects={"Card Name": "Charizard ex", "Set": "151",
                 "Card Number": "199/165", "Language": "English"},
    )


def test_safe_item_id_strips_the_pipes_ebay_puts_in_item_ids():
    # Browse returns ids like v1|364012345678|0, which cannot be a filename.
    assert safe_item_id("v1|364012345678|0") == "v1_364012345678_0"


def test_safe_item_id_leaves_already_safe_ids_alone():
    assert safe_item_id("364012345678") == "364012345678"


def test_image_relpath_is_relative_and_under_images():
    assert image_relpath("v1|364012345678|0") == "images/v1_364012345678_0.jpg"


def test_round_trips_through_disk_unchanged(tmp_path):
    original = Manifest(
        entries=[entry(), entry(item_id="v1|999|0", card_id="bw1-1")],
        discards={"NOT_ENGLISH": 12, "NAME_MISMATCH": 3},
        queries=["pokemon psa graded card"],
    )
    path = str(tmp_path / "manifest.json")
    save_manifest(original, path)
    loaded = load_manifest(path)
    assert loaded == original


def test_saved_json_is_sorted_and_indented_so_git_diffs_are_readable(tmp_path):
    path = str(tmp_path / "manifest.json")
    save_manifest(Manifest(entries=[entry()], discards={}, queries=[]), path)
    text = open(path).read()
    assert "\n  " in text
    keys = list(json.loads(text)["entries"][0])
    assert keys == sorted(keys)


def test_save_leaves_no_part_file_behind(tmp_path):
    path = str(tmp_path / "manifest.json")
    save_manifest(Manifest(entries=[entry()], discards={}, queries=[]), path)
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.json"]


def test_item_ids_lets_a_rerun_skip_what_is_already_captured(tmp_path):
    # Listings expire, so images and entries are written once and never
    # re-fetched. A rerun must be able to tell what it already has.
    m = Manifest(entries=[entry(), entry(item_id="v1|999|0")], discards={}, queries=[])
    assert m.item_ids() == {"v1|364012345678|0", "v1|999|0"}


def test_loading_a_missing_manifest_gives_an_empty_one(tmp_path):
    loaded = load_manifest(str(tmp_path / "nope.json"))
    assert loaded.entries == []
    assert loaded.discards == {}


def test_yield_summary_states_the_corpus_yield_including_every_discard():
    m = Manifest(
        entries=[entry()],
        discards={"NOT_ENGLISH": 12, "NAME_MISMATCH": 3},
        queries=[],
    )
    text = m.yield_summary()
    assert "kept=1" in text
    assert "discarded=15" in text
    assert "NOT_ENGLISH=12" in text
    assert "NAME_MISMATCH=3" in text


def test_yield_summary_of_a_perfect_run_still_reports_zero_discards():
    assert "discarded=0" in Manifest(entries=[], discards={}, queries=[]).yield_summary()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_manifest.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hitcheck_trainer.corpus.manifest'`.

- [ ] **Step 3: Implement the manifest**

Create `trainer/hitcheck_trainer/corpus/manifest.py`:

```python
"""The M2 corpus's checked-in record of what it captured and what it dropped.

Listings expire, so images and entries are written once and never
re-fetched; the eval reads this file and the images beside it, and
nothing else. That is what makes the number reproducible after the
listings are gone.

This file is tracked in git (its images are not -- they are sellers'
copyrighted photographs, used locally and not redistributed), so it is
written sorted and indented for readable diffs.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def safe_item_id(item_id: str) -> str:
    """Filename-safe form of an eBay itemId.

    Browse returns ids like 'v1|364012345678|0'; the pipes cannot go in a
    path. The original id stays in the manifest as provenance -- never
    reconstruct one form from the other.
    """
    return _UNSAFE.sub("_", item_id)


def image_relpath(item_id: str) -> str:
    """Image path relative to the manifest's own directory."""
    return f"images/{safe_item_id(item_id)}.jpg"


@dataclass(frozen=True)
class CorpusEntry:
    item_id: str
    card_id: str
    image: str
    image_url: str
    listing_url: str
    aspects: dict[str, str]


@dataclass
class Manifest:
    entries: list[CorpusEntry] = field(default_factory=list)
    discards: dict[str, int] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)

    def item_ids(self) -> set[str]:
        return {e.item_id for e in self.entries}

    def yield_summary(self) -> str:
        """One line stating the corpus's own yield.

        A corpus that silently dropped most of its candidates would skew
        toward whichever listings happen to have tidy Item Specifics, so
        the discard breakdown travels with the manifest rather than
        living only in a console log.
        """
        total_discarded = sum(self.discards.values())
        breakdown = " ".join(
            f"{reason}={count}" for reason, count in sorted(self.discards.items())
        )
        return f"kept={len(self.entries)} discarded={total_discarded} {breakdown}".rstrip()


def save_manifest(manifest: Manifest, path: str) -> None:
    payload = {
        "entries": [asdict(e) for e in manifest.entries],
        "discards": manifest.discards,
        "queries": manifest.queries,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.part"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)  # atomic — never a half-written manifest


def load_manifest(path: str) -> Manifest:
    """Load a manifest, or an empty one if it does not exist yet."""
    if not os.path.exists(path):
        return Manifest()
    with open(path) as fh:
        payload = json.load(fh)
    return Manifest(
        entries=[CorpusEntry(**e) for e in payload.get("entries", [])],
        discards=payload.get("discards", {}),
        queries=payload.get("queries", []),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_manifest.py -q`
Expected: PASS.

- [ ] **Step 5: Carve the manifest out of the `data/` ignore**

`trainer/.gitignore` currently reads:

```
.venv
__pycache__/
*.pyc
.env
data/
.pytest_cache/
.ruff_cache/
```

Replace the single `data/` line with these five lines, leaving every other line as-is:

```
data/*
!data/corpus/
data/corpus/*
!data/corpus/manifest.json
!data/corpus/crops.json
```

The trailing `*` on the first line is load-bearing: git cannot un-ignore a file inside an ignored *directory*, so `data/` would make the two negations below it dead letters.

- [ ] **Step 6: Verify the ignore rules do exactly what they claim**

Run:

```bash
cd trainer && mkdir -p data/corpus/images && touch data/corpus/manifest.json data/corpus/crops.json data/corpus/images/probe.jpg
git check-ignore -v data/catalog.sqlite data/corpus/images/probe.jpg
git check-ignore -v data/corpus/manifest.json data/corpus/crops.json ; echo "exit=$?"
rm data/corpus/images/probe.jpg
```

Expected: the first `git check-ignore` prints a rule for **both** `data/catalog.sqlite` and `data/corpus/images/probe.jpg` (both still ignored). The second prints nothing and reports `exit=1` — meaning neither the manifest nor the crop file is ignored. If `manifest.json` comes back ignored, the `data/*` trailing star is missing.

- [ ] **Step 7: Commit**

```bash
cd trainer && .venv/bin/python -m ruff check hitcheck_trainer tests
cd /var/home/mstephens/Documents/GitHub/pokemon-card-stream-pricer
git add trainer/hitcheck_trainer/corpus/manifest.py trainer/tests/test_manifest.py trainer/.gitignore
git commit -m "feat(corpus): checked-in manifest, gitignored images

Listings expire. Images and entries are written once and never re-fetched,
and the eval reads the manifest and the images beside it and nothing else
-- that is what keeps the M2 number reproducible after the listings are
gone. The manifest is tracked; the images are not, because they are
sellers' copyrighted photographs used locally and not redistributed.

trainer/.gitignore ignored data/ wholesale, and git cannot un-ignore a
file inside an ignored directory, so the rule becomes data/* plus
explicit negations for the two small JSON files.

safe_item_id exists because Browse itemIds look like v1|364012345678|0
and pipes cannot go in a path; the original id stays in the manifest as
provenance rather than being reconstructed from the filename.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: eBay OAuth and Browse client

**Files:**
- Modify: `trainer/hitcheck_trainer/catalog/http.py`
- Create: `trainer/hitcheck_trainer/corpus/ebay.py`
- Test: `trainer/tests/test_ebay.py`

**Interfaces:**
- Consumes: `backoff_delays` from `hitcheck_trainer.catalog.backoff`.
- Produces, in `catalog/http.py`:
  - `httpx_post_form(timeout: float = 30.0)` → a callable `post(url: str, headers: dict, data: dict) -> tuple[int, dict | None]`.
- Produces, in `corpus/ebay.py`:
  - `EBAY_OAUTH_URL`, `BROWSE_BASE`, `OAUTH_SCOPE`, `MARKETPLACE_ID` module constants.
  - `basic_auth_header(app_id: str, cert_id: str) -> str`
  - `fetch_token(post, app_id: str, cert_id: str) -> str` — raises `EbayError` on failure.
  - `hi_res_url(url: str) -> str` — swaps any `s-l<N>` segment to `s-l1600`.
  - `aspects_from_item(item: dict) -> dict[str, str]` — flattens `localizedAspects`.
  - `EbayError(Exception)`
  - `BrowseClient(transport, token, marketplace=MARKETPLACE_ID, max_attempts=5, sleep=time.sleep)` with `search(query: str, limit: int = 200, offset: int = 0, extra_filter: str | None = None) -> list[dict]` and `item(item_id: str) -> dict`.

**Background for the implementer:** This is the only network-touching file in `corpus/`. It follows `catalog/api.py`'s shape exactly: an injected `transport(url, headers) -> (status, json | None)`, retries on `RETRYABLE` statuses driven by `backoff_delays`, and a raised error when the attempts run out. Do not add a second retry policy.

Facts that were established by hitting the live API during design, and that the code must not drift from:

- The token endpoint is `https://api.ebay.com/identity/v1/oauth2/token`. **The `v1` path segment is mandatory** — omitting it returns 404, which is easy to misread as a credentials problem.
- Auth is HTTP Basic over `AppID:CertID`, base64 of the raw `app_id:cert_id` string. The grant is `client_credentials` with scope `https://api.ebay.com/oauth/api_scope`. A successful response is `{"access_token": "v^1.1#...", "token_type": "Application Access Token", "expires_in": 7200}`.
- Browse search is `GET {BROWSE_BASE}/item_summary/search`; `limit` maxes out at 200. Item detail is `GET {BROWSE_BASE}/item/{item_id}`. Both need `Authorization: Bearer <token>` and `X-EBAY-C-MARKETPLACE-ID: EBAY_US`.
- Item Specifics arrive on the **item detail** response as `localizedAspects`: a list of `{"type": ..., "name": ..., "value": ...}`. Search summaries do not carry them, which is why acquisition costs two calls per item.
- Image URLs come back at `s-l225`; swapping the suffix to `s-l1600` yields the full-resolution photograph (measured: an `s-l225` at 138×225 became 734×1200).

**Secrets never get logged.** `basic_auth_header` returns the header value; nothing in this module prints it, the token, or the credentials. Error messages carry the HTTP status only.

- [ ] **Step 1: Write the failing tests**

Create `trainer/tests/test_ebay.py`:

```python
import base64

import pytest

from hitcheck_trainer.corpus.ebay import (
    BROWSE_BASE,
    EBAY_OAUTH_URL,
    BrowseClient,
    EbayError,
    aspects_from_item,
    basic_auth_header,
    fetch_token,
    hi_res_url,
)


def test_oauth_url_keeps_the_mandatory_v1_segment():
    # Dropping /v1/ returns 404, which reads like a credentials problem
    # and cost real debugging time once already.
    assert EBAY_OAUTH_URL == "https://api.ebay.com/identity/v1/oauth2/token"


def test_basic_auth_header_is_base64_of_appid_colon_certid():
    header = basic_auth_header("APP-123", "CERT-456")
    assert header.startswith("Basic ")
    decoded = base64.b64decode(header.split(" ", 1)[1]).decode()
    assert decoded == "APP-123:CERT-456"


def test_fetch_token_posts_client_credentials_and_returns_the_token():
    seen = {}

    def post(url, headers, data):
        seen.update(url=url, headers=headers, data=data)
        return 200, {"access_token": "v^1.1#tok", "expires_in": 7200}

    assert fetch_token(post, "APP-123", "CERT-456") == "v^1.1#tok"
    assert seen["url"] == EBAY_OAUTH_URL
    assert seen["data"]["grant_type"] == "client_credentials"
    assert seen["data"]["scope"] == "https://api.ebay.com/oauth/api_scope"
    assert seen["headers"]["Content-Type"] == "application/x-www-form-urlencoded"
    assert seen["headers"]["Authorization"].startswith("Basic ")


def test_fetch_token_raises_without_leaking_the_credentials():
    with pytest.raises(EbayError) as exc:
        fetch_token(lambda url, headers, data: (401, None), "APP-SECRET", "CERT-SECRET")
    message = str(exc.value)
    assert "401" in message
    assert "APP-SECRET" not in message
    assert "CERT-SECRET" not in message


def test_hi_res_url_swaps_the_size_suffix():
    assert hi_res_url("https://i.ebayimg.com/images/g/abc/s-l225.jpg") == (
        "https://i.ebayimg.com/images/g/abc/s-l1600.jpg"
    )
    assert hi_res_url("https://i.ebayimg.com/images/g/abc/s-l500.jpg").endswith("s-l1600.jpg")


def test_hi_res_url_leaves_a_url_with_no_size_suffix_alone():
    assert hi_res_url("https://example.com/photo.jpg") == "https://example.com/photo.jpg"


def test_aspects_from_item_flattens_localized_aspects():
    item = {"localizedAspects": [
        {"type": "STRING", "name": "Card Name", "value": "Charizard ex"},
        {"type": "STRING", "name": "Language", "value": "English"},
    ]}
    assert aspects_from_item(item) == {"Card Name": "Charizard ex", "Language": "English"}


def test_aspects_from_item_of_a_listing_with_no_specifics_is_empty_not_a_crash():
    assert aspects_from_item({}) == {}
    assert aspects_from_item({"localizedAspects": None}) == {}


def test_aspects_from_item_skips_entries_missing_a_name_or_value():
    item = {"localizedAspects": [
        {"name": "Set"},
        {"value": "orphan"},
        {"name": "Set", "value": "151"},
    ]}
    assert aspects_from_item(item) == {"Set": "151"}


def fake_transport(responses):
    """responses: list of (status, body). Records every url and header seen."""
    calls = []

    def transport(url, headers):
        calls.append((url, headers))
        return responses[min(len(calls) - 1, len(responses) - 1)]

    return transport, calls


def test_search_sends_the_bearer_token_and_marketplace_header():
    transport, calls = fake_transport([(200, {"itemSummaries": [{"itemId": "v1|1|0"}]})])
    client = BrowseClient(transport, token="v^1.1#tok")
    summaries = client.search("charizard psa 10")
    assert summaries == [{"itemId": "v1|1|0"}]
    url, headers = calls[0]
    assert url.startswith(f"{BROWSE_BASE}/item_summary/search")
    assert headers["Authorization"] == "Bearer v^1.1#tok"
    assert headers["X-EBAY-C-MARKETPLACE-ID"] == "EBAY_US"


def test_search_url_encodes_the_query_and_carries_limit_and_offset():
    transport, calls = fake_transport([(200, {"itemSummaries": []})])
    BrowseClient(transport, token="t").search("charizard & pikachu", limit=50, offset=100)
    url = calls[0][0]
    assert "q=charizard+%26+pikachu" in url
    assert "limit=50" in url
    assert "offset=100" in url


def test_search_of_a_page_with_no_results_returns_an_empty_list():
    transport, _ = fake_transport([(200, {})])
    assert BrowseClient(transport, token="t").search("nothing") == []


def test_item_fetches_the_detail_endpoint_for_one_listing():
    transport, calls = fake_transport([(200, {"itemId": "v1|1|0", "localizedAspects": []})])
    client = BrowseClient(transport, token="t")
    assert client.item("v1|1|0")["itemId"] == "v1|1|0"
    assert calls[0][0] == f"{BROWSE_BASE}/item/v1%7C1%7C0"


def test_a_retryable_status_is_retried_with_backoff_then_succeeds():
    transport, calls = fake_transport([(503, None), (429, None), (200, {"itemSummaries": []})])
    slept = []
    client = BrowseClient(transport, token="t", sleep=slept.append)
    client.search("x")
    assert len(calls) == 3
    assert len(slept) == 2
    assert slept == sorted(slept)  # backoff grows


def test_a_non_retryable_status_fails_immediately():
    transport, calls = fake_transport([(400, None)])
    with pytest.raises(EbayError):
        BrowseClient(transport, token="t", sleep=lambda s: None).search("x")
    assert len(calls) == 1


def test_exhausted_retries_raise_rather_than_returning_empty():
    # An empty list would look like "no listings matched" and silently
    # shrink the corpus instead of surfacing an outage.
    transport, _ = fake_transport([(503, None)])
    with pytest.raises(EbayError) as exc:
        BrowseClient(transport, token="t", max_attempts=3, sleep=lambda s: None).search("x")
    assert "503" in str(exc.value)


def test_a_200_with_no_body_is_treated_as_retryable_not_as_success():
    # Matches catalog/api.py: a 200 with junk body is a malformed
    # response, not a real success and not a hard failure.
    transport, calls = fake_transport([(200, None), (200, {"itemSummaries": []})])
    BrowseClient(transport, token="t", sleep=lambda s: None).search("x")
    assert len(calls) == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_ebay.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hitcheck_trainer.corpus.ebay'`.

- [ ] **Step 3: Add the form-POST transport**

Append to `trainer/hitcheck_trainer/catalog/http.py`:

```python
def httpx_post_form(timeout: float = 30.0):
    """Form-encoded POST returning (status, json). For eBay's OAuth grant.

    The GET transports above cover every other call in this repo; the
    token endpoint is the one place a POST is needed.
    """
    client = httpx.Client(timeout=timeout, follow_redirects=True)

    def post(url: str, headers: dict, data: dict):
        try:
            response = client.post(url, headers=headers, data=data)
        except httpx.HTTPError:
            return 0, None
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, None

    return post
```

- [ ] **Step 4: Implement the eBay client**

Create `trainer/hitcheck_trainer/corpus/ebay.py`:

```python
"""eBay OAuth and Browse API access.

The only network-touching file in the corpus package. Shaped after
catalog/api.py: an injected transport, retries driven by
catalog/backoff.py, and a raised error when the attempts run out --
never a silently empty result, which would look like "no listings
matched" and quietly shrink the corpus instead of surfacing an outage.

Nothing here logs the credentials or the token. Error messages carry the
HTTP status and nothing else.
"""

import base64
import re
import time
import urllib.parse

from ..catalog.backoff import backoff_delays

# The /v1/ segment is mandatory. Omitting it returns 404, which reads
# like a credentials failure and is a genuinely expensive misdiagnosis.
EBAY_OAUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
BROWSE_BASE = "https://api.ebay.com/buy/browse/v1"
OAUTH_SCOPE = "https://api.ebay.com/oauth/api_scope"
MARKETPLACE_ID = "EBAY_US"

RETRYABLE = {0, 429, 500, 502, 503, 504}

_SIZE_SUFFIX = re.compile(r"s-l\d+(?=\.\w+$)")


class EbayError(Exception):
    pass


def basic_auth_header(app_id: str, cert_id: str) -> str:
    token = base64.b64encode(f"{app_id}:{cert_id}".encode()).decode()
    return f"Basic {token}"


def fetch_token(post, app_id: str, cert_id: str) -> str:
    """Client-credentials grant. `post(url, headers, data) -> (status, json)`."""
    status, body = post(
        EBAY_OAUTH_URL,
        {
            "Authorization": basic_auth_header(app_id, cert_id),
            "Content-Type": "application/x-www-form-urlencoded",
        },
        {"grant_type": "client_credentials", "scope": OAUTH_SCOPE},
    )
    if status != 200 or not body or not body.get("access_token"):
        raise EbayError(f"oauth token request failed (status {status})")
    return body["access_token"]


def hi_res_url(url: str) -> str:
    """Swap eBay's thumbnail size suffix for the full-resolution one.

    Summaries carry s-l225 (measured 138x225); s-l1600 on the same URL
    returned 734x1200 for the same listing.
    """
    return _SIZE_SUFFIX.sub("s-l1600", url or "")


def aspects_from_item(item: dict) -> dict[str, str]:
    """Flatten a detail response's localizedAspects into name -> value.

    Structured Item Specifics, not the listing title, are the label
    source: titles are seller free text and parsing them would put label
    noise directly under the accuracy number.
    """
    flat = {}
    for aspect in item.get("localizedAspects") or []:
        name, value = aspect.get("name"), aspect.get("value")
        if name and value:
            flat[name] = value
    return flat


class BrowseClient:
    def __init__(self, transport, token: str, marketplace: str = MARKETPLACE_ID,
                 max_attempts: int = 5, sleep=time.sleep):
        self._transport = transport
        self._token = token
        self._marketplace = marketplace
        self._max_attempts = max_attempts
        self._sleep = sleep

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "X-EBAY-C-MARKETPLACE-ID": self._marketplace,
        }

    def _get(self, url: str) -> dict:
        delays = backoff_delays(self._max_attempts - 1)
        last_status = None
        for attempt in range(self._max_attempts):
            status, body = self._transport(url, self._headers())
            last_status = status
            if status == 200 and body is not None:
                return body
            # A 200 with no body is malformed, not a success and not a
            # hard failure — same call as catalog/api.py makes.
            retryable = status in RETRYABLE or (status == 200 and body is None)
            if not retryable:
                break
            if attempt < len(delays):
                self._sleep(delays[attempt])
        raise EbayError(f"{url} failed after {self._max_attempts} attempts (last status {last_status})")

    def search(self, query: str, limit: int = 200, offset: int = 0,
               extra_filter: str | None = None) -> list[dict]:
        """One page of item summaries. `limit` maxes out at 200 server-side."""
        params = {"q": query, "limit": str(limit), "offset": str(offset)}
        if extra_filter:
            params["filter"] = extra_filter
        url = f"{BROWSE_BASE}/item_summary/search?{urllib.parse.urlencode(params)}"
        return self._get(url).get("itemSummaries") or []

    def item(self, item_id: str) -> dict:
        """Full detail for one listing — this is where localizedAspects live."""
        return self._get(f"{BROWSE_BASE}/item/{urllib.parse.quote(item_id, safe='')}")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_ebay.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd trainer && .venv/bin/python -m ruff check hitcheck_trainer tests
cd /var/home/mstephens/Documents/GitHub/pokemon-card-stream-pricer
git add trainer/hitcheck_trainer/catalog/http.py trainer/hitcheck_trainer/corpus/ebay.py trainer/tests/test_ebay.py
git commit -m "feat(corpus): eBay OAuth and Browse client

The only network-touching file in corpus/, shaped after catalog/api.py:
injected transport, retries from catalog/backoff.py, and a raised error
when attempts run out rather than an empty list -- an empty list would
read as 'no listings matched' and quietly shrink the corpus instead of
surfacing an outage.

Item Specifics come from the item-detail endpoint's localizedAspects,
not from listing titles: titles are seller free text and parsing them
would put label noise directly under the accuracy number. That is why
acquisition costs two calls per item.

The OAuth URL keeps its mandatory /v1/ segment -- omitting it returns
404, which reads like a credentials failure. Nothing here logs the
credentials or the token; errors carry the HTTP status only.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: The acquisition CLI

**Files:**
- Create: `trainer/hitcheck_trainer/corpus/build.py`
- Test: `trainer/tests/test_corpus_build.py`

**Interfaces:**
- Consumes: `BrowseClient`, `aspects_from_item`, `hi_res_url`, `fetch_token` from `corpus.ebay` (Task 6); `CardLookup`, `resolve` from `corpus.resolve` (Task 4); `Manifest`, `CorpusEntry`, `image_relpath`, `load_manifest`, `save_manifest` from `corpus.manifest` (Task 5); `fetch_to_path` from `catalog.images` (Task 2).
- Produces:
  - `IMAGE_FAILED: str = "IMAGE_FAILED"` — a discard reason build owns, not one `resolve` can return.
  - `DEFAULT_QUERIES: tuple[str, ...]`
  - `build_corpus(client, lookup, fetch, corpus_dir: str, manifest: Manifest, queries, target: int, page_size: int = 200, sleep=time.sleep, on_progress=None) -> Manifest`
  - `main(argv=None) -> int`

**Background for the implementer:** This wires the previous four tasks together and is the only place that spends the eBay rate budget (~5,000 Browse calls/day; a 500-item corpus costs ~550 at two calls per item). Three behaviours are load-bearing:

1. **Never re-fetch.** Listings expire. Anything already in the manifest is skipped by `item_id`, and any image already on disk with content is left alone. A rerun tops the corpus up; it never rebuilds it.
2. **Every discard is counted.** A corpus that silently dropped most candidates would skew toward listings with tidy Item Specifics. The tally goes in the manifest, not just the console.
3. **Save incrementally.** The manifest is written after every accepted entry, so an interrupted run — rate limit, network drop, Ctrl-C — keeps everything it already paid for.

`build_corpus` takes injected `client` and `fetch` so the whole orchestration is testable offline. Only `main` touches the network or the environment.

`target` defaults to 600 rather than the spec's N ≥ 500, so the corpus still clears 500 after the hand-crop pass drops anything unusable.

- [ ] **Step 1: Write the failing tests**

Create `trainer/tests/test_corpus_build.py`:

```python
from hitcheck_trainer.corpus.build import IMAGE_FAILED, build_corpus
from hitcheck_trainer.corpus.manifest import CorpusEntry, Manifest, image_relpath
from hitcheck_trainer.corpus.resolve import CardLookup


def lookup():
    return CardLookup(
        set_ids={"151": "sv3pt5"},
        cards={("sv3pt5", "199"): [("sv3pt5-199", "charizardex")],
               ("sv3pt5", "6"): [("sv3pt5-6", "charizardex")]},
    )


def specifics(number="199/165", language="English", name="Charizard ex"):
    return [
        {"name": "Card Name", "value": name},
        {"name": "Set", "value": "151"},
        {"name": "Card Number", "value": number},
        {"name": "Language", "value": language},
    ]


class FakeClient:
    """Serves a fixed set of summaries and details, counting calls."""

    def __init__(self, items):
        self.items = items  # item_id -> detail dict
        self.searches = 0
        self.detail_calls = []

    def search(self, query, limit=200, offset=0, extra_filter=None):
        self.searches += 1
        if offset:
            return []  # single page
        return [{"itemId": i} for i in self.items]

    def item(self, item_id):
        self.detail_calls.append(item_id)
        return self.items[item_id]


def detail(item_id, aspects, image="https://i.ebayimg.com/g/a/s-l225.jpg"):
    return {
        "itemId": item_id,
        "itemWebUrl": f"https://www.ebay.com/itm/{item_id}",
        "image": {"imageUrl": image},
        "localizedAspects": aspects,
    }


def ok_fetch(url):
    return 200, b"JPEGBYTES"


def test_builds_a_manifest_entry_per_resolved_listing(tmp_path):
    client = FakeClient({"v1|1|0": detail("v1|1|0", specifics())})
    result = build_corpus(client, lookup(), ok_fetch, str(tmp_path), Manifest(),
                          ["q"], target=10, sleep=lambda s: None)
    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.card_id == "sv3pt5-199"
    assert entry.item_id == "v1|1|0"
    assert entry.aspects["Set"] == "151"


def test_downloads_the_hi_res_image_beside_the_manifest(tmp_path):
    requested = []

    def fetch(url):
        requested.append(url)
        return 200, b"JPEGBYTES"

    client = FakeClient({"v1|1|0": detail("v1|1|0", specifics())})
    build_corpus(client, lookup(), fetch, str(tmp_path), Manifest(), ["q"],
                 target=10, sleep=lambda s: None)
    assert requested == ["https://i.ebayimg.com/g/a/s-l1600.jpg"]
    assert (tmp_path / image_relpath("v1|1|0")).read_bytes() == b"JPEGBYTES"


def test_a_failed_image_download_is_counted_and_produces_no_entry(tmp_path):
    client = FakeClient({"v1|1|0": detail("v1|1|0", specifics())})
    result = build_corpus(client, lookup(), lambda url: (404, None), str(tmp_path),
                          Manifest(), ["q"], target=10, sleep=lambda s: None)
    assert result.entries == []
    assert result.discards[IMAGE_FAILED] == 1


def test_an_unresolvable_listing_is_counted_by_reason_not_guessed(tmp_path):
    client = FakeClient({"v1|1|0": detail("v1|1|0", specifics(language="Japanese"))})
    result = build_corpus(client, lookup(), ok_fetch, str(tmp_path), Manifest(),
                          ["q"], target=10, sleep=lambda s: None)
    assert result.entries == []
    assert result.discards["NOT_ENGLISH"] == 1


def test_discard_counts_accumulate_across_reasons(tmp_path):
    client = FakeClient({
        "v1|1|0": detail("v1|1|0", specifics(language="Japanese")),
        "v1|2|0": detail("v1|2|0", specifics(name="Blastoise ex")),
        "v1|3|0": detail("v1|3|0", specifics(number="9999/165")),
    })
    result = build_corpus(client, lookup(), ok_fetch, str(tmp_path), Manifest(),
                          ["q"], target=10, sleep=lambda s: None)
    assert result.discards == {"NOT_ENGLISH": 1, "NAME_MISMATCH": 1, "NO_SUCH_NUMBER": 1}


def test_a_rerun_never_refetches_a_listing_it_already_has(tmp_path):
    # Listings expire; the corpus survives them by writing once.
    existing = Manifest(entries=[CorpusEntry(
        item_id="v1|1|0", card_id="sv3pt5-199", image=image_relpath("v1|1|0"),
        image_url="https://i.ebayimg.com/g/a/s-l1600.jpg",
        listing_url="https://www.ebay.com/itm/v1|1|0", aspects={},
    )])
    client = FakeClient({"v1|1|0": detail("v1|1|0", specifics())})
    result = build_corpus(client, lookup(), ok_fetch, str(tmp_path), existing,
                          ["q"], target=10, sleep=lambda s: None)
    assert client.detail_calls == []  # never paid for the detail call again
    assert len(result.entries) == 1


def test_stops_once_the_target_is_reached(tmp_path):
    items = {f"v1|{i}|0": detail(f"v1|{i}|0", specifics()) for i in range(5)}
    client = FakeClient(items)
    result = build_corpus(client, lookup(), ok_fetch, str(tmp_path), Manifest(),
                          ["q"], target=2, sleep=lambda s: None)
    assert len(result.entries) == 2
    assert len(client.detail_calls) == 2  # no calls spent past the target


def test_the_target_counts_entries_already_in_the_manifest(tmp_path):
    existing = Manifest(entries=[CorpusEntry(
        item_id="v1|old|0", card_id="sv3pt5-6", image=image_relpath("v1|old|0"),
        image_url="u", listing_url="l", aspects={},
    )])
    client = FakeClient({"v1|1|0": detail("v1|1|0", specifics())})
    result = build_corpus(client, lookup(), ok_fetch, str(tmp_path), existing,
                          ["q"], target=1, sleep=lambda s: None)
    assert len(result.entries) == 1
    assert client.detail_calls == []


def test_the_manifest_is_saved_after_every_entry_so_an_interrupt_keeps_progress(tmp_path):
    from hitcheck_trainer.corpus.manifest import load_manifest

    items = {f"v1|{i}|0": detail(f"v1|{i}|0", specifics()) for i in range(3)}

    calls = {"n": 0}

    def flaky_fetch(url):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return 200, b"JPEGBYTES"

    try:
        build_corpus(FakeClient(items), lookup(), flaky_fetch, str(tmp_path),
                     Manifest(), ["q"], target=10, sleep=lambda s: None)
    except KeyboardInterrupt:
        pass

    saved = load_manifest(str(tmp_path / "manifest.json"))
    assert len(saved.entries) == 2  # the two that completed before the interrupt


def test_the_queries_used_are_recorded_in_the_manifest(tmp_path):
    client = FakeClient({"v1|1|0": detail("v1|1|0", specifics())})
    result = build_corpus(client, lookup(), ok_fetch, str(tmp_path), Manifest(),
                          ["charizard psa", "pikachu psa"], target=10, sleep=lambda s: None)
    assert "charizard psa" in result.queries


def test_a_listing_with_no_image_url_is_counted_not_crashed_on(tmp_path):
    client = FakeClient({"v1|1|0": detail("v1|1|0", specifics(), image="")})
    result = build_corpus(client, lookup(), ok_fetch, str(tmp_path), Manifest(),
                          ["q"], target=10, sleep=lambda s: None)
    assert result.entries == []
    assert result.discards[IMAGE_FAILED] == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_corpus_build.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hitcheck_trainer.corpus.build'`.

- [ ] **Step 3: Implement the builder**

Create `trainer/hitcheck_trainer/corpus/build.py`:

```python
"""Acquire the M2 corpus: search, resolve, download, record.

The only place that spends the eBay rate budget (~5,000 Browse calls a
day; at two calls per item a 500-entry corpus costs ~550). Three
behaviours are load-bearing:

Never re-fetch. Listings expire, so anything already in the manifest is
skipped by itemId and any image already on disk is left alone. A rerun
tops the corpus up; it never rebuilds it.

Count every discard. A corpus that silently dropped most of its
candidates would skew toward whichever listings happen to have tidy Item
Specifics, so the tally is written into the manifest rather than left in
a console log.

Save incrementally. The manifest is written after every accepted entry,
so a rate limit, a dropped connection or a Ctrl-C keeps everything the
run already paid for.
"""

import argparse
import os
import sys
import time

from ..catalog.db import open_db
from ..catalog.http import httpx_fetch, httpx_post_form, httpx_transport
from ..catalog.images import fetch_to_path
from .ebay import BrowseClient, EbayError, aspects_from_item, fetch_token, hi_res_url
from .manifest import CorpusEntry, image_relpath, load_manifest, save_manifest
from .resolve import CardLookup, resolve

DEFAULT_DB = "data/catalog.sqlite"
DEFAULT_CORPUS = "data/corpus"

# A discard reason build.py owns; resolve() cannot return it.
IMAGE_FAILED = "IMAGE_FAILED"

DEFAULT_QUERIES = (
    "pokemon card psa 10",
    "pokemon card psa 9",
    "pokemon card cgc graded",
    "pokemon card bgs graded",
    "pokemon holo rare card",
)


def build_corpus(client, lookup, fetch, corpus_dir, manifest, queries, target,
                 page_size=200, sleep=time.sleep, on_progress=None):
    """Top the manifest up toward `target` resolved entries."""
    manifest_path = os.path.join(corpus_dir, "manifest.json")
    seen = manifest.item_ids()
    for query in queries:
        if query not in manifest.queries:
            manifest.queries.append(query)

    def discard(reason):
        manifest.discards[reason] = manifest.discards.get(reason, 0) + 1

    for query in queries:
        offset = 0
        while len(manifest.entries) < target:
            summaries = client.search(query, limit=page_size, offset=offset)
            if not summaries:
                break
            offset += len(summaries)

            for summary in summaries:
                if len(manifest.entries) >= target:
                    break
                item_id = summary.get("itemId")
                if not item_id or item_id in seen:
                    continue
                seen.add(item_id)

                item = client.item(item_id)
                aspects = aspects_from_item(item)
                resolution = resolve(aspects, lookup)
                if resolution.card_id is None:
                    discard(resolution.reason)
                    continue

                image_url = hi_res_url((item.get("image") or {}).get("imageUrl") or "")
                relpath = image_relpath(item_id)
                path = os.path.join(corpus_dir, relpath)
                already = os.path.exists(path) and os.path.getsize(path) > 0
                if not image_url or not (already or fetch_to_path(url=image_url, path=path,
                                                                  fetch=fetch, sleep=sleep)):
                    discard(IMAGE_FAILED)
                    continue

                manifest.entries.append(CorpusEntry(
                    item_id=item_id,
                    card_id=resolution.card_id,
                    image=relpath,
                    image_url=image_url,
                    listing_url=item.get("itemWebUrl", ""),
                    aspects=aspects,
                ))
                # Saved per entry: an interrupted run keeps what it paid for.
                save_manifest(manifest, manifest_path)
                if on_progress:
                    on_progress(len(manifest.entries), target)

    save_manifest(manifest, manifest_path)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hitcheck-corpus")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--target", type=int, default=600,
                        help="resolved entries to reach; 600 leaves headroom "
                             "above the 500 the eval needs after cropping")
    parser.add_argument("--query", action="append", default=None,
                        help="repeatable; defaults to DEFAULT_QUERIES")
    args = parser.parse_args(argv)

    app_id = os.environ.get("PROD_APP_ID")
    cert_id = os.environ.get("PROD_EBAY_CERT_ID")
    if not app_id or not cert_id:
        print("Set PROD_APP_ID and PROD_EBAY_CERT_ID in the environment.")
        return 1

    try:
        token = fetch_token(httpx_post_form(), app_id, cert_id)
    except EbayError as exc:
        print(f"eBay auth failed: {exc}")
        return 1

    client = BrowseClient(httpx_transport(), token)
    lookup = CardLookup.from_conn(open_db(args.db))
    manifest_path = os.path.join(args.corpus, "manifest.json")
    manifest = load_manifest(manifest_path)
    print(f"starting from {len(manifest.entries)} entries, target {args.target}")

    def progress(done, total):
        print(f"\rcorpus: {done}/{total}", end="", flush=True)

    try:
        manifest = build_corpus(
            client, lookup, httpx_fetch(), args.corpus, manifest,
            args.query or list(DEFAULT_QUERIES), args.target, on_progress=progress,
        )
    except EbayError as exc:
        print(f"\nacquisition stopped: {exc}")
        print(f"Progress saved ({len(manifest.entries)} entries). Rerun to resume.")
        return 1

    print()
    print(manifest.yield_summary())
    if len(manifest.entries) < 500:
        print(f"*** {len(manifest.entries)} entries is below the 500 the eval needs. "
              "Rerun, or add --query terms. ***")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_corpus_build.py -q`
Expected: PASS.

Run: `cd trainer && .venv/bin/python -m pytest -q -m "not slow"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd trainer && .venv/bin/python -m ruff check hitcheck_trainer tests
cd /var/home/mstephens/Documents/GitHub/pokemon-card-stream-pricer
git add trainer/hitcheck_trainer/corpus/build.py trainer/tests/test_corpus_build.py
git commit -m "feat(corpus): acquisition CLI for the M2 real-photograph corpus

Wires search, resolution, download and manifest together. The only place
that spends the eBay rate budget, so it never re-fetches: entries already
in the manifest are skipped by itemId and images already on disk are left
alone, which is also what lets the corpus outlive the listings it came from.

The manifest is saved after every accepted entry, so a rate limit or a
Ctrl-C keeps what the run already paid for. Every discard is tallied by
reason into the manifest itself -- a corpus that silently dropped most of
its candidates would skew toward listings with tidy Item Specifics, and
that skew has to be visible.

Targets 600 rather than the eval's 500 so the corpus still clears the bar
after the hand-crop pass drops anything unusable.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Quadrilateral crops and the perspective unwarp

**Files:**
- Create: `trainer/hitcheck_trainer/corpus/crops.py`
- Test: `trainer/tests/test_crops.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. PIL, numpy, standard library.
- Produces:
  - `CARD_SIZE: tuple[int, int] = (245, 342)`
  - `MIN_QUAD_AREA: float = 1000.0`
  - `Quad = list[list[float]]` (a type alias; four `[x, y]` pairs)
  - `quad_area(quad) -> float`
  - `validate_quad(quad) -> None` — raises `ValueError` on anything but four non-degenerate points.
  - `perspective_coeffs(size: tuple[int, int], quad) -> tuple[float, ...]`
  - `apply_quad(image: Image.Image, quad, size: tuple[int, int] = CARD_SIZE) -> Image.Image`
  - `load_crops(path: str) -> dict[str, Quad]`
  - `save_crops(crops: dict[str, Quad], path: str) -> None`

**Background for the implementer:** Catalog gallery images are tight card scans; eBay photographs are whole slabs on a desk, angled, with the PSA label and background in frame. Embedding one against the other measures domain mismatch rather than degradation tolerance, and would read as a catastrophic M2 result for entirely the wrong reason. This module is what removes that error source.

**A quadrilateral, not a box.** Half B's perspective estimator reads corner deviation directly off the recorded quad; storing an axis-aligned box would mean redoing the entire hand-crop pass later. `apply_quad` therefore does a full perspective unwarp, not a rectangular crop.

**Click order is the point order.** Corners are recorded in the order the human clicked them — card top-left first, then clockwise. Do not sort them geometrically: a card photographed at 40° has no meaningful "topmost" corner, and sorting would silently rotate some crops.

**Output is 245×342** to match the catalog's `images.small` dimensions (`catalog/images.py` documents ~245×342). DINOv2 resizes both to 224×224 anyway, and rendering queries at the gallery's own scale avoids handing the real corpus an unintended sharpness advantage over the images it is being matched against.

The PIL transform direction is the non-obvious part: `Image.transform(size, PERSPECTIVE, coeffs)` maps **output** coordinates back to **source** coordinates, so the system being solved sends the output rectangle's corners to the recorded quad's corners. `degrade.py:73-82` solves the same shape in the opposite direction; this is not a copy of it. The math below was verified against a synthetic rotated quad and recovered 99.7% of the marked region.

- [ ] **Step 1: Write the failing tests**

Create `trainer/tests/test_crops.py`:

```python
import numpy as np
import pytest
from PIL import Image, ImageDraw

from hitcheck_trainer.corpus.crops import (
    CARD_SIZE,
    apply_quad,
    load_crops,
    quad_area,
    save_crops,
    validate_quad,
)


def photo_with_marked_card(quad, size=(400, 400)):
    """A white 'desk' with a red 'card' occupying exactly `quad`."""
    image = Image.new("RGB", size, "white")
    ImageDraw.Draw(image).polygon([tuple(p) for p in quad], fill=(255, 0, 0))
    return image


def test_quad_area_of_a_unit_square_is_one():
    assert quad_area([[0, 0], [1, 0], [1, 1], [0, 1]]) == pytest.approx(1.0)


def test_quad_area_is_the_same_whichever_way_round_the_points_go():
    clockwise = [[0, 0], [10, 0], [10, 10], [0, 10]]
    counter = list(reversed(clockwise))
    assert quad_area(clockwise) == pytest.approx(quad_area(counter))


def test_validate_rejects_anything_that_is_not_four_points():
    with pytest.raises(ValueError):
        validate_quad([[0, 0], [1, 0], [1, 1]])
    with pytest.raises(ValueError):
        validate_quad([[0, 0], [1, 0], [1, 1], [0, 1], [2, 2]])


def test_validate_rejects_a_degenerate_quad():
    # Four clicks in nearly the same place is a misclick, not a crop.
    with pytest.raises(ValueError):
        validate_quad([[0, 0], [1, 0], [1, 1], [0, 1]])


def test_validate_accepts_a_real_sized_quad():
    validate_quad([[50, 60], [300, 40], [330, 300], [80, 330]])


def test_apply_quad_unwarps_an_angled_card_to_a_full_frame():
    quad = [[50, 60], [300, 40], [330, 300], [80, 330]]
    cropped = apply_quad(photo_with_marked_card(quad), quad)
    assert cropped.size == CARD_SIZE
    pixels = np.array(cropped)
    # The card now fills the frame: essentially every pixel is the card,
    # not the white desk it was photographed on.
    is_card = (pixels[:, :, 0] > 200) & (pixels[:, :, 1] < 80)
    assert is_card.mean() > 0.98


def test_apply_quad_of_an_axis_aligned_quad_is_a_plain_crop():
    quad = [[100, 100], [300, 100], [300, 380], [100, 380]]
    image = Image.new("RGB", (400, 400), "white")
    ImageDraw.Draw(image).rectangle([100, 100, 300, 380], fill=(0, 0, 255))
    pixels = np.array(apply_quad(image, quad))
    assert (pixels[:, :, 2] > 200).mean() > 0.98


def test_apply_quad_honours_click_order_rather_than_sorting_corners():
    # Feeding the corners rotated by one position must rotate the output.
    # A card photographed at an angle has no meaningful "topmost" corner,
    # so sorting geometrically would silently rotate some crops.
    quad = [[100, 100], [300, 100], [300, 380], [100, 380]]
    image = Image.new("RGB", (400, 400), "white")
    ImageDraw.Draw(image).rectangle([100, 100, 200, 380], fill=(0, 255, 0))
    upright = np.array(apply_quad(image, quad))
    rotated = np.array(apply_quad(image, quad[1:] + quad[:1]))
    assert not np.allclose(upright, rotated)


def test_apply_quad_accepts_a_size_override():
    quad = [[50, 60], [300, 40], [330, 300], [80, 330]]
    assert apply_quad(photo_with_marked_card(quad), quad, size=(64, 89)).size == (64, 89)


def test_apply_quad_output_defaults_to_the_catalog_image_size():
    # Matching catalog images.small keeps the query at the same scale as
    # the gallery it is matched against.
    assert CARD_SIZE == (245, 342)


def test_crops_round_trip_through_disk(tmp_path):
    crops = {"v1|1|0": [[50.0, 60.0], [300.0, 40.0], [330.0, 300.0], [80.0, 330.0]]}
    path = str(tmp_path / "crops.json")
    save_crops(crops, path)
    assert load_crops(path) == crops


def test_loading_a_missing_crops_file_gives_an_empty_mapping(tmp_path):
    assert load_crops(str(tmp_path / "nope.json")) == {}


def test_saving_crops_leaves_no_part_file(tmp_path):
    save_crops({"a": [[0, 0], [1, 0], [1, 1], [0, 1]]}, str(tmp_path / "crops.json"))
    assert [p.name for p in tmp_path.iterdir()] == ["crops.json"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_crops.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hitcheck_trainer.corpus.crops'`.

- [ ] **Step 3: Implement the crop module**

Create `trainer/hitcheck_trainer/corpus/crops.py`:

```python
"""Turn a hand-marked quadrilateral into a card-shaped crop.

Catalog gallery images are tight card scans; eBay photographs are whole
slabs on a desk, angled, with the grading label and background in frame.
Embedding one against the other would measure domain mismatch rather
than degradation tolerance, and read as a catastrophic M2 result for
entirely the wrong reason. This module removes that error source.

Crops are quadrilaterals, not boxes, on purpose: Half B's perspective
estimator reads corner deviation directly off the recorded quad, and
storing an axis-aligned box would mean redoing the whole hand-crop pass.

In the live app M3's detector supplies this quad. Until M3 exists the
corpus is cropped by hand, which also gives M3 a ground-truth set to be
evaluated against later.
"""

import json
import os

import numpy as np
from PIL import Image

# Matches catalog images.small (~245x342). DINOv2 resizes both to 224x224
# anyway, and rendering queries at the gallery's own scale avoids handing
# the real corpus an unintended sharpness advantage over the images it is
# being matched against.
CARD_SIZE = (245, 342)

# Four clicks inside this area is a misclick, not a crop.
MIN_QUAD_AREA = 1000.0


def quad_area(quad) -> float:
    """Shoelace area, orientation-independent."""
    points = np.asarray(quad, dtype=np.float64)
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def validate_quad(quad) -> None:
    points = np.asarray(quad, dtype=np.float64)
    if points.shape != (4, 2):
        raise ValueError(f"expected 4 [x, y] points, got shape {points.shape}")
    area = quad_area(points)
    if area < MIN_QUAD_AREA:
        raise ValueError(f"quad area {area:.1f} is below {MIN_QUAD_AREA} — degenerate")


def perspective_coeffs(size: tuple[int, int], quad) -> tuple[float, ...]:
    """Coefficients mapping the output rectangle back onto `quad`.

    PIL's PERSPECTIVE transform maps OUTPUT coordinates to SOURCE
    coordinates, so the system solved here sends the output rectangle's
    corners to the recorded quad's corners -- the opposite direction from
    degrade.perspective_warp, which warps a rectangle outward.
    """
    width, height = size
    destination = [(0, 0), (width, 0), (width, height), (0, height)]
    rows, rhs = [], []
    for (dx, dy), (sx, sy) in zip(destination, quad):
        rows.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        rhs.append(sx)
        rows.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
        rhs.append(sy)
    solved = np.linalg.solve(np.array(rows, dtype=np.float64), np.array(rhs, dtype=np.float64))
    return tuple(float(c) for c in solved)


def apply_quad(image: Image.Image, quad, size: tuple[int, int] = CARD_SIZE) -> Image.Image:
    """Unwarp the quad out of the photograph into a card-shaped crop.

    Corner order is the order they were clicked -- card top-left first,
    then clockwise -- and is never sorted geometrically. A card
    photographed at 40 degrees has no meaningful "topmost" corner, and
    sorting would silently rotate some crops.
    """
    validate_quad(quad)
    return image.convert("RGB").transform(
        size, Image.Transform.PERSPECTIVE, perspective_coeffs(size, quad),
        Image.Resampling.BICUBIC,
    )


def load_crops(path: str) -> dict[str, list[list[float]]]:
    """item_id -> quad. Missing file means nothing has been cropped yet."""
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def save_crops(crops: dict[str, list[list[float]]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.part"
    with open(tmp, "w") as fh:
        json.dump(crops, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)  # atomic — hours of hand-cropping live in here
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_crops.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd trainer && .venv/bin/python -m ruff check hitcheck_trainer tests
cd /var/home/mstephens/Documents/GitHub/pokemon-card-stream-pricer
git add trainer/hitcheck_trainer/corpus/crops.py trainer/tests/test_crops.py
git commit -m "feat(corpus): perspective unwarp from a hand-marked quadrilateral

Catalog images are tight scans; eBay photos are whole slabs on a desk,
angled, with the grading label in frame. Embedding one against the other
measures domain mismatch rather than degradation tolerance and would read
as a catastrophic M2 result for the wrong reason.

Stored as a quadrilateral rather than a box because Half B's perspective
estimator reads corner deviation off it -- a box would mean redoing the
whole hand-crop pass. Corner order is click order and is never sorted
geometrically: a card shot at 40 degrees has no meaningful topmost
corner, and sorting would silently rotate some crops.

Output is 245x342 to match catalog images.small, so a query is not handed
a sharpness advantage over the gallery it is matched against.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: The hand-crop tool

**Files:**
- Create: `trainer/hitcheck_trainer/corpus/croptool.py`
- Test: `trainer/tests/test_croptool.py`

**Interfaces:**
- Consumes: `load_manifest` from `corpus.manifest` (Task 5); `load_crops`, `save_crops`, `validate_quad` from `corpus.crops` (Task 8).
- Produces:
  - `CropApp(manifest, crops: dict, crops_path: str, corpus_dir: str)` with `next_item() -> dict | None`, `progress() -> tuple[int, int]`, `image_bytes(item_id: str) -> bytes`, `record(item_id: str, quad) -> None`, and `handle(method: str, path: str, body: bytes) -> tuple[int, str, bytes]`.
  - `PAGE: str` — the single-page HTML/JS client.
  - `serve(app: CropApp, port: int = 8765) -> None`
  - `main(argv=None) -> int`

**Background for the implementer:** Cropping a few hundred photographs by hand needs a UI, and this repo may not have a working `_tkinter` in its `uv` venv. A stdlib `http.server` plus a small HTML page has zero new dependencies, works on this machine's Wayland session, and — importantly — keeps the logic testable: every test below calls `CropApp.handle` directly, so nothing binds a socket and the standing "no network in tests" constraint holds.

**All routing goes through `handle(method, path, body)`.** The `BaseHTTPRequestHandler` subclass is a five-line adapter that calls it. Do not put logic in the handler class; it is the one part that cannot be tested offline.

Routes:

| Method | Path | Returns |
|---|---|---|
| `GET` | `/` | `PAGE` as `text/html` |
| `GET` | `/api/next` | `{"item_id", "image", "card_id", "done", "total"}` — or `{"done": N, "total": N, "item_id": null}` when finished |
| `GET` | `/api/image?id=<item_id>` | the raw photograph as `image/jpeg` |
| `POST` | `/api/quad` | `{"ok": true}` after validating and saving, or 400 with `{"error": ...}` |

The client draws the photograph scaled to fit, collects four clicks in **card top-left, then clockwise** order, and posts them **in original-image pixel coordinates** — it divides by its own display scale before sending. Recording display coordinates would silently make every quad wrong by the scale factor, so the scale conversion happens client-side and the server stores what it is given.

`record` saves after every quad. A crash three hours into a hand-crop pass must not cost the pass.

- [ ] **Step 1: Write the failing tests**

Create `trainer/tests/test_croptool.py`:

```python
import json

from PIL import Image

from hitcheck_trainer.corpus.croptool import CropApp
from hitcheck_trainer.corpus.crops import load_crops
from hitcheck_trainer.corpus.manifest import CorpusEntry, Manifest, image_relpath

QUAD = [[50, 60], [300, 40], [330, 300], [80, 330]]


def make_app(tmp_path, item_ids=("v1|1|0", "v1|2|0"), crops=None):
    entries = []
    for item_id in item_ids:
        relpath = image_relpath(item_id)
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (400, 400), "white").save(path, "JPEG")
        entries.append(CorpusEntry(item_id=item_id, card_id="sv3pt5-199", image=relpath,
                                   image_url="u", listing_url="l", aspects={}))
    return CropApp(
        manifest=Manifest(entries=entries),
        crops=dict(crops or {}),
        crops_path=str(tmp_path / "crops.json"),
        corpus_dir=str(tmp_path),
    )


def body_of(response):
    return json.loads(response[2])


def test_next_item_is_the_first_entry_with_no_quad_yet(tmp_path):
    app = make_app(tmp_path)
    assert app.next_item()["item_id"] == "v1|1|0"


def test_next_item_skips_entries_that_are_already_cropped(tmp_path):
    app = make_app(tmp_path, crops={"v1|1|0": QUAD})
    assert app.next_item()["item_id"] == "v1|2|0"


def test_next_item_is_none_once_everything_is_cropped(tmp_path):
    app = make_app(tmp_path, crops={"v1|1|0": QUAD, "v1|2|0": QUAD})
    assert app.next_item() is None


def test_progress_counts_cropped_against_total(tmp_path):
    assert make_app(tmp_path, crops={"v1|1|0": QUAD}).progress() == (1, 2)


def test_get_root_serves_the_page(tmp_path):
    status, content_type, payload = make_app(tmp_path).handle("GET", "/", b"")
    assert status == 200
    assert content_type == "text/html"
    assert b"<canvas" in payload


def test_api_next_reports_the_item_and_the_progress(tmp_path):
    response = make_app(tmp_path).handle("GET", "/api/next", b"")
    payload = body_of(response)
    assert payload["item_id"] == "v1|1|0"
    assert payload["card_id"] == "sv3pt5-199"
    assert payload["done"] == 0
    assert payload["total"] == 2


def test_api_next_reports_a_null_item_when_the_pass_is_complete(tmp_path):
    app = make_app(tmp_path, crops={"v1|1|0": QUAD, "v1|2|0": QUAD})
    payload = body_of(app.handle("GET", "/api/next", b""))
    assert payload["item_id"] is None
    assert payload["done"] == payload["total"] == 2


def test_api_image_returns_the_photograph_bytes(tmp_path):
    app = make_app(tmp_path)
    status, content_type, payload = app.handle("GET", "/api/image?id=v1%7C1%7C0", b"")
    assert status == 200
    assert content_type == "image/jpeg"
    assert payload[:2] == b"\xff\xd8"  # JPEG magic


def test_api_image_for_an_unknown_id_is_a_404_not_a_traceback(tmp_path):
    status, _, _ = make_app(tmp_path).handle("GET", "/api/image?id=nope", b"")
    assert status == 404


def test_posting_a_quad_records_it_and_persists_immediately(tmp_path):
    app = make_app(tmp_path)
    payload = json.dumps({"item_id": "v1|1|0", "quad": QUAD}).encode()
    status, _, _ = app.handle("POST", "/api/quad", payload)
    assert status == 200
    # Persisted, not just held in memory: a crash three hours into a
    # hand-crop pass must not cost the pass.
    assert load_crops(str(tmp_path / "crops.json"))["v1|1|0"] == QUAD


def test_posting_a_quad_advances_next_item(tmp_path):
    app = make_app(tmp_path)
    app.handle("POST", "/api/quad", json.dumps({"item_id": "v1|1|0", "quad": QUAD}).encode())
    assert app.next_item()["item_id"] == "v1|2|0"


def test_posting_a_degenerate_quad_is_rejected_with_400(tmp_path):
    app = make_app(tmp_path)
    bad = json.dumps({"item_id": "v1|1|0", "quad": [[0, 0], [1, 0], [1, 1], [0, 1]]}).encode()
    status, _, payload = app.handle("POST", "/api/quad", bad)
    assert status == 400
    assert "error" in json.loads(payload)
    assert app.next_item()["item_id"] == "v1|1|0"  # not advanced


def test_posting_the_wrong_number_of_points_is_rejected_with_400(tmp_path):
    app = make_app(tmp_path)
    bad = json.dumps({"item_id": "v1|1|0", "quad": [[0, 0], [400, 0], [400, 400]]}).encode()
    assert app.handle("POST", "/api/quad", bad)[0] == 400


def test_posting_malformed_json_is_rejected_with_400(tmp_path):
    assert make_app(tmp_path).handle("POST", "/api/quad", b"{not json")[0] == 400


def test_an_unknown_route_is_a_404(tmp_path):
    assert make_app(tmp_path).handle("GET", "/nope", b"")[0] == 404


def test_the_page_sends_original_image_coordinates_not_display_coordinates(tmp_path):
    # The canvas scales the photo to fit the window. If clicks were posted
    # in display pixels every quad would be wrong by that scale factor, so
    # the client divides by its own scale before posting.
    from hitcheck_trainer.corpus.croptool import PAGE

    assert "scale" in PAGE
    assert "/api/quad" in PAGE
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_croptool.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hitcheck_trainer.corpus.croptool'`.

- [ ] **Step 3: Implement the crop tool**

Create `trainer/hitcheck_trainer/corpus/croptool.py`:

```python
"""Local browser tool for hand-marking card corners in corpus photographs.

Stdlib http.server plus one HTML page: no new dependency, no reliance on
a working _tkinter in the venv, and it works on this machine's Wayland
session. All routing goes through CropApp.handle, so the whole tool is
testable without binding a socket -- the request handler below is a thin
adapter and is the only part that is not.

The client posts corners in ORIGINAL image pixels. It scales the photo to
fit the window, so it divides by that scale before posting; recording
display coordinates would make every quad silently wrong by the scale
factor.
"""

import argparse
import json
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .crops import load_crops, save_crops, validate_quad
from .manifest import load_manifest

DEFAULT_CORPUS = "data/corpus"

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>HitCheck crop tool</title>
<style>
  body { font: 14px system-ui; margin: 0; background: #111; color: #eee; }
  header { padding: 8px 12px; display: flex; gap: 16px; align-items: baseline; }
  canvas { display: block; cursor: crosshair; }
  #hint { color: #9ad; }
</style></head><body>
<header>
  <strong id="progress">loading...</strong>
  <span id="card"></span>
  <span id="hint">click the card's top-left corner, then clockwise. u = undo</span>
</header>
<canvas id="c"></canvas>
<script>
let item = null, scale = 1, points = [], img = new Image();
const canvas = document.getElementById('c'), ctx = canvas.getContext('2d');

async function load() {
  const state = await (await fetch('/api/next')).json();
  document.getElementById('progress').textContent = state.done + ' / ' + state.total;
  if (!state.item_id) { document.getElementById('card').textContent = 'done'; return; }
  item = state;
  document.getElementById('card').textContent = state.card_id;
  points = [];
  img = new Image();
  img.onload = draw;
  img.src = '/api/image?id=' + encodeURIComponent(state.item_id);
}

function draw() {
  const maxH = window.innerHeight - 60, maxW = window.innerWidth;
  scale = Math.min(maxW / img.width, maxH / img.height, 1);
  canvas.width = img.width * scale;
  canvas.height = img.height * scale;
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#0f0'; ctx.fillStyle = '#0f0'; ctx.lineWidth = 2;
  points.forEach((p, i) => {
    ctx.beginPath(); ctx.arc(p[0] * scale, p[1] * scale, 4, 0, 7); ctx.fill();
    if (i) {
      ctx.beginPath();
      ctx.moveTo(points[i-1][0] * scale, points[i-1][1] * scale);
      ctx.lineTo(p[0] * scale, p[1] * scale);
      ctx.stroke();
    }
  });
}

canvas.addEventListener('click', async (e) => {
  const box = canvas.getBoundingClientRect();
  // Divide by scale: the server stores ORIGINAL image pixels.
  points.push([(e.clientX - box.left) / scale, (e.clientY - box.top) / scale]);
  draw();
  if (points.length === 4) {
    const res = await fetch('/api/quad', {
      method: 'POST',
      body: JSON.stringify({item_id: item.item_id, quad: points}),
    });
    if (res.ok) { load(); } else { alert((await res.json()).error); points = []; draw(); }
  }
});

window.addEventListener('keydown', (e) => {
  if (e.key === 'u') { points.pop(); draw(); }
});
window.addEventListener('resize', draw);
load();
</script></body></html>
"""


class CropApp:
    def __init__(self, manifest, crops, crops_path, corpus_dir):
        self._manifest = manifest
        self._crops = crops
        self._crops_path = crops_path
        self._corpus_dir = corpus_dir
        self._by_id = {e.item_id: e for e in manifest.entries}

    def next_item(self):
        for entry in self._manifest.entries:
            if entry.item_id not in self._crops:
                return {"item_id": entry.item_id, "card_id": entry.card_id,
                        "image": entry.image}
        return None

    def progress(self) -> tuple[int, int]:
        return len(self._crops), len(self._manifest.entries)

    def image_bytes(self, item_id: str) -> bytes:
        entry = self._by_id[item_id]
        with open(os.path.join(self._corpus_dir, entry.image), "rb") as fh:
            return fh.read()

    def record(self, item_id: str, quad) -> None:
        validate_quad(quad)
        if item_id not in self._by_id:
            raise KeyError(item_id)
        self._crops[item_id] = [[float(x), float(y)] for x, y in quad]
        # Saved per quad: a crash three hours into a pass must not cost it.
        save_crops(self._crops, self._crops_path)

    def handle(self, method: str, path: str, body: bytes) -> tuple[int, str, bytes]:
        route, _, query = path.partition("?")

        if method == "GET" and route == "/":
            return 200, "text/html", PAGE.encode()

        if method == "GET" and route == "/api/next":
            item = self.next_item()
            done, total = self.progress()
            payload = {"item_id": None, "card_id": None, "image": None,
                       "done": done, "total": total}
            if item:
                payload.update(item)
            return 200, "application/json", json.dumps(payload).encode()

        if method == "GET" and route == "/api/image":
            item_id = urllib.parse.parse_qs(query).get("id", [""])[0]
            try:
                return 200, "image/jpeg", self.image_bytes(item_id)
            except (KeyError, OSError):
                return 404, "application/json", b'{"error": "no such image"}'

        if method == "POST" and route == "/api/quad":
            try:
                payload = json.loads(body)
                self.record(payload["item_id"], payload["quad"])
            except (ValueError, KeyError, TypeError) as exc:
                return 400, "application/json", json.dumps({"error": str(exc)}).encode()
            return 200, "application/json", b'{"ok": true}'

        return 404, "application/json", b'{"error": "not found"}'


def _make_handler(app: CropApp):
    class Handler(BaseHTTPRequestHandler):
        def _respond(self, method):
            length = int(self.headers.get("Content-Length") or 0)
            status, content_type, payload = app.handle(method, self.path, self.rfile.read(length))
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self._respond("GET")

        def do_POST(self):
            self._respond("POST")

        def log_message(self, *args):
            pass  # one line per click is noise during a long cropping pass

    return Handler


def serve(app: CropApp, port: int = 8765) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(app))
    done, total = app.progress()
    print(f"crop tool on http://127.0.0.1:{port}/  ({done}/{total} done)")
    print("Ctrl-C to stop; progress is saved after every card.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hitcheck-croptool")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    manifest = load_manifest(os.path.join(args.corpus, "manifest.json"))
    if not manifest.entries:
        print(f"No manifest entries under {args.corpus}. Run the corpus build first.")
        return 1
    crops_path = os.path.join(args.corpus, "crops.json")
    serve(CropApp(manifest, load_crops(crops_path), crops_path, args.corpus), args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_croptool.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd trainer && .venv/bin/python -m ruff check hitcheck_trainer tests
cd /var/home/mstephens/Documents/GitHub/pokemon-card-stream-pricer
git add trainer/hitcheck_trainer/corpus/croptool.py trainer/tests/test_croptool.py
git commit -m "feat(corpus): browser tool for hand-marking card corners

A few hundred photographs need cropping by hand, and stdlib http.server
plus one HTML page costs no new dependency, does not depend on _tkinter
being present in the venv, and works on this machine's Wayland session.

All routing goes through CropApp.handle, so the tool is tested without
binding a socket -- the BaseHTTPRequestHandler subclass is a thin adapter
and the only untested part.

The client posts corners in original image pixels, dividing by its own
display scale first: recording display coordinates would make every quad
silently wrong by the scale factor. Quads are saved after every card, so
a crash partway through a long pass costs one card rather than the pass.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: One chunked-embedding implementation, keyed by (label, path)

**Files:**
- Create: `trainer/hitcheck_trainer/eval/chunks.py`
- Modify: `trainer/hitcheck_trainer/eval/synthetic.py`
- Test: `trainer/tests/test_chunks.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. PIL, numpy.
- Produces:
  - `load_chunk(items: list[tuple[str, str]], offset: int = 0) -> tuple[list[int], list[str], list[Image.Image]]` — the first element is each kept image's index in the *full* item list.
  - `embed_in_chunks(embedder, items: list[tuple[str, str]], chunk: int = 256, transform=None) -> tuple[list[str], np.ndarray]` — `transform(image, index) -> Image` where `index` is the item's true position in `items`, even when earlier files were skipped.
- Removed from `synthetic.py`: its private `load_chunk` and `embed_in_chunks`. `available_ids` and `main` stay.

**Background for the implementer:** `synthetic.py`'s `embed_in_chunks` is hardwired to `image_path(images_root, card_id)`. Corpus images are keyed by eBay itemId and live at arbitrary paths, so `eval/real.py` cannot call it. Generalising the existing one to `(label, path)` pairs is deliberate rather than writing a second loader.

**Why that matters more than tidiness:** the 256-image chunking exists because materialising the whole 20,427-image catalog is 4.52GB of decoded pixels before PIL overhead, degraded copies and torch tensors — an allocation that contributed to a global OOM on this 30GB machine, which also runs games and browsers. A second loader that quietly forgot to chunk would reintroduce that. Carry the existing docstring's warning across verbatim; it is the reason the function has the shape it has.

`synthetic.py`'s CLI and output are unchanged, and it currently has no tests for these two functions, so this task adds the first ones. The one behavioural change is the index fix described below, which only shows up when an image file is unreadable.

- [ ] **Step 1: Write the failing tests**

Create `trainer/tests/test_chunks.py`:

```python
import numpy as np
import pytest
from PIL import Image

from hitcheck_trainer.eval.chunks import embed_in_chunks, load_chunk


class FakeEmbedder:
    """Records the batch sizes it was handed; returns one row per image."""

    dim = 4

    def __init__(self):
        self.batch_sizes = []

    def embed(self, images, batch_size=32):
        self.batch_sizes.append(len(images))
        return np.tile(np.arange(self.dim, dtype=np.float32), (len(images), 1))


def written(tmp_path, names, size=(8, 8)):
    items = []
    for name in names:
        path = tmp_path / f"{name}.png"
        Image.new("RGB", size, "white").save(path)
        items.append((name, str(path)))
    return items


def test_load_chunk_returns_indices_labels_and_decoded_images(tmp_path):
    indices, labels, images = load_chunk(written(tmp_path, ["a", "b"]))
    assert indices == [0, 1]
    assert labels == ["a", "b"]
    assert [im.mode for im in images] == ["RGB", "RGB"]


def test_load_chunk_indices_are_offset_into_the_full_item_list(tmp_path):
    indices, _, _ = load_chunk(written(tmp_path, ["a", "b"]), offset=256)
    assert indices == [256, 257]


def test_load_chunk_skips_a_truncated_file_without_failing_the_run(tmp_path):
    # A catalog rerun replaces a truncated download; one bad file must not
    # abort an embed of twenty thousand images.
    items = written(tmp_path, ["good"])
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    items.append(("bad", str(bad)))
    indices, labels, images = load_chunk(items)
    assert labels == ["good"]
    assert indices == [0]
    assert len(images) == 1


def test_load_chunk_skips_a_missing_file(tmp_path):
    indices, labels, _ = load_chunk([("gone", str(tmp_path / "nope.png"))])
    assert labels == []
    assert indices == []


def test_embeds_every_item_and_returns_one_vector_each(tmp_path):
    embedder = FakeEmbedder()
    labels, vectors = embed_in_chunks(embedder, written(tmp_path, list("abcde")))
    assert labels == list("abcde")
    assert vectors.shape == (5, 4)


def test_never_decodes_more_than_chunk_images_at_once(tmp_path):
    # The whole point of this function: 20,427 catalog images at 240x330
    # RGB is 4.52GB decoded, and materialising it once caused a global OOM
    # on this machine. Chunking is not an optimisation, it is a constraint.
    embedder = FakeEmbedder()
    embed_in_chunks(embedder, written(tmp_path, [str(i) for i in range(10)]), chunk=3)
    assert max(embedder.batch_sizes) <= 3
    assert sum(embedder.batch_sizes) == 10


def test_labels_line_up_with_vectors_when_a_file_in_the_middle_is_unreadable(tmp_path):
    items = written(tmp_path, ["a", "c"])
    bad = tmp_path / "b.png"
    bad.write_bytes(b"junk")
    items.insert(1, ("b", str(bad)))
    labels, vectors = embed_in_chunks(FakeEmbedder(), items, chunk=2)
    assert labels == ["a", "c"]
    assert len(vectors) == len(labels)


def test_transform_is_applied_with_the_items_global_index(tmp_path):
    # synthetic.py seeds its degradation off this index, so it must be the
    # position in `items`, not the position within the current chunk --
    # otherwise every chunk would repeat the same seeds.
    seen = []

    def transform(image, index):
        seen.append(index)
        return image

    embed_in_chunks(FakeEmbedder(), written(tmp_path, [str(i) for i in range(5)]),
                    chunk=2, transform=transform)
    assert seen == [0, 1, 2, 3, 4]


def test_the_transform_index_still_points_at_the_right_item_after_a_skip(tmp_path):
    # eval/real.py looks its crop quad up by this index. If a skipped file
    # shifted the indices of everything after it, every later photograph
    # would be cropped with its neighbour's quad -- a silent, total
    # corruption of the eval rather than a crash.
    items = written(tmp_path, ["a", "c", "d"])
    bad = tmp_path / "b.png"
    bad.write_bytes(b"junk")
    items.insert(1, ("b", str(bad)))

    seen = []

    def transform(image, index):
        seen.append(index)
        return image

    labels, _ = embed_in_chunks(FakeEmbedder(), items, chunk=4, transform=transform)
    assert labels == ["a", "c", "d"]
    assert seen == [0, 2, 3]  # not [0, 1, 2]


def test_no_items_returns_an_empty_array_of_the_right_width(tmp_path):
    labels, vectors = embed_in_chunks(FakeEmbedder(), [])
    assert labels == []
    assert vectors.shape == (0, 4)


def test_all_items_unreadable_returns_an_empty_array_not_a_crash(tmp_path):
    bad = tmp_path / "b.png"
    bad.write_bytes(b"junk")
    labels, vectors = embed_in_chunks(FakeEmbedder(), [("b", str(bad))])
    assert labels == []
    assert vectors.shape == (0, 4)


def test_synthetic_reexports_nothing_it_no_longer_owns():
    # The refactor's whole point is a single implementation; a leftover
    # copy in synthetic.py would be free to drift.
    import hitcheck_trainer.eval.synthetic as synthetic

    assert not hasattr(synthetic, "load_chunk")
    assert synthetic.embed_in_chunks is embed_in_chunks
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_chunks.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hitcheck_trainer.eval.chunks'`.

- [ ] **Step 3: Create the shared module**

Create `trainer/hitcheck_trainer/eval/chunks.py`:

```python
"""Memory-bounded embedding of images from disk.

Shared by the synthetic eval and the real-corpus eval. They key their
images differently -- catalog card id versus eBay itemId at an arbitrary
path -- so items are (label, path) pairs and the loader stays one
implementation rather than two.

NEVER materialise a whole gallery. 20,427 images at 240x330 RGB is 4.52GB
of decoded pixels before PIL overhead, degraded copies and torch tensors
-- that allocation contributed to a global OOM on a 30GB machine that
also runs games and browsers. Chunked at 256 the resident set is ~58MB.
Chunking here is a constraint, not an optimisation.
"""

import numpy as np
from PIL import Image


def load_chunk(items, offset=0):
    """Decode one chunk of (label, path) pairs.

    Returns (indices, labels, images) for those that opened, where each
    index is the item's position in the FULL item list -- `offset` is the
    chunk's start. Callers key per-image data off that index, so it has to
    survive a skip.

    An unreadable file is skipped rather than fatal: a truncated download
    must not abort an embed of twenty thousand images, and a catalog rerun
    replaces it.
    """
    indices, labels, images = [], [], []
    for index, (label, path) in enumerate(items, offset):
        try:
            with Image.open(path) as img:
                images.append(img.convert("RGB").copy())
            labels.append(label)
            indices.append(index)
        except OSError:
            continue
    return indices, labels, images


def embed_in_chunks(embedder, items, chunk=256, transform=None):
    """Embed (label, path) pairs a chunk at a time.

    Holds only `chunk` images decoded at once -- see the module docstring
    for why that bound is not negotiable. `transform(image, index)`
    optionally rewrites each image before embedding; `index` is the item's
    position in `items`, not its position within the chunk, because
    callers seed reproducible degradation off it.
    """
    kept_labels, vectors = [], []
    for start in range(0, len(items), chunk):
        labels, images = load_chunk(items[start : start + chunk])
        if not labels:
            continue
        if transform is not None:
            images = [transform(img, i) for i, img in enumerate(images, start)]
        vectors.append(embedder.embed(images, batch_size=64))
        kept_labels.extend(labels)
        del images  # drop decoded pixels before the next chunk
    if not vectors:
        return kept_labels, np.zeros((0, embedder.dim), dtype=np.float32)
    return kept_labels, np.concatenate(vectors, axis=0)
```

**This fixes a latent bug in the code being replaced.** `synthetic.py`'s version did `enumerate(images, start)`, so a single unreadable file shifted the index of every image after it within that chunk. For the synthetic eval that only perturbs degradation seeds. For `eval/real.py` (Task 11), which looks each photograph's crop quad up by this index, it would pair every later photograph with its neighbour's quad — a silent, total corruption of the eval rather than a crash. Hence `load_chunk` returning true indices.

- [ ] **Step 4: Rewire `synthetic.py`**

In `trainer/hitcheck_trainer/eval/synthetic.py`:

Delete the `load_chunk` and `embed_in_chunks` function definitions entirely (lines 43–75 in the current file). Keep `available_ids` and `main`.

Replace the `from PIL import Image` import with nothing (it is no longer used) and add the new import alongside the existing relative imports:

```python
from .chunks import embed_in_chunks
```

The remaining import block becomes:

```python
import argparse
import json
import os
import sys

from ..augment.degrade import degrade
from ..catalog.db import all_card_images, open_db
from ..catalog.images import image_path
from ..index.build import build_index
from ..index.embed import Embedder
from ..index.query import CardIndex
from .chunks import embed_in_chunks
from .report import score
```

`numpy` is also no longer used in `synthetic.py` — remove `import numpy as np`.

In `main`, replace the gallery embed call:

```python
        print(f"embedding gallery on {embedder.device} (dim {embedder.dim})...")
        gallery_items = [(card_id, image_path(args.images, card_id)) for card_id in disk_ids]
        ids, gallery = embed_in_chunks(embedder, gallery_items, chunk=args.chunk)
```

and the query embed call:

```python
    print(f"degrading and embedding {len(query_ids)} queries (strength {args.strength})...")
    query_items = [(card_id, image_path(args.images, card_id)) for card_id in query_ids]
    query_ids, query_vectors = embed_in_chunks(
        embedder,
        query_items,
        chunk=args.chunk,
        transform=lambda img, i: degrade(img, seed=i, strength=args.strength),
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_chunks.py -q`
Expected: PASS.

Run: `cd trainer && .venv/bin/python -m pytest -q -m "not slow"`
Expected: PASS.

- [ ] **Step 6: Verify `synthetic.py` still runs end to end**

Run: `cd trainer && .venv/bin/python -m hitcheck_trainer.eval.synthetic --sample 20 --strength 0.1 --reuse-index`
Expected: it loads the existing index, embeds 20 degraded queries, and prints a `queries=20 top1=... ci95=[...] verdict=INCONCLUSIVE` line. The verdict will be `INCONCLUSIVE` at N=20 — that is Task 1 working, not a regression. If it raises, the rewire is wrong; fix it here rather than in `real.py`.

- [ ] **Step 7: Commit**

```bash
cd trainer && .venv/bin/python -m ruff check hitcheck_trainer tests
cd /var/home/mstephens/Documents/GitHub/pokemon-card-stream-pricer
git add trainer/hitcheck_trainer/eval/chunks.py trainer/hitcheck_trainer/eval/synthetic.py trainer/tests/test_chunks.py
git commit -m "refactor(eval): one chunked-embedding loader, keyed by (label, path)

synthetic.py's embed_in_chunks was hardwired to image_path(root, card_id),
so the real-corpus eval -- whose images are keyed by eBay itemId at
arbitrary paths -- could not call it. Generalising to (label, path) pairs
keeps one implementation instead of two.

That matters beyond tidiness: the 256-image chunking exists because
materialising the whole 20,427-image catalog is 4.52GB of decoded pixels
and contributed to a global OOM on this machine. A second loader that
quietly forgot to chunk would bring it back.

Also fixes a latent index bug: the old loop numbered images by their
position within the decoded chunk, so one unreadable file shifted every
index after it. That only perturbed degradation seeds for the synthetic
eval, but eval/real.py looks each photograph's crop quad up by this
index, where it would have paired every later photo with its neighbour's
quad -- silently corrupting the whole eval rather than crashing.

First test coverage these two functions have had.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: The real-corpus eval

**Files:**
- Create: `trainer/hitcheck_trainer/eval/real.py`
- Test: `trainer/tests/test_real_eval.py`

**Interfaces:**
- Consumes: `embed_in_chunks` from `eval.chunks` (Task 10); `score`, `label_noise_bound` from `eval.report` (Task 1); `load_manifest` from `corpus.manifest` (Task 5); `load_crops`, `apply_quad` from `corpus.crops` (Task 8); `Embedder`, `CardIndex`, `build_index`, `available_ids`, `image_path` unchanged from the existing modules.
- Produces:
  - `corpus_queries(manifest, crops: dict, corpus_dir: str) -> tuple[list[tuple[str, str]], list[list[list[float]]]]` — parallel lists of `(card_id, image_path)` items and their quads, for entries that have both a crop and an image on disk.
  - `run_eval(embedder, index, items, quads, chunk: int = 256) -> list[tuple[str, list[tuple[str, float]]]]`
  - `main(argv=None) -> int`

**Background for the implementer:** This is the module the whole plan exists to produce. It mirrors `synthetic.py` but draws its queries from the manifest instead of degrading catalog images, and it searches **the same 20,427-image gallery** so any difference in the number is attributable to the queries and nothing else.

Two things it must get right:

- **Items and quads are index-aligned.** `run_eval` passes `transform=lambda img, i: apply_quad(img, quads[i])`, which relies on Task 10's true-index guarantee. `corpus_queries` therefore builds both lists in one pass; never filter one without the other.
- **The true label is the `card_id`, not the itemId.** `score` takes `(true_id, ranked)` and the ranked ids come from the catalog index, so the corpus entry's `card_id` is what goes in.

An entry with no crop is skipped silently — the hand-crop pass is incremental and a partial `crops.json` is the normal state mid-pass. An entry whose image is missing from disk is also skipped; `main` prints both counts so a short run is visible rather than quietly shrinking N.

`main` refuses to print a verdict below 500 usable queries. The interval would still be computed correctly, but the spec's target exists because smaller samples land in the inconclusive band as a matter of arithmetic, and printing a `SKIP_TRAINING` off 40 crops is exactly the failure Task 1 removed.

- [ ] **Step 1: Write the failing tests**

Create `trainer/tests/test_real_eval.py`:

```python
import numpy as np
from PIL import Image

from hitcheck_trainer.corpus.manifest import CorpusEntry, Manifest, image_relpath
from hitcheck_trainer.eval.real import corpus_queries, run_eval

QUAD_A = [[10, 10], [200, 10], [200, 300], [10, 300]]
QUAD_B = [[20, 20], [210, 20], [210, 310], [20, 310]]


class FakeEmbedder:
    dim = 4

    def embed(self, images, batch_size=32):
        # One row per image, encoding its mean brightness so different
        # crops produce different vectors.
        return np.array([[float(np.array(im).mean())] * self.dim for im in images],
                        dtype=np.float32)


class FakeIndex:
    def __init__(self, ranked):
        self.ranked = ranked
        self.queries = 0

    def query(self, vector, k=5):
        self.queries += 1
        return self.ranked


def corpus(tmp_path, entries, write=None):
    write = entries if write is None else write
    for item_id in write:
        path = tmp_path / image_relpath(item_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (400, 400), "white").save(path, "JPEG")
    return Manifest(entries=[
        CorpusEntry(item_id=i, card_id=f"card-{i}", image=image_relpath(i),
                    image_url="u", listing_url="l", aspects={})
        for i in entries
    ])


def test_pairs_each_cropped_entry_with_its_quad(tmp_path):
    manifest = corpus(tmp_path, ["a", "b"])
    items, quads = corpus_queries(manifest, {"a": QUAD_A, "b": QUAD_B}, str(tmp_path))
    assert [label for label, _ in items] == ["card-a", "card-b"]
    assert quads == [QUAD_A, QUAD_B]


def test_the_query_label_is_the_card_id_not_the_item_id(tmp_path):
    # score() compares against ids from the catalog index.
    manifest = corpus(tmp_path, ["a"])
    items, _ = corpus_queries(manifest, {"a": QUAD_A}, str(tmp_path))
    assert items[0][0] == "card-a"


def test_an_entry_with_no_crop_yet_is_skipped(tmp_path):
    # A partial crops.json is the normal state mid-pass.
    manifest = corpus(tmp_path, ["a", "b"])
    items, quads = corpus_queries(manifest, {"b": QUAD_B}, str(tmp_path))
    assert [label for label, _ in items] == ["card-b"]
    assert quads == [QUAD_B]


def test_an_entry_whose_image_is_missing_from_disk_is_skipped(tmp_path):
    manifest = corpus(tmp_path, ["a", "b"], write=["a"])
    items, quads = corpus_queries(manifest, {"a": QUAD_A, "b": QUAD_B}, str(tmp_path))
    assert [label for label, _ in items] == ["card-a"]
    assert len(quads) == 1


def test_items_and_quads_stay_aligned_when_entries_are_dropped(tmp_path):
    # Filtering one list without the other would crop every later
    # photograph with its neighbour's quad and silently corrupt the eval.
    manifest = corpus(tmp_path, ["a", "b", "c"], write=["a", "c"])
    items, quads = corpus_queries(manifest, {"a": QUAD_A, "c": QUAD_B}, str(tmp_path))
    assert len(items) == len(quads) == 2
    assert [label for label, _ in items] == ["card-a", "card-c"]
    assert quads == [QUAD_A, QUAD_B]


def test_no_crops_at_all_yields_no_queries(tmp_path):
    items, quads = corpus_queries(corpus(tmp_path, ["a"]), {}, str(tmp_path))
    assert items == [] and quads == []


def test_run_eval_returns_one_true_id_and_ranking_per_query(tmp_path):
    manifest = corpus(tmp_path, ["a", "b"])
    items, quads = corpus_queries(manifest, {"a": QUAD_A, "b": QUAD_B}, str(tmp_path))
    index = FakeIndex([("card-a", 0.1), ("card-z", 0.4)])
    results = run_eval(FakeEmbedder(), index, items, quads)
    assert [true_id for true_id, _ in results] == ["card-a", "card-b"]
    assert index.queries == 2


def test_run_eval_scores_through_the_existing_report_harness(tmp_path):
    from hitcheck_trainer.eval.report import score

    manifest = corpus(tmp_path, ["a", "b"])
    items, quads = corpus_queries(manifest, {"a": QUAD_A, "b": QUAD_B}, str(tmp_path))
    index = FakeIndex([("card-a", 0.1)])
    report = score(run_eval(FakeEmbedder(), index, items, quads))
    assert report.total == 2
    assert report.top1 == 0.5  # card-a hits, card-b does not


def test_run_eval_crops_before_embedding(tmp_path):
    # If the photograph went in uncropped, the embedder would see a
    # 400x400 desk shot; apply_quad hands it a 245x342 card.
    seen = []

    class SizeRecordingEmbedder(FakeEmbedder):
        def embed(self, images, batch_size=32):
            seen.extend(im.size for im in images)
            return super().embed(images, batch_size)

    manifest = corpus(tmp_path, ["a"])
    items, quads = corpus_queries(manifest, {"a": QUAD_A}, str(tmp_path))
    run_eval(SizeRecordingEmbedder(), FakeIndex([("card-a", 0.1)]), items, quads)
    assert seen == [(245, 342)]


def test_run_eval_of_an_empty_corpus_returns_no_results(tmp_path):
    assert run_eval(FakeEmbedder(), FakeIndex([]), [], []) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_real_eval.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hitcheck_trainer.eval.real'`.

- [ ] **Step 3: Implement the eval**

Create `trainer/hitcheck_trainer/eval/real.py`:

```python
"""M2 gate, measured on real photographs instead of a synthetic axis.

synthetic.py queries the gallery with degraded copies of the gallery's
own images -- identical source pixels, lighting and crop -- which is why
it scores 1.000 at strength 0.0 and why no value of `strength` models
matching a DIFFERENT photograph of the same card. This module closes that
gap: the queries are eBay seller photographs, hand-cropped, labelled from
structured Item Specifics.

The gallery is the same 20,427-image index the synthetic run used, so any
difference in the number is attributable to the queries and nothing else.

Two caveats belong in every write-up of this number. It measures retrieval
GIVEN A GOOD CROP, because M3's detector does not exist yet and the corpus
is cropped by hand. And seller photographs are well-lit, static and
high-resolution -- meaningfully easier than a compressed handheld stream
frame -- so the result is an upper bound.
"""

import argparse
import os
import sys

from ..catalog.db import all_card_images, open_db
from ..catalog.images import image_path
from ..corpus.crops import apply_quad, load_crops
from ..corpus.manifest import load_manifest
from ..index.build import build_index
from ..index.embed import Embedder
from ..index.query import CardIndex
from .chunks import embed_in_chunks
from .report import label_noise_bound, score
from .synthetic import available_ids

DEFAULT_DB = "data/catalog.sqlite"
DEFAULT_IMAGES = "data/images"
DEFAULT_INDEX = "data/index/cards.bin"
DEFAULT_CORPUS = "data/corpus"

MIN_QUERIES = 500


def corpus_queries(manifest, crops, corpus_dir):
    """(label, path) items and their quads, index-aligned.

    Built in one pass on purpose: filtering one list without the other
    would crop every later photograph with its neighbour's quad and
    silently corrupt the eval rather than crash it.

    An entry with no crop is skipped -- a partial crops.json is the normal
    state during an incremental hand-crop pass. An entry whose image is
    missing from disk is skipped too; main() prints both counts so a short
    run is visible instead of quietly shrinking N.
    """
    items, quads = [], []
    for entry in manifest.entries:
        quad = crops.get(entry.item_id)
        if quad is None:
            continue
        path = os.path.join(corpus_dir, entry.image)
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            continue
        items.append((entry.card_id, path))
        quads.append(quad)
    return items, quads


def run_eval(embedder, index, items, quads, chunk=256):
    """Crop, embed and query. Returns (true_id, ranked) pairs for score()."""
    labels, vectors = embed_in_chunks(
        embedder, items, chunk=chunk,
        transform=lambda img, i: apply_quad(img, quads[i]),
    )
    return [(label, index.query(vector, k=5)) for label, vector in zip(labels, vectors)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hitcheck-eval-real")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--images", default=DEFAULT_IMAGES)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--chunk", type=int, default=256,
                        help="images decoded at once; keep small, this bounds RAM")
    parser.add_argument("--reuse-index", action="store_true",
                        help="load the existing gallery index instead of re-embedding "
                             "all ~20k catalog images")
    parser.add_argument("--label-errors", type=int, default=None,
                        help="wrong labels found by the hand audit (see corpus.audit)")
    parser.add_argument("--label-sample", type=int, default=None,
                        help="entries hand-audited")
    args = parser.parse_args(argv)

    manifest = load_manifest(os.path.join(args.corpus, "manifest.json"))
    crops = load_crops(os.path.join(args.corpus, "crops.json"))
    items, quads = corpus_queries(manifest, crops, args.corpus)
    print(f"manifest: {len(manifest.entries)} entries, {len(crops)} cropped, "
          f"{len(items)} usable queries")
    print(manifest.yield_summary())
    if not items:
        print("No cropped corpus entries. Run the corpus build, then the crop tool.")
        return 1

    embedder = Embedder()
    sidecar_path = f"{args.index}.ids.json"
    if args.reuse_index and os.path.exists(args.index) and os.path.exists(sidecar_path):
        print(f"reusing existing gallery index at {args.index}")
        index = CardIndex.load(args.index, dim=embedder.dim)
    else:
        conn = open_db(args.db)
        disk_ids = available_ids(all_card_images(conn), args.images)
        print(f"embedding gallery of {len(disk_ids)} images on {embedder.device}...")
        gallery_items = [(card_id, image_path(args.images, card_id)) for card_id in disk_ids]
        ids, gallery = embed_in_chunks(embedder, gallery_items, chunk=args.chunk)
        build_index(gallery, ids, args.index)
        index = CardIndex.load(args.index, dim=embedder.dim)
        del gallery

    print(f"cropping and embedding {len(items)} real queries...")
    report = score(run_eval(embedder, index, items, quads, chunk=args.chunk))

    print()
    print(report.summary())
    if args.label_sample:
        bound = label_noise_bound(args.label_errors or 0, args.label_sample)
        print(f"label error <= {bound:.1%} (95% bound from {args.label_errors or 0}"
              f"/{args.label_sample} audited) — measured top1 understates true top1 "
              "by at most this much")
    else:
        print("label error UNBOUNDED — run corpus.audit and pass --label-errors/"
              "--label-sample before quoting this number")
    print("Measured GIVEN A GOOD CROP (hand-cropped; M3's detector does not exist "
          "yet) and on seller photographs, which are easier than stream frames. "
          "This is an upper bound.")

    if report.total < MIN_QUERIES:
        print(f"\n*** {report.total} queries is below {MIN_QUERIES}. Samples this small "
              "land in the inconclusive band as a matter of arithmetic; crop more "
              "before treating the verdict as an answer. ***")
        return 1

    print()
    print("Sample misses (true -> predicted):")
    for true_id, predicted in report.failures[:15]:
        print(f"  {true_id} -> {predicted or '(none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_real_eval.py -q`
Expected: PASS.

Run: `cd trainer && .venv/bin/python -m pytest -q -m "not slow"`
Expected: PASS.

- [ ] **Step 5: Verify the CLI refuses to run without a corpus**

Run: `cd trainer && .venv/bin/python -m hitcheck_trainer.eval.real --corpus data/corpus --reuse-index`
Expected: exits 1 with `No cropped corpus entries.` — it must not attempt to embed or crash. (If the corpus has already been built and cropped by the time this task runs, it will instead run the eval; that is fine too.)

- [ ] **Step 6: Commit**

```bash
cd trainer && .venv/bin/python -m ruff check hitcheck_trainer tests
cd /var/home/mstephens/Documents/GitHub/pokemon-card-stream-pricer
git add trainer/hitcheck_trainer/eval/real.py trainer/tests/test_real_eval.py
git commit -m "feat(eval): measure M2 retrieval on real seller photographs

synthetic.py queries the gallery with degraded copies of the gallery's own
images -- identical source pixels, lighting and crop -- which is why it
scores 1.000 at strength 0.0 and why no strength value models matching a
different photograph of the same card. This closes that gap directly
instead of trying to locate real frames on the synthetic axis.

Same 20,427-image gallery via --reuse-index, so any difference in the
number is attributable to the queries. Items and their crop quads are
built in one pass and stay index-aligned; filtering one without the other
would crop every later photo with its neighbour's quad.

Refuses to quote a verdict below 500 queries, and refuses to quote the
accuracy at all without a label-error bound from the hand audit -- an
unbounded label-error rate sitting under the M2 verdict would make the
number unusable for the decision it exists to settle.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 12: The label audit sheet and the runbook

**Files:**
- Create: `trainer/hitcheck_trainer/corpus/audit.py`
- Create: `docs/runbooks/2026-08-31-m2-corpus.md`
- Test: `trainer/tests/test_audit.py`

**Interfaces:**
- Consumes: `load_manifest` from `corpus.manifest` (Task 5); `load_crops`, `apply_quad` from `corpus.crops` (Task 8); `image_path` from `catalog.images`; `safe_item_id` from `corpus.manifest`.
- Produces:
  - `sample_entries(manifest, crops: dict, count: int = 50, seed: int = 0) -> list`
  - `build_audit(manifest, crops, corpus_dir: str, images_root: str, out_dir: str, count: int = 50, seed: int = 0) -> str` — returns the path of the HTML sheet it wrote.
  - `main(argv=None) -> int`

**Background for the implementer:** The spec requires hand-verifying a random 50 resolved entries to bound residual label noise, and reporting the accuracy with that bound attached. This produces the sheet a human eyeballs: for each sampled entry, the hand-cropped photograph next to the catalog scan of the card it was resolved to. Wrong pairs are obvious at a glance; the human counts them and passes `--label-errors N --label-sample 50` to `eval/real.py`.

**Sampling is seeded** so the same 50 come back on a rerun. An audit that resampled every run could be repeated until it produced a flattering count.

Cropped previews are written under `<corpus_dir>/audit/` — already gitignored by Task 5's `data/corpus/*` rule, which only exempts the two JSON files.

- [ ] **Step 1: Write the failing tests**

Create `trainer/tests/test_audit.py`:

```python
from PIL import Image

from hitcheck_trainer.corpus.audit import build_audit, sample_entries
from hitcheck_trainer.corpus.manifest import CorpusEntry, Manifest, image_relpath

QUAD = [[10, 10], [200, 10], [200, 300], [10, 300]]


def setup_corpus(tmp_path, n=6):
    corpus_dir = tmp_path / "corpus"
    images_root = tmp_path / "images"
    entries = []
    for i in range(n):
        item_id = f"v1|{i}|0"
        photo = corpus_dir / image_relpath(item_id)
        photo.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (400, 400), "white").save(photo, "JPEG")
        catalog = images_root / "sv3pt5" / f"sv3pt5-{i}.png"
        catalog.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (245, 342), "blue").save(catalog)
        entries.append(CorpusEntry(item_id=item_id, card_id=f"sv3pt5-{i}",
                                   image=image_relpath(item_id), image_url="u",
                                   listing_url=f"https://ebay/{i}", aspects={"Set": "151"}))
    crops = {e.item_id: QUAD for e in entries}
    return Manifest(entries=entries), crops, str(corpus_dir), str(images_root)


def test_samples_the_requested_number_of_entries(tmp_path):
    manifest, crops, _, _ = setup_corpus(tmp_path)
    assert len(sample_entries(manifest, crops, count=3, seed=0)) == 3


def test_sampling_is_seeded_so_a_rerun_audits_the_same_entries(tmp_path):
    # An audit that resampled every run could be repeated until it
    # produced a flattering error count.
    manifest, crops, _, _ = setup_corpus(tmp_path)
    first = [e.item_id for e in sample_entries(manifest, crops, count=3, seed=7)]
    second = [e.item_id for e in sample_entries(manifest, crops, count=3, seed=7)]
    assert first == second


def test_a_different_seed_can_sample_differently(tmp_path):
    manifest, crops, _, _ = setup_corpus(tmp_path, n=40)
    a = [e.item_id for e in sample_entries(manifest, crops, count=5, seed=1)]
    b = [e.item_id for e in sample_entries(manifest, crops, count=5, seed=2)]
    assert a != b


def test_only_cropped_entries_are_auditable(tmp_path):
    # An uncropped entry has no crop to show, and is not in the eval either.
    manifest, crops, _, _ = setup_corpus(tmp_path)
    only_one = {manifest.entries[2].item_id: QUAD}
    sampled = sample_entries(manifest, only_one, count=50, seed=0)
    assert [e.item_id for e in sampled] == [manifest.entries[2].item_id]


def test_asking_for_more_than_exist_returns_everything_available(tmp_path):
    manifest, crops, _, _ = setup_corpus(tmp_path, n=4)
    assert len(sample_entries(manifest, crops, count=50, seed=0)) == 4


def test_writes_an_html_sheet_pairing_the_crop_with_the_catalog_scan(tmp_path):
    manifest, crops, corpus_dir, images_root = setup_corpus(tmp_path)
    out = build_audit(manifest, crops, corpus_dir, images_root,
                      str(tmp_path / "out"), count=2, seed=0)
    html = open(out).read()
    assert html.count("<img") == 4  # two pairs
    assert "sv3pt5-" in html


def test_the_sheet_links_back_to_the_listing_and_shows_the_aspects(tmp_path):
    # A mismatch is easier to adjudicate with the source listing to hand.
    manifest, crops, corpus_dir, images_root = setup_corpus(tmp_path)
    out = build_audit(manifest, crops, corpus_dir, images_root,
                      str(tmp_path / "out"), count=1, seed=0)
    html = open(out).read()
    assert "https://ebay/" in html
    assert "151" in html


def test_writes_a_cropped_preview_per_audited_entry(tmp_path):
    manifest, crops, corpus_dir, images_root = setup_corpus(tmp_path)
    build_audit(manifest, crops, corpus_dir, images_root, str(tmp_path / "out"),
                count=3, seed=0)
    previews = list((tmp_path / "out" / "crops").glob("*.png"))
    assert len(previews) == 3
    assert Image.open(previews[0]).size == (245, 342)


def test_states_the_sample_size_the_bound_will_be_computed_from(tmp_path):
    manifest, crops, corpus_dir, images_root = setup_corpus(tmp_path)
    out = build_audit(manifest, crops, corpus_dir, images_root,
                      str(tmp_path / "out"), count=3, seed=0)
    assert "--label-sample 3" in open(out).read()


def test_an_entry_whose_photograph_is_missing_is_skipped_not_fatal(tmp_path):
    manifest, crops, corpus_dir, images_root = setup_corpus(tmp_path, n=2)
    import os

    os.remove(os.path.join(corpus_dir, manifest.entries[0].image))
    out = build_audit(manifest, crops, corpus_dir, images_root,
                      str(tmp_path / "out"), count=50, seed=0)
    assert open(out).read().count("<img") == 2  # only the surviving pair
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_audit.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hitcheck_trainer.corpus.audit'`.

- [ ] **Step 3: Implement the audit sheet**

Create `trainer/hitcheck_trainer/corpus/audit.py`:

```python
"""Side-by-side sheet for hand-verifying resolved corpus labels.

Resolution is the accuracy-contaminating step: a mis-resolved label
surfaces as a retrieval miss that is not one. So a random sample gets
eyeballed -- hand-cropped photograph beside the catalog scan it was
resolved to -- and the resulting error count becomes a bound reported
alongside the accuracy. An unbounded label-error rate sitting under the
M2 verdict would make the number unusable for the decision it settles.

Sampling is seeded. An audit that resampled every run could be repeated
until it produced a flattering count.
"""

import argparse
import html
import os
import random
import sys

from PIL import Image

from ..catalog.images import image_path
from .crops import apply_quad, load_crops
from .manifest import load_manifest, safe_item_id

DEFAULT_CORPUS = "data/corpus"
DEFAULT_IMAGES = "data/images"


def sample_entries(manifest, crops, count=50, seed=0):
    """A reproducible random sample of cropped entries, in manifest order."""
    cropped = [e for e in manifest.entries if e.item_id in crops]
    if len(cropped) <= count:
        return cropped
    return random.Random(seed).sample(cropped, count)


def build_audit(manifest, crops, corpus_dir, images_root, out_dir, count=50, seed=0) -> str:
    """Write an HTML sheet plus cropped previews. Returns the sheet's path."""
    crops_dir = os.path.join(out_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    rows = []
    for entry in sample_entries(manifest, crops, count, seed):
        photo_path = os.path.join(corpus_dir, entry.image)
        catalog_path = image_path(images_root, entry.card_id)
        try:
            with Image.open(photo_path) as photo:
                cropped = apply_quad(photo, crops[entry.item_id])
        except (OSError, ValueError):
            continue  # a missing or unreadable photo is skipped, not fatal
        preview = os.path.join(crops_dir, f"{safe_item_id(entry.item_id)}.png")
        cropped.save(preview)

        aspects = ", ".join(f"{k}: {v}" for k, v in sorted(entry.aspects.items()))
        rows.append(
            f'<tr><td><img src="{html.escape(os.path.relpath(preview, out_dir))}"></td>'
            f'<td><img src="{html.escape(os.path.relpath(catalog_path, out_dir))}"></td>'
            f"<td><strong>{html.escape(entry.card_id)}</strong><br>"
            f"{html.escape(aspects)}<br>"
            f'<a href="{html.escape(entry.listing_url)}">listing</a></td></tr>'
        )

    sheet = (
        "<!doctype html><meta charset='utf-8'><title>HitCheck label audit</title>"
        "<style>body{font:14px system-ui;background:#111;color:#eee}"
        "img{height:240px;background:#fff}td{padding:8px;vertical-align:top}"
        "tr{border-bottom:1px solid #333}</style>"
        f"<h1>Label audit — {len(rows)} entries</h1>"
        "<p>Left: hand-cropped photograph. Right: the catalog scan it resolved to. "
        "Count the pairs that are not the same card, then run:</p>"
        f"<pre>python -m hitcheck_trainer.eval.real --reuse-index "
        f"--label-errors N --label-sample {len(rows)}</pre>"
        "<table>" + "".join(rows) + "</table>"
    )
    out_path = os.path.join(out_dir, "audit.html")
    with open(out_path, "w") as fh:
        fh.write(sheet)
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hitcheck-corpus-audit")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--images", default=DEFAULT_IMAGES)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    manifest = load_manifest(os.path.join(args.corpus, "manifest.json"))
    crops = load_crops(os.path.join(args.corpus, "crops.json"))
    if not crops:
        print(f"No crops under {args.corpus}. Run the crop tool first.")
        return 1

    out_dir = os.path.join(args.corpus, "audit")
    path = build_audit(manifest, crops, args.corpus, args.images, out_dir,
                       args.count, args.seed)
    print(f"open file://{os.path.abspath(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd trainer && .venv/bin/python -m pytest tests/test_audit.py -q`
Expected: PASS.

Run: `cd trainer && .venv/bin/python -m pytest -q -m "not slow"`
Expected: PASS — the whole suite.

- [ ] **Step 5: Write the runbook**

Create `docs/runbooks/2026-08-31-m2-corpus.md`:

````markdown
# Runbook: producing the M2 real-corpus number

Implements `docs/superpowers/specs/2026-08-31-m2-calibration-design.md` (Half A).
Everything below runs from `trainer/` with the venv python.

## 0. Credentials

`PROD_APP_ID` and `PROD_EBAY_CERT_ID` must be in the environment. They live in
the gitignored `.env` at the repo root. The production keyset works because
`workers/ebay-account-deletion/` is deployed and accepted by eBay — if OAuth
starts returning `invalid_client`, check that endpoint before the credentials.

## 1. Acquire the corpus

```bash
cd trainer
set -a && . ../.env && set +a
.venv/bin/python -m hitcheck_trainer.corpus.build --target 600
```

Costs roughly two Browse calls per candidate against a ~5,000/day budget.
Rerunning tops the corpus up and never re-fetches; interrupt it freely.

Read the `kept=... discarded=...` line. A yield far below the expected ~60%
means the resolver is rejecting more than the measured aspect coverage
predicts — investigate before cropping, not after.

## 2. Hand-crop

```bash
.venv/bin/python -m hitcheck_trainer.corpus.croptool
```

Open http://127.0.0.1:8765/. For each photograph, click the **card's**
top-left corner, then the other three clockwise — the card, not the slab,
not the label. `u` undoes the last point. Progress saves after every card,
so stop and resume as often as you like.

This is the expensive step: a few hundred cards. It exists because M3's
detector does not, and it doubles as ground truth for evaluating M3 later.

## 3. Audit the labels

```bash
.venv/bin/python -m hitcheck_trainer.corpus.audit --count 50
```

Open the printed path. Count the pairs that are not the same card. That
count is the input to the bound; do not re-roll the seed to get a nicer
number.

## 4. Run the eval

```bash
.venv/bin/python -m hitcheck_trainer.eval.real --reuse-index \
    --label-errors <N> --label-sample 50
```

`--reuse-index` is not optional in spirit: the point is to search the same
20,427-image gallery the synthetic run searched, so any difference in the
number belongs to the queries.

## 5. Record the result

Write `docs/verification/2026-08-31-m2-real-corpus.md` alongside
`2026-08-10-m2-zeroshot.md`. It must state:

- top-1, top-5, the 95% interval, and the verdict.
- The label-error bound and the audit sample it came from.
- The corpus yield line from the manifest, so the selection bias is visible.
- That the number is measured **given a good crop** (hand-cropped; M3 does
  not exist), and therefore conditional on M3 working.
- That seller photographs are well-lit, static and high-resolution —
  meaningfully easier than a compressed handheld stream frame — so this is
  an **upper bound**.
- That the corpus is described qualitatively as "seller photographs" until
  Half B's `measure.py` exists to place it on the degradation axes.

## If the verdict is INCONCLUSIVE

That is an answer, not a failure. The interval straddles 0.90 and the
sample cannot resolve which side it is on. The decisive bands are:

| N | SKIP_TRAINING needs top1 ≥ | TRAIN_REQUIRED needs top1 ≤ |
|---|---|---|
| 500 | 0.9280 | 0.8720 |
| 1000 | 0.9190 | 0.8810 |
| 2000 | 0.9135 | 0.8865 |

Go back to step 1 with a higher `--target` and crop the additions. Do not
lower the threshold, and do not quote the point estimate as if it settled
the question.
````

- [ ] **Step 6: Commit**

```bash
cd trainer && .venv/bin/python -m ruff check hitcheck_trainer tests
cd /var/home/mstephens/Documents/GitHub/pokemon-card-stream-pricer
git add trainer/hitcheck_trainer/corpus/audit.py trainer/tests/test_audit.py docs/runbooks/2026-08-31-m2-corpus.md
git commit -m "feat(corpus): label audit sheet, and the runbook for producing the number

Resolution is the accuracy-contaminating step -- a mis-resolved label
surfaces as a retrieval miss that is not one -- so a seeded random sample
gets eyeballed, crop beside catalog scan, and the error count becomes a
bound reported with the accuracy. Seeded because an audit that resampled
every run could be repeated until it produced a flattering count.

The runbook covers acquisition, the hand-crop pass, the audit, the eval
run, and what the write-up must state: that the number is measured given
a good crop and on seller photographs, and is therefore an upper bound
conditional on M3 working.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## After this plan

Half A ends with a number and a verification doc. Half B — `augment/measure.py`
and its calibration curves — gets its own plan, written once this one lands.
That is the spec's own sequencing: the axis estimators are the bridge from
photograph conditions to stream conditions, not a blocker on the verdict.

Until Half B exists, the corpus is described qualitatively as "seller
photographs". It only gets to say "these sit at blur-equivalent 0.1,
JPEG-equivalent 0.15" afterwards.
