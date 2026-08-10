# HitCheck — Design Spec

**Date:** 2026-08-10
**Status:** Approved

## Summary

HitCheck is a cross-platform desktop app for **buyers** watching live card-selling
streams (eBay Live, TikTok Live). It screen-captures the stream video, identifies
the Pokémon card the seller is holding up, reads the grade if it's slabbed,
looks up market value, OCRs the current bid off the stream, and shows the
spread in a local overlay — so the viewer can decide whether to bid before the
lot closes.

The overlay is **private to the user**. Nothing is composited into anyone's
stream; HitCheck is a read-only consumer of pixels already on screen.

## Context and constraints

The user is a viewer, not a seller. This drives most of the design:

- **No control over input quality.** Whatever glare, motion blur, framing and
  H.264 compression the seller's setup produces is what we get. We cannot ask
  for better lighting or a steadier hand.
- **Latency is a hard requirement, not a preference.** Live lots close in
  seconds. An identification that arrives after the hammer is worthless.
- **Screen capture is the only possible source.** No capture card, no direct
  camera access — we read the decoded stream as displayed.
- **The consumer of the answer is a person making a money decision.** A
  confidently wrong price is worse than "identifying…".

## Goals

- Identify a Pokémon card held to camera from a compressed live stream, in
  under ~1 second from the moment it is held steady.
- Detect slabs and read the grade (PSA / CGC / BGS).
- Show market value alongside the current bid, and the headroom between them.
- Work on eBay Live and TikTok Live from day one.
- Ship as Linux AppImage, Windows installer, and macOS DMG.

## Non-goals

- Placing bids, or any interaction with the marketplace. Read-only.
- Broadcasting anything to a stream.
- Sports cards or other TCGs. The catalog interface leaves room, but v1 is
  Pokémon.
- Sold-comp history of our own. The user is a buyer; there is no proprietary
  sales data to accumulate.

## Naming

Product name **HitCheck** — "hit" is collector slang for a valuable pull, and
"check" carries both *checking for a hit* and *price check*.

Repo: `hitcheck`, public.
Description: "Real-time Pokémon card and slab pricing overlay for live streams."
Topics: `pokemon-cards`, `computer-vision`, `screen-capture`, `tcgplayer`,
`card-grading`, `electron`.

"Pokémon" appears in the description and topics — ordinary nominative use — but
deliberately not in the product name. `Poké-`-prefixed names are the specific
construction Nintendo/TPCi enforce against, and a neutral name also leaves room
to point the camera at other card types later.

## Stack

**App:** Electron + TypeScript, `onnxruntime-node` for inference, `sharp` for
image ops, packaged with electron-builder (AppImage / NSIS / DMG).

**Trainer:** Python + PyTorch, exporting ONNX. Never ships to users.

### Why Electron over Tauri

Tauri's smaller bundle is real (~50MB vs ~200MB) but the deciding factor is
capture. Tauri's Linux webview is WebKitGTK, which does not provide
`getDisplayMedia`, so a Tauri build requires three native capture backends —
`ashpd` + `pipewire-rs` on Wayland, Windows Graphics Capture, and
ScreenCaptureKit — with the fiddliest of the three on the primary development
machine. Electron inherits Chromium's three capture backends behind one web API.

Sibling projects in the same workspace settle it further: `axistream`, `otto`
and `sai` are all Electron; `otto` already ships `onnxruntime-node` 1.21.0 and
`sharp` in an Electron app building all three targets. The capture layer stays
behind an interface, so a Tauri port remains a contained swap if bundle size
ever becomes a real constraint.

### Why PyTorch stays out of the app

PyTorch + CUDA cannot ship cross-platform — Macs have no NVIDIA, and a bundled
Torch+CUDA AppImage runs 2–4GB. Training runs on the developer's RTX 4070 Ti
and emits ONNX; the app runs ONNX Runtime with per-platform execution providers
(CUDA/TensorRT on NVIDIA, CoreML on Apple Silicon, DirectML on Windows AMD, CPU
fallback everywhere).

## Prior art in this workspace

Findings already established by sibling projects, reused rather than rediscovered:

- **Wayland portal approval persists.** `axistream` proved on this exact machine
  (Bazzite / KDE Kinoite / Wayland) that the xdg-desktop-portal screen-share
  approval stores a restore token and re-launches silently with no prompt.
- **`desktopCapturer.getSources()` thumbnails are blank on Wayland.** This is
  why `sai` needs `spectacle`/`grim`/`screencapture` fallbacks. That is the
  *screenshot* API; HitCheck needs continuous video and uses
  `getDisplayMedia()`, which is portal-backed and returns a live MediaStream.
  HitCheck therefore does not inherit those fallbacks.
- **The OBS sidecar approach is not needed.** `axistream` bundles OBS (537MB) to
  solve encoding and streaming. HitCheck only reads pixels.

## Architecture

One Electron app. Capture and inference in the main process (or a utility
process), overlay in a renderer.

```
getDisplayMedia ─┬─► card loop (~15fps) ─► detect ─► stability gate ─► identify ─► price
                 └─► bid loop  (~2fps)  ─► OCR bid region ──────────────────────┐
                                                                                 ▼
                                                            overlay (transparent, click-through)
```

### Modules

| Module | Responsibility | Depends on |
|---|---|---|
| `capture/` | Platform-agnostic frame source from a user-selected window/region | Electron only |
| `vision/` | detect → stability gate → identify → grade. Image in, card ID out. | ONNX models, index |
| `catalog/` | Card DB sync, local SQLite, embedding index build + query | Card data API |
| `pricing/` | `PriceSource` interface + implementations, cache-first | Network |
| `overlay/` | Render card, value, bid, spread, confidence | Pipeline state |
| `calibration/` | Per-platform screen-region config (bid rectangle) | Persisted config |
| `app.ts` | Wiring, lifecycle | All of the above |

`vision/` takes an image and returns a card identity. It knows nothing about
screens, streams, or prices, and is testable against still images alone.

### The card loop

1. **Detect** — YOLOv11n fine-tuned on four classes: `raw_card`, `psa_slab`,
   `cgc_slab`, `bgs_slab`. ~3ms/frame. Grading company falls out of detection
   for free, since slab labels are visually distinct.
2. **Stability gate** — do not identify every frame. Proceed only when a
   detection holds position across N frames *and* passes a sharpness check
   (variance-of-Laplacian). The seller waves the card around; compute is spent
   only on the moment it is held steady toward camera. Costs ~250ms of
   deliberate latency, and avoids identifying motion blur.
3. **Identify** — perspective-correct the detected quad, embed, and
   approximate-nearest-neighbor against a prebuilt index of the full card
   catalog. **Retrieval, not classification** — no 20,000-class model.
   Deliberately not OCR-first: set numbers are a handful of pixels tall through
   stream compression, and visual matching degrades gracefully where text
   recognition fails outright.
4. **Grade** — for slab classes, crop the label region and OCR it; parse
   `PSA 10`, `CGC 9.5`, `BGS 9.5`.
5. **Price** — cache-first against local SQLite; async prefetch fires the moment
   identity is known.

### The bid loop

Runs independently at ~2fps against a **user-calibrated screen rectangle**. The
user drags a box over the bid number once per platform; it persists. Not
hardcoded coordinates — eBay and TikTok will redesign, and that must not be a
code change.

### Latency budget

| Stage | Cost |
|---|---|
| Stability gate | ~250ms (deliberate) |
| Detect + embed + ANN | ~20ms |
| Grade OCR | ~50ms |
| Price (cache hit) | ~0ms |
| Price (cache miss) | 300–500ms, async |

Card held steady → number on screen in well under one second on a cache hit.

### Confidence gate

When retrieval confidence is below threshold, the overlay shows "identifying…"
rather than a card and a price. When the top candidates are close together, it
shows the top 3 rather than committing. The failure mode being designed against
is the user bidding real money on a confidently wrong match.

## Pricing

All sources sit behind a single `PriceSource` interface.

**Raw / ungraded — pokemontcg.io.** Free API key, 1,000 requests/day, and it
embeds TCGplayer prices in every card response (updated hourly) plus Cardmarket
in EUR and 1/7/30-day trend averages. This satisfies the TCGplayer requirement
without the partner-approval gauntlet. The same API also serves the card
database and images that `catalog/` needs — one dependency, two jobs.

**Graded — eBay Browse API (free tier) initially.** Returns *active* listings,
not sold ones. Asking prices skew high because unsold optimists sit at the top
of the results, so the displayed figure is the median of the interquartile range
with outliers discarded, and it is **labeled in the UI as asks, not sales**.

**PriceCharting ($50/mo) is the deferred upgrade.** It is the correct source for
graded comps. Deferring it makes the spend an evidence-based decision after a
few real streams rather than an upfront cost. Because it sits behind
`PriceSource`, adopting it is a config change.

eBay's sold-comp data (Marketplace Insights API) is gated to approved partners,
and eBay's consumer Price Guide with real graded sold history is UI-only with no
API. Neither is available to us.

Caching is aggressive: card metadata and images are pulled once and stored
locally, so the daily quota is barely touched during normal use.

## The training half

Two distinct jobs, and only one clearly needs training.

### Detection — synthetic-first

Generic detectors do not know "graded slab." Training data is generated rather
than collected: take catalog card images, apply random perspective warps,
composite onto random backgrounds and onto slab-frame templates. This yields
unlimited labeled boxes for free. Fine-tune YOLOv11n on synthetic, then correct
a few hundred real frames pulled from recorded streams. Roughly a day or two of
GPU time, not a research project.

### Identification — measure before training

Identification is retrieval. Modern self-supervised embeddings may already be
sufficient with **zero training**: DINOv2 was built for instance retrieval, and
Pokémon cards are high-texture, visually distinctive objects — close to the
ideal case.

**M2 is therefore a measurement, not an implementation.** Build the index with
off-the-shelf DINOv2 embeddings, feed it a few hundred hand-labeled real stream
frames, and measure top-1 and top-5 accuracy.

- If top-1 lands above ~90%, the identification training project is unnecessary
  and the app ships far earlier.
- If it is poor, fine-tune with metric learning (ArcFace) over the augmented
  catalog — and the labeled eval set built for the measurement is exactly what
  that training needs anyway.

Training a card identifier is the most expensive item in this project and may be
redundant. The measurement is cheap and decides it.

### The augmentation pipeline is the real asset

Catalog images are clean flatbed scans. Real input is a card under stream
lighting, at an angle, behind slab plastic, through H.264. Bridging that domain
gap determines whether any of this works, and it is required for both zero-shot
evaluation and any future fine-tuning:

- random perspective warp and rotation
- specular highlight / glare overlays (slab plastic)
- motion blur
- H.264 / JPEG compression artifacts at stream-realistic bitrates
- white balance and exposure shifts
- partial occlusion by fingers

### Grade and bid OCR — no training

Slab labels and bid overlays are high-contrast printed text in fixed layouts.
Off-the-shelf OCR plus regexes suffices.

### Trainer output contract

`trainer/` emits ONNX model files and a prebuilt embedding index. Nothing else
crosses the boundary into the app.

## Milestones

Ordered by risk retired per unit of work.

| # | Milestone | Proves |
|---|---|---|
| **M0** | Electron shell: `getDisplayMedia` capture of a chosen window + transparent click-through overlay drawing a box over it | Wayland capture *and* overlay together — highest unknown-unknown density |
| **M1** | Catalog sync: card DB + images → local SQLite + embedding index | Data foundation; also the training corpus |
| **M2** | Zero-shot retrieval eval on labeled real stream frames | The train / don't-train decision |
| **M3** | Synthetic-trained detector + stability gate, running live | The real-time loop, end to end |
| **M4** | `PriceSource` + cache-first lookup | Numbers on screen |
| **M5** | Slab classes + grade OCR | Graded cards |
| **M6** | Bid-region calibration + OCR + spread display | The actual bidding decision |
| **M7** | Fine-tune if M2 requires it; packaging for all three targets | Ship |

**M0 goes first deliberately.** Instinct says start with the CV. But
capture-plus-overlay on Wayland is where this project would die, and the
`axistream` findings say that path is *survivable*, not *free*. One evening to
prove it beats discovering it in week six.

## Testing

- `vision/` is tested against fixed still images with known answers — no screen,
  no stream, no network.
- Backend selection, stability-gate logic, blank-frame detection, bid-string
  parsing and price aggregation are pure functions with unit tests.
- Retrieval accuracy is a measured metric with a checked-in labeled eval set,
  tracked across model changes rather than asserted.
- `PriceSource` implementations are tested against recorded fixtures, not live
  APIs.
- Vitest, fork pool capped at 2 workers (workspace convention).

## Risks

| Risk | Mitigation |
|---|---|
| Overlay refused on GNOME (no true always-on-top) | Side-panel fallback window. Affects other users, not the primary machine — KWin behaves. |
| Near-identical reprints confuse retrieval | Show top-3 when confidence is close; set-number OCR as tiebreaker only when the crop is sharp enough to be worth it |
| pokemontcg.io succeeded by a commercial product | Works and is free today; `CardCatalog` interface keeps the swap cheap |
| eBay/TikTok layout drift breaks bid OCR | User-calibrated rectangles, never hardcoded constants |
| Lots close faster than the stability gate resolves | M3 logs time-to-identify so the gate is tuned with data, not guesses |
| Graded "asks" mislead the user | Labeled as asks in the UI; PriceCharting upgrade path is one config change |

## Open items

None blocking. Deferred by decision:

- PriceCharting subscription — revisit after real-stream use.
- Fine-tuned identification model — gated on the M2 measurement.
- Non-Pokémon card types — out of scope for v1.
