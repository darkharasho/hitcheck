## Project Context

HitCheck is a cross-platform desktop app for **buyers** watching live card-selling
streams (eBay Live, TikTok Live). It screen-captures the stream video, identifies
the Pokémon card the seller is holding to camera, reads the slab grade
(PSA/CGC/BGS), looks up market value, OCRs the current bid off the screen, and
displays the spread in a private local overlay so the viewer can decide whether
to bid before the lot closes. The user is a viewer, not a seller — nothing is
ever composited into anyone's stream.

Two halves: `app/` (Electron + TypeScript + `onnxruntime-node`, shipping as
AppImage/NSIS/DMG) and `trainer/` (Python + PyTorch, dev-GPU only, emits ONNX
models and a prebuilt embedding index — never ships).

Design spec: `docs/superpowers/specs/2026-08-10-hitcheck-design.md`.

## Key decisions

- **Electron, not Tauri.** Tauri's Linux webview (WebKitGTK) lacks
  `getDisplayMedia`, which would force three native capture backends. Electron
  inherits Chromium's. Capture stays behind an interface so a port stays possible.
- **Identification is retrieval, not classification.** Embed the card crop, ANN
  search against a prebuilt catalog index. Not a 20k-class model, and not
  OCR-first — set numbers are a few pixels tall through stream compression.
- **Measure before training.** Zero-shot DINOv2 retrieval accuracy is evaluated
  (M2) before any identification model is trained. It may make training redundant.
- **Stability gate over per-frame inference.** Identify only when a detection
  holds steady and passes a sharpness check. ~250ms of deliberate latency, buys
  accuracy on a shaky handheld card.
- **Latency is a hard requirement.** Live lots close in seconds; a late answer is
  worthless. Target is sub-second from steady-hold to price on screen.
- **Confidence gate.** Below threshold, show "identifying…" — never a wrong price.
  The user is spending real money on this number.

## Prior art in sibling repos

Reuse rather than rediscover — `../axistream`, `../sai`, `../otto`:

- Wayland portal approval persists via restore token (proven on this machine by
  axistream: Bazzite/KDE/Wayland).
- `desktopCapturer.getSources()` thumbnails come back blank on Wayland — that's
  the *screenshot* API. Use `getDisplayMedia()` for continuous video; it's
  portal-backed and works.
- `otto` already ships `onnxruntime-node` 1.21.0 + `sharp` in Electron with
  AppImage/NSIS/DMG targets. Copy that packaging setup.
- Do **not** adopt axistream's OBS sidecar — that solves encoding/streaming,
  which HitCheck doesn't do.

## Environment

Dev machine is Bazzite (Fedora Atomic, KDE Kinoite), Wayland, RTX 4070 Ti 12GB.
The OS is immutable — Python/CUDA work goes in a distrobox/podman container or a
`uv` venv, never `dnf install`.

## Conventions

- Workflow: design spec (`docs/superpowers/specs/`) → implementation plan
  (`docs/superpowers/plans/`) → feature branch → merge to main.
- Tests: vitest with fork pool capped at 2 workers.
- `vision/` must stay testable against still images — no screen, stream, or
  network dependency.
