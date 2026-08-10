# HitCheck

Real-time Pokémon card and slab pricing overlay for live streams.

HitCheck is a desktop app for **buyers** watching live card-selling streams
(eBay Live, TikTok Live). It screen-captures the stream, identifies the card the
seller is holding up, reads the grade if it's slabbed (PSA / CGC / BGS), looks up
market value, reads the current bid off the screen, and shows you the spread —
so you can decide whether to bid before the lot closes.

The overlay is private to you. Nothing is composited into anyone's stream.

## Status

Design complete, implementation not started. See
[the design spec](docs/superpowers/specs/2026-08-10-hitcheck-design.md).

## How it works

```
getDisplayMedia ─┬─► card loop ─► detect ─► stability gate ─► identify ─► price
                 └─► bid loop  ─► OCR bid region ──────────────────────────┐
                                                                            ▼
                                                       overlay (transparent, click-through)
```

Identification is **retrieval, not classification** — the detected card is
embedded and matched against a prebuilt index of the full card catalog, which
survives stream compression far better than reading set numbers as text.

## Two halves

- **`app/`** — Electron + TypeScript, ONNX Runtime. Ships as AppImage, NSIS
  installer, and DMG.
- **`trainer/`** — Python + PyTorch. Runs on a dev GPU, emits ONNX models and a
  prebuilt index. Never ships to users.

## Pricing sources

Raw prices come from [pokemontcg.io](https://docs.pokemontcg.io) (free tier,
embeds TCGplayer data). Graded prices start from eBay Browse API active listings
— **asking prices, labeled as such** — with PriceCharting available as a paid
upgrade behind the same interface.

## License

TBD.

---

Not affiliated with, endorsed by, or sponsored by Nintendo, The Pokémon Company,
eBay, TCGplayer, or any grading company.
