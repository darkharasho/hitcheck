# Lookup window: rendered price pages instead of graded price APIs

**Date:** 2026-08-31
**Status:** Approved, not implemented
**Supersedes:** the graded half of the Pricing section in
`2026-08-10-hitcheck-design.md`

## The problem

The original design puts a *number* on screen for every identified card. That
works for raw singles — pokemontcg.io returns TCGplayer prices in every card
response, free. It does not work for graded slabs. eBay's sold-comp data
(Marketplace Insights) is gated to approved partners, eBay's consumer Price
Guide is UI-only, and PriceCharting's API is $50/mo. The design's answer was
eBay Browse *active asks*, labeled as asks — a weaker number that the user has
to mentally discount.

There is a third option the original design did not consider: render the price
page the user would have opened anyway, and let them read it.

## Decision

Keep numeric pricing where a free source exists. Add a second, browser-shaped
answer where one does not.

| Detection | Overlay shows | Lookup window loads |
|---|---|---|
| Raw single | Name + numeric price from `PriceSource` | TCGplayer product page (exact deep link) |
| Graded slab | Name + confidence, no number | PriceCharting product page (every grade) |
| Sealed product | Product name | TCGplayer search |

eBay Browse active asks are **dropped**. A PriceCharting page carries real sold
comps for every grade; an ask-median was only ever a proxy for it.

### Why a window and not DOM extraction

A considered and rejected option was to render the page and scrape a number out
of it, preserving M6's automatic spread. Rejected because a broken selector
yields a *wrong number* rather than *no number*, and the failure mode this
project designs against is the user bidding real money on a confident wrong
answer. A rendered page cannot be confidently wrong: a Clefairy page under a
Charizard slab is obvious in a glance.

The consequence is real and accepted: **there is no automatic spread for graded
cards.** M6 computes a spread for raw singles only. For slabs the user reads
the grade row and compares it to the OCR'd bid themselves.

### Not scraping

The lookup window is a browser loading public pages in the user's own session,
at human pace, one page per lot. It does not collect, store, or aggregate page
content. This distinction is deliberate and is the reason DOM extraction was
rejected above.

## Surface

A second `BrowserWindow`, created once at app start and thereafter only
navigated. Not a panel inside the overlay: the overlay is transparent and
click-through by design and cannot host an interactive page without giving that
up. A separate window also parks on a second monitor beside the stream, and
never steals focus from it the way `shell.openExternal` would.

Creating it at startup keeps window creation and cold page-load off the critical
path. When the stability gate fires, the only work left is `loadURL`.

## Modules

New code lives in `app/src/main/lookup/`. `pricing/` keeps its original job —
`PriceSource`, pokemontcg.io, raw singles — unchanged.

| Module | Responsibility | Depends on |
|---|---|---|
| `lookup/url.ts` | Pure: card or product record → destination URL | nothing |
| `lookup/router.ts` | Pure: classified detection + top-3 candidates → ordered destinations | `url.ts` |
| `lookup/window.ts` | `BrowserWindow` lifecycle: create, navigate, report load outcome, persist session | Electron |

`url.ts` and `router.ts` hold the entire decision surface and are pure functions
over plain records — no network, no screen, no Electron, testable exactly like
`vision/`. `window.ts` is a thin imperative shell.

`router.ts` stays pure, which means it cannot ask the network whether a slug
resolves. It emits an *ordered list* of destinations; `window.ts` walks that
list. The router does not learn what happened. This is deliberate: it keeps the
churn-prone slug rules under fast unit tests.

## URL construction

### TCGplayer singles — already solved

pokemontcg.io returns `tcgplayer.url`, an exact product deep link, in every card
payload. `catalog/db.py` already persists the untouched payload in `raw_json`,
so this is on disk for all 20,427 cards today.

**Migration:** add a `tcgplayer_url` column and backfill from `raw_json`. No
re-sync, no additional API calls. Cards without the field fall back to a
constructed TCGplayer search from name + set.

### PriceCharting slabs — construct, and let their site catch misses

Verified against the live site on 2026-08-31:

- `/game/pokemon-base-set/charizard-4` resolves on a first-guess slug
  (`/game/pokemon-<set-slug>/<name-slug>-<number>`), returns HTTP 200, and the
  page carries Ungraded, PSA 9, PSA 10, BGS and CGC rows.
- A deliberately invalid slug returns **HTTP 200 and redirects to
  `/search-products`**, with the slug's own words as the query.

So PriceCharting never 404s, and implements the fallback itself. Two
consequences:

1. **No soft-404 detection is needed.** The originally planned
   `executeJavaScript` content-marker check is unnecessary. Detecting a miss is
   just comparing the final URL's path against `/search-products` after
   `did-navigate`.
2. **Our own search URL is still worth having.** Their auto-search derives the
   query from the slug's trailing segment only and drops the `/game/<set>`
   context — a wrong-set slug auto-searches without the set name. On detecting
   the redirect, re-navigate to a search URL built from name + set + number,
   which retains it.

Because the slug already covers every grade, **the slab grade is not part of the
URL.** M5's grade OCR becomes a highlight hint on an already-loaded page. This
feature does not depend on M5 and can ship before it.

### Sealed products

No external product catalog is required. Set names already exist in the catalog
(`set_name`, ~170 of them) and product type is a small closed vocabulary:
Booster Box, Elite Trainer Box, Booster Bundle, Blister, Collection Box, Tin,
Booster Pack. The cross product, generated locally, is the sealed catalog.

Identification is OCR-first — set name and product type are the largest text on
the package, unlike a card's set number, which is a few pixels tall through
stream compression. OCR output matches against the `(set × type)` list; the
match resolves to a TCGplayer search URL. Sealed gets search rather than deep
links because TCGplayer sealed product IDs are not in any data we hold; the
extra click lands on the product type whose lots close slowest.

## The classifier this depends on

Routing needs a slab / sealed / raw-single decision before a URL can be built,
and nothing produces one today. M3's synthetic-first detector is trained to find
a card-shaped region; it does not label what kind of object that region is.

M4 therefore needs a **two-way** slab-vs-raw head on the existing detector —
sealed is out of scope until M4.5, and a sealed box is far enough from a card
silhouette that M3's detector will simply not fire on it, which fails safe. The
slab case is well suited to synthetic training data for the same reason cards
are: slab frames are a small set of rigid, high-contrast templates already used
by M3's compositing pipeline.

If the two-way head is not ready, M4 can ship routing everything to TCGplayer
singles and treat slab support as the head's acceptance test.

## Confidence and ambiguity

The spec's confidence gate exists because a wrong *number* costs money. A wrong
*page* does not — it is self-evidently wrong and asserts nothing. The gate is
therefore not inherited here.

- The window navigates to top-1 **regardless of confidence**.
- The overlay shows the candidate name with a confidence badge.
- The two runner-up candidates appear in the overlay as buttons that
  re-navigate the window.

This converts ambiguity the model cannot resolve — the near-identical reprint
case — into a two-second human comparison against the page's own card image.
The strict confidence gate remains in force for the numeric raw-single price,
where it belongs.

## Data flow

```
stability gate fires
  ├─► classify: slab | sealed | raw single
  ├─► pre-warm lookup window (parallel, off critical path)
  └─► retrieval ─► top-3 candidates
                    ├─► overlay: name + confidence badge + 2 runner-up buttons
                    └─► router(top-1) ─► window.navigate(ordered destinations)
                                          └─ landed on /search-products?
                                             ─► navigate(our search URL)
```

The overlay never blocks on the page. The card name appears at retrieval speed;
the window catches up independently.

## Failure modes

| Failure | Behavior |
|---|---|
| Constructed slug misses | PriceCharting redirects to its own search; we detect the redirect and re-navigate to a set-aware search URL |
| Network down | Browser error page in the window; overlay still names the card. Visibly degraded, nothing asserted |
| Wrong identification | Wrong card's page is obvious; runner-up buttons re-navigate |
| Card has no `tcgplayer_url` | Constructed TCGplayer search from name + set |
| OCR misreads a sealed box | Lands on a TCGplayer search for the wrong set; visible, and the search box is editable |
| PriceCharting changes its slug scheme | Every lookup falls through to search. Degraded, not broken — equivalent to search-always |

## Testing

- `url.ts` — table-driven fixtures covering normal cards, promos, Japanese sets,
  names containing `/`, `'` and `é`, and cards missing `tcgplayer_url`. This is
  where slug rules are pinned.
- `router.ts` — classification × candidate list → expected ordered destinations.
- `window.ts` — no unit tests. Its load-outcome classification (final URL →
  hit / redirected-to-search / failed) is extracted as a pure function and
  tested against recorded URL pairs. The `BrowserWindow` glue gets one manual
  smoke check.
- No test in this feature touches the network.

## Milestone impact

M4 is reshaped and split; M5 is decoupled.

| # | Milestone | Change |
|---|---|---|
| **M4** | `PriceSource` (raw singles) + lookup window for raw and slab | Was "`PriceSource` + cache-first lookup". eBay Browse dropped |
| **M4.5** | Sealed product path | New. Depends on the OCR engine that M6 introduces |
| **M5** | Slab classes + grade OCR | No longer blocks price display; grade becomes a highlight hint |
| **M6** | Bid-region OCR + spread | Spread is computed for raw singles only |

M4 ships the described experience — see a slab, get its grade table — without
waiting on OCR.

## Risks

| Risk | Mitigation |
|---|---|
| No automatic spread for graded cards | Accepted. The alternative was DOM extraction, whose failure mode is a wrong number |
| Slug scheme drifts for promos and Japanese sets | Fallback is search; `url.ts` fixtures cover these cases explicitly |
| Second window is awkward on a single-monitor setup | Position is user-configurable; single-monitor use is a known rough edge, not addressed in M4 |
| A learned `card_id → slug` cache was deferred | Deliberate. Build it once the fallback rate is observable, not before |
