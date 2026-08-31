# Lookup Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a card is identified, a second always-open browser window
navigates to that card's price page — PriceCharting for slabs, TCGplayer for
raw singles — with the overlay offering the runner-up candidates as one-click
re-navigations.

**Architecture:** All routing logic is pure functions over plain records in
`app/src/main/lookup/` (`slug.ts`, `url.ts`, `router.ts`, `outcome.ts`) with no
Electron, no network, and no screen dependency — tested exactly like `vision/`.
A thin imperative shell (`window.ts`) owns the `BrowserWindow`, walks the
ordered destination list the router produces, and reports what it landed on.

**Tech Stack:** TypeScript (ESM, strict), Electron 36, vitest (forks pool, max 2
workers). Trainer side: Python 3.12, sqlite3 stdlib, pytest.

**Spec:** `docs/superpowers/specs/2026-08-31-lookup-window-design.md`

## Global Constraints

- Node/TS files use ESM with no file extensions on relative imports, matching
  existing `app/src` code (`import { rankSources } from './filter'`).
- Tests are vitest, colocated as `<module>.test.ts` beside the module.
- Run tests as `npm test` from `app/` — `vitest.config.ts` already caps the
  fork pool at `maxForks: 2`. Never raise it.
- No test in this plan may touch the network, the screen, or Electron.
- Trainer tests run as `uv run pytest` from `trainer/`, line-length 100 (ruff).
- The lookup window must never be created on the critical path — it is created
  once at app startup and thereafter only navigated.
- The app never extracts prices from a rendered page. Rendering is the answer;
  DOM scraping was rejected in the spec and must not be reintroduced.

## Scope

This plan covers **M4 only**: the lookup window for raw singles and slabs.

Two things the spec describes are explicitly **out of scope** here and get
their own plans:

- **The slab-vs-raw classifier head** (a trainer-side change to M3's detector).
  This plan treats the classification as an *input*: `routeCard` takes a
  `Classification` argument. Until the head exists, callers pass
  `'raw-single'`, which is the spec's documented fallback.
- **Sealed products (M4.5)**, which depend on the OCR engine M6 introduces.

## Findings that change the spec

Two spec statements were checked against live services on 2026-08-31 and one is
wrong. Task 1 corrects the spec text.

1. **`tcgplayer.url` is not a direct deep link.** It is
   `https://prices.pokemontcg.io/tcgplayer/<card-id>`, a redirector that lands
   on the real product page carrying a *third party's* affiliate and tracking
   parameters (`utm_campaign=Scrydex`, `irclickid=...`). The bare
   `https://www.tcgplayer.com/product/42382` works fine without them. Task 9
   caches the resolved bare URL so the redirector is hit at most once per card.
2. **266 of 20,479 cards have no `tcgplayer.url`** (20,213 do). The search
   fallback is a real path, not a theoretical one.

Empirically measured PriceCharting slug behaviour (14 cards sampled):
`pokemon-<slugified set name>/<slugified card name>-<number>` resolved for 12.
`Base` requires the slug `pokemon-base-set` (`pokemon-base` misses). `Kalos
Starter Set` misses and falls through to search. PriceCharting never 404s — a
bad slug returns HTTP 200 and redirects to `/search-products`.

---

### Task 1: Catalog `tcgplayer_url` column, and spec correction

**Files:**
- Modify: `trainer/hitcheck_trainer/catalog/db.py`
- Test: `trainer/tests/test_db.py`
- Modify: `docs/superpowers/specs/2026-08-31-lookup-window-design.md`

**Interfaces:**
- Consumes: nothing
- Produces: a `tcgplayer_url TEXT` column on the `cards` table, populated for
  existing rows by `backfill_tcgplayer_urls(conn) -> int` (returns rows
  updated), and populated for new rows by the existing `upsert_cards`.

- [ ] **Step 1: Write the failing tests**

Append to `trainer/tests/test_db.py`:

```python
def test_upsert_stores_tcgplayer_url(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    upsert_cards(conn, [{
        "id": "base1-4", "name": "Charizard", "number": "4",
        "set": {"id": "base1", "name": "Base"},
        "tcgplayer": {"url": "https://prices.pokemontcg.io/tcgplayer/base1-4"},
    }])
    assert get_card(conn, "base1-4")["tcgplayer_url"] == \
        "https://prices.pokemontcg.io/tcgplayer/base1-4"


def test_upsert_tolerates_missing_tcgplayer_block(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    upsert_cards(conn, [{"id": "x-1", "name": "X", "set": {"id": "x"}}])
    assert get_card(conn, "x-1")["tcgplayer_url"] is None


def test_backfill_populates_from_raw_json(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    upsert_cards(conn, [{
        "id": "base1-4", "name": "Charizard",
        "set": {"id": "base1", "name": "Base"},
        "tcgplayer": {"url": "https://prices.pokemontcg.io/tcgplayer/base1-4"},
    }])
    conn.execute("UPDATE cards SET tcgplayer_url = NULL")
    conn.commit()

    assert backfill_tcgplayer_urls(conn) == 1
    assert get_card(conn, "base1-4")["tcgplayer_url"] == \
        "https://prices.pokemontcg.io/tcgplayer/base1-4"


def test_backfill_is_idempotent(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    upsert_cards(conn, [{
        "id": "base1-4", "name": "Charizard",
        "set": {"id": "base1", "name": "Base"},
        "tcgplayer": {"url": "https://prices.pokemontcg.io/tcgplayer/base1-4"},
    }])
    backfill_tcgplayer_urls(conn)
    assert backfill_tcgplayer_urls(conn) == 0
```

Add `backfill_tcgplayer_urls` to the existing import line at the top of
`trainer/tests/test_db.py`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd trainer && uv run pytest tests/test_db.py -v`
Expected: FAIL — `ImportError: cannot import name 'backfill_tcgplayer_urls'`

- [ ] **Step 3: Add the column and the migration**

In `db.py`, add the column to `SCHEMA` inside `CREATE TABLE cards`, after
`image_small TEXT,`:

```python
    tcgplayer_url TEXT,
```

`CREATE TABLE IF NOT EXISTS` will not alter an existing table, so `open_db`
needs an explicit migration. Add this function and call it from `open_db`:

```python
def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created.

    `CREATE TABLE IF NOT EXISTS` silently does nothing on an existing table,
    so new columns must be added here or a synced catalog would keep the old
    schema forever.
    """
    have = {r["name"] for r in conn.execute("PRAGMA table_info(cards)")}
    if "tcgplayer_url" not in have:
        conn.execute("ALTER TABLE cards ADD COLUMN tcgplayer_url TEXT")
        conn.commit()
```

In `open_db`, after `conn.executescript(SCHEMA)` and its `commit()`, add:

```python
    _migrate(conn)
```

- [ ] **Step 4: Store the URL on upsert**

In `upsert_cards`, before building `rows.append(...)`, add alongside the
existing `card_set` and `images` locals:

```python
        tcgplayer = card.get("tcgplayer") or {}
```

Add `tcgplayer.get("url"),` to the tuple, immediately after
`images.get("small"),` and before the `json.dumps(...)` line. Then update the
SQL: add `tcgplayer_url` to the column list after `image_small`, add one more
`?` to `VALUES`, and add `tcgplayer_url=excluded.tcgplayer_url,` to the
`DO UPDATE SET` clause.

The resulting statement:

```python
            INSERT INTO cards (id, name, number, rarity, supertype, artist,
                               set_id, set_name, set_series, set_release,
                               image_small, tcgplayer_url, raw_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, number=excluded.number, rarity=excluded.rarity,
                supertype=excluded.supertype, artist=excluded.artist,
                set_id=excluded.set_id, set_name=excluded.set_name,
                set_series=excluded.set_series, set_release=excluded.set_release,
                image_small=excluded.image_small,
                tcgplayer_url=excluded.tcgplayer_url, raw_json=excluded.raw_json
```

- [ ] **Step 5: Write the backfill**

Add to `db.py`:

```python
def backfill_tcgplayer_urls(conn: sqlite3.Connection) -> int:
    """Populate tcgplayer_url from the raw payload already on disk.

    The full API response was stored in raw_json from the first sync, so this
    needs no network access and no re-sync. Returns the number of rows updated.
    """
    updates = []
    for row in conn.execute(
        "SELECT id, raw_json FROM cards WHERE tcgplayer_url IS NULL"
    ).fetchall():
        url = (json.loads(row["raw_json"]).get("tcgplayer") or {}).get("url")
        if url:
            updates.append((url, row["id"]))
    conn.executemany("UPDATE cards SET tcgplayer_url = ? WHERE id = ?", updates)
    conn.commit()
    return len(updates)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd trainer && uv run pytest tests/test_db.py -v`
Expected: PASS, including the pre-existing tests in that file.

- [ ] **Step 7: Run the backfill against the real catalog**

Run:
```bash
cd trainer && uv run python -c "
from hitcheck_trainer.catalog.db import open_db, backfill_tcgplayer_urls
c = open_db('data/catalog.sqlite')
print('updated:', backfill_tcgplayer_urls(c))
print('with url:', c.execute('SELECT COUNT(*) FROM cards WHERE tcgplayer_url IS NOT NULL').fetchone()[0])
"
```
Expected: `updated: 20213` and `with url: 20213`. If the second number differs
from the first on a fresh run, the migration did not apply — stop and
investigate before continuing.

- [ ] **Step 8: Correct the spec**

In `docs/superpowers/specs/2026-08-31-lookup-window-design.md`, replace the
paragraph under "### TCGplayer singles — already solved" that claims
`tcgplayer.url` is "an exact product deep link" with:

```markdown
pokemontcg.io returns `tcgplayer.url` in every card payload — verified present
for 20,213 of 20,479 cards. It is **not** a direct product link: it is
`https://prices.pokemontcg.io/tcgplayer/<card-id>`, a redirector that lands on
the real product page carrying a third party's affiliate and tracking
parameters (`utm_campaign=Scrydex`, `irclickid=...`). The bare
`https://www.tcgplayer.com/product/<id>` resolves fine without them.

Consequence: every lookup through the redirector is logged by a third party and
credits them for any resulting purchase. The resolved bare URL is therefore
cached per card on first use, so the redirector is hit at most once per card
rather than once per lot. Cards without the field (266 of them) fall back to a
constructed TCGplayer search.
```

- [ ] **Step 9: Commit**

```bash
git add trainer/hitcheck_trainer/catalog/db.py trainer/tests/test_db.py \
        docs/superpowers/specs/2026-08-31-lookup-window-design.md
git commit -m "feat(catalog): store tcgplayer_url and backfill from raw_json"
```

---

### Task 2: Slug rules

**Files:**
- Create: `app/src/main/lookup/types.ts`
- Create: `app/src/main/lookup/slug.ts`
- Test: `app/src/main/lookup/slug.test.ts`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `type CardRecord = { id: string; name: string; number: string | null; setName: string; tcgplayerUrl: string | null }`
  - `type Classification = 'slab' | 'raw-single'`
  - `slugify(value: string): string`
  - `priceChartingSetSlug(setName: string): string` — returns the full path
    segment including the `pokemon-` prefix, e.g. `pokemon-base-set`

- [ ] **Step 1: Write the failing test**

Create `app/src/main/lookup/slug.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { slugify, priceChartingSetSlug } from './slug'

describe('slugify', () => {
  it('lowercases and hyphenates', () => {
    expect(slugify('Vivid Voltage')).toBe('vivid-voltage')
  })

  it('drops apostrophes rather than hyphenating them', () => {
    expect(slugify("Team Rocket's Meowth")).toBe('team-rockets-meowth')
  })

  it('drops typographic apostrophes too', () => {
    expect(slugify('Farfetch’d')).toBe('farfetchd')
  })

  it('strips accents', () => {
    expect(slugify('Flabébé')).toBe('flabebe')
  })

  it('spells out ampersands', () => {
    expect(slugify('Scarlet & Violet')).toBe('scarlet-and-violet')
  })

  it('collapses runs of punctuation into a single hyphen', () => {
    expect(slugify('Hidden Fates: Shiny Vault')).toBe('hidden-fates-shiny-vault')
  })

  it('never leaves leading or trailing hyphens', () => {
    expect(slugify('  Fossil!  ')).toBe('fossil')
  })

  it('returns an empty string for input with no alphanumerics', () => {
    expect(slugify('!!!')).toBe('')
  })
})

describe('priceChartingSetSlug', () => {
  it('prefixes the slugified set name', () => {
    expect(priceChartingSetSlug('Vivid Voltage')).toBe('pokemon-vivid-voltage')
  })

  // Measured 2026-08-31: pokemon-base misses and falls through to search;
  // pokemon-base-set resolves. This is the only override the sample found.
  it('applies the Base override', () => {
    expect(priceChartingSetSlug('Base')).toBe('pokemon-base-set')
  })

  it('does not apply the Base override to sets merely starting with Base', () => {
    expect(priceChartingSetSlug('Base Set 2')).toBe('pokemon-base-set-2')
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd app && npm test -- src/main/lookup/slug.test.ts`
Expected: FAIL — cannot resolve `./slug`

- [ ] **Step 3: Write the types**

Create `app/src/main/lookup/types.ts`:

```typescript
/** The subset of a catalog row the lookup layer needs to build a URL. */
export type CardRecord = {
  id: string
  name: string
  /** pokemontcg.io card number, e.g. "4" or "SV49". Null for cards without one. */
  number: string | null
  setName: string
  /** pokemontcg.io redirector URL, or null for the 266 cards lacking one. */
  tcgplayerUrl: string | null
}

/**
 * What kind of object the detector found. Sealed products are M4.5 and are
 * deliberately absent — adding them here is a signal that the sealed plan has
 * started.
 */
export type Classification = 'slab' | 'raw-single'
```

- [ ] **Step 4: Write the slug rules**

Create `app/src/main/lookup/slug.ts`:

```typescript
/**
 * PriceCharting set slugs that do not follow the default rule.
 *
 * Derived empirically on 2026-08-31 by requesting constructed slugs and
 * checking whether the site redirected to /search-products. Of 14 sets
 * sampled, only "Base" needed an override. Add entries here as real misses
 * are observed — do not guess.
 */
const SET_SLUG_OVERRIDES: Record<string, string> = {
  Base: 'base-set',
}

export function slugify(value: string): string {
  return value
    .toLowerCase()
    .normalize('NFD')
    // Strip combining marks left behind by NFD, so "é" becomes "e".
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/&/g, ' and ')
    // Apostrophes vanish rather than becoming hyphens: PriceCharting slugs
    // "Team Rocket's Meowth" as team-rockets-meowth, not team-rocket-s-meowth.
    .replace(/['’]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

export function priceChartingSetSlug(setName: string): string {
  const override = SET_SLUG_OVERRIDES[setName]
  return `pokemon-${override ?? slugify(setName)}`
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd app && npm test -- src/main/lookup/slug.test.ts`
Expected: PASS, 11 tests.

- [ ] **Step 6: Commit**

```bash
git add app/src/main/lookup/types.ts app/src/main/lookup/slug.ts \
        app/src/main/lookup/slug.test.ts
git commit -m "feat(lookup): slug rules for PriceCharting URLs"
```

---

### Task 3: URL builders

**Files:**
- Create: `app/src/main/lookup/url.ts`
- Test: `app/src/main/lookup/url.test.ts`

**Interfaces:**
- Consumes: `CardRecord` from `./types`, `slugify` and `priceChartingSetSlug`
  from `./slug`
- Produces:
  - `priceChartingProductUrl(card: CardRecord): string`
  - `priceChartingSearchUrl(card: CardRecord): string`
  - `tcgplayerSearchUrl(card: CardRecord): string`

- [ ] **Step 1: Write the failing test**

Create `app/src/main/lookup/url.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import {
  priceChartingProductUrl,
  priceChartingSearchUrl,
  tcgplayerSearchUrl,
} from './url'
import type { CardRecord } from './types'

const card = (over: Partial<CardRecord> = {}): CardRecord => ({
  id: 'base1-4',
  name: 'Charizard',
  number: '4',
  setName: 'Base',
  tcgplayerUrl: 'https://prices.pokemontcg.io/tcgplayer/base1-4',
  ...over,
})

describe('priceChartingProductUrl', () => {
  // Every expectation below was confirmed to resolve on the live site
  // (no redirect to /search-products) on 2026-08-31.
  it('builds the verified Base Set Charizard URL', () => {
    expect(priceChartingProductUrl(card())).toBe(
      'https://www.pricecharting.com/game/pokemon-base-set/charizard-4',
    )
  })

  it('builds a modern set URL', () => {
    expect(priceChartingProductUrl(card({
      id: 'swsh4-188', name: 'Pikachu VMAX', number: '188', setName: 'Vivid Voltage',
    }))).toBe(
      'https://www.pricecharting.com/game/pokemon-vivid-voltage/pikachu-vmax-188',
    )
  })

  it('slugifies apostrophes in card names', () => {
    expect(priceChartingProductUrl(card({
      id: 'basep-18', name: "Team Rocket's Meowth", number: '18',
      setName: 'Wizards Black Star Promos',
    }))).toBe(
      'https://www.pricecharting.com/game/pokemon-wizards-black-star-promos/team-rockets-meowth-18',
    )
  })

  it('slugifies alphanumeric card numbers', () => {
    expect(priceChartingProductUrl(card({ number: 'SV49' }))).toBe(
      'https://www.pricecharting.com/game/pokemon-base-set/charizard-sv49',
    )
  })

  it('omits the number segment when the card has none', () => {
    expect(priceChartingProductUrl(card({ number: null }))).toBe(
      'https://www.pricecharting.com/game/pokemon-base-set/charizard',
    )
  })
})

describe('priceChartingSearchUrl', () => {
  it('includes name, set and number so set context is not lost', () => {
    expect(priceChartingSearchUrl(card())).toBe(
      'https://www.pricecharting.com/search-products?type=prices&q=Charizard+Base+4',
    )
  })

  it('omits a null number', () => {
    expect(priceChartingSearchUrl(card({ number: null }))).toBe(
      'https://www.pricecharting.com/search-products?type=prices&q=Charizard+Base',
    )
  })
})

describe('tcgplayerSearchUrl', () => {
  it('searches the pokemon product line by name and set', () => {
    expect(tcgplayerSearchUrl(card())).toBe(
      'https://www.tcgplayer.com/search/pokemon/product?q=Charizard+Base',
    )
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd app && npm test -- src/main/lookup/url.test.ts`
Expected: FAIL — cannot resolve `./url`

- [ ] **Step 3: Write the implementation**

Create `app/src/main/lookup/url.ts`:

```typescript
import { slugify, priceChartingSetSlug } from './slug'
import type { CardRecord } from './types'

const PRICECHARTING = 'https://www.pricecharting.com'
const TCGPLAYER = 'https://www.tcgplayer.com'

/** Human-readable query terms, in the order a person would type them. */
function queryTerms(card: CardRecord, includeNumber: boolean): string {
  return [card.name, card.setName, includeNumber ? card.number : null]
    .filter((part): part is string => Boolean(part))
    .join(' ')
}

/**
 * Best-guess direct product page. PriceCharting never 404s — a wrong slug
 * returns 200 and redirects to /search-products — so this is always safe to
 * try first. See classifyLanding in ./outcome for how a miss is detected.
 */
export function priceChartingProductUrl(card: CardRecord): string {
  const name = slugify(card.name)
  const tail = card.number ? `${name}-${slugify(card.number)}` : name
  return `${PRICECHARTING}/game/${priceChartingSetSlug(card.setName)}/${tail}`
}

/**
 * Our own search URL, used when the constructed slug misses. Worth having
 * even though PriceCharting auto-searches on a miss: their auto-search derives
 * the query from the slug's trailing segment alone and drops the set name, so
 * a wrong-set slug would search without set context. This keeps it.
 */
export function priceChartingSearchUrl(card: CardRecord): string {
  const params = new URLSearchParams({ type: 'prices', q: queryTerms(card, true) })
  return `${PRICECHARTING}/search-products?${params}`
}

/** Fallback for the 266 catalog cards with no pokemontcg.io TCGplayer link. */
export function tcgplayerSearchUrl(card: CardRecord): string {
  const params = new URLSearchParams({ q: queryTerms(card, false) })
  return `${TCGPLAYER}/search/pokemon/product?${params}`
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd app && npm test -- src/main/lookup/url.test.ts`
Expected: PASS, 8 tests.

`URLSearchParams` encodes spaces as `+`, which is what both sites use and what
the expectations above assert. If a test fails showing `%20`, the
implementation built the query by string concatenation instead — fix the
implementation, not the test.

- [ ] **Step 5: Commit**

```bash
git add app/src/main/lookup/url.ts app/src/main/lookup/url.test.ts
git commit -m "feat(lookup): PriceCharting and TCGplayer URL builders"
```

---

### Task 4: The router

**Files:**
- Create: `app/src/main/lookup/router.ts`
- Test: `app/src/main/lookup/router.test.ts`

**Interfaces:**
- Consumes: `CardRecord` and `Classification` from `./types`; all three
  builders from `./url`
- Produces:
  - `type Destination = { url: string; kind: 'product' | 'search' }`
  - `routeCard(card: CardRecord, classification: Classification): Destination[]`
    — ordered, always non-empty, best guess first

- [ ] **Step 1: Write the failing test**

Create `app/src/main/lookup/router.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { routeCard } from './router'
import type { CardRecord } from './types'

const card = (over: Partial<CardRecord> = {}): CardRecord => ({
  id: 'base1-4',
  name: 'Charizard',
  number: '4',
  setName: 'Base',
  tcgplayerUrl: 'https://prices.pokemontcg.io/tcgplayer/base1-4',
  ...over,
})

describe('routeCard', () => {
  it('sends slabs to PriceCharting, product first then search', () => {
    const out = routeCard(card(), 'slab')
    expect(out.map(d => d.kind)).toEqual(['product', 'search'])
    expect(out[0].url).toContain('/game/pokemon-base-set/charizard-4')
    expect(out[1].url).toContain('/search-products')
  })

  it('sends raw singles to the TCGplayer link from the catalog', () => {
    const out = routeCard(card(), 'raw-single')
    expect(out).toEqual([
      { url: 'https://prices.pokemontcg.io/tcgplayer/base1-4', kind: 'product' },
    ])
  })

  it('falls back to TCGplayer search when the card has no link', () => {
    const out = routeCard(card({ tcgplayerUrl: null }), 'raw-single')
    expect(out.map(d => d.kind)).toEqual(['search'])
    expect(out[0].url).toContain('tcgplayer.com/search/pokemon/product')
  })

  it('treats an empty-string link as missing', () => {
    const out = routeCard(card({ tcgplayerUrl: '' }), 'raw-single')
    expect(out.map(d => d.kind)).toEqual(['search'])
  })

  it('always returns at least one destination', () => {
    for (const classification of ['slab', 'raw-single'] as const) {
      expect(routeCard(card({ tcgplayerUrl: null }), classification).length)
        .toBeGreaterThan(0)
    }
  })

  it('is pure — the same input yields an equal result', () => {
    expect(routeCard(card(), 'slab')).toEqual(routeCard(card(), 'slab'))
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd app && npm test -- src/main/lookup/router.test.ts`
Expected: FAIL — cannot resolve `./router`

- [ ] **Step 3: Write the implementation**

Create `app/src/main/lookup/router.ts`:

```typescript
import {
  priceChartingProductUrl,
  priceChartingSearchUrl,
  tcgplayerSearchUrl,
} from './url'
import type { CardRecord, Classification } from './types'

export type Destination = { url: string; kind: 'product' | 'search' }

/**
 * Where to send the lookup window, best guess first.
 *
 * Deliberately pure: it cannot ask the network whether a slug resolves, so it
 * emits an ordered list and lets the window walk it. That keeps the
 * churn-prone slug rules under fast unit tests, at the cost of the router not
 * learning what actually happened.
 */
export function routeCard(
  card: CardRecord,
  classification: Classification,
): Destination[] {
  if (classification === 'slab') {
    return [
      { url: priceChartingProductUrl(card), kind: 'product' },
      { url: priceChartingSearchUrl(card), kind: 'search' },
    ]
  }
  return card.tcgplayerUrl
    ? [{ url: card.tcgplayerUrl, kind: 'product' }]
    : [{ url: tcgplayerSearchUrl(card), kind: 'search' }]
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd app && npm test -- src/main/lookup/router.test.ts`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add app/src/main/lookup/router.ts app/src/main/lookup/router.test.ts
git commit -m "feat(lookup): pure destination router"
```

---

### Task 5: Landing classification and tracking-parameter stripping

**Files:**
- Create: `app/src/main/lookup/outcome.ts`
- Test: `app/src/main/lookup/outcome.test.ts`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `type LoadOutcome = 'product' | 'search-fallback'`
  - `classifyLanding(finalUrl: string): LoadOutcome`
  - `stripTracking(url: string): string`

- [ ] **Step 1: Write the failing test**

Create `app/src/main/lookup/outcome.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { classifyLanding, stripTracking } from './outcome'

describe('classifyLanding', () => {
  it('recognises a PriceCharting product page', () => {
    expect(classifyLanding(
      'https://www.pricecharting.com/game/pokemon-base-set/charizard-4',
    )).toBe('product')
  })

  it('recognises the search page PriceCharting redirects misses to', () => {
    expect(classifyLanding(
      'https://www.pricecharting.com/search-products?type=prices&q=weedle+1',
    )).toBe('search-fallback')
  })

  // Measured 2026-08-31: a near-miss slug lands on the right product page but
  // keeps a ?q= parameter. That is a hit, not a fallback — match on path only.
  it('treats a product page carrying a q parameter as a product page', () => {
    expect(classifyLanding(
      'https://www.pricecharting.com/game/pokemon-vivid-voltage/pikachu-vmax-188?q=pikachu',
    )).toBe('product')
  })

  it('recognises a TCGplayer search page', () => {
    expect(classifyLanding(
      'https://www.tcgplayer.com/search/pokemon/product?q=Charizard+Base',
    )).toBe('search-fallback')
  })

  it('recognises a TCGplayer product page', () => {
    expect(classifyLanding('https://www.tcgplayer.com/product/42382')).toBe('product')
  })

  it('treats an unparseable URL as a fallback rather than throwing', () => {
    expect(classifyLanding('not a url')).toBe('search-fallback')
  })
})

describe('stripTracking', () => {
  it('removes the affiliate parameters the pokemontcg.io redirector adds', () => {
    expect(stripTracking(
      'https://www.tcgplayer.com/product/42382?irclickid=abc&sharedid=&irpid=4944541'
      + '&irgwc=1&afsrc=1&utm_source=impact&utm_medium=affiliate&utm_campaign=Scrydex',
    )).toBe('https://www.tcgplayer.com/product/42382')
  })

  it('leaves a URL with no tracking parameters untouched', () => {
    expect(stripTracking('https://www.tcgplayer.com/product/42382'))
      .toBe('https://www.tcgplayer.com/product/42382')
  })

  it('keeps meaningful parameters while dropping tracking ones', () => {
    expect(stripTracking(
      'https://www.pricecharting.com/search-products?type=prices&q=x&utm_source=impact',
    )).toBe('https://www.pricecharting.com/search-products?type=prices&q=x')
  })

  it('returns an unparseable URL unchanged', () => {
    expect(stripTracking('not a url')).toBe('not a url')
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd app && npm test -- src/main/lookup/outcome.test.ts`
Expected: FAIL — cannot resolve `./outcome`

- [ ] **Step 3: Write the implementation**

Create `app/src/main/lookup/outcome.ts`:

```typescript
export type LoadOutcome = 'product' | 'search-fallback'

/** Path prefixes that mean "we did not land on a specific product". */
const SEARCH_PATHS = ['/search-products', '/search/']

const TRACKING_PARAMS = [
  'irclickid', 'sharedid', 'irpid', 'irgwc', 'afsrc',
  'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
]

/**
 * What did the window actually land on?
 *
 * Matches on path only. A near-miss PriceCharting slug resolves to the right
 * product page but keeps a ?q= parameter, and that is a hit — keying off the
 * query string would misreport it as a fallback.
 */
export function classifyLanding(finalUrl: string): LoadOutcome {
  let path: string
  try {
    path = new URL(finalUrl).pathname
  } catch {
    return 'search-fallback'
  }
  return SEARCH_PATHS.some(p => path.startsWith(p)) ? 'search-fallback' : 'product'
}

/**
 * Drop affiliate and campaign parameters. The pokemontcg.io redirector appends
 * a third party's tracking to every TCGplayer landing; the bare product URL
 * works without it and is what gets cached.
 */
export function stripTracking(url: string): string {
  let parsed: URL
  try {
    parsed = new URL(url)
  } catch {
    return url
  }
  for (const param of TRACKING_PARAMS) parsed.searchParams.delete(param)
  // Drop a now-empty '?' so cached URLs compare equal to hand-written ones.
  if (![...parsed.searchParams.keys()].length) parsed.search = ''
  return parsed.toString()
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd app && npm test -- src/main/lookup/outcome.test.ts`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
git add app/src/main/lookup/outcome.ts app/src/main/lookup/outcome.test.ts
git commit -m "feat(lookup): landing classification and tracking-param stripping"
```

---

### Task 6: The lookup window

**Files:**
- Create: `app/src/main/lookup/window.ts`
- Modify: `app/src/main/index.ts:41-45` (the `app.whenReady()` block)

**Interfaces:**
- Consumes: `Destination` from `./router`; `classifyLanding` from `./outcome`
- Produces:
  - `createLookupWindow(): BrowserWindow`
  - `navigateLookup(destinations: Destination[]): Promise<void>`

There are no unit tests in this task. Its testable logic already lives in
`outcome.ts`; what remains is Electron glue, verified by the smoke script in
Step 4. This mirrors how `overlay/window.ts` is treated — pure `bounds.ts` is
tested, the `BrowserWindow` shell is not.

- [ ] **Step 1: Write the window module**

Create `app/src/main/lookup/window.ts`:

```typescript
import { BrowserWindow } from 'electron'
import { classifyLanding } from './outcome'
import type { Destination } from './router'

let lookup: BrowserWindow | null = null

/**
 * Created once at startup, never per lot. A live auction closes in seconds, so
 * window creation and a cold page load must not sit on the critical path —
 * when the stability gate fires the only remaining work is loadURL.
 */
export function createLookupWindow(): BrowserWindow {
  const outgoing = lookup
  if (outgoing && !outgoing.isDestroyed()) outgoing.close()

  const win = new BrowserWindow({
    width: 900,
    height: 1000,
    show: false,
    title: 'HitCheck — Prices',
    webPreferences: {
      // No preload and no node integration: this window renders third-party
      // pages and must have no bridge into the app.
      contextIsolation: true,
      nodeIntegration: false,
      // A named partition persists cookies across restarts, so a TCGplayer
      // login survives. Deliberately not the default session, which the
      // capture half uses.
      partition: 'persist:lookup',
    },
  })

  win.on('closed', () => { if (lookup === win) lookup = null })
  lookup = win
  return win
}

/**
 * Walk the router's ordered destinations, stopping at the first that lands on
 * a product page. A `product`-kind destination that redirects to a search page
 * is a slug miss, and the next destination is our own set-aware search.
 */
export async function navigateLookup(destinations: Destination[]): Promise<void> {
  if (!lookup || lookup.isDestroyed()) createLookupWindow()
  const win = lookup!

  for (const destination of destinations) {
    try {
      await win.loadURL(destination.url)
    } catch {
      // Network failure or an aborted load: the window shows the browser's own
      // error page. Try the next destination; if there is none, that error page
      // is what the user sees, which is honest.
      continue
    }
    win.showInactive()
    if (classifyLanding(win.webContents.getURL()) === 'product') return
  }
  win.showInactive()
}
```

- [ ] **Step 2: Create the window at startup**

In `app/src/main/index.ts`, add to the imports:

```typescript
import { createLookupWindow } from './lookup/window'
```

and add `createLookupWindow()` as the last call inside `app.whenReady()`:

```typescript
app.whenReady().then(() => {
  registerIpc()
  registerDisplayMediaHandler()
  createWindow()
  createLookupWindow()
})
```

- [ ] **Step 3: Run the existing suite to confirm nothing regressed**

Run: `cd app && npm test && npm run typecheck`
Expected: all existing tests PASS, no type errors.

- [ ] **Step 4: Smoke-test the window against the live sites**

Create `app/scripts/lookup-smoke.mjs`:

```javascript
// Manual smoke check: does the lookup window actually reach both sites, and
// does a deliberately wrong slug fall through to our search URL?
// Run: cd app && npx electron scripts/lookup-smoke.mjs
import { app } from 'electron'
import { createLookupWindow, navigateLookup } from '../out/main/lookup/window.js'

const CASES = [
  ['slab hit', [
    { url: 'https://www.pricecharting.com/game/pokemon-base-set/charizard-4', kind: 'product' },
    { url: 'https://www.pricecharting.com/search-products?type=prices&q=Charizard+Base+4', kind: 'search' },
  ]],
  ['slab miss falls through', [
    { url: 'https://www.pricecharting.com/game/pokemon-nonesuch/nothing-9999', kind: 'product' },
    { url: 'https://www.pricecharting.com/search-products?type=prices&q=Weedle+Kalos+Starter+Set+1', kind: 'search' },
  ]],
  ['raw single', [
    { url: 'https://prices.pokemontcg.io/tcgplayer/base1-4', kind: 'product' },
  ]],
]

app.whenReady().then(async () => {
  const win = createLookupWindow()
  for (const [label, destinations] of CASES) {
    await navigateLookup(destinations)
    console.log(`${label}: ${win.webContents.getURL()}`)
    await new Promise(r => setTimeout(r, 2000))
  }
  app.quit()
})
```

Run:
```bash
cd app && npm run build && npx electron scripts/lookup-smoke.mjs
```

Expected output — three lines, and a visible window:
- `slab hit: https://www.pricecharting.com/game/pokemon-base-set/charizard-4`
- `slab miss falls through: https://www.pricecharting.com/search-products?...q=Weedle...`
  (the *second* destination, proving fall-through fired — not PriceCharting's
  own auto-search, which would show `q=nothing+9999`)
- `raw single: https://www.tcgplayer.com/product/42382?...` (tracking params
  present; Task 9 removes them)

If the second line shows `q=nothing+9999`, `classifyLanding` is being fed the
requested URL rather than `webContents.getURL()` — fix `navigateLookup`.

- [ ] **Step 5: Commit**

```bash
git add app/src/main/lookup/window.ts app/src/main/index.ts app/scripts/lookup-smoke.mjs
git commit -m "feat(lookup): browser window created at startup, walks destinations"
```

---

### Task 7: IPC wiring

**Files:**
- Modify: `app/src/main/ipc.ts`
- Modify: `app/src/preload/index.ts`

**Interfaces:**
- Consumes: `routeCard` from `../lookup/router`, `navigateLookup` from
  `../lookup/window`, `CardRecord` and `Classification` from `../lookup/types`
- Produces: IPC channel `hitcheck:lookupCard`, exposed to the renderer as
  `window.hitcheck.lookupCard(card: CardRecord, classification: Classification): Promise<void>`

- [ ] **Step 1: Add the handler**

In `app/src/main/ipc.ts`, add imports:

```typescript
import { routeCard } from './lookup/router'
import { navigateLookup } from './lookup/window'
import type { CardRecord, Classification } from './lookup/types'
```

and add inside `registerIpc()`:

```typescript
  ipcMain.handle(
    'hitcheck:lookupCard',
    (_e, card: CardRecord, classification: Classification) =>
      navigateLookup(routeCard(card, classification)),
  )
```

- [ ] **Step 2: Expose it to the renderer**

In `app/src/preload/index.ts`, add to the `exposeInMainWorld` object:

```typescript
  lookupCard: (
    card: {
      id: string
      name: string
      number: string | null
      setName: string
      tcgplayerUrl: string | null
    },
    classification: 'slab' | 'raw-single',
  ) => ipcRenderer.invoke('hitcheck:lookupCard', card, classification),
```

The shape is repeated inline rather than imported: the preload bundle is a
separate electron-vite build target from `main`, and importing across that
boundary pulls main-process code into the preload bundle.

- [ ] **Step 3: Typecheck and test**

Run: `cd app && npm run typecheck && npm test`
Expected: no type errors, all existing tests PASS.

- [ ] **Step 4: Commit**

```bash
git add app/src/main/ipc.ts app/src/preload/index.ts
git commit -m "feat(lookup): expose lookupCard over IPC"
```

---

### Task 8: Candidate switching in the overlay

**Files:**
- Create: `app/src/renderer/overlay/candidates.ts`
- Test: `app/src/renderer/overlay/candidates.test.ts`

**Interfaces:**
- Consumes: `CardRecord` shape (structurally, declared locally — the renderer
  is a separate build target from `main`, same reason as Task 7)
- Produces:
  - `type Candidate = { card: CardRecord; score: number }`
  - `formatCandidates(candidates: Candidate[], threshold: number): CandidateView`
    where `CandidateView = { primary: CardRecord; confidence: 'confident' | 'uncertain'; alternates: CardRecord[] }`

This is the pure part of the overlay's candidate UI: given the retrieval's
top-3 and a threshold, decide what the badge says and which cards become
re-navigation buttons. Rendering it into DOM is deferred to M6, when the
overlay gains real content beyond a box.

- [ ] **Step 1: Write the failing test**

Create `app/src/renderer/overlay/candidates.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { formatCandidates } from './candidates'
import type { Candidate } from './candidates'

const cand = (id: string, name: string, score: number): Candidate => ({
  card: { id, name, number: '4', setName: 'Base', tcgplayerUrl: null },
  score,
})

describe('formatCandidates', () => {
  it('promotes the highest-scoring candidate to primary', () => {
    const view = formatCandidates(
      [cand('a', 'Alakazam', 0.4), cand('b', 'Charizard', 0.9)], 0.7,
    )
    expect(view.primary.name).toBe('Charizard')
  })

  it('marks a score above the threshold as confident', () => {
    expect(formatCandidates([cand('b', 'Charizard', 0.9)], 0.7).confidence)
      .toBe('confident')
  })

  it('marks a score below the threshold as uncertain', () => {
    expect(formatCandidates([cand('b', 'Charizard', 0.5)], 0.7).confidence)
      .toBe('uncertain')
  })

  it('treats a score exactly at the threshold as confident', () => {
    expect(formatCandidates([cand('b', 'Charizard', 0.7)], 0.7).confidence)
      .toBe('confident')
  })

  it('offers the runners-up as alternates, best first', () => {
    const view = formatCandidates(
      [cand('a', 'Abra', 0.3), cand('c', 'Charizard', 0.9), cand('b', 'Blastoise', 0.6)],
      0.7,
    )
    expect(view.alternates.map(c => c.name)).toEqual(['Blastoise', 'Abra'])
  })

  it('caps alternates at two', () => {
    const view = formatCandidates(
      [cand('a', 'A', 0.9), cand('b', 'B', 0.8), cand('c', 'C', 0.7), cand('d', 'D', 0.6)],
      0.7,
    )
    expect(view.alternates).toHaveLength(2)
  })

  it('returns no alternates for a single candidate', () => {
    expect(formatCandidates([cand('b', 'Charizard', 0.9)], 0.7).alternates).toEqual([])
  })

  it('throws on an empty candidate list rather than inventing a primary', () => {
    expect(() => formatCandidates([], 0.7)).toThrow(/at least one candidate/)
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd app && npm test -- src/renderer/overlay/candidates.test.ts`
Expected: FAIL — cannot resolve `./candidates`

- [ ] **Step 3: Write the implementation**

Create `app/src/renderer/overlay/candidates.ts`:

```typescript
type CardRecord = {
  id: string
  name: string
  number: string | null
  setName: string
  tcgplayerUrl: string | null
}

export type Candidate = { card: CardRecord; score: number }

export type CandidateView = {
  primary: CardRecord
  confidence: 'confident' | 'uncertain'
  alternates: CardRecord[]
}

const MAX_ALTERNATES = 2

/**
 * Decide what the overlay shows for a set of retrieval candidates.
 *
 * The window navigates to the primary regardless of confidence. That is
 * deliberate and differs from the numeric price path: a wrong page is
 * self-evidently wrong and asserts nothing, whereas a wrong number costs money.
 * The badge tells the user how much to trust it; the alternates let them fix it
 * in one click when the model cannot separate near-identical reprints.
 */
export function formatCandidates(
  candidates: Candidate[],
  threshold: number,
): CandidateView {
  if (candidates.length === 0) {
    throw new Error('formatCandidates requires at least one candidate')
  }
  const ranked = [...candidates].sort((a, b) => b.score - a.score)
  const [best, ...rest] = ranked
  return {
    primary: best.card,
    confidence: best.score >= threshold ? 'confident' : 'uncertain',
    alternates: rest.slice(0, MAX_ALTERNATES).map(c => c.card),
  }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd app && npm test -- src/renderer/overlay/candidates.test.ts`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add app/src/renderer/overlay/candidates.ts app/src/renderer/overlay/candidates.test.ts
git commit -m "feat(overlay): candidate view with confidence badge and alternates"
```

---

### Task 9: Resolved-URL cache

**Files:**
- Create: `app/src/main/lookup/cache.ts`
- Test: `app/src/main/lookup/cache.test.ts`
- Modify: `app/src/main/lookup/window.ts`

**Interfaces:**
- Consumes: `stripTracking` from `./outcome`
- Produces:
  - `type ResolvedCache = { get(cardId: string): string | undefined; set(cardId: string, url: string): void; size(): number }`
  - `createResolvedCache(limit?: number): ResolvedCache`
  - `resolvedDestinations(cardId: string, destinations: Destination[], cache: ResolvedCache): Destination[]`

Motivation is in Task 1's findings: the pokemontcg.io redirector logs every
lookup with a third party and adds an affiliate credit. Caching the resolved
bare URL means it is hit at most once per card. Slab streams repeat cards
heavily, so this converts most lookups into a direct load.

- [ ] **Step 1: Write the failing test**

Create `app/src/main/lookup/cache.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { createResolvedCache, resolvedDestinations } from './cache'
import type { Destination } from './router'

const dest = (url: string, kind: Destination['kind'] = 'product'): Destination =>
  ({ url, kind })

describe('createResolvedCache', () => {
  it('returns undefined for an unseen card', () => {
    expect(createResolvedCache().get('base1-4')).toBeUndefined()
  })

  it('stores and returns a resolved URL', () => {
    const cache = createResolvedCache()
    cache.set('base1-4', 'https://www.tcgplayer.com/product/42382')
    expect(cache.get('base1-4')).toBe('https://www.tcgplayer.com/product/42382')
  })

  it('strips tracking parameters on the way in', () => {
    const cache = createResolvedCache()
    cache.set('base1-4', 'https://www.tcgplayer.com/product/42382?utm_campaign=Scrydex')
    expect(cache.get('base1-4')).toBe('https://www.tcgplayer.com/product/42382')
  })

  it('evicts the oldest entry past the limit', () => {
    const cache = createResolvedCache(2)
    cache.set('a', 'https://x/1')
    cache.set('b', 'https://x/2')
    cache.set('c', 'https://x/3')
    expect(cache.get('a')).toBeUndefined()
    expect(cache.get('c')).toBe('https://x/3')
    expect(cache.size()).toBe(2)
  })

  it('re-setting an existing card does not grow the cache', () => {
    const cache = createResolvedCache(2)
    cache.set('a', 'https://x/1')
    cache.set('a', 'https://x/2')
    expect(cache.size()).toBe(1)
    expect(cache.get('a')).toBe('https://x/2')
  })
})

describe('resolvedDestinations', () => {
  it('prepends the cached URL when one exists', () => {
    const cache = createResolvedCache()
    cache.set('base1-4', 'https://www.tcgplayer.com/product/42382')
    const out = resolvedDestinations(
      'base1-4', [dest('https://prices.pokemontcg.io/tcgplayer/base1-4')], cache,
    )
    expect(out.map(d => d.url)).toEqual([
      'https://www.tcgplayer.com/product/42382',
      'https://prices.pokemontcg.io/tcgplayer/base1-4',
    ])
  })

  it('returns the destinations untouched on a cache miss', () => {
    const given = [dest('https://prices.pokemontcg.io/tcgplayer/base1-4')]
    expect(resolvedDestinations('base1-4', given, createResolvedCache()))
      .toEqual(given)
  })

  it('does not mutate the destinations it was given', () => {
    const cache = createResolvedCache()
    cache.set('base1-4', 'https://www.tcgplayer.com/product/42382')
    const given = [dest('https://prices.pokemontcg.io/tcgplayer/base1-4')]
    resolvedDestinations('base1-4', given, cache)
    expect(given).toHaveLength(1)
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd app && npm test -- src/main/lookup/cache.test.ts`
Expected: FAIL — cannot resolve `./cache`

- [ ] **Step 3: Write the implementation**

Create `app/src/main/lookup/cache.ts`:

```typescript
import { stripTracking } from './outcome'
import type { Destination } from './router'

export type ResolvedCache = {
  get(cardId: string): string | undefined
  set(cardId: string, url: string): void
  size(): number
}

const DEFAULT_LIMIT = 2000

/**
 * card id -> the bare product URL that card actually resolved to.
 *
 * In-memory and bounded. A Map preserves insertion order, so the first key is
 * the oldest and eviction is a single shift. Deliberately not persisted: a
 * stale mapping to a delisted product would be worse than one extra redirect,
 * and a session's worth of caching already removes almost all repeat hits.
 */
export function createResolvedCache(limit: number = DEFAULT_LIMIT): ResolvedCache {
  const entries = new Map<string, string>()
  return {
    get: cardId => entries.get(cardId),
    set(cardId, url) {
      entries.delete(cardId)
      entries.set(cardId, stripTracking(url))
      if (entries.size > limit) {
        const oldest = entries.keys().next().value
        if (oldest !== undefined) entries.delete(oldest)
      }
    },
    size: () => entries.size,
  }
}

/** Put a previously-resolved URL at the front, keeping the rest as fallbacks. */
export function resolvedDestinations(
  cardId: string,
  destinations: Destination[],
  cache: ResolvedCache,
): Destination[] {
  const cached = cache.get(cardId)
  return cached ? [{ url: cached, kind: 'product' }, ...destinations] : destinations
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd app && npm test -- src/main/lookup/cache.test.ts`
Expected: PASS, 8 tests.

- [ ] **Step 5: Wire the cache into the window**

In `app/src/main/lookup/window.ts`, add the import:

```typescript
import { createResolvedCache, resolvedDestinations } from './cache'
```

Add a module-level cache beside the `lookup` binding:

```typescript
const resolved = createResolvedCache()
```

Change `navigateLookup` to take the card id and record what it landed on.
Replace the whole function with:

```typescript
export async function navigateLookup(
  cardId: string,
  destinations: Destination[],
): Promise<void> {
  if (!lookup || lookup.isDestroyed()) createLookupWindow()
  const win = lookup!

  for (const destination of resolvedDestinations(cardId, destinations, resolved)) {
    try {
      await win.loadURL(destination.url)
    } catch {
      continue
    }
    win.showInactive()
    const landed = win.webContents.getURL()
    if (classifyLanding(landed) === 'product') {
      resolved.set(cardId, landed)
      return
    }
  }
  win.showInactive()
}
```

- [ ] **Step 6: Update the caller and the smoke script**

In `app/src/main/ipc.ts`, pass the card id:

```typescript
      navigateLookup(card.id, routeCard(card, classification)),
```

In `app/scripts/lookup-smoke.mjs`, change each `navigateLookup(destinations)`
call to `navigateLookup(label, destinations)` — using the case label as a
stand-in card id keeps the three cases in separate cache entries.

- [ ] **Step 7: Verify the whole suite and the smoke script**

Run: `cd app && npm test && npm run typecheck`
Expected: all tests PASS, no type errors.

Run: `cd app && npm run build && npx electron scripts/lookup-smoke.mjs`
Expected: the same three lines as Task 6, except the `raw single` line now ends
at `https://www.tcgplayer.com/product/42382` with no tracking parameters —
because the first load resolved, was stripped, and was cached.

- [ ] **Step 8: Commit**

```bash
git add app/src/main/lookup/cache.ts app/src/main/lookup/cache.test.ts \
        app/src/main/lookup/window.ts app/src/main/ipc.ts app/scripts/lookup-smoke.mjs
git commit -m "feat(lookup): cache resolved product URLs, drop affiliate tracking"
```

---

## Done when

- `cd app && npm test` passes, including 51 new tests across six new test
  files (`slug`, `url`, `router`, `outcome`, `candidates`, `cache`).
- `cd app && npm run typecheck` is clean.
- `cd trainer && uv run pytest` passes.
- `data/catalog.sqlite` has `tcgplayer_url` populated for 20,213 cards.
- The smoke script shows a PriceCharting product page, a fall-through to our
  own set-aware search on a bad slug, and a tracking-free TCGplayer product URL.

## Deliberately not built

- **Sealed products.** M4.5, blocked on M6's OCR engine. Adding `'sealed'` to
  `Classification` is the signal that plan has begun.
- **The slab-vs-raw classifier head.** Trainer-side, its own plan. Until it
  lands, callers pass `'raw-single'` and slab routing is exercised only by the
  unit tests and the smoke script.
- **Rendering the candidate view into overlay DOM.** Task 8 builds and tests
  the decision; the overlay has no content surface beyond a box until M6.
- **Persisting the resolved-URL cache to disk.** In-memory only, for the reason
  given in `cache.ts`.
- **`PriceSource` and the numeric raw-single price.** The spec's M4 pairs the
  lookup window with a number in the overlay for raw singles. It is left out
  here for two reasons: the pokemontcg.io prices are already sitting in
  `raw_json` so no fetching work is required, and the overlay has no surface
  to render a number on until M6. It is a small, independent plan that can
  run in parallel with this one.
- **A learned `card_id → PriceCharting slug` table.** The spec defers this
  until the fallback rate is observable. `SET_SLUG_OVERRIDES` is where measured
  misses go in the meantime.
