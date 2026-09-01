# Hosted crop tool

The trainer's desktop crop tool, opened to a handful of named people so the
hand-crop pass has more than one pair of hands on it.

The corpus needs several hundred hand-marked cards before `eval/real.py`
will return a verdict, and marking a card is irreducibly manual. Sharing it
is the only way to go faster.

**These crops are the ground truth the M2 train/don't-train number is
computed from.** A quietly wrong crop — the slab instead of the card, or
corner 1 on the wrong corner — does not fail anything. It moves the number.
That is what the calibration set and the pull-time revalidation are for.

## What lives where

| | |
|---|---|
| `src/index.js`  | routing, the work queue, admin endpoints |
| `src/page.js`   | the crop page, ported from `trainer/…/croptool.py` |
| `src/quad.js`   | fast feedback for the cropper — **not** the authority |
| `src/access.js` | Cloudflare Access identity, signature-verified |
| `schema.sql`    | D1: items, crops, skips |
| `quad-cases.json` | truth table both validators are tested against |

The gate that decides what enters `crops.json` is `validate_quad` in the
trainer, re-run over every quad at pull time. `src/quad.js` exists so a
cropper hears about a bad quad while the card is still on screen. Both are
pinned to `quad-cases.json` by their own suites, so the two cannot drift.

## First deploy

```bash
cd workers/croptool
npm install

npx wrangler d1 create hitcheck-crops      # paste database_id into wrangler.jsonc
npx wrangler r2 bucket create hitcheck-corpus
npm run schema                             # applies schema.sql to the remote D1
```

Then create the Access application, in **Zero Trust → Access →
Applications → Add → Self-hosted**:

- Application domain: `crop.axi.link`
- Policy: *Allow*, selector **Emails**, listing each cropper

Copy two values into `wrangler.jsonc`:

- `ACCESS_AUD` — the application's **Application Audience (AUD) Tag**
- `ACCESS_TEAM_DOMAIN` — your `<team>.cloudflareaccess.com`

Both are mandatory. The worker refuses to serve a photograph without them
rather than falling back to open — an unset Access config does not fail
closed on its own, it just means nobody is checked.

Finally, the script credential and the deploy:

```bash
npx wrangler secret put ADMIN_TOKEN        # any long random string
npm run deploy
```

`/api/admin/*` authenticates with that bearer token, not with Access. If
the Access policy covers those paths too, either add a **Bypass** policy for
`/api/admin/*` or create a **service token** and export
`CF_ACCESS_CLIENT_ID` / `CF_ACCESS_CLIENT_SECRET` alongside the push.

## Filling and draining it

From `trainer/`, with `CROPTOOL_ADMIN_TOKEN` in the environment:

```bash
.venv/bin/python -m hitcheck_trainer.corpus.sync push
```

Push uploads the photographs to R2 and seeds the queue. **Cards you have
already marked locally become the calibration set** — they are the only
quads whose provenance is known, so they are the reference everyone else is
measured against. Mark a handful with the desktop tool before the first
push; `push` refuses to run with none.

Croppers open <https://crop.axi.link/>. Everyone marks the calibration
cards first (`CALIBRATION_N`, default 5), then gets corpus cards on a lease
(`CLAIM_MINUTES`, default 10) so nobody duplicates anybody's work.

```bash
.venv/bin/python -m hitcheck_trainer.corpus.sync pull
```

Pull prints each cropper's mean corner distance from the reference, as a
fraction of the card's diagonal, and flags anything over 5% — roughly a
corner sitting on the slab edge rather than on the card. **Read that table
before trusting a pass.** It is the only warning available for a cropper who
has misunderstood the job, because their crops are geometrically valid.

Then it revalidates every corpus quad through `validate_quad` and merges the
survivors into `crops.json`. Rejections are printed with the cropper named,
so one person's pass can be withdrawn without redoing everyone else's. Local
crops always win: a remote re-mark must not redefine the yardstick.

## Running the tests

```bash
npm test
```

The queue's SQL runs against real SQLite via `node:sqlite` (`test/d1.js`), so
the claim statement is tested without a deploy. `test/d1.js` mirrors D1's
`bind()`-returns-a-new-statement semantics on purpose: a shim that mutated
in place would make the batched upsert look fine while writing the last row
repeatedly.

## Costs and caveats

Free tier throughout at this size — 161 photographs is ~80MB against R2's
10GB, and a few hundred crops is nothing to D1.

This puts eBay sellers' listing photographs on a domain you control. Access
keeps them behind a login for named people, which is what makes that
defensible; do not remove it to "make sharing easier".
