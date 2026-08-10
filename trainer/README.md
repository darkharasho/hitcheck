# hitcheck-trainer

Python package that syncs the Pokémon card catalog, builds an embedding
index, and evaluates retrieval accuracy for HitCheck.

## Setup

```bash
cd trainer
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Testing

```bash
cd trainer
uv run pytest tests/ -v
```

## Why retries matter

The upstream card API (pokemontcg.io) was measured at a ~50% HTTP 500
failure rate on 2026-08-10. Retry and resumability are first-class design
concerns throughout this package, not defensive extras.

## Catalog sync

Pulls the full card catalog (metadata + images) to local disk under
`data/`. Copy `.env.example` to `.env` and optionally set
`POKEMONTCG_API_KEY` (raises the rate limit; the sync works without it).

```bash
cd trainer
uv run python -m hitcheck_trainer.catalog.cli sync
```

The upstream API fails often enough that a single run is not expected to
finish the whole catalog. The CLI checkpoints metadata progress page by
page and skips images already on disk, so **rerun the command until it
exits 0** — each run resumes from where the last one stopped instead of
starting over. A non-zero exit means metadata is short of the catalog
total, or one or more images failed to download; the printed summary
says which and by how much. See
`docs/verification/2026-08-10-m1-catalog.md` for a real run's results.
