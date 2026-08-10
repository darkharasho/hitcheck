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
