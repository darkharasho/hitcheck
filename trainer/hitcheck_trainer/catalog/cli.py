"""Catalog sync entry point.

Run repeatedly. The upstream API fails often; each run resumes from the
last checkpoint and re-attempts whatever is still missing.
"""

import argparse
import os
import sys

from .api import CatalogApi, CatalogApiError
from .db import all_card_images, card_count, open_db
from .http import httpx_fetch, httpx_transport
from .images import download_images
from .sync import SyncIncompleteError, sync_catalog

DEFAULT_DB = "data/catalog.sqlite"
DEFAULT_IMAGES = "data/images"


def _progress(label: str):
    def report(done: int, total: int):
        pct = (done / total * 100) if total else 0.0
        print(f"\r{label}: {done}/{total} ({pct:.1f}%)", end="", flush=True)

    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hitcheck-catalog")
    parser.add_argument("command", choices=["sync"])
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--images", default=DEFAULT_IMAGES)
    args = parser.parse_args(argv)

    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    conn = open_db(args.db)
    api = CatalogApi(httpx_transport(), api_key=os.environ.get("POKEMONTCG_API_KEY"))

    print("Syncing card metadata...")
    try:
        total = sync_catalog(api, conn, on_progress=_progress("cards"))
        print(f"\nMetadata complete: {total} cards")
    except SyncIncompleteError as exc:
        print(f"\nMetadata sync incomplete (truncated read): {exc}")
        print(f"Progress saved ({card_count(conn)} cards). Rerun to resume.")
        return 1
    except CatalogApiError as exc:
        print(f"\nMetadata sync interrupted (retries exhausted): {exc}")
        print(f"Progress saved ({card_count(conn)} cards). Rerun to resume.")
        return 1

    print("Downloading card images...")
    pairs = all_card_images(conn)
    got, skipped = download_images(pairs, args.images, httpx_fetch(), on_progress=_progress("images"))
    failed = len(pairs) - got - skipped
    print(f"\nImages: {got} downloaded, {skipped} already present, {failed} failed")

    exit_code = 0
    if failed > 0:
        print(
            f"*** INCOMPLETE: {failed} image(s) failed to download. "
            "Rerun the sync to retry them. ***"
        )
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
