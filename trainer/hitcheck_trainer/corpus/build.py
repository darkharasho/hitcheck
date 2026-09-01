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
