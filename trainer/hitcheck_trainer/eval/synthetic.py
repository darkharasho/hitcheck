"""M2 gate: does zero-shot DINOv2 retrieval identify degraded cards?

Builds the index from clean catalog images, then queries it with the
same cards put through the stream-degradation pipeline. This is an
optimistic bound — real frames add framing and background noise this
does not simulate — so a poor result here is decisive, while a good one
still needs confirming against real labelled frames.
"""

import argparse
import json
import os
import sys

from ..augment.degrade import degrade
from ..catalog.db import all_card_images, open_db
from ..catalog.images import image_path
from ..index.build import build_index
from ..index.embed import Embedder
from ..index.query import CardIndex
from .chunks import embed_in_chunks
from .report import score

DEFAULT_DB = "data/catalog.sqlite"
DEFAULT_IMAGES = "data/images"
DEFAULT_INDEX = "data/index/cards.bin"


def available_ids(pairs, images_root):
    """Card ids whose image file actually exists on disk, in catalog order.

    Cheap — stats files, decodes nothing.
    """
    return [
        card_id
        for card_id, _ in pairs
        if os.path.exists(image_path(images_root, card_id))
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hitcheck-eval-synthetic")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--images", default=DEFAULT_IMAGES)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--sample", type=int, default=500, help="queries to evaluate")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--chunk", type=int, default=256,
                        help="images decoded at once; keep small, this bounds RAM")
    parser.add_argument("--reuse-index", action="store_true",
                        help="load an existing gallery index instead of re-embedding "
                             "all ~20k catalog images (skips the expensive step when "
                             "a valid index is already on disk)")
    args = parser.parse_args(argv)

    conn = open_db(args.db)
    pairs = all_card_images(conn)
    print(f"catalog: {len(pairs)} cards with images")

    disk_ids = available_ids(pairs, args.images)
    print(f"{len(disk_ids)} images present on disk")
    if not disk_ids:
        print("No images found. Run the catalog sync first.")
        return 1

    embedder = Embedder()

    sidecar_path = f"{args.index}.ids.json"
    if args.reuse_index and os.path.exists(args.index) and os.path.exists(sidecar_path):
        print(f"reusing existing gallery index at {args.index} (skipping gallery embed)")
        index = CardIndex.load(args.index, dim=embedder.dim)
        with open(sidecar_path) as fh:
            ids = json.load(fh)["ids"]
    else:
        print(f"embedding gallery on {embedder.device} (dim {embedder.dim})...")
        gallery_items = [(card_id, image_path(args.images, card_id)) for card_id in disk_ids]
        ids, gallery = embed_in_chunks(embedder, gallery_items, chunk=args.chunk)
        build_index(gallery, ids, args.index)
        index = CardIndex.load(args.index, dim=embedder.dim)
        del gallery

    step = max(1, len(ids) // args.sample)
    query_ids = ids[::step][: args.sample]

    print(f"degrading and embedding {len(query_ids)} queries (strength {args.strength})...")
    query_items = [(card_id, image_path(args.images, card_id)) for card_id in query_ids]
    query_ids, query_vectors = embed_in_chunks(
        embedder,
        query_items,
        chunk=args.chunk,
        transform=lambda img, i: degrade(img, seed=i, strength=args.strength),
    )

    results = [
        (card_id, index.query(vector, k=5))
        for card_id, vector in zip(query_ids, query_vectors)
    ]
    report = score(results)

    print()
    print(report.summary())
    print()
    print("Sample misses (true -> predicted):")
    for true_id, predicted in report.failures[:15]:
        print(f"  {true_id} -> {predicted or '(none)'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
