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

import numpy as np
from PIL import Image

from ..augment.degrade import degrade
from ..catalog.db import all_card_images, open_db
from ..catalog.images import image_path
from ..index.build import build_index
from ..index.embed import Embedder
from ..index.query import CardIndex
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


def load_chunk(card_ids, images_root):
    """Decode one chunk of images. Returns (ids, images) for those that opened."""
    ids, images = [], []
    for card_id in card_ids:
        try:
            with Image.open(image_path(images_root, card_id)) as img:
                images.append(img.convert("RGB").copy())
            ids.append(card_id)
        except OSError:
            continue  # truncated file; a catalog rerun replaces it
    return ids, images


def embed_in_chunks(embedder, card_ids, images_root, chunk=256, transform=None):
    """Embed images a chunk at a time, holding only `chunk` decoded at once.

    NEVER materialise the whole catalog. 20,427 images at 240x330 RGB is
    4.52GB of decoded pixels before PIL overhead, degraded copies and torch
    tensors — that allocation contributed to a global OOM on a 30GB machine
    that also runs games and browsers. Chunked at 256 the resident set is
    ~58MB. `transform` optionally degrades each image before embedding.
    """
    kept_ids, vectors = [], []
    for start in range(0, len(card_ids), chunk):
        ids, images = load_chunk(card_ids[start : start + chunk], images_root)
        if not ids:
            continue
        if transform is not None:
            images = [transform(img, i) for i, img in enumerate(images, start)]
        vectors.append(embedder.embed(images, batch_size=64))
        kept_ids.extend(ids)
        del images  # drop decoded pixels before the next chunk
    return kept_ids, np.concatenate(vectors, axis=0) if vectors else np.zeros((0, embedder.dim), dtype=np.float32)


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
        ids, gallery = embed_in_chunks(embedder, disk_ids, args.images, chunk=args.chunk)
        build_index(gallery, ids, args.index)
        index = CardIndex.load(args.index, dim=embedder.dim)
        del gallery

    step = max(1, len(ids) // args.sample)
    query_ids = ids[::step][: args.sample]

    print(f"degrading and embedding {len(query_ids)} queries (strength {args.strength})...")
    query_ids, query_vectors = embed_in_chunks(
        embedder,
        query_ids,
        args.images,
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
