"""M2 gate, measured on real photographs instead of a synthetic axis.

synthetic.py queries the gallery with degraded copies of the gallery's
own images -- identical source pixels, lighting and crop -- which is why
it scores 1.000 at strength 0.0 and why no value of `strength` models
matching a DIFFERENT photograph of the same card. This module closes that
gap: the queries are eBay seller photographs, hand-cropped, labelled from
structured Item Specifics.

The gallery is the same 20,427-image index the synthetic run used, so any
difference in the number is attributable to the queries and nothing else.

Two caveats belong in every write-up of this number. It measures retrieval
GIVEN A GOOD CROP, because M3's detector does not exist yet and the corpus
is cropped by hand. And seller photographs are well-lit, static and
high-resolution -- meaningfully easier than a compressed handheld stream
frame -- so the result is an upper bound.
"""

import argparse
import os
import sys

from ..catalog.db import all_card_images, open_db
from ..catalog.images import image_path
from ..corpus.crops import apply_quad, load_crops
from ..corpus.manifest import load_manifest
from ..index.build import build_index
from ..index.embed import Embedder
from ..index.query import CardIndex
from .chunks import embed_in_chunks
from .report import label_noise_bound, score
from .synthetic import available_ids

DEFAULT_DB = "data/catalog.sqlite"
DEFAULT_IMAGES = "data/images"
DEFAULT_INDEX = "data/index/cards.bin"
DEFAULT_CORPUS = "data/corpus"

MIN_QUERIES = 500


def corpus_queries(manifest, crops, corpus_dir):
    """(label, path) items and their quads, index-aligned.

    Built in one pass on purpose: filtering one list without the other
    would crop every later photograph with its neighbour's quad and
    silently corrupt the eval rather than crash it.

    An entry with no crop is skipped -- a partial crops.json is the normal
    state during an incremental hand-crop pass. An entry whose image is
    missing from disk is skipped too; main() prints both counts so a short
    run is visible instead of quietly shrinking N.
    """
    items, quads = [], []
    for entry in manifest.entries:
        quad = crops.get(entry.item_id)
        if quad is None:
            continue
        path = os.path.join(corpus_dir, entry.image)
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            continue
        items.append((entry.card_id, path))
        quads.append(quad)
    return items, quads


def run_eval(embedder, index, items, quads, chunk=256):
    """Crop, embed and query. Returns (true_id, ranked) pairs for score()."""
    labels, vectors = embed_in_chunks(
        embedder, items, chunk=chunk,
        transform=lambda img, i: apply_quad(img, quads[i]),
    )
    return [(label, index.query(vector, k=5)) for label, vector in zip(labels, vectors)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hitcheck-eval-real")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--images", default=DEFAULT_IMAGES)
    parser.add_argument("--index", default=DEFAULT_INDEX)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--chunk", type=int, default=256,
                        help="images decoded at once; keep small, this bounds RAM")
    parser.add_argument("--reuse-index", action="store_true",
                        help="load the existing gallery index instead of re-embedding "
                             "all ~20k catalog images")
    parser.add_argument("--label-errors", type=int, default=None,
                        help="wrong labels found by the hand audit (see corpus.audit)")
    parser.add_argument("--label-sample", type=int, default=None,
                        help="entries hand-audited")
    args = parser.parse_args(argv)

    manifest = load_manifest(os.path.join(args.corpus, "manifest.json"))
    crops = load_crops(os.path.join(args.corpus, "crops.json"))
    items, quads = corpus_queries(manifest, crops, args.corpus)
    print(f"manifest: {len(manifest.entries)} entries, {len(crops)} cropped, "
          f"{len(items)} usable queries")
    print(manifest.yield_summary())
    if not items:
        print("No cropped corpus entries. Run the corpus build, then the crop tool.")
        return 1

    embedder = Embedder()
    sidecar_path = f"{args.index}.ids.json"
    if args.reuse_index and os.path.exists(args.index) and os.path.exists(sidecar_path):
        print(f"reusing existing gallery index at {args.index}")
        index = CardIndex.load(args.index, dim=embedder.dim)
    else:
        conn = open_db(args.db)
        disk_ids = available_ids(all_card_images(conn), args.images)
        print(f"embedding gallery of {len(disk_ids)} images on {embedder.device}...")
        gallery_items = [(card_id, image_path(args.images, card_id)) for card_id in disk_ids]
        ids, gallery = embed_in_chunks(embedder, gallery_items, chunk=args.chunk)
        build_index(gallery, ids, args.index)
        index = CardIndex.load(args.index, dim=embedder.dim)
        del gallery

    print(f"cropping and embedding {len(items)} real queries...")
    report = score(run_eval(embedder, index, items, quads, chunk=args.chunk))

    print()
    print(report.summary())
    if args.label_sample:
        bound = label_noise_bound(args.label_errors or 0, args.label_sample)
        print(f"label error <= {bound:.1%} (95% bound from {args.label_errors or 0}"
              f"/{args.label_sample} audited) — measured top1 understates true top1 "
              "by at most this much")
    else:
        print("label error UNBOUNDED — run corpus.audit and pass --label-errors/"
              "--label-sample before quoting this number")
    print("Measured GIVEN A GOOD CROP (hand-cropped; M3's detector does not exist "
          "yet) and on seller photographs, which are easier than stream frames. "
          "This is an upper bound.")

    if report.total < MIN_QUERIES:
        print(f"\n*** {report.total} queries is below {MIN_QUERIES}. Samples this small "
              "land in the inconclusive band as a matter of arithmetic; crop more "
              "before treating the verdict as an answer. ***")
        return 1

    print()
    print("Sample misses (true -> predicted):")
    for true_id, predicted in report.failures[:15]:
        print(f"  {true_id} -> {predicted or '(none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
