"""Side-by-side sheet for hand-verifying resolved corpus labels.

Resolution is the accuracy-contaminating step: a mis-resolved label
surfaces as a retrieval miss that is not one. So a random sample gets
eyeballed -- hand-cropped photograph beside the catalog scan it was
resolved to -- and the resulting error count becomes a bound reported
alongside the accuracy. An unbounded label-error rate sitting under the
M2 verdict would make the number unusable for the decision it settles.

Sampling is seeded. An audit that resampled every run could be repeated
until it produced a flattering count. The candidate list is sorted by
item_id before sampling, so the sample depends only on the seed and the
set of cropped entries -- never on manifest list order or crops.json
dict/key order, either of which could otherwise be reshuffled to
resample under cover of "the seed didn't change".
"""

import argparse
import html
import os
import random
import sys

from PIL import Image

from ..catalog.images import image_path
from .crops import apply_quad, load_crops
from .manifest import load_manifest, safe_item_id

DEFAULT_CORPUS = "data/corpus"
DEFAULT_IMAGES = "data/images"


def sample_entries(manifest, crops, count=50, seed=0):
    """A reproducible random sample of cropped entries.

    Sorted by item_id before sampling so the result depends only on the
    seed and the set of cropped entries -- never on manifest order or
    crops dict iteration order.
    """
    cropped = sorted((e for e in manifest.entries if e.item_id in crops), key=lambda e: e.item_id)
    if len(cropped) <= count:
        return cropped
    return random.Random(seed).sample(cropped, count)


def build_audit(manifest, crops, corpus_dir, images_root, out_dir, count=50, seed=0) -> str:
    """Write an HTML sheet plus cropped previews. Returns the sheet's path."""
    crops_dir = os.path.join(out_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    rows = []
    for entry in sample_entries(manifest, crops, count, seed):
        photo_path = os.path.join(corpus_dir, entry.image)
        catalog_path = image_path(images_root, entry.card_id)
        try:
            with Image.open(photo_path) as photo:
                cropped = apply_quad(photo, crops[entry.item_id])
        except (OSError, ValueError):
            continue  # a missing or unreadable photo is skipped, not fatal
        preview = os.path.join(crops_dir, f"{safe_item_id(entry.item_id)}.png")
        cropped.save(preview)

        aspects = ", ".join(f"{k}: {v}" for k, v in sorted(entry.aspects.items()))
        rows.append(
            f'<tr><td><img src="{html.escape(os.path.relpath(preview, out_dir))}"></td>'
            f'<td><img src="{html.escape(os.path.relpath(catalog_path, out_dir))}"></td>'
            f"<td><strong>{html.escape(entry.card_id)}</strong><br>"
            f"{html.escape(aspects)}<br>"
            f'<a href="{html.escape(entry.listing_url)}">listing</a></td></tr>'
        )

    sheet = (
        "<!doctype html><meta charset='utf-8'><title>HitCheck label audit</title>"
        "<style>body{font:14px system-ui;background:#111;color:#eee}"
        "img{height:240px;background:#fff}td{padding:8px;vertical-align:top}"
        "tr{border-bottom:1px solid #333}</style>"
        f"<h1>Label audit — {len(rows)} entries</h1>"
        "<p>Left: hand-cropped photograph. Right: the catalog scan it resolved to. "
        "Count the pairs that are not the same card, then run:</p>"
        f"<pre>python -m hitcheck_trainer.eval.real --reuse-index "
        f"--label-errors N --label-sample {len(rows)}</pre>"
        "<table>" + "".join(rows) + "</table>"
    )
    out_path = os.path.join(out_dir, "audit.html")
    with open(out_path, "w") as fh:
        fh.write(sheet)
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hitcheck-corpus-audit")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--images", default=DEFAULT_IMAGES)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    manifest = load_manifest(os.path.join(args.corpus, "manifest.json"))
    crops = load_crops(os.path.join(args.corpus, "crops.json"))
    if not crops:
        print(f"No crops under {args.corpus}. Run the crop tool first.")
        return 1

    out_dir = os.path.join(args.corpus, "audit")
    path = build_audit(manifest, crops, args.corpus, args.images, out_dir,
                       args.count, args.seed)
    print(f"open file://{os.path.abspath(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
