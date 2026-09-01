"""Push the corpus to the hosted crop tool, and pull hand-marked quads back.

The hosted tool at crop.axi.link exists to put more than one pair of hands
on the hand-crop pass. It is a work queue, not an authority: every quad it
collects is re-run through validate_quad here before it is allowed into
crops.json, so the contract that decides whether a crop is mirrored has
exactly one implementation that counts.

The other thing this module does is measure the croppers. Everyone marks
the same handful of reference cards first, and `pull` prints how far each
person's corners landed from the reference. That number is the only warning
available for the failure that matters -- someone marking the slab instead
of the card, or starting from the wrong corner -- because such a crop is
geometrically valid and merely shifts the M2 accuracy figure.

Credentials come from the environment and are never printed:
  CROPTOOL_ADMIN_TOKEN                     -- the worker's ADMIN_TOKEN secret
  CF_ACCESS_CLIENT_ID / _SECRET (optional) -- an Access service token, if the
                                              Access policy covers /api/admin
"""

import argparse
import os
import sys

import httpx
import numpy as np

from .crops import load_crops, load_skips, save_crops, save_skips, validate_quad
from .manifest import load_manifest

DEFAULT_CORPUS = "data/corpus"
DEFAULT_URL = "https://crop.axi.link"
ITEM_BATCH = 100
# Above this, a cropper's corners are far enough from the reference that
# their pass should be looked at before it is trusted. Expressed as a
# fraction of the reference card's diagonal so it is scale-free: 5% of a
# card's diagonal is roughly a corner sitting on the slab edge rather than
# on the card.
DISAGREEMENT_WARN = 0.05


def corner_disagreement(reference, candidate) -> float:
    """Mean corner distance between two quads, as a fraction of the diagonal.

    Compared corner-for-corner rather than as sets: two quads covering the
    same card with the corners started from different places are exactly
    the mistake this is looking for, and set comparison would call them
    identical.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    diagonal = float(np.linalg.norm(ref[2] - ref[0]))
    if diagonal == 0:
        return float("inf")
    return float(np.linalg.norm(cand - ref, axis=1).mean() / diagonal)


def _client(url: str) -> httpx.Client:
    token = os.environ.get("CROPTOOL_ADMIN_TOKEN")
    if not token:
        raise SystemExit("CROPTOOL_ADMIN_TOKEN is not set")
    headers = {"Authorization": f"Bearer {token}"}
    access_id = os.environ.get("CF_ACCESS_CLIENT_ID")
    access_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET")
    if access_id and access_secret:
        headers["CF-Access-Client-Id"] = access_id
        headers["CF-Access-Client-Secret"] = access_secret
    return httpx.Client(base_url=url, headers=headers, timeout=60.0)


def push(corpus_dir: str, url: str) -> int:
    manifest = load_manifest(os.path.join(corpus_dir, "manifest.json"))
    if not manifest.entries:
        print(f"No manifest entries under {corpus_dir}. Run the corpus build first.")
        return 1

    # Cards already marked locally become the calibration set: they are the
    # only quads whose provenance is known, which is what makes them usable
    # as a reference for everybody else.
    reference = load_crops(os.path.join(corpus_dir, "crops.json"))
    if not reference:
        print("No local crops yet — mark a few cards with the desktop tool first;")
        print("they become the reference every other cropper is measured against.")
        return 1

    items = [
        {
            "item_id": entry.item_id,
            "card_id": entry.card_id,
            "image": entry.image,
            "calibration": entry.item_id in reference,
        }
        for entry in manifest.entries
    ]

    with _client(url) as client:
        for start in range(0, len(items), ITEM_BATCH):
            batch = items[start : start + ITEM_BATCH]
            response = client.post("/api/admin/items", json=batch)
            response.raise_for_status()
            print(f"items: {min(start + ITEM_BATCH, len(items))}/{len(items)}")

        uploaded = skipped = 0
        for entry in manifest.entries:
            path = os.path.join(corpus_dir, entry.image)
            if not os.path.exists(path):
                continue
            # Ask before sending: a re-push after the next acquisition batch
            # should cost one small call per known photograph, not eighty
            # megabytes of re-upload.
            present = client.get("/api/admin/image", params={"key": entry.image})
            present.raise_for_status()
            if present.json().get("present"):
                skipped += 1
                continue
            with open(path, "rb") as fh:
                response = client.put(
                    "/api/admin/image",
                    params={"key": entry.image},
                    content=fh.read(),
                    headers={"content-type": "image/jpeg"},
                )
            response.raise_for_status()
            uploaded += 1
            if uploaded % 25 == 0:
                print(f"photographs: {uploaded} uploaded, {skipped} already there")

    print(f"pushed {len(items)} items ({sum(i['calibration'] for i in items)} calibration), "
          f"{uploaded} photographs uploaded, {skipped} already present")
    return 0


def calibration_errors(remote_crops, calibration_ids, reference):
    """cropper -> list of per-card disagreements against the reference."""
    by_cropper: dict[str, list[float]] = {}
    for crop in remote_crops:
        if crop["item_id"] not in calibration_ids or crop["item_id"] not in reference:
            continue
        by_cropper.setdefault(crop["cropper"], []).append(
            corner_disagreement(reference[crop["item_id"]], crop["quad"])
        )
    return by_cropper


def merge_remote_crops(payload, reference):
    """Fold the worker's quads into the local crops, refusing invalid ones.

    Returns (merged, rejected, duplicates). Every remote quad goes through
    validate_quad here: the worker's JavaScript check is for fast feedback,
    and this is the gate that decides what the M2 number is computed from.
    """
    calibration_ids = set(payload["calibration"])

    # Earliest wins: a second quad on the same corpus card means a lease
    # expired while someone was still working, not a correction.
    best: dict[str, dict] = {}
    corpus_crops = [c for c in payload["crops"] if c["item_id"] not in calibration_ids]
    for crop in corpus_crops:
        current = best.get(crop["item_id"])
        if current is None or crop["at"] < current["at"]:
            best[crop["item_id"]] = crop

    merged, rejected = dict(reference), []
    for item_id, crop in sorted(best.items()):
        # Local crops win: they are the references, and re-marking one
        # remotely must not quietly redefine what everyone is measured on.
        if item_id in reference:
            continue
        try:
            validate_quad(crop["quad"])
        except ValueError as exc:
            rejected.append((item_id, crop["cropper"], str(exc)))
            continue
        merged[item_id] = [[float(x), float(y)] for x, y in crop["quad"]]

    return merged, rejected, len(corpus_crops) - len(best)


def _report_calibration(remote_crops, calibration_ids, reference) -> None:
    by_cropper = calibration_errors(remote_crops, calibration_ids, reference)
    if not by_cropper:
        print("\nNo calibration crops yet.")
        return

    print("\nCalibration — mean corner distance from the reference, as a")
    print("fraction of the card's diagonal. Look at anything over "
          f"{DISAGREEMENT_WARN:.0%}:")
    for cropper, errors in sorted(by_cropper.items()):
        worst = max(errors)
        mean = sum(errors) / len(errors)
        flag = "  <-- CHECK THIS PASS" if worst > DISAGREEMENT_WARN else ""
        print(f"  {cropper:<40} n={len(errors):<3} mean={mean:.1%} worst={worst:.1%}{flag}")


def pull(corpus_dir: str, url: str) -> int:
    crops_path = os.path.join(corpus_dir, "crops.json")
    skips_path = os.path.join(corpus_dir, "skipped.json")
    reference = load_crops(crops_path)

    with _client(url) as client:
        response = client.get("/api/admin/crops")
        response.raise_for_status()
        payload = response.json()

    calibration_ids = set(payload["calibration"])
    _report_calibration(payload["crops"], calibration_ids, reference)
    merged, rejected, duplicates = merge_remote_crops(payload, reference)

    skips = load_skips(skips_path)
    skips.update(s["item_id"] for s in payload["skips"] if s["item_id"] not in calibration_ids)

    save_crops(merged, crops_path)
    save_skips(skips, skips_path)

    print(f"\ncrops.json: {len(reference)} -> {len(merged)}")
    print(f"skipped.json: {len(skips)} items")
    if duplicates:
        print(f"{duplicates} cards were marked by more than one person (lease expiry); "
              "kept the earliest")
    if rejected:
        print(f"\n{len(rejected)} quads failed validation and were NOT merged:")
        for item_id, cropper, reason in rejected:
            print(f"  {item_id} ({cropper}): {reason}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hitcheck-corpus-sync")
    parser.add_argument("command", choices=["push", "pull"])
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args(argv)
    return push(args.corpus, args.url) if args.command == "push" else pull(args.corpus, args.url)


if __name__ == "__main__":
    sys.exit(main())
