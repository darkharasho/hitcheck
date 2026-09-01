"""The M2 corpus's checked-in record of what it captured and what it dropped.

Listings expire, so images and entries are written once and never
re-fetched; the eval reads this file and the images beside it, and
nothing else. That is what makes the number reproducible after the
listings are gone.

This file is tracked in git (its images are not -- they are sellers'
copyrighted photographs, used locally and not redistributed), so it is
written sorted and indented for readable diffs.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")


def safe_item_id(item_id: str) -> str:
    """Filename-safe form of an eBay itemId.

    Browse returns ids like 'v1|364012345678|0'; the pipes cannot go in a
    path. The original id stays in the manifest as provenance -- never
    reconstruct one form from the other.
    """
    return _UNSAFE.sub("_", item_id)


def image_relpath(item_id: str) -> str:
    """Image path relative to the manifest's own directory."""
    return f"images/{safe_item_id(item_id)}.jpg"


@dataclass(frozen=True)
class CorpusEntry:
    item_id: str
    card_id: str
    image: str
    image_url: str
    listing_url: str
    aspects: dict[str, str]


@dataclass
class Manifest:
    entries: list[CorpusEntry] = field(default_factory=list)
    discards: dict[str, int] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)

    def item_ids(self) -> set[str]:
        return {e.item_id for e in self.entries}

    def yield_summary(self) -> str:
        """One line stating the corpus's own yield.

        A corpus that silently dropped most of its candidates would skew
        toward whichever listings happen to have tidy Item Specifics, so
        the discard breakdown travels with the manifest rather than
        living only in a console log.
        """
        total_discarded = sum(self.discards.values())
        breakdown = " ".join(
            f"{reason}={count}" for reason, count in sorted(self.discards.items())
        )
        return f"kept={len(self.entries)} discarded={total_discarded} {breakdown}".rstrip()


def save_manifest(manifest: Manifest, path: str) -> None:
    payload = {
        "entries": [asdict(e) for e in manifest.entries],
        "discards": manifest.discards,
        "queries": manifest.queries,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.part"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)  # atomic — never a half-written manifest


def load_manifest(path: str) -> Manifest:
    """Load a manifest, or an empty one if it does not exist yet."""
    if not os.path.exists(path):
        return Manifest()
    with open(path) as fh:
        payload = json.load(fh)
    return Manifest(
        entries=[CorpusEntry(**e) for e in payload.get("entries", [])],
        discards=payload.get("discards", {}),
        queries=payload.get("queries", []),
    )
