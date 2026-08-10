"""Approximate nearest-neighbour index over card embeddings.

Identification is retrieval, not classification — there is no 20k-class
model, just a nearest-neighbour lookup in embedding space.
"""

import hashlib
import json
import os

import hnswlib
import numpy as np

# ASCII unit separator between hashed ids — not a character card ids use.
_ID_SEPARATOR = b"\x1f"


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows so cosine distance behaves. Zero rows stay zero."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def fingerprint(ids: list[str], dim: int, count: int, vectors: np.ndarray) -> str:
    """Content hash tying a `.ids.json` sidecar to one specific index build.

    hnswlib's `load_index(path, max_elements=N)` treats `max_elements` as a
    growth ceiling, not validation — a sidecar with the wrong id count (or
    the right count but from an unrelated build) loads without complaint,
    and every subsequent query silently returns the wrong card. This hash
    covers dimension, count, exact id order, and the vectors themselves, so
    any of those drifting apart is caught on load instead of surfacing later
    as mysteriously bad accuracy.

    Vectors must be hnswlib's own stored representation (via
    `index.get_items(...)`), not our own `normalize()` output — hnswlib
    renormalizes internally for cosine space and the two are only equal to
    float32 rounding tolerance, not bit-for-bit. Hashing hnswlib's own
    values means the build-time and load-time hash always agree exactly.
    """
    hasher = hashlib.sha256()
    hasher.update(f"{dim}:{count}:".encode())
    for card_id in ids:
        hasher.update(card_id.encode())
        hasher.update(_ID_SEPARATOR)
    hasher.update(np.ascontiguousarray(vectors, dtype=np.float32).tobytes())
    return hasher.hexdigest()


def build_index(vectors: np.ndarray, ids: list[str], path: str) -> None:
    if len(ids) != len(vectors):
        raise ValueError(f"got {len(vectors)} vectors but {len(ids)} ids")

    count, dim = vectors.shape
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=count, ef_construction=200, M=32)
    index.add_items(normalize(vectors), np.arange(count))
    index.set_ef(64)

    # Hash what hnswlib actually stored (see fingerprint()'s docstring for
    # why this must come from get_items rather than our own normalize()).
    stored = np.array(index.get_items(list(range(count))), dtype=np.float32)
    sidecar = {"ids": ids, "fingerprint": fingerprint(ids, dim, count, stored)}

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Write via temp file + atomic rename, same pattern as
    # catalog/images.py's `.part` downloads: a reader never observes a
    # half-written .bin or sidecar at the canonical path. This can't make
    # the two-file rename atomic as a pair — a crash between the two
    # os.replace calls can still leave a fresh index next to a stale
    # sidecar — but CardIndex.load()'s fingerprint check catches that case
    # and refuses to load rather than silently mis-mapping ids.
    index_tmp = f"{path}.part"
    index.save_index(index_tmp)
    os.replace(index_tmp, path)

    sidecar_path = f"{path}.ids.json"
    sidecar_tmp = f"{sidecar_path}.part"
    with open(sidecar_tmp, "w") as fh:
        json.dump(sidecar, fh)
    os.replace(sidecar_tmp, sidecar_path)
