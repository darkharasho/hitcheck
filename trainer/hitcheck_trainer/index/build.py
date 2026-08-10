"""Approximate nearest-neighbour index over card embeddings.

Identification is retrieval, not classification — there is no 20k-class
model, just a nearest-neighbour lookup in embedding space.
"""

import json
import os

import hnswlib
import numpy as np


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows so cosine distance behaves. Zero rows stay zero."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def build_index(vectors: np.ndarray, ids: list[str], path: str) -> None:
    if len(ids) != len(vectors):
        raise ValueError(f"got {len(vectors)} vectors but {len(ids)} ids")

    count, dim = vectors.shape
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=count, ef_construction=200, M=32)
    index.add_items(normalize(vectors), np.arange(count))
    index.set_ef(64)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    index.save_index(path)
    with open(f"{path}.ids.json", "w") as fh:
        json.dump(ids, fh)
