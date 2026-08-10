import json

import hnswlib
import numpy as np

from .build import normalize


class CardIndex:
    def __init__(self, index: "hnswlib.Index", ids: list[str]):
        self._index = index
        self._ids = ids

    @classmethod
    def load(cls, path: str, dim: int) -> "CardIndex":
        index = hnswlib.Index(space="cosine", dim=dim)
        with open(f"{path}.ids.json") as fh:
            ids = json.load(fh)
        index.load_index(path, max_elements=len(ids))
        index.set_ef(64)
        return cls(index, ids)

    def query(self, vector: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        k = min(k, len(self._ids))
        query_vec = normalize(np.asarray(vector, dtype=np.float32)[None])
        labels, distances = self._index.knn_query(query_vec, k=k)
        return [(self._ids[int(i)], float(d)) for i, d in zip(labels[0], distances[0])]
