import json

import hnswlib
import numpy as np

from .build import fingerprint, normalize


class CardIndex:
    def __init__(self, index: "hnswlib.Index", ids: list[str]):
        self._index = index
        self._ids = ids

    @classmethod
    def load(cls, path: str, dim: int) -> "CardIndex":
        sidecar_path = f"{path}.ids.json"
        with open(sidecar_path) as fh:
            sidecar = json.load(fh)
        ids = sidecar["ids"]

        index = hnswlib.Index(space="cosine", dim=dim)
        index.load_index(path, max_elements=len(ids))
        index.set_ef(64)

        # load_index's max_elements is only a growth ceiling, not
        # validation — it silently accepts a value smaller or larger than
        # what's actually stored. get_current_count() reports the real
        # number of vectors in the .bin, which is what must agree with the
        # sidecar.
        actual_count = index.get_current_count()
        if actual_count != len(ids):
            raise ValueError(
                f"index/sidecar mismatch: {path} contains {actual_count} vectors "
                f"but {sidecar_path} lists {len(ids)} ids — they are out of sync "
                "(rebuild the index)"
            )

        stored = np.array(index.get_items(list(range(actual_count))), dtype=np.float32)
        expected = fingerprint(ids, dim, actual_count, stored)
        if expected != sidecar.get("fingerprint"):
            raise ValueError(
                f"index/sidecar mismatch: {sidecar_path} does not match {path} "
                "(fingerprint mismatch) — they appear to come from different "
                "builds (rebuild the index)"
            )

        return cls(index, ids)

    def query(self, vector: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        k = min(k, len(self._ids))
        query_vec = normalize(np.asarray(vector, dtype=np.float32)[None])
        labels, distances = self._index.knn_query(query_vec, k=k)
        return [(self._ids[int(i)], float(d)) for i, d in zip(labels[0], distances[0])]
