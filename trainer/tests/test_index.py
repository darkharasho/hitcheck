import numpy as np
import pytest

from hitcheck_trainer.index.build import build_index, normalize
from hitcheck_trainer.index.query import CardIndex


def vectors(n=20, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    return normalize(rng.normal(size=(n, dim)).astype(np.float32))


def test_normalize_produces_unit_vectors():
    out = normalize(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_normalize_leaves_zero_vectors_finite():
    out = normalize(np.zeros((2, 4), dtype=np.float32))
    assert np.isfinite(out).all()


def test_query_finds_the_exact_vector_first(tmp_path):
    vecs = vectors()
    ids = [f"c-{i}" for i in range(len(vecs))]
    path = str(tmp_path / "idx.bin")
    build_index(vecs, ids, path)

    index = CardIndex.load(path, dim=vecs.shape[1])
    assert index.query(vecs[7], k=5)[0][0] == "c-7"


def test_query_returns_k_results_nearest_first(tmp_path):
    vecs = vectors()
    ids = [f"c-{i}" for i in range(len(vecs))]
    path = str(tmp_path / "idx.bin")
    build_index(vecs, ids, path)

    results = CardIndex.load(path, dim=vecs.shape[1]).query(vecs[3], k=5)
    assert len(results) == 5
    distances = [d for _, d in results]
    assert distances == sorted(distances)


def test_a_perturbed_vector_still_retrieves_its_source(tmp_path):
    vecs = vectors()
    ids = [f"c-{i}" for i in range(len(vecs))]
    path = str(tmp_path / "idx.bin")
    build_index(vecs, ids, path)

    rng = np.random.default_rng(99)
    noisy = normalize((vecs[11] + rng.normal(scale=0.05, size=vecs.shape[1])).astype(np.float32)[None])[0]
    assert CardIndex.load(path, dim=vecs.shape[1]).query(noisy, k=3)[0][0] == "c-11"


def test_k_larger_than_the_index_returns_everything(tmp_path):
    vecs = vectors(n=3)
    path = str(tmp_path / "idx.bin")
    build_index(vecs, ["a", "b", "c"], path)
    assert len(CardIndex.load(path, dim=vecs.shape[1]).query(vecs[0], k=50)) == 3


def test_build_rejects_mismatched_ids():
    with pytest.raises(ValueError):
        build_index(vectors(n=3), ["a", "b"], "/tmp/unused.bin")


def test_id_mapping_is_stable_across_save_and_load(tmp_path):
    """Deliberately verify index-position -> card-id mapping is correct and
    stable, not just that querying happens to work. An off-by-one or unstable
    ordering here would silently return the wrong card at identification time."""
    n, dim = 30, 12
    rng = np.random.default_rng(7)
    # Use well-separated basis-like vectors so nearest neighbour is unambiguous.
    raw = rng.normal(size=(n, dim)).astype(np.float32) * 0.01
    for i in range(n):
        raw[i, i % dim] += 10.0 + i
    vecs = normalize(raw)
    ids = [f"card-{i:03d}" for i in range(n)]
    path = str(tmp_path / "idx.bin")
    build_index(vecs, ids, path)

    index = CardIndex.load(path, dim=dim)
    for i in range(n):
        result = index.query(vecs[i], k=1)
        assert result[0][0] == ids[i], f"position {i} mapped to wrong id: {result[0][0]} != {ids[i]}"


def test_sidecar_ids_file_matches_input_order(tmp_path):
    """The .ids.json sidecar must preserve exact input order — CardIndex.load
    relies on hnswlib's internal integer label matching this list's index."""
    import json

    ids = ["z-card", "a-card", "m-card"]
    vecs = vectors(n=3)
    path = str(tmp_path / "idx.bin")
    build_index(vecs, ids, path)

    with open(f"{path}.ids.json") as fh:
        saved_ids = json.load(fh)
    assert saved_ids == ids
