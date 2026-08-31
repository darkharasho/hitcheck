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
        saved = json.load(fh)
    assert saved["ids"] == ids


# --- Fix round 1: load() must not trust an unverified sidecar -------------
#
# hnswlib's load_index(path, max_elements=N) treats max_elements as a future
# growth ceiling, not a validation of what's actually stored — passing a
# smaller or larger value than the real element count is silently ignored.
# So `.ids.json` and the `.bin` file can drift apart (fewer/more ids than
# vectors, or a same-length sidecar from a completely different build) with
# nothing catching it, and every query after that silently returns the wrong
# card. These tests pin down that CardIndex.load() detects and rejects all
# of that instead of mismapping silently.


def test_sidecar_with_fewer_ids_than_index_raises_descriptive_error(tmp_path):
    import json

    vecs = vectors(n=5)
    ids = [f"c-{i}" for i in range(5)]
    path = str(tmp_path / "idx.bin")
    build_index(vecs, ids, path)

    sidecar_path = f"{path}.ids.json"
    with open(sidecar_path) as fh:
        sidecar = json.load(fh)
    sidecar["ids"] = sidecar["ids"][:3]  # fewer ids than the index actually holds
    with open(sidecar_path, "w") as fh:
        json.dump(sidecar, fh)

    with pytest.raises(ValueError, match="mismatch"):
        CardIndex.load(path, dim=vecs.shape[1])


def test_sidecar_with_more_ids_than_index_raises_descriptive_error(tmp_path):
    import json

    vecs = vectors(n=5)
    ids = [f"c-{i}" for i in range(5)]
    path = str(tmp_path / "idx.bin")
    build_index(vecs, ids, path)

    sidecar_path = f"{path}.ids.json"
    with open(sidecar_path) as fh:
        sidecar = json.load(fh)
    sidecar["ids"] = sidecar["ids"] + ["extra-1", "extra-2"]  # more ids than vectors
    with open(sidecar_path, "w") as fh:
        json.dump(sidecar, fh)

    with pytest.raises(ValueError, match="mismatch"):
        CardIndex.load(path, dim=vecs.shape[1])


def test_same_length_sidecar_from_different_build_is_rejected(tmp_path):
    """The dangerous case: count matches, but the sidecar belongs to a
    different build entirely. A count check alone can't catch this — only a
    content fingerprint can."""
    import shutil

    vecs_a = vectors(n=5, seed=1)
    ids_a = [f"a-{i}" for i in range(5)]
    path_a = str(tmp_path / "a.bin")
    build_index(vecs_a, ids_a, path_a)

    vecs_b = vectors(n=5, seed=2)
    ids_b = [f"b-{i}" for i in range(5)]
    path_b = str(tmp_path / "b.bin")
    build_index(vecs_b, ids_b, path_b)

    # Swap in build B's same-length sidecar next to build A's index.
    shutil.copy(f"{path_b}.ids.json", f"{path_a}.ids.json")

    with pytest.raises(ValueError, match="mismatch"):
        CardIndex.load(path_a, dim=vecs_a.shape[1])


def test_a_crash_between_index_and_sidecar_writes_does_not_leave_a_silently_stale_pair(
    tmp_path, monkeypatch
):
    """Simulate a process crash between the index rename and the sidecar
    rename during a rebuild. The .bin at `path` ends up belonging to the new
    build while the sidecar is still the old build's — same length, wrong
    pairing. Loading must refuse rather than silently mis-map card ids."""
    import hitcheck_trainer.index.build as build_mod

    vecs_a = vectors(n=5, seed=1)
    ids_a = [f"a-{i}" for i in range(5)]
    path = str(tmp_path / "idx.bin")
    build_index(vecs_a, ids_a, path)  # first, complete build

    vecs_b = vectors(n=5, seed=2)
    ids_b = [f"b-{i}" for i in range(5)]

    real_replace = build_mod.os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated crash between index and sidecar rename")
        return real_replace(src, dst)

    monkeypatch.setattr(build_mod.os, "replace", flaky_replace)

    with pytest.raises(OSError):
        build_index(vecs_b, ids_b, path)  # crashes after index rename, before sidecar rename

    monkeypatch.undo()

    with pytest.raises(ValueError, match="mismatch"):
        CardIndex.load(path, dim=vecs_a.shape[1])
