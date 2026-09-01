import numpy as np
from PIL import Image

from hitcheck_trainer.corpus.manifest import CorpusEntry, Manifest, image_relpath
from hitcheck_trainer.eval.real import corpus_queries, run_eval

QUAD_A = [[10, 10], [200, 10], [200, 300], [10, 300]]
QUAD_B = [[20, 20], [210, 20], [210, 310], [20, 310]]


class FakeEmbedder:
    dim = 4

    def embed(self, images, batch_size=32):
        # One row per image, encoding its mean brightness so different
        # crops produce different vectors.
        return np.array([[float(np.array(im).mean())] * self.dim for im in images],
                        dtype=np.float32)


class FakeIndex:
    def __init__(self, ranked):
        self.ranked = ranked
        self.queries = 0

    def query(self, vector, k=5):
        self.queries += 1
        return self.ranked


def corpus(tmp_path, entries, write=None):
    write = entries if write is None else write
    for item_id in write:
        path = tmp_path / image_relpath(item_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (400, 400), "white").save(path, "JPEG")
    return Manifest(entries=[
        CorpusEntry(item_id=i, card_id=f"card-{i}", image=image_relpath(i),
                    image_url="u", listing_url="l", aspects={})
        for i in entries
    ])


def test_pairs_each_cropped_entry_with_its_quad(tmp_path):
    manifest = corpus(tmp_path, ["a", "b"])
    items, quads = corpus_queries(manifest, {"a": QUAD_A, "b": QUAD_B}, str(tmp_path))
    assert [label for label, _ in items] == ["card-a", "card-b"]
    assert quads == [QUAD_A, QUAD_B]


def test_the_query_label_is_the_card_id_not_the_item_id(tmp_path):
    # score() compares against ids from the catalog index.
    manifest = corpus(tmp_path, ["a"])
    items, _ = corpus_queries(manifest, {"a": QUAD_A}, str(tmp_path))
    assert items[0][0] == "card-a"


def test_an_entry_with_no_crop_yet_is_skipped(tmp_path):
    # A partial crops.json is the normal state mid-pass.
    manifest = corpus(tmp_path, ["a", "b"])
    items, quads = corpus_queries(manifest, {"b": QUAD_B}, str(tmp_path))
    assert [label for label, _ in items] == ["card-b"]
    assert quads == [QUAD_B]


def test_an_entry_whose_image_is_missing_from_disk_is_skipped(tmp_path):
    manifest = corpus(tmp_path, ["a", "b"], write=["a"])
    items, quads = corpus_queries(manifest, {"a": QUAD_A, "b": QUAD_B}, str(tmp_path))
    assert [label for label, _ in items] == ["card-a"]
    assert len(quads) == 1


def test_items_and_quads_stay_aligned_when_entries_are_dropped(tmp_path):
    # Filtering one list without the other would crop every later
    # photograph with its neighbour's quad and silently corrupt the eval.
    manifest = corpus(tmp_path, ["a", "b", "c"], write=["a", "c"])
    items, quads = corpus_queries(manifest, {"a": QUAD_A, "c": QUAD_B}, str(tmp_path))
    assert len(items) == len(quads) == 2
    assert [label for label, _ in items] == ["card-a", "card-c"]
    assert quads == [QUAD_A, QUAD_B]


def test_no_crops_at_all_yields_no_queries(tmp_path):
    items, quads = corpus_queries(corpus(tmp_path, ["a"]), {}, str(tmp_path))
    assert items == [] and quads == []


def test_run_eval_returns_one_true_id_and_ranking_per_query(tmp_path):
    manifest = corpus(tmp_path, ["a", "b"])
    items, quads = corpus_queries(manifest, {"a": QUAD_A, "b": QUAD_B}, str(tmp_path))
    index = FakeIndex([("card-a", 0.1), ("card-z", 0.4)])
    results = run_eval(FakeEmbedder(), index, items, quads)
    assert [true_id for true_id, _ in results] == ["card-a", "card-b"]
    assert index.queries == 2


def test_run_eval_scores_through_the_existing_report_harness(tmp_path):
    from hitcheck_trainer.eval.report import score

    manifest = corpus(tmp_path, ["a", "b"])
    items, quads = corpus_queries(manifest, {"a": QUAD_A, "b": QUAD_B}, str(tmp_path))
    index = FakeIndex([("card-a", 0.1)])
    report = score(run_eval(FakeEmbedder(), index, items, quads))
    assert report.total == 2
    assert report.top1 == 0.5  # card-a hits, card-b does not


def test_run_eval_crops_before_embedding(tmp_path):
    # If the photograph went in uncropped, the embedder would see a
    # 400x400 desk shot; apply_quad hands it a 245x342 card.
    seen = []

    class SizeRecordingEmbedder(FakeEmbedder):
        def embed(self, images, batch_size=32):
            seen.extend(im.size for im in images)
            return super().embed(images, batch_size)

    manifest = corpus(tmp_path, ["a"])
    items, quads = corpus_queries(manifest, {"a": QUAD_A}, str(tmp_path))
    run_eval(SizeRecordingEmbedder(), FakeIndex([("card-a", 0.1)]), items, quads)
    assert seen == [(245, 342)]


def test_run_eval_of_an_empty_corpus_returns_no_results(tmp_path):
    assert run_eval(FakeEmbedder(), FakeIndex([]), [], []) == []
