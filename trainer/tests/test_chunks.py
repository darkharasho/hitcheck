import numpy as np
from PIL import Image

from hitcheck_trainer.eval.chunks import embed_in_chunks, load_chunk


class FakeEmbedder:
    """Records the batch sizes it was handed; returns one row per image."""

    dim = 4

    def __init__(self):
        self.batch_sizes = []

    def embed(self, images, batch_size=32):
        self.batch_sizes.append(len(images))
        return np.tile(np.arange(self.dim, dtype=np.float32), (len(images), 1))


def written(tmp_path, names, size=(8, 8)):
    items = []
    for name in names:
        path = tmp_path / f"{name}.png"
        Image.new("RGB", size, "white").save(path)
        items.append((name, str(path)))
    return items


def test_load_chunk_returns_indices_labels_and_decoded_images(tmp_path):
    indices, labels, images = load_chunk(written(tmp_path, ["a", "b"]))
    assert indices == [0, 1]
    assert labels == ["a", "b"]
    assert [im.mode for im in images] == ["RGB", "RGB"]


def test_load_chunk_indices_are_offset_into_the_full_item_list(tmp_path):
    indices, _, _ = load_chunk(written(tmp_path, ["a", "b"]), offset=256)
    assert indices == [256, 257]


def test_load_chunk_skips_a_truncated_file_without_failing_the_run(tmp_path):
    # A catalog rerun replaces a truncated download; one bad file must not
    # abort an embed of twenty thousand images.
    items = written(tmp_path, ["good"])
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    items.append(("bad", str(bad)))
    indices, labels, images = load_chunk(items)
    assert labels == ["good"]
    assert indices == [0]
    assert len(images) == 1


def test_load_chunk_skips_a_missing_file(tmp_path):
    indices, labels, _ = load_chunk([("gone", str(tmp_path / "nope.png"))])
    assert labels == []
    assert indices == []


def test_embeds_every_item_and_returns_one_vector_each(tmp_path):
    embedder = FakeEmbedder()
    labels, vectors = embed_in_chunks(embedder, written(tmp_path, list("abcde")))
    assert labels == list("abcde")
    assert vectors.shape == (5, 4)


def test_never_decodes_more_than_chunk_images_at_once(tmp_path):
    # The whole point of this function: 20,427 catalog images at 240x330
    # RGB is 4.52GB decoded, and materialising it once caused a global OOM
    # on this machine. Chunking is not an optimisation, it is a constraint.
    embedder = FakeEmbedder()
    embed_in_chunks(embedder, written(tmp_path, [str(i) for i in range(10)]), chunk=3)
    assert max(embedder.batch_sizes) <= 3
    assert sum(embedder.batch_sizes) == 10


def test_labels_line_up_with_vectors_when_a_file_in_the_middle_is_unreadable(tmp_path):
    items = written(tmp_path, ["a", "c"])
    bad = tmp_path / "b.png"
    bad.write_bytes(b"junk")
    items.insert(1, ("b", str(bad)))
    labels, vectors = embed_in_chunks(FakeEmbedder(), items, chunk=2)
    assert labels == ["a", "c"]
    assert len(vectors) == len(labels)


def test_transform_is_applied_with_the_items_global_index(tmp_path):
    # synthetic.py seeds its degradation off this index, so it must be the
    # position in `items`, not the position within the current chunk --
    # otherwise every chunk would repeat the same seeds.
    seen = []

    def transform(image, index):
        seen.append(index)
        return image

    embed_in_chunks(FakeEmbedder(), written(tmp_path, [str(i) for i in range(5)]),
                    chunk=2, transform=transform)
    assert seen == [0, 1, 2, 3, 4]


def test_the_transform_index_still_points_at_the_right_item_after_a_skip(tmp_path):
    # eval/real.py looks its crop quad up by this index. If a skipped file
    # shifted the indices of everything after it, every later photograph
    # would be cropped with its neighbour's quad -- a silent, total
    # corruption of the eval rather than a crash.
    items = written(tmp_path, ["a", "c", "d"])
    bad = tmp_path / "b.png"
    bad.write_bytes(b"junk")
    items.insert(1, ("b", str(bad)))

    seen = []

    def transform(image, index):
        seen.append(index)
        return image

    labels, _ = embed_in_chunks(FakeEmbedder(), items, chunk=4, transform=transform)
    assert labels == ["a", "c", "d"]
    assert seen == [0, 2, 3]  # not [0, 1, 2]


def test_the_transform_index_survives_a_chunk_boundary_after_an_earlier_skip(tmp_path):
    # Every other skip test keeps the skip and its survivors inside a single
    # chunk (<=4 items, chunk=2 or 4), so offset arithmetic that is correct in
    # chunk 0 but off by `chunk` in chunk 1 would still pass all of them. This
    # spans three chunks -- the skip lands in chunk 0, and chunks 1 and 2 hold
    # further readable items -- so a boundary-crossing offset bug shows up.
    items = written(tmp_path, ["a"])
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"junk")
    items.append(("bad", str(bad)))
    items += written(tmp_path, ["c", "d", "e"])

    seen = []

    def transform(image, index):
        seen.append(index)
        return image

    labels, vectors = embed_in_chunks(FakeEmbedder(), items, chunk=2, transform=transform)
    assert labels == ["a", "c", "d", "e"]
    assert len(vectors) == len(labels)
    assert seen == [0, 2, 3, 4]  # not [0, 1, 2, 3]


def test_no_items_returns_an_empty_array_of_the_right_width(tmp_path):
    labels, vectors = embed_in_chunks(FakeEmbedder(), [])
    assert labels == []
    assert vectors.shape == (0, 4)


def test_all_items_unreadable_returns_an_empty_array_not_a_crash(tmp_path):
    bad = tmp_path / "b.png"
    bad.write_bytes(b"junk")
    labels, vectors = embed_in_chunks(FakeEmbedder(), [("b", str(bad))])
    assert labels == []
    assert vectors.shape == (0, 4)


def test_synthetic_reexports_nothing_it_no_longer_owns():
    # The refactor's whole point is a single implementation; a leftover
    # copy in synthetic.py would be free to drift.
    from hitcheck_trainer.eval import synthetic

    assert not hasattr(synthetic, "load_chunk")
    assert synthetic.embed_in_chunks is embed_in_chunks
