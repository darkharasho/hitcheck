from PIL import Image

from hitcheck_trainer.corpus.audit import build_audit, sample_entries
from hitcheck_trainer.corpus.manifest import CorpusEntry, Manifest, image_relpath

QUAD = [[10, 10], [200, 10], [200, 300], [10, 300]]


def setup_corpus(tmp_path, n=6):
    corpus_dir = tmp_path / "corpus"
    images_root = tmp_path / "images"
    entries = []
    for i in range(n):
        item_id = f"v1|{i}|0"
        photo = corpus_dir / image_relpath(item_id)
        photo.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (400, 400), "white").save(photo, "JPEG")
        catalog = images_root / "sv3pt5" / f"sv3pt5-{i}.png"
        catalog.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (245, 342), "blue").save(catalog)
        entries.append(CorpusEntry(item_id=item_id, card_id=f"sv3pt5-{i}",
                                   image=image_relpath(item_id), image_url="u",
                                   listing_url=f"https://ebay/{i}", aspects={"Set": "151"}))
    crops = {e.item_id: QUAD for e in entries}
    return Manifest(entries=entries), crops, str(corpus_dir), str(images_root)


def test_samples_the_requested_number_of_entries(tmp_path):
    manifest, crops, _, _ = setup_corpus(tmp_path)
    assert len(sample_entries(manifest, crops, count=3, seed=0)) == 3


def test_sampling_is_seeded_so_a_rerun_audits_the_same_entries(tmp_path):
    # An audit that resampled every run could be repeated until it
    # produced a flattering error count.
    manifest, crops, _, _ = setup_corpus(tmp_path)
    first = [e.item_id for e in sample_entries(manifest, crops, count=3, seed=7)]
    second = [e.item_id for e in sample_entries(manifest, crops, count=3, seed=7)]
    assert first == second


def test_sampling_is_independent_of_input_ordering(tmp_path):
    # Neither dict ordering of `crops` nor list ordering of the manifest's
    # entries may change who gets sampled -- otherwise the "same seed"
    # guarantee is hollow: reordering crops.json on disk (or the dict's
    # insertion order from a JSON reload) could be used to reshuffle the
    # sample while keeping the seed fixed.
    manifest, crops, _, _ = setup_corpus(tmp_path, n=40)

    reversed_manifest = Manifest(entries=list(reversed(manifest.entries)))
    reversed_crops = dict(reversed(list(crops.items())))

    forward = [e.item_id for e in sample_entries(manifest, crops, count=5, seed=3)]
    backward = [e.item_id for e in sample_entries(reversed_manifest, reversed_crops,
                                                    count=5, seed=3)]
    assert forward == backward


def test_a_different_seed_can_sample_differently(tmp_path):
    manifest, crops, _, _ = setup_corpus(tmp_path, n=40)
    a = [e.item_id for e in sample_entries(manifest, crops, count=5, seed=1)]
    b = [e.item_id for e in sample_entries(manifest, crops, count=5, seed=2)]
    assert a != b


def test_only_cropped_entries_are_auditable(tmp_path):
    # An uncropped entry has no crop to show, and is not in the eval either.
    manifest, _, _, _ = setup_corpus(tmp_path)
    only_one = {manifest.entries[2].item_id: QUAD}
    sampled = sample_entries(manifest, only_one, count=50, seed=0)
    assert [e.item_id for e in sampled] == [manifest.entries[2].item_id]


def test_asking_for_more_than_exist_returns_everything_available(tmp_path):
    manifest, crops, _, _ = setup_corpus(tmp_path, n=4)
    assert len(sample_entries(manifest, crops, count=50, seed=0)) == 4


def test_writes_an_html_sheet_pairing_the_crop_with_the_catalog_scan(tmp_path):
    manifest, crops, corpus_dir, images_root = setup_corpus(tmp_path)
    out = build_audit(manifest, crops, corpus_dir, images_root,
                      str(tmp_path / "out"), count=2, seed=0)
    with open(out) as fh:
        html = fh.read()
    assert html.count("<img") == 4  # two pairs
    assert "sv3pt5-" in html


def test_the_sheet_links_back_to_the_listing_and_shows_the_aspects(tmp_path):
    # A mismatch is easier to adjudicate with the source listing to hand.
    manifest, crops, corpus_dir, images_root = setup_corpus(tmp_path)
    out = build_audit(manifest, crops, corpus_dir, images_root,
                      str(tmp_path / "out"), count=1, seed=0)
    with open(out) as fh:
        html = fh.read()
    assert "https://ebay/" in html
    assert "151" in html


def test_writes_a_cropped_preview_per_audited_entry(tmp_path):
    manifest, crops, corpus_dir, images_root = setup_corpus(tmp_path)
    build_audit(manifest, crops, corpus_dir, images_root, str(tmp_path / "out"),
                count=3, seed=0)
    previews = list((tmp_path / "out" / "crops").glob("*.png"))
    assert len(previews) == 3
    assert Image.open(previews[0]).size == (245, 342)


def test_states_the_sample_size_the_bound_will_be_computed_from(tmp_path):
    manifest, crops, corpus_dir, images_root = setup_corpus(tmp_path)
    out = build_audit(manifest, crops, corpus_dir, images_root,
                      str(tmp_path / "out"), count=3, seed=0)
    with open(out) as fh:
        assert "--label-sample 3" in fh.read()


def test_an_entry_whose_photograph_is_missing_is_skipped_not_fatal(tmp_path):
    manifest, crops, corpus_dir, images_root = setup_corpus(tmp_path, n=2)
    import os

    os.remove(os.path.join(corpus_dir, manifest.entries[0].image))
    out = build_audit(manifest, crops, corpus_dir, images_root,
                      str(tmp_path / "out"), count=50, seed=0)
    with open(out) as fh:
        assert fh.read().count("<img") == 2  # only the surviving pair
