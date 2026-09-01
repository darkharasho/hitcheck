import json

from hitcheck_trainer.corpus.manifest import (
    CorpusEntry,
    Manifest,
    image_relpath,
    load_manifest,
    safe_item_id,
    save_manifest,
)


def entry(item_id="v1|364012345678|0", card_id="sv3pt5-199"):
    return CorpusEntry(
        item_id=item_id,
        card_id=card_id,
        image=image_relpath(item_id),
        image_url="https://i.ebayimg.com/images/g/abc/s-l1600.jpg",
        listing_url="https://www.ebay.com/itm/364012345678",
        aspects={"Card Name": "Charizard ex", "Set": "151",
                 "Card Number": "199/165", "Language": "English"},
    )


def test_safe_item_id_strips_the_pipes_ebay_puts_in_item_ids():
    # Browse returns ids like v1|364012345678|0, which cannot be a filename.
    assert safe_item_id("v1|364012345678|0") == "v1_364012345678_0"


def test_safe_item_id_leaves_already_safe_ids_alone():
    assert safe_item_id("364012345678") == "364012345678"


def test_image_relpath_is_relative_and_under_images():
    assert image_relpath("v1|364012345678|0") == "images/v1_364012345678_0.jpg"


def test_round_trips_through_disk_unchanged(tmp_path):
    original = Manifest(
        entries=[entry(), entry(item_id="v1|999|0", card_id="bw1-1")],
        discards={"NOT_ENGLISH": 12, "NAME_MISMATCH": 3},
        queries=["pokemon psa graded card"],
    )
    path = str(tmp_path / "manifest.json")
    save_manifest(original, path)
    loaded = load_manifest(path)
    assert loaded == original


def test_saved_json_is_sorted_and_indented_so_git_diffs_are_readable(tmp_path):
    path = str(tmp_path / "manifest.json")
    save_manifest(Manifest(entries=[entry()], discards={}, queries=[]), path)
    with open(path) as fh:
        text = fh.read()
    assert "\n  " in text
    keys = list(json.loads(text)["entries"][0])
    assert keys == sorted(keys)


def test_save_leaves_no_part_file_behind(tmp_path):
    path = str(tmp_path / "manifest.json")
    save_manifest(Manifest(entries=[entry()], discards={}, queries=[]), path)
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.json"]


def test_item_ids_lets_a_rerun_skip_what_is_already_captured(tmp_path):
    # Listings expire, so images and entries are written once and never
    # re-fetched. A rerun must be able to tell what it already has.
    m = Manifest(entries=[entry(), entry(item_id="v1|999|0")], discards={}, queries=[])
    assert m.item_ids() == {"v1|364012345678|0", "v1|999|0"}


def test_loading_a_missing_manifest_gives_an_empty_one(tmp_path):
    loaded = load_manifest(str(tmp_path / "nope.json"))
    assert loaded.entries == []
    assert loaded.discards == {}


def test_yield_summary_states_the_corpus_yield_including_every_discard():
    m = Manifest(
        entries=[entry()],
        discards={"NOT_ENGLISH": 12, "NAME_MISMATCH": 3},
        queries=[],
    )
    text = m.yield_summary()
    assert "kept=1" in text
    assert "discarded=15" in text
    assert "NOT_ENGLISH=12" in text
    assert "NAME_MISMATCH=3" in text


def test_yield_summary_of_a_perfect_run_still_reports_zero_discards():
    assert "discarded=0" in Manifest(entries=[], discards={}, queries=[]).yield_summary()
