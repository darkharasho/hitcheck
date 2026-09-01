import io

from PIL import Image

from hitcheck_trainer.corpus.build import IMAGE_FAILED, build_corpus
from hitcheck_trainer.corpus.manifest import CorpusEntry, Manifest, image_relpath
from hitcheck_trainer.corpus.resolve import CardLookup


def lookup():
    return CardLookup(
        set_ids={"151": "sv3pt5"},
        cards={("sv3pt5", "199"): [("sv3pt5-199", "charizardex")],
               ("sv3pt5", "6"): [("sv3pt5-6", "charizardex")]},
    )


def specifics(number="199/165", language="English", name="Charizard ex"):
    return [
        {"name": "Card Name", "value": name},
        {"name": "Set", "value": "151"},
        {"name": "Card Number", "value": number},
        {"name": "Language", "value": language},
    ]


class FakeClient:
    """Serves a fixed set of summaries and details, counting calls."""

    def __init__(self, items):
        self.items = items  # item_id -> detail dict
        self.searches = 0
        self.detail_calls = []

    def search(self, query, limit=200, offset=0, extra_filter=None):
        self.searches += 1
        if offset:
            return []  # single page
        return [{"itemId": i} for i in self.items]

    def item(self, item_id):
        self.detail_calls.append(item_id)
        return self.items[item_id]


def detail(item_id, aspects, image="https://i.ebayimg.com/g/a/s-l225.jpg"):
    return {
        "itemId": item_id,
        "itemWebUrl": f"https://www.ebay.com/itm/{item_id}",
        "image": {"imageUrl": image},
        "localizedAspects": aspects,
    }


def jpeg_bytes(size=(20, 28), colour="red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, "JPEG")
    return buffer.getvalue()


JPEG = jpeg_bytes()


def ok_fetch(url):
    return 200, JPEG


def test_builds_a_manifest_entry_per_resolved_listing(tmp_path):
    client = FakeClient({"v1|1|0": detail("v1|1|0", specifics())})
    result = build_corpus(client, lookup(), ok_fetch, str(tmp_path), Manifest(),
                          ["q"], target=10, sleep=lambda s: None)
    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.card_id == "sv3pt5-199"
    assert entry.item_id == "v1|1|0"
    assert entry.aspects["Set"] == "151"


def test_downloads_the_hi_res_image_beside_the_manifest(tmp_path):
    requested = []

    def fetch(url):
        requested.append(url)
        return 200, JPEG

    client = FakeClient({"v1|1|0": detail("v1|1|0", specifics())})
    build_corpus(client, lookup(), fetch, str(tmp_path), Manifest(), ["q"],
                 target=10, sleep=lambda s: None)
    assert requested == ["https://i.ebayimg.com/g/a/s-l1600.jpg"]
    assert (tmp_path / image_relpath("v1|1|0")).read_bytes() == JPEG


def test_a_failed_image_download_is_counted_and_produces_no_entry(tmp_path):
    client = FakeClient({"v1|1|0": detail("v1|1|0", specifics())})
    result = build_corpus(client, lookup(), lambda url: (404, None), str(tmp_path),
                          Manifest(), ["q"], target=10, sleep=lambda s: None)
    assert result.entries == []
    assert result.discards[IMAGE_FAILED] == 1


def test_an_unresolvable_listing_is_counted_by_reason_not_guessed(tmp_path):
    client = FakeClient({"v1|1|0": detail("v1|1|0", specifics(language="Japanese"))})
    result = build_corpus(client, lookup(), ok_fetch, str(tmp_path), Manifest(),
                          ["q"], target=10, sleep=lambda s: None)
    assert result.entries == []
    assert result.discards["NOT_ENGLISH"] == 1


def test_discard_counts_accumulate_across_reasons(tmp_path):
    client = FakeClient({
        "v1|1|0": detail("v1|1|0", specifics(language="Japanese")),
        "v1|2|0": detail("v1|2|0", specifics(name="Blastoise ex")),
        "v1|3|0": detail("v1|3|0", specifics(number="9999/165")),
    })
    result = build_corpus(client, lookup(), ok_fetch, str(tmp_path), Manifest(),
                          ["q"], target=10, sleep=lambda s: None)
    assert result.discards == {"NOT_ENGLISH": 1, "NAME_MISMATCH": 1, "NO_SUCH_NUMBER": 1}


def test_a_rerun_never_refetches_a_listing_it_already_has(tmp_path):
    # Listings expire; the corpus survives them by writing once.
    existing = Manifest(entries=[CorpusEntry(
        item_id="v1|1|0", card_id="sv3pt5-199", image=image_relpath("v1|1|0"),
        image_url="https://i.ebayimg.com/g/a/s-l1600.jpg",
        listing_url="https://www.ebay.com/itm/v1|1|0", aspects={},
    )])
    client = FakeClient({"v1|1|0": detail("v1|1|0", specifics())})
    result = build_corpus(client, lookup(), ok_fetch, str(tmp_path), existing,
                          ["q"], target=10, sleep=lambda s: None)
    assert client.detail_calls == []  # never paid for the detail call again
    assert len(result.entries) == 1


def test_stops_once_the_target_is_reached(tmp_path):
    items = {f"v1|{i}|0": detail(f"v1|{i}|0", specifics()) for i in range(5)}
    client = FakeClient(items)
    result = build_corpus(client, lookup(), ok_fetch, str(tmp_path), Manifest(),
                          ["q"], target=2, sleep=lambda s: None)
    assert len(result.entries) == 2
    assert len(client.detail_calls) == 2  # no calls spent past the target


def test_the_target_counts_entries_already_in_the_manifest(tmp_path):
    existing = Manifest(entries=[CorpusEntry(
        item_id="v1|old|0", card_id="sv3pt5-6", image=image_relpath("v1|old|0"),
        image_url="u", listing_url="l", aspects={},
    )])
    client = FakeClient({"v1|1|0": detail("v1|1|0", specifics())})
    result = build_corpus(client, lookup(), ok_fetch, str(tmp_path), existing,
                          ["q"], target=1, sleep=lambda s: None)
    assert len(result.entries) == 1
    assert client.detail_calls == []


def test_the_manifest_is_saved_after_every_entry_so_an_interrupt_keeps_progress(tmp_path):
    from hitcheck_trainer.corpus.manifest import load_manifest

    items = {f"v1|{i}|0": detail(f"v1|{i}|0", specifics()) for i in range(3)}

    calls = {"n": 0}

    def flaky_fetch(url):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return 200, JPEG

    try:
        build_corpus(FakeClient(items), lookup(), flaky_fetch, str(tmp_path),
                     Manifest(), ["q"], target=10, sleep=lambda s: None)
    except KeyboardInterrupt:
        pass

    saved = load_manifest(str(tmp_path / "manifest.json"))
    assert len(saved.entries) == 2  # the two that completed before the interrupt


def test_the_queries_used_are_recorded_in_the_manifest(tmp_path):
    client = FakeClient({"v1|1|0": detail("v1|1|0", specifics())})
    result = build_corpus(client, lookup(), ok_fetch, str(tmp_path), Manifest(),
                          ["charizard psa", "pikachu psa"], target=10, sleep=lambda s: None)
    assert "charizard psa" in result.queries


def test_a_listing_with_no_image_url_is_counted_not_crashed_on(tmp_path):
    client = FakeClient({"v1|1|0": detail("v1|1|0", specifics(), image="")})
    result = build_corpus(client, lookup(), ok_fetch, str(tmp_path), Manifest(),
                          ["q"], target=10, sleep=lambda s: None)
    assert result.entries == []
    assert result.discards[IMAGE_FAILED] == 1


def test_a_body_that_is_not_a_decodable_image_is_discarded_not_manifested(tmp_path):
    # eBay can answer 200 with an HTML error page or a placeholder body.
    # fetch_to_path only checks the bytes are non-empty, so such a body used
    # to enter the manifest as a valid entry -- and then permanently jam the
    # hand-crop tool, whose img.onload never fires on it.
    html_body = b"<!doctype html><html><body>Sorry, this page is unavailable</body></html>"
    client = FakeClient({"v1|1|0": detail("v1|1|0", specifics())})
    result = build_corpus(client, lookup(), lambda url: (200, html_body), str(tmp_path),
                          Manifest(), ["q"], target=10, sleep=lambda s: None)
    assert result.entries == []
    assert result.discards[IMAGE_FAILED] == 1
    # The junk is removed, so a rerun refetches rather than treating the
    # file on disk as an already-completed download.
    assert not (tmp_path / image_relpath("v1|1|0")).exists()


def test_an_undecodable_image_already_on_disk_is_discarded_on_rerun(tmp_path):
    # The resume path skips the download when a non-empty file already
    # exists; the decode check has to cover that path too.
    path = tmp_path / image_relpath("v1|1|0")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"<html>not an image</html>")
    client = FakeClient({"v1|1|0": detail("v1|1|0", specifics())})
    result = build_corpus(client, lookup(), lambda url: (200, JPEG), str(tmp_path),
                          Manifest(), ["q"], target=10, sleep=lambda s: None)
    assert result.entries == []
    assert result.discards[IMAGE_FAILED] == 1


def test_a_decodable_image_is_kept(tmp_path):
    client = FakeClient({"v1|1|0": detail("v1|1|0", specifics())})
    result = build_corpus(client, lookup(), lambda url: (200, jpeg_bytes((40, 56), "blue")),
                          str(tmp_path), Manifest(), ["q"], target=10, sleep=lambda s: None)
    assert len(result.entries) == 1
    assert result.discards == {}
