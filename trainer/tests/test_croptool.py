import json

from PIL import Image

from hitcheck_trainer.corpus.crops import load_crops, load_skips
from hitcheck_trainer.corpus.croptool import CropApp
from hitcheck_trainer.corpus.manifest import CorpusEntry, Manifest, image_relpath

QUAD = [[50, 60], [300, 40], [330, 300], [80, 330]]


def make_app(tmp_path, item_ids=("v1|1|0", "v1|2|0"), crops=None):
    entries = []
    for item_id in item_ids:
        relpath = image_relpath(item_id)
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (400, 400), "white").save(path, "JPEG")
        entries.append(
            CorpusEntry(
                item_id=item_id,
                card_id="sv3pt5-199",
                image=relpath,
                image_url="u",
                listing_url="l",
                aspects={},
            )
        )
    return CropApp(
        manifest=Manifest(entries=entries),
        crops=dict(crops or {}),
        crops_path=str(tmp_path / "crops.json"),
        corpus_dir=str(tmp_path),
    )


def body_of(response):
    return json.loads(response[2])


def test_next_item_is_the_first_entry_with_no_quad_yet(tmp_path):
    app = make_app(tmp_path)
    assert app.next_item()["item_id"] == "v1|1|0"


def test_next_item_skips_entries_that_are_already_cropped(tmp_path):
    app = make_app(tmp_path, crops={"v1|1|0": QUAD})
    assert app.next_item()["item_id"] == "v1|2|0"


def test_next_item_is_none_once_everything_is_cropped(tmp_path):
    app = make_app(tmp_path, crops={"v1|1|0": QUAD, "v1|2|0": QUAD})
    assert app.next_item() is None


def test_progress_counts_cropped_against_total(tmp_path):
    assert make_app(tmp_path, crops={"v1|1|0": QUAD}).progress() == (1, 2)


def test_get_root_serves_the_page(tmp_path):
    status, content_type, payload = make_app(tmp_path).handle("GET", "/", b"")
    assert status == 200
    assert content_type == "text/html"
    assert b"<canvas" in payload


def test_api_next_reports_the_item_and_the_progress(tmp_path):
    response = make_app(tmp_path).handle("GET", "/api/next", b"")
    payload = body_of(response)
    assert payload["item_id"] == "v1|1|0"
    assert payload["card_id"] == "sv3pt5-199"
    assert payload["done"] == 0
    assert payload["total"] == 2


def test_api_next_reports_a_null_item_when_the_pass_is_complete(tmp_path):
    app = make_app(tmp_path, crops={"v1|1|0": QUAD, "v1|2|0": QUAD})
    payload = body_of(app.handle("GET", "/api/next", b""))
    assert payload["item_id"] is None
    assert payload["done"] == payload["total"] == 2


def test_api_image_returns_the_photograph_bytes(tmp_path):
    app = make_app(tmp_path)
    status, content_type, payload = app.handle("GET", "/api/image?id=v1%7C1%7C0", b"")
    assert status == 200
    assert content_type == "image/jpeg"
    assert payload[:2] == b"\xff\xd8"  # JPEG magic


def test_api_image_for_an_unknown_id_is_a_404_not_a_traceback(tmp_path):
    status, _, _ = make_app(tmp_path).handle("GET", "/api/image?id=nope", b"")
    assert status == 404


def test_posting_a_quad_records_it_and_persists_immediately(tmp_path):
    app = make_app(tmp_path)
    payload = json.dumps({"item_id": "v1|1|0", "quad": QUAD}).encode()
    status, _, _ = app.handle("POST", "/api/quad", payload)
    assert status == 200
    # Persisted, not just held in memory: a crash three hours into a
    # hand-crop pass must not cost the pass.
    assert load_crops(str(tmp_path / "crops.json"))["v1|1|0"] == QUAD


def test_posting_a_quad_advances_next_item(tmp_path):
    app = make_app(tmp_path)
    app.handle("POST", "/api/quad", json.dumps({"item_id": "v1|1|0", "quad": QUAD}).encode())
    assert app.next_item()["item_id"] == "v1|2|0"


def test_posting_a_degenerate_quad_is_rejected_with_400(tmp_path):
    app = make_app(tmp_path)
    bad = json.dumps({"item_id": "v1|1|0", "quad": [[0, 0], [1, 0], [1, 1], [0, 1]]}).encode()
    status, _, payload = app.handle("POST", "/api/quad", bad)
    assert status == 400
    assert "error" in json.loads(payload)
    assert app.next_item()["item_id"] == "v1|1|0"  # not advanced


def test_posting_a_counter_clockwise_quad_is_rejected_and_tells_the_operator_why(tmp_path):
    # The 400 body is what the client alerts, so the winding message has to
    # survive the round trip -- otherwise the operator sees a rejection with
    # no idea that re-clicking the other way round is the fix.
    app = make_app(tmp_path)
    bad = json.dumps({"item_id": "v1|1|0", "quad": list(reversed(QUAD))}).encode()
    status, _, payload = app.handle("POST", "/api/quad", bad)
    assert status == 400
    assert "counter-clockwise" in json.loads(payload)["error"]
    assert app.next_item()["item_id"] == "v1|1|0"  # not advanced


def test_posting_the_wrong_number_of_points_is_rejected_with_400(tmp_path):
    app = make_app(tmp_path)
    bad = json.dumps({"item_id": "v1|1|0", "quad": [[0, 0], [400, 0], [400, 400]]}).encode()
    assert app.handle("POST", "/api/quad", bad)[0] == 400


def test_posting_malformed_json_is_rejected_with_400(tmp_path):
    assert make_app(tmp_path).handle("POST", "/api/quad", b"{not json")[0] == 400


def test_an_unknown_route_is_a_404(tmp_path):
    assert make_app(tmp_path).handle("GET", "/nope", b"")[0] == 404


def test_the_page_sends_original_image_coordinates_not_display_coordinates(tmp_path):
    # The canvas scales the photo to fit the window. If clicks were posted
    # in display pixels every quad would be wrong by that scale factor, so
    # the client divides by its own scale before posting.
    from hitcheck_trainer.corpus.croptool import PAGE

    assert "scale" in PAGE
    assert "/api/quad" in PAGE


def test_skipping_an_item_advances_past_it(tmp_path):
    # One unrenderable photograph must never be able to trap the pass: with
    # no skip control /api/next returns the same jammed entry forever.
    app = make_app(tmp_path)
    status, _, _ = app.handle("POST", "/api/skip", json.dumps({"item_id": "v1|1|0"}).encode())
    assert status == 200
    assert app.next_item()["item_id"] == "v1|2|0"


def test_a_skip_persists_so_the_next_run_does_not_serve_it_again(tmp_path):
    app = make_app(tmp_path)
    app.handle("POST", "/api/skip", json.dumps({"item_id": "v1|1|0"}).encode())

    reopened = CropApp(
        manifest=app._manifest,
        crops={},
        crops_path=str(tmp_path / "crops.json"),
        corpus_dir=str(tmp_path),
        skips=load_skips(str(tmp_path / "skipped.json")),
        skips_path=str(tmp_path / "skipped.json"),
    )
    assert reopened.next_item()["item_id"] == "v1|2|0"


def test_skipping_an_unknown_item_is_a_400_not_a_traceback(tmp_path):
    app = make_app(tmp_path)
    status, _, _ = app.handle("POST", "/api/skip", json.dumps({"item_id": "nope"}).encode())
    assert status == 400


def test_a_skipped_item_is_not_counted_as_cropped(tmp_path):
    # It has no quad, so counting it as done would overstate the crop pass.
    app = make_app(tmp_path)
    app.handle("POST", "/api/skip", json.dumps({"item_id": "v1|1|0"}).encode())
    assert app.progress() == (0, 2)
    assert load_crops(str(tmp_path / "crops.json")) == {}


def test_the_page_offers_the_skip_control_and_says_so_on_screen(tmp_path):
    from hitcheck_trainer.corpus.croptool import PAGE

    assert "/api/skip" in PAGE
    assert "skip" in PAGE.split("<script>")[0]  # documented in the header hint
