from hitcheck_trainer.catalog.db import (
    all_card_images,
    card_count,
    get_card,
    get_sync_state,
    open_db,
    set_sync_state,
    upsert_cards,
)

CARD = {
    "id": "pl3-1",
    "name": "Aggron",
    "number": "1",
    "rarity": "Rare Holo",
    "supertype": "Pokémon",
    "artist": "Kagemaru Himeno",
    "images": {"small": "https://images.pokemontcg.io/pl3/1.png"},
    "set": {"id": "pl3", "name": "Supreme Victors", "series": "Platinum", "releaseDate": "2009/08/19"},
}


def test_open_db_is_idempotent(tmp_path):
    path = str(tmp_path / "c.sqlite")
    open_db(path).close()
    conn = open_db(path)
    assert card_count(conn) == 0


def test_upsert_inserts_a_card(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    assert upsert_cards(conn, [CARD]) == 1
    assert card_count(conn) == 1


def test_upsert_is_idempotent_on_repeat(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    upsert_cards(conn, [CARD])
    upsert_cards(conn, [CARD])
    assert card_count(conn) == 1


def test_upsert_updates_changed_fields(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    upsert_cards(conn, [CARD])
    upsert_cards(conn, [{**CARD, "rarity": "Rare Holo EX"}])
    assert get_card(conn, "pl3-1")["rarity"] == "Rare Holo EX"


def test_get_card_returns_flattened_set_fields(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    upsert_cards(conn, [CARD])
    row = get_card(conn, "pl3-1")
    assert row["set_id"] == "pl3"
    assert row["set_name"] == "Supreme Victors"
    assert row["image_small"] == "https://images.pokemontcg.io/pl3/1.png"


def test_get_card_returns_none_for_unknown_id(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    assert get_card(conn, "nope-1") is None


def test_card_missing_optional_fields_is_still_stored(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    assert upsert_cards(conn, [{"id": "x-1", "name": "Bare"}]) == 1
    row = get_card(conn, "x-1")
    assert row["rarity"] is None and row["image_small"] is None


def test_all_card_images_skips_cards_without_images(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    upsert_cards(conn, [CARD, {"id": "x-1", "name": "Bare"}])
    assert all_card_images(conn) == [("pl3-1", "https://images.pokemontcg.io/pl3/1.png")]


def test_sync_state_roundtrips(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    assert get_sync_state(conn, "last_page") is None
    set_sync_state(conn, "last_page", "12")
    assert get_sync_state(conn, "last_page") == "12"
    set_sync_state(conn, "last_page", "13")
    assert get_sync_state(conn, "last_page") == "13"
