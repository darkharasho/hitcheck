import sqlite3

import pytest

from hitcheck_trainer.catalog.db import (
    all_card_images,
    backfill_tcgplayer_urls,
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
    conn = open_db(path)
    upsert_cards(conn, [CARD])
    conn.close()
    conn = open_db(path)
    assert card_count(conn) == 1
    assert get_card(conn, "pl3-1") is not None


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


def test_explicit_null_name_stores_as_empty_string(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    assert upsert_cards(conn, [{"id": "n-1", "name": None}]) == 1
    row = get_card(conn, "n-1")
    assert row["name"] == ""


def test_failed_batch_rolls_back_entirely(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    # Force a genuine sqlite3.IntegrityError on the second row of a batch,
    # independent of any particular column's NOT NULL guard, via a trigger
    # that poisons inserts for a specific id.
    conn.execute(
        "CREATE TRIGGER poison_b1 BEFORE INSERT ON cards "
        "WHEN NEW.id = 'b-1' BEGIN SELECT RAISE(ABORT, 'poison row'); END"
    )
    good = {"id": "g-1", "name": "Good"}
    bad = {"id": "b-1", "name": "Bad"}

    with pytest.raises(sqlite3.IntegrityError):
        upsert_cards(conn, [good, bad])
    assert get_card(conn, "g-1") is None

    # Reproduce the reviewer's exact repro: a later, unrelated successful
    # upsert_cards call must not silently commit the rolled-back row.
    upsert_cards(conn, [{"id": "z-1", "name": "Unrelated"}])
    assert get_card(conn, "g-1") is None
    assert get_card(conn, "z-1") is not None


def test_sync_state_roundtrips(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    assert get_sync_state(conn, "last_page") is None
    set_sync_state(conn, "last_page", "12")
    assert get_sync_state(conn, "last_page") == "12"
    set_sync_state(conn, "last_page", "13")
    assert get_sync_state(conn, "last_page") == "13"


def test_upsert_stores_tcgplayer_url(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    upsert_cards(conn, [{
        "id": "base1-4", "name": "Charizard", "number": "4",
        "set": {"id": "base1", "name": "Base"},
        "tcgplayer": {"url": "https://prices.pokemontcg.io/tcgplayer/base1-4"},
    }])
    assert get_card(conn, "base1-4")["tcgplayer_url"] == \
        "https://prices.pokemontcg.io/tcgplayer/base1-4"


def test_upsert_tolerates_missing_tcgplayer_block(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    upsert_cards(conn, [{"id": "x-1", "name": "X", "set": {"id": "x"}}])
    assert get_card(conn, "x-1")["tcgplayer_url"] is None


def test_backfill_populates_from_raw_json(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    upsert_cards(conn, [{
        "id": "base1-4", "name": "Charizard",
        "set": {"id": "base1", "name": "Base"},
        "tcgplayer": {"url": "https://prices.pokemontcg.io/tcgplayer/base1-4"},
    }])
    conn.execute("UPDATE cards SET tcgplayer_url = NULL")
    conn.commit()

    assert backfill_tcgplayer_urls(conn) == 1
    assert get_card(conn, "base1-4")["tcgplayer_url"] == \
        "https://prices.pokemontcg.io/tcgplayer/base1-4"


def test_backfill_is_idempotent(tmp_path):
    conn = open_db(str(tmp_path / "c.sqlite"))
    upsert_cards(conn, [{
        "id": "base1-4", "name": "Charizard",
        "set": {"id": "base1", "name": "Base"},
        "tcgplayer": {"url": "https://prices.pokemontcg.io/tcgplayer/base1-4"},
    }])
    backfill_tcgplayer_urls(conn)
    assert backfill_tcgplayer_urls(conn) == 0

