"""Local catalog store.

Card JSON is flattened into columns for the fields retrieval and pricing
care about, with the untouched payload kept in `raw_json` so nothing is
lost if a later milestone needs a field this schema didn't anticipate.
"""

import json
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    number        TEXT,
    rarity        TEXT,
    supertype     TEXT,
    artist        TEXT,
    set_id        TEXT,
    set_name      TEXT,
    set_series    TEXT,
    set_release   TEXT,
    image_small   TEXT,
    raw_json      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cards_set ON cards(set_id);
CREATE INDEX IF NOT EXISTS idx_cards_name ON cards(name);

CREATE TABLE IF NOT EXISTS sync_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def upsert_cards(conn: sqlite3.Connection, cards: list[dict]) -> int:
    rows = []
    for card in cards:
        card_set = card.get("set") or {}
        images = card.get("images") or {}
        rows.append(
            (
                card["id"],
                card.get("name", ""),
                card.get("number"),
                card.get("rarity"),
                card.get("supertype"),
                card.get("artist"),
                card_set.get("id"),
                card_set.get("name"),
                card_set.get("series"),
                card_set.get("releaseDate"),
                images.get("small"),
                json.dumps(card, separators=(",", ":")),
            )
        )
    conn.executemany(
        """
        INSERT INTO cards (id, name, number, rarity, supertype, artist,
                           set_id, set_name, set_series, set_release,
                           image_small, raw_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name, number=excluded.number, rarity=excluded.rarity,
            supertype=excluded.supertype, artist=excluded.artist,
            set_id=excluded.set_id, set_name=excluded.set_name,
            set_series=excluded.set_series, set_release=excluded.set_release,
            image_small=excluded.image_small, raw_json=excluded.raw_json
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def card_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]


def get_card(conn: sqlite3.Connection, card_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
    return dict(row) if row else None


def all_card_images(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    rows = conn.execute(
        "SELECT id, image_small FROM cards WHERE image_small IS NOT NULL ORDER BY id"
    ).fetchall()
    return [(r["id"], r["image_small"]) for r in rows]


def set_sync_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO sync_state (key, value) VALUES (?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def get_sync_state(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM sync_state WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None
