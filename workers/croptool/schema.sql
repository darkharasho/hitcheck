-- Corpus state for the hosted crop tool.
--
-- D1 rather than KV: handing out work needs "the next card nobody else is
-- on" to be a single atomic decision, and KV's eventual consistency would
-- hand the same card to two people who opened the tool at the same moment.

CREATE TABLE IF NOT EXISTS items (
  item_id       TEXT PRIMARY KEY,
  card_id       TEXT NOT NULL,
  image         TEXT NOT NULL,
  -- A calibration item is one the project owner has already marked. Every
  -- cropper does the same ones first, and their quads are compared against
  -- the owner's rather than merged into the corpus.
  calibration   INTEGER NOT NULL DEFAULT 0,
  claimed_by    TEXT,
  claimed_until INTEGER
);

-- Keyed by (item_id, cropper), not item_id alone: a calibration card
-- deliberately collects one quad per person, and that is the whole
-- agreement signal.
CREATE TABLE IF NOT EXISTS crops (
  item_id TEXT NOT NULL,
  cropper TEXT NOT NULL,
  quad    TEXT NOT NULL,
  at      INTEGER NOT NULL,
  PRIMARY KEY (item_id, cropper)
);

CREATE TABLE IF NOT EXISTS skips (
  item_id TEXT NOT NULL,
  cropper TEXT NOT NULL,
  at      INTEGER NOT NULL,
  PRIMARY KEY (item_id, cropper)
);

CREATE INDEX IF NOT EXISTS items_open ON items (calibration, claimed_until);
CREATE INDEX IF NOT EXISTS crops_by_cropper ON crops (cropper);
