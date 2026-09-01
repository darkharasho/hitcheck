/**
 * Enough of the D1 binding to run the queue's real SQL against real SQLite.
 *
 * The statements that hand out work are the part of this worker most likely
 * to be subtly wrong -- a claim that fails to exclude a card someone else
 * is already on, or a calibration count that waves a new cropper straight
 * through -- and none of that is visible from reading the JavaScript.
 * Running it against node:sqlite tests the SQL itself, with no network and
 * no deploy.
 *
 * bind() returns a NEW statement here because that is what D1 does. A shim
 * that mutated and returned itself would make the batched upsert in
 * handleAdmin appear to work while writing the last row N times.
 */
import { DatabaseSync } from 'node:sqlite'
import { readFileSync } from 'node:fs'

export function testDb() {
  const db = new DatabaseSync(':memory:')
  db.exec(readFileSync(new URL('../schema.sql', import.meta.url), 'utf8'))

  const make = (sql, args) => ({
    sql,
    args,
    bind: (...next) => make(sql, next),
    async first() {
      return db.prepare(sql).get(...args) ?? null
    },
    async all() {
      return { results: db.prepare(sql).all(...args) }
    },
    async run() {
      db.prepare(sql).run(...args)
      return {}
    },
  })

  return {
    raw: db,
    prepare: (sql) => make(sql, []),
    async batch(statements) {
      db.exec('BEGIN')
      try {
        for (const s of statements) db.prepare(s.sql).run(...s.args)
        db.exec('COMMIT')
      } catch (error) {
        db.exec('ROLLBACK')
        throw error
      }
      return []
    },
  }
}

export function seed(db, items) {
  for (const i of items) {
    db.raw
      .prepare('INSERT INTO items (item_id, card_id, image, calibration) VALUES (?, ?, ?, ?)')
      .run(i.item_id, i.card_id ?? 'card-x', i.image ?? `images/${i.item_id}.jpg`,
           i.calibration ? 1 : 0)
  }
}
