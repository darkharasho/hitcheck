import { test } from 'node:test'
import assert from 'node:assert/strict'
import { testDb, seed } from './d1.js'
import { nextItem, progress, calibrationState } from '../src/index.js'

const env = (DB) => ({ DB, CALIBRATION_N: '2', CLAIM_MINUTES: '10' })
const NOW = 1_000_000

function withCorpus({ calibration = 0, corpus = 3 } = {}) {
  const db = testDb()
  const items = []
  for (let i = 0; i < calibration; i++) items.push({ item_id: `cal-${i}`, calibration: 1 })
  for (let i = 0; i < corpus; i++) items.push({ item_id: `corpus-${i}` })
  seed(db, items)
  return db
}

const cropped = (db, item_id, cropper) =>
  db.raw
    .prepare('INSERT INTO crops (item_id, cropper, quad, at) VALUES (?, ?, ?, ?)')
    .run(item_id, cropper, '[]', 1)

test('a new cropper is handed calibration cards before any corpus card', async () => {
  // The whole point of calibration is that it comes first. Serving corpus
  // work to someone unmeasured is the failure this ordering prevents.
  const db = withCorpus({ calibration: 3 })
  const { item } = await nextItem(env(db), 'new@x.y', NOW)
  assert.equal(item.item_id, 'cal-0')
})

test('calibration cards are not claimed, so two croppers get the same one', async () => {
  // Everyone must mark the SAME reference cards -- that is the comparison.
  // A lease here would starve whoever arrived second.
  const db = withCorpus({ calibration: 3 })
  const a = await nextItem(env(db), 'a@x.y', NOW)
  const b = await nextItem(env(db), 'b@x.y', NOW)
  assert.equal(a.item.item_id, 'cal-0')
  assert.equal(b.item.item_id, 'cal-0')
})

test('a cropper moves on to the corpus once the calibration quota is met', async () => {
  const db = withCorpus({ calibration: 3 })
  cropped(db, 'cal-0', 'a@x.y')
  cropped(db, 'cal-1', 'a@x.y')
  const { item } = await nextItem(env(db), 'a@x.y', NOW) // CALIBRATION_N is 2
  assert.equal(item.item_id, 'corpus-0')
})

test('one cropper finishing calibration does not release another', async () => {
  const db = withCorpus({ calibration: 3 })
  cropped(db, 'cal-0', 'a@x.y')
  cropped(db, 'cal-1', 'a@x.y')
  const { item } = await nextItem(env(db), 'b@x.y', NOW)
  assert.equal(item.item_id, 'cal-0')
})

test('a skipped calibration card still counts toward the quota', async () => {
  // Otherwise an unloadable reference photograph traps that cropper on the
  // calibration set forever with no way past it.
  const db = withCorpus({ calibration: 3 })
  db.raw.prepare('INSERT INTO skips (item_id, cropper, at) VALUES (?, ?, ?)')
    .run('cal-0', 'a@x.y', 1)
  cropped(db, 'cal-1', 'a@x.y')
  const { item } = await nextItem(env(db), 'a@x.y', NOW)
  assert.equal(item.item_id, 'corpus-0')
})

test('the quota never exceeds the number of calibration cards that exist', async () => {
  // With one reference card and CALIBRATION_N of 2, a naive count would
  // demand a second card that is not there and hand out nothing at all.
  const db = withCorpus({ calibration: 1 })
  const state = await calibrationState(env(db), 'a@x.y')
  assert.equal(state.total, 1)
  cropped(db, 'cal-0', 'a@x.y')
  const { item } = await nextItem(env(db), 'a@x.y', NOW)
  assert.equal(item.item_id, 'corpus-0')
})

test('two croppers are never handed the same corpus card', async () => {
  const db = withCorpus()
  const a = await nextItem(env(db), 'a@x.y', NOW)
  const b = await nextItem(env(db), 'b@x.y', NOW)
  assert.notEqual(a.item.item_id, b.item.item_id)
})

test('reloading hands a cropper back the card they already hold', async () => {
  // A page refresh must not burn through the queue leaving claimed,
  // unmarked cards behind it.
  const db = withCorpus()
  const first = await nextItem(env(db), 'a@x.y', NOW)
  const again = await nextItem(env(db), 'a@x.y', NOW + 1000)
  assert.equal(again.item.item_id, first.item.item_id)
})

test('an expired claim returns the card to the pool', async () => {
  // Someone who opens the tool and walks away must not remove a card from
  // the corpus permanently.
  const db = withCorpus({ corpus: 1 })
  await nextItem(env(db), 'gone@x.y', NOW)
  assert.equal((await nextItem(env(db), 'b@x.y', NOW + 60_000)).item, null)
  const later = await nextItem(env(db), 'b@x.y', NOW + 11 * 60_000)
  assert.equal(later.item.item_id, 'corpus-0')
})

test('a marked card is never handed out again, to anyone', async () => {
  const db = withCorpus({ corpus: 1 })
  cropped(db, 'corpus-0', 'a@x.y')
  assert.equal((await nextItem(env(db), 'b@x.y', NOW)).item, null)
})

test('a skipped card is never handed out again, to anyone', async () => {
  // A photograph that will not decode is unusable for everybody, so a skip
  // retires it rather than passing it round the group.
  const db = withCorpus({ corpus: 1 })
  db.raw.prepare('INSERT INTO skips (item_id, cropper, at) VALUES (?, ?, ?)')
    .run('corpus-0', 'a@x.y', 1)
  assert.equal((await nextItem(env(db), 'b@x.y', NOW)).item, null)
})

test('progress counts corpus cards only, and each card once', async () => {
  // Calibration cards collect one quad per person by design. Counting them
  // would show progress past 100% and hide how much work is actually left.
  const db = withCorpus({ calibration: 2, corpus: 3 })
  cropped(db, 'cal-0', 'a@x.y')
  cropped(db, 'cal-0', 'b@x.y')
  cropped(db, 'corpus-0', 'a@x.y')
  // Two people on one corpus card happens when a lease expires mid-card.
  // Counting rows rather than cards would report 2/3 done from one card.
  cropped(db, 'corpus-0', 'b@x.y')
  const counts = await progress(env(db))
  assert.deepEqual([counts.done, counts.total], [1, 3])
})
