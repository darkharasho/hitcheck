import { test } from 'node:test'
import assert from 'node:assert/strict'
import { testDb } from './d1.js'
import worker from '../src/index.js'

function bucket() {
  const store = new Map()
  return {
    store,
    async put(key, body) {
      store.set(key, body instanceof ReadableStream ? await new Response(body).arrayBuffer() : body)
    },
    async get(key) {
      return store.has(key) ? { body: store.get(key) } : null
    },
    async head(key) {
      return store.has(key) ? { key } : null
    },
  }
}

const env = () => ({
  DB: testDb(),
  PHOTOS: bucket(),
  ADMIN_TOKEN: 'secret',
  ACCESS_TEAM_DOMAIN: 'team.cloudflareaccess.com',
  ACCESS_AUD: 'aud-1',
})

const call = (path, init = {}, token = 'secret') =>
  worker.fetch(
    new Request(`https://crop.axi.link${path}`, {
      ...init,
      headers: { Authorization: `Bearer ${token}`, ...(init.headers || {}) },
    }),
    init.env,
  )

test('an admin route without the token is refused', async () => {
  const e = env()
  const res = await call('/api/admin/crops', { env: e }, 'wrong')
  assert.equal(res.status, 401)
})

test('an unset ADMIN_TOKEN does not make the empty string a credential', async () => {
  // The failure mode is a deploy where `wrangler secret put` was never run:
  // string equality alone would accept "Bearer undefined" or "Bearer ".
  const e = { ...env(), ADMIN_TOKEN: undefined }
  assert.equal((await call('/api/admin/crops', { env: e }, '')).status, 401)
  assert.equal((await call('/api/admin/crops', { env: e }, 'undefined')).status, 401)
})

test('a batch of items is upserted row by row, not collapsed into the last one', async () => {
  // D1's bind() returns a new statement; reusing one builder across a map
  // would write the final row N times and silently lose the corpus.
  const e = env()
  const items = [
    { item_id: 'a', card_id: 'c-a', image: 'images/a.jpg' },
    { item_id: 'b', card_id: 'c-b', image: 'images/b.jpg', calibration: true },
  ]
  const res = await call('/api/admin/items', {
    env: e,
    method: 'POST',
    body: JSON.stringify(items),
  })
  assert.equal(res.status, 200)
  // Spread: node:sqlite returns null-prototype rows, which strict deep
  // equality distinguishes from object literals.
  const rows = e.DB.raw
    .prepare('SELECT item_id, card_id, calibration FROM items ORDER BY item_id')
    .all()
    .map((row) => ({ ...row }))
  assert.deepEqual(rows, [
    { item_id: 'a', card_id: 'c-a', calibration: 0 },
    { item_id: 'b', card_id: 'c-b', calibration: 1 },
  ])
})

test('re-pushing an item updates it rather than failing on the primary key', async () => {
  // The corpus grows in acquisition batches, so push runs again over cards
  // that are already there -- including to promote one to calibration.
  const e = env()
  const push = (calibration) =>
    call('/api/admin/items', {
      env: e,
      method: 'POST',
      body: JSON.stringify([{ item_id: 'a', card_id: 'c-a', image: 'images/a.jpg', calibration }]),
    })
  await push(false)
  assert.equal((await push(true)).status, 200)
  assert.equal(e.DB.raw.prepare('SELECT COUNT(*) AS n FROM items').get().n, 1)
  assert.equal(e.DB.raw.prepare('SELECT calibration FROM items').get().calibration, 1)
})

test('a photograph round-trips through the bucket and reports as present', async () => {
  const e = env()
  assert.equal(
    (await (await call('/api/admin/image?key=images/a.jpg', { env: e })).json()).present,
    false,
  )
  await call('/api/admin/image?key=images/a.jpg', { env: e, method: 'PUT', body: 'jpegbytes' })
  assert.equal(
    (await (await call('/api/admin/image?key=images/a.jpg', { env: e })).json()).present,
    true,
  )
})

test('the crops dump parses quads and names the calibration items', async () => {
  // This payload is what sync.pull validates and merges; a quad left as a
  // string would fail far away from here.
  const e = env()
  e.DB.raw.prepare('INSERT INTO items (item_id, card_id, image, calibration) VALUES (?,?,?,1)')
    .run('cal-0', 'c', 'images/c.jpg')
  e.DB.raw.prepare('INSERT INTO crops (item_id, cropper, quad, at) VALUES (?,?,?,?)')
    .run('cal-0', 'a@x.y', '[[0,0],[1,0],[1,1],[0,1]]', 5)
  const payload = await (await call('/api/admin/crops', { env: e })).json()
  assert.deepEqual(payload.calibration, ['cal-0'])
  assert.deepEqual(payload.crops[0].quad, [[0, 0], [1, 0], [1, 1], [0, 1]])
})

test('a human route with no Access assertion is refused, not served', async () => {
  // Admin auth must not leak into the pages that show photographs: the
  // bearer token is a script credential, not a login.
  const res = await worker.fetch(
    new Request('https://crop.axi.link/', { headers: { Authorization: 'Bearer secret' } }),
    env(),
  )
  assert.equal(res.status, 403)
})
