/**
 * Hosted crop tool: the trainer's local croptool, opened to a few people.
 *
 * The corpus needs several hundred hand-marked cards and the marking is
 * irreducibly manual, so the only way to go faster is more hands. What
 * hosting adds beyond the desktop tool is therefore not features but the
 * three things sharing the work requires:
 *
 *   - identity, so a crop can be attributed and, if need be, withdrawn;
 *   - leases, so two people opening the tool are not handed the same card;
 *   - calibration, so a new cropper's understanding of "the card, not the
 *     slab" is measured against a reference before their work counts.
 *
 * These crops are the ground truth the M2 train/don't-train number is
 * computed from. A quietly wrong crop does not fail anything -- it moves
 * the number. Hence the checks, and hence validate_quad in the trainer
 * re-running over every quad at pull time regardless of what passed here.
 */
import { PAGE } from './page.js'
import { quadError } from './quad.js'
import { authenticate } from './access.js'

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  })

const fail = (message, status = 400) => json({ error: message }, status)

async function body(request) {
  try {
    return await request.json()
  } catch {
    throw new Error('body is not JSON')
  }
}

function adminAuthorised(request, env) {
  const header = request.headers.get('Authorization') || ''
  // Compared with an explicit length check first: an unset ADMIN_TOKEN must
  // not make the empty string a valid credential.
  return Boolean(env.ADMIN_TOKEN) && header === `Bearer ${env.ADMIN_TOKEN}`
}

export async function calibrationState(env, cropper) {
  const target = Number(env.CALIBRATION_N || 5)
  const row = await env.DB.prepare(
    `SELECT
       (SELECT COUNT(*) FROM items WHERE calibration = 1) AS total,
       (SELECT COUNT(*) FROM items i WHERE i.calibration = 1
          AND (EXISTS (SELECT 1 FROM crops c WHERE c.item_id = i.item_id AND c.cropper = ?1)
            OR EXISTS (SELECT 1 FROM skips s WHERE s.item_id = i.item_id AND s.cropper = ?1)))
         AS done`,
  )
    .bind(cropper)
    .first()
  return { total: Math.min(target, row.total), done: row.done }
}

export async function nextItem(env, cropper, now) {
  const calibration = await calibrationState(env, cropper)

  if (calibration.done < calibration.total) {
    const item = await env.DB.prepare(
      `SELECT item_id, card_id, image FROM items i
        WHERE i.calibration = 1
          AND NOT EXISTS (SELECT 1 FROM crops c WHERE c.item_id = i.item_id AND c.cropper = ?1)
          AND NOT EXISTS (SELECT 1 FROM skips s WHERE s.item_id = i.item_id AND s.cropper = ?1)
        ORDER BY i.item_id LIMIT 1`,
    )
      .bind(cropper)
      .first()
    // Calibration cards are not claimed: everyone is meant to mark the same
    // ones, so a lease would starve the second person to arrive.
    if (item) return { item, calibration }
  }

  const until = now + Number(env.CLAIM_MINUTES || 10) * 60_000
  // Claim and hand out in one statement. Two statements would leave a
  // window in which two croppers both read the same unclaimed row.
  const item = await env.DB.prepare(
    `UPDATE items SET claimed_by = ?1, claimed_until = ?2
      WHERE item_id = (
        SELECT i.item_id FROM items i
         WHERE i.calibration = 0
           AND (i.claimed_until IS NULL OR i.claimed_until < ?3 OR i.claimed_by = ?1)
           AND NOT EXISTS (SELECT 1 FROM crops c WHERE c.item_id = i.item_id)
           AND NOT EXISTS (SELECT 1 FROM skips s WHERE s.item_id = i.item_id)
         ORDER BY (i.claimed_by = ?1) DESC, i.item_id
         LIMIT 1)
      RETURNING item_id, card_id, image`,
  )
    .bind(cropper, until, now)
    .first()
  return { item, calibration }
}

export async function progress(env) {
  return await env.DB.prepare(
    `SELECT
       (SELECT COUNT(*) FROM items WHERE calibration = 0) AS total,
       -- DISTINCT: a lease can expire mid-card, leaving two people's quads
       -- on one corpus card. Counting rows would push done past total.
       (SELECT COUNT(DISTINCT c.item_id) FROM crops c JOIN items i ON i.item_id = c.item_id
         WHERE i.calibration = 0) AS done`,
  ).first()
}

async function handleHuman(request, url, env, cropper, now) {
  if (request.method === 'GET' && url.pathname === '/') {
    return new Response(PAGE, {
      headers: { 'content-type': 'text/html; charset=utf-8' },
    })
  }

  if (request.method === 'GET' && url.pathname === '/api/next') {
    const [{ item, calibration }, counts] = await Promise.all([
      nextItem(env, cropper, now),
      progress(env),
    ])
    return json({
      item_id: item?.item_id ?? null,
      card_id: item?.card_id ?? null,
      image: item?.image ?? null,
      calibration: Boolean(item) && calibration.done < calibration.total,
      calibration_done: calibration.done + 1,
      calibration_total: calibration.total,
      cropper,
      done: counts.done,
      total: counts.total,
    })
  }

  if (request.method === 'GET' && url.pathname === '/api/image') {
    const row = await env.DB.prepare('SELECT image FROM items WHERE item_id = ?1')
      .bind(url.searchParams.get('id') || '')
      .first()
    if (!row) return fail('no such image', 404)
    const object = await env.PHOTOS.get(row.image)
    if (!object) return fail('photograph was never uploaded', 404)
    return new Response(object.body, {
      headers: {
        'content-type': 'image/jpeg',
        // private: these are other people's listing photographs behind an
        // Access login, and must not be cached by anything shared.
        'cache-control': 'private, max-age=3600',
      },
    })
  }

  if (request.method === 'GET' && url.pathname === '/api/heartbeat') {
    await env.DB.prepare(
      `UPDATE items SET claimed_until = ?2
        WHERE item_id = ?1 AND claimed_by = ?3 AND calibration = 0`,
    )
      .bind(
        url.searchParams.get('id') || '',
        now + Number(env.CLAIM_MINUTES || 10) * 60_000,
        cropper,
      )
      .run()
    return json({ ok: true })
  }

  if (request.method === 'POST' && url.pathname === '/api/quad') {
    const payload = await body(request)
    const problem = quadError(payload.quad)
    if (problem) return fail(problem)
    const known = await env.DB.prepare('SELECT 1 FROM items WHERE item_id = ?1')
      .bind(payload.item_id)
      .first()
    if (!known) return fail('no such item', 404)
    // Upsert, not insert: re-marking a card you already did is a correction,
    // and should replace your quad rather than fail on the primary key.
    await env.DB.prepare(
      `INSERT INTO crops (item_id, cropper, quad, at) VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT (item_id, cropper) DO UPDATE SET quad = ?3, at = ?4`,
    )
      .bind(payload.item_id, cropper, JSON.stringify(payload.quad), now)
      .run()
    return json({ ok: true })
  }

  if (request.method === 'POST' && url.pathname === '/api/skip') {
    const payload = await body(request)
    await env.DB.prepare(
      `INSERT INTO skips (item_id, cropper, at) VALUES (?1, ?2, ?3)
         ON CONFLICT (item_id, cropper) DO NOTHING`,
    )
      .bind(payload.item_id, cropper, now)
      .run()
    return json({ ok: true })
  }

  return fail('not found', 404)
}

async function handleAdmin(request, url, env) {
  if (!adminAuthorised(request, env)) return fail('unauthorised', 401)

  if (request.method === 'POST' && url.pathname === '/api/admin/items') {
    const items = await body(request)
    if (!Array.isArray(items)) return fail('expected a list of items')
    const statement = env.DB.prepare(
      `INSERT INTO items (item_id, card_id, image, calibration) VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT (item_id) DO UPDATE SET
           card_id = ?2, image = ?3, calibration = ?4`,
    )
    await env.DB.batch(
      items.map((i) =>
        statement.bind(i.item_id, i.card_id, i.image, i.calibration ? 1 : 0),
      ),
    )
    return json({ ok: true, upserted: items.length })
  }

  if (request.method === 'PUT' && url.pathname === '/api/admin/image') {
    const key = url.searchParams.get('key')
    if (!key) return fail('missing key')
    await env.PHOTOS.put(key, request.body)
    return json({ ok: true, key })
  }

  if (request.method === 'GET' && url.pathname === '/api/admin/image') {
    // Lets the push script skip photographs already uploaded, so a re-push
    // after another acquisition batch costs one HEAD-shaped call per known
    // image instead of re-sending eighty megabytes.
    const key = url.searchParams.get('key')
    const object = key ? await env.PHOTOS.head(key) : null
    return json({ present: Boolean(object) })
  }

  if (request.method === 'GET' && url.pathname === '/api/admin/crops') {
    const [crops, skips, items] = await Promise.all([
      env.DB.prepare('SELECT item_id, cropper, quad, at FROM crops').all(),
      env.DB.prepare('SELECT item_id, cropper, at FROM skips').all(),
      env.DB.prepare('SELECT item_id, calibration FROM items').all(),
    ])
    return json({
      crops: crops.results.map((c) => ({ ...c, quad: JSON.parse(c.quad) })),
      skips: skips.results,
      calibration: items.results.filter((i) => i.calibration).map((i) => i.item_id),
    })
  }

  return fail('not found', 404)
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url)

    if (url.pathname.startsWith('/api/admin/')) {
      try {
        return await handleAdmin(request, url, env)
      } catch (error) {
        return fail(String(error.message || error), 400)
      }
    }

    let cropper
    try {
      cropper = await authenticate(request, env)
    } catch (error) {
      // 403 rather than 401: Access has already done the challenge, so a
      // failure here is a misconfiguration or a forged header, and a
      // browser retry would loop rather than help.
      return new Response(`not authorised: ${error.message}`, { status: 403 })
    }

    try {
      return await handleHuman(request, url, env, cropper, Date.now())
    } catch (error) {
      return fail(String(error.message || error), 400)
    }
  },
}
