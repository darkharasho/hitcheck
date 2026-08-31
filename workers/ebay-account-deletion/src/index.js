/**
 * eBay Marketplace Account Deletion notification endpoint.
 *
 * eBay will not authenticate a production keyset until the application
 * points at a reachable HTTPS endpoint for account-deletion notifications.
 * HitCheck reads public listing images and card metadata and stores no eBay
 * user data, so there is nothing here to delete on request -- this endpoint
 * exists to satisfy the requirement, not to do work.
 *
 * Routes:
 *   GET  /ebay/account-deletion?challenge_code=X -> 200 {"challengeResponse": ...}
 *   POST /ebay/account-deletion                  -> 200, body discarded
 *   anything else                                -> 404
 *
 * Deliberately no signature verification on the POST. eBay signs
 * notifications with a key fetched from their Notification API, which needs
 * an OAuth token -- the very thing this endpoint exists to unblock. More to
 * the point, the handler stores nothing and deletes nothing, so there is no
 * state for a forged notification to corrupt. Verifying a signature to
 * protect a no-op would be ceremony, not security.
 */
import { challengeResponse } from './challenge.js'

const PATH = '/ebay/account-deletion'

export default {
  async fetch(request, env) {
    const url = new URL(request.url)
    if (url.pathname !== PATH) {
      return new Response('not found', { status: 404 })
    }

    if (request.method === 'GET') {
      const code = url.searchParams.get('challenge_code')
      if (!code) {
        return new Response('missing challenge_code', { status: 400 })
      }
      // Fail loudly rather than hashing the string "undefined" into a
      // response that looks fine here and fails opaquely on eBay's side.
      if (!env.EBAY_VERIFICATION_TOKEN || !env.EBAY_ENDPOINT_URL) {
        return new Response('endpoint not configured', { status: 500 })
      }
      const hash = await challengeResponse(
        code,
        env.EBAY_VERIFICATION_TOKEN,
        env.EBAY_ENDPOINT_URL,
      )
      return new Response(JSON.stringify({ challengeResponse: hash }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    }

    if (request.method === 'POST') {
      // Acknowledged and dropped: nothing is stored, so nothing is deleted.
      return new Response(null, { status: 200 })
    }

    return new Response('method not allowed', {
      status: 405,
      headers: { allow: 'GET, POST' },
    })
  },
}
