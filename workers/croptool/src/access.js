/**
 * Cloudflare Access identity, verified rather than trusted.
 *
 * Access puts the authenticated email in a header, and reading that header
 * is the tempting one-liner. It is only meaningful if Access is actually in
 * front of the worker: a route with no Access application still receives
 * whatever headers the client chose to send, so header-trusting code is
 * wide open exactly in the case that matters -- a misconfigured deploy that
 * looks like it is working.
 *
 * So the signed assertion is verified instead, against the team's public
 * keys, checking signature, audience and expiry. Missing configuration is
 * an error, not a fallback to open.
 */

const CERTS_TTL_MS = 60 * 60 * 1000

// Module scope, so the fetch is amortised across requests on a warm isolate
// rather than run per photograph.
let cachedCerts = null

function base64UrlToBytes(value) {
  const padded = value.replace(/-/g, '+').replace(/_/g, '/')
  const binary = atob(padded + '='.repeat((4 - (padded.length % 4)) % 4))
  return Uint8Array.from(binary, (c) => c.charCodeAt(0))
}

function base64UrlToJson(value) {
  return JSON.parse(new TextDecoder().decode(base64UrlToBytes(value)))
}

export function parseJwt(token) {
  const parts = token.split('.')
  if (parts.length !== 3) throw new Error('malformed token')
  return {
    header: base64UrlToJson(parts[0]),
    payload: base64UrlToJson(parts[1]),
    signature: base64UrlToBytes(parts[2]),
    signed: new TextEncoder().encode(parts[0] + '.' + parts[1]),
  }
}

async function teamKeys(teamDomain, now, fetchImpl = fetch) {
  if (cachedCerts && cachedCerts.domain === teamDomain && cachedCerts.expires > now) {
    return cachedCerts.keys
  }
  const res = await fetchImpl(`https://${teamDomain}/cdn-cgi/access/certs`)
  if (!res.ok) throw new Error(`access certs unavailable (${res.status})`)
  const { keys } = await res.json()
  cachedCerts = { domain: teamDomain, keys, expires: now + CERTS_TTL_MS }
  return keys
}

/**
 * Return the authenticated email, or throw.
 *
 * `env.ACCESS_TEAM_DOMAIN` and `env.ACCESS_AUD` are both mandatory. Treating
 * an absent audience as "skip the check" would turn a forgotten variable
 * into a public bucket of other people's listing photographs.
 */
export async function authenticate(request, env, { now = Date.now(), fetchImpl = fetch } = {}) {
  if (!env.ACCESS_TEAM_DOMAIN || !env.ACCESS_AUD) {
    throw new Error('Access is not configured on this worker')
  }
  const token =
    request.headers.get('Cf-Access-Jwt-Assertion') ||
    (request.headers.get('Cookie') || '').match(/(?:^|;\s*)CF_Authorization=([^;]+)/)?.[1]
  if (!token) throw new Error('no Access assertion')

  const { header, payload, signature, signed } = parseJwt(token)
  const audiences = Array.isArray(payload.aud) ? payload.aud : [payload.aud]
  if (!audiences.includes(env.ACCESS_AUD)) throw new Error('assertion is for another application')
  if (!payload.exp || payload.exp * 1000 <= now) throw new Error('assertion has expired')

  const jwk = (await teamKeys(env.ACCESS_TEAM_DOMAIN, now, fetchImpl)).find(
    (k) => k.kid === header.kid,
  )
  if (!jwk) throw new Error('assertion signed by an unknown key')

  const key = await crypto.subtle.importKey(
    'jwk',
    jwk,
    { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' },
    false,
    ['verify'],
  )
  if (!(await crypto.subtle.verify('RSASSA-PKCS1-v1_5', key, signature, signed))) {
    throw new Error('assertion signature does not verify')
  }

  const email = payload.email || payload.common_name
  if (!email) throw new Error('assertion carries no identity')
  return email
}

export function resetCertCache() {
  cachedCerts = null
}
