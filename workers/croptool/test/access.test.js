import { test } from 'node:test'
import assert from 'node:assert/strict'
import { authenticate, parseJwt, resetCertCache } from '../src/access.js'

const b64url = (value) =>
  Buffer.from(JSON.stringify(value)).toString('base64url')

const token = (payload, kid = 'k1') =>
  `${b64url({ alg: 'RS256', kid })}.${b64url(payload)}.${Buffer.from('sig').toString('base64url')}`

const request = (headers = {}) => new Request('https://crop.axi.link/', { headers })

const env = { ACCESS_TEAM_DOMAIN: 'team.cloudflareaccess.com', ACCESS_AUD: 'aud-1' }

test('a token decodes into its header and payload', () => {
  const { header, payload } = parseJwt(token({ email: 'a@b.c', aud: 'aud-1' }))
  assert.equal(header.kid, 'k1')
  assert.equal(payload.email, 'a@b.c')
})

test('missing Access configuration is an error, not an open door', async () => {
  // The failure this guards is a deploy where the Access application was
  // never attached: the header check would pass trivially and every
  // photograph would be public.
  await assert.rejects(
    authenticate(request({ 'Cf-Access-Jwt-Assertion': token({ email: 'a@b.c' }) }), {}),
    /not configured/,
  )
})

test('a request with no assertion is rejected', async () => {
  await assert.rejects(authenticate(request(), env), /no Access assertion/)
})

test('an assertion for another application is rejected', async () => {
  // Access tokens are per-application. Without the audience check, a token
  // minted for any other app on the same team would authenticate here.
  const assertion = token({ email: 'a@b.c', aud: 'aud-2', exp: 4e9 })
  await assert.rejects(
    authenticate(request({ 'Cf-Access-Jwt-Assertion': assertion }), env),
    /another application/,
  )
})

test('an expired assertion is rejected before any network call', async () => {
  resetCertCache()
  const assertion = token({ email: 'a@b.c', aud: 'aud-1', exp: 1000 })
  await assert.rejects(
    authenticate(request({ 'Cf-Access-Jwt-Assertion': assertion }), env, {
      now: 2_000_000,
      fetchImpl: () => assert.fail('expiry must be checked without fetching certs'),
    }),
    /expired/,
  )
})

test('an assertion signed by an unknown key is rejected', async () => {
  resetCertCache()
  const assertion = token({ email: 'a@b.c', aud: 'aud-1', exp: 4e9 }, 'other-kid')
  await assert.rejects(
    authenticate(request({ 'Cf-Access-Jwt-Assertion': assertion }), env, {
      fetchImpl: async () => new Response(JSON.stringify({ keys: [{ kid: 'k1' }] })),
    }),
    /unknown key/,
  )
})

test('the cookie carries the assertion when the header does not', async () => {
  // Access sets CF_Authorization on the browser; the header is only present
  // on the first hop. Reading just the header would 403 every page reload.
  resetCertCache()
  const assertion = token({ email: 'a@b.c', aud: 'aud-2', exp: 4e9 })
  await assert.rejects(
    authenticate(request({ Cookie: `CF_Authorization=${assertion}; other=1` }), env),
    /another application/,
  )
})
