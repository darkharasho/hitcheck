import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { challengeResponse } from './challenge.js'

const sha = (s) => createHash('sha256').update(s, 'utf8').digest('hex')

test('concatenates code + token + endpoint in that exact order', async () => {
  const got = await challengeResponse('CODE', 'TOKEN', 'https://e/p')
  assert.equal(got, sha('CODETOKENhttps://e/p'))
  // Order matters to eBay; a permuted concatenation must not collide.
  assert.notEqual(got, sha('TOKENCODEhttps://e/p'))
})

test('returns lowercase hex of the full 32-byte digest', async () => {
  const got = await challengeResponse('a', 'b', 'c')
  assert.match(got, /^[0-9a-f]{64}$/)
})

test('a trailing slash on the endpoint changes the hash', async () => {
  // The failure mode this guards: a URL that differs from the registered one
  // by a single character still produces a valid-looking response.
  const bare = await challengeResponse('x', 'y', 'https://axi.link/ebay/account-deletion')
  const slash = await challengeResponse('x', 'y', 'https://axi.link/ebay/account-deletion/')
  assert.notEqual(bare, slash)
})

test('handles non-ASCII input as UTF-8', async () => {
  assert.equal(await challengeResponse('nñ', 'tök', 'https://e/é'),
               sha('nñtökhttps://e/é'))
})

test('an empty challenge code still hashes deterministically', async () => {
  assert.equal(await challengeResponse('', 'T', 'U'), sha('TU'))
})
