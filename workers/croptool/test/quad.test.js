import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { quadError, signedQuadArea } from '../src/quad.js'

const cases = JSON.parse(
  readFileSync(new URL('../quad-cases.json', import.meta.url), 'utf8'),
).cases

test('every shared case gets the verdict the trainer gives it', () => {
  // The same file is read by trainer/tests/test_quad_cases.py. If the two
  // validators ever disagree about the winding contract, one of these two
  // suites fails rather than a mirrored crop entering the corpus silently.
  for (const { why, quad, valid } of cases) {
    assert.equal(quadError(quad) === null, valid, why)
  }
})

test('a screen-clockwise walk is positive in y-down image space', () => {
  // Pins the sign convention empirically rather than by reasoning: getting
  // it backwards would reject exactly the correct corner order.
  const clockwise = [[0, 0], [100, 0], [100, 140], [0, 140]]
  assert.ok(signedQuadArea(clockwise) > 0)
  assert.ok(signedQuadArea([...clockwise].reverse()) < 0)
})

test('the anticlockwise message names the corner order, not the shape', () => {
  // A rectangle marked the wrong way round passes area and self-intersection,
  // so the only useful thing to say is where corner 1 belongs.
  const message = quadError([[0, 140], [100, 140], [100, 0], [0, 0]])
  assert.match(message, /anticlockwise/)
  assert.match(message, /top-left/)
})

test('non-numeric corners are rejected rather than coerced', () => {
  // JSON from a browser is not to be trusted: "10" would multiply into a
  // plausible area and store a quad the trainer then chokes on.
  assert.ok(quadError([['10', 20], [210, 20], [210, 320], [10, 320]]))
  assert.ok(quadError([[NaN, 20], [210, 20], [210, 320], [10, 320]]))
})
