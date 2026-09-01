/**
 * Quad validation, for fast feedback -- NOT the authority.
 *
 * The gate that decides what enters the corpus is validate_quad in the
 * trainer, re-run over every quad at pull time. This copy exists so a
 * cropper hears "you went round anticlockwise" while the card is still on
 * screen, instead of a fortnight later when the pull runs.
 *
 * Because it is a second implementation of the same contract, both sides
 * are pinned to ../quad-cases.json by their own test suites. Change the
 * rule in one language and the other language's tests fail.
 */

export const MIN_QUAD_AREA = 1000

/**
 * Shoelace area WITH sign -- the winding discriminator.
 *
 * Image coordinates are y-down, which flips the textbook (y-up) sense of
 * the shoelace sign: a walk that looks clockwise ON SCREEN comes out
 * positive here.
 */
export function signedQuadArea(quad) {
  let sum = 0
  for (let i = 0; i < quad.length; i++) {
    const [x0, y0] = quad[i]
    const [x1, y1] = quad[(i + 1) % quad.length]
    sum += x0 * y1 - y0 * x1
  }
  return sum / 2
}

function cross(o, a, b) {
  return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

// A simple quadrilateral is exactly the one whose two DIAGONALS cross. In a
// bow-tie they do not -- a pair of edges does instead.
function diagonalsCross(p) {
  const d1 = cross(p[0], p[2], p[1]) * cross(p[0], p[2], p[3])
  const d2 = cross(p[1], p[3], p[0]) * cross(p[1], p[3], p[2])
  return d1 < 0 && d2 < 0
}

/** Return an error string, or null when the quad is acceptable. */
export function quadError(quad) {
  if (!Array.isArray(quad) || quad.length !== 4) return 'expected 4 [x, y] points'
  for (const point of quad) {
    if (!Array.isArray(point) || point.length !== 2) return 'expected 4 [x, y] points'
    if (!point.every((n) => typeof n === 'number' && Number.isFinite(n))) {
      return 'corner coordinates must be finite numbers'
    }
  }
  const signed = signedQuadArea(quad)
  if (Math.abs(signed) < MIN_QUAD_AREA) return 'that box is too small to be a card'
  if (!diagonalsCross(quad)) return 'the outline crosses itself — check the corner order'
  if (signed < 0) {
    return 'corners run anticlockwise — corner 1 goes on the card’s top-left, then round'
  }
  return null
}
