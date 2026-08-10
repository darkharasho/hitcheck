import { describe, it, expect } from 'vitest'
import { computeOverlayBounds } from './bounds'

describe('computeOverlayBounds', () => {
  it('returns the source rect when inset is zero', () => {
    expect(computeOverlayBounds({ x: 10, y: 20, width: 800, height: 600 }, 0))
      .toEqual({ x: 10, y: 20, width: 800, height: 600 })
  })

  it('shrinks the rect symmetrically by the inset', () => {
    expect(computeOverlayBounds({ x: 0, y: 0, width: 100, height: 100 }, 10))
      .toEqual({ x: 10, y: 10, width: 80, height: 80 })
  })

  it('clamps to a minimum 1x1 rather than going negative', () => {
    expect(computeOverlayBounds({ x: 0, y: 0, width: 10, height: 10 }, 50))
      .toEqual({ x: 5, y: 5, width: 1, height: 1 })
  })
})
