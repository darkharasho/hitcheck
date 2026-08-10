import { describe, it, expect } from 'vitest'
import { isBlankFrame } from './blank'

const solid = (r: number, g: number, b: number, px = 64) => {
  const out = new Uint8ClampedArray(px * 4)
  for (let i = 0; i < px; i++) out.set([r, g, b, 255], i * 4)
  return out
}

const noisy = (px = 64) => {
  const out = new Uint8ClampedArray(px * 4)
  for (let i = 0; i < px; i++) {
    const v = (i * 37) % 256
    out.set([v, 255 - v, (v * 3) % 256, 255], i * 4)
  }
  return out
}

describe('isBlankFrame', () => {
  it('flags an all-black frame', () => {
    expect(isBlankFrame(solid(0, 0, 0))).toBe(true)
  })

  it('flags an all-white frame — uniform is uniform', () => {
    expect(isBlankFrame(solid(255, 255, 255))).toBe(true)
  })

  it('flags a uniform mid-grey frame', () => {
    expect(isBlankFrame(solid(128, 128, 128))).toBe(true)
  })

  it('does not flag a frame with real variation', () => {
    expect(isBlankFrame(noisy())).toBe(false)
  })

  it('treats an empty buffer as blank rather than throwing', () => {
    expect(isBlankFrame(new Uint8ClampedArray(0))).toBe(true)
  })

  it('respects a custom variance threshold', () => {
    expect(isBlankFrame(noisy(), { varianceThreshold: 1_000_000 })).toBe(true)
  })
})
