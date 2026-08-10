import { describe, it, expect } from 'vitest'
import { BlankRateWindow, formatPercent, isStallGap, isTrackEnded } from './health'

describe('BlankRateWindow', () => {
  it('reports null before any samples are pushed', () => {
    expect(new BlankRateWindow(4).rate()).toBeNull()
  })

  it('computes the rate over pushed samples while under capacity', () => {
    const w = new BlankRateWindow(4)
    w.push(true)
    w.push(false)
    w.push(false)
    expect(w.rate()).toBeCloseTo(1 / 3)
  })

  it('evicts the oldest sample once the window is full', () => {
    const w = new BlankRateWindow(3)
    w.push(true)
    w.push(true)
    w.push(true)
    expect(w.rate()).toBe(1)
    w.push(false) // evicts the first `true`
    w.push(false) // evicts the second `true`
    expect(w.rate()).toBeCloseTo(1 / 3)
  })

  it('rejects a non-positive window size', () => {
    expect(() => new BlankRateWindow(0)).toThrow()
  })
})

describe('formatPercent', () => {
  it('renders n/a for a null rate instead of NaN%', () => {
    expect(formatPercent(null)).toBe('n/a')
  })

  it('rounds to the nearest percent', () => {
    expect(formatPercent(1 / 3)).toBe('33%')
    expect(formatPercent(0)).toBe('0%')
    expect(formatPercent(1)).toBe('100%')
  })
})

describe('isStallGap', () => {
  it('is false for a normal ~60fps gap', () => {
    expect(isStallGap(16)).toBe(false)
  })

  it('is false exactly at the threshold', () => {
    expect(isStallGap(2000)).toBe(false)
  })

  it('is true just past the threshold', () => {
    expect(isStallGap(2001)).toBe(true)
  })

  it('honors a custom threshold', () => {
    expect(isStallGap(600, 500)).toBe(true)
    expect(isStallGap(400, 500)).toBe(false)
  })
})

describe('isTrackEnded', () => {
  it('is true when readyState is "ended"', () => {
    expect(isTrackEnded({ readyState: 'ended' })).toBe(true)
  })

  it('is false when readyState is "live"', () => {
    expect(isTrackEnded({ readyState: 'live' })).toBe(false)
  })
})
