import { describe, it, expect } from 'vitest'
import { FrameSampler } from './sampler'

describe('FrameSampler', () => {
  it('samples the very first frame', () => {
    expect(new FrameSampler(15).shouldSample(0)).toBe(true)
  })

  it('rejects a frame arriving before the interval elapses', () => {
    const s = new FrameSampler(10) // 100ms interval
    s.shouldSample(1000)
    expect(s.shouldSample(1050)).toBe(false)
  })

  it('accepts a frame once the interval has elapsed', () => {
    const s = new FrameSampler(10)
    s.shouldSample(1000)
    expect(s.shouldSample(1100)).toBe(true)
  })

  it('does not drift — spacing is measured from the accepted frame', () => {
    const s = new FrameSampler(10)
    s.shouldSample(1000)
    s.shouldSample(1150) // accepted
    expect(s.shouldSample(1200)).toBe(false)
    expect(s.shouldSample(1250)).toBe(true)
  })

  it('treats any fps <= 0 as "sample everything"', () => {
    const s = new FrameSampler(0)
    expect(s.shouldSample(0)).toBe(true)
    expect(s.shouldSample(1)).toBe(true)
  })
})
