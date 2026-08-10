import { describe, it, expect } from 'vitest'
import { rankSources } from './filter'
import type { CaptureSource } from './types'

const src = (id: string, name: string, kind: 'window' | 'screen' = 'window'): CaptureSource =>
  ({ id, name, kind })

describe('rankSources', () => {
  it('drops HitCheck\'s own windows', () => {
    const out = rankSources([src('1', 'HitCheck'), src('2', 'Firefox')], 'HitCheck')
    expect(out.map(s => s.name)).toEqual(['Firefox'])
  })

  it('ranks likely stream windows above unrelated ones', () => {
    const out = rankSources(
      [src('1', 'Terminal'), src('2', 'eBay Live — Chromium'), src('3', 'Files')],
      'HitCheck',
    )
    expect(out[0].name).toBe('eBay Live — Chromium')
  })

  it('matches stream keywords case-insensitively', () => {
    const out = rankSources([src('1', 'Notes'), src('2', 'tiktok live')], 'HitCheck')
    expect(out[0].name).toBe('tiktok live')
  })

  it('places screens after windows', () => {
    const out = rankSources([src('1', 'Screen 1', 'screen'), src('2', 'Firefox')], 'HitCheck')
    expect(out.map(s => s.kind)).toEqual(['window', 'screen'])
  })

  it('returns an empty array when everything is filtered out', () => {
    expect(rankSources([src('1', 'HitCheck')], 'HitCheck')).toEqual([])
  })
})
