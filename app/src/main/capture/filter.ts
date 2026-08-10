import type { CaptureSource } from './types'

const STREAM_HINTS = ['ebay', 'tiktok', 'live', 'stream', 'whatnot']

function score(source: CaptureSource): number {
  const name = source.name.toLowerCase()
  const hinted = STREAM_HINTS.some(hint => name.includes(hint)) ? 0 : 1
  const kindRank = source.kind === 'window' ? 0 : 1
  return kindRank * 10 + hinted
}

/**
 * Remove HitCheck's own windows and order the rest so likely stream
 * windows surface first. Stable within equal scores.
 */
export function rankSources(sources: CaptureSource[], selfTitle: string): CaptureSource[] {
  const self = selfTitle.toLowerCase()
  return sources
    .filter(s => !s.name.toLowerCase().includes(self))
    .map((s, i) => ({ s, i }))
    .sort((a, b) => score(a.s) - score(b.s) || a.i - b.i)
    .map(({ s }) => s)
}
