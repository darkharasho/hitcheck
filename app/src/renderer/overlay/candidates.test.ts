import { describe, it, expect } from 'vitest'
import { formatCandidates } from './candidates'
import type { Candidate } from './candidates'

const cand = (id: string, name: string, score: number): Candidate => ({
  card: { id, name, number: '4', setName: 'Base', tcgplayerUrl: null },
  score,
})

describe('formatCandidates', () => {
  it('promotes the highest-scoring candidate to primary', () => {
    const view = formatCandidates(
      [cand('a', 'Alakazam', 0.4), cand('b', 'Charizard', 0.9)], 0.7,
    )
    expect(view.primary.name).toBe('Charizard')
  })

  it('marks a score above the threshold as confident', () => {
    expect(formatCandidates([cand('b', 'Charizard', 0.9)], 0.7).confidence)
      .toBe('confident')
  })

  it('marks a score below the threshold as uncertain', () => {
    expect(formatCandidates([cand('b', 'Charizard', 0.5)], 0.7).confidence)
      .toBe('uncertain')
  })

  it('treats a score exactly at the threshold as confident', () => {
    expect(formatCandidates([cand('b', 'Charizard', 0.7)], 0.7).confidence)
      .toBe('confident')
  })

  it('offers the runners-up as alternates, best first', () => {
    const view = formatCandidates(
      [cand('a', 'Abra', 0.3), cand('c', 'Charizard', 0.9), cand('b', 'Blastoise', 0.6)],
      0.7,
    )
    expect(view.alternates.map(c => c.name)).toEqual(['Blastoise', 'Abra'])
  })

  it('caps alternates at two', () => {
    const view = formatCandidates(
      [cand('a', 'A', 0.9), cand('b', 'B', 0.8), cand('c', 'C', 0.7), cand('d', 'D', 0.6)],
      0.7,
    )
    expect(view.alternates).toHaveLength(2)
  })

  it('returns no alternates for a single candidate', () => {
    expect(formatCandidates([cand('b', 'Charizard', 0.9)], 0.7).alternates).toEqual([])
  })

  it('throws on an empty candidate list rather than inventing a primary', () => {
    expect(() => formatCandidates([], 0.7)).toThrow(/at least one candidate/)
  })
})
