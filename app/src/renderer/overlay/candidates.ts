type CardRecord = {
  id: string
  name: string
  number: string | null
  setName: string
  tcgplayerUrl: string | null
}

export type Candidate = { card: CardRecord; score: number }

export type CandidateView = {
  primary: CardRecord
  confidence: 'confident' | 'uncertain'
  alternates: CardRecord[]
}

const MAX_ALTERNATES = 2

/**
 * Decide what the overlay shows for a set of retrieval candidates.
 *
 * The window navigates to the primary regardless of confidence. That is
 * deliberate and differs from the numeric price path: a wrong page is
 * self-evidently wrong and asserts nothing, whereas a wrong number costs money.
 * The badge tells the user how much to trust it; the alternates let them fix it
 * in one click when the model cannot separate near-identical reprints.
 */
export function formatCandidates(
  candidates: Candidate[],
  threshold: number,
): CandidateView {
  if (candidates.length === 0) {
    throw new Error('formatCandidates requires at least one candidate')
  }
  const ranked = [...candidates].sort((a, b) => b.score - a.score)
  const [best, ...rest] = ranked
  return {
    primary: best.card,
    confidence: best.score >= threshold ? 'confident' : 'uncertain',
    alternates: rest.slice(0, MAX_ALTERNATES).map(c => c.card),
  }
}
