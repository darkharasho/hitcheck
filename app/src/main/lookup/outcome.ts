export type LoadOutcome = 'product' | 'search-fallback'

/** Path prefixes that mean "we did not land on a specific product". */
const SEARCH_PATHS = ['/search-products', '/search/']

const TRACKING_PARAMS = [
  'irclickid', 'sharedid', 'irpid', 'irgwc', 'afsrc',
  'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
]

/**
 * What did the window actually land on?
 *
 * Matches on path only. A near-miss PriceCharting slug resolves to the right
 * product page but keeps a ?q= parameter, and that is a hit -- keying off the
 * query string would misreport it as a fallback.
 */
export function classifyLanding(finalUrl: string): LoadOutcome {
  let path: string
  try {
    path = new URL(finalUrl).pathname
  } catch {
    return 'search-fallback'
  }
  return SEARCH_PATHS.some(p => path.startsWith(p)) ? 'search-fallback' : 'product'
}

/**
 * Drop affiliate and campaign parameters. The pokemontcg.io redirector appends
 * a third party's tracking to every TCGplayer landing; the bare product URL
 * works without it and is what gets cached.
 */
export function stripTracking(url: string): string {
  let parsed: URL
  try {
    parsed = new URL(url)
  } catch {
    return url
  }
  for (const param of TRACKING_PARAMS) parsed.searchParams.delete(param)
  // Drop a now-empty '?' so cached URLs compare equal to hand-written ones.
  if (![...parsed.searchParams.keys()].length) parsed.search = ''
  return parsed.toString()
}
