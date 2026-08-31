import {
  priceChartingProductUrl,
  priceChartingSearchUrl,
  tcgplayerSearchUrl,
} from './url'
import type { CardRecord, Classification } from './types'

export type Destination = { url: string; kind: 'product' | 'search' }

/**
 * Where to send the lookup window, best guess first.
 *
 * Deliberately pure: it cannot ask the network whether a slug resolves, so it
 * emits an ordered list and lets the window walk it. That keeps the
 * churn-prone slug rules under fast unit tests, at the cost of the router not
 * learning what actually happened.
 */
export function routeCard(
  card: CardRecord,
  classification: Classification,
): Destination[] {
  if (classification === 'slab') {
    return [
      { url: priceChartingProductUrl(card), kind: 'product' },
      { url: priceChartingSearchUrl(card), kind: 'search' },
    ]
  }
  return card.tcgplayerUrl
    ? [{ url: card.tcgplayerUrl, kind: 'product' }]
    : [{ url: tcgplayerSearchUrl(card), kind: 'search' }]
}
