import { slugify, priceChartingSetSlug } from './slug'
import type { CardRecord } from './types'

const PRICECHARTING = 'https://www.pricecharting.com'
const TCGPLAYER = 'https://www.tcgplayer.com'

/** Human-readable query terms, in the order a person would type them. */
function queryTerms(card: CardRecord, includeNumber: boolean): string {
  return [card.name, card.setName, includeNumber ? card.number : null]
    .filter((part): part is string => Boolean(part))
    .join(' ')
}

/**
 * Best-guess direct product page. PriceCharting never 404s — a wrong slug
 * returns 200 and redirects to /search-products — so this is always safe to
 * try first. See classifyLanding in ./outcome for how a miss is detected.
 */
export function priceChartingProductUrl(card: CardRecord): string {
  const name = slugify(card.name)
  const tail = card.number ? `${name}-${slugify(card.number)}` : name
  return `${PRICECHARTING}/game/${priceChartingSetSlug(card.setName)}/${tail}`
}

/**
 * Our own search URL, used when the constructed slug misses. Worth having
 * even though PriceCharting auto-searches on a miss: their auto-search derives
 * the query from the slug's trailing segment alone and drops the set name, so
 * a wrong-set slug would search without set context. This keeps it.
 */
export function priceChartingSearchUrl(card: CardRecord): string {
  const params = new URLSearchParams({ type: 'prices', q: queryTerms(card, true) })
  return `${PRICECHARTING}/search-products?${params}`
}

/** Fallback for the 266 catalog cards with no pokemontcg.io TCGplayer link. */
export function tcgplayerSearchUrl(card: CardRecord): string {
  const params = new URLSearchParams({ q: queryTerms(card, false) })
  return `${TCGPLAYER}/search/pokemon/product?${params}`
}
