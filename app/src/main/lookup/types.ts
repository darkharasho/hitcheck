/** The subset of a catalog row the lookup layer needs to build a URL. */
export type CardRecord = {
  id: string
  name: string
  /** pokemontcg.io card number, e.g. "4" or "SV49". Null for cards without one. */
  number: string | null
  setName: string
  /** pokemontcg.io redirector URL, or null for the 266 cards lacking one. */
  tcgplayerUrl: string | null
}

/**
 * What kind of object the detector found. Sealed products are M4.5 and are
 * deliberately absent — adding them here is a signal that the sealed plan has
 * started.
 */
export type Classification = 'slab' | 'raw-single'
