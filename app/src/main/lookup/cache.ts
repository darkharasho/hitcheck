import { stripTracking } from './outcome'
import type { Destination } from './router'
import type { Classification } from './types'

export type ResolvedCache = {
  get(key: string): string | undefined
  set(key: string, url: string): void
  size(): number
}

const DEFAULT_LIMIT = 2000

/**
 * The cache key for one card looked up under one classification.
 *
 * The same card id routes to two different sites depending on how it was
 * classified -- PriceCharting for a slab, TCGplayer for a raw single -- so a
 * key of the card id alone would let a raw-single resolution be replayed as
 * the front destination of a later slab lookup and show a raw price for a
 * graded card. The classification comes first because it is a closed set of
 * two literals while a card id is free-form upstream text; that ordering
 * keeps the prefix unambiguous.
 */
export function cacheKey(cardId: string, classification: Classification): string {
  return `${classification}:${cardId}`
}

/**
 * cache key -> the bare product URL that card actually resolved to.
 *
 * In-memory and bounded. A Map preserves insertion order, so eviction is a
 * single shift of the first key. Eviction is therefore FIFO by first
 * insertion, not LRU: a cache *hit* never re-sets (the hit makes
 * `landed === loaded`, so `shouldCacheResolved` suppresses the write), so
 * nothing in production ever moves an entry back. The `delete` before `set`
 * below is kept only so that a caller which does overwrite an existing key
 * gets the fresher entry ordered last rather than evicted on its old
 * position; it is not, on its own, an LRU policy.
 *
 * Deliberately not persisted: a stale mapping to a delisted product would be
 * worse than one extra redirect, and a session's worth of caching already
 * removes almost all repeat hits.
 */
export function createResolvedCache(limit: number = DEFAULT_LIMIT): ResolvedCache {
  const entries = new Map<string, string>()
  return {
    get: key => entries.get(key),
    set(key, url) {
      entries.delete(key)
      entries.set(key, stripTracking(url))
      if (entries.size > limit) {
        const oldest = entries.keys().next().value
        if (oldest !== undefined) entries.delete(oldest)
      }
    },
    size: () => entries.size,
  }
}

/**
 * Is the landed URL worth caching against the destination that was loaded?
 *
 * Guards against caching a redirector URL that never actually redirected: if
 * `loadURL`'s promise resolves before the third-party redirect settles,
 * `classifyLanding` sees the un-redirected URL and (correctly, by its own
 * rules) calls it a product page. Caching that URL would poison the cache
 * with the exact address we are trying to stop hitting. A plain equality
 * check catches it for free -- no redirect happened, so nothing changed.
 *
 * Deliberately no URL normalization: a trailing-slash-only difference still
 * counts as "differs" and is cached. The window navigated somewhere; treating
 * that as a non-event would need a rule for what counts as equivalent, and
 * that is more surface area than this guard is trying to own.
 */
export function shouldCacheResolved(loadedUrl: string, landedUrl: string): boolean {
  return landedUrl !== loadedUrl
}

/** Put a previously-resolved URL at the front, keeping the rest as fallbacks. */
export function resolvedDestinations(
  key: string,
  destinations: Destination[],
  cache: ResolvedCache,
): Destination[] {
  const cached = cache.get(key)
  return cached ? [{ url: cached, kind: 'product' }, ...destinations] : destinations
}
