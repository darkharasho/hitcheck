import { stripTracking } from './outcome'
import type { Destination } from './router'

export type ResolvedCache = {
  get(cardId: string): string | undefined
  set(cardId: string, url: string): void
  size(): number
}

const DEFAULT_LIMIT = 2000

/**
 * card id -> the bare product URL that card actually resolved to.
 *
 * In-memory and bounded. A Map preserves insertion order, so the first key is
 * the oldest and eviction is a single shift. Deliberately not persisted: a
 * stale mapping to a delisted product would be worse than one extra redirect,
 * and a session's worth of caching already removes almost all repeat hits.
 */
export function createResolvedCache(limit: number = DEFAULT_LIMIT): ResolvedCache {
  const entries = new Map<string, string>()
  return {
    get: cardId => entries.get(cardId),
    set(cardId, url) {
      entries.delete(cardId)
      entries.set(cardId, stripTracking(url))
      if (entries.size > limit) {
        const oldest = entries.keys().next().value
        if (oldest !== undefined) entries.delete(oldest)
      }
    },
    size: () => entries.size,
  }
}

/** Put a previously-resolved URL at the front, keeping the rest as fallbacks. */
export function resolvedDestinations(
  cardId: string,
  destinations: Destination[],
  cache: ResolvedCache,
): Destination[] {
  const cached = cache.get(cardId)
  return cached ? [{ url: cached, kind: 'product' }, ...destinations] : destinations
}
