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
  cardId: string,
  destinations: Destination[],
  cache: ResolvedCache,
): Destination[] {
  const cached = cache.get(cardId)
  return cached ? [{ url: cached, kind: 'product' }, ...destinations] : destinations
}
