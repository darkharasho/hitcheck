import { describe, it, expect } from 'vitest'
import { createResolvedCache, resolvedDestinations } from './cache'
import type { Destination } from './router'

const dest = (url: string, kind: Destination['kind'] = 'product'): Destination =>
  ({ url, kind })

describe('createResolvedCache', () => {
  it('returns undefined for an unseen card', () => {
    expect(createResolvedCache().get('base1-4')).toBeUndefined()
  })

  it('stores and returns a resolved URL', () => {
    const cache = createResolvedCache()
    cache.set('base1-4', 'https://www.tcgplayer.com/product/42382')
    expect(cache.get('base1-4')).toBe('https://www.tcgplayer.com/product/42382')
  })

  it('strips tracking parameters on the way in', () => {
    const cache = createResolvedCache()
    cache.set('base1-4', 'https://www.tcgplayer.com/product/42382?utm_campaign=Scrydex')
    expect(cache.get('base1-4')).toBe('https://www.tcgplayer.com/product/42382')
  })

  it('evicts the oldest entry past the limit', () => {
    const cache = createResolvedCache(2)
    cache.set('a', 'https://x/1')
    cache.set('b', 'https://x/2')
    cache.set('c', 'https://x/3')
    expect(cache.get('a')).toBeUndefined()
    expect(cache.get('c')).toBe('https://x/3')
    expect(cache.size()).toBe(2)
  })

  it('re-setting an existing card does not grow the cache', () => {
    const cache = createResolvedCache(2)
    cache.set('a', 'https://x/1')
    cache.set('a', 'https://x/2')
    expect(cache.size()).toBe(1)
    expect(cache.get('a')).toBe('https://x/2')
  })
})

describe('resolvedDestinations', () => {
  it('prepends the cached URL when one exists', () => {
    const cache = createResolvedCache()
    cache.set('base1-4', 'https://www.tcgplayer.com/product/42382')
    const out = resolvedDestinations(
      'base1-4', [dest('https://prices.pokemontcg.io/tcgplayer/base1-4')], cache,
    )
    expect(out.map(d => d.url)).toEqual([
      'https://www.tcgplayer.com/product/42382',
      'https://prices.pokemontcg.io/tcgplayer/base1-4',
    ])
  })

  it('returns the destinations untouched on a cache miss', () => {
    const given = [dest('https://prices.pokemontcg.io/tcgplayer/base1-4')]
    expect(resolvedDestinations('base1-4', given, createResolvedCache()))
      .toEqual(given)
  })

  it('does not mutate the destinations it was given', () => {
    const cache = createResolvedCache()
    cache.set('base1-4', 'https://www.tcgplayer.com/product/42382')
    const given = [dest('https://prices.pokemontcg.io/tcgplayer/base1-4')]
    resolvedDestinations('base1-4', given, cache)
    expect(given).toHaveLength(1)
  })
})
