import { describe, it, expect } from 'vitest'
import { classifyLanding, stripTracking } from './outcome'

describe('classifyLanding', () => {
  it('recognises a PriceCharting product page', () => {
    expect(classifyLanding(
      'https://www.pricecharting.com/game/pokemon-base-set/charizard-4',
    )).toBe('product')
  })

  it('recognises the search page PriceCharting redirects misses to', () => {
    expect(classifyLanding(
      'https://www.pricecharting.com/search-products?type=prices&q=weedle+1',
    )).toBe('search-fallback')
  })

  // Measured 2026-08-31: a near-miss slug lands on the right product page but
  // keeps a ?q= parameter. That is a hit, not a fallback -- match on path only.
  it('treats a product page carrying a q parameter as a product page', () => {
    expect(classifyLanding(
      'https://www.pricecharting.com/game/pokemon-vivid-voltage/pikachu-vmax-188?q=pikachu',
    )).toBe('product')
  })

  it('recognises a TCGplayer search page', () => {
    expect(classifyLanding(
      'https://www.tcgplayer.com/search/pokemon/product?q=Charizard+Base',
    )).toBe('search-fallback')
  })

  it('recognises a TCGplayer product page', () => {
    expect(classifyLanding('https://www.tcgplayer.com/product/42382')).toBe('product')
  })

  it('treats an unparseable URL as a fallback rather than throwing', () => {
    expect(classifyLanding('not a url')).toBe('search-fallback')
  })
})

describe('stripTracking', () => {
  it('removes the affiliate parameters the pokemontcg.io redirector adds', () => {
    expect(stripTracking(
      'https://www.tcgplayer.com/product/42382?irclickid=abc&sharedid=&irpid=4944541'
      + '&irgwc=1&afsrc=1&utm_source=impact&utm_medium=affiliate&utm_campaign=Scrydex',
    )).toBe('https://www.tcgplayer.com/product/42382')
  })

  it('leaves a URL with no tracking parameters untouched', () => {
    expect(stripTracking('https://www.tcgplayer.com/product/42382'))
      .toBe('https://www.tcgplayer.com/product/42382')
  })

  it('keeps meaningful parameters while dropping tracking ones', () => {
    expect(stripTracking(
      'https://www.pricecharting.com/search-products?type=prices&q=x&utm_source=impact',
    )).toBe('https://www.pricecharting.com/search-products?type=prices&q=x')
  })

  it('returns an unparseable URL unchanged', () => {
    expect(stripTracking('not a url')).toBe('not a url')
  })
})
