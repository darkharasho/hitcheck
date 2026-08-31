import { describe, it, expect } from 'vitest'
import { routeCard } from './router'
import type { CardRecord } from './types'

const card = (over: Partial<CardRecord> = {}): CardRecord => ({
  id: 'base1-4',
  name: 'Charizard',
  number: '4',
  setName: 'Base',
  tcgplayerUrl: 'https://prices.pokemontcg.io/tcgplayer/base1-4',
  ...over,
})

describe('routeCard', () => {
  it('sends slabs to PriceCharting, product first then search', () => {
    const out = routeCard(card(), 'slab')
    expect(out.map(d => d.kind)).toEqual(['product', 'search'])
    expect(out[0].url).toContain('/game/pokemon-base-set/charizard-4')
    expect(out[1].url).toContain('/search-products')
  })

  it('sends raw singles to the TCGplayer link from the catalog', () => {
    const out = routeCard(card(), 'raw-single')
    expect(out).toEqual([
      { url: 'https://prices.pokemontcg.io/tcgplayer/base1-4', kind: 'product' },
    ])
  })

  it('falls back to TCGplayer search when the card has no link', () => {
    const out = routeCard(card({ tcgplayerUrl: null }), 'raw-single')
    expect(out.map(d => d.kind)).toEqual(['search'])
    expect(out[0].url).toContain('tcgplayer.com/search/pokemon/product')
  })

  it('treats an empty-string link as missing', () => {
    const out = routeCard(card({ tcgplayerUrl: '' }), 'raw-single')
    expect(out.map(d => d.kind)).toEqual(['search'])
  })

  it('always returns at least one destination', () => {
    for (const classification of ['slab', 'raw-single'] as const) {
      expect(routeCard(card({ tcgplayerUrl: null }), classification).length)
        .toBeGreaterThan(0)
    }
  })

  it('is pure \u2014 the same input yields an equal result', () => {
    expect(routeCard(card(), 'slab')).toEqual(routeCard(card(), 'slab'))
  })
})
