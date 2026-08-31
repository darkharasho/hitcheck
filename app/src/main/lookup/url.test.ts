import { describe, it, expect } from 'vitest'
import {
  priceChartingProductUrl,
  priceChartingSearchUrl,
  tcgplayerSearchUrl,
} from './url'
import type { CardRecord } from './types'

const card = (over: Partial<CardRecord> = {}): CardRecord => ({
  id: 'base1-4',
  name: 'Charizard',
  number: '4',
  setName: 'Base',
  tcgplayerUrl: 'https://prices.pokemontcg.io/tcgplayer/base1-4',
  ...over,
})

describe('priceChartingProductUrl', () => {
  // Every expectation below was confirmed to resolve on the live site
  // (no redirect to /search-products) on 2026-08-31.
  it('builds the verified Base Set Charizard URL', () => {
    expect(priceChartingProductUrl(card())).toBe(
      'https://www.pricecharting.com/game/pokemon-base-set/charizard-4',
    )
  })

  it('builds a modern set URL', () => {
    expect(priceChartingProductUrl(card({
      id: 'swsh4-188', name: 'Pikachu VMAX', number: '188', setName: 'Vivid Voltage',
    }))).toBe(
      'https://www.pricecharting.com/game/pokemon-vivid-voltage/pikachu-vmax-188',
    )
  })

  it('slugifies apostrophes in card names', () => {
    expect(priceChartingProductUrl(card({
      id: 'basep-18', name: 'Team Rocket\u2019s Meowth', number: '18',
      setName: 'Wizards Black Star Promos',
    }))).toBe(
      'https://www.pricecharting.com/game/pokemon-wizards-black-star-promos/team-rockets-meowth-18',
    )
  })

  it('slugifies alphanumeric card numbers', () => {
    expect(priceChartingProductUrl(card({ number: 'SV49' }))).toBe(
      'https://www.pricecharting.com/game/pokemon-base-set/charizard-sv49',
    )
  })

  it('omits the number segment when the card has none', () => {
    expect(priceChartingProductUrl(card({ number: null }))).toBe(
      'https://www.pricecharting.com/game/pokemon-base-set/charizard',
    )
  })
})

describe('priceChartingSearchUrl', () => {
  it('includes name, set and number so set context is not lost', () => {
    expect(priceChartingSearchUrl(card())).toBe(
      'https://www.pricecharting.com/search-products?type=prices&q=Charizard+Base+4',
    )
  })

  it('omits a null number', () => {
    expect(priceChartingSearchUrl(card({ number: null }))).toBe(
      'https://www.pricecharting.com/search-products?type=prices&q=Charizard+Base',
    )
  })
})

describe('tcgplayerSearchUrl', () => {
  it('searches the pokemon product line by name and set', () => {
    expect(tcgplayerSearchUrl(card())).toBe(
      'https://www.tcgplayer.com/search/pokemon/product?q=Charizard+Base',
    )
  })
})
