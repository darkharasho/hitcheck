import { describe, it, expect } from 'vitest'
import { slugify, priceChartingSetSlug } from './slug'

describe('slugify', () => {
  it('lowercases and hyphenates', () => {
    expect(slugify('Vivid Voltage')).toBe('vivid-voltage')
  })

  it('drops apostrophes rather than hyphenating them', () => {
    expect(slugify("Team Rocket's Meowth")).toBe('team-rockets-meowth')
  })

  it('drops typographic apostrophes too', () => {
    expect(slugify("Farfetch'd")).toBe('farfetchd')
  })

  it('strips accents', () => {
    expect(slugify('Flabébé')).toBe('flabebe')
  })

  it('spells out ampersands', () => {
    expect(slugify('Scarlet & Violet')).toBe('scarlet-and-violet')
  })

  it('collapses runs of punctuation into a single hyphen', () => {
    expect(slugify('Hidden Fates: Shiny Vault')).toBe('hidden-fates-shiny-vault')
  })

  it('never leaves leading or trailing hyphens', () => {
    expect(slugify('  Fossil!  ')).toBe('fossil')
  })

  it('returns an empty string for input with no alphanumerics', () => {
    expect(slugify('!!!')).toBe('')
  })
})

describe('priceChartingSetSlug', () => {
  it('prefixes the slugified set name', () => {
    expect(priceChartingSetSlug('Vivid Voltage')).toBe('pokemon-vivid-voltage')
  })

  // Measured 2026-08-31: pokemon-base misses and falls through to search;
  // pokemon-base-set resolves. This is the only override the sample found.
  it('applies the Base override', () => {
    expect(priceChartingSetSlug('Base')).toBe('pokemon-base-set')
  })

  it('does not apply the Base override to sets merely starting with Base', () => {
    expect(priceChartingSetSlug('Base Set 2')).toBe('pokemon-base-set-2')
  })
})
