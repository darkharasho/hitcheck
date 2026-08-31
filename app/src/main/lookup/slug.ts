/**
 * PriceCharting set slugs that do not follow the default rule.
 *
 * Derived empirically on 2026-08-31 by requesting constructed slugs and
 * checking whether the site redirected to /search-products. Of 14 sets
 * sampled, only "Base" needed an override. Add entries here as real misses
 * are observed — do not guess.
 */
const SET_SLUG_OVERRIDES: Record<string, string> = {
  Base: 'base-set',
}

export function slugify(value: string): string {
  return value
    .toLowerCase()
    .normalize('NFD')
    // Strip combining marks left behind by NFD, so "é" becomes "e".
    .replace(/[̀-ͯ]/g, '')
    .replace(/&/g, ' and ')
    // Apostrophes vanish rather than becoming hyphens: PriceCharting slugs
    // "Team Rocket's Meowth" as team-rockets-meowth, not team-rocket-s-meowth.
    .replace(/['']/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

export function priceChartingSetSlug(setName: string): string {
  const override = SET_SLUG_OVERRIDES[setName]
  return `pokemon-${override ?? slugify(setName)}`
}
