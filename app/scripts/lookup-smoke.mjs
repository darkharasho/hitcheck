// Manual smoke check: does the lookup window actually reach both sites, and
// does a deliberately wrong slug fall through to our search URL?
// Run: cd app && npm run build && npx electron scripts/lookup-smoke.mjs
// (the build step is required -- this imports from out/, which is gitignored)
import { app } from 'electron'
import { createLookupWindow, navigateLookup } from '../out/main/lookup/window.js'

// Each case passes its own cache key, exactly as ipc.ts composes it with
// cacheKey(cardId, classification) -- distinct keys keep the resolved-URL
// cache from replaying one case's landing page into another's.
const CASES = [
  ['slab hit', 'slab:base1-4', [
    { url: 'https://www.pricecharting.com/game/pokemon-base-set/charizard-4', kind: 'product' },
    { url: 'https://www.pricecharting.com/search-products?type=prices&q=Charizard+Base+4', kind: 'search' },
  ]],
  ['slab miss falls through', 'slab:xy0-nonesuch', [
    { url: 'https://www.pricecharting.com/game/pokemon-nonesuch/nothing-9999', kind: 'product' },
    { url: 'https://www.pricecharting.com/search-products?type=prices&q=Weedle+Kalos+Starter+Set+1', kind: 'search' },
  ]],
  ['raw single', 'raw-single:base1-4', [
    { url: 'https://prices.pokemontcg.io/tcgplayer/base1-4', kind: 'product' },
  ]],
]

app.whenReady().then(async () => {
  const win = createLookupWindow()
  for (const [label, key, destinations] of CASES) {
    await navigateLookup(key, destinations)
    console.log(`${label}: ${win.webContents.getURL()}`)
    await new Promise(r => setTimeout(r, 2000))
  }
  app.quit()
})
