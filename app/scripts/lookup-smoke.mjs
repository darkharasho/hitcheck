// Manual smoke check: does the lookup window actually reach both sites, and
// does a deliberately wrong slug fall through to our search URL?
// Run: cd app && npx electron scripts/lookup-smoke.mjs
import { app } from 'electron'
import { createLookupWindow, navigateLookup } from '../out/main/lookup/window.js'

const CASES = [
  ['slab hit', [
    { url: 'https://www.pricecharting.com/game/pokemon-base-set/charizard-4', kind: 'product' },
    { url: 'https://www.pricecharting.com/search-products?type=prices&q=Charizard+Base+4', kind: 'search' },
  ]],
  ['slab miss falls through', [
    { url: 'https://www.pricecharting.com/game/pokemon-nonesuch/nothing-9999', kind: 'product' },
    { url: 'https://www.pricecharting.com/search-products?type=prices&q=Weedle+Kalos+Starter+Set+1', kind: 'search' },
  ]],
  ['raw single', [
    { url: 'https://prices.pokemontcg.io/tcgplayer/base1-4', kind: 'product' },
  ]],
]

app.whenReady().then(async () => {
  const win = createLookupWindow()
  for (const [label, destinations] of CASES) {
    await navigateLookup(label, destinations)
    console.log(`${label}: ${win.webContents.getURL()}`)
    await new Promise(r => setTimeout(r, 2000))
  }
  app.quit()
})
