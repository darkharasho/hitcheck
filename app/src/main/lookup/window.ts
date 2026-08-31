import { BrowserWindow } from 'electron'
import { createResolvedCache, resolvedDestinations } from './cache'
import { classifyLanding } from './outcome'
import type { Destination } from './router'

let lookup: BrowserWindow | null = null
const resolved = createResolvedCache()

/**
 * Created once at startup, never per lot. A live auction closes in seconds, so
 * window creation and a cold page load must not sit on the critical path —
 * when the stability gate fires the only remaining work is loadURL.
 */
export function createLookupWindow(): BrowserWindow {
  const outgoing = lookup
  if (outgoing && !outgoing.isDestroyed()) outgoing.close()

  const win = new BrowserWindow({
    width: 900,
    height: 1000,
    show: false,
    title: 'HitCheck \u2014 Prices',
    webPreferences: {
      // No preload and no node integration: this window renders third-party
      // pages and must have no bridge into the app.
      contextIsolation: true,
      nodeIntegration: false,
      // A named partition persists cookies across restarts, so a TCGplayer
      // login survives. Deliberately not the default session, which the
      // capture half uses.
      partition: 'persist:lookup',
    },
  })

  win.on('closed', () => { if (lookup === win) lookup = null })
  lookup = win
  return win
}

/**
 * Walk the router's ordered destinations, stopping at the first that lands on
 * a product page. A `product`-kind destination that redirects to a search page
 * is a slug miss, and the next destination is our own set-aware search.
 */
export async function navigateLookup(
  cardId: string,
  destinations: Destination[],
): Promise<void> {
  const win = !lookup || lookup.isDestroyed() ? createLookupWindow() : lookup

  for (const destination of resolvedDestinations(cardId, destinations, resolved)) {
    try {
      await win.loadURL(destination.url)
    } catch {
      // Network failure or an aborted load: the window shows the browser's own
      // error page. Try the next destination; if there is none, that error page
      // is what the user sees, which is honest.
      continue
    }
    win.showInactive()
    const landed = win.webContents.getURL()
    if (classifyLanding(landed) === 'product') {
      resolved.set(cardId, landed)
      return
    }
  }
  win.showInactive()
}
