import { BrowserWindow } from 'electron'
import { createResolvedCache, resolvedDestinations, shouldCacheResolved } from './cache'
import { classifyLanding } from './outcome'
import type { Destination } from './router'

let lookup: BrowserWindow | null = null
const resolved = createResolvedCache()

/**
 * Bumped by every navigateLookup call so a superseded one can bow out.
 *
 * ipcMain.handle does not serialize invocations and lots close in seconds, so
 * two calls overlapping is the normal case, not an edge. Chromium aborts a
 * pending load when a newer loadURL starts, which makes the older call's
 * promise reject; without this counter that older call reads the abort as
 * "this destination failed" and loads its *next* destination, aborting the
 * newer lot and leaving the window on the previous lot's page. Abandoning the
 * older walk after every await is the whole fix: no queue and no settle delay,
 * so it costs the newer lot nothing.
 */
let generation = 0

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
  key: string,
  destinations: Destination[],
): Promise<void> {
  const mine = ++generation
  const superseded = (): boolean => generation !== mine
  const win = !lookup || lookup.isDestroyed() ? createLookupWindow() : lookup

  for (const destination of resolvedDestinations(key, destinations, resolved)) {
    try {
      await win.loadURL(destination.url)
    } catch {
      // A newer lookup started while this load was in flight: Chromium aborted
      // it on purpose and the newer call owns the window now. Stop quietly --
      // no fall-through, no cache write, no showInactive.
      if (superseded()) return
      // Otherwise a genuine network failure or aborted load: the window shows
      // the browser's own error page. Try the next destination; if there is
      // none, that error page is what the user sees, which is honest.
      continue
    }
    if (superseded()) return
    win.showInactive()
    const landed = win.webContents.getURL()
    if (classifyLanding(landed) === 'product') {
      if (shouldCacheResolved(destination.url, landed)) resolved.set(key, landed)
      return
    }
  }
  if (superseded()) return
  win.showInactive()
}
