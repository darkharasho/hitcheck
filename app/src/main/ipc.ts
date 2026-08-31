import { ipcMain } from 'electron'
import { listSources } from './capture/sources'
import { showOverlayBox } from './overlay/window'
import type { Rect } from './overlay/bounds'
import { cacheKey } from './lookup/cache'
import { routeCard } from './lookup/router'
import { navigateLookup } from './lookup/window'
import type { CardRecord, Classification } from './lookup/types'

export function registerIpc(): void {
  ipcMain.handle('hitcheck:listSources', () => listSources())
  ipcMain.handle('hitcheck:showOverlay', (_e, bounds: Rect) => { showOverlayBox(bounds) })
  ipcMain.handle(
    'hitcheck:lookupCard',
    (_e, card: CardRecord, classification: Classification) =>
      navigateLookup(cacheKey(card.id, classification), routeCard(card, classification)),
  )
}
