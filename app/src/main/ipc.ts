import { ipcMain } from 'electron'
import { listSources } from './capture/sources'
import { showOverlayBox } from './overlay/window'
import type { Rect } from './overlay/bounds'

export function registerIpc(): void {
  ipcMain.handle('hitcheck:listSources', () => listSources())
  ipcMain.handle('hitcheck:showOverlay', (_e, bounds: Rect) => { showOverlayBox(bounds) })
}
