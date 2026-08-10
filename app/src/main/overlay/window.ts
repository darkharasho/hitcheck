import { BrowserWindow } from 'electron'
import { join } from 'node:path'
import type { Rect } from './bounds'

let overlay: BrowserWindow | null = null

export function createOverlayWindow(bounds: Rect): BrowserWindow {
  overlay = new BrowserWindow({
    ...bounds,
    frame: false,
    transparent: true,
    hasShadow: false,
    resizable: false,
    movable: false,
    skipTaskbar: true,
    focusable: false,
    alwaysOnTop: true,
    webPreferences: { contextIsolation: true },
  })

  // 'screen-saver' is the highest practical level and is what keeps the
  // overlay above a fullscreen browser playing the stream.
  overlay.setAlwaysOnTop(true, 'screen-saver')
  overlay.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true })
  // Clicks pass straight through to the stream underneath. `forward`
  // keeps mouse-move events flowing so hover effects remain possible later.
  overlay.setIgnoreMouseEvents(true, { forward: true })

  overlay.loadFile(join(import.meta.dirname, '../renderer/overlay/overlay.html'))
  overlay.on('closed', () => { overlay = null })
  return overlay
}

export function showOverlayBox(bounds: Rect): void {
  if (!overlay || overlay.isDestroyed()) createOverlayWindow(bounds)
  else overlay.setBounds(bounds)
  overlay?.showInactive()
}
