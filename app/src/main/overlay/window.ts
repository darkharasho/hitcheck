import { BrowserWindow } from 'electron'
import { join } from 'node:path'
import type { Rect } from './bounds'

let overlay: BrowserWindow | null = null

export function createOverlayWindow(bounds: Rect): BrowserWindow {
  // Guard against leaking a window: if a live overlay already exists (e.g. this
  // is called directly rather than through showOverlayBox), close it before
  // replacing the reference.
  if (overlay && !overlay.isDestroyed()) overlay.close()

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

  // The boolean is what keeps the overlay on top on Linux/X11 and Wayland
  // compositors that support layer-shell-style always-on-top hints; the
  // 'screen-saver' level argument is macOS/Windows-only per Electron's docs
  // and has no effect here, but is harmless to pass and helps on those
  // platforms.
  overlay.setAlwaysOnTop(true, 'screen-saver')
  // `visibleOnFullScreen` is macOS-only and does nothing on Linux; on Linux,
  // staying above a fullscreen window depends on setAlwaysOnTop above and the
  // window manager/compositor's own always-on-top handling. Kept for macOS.
  overlay.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true })
  // Clicks pass straight through to the stream underneath. `forward`
  // keeps mouse-move events flowing so hover effects remain possible later.
  overlay.setIgnoreMouseEvents(true, { forward: true })

  if (process.env.ELECTRON_RENDERER_URL) {
    overlay.loadURL(`${process.env.ELECTRON_RENDERER_URL}/overlay/overlay.html`)
  } else {
    overlay.loadFile(join(import.meta.dirname, '../renderer/overlay/overlay.html'))
  }
  overlay.on('closed', () => { overlay = null })
  return overlay
}

export function showOverlayBox(bounds: Rect): void {
  if (!overlay || overlay.isDestroyed()) createOverlayWindow(bounds)
  else overlay.setBounds(bounds)
  overlay?.showInactive()
}
