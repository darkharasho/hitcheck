import { app, BrowserWindow, session, desktopCapturer } from 'electron'
import { join } from 'node:path'
import { registerIpc } from './ipc'
import { createLookupWindow } from './lookup/window'

// Wayland/PipeWire screen capture. Harmless where already default.
if (process.platform === 'linux') {
  app.commandLine.appendSwitch('enable-features', 'WebRTCPipeWireCapturer')
}

function registerDisplayMediaHandler(): void {
  // On Wayland, the xdg-desktop-portal handshake is driven by the
  // WebRTCPipeWireCapturer command-line switch enabled above, combined with
  // this desktopCapturer.getSources() call — that's what triggers PipeWire
  // and surfaces the portal's own picker dialog to the user.
  //
  // Note: `useSystemPicker` (Electron's built-in system-picker option) is
  // experimental and macOS 15+ only per Electron's type declarations; it has
  // no effect on Linux/Wayland, so it is intentionally omitted here.
  session.defaultSession.setDisplayMediaRequestHandler((_request, callback) => {
    desktopCapturer
      .getSources({ types: ['window', 'screen'] })
      .then(sources => {
        if (sources.length === 0) return callback({})
        callback({ video: sources[0] })
      })
      .catch(() => callback({}))
  })
}

// Tracked so the darwin 'activate' handler below can tell "no main window" from
// "only the lookup window is left" -- BrowserWindow.getAllWindows() counts the
// always-open lookup window, so it can never reach zero on its own.
let mainWindow: BrowserWindow | null = null

function createWindow(): void {
  const win = new BrowserWindow({
    width: 1100,
    height: 720,
    webPreferences: { preload: join(import.meta.dirname, '../preload/index.js') },
  })
  if (process.env.ELECTRON_RENDERER_URL) win.loadURL(process.env.ELECTRON_RENDERER_URL)
  else win.loadFile(join(import.meta.dirname, '../renderer/index.html'))

  mainWindow = win

  // The lookup window is created once at startup and stays open (hidden or
  // shown) for the app's whole life, so Electron's window count never drops
  // to zero on its own -- 'window-all-closed' below would otherwise never
  // fire once the user closes this, the only window they can see and close.
  // Quitting here, tied to this specific window, keeps that shutdown path
  // reachable without touching the lookup window's own lifecycle.
  win.on('closed', () => {
    if (mainWindow === win) mainWindow = null
    if (process.platform !== 'darwin') app.quit()
  })
}

app.whenReady().then(() => {
  registerIpc()
  registerDisplayMediaHandler()
  createWindow()
  createLookupWindow()

  // macOS convention: the app stays alive with no main window, so clicking the
  // dock icon must be able to bring one back. Without this, closing the main
  // window on darwin leaves the user with only the always-open lookup window --
  // a stray third-party price page and no route back into HitCheck.
  //
  // Registered inside whenReady, not at module scope: on macOS 'activate' can
  // fire for a first launch before 'ready', and constructing a BrowserWindow
  // before the app is ready throws in the main process.
  app.on('activate', () => {
    if (!mainWindow || mainWindow.isDestroyed()) createWindow()
  })
})
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit() })
