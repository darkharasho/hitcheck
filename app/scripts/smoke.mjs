// Runtime smoke check for the preload bridge.
//
// Boots the built app with a hidden BrowserWindow using the real preload
// script, then asks the renderer (via executeJavaScript, which runs in the
// page's main world) whether `window.hitcheck` was actually exposed by
// contextBridge. File-existence checks are not sufficient here: a preload
// script can exist on disk and still fail to execute (e.g. an ESM preload
// under a sandboxed renderer silently no-ops). This script proves the
// bridge runs, not just that the file is present.
//
// Exit code 0: window.hitcheck is a non-null object (bridge executed).
// Exit code 1: anything else (bridge missing, preload crashed, or error).
// Always quits Electron before exiting, leaving no stray process.

import { app, BrowserWindow } from 'electron'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = fileURLToPath(new URL('.', import.meta.url))
const preloadPath = join(__dirname, '../out/preload/index.js')

async function main() {
  await app.whenReady()

  const win = new BrowserWindow({
    show: false,
    webPreferences: {
      preload: preloadPath,
      // Intentionally omit `sandbox`/`nodeIntegration` so this exercises the
      // same webPreferences (and Electron's default sandboxed-renderer
      // behavior) as the real app in src/main/index.ts.
    },
  })

  await win.loadURL('data:text/html,<html><body></body></html>')

  const result = await win.webContents.executeJavaScript(
    'JSON.stringify({ type: typeof window.hitcheck, isNull: window.hitcheck === null })',
  )
  const { type, isNull } = JSON.parse(result)
  const ok = type === 'object' && !isNull

  console.log(`preload path: ${preloadPath}`)
  console.log(`typeof window.hitcheck = ${type}, isNull = ${isNull}`)
  console.log(ok ? 'PASS: window.hitcheck is a non-null object' : 'FAIL: window.hitcheck was not exposed')

  return ok ? 0 : 1
}

main()
  .then((code) => app.exit(code))
  .catch((err) => {
    console.error('smoke check errored:', err)
    app.exit(1)
  })
