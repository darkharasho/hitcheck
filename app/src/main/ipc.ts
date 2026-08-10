import { ipcMain } from 'electron'
import { listSources } from './capture/sources'

export function registerIpc(): void {
  ipcMain.handle('hitcheck:listSources', () => listSources())
}
