import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('hitcheck', {
  listSources: () => ipcRenderer.invoke('hitcheck:listSources'),
})
