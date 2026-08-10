import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('hitcheck', {
  listSources: () => ipcRenderer.invoke('hitcheck:listSources'),
  showOverlay: (bounds: { x: number; y: number; width: number; height: number }) =>
    ipcRenderer.invoke('hitcheck:showOverlay', bounds),
})
