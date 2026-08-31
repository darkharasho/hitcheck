import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('hitcheck', {
  listSources: () => ipcRenderer.invoke('hitcheck:listSources'),
  showOverlay: (bounds: { x: number; y: number; width: number; height: number }) =>
    ipcRenderer.invoke('hitcheck:showOverlay', bounds),
  lookupCard: (
    card: {
      id: string
      name: string
      number: string | null
      setName: string
      tcgplayerUrl: string | null
    },
    classification: 'slab' | 'raw-single',
  ) => ipcRenderer.invoke('hitcheck:lookupCard', card, classification),
})
