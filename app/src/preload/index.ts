import { contextBridge } from 'electron'

contextBridge.exposeInMainWorld('hitcheck', {})
