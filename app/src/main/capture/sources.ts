import { desktopCapturer } from 'electron'
import { rankSources } from './filter'
import type { CaptureSource } from './types'

export async function listSources(): Promise<CaptureSource[]> {
  const raw = await desktopCapturer.getSources({
    types: ['window', 'screen'],
    thumbnailSize: { width: 0, height: 0 },
  })
  const mapped: CaptureSource[] = raw.map(s => ({
    id: s.id,
    name: s.name,
    kind: s.id.startsWith('screen:') ? 'screen' : 'window',
  }))
  return rankSources(mapped, 'HitCheck')
}
