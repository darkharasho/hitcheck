import { startCapture } from './capture/stream'
import { FrameSampler } from './capture/sampler'
import { isBlankFrame } from './capture/blank'

const status = document.getElementById('status')!
const video = document.getElementById('preview') as HTMLVideoElement
const canvas = document.createElement('canvas')
const ctx = canvas.getContext('2d', { willReadFrequently: true })!

document.getElementById('start')!.addEventListener('click', async () => {
  try {
    status.textContent = 'starting capture...'
    const stream = await startCapture()
    video.srcObject = stream
    await video.play()

    canvas.width = 320
    canvas.height = 180

    const sampler = new FrameSampler(15)
    let sampled = 0
    let blank = 0

    const tick = (t: number) => {
      try {
        if (sampler.shouldSample(t) && video.videoWidth > 0) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
          const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height)
          sampled += 1
          if (isBlankFrame(data)) blank += 1
        }
        const pct = sampled > 0 ? `${Math.round((blank / sampled) * 100)}%` : 'n/a'
        status.textContent = `sampled ${sampled} · blank ${blank} (${pct})`
      } catch (err) {
        status.textContent = `frame loop error: ${err instanceof Error ? err.message : String(err)}`
        return
      }
      requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)

    await (window as any).hitcheck.showOverlay({ x: 200, y: 200, width: 520, height: 380 })
  } catch (err) {
    status.textContent = `capture failed: ${err instanceof Error ? err.message : String(err)}`
  }
})

document.getElementById('overlay')!.addEventListener('click', () => {
  ;(window as any).hitcheck.showOverlay({ x: 200, y: 200, width: 520, height: 380 })
})
