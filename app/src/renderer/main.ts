import { startCapture } from './capture/stream'
import { FrameSampler } from './capture/sampler'

const status = document.getElementById('status')!
const video = document.getElementById('preview') as HTMLVideoElement

document.getElementById('start')!.addEventListener('click', async () => {
  try {
    status.textContent = 'starting capture...'
    const stream = await startCapture()
    video.srcObject = stream

    const sampler = new FrameSampler(15)
    let sampled = 0
    const tick = (t: number) => {
      if (sampler.shouldSample(t)) sampled += 1
      status.textContent = `sampled ${sampled} frames`
      requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)
  } catch (err) {
    status.textContent = `capture failed: ${err instanceof Error ? err.message : String(err)}`
  }
})
