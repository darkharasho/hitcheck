import { startCapture } from './capture/stream'
import { FrameSampler } from './capture/sampler'
import { isBlankFrame } from './capture/blank'
import { BlankRateWindow, formatPercent, isStallGap, isTrackEnded } from './capture/health'

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
    const recentWindow = new BlankRateWindow()
    let sampled = 0
    let blank = 0
    let stalls = 0
    let lastTickMs: number | null = null
    let running = true

    const [track] = stream.getVideoTracks()

    // Distinct terminal states, set from event listeners so we notice the
    // stream ending even if the rAF loop itself is throttled at that moment.
    const stopForEndedCapture = (reason: string) => {
      if (!running) return
      running = false
      status.textContent = `capture ended — ${reason} after ${sampled} sampled frame${sampled === 1 ? '' : 's'}`
    }

    track?.addEventListener('ended', () => stopForEndedCapture('stream stopped'))
    video.addEventListener('pause', () => {
      if (running) stopForEndedCapture('preview paused unexpectedly')
    })

    const tick = (t: number) => {
      if (!running) return

      // Belt-and-suspenders: drawImage on an ended track usually does not
      // throw, it just keeps drawing the last frame — so check readyState
      // explicitly rather than relying on an exception.
      if (track && isTrackEnded(track)) {
        stopForEndedCapture('stream stopped')
        return
      }

      if (lastTickMs !== null && isStallGap(t - lastTickMs)) {
        stalls += 1
      }
      lastTickMs = t

      try {
        if (sampler.shouldSample(t) && video.videoWidth > 0) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
          const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height)
          sampled += 1
          const blankFrame = isBlankFrame(data)
          if (blankFrame) blank += 1
          recentWindow.push(blankFrame)
        }
        const cumulativePct = sampled > 0 ? formatPercent(blank / sampled) : 'n/a'
        const recentPct = formatPercent(recentWindow.rate())
        const stallSuffix = stalls > 0 ? ` · stalls ${stalls}` : ''
        status.textContent = `sampled ${sampled} · blank ${blank} (all ${cumulativePct}, recent ${recentPct})${stallSuffix}`
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
