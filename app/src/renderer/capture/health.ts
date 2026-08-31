/**
 * Pure helpers backing the M0 acceptance status line.
 *
 * The status line is the human's only signal during a live acceptance run,
 * so the logic that decides what it says needs to be testable outside a
 * requestAnimationFrame callback.
 */

/** Default number of most-recent sampled frames used for the "recent" blank rate. */
export const DEFAULT_WINDOW_SIZE = 100

/** Gap between successive rAF ticks, in ms, above which we treat sampling as stalled. */
export const STALL_THRESHOLD_MS = 2000

/**
 * Bounded ring buffer of recent blank/non-blank judgements, used to compute
 * a "recent" blank rate that does not get diluted by a long healthy run
 * before a late-run failure.
 */
export class BlankRateWindow {
  private readonly buffer: boolean[]
  private index = 0
  private count = 0

  constructor(private readonly size: number = DEFAULT_WINDOW_SIZE) {
    if (size <= 0) throw new Error('BlankRateWindow size must be > 0')
    this.buffer = new Array(size)
  }

  push(isBlank: boolean): void {
    this.buffer[this.index] = isBlank
    this.index = (this.index + 1) % this.size
    if (this.count < this.size) this.count += 1
  }

  /** Fraction (0..1) of blank frames in the window, or null if no samples yet. */
  rate(): number | null {
    if (this.count === 0) return null
    let blanks = 0
    for (let i = 0; i < this.count; i++) {
      if (this.buffer[i]) blanks += 1
    }
    return blanks / this.count
  }
}

/** Formats a 0..1 rate as a rounded percentage string, or 'n/a' before any samples exist. */
export function formatPercent(rate: number | null): string {
  if (rate === null) return 'n/a'
  return `${Math.round(rate * 100)}%`
}

/**
 * True when the wall-clock gap between two rAF ticks is large enough that
 * sampling was almost certainly suspended (window minimised/occluded,
 * tab throttled, etc.) rather than just running at a normal frame cadence.
 */
export function isStallGap(deltaMs: number, thresholdMs: number = STALL_THRESHOLD_MS): boolean {
  return deltaMs > thresholdMs
}

/** True when a media track has ended (screen-share revoked, source window closed, etc.). */
export function isTrackEnded(track: Pick<MediaStreamTrack, 'readyState'>): boolean {
  return track.readyState === 'ended'
}
