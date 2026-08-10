/**
 * Decides whether a frame arriving at `nowMs` should be processed,
 * throttling to approximately `fps`. Time is injected so this is
 * testable without a clock.
 */
export class FrameSampler {
  private readonly intervalMs: number
  private lastMs: number | null = null

  constructor(fps: number) {
    this.intervalMs = fps > 0 ? 1000 / fps : 0
  }

  shouldSample(nowMs: number): boolean {
    if (this.lastMs === null || nowMs - this.lastMs >= this.intervalMs) {
      this.lastMs = nowMs
      return true
    }
    return false
  }
}
