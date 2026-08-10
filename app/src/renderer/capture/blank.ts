const DEFAULT_VARIANCE_THRESHOLD = 25

/**
 * A frame is "blank" when its luminance carries essentially no variance —
 * an all-black, all-white, or otherwise flat image. Catches the silent
 * Wayland failure mode where capture succeeds but delivers no picture.
 *
 * `pixels` is RGBA, 4 bytes per pixel, as produced by `getImageData`.
 */
export function isBlankFrame(
  pixels: Uint8ClampedArray,
  opts: { varianceThreshold?: number } = {},
): boolean {
  const threshold = opts.varianceThreshold ?? DEFAULT_VARIANCE_THRESHOLD
  const count = Math.floor(pixels.length / 4)
  if (count === 0) return true

  let sum = 0
  let sumSquares = 0
  for (let i = 0; i < count; i++) {
    const o = i * 4
    // Rec. 601 luma — cheap and adequate for a presence check.
    const luma = 0.299 * pixels[o] + 0.587 * pixels[o + 1] + 0.114 * pixels[o + 2]
    sum += luma
    sumSquares += luma * luma
  }
  const mean = sum / count
  const variance = sumSquares / count - mean * mean
  return variance < threshold
}
