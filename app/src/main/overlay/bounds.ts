export type Rect = { x: number; y: number; width: number; height: number }

/**
 * Shrink a rect symmetrically by `inset` pixels on every side.
 * Never returns a degenerate rect — clamps to 1x1 and keeps it centred.
 */
export function computeOverlayBounds(source: Rect, inset: number): Rect {
  const width = Math.max(1, source.width - inset * 2)
  const height = Math.max(1, source.height - inset * 2)
  const x = source.x + Math.round((source.width - width) / 2)
  const y = source.y + Math.round((source.height - height) / 2)
  return { x, y, width, height }
}
