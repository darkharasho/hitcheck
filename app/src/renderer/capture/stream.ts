/**
 * Acquire a capture stream.
 *
 * Source selection is owned by the main process's display-media handler,
 * which defers to the compositor's own picker (the xdg-desktop-portal
 * dialog on Wayland). The renderer therefore does not choose a source —
 * `listSources()` exists to show the user what is available, not to
 * select on their behalf.
 */
export async function startCapture(): Promise<MediaStream> {
  return navigator.mediaDevices.getDisplayMedia({
    video: { frameRate: { ideal: 30 } },
    audio: false,
  })
}
