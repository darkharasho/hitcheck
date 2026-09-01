"""Local browser tool for hand-marking card corners in corpus photographs.

Stdlib http.server plus one HTML page: no new dependency, no reliance on
a working _tkinter in the venv, and it works on this machine's Wayland
session. All routing goes through CropApp.handle, so the whole tool is
testable without binding a socket -- the request handler below is a thin
adapter and is the only part that is not.

The client posts corners in ORIGINAL image pixels. It scales the photo to
fit the window, so it divides by that scale before posting; recording
display coordinates would make every quad silently wrong by the scale
factor.
"""

import argparse
import json
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .crops import load_crops, load_skips, save_crops, save_skips, validate_quad
from .manifest import load_manifest

DEFAULT_CORPUS = "data/corpus"
SKIPS_FILE = "skipped.json"

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>HitCheck crop tool</title>
<style>
  body { font: 14px system-ui; margin: 0; background: #111; color: #eee; }
  header { padding: 8px 12px; display: flex; gap: 16px; align-items: baseline; }
  canvas { display: block; cursor: crosshair; }
  #hint { color: #9ad; }
  #save { font: inherit; padding: 4px 12px; }
  #save:disabled { opacity: 0.4; }
</style></head><body>
<header>
  <strong id="progress">loading...</strong>
  <span id="card"></span>
  <span id="hint">drag a box around the card, then put corner 1 on the card's
  TOP-LEFT and the rest to match. space = save, u = start over,
  s = skip (photo will not load / no card visible)</span>
  <button id="save" disabled>Save (space)</button>
</header>
<canvas id="c"></canvas>
<script>
// Corner 1 is drawn yellow because its placement is the one thing no
// server-side check can catch: a quad rotated a quarter turn is still
// simple and still clockwise, and yields a sideways crop that retrieves
// nothing. Winding itself is safe by construction -- the rubber band
// emits TL, TR, BR, BL, which is positive signed area in y-down image
// space -- so validate_quad's clockwise rejection now only fires when a
// handle is dragged past its neighbours.
let item = null, scale = 1, points = [], drag = null, grabbed = null, img = new Image();
const canvas = document.getElementById('c'), ctx = canvas.getContext('2d');
const saveButton = document.getElementById('save');
const HANDLE_R = 7;

function saveable() {
  return points.length === 4 && Boolean(item);
}

async function save() {
  if (!saveable()) return;
  const res = await fetch('/api/quad', {
    method: 'POST',
    body: JSON.stringify({item_id: item.item_id, quad: points}),
  });
  // On rejection the quad is LEFT on screen -- the operator drags the
  // offending handle rather than re-marking the card from scratch.
  if (res.ok) { load(); } else { alert((await res.json()).error); }
}

// mousedown prevented rather than click handled alone: a button that takes
// focus treats the next space as a press of itself, and the quad would go up
// twice -- once from the keydown handler, once from the button.
saveButton.addEventListener('mousedown', (e) => e.preventDefault());
saveButton.addEventListener('click', save);

async function load() {
  const state = await (await fetch('/api/next')).json();
  document.getElementById('progress').textContent = state.done + ' / ' + state.total;
  if (!state.item_id) {
    item = null; saveButton.disabled = true;
    document.getElementById('card').textContent = 'done'; return;
  }
  item = state;
  document.getElementById('card').textContent = state.card_id;
  points = []; drag = null; grabbed = null;
  img = new Image();
  img.onload = draw;
  img.src = '/api/image?id=' + encodeURIComponent(state.item_id);
}

function draw() {
  // Every path that changes the quad already calls draw(), so the button's
  // enabled state is derived here rather than tracked separately.
  saveButton.disabled = !saveable();
  const maxH = window.innerHeight - 60, maxW = window.innerWidth;
  scale = Math.min(maxW / img.width, maxH / img.height, 1);
  canvas.width = img.width * scale;
  canvas.height = img.height * scale;
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  if (drag) {
    ctx.strokeStyle = '#0f0'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    const [a, b] = drag;
    ctx.strokeRect(a[0] * scale, a[1] * scale, (b[0] - a[0]) * scale, (b[1] - a[1]) * scale);
    ctx.setLineDash([]);
    return;
  }
  if (points.length !== 4) return;

  ctx.strokeStyle = '#0f0'; ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((p, i) => i ? ctx.lineTo(p[0] * scale, p[1] * scale)
                             : ctx.moveTo(p[0] * scale, p[1] * scale));
  ctx.closePath(); ctx.stroke();
  points.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(p[0] * scale, p[1] * scale, HANDLE_R, 0, 7);
    ctx.fillStyle = i === 0 ? '#ff0' : '#0f0';
    ctx.fill();
    ctx.fillStyle = '#000'; ctx.font = 'bold 11px system-ui';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(String(i + 1), p[0] * scale, p[1] * scale);
  });
}

function at(e) {
  const box = canvas.getBoundingClientRect();
  // Divide by scale: the server stores ORIGINAL image pixels.
  return [(e.clientX - box.left) / scale, (e.clientY - box.top) / scale];
}

canvas.addEventListener('mousedown', (e) => {
  const p = at(e);
  // Grab radius is in SCREEN pixels, so it stays clickable on a photo
  // scaled down to a third of its size.
  const near = points.findIndex(q => Math.hypot(q[0] - p[0], q[1] - p[1]) * scale < HANDLE_R * 2);
  if (near >= 0) { grabbed = near; return; }
  points = []; drag = [p, p];
});

window.addEventListener('mousemove', (e) => {
  if (grabbed !== null) { points[grabbed] = at(e); draw(); }
  else if (drag) { drag[1] = at(e); draw(); }
});

window.addEventListener('mouseup', () => {
  if (grabbed !== null) { grabbed = null; return; }
  if (!drag) return;
  const [a, b] = drag;
  const x0 = Math.min(a[0], b[0]), x1 = Math.max(a[0], b[0]);
  const y0 = Math.min(a[1], b[1]), y1 = Math.max(a[1], b[1]);
  drag = null;
  // A stray click is a zero-size band, not a crop. Below a few pixels
  // there is nothing to drag handles off of, so discard it rather than
  // leave four coincident handles on the canvas.
  points = (x1 - x0 > 8 && y1 - y0 > 8) ? [[x0, y0], [x1, y0], [x1, y1], [x0, y1]] : [];
  draw();
});

window.addEventListener('keydown', async (e) => {
  if (e.key === 'u') { points = []; drag = null; draw(); }
  if (e.key === ' ' && saveable()) {
    // Submit is a keypress or the button, not the fourth click: the whole
    // point of the handles is to adjust after seeing the outline closed.
    e.preventDefault();
    save();
  }
  // A photograph that never decodes leaves the canvas blank and unclickable,
  // and /api/next would otherwise hand it back forever. Skipping is recorded
  // server-side so the next run does not serve it again either.
  if (e.key === 's' && item) {
    const res = await fetch('/api/skip', {
      method: 'POST',
      body: JSON.stringify({item_id: item.item_id}),
    });
    if (res.ok) { load(); } else { alert((await res.json()).error); }
  }
});
window.addEventListener('resize', draw);
load();
</script></body></html>
"""


class CropApp:
    def __init__(self, manifest, crops, crops_path, corpus_dir, skips=None, skips_path=None):
        self._manifest = manifest
        self._crops = crops
        self._crops_path = crops_path
        self._corpus_dir = corpus_dir
        self._skips = set(skips or ())
        self._skips_path = skips_path or os.path.join(corpus_dir, SKIPS_FILE)
        self._by_id = {e.item_id: e for e in manifest.entries}

    def next_item(self):
        for entry in self._manifest.entries:
            if entry.item_id not in self._crops and entry.item_id not in self._skips:
                return {"item_id": entry.item_id, "card_id": entry.card_id, "image": entry.image}
        return None

    def progress(self) -> tuple[int, int]:
        return len(self._crops), len(self._manifest.entries)

    def image_bytes(self, item_id: str) -> bytes:
        entry = self._by_id[item_id]
        with open(os.path.join(self._corpus_dir, entry.image), "rb") as fh:
            return fh.read()

    def record(self, item_id: str, quad) -> None:
        validate_quad(quad)
        if item_id not in self._by_id:
            raise KeyError(item_id)
        self._crops[item_id] = [[float(x), float(y)] for x, y in quad]
        # Saved per quad: a crash three hours into a pass must not cost it.
        save_crops(self._crops, self._crops_path)

    def skip(self, item_id: str) -> None:
        """Mark an item unusable so the pass can move past it.

        Recorded to disk immediately, and to its own file rather than
        crops.json -- a skip has no quad, and must not be counted as
        cropped progress.
        """
        if item_id not in self._by_id:
            raise KeyError(item_id)
        self._skips.add(item_id)
        save_skips(self._skips, self._skips_path)

    def handle(self, method: str, path: str, body: bytes) -> tuple[int, str, bytes]:
        route, _, query = path.partition("?")

        if method == "GET" and route == "/":
            return 200, "text/html", PAGE.encode()

        if method == "GET" and route == "/api/next":
            item = self.next_item()
            done, total = self.progress()
            payload = {
                "item_id": None,
                "card_id": None,
                "image": None,
                "done": done,
                "total": total,
            }
            if item:
                payload.update(item)
            return 200, "application/json", json.dumps(payload).encode()

        if method == "GET" and route == "/api/image":
            item_id = urllib.parse.parse_qs(query).get("id", [""])[0]
            try:
                return 200, "image/jpeg", self.image_bytes(item_id)
            except (KeyError, OSError):
                return 404, "application/json", b'{"error": "no such image"}'

        if method == "POST" and route == "/api/skip":
            try:
                self.skip(json.loads(body)["item_id"])
            except (ValueError, KeyError, TypeError) as exc:
                return 400, "application/json", json.dumps({"error": str(exc)}).encode()
            return 200, "application/json", b'{"ok": true}'

        if method == "POST" and route == "/api/quad":
            try:
                payload = json.loads(body)
                self.record(payload["item_id"], payload["quad"])
            except (ValueError, KeyError, TypeError) as exc:
                return 400, "application/json", json.dumps({"error": str(exc)}).encode()
            return 200, "application/json", b'{"ok": true}'

        return 404, "application/json", b'{"error": "not found"}'


def _make_handler(app: CropApp):
    class Handler(BaseHTTPRequestHandler):
        def _respond(self, method):
            length = int(self.headers.get("Content-Length") or 0)
            status, content_type, payload = app.handle(method, self.path, self.rfile.read(length))
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self._respond("GET")

        def do_POST(self):
            self._respond("POST")

        def log_message(self, *args):
            pass  # one line per click is noise during a long cropping pass

    return Handler


def serve(app: CropApp, port: int = 8765) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(app))
    done, total = app.progress()
    print(f"crop tool on http://127.0.0.1:{port}/  ({done}/{total} done)")
    print("Ctrl-C to stop; progress is saved after every card.")
    print("Drag a box around the card, adjust the four corners, space or Save.")
    print("Press s on a photo that will not load — it is skipped for good.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hitcheck-croptool")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    manifest = load_manifest(os.path.join(args.corpus, "manifest.json"))
    if not manifest.entries:
        print(f"No manifest entries under {args.corpus}. Run the corpus build first.")
        return 1
    crops_path = os.path.join(args.corpus, "crops.json")
    skips_path = os.path.join(args.corpus, SKIPS_FILE)
    app = CropApp(manifest, load_crops(crops_path), crops_path, args.corpus,
                  skips=load_skips(skips_path), skips_path=skips_path)
    serve(app, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
