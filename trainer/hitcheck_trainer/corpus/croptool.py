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

from .crops import load_crops, save_crops, validate_quad
from .manifest import load_manifest

DEFAULT_CORPUS = "data/corpus"

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>HitCheck crop tool</title>
<style>
  body { font: 14px system-ui; margin: 0; background: #111; color: #eee; }
  header { padding: 8px 12px; display: flex; gap: 16px; align-items: baseline; }
  canvas { display: block; cursor: crosshair; }
  #hint { color: #9ad; }
</style></head><body>
<header>
  <strong id="progress">loading...</strong>
  <span id="card"></span>
  <span id="hint">click the card's top-left corner, then clockwise. u = undo</span>
</header>
<canvas id="c"></canvas>
<script>
let item = null, scale = 1, points = [], img = new Image();
const canvas = document.getElementById('c'), ctx = canvas.getContext('2d');

async function load() {
  const state = await (await fetch('/api/next')).json();
  document.getElementById('progress').textContent = state.done + ' / ' + state.total;
  if (!state.item_id) { document.getElementById('card').textContent = 'done'; return; }
  item = state;
  document.getElementById('card').textContent = state.card_id;
  points = [];
  img = new Image();
  img.onload = draw;
  img.src = '/api/image?id=' + encodeURIComponent(state.item_id);
}

function draw() {
  const maxH = window.innerHeight - 60, maxW = window.innerWidth;
  scale = Math.min(maxW / img.width, maxH / img.height, 1);
  canvas.width = img.width * scale;
  canvas.height = img.height * scale;
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#0f0'; ctx.fillStyle = '#0f0'; ctx.lineWidth = 2;
  points.forEach((p, i) => {
    ctx.beginPath(); ctx.arc(p[0] * scale, p[1] * scale, 4, 0, 7); ctx.fill();
    if (i) {
      ctx.beginPath();
      ctx.moveTo(points[i-1][0] * scale, points[i-1][1] * scale);
      ctx.lineTo(p[0] * scale, p[1] * scale);
      ctx.stroke();
    }
  });
}

canvas.addEventListener('click', async (e) => {
  const box = canvas.getBoundingClientRect();
  // Divide by scale: the server stores ORIGINAL image pixels.
  points.push([(e.clientX - box.left) / scale, (e.clientY - box.top) / scale]);
  draw();
  if (points.length === 4) {
    const res = await fetch('/api/quad', {
      method: 'POST',
      body: JSON.stringify({item_id: item.item_id, quad: points}),
    });
    if (res.ok) { load(); } else { alert((await res.json()).error); points = []; draw(); }
  }
});

window.addEventListener('keydown', (e) => {
  if (e.key === 'u') { points.pop(); draw(); }
});
window.addEventListener('resize', draw);
load();
</script></body></html>
"""


class CropApp:
    def __init__(self, manifest, crops, crops_path, corpus_dir):
        self._manifest = manifest
        self._crops = crops
        self._crops_path = crops_path
        self._corpus_dir = corpus_dir
        self._by_id = {e.item_id: e for e in manifest.entries}

    def next_item(self):
        for entry in self._manifest.entries:
            if entry.item_id not in self._crops:
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
    serve(CropApp(manifest, load_crops(crops_path), crops_path, args.corpus), args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
