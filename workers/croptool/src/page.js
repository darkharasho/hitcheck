/**
 * The crop page, ported from the trainer's local croptool.
 *
 * Deliberately the same interaction and the same /api contract, so a card
 * marked here and a card marked on the desktop tool produce the same quad.
 * What is new is only what being hosted requires: who you are, whether the
 * card in front of you is a calibration card, and a lease countdown.
 */
export const PAGE = `<!doctype html>
<html><head><meta charset="utf-8"><title>HitCheck crop tool</title>
<style>
  body { font: 14px system-ui; margin: 0; background: #111; color: #eee; }
  header { padding: 8px 12px; display: flex; gap: 16px; align-items: baseline;
           flex-wrap: wrap; }
  canvas { display: block; cursor: crosshair; }
  #hint { color: #9ad; }
  #save { font: inherit; padding: 4px 12px; }
  #save:disabled { opacity: 0.4; }
  #who { margin-left: auto; color: #777; }
  #banner { display: none; padding: 6px 12px; background: #4a3c00; color: #ffd; }
  #banner.on { display: block; }
  #error { display: none; padding: 6px 12px; background: #5a1a1a; color: #fdd; }
  #error.on { display: block; }
</style></head><body>
<header>
  <strong id="progress">loading...</strong>
  <span id="card"></span>
  <span id="hint">drag a box around the card, then put corner 1 on the card's
  TOP-LEFT and the rest to match. space = save, u = start over,
  s = skip (photo will not load / no card visible)</span>
  <button id="save" disabled>Save (space)</button>
  <span id="who"></span>
</header>
<div id="banner"></div>
<div id="error"></div>
<canvas id="c"></canvas>
<script>
// Corner 1 is drawn yellow because its placement is the one thing no
// server-side check can catch: a quad rotated a quarter turn is still
// simple and still clockwise, and yields a sideways crop that retrieves
// nothing. Winding itself is safe by construction -- the rubber band
// emits TL, TR, BR, BL, which is positive signed area in y-down image
// space -- so the anticlockwise rejection only fires when a handle is
// dragged past its neighbours.
let item = null, scale = 1, points = [], drag = null, grabbed = null, img = new Image();
const canvas = document.getElementById('c'), ctx = canvas.getContext('2d');
const HANDLE_R = 7;

function say(message) {
  const box = document.getElementById('error');
  box.textContent = message || '';
  box.className = message ? 'on' : '';
}

async function load() {
  say('');
  const res = await fetch('/api/next');
  if (!res.ok) { say('could not load: ' + (await res.text())); return; }
  const state = await res.json();
  document.getElementById('who').textContent = state.cropper || '';
  document.getElementById('progress').textContent = state.done + ' / ' + state.total;
  const banner = document.getElementById('banner');
  // Calibration cards are announced, not hidden. Someone who knows this one
  // is being compared marks it the way they mean to mark all of them.
  banner.textContent = state.calibration
    ? 'Calibration card ' + state.calibration_done + ' of ' + state.calibration_total +
      ' — this one is checked against a reference crop before your work counts.'
    : '';
  banner.className = state.calibration ? 'on' : '';
  if (!state.item_id) {
    // Cleared, so the heartbeat below stops renewing a lease on a card that
    // is already marked.
    item = null;
    saveButton.disabled = true;
    document.getElementById('card').textContent = 'done — nothing left to crop';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  item = state;
  document.getElementById('card').textContent = state.card_id;
  points = []; drag = null; grabbed = null;
  img = new Image();
  img.onload = draw;
  img.onerror = () => say('this photograph will not load — press s to skip it');
  img.src = '/api/image?id=' + encodeURIComponent(state.item_id);
}

function draw() {
  // Every path that changes the quad already calls draw(), so the button's
  // enabled state is derived here rather than tracked separately.
  saveButton.disabled = !saveable();
  const maxH = window.innerHeight - 100, maxW = window.innerWidth;
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

function saveable() {
  return points.length === 4 && Boolean(item);
}

async function post(path, body) {
  const res = await fetch(path, { method: 'POST', body: JSON.stringify(body) });
  if (res.ok) { load(); return; }
  // On rejection the quad is LEFT on screen -- the operator drags the
  // offending handle rather than re-marking the card from scratch.
  say((await res.json().catch(() => ({}))).error || 'rejected');
}

function save() {
  if (!saveable()) return;
  post('/api/quad', { item_id: item.item_id, quad: points });
}

const saveButton = document.getElementById('save');
// mousedown prevented rather than click handled alone: a button that takes
// focus treats the next space as a press of itself, and the quad would go up
// twice -- once from the keydown handler, once from the button.
saveButton.addEventListener('mousedown', (e) => e.preventDefault());
saveButton.addEventListener('click', save);

window.addEventListener('keydown', (e) => {
  if (e.key === 'u') { points = []; drag = null; say(''); draw(); }
  if (e.key === ' ' && saveable()) {
    // Submit is a keypress or the button, not the fourth click: the whole
    // point of the handles is to adjust after seeing the outline closed.
    e.preventDefault();
    save();
  }
  // A photograph that never decodes leaves the canvas blank and unclickable,
  // and /api/next would otherwise hand it back forever.
  if (e.key === 's' && item) post('/api/skip', { item_id: item.item_id });
});
window.addEventListener('resize', draw);
load();
// The lease is held for minutes, not hours. Reloading keeps a card reserved
// while its cropper is actually looking at it.
setInterval(() => { if (item) fetch('/api/heartbeat?id=' + encodeURIComponent(item.item_id)); },
            60000);
</script></body></html>
`
