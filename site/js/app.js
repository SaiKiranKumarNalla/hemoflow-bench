// app.js — orchestrates the hemoflow interactive corpus viewer.
import { loadFamily, interpolate, deriveField } from "./interpolator.js";
import { makeMapper, computeStreamlines, RBCParticles } from "./rbc_streamlines.js";

const $ = id => document.getElementById(id);
const DATA = "data/";
let INDEX = null;
let FAM = null;            // current family decoded object
let STATE = {
  family: "G2", variant: "V4", field: "phi",
  geo: {}, Re: 10, phi_in: 0.3, h_um: 110,
  rbcCount: 280, showLines: true, showRBC: true, phiWeight: true, expandY: true,
  probeX: null,
};
let interp = null;         // current interpolated result
let particles = null;
let streamlines = [];
let bgCanvas = null;       // cached offscreen heatmap canvas
let activeRange = [0, 1];
let activeDiv = false;
let raf = null;

// ---------- color maps ----------
function lerp(a, b, t) { return a + (b - a) * t; }
function mix(c1, c2, t) { return [lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t)]; }
const SEQ = [[5, 8, 15], [19, 78, 74], [45, 212, 191], [251, 191, 36], [225, 29, 72]];
const DIV = [[33, 102, 172], [247, 247, 247], [225, 29, 72]];
// High-contrast CFL palette for phi: deep navy -> bright plasma cyan (CFL!) -> pale mist -> amber -> deep RBC crimson
const PHI_CMAP = [
  [8, 48, 107],
  [14, 165, 233],
  [165, 243, 252],
  [253, 224, 71],
  [244, 63, 94],
  [159, 18, 57],
];
function cmap(t, mode = "seq") {
  t = Math.max(0, Math.min(1, t));
  const stops = mode === "div" ? DIV : (mode === "phi" ? PHI_CMAP : SEQ);
  const x = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  return mix(stops[i], stops[i + 1], x - i);
}

// ---------- index + controls ----------
async function boot() {
  const resp = await fetch(DATA + "index.json");
  INDEX = await resp.json();
  $("status-pill").textContent = `${INDEX.n_runs_total} runs · ${(INDEX.total_bundle_bytes / 1e6).toFixed(1)} MB`;
  const fs = $("family-select");
  for (const fam of Object.keys(INDEX.families)) {
    const o = document.createElement("option");
    o.value = fam; o.textContent = `${fam} — ${INDEX.families[fam].name}`;
    fs.appendChild(o);
  }
  fs.value = STATE.family;
  $("variant-select").value = STATE.variant;
  $("field-select").value = STATE.field;

  fs.onchange = async () => { STATE.family = fs.value; STATE.probeX = null; buildSliders(false); await loadAndSnapDefault(); };
  $("variant-select").onchange = e => { STATE.variant = e.target.value; refresh(); };
  $("field-select").onchange = e => { STATE.field = e.target.value; rebuildFieldTexture(); renderFrame(); };
  $("rbc-count").oninput = e => { STATE.rbcCount = +e.target.value; $("rbc-count-val").textContent = e.target.value; resetParticles(); };
  $("show-streamlines").onchange = e => { STATE.showLines = e.target.checked; renderFrame(); };
  $("show-rbcs").onchange = e => { STATE.showRBC = e.target.checked; renderFrame(); };
  $("phi-weight").onchange = e => { STATE.phiWeight = e.target.checked; resetParticles(); };
  $("expand-y").onchange = e => { STATE.expandY = e.target.checked; renderFrame(); };
  $("snap-btn").onclick = snapToNearest;
  $("exact-run-select").onchange = e => { if (e.target.value) applyExactRun(e.target.value); };
  $("copy-btn").onclick = copySnippet;

  buildSliders(false);
  await loadAndSnapDefault();

  const cv = $("main-canvas");
  cv.addEventListener("mousemove", onProbe);
  cv.addEventListener("mouseleave", () => { STATE.probeX = null; renderFrame(); drawCharts(); });
  animate();
}

function familyMeta() { return INDEX.families[STATE.family]; }

function buildSliders(preserveState = false) {
  const meta = familyMeta();
  $("family-desc").textContent = meta.description;
  $("family-topo").textContent = `topology: ${meta.topology} · ${meta.n_runs} runs · ${meta.variants.join("/")}`;
  const gs = $("geo-sliders"); gs.innerHTML = "";
  if (!preserveState) STATE.geo = {};
  for (const p of meta.params) {
    const v = (preserveState && STATE.geo[p.name] != null) ? STATE.geo[p.name] : p.default;
    STATE.geo[p.name] = v;
    gs.appendChild(sliderRow(p.label || p.name, p.min, p.max, p.step || (p.max - p.min) / 100, v, val => {
      STATE.geo[p.name] = p.int ? Math.round(val) : val; refresh();
    }, p.int));
  }
  const fs = $("flow-sliders"); fs.innerHTML = "";
  for (const p of INDEX.flow_params) {
    const key = p.name;
    const v = STATE[key] != null ? STATE[key] : p.default;
    STATE[key] = v;
    fs.appendChild(sliderRow(p.label, p.min, p.max, p.step || 0.1, v, val => {
      STATE[key] = val; refresh();
    }));
  }
}

function sliderRow(label, min, max, step, val, oninput, isInt = false) {
  const div = document.createElement("div");
  div.className = "slider-row";
  const lab = document.createElement("label");
  lab.innerHTML = `<span>${label}</span><span class="val">${isInt ? Math.round(val) : (+val).toFixed(3)}</span>`;
  const inp = document.createElement("input");
  inp.type = "range"; inp.min = min; inp.max = max; inp.step = step; inp.value = val;
  inp.oninput = () => { const v = +inp.value; lab.querySelector(".val").textContent = isInt ? Math.round(v) : v.toFixed(3); oninput(v); };
  div.appendChild(lab); div.appendChild(inp);
  return div;
}

// ---------- refresh: load family, snap default run, interpolate, render ----------
async function loadAndSnapDefault() {
  FAM = await loadFamily(STATE.family, DATA);
  doInterpolate();
  // On initial load / family switch, snap to the closest exact FEniCSx V4 run so the CFL is crisp & unblurred
  if (interp?.nearest) {
    const r = interp.nearest;
    STATE.geo = { ...r.geometry_params };
    if (r.Re != null) STATE.Re = r.Re;
    if (r.phi_in != null) STATE.phi_in = r.phi_in;
    if (r.h_um != null) STATE.h_um = r.h_um;
    buildSliders(true);
  }
  await refresh();
}

async function refresh() {
  FAM = await loadFamily(STATE.family, DATA);
  doInterpolate();
  populateExactRuns();
  updateNearest();
  updateSnippet();
  updateQoI();
  rebuildFieldTexture();
  resetParticles();
  if (interp) {
    if (STATE.probeX == null || STATE.probeX < interp.x0 || STATE.probeX > interp.x1) {
      STATE.probeX = interp.x0 + 0.5 * (interp.x1 - interp.x0);
    }
  }
  renderFrame();
  drawCharts();
}

function targetParams() {
  return { geometry_params: { ...STATE.geo }, Re: STATE.Re, phi_in: STATE.phi_in };
}

function doInterpolate() {
  const t = targetParams();
  if (STATE.variant === "V4-V1") {
    const a = interpolate(FAM, "V4", t);
    const b = interpolate(FAM, "V1", t);
    if (!a || !b) { interp = a || b; return; }
    const fields = {};
    for (const f of Object.keys(a.fields)) fields[f] = new Float32Array(a.fields[f].length);
    for (let i = 0; i < fields.ux.length; i++) {
      for (const f of Object.keys(fields)) fields[f][i] = (a.fields[f][i] || 0) - (b.fields[f] ? b.fields[f][i] : 0);
    }
    interp = { ...a, fields, nearest: a.nearest, dist: a.dist, exact: null, delta: true };
  } else {
    interp = interpolate(FAM, STATE.variant, t);
  }
  if (interp) {
    const map = makeMapper(interp.nx, interp.ny, interp.x0, interp.x1, interp.y0, interp.y1);
    interp.map = map;
    streamlines = computeStreamlines(map, interp.fields.ux, interp.fields.uy, interp.mask, { nSeeds: 44, ds: 0.05 });
  }
}

// ---------- geometry + rendering ----------
function viewTransform() {
  const cv = $("main-canvas");
  const W = cv.width, H = cv.height;
  const dx = Math.max(1e-6, interp.x1 - interp.x0), dy = Math.max(1e-6, interp.y1 - interp.y0);
  const pad = 16;
  const sx = (W - 2 * pad) / dx;
  const syIso = (H - 2 * pad) / dy;
  // Auto-expand Y (up to 3.2x sx) for long thin channels (dx/dy > 2.8) so the wall CFL band is clearly legible
  let sX = Math.min(sx, syIso);
  let sY = sX;
  if (STATE.expandY && dx / dy > 2.8) {
    sX = sx;
    sY = Math.min(syIso, sX * 3.2);
  }
  const ox = (W - dx * sX) / 2, oy = (H - dy * sY) / 2;
  return { W, H, sX, sY, ox, oy, dx, dy };
}

function toPx(x, y) {
  const v = viewTransform();
  return [v.ox + (x - interp.x0) * v.sX, v.oy + (interp.y1 - y) * v.sY];
}

function fieldArray() {
  return deriveField(STATE.field, interp.fields, interp.nx, interp.ny, interp.x0, interp.x1, interp.y0, interp.y1, interp.mask);
}

function fieldRange(arr) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < arr.length; i++) if (interp.mask[i]) { if (arr[i] < lo) lo = arr[i]; if (arr[i] > hi) hi = arr[i]; }
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  // For phi, anchor lower colorbar near zero (0 .. max(phi)) so blue CFL stands out unmistakably
  if (STATE.field === "phi" && STATE.variant !== "V4-V1") {
    lo = Math.min(lo, 0.0);
  }
  if (hi - lo < 1e-9) hi = lo + 1e-6;
  return [lo, hi];
}

function rebuildFieldTexture() {
  if (!interp) return;
  const arr = fieldArray();
  const [lo, hi] = fieldRange(arr);
  activeRange = [lo, hi];
  activeDiv = STATE.field === "p" || STATE.field === "vort" || STATE.variant === "V4-V1";
  const mode = activeDiv ? "div" : (STATE.field === "phi" ? "phi" : "seq");
  const nx = interp.nx, ny = interp.ny;
  const off = document.createElement("canvas");
  off.width = nx; off.height = ny;
  const octx = off.getContext("2d");
  const img = octx.createImageData(nx, ny);
  // 1) Color all fluid pixels cleanly
  for (let j = 0; j < ny; j++) {
    const outJ = ny - 1 - j;
    for (let i = 0; i < nx; i++) {
      const idx = j * nx + i;
      const p = (outJ * nx + i) * 4;
      if (!interp.mask[idx]) {
        img.data[p] = 0; img.data[p + 1] = 0; img.data[p + 2] = 0; img.data[p + 3] = 0;
        continue;
      }
      let t = (arr[idx] - lo) / (hi - lo);
      if (mode === "div") {
        const mAbs = Math.max(Math.abs(lo), Math.abs(hi), 1e-9);
        t = 0.5 + 0.5 * Math.max(-1, Math.min(1, arr[idx] / mAbs));
      } else if (mode === "phi") {
        t = Math.pow(Math.max(0, Math.min(1, t)), 1.25);
      }
      const [r, g, b] = cmap(t, mode);
      img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = 255;
    }
  }
  // 2) Bleed neighbor fluid RGB into adjacent transparent solid pixels (alpha stays 0) so bilinear
  // drawImage() anti-aliases slanted diagonal walls without dark muddy edge fringes.
  for (let j = 0; j < ny; j++) {
    const outJ = ny - 1 - j;
    for (let i = 0; i < nx; i++) {
      const idx = j * nx + i;
      if (interp.mask[idx]) continue;
      let rSum = 0, gSum = 0, bSum = 0, cnt = 0;
      for (const [di, dj] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
        const ni = i + di, nj = j + dj;
        if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
        const nIdx = nj * nx + ni;
        if (!interp.mask[nIdx]) continue;
        const np = ((ny - 1 - nj) * nx + ni) * 4;
        rSum += img.data[np]; gSum += img.data[np + 1]; bSum += img.data[np + 2]; cnt++;
      }
      if (cnt > 0) {
        const p = (outJ * nx + i) * 4;
        img.data[p] = (rSum / cnt) | 0;
        img.data[p + 1] = (gSum / cnt) | 0;
        img.data[p + 2] = (bSum / cnt) | 0;
        img.data[p + 3] = 0;
      }
    }
  }
  octx.putImageData(img, 0, 0);
  bgCanvas = off;
  drawColorbar(lo, hi, mode);
}

function drawColorbar(lo, hi, mode) {
  const cb = $("colorbar"), c = cb.getContext("2d");
  const W = cb.width, H = cb.height;
  c.clearRect(0, 0, W, H);
  for (let y = 0; y < H; y++) {
    const t = 1 - y / (H - 1);
    const [r, g, b] = cmap(t, mode);
    c.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
    c.fillRect(0, y, W, 1);
  }
  c.fillStyle = "rgba(5,8,15,0.75)";
  c.fillRect(0, 0, W, 16);
  c.fillRect(0, H - 16, W, 16);
  c.fillStyle = "#dbe7f5"; c.font = "9px monospace";
  const fmt = v => Math.abs(v) >= 100 || (Math.abs(v) > 0 && Math.abs(v) < 0.01) ? v.toExponential(1) : v.toFixed(2);
  c.fillText(fmt(hi), 2, 11);
  c.fillText(fmt(lo), 2, H - 4);
}

function renderFrame() {
  if (!interp || !bgCanvas) return;
  const cv = $("main-canvas");
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#08111f";
  ctx.fillRect(0, 0, cv.width, cv.height);
  const v = viewTransform();
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(bgCanvas, v.ox, v.oy, v.dx * v.sX, v.dy * v.sY);

  if (STATE.showLines) {
    ctx.strokeStyle = "rgba(255,255,255,0.32)";
    ctx.lineWidth = 0.8;
    for (const ln of streamlines) {
      ctx.beginPath();
      for (let k = 0; k < ln.length; k++) {
        const px = v.ox + (ln[k][0] - interp.x0) * v.sX;
        const py = v.oy + (interp.y1 - ln[k][1]) * v.sY;
        if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    }
  }
  if (STATE.showRBC && particles) {
    particles.draw(ctx, (x, y) => [v.ox + (x - interp.x0) * v.sX, v.oy + (interp.y1 - y) * v.sY], v.sY);
  }
  if (STATE.probeX != null) {
    const px = v.ox + (STATE.probeX - interp.x0) * v.sX;
    ctx.strokeStyle = "rgba(251,191,36,0.85)";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(px, v.oy); ctx.lineTo(px, v.oy + v.dy * v.sY);
    ctx.stroke();
    ctx.setLineDash([]);
    $("probe-label").textContent = `probe x* = ${STATE.probeX.toFixed(2)}` + (v.sY > v.sX * 1.05 ? ` (Y ×${(v.sY / v.sX).toFixed(1)})` : "");
  } else {
    $("probe-label").textContent = "hover channel to probe u*(y), φ(y)" + (v.sY > v.sX * 1.05 ? ` · Y ×${(v.sY / v.sX).toFixed(1)}` : "");
  }
}

function resetParticles() {
  if (!interp) return;
  particles = new RBCParticles(STATE.rbcCount);
  const shear = deriveField("shear", interp.fields, interp.nx, interp.ny, interp.x0, interp.x1, interp.y0, interp.y1, interp.mask);
  particles.setShear(shear);
  particles.reset(interp.map, interp.fields.ux, interp.fields.uy, interp.fields.phi, interp.mask,
    { phiWeight: STATE.phiWeight && STATE.variant === "V4" });
}

function animate() {
  if (particles) { particles.step(0.016, 1.0); }
  renderFrame();
  raf = requestAnimationFrame(animate);
}

// ---------- cross-section probe ----------
function onProbe(e) {
  if (!interp) return;
  const cv = $("main-canvas");
  const r = cv.getBoundingClientRect();
  const px = (e.clientX - r.left) * (cv.width / r.width);
  const v = viewTransform();
  const x = interp.x0 + (px - v.ox) / v.sX;
  if (x < interp.x0 || x > interp.x1) { STATE.probeX = null; }
  else STATE.probeX = x;
  drawCharts();
}

function drawCharts() {
  if (!interp || STATE.probeX == null) return;
  const fi = interp.map.ix(STATE.probeX);
  const i = Math.max(0, Math.min(interp.nx - 1, Math.round(fi)));
  plotChart($("chart-u"), interp, i, "ux", "#2dd4bf", "u*(y)");
  if (interp.fields.phi) plotChart($("chart-phi"), interp, i, "phi", "#fb7185", "φ(y)");
  else {
    const cv = $("chart-phi"), ctx = cv.getContext("2d");
    ctx.fillStyle = "#05080f"; ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = "#8aa0bd"; ctx.font = "11px monospace";
    ctx.fillText("φ(y) available in V4 (single-phase variant selected)", 18, cv.height / 2);
  }
}

function plotChart(cv, interpObj, i, field, color, label) {
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.fillStyle = "#05080f"; ctx.fillRect(0, 0, W, H);
  const ny = interpObj.ny, nx = interpObj.nx;
  const vals = [];
  let jMin = ny, jMax = -1;
  let lo = Infinity, hi = -Infinity;
  for (let j = 0; j < ny; j++) {
    const ok = interpObj.mask[j * nx + i];
    const v = ok ? interpObj.fields[field][j * nx + i] : NaN;
    vals.push(v);
    if (isFinite(v)) {
      jMin = Math.min(jMin, j);
      jMax = Math.max(jMax, j);
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  }
  if (!isFinite(lo) || jMax < jMin) return;
  if (field === "phi") lo = Math.min(lo, 0.0);
  if (hi - lo < 1e-6) { lo -= 0.05; hi += 0.05; }
  const jSpan = Math.max(1, jMax - jMin);
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
  let started = false;
  for (let j = jMin; j <= jMax; j++) {
    if (!isFinite(vals[j])) { started = false; continue; }
    const x = (vals[j] - lo) / (hi - lo) * (W - 36) + 12;
    const y = H - ((j - jMin) / jSpan) * (H - 30) - 6;
    if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.fillStyle = "#8aa0bd"; ctx.font = "10px monospace";
  ctx.fillText(`${label}  [${lo.toFixed(3)}, ${hi.toFixed(3)}]`, 10, 13);
}

// ---------- exact runs / snap ----------
function populateExactRuns() {
  const sel = $("exact-run-select");
  sel.innerHTML = "";
  const runs = FAM.runs.filter(r => r.variant === (STATE.variant === "V4-V1" ? "V4" : STATE.variant));
  for (const r of runs) {
    const o = document.createElement("option");
    o.value = r.id;
    o.textContent = `${r.id} · Re=${r.Re?.toFixed(1)} φ=${r.phi_in?.toFixed(3)}`;
    sel.appendChild(o);
  }
  if (interp?.exact) sel.value = interp.exact.id;
  else if (interp?.nearest) sel.value = interp.nearest.id;
}

function updateNearest() {
  if (!interp) return;
  $("nearest-dist").textContent = interp.dist != null ? interp.dist.toFixed(3) : "–";
  const badge = $("exact-badge");
  if (interp.exact) { badge.textContent = `exact CFD run: ${interp.exact.id}`; badge.classList.add("exact"); }
  else { badge.textContent = "RBF-interpolated across corpus"; badge.classList.remove("exact"); }
}

function snapToNearest() {
  if (!interp || !interp.nearest) return;
  applyExactRun(interp.nearest.id);
}

function applyExactRun(id) {
  const r = FAM.runs.find(x => x.id === id);
  if (!r) return;
  STATE.geo = { ...r.geometry_params };
  if (r.Re != null) STATE.Re = r.Re;
  if (r.phi_in != null) STATE.phi_in = r.phi_in;
  if (r.h_um != null) STATE.h_um = r.h_um;
  buildSliders(true);
  refresh();
}

// ---------- QoI cards ----------
function updateQoI() {
  const box = $("qoi-cards"); box.innerHTML = "";
  if (!interp || !interp.nearest) return;
  const q = interp.nearest.qoi || {};
  const cards = [
    ["dp*", q.dp_star], ["xr/h1", q.xr_over_h1], ["cfl/h", q.cfl_over_h_local],
    ["f_window", q.f_window], ["max φ", q.max_phi], ["min φ", q.min_phi],
    ["max τw*", q.max_wall_tau_star], ["max slip", q.max_slip],
    ["Re", interp.nearest.Re], ["φ_in", interp.nearest.phi_in], ["h1 (µm)", interp.nearest.h_um],
  ];
  for (const [k, v] of cards) {
    const d = document.createElement("div");
    d.className = "qoi" + (k === "f_window" && v != null && v > 0.05 ? " warn" : "");
    d.innerHTML = `<div class="k">${k}</div><div class="v">${v == null ? "–" : (+v).toPrecision(4)}</div>`;
    box.appendChild(d);
  }
}

// ---------- python snippet ----------
function updateSnippet() {
  const geo = Object.entries(STATE.geo).filter(([k]) => !k.startsWith("__"))
    .map(([k, v]) => `"${k}": ${typeof v === "number" ? +v.toFixed(4) : JSON.stringify(v)}`).join(", ");
  const vmap = { V1: "v1_newtonian", V2: "v2_carreau", V4: "v4_wu", "V4-V1": "v4_wu" };
  const preset = vmap[STATE.variant] || "v4_wu";
  const dHat = (8.0 / Math.max(STATE.h_um || 110, 1)).toFixed(4);
  const code = `import hemoflow

cfg = hemoflow.EulerEulerConfig.${preset}(
    re_p=${(+STATE.Re).toFixed(2)},
    d_hat=${dHat},                 # 8 um / h1
    phi_in=${(+STATE.phi_in).toFixed(3)},
)
# geometry ${STATE.family} (${familyMeta().name}), h1 = ${(+STATE.h_um).toFixed(1)} um
params = {${geo}}

# host-safe NumPy closure check:
from hemoflow.closures import viscosity as V
import numpy as np
phi = np.linspace(0.01, 0.6, 50)
print("mu_hat(phi) =", V.${STATE.variant === "V1" ? "newtonian" : STATE.variant === "V2" ? "bird_carreau" : "wu_high_shear"}(phi))

# full FEniCSx solve (requires dolfinx/dolfinx:v0.9.0 container):
# from hemoflow.solver.stationary import EulerEulerSolver
# res = EulerEulerSolver(cfg).solve("${STATE.family}", geometry_params=params)
# res.plot_streamlines_with_rbcs()
`;
  $("python-snippet").textContent = code;
}

function copySnippet() {
  navigator.clipboard.writeText($("python-snippet").textContent);
  $("copy-btn").textContent = "Copied";
  setTimeout(() => $("copy-btn").textContent = "Copy", 1200);
}

boot();
