// interpolator.js — lazy load + decode site/data/G<k>.bin.gz and interpolate runs.
// Uses the browser-native DecompressionStream('gzip'); no build step.

const cache = new Map(); // family -> decoded family object

export async function gunzipBytes(bytes) {
  const ds = new DecompressionStream("gzip");
  const stream = new Blob([bytes]).stream().pipeThrough(ds);
  const buf = await new Response(stream).arrayBuffer();
  return new Uint8Array(buf);
}

function b64ToU8(s) {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function b64ToF32(s) {
  const u8 = b64ToU8(s);
  return new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4);
}

export function decodeRun(run) {
  const nx = run.nx, ny = run.ny;
  const x = b64ToF32(run.x);
  const y = b64ToF32(run.y);
  const maskBytes = b64ToU8(run.mask);
  const mask = new Uint8Array(nx * ny);
  for (let i = 0; i < maskBytes.length * 8 && i < nx * ny; i++) {
    mask[i] = (maskBytes[i >> 3] >> (7 - (i & 7))) & 1;
  }
  const fields = {};
  for (const [name, f] of Object.entries(run.fields)) {
    const q = b64ToU8(f.data);
    const arr = new Float32Array(nx * ny);
    const lo = f.min, hi = f.max, span = hi - lo;
    for (let i = 0; i < nx * ny; i++) arr[i] = lo + (q[i] / 255) * span;
    fields[name] = arr;
  }
  return { ...run, nx, ny, x, y, mask, fields };
}

export async function loadFamily(family, dataBase = "data/") {
  if (cache.has(family)) return cache.get(family);
  const resp = await fetch(`${dataBase}${family}.bin.gz`);
  if (!resp.ok) throw new Error(`failed to load ${family}.bin.gz`);
  const gz = new Uint8Array(await resp.arrayBuffer());
  const raw = await gunzipBytes(gz);
  const text = new TextDecoder().decode(raw);
  const obj = JSON.parse(text);
  obj.runs = obj.runs.map(decodeRun);
  // observed ranges for feature normalization
  const keys = new Set();
  for (const r of obj.runs) for (const k of Object.keys(r.geometry_params || {})) keys.add(k);
  const ranges = { Re: [Infinity, -Infinity], phi_in: [Infinity, -Infinity] };
  for (const k of keys) ranges[k] = [Infinity, -Infinity];
  for (const r of obj.runs) {
    if (r.Re != null) { ranges.Re[0] = Math.min(ranges.Re[0], r.Re); ranges.Re[1] = Math.max(ranges.Re[1], r.Re); }
    if (r.phi_in != null) { ranges.phi_in[0] = Math.min(ranges.phi_in[0], r.phi_in); ranges.phi_in[1] = Math.max(ranges.phi_in[1], r.phi_in); }
    for (const k of keys) {
      const v = (r.geometry_params || {})[k];
      if (typeof v === "number") { ranges[k][0] = Math.min(ranges[k][0], v); ranges[k][1] = Math.max(ranges[k][1], v); }
    }
  }
  for (const k of Object.keys(ranges)) {
    if (ranges[k][0] === Infinity) ranges[k] = [0, 1];
    if (ranges[k][1] - ranges[k][0] < 1e-12) ranges[k] = [ranges[k][0], ranges[k][0] + 1];
  }
  obj.paramKeys = [...keys];
  obj.ranges = ranges;
  cache.set(family, obj);
  return obj;
}

// feature vector for a target {geometry_params, Re, phi_in}
function features(obj, geo, Re, phiIn) {
  const f = [];
  for (const k of obj.paramKeys) {
    const [lo, hi] = obj.ranges[k];
    const v = (geo && geo[k] != null) ? geo[k] : lo;
    f.push((v - lo) / (hi - lo));
  }
  const [rlo, rhi] = obj.ranges.Re; f.push((Re - rlo) / (rhi - rlo));
  const [plo, phi] = obj.ranges.phi_in; f.push((phiIn - plo) / (phi - plo));
  return f;
}

function runFeatures(obj, r) {
  const f = [];
  for (const k of obj.paramKeys) {
    const [lo, hi] = obj.ranges[k];
    const v = (r.geometry_params || {})[k];
    f.push((typeof v === "number" ? (v - lo) : 0) / (hi - lo));
  }
  const [rlo, rhi] = obj.ranges.Re; f.push(((r.Re ?? rlo) - rlo) / (rhi - rlo));
  const [plo, phi] = obj.ranges.phi_in; f.push(((r.phi_in ?? plo) - plo) / (phi - plo));
  return f;
}

function dist2(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return s;
}

/**
 * Interpolate top-k nearest runs of a family/variant at target params using sharp RBF weights.
 * Returns { nx, ny, x0, x1, y0, y1, mask(Uint8), fields:{name:Float32Array}, nearest, dist, exact }
 */
export function interpolate(obj, variant, target, sigma = 0.16, topK = 4) {
  const runs = obj.runs.filter(r => r.variant === variant);
  if (!runs.length) return null;
  const tf = features(obj, target.geometry_params, target.Re, target.phi_in);
  const scored = [];
  for (const r of runs) {
    const d2 = dist2(tf, runFeatures(obj, r));
    scored.push({ r, d2, w: Math.exp(-d2 / (2 * sigma * sigma)) });
  }
  scored.sort((a, b) => a.d2 - b.d2);
  const nearest = scored[0].r;
  const best = scored[0].d2;
  const nx = runs[0].nx, ny = runs[0].ny, n = nx * ny;

  // Exact or near-exact run: return unblurred raw decoded fields
  if (best < 1e-8) {
    let x0 = nearest.x[0], x1 = nearest.x[nx - 1], y0 = nearest.y[0], y1 = nearest.y[ny - 1];
    return { nx, ny, x0, x1, y0, y1, mask: nearest.mask, fields: nearest.fields, nearest, dist: Math.sqrt(best), exact: nearest };
  }

  const sub = scored.slice(0, Math.min(topK, scored.length));
  let wsum = 0; for (const s of sub) wsum += s.w;
  if (wsum < 1e-12) { for (const s of sub) s.w = 1 / (s.d2 + 1e-6); wsum = 0; for (const s of sub) wsum += s.w; }

  const mask = new Uint8Array(n);
  const fieldNames = Object.keys(runs[0].fields);
  const out = {};
  for (const f of fieldNames) out[f] = new Float32Array(n);
  let x0 = 0, x1 = 0, y0 = 0, y1 = 0, wacc = 0;
  for (const s of sub) {
    const r = s.r, w = s.w;
    x0 += w * r.x[0]; x1 += w * r.x[nx - 1]; y0 += w * r.y[0]; y1 += w * r.y[ny - 1]; wacc += w;
  }
  x0 /= wacc; x1 /= wacc; y0 /= wacc; y1 /= wacc;

  for (let i = 0; i < n; i++) {
    let mw = 0, wm = 0;
    const acc = {};
    for (const f of fieldNames) acc[f] = 0;
    for (const s of sub) {
      const r = s.r, w = s.w;
      if (r.mask[i]) { mw += w; for (const f of fieldNames) acc[f] += w * r.fields[f][i]; }
      wm += w;
    }
    if (wm > 0) mask[i] = (mw / wm) >= 0.5 ? 1 : 0;
    if (mw > 1e-12) for (const f of fieldNames) out[f][i] = acc[f] / mw;
    else for (const f of fieldNames) out[f][i] = 0;
  }
  return { nx, ny, x0, x1, y0, y1, mask, fields: out, nearest, dist: Math.sqrt(best), exact: null };
}

/** Compute a derived field (speed/shear/vort/p/phi) using true physical dx, dy spacing. */
export function deriveField(name, f, nx, ny, x0 = 0, x1 = 1, y0 = 0, y1 = 1, mask = null) {
  const ux = f.ux, uy = f.uy;
  const out = new Float32Array(nx * ny);
  const dx = nx > 1 ? (x1 - x0) / (nx - 1) : 1;
  const dy = ny > 1 ? (y1 - y0) / (ny - 1) : 1;
  if (name === "speed") {
    for (let i = 0; i < nx * ny; i++) out[i] = (!mask || mask[i]) ? Math.hypot(ux[i], uy[i]) : 0;
  } else if (name === "shear" || name === "vort") {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const idx = j * nx + i;
        if (mask && !mask[idx]) continue;
        const i0 = Math.max(0, i - 1), i1 = Math.min(nx - 1, i + 1);
        const j0 = Math.max(0, j - 1), j1 = Math.min(ny - 1, j + 1);
        const spanX = (i1 - i0) * dx || dx;
        const spanY = (j1 - j0) * dy || dy;
        const dudx = (ux[j * nx + i1] - ux[j * nx + i0]) / spanX;
        const dudy = (ux[j1 * nx + i] - ux[j0 * nx + i]) / spanY;
        const dvdx = (uy[j * nx + i1] - uy[j * nx + i0]) / spanX;
        const dvdy = (uy[j1 * nx + i] - uy[j0 * nx + i]) / spanY;
        if (name === "shear") {
          const Dxy = 0.5 * (dudy + dvdx);
          out[idx] = Math.sqrt(2 * (dudx * dudx + dvdy * dvdy) + 4 * Dxy * Dxy);
        } else {
          out[idx] = dvdx - dudy;
        }
      }
    }
  } else if (name === "p") {
    return f.p;
  } else if (name === "phi") {
    return f.phi || new Float32Array(nx * ny);
  }
  return out;
}
