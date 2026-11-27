// rbc_streamlines.js — RK4 streamlines + animated biconcave RBC glyphs distributed on streamlines.

export function makeMapper(nx, ny, x0, x1, y0, y1) {
  return {
    nx, ny, x0, x1, y0, y1,
    gx: i => x0 + (i / (nx - 1)) * (x1 - x0),
    gy: j => y0 + (j / (ny - 1)) * (y1 - y0),
    ix: x => (x - x0) / (x1 - x0) * (nx - 1),
    jy: y => (y - y0) / (y1 - y0) * (ny - 1),
  };
}

/** Bilinear sample of a grid field at physical (x,y); uses fluid-masked corner weighting so thin slanted arms never drop streamlines. */
export function sample(map, arr, mask, x, y) {
  const fi = map.ix(x), fj = map.jy(y);
  if (fi < 0 || fi > map.nx - 1 || fj < 0 || fj > map.ny - 1) return NaN;
  const iN = Math.min(map.nx - 1, Math.max(0, Math.round(fi)));
  const jN = Math.min(map.ny - 1, Math.max(0, Math.round(fj)));
  if (!mask[jN * map.nx + iN]) return NaN;
  const i = Math.min(map.nx - 2, Math.max(0, Math.floor(fi)));
  const j = Math.min(map.ny - 2, Math.max(0, Math.floor(fj)));
  const tx = fi - i, ty = fj - j;
  const idx00 = j * map.nx + i, idx10 = idx00 + 1;
  const idx01 = (j + 1) * map.nx + i, idx11 = idx01 + 1;
  const w00 = mask[idx00] ? (1 - tx) * (1 - ty) : 0;
  const w10 = mask[idx10] ? tx * (1 - ty) : 0;
  const w01 = mask[idx01] ? (1 - tx) * ty : 0;
  const w11 = mask[idx11] ? tx * ty : 0;
  const wsum = w00 + w10 + w01 + w11;
  if (wsum < 1e-6) return NaN;
  return (arr[idx00] * w00 + arr[idx10] * w10 + arr[idx01] * w01 + arr[idx11] * w11) / wsum;
}

export function velocityAt(map, ux, uy, mask, x, y) {
  const a = sample(map, ux, mask, x, y), b = sample(map, uy, mask, x, y);
  if (!isFinite(a) || !isFinite(b)) return null;
  const s = Math.hypot(a, b);
  if (s < 1e-7) return null;
  return [a, b, s];
}

/** RK4 unit-tangent streamlines from dense inlet + internal vortex/branch seeds. */
export function computeStreamlines(map, ux, uy, mask, { nSeeds = 48, ds = 0.05, maxSteps = 2200 } = {}) {
  const lines = [];
  const xSpan = map.x1 - map.x0;
  const ySpan = map.y1 - map.y0;
  const stepDist = Math.max(0.02, ds * Math.min(xSpan, ySpan) * 0.08);
  const seeds = [];
  // Dense inlet seeds
  const xIn = map.x0 + 0.008 * xSpan;
  for (let s = 0; s < nSeeds; s++) {
    seeds.push([xIn, map.y0 + (s + 0.5) / nSeeds * ySpan]);
  }
  // Additional axial station seeds to illuminate downstream bifurcations & recirculation vortices
  for (const frac of [0.22, 0.45, 0.68]) {
    const xS = map.x0 + frac * xSpan;
    for (let s = 0; s < Math.max(10, Math.floor(nSeeds / 3)); s++) {
      seeds.push([xS, map.y0 + (s + 0.5) / Math.max(10, Math.floor(nSeeds / 3)) * ySpan]);
    }
  }

  function unitVel(x, y) {
    const v = velocityAt(map, ux, uy, mask, x, y);
    if (!v) return null;
    return [v[0] / v[2], v[1] / v[2]];
  }

  for (const [sx, sy] of seeds) {
    if (!unitVel(sx, sy)) continue;
    let x = sx, y = sy;
    const pts = [[x, y]];
    for (let k = 0; k < maxSteps; k++) {
      const k1 = unitVel(x, y); if (!k1) break;
      const k2 = unitVel(x + 0.5 * stepDist * k1[0], y + 0.5 * stepDist * k1[1]); if (!k2) break;
      const k3 = unitVel(x + 0.5 * stepDist * k2[0], y + 0.5 * stepDist * k2[1]); if (!k3) break;
      const k4 = unitVel(x + stepDist * k3[0], y + stepDist * k3[1]); if (!k4) break;
      const nxPos = x + stepDist * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6;
      const nyPos = y + stepDist * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6;
      if (nxPos < map.x0 || nxPos > map.x1 || nyPos < map.y0 || nyPos > map.y1) break;
      if (!unitVel(nxPos, nyPos)) break;
      x = nxPos; y = nyPos;
      pts.push([x, y]);
      // Break if loop closes near start (recirculation vortex)
      if (pts.length > 20 && Math.hypot(x - sx, y - sy) < stepDist * 0.9) break;
    }
    if (pts.length >= 4) lines.push(pts);
  }
  return lines;
}

/**
 * Animated biconcave RBC glyphs moving explicitly on streamlines / flow field.
 *
 * When phiWeight is active (V4), RBCs are seeded and retained via rejection sampling
 * proportional to local phi / phi_max — making the Cell-Free Layer (CFL) near walls,
 * core enrichment, and bifurcation plasma skimming directly visible relative to uniform V1/V2.
 */
export class RBCParticles {
  constructor(count) {
    this.count = count;
    this.p = [];
    this.rng = Math.random;
    this._shear = null;
  }

  _spawnAlongField(inletOnly = false) {
    const map = this.map;
    const pool = inletOnly && this.inletCells.length ? this.inletCells : this.fluidCells;
    if (!pool.length) return null;
    const dxCell = (map.x1 - map.x0) / Math.max(map.nx - 1, 1);
    const dyCell = (map.y1 - map.y0) / Math.max(map.ny - 1, 1);
    for (let attempt = 0; attempt < 150; attempt++) {
      const idx = pool[(this.rng() * pool.length) | 0];
      const i = idx % map.nx, j = (idx / map.nx) | 0;
      const x = map.gx(i) + (this.rng() - 0.5) * dxCell;
      const y = map.gy(j) + (this.rng() - 0.5) * dyCell;
      const v = velocityAt(map, this.ux, this.uy, this.mask, x, y);
      if (!v) continue;
      if (this.phiWeight && this.phi) {
        const ph = sample(map, this.phi, this.mask, x, y);
        if (!isFinite(ph)) continue;
        // Contrast boost around CFL wall depletion (keeps dense core, rarefies depleted wall layer)
        const ratio = Math.max(0, ph / Math.max(this.phiMax, 1e-6));
        const acceptProb = this.phiMin < this.phiMax * 0.92
          ? Math.pow(Math.max(0, (ph - 0.55 * this.phiMin) / Math.max(this.phiMax - 0.55 * this.phiMin, 1e-5)), 1.35)
          : ratio;
        if (this.rng() > acceptProb) continue;
      }
      const sh = this._shear ? sample(map, this._shear, this.mask, x, y) : 0;
      const shear = isFinite(sh) ? sh : 0;
      return {
        x, y,
        rot: Math.atan2(v[1], v[0]),
        elong: Math.min(0.65, Math.abs(shear) * 0.015),
        shear,
        age: this.rng() * 2,
      };
    }
    return null;
  }

  reset(map, ux, uy, phi, mask, { phiWeight = true } = {}) {
    this.map = map; this.ux = ux; this.uy = uy; this.phi = phi; this.mask = mask;
    this.phiWeight = phiWeight;
    this.fluidCells = [];
    this.inletCells = [];
    let phiMax = 1e-9, phiMin = Infinity;
    for (let idx = 0; idx < mask.length; idx++) {
      if (!mask[idx]) continue;
      this.fluidCells.push(idx);
      const i = idx % map.nx;
      if (i <= Math.max(2, Math.floor(0.03 * map.nx))) this.inletCells.push(idx);
      if (phi) {
        phiMax = Math.max(phiMax, phi[idx]);
        phiMin = Math.min(phiMin, phi[idx]);
      }
    }
    if (!phi) phiMin = 0;
    this.phiMax = phiMax;
    this.phiMin = isFinite(phiMin) ? phiMin : 0;
    this.p = [];
    for (let k = 0; k < this.count; k++) {
      const cell = this._spawnAlongField(false);
      if (cell) this.p.push(cell);
    }
  }

  setShear(shearArr) { this._shear = shearArr; }

  step(dt, speedMult = 1.0) {
    if (!this.map) return;
    const map = this.map;
    // Scale step cleanly to domain length so particles traverse channel in ~5-8s at speedMult=1
    const xSpan = map.x1 - map.x0;
    const advScale = dt * speedMult * (xSpan / 6.0);
    for (let idx = 0; idx < this.p.length; idx++) {
      const q = this.p[idx];
      const v1 = velocityAt(map, this.ux, this.uy, this.mask, q.x, q.y);
      if (!v1) {
        const rep = this._spawnAlongField(true);
        if (rep) this.p[idx] = rep;
        continue;
      }
      // RK2 midpoint advection proportional to true local u*(x,y)
      const xm = q.x + 0.5 * v1[0] * advScale;
      const ym = q.y + 0.5 * v1[1] * advScale;
      const v2 = velocityAt(map, this.ux, this.uy, this.mask, xm, ym) || v1;
      q.x += v2[0] * advScale;
      q.y += v2[1] * advScale;
      if (q.x > map.x1 - 0.005 * xSpan || q.x < map.x0 || q.y < map.y0 || q.y > map.y1 ||
          !velocityAt(map, this.ux, this.uy, this.mask, q.x, q.y)) {
        const rep = this._spawnAlongField(true);
        if (rep) this.p[idx] = rep;
        continue;
      }
      // In V4, prevent numerical RK drift into deep depleted CFL walls if spawned in core
      if (this.phiWeight && this.phi) {
        const ph = sample(map, this.phi, this.mask, q.x, q.y);
        if (isFinite(ph) && ph < this.phiMin + 0.08 * (this.phiMax - this.phiMin) && this.rng() < 0.08) {
          const rep = this._spawnAlongField(false);
          if (rep) { this.p[idx] = rep; continue; }
        }
      }
      q.rot = Math.atan2(v2[1], v2[0]);
      const sh = this._shear ? sample(map, this._shear, this.mask, q.x, q.y) : 0;
      q.shear = isFinite(sh) ? sh : 0;
      q.elong = Math.min(0.65, Math.abs(q.shear) * 0.015);
    }
  }

  draw(ctx, toPx) {
    const tauLo = 8.0; // elevated shear-rate highlight threshold in star units
    for (const q of this.p) {
      const [px, py] = toPx(q.x, q.y);
      const L = 6.2 * (1 + q.elong), W = 3.0 * (1 - 0.25 * q.elong);
      ctx.save();
      ctx.translate(px, py);
      ctx.rotate(q.rot);
      // Outer biconcave discoid
      ctx.beginPath();
      ctx.ellipse(0, 0, L, W, 0, 0, Math.PI * 2);
      ctx.fillStyle = q.shear > tauLo ? "rgba(244,63,94,0.92)" : "rgba(225,29,72,0.86)";
      ctx.fill();
      ctx.lineWidth = q.shear > tauLo ? 1.5 : 0.7;
      ctx.strokeStyle = q.shear > tauLo ? "#fbbf24" : "rgba(255,255,255,0.65)";
      ctx.stroke();
      // Central pallor dimple
      ctx.beginPath();
      ctx.ellipse(0, 0, L * 0.42, W * 0.45, 0, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(110,10,36,0.58)";
      ctx.fill();
      ctx.restore();
    }
  }
}
