"""FlowResult: stores a solved 2D flow field on a resampled grid + native arrays, with
NPZ round-trip, streamlines (RK4) and an optional matplotlib RBC-overlay plot.

Host-safe: only numpy/scipy at import; matplotlib is imported lazily inside the plot method.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["FlowResult"]


@dataclass
class FlowResult:
    """A solved (or loaded) 2D Euler-Euler flow field.

    Grid arrays (star units, h1*=1, U_mean*=1): ``grid_x`` (nx,), ``grid_y`` (ny,),
    ``grid_mask`` (ny, nx) bool (True = fluid), ``grid_ux``/``grid_uy``/``grid_p`` (ny, nx),
    ``grid_phi`` (ny, nx) (V4 only, else None).  Native arrays (cell/DOF) are optional.
    """
    grid_x: np.ndarray
    grid_y: np.ndarray
    grid_mask: np.ndarray
    grid_ux: np.ndarray
    grid_uy: np.ndarray
    grid_p: np.ndarray
    grid_phi: np.ndarray | None = None
    native: dict[str, np.ndarray] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.grid_x = np.asarray(self.grid_x, dtype=np.float64)
        self.grid_y = np.asarray(self.grid_y, dtype=np.float64)
        self.grid_mask = np.asarray(self.grid_mask, dtype=bool)
        for name in ("grid_ux", "grid_uy", "grid_p"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        if self.grid_phi is not None:
            self.grid_phi = np.asarray(self.grid_phi, dtype=np.float64)
        ny, nx = self.grid_mask.shape
        if self.grid_x.shape[0] != nx or self.grid_y.shape[0] != ny:
            raise ValueError("grid_x/grid_y length must match grid_mask shape")
        for name in ("grid_ux", "grid_uy", "grid_p"):
            if getattr(self, name).shape != (ny, nx):
                raise ValueError(f"{name} shape must be (ny, nx) = {(ny, nx)}")

    # ---- derived fields ----
    def speed(self) -> np.ndarray:
        return np.hypot(self.grid_ux, self.grid_uy)

    def shear_rate(self) -> np.ndarray:
        """gamma_dot* = sqrt(2 D:D) on the grid (central differences, masked)."""
        ux, uy = self.grid_ux, self.grid_uy
        dUx_dy, dUx_dx = np.gradient(ux, self.grid_y, self.grid_x, axis=(0, 1))
        dUy_dy, dUy_dx = np.gradient(uy, self.grid_y, self.grid_x, axis=(0, 1))
        dudx, dudy = dUx_dx, dUx_dy
        dvdx, dvdy = dUy_dx, dUy_dy
        D = 0.5 * np.array([[dudx + dudx, dudy + dvdx], [dudy + dvdx, dvdy + dvdy]])  # not used directly
        # 2 D:D = dudx^2 + dvdy^2 + 0.5*(dudy+dvdx)^2 ... use full: 2*(Dxx^2+Dyy^2)+4*Dxy^2
        Dxx, Dyy, Dxy = dudx, dvdy, 0.5 * (dudy + dvdx)
        g2 = 2.0 * (Dxx**2 + Dyy**2) + 4.0 * Dxy**2
        g = np.sqrt(g2)
        g[~self.grid_mask] = 0.0
        return g

    def vorticity(self) -> np.ndarray:
        """Out-of-plane vorticity omega_z = d(uy)/dx - d(ux)/dy on the masked grid."""
        dUx_dy, _ = np.gradient(self.grid_ux, self.grid_y, self.grid_x, axis=(0, 1))
        _, dUy_dx = np.gradient(self.grid_uy, self.grid_y, self.grid_x, axis=(0, 1))
        w = dUy_dx - dUx_dy
        w[~self.grid_mask] = 0.0
        return w

    def summary(self) -> dict[str, Any]:
        """Scalar QoI summary (dp*, max/min phi, max |u|, etc.)."""
        m = self.grid_mask
        out = {
            "family": self.metadata.get("family"),
            "variant": self.metadata.get("variant"),
            "nx": int(self.grid_x.size), "ny": int(self.grid_y.size),
            "n_fluid_cells": int(m.sum()),
            "umax": float(self.speed()[m].max()) if m.any() else 0.0,
            "p_min": float(self.grid_p[m].min()) if m.any() else 0.0,
            "p_max": float(self.grid_p[m].max()) if m.any() else 0.0,
            "dp_star": float(self.grid_p[m].max() - self.grid_p[m].min()) if m.any() else 0.0,
        }
        if self.grid_phi is not None:
            out["phi_max"] = float(self.grid_phi[m].max()) if m.any() else 0.0
            out["phi_min"] = float(self.grid_phi[m].min()) if m.any() else 0.0
            out["phi_mean"] = float(self.grid_phi[m].mean()) if m.any() else 0.0
        out.update({k: v for k, v in self.metadata.items() if k not in out})
        return out

    # ---- NPZ round-trip ----
    def to_npz(self, path) -> None:
        """Write fields + params/metadata JSON in a single .npz (round-trip with :meth:`from_npz`)."""
        import json
        arrays = {
            "grid_x": self.grid_x, "grid_y": self.grid_y, "grid_mask": self.grid_mask,
            "grid_ux": self.grid_ux, "grid_uy": self.grid_uy, "grid_p": self.grid_p,
        }
        if self.grid_phi is not None:
            arrays["grid_phi"] = self.grid_phi
        for k, v in self.native.items():
            arrays[k] = v
        arrays["_params_json"] = np.array(json.dumps(self.params, default=str))
        arrays["_meta_json"] = np.array(json.dumps(self.metadata, default=str))
        np.savez(path, **arrays)

    @classmethod
    def from_npz(cls, path) -> "FlowResult":
        import json
        z = np.load(path, allow_pickle=False)
        native = {k: z[k] for k in z.files
                  if k not in ("grid_x", "grid_y", "grid_mask", "grid_ux", "grid_uy",
                               "grid_p", "grid_phi", "_params_json", "_meta_json")}
        params = json.loads(str(z["_params_json"])) if "_params_json" in z.files else {}
        meta = json.loads(str(z["_meta_json"])) if "_meta_json" in z.files else {}
        return cls(
            grid_x=z["grid_x"], grid_y=z["grid_y"], grid_mask=z["grid_mask"],
            grid_ux=z["grid_ux"], grid_uy=z["grid_uy"], grid_p=z["grid_p"],
            grid_phi=z["grid_phi"] if "grid_phi" in z.files else None,
            native=native, params=params, metadata=meta,
        )

    # ---- streamlines (RK4 on the resampled grid, masked) ----
    def _bilinear(self, field: np.ndarray, x: float, y: float):
        """Bilinear interpolation of ``field`` (ny, nx) at (x, y); NaN outside fluid/extent."""
        gx, gy = self.grid_x, self.grid_y
        nx, ny = gx.size, gy.size
        if x < gx[0] or x > gx[-1] or y < gy[0] or y > gy[-1]:
            return np.nan
        i = int(np.clip(np.searchsorted(gx, x) - 1, 0, nx - 2))
        j = int(np.clip(np.searchsorted(gy, y) - 1, 0, ny - 2))
        tx = (x - gx[i]) / (gx[i + 1] - gx[i]) if gx[i + 1] != gx[i] else 0.0
        ty = (y - gy[j]) / (gy[j + 1] - gy[j]) if gy[j + 1] != gy[j] else 0.0
        f = field[j, i] * (1 - tx) * (1 - ty) + field[j, i + 1] * tx * (1 - ty) \
            + field[j + 1, i] * (1 - tx) * ty + field[j + 1, i + 1] * tx * ty
        # mask check at the 4 corners: require at least the cell to be fluid
        m = (self.grid_mask[j, i] and self.grid_mask[j, i + 1]
             and self.grid_mask[j + 1, i] and self.grid_mask[j + 1, i + 1])
        return f if m else np.nan

    def velocity_at(self, x: float, y: float):
        ux = self._bilinear(self.grid_ux, x, y)
        uy = self._bilinear(self.grid_uy, x, y)
        return ux, uy

    def compute_streamlines(self, seeds=None, n_seeds: int = 60, ds: float = 0.05,
                            max_steps: int = 4000, rk: int = 4) -> list[np.ndarray]:
        """Integrate streamlines from inlet/vortex seeds with RK2/RK4, respecting the mask.

        ``seeds``: optional list of (x, y).  Default: dense seeds along the inlet (x = grid_x[0]).
        Returns a list of ``(N, 2)`` point arrays.  Stops at outlets, walls, or stagnation.
        """
        gx, gy = self.grid_x, self.grid_y
        if seeds is None:
            x0 = gx[0] + 1e-3 * (gx[-1] - gx[0])
            ys = np.linspace(gy[0] + 0.02, gy[-1] - 0.02, n_seeds)
            seeds = [(x0, float(y)) for y in ys]

        def vel(x, y):
            ux, uy = self.velocity_at(x, y)
            if not (np.isfinite(ux) and np.isfinite(uy)):
                return None
            s = np.hypot(ux, uy)
            if s < 1e-9:
                return None
            return ux, uy, s

        lines: list[np.ndarray] = []
        for (x0, y0) in seeds:
            pts = [(x0, y0)]
            x, y = x0, y0
            for _ in range(max_steps):
                v0 = vel(x, y)
                if v0 is None:
                    break
                if rk == 4:
                    k1u, k1v, _ = v0
                    v2 = vel(x + 0.5 * ds * k1u, y + 0.5 * ds * k1v)
                    if v2 is None: break
                    k2u, k2v, _ = v2
                    v3 = vel(x + 0.5 * ds * k2u, y + 0.5 * ds * k2v)
                    if v3 is None: break
                    k3u, k3v, _ = v3
                    v4 = vel(x + ds * k3u, y + ds * k3v)
                    if v4 is None: break
                    k4u, k4v, _ = v4
                    x += ds * (k1u + 2 * k2u + 2 * k3u + k4u) / 6.0
                    y += ds * (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
                else:
                    k1u, k1v, _ = v0
                    v2 = vel(x + ds * k1u, y + ds * k1v)
                    if v2 is None: break
                    k2u, k2v, _ = v2
                    x += 0.5 * ds * (k1u + k2u)
                    y += 0.5 * ds * (k1v + k2v)
                if not (gx[0] <= x <= gx[-1] and gy[0] <= y <= gy[-1]):
                    break
                pts.append((x, y))
                if len(pts) > 1 and np.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]) < 1e-7:
                    break
            if len(pts) > 1:
                lines.append(np.asarray(pts))
        return lines

    def plot_streamlines_with_rbcs(self, ax=None, n_rbcs: int = 200, seed: int = 0,
                                   weighted_by_phi: bool = True, **kwargs):
        """Draw streamlines + explicit biconcave RBC markers (matplotlib, lazy import).

        In two-phase (V4) mode with ``weighted_by_phi``, RBC glyph placement is weighted by
        local ``phi / phi_max`` (rejection sampling), revealing the cell-free layer; in
        single-phase mode RBCs are distributed uniformly by flux.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from matplotlib.collections import PatchCollection
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        m = self.grid_mask
        X, Y = np.meshgrid(self.grid_x, self.grid_y)
        ax.contourf(X, Y, np.where(m, self.speed(), np.nan), levels=24, cmap="viridis", alpha=0.85)
        lines = self.compute_streamlines(n_seeds=40)
        for ln in lines:
            ax.plot(ln[:, 0], ln[:, 1], color="white", lw=0.4, alpha=0.35)
        # RBC glyphs
        rng = np.random.default_rng(seed)
        phi = self.grid_phi if (weighted_by_phi and self.grid_phi is not None) else None
        phi_max = float(np.nanmax(self.grid_phi[m])) if phi is not None and m.any() else 1.0
        placed = 0
        attempts = 0
        while placed < n_rbcs and attempts < n_rbcs * 60:
            attempts += 1
            i = rng.integers(0, self.grid_x.size)
            j = rng.integers(0, self.grid_y.size)
            if not m[j, i]:
                continue
            if phi is not None:
                p = float(phi[j, i]) / (phi_max if phi_max > 0 else 1.0)
                if rng.random() > p:
                    continue
            x, y = float(self.grid_x[i]), float(self.grid_y[j])
            ux, uy = self.grid_ux[j, i], self.grid_uy[j, i]
            ang = np.degrees(np.arctan2(uy, ux))
            ax.add_patch(Ellipse((x, y), 0.18, 0.09, angle=ang, fc="#e63946", ec="white", lw=0.3, alpha=0.9))
            placed += 1
        ax.set_aspect("equal")
        ax.set_xlim(self.grid_x[0], self.grid_x[-1])
        ax.set_ylim(self.grid_y[0], self.grid_y[-1])
        ax.set_title("streamlines + RBCs")
        return ax
