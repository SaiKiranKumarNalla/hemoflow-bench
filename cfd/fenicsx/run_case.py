"""Single-case steady 2D FEniCSx Euler–Euler runner, continuation solver, and .npz extractor.

Runs inside ``ghcr.io/fenics/dolfinx/dolfinx:v0.9.0`` (or any environment with dolfinx 0.9+):
1. Builds the exact same quadrilateral mesh as OpenFOAM ``blockMesh`` via
   ``cfd.fenicsx.mesh_from_spec``.
2. Applies developed inlet velocity (Poiseuille; tube profile for KG) and uniform ``phi_in``
   for V3 ``EE-DL`` (V4 ``EE-full`` developed precursor / extra closures left as TODO hook).
3. Solves the monolithic nonlinear two-fluid residual from ``cfd.fenicsx.twofluid`` via
   Newton + continuation: ``Re_p: 1 -> target``, then ``phi_in: 0.01 -> target`` in <= 4 steps
   each, with adaptive step halving on failure (max 3 halvings per segment).
4. Extracts cell-wise fields, fixed per-family resampling grid + exact fluid mask, outlet
   profiles, and scalars (``dp_star``, ``xr_over_h1``, ``max_wall_tau_star``, ``f_window``,
   ``outlet_phi_mean``, Sezer Eq. 21 ``cfl_thickness_star``) to ``<run_id>.npz`` using the
   OpenFOAM schema plus multiphase fields (``phi``, ``U_rbc_star``, ``grid_phi``, ``grid_urbc_*``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import dolfinx
import numpy as np
import ufl
from dolfinx import fem, geometry, mesh
from mpi4py import MPI

# Ensure repo root is importable whether invoked as script or module.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cfd.fenicsx import mesh_from_spec as mfs  # noqa: E402
from cfd.fenicsx import twofluid as tf  # noqa: E402
from cfd.gen.geometry import MeshSpec, SampleLine  # noqa: E402
from rbc.postprocess import reattachment_length, trapezoid_weights, window_fraction  # noqa: E402

# Default star-unit shear window (matches campaign/run_batch.py).
DEFAULT_TAU_STAR_LO = 30.0 / 17.25
DEFAULT_TAU_STAR_HI = 60.0 / 17.25

# Fixed resampling grid dimensions (Nx, Ny) per family — matches campaign/run_batch.py
# so FEniCSx V3/V4 and OpenFOAM V1/V2 pair pixel-for-pixel.
FAMILY_GRID_SHAPES: dict[str, tuple[int, int]] = {
    "G1": (256, 64),
    "G2": (256, 96),
    "G3": (256, 64),
    "G4": (192, 192),
    "G5": (192, 192),
    "G6": (192, 192),
    "G7": (256, 192),
    "KG": (256, 96),
}

# Expansion-type families for reattachment length extraction.
EXPANSION_REATTACHMENT_SPECS: dict[str, tuple[str, str | None]] = {
    "G2": ("nearWallLower", None),
    "G3": ("nearWallDownstream", "L_t"),
    "G7": ("nearWallExpansion", None),
    "KG": ("nearWallOuter", None),
}


def fenicsx_code_version(root: Path | str | None = None) -> str:
    """Return SHA-256 provenance hash over all ``cfd/fenicsx/*.py`` files."""
    base = Path(root).resolve() if root is not None else _REPO_ROOT.resolve()
    fenicsx_dir = base / "cfd" / "fenicsx"
    files = sorted(p for p in fenicsx_dir.glob("*.py") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No .py files found in {fenicsx_dir}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(base).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def peak_ram_mb() -> float:
    """Return process peak resident set size in MiB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


# ---------------------------------------------------------------------------
# Exact fluid mask & FEM probing (zero scipy dependency)
# ---------------------------------------------------------------------------

def _points_in_convex_quad(
    px: np.ndarray,
    py: np.ndarray,
    quad: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]],
    tol: float = 1e-9,
) -> np.ndarray:
    inside = np.ones(px.shape, dtype=bool)
    for k in range(4):
        x0, y0 = quad[k]
        x1, y1 = quad[(k + 1) % 4]
        cross = (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)
        inside &= cross >= -tol
    return inside


def build_fluid_mask(spec: MeshSpec, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """Build exact 2D fluid boolean mask ``(Ny, Nx)`` matching ``campaign/run_batch.py``."""
    xx, yy = np.meshgrid(grid_x, grid_y)
    mask = np.zeros(xx.shape, dtype=bool)

    if spec.family == "G6":
        Rc = float(spec.params["Rc_ratio"])
        n_bends = int(spec.params["n_bends"])
        L_straight = float(spec.params.get("L_straight", 2.0))
        r_in = Rc - 0.5
        r_out = Rc + 0.5
        tol = 1e-9

        curved_indices: set[int] = set()
        for k in range(n_bends):
            curved_indices.add(1 + 3 * k)
            curved_indices.add(1 + 3 * k + 1)

        for idx, blk in enumerate(spec.blocks_2d):
            if idx in curved_indices:
                continue
            quad = tuple(spec.vertices_2d[i] for i in blk.verts)
            mask |= _points_in_convex_quad(xx, yy, quad, tol=tol)  # type: ignore[arg-type]

        for k in range(n_bends):
            yc = 2.0 * k * Rc + 0.5 + Rc
            if k % 2 == 0:
                cx = 0.0
                r = np.hypot(xx - cx, yy - yc)
                mask |= (xx >= cx - tol) & (r >= r_in - tol) & (r <= r_out + tol)
            else:
                cx = -L_straight
                r = np.hypot(xx - cx, yy - yc)
                mask |= (xx <= cx + tol) & (r >= r_in - tol) & (r <= r_out + tol)
        return mask

    for blk in spec.blocks_2d:
        quad = tuple(spec.vertices_2d[i] for i in blk.verts)
        mask |= _points_in_convex_quad(xx, yy, quad)  # type: ignore[arg-type]
    return mask


def _locate_points_cells(
    msh: mesh.Mesh,
    tree: geometry.BoundingBoxTree,
    pts_2d: np.ndarray,
    cell_midpoints_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map 2D query points to containing dolfinx cells, with nearest-cell fallback on boundary eps."""
    n_pts = pts_2d.shape[0]
    pts_3d = np.zeros((n_pts, 3), dtype=np.float64)
    pts_3d[:, :2] = pts_2d
    candidates = geometry.compute_collisions_points(tree, pts_3d)
    colliding = geometry.compute_colliding_cells(msh, candidates, pts_3d)

    cell_ids = np.empty(n_pts, dtype=np.int32)
    missed: list[int] = []
    for i in range(n_pts):
        links = colliding.links(i)
        if len(links) > 0:
            cell_ids[i] = int(links[0])
        else:
            missed.append(i)

    if missed:
        # Rare boundary points (e.g. curved arc chord epsilon): snap to nearest cell midpoint.
        miss_pts = pts_2d[missed]
        # Chunked L2 search to keep memory bounded.
        chunk = 512
        for start in range(0, len(missed), chunk):
            sub_idx = missed[start : start + chunk]
            sub_pts = miss_pts[start : start + chunk]
            diff = sub_pts[:, None, :] - cell_midpoints_2d[None, :, :]
            dist2 = np.sum(diff * diff, axis=2)
            nearest = np.argmin(dist2, axis=1)
            for k, pt_i in enumerate(sub_idx):
                c_idx = int(nearest[k])
                cell_ids[pt_i] = c_idx
                pts_3d[pt_i, 0] = cell_midpoints_2d[c_idx, 0]
                pts_3d[pt_i, 1] = cell_midpoints_2d[c_idx, 1]

    return pts_3d, cell_ids


def _eval_fem(func: fem.Function, pts_3d: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
    """Evaluate a dolfinx ``Function`` at ``pts_3d`` with pre-located ``cell_ids``."""
    vals = func.eval(pts_3d, cell_ids)
    vals = np.asarray(vals, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals[:, None]
    return vals


# ---------------------------------------------------------------------------
# Boundary conditions, continuation, and nonlinear solve
# ---------------------------------------------------------------------------

def _tube_profile(r: np.ndarray) -> np.ndarray:
    """Developed Hagen-Poiseuille profile for KG axisymmetric tube (r in [0, 0.5], mean 1)."""
    return 2.0 * (1.0 - (r / 0.5) ** 2)


def _build_bcs(
    W: fem.FunctionSpace,
    tags: mesh.MeshTags,
    phi_in: float,
    axisym: bool,
    variant: str,
) -> list[fem.DirichletBC]:
    """Build Dirichlet BCs for V3 ``EE-DL`` (developed Poiseuille + uniform ``phi_in``).

    TODO (V4 ``EE-full``): replace uniform ``phi_in`` + Poiseuille with a 1D/G1 developed
    precursor profile when V4 closures are added to ``cfd/fenicsx/twofluid.py``.
    """
    if variant == "EE-full":
        raise NotImplementedError(
            "V4 EE-full closures and developed precursor inlet are not yet implemented (TODO)."
        )
    if variant != "EE-DL":
        raise ValueError(f"Unsupported variant '{variant}'; expected 'EE-DL' or 'EE-full'")

    inlet_prof = _tube_profile if axisym else tf.poiseuille
    bcs = tf.boundary_conditions(W, tags, phi_in, inlet_profile=inlet_prof)
    if axisym:
        for sub in (0, 1):
            Vr, _ = W.sub(sub).sub(1).collapse()
            zero = fem.Function(Vr)
            dofs = fem.locate_dofs_topological((W.sub(sub).sub(1), Vr), 1, tags.find(mfs.AXIS))
            bcs.append(fem.dirichletbc(zero, dofs, W.sub(sub).sub(1)))
    return bcs


def _build_continuation_schedule(
    re_target: float,
    phi_target: float,
    *,
    re_start: float = 1.0,
    phi_start: float = 0.01,
    max_re_steps: int = 4,
    max_phi_steps: int = 4,
) -> list[tuple[float, float]]:
    """Build ``(Re_p, phi_in)`` continuation ladder: Re 1 -> target, then phi 0.01 -> target (<= 4 steps each)."""
    re0 = min(re_start, re_target)
    phi0 = min(phi_start, phi_target)

    # Phase 1: Re continuation at phi = phi0.
    if re_target <= re0 * (1.0 + 1e-12):
        re_vals = [re_target]
    else:
        ratio = re_target / max(re0, 1e-6)
        n_re = min(max_re_steps, max(1, int(math.ceil(math.log(ratio) / math.log(4.0)))))
        re_vals = [float(v) for v in np.geomspace(re0, re_target, n_re + 1)]
        re_vals[-1] = re_target

    schedule: list[tuple[float, float]] = [(re_v, phi0) for re_v in re_vals]

    # Phase 2: phi continuation at Re = re_target.
    if phi_target > phi0 + 1e-12:
        delta_phi = phi_target - phi0
        n_phi = min(max_phi_steps, max(1, int(math.ceil(delta_phi / 0.12))))
        phi_vals = [float(v) for v in np.linspace(phi0, phi_target, n_phi + 1)][1:]
        phi_vals[-1] = phi_target
        for pv in phi_vals:
            schedule.append((re_target, pv))

    return schedule


def _initialize_solution(
    W: fem.FunctionSpace,
    U: fem.Function,
    spec: MeshSpec,
    re_init: float,
    phi_init: float,
    axisym: bool,
    viscosity: str,
) -> None:
    """Populate a clean starting guess for the first continuation step."""
    U.x.array[:] = 0.0
    if spec.family.upper() == "G1":
        mu = float(tf.mixture_viscosity_hat(phi_init)) if viscosity == "wu-high-shear" else 1.0
        L = float(spec.entry_length)
        for sub in (0, 1):
            U.sub(sub).interpolate(lambda x: np.vstack([tf.poiseuille(x[1]), np.zeros_like(x[0])]))
        U.sub(2).interpolate(lambda x: 12.0 * mu * (L - x[0]))
    elif axisym:
        for sub in (0, 1):
            U.sub(sub).interpolate(
                lambda x: np.vstack(
                    [
                        np.where(x[1] < 0.5, _tube_profile(np.minimum(x[1], 0.5)), 0.0),
                        np.zeros_like(x[0]),
                    ]
                )
            )
    else:
        # Smooth inlet-channel Poiseuille guess in entry section, zero elsewhere.
        for sub in (0, 1):
            U.sub(sub).interpolate(
                lambda x: np.vstack(
                    [
                        np.where(
                            (x[0] <= 0.0) & (x[1] >= 0.0) & (x[1] <= 1.0),
                            tf.poiseuille(np.clip(x[1], 0.0, 1.0)),
                            0.0,
                        ),
                        np.zeros_like(x[0]),
                    ]
                )
            )
    U.sub(3).interpolate(lambda x: np.full_like(x[0], phi_init))
    U.x.scatter_forward()


def solve_with_continuation(
    W: fem.FunctionSpace,
    tags: mesh.MeshTags,
    spec: MeshSpec,
    *,
    re_p: float,
    d_hat: float,
    phi_in: float,
    variant: str = "EE-DL",
    viscosity: str = "wu-high-shear",
    lift: bool = True,
    supg: bool = False,
    phi_diff: float = 1.0e-4,
    axisym: bool = False,
    rtol: float = 1e-9,
    max_it: int = 30,
    max_halvings: int = 3,
) -> dict[str, Any]:
    """Solve steady two-fluid equations with Re/phi continuation and adaptive step halving."""
    U = fem.Function(W)
    schedule = _build_continuation_schedule(re_p, phi_in)
    re0, phi0 = schedule[0]

    _initialize_solution(W, U, spec, re0, phi0, axisym=axisym, viscosity=viscosity)

    # Queue of (re_next, phi_next, halving_depth)
    queue: list[tuple[float, float, int]] = [(r, p, 0) for (r, p) in schedule]
    prev_re, prev_phi = 0.0, phi0
    has_converged_state = False

    total_newton_its = 0
    total_halvings = 0
    step_history: list[dict[str, Any]] = []

    while queue:
        re_tgt, phi_tgt, depth = queue.pop(0)
        U_backup = U.x.array.copy()
        t_step = time.monotonic()

        try:
            bcs = _build_bcs(W, tags, phi_tgt, axisym=axisym, variant=variant)
            if not has_converged_state:
                for bc in bcs:
                    bc.set(U.x.array)
                U.x.scatter_forward()

            prm = tf.TwoFluidParams(
                re_p=re_tgt,
                d_hat=d_hat,
                phi_in=phi_tgt,
                variant=variant,
                lift=lift,
                supg=supg,
                phi_diff=phi_diff,
                axisym=axisym,
                viscosity=viscosity,
            )
            F = tf.residual(W, U, prm, None)
            n_it, ok = tf.solve(W, U, F, bcs, rtol=rtol, max_it=max_it)
            if not ok or not np.all(np.isfinite(U.x.array)):
                raise RuntimeError(f"Newton returned converged={ok} / non-finite values")
        except Exception as exc:  # noqa: BLE001
            # Restore last valid state and attempt step halving.
            U.x.array[:] = U_backup
            U.x.scatter_forward()
            if depth >= max_halvings:
                return {
                    "converged": False,
                    "reason": (
                        f"newton_failed_after_{max_halvings}_halvings at "
                        f"(Re={re_tgt:.4g}, phi={phi_tgt:.4g}): {type(exc).__name__}: {exc}"
                    ),
                    "U": U,
                    "newton_iterations": total_newton_its,
                    "continuation_steps": step_history,
                    "n_halvings": total_halvings,
                }

            total_halvings += 1
            if not has_converged_state:
                # First step failed: try a lower starting Re (down to 0.1 * re_tgt).
                re_mid = 0.5 * re_tgt
                queue.insert(0, (re_tgt, phi_tgt, depth + 1))
                queue.insert(0, (re_mid, phi_tgt, depth + 1))
            else:
                re_mid = 0.5 * (prev_re + re_tgt)
                phi_mid = 0.5 * (prev_phi + phi_tgt)
                queue.insert(0, (re_tgt, phi_tgt, depth + 1))
                queue.insert(0, (re_mid, phi_mid, depth + 1))
            continue

        step_wall = time.monotonic() - t_step
        total_newton_its += int(n_it)
        has_converged_state = True
        prev_re, prev_phi = re_tgt, phi_tgt
        step_history.append(
            {
                "re_p": round(float(re_tgt), 6),
                "phi_in": round(float(phi_tgt), 6),
                "newton_its": int(n_it),
                "depth": int(depth),
                "wall_seconds": round(float(step_wall), 4),
            }
        )

    return {
        "converged": True,
        "reason": None,
        "U": U,
        "newton_iterations": total_newton_its,
        "continuation_steps": step_history,
        "n_halvings": total_halvings,
    }


# ---------------------------------------------------------------------------
# Post-processing: scalars, CFL (Sezer Eq. 21), fixed grid, .npz writer
# ---------------------------------------------------------------------------

def _outlet_section_specs(spec: MeshSpec) -> list[tuple[str, tuple[float, float], float]]:
    """Return ``(sample_line_name, (nx, ny), h_local)`` for each outlet section."""
    fam = spec.family.upper()
    if fam in ("G1", "G2", "G3", "KG"):
        return [("outletProfile", (1.0, 0.0), float(spec.outlet_heights[0]))]
    if fam == "G6":
        n_bends = int(spec.params["n_bends"])
        normal = (-1.0, 0.0) if (n_bends % 2 == 1) else (1.0, 0.0)
        return [("outletProfile", normal, float(spec.outlet_heights[0]))]
    if fam in ("G4", "G5", "G7"):
        theta = math.radians(float(spec.params["theta_deg"]))
        ct, st = math.cos(theta), math.sin(theta)
        return [
            ("daughter1Mid", (ct, st), float(spec.outlet_heights[0])),
            ("daughter2Mid", (ct, -st), float(spec.outlet_heights[1])),
        ]
    raise ValueError(f"No outlet section specification for family '{fam}'")


def _sample_line_points(sl: SampleLine, n_pts: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(coord, pts_2d)`` along ``sl`` with ``n_pts`` uniformly spaced points."""
    sx, sy, _ = sl.start
    ex, ey, _ = sl.end
    t = np.linspace(0.0, 1.0, n_pts, dtype=np.float64)
    px = sx + t * (ex - sx)
    py = sy + t * (ey - sy)
    if sl.axis == "x":
        coord = px
    elif sl.axis == "y":
        coord = py
    else:
        coord = t * math.hypot(ex - sx, ey - sy)
    return coord, np.column_stack([px, py])


def _safe_sym_grad(v: Any, axisym: bool) -> Any:
    """Symmetric velocity gradient safe at ``r = 0`` for axisymmetric DG1 interpolation."""
    if not axisym:
        return ufl.sym(ufl.grad(v))
    r = ufl.SpatialCoordinate(ufl.domain.extract_unique_domain(v))[1]
    hoop = ufl.conditional(ufl.gt(r, 1e-10), v[1] / r, v[1].dx(1))
    g3 = ufl.as_tensor(
        [
            [v[0].dx(0), v[0].dx(1), 0.0],
            [v[1].dx(0), v[1].dx(1), 0.0],
            [0.0, 0.0, hoop],
        ]
    )
    return ufl.sym(g3)


def compute_cfl_sezer_eq21(
    msh: mesh.Mesh,
    tree: geometry.BoundingBoxTree,
    cell_midpoints_2d: np.ndarray,
    phi_fn: fem.Function,
    sl: SampleLine,
    h_local: float,
    *,
    axisym: bool = False,
    n_samples: int = 400,
    wall_margin_frac: float = 0.02,
) -> dict[str, float | None]:
    """Compute CFL thickness using Sezer et al. (2026) Eq. 21 (knowledge/15, bench_zhao.py).

    Sezer Eq. 21 defines the cell-free layer thickness as the distance from the solid wall
    to the first inflection point (sign change of ``d^2 phi / ds^2``) moving inward from the
    wall, excluding the outer 2 % (``|y/h| > 0.48``) to avoid boundary-node artifacts.

    Returns both star-unit ``cfl_thickness_star`` (``delta_CFL / h1``) and locally normalized
    ``cfl_over_h_local`` (``delta_CFL / h_local``). For uniform ``phi`` (variation < 1e-8),
    returns ``0.0``.
    """
    sx, sy, _ = sl.start
    ex, ey, _ = sl.end
    dx, dy = ex - sx, ey - sy
    seg_len = math.hypot(dx, dy)
    if seg_len <= 1e-12:
        return {"cfl_thickness_star": None, "cfl_over_h_local": None, "cfl_bot_star": None, "cfl_top_star": None}

    ux, uy = dx / seg_len, dy / seg_len
    # Reconstruct full wall-to-wall segment centered on sl (since _line_pts pulls endpoints inward by eps).
    mx, my = 0.5 * (sx + ex), 0.5 * (sy + ey)

    if axisym:
        # Axisymmetric KG: r in [0, R_wall = 0.5 * h_local], exclude r/D > 0.48 -> r <= 0.96 * R_wall.
        r_wall = 0.5 * h_local
        r_max = (1.0 - 2.0 * wall_margin_frac) * r_wall
        rs = np.linspace(0.0, r_max, n_samples, dtype=np.float64)
        x_const = 0.5 * (sx + ex)
        pts_2d = np.column_stack([np.full_like(rs, x_const), rs])
        pts_3d, cids = _locate_points_cells(msh, tree, pts_2d, cell_midpoints_2d)
        vals = _eval_fem(phi_fn, pts_3d, cids)[:, 0]
        if float(np.ptp(vals)) / max(float(np.mean(np.abs(vals))), 1e-12) < 1e-3:
            return {"cfl_thickness_star": 0.0, "cfl_over_h_local": 0.0, "cfl_bot_star": None, "cfl_top_star": 0.0}
        d2 = np.gradient(np.gradient(vals, rs), rs)
        d2_scale = float(np.max(np.abs(d2)))
        if d2_scale < 1e-10:
            return {"cfl_thickness_star": 0.0, "cfl_over_h_local": 0.0, "cfl_bot_star": None, "cfl_top_star": 0.0}
        sign = np.where(np.abs(d2) > 1e-6 * d2_scale, np.sign(d2), 0.0)
        cfl_star: float | None = None
        last_nz = 0.0
        for k in range(len(rs) - 1, 0, -1):
            if sign[k] != 0.0:
                if last_nz != 0.0 and sign[k] != last_nz:
                    cfl_star = float(max(0.0, r_wall - rs[k]))
                    break
                last_nz = sign[k]
        return {
            "cfl_thickness_star": cfl_star,
            "cfl_over_h_local": (cfl_star / h_local) if cfl_star is not None else None,
            "cfl_bot_star": None,
            "cfl_top_star": cfl_star,
        }

    # Planar channel: s in [-0.48 * h_local, +0.48 * h_local] around channel centerline.
    s_lim = (0.5 - wall_margin_frac) * h_local
    s_half = np.linspace(0.0, s_lim, n_samples, dtype=np.float64)

    def _walk_half(sign_dir: float) -> float | None:
        # sign_dir = +1: centerline -> upper wall; -1: centerline -> lower wall.
        coords = sign_dir * s_half
        pts_2d = np.column_stack([mx + coords * ux, my + coords * uy])
        pts_3d, cids = _locate_points_cells(msh, tree, pts_2d, cell_midpoints_2d)
        vals = _eval_fem(phi_fn, pts_3d, cids)[:, 0]
        if float(np.ptp(vals)) / max(float(np.mean(np.abs(vals))), 1e-12) < 1e-3:
            return 0.0
        d2 = np.gradient(np.gradient(vals, s_half), s_half)
        d2_scale = float(np.max(np.abs(d2)))
        if d2_scale < 1e-10:
            return 0.0
        sgn = np.where(np.abs(d2) > 1e-6 * d2_scale, np.sign(d2), 0.0)
        last_nz = 0.0
        for k in range(len(s_half) - 1, 0, -1):
            if sgn[k] != 0.0:
                if last_nz != 0.0 and sgn[k] != last_nz:
                    return float(max(0.0, 0.5 * h_local - s_half[k]))
                last_nz = sgn[k]
        return None

    cfl_bot = _walk_half(-1.0)
    cfl_top = _walk_half(+1.0)
    valid = [c for c in (cfl_bot, cfl_top) if c is not None]
    cfl_mean = float(np.mean(valid)) if valid else None
    return {
        "cfl_thickness_star": cfl_mean,
        "cfl_over_h_local": (cfl_mean / h_local) if cfl_mean is not None else None,
        "cfl_bot_star": cfl_bot,
        "cfl_top_star": cfl_top,
    }


def _extract_and_write_npz(
    *,
    msh: mesh.Mesh,
    tags: mesh.MeshTags,
    spec: MeshSpec,
    U: fem.Function,
    run_id: str,
    record: Mapping[str, Any],
    re_p: float,
    d_hat: float,
    phi_in: float,
    variant: str,
    viscosity: str,
    lift: bool,
    supg: bool,
    phi_diff: float,
    axisym: bool,
    solve_info: Mapping[str, Any],
    wall_seconds: float,
    out_npz_path: Path,
    tau_lo: float = DEFAULT_TAU_STAR_LO,
    tau_hi: float = DEFAULT_TAU_STAR_HI,
) -> dict[str, Any]:
    """Extract cell fields, fixed grid, outlet profiles, scalars, and write ``<run_id>.npz``."""
    family = spec.family.upper()
    n_cells = int(msh.topology.index_map(2).size_local)
    W = U.function_space
    n_dofs = int(W.dofmap.index_map.size_global * W.dofmap.index_map_bs)

    u_fn, w_fn, p_fn, phi_fn = (U.sub(i).collapse() for i in range(4))

    # Cell midpoints and exact cell areas (volumes in 2D z_thickness = 1).
    cell_indices = np.arange(n_cells, dtype=np.int32)
    mid_3d = mesh.compute_midpoints(msh, 2, cell_indices)
    x_star = mid_3d[:, 0].astype(np.float64)
    y_star = mid_3d[:, 1].astype(np.float64)
    cell_mid_2d = mid_3d[:, :2].astype(np.float64)

    DG0 = fem.functionspace(msh, ("DG", 0))
    DG0_vec = fem.functionspace(msh, ("DG", 0, (2,)))
    DG1 = fem.functionspace(msh, ("DG", 1))

    # Exact cell areas via DG0 assembly of integral(1 dx).
    area_vec = fem.assemble_scalar  # unused alias check avoided
    _ = area_vec
    v0 = ufl.TestFunction(DG0)
    cell_volumes = np.asarray(
        fem.petsc.assemble_vector(fem.form(v0 * ufl.dx)).array, dtype=np.float64
    ).copy()

    # Shear rate and mixture shear stress as DG1 functions (exact for P2 velocity gradients).
    d_u = _safe_sym_grad(u_fn, axisym)
    d_w = _safe_sym_grad(w_fn, axisym)
    gdot_u_expr = ufl.sqrt(2.0 * ufl.inner(d_u, d_u) + 1e-24)
    gdot_w_expr = ufl.sqrt(2.0 * ufl.inner(d_w, d_w) + 1e-24)
    phi_reg = ufl.min_value(ufl.max_value(phi_fn, 0.0), 1.0)
    mu_expr = tf.mixture_viscosity_hat(phi_reg) if viscosity == "wu-high-shear" else 1.0
    tau_expr = mu_expr * ((1.0 - phi_reg) * gdot_u_expr + phi_reg * gdot_w_expr)

    ip_dg1 = DG1.element.interpolation_points()
    gdot_p_dg1 = fem.Function(DG1)
    gdot_p_dg1.interpolate(fem.Expression(gdot_u_expr, ip_dg1))
    gdot_r_dg1 = fem.Function(DG1)
    gdot_r_dg1.interpolate(fem.Expression(gdot_w_expr, ip_dg1))
    tau_dg1 = fem.Function(DG1)
    tau_dg1.interpolate(fem.Expression(tau_expr, ip_dg1))

    # Cell-center field arrays (exact cell-count pairing with OpenFOAM).
    ip_dg0 = DG0.element.interpolation_points()
    ip_dg0_v = DG0_vec.element.interpolation_points()

    def _interp_dg0_scalar(expr_or_fn: Any) -> np.ndarray:
        f = fem.Function(DG0)
        f.interpolate(fem.Expression(expr_or_fn, ip_dg0))
        return np.asarray(f.x.array, dtype=np.float64).copy()

    def _interp_dg0_vec3(fn_2d: fem.Function) -> np.ndarray:
        f = fem.Function(DG0_vec)
        f.interpolate(fem.Expression(fn_2d, ip_dg0_v))
        arr2 = np.asarray(f.x.array, dtype=np.float64).reshape((n_cells, 2))
        out = np.zeros((n_cells, 3), dtype=np.float64)
        out[:, :2] = arr2
        return out

    u_plasma_star = _interp_dg0_vec3(u_fn)
    u_rbc_star = _interp_dg0_vec3(w_fn)
    phi_cell = _interp_dg0_scalar(phi_fn)
    p_star = _interp_dg0_scalar(p_fn)
    gdot_plasma_star = _interp_dg0_scalar(gdot_u_expr)
    gdot_rbc_star = _interp_dg0_scalar(gdot_w_expr)
    tau_star = _interp_dg0_scalar(tau_expr)
    gdot_star = gdot_plasma_star
    u_star = (1.0 - phi_cell[:, None]) * u_plasma_star + phi_cell[:, None] * u_rbc_star

    # 1. Pressure drop dp* = area-mean p on INLET minus area-mean p on OUTLET (exact boundary integral).
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    jac = ufl.SpatialCoordinate(msh)[1] if axisym else 1.0
    inlet_area = float(fem.assemble_scalar(fem.form(1.0 * jac * ds(mfs.INLET))))
    outlet_area = float(fem.assemble_scalar(fem.form(1.0 * jac * ds(mfs.OUTLET))))
    p_inlet = float(fem.assemble_scalar(fem.form(p_fn * jac * ds(mfs.INLET)))) / max(inlet_area, 1e-15)
    p_outlet = float(fem.assemble_scalar(fem.form(p_fn * jac * ds(mfs.OUTLET)))) / max(outlet_area, 1e-15)
    dp_star = float(p_inlet - p_outlet)

    # Bounding-box tree for exact FEM evaluation on resampling grid & sample lines.
    tree = geometry.bb_tree(msh, 2)

    # 2. Fixed per-family resampling grid + exact fluid mask.
    nx_g, ny_g = FAMILY_GRID_SHAPES[family]
    xs_v = [pt[0] for pt in spec.vertices_2d]
    ys_v = [pt[1] for pt in spec.vertices_2d]
    x_min, x_max = min(xs_v), max(xs_v)
    y_min, y_max = min(ys_v), max(ys_v)
    if family == "G6":
        Rc = float(spec.params["Rc_ratio"])
        r_out = Rc + 0.5
        L_straight = float(spec.params.get("L_straight", 2.0))
        x_max = max(x_max, r_out)
        if int(spec.params["n_bends"]) >= 2:
            x_min = min(x_min, -L_straight - r_out)

    grid_x = np.linspace(x_min, x_max, nx_g, dtype=np.float64)
    grid_y = np.linspace(y_min, y_max, ny_g, dtype=np.float64)
    grid_mask = build_fluid_mask(spec, grid_x, grid_y)

    xx, yy = np.meshgrid(grid_x, grid_y)
    pts_dst_2d = np.column_stack([xx[grid_mask], yy[grid_mask]])

    grid_data: dict[str, np.ndarray] = {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_mask": grid_mask,
    }
    grid_field_names = (
        "ux", "uy", "u_plasma_x", "u_plasma_y", "urbc_x", "urbc_y", "phi", "p", "gdot", "tau"
    )
    if pts_dst_2d.shape[0] > 0:
        g_pts_3d, g_cids = _locate_points_cells(msh, tree, pts_dst_2d, cell_mid_2d)
        u_g = _eval_fem(u_fn, g_pts_3d, g_cids)
        w_g = _eval_fem(w_fn, g_pts_3d, g_cids)
        phi_g = _eval_fem(phi_fn, g_pts_3d, g_cids)[:, 0]
        p_g = _eval_fem(p_fn, g_pts_3d, g_cids)[:, 0]
        gdot_g = _eval_fem(gdot_p_dg1, g_pts_3d, g_cids)[:, 0]
        tau_g = _eval_fem(tau_dg1, g_pts_3d, g_cids)[:, 0]
        umix_g = (1.0 - phi_g[:, None]) * u_g + phi_g[:, None] * w_g

        g_vals = {
            "ux": umix_g[:, 0],
            "uy": umix_g[:, 1],
            "u_plasma_x": u_g[:, 0],
            "u_plasma_y": u_g[:, 1],
            "urbc_x": w_g[:, 0],
            "urbc_y": w_g[:, 1],
            "phi": phi_g,
            "p": p_g,
            "gdot": gdot_g,
            "tau": tau_g,
        }
        for name in grid_field_names:
            arr = np.zeros((ny_g, nx_g), dtype=np.float32)
            arr[grid_mask] = g_vals[name].astype(np.float32)
            grid_data[f"grid_{name}"] = arr
    else:
        for name in grid_field_names:
            grid_data[f"grid_{name}"] = np.zeros((ny_g, nx_g), dtype=np.float32)

    # 3. Reattachment length x_r / h1 (or x_r / d1 for KG) for expansion families.
    sl_by_name = {sl.name: sl for sl in spec.sample_lines}
    xr_over_h1: float | None = None
    reattach_note: str | None = None
    if family in EXPANSION_REATTACHMENT_SPECS:
        line_name, step_key = EXPANSION_REATTACHMENT_SPECS[family]
        x_step = float(spec.params[step_key]) if step_key is not None else 0.0
        sl_nw = sl_by_name[line_name]
        coord_nw, pts_nw = _sample_line_points(sl_nw, max(400, sl_nw.n_points))
        nw_3d, nw_cids = _locate_points_cells(msh, tree, pts_nw, cell_mid_2d)
        u_nw = _eval_fem(u_fn, nw_3d, nw_cids)
        try:
            xr_over_h1 = float(reattachment_length(coord_nw, u_nw[:, 0], x_step=x_step))
        except ValueError as exc:
            xr_over_h1 = None
            reattach_note = str(exc)

    # 4. Max wall shear stress tau* evaluated on WALL facet midpoints (DG1 is exact for P2 velocity).
    wall_facets = tags.find(mfs.WALL)
    if len(wall_facets) > 0:
        msh.topology.create_connectivity(1, 2)
        f2c = msh.topology.connectivity(1, 2)
        w_mids = mesh.compute_midpoints(msh, 1, wall_facets)
        w_cells = np.array([int(f2c.links(int(f))[0]) for f in wall_facets], dtype=np.int32)
        w_tau = _eval_fem(tau_dg1, w_mids, w_cells)[:, 0]
        max_wall_tau_star = float(np.max(w_tau))
        max_wall_tau_source = "wall_facets_dg1"
    else:
        max_wall_tau_star = float(np.max(tau_star))
        max_wall_tau_source = "internal_fallback"

    # 5. Outlet-section profiles, RBC flux-weighted window fraction, and Sezer Eq. 21 CFL thickness.
    out_s_list: list[np.ndarray] = []
    out_un_list: list[np.ndarray] = []
    out_up_n_list: list[np.ndarray] = []
    out_phi_list: list[np.ndarray] = []
    out_gdot_list: list[np.ndarray] = []
    out_tau_list: list[np.ndarray] = []
    out_w_list: list[np.ndarray] = []
    out_branch_list: list[np.ndarray] = []
    cfl_branch_vals: list[dict[str, float | None]] = []

    for branch_idx, (line_name, (nx_dir, ny_dir), h_local) in enumerate(_outlet_section_specs(spec)):
        sl = sl_by_name[line_name]
        coord, pts_2d = _sample_line_points(sl, max(100, sl.n_points))
        order = np.argsort(coord)
        coord = coord[order]
        pts_2d = pts_2d[order]

        l_3d, l_cids = _locate_points_cells(msh, tree, pts_2d, cell_mid_2d)
        u_s = _eval_fem(u_fn, l_3d, l_cids)
        w_s = _eval_fem(w_fn, l_3d, l_cids)
        phi_s = np.clip(_eval_fem(phi_fn, l_3d, l_cids)[:, 0], 0.0, 1.0)
        gdot_s = np.maximum(0.0, _eval_fem(gdot_p_dg1, l_3d, l_cids)[:, 0])
        tau_s = np.maximum(0.0, _eval_fem(tau_dg1, l_3d, l_cids)[:, 0])

        u_rbc_n = w_s[:, 0] * nx_dir + w_s[:, 1] * ny_dir
        u_pla_n = u_s[:, 0] * nx_dir + u_s[:, 1] * ny_dir

        wts = trapezoid_weights(coord)
        if axisym:
            wts = 2.0 * math.pi * np.maximum(0.0, pts_2d[:, 1]) * wts

        out_s_list.append(coord)
        out_un_list.append(u_rbc_n)
        out_up_n_list.append(u_pla_n)
        out_phi_list.append(phi_s)
        out_gdot_list.append(gdot_s)
        out_tau_list.append(tau_s)
        out_w_list.append(wts)
        out_branch_list.append(np.full(coord.shape, branch_idx, dtype=np.int32))

        cfl_branch_vals.append(
            compute_cfl_sezer_eq21(
                msh, tree, cell_mid_2d, phi_fn, sl, h_local, axisym=axisym
            )
        )

    outlet_s = np.concatenate(out_s_list)
    outlet_u_n = np.concatenate(out_un_list)
    outlet_u_plasma_n = np.concatenate(out_up_n_list)
    outlet_phi = np.concatenate(out_phi_list)
    outlet_gdot = np.concatenate(out_gdot_list)
    outlet_tau = np.concatenate(out_tau_list)
    outlet_weights = np.concatenate(out_w_list)
    outlet_branch_id = np.concatenate(out_branch_list)

    try:
        f_window = float(
            window_fraction(outlet_tau, outlet_u_n, outlet_weights, phi=outlet_phi, lo=tau_lo, hi=tau_hi)
        )
    except ValueError:
        f_window = 0.0

    w_sum = float(np.sum(outlet_weights))
    outlet_phi_mean = (
        float(np.sum(outlet_weights * outlet_phi) / w_sum)
        if w_sum > 0.0
        else float(np.mean(outlet_phi))
    )

    valid_cfl = [c["cfl_thickness_star"] for c in cfl_branch_vals if c["cfl_thickness_star"] is not None]
    valid_cfl_norm = [c["cfl_over_h_local"] for c in cfl_branch_vals if c["cfl_over_h_local"] is not None]
    cfl_thickness_star = float(np.mean(valid_cfl)) if valid_cfl else None
    cfl_over_h_local = float(np.mean(valid_cfl_norm)) if valid_cfl_norm else None

    phi_dof_min = float(np.min(phi_fn.x.array))
    phi_dof_max = float(np.max(phi_fn.x.array))
    max_slip = float(np.max(np.abs(w_fn.x.array - u_fn.x.array)))
    ram_mb = round(peak_ram_mb(), 2)

    scalars: dict[str, float | None] = {
        "dp_star": dp_star,
        "xr_over_h1": xr_over_h1,
        "max_wall_tau_star": max_wall_tau_star,
        "f_window": f_window,
        "cfl_thickness_star": cfl_thickness_star,
        "cfl_over_h_local": cfl_over_h_local,
        "outlet_phi_mean": outlet_phi_mean,
        "min_phi": phi_dof_min,
        "max_phi": phi_dof_max,
        "max_slip": max_slip,
    }

    code_ver = fenicsx_code_version()
    params_payload: dict[str, Any] = {
        "run_id": run_id,
        "case_id": record.get("case_id"),
        "tier": record.get("tier", "paired"),
        "family": family,
        "level": int(spec.level),
        "solver": "fenicsx",
        "dolfinx_version": str(dolfinx.__version__),
        "variant": variant,
        "viscosity": viscosity,
        "Re": float(re_p),
        "re_p": float(re_p),
        "d_hat": float(d_hat),
        "d_rbc_over_h": float(d_hat),
        "phi": float(phi_in),
        "phi_in": float(phi_in),
        "lift": bool(lift),
        "supg": bool(supg),
        "phi_diff": float(phi_diff),
        "axisym": bool(axisym),
        "geometry": dict(spec.params),
        "n_cells": n_cells,
        "n_dofs": n_dofs,
        "status": "ok",
        "reason": None,
        "converged": True,
        "newton_iterations": int(solve_info["newton_iterations"]),
        "n_continuation_steps": len(solve_info["continuation_steps"]),
        "n_halvings": int(solve_info["n_halvings"]),
        "continuation_steps": solve_info["continuation_steps"],
        "wall_seconds": round(float(wall_seconds), 4),
        "peak_ram_mb": ram_mb,
        "pressure_scaling": "viscous_plasma (p* = p h1 / (mu_p U))",
        "stress_definition": "mixture_star: mu_hat(phi) * ((1 - phi)*gdot_plasma* + phi*gdot_rbc*)",
        "window_weighting": "rbc_flux (phi * w_n * dA)",
        "cfl_definition": {
            "method": "sezer2026_eq21_inflection",
            "wall_margin_frac": 0.02,
            "branches": cfl_branch_vals,
        },
        "max_wall_tau_source": max_wall_tau_source,
        "tau_window_star": [float(tau_lo), float(tau_hi)],
        "code_version": code_ver,
        "scalars": scalars,
    }

    # Propagate paired-design provenance metadata when present.
    for key in (
        "design_id",
        "pair_id",
        "h_um",
        "U_mps",
        "mu_carrier",
        "mu_eff",
        "rho",
        "d_rbc_um",
        "tau_ref",
        "t_ref",
        "gdot_ref",
        "seed",
    ):
        if key in record:
            params_payload[key] = record[key]
    if reattach_note is not None:
        params_payload["reattachment_note"] = reattach_note

    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz_path,
        x_star=x_star,
        y_star=y_star,
        cell_volumes=cell_volumes,
        U_star=u_star,
        U_plasma_star=u_plasma_star,
        U_rbc_star=u_rbc_star,
        phi=phi_cell,
        phi_star=phi_cell,
        p_star=p_star,
        gdot_star=gdot_star,
        gdot_plasma_star=gdot_plasma_star,
        gdot_rbc_star=gdot_rbc_star,
        tau_star=tau_star,
        outlet_s=outlet_s,
        outlet_u_n=outlet_u_n,
        outlet_u_plasma_n=outlet_u_plasma_n,
        outlet_phi=outlet_phi,
        outlet_gdot=outlet_gdot,
        outlet_tau=outlet_tau,
        outlet_weights=outlet_weights,
        outlet_branch_id=outlet_branch_id,
        params=json.dumps(params_payload),
        **grid_data,
    )

    sha256_hex = compute_file_sha256(out_npz_path)
    return {
        "run_id": run_id,
        "case_id": record.get("case_id"),
        "family": family,
        "level": int(spec.level),
        "status": "ok",
        "reason": None,
        "newton_iterations": int(solve_info["newton_iterations"]),
        "n_continuation_steps": len(solve_info["continuation_steps"]),
        "n_halvings": int(solve_info["n_halvings"]),
        "wall_seconds": round(float(wall_seconds), 4),
        "peak_ram_mb": ram_mb,
        "params": params_payload,
        "scalars": scalars,
        "sha256": sha256_hex,
        "npz_path": str(out_npz_path),
        "code_version": code_ver,
    }


def run_case(
    *,
    run_id: str,
    family: str,
    geometry_params: Mapping[str, Any],
    level: int,
    re_p: float,
    d_hat: float,
    phi_in: float,
    out_npz_path: Path,
    variant: str = "EE-DL",
    viscosity: str = "wu-high-shear",
    lift: bool = True,
    supg: bool = False,
    phi_diff: float = 1.0e-4,
    tau_lo: float = DEFAULT_TAU_STAR_LO,
    tau_hi: float = DEFAULT_TAU_STAR_HI,
    rtol: float = 1e-9,
    max_it: int = 30,
    max_halvings: int = 3,
    record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one FEniCSx Euler–Euler case end-to-end and write ``<run_id>.npz`` on success."""
    t0 = time.monotonic()
    fam = family.upper()
    axisym = fam == "KG"
    rec = dict(record) if record is not None else {}
    code_ver = fenicsx_code_version()

    try:
        if variant == "EE-full":
            raise NotImplementedError(
                "V4 EE-full closures and developed precursor inlet are not yet implemented (TODO)."
            )

        msh, tags, spec = mfs.build_dolfinx_mesh(fam, geometry_params, level=level, comm=MPI.COMM_WORLD)
        W = tf.build_space(msh)
        solve_info = solve_with_continuation(
            W,
            tags,
            spec,
            re_p=float(re_p),
            d_hat=float(d_hat),
            phi_in=float(phi_in),
            variant=variant,
            viscosity=viscosity,
            lift=lift,
            supg=supg,
            phi_diff=phi_diff,
            axisym=axisym,
            rtol=rtol,
            max_it=max_it,
            max_halvings=max_halvings,
        )
        wall_s = time.monotonic() - t0
        if not solve_info["converged"]:
            ram_mb = round(peak_ram_mb(), 2)
            return {
                "run_id": run_id,
                "case_id": rec.get("case_id"),
                "family": fam,
                "level": int(level),
                "status": "failed",
                "reason": solve_info["reason"],
                "newton_iterations": int(solve_info["newton_iterations"]),
                "n_continuation_steps": len(solve_info["continuation_steps"]),
                "n_halvings": int(solve_info["n_halvings"]),
                "wall_seconds": round(wall_s, 4),
                "peak_ram_mb": ram_mb,
                "params": {
                    **rec,
                    "run_id": run_id,
                    "family": fam,
                    "level": int(level),
                    "solver": "fenicsx",
                    "variant": variant,
                    "viscosity": viscosity,
                    "re_p": float(re_p),
                    "d_hat": float(d_hat),
                    "phi_in": float(phi_in),
                    "geometry": dict(spec.params),
                    "status": "failed",
                    "reason": solve_info["reason"],
                    "newton_iterations": int(solve_info["newton_iterations"]),
                    "continuation_steps": solve_info["continuation_steps"],
                    "wall_seconds": round(wall_s, 4),
                    "peak_ram_mb": ram_mb,
                    "code_version": code_ver,
                },
                "scalars": None,
                "sha256": None,
                "npz_path": None,
                "code_version": code_ver,
            }

        return _extract_and_write_npz(
            msh=msh,
            tags=tags,
            spec=spec,
            U=solve_info["U"],
            run_id=run_id,
            record=rec,
            re_p=float(re_p),
            d_hat=float(d_hat),
            phi_in=float(phi_in),
            variant=variant,
            viscosity=viscosity,
            lift=lift,
            supg=supg,
            phi_diff=phi_diff,
            axisym=axisym,
            solve_info=solve_info,
            wall_seconds=wall_s,
            out_npz_path=Path(out_npz_path),
            tau_lo=tau_lo,
            tau_hi=tau_hi,
        )
    except Exception as exc:  # noqa: BLE001
        wall_s = time.monotonic() - t0
        ram_mb = round(peak_ram_mb(), 2)
        reason = f"{type(exc).__name__}: {exc}"
        return {
            "run_id": run_id,
            "case_id": rec.get("case_id"),
            "family": fam,
            "level": int(level),
            "status": "failed",
            "reason": reason,
            "newton_iterations": 0,
            "n_continuation_steps": 0,
            "n_halvings": 0,
            "wall_seconds": round(wall_s, 4),
            "peak_ram_mb": ram_mb,
            "params": {
                **rec,
                "run_id": run_id,
                "family": fam,
                "level": int(level),
                "solver": "fenicsx",
                "variant": variant,
                "viscosity": viscosity,
                "re_p": float(re_p),
                "d_hat": float(d_hat),
                "phi_in": float(phi_in),
                "geometry": dict(geometry_params),
                "status": "failed",
                "reason": reason,
                "wall_seconds": round(wall_s, 4),
                "peak_ram_mb": ram_mb,
                "code_version": code_ver,
            },
            "scalars": None,
            "sha256": None,
            "npz_path": None,
            "code_version": code_ver,
        }


def run_case_from_record(
    record: Mapping[str, Any],
    *,
    out_npz_path: Path,
    level_override: int | None = None,
    tau_lo: float = DEFAULT_TAU_STAR_LO,
    tau_hi: float = DEFAULT_TAU_STAR_HI,
) -> dict[str, Any]:
    """Run a single FEniCSx case from a JSONL design record dict."""
    run_id = str(record["run_id"])
    family = str(record["family"]).upper()
    level = int(level_override if level_override is not None else record.get("level", 2))
    geom = dict(record.get("geometry", record))
    re_p = float(record.get("re_p", record.get("Re")))
    d_hat = float(record.get("d_hat", record.get("d_rbc_over_h", record.get("d_p_star", 0.08))))
    phi_in = float(record.get("phi_in", record.get("phi")))
    variant = str(record.get("variant", "EE-DL"))
    viscosity = str(record.get("viscosity", "wu-high-shear"))
    lift = bool(record.get("lift", True))
    supg = bool(record.get("supg", False))
    phi_diff = float(record.get("phi_diff", 1.0e-4))

    return run_case(
        run_id=run_id,
        family=family,
        geometry_params=geom,
        level=level,
        re_p=re_p,
        d_hat=d_hat,
        phi_in=phi_in,
        out_npz_path=out_npz_path,
        variant=variant,
        viscosity=viscosity,
        lift=lift,
        supg=supg,
        phi_diff=phi_diff,
        tau_lo=tau_lo,
        tau_hi=tau_hi,
        record=record,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single FEniCSx steady 2D two-fluid Euler–Euler case."
    )
    parser.add_argument("--record-json", type=Path, default=None, help="JSON file with one design record")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--geometry-json", type=str, default="{}")
    parser.add_argument("--level", type=int, default=None, choices=(0, 1, 2))
    parser.add_argument("--re-p", type=float, default=None)
    parser.add_argument("--d-hat", type=float, default=0.08)
    parser.add_argument("--phi-in", type=float, default=None)
    parser.add_argument("--variant", type=str, default="EE-DL")
    parser.add_argument("--viscosity", type=str, default="wu-high-shear")
    parser.add_argument("--no-lift", action="store_true")
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, default=None, help="Write structured result JSON here")
    parser.add_argument("--tau-lo", type=float, default=DEFAULT_TAU_STAR_LO)
    parser.add_argument("--tau-hi", type=float, default=DEFAULT_TAU_STAR_HI)
    args = parser.parse_args()

    if args.record_json is not None:
        rec = json.loads(args.record_json.read_text(encoding="utf-8"))
        res = run_case_from_record(
            rec,
            out_npz_path=args.out_npz,
            level_override=args.level,
            tau_lo=args.tau_lo,
            tau_hi=args.tau_hi,
        )
    else:
        if not args.run_id or not args.family or args.re_p is None or args.phi_in is None:
            parser.error("Either --record-json or (--run-id, --family, --re-p, --phi-in) is required")
        geom = json.loads(args.geometry_json)
        res = run_case(
            run_id=args.run_id,
            family=args.family,
            geometry_params=geom,
            level=args.level if args.level is not None else 2,
            re_p=args.re_p,
            d_hat=args.d_hat,
            phi_in=args.phi_in,
            out_npz_path=args.out_npz,
            variant=args.variant,
            viscosity=args.viscosity,
            lift=not args.no_lift,
            tau_lo=args.tau_lo,
            tau_hi=args.tau_hi,
        )

    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "run_id": res["run_id"],
                "family": res["family"],
                "level": res["level"],
                "status": res["status"],
                "reason": res["reason"],
                "newton_iterations": res["newton_iterations"],
                "n_continuation_steps": res["n_continuation_steps"],
                "n_halvings": res["n_halvings"],
                "wall_seconds": res["wall_seconds"],
                "peak_ram_mb": res["peak_ram_mb"],
                "scalars": res["scalars"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
