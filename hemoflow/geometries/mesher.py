"""hemoflow graded quad-mesh builders (host-safe NumPy core + lazy FEniCSx).

Ported verbatim (release-guard: 2D/2.5D only) from ``cfd/fenicsx/mesh_graded.py``.
Boundary facet tags: ``INLET = 1``, ``OUTLET = 2`` (all outlets share tag 2),
``WALL = 3``, ``AXIS = 4``.

The pure-NumPy entry point ``graded_mesh_arrays(family, params, refine=1.0)`` returns
``(nodes, cells, tagged_midpoints, min_wall_dy)`` and runs on any host without dolfinx.
``build_dolfinx_mesh`` / ``custom_mesh`` / ``from_gmsh`` import dolfinx lazily.
"""
from __future__ import annotations

import importlib.util
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from hemoflow.geometries.builders import MeshSpec, Point2D, build_geometry
from hemoflow.geometries.builders import build_g1

INLET = 1
OUTLET = 2
WALL = 3
AXIS = 4

def _has_dolfinx() -> bool:
    return importlib.util.find_spec("dolfinx") is not None



# Clean NumPy-only equivalents (verbatim logic of mesh_from_spec) for host pytest.
def _circular_arc_points(
    p0: Point2D,
    p1: Point2D,
    pm: Point2D,
    t: np.ndarray,
) -> np.ndarray:
    """Sample circular arc from ``p0`` to ``p1`` through ``pm`` (blockMesh ``arc``)."""
    ax, ay = p0
    bx, by = p1
    mx, my = pm
    d = 2.0 * (ax * (my - by) + mx * (by - ay) + bx * (ay - my))
    if abs(d) < 1e-14:
        return np.column_stack([(1.0 - t) * ax + t * bx, (1.0 - t) * ay + t * by])
    a2 = ax * ax + ay * ay
    m2 = mx * mx + my * my
    b2 = bx * bx + by * by
    ox = (a2 * (my - by) + m2 * (by - ay) + b2 * (ay - my)) / d
    oy = (a2 * (bx - mx) + m2 * (ax - bx) + b2 * (mx - ax)) / d
    radius = math.hypot(ax - ox, ay - oy)
    th_a = math.atan2(ay - oy, ax - ox)
    th_m = math.atan2(my - oy, mx - ox)
    th_b = math.atan2(by - oy, bx - ox)
    d1 = math.atan2(math.sin(th_m - th_a), math.cos(th_m - th_a))
    d2 = math.atan2(math.sin(th_b - th_m), math.cos(th_b - th_m))
    sweep = d1 + d2
    thetas = th_a + t * sweep
    pts = np.column_stack([ox + radius * np.cos(thetas), oy + radius * np.sin(thetas)])
    pts[0] = (ax, ay)
    pts[-1] = (bx, by)
    return pts

def _edge_curve(
    verts_2d: tuple[Point2D, ...],
    arc_map: Mapping[tuple[int, int], Point2D],
    u: int,
    v: int,
    t: np.ndarray,
) -> np.ndarray:
    p0 = verts_2d[u]
    p1 = verts_2d[v]
    if (u, v) in arc_map:
        return _circular_arc_points(p0, p1, arc_map[(u, v)], t)
    return np.column_stack(
        [(1.0 - t) * p0[0] + t * p1[0], (1.0 - t) * p0[1] + t * p1[1]]
    )

def _extract_2d_arcs(spec: MeshSpec) -> dict[tuple[int, int], Point2D]:
    n2d = len(spec.vertices_2d)
    arc_map: dict[tuple[int, int], Point2D] = {}
    for u, v, mid_3d in spec.arcs:
        if u < n2d and v < n2d:
            mid = (float(mid_3d[0]), float(mid_3d[1]))
            arc_map[(u, v)] = mid
            arc_map[(v, u)] = mid
    return arc_map

def _patch_name_to_tag(patch_name: str, patch_type: str) -> int | None:
    lname = patch_name.lower()
    if lname == "inlet":
        return INLET
    if lname.startswith("outlet"):
        return OUTLET
    if lname == "axis":
        return AXIS
    if lname == "walls" or patch_type.lower() == "wall":
        return WALL
    return None

def _extract_2d_boundary_edge_tags(spec: MeshSpec) -> dict[tuple[int, int], int]:
    n2d = len(spec.vertices_2d)
    edge_tags: dict[tuple[int, int], int] = {}
    for patch in spec.patches:
        if patch.patch_type == "empty":
            continue
        tag = _patch_name_to_tag(patch.name, patch.patch_type)
        if tag is None:
            continue
        for face in patch.faces:
            u, v = int(face[0]), int(face[1])
            if u < n2d and v < n2d:
                edge_tags[(u, v)] = tag
    return edge_tags

class _SpatialPointMerger:
    """O(N) 2D point deduplicator with tolerance ``tol`` using a 3x3 hash-bucket lookup."""

    def __init__(self, tol: float = 1e-9) -> None:
        self.tol = tol
        self.tol2 = tol * tol
        self.cell_size = max(10.0 * tol, 1e-8)
        self.buckets: dict[tuple[int, int], list[int]] = {}
        self.coords: list[tuple[float, float]] = []

    def add(self, x: float, y: float) -> int:
        bx = int(math.floor(x / self.cell_size))
        by = int(math.floor(y / self.cell_size))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for idx in self.buckets.get((bx + dx, by + dy), ()):
                    ox, oy = self.coords[idx]
                    if (x - ox) ** 2 + (y - oy) ** 2 <= self.tol2:
                        return idx
        new_id = len(self.coords)
        self.coords.append((x, y))
        self.buckets.setdefault((bx, by), []).append(new_id)
        return new_id

    def as_array(self) -> np.ndarray:
        return np.asarray(self.coords, dtype=np.float64)

def build_mesh_spec(family: str, params: Mapping[str, Any], level: int = 1) -> MeshSpec:
    """Build ``MeshSpec`` for any supported family (identical to mesh_from_spec)."""
    key = family.upper()
    if key == "G1":
        return build_g1(params, level=level)
    return build_geometry(key, params, level=level)

def expected_domain_area(spec: MeshSpec) -> float:
    """Exact 2D analytical area of the geometry represented by ``spec``."""
    fam = spec.family.upper()
    if fam == "G6":
        Rc = float(spec.params["Rc_ratio"])
        n_bends = int(spec.params["n_bends"])
        L_in = float(spec.params.get("L_in", 5.0))
        L_straight = float(spec.params.get("L_straight", 2.0))
        L_out = float(spec.params.get("L_out_ratio", 10.0))
        straight_len = L_in + (n_bends - 1) * L_straight + L_out
        return float(straight_len * 1.0 + n_bends * math.pi * Rc)
    total = 0.0
    for blk in spec.blocks_2d:
        p0, p1, p2, p3 = (spec.vertices_2d[i] for i in blk.verts)
        a1 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        a2 = (p2[0] - p0[0]) * (p3[1] - p0[1]) - (p2[1] - p0[1]) * (p3[0] - p0[0])
        total += 0.5 * (a1 + a2)
    return float(total)


# ---------------------------------------------------------------------------
# T26 Option 1: opt-in corner fillets (spec.params["fillet_r"], default 0 = off)
# ---------------------------------------------------------------------------

FILLET_FAMILIES = frozenset({"G3", "G4", "G5"})


def _point_in_convex_quad(px: float, py: float, quad: tuple[Point2D, ...]) -> bool:
    """True iff ``(px, py)`` is inside (or on) the CCW convex quad (tolerance 1e-12)."""
    for k in range(4):
        ax, ay = quad[k]
        bx, by = quad[(k + 1) % 4]
        if (bx - ax) * (py - ay) - (by - ay) * (px - ax) < -1e-12:
            return False
    return True


def _fillet_corners(spec: MeshSpec, fillet_r: float) -> list[dict[str, Any]]:
    """Detect the re-entrant (solid-side) wall corners of G3/G4/G5 and their fillet data.

    A 2D vertex with exactly two straight WALL edges whose smaller-angle bisector points
    into the solid (i.e. just outside every block quad) is a re-entrant fluid corner:
    the four stenosis corners of G3 ``(0,c), (0,1-c), (L_t,c), (L_t,1-c)`` and the
    bifurcation outer junction corners ``T1``/``T2`` plus carina apex ``A`` of G4/G5.
    Convex fluid corners (bisector inside a block) are NOT filleted.

    For each corner with wall tangents ``t_a``/``t_b`` (unit, pointing away from the
    corner along the two wall edges) and solid-side angle ``alpha`` between them, the
    fillet of radius ``r`` is the circular arc tangent to both walls with centre
    ``O = V + (r / sin(alpha/2)) * bhat`` on the solid bisector ``bhat`` and tangent
    length ``L = r / tan(alpha/2)``.  The effective radius is capped at
    ``r0 = min(fillet_r, 0.4 * min(len_a, len_b))`` (for G3 this is the task rule
    ``min(fillet_r, 0.4*c, 0.25*L_t)`` since ``min(len_a, len_b) = c`` there).
    """
    if fillet_r <= 0.0:
        return []
    fam = spec.family.upper()
    if fam not in FILLET_FAMILIES:
        raise ValueError(
            f"fillet_r > 0 only supported for families {sorted(FILLET_FAMILIES)}, got {fam}"
        )
    arc_map = _extract_2d_arcs(spec)
    edge_tags = _extract_2d_boundary_edge_tags(spec)
    verts = spec.vertices_2d
    incident: dict[int, list[tuple[int, int]]] = {}
    for (u, v), tag in edge_tags.items():
        if tag != WALL:
            continue
        incident.setdefault(u, []).append((u, v))
        incident.setdefault(v, []).append((u, v))
    scale = max(max(abs(x), abs(y)) for x, y in verts)
    eps = 1e-6 * max(scale, 1.0)
    quads = [tuple(verts[i] for i in blk.verts) for blk in spec.blocks_2d]
    corners: list[dict[str, Any]] = []
    for v, edges in sorted(incident.items()):
        if len(edges) != 2:
            continue
        e_a, e_b = edges
        if e_a in arc_map or e_b in arc_map:
            continue  # only straight wall edges are filleted
        other_a = e_a[1] if e_a[0] == v else e_a[0]
        other_b = e_b[1] if e_b[0] == v else e_b[0]
        V = verts[v]
        pa, pb = verts[other_a], verts[other_b]
        len_a = math.hypot(pa[0] - V[0], pa[1] - V[1])
        len_b = math.hypot(pb[0] - V[0], pb[1] - V[1])
        if len_a < 1e-12 or len_b < 1e-12:
            continue
        t_a = ((pa[0] - V[0]) / len_a, (pa[1] - V[1]) / len_a)
        t_b = ((pb[0] - V[0]) / len_b, (pb[1] - V[1]) / len_b)
        cos_a = max(-1.0, min(1.0, t_a[0] * t_b[0] + t_a[1] * t_b[1]))
        alpha = math.acos(cos_a)
        if alpha < 1e-6:
            continue
        bis = (t_a[0] + t_b[0], t_a[1] + t_b[1])
        nb = math.hypot(bis[0], bis[1])
        if nb < 1e-9:
            continue  # straight wall (alpha ~ pi), not a corner
        bhat = (bis[0] / nb, bis[1] / nb)
        # Solid-side test: V + eps*bhat inside any block quad -> bisector is the fluid side.
        if any(_point_in_convex_quad(V[0] + eps * bhat[0], V[1] + eps * bhat[1], q) for q in quads):
            continue  # convex fluid corner: not filleted
        r0 = min(fillet_r, 0.4 * min(len_a, len_b))
        if r0 <= 0.0:
            continue
        half = 0.5 * alpha
        corners.append(
            {
                "v": v,
                "V": V,
                "e_a": e_a,
                "e_b": e_b,
                "len_a": len_a,
                "len_b": len_b,
                "half": half,
                "bhat": bhat,
                "t_a": t_a,
                "t_b": t_b,
            }
        )
    return corners


def _fillet_arc(corner: dict[str, Any], L: float) -> dict[str, Any]:
    """Complete fillet arc data for a tangent length ``L`` (radius ``r = L*tan(half)``)."""
    r = L * math.tan(corner["half"])
    V, bhat = corner["V"], corner["bhat"]
    O = (V[0] + (r / math.sin(corner["half"])) * bhat[0], V[1] + (r / math.sin(corner["half"])) * bhat[1])
    t_a, t_b = corner["t_a"], corner["t_b"]
    P_a = (V[0] + L * t_a[0], V[1] + L * t_a[1])
    P_b = (V[0] + L * t_b[0], V[1] + L * t_b[1])
    th_a = math.atan2(P_a[1] - O[1], P_a[0] - O[0])
    th_b = math.atan2(P_b[1] - O[1], P_b[0] - O[0])
    dth = (th_b - th_a + math.pi) % (2.0 * math.pi) - math.pi  # short sweep toward V
    return {**corner, "L": L, "r": r, "O": O, "th_a": th_a, "dth": dth}


def _fillet_arc_point(corner: dict[str, Any], side: str, s: float) -> tuple[float, float]:
    """Map a wall node at distance ``s`` from the corner onto the fillet arc.

    ``side`` is ``"a"`` or ``"b"`` (which of the two wall edges the node lies on).
    The corner itself (``s = 0`` on both edges) maps to the arc midpoint (``u = 0.5``),
    the tangent points (``s = L``) map to the arc ends ``u = 0`` / ``u = 1``.
    """
    L = corner["L"]
    u = 0.5 * (L - s) / L if side == "a" else 0.5 + 0.5 * s / L
    th = corner["th_a"] + u * corner["dth"]
    return (corner["O"][0] + corner["r"] * math.cos(th), corner["O"][1] + corner["r"] * math.sin(th))


# ---------------------------------------------------------------------------
# 1D grading helpers (matching bench_zhao.py)
# ---------------------------------------------------------------------------

def _tanh2(n: int, b: float) -> np.ndarray:
    """``n`` intervals on [0, 1], clustered at BOTH ends (tanh)."""
    s = np.linspace(-1.0, 1.0, n + 1)
    return 0.5 * (1.0 + np.tanh(b * s) / np.tanh(b))


def _geo(n: int, r: float) -> np.ndarray:
    """``n`` intervals on [0, 1], finest at 0, growth ratio ``r`` per interval."""
    w = r ** np.arange(n)
    return np.concatenate([[0.0], np.cumsum(w) / w.sum()])


def _geo_rev(n: int, r: float) -> np.ndarray:
    """``n`` intervals on [0, 1], finest at 1 (coarsening from 0)."""
    return 1.0 - _geo(n, r)[::-1]


def _uniform(n: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n + 1, dtype=np.float64)


def _refine_helpers(refine: float):
    """Return ``(c, g)``: cell-count scaler and growth-ratio scaler (bench_zhao.py line 56 rule)."""
    if refine <= 0.0:
        raise ValueError(f"refine must be > 0, got {refine}")
    c = lambda k: max(4, int(round(k * refine)))  # noqa: E731
    g = lambda r0: r0 ** (1.0 / refine)  # noqa: E731
    return c, g


# ---------------------------------------------------------------------------
# Family-specific conforming block schedules
# ---------------------------------------------------------------------------

def graded_block_schedule(spec: MeshSpec, refine: float = 1.0) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Per-block ``(nx, ny, xis, etas)`` graded schedule for family ``G1``-``G7``.

    Every shared block edge uses identical node coordinates (shared edges run either along a
    common ``etas`` array on all blocks of a family, or along a common ``xis`` array within a
    block group), so the mesh stays 100% conforming.
    """
    fam = spec.family.upper()
    c, g = _refine_helpers(refine)
    nb = len(spec.blocks_2d)

    if fam == "G1":
        # 1 block: straight channel [0, L] x [0, 1].
        nx = c(60)
        ny = c(16)
        return [(nx, ny, _uniform(nx), _tanh2(ny, 2.2))]

    if fam == "G2":
        # 4 blocks: blk 0 inlet [-L_in, 0] x [0, 1]; blk 1,2,3 downstream [0, L_out] x
        # lower recess / core / upper recess.
        L_in = float(spec.entry_length)
        L_out = float(spec.outlet_lengths[0])
        nx_in = c(max(20, min(40, round(4.0 * L_in))))
        nx_out = c(max(50, min(90, round(3.0 * L_out))))
        ny_core = c(16)
        ny_outer = c(10)
        xis_in = _geo_rev(nx_in, g(1.06))    # finest at step x = 0
        xis_out = _geo(nx_out, g(1.05))       # finest at step x = 0
        etas_core = _tanh2(ny_core, 2.2)
        etas_outer = _tanh2(ny_outer, 2.0)
        return [
            (nx_in, ny_core, xis_in, etas_core),
            (nx_out, ny_outer, xis_out, etas_outer),
            (nx_out, ny_core, xis_out, etas_core),
            (nx_out, ny_outer, xis_out, etas_outer),
        ]

    if fam == "G3":
        # 7 blocks: blk 0,1,2 upstream [-L_in, 0]; blk 3 throat [0, L_t];
        # blk 4,5,6 downstream [L_t, L_t + L_out].
        nx_in = c(24)
        nx_t = c(max(16, round(8.0 * float(spec.params["L_t"]))))
        nx_out = c(56)
        ny_core = c(14)
        ny_side = c(8)
        xis_in = _geo_rev(nx_in, g(1.06))     # finest at contraction x = 0
        xis_t = _tanh2(nx_t, 1.5)             # clustered at both contraction & expansion corners
        xis_out = _geo(nx_out, g(1.05))       # finest at expansion x = L_t
        etas_core = _tanh2(ny_core, 2.2)
        etas_side = _tanh2(ny_side, 2.0)
        return [
            (nx_in, ny_side, xis_in, etas_side),
            (nx_in, ny_core, xis_in, etas_core),
            (nx_in, ny_side, xis_in, etas_side),
            (nx_t, ny_core, xis_t, etas_core),
            (nx_out, ny_side, xis_out, etas_side),
            (nx_out, ny_core, xis_out, etas_core),
            (nx_out, ny_side, xis_out, etas_side),
        ]

    if fam in ("G4", "G5"):
        # 8 blocks: blk 0,1 parent halves; blk 2,3 junction; blk 4,5 daughter entry;
        # blk 6,7 daughter branches.  Shared ny on ALL 8 blocks -> exact edge conformity.
        ny = c(14)
        etas = _tanh2(ny, 2.2)
        nx_par = c(24)
        nx_j = c(12)
        nx_de = c(10)
        nx_br = c(40)
        sched: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        for i in range(nb):
            if i in (0, 1):
                sched.append((nx_par, ny, _geo_rev(nx_par, g(1.06)), etas))  # finest at junction
            elif i in (2, 3):
                sched.append((nx_j, ny, _uniform(nx_j), etas))
            elif i in (4, 5):
                sched.append((nx_de, ny, _uniform(nx_de), etas))
            else:
                sched.append((nx_br, ny, _geo(nx_br, g(1.05)), etas))  # finest at junction
        return sched

    if fam == "G6":
        # Serpentine: blk 0 inlet; for each bend k two 90-deg arc blocks (1+3k, 2+3k) +
        # straight connector (3+3k); the final block is the outlet.  Shared ny on ALL blocks.
        n_bends = int(spec.params["n_bends"])
        if nb != 1 + 3 * n_bends:
            raise ValueError(f"G6: expected {1 + 3 * n_bends} blocks for n_bends={n_bends}, got {nb}")
        ny = c(16)
        etas = _tanh2(ny, 2.2)
        nx_in = c(24)
        sched = [(nx_in, ny, _geo_rev(nx_in, g(1.06)), etas)]  # finest at bend entry
        for k in range(n_bends):
            for _j in (1 + 3 * k, 2 + 3 * k):  # curved arc blocks
                nx = c(20)
                sched.append((nx, ny, _uniform(nx), etas))
            idx = 3 + 3 * k
            if idx == nb - 1:  # last block = outlet
                nx = c(48)
                sched.append((nx, ny, _geo(nx, g(1.05)), etas))  # finest at bend exit
            else:  # straight connector
                nx = c(14)
                sched.append((nx, ny, _uniform(nx), etas))
        return sched

    if fam == "G7":
        # 18 blocks: 0,1 inlet; 2..5 expansion; 6..9 junction; 10..13 daughter entry;
        # 14..17 daughter outlets.  Shared ny on ALL 18 blocks -> 100% conforming.
        ny = c(10)
        etas = _tanh2(ny, 2.0)
        sched = []
        for i in range(nb):
            if i in (0, 1):
                nx = c(20)
                sched.append((nx, ny, _geo_rev(nx, g(1.06)), etas))  # finest at step
            elif 2 <= i <= 5:
                nx = c(36)
                sched.append((nx, ny, _tanh2(nx, 1.5), etas))  # clustered at step & junction
            elif 6 <= i <= 13:
                nx = c(10)
                sched.append((nx, ny, _uniform(nx), etas))
            else:
                nx = c(40)
                sched.append((nx, ny, _geo(nx, g(1.05)), etas))  # finest at junction
        return sched

    if fam in ("G8", "G9", "G10", "G11", "G12", "G13", "KG"):
        # T36 strip families: every block keeps its (nx, ny); xis is uniform per
        # strip (identical for all bands of a strip -> conforming horizontal edges)
        # and etas is tanh-clustered at both band edges per band (identical across
        # strips -> conforming vertical edges; clusters at outer walls AND island
        # walls for G11/G13 so wall_psi resolves the island).
        sched = []
        for blk in spec.blocks_2d:
            nx = c(blk.nx)
            ny = c(blk.ny)
            sched.append((nx, ny, _uniform(nx), _tanh2(ny, 2.2)))
        return sched

    raise ValueError(f"graded_block_schedule: unsupported family '{fam}' (supported: G1-G13, KG)")


# ---------------------------------------------------------------------------
# Pure-NumPy graded mesh arrays (host-testable)
# ---------------------------------------------------------------------------

def _quad_area(p00, p10, p01, p11) -> float:
    """Signed area of quad with vertex ordering (0,0), (1,0), (0,1), (1,1) (CCW positive)."""
    x = (p00[0], p10[0], p11[0], p01[0])
    y = (p00[1], p10[1], p11[1], p01[1])
    s = 0.0
    for k in range(4):
        s += x[k] * y[(k + 1) % 4] - x[(k + 1) % 4] * y[k]
    return 0.5 * s


def graded_mesh_arrays_from_spec(
    spec: MeshSpec,
    refine: float = 1.0,
    merge_tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, int]], float]:
    """Build graded quad-mesh arrays from ``spec`` without dolfinx.

    Returns ``(nodes, cells, tagged_midpoints, min_wall_dy)`` where
    - ``nodes``: ``(n_nodes, 2)`` float64 node coordinates (block nodes merged at shared edges),
    - ``cells``: ``(n_cells, 4)`` int64 connectivity in dolfinx quad order
      ``(0,0), (1,0), (0,1), (1,1)``,
    - ``tagged_midpoints``: list of ``(mx, my, tag)`` exterior sub-edge midpoints
      (``INLET=1``, ``OUTLET=2``, ``WALL=3``),
    - ``min_wall_dy``: minimum wall-normal thickness ``2*area/|edge|`` over all WALL
      sub-edges (the finest near-wall cell spacing).
    """
    arc_map = _extract_2d_arcs(spec)
    edge_tags = _extract_2d_boundary_edge_tags(spec)
    verts_2d = spec.vertices_2d
    sched = graded_block_schedule(spec, refine=refine)
    if len(sched) != len(spec.blocks_2d):
        raise ValueError(
            f"graded_block_schedule({spec.family}) returned {len(sched)} blocks, "
            f"spec has {len(spec.blocks_2d)}"
        )

    # T26 Option 1: opt-in corner fillets (fillet_r = 0 -> no corners, no moves).
    fillet_r = float(spec.params.get("fillet_r", 0.0))
    raw_corners = _fillet_corners(spec, fillet_r)
    edge_fillet: dict[tuple[int, int], tuple[dict[str, Any], str]] = {}
    if raw_corners:
        fillet_edge_set = {e for c in raw_corners for e in (c["e_a"], c["e_b"])}
        # Cap the tangent length L so the arc never spans more than the finest node
        # segment on either wall edge (keeps every quad Jacobian strictly positive).
        seg_min: dict[tuple[int, int], float] = {}
        for blk, (nx, ny, xis, etas) in zip(spec.blocks_2d, sched):
            v0, v1, v2, v3 = blk.verts
            for (a, b), n, par in (
                ((v0, v1), nx, xis),
                ((v1, v2), ny, etas),
                ((v2, v3), nx, xis),
                ((v3, v0), ny, etas),
            ):
                if (a, b) not in fillet_edge_set:
                    continue
                ln = math.hypot(verts_2d[b][0] - verts_2d[a][0], verts_2d[b][1] - verts_2d[a][1])
                d = float(np.min(np.diff(par))) * ln if n > 1 else ln
                seg_min[(a, b)] = min(seg_min.get((a, b), ln), d)
        for c in raw_corners:
            r0 = min(fillet_r, 0.4 * min(c["len_a"], c["len_b"]))
            L0 = r0 / math.tan(c["half"])
            cap = min(seg_min.get(c["e_a"], L0), seg_min.get(c["e_b"], L0))
            corner = _fillet_arc(c, min(L0, 0.8 * cap))
            edge_fillet[corner["e_a"]] = (corner, "a")
            edge_fillet[corner["e_b"]] = (corner, "b")
    moves: dict[int, tuple[float, float]] = {}
    boundary_segs: list[tuple[int, int, int, tuple[int, int, int, int]]] = []

    merger = _SpatialPointMerger(tol=merge_tol)
    cells_list: list[np.ndarray] = []

    for blk, (nx, ny, xis, etas) in zip(spec.blocks_2d, sched):
        v0, v1, v2, v3 = blk.verts

        c_bot = _edge_curve(verts_2d, arc_map, v0, v1, xis)    # (nx+1, 2), eta=0
        c_top = _edge_curve(verts_2d, arc_map, v3, v2, xis)    # (nx+1, 2), eta=1
        c_left = _edge_curve(verts_2d, arc_map, v0, v3, etas)  # (ny+1, 2), xi=0
        c_right = _edge_curve(verts_2d, arc_map, v1, v2, etas) # (ny+1, 2), xi=1

        p00 = np.asarray(verts_2d[v0], dtype=np.float64)
        p10 = np.asarray(verts_2d[v1], dtype=np.float64)
        p11 = np.asarray(verts_2d[v2], dtype=np.float64)
        p01 = np.asarray(verts_2d[v3], dtype=np.float64)

        # Bilinear Coons patch: P(xi, eta) of shape (nx+1, ny+1, 2).
        xi_g = xis[:, None, None]
        eta_g = etas[None, :, None]
        pts = (
            (1.0 - eta_g) * c_bot[:, None, :]
            + eta_g * c_top[:, None, :]
            + (1.0 - xi_g) * c_left[None, :, :]
            + xi_g * c_right[None, :, :]
            - (
                (1.0 - xi_g) * (1.0 - eta_g) * p00
                + xi_g * (1.0 - eta_g) * p10
                + (1.0 - xi_g) * eta_g * p01
                + xi_g * eta_g * p11
            )
        )

        nid = np.empty((nx + 1, ny + 1), dtype=np.int64)
        for i in range(nx + 1):
            for j in range(ny + 1):
                nid[i, j] = merger.add(float(pts[i, j, 0]), float(pts[i, j, 1]))

        # dolfinx / basix tensor-product quad vertex ordering: (0,0), (1,0), (0,1), (1,1).
        blk_cells = np.empty((nx * ny, 4), dtype=np.int64)
        blk_cells[:, 0] = nid[:-1, :-1].ravel()
        blk_cells[:, 1] = nid[1:, :-1].ravel()
        blk_cells[:, 2] = nid[:-1, 1:].ravel()
        blk_cells[:, 3] = nid[1:, 1:].ravel()
        cells_list.append(blk_cells)

        # Boundary sub-edges: (edge_key, node_ids (n+1 along the side), param-from-a (n+1),
        # adjacent cell flat index per segment).  Midpoints / wall thicknesses are computed
        # AFTER the fillet node moves from the merged node array (see below).
        side_defs = (
            ((v0, v1), nid[:, 0], xis, np.arange(nx)),                        # bottom: nodes (i, 0)
            ((v1, v2), nid[nx, :], etas, np.arange(ny) * nx + (nx - 1)),      # right: nodes (nx, j)
            ((v2, v3), nid[:, ny], 1.0 - xis, (ny - 1) * nx + np.arange(nx)),  # top: nodes (i, ny), param from v2
            ((v3, v0), nid[0, :], 1.0 - etas, np.arange(ny) * nx),            # left: nodes (0, j), param from v3
        )
        for edge_key, node_ids, par_a, cell_loc in side_defs:
            tag = edge_tags.get(edge_key)
            if tag is None:
                continue
            # T26 fillet: move wall nodes within the tangent length L of a sharp corner
            # onto the fillet arc (merged node ids -> conforming shared edges stay exact).
            ff = edge_fillet.get(edge_key)
            if ff is not None:
                corner, side = ff
                elen = corner["len_a"] if side == "a" else corner["len_b"]
                v_corner = corner["v"]
                a_end = edge_key[0]
                for k in range(len(par_a)):
                    s = float(par_a[k]) * elen if v_corner == a_end else (1.0 - float(par_a[k])) * elen
                    if s <= corner["L"] * (1.0 + 1e-12):
                        moves[int(node_ids[k])] = _fillet_arc_point(corner, side, s)
            for s in range(len(cell_loc)):
                ci = int(cell_loc[s])
                i, j = ci % nx, ci // nx
                quad = (int(nid[i, j]), int(nid[i + 1, j]), int(nid[i, j + 1]), int(nid[i + 1, j + 1]))
                boundary_segs.append((int(node_ids[s]), int(node_ids[s + 1]), int(tag), quad))

    all_cells = np.vstack(cells_list)
    all_nodes = merger.as_array().copy()
    if moves:
        for nid_, tgt in moves.items():
            all_nodes[nid_] = tgt
    tagged_midpoints: list[tuple[float, float, int]] = []
    wall_h: list[float] = []
    for na, nb, tag, quad in boundary_segs:
        a_pt = all_nodes[na]
        b_pt = all_nodes[nb]
        tagged_midpoints.append(
            (0.5 * (a_pt[0] + b_pt[0]), 0.5 * (a_pt[1] + b_pt[1]), tag)
        )
        if tag == WALL:
            seg_len = math.hypot(b_pt[0] - a_pt[0], b_pt[1] - a_pt[1])
            if seg_len > 0.0:
                area = _quad_area(all_nodes[quad[0]], all_nodes[quad[1]], all_nodes[quad[2]], all_nodes[quad[3]])
                wall_h.append(2.0 * area / seg_len)
    min_wall_dy = float(min(wall_h)) if wall_h else float("nan")
    return all_nodes, all_cells, tagged_midpoints, min_wall_dy


# ---------------------------------------------------------------------------
# Test / analysis helpers (pure NumPy)
# ---------------------------------------------------------------------------

def cell_signed_areas(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Signed area of every quad cell (vertex order (0,0), (1,0), (0,1), (1,1); CCW positive)."""
    p = nodes[cells]
    p00, p10, p01, p11 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    x = np.stack([p00[:, 0], p10[:, 0], p11[:, 0], p01[:, 0]], axis=1)
    y = np.stack([p00[:, 1], p10[:, 1], p11[:, 1], p01[:, 1]], axis=1)
    return 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)


def edge_use_counts(cells: np.ndarray) -> dict[tuple[int, int], int]:
    """Map each undirected cell edge to the number of cells using it."""
    counts: dict[tuple[int, int], int] = {}
    for c in cells:
        for a, b in ((int(c[0]), int(c[1])), (int(c[1]), int(c[3])),
                     (int(c[3]), int(c[2])), (int(c[2]), int(c[0]))):
            key = (a, b) if a < b else (b, a)
            counts[key] = counts.get(key, 0) + 1
    return counts


def wall_normal_thicknesses(
    nodes: np.ndarray,
    cells: np.ndarray,
    tagged_midpoints: list[tuple[float, float, int]],
) -> list[tuple[float, bool, int, float, float]]:
    """Per WALL sub-edge ``(h, is_horizontal, tag, mx, my)`` with ``h = 2*area/|edge|``."""
    edge_cell: dict[tuple[int, int], list[int]] = {}
    for ci, c in enumerate(cells):
        for a, b in ((int(c[0]), int(c[1])), (int(c[1]), int(c[3])),
                     (int(c[3]), int(c[2])), (int(c[2]), int(c[0]))):
            key = (a, b) if a < b else (b, a)
            edge_cell.setdefault(key, []).append(ci)
    areas = cell_signed_areas(nodes, cells)
    tol = 1e-9
    cell_size = 1e-8
    buckets: dict[tuple[int, int], list[tuple[float, float, tuple[int, int], list[int]]]] = {}
    for e, cl in edge_cell.items():
        a, b = e
        mx = 0.5 * (nodes[a, 0] + nodes[b, 0])
        my = 0.5 * (nodes[a, 1] + nodes[b, 1])
        buckets.setdefault((int(math.floor(mx / cell_size)), int(math.floor(my / cell_size))), []).append(
            (mx, my, e, cl)
        )
    out: list[tuple[float, bool, int, float, float]] = []
    for mx, my, tag in tagged_midpoints:
        if tag != WALL:
            continue
        bx = int(math.floor(mx / cell_size))
        by = int(math.floor(my / cell_size))
        found = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for emx, emy, e, cl in buckets.get((bx + dx, by + dy), ()):
                    if (emx - mx) ** 2 + (emy - my) ** 2 <= tol * tol:
                        found = (e, cl)
                        break
                if found:
                    break
            if found:
                break
        if found is None:
            raise RuntimeError("WALL tagged midpoint not found among cell edges")
        e, cl = found
        a, b = e
        ex = nodes[b, 0] - nodes[a, 0]
        ey = nodes[b, 1] - nodes[a, 1]
        elen = math.hypot(ex, ey)
        h = 2.0 * float(areas[cl[0]]) / elen
        out.append((h, abs(ex) >= abs(ey), tag, mx, my))
    return out


# ---------------------------------------------------------------------------
# dolfinx mesh construction (Docker only)
# ---------------------------------------------------------------------------

def build_graded_dolfinx_mesh(
    family: str,
    params: Mapping[str, Any],
    refine: float = 1.0,
    comm=None,
):
    """Build the graded dolfinx quad ``Mesh`` + exterior facet ``MeshTags`` + ``MeshSpec``.

    Returns ``(msh, tags, spec)``.  ``spec`` is always built at ``level=1``; ``refine`` scales
    the graded cell schedules.  Facet tagging matches ``mesh_from_spec.mesh_from_spec``
    exactly (exterior facet midpoint matching against tagged block-boundary sub-edges).
    """
    import basix.ufl
    import ufl
    from dolfinx import mesh
    from mpi4py import MPI

    if comm is None:
        comm = MPI.COMM_WORLD

    spec = build_mesh_spec(family, params, level=1)
    nodes, cells, tagged_midpoints, _min_wall_dy = graded_mesh_arrays(spec, refine=refine)

    domain = ufl.Mesh(basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,)))
    msh = mesh.create_mesh(comm, cells, nodes, domain)

    # Tag exterior facets by matching midpoints against the exact block-boundary sub-edges.
    merge_tol = 1e-9
    fdim = 1
    msh.topology.create_connectivity(fdim, 2)
    bfacets = mesh.exterior_facet_indices(msh.topology)
    bmid = mesh.compute_midpoints(msh, fdim, bfacets)

    match_tol = max(1e-7, 100.0 * merge_tol)
    match_tol2 = match_tol * match_tol
    bucket_h = 10.0 * match_tol
    mid_buckets: dict[tuple[int, int], list[tuple[float, float, int]]] = {}
    for mx, my, tag in tagged_midpoints:
        bx = int(math.floor(mx / bucket_h))
        by = int(math.floor(my / bucket_h))
        mid_buckets.setdefault((bx, by), []).append((mx, my, tag))

    facet_tags = np.empty(len(bfacets), dtype=np.int32)
    unmatched = 0
    for idx in range(len(bfacets)):
        fx, fy = float(bmid[idx, 0]), float(bmid[idx, 1])
        bx = int(math.floor(fx / bucket_h))
        by = int(math.floor(fy / bucket_h))
        best_tag = -1
        best_d2 = match_tol2
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for mx, my, tag in mid_buckets.get((bx + dx, by + dy), ()):
                    d2 = (fx - mx) ** 2 + (fy - my) ** 2
                    if d2 <= best_d2:
                        best_d2 = d2
                        best_tag = tag
        if best_tag < 0:
            unmatched += 1
            best_tag = WALL
        facet_tags[idx] = best_tag

    if unmatched > 0:
        raise RuntimeError(
            f"build_graded_dolfinx_mesh({spec.family}, refine={refine}): "
            f"{unmatched}/{len(bfacets)} exterior facets failed boundary-edge tag matching"
        )

    order = np.argsort(bfacets)
    tags = mesh.meshtags(msh, fdim, bfacets[order], facet_tags[order])
    return msh, tags, spec


# ---------------------------------------------------------------------------
# Public host-safe wrapper + lazy FEniCSx constructors
# ---------------------------------------------------------------------------

def graded_mesh_arrays(family: str, params, refine: float = 1.0, merge_tol: float = 1e-9):
    """Build graded quad-mesh arrays for ``family`` with geometry ``params`` (pure NumPy).

    Returns ``(nodes, cells, tagged_midpoints, min_wall_dy)`` -- see
    :func:`graded_mesh_arrays_from_spec`.  ``family`` in {'G1'..'G13', 'KG'}.
    """
    spec = build_mesh_spec(family, params, level=1)
    return graded_mesh_arrays_from_spec(spec, refine=refine, merge_tol=merge_tol)


def build_dolfinx_mesh(family: str, params, refine: float = 1.0, comm=None):
    """Lazy FEniCSx: graded dolfinx quad ``Mesh`` + facet ``MeshTags`` + ``MeshSpec``.

    Requires dolfinx/ufl/basix/mpi4py (e.g. the ``dolfinx/dolfinx:v0.9.0`` Docker image).
    Returns ``(msh, tags, spec)``.
    """
    return build_graded_dolfinx_mesh(family, params, refine=refine, comm=comm)


def custom_mesh(msh, facet_tags):
    """Lazy FEniCSx: wrap a user-supplied dolfinx ``Mesh`` + ``MeshTags`` as a solve target.

    Returns ``(msh, facet_tags)`` unchanged so external users can plug in any custom
    domain; ``EulerEulerSolver.solve`` accepts either a ``(msh, tags)`` tuple, a family
    string, or a ``MeshSpec``.
    """
    return msh, facet_tags


def from_gmsh(path, comm=None):
    """Lazy FEniCSx: read a Gmsh file into ``(msh, facet_tags)`` via ``dolfinx.io.gmshio``.

    Facet tags are read from the Gmsh physical groups; ``INLET/OUTLET/WALL`` are the
    numeric tags 1/2/3.  Requires dolfinx + mpi4py + gmsh.
    """
    from mpi4py import MPI
    from dolfinx.io import gmshio
    if comm is None:
        comm = MPI.COMM_WORLD
    if hasattr(gmshio, "read_from_msh"):
        msh, _cell_tags, facet_tags = gmshio.read_from_msh(str(path), comm, 0)
    else:  # dolfinx >= 0.8 fallback
        import gmsh
        gmsh.initialize()
        gmsh.open(str(path))
        msh, _cell_tags, facet_tags = gmshio.model_to_mesh(gmsh, comm, 0, gdim=2)
        gmsh.finalize()
    return msh, facet_tags
