"""Build a dolfinx 2D quadrilateral mesh from ``cfd.gen.geometry`` MeshSpec blocks.

Reproduces the exact vertices, block subdivisions, simpleGrading, and circular-arc edges
of OpenFOAM ``blockMesh`` so FEniCSx Euler–Euler V3/V4 runs share the exact same mesh
as paired OpenFOAM V1/V2 runs.

Boundary facet tags (matching ``cfd/fenicsx/twofluid.py`` and ``bench_kg.py``):
- ``INLET  = 1``
- ``OUTLET = 2`` (all outlets share tag 2 -> equal natural pressure p = 0)
- ``WALL   = 3``
- ``AXIS   = 4`` (KG axisymmetric centerline r = 0)
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import basix.ufl
import numpy as np
import ufl
from dolfinx import mesh
from mpi4py import MPI

from cfd.gen.geometry import MeshSpec, Point2D, build_geometry
from cfd.gen.m2b import build_g1

INLET = 1
OUTLET = 2
WALL = 3
AXIS = 4


def _simple_grading_1d(n_cells: int, grading: float) -> np.ndarray:
    """Return ``n_cells + 1`` normalized coordinates in ``[0, 1]`` matching OpenFOAM simpleGrading.

    OpenFOAM ``simpleGrading`` defines ``grading = dx_last / dx_first``.
    """
    if n_cells < 1:
        raise ValueError(f"n_cells must be >= 1, got {n_cells}")
    if grading <= 0.0:
        raise ValueError(f"grading must be > 0, got {grading}")
    if n_cells == 1 or abs(grading - 1.0) < 1e-12:
        return np.linspace(0.0, 1.0, n_cells + 1, dtype=np.float64)
    r = grading ** (1.0 / (n_cells - 1))
    idx = np.arange(n_cells + 1, dtype=np.float64)
    t = (r**idx - 1.0) / (r**n_cells - 1.0)
    t[0] = 0.0
    t[-1] = 1.0
    return t


def _circular_arc_points(
    p0: Point2D,
    p1: Point2D,
    pm: Point2D,
    t: np.ndarray,
) -> np.ndarray:
    """Sample circular arc from ``p0`` to ``p1`` passing through interior point ``pm``.

    Matches OpenFOAM ``blockMesh`` ``arc`` interpolation along normalized parameter ``t`` in ``[0, 1]``.
    Falls back to linear interpolation if the three points are collinear.
    """
    ax, ay = p0
    bx, by = p1
    mx, my = pm

    # Circumcenter of triangle (p0, pm, p1).
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

    # Directed angular sweep passing through pm (valid for any arc < 360 deg).
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
    """Evaluate 2D edge curve from vertex ``u`` to ``v`` at parameter values ``t`` in ``[0, 1]``."""
    p0 = verts_2d[u]
    p1 = verts_2d[v]
    if (u, v) in arc_map:
        return _circular_arc_points(p0, p1, arc_map[(u, v)], t)
    return np.column_stack(
        [(1.0 - t) * p0[0] + t * p1[0], (1.0 - t) * p0[1] + t * p1[1]]
    )


def _extract_2d_arcs(spec: MeshSpec) -> dict[tuple[int, int], Point2D]:
    """Extract 2D circular arcs on the ``z = 0`` plane from ``spec.arcs``."""
    n2d = len(spec.vertices_2d)
    arc_map: dict[tuple[int, int], Point2D] = {}
    for u, v, mid_3d in spec.arcs:
        if u < n2d and v < n2d:
            mid = (float(mid_3d[0]), float(mid_3d[1]))
            arc_map[(u, v)] = mid
            arc_map[(v, u)] = mid
    return arc_map


def _patch_name_to_tag(patch_name: str, patch_type: str) -> int | None:
    """Map an OpenFOAM side patch name/type to a FEniCSx facet tag."""
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
    """Map directed 2D exterior block edges ``(u, v)`` to facet tags."""
    if spec.family.upper() == "KG":
        # KG's 3D wedge patches use 13 wedge vertices, while blocks_2d uses 8 meridian vertices:
        # 0:(-L_in,0), 1:(0,0), 2:(L_out,0), 3:(-L_in,r1), 4:(0,r1), 5:(L_out,r1), 6:(0,r2), 7:(L_out,r2)
        return {
            (3, 0): INLET,
            (2, 5): OUTLET,
            (5, 7): OUTLET,
            (0, 1): AXIS,
            (1, 2): AXIS,
            (4, 3): WALL,
            (6, 4): WALL,
            (7, 6): WALL,
        }

    n2d = len(spec.vertices_2d)
    edge_tags: dict[tuple[int, int], int] = {}
    for patch in spec.patches:
        if patch.patch_type == "empty":
            continue
        tag = _patch_name_to_tag(patch.name, patch.patch_type)
        if tag is None:
            continue
        for face in patch.faces:
            # _extrude_2d_mesh builds side faces as (u, v, v + n2d, u + n2d).
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
    """Build ``MeshSpec`` for any supported family ``G1``–``G7`` or ``KG``."""
    key = family.upper()
    if key == "G1":
        return build_g1(params, level=level)
    return build_geometry(key, params, level=level)


def expected_cell_count(spec: MeshSpec) -> int:
    """Exact total number of 2D cells in ``spec`` (equals blockMesh cell count)."""
    return int(sum(blk.nx * blk.ny for blk in spec.blocks_2d))


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
        # Each 180-deg annulus of width 1 and centerline radius Rc has area pi * Rc.
        return float(straight_len * 1.0 + n_bends * math.pi * Rc)

    total = 0.0
    for blk in spec.blocks_2d:
        p0, p1, p2, p3 = (spec.vertices_2d[i] for i in blk.verts)
        a1 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        a2 = (p2[0] - p0[0]) * (p3[1] - p0[1]) - (p2[1] - p0[1]) * (p3[0] - p0[0])
        total += 0.5 * (a1 + a2)
    return float(total)


def mesh_from_spec(
    spec: MeshSpec,
    comm: MPI.Comm = MPI.COMM_WORLD,
    merge_tol: float = 1e-9,
) -> tuple[mesh.Mesh, mesh.MeshTags]:
    """Construct a dolfinx quadrilateral ``Mesh`` and exterior facet ``MeshTags`` from ``spec``.

    Every block is mapped with a bilinear Coons patch (transfinite interpolation) supporting
    arbitrary ``simpleGrading`` and circular ``arcs`` (e.g. G6 serpentines). Shared nodes
    across block interfaces are merged cleanly, and every exterior facet is tagged with
    ``INLET=1``, ``OUTLET=2``, ``WALL=3``, or ``AXIS=4``.
    """
    arc_map = _extract_2d_arcs(spec)
    edge_tags = _extract_2d_boundary_edge_tags(spec)
    verts_2d = spec.vertices_2d

    merger = _SpatialPointMerger(tol=merge_tol)
    cells_list: list[np.ndarray] = []
    # Tagged boundary sub-edge midpoints for exact, renumbering-invariant facet matching.
    tagged_midpoints: list[tuple[float, float, int]] = []

    for blk in spec.blocks_2d:
        v0, v1, v2, v3 = blk.verts
        nx, ny = int(blk.nx), int(blk.ny)
        gx, gy = float(blk.grading[0]), float(blk.grading[1])

        xis = _simple_grading_1d(nx, gx)
        etas = _simple_grading_1d(ny, gy)

        c_bot = _edge_curve(verts_2d, arc_map, v0, v1, xis)   # (nx+1, 2), eta=0
        c_top = _edge_curve(verts_2d, arc_map, v3, v2, xis)   # (nx+1, 2), eta=1
        c_left = _edge_curve(verts_2d, arc_map, v0, v3, etas) # (ny+1, 2), xi=0
        c_right = _edge_curve(verts_2d, arc_map, v1, v2, etas)# (ny+1, 2), xi=1

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

        # Merge block nodes into global node table.
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

        # Record boundary sub-edge midpoints for any side belonging to an exterior patch.
        side_defs = (
            ((v0, v1), pts[:-1, 0], pts[1:, 0]),     # bottom: j=0, i=0..nx-1
            ((v1, v2), pts[nx, :-1], pts[nx, 1:]),   # right:  i=nx, j=0..ny-1
            ((v2, v3), pts[1:, ny], pts[:-1, ny]),   # top:    j=ny, i=nx..1
            ((v3, v0), pts[0, 1:], pts[0, :-1]),     # left:   i=0, j=ny..1
        )
        for edge_key, seg_a, seg_b in side_defs:
            tag = edge_tags.get(edge_key)
            if tag is not None:
                mids = 0.5 * (seg_a + seg_b)
                for mx, my in mids:
                    tagged_midpoints.append((float(mx), float(my), int(tag)))

    all_cells = np.vstack(cells_list)
    all_nodes = merger.as_array()

    domain = ufl.Mesh(basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,)))
    msh = mesh.create_mesh(comm, all_cells, all_nodes, domain)

    # Tag exterior facets by matching midpoints against the exact block-boundary sub-edges.
    fdim = 1
    msh.topology.create_connectivity(fdim, 2)
    bfacets = mesh.exterior_facet_indices(msh.topology)
    bmid = mesh.compute_midpoints(msh, fdim, bfacets)

    # Spatial hash of expected boundary sub-edge midpoints.
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
            f"mesh_from_spec({spec.family}, level={spec.level}): "
            f"{unmatched}/{len(bfacets)} exterior facets failed boundary-edge tag matching"
        )

    order = np.argsort(bfacets)
    tags = mesh.meshtags(msh, fdim, bfacets[order], facet_tags[order])
    return msh, tags


def build_dolfinx_mesh(
    family: str,
    params: Mapping[str, Any],
    level: int = 1,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> tuple[mesh.Mesh, mesh.MeshTags, MeshSpec]:
    """Build ``MeshSpec`` and its matching dolfinx quad mesh + facet tags in one call."""
    spec = build_mesh_spec(family, params, level=level)
    msh, tags = mesh_from_spec(spec, comm=comm)
    return msh, tags, spec
