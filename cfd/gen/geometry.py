"""Nondimensional 2D OpenFOAM blockMesh geometry generators for families G2–G7.

All lengths are in star units with inlet height h1* = 1.0.
Every family places the inlet on x = x_in with y in [0, 1] (centerline y = 0.5)
so the fully developed Newtonian parabolic profile u*(y*) = 6 y* (1 - y*)
applies uniformly across all families.

Mesh refinement levels: level in {0, 1, 2} with refinement ratio r = 1.5
in both in-plane directions (for Roache GCI studies, plan/CORPUS_PLAN.md §0).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

Point2D = tuple[float, float]
Point3D = tuple[float, float, float]
Edge2D = tuple[int, int]
Face3D = tuple[int, int, int, int]
Hex3D = tuple[int, int, int, int, int, int, int, int]


@dataclass(frozen=True)
class Block2D:
    """A counter-clockwise 2D quad block extruded one cell thick in +z."""

    verts: tuple[int, int, int, int]
    nx: int
    ny: int
    grading: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class ArcEdge:
    """A curved circular arc edge in 2D (extruded to both z=0 and z=1)."""

    u: int
    v: int
    midpoint: Point2D


@dataclass(frozen=True)
class SampleLine:
    """A 1D sampling line in star units (midplane z = 0.5)."""

    name: str
    axis: str  # "x", "y", or "distance"
    start: Point3D
    end: Point3D
    n_points: int = 80


@dataclass(frozen=True)
class PatchSpec:
    """An OpenFOAM boundary patch."""

    name: str
    patch_type: str  # "patch", "wall", or "empty"
    faces: tuple[Face3D, ...]


@dataclass(frozen=True)
class MeshSpec:
    """Complete 3D extruded 2D blockMesh specification in star units."""

    family: str
    level: int
    params: dict[str, float | int]
    vertices_2d: tuple[Point2D, ...]
    blocks_2d: tuple[Block2D, ...]
    vertices: tuple[Point3D, ...]
    blocks: tuple[tuple[Hex3D, tuple[int, int, int], tuple[float, float, float]], ...]
    arcs: tuple[tuple[int, int, Point3D], ...]
    patches: tuple[PatchSpec, ...]
    sample_lines: tuple[SampleLine, ...]
    inlet_height: float
    outlet_heights: tuple[float, ...]
    entry_length: float
    outlet_lengths: tuple[float, ...]


def _scale_cells(base_cells: int, level: int, minimum: int = 4) -> int:
    """Scale cell count by r^level with r = 1.5 for level in {0, 1, 2, 3}."""
    if level not in (0, 1, 2, 3):
        raise ValueError(f"mesh level must be in {{0, 1, 2, 3}}, got {level}")
    return max(minimum, int(round(base_cells * (1.5**level))))


def _base_cells_for_length(length: float, cells_per_unit: float, minimum: int = 4) -> int:
    """Choose a base cell count (level 0) rounded to an even integer for clean 1.5x scaling."""
    raw = max(float(minimum), length * cells_per_unit)
    # Round to nearest multiple of 4 when >= 4 so 1.5 and 2.25 give exact integers.
    return max(minimum, int(round(raw / 4.0)) * 4)


def _cross_2d(o: Point2D, a: Point2D, b: Point2D) -> float:
    """2D signed cross product (a - o) x (b - o)."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _quad_area_2d(pts: tuple[Point2D, Point2D, Point2D, Point2D]) -> float:
    """Signed area of a 2D quad (positive iff counter-clockwise)."""
    p0, p1, p2, p3 = pts
    return 0.5 * (_cross_2d(p0, p1, p2) + _cross_2d(p0, p2, p3))


def _check_convex_ccw(
    verts_2d: list[Point2D], block: Block2D, block_idx: int
) -> None:
    """Raise ValueError if a 2D block is not strictly convex and counter-clockwise."""
    i0, i1, i2, i3 = block.verts
    pts = (verts_2d[i0], verts_2d[i1], verts_2d[i2], verts_2d[i3])
    for k in range(4):
        prev_pt = pts[(k - 1) % 4]
        curr_pt = pts[k]
        next_pt = pts[(k + 1) % 4]
        c = _cross_2d(curr_pt, next_pt, prev_pt)
        # Note: (next - curr) x (prev - curr) > 0? Wait:
        # CCW interior turn: (curr - prev) x (next - curr) > 0.
        turn = (curr_pt[0] - prev_pt[0]) * (next_pt[1] - curr_pt[1]) - (
            curr_pt[1] - prev_pt[1]
        ) * (next_pt[0] - curr_pt[0])
        if turn <= 1e-12:
            raise ValueError(
                f"Block {block_idx} verts={block.verts} is not strictly convex CCW "
                f"at corner {k} (turn={turn:.3e}, cross={c:.3e}, pts={pts})"
            )
    if _quad_area_2d(pts) <= 1e-12:
        raise ValueError(f"Block {block_idx} has non-positive area")


def _extrude_2d_mesh(
    *,
    family: str,
    level: int,
    params: dict[str, float | int],
    verts_2d: list[Point2D],
    blocks_2d: list[Block2D],
    patch_edges: list[tuple[str, str, list[Edge2D]]],
    sample_lines: list[SampleLine],
    inlet_height: float,
    outlet_heights: tuple[float, ...],
    entry_length: float,
    outlet_lengths: tuple[float, ...],
    arcs_2d: list[ArcEdge] | None = None,
    z_thickness: float = 1.0,
) -> MeshSpec:
    """Extrude a validated 2D multi-block mesh one cell thick in +z.

    Verifies:
    - Every 2D block is strictly convex and CCW.
    - Every shared internal edge has matching cell counts in opposite directions.
    - Patch edge lists partition the exterior boundary edges exactly once.
    """
    n_verts = len(verts_2d)
    if n_verts < 4 or not blocks_2d:
        raise ValueError("Mesh must have at least 4 vertices and 1 block")

    # Validate block convexity and collect directed side edges with their cell counts.
    edge_counts: Counter[Edge2D] = Counter()
    edge_cells: dict[Edge2D, int] = {}

    for idx, blk in enumerate(blocks_2d):
        _check_convex_ccw(verts_2d, blk, idx)
        if blk.nx < 1 or blk.ny < 1:
            raise ValueError(f"Block {idx} has invalid cell counts ({blk.nx}, {blk.ny})")
        v0, v1, v2, v3 = blk.verts
        blk_edges = (
            ((v0, v1), blk.nx),
            ((v1, v2), blk.ny),
            ((v2, v3), blk.nx),
            ((v3, v0), blk.ny),
        )
        for edge, n_cells in blk_edges:
            edge_counts[edge] += 1
            if edge_counts[edge] > 1:
                raise ValueError(
                    f"Duplicate directed edge {edge} in block {idx} (overlapping blocks)"
                )
            edge_cells[edge] = n_cells

    # Check shared internal edges and identify exterior boundary edges.
    boundary_edges: set[Edge2D] = set()
    for (u, v), n_cells in edge_cells.items():
        rev = (v, u)
        if rev in edge_cells:
            if edge_cells[rev] != n_cells:
                raise ValueError(
                    f"Cell count mismatch across shared edge ({u}, {v}): "
                    f"{n_cells} vs {edge_cells[rev]}"
                )
        else:
            boundary_edges.add((u, v))

    # Build 3D vertices and blocks.
    verts_3d: list[Point3D] = [(x, y, 0.0) for (x, y) in verts_2d] + [
        (x, y, z_thickness) for (x, y) in verts_2d
    ]
    blocks_3d: list[tuple[Hex3D, tuple[int, int, int], tuple[float, float, float]]] = []
    front_and_back_faces: list[Face3D] = []

    for blk in blocks_2d:
        a, b, c, d = blk.verts
        hex_verts: Hex3D = (a, b, c, d, a + n_verts, b + n_verts, c + n_verts, d + n_verts)
        blocks_3d.append((hex_verts, (blk.nx, blk.ny, 1), blk.grading))
        # Outward normals: back (z=0) is CW in xy (-z), front (z=z_thickness) is CCW (+z).
        front_and_back_faces.append((a, d, c, b))
        front_and_back_faces.append((a + n_verts, b + n_verts, c + n_verts, d + n_verts))

    # Build side patches and verify exact partition of boundary edges.
    assigned_edges: set[Edge2D] = set()
    patches: list[PatchSpec] = []
    for patch_name, patch_type, edges in patch_edges:
        faces: list[Face3D] = []
        for u, v in edges:
            if (u, v) not in boundary_edges:
                raise ValueError(
                    f"Patch '{patch_name}' references edge {(u, v)} which is not "
                    f"an outward boundary edge"
                )
            if (u, v) in assigned_edges:
                raise ValueError(
                    f"Boundary edge {(u, v)} assigned to multiple patches (dup in '{patch_name}')"
                )
            assigned_edges.add((u, v))
            # Outward normal for CCW directed edge (u -> v) extruded in +z.
            faces.append((u, v, v + n_verts, u + n_verts))
        patches.append(PatchSpec(name=patch_name, patch_type=patch_type, faces=tuple(faces)))

    unassigned = boundary_edges - assigned_edges
    if unassigned:
        raise ValueError(f"Unassigned boundary edges in {family}: {sorted(unassigned)}")

    patches.append(
        PatchSpec(
            name="frontAndBack",
            patch_type="empty",
            faces=tuple(front_and_back_faces),
        )
    )

    # Extrude 2D arcs to both z=0 and z=z_thickness.
    arcs_3d: list[tuple[int, int, Point3D]] = []
    for arc in arcs_2d or []:
        mx, my = arc.midpoint
        arcs_3d.append((arc.u, arc.v, (mx, my, 0.0)))
        arcs_3d.append((arc.u + n_verts, arc.v + n_verts, (mx, my, z_thickness)))

    return MeshSpec(
        family=family,
        level=level,
        params=params,
        vertices_2d=tuple(verts_2d),
        blocks_2d=tuple(blocks_2d),
        vertices=tuple(verts_3d),
        blocks=tuple(blocks_3d),
        arcs=tuple(arcs_3d),
        patches=tuple(patches),
        sample_lines=tuple(sample_lines),
        inlet_height=inlet_height,
        outlet_heights=outlet_heights,
        entry_length=entry_length,
        outlet_lengths=outlet_lengths,
    )


def _line_pts(x0: float, y0: float, x1: float, y1: float, eps: float = 0.02) -> tuple[Point3D, Point3D]:
    """Pull endpoints inward by eps along a 2D line segment so samples stay inside the mesh."""
    dx = x1 - x0
    dy = y1 - y0
    dist = math.hypot(dx, dy)
    if dist <= 2.0 * eps:
        eps = 0.05 * dist
    ux = dx / dist
    uy = dy / dist
    return (
        (x0 + eps * ux, y0 + eps * uy, 0.5),
        (x1 - eps * ux, y1 - eps * uy, 0.5),
    )


def build_g2(params: Mapping[str, Any], level: int = 1) -> MeshSpec:
    """Build G2 (symmetric sudden expansion) in star units (h1 = 1).

    Parameters
    ----------
    params:
        E : float in [1.5, 4.0]
            Expansion ratio h2 / h1.
        L_in : float >= 5.0, optional (default 5.0)
            Entry length in star units.
        L_out_ratio : float >= 10.0, optional (default 10.0)
            Outlet length divided by outlet height E (L_out = L_out_ratio * E).
    level : int in {0, 1, 2}
        Grid refinement level (refinement ratio r = 1.5).
    """
    E = float(params["E"])
    L_in = float(params.get("L_in", 5.0))
    L_out_ratio = float(params.get("L_out_ratio", 10.0))
    if not (1.5 <= E <= 4.0):
        raise ValueError(f"G2 E must be in [1.5, 4.0], got {E}")
    if L_in < 5.0:
        raise ValueError(f"G2 L_in must be >= 5.0, got {L_in}")
    if L_out_ratio < 10.0:
        raise ValueError(f"G2 L_out_ratio must be >= 10.0, got {L_out_ratio}")

    s = 0.5 * (E - 1.0)
    L_out = L_out_ratio * E

    verts_2d: list[Point2D] = [
        (-L_in, 0.0),      # 0
        (-L_in, 1.0),      # 1
        (0.0, -s),         # 2
        (0.0, 0.0),        # 3
        (0.0, 1.0),        # 4
        (0.0, 1.0 + s),    # 5
        (L_out, -s),       # 6
        (L_out, 0.0),      # 7
        (L_out, 1.0),      # 8
        (L_out, 1.0 + s),  # 9
    ]

    ny_in = _scale_cells(16, level)
    ny_step = _scale_cells(_base_cells_for_length(s, 16.0, minimum=4), level)
    nx_in = _scale_cells(_base_cells_for_length(L_in, 8.0, minimum=16), level)
    nx_out = _scale_cells(_base_cells_for_length(L_out, 6.0, minimum=40), level)

    blocks_2d = [
        Block2D((0, 3, 4, 1), nx_in, ny_in),
        Block2D((2, 6, 7, 3), nx_out, ny_step),
        Block2D((3, 7, 8, 4), nx_out, ny_in),
        Block2D((4, 8, 9, 5), nx_out, ny_step),
    ]

    patch_edges = [
        ("inlet", "patch", [(1, 0)]),
        ("outlet", "patch", [(6, 7), (7, 8), (8, 9)]),
        ("walls", "wall", [(0, 3), (3, 2), (2, 6), (9, 5), (5, 4), (4, 1)]),
    ]

    sample_lines = [
        SampleLine("inletProfile", "y", *_line_pts(-0.5, 0.0, -0.5, 1.0), n_points=80),
        SampleLine(
            "nearWallLower",
            "x",
            (0.02, -s + 0.02, 0.5),
            (L_out - 0.02, -s + 0.02, 0.5),
            n_points=200,
        ),
        SampleLine(
            "centerline",
            "x",
            (-L_in + 0.02, 0.5, 0.5),
            (L_out - 0.02, 0.5, 0.5),
            n_points=200,
        ),
        SampleLine(
            "outletProfile",
            "y",
            *_line_pts(L_out - 1.0, -s, L_out - 1.0, 1.0 + s),
            n_points=100,
        ),
    ]

    return _extrude_2d_mesh(
        family="G2",
        level=level,
        params={"E": E, "L_in": L_in, "L_out_ratio": L_out_ratio},
        verts_2d=verts_2d,
        blocks_2d=blocks_2d,
        patch_edges=patch_edges,
        sample_lines=sample_lines,
        inlet_height=1.0,
        outlet_heights=(E,),
        entry_length=L_in,
        outlet_lengths=(L_out,),
    )


def build_g3(params: Mapping[str, Any], level: int = 1) -> MeshSpec:
    """Build G3 (symmetric contraction–expansion / stenosis) in star units (h1 = 1).

    Parameters
    ----------
    params:
        throat_ratio : float in [0.3, 0.8]
            Throat height h_t / h1.
        L_t : float in [1.0, 5.0]
            Throat length L_t / h1.
        L_in : float >= 5.0, optional (default 5.0)
        L_out_ratio : float >= 10.0, optional (default 10.0)
    level : int in {0, 1, 2}
    """
    throat_ratio = float(params["throat_ratio"])
    L_t = float(params["L_t"])
    L_in = float(params.get("L_in", 5.0))
    L_out_ratio = float(params.get("L_out_ratio", 10.0))

    if not (0.3 <= throat_ratio <= 0.8):
        raise ValueError(f"G3 throat_ratio must be in [0.3, 0.8], got {throat_ratio}")
    if not (1.0 <= L_t <= 5.0):
        raise ValueError(f"G3 L_t must be in [1.0, 5.0], got {L_t}")
    if L_in < 5.0:
        raise ValueError(f"G3 L_in must be >= 5.0, got {L_in}")
    if L_out_ratio < 10.0:
        raise ValueError(f"G3 L_out_ratio must be >= 10.0, got {L_out_ratio}")

    c = 0.5 * (1.0 - throat_ratio)
    L_out = L_out_ratio * 1.0  # outlet height h_out = 1

    xs = (-L_in, 0.0, L_t, L_t + L_out)
    ys = (0.0, c, 1.0 - c, 1.0)
    verts_2d: list[Point2D] = [(x, y) for x in xs for y in ys]
    # Index helper: col * 4 + row (col in 0..3, row in 0..3)

    ny_throat = _scale_cells(_base_cells_for_length(throat_ratio, 20.0, minimum=8), level)
    ny_side = _scale_cells(_base_cells_for_length(c, 20.0, minimum=4), level)
    nx_in = _scale_cells(_base_cells_for_length(L_in, 8.0, minimum=16), level)
    nx_t = _scale_cells(_base_cells_for_length(L_t, 12.0, minimum=8), level)
    nx_out = _scale_cells(_base_cells_for_length(L_out, 8.0, minimum=40), level)

    blocks_2d = [
        # Inlet (3 stacked bands)
        Block2D((0, 4, 5, 1), nx_in, ny_side),
        Block2D((1, 5, 6, 2), nx_in, ny_throat),
        Block2D((2, 6, 7, 3), nx_in, ny_side),
        # Throat (middle band)
        Block2D((5, 9, 10, 6), nx_t, ny_throat),
        # Outlet (3 stacked bands)
        Block2D((8, 12, 13, 9), nx_out, ny_side),
        Block2D((9, 13, 14, 10), nx_out, ny_throat),
        Block2D((10, 14, 15, 11), nx_out, ny_side),
    ]

    patch_edges = [
        ("inlet", "patch", [(1, 0), (2, 1), (3, 2)]),
        ("outlet", "patch", [(12, 13), (13, 14), (14, 15)]),
        (
            "walls",
            "wall",
            [
                (0, 4), (4, 5), (5, 9), (9, 8), (8, 12),
                (15, 11), (11, 10), (10, 6), (6, 7), (7, 3),
            ],
        ),
    ]

    sample_lines = [
        SampleLine("inletProfile", "y", *_line_pts(-0.5, 0.0, -0.5, 1.0), n_points=80),
        SampleLine(
            "throatMid",
            "y",
            *_line_pts(0.5 * L_t, c, 0.5 * L_t, 1.0 - c),
            n_points=80,
        ),
        SampleLine(
            "nearWallDownstream",
            "x",
            (L_t + 0.02, 0.02, 0.5),
            (L_t + L_out - 0.02, 0.02, 0.5),
            n_points=200,
        ),
        SampleLine(
            "centerline",
            "x",
            (-L_in + 0.02, 0.5, 0.5),
            (L_t + L_out - 0.02, 0.5, 0.5),
            n_points=200,
        ),
        SampleLine(
            "outletProfile",
            "y",
            *_line_pts(L_t + L_out - 1.0, 0.0, L_t + L_out - 1.0, 1.0),
            n_points=80,
        ),
    ]

    return _extrude_2d_mesh(
        family="G3",
        level=level,
        params={
            "throat_ratio": throat_ratio,
            "L_t": L_t,
            "L_in": L_in,
            "L_out_ratio": L_out_ratio,
        },
        verts_2d=verts_2d,
        blocks_2d=blocks_2d,
        patch_edges=patch_edges,
        sample_lines=sample_lines,
        inlet_height=1.0,
        outlet_heights=(1.0,),
        entry_length=L_in,
        outlet_lengths=(L_out,),
    )


def _build_bifurcation_core(
    *,
    family: str,
    level: int,
    params: dict[str, float | int],
    theta_deg: float,
    h_d1: float,
    h_d2: float,
    y_split: float,
    L_in: float,
    L_out_ratio: float,
) -> MeshSpec:
    """Shared structured blockMesh builder for symmetric (G4) and asymmetric (G5) bifurcations.

    Parent enters along +x on y in [0, 1] (height 1) from x = -L_in.
    Upper daughter has width h_d1 at angle +theta_deg; lower daughter has width h_d2 at -theta_deg.
    Uses either a 6-block single-trapezoid junction (when s_A >= 0.25 * h_d, giving low
    non-orthogonality at acute angles with wide daughters) or an 8-block mitered-trapezoid
    junction (guaranteed convex and well-conditioned up to theta = 90 deg T-junctions).
    """
    theta = math.radians(theta_deg)
    ct = math.cos(theta)
    st = math.sin(theta)

    # Unit tangents and left-normals for upper (1) and lower (2) daughters.
    t1 = (ct, st)
    n1 = (-st, ct)   # points from inner wall toward outer wall of upper daughter
    t2 = (ct, -st)
    n2 = (st, ct)    # points from outer wall toward inner wall of lower daughter

    h_p1 = 1.0 - y_split  # upper parent half height
    h_p2 = y_split        # lower parent half height

    # Place lower outer corner at x_T2 = 0. Shift x_T1 so both inner walls meet at one apex A(x_A, y_split).
    # Upper inner wall hits y = y_split at x = x_T1 + (h_d1 - h_p1 * ct) / st.
    # Lower inner wall hits y = y_split at x = x_T2 + (h_d2 - h_p2 * ct) / st.
    dx_1 = (h_d1 - h_p1 * ct) / st
    dx_2 = (h_d2 - h_p2 * ct) / st

    # Choose x_T1, x_T2 >= 0 with min(x_T1, x_T2) = 0 so x_A = max(dx_1, dx_2).
    if dx_2 >= dx_1:
        x_T2 = 0.0
        x_T1 = dx_2 - dx_1
        x_A = dx_2
    else:
        x_T1 = 0.0
        x_T2 = dx_1 - dx_2
        x_A = dx_1

    # Inner-wall axial coordinates of apex A relative to T1 and T2.
    s_A1 = (h_d1 * ct - h_p1) / st
    s_A2 = (h_d2 * ct - h_p2) / st

    L_d1 = L_out_ratio * h_d1
    L_d2 = L_out_ratio * h_d2

    ny_1 = _scale_cells(_base_cells_for_length(h_p1, 16.0, minimum=8), level)
    ny_2 = _scale_cells(_base_cells_for_length(h_p2, 16.0, minimum=8), level)
    nx_d1 = _scale_cells(_base_cells_for_length(L_d1, 6.0, minimum=36), level)
    nx_d2 = _scale_cells(_base_cells_for_length(L_d2, 6.0, minimum=36), level)

    # Regime check: single-trapezoid junction when BOTH branches have s_A >= 0.25 * h_d
    # AND x_T1 == x_T2 == 0 (symmetric acute wide-daughter case).
    use_direct = (
        abs(x_T1 - x_T2) < 1e-12
        and s_A1 >= 0.25 * h_d1
        and s_A2 >= 0.25 * h_d2
    )

    if use_direct:
        # 6-block topology: parent ends at x = 0, junction quads (C0, A, O_A1, T1) & (T2, O_A2, A, C0)
        # connect directly into orthogonal daughter channels.
        nx_in = _scale_cells(_base_cells_for_length(L_in, 8.0, minimum=16), level)
        nx_j = _scale_cells(_base_cells_for_length(max(x_A, s_A1), 10.0, minimum=6), level)

        T1 = (0.0, 1.0)
        T2 = (0.0, 0.0)
        C0 = (0.0, y_split)
        A = (x_A, y_split)
        O_A1 = (T1[0] + s_A1 * t1[0], T1[1] + s_A1 * t1[1])
        O_A2 = (T2[0] + s_A2 * t2[0], T2[1] + s_A2 * t2[1])

        I1_end = (A[0] + L_d1 * t1[0], A[1] + L_d1 * t1[1])
        O1_end = (O_A1[0] + L_d1 * t1[0], O_A1[1] + L_d1 * t1[1])
        O2_end = (O_A2[0] + L_d2 * t2[0], O_A2[1] + L_d2 * t2[1])
        I2_end = (A[0] + L_d2 * t2[0], A[1] + L_d2 * t2[1])

        verts_2d: list[Point2D] = [
            (-L_in, 0.0),      # 0: In_bot
            (-L_in, y_split),  # 1: In_mid
            (-L_in, 1.0),      # 2: In_top
            T2,                # 3
            C0,                # 4
            T1,                # 5
            A,                 # 6
            O_A1,              # 7
            O_A2,              # 8
            I1_end,            # 9
            O1_end,            # 10
            O2_end,            # 11
            I2_end,            # 12
        ]

        blocks_2d = [
            Block2D((0, 3, 4, 1), nx_in, ny_2),   # parent lower
            Block2D((1, 4, 5, 2), nx_in, ny_1),   # parent upper
            Block2D((4, 6, 7, 5), nx_j, ny_1),    # junction upper
            Block2D((3, 8, 6, 4), nx_j, ny_2),    # junction lower
            Block2D((6, 9, 10, 7), nx_d1, ny_1),  # daughter 1 (upper)
            Block2D((8, 11, 12, 6), nx_d2, ny_2), # daughter 2 (lower)
        ]

        patch_edges = [
            ("inlet", "patch", [(1, 0), (2, 1)]),
            ("outlet1", "patch", [(9, 10)]),
            ("outlet2", "patch", [(11, 12)]),
            (
                "walls",
                "wall",
                [
                    (0, 3), (3, 8), (8, 11),
                    (12, 6), (6, 9),
                    (10, 7), (7, 5), (5, 2),
                ],
            ),
        ]

        d1_mid_s = 0.5 * L_d1
        d2_mid_s = 0.5 * L_d2
        d1_start = (A[0] + d1_mid_s * t1[0], A[1] + d1_mid_s * t1[1])
        d1_end = (O_A1[0] + d1_mid_s * t1[0], O_A1[1] + d1_mid_s * t1[1])
        d2_start = (O_A2[0] + d2_mid_s * t2[0], O_A2[1] + d2_mid_s * t2[1])
        d2_end = (A[0] + d2_mid_s * t2[0], A[1] + d2_mid_s * t2[1])
        entry_len = L_in
    else:
        # 8-block mitered-trapezoid topology: parent straight channel runs from x = -L_in
        # to x_j = -L_buf (so L_in_straight = L_in - L_buf, total parent length to corners >= L_in).
        L_buf = 0.5
        x_in = -(L_in + L_buf)  # guarantees straight entry length >= L_in before junction block
        x_j = -L_buf

        T1 = (x_T1, 1.0)
        T2 = (x_T2, 0.0)
        P1 = (x_j, 1.0)
        C0 = (x_j, y_split)
        P2 = (x_j, 0.0)
        A = (x_A, y_split)

        # Place orthogonal daughter entrance planes past both corner T and apex A.
        s_cut1 = max(0.0, s_A1) + 0.5 * h_d1
        s_cut2 = max(0.0, s_A2) + 0.5 * h_d2

        O1 = (T1[0] + s_cut1 * t1[0], T1[1] + s_cut1 * t1[1])
        I1 = (O1[0] - h_d1 * n1[0], O1[1] - h_d1 * n1[1])
        O2 = (T2[0] + s_cut2 * t2[0], T2[1] + s_cut2 * t2[1])
        I2 = (O2[0] + h_d2 * n2[0], O2[1] + h_d2 * n2[1])

        I1_end = (I1[0] + L_d1 * t1[0], I1[1] + L_d1 * t1[1])
        O1_end = (O1[0] + L_d1 * t1[0], O1[1] + L_d1 * t1[1])
        O2_end = (O2[0] + L_d2 * t2[0], O2[1] + L_d2 * t2[1])
        I2_end = (I2[0] + L_d2 * t2[0], I2[1] + L_d2 * t2[1])

        nx_in = _scale_cells(_base_cells_for_length(L_in, 8.0, minimum=16), level)
        nx_j_in = _scale_cells(_base_cells_for_length(x_A - x_j, 12.0, minimum=6), level)
        nx_j1 = _scale_cells(_base_cells_for_length(s_cut1, 12.0, minimum=6), level)
        nx_j2 = _scale_cells(_base_cells_for_length(s_cut2, 12.0, minimum=6), level)

        verts_2d = [
            (x_in, 0.0),       # 0: In_bot
            (x_in, y_split),   # 1: In_mid
            (x_in, 1.0),       # 2: In_top
            P2,                # 3
            C0,                # 4
            P1,                # 5
            T2,                # 6
            A,                 # 7
            T1,                # 8
            I1,                # 9
            O1,                # 10
            O2,                # 11
            I2,                # 12
            I1_end,            # 13
            O1_end,            # 14
            O2_end,            # 15
            I2_end,            # 16
        ]

        blocks_2d = [
            Block2D((0, 3, 4, 1), nx_in, ny_2),     # parent lower
            Block2D((1, 4, 5, 2), nx_in, ny_1),     # parent upper
            Block2D((4, 7, 8, 5), nx_j_in, ny_1),   # miter entry upper
            Block2D((3, 6, 7, 4), nx_j_in, ny_2),   # miter entry lower
            Block2D((7, 9, 10, 8), nx_j1, ny_1),    # miter exit upper
            Block2D((6, 11, 12, 7), nx_j2, ny_2),   # miter exit lower
            Block2D((9, 13, 14, 10), nx_d1, ny_1),  # daughter 1 (upper)
            Block2D((11, 15, 16, 12), nx_d2, ny_2), # daughter 2 (lower)
        ]

        patch_edges = [
            ("inlet", "patch", [(1, 0), (2, 1)]),
            ("outlet1", "patch", [(13, 14)]),
            ("outlet2", "patch", [(15, 16)]),
            (
                "walls",
                "wall",
                [
                    (0, 3), (3, 6), (6, 11), (11, 15),
                    (16, 12), (12, 7), (7, 9), (9, 13),
                    (14, 10), (10, 8), (8, 5), (5, 2),
                ],
            ),
        ]

        d1_mid_s = 0.5 * L_d1
        d2_mid_s = 0.5 * L_d2
        d1_start = (I1[0] + d1_mid_s * t1[0], I1[1] + d1_mid_s * t1[1])
        d1_end = (O1[0] + d1_mid_s * t1[0], O1[1] + d1_mid_s * t1[1])
        d2_start = (O2[0] + d2_mid_s * t2[0], O2[1] + d2_mid_s * t2[1])
        d2_end = (I2[0] + d2_mid_s * t2[0], I2[1] + d2_mid_s * t2[1])
        entry_len = L_in + L_buf

    sample_lines = [
        SampleLine("inletProfile", "y", *_line_pts(-1.0, 0.0, -1.0, 1.0), n_points=80),
        SampleLine(
            "daughter1Mid",
            "distance",
            *_line_pts(d1_start[0], d1_start[1], d1_end[0], d1_end[1]),
            n_points=80,
        ),
        SampleLine(
            "daughter2Mid",
            "distance",
            *_line_pts(d2_start[0], d2_start[1], d2_end[0], d2_end[1]),
            n_points=80,
        ),
        SampleLine(
            "parentCenterline",
            "x",
            (-L_in + 0.05, y_split, 0.5),
            (x_A - 0.05, y_split, 0.5),
            n_points=120,
        ),
    ]

    return _extrude_2d_mesh(
        family=family,
        level=level,
        params=params,
        verts_2d=verts_2d,
        blocks_2d=blocks_2d,
        patch_edges=patch_edges,
        sample_lines=sample_lines,
        inlet_height=1.0,
        outlet_heights=(h_d1, h_d2),
        entry_length=entry_len,
        outlet_lengths=(L_d1, L_d2),
    )


def build_g4(params: Mapping[str, Any], level: int = 1) -> MeshSpec:
    """Build G4 (symmetric Y/T bifurcation) in star units (h1 = 1).

    Parameters
    ----------
    params:
        theta_deg : float in [20.0, 90.0]
            Half-angle between branch centerline and parent axis [deg].
        daughter_ratio : float in [0.5, 1.0]
            Daughter-to-parent width ratio h_d / h1.
        L_in : float >= 5.0, optional (default 5.0)
        L_out_ratio : float >= 10.0, optional (default 10.0)
    level : int in {0, 1, 2}
    """
    theta_deg = float(params["theta_deg"])
    daughter_ratio = float(params["daughter_ratio"])
    L_in = float(params.get("L_in", 5.0))
    L_out_ratio = float(params.get("L_out_ratio", 10.0))

    if not (20.0 <= theta_deg <= 90.0):
        raise ValueError(f"G4 theta_deg must be in [20, 90], got {theta_deg}")
    if not (0.5 <= daughter_ratio <= 1.0):
        raise ValueError(f"G4 daughter_ratio must be in [0.5, 1.0], got {daughter_ratio}")
    if L_in < 5.0:
        raise ValueError(f"G4 L_in must be >= 5.0, got {L_in}")
    if L_out_ratio < 10.0:
        raise ValueError(f"G4 L_out_ratio must be >= 10.0, got {L_out_ratio}")

    return _build_bifurcation_core(
        family="G4",
        level=level,
        params={
            "theta_deg": theta_deg,
            "daughter_ratio": daughter_ratio,
            "L_in": L_in,
            "L_out_ratio": L_out_ratio,
        },
        theta_deg=theta_deg,
        h_d1=daughter_ratio,
        h_d2=daughter_ratio,
        y_split=0.5,
        L_in=L_in,
        L_out_ratio=L_out_ratio,
    )


def build_g5(params: Mapping[str, Any], level: int = 1) -> MeshSpec:
    """Build G5 (asymmetric bifurcation) in star units (h1 = 1).

    Area-preserving daughter split: h_d1 + h_d2 = h1 = 1 with h_d1 / h_d2 = width_ratio,
    so h_d2 = 1 / (1 + width_ratio) and h_d1 = width_ratio / (1 + width_ratio).

    Parameters
    ----------
    params:
        width_ratio : float in [0.5, 1.0]
            Asymmetric daughter width ratio h_d1 / h_d2.
        theta_deg : float in [30.0, 90.0]
            Branch half-angle [deg].
        L_in : float >= 5.0, optional (default 5.0)
        L_out_ratio : float >= 10.0, optional (default 10.0)
    level : int in {0, 1, 2}
    """
    width_ratio = float(params["width_ratio"])
    theta_deg = float(params["theta_deg"])
    L_in = float(params.get("L_in", 5.0))
    L_out_ratio = float(params.get("L_out_ratio", 10.0))

    if not (0.5 <= width_ratio <= 1.0):
        raise ValueError(f"G5 width_ratio must be in [0.5, 1.0], got {width_ratio}")
    if not (30.0 <= theta_deg <= 90.0):
        raise ValueError(f"G5 theta_deg must be in [30, 90], got {theta_deg}")
    if L_in < 5.0:
        raise ValueError(f"G5 L_in must be >= 5.0, got {L_in}")
    if L_out_ratio < 10.0:
        raise ValueError(f"G5 L_out_ratio must be >= 10.0, got {L_out_ratio}")

    h_d2 = 1.0 / (1.0 + width_ratio)
    h_d1 = width_ratio * h_d2

    return _build_bifurcation_core(
        family="G5",
        level=level,
        params={
            "width_ratio": width_ratio,
            "theta_deg": theta_deg,
            "L_in": L_in,
            "L_out_ratio": L_out_ratio,
        },
        theta_deg=theta_deg,
        h_d1=h_d1,
        h_d2=h_d2,
        y_split=h_d2,
        L_in=L_in,
        L_out_ratio=L_out_ratio,
    )


def build_g6(params: Mapping[str, Any], level: int = 1) -> MeshSpec:
    """Build G6 (2D serpentine with 1–4 180-deg U-bends using blockMesh arcs) in star units.

    Parameters
    ----------
    params:
        Rc_ratio : float in [1.5, 6.0]
            Centerline bend radius R_c / h1.
        n_bends : int in {1, 2, 3, 4}
            Number of 180-degree U-bends.
        L_in : float >= 5.0, optional (default 5.0)
        L_straight : float >= 0.5, optional (default 2.0)
            Straight leg length between consecutive bends.
        L_out_ratio : float >= 10.0, optional (default 10.0)
    level : int in {0, 1, 2}
    """
    Rc = float(params["Rc_ratio"])
    n_bends = int(params["n_bends"])
    L_in = float(params.get("L_in", 5.0))
    L_straight = float(params.get("L_straight", 2.0))
    L_out_ratio = float(params.get("L_out_ratio", 10.0))

    if not (1.5 <= Rc <= 6.0):
        raise ValueError(f"G6 Rc_ratio must be in [1.5, 6.0], got {Rc}")
    if n_bends not in (1, 2, 3, 4):
        raise ValueError(f"G6 n_bends must be in {{1, 2, 3, 4}}, got {n_bends}")
    if L_in < 5.0:
        raise ValueError(f"G6 L_in must be >= 5.0, got {L_in}")
    if L_straight < 0.5:
        raise ValueError(f"G6 L_straight must be >= 0.5, got {L_straight}")
    if L_out_ratio < 10.0:
        raise ValueError(f"G6 L_out_ratio must be >= 10.0, got {L_out_ratio}")

    r_in = Rc - 0.5
    r_out = Rc + 0.5
    L_out = L_out_ratio * 1.0

    ny = _scale_cells(16, level)
    nx_in = _scale_cells(_base_cells_for_length(L_in, 8.0, minimum=16), level)
    nx_leg = _scale_cells(_base_cells_for_length(L_straight, 8.0, minimum=8), level)
    nx_out = _scale_cells(_base_cells_for_length(L_out, 6.0, minimum=40), level)
    # Quarter-circle arc length is 0.5 * pi * Rc
    nx_quad = _scale_cells(_base_cells_for_length(0.5 * math.pi * Rc, 8.0, minimum=12), level)

    verts_2d: list[Point2D] = []
    blocks_2d: list[Block2D] = []
    arcs_2d: list[ArcEdge] = []
    wall_edges: list[Edge2D] = []

    def add_v(pt: Point2D) -> int:
        verts_2d.append(pt)
        return len(verts_2d) - 1

    # Maintain a strip of cross-section pairs (v_right, v_left) relative to local flow direction:
    # For any block advancing along flow: (v_r0, v_r1, v_l1, v_l0) is CCW when
    # flow goes from 0 -> 1 and left-normal points from v_r -> v_l!
    # Check: at inlet (flow +x), v_r = bottom (y=0), v_l = top (y=1).
    # (0, 1, 2, 3) -> v0=(-L_in, 0), v1=(0, 0), v2=(0, 1), v3=(-L_in, 1): CCW!
    vr_curr = add_v((-L_in, 0.0))
    vl_curr = add_v((-L_in, 1.0))
    inlet_edge: Edge2D = (vl_curr, vr_curr)

    vr_next = add_v((0.0, 0.0))
    vl_next = add_v((0.0, 1.0))
    blocks_2d.append(Block2D((vr_curr, vr_next, vl_next, vl_curr), nx_in, ny))
    wall_edges.extend([(vr_curr, vr_next), (vl_next, vl_curr)])
    vr_curr, vl_curr = vr_next, vl_next

    bend1_apex_pts: tuple[Point2D, Point2D] = ((r_in, 0.5 + Rc), (r_out, 0.5 + Rc))

    for k in range(n_bends):
        # Current channel centerline y = 0.5 + 2*k*Rc
        y_bot = 2.0 * k * Rc
        yc = y_bot + 0.5 + Rc  # center of curvature y for bend k

        if k % 2 == 0:
            # Flowing in +x, turning CCW upward (+y) around (0, yc)
            # Currently: vr_curr = (0, y_bot) [outer], vl_curr = (0, y_bot+1) [inner].
            cx = 0.0
            # Quarter 1: -pi/2 -> 0
            vr_mid = add_v((cx + r_out, yc))
            vl_mid = add_v((cx + r_in, yc))
            blocks_2d.append(Block2D((vr_curr, vr_mid, vl_mid, vl_curr), nx_quad, ny))
            wall_edges.extend([(vr_curr, vr_mid), (vl_mid, vl_curr)])
            ang1 = -0.25 * math.pi
            arcs_2d.append(ArcEdge(vr_curr, vr_mid, (cx + r_out * math.cos(ang1), yc + r_out * math.sin(ang1))))
            arcs_2d.append(ArcEdge(vl_curr, vl_mid, (cx + r_in * math.cos(ang1), yc + r_in * math.sin(ang1))))
            if k == 0:
                bend1_apex_pts = ((cx + r_in, yc), (cx + r_out, yc))

            # Quarter 2: 0 -> +pi/2
            vr_end = add_v((cx, yc + r_out))
            vl_end = add_v((cx, yc + r_in))
            blocks_2d.append(Block2D((vr_mid, vr_end, vl_end, vl_mid), nx_quad, ny))
            wall_edges.extend([(vr_mid, vr_end), (vl_end, vl_mid)])
            ang2 = 0.25 * math.pi
            arcs_2d.append(ArcEdge(vr_mid, vr_end, (cx + r_out * math.cos(ang2), yc + r_out * math.sin(ang2))))
            arcs_2d.append(ArcEdge(vl_mid, vl_end, (cx + r_in * math.cos(ang2), yc + r_in * math.sin(ang2))))
            vr_curr, vl_curr = vr_end, vl_end

            # Straight segment going in -x (right-of-flow is top y = yc + r_out, left-of-flow is bottom y = yc + r_in)
            seg_len = L_out if k == n_bends - 1 else L_straight
            seg_nx = nx_out if k == n_bends - 1 else nx_leg
            vr_s = add_v((cx - seg_len, yc + r_out))
            vl_s = add_v((cx - seg_len, yc + r_in))
            blocks_2d.append(Block2D((vr_curr, vr_s, vl_s, vl_curr), seg_nx, ny))
            wall_edges.extend([(vr_curr, vr_s), (vl_s, vl_curr)])
            vr_curr, vl_curr = vr_s, vl_s
        else:
            # Flowing in -x, turning CW upward (+y) around (-L_straight, yc)
            # Currently: vr_curr = (-L_straight, y_bot+1) [inner!], vl_curr = (-L_straight, y_bot) [outer!].
            cx = -L_straight
            # Quarter 1: -pi/2 -> -pi (CW)
            vr_mid = add_v((cx - r_in, yc))
            vl_mid = add_v((cx - r_out, yc))
            blocks_2d.append(Block2D((vr_curr, vr_mid, vl_mid, vl_curr), nx_quad, ny))
            wall_edges.extend([(vr_curr, vr_mid), (vl_mid, vl_curr)])
            ang1 = -0.75 * math.pi
            arcs_2d.append(ArcEdge(vr_curr, vr_mid, (cx + r_in * math.cos(ang1), yc + r_in * math.sin(ang1))))
            arcs_2d.append(ArcEdge(vl_curr, vl_mid, (cx + r_out * math.cos(ang1), yc + r_out * math.sin(ang1))))

            # Quarter 2: -pi -> -3*pi/2 (i.e. +pi/2)
            vr_end = add_v((cx, yc + r_in))
            vl_end = add_v((cx, yc + r_out))
            blocks_2d.append(Block2D((vr_mid, vr_end, vl_end, vl_mid), nx_quad, ny))
            wall_edges.extend([(vr_mid, vr_end), (vl_end, vl_mid)])
            ang2 = 0.75 * math.pi
            arcs_2d.append(ArcEdge(vr_mid, vr_end, (cx + r_in * math.cos(ang2), yc + r_in * math.sin(ang2))))
            arcs_2d.append(ArcEdge(vl_mid, vl_end, (cx + r_out * math.cos(ang2), yc + r_out * math.sin(ang2))))
            vr_curr, vl_curr = vr_end, vl_end

            # Straight segment going in +x
            seg_len = L_out if k == n_bends - 1 else L_straight
            seg_nx = nx_out if k == n_bends - 1 else nx_leg
            vr_s = add_v((cx + seg_len, yc + r_in))
            vl_s = add_v((cx + seg_len, yc + r_out))
            blocks_2d.append(Block2D((vr_curr, vr_s, vl_s, vl_curr), seg_nx, ny))
            wall_edges.extend([(vr_curr, vr_s), (vl_s, vl_curr)])
            vr_curr, vl_curr = vr_s, vl_s

    outlet_edge: Edge2D = (vr_curr, vl_curr)
    out_r = verts_2d[vr_curr]
    out_l = verts_2d[vl_curr]
    out_x_sample = out_r[0] + (1.0 if n_bends % 2 == 1 else -1.0)

    patch_edges = [
        ("inlet", "patch", [inlet_edge]),
        ("outlet", "patch", [outlet_edge]),
        ("walls", "wall", wall_edges),
    ]

    sample_lines = [
        SampleLine("inletProfile", "y", *_line_pts(-0.5, 0.0, -0.5, 1.0), n_points=80),
        SampleLine(
            "bend1Apex",
            "x",
            *_line_pts(bend1_apex_pts[0][0], bend1_apex_pts[0][1], bend1_apex_pts[1][0], bend1_apex_pts[1][1]),
            n_points=80,
        ),
        SampleLine(
            "outletProfile",
            "y",
            *_line_pts(out_x_sample, min(out_r[1], out_l[1]), out_x_sample, max(out_r[1], out_l[1])),
            n_points=80,
        ),
    ]

    return _extrude_2d_mesh(
        family="G6",
        level=level,
        params={
            "Rc_ratio": Rc,
            "n_bends": n_bends,
            "L_in": L_in,
            "L_straight": L_straight,
            "L_out_ratio": L_out_ratio,
        },
        verts_2d=verts_2d,
        blocks_2d=blocks_2d,
        patch_edges=patch_edges,
        sample_lines=sample_lines,
        inlet_height=1.0,
        outlet_heights=(1.0,),
        entry_length=L_in,
        outlet_lengths=(L_out,),
        arcs_2d=arcs_2d,
    )


def build_g7(params: Mapping[str, Any], level: int = 1) -> MeshSpec:
    """Build G7 (network cell: sudden expansion -> symmetric Y bifurcation) in star units.

    Inlet (h1 = 1) expands symmetrically at x = 0 to width E over chamber length L_mid,
    then bifurcates symmetrically at half-angle theta_deg into two daughters of width
    h_d = 0.5 * E (area-preserving split of the expanded chamber) with outlet length
    L_out = L_out_ratio * h_d >= 10 * h_d.

    Parameters
    ----------
    params:
        E : float in [1.5, 4.0]
            Expansion ratio h2 / h1.
        theta_deg : float in [20.0, 90.0]
            Bifurcation half-angle [deg].
        L_in : float >= 5.0, optional (default 5.0)
        L_mid : float >= 2.0, optional (default max(4.0, 2.5 * E))
        L_out_ratio : float >= 10.0, optional (default 10.0)
    level : int in {0, 1, 2}
    """
    E = float(params["E"])
    theta_deg = float(params["theta_deg"])
    L_in = float(params.get("L_in", 5.0))
    L_mid = float(params.get("L_mid", max(4.0, 2.5 * E)))
    L_out_ratio = float(params.get("L_out_ratio", 10.0))

    if not (1.5 <= E <= 4.0):
        raise ValueError(f"G7 E must be in [1.5, 4.0], got {E}")
    if not (20.0 <= theta_deg <= 90.0):
        raise ValueError(f"G7 theta_deg must be in [20, 90], got {theta_deg}")
    if L_in < 5.0:
        raise ValueError(f"G7 L_in must be >= 5.0, got {L_in}")
    if L_mid < 2.0:
        raise ValueError(f"G7 L_mid must be >= 2.0, got {L_mid}")
    if L_out_ratio < 10.0:
        raise ValueError(f"G7 L_out_ratio must be >= 10.0, got {L_out_ratio}")

    s = 0.5 * (E - 1.0)
    h_d = 0.5 * E
    L_d = L_out_ratio * h_d

    theta = math.radians(theta_deg)
    ct = math.cos(theta)
    st = math.sin(theta)
    t1 = (ct, st)
    n1 = (-st, ct)
    t2 = (ct, -st)
    n2 = (st, ct)

    # Bifurcation corners at x_b = L_mid.
    # For h_d = 0.5 * E, apex sits at x_A = x_b + h_d * tan(theta/2).
    L_buf = 0.5 * h_d
    x_j = L_mid - L_buf  # start of miter entry blocks (still > 0 since L_mid >= max(4, 2.5*E) > 0.25*E)
    if x_j <= 0.5:
        raise ValueError(f"G7 L_mid={L_mid} too short for expansion chamber at E={E}")

    x_A = L_mid + h_d * math.tan(0.5 * theta)
    # Orthogonal daughter entrance cut at s_cut = 0.5 * h_d past outer corners T1, T2.
    s_cut = 0.5 * h_d

    T1 = (L_mid, 1.0 + s)
    T2 = (L_mid, -s)
    A = (x_A, 0.5)

    # Split fraction along half-channel width h_d = 0.5 * E:
    # inner band is width 0.5 (ratio f_in = 1/E), outer band is width s (ratio 1 - 1/E).
    f_in = 1.0 / E

    # Miter diagonal points (interpolated along A -> T1 and A -> T2):
    M_top = (A[0] + f_in * (T1[0] - A[0]), A[1] + f_in * (T1[1] - A[1]))
    M_bot = (A[0] + f_in * (T2[0] - A[0]), A[1] + f_in * (T2[1] - A[1]))

    # Orthogonal daughter entrance cross-sections (3 points each: inner, split, outer):
    O1 = (T1[0] + s_cut * t1[0], T1[1] + s_cut * t1[1])
    I1 = (O1[0] - h_d * n1[0], O1[1] - h_d * n1[1])
    S1 = (I1[0] + f_in * h_d * n1[0], I1[1] + f_in * h_d * n1[1])

    O2 = (T2[0] + s_cut * t2[0], T2[1] + s_cut * t2[1])
    I2 = (O2[0] + h_d * n2[0], O2[1] + h_d * n2[1])
    S2 = (I2[0] - f_in * h_d * n2[0], I2[1] - f_in * h_d * n2[1])

    # Daughter outlet cross-sections:
    I1_end = (I1[0] + L_d * t1[0], I1[1] + L_d * t1[1])
    S1_end = (S1[0] + L_d * t1[0], S1[1] + L_d * t1[1])
    O1_end = (O1[0] + L_d * t1[0], O1[1] + L_d * t1[1])

    O2_end = (O2[0] + L_d * t2[0], O2[1] + L_d * t2[1])
    S2_end = (S2[0] + L_d * t2[0], S2[1] + L_d * t2[1])
    I2_end = (I2[0] + L_d * t2[0], I2[1] + L_d * t2[1])

    # Vertices list:
    verts_2d: list[Point2D] = [
        # Inlet x = -L_in (0..2)
        (-L_in, 0.0),      # 0
        (-L_in, 0.5),      # 1
        (-L_in, 1.0),      # 2
        # Step x = 0 (3..7)
        (0.0, -s),         # 3
        (0.0, 0.0),        # 4
        (0.0, 0.5),        # 5
        (0.0, 1.0),        # 6
        (0.0, 1.0 + s),    # 7
        # Miter entry x = x_j (8..12)
        (x_j, -s),         # 8
        (x_j, 0.0),        # 9
        (x_j, 0.5),        # 10
        (x_j, 1.0),        # 11
        (x_j, 1.0 + s),    # 12
        # Miter diagonal (13..17): T2, M_bot, A, M_top, T1
        T2,                # 13
        M_bot,             # 14
        A,                 # 15
        M_top,             # 16
        T1,                # 17
        # Upper daughter entrance (18..20): I1, S1, O1
        I1,                # 18
        S1,                # 19
        O1,                # 20
        # Lower daughter entrance (21..23): O2, S2, I2
        O2,                # 21
        S2,                # 22
        I2,                # 23
        # Upper daughter outlet (24..26): I1_end, S1_end, O1_end
        I1_end,            # 24
        S1_end,            # 25
        O1_end,            # 26
        # Lower daughter outlet (27..29): O2_end, S2_end, I2_end
        O2_end,            # 27
        S2_end,            # 28
        I2_end,            # 29
    ]

    ny_in_half = _scale_cells(8, level)
    ny_step = _scale_cells(_base_cells_for_length(s, 12.0, minimum=4), level)
    nx_in = _scale_cells(_base_cells_for_length(L_in, 8.0, minimum=16), level)
    nx_mid = _scale_cells(_base_cells_for_length(x_j, 6.0, minimum=16), level)
    nx_j_in = _scale_cells(_base_cells_for_length(x_A - x_j, 8.0, minimum=6), level)
    nx_j_out = _scale_cells(_base_cells_for_length(s_cut, 8.0, minimum=6), level)
    nx_d = _scale_cells(_base_cells_for_length(L_d, 5.0, minimum=36), level)

    blocks_2d = [
        # Inlet (2 bands)
        Block2D((0, 4, 5, 1), nx_in, ny_in_half),
        Block2D((1, 5, 6, 2), nx_in, ny_in_half),
        # Expansion chamber (4 bands)
        Block2D((3, 8, 9, 4), nx_mid, ny_step),
        Block2D((4, 9, 10, 5), nx_mid, ny_in_half),
        Block2D((5, 10, 11, 6), nx_mid, ny_in_half),
        Block2D((6, 11, 12, 7), nx_mid, ny_step),
        # Miter entry (4 bands)
        Block2D((8, 13, 14, 9), nx_j_in, ny_step),
        Block2D((9, 14, 15, 10), nx_j_in, ny_in_half),
        Block2D((10, 15, 16, 11), nx_j_in, ny_in_half),
        Block2D((11, 16, 17, 12), nx_j_in, ny_step),
        # Miter exit upper (2 bands)
        Block2D((15, 18, 19, 16), nx_j_out, ny_in_half),
        Block2D((16, 19, 20, 17), nx_j_out, ny_step),
        # Miter exit lower (2 bands)
        Block2D((13, 21, 22, 14), nx_j_out, ny_step),
        Block2D((14, 22, 23, 15), nx_j_out, ny_in_half),
        # Upper daughter (2 bands)
        Block2D((18, 24, 25, 19), nx_d, ny_in_half),
        Block2D((19, 25, 26, 20), nx_d, ny_step),
        # Lower daughter (2 bands)
        Block2D((21, 27, 28, 22), nx_d, ny_step),
        Block2D((22, 28, 29, 23), nx_d, ny_in_half),
    ]

    patch_edges = [
        ("inlet", "patch", [(1, 0), (2, 1)]),
        ("outlet1", "patch", [(24, 25), (25, 26)]),
        ("outlet2", "patch", [(27, 28), (28, 29)]),
        (
            "walls",
            "wall",
            [
                # Lower outer wall
                (0, 4), (4, 3), (3, 8), (8, 13), (13, 21), (21, 27),
                # Inner carina walls
                (29, 23), (23, 15), (15, 18), (18, 24),
                # Upper outer wall
                (26, 20), (20, 17), (17, 12), (12, 7), (7, 6), (6, 2),
            ],
        ),
    ]

    d_mid = 0.5 * L_d
    d1_p0 = (I1[0] + d_mid * t1[0], I1[1] + d_mid * t1[1])
    d1_p1 = (O1[0] + d_mid * t1[0], O1[1] + d_mid * t1[1])
    d2_p0 = (O2[0] + d_mid * t2[0], O2[1] + d_mid * t2[1])
    d2_p1 = (I2[0] + d_mid * t2[0], I2[1] + d_mid * t2[1])

    sample_lines = [
        SampleLine("inletProfile", "y", *_line_pts(-0.5, 0.0, -0.5, 1.0), n_points=80),
        SampleLine(
            "expansionMid",
            "y",
            *_line_pts(0.5 * x_j, -s, 0.5 * x_j, 1.0 + s),
            n_points=100,
        ),
        SampleLine(
            "nearWallExpansion",
            "x",
            (0.02, -s + 0.02, 0.5),
            (x_j - 0.02, -s + 0.02, 0.5),
            n_points=120,
        ),
        SampleLine(
            "daughter1Mid",
            "distance",
            *_line_pts(d1_p0[0], d1_p0[1], d1_p1[0], d1_p1[1]),
            n_points=80,
        ),
        SampleLine(
            "daughter2Mid",
            "distance",
            *_line_pts(d2_p0[0], d2_p0[1], d2_p1[0], d2_p1[1]),
            n_points=80,
        ),
    ]

    return _extrude_2d_mesh(
        family="G7",
        level=level,
        params={
            "E": E,
            "theta_deg": theta_deg,
            "L_in": L_in,
            "L_mid": L_mid,
            "L_out_ratio": L_out_ratio,
        },
        verts_2d=verts_2d,
        blocks_2d=blocks_2d,
        patch_edges=patch_edges,
        sample_lines=sample_lines,
        inlet_height=1.0,
        outlet_heights=(h_d, h_d),
        entry_length=L_in,
        outlet_lengths=(L_d, L_d),
    )


def build_kg(params: Mapping[str, Any], level: int = 1) -> MeshSpec:
    """Build KG (Karino & Goldsmith 1977 via Wu et al. 2015 §3.1) axisymmetric wedge mesh.

    Tubular sudden expansion in star units with length scale = inlet diameter d1 = 2 * R1
    (so inlet radius r1* = 0.5, outlet radius r2* = 0.5 * E, diameter ratio E = D2 / d1 = R2 / R1).

    Primary-source geometry (Wu et al. 2015, IJES 95:49-59, PMC4521229 Fig. 2 caption):
      - Inlet radius R1 = 0.0755 mm (d1 = 151 um), outlet radius R2 = 0.252 mm (D2 = 504 um)
        => E = 252 / 75.5 = 3.337748... (task prompt / knowledge/12 row 15 noted 77.5->252 um,
        E = 3.25; E is parameterized so both 3.337748 and 3.25 can be run).
      - Upstream length 1 mm -> L_in* = 1000 / 151 = 6.622517 (>= 5 d1).
      - Downstream length 5 mm (per T2 prompt; Wu 2015 Fig. 2 states 2 mm = 13.245 d1)
        -> L_out* = 5000 / 151 = 33.112583.
      - OpenFOAM axisymmetric wedge spanning [-wedge_angle/2, +wedge_angle/2] around the x-axis
        (default wedge_angle_deg = 5.0 deg), with ``wedgeFront``/``wedgeBack`` (type ``wedge``)
        and collapsed ``axis`` (type ``empty``).

    Parameters
    ----------
    params : Mapping[str, Any]
        Optional keys:
        - ``E`` : float in [1.5, 5.0], default 252.0 / 75.5 (~3.337748)
        - ``L_in`` : float >= 5.0, default 1000.0 / 151.0 (~6.622517)
        - ``L_out`` : float >= 10.0, default 5000.0 / 151.0 (~33.112583)
        - ``wedge_angle_deg`` : float in (0.5, 5.0], default 5.0
    level : int in {0, 1, 2}
        Mesh refinement level (r = 1.5 per level).
    """
    E = float(params.get("E", 252.0 / 75.5))
    L_in = float(params.get("L_in", 1000.0 / 151.0))
    L_out = float(params.get("L_out", 5000.0 / 151.0))
    wedge_angle_deg = float(params.get("wedge_angle_deg", 5.0))

    if not (1.5 <= E <= 5.0):
        raise ValueError(f"KG E must be in [1.5, 5.0], got {E}")
    if L_in < 5.0:
        raise ValueError(f"KG L_in must be >= 5.0, got {L_in}")
    if L_out < 10.0:
        raise ValueError(f"KG L_out must be >= 10.0, got {L_out}")
    if not (0.5 <= wedge_angle_deg <= 5.0):
        raise ValueError(f"KG wedge_angle_deg must be in [0.5, 5.0], got {wedge_angle_deg}")

    r1 = 0.5
    r2 = 0.5 * E
    half_angle = math.radians(0.5 * wedge_angle_deg)
    ca = math.cos(half_angle)
    sa = math.sin(half_angle)

    # 2D meridian plane (x*, r*) representation for fixed-grid resampling and masks.
    verts_2d: tuple[Point2D, ...] = (
        (-L_in, 0.0),   # 0
        (0.0, 0.0),     # 1
        (L_out, 0.0),   # 2
        (-L_in, r1),    # 3
        (0.0, r1),      # 4
        (L_out, r1),    # 5
        (0.0, r2),      # 6
        (L_out, r2),    # 7
    )

    nr_in = _scale_cells(16, level)
    nr_step = _scale_cells(_base_cells_for_length(r2 - r1, 24.0, minimum=12), level)
    nx_in = _scale_cells(_base_cells_for_length(L_in, 8.0, minimum=16), level)
    nx_out = _scale_cells(_base_cells_for_length(L_out, 8.0, minimum=64), level)

    blocks_2d: tuple[Block2D, ...] = (
        Block2D((0, 1, 4, 3), nx_in, nr_in),
        Block2D((1, 2, 5, 4), nx_out, nr_in),
        Block2D((4, 5, 7, 6), nx_out, nr_step),
    )

    # 3D wedge vertices: axis at (x, 0, 0); wedgeBack at z = -r*sin(a); wedgeFront at z = +r*sin(a).
    verts_3d: tuple[Point3D, ...] = (
        (-L_in, 0.0, 0.0),          # 0: axis inlet
        (0.0, 0.0, 0.0),            # 1: axis step
        (L_out, 0.0, 0.0),          # 2: axis outlet
        (-L_in, r1 * ca, -r1 * sa), # 3: r1 inlet back
        (0.0, r1 * ca, -r1 * sa),   # 4: r1 step back
        (L_out, r1 * ca, -r1 * sa), # 5: r1 outlet back
        (-L_in, r1 * ca, +r1 * sa), # 6: r1 inlet front
        (0.0, r1 * ca, +r1 * sa),   # 7: r1 step front
        (L_out, r1 * ca, +r1 * sa), # 8: r1 outlet front
        (0.0, r2 * ca, -r2 * sa),   # 9: r2 step back
        (L_out, r2 * ca, -r2 * sa), # 10: r2 outlet back
        (0.0, r2 * ca, +r2 * sa),   # 11: r2 step front
        (L_out, r2 * ca, +r2 * sa), # 12: r2 outlet front
    )

    blocks_3d: tuple[tuple[Hex3D, tuple[int, int, int], tuple[float, float, float]], ...] = (
        ((0, 1, 4, 3, 0, 1, 7, 6), (nx_in, nr_in, 1), (1.0, 1.0, 1.0)),
        ((1, 2, 5, 4, 1, 2, 8, 7), (nx_out, nr_in, 1), (1.0, 1.0, 1.0)),
        ((4, 5, 10, 9, 7, 8, 12, 11), (nx_out, nr_step, 1), (1.0, 1.0, 1.0)),
    )

    patches: tuple[PatchSpec, ...] = (
        PatchSpec("inlet", "patch", ((3, 0, 0, 6),)),
        PatchSpec("outlet", "patch", ((2, 5, 8, 2), (5, 10, 12, 8))),
        PatchSpec("walls", "wall", ((4, 3, 6, 7), (9, 4, 7, 11), (10, 9, 11, 12))),
        PatchSpec("axis", "empty", ((0, 1, 1, 0), (1, 2, 2, 1))),
        PatchSpec("wedgeBack", "wedge", ((0, 3, 4, 1), (1, 4, 5, 2), (4, 9, 10, 5))),
        PatchSpec("wedgeFront", "wedge", ((0, 1, 7, 6), (1, 2, 8, 7), (7, 8, 12, 11))),
    )

    # Sample lines on symmetry midplane z = 0 (strictly inside the wedge for y > 0).
    y_r1 = r1 * ca
    y_r2 = r2 * ca
    eps_r = 0.005
    near_wall_y = y_r2 - 0.015
    sample_lines: tuple[SampleLine, ...] = (
        SampleLine(
            "inletProfile",
            "y",
            (-0.5, eps_r, 0.0),
            (-0.5, y_r1 - eps_r, 0.0),
            n_points=80,
        ),
        SampleLine(
            "nearWallOuter",
            "x",
            (0.02, near_wall_y, 0.0),
            (L_out - 0.02, near_wall_y, 0.0),
            n_points=400,
        ),
        SampleLine(
            "centerline",
            "x",
            (-L_in + 0.02, eps_r, 0.0),
            (L_out - 0.02, eps_r, 0.0),
            n_points=200,
        ),
        SampleLine(
            "outletProfile",
            "y",
            (L_out - 1.0, eps_r, 0.0),
            (L_out - 1.0, y_r2 - eps_r, 0.0),
            n_points=100,
        ),
    )

    return MeshSpec(
        family="KG",
        level=level,
        params={
            "E": E,
            "L_in": L_in,
            "L_out": L_out,
            "wedge_angle_deg": wedge_angle_deg,
        },
        vertices_2d=verts_2d,
        blocks_2d=blocks_2d,
        vertices=verts_3d,
        blocks=blocks_3d,
        arcs=(),
        patches=patches,
        sample_lines=sample_lines,
        inlet_height=1.0,          # inlet diameter d1* = 1.0 (radius = 0.5)
        outlet_heights=(E,),       # outlet diameter D2* = E (radius = 0.5 * E)
        entry_length=L_in,
        outlet_lengths=(L_out,),
    )


def build_geometry(family: str, params: Mapping[str, Any], level: int = 1) -> MeshSpec:
    """Dispatch geometry construction by family ID in {'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'KG'}."""
    builders = {
        "G2": build_g2,
        "G3": build_g3,
        "G4": build_g4,
        "G5": build_g5,
        "G6": build_g6,
        "G7": build_g7,
        "KG": build_kg,
    }
    key = family.upper()
    if key not in builders:
        raise ValueError(f"Unsupported geometry family '{family}'; expected one of {sorted(builders)}")
    return builders[key](params, level=level)


def _fmt_float(val: float) -> str:
    """Format a float cleanly for OpenFOAM ASCII dictionaries."""
    if abs(val) < 1e-13:
        return "0"
    return f"{val:.8g}"


def format_block_mesh_dict(mesh: MeshSpec) -> str:
    """Render a MeshSpec into an OpenFOAM v2306 system/blockMeshDict string."""
    lines: list[str] = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       dictionary;",
        "    object      blockMeshDict;",
        "}",
        "",
        "scale 1;",
        "",
        "vertices",
        "(",
    ]
    for idx, (x, y, z) in enumerate(mesh.vertices):
        lines.append(f"    ({_fmt_float(x)} {_fmt_float(y)} {_fmt_float(z)})  // {idx}")
    lines.extend([");", "", "blocks", "("])

    for hex_verts, (nx, ny, nz), (gx, gy, gz) in mesh.blocks:
        v_str = " ".join(str(v) for v in hex_verts)
        lines.append(
            f"    hex ({v_str}) ({nx} {ny} {nz}) simpleGrading "
            f"({_fmt_float(gx)} {_fmt_float(gy)} {_fmt_float(gz)})"
        )
    lines.extend([");", "", "edges", "("])
    for u, v, (mx, my, mz) in mesh.arcs:
        lines.append(f"    arc {u} {v} ({_fmt_float(mx)} {_fmt_float(my)} {_fmt_float(mz)})")
    lines.extend([");", "", "boundary", "("])

    for patch in mesh.patches:
        lines.append(f"    {patch.name}")
        lines.append("    {")
        lines.append(f"        type {patch.patch_type};")
        lines.append("        faces")
        lines.append("        (")
        for face in patch.faces:
            f_str = " ".join(str(v) for v in face)
            lines.append(f"            ({f_str})")
        lines.append("        );")
        lines.append("    }")
    lines.extend([");", "", "mergePatchPairs", "(", ");", ""])
    return "\n".join(lines)
