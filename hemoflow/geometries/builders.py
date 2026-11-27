"""hemoflow geometry builders: exact 2D microfluidic families G1-G13 + KG (star units, h1* = 1).

Ported verbatim (release-guard: 2D/2.5D only) from the rbc-shear-surrogate reference
``cfd/gen/geometry.py`` and ``cfd/gen/m2b.py`` (``build_g1``).  All lengths in star units
with inlet height h1* = 1.0; inlet on x = x_in with y in [0, 1] (centerline y = 0.5), so
the developed Newtonian parabolic profile u*(y*) = 6 y* (1 - y*) applies across families.

Pure Python (math/dataclasses only) -> importable on any host without dolfinx/ufl.
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


def _read_fillet_r(params: Mapping[str, Any], family: str) -> float:
    """T26 Option 1 (opt-in corner fillets): read ``fillet_r`` from params.

    ``fillet_r = 0.0`` (default) keeps the geometry 100% identical to the unfilleted
    one.  Only G3/G4/G5 support fillets; other families raise if ``fillet_r > 0``.
    """
    fillet_r = float(params.get("fillet_r", 0.0))
    if fillet_r < 0.0:
        raise ValueError(f"{family} fillet_r must be >= 0, got {fillet_r}")
    if fillet_r > 0.0 and family not in ("G3", "G4", "G5"):
        raise ValueError(f"{family}: fillet_r > 0 is only supported for G3/G4/G5, got {fillet_r}")
    return fillet_r


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
    _read_fillet_r(params, "G2")  # raises if fillet_r > 0 (unsupported family)

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
    fillet_r = _read_fillet_r(params, "G3")  # T26 Option 1: opt-in, 0 = unfilleted (identical)

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
            "fillet_r": fillet_r,
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
            "fillet_r": _read_fillet_r(params, "G4"),  # T26 Option 1: opt-in, 0 = unfilleted
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
            "fillet_r": _read_fillet_r(params, "G5"),  # T26 Option 1: opt-in, 0 = unfilleted
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
    _read_fillet_r(params, "G6")  # raises if fillet_r > 0 (unsupported family)

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
    _read_fillet_r(params, "G7")  # raises if fillet_r > 0 (unsupported family)

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
    _read_fillet_r(params, "KG")  # raises if fillet_r > 0 (unsupported family)

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


# ---------------------------------------------------------------------------
# T36: families G8-G13 (plan/tasks/T36_mistral_g8_g13_corpus.md)
#
# Shared construction: a "strip mesh" is a sequence of vertical stations; every
# station carries a list of y-levels (bottom -> top) and consecutive stations are
# connected by one quad block per fluid band.  Walls are finely segmented smooth
# polylines (C1 cosine ramps / stadium caps), so every block stays strictly convex
# CCW and the union-of-quads fluid mask is exact.  Internal islands (G11, G13)
# split the bands; their wall edges go to a separate ``wall``-type patch
# ``wallsIsland`` (tagged WALL by the FEniCSx adapter).
# ---------------------------------------------------------------------------


def _ss(u: float) -> float:
    """C1 cosine easing on [0, 1]: 0 -> 1 with zero slope at both ends."""
    u = min(1.0, max(0.0, u))
    return 0.5 - 0.5 * math.cos(math.pi * u)


def _lerp(a: float, b: float, f: float) -> float:
    return (1.0 - f) * a + f * b


class _StripMesh:
    """Incremental vertical-strip 2D block mesh builder.

    Stations are added with :meth:`station`; each subsequent :meth:`strip` call
    connects the last two stations with one convex CCW quad block per band.
    """

    def __init__(self) -> None:
        self.verts: list[Point2D] = []
        self.blocks: list[Block2D] = []
        self.stations: list[list[int]] = []
        self.xs: list[float] = []
        self.bottom_wall: list[Edge2D] = []  # outer bottom wall, left -> right
        self.top_wall: list[Edge2D] = []     # outer top wall, right -> left
        self.island_wall: list[Edge2D] = []  # internal island wall edges
        self.inlet_edges: list[Edge2D] = []
        self.outlet_edges: dict[str, list[Edge2D]] = {}

    def station(self, x: float, levels: list[float]) -> int:
        """Add a station at ``x`` with y-levels bottom -> top; returns its index.

        ``x`` need not be globally increasing (G13 branches revisit x-ranges);
        each strip is connected explicitly via :meth:`strip_between`.
        """
        ids: list[int] = []
        for y in levels:
            self.verts.append((x, y))
            ids.append(len(self.verts) - 1)
        self.stations.append(ids)
        self.xs.append(x)
        return len(self.stations) - 1

    def station_pts(self, pts: list[Point2D]) -> int:
        """Add a station from explicit 2D points (bottom -> top; may be slanted)."""
        ids: list[int] = []
        for (x, y) in pts:
            self.verts.append((x, y))
            ids.append(len(self.verts) - 1)
        self.stations.append(ids)
        self.xs.append(pts[0][0])
        return len(self.stations) - 1

    def finish_inlet(self) -> None:
        """Collect the inlet patch edges of station 0 (top -> bottom per band)."""
        s0 = self.stations[0]
        self.inlet_edges = [(s0[k + 1], s0[k]) for k in range(len(s0) - 1)]

    def strip(
        self,
        bands: list[tuple[int, int, int, int]],
        nx: int,
        nys: list[int],
        island: bool = False,
        outlet: str | None = None,
    ) -> None:
        """Connect the last two stations (convenience wrapper of :meth:`strip_between`)."""
        if len(self.stations) < 2:
            raise ValueError("need at least two stations before strip()")
        self.strip_between(len(self.stations) - 2, len(self.stations) - 1, bands, nx, nys,
                           island=island, outlet=outlet)

    def strip_between(
        self,
        i: int,
        j: int,
        bands: list[tuple[int, int, int, int]],
        nx: int,
        nys: list[int],
        island: bool = False,
        outlet: str | None = None,
    ) -> None:
        """Connect station ``i`` (left) to station ``j`` (right) with one quad per band.

        ``bands`` entries are ``(i0, i1, j0, j1)`` level indices into the left
        (i*) and right (j*) station; block = (s_left[i0], s_right[j0], s_right[j1],
        s_left[i1]).  ``nys`` is the per-band ny (aligned with ``bands``).
        ``island=True`` marks the strip as island-carved: all inner band edges
        (band b top for b < nb-1, band b bottom for b > 0) are island walls.
        ``outlet`` names the outlet patch for the right-side outer edges.
        """
        if len(bands) != len(nys):
            raise ValueError("bands and nys must be aligned")
        s0 = self.stations[i]
        s1 = self.stations[j]
        nb = len(bands)
        for b, (i0, i1, j0, j1) in enumerate(bands):
            self.blocks.append(Block2D((s0[i0], s1[j0], s1[j1], s0[i1]), nx, nys[b]))
        i0, _i1, j0, _j1 = bands[0]
        self.bottom_wall.append((s0[i0], s1[j0]))
        _i0, i1, _j0, j1 = bands[-1]
        self.top_wall.append((s1[j1], s0[i1]))
        if island:
            for b in range(nb - 1):  # band b top edge (right -> left)
                _i0, i1, _j0, j1 = bands[b]
                self.island_wall.append((s1[j1], s0[i1]))
            for b in range(1, nb):  # band b bottom edge (left -> right)
                i0, _i1, j0, _j1 = bands[b]
                self.island_wall.append((s0[i0], s1[j0]))
        if outlet is not None:
            for (_i0, _i1, j0, j1) in bands:
                self.outlet_edges.setdefault(outlet, []).append((s1[j0], s1[j1]))

    def patch_edges(self) -> list[tuple[str, str, list[Edge2D]]]:
        """Assemble the patch edge list (inlet / outlets / walls / wallsIsland)."""
        pe: list[tuple[str, str, list[Edge2D]]] = [("inlet", "patch", list(self.inlet_edges))]
        for name in sorted(self.outlet_edges):
            pe.append((name, "patch", list(self.outlet_edges[name])))
        pe.append(("walls", "wall", self.bottom_wall + self.top_wall))
        if self.island_wall:
            pe.append(("wallsIsland", "wall", list(self.island_wall)))
        return pe


def _strip_nx(x0: float, x1: float, cells_per_unit: float, level: int) -> int:
    return _scale_cells(_base_cells_for_length(x1 - x0, cells_per_unit, minimum=4), level)


def _island_half_thickness(x: float, x0: float, x1: float, r: float, L_nose: float) -> float:
    """C1 cosine-tapered island half-thickness: 0 at the tips, ``r`` in the straight body.

    The nose/tail are cosine tapers of length ``L_nose`` (max wall slope
    r*pi/(2*L_nose) ~ 0.31 for r = 0.10, L_nose = 0.5), so the polyline wall is smooth
    (no sharp apex) and block cells stay well-conditioned (unlike a semicircular cap,
    whose tip strip is a near-vertical sliver that fails checkMesh skewness).
    """
    if x <= x0 or x >= x1:
        return 0.0
    if x <= x0 + L_nose:
        return r * _ss((x - x0) / L_nose)
    if x >= x1 - L_nose:
        return r * (1.0 - _ss((x - (x1 - L_nose)) / L_nose))
    return r


def build_g8(params: Mapping[str, Any], level: int = 1) -> MeshSpec:
    """Build G8 (smooth Venturi / extensional nozzle, 1->1) in star units (h1 = 1).

    Symmetric C1 cosine converging-diverging wall: half-width w(x) eases from 0.5
    at x = 0 to 0.5*throat_ratio at x = L_conv, back to 0.5 at x = L_conv + L_div,
    then a straight outlet of height 1.

    Parameters
    ----------
    params:
        throat_ratio : float in [0.50, 0.85]  (throat height h_t / h1; w_min/h1 >= 0.50)
        L_conv : float in [1.5, 4.0]
        L_div : float in [2.5, 7.0]
        L_in : float >= 5.0, optional (default 5.0)
        L_out_ratio : float >= 10.0, optional (default 10.0)
    level : int in {0, 1, 2, 3}
    """
    throat_ratio = float(params["throat_ratio"])
    L_conv = float(params["L_conv"])
    L_div = float(params["L_div"])
    L_in = float(params.get("L_in", 5.0))
    L_out_ratio = float(params.get("L_out_ratio", 10.0))
    if not (0.50 <= throat_ratio <= 0.85):
        raise ValueError(f"G8 throat_ratio must be in [0.50, 0.85], got {throat_ratio}")
    if not (1.5 <= L_conv <= 4.0):
        raise ValueError(f"G8 L_conv must be in [1.5, 4.0], got {L_conv}")
    if not (2.5 <= L_div <= 7.0):
        raise ValueError(f"G8 L_div must be in [2.5, 7.0], got {L_div}")
    if L_in < 5.0:
        raise ValueError(f"G8 L_in must be >= 5.0, got {L_in}")
    if L_out_ratio < 10.0:
        raise ValueError(f"G8 L_out_ratio must be >= 10.0, got {L_out_ratio}")
    _read_fillet_r(params, "G8")

    L_out = L_out_ratio * 1.0
    x_div_end = L_conv + L_div
    x_end = x_div_end + L_out

    def half_w(x: float) -> float:
        if x <= 0.0:
            return 0.5
        if x <= L_conv:
            return 0.5 * (1.0 + (throat_ratio - 1.0) * _ss(x / L_conv))
        if x <= x_div_end:
            return 0.5 * (1.0 + (throat_ratio - 1.0) * (1.0 - _ss((x - L_conv) / L_div)))
        return 0.5

    nseg = 8
    xs = [-L_in, 0.0]
    xs += [L_conv * (k + 1) / nseg for k in range(nseg)]
    xs += [L_conv + L_div * (k + 1) / nseg for k in range(nseg)]
    xs += [x_div_end + L_out * (k + 1) / 4.0 for k in range(4)]

    sm = _StripMesh()
    for x in xs:
        w = half_w(x)
        sm.station(x, [0.5 - w, 0.5 + w])
    sm.finish_inlet()
    ny = _scale_cells(_base_cells_for_length(1.0, 12.0, minimum=8), level)
    for i in range(len(xs) - 1):
        sm.strip_between(i, i + 1, [(0, 1, 0, 1)], _strip_nx(xs[i], xs[i + 1], 6.0, level), [ny],
                         outlet=("outlet" if i == len(xs) - 2 else None))

    y_throat_lo = 0.5 - 0.5 * throat_ratio
    y_throat_hi = 0.5 + 0.5 * throat_ratio
    sample_lines = [
        SampleLine("inletProfile", "y", *_line_pts(-0.5, 0.0, -0.5, 1.0), n_points=80),
        SampleLine("throatMid", "y", *_line_pts(L_conv, y_throat_lo, L_conv, y_throat_hi), n_points=80),
        SampleLine("centerline", "x", (-L_in + 0.02, 0.5, 0.5), (x_end - 0.02, 0.5, 0.5), n_points=200),
        SampleLine("outletProfile", "y", *_line_pts(x_end - 1.0, 0.0, x_end - 1.0, 1.0), n_points=80),
    ]
    return _extrude_2d_mesh(
        family="G8",
        level=level,
        params={"throat_ratio": throat_ratio, "L_conv": L_conv, "L_div": L_div,
                "L_in": L_in, "L_out_ratio": L_out_ratio},
        verts_2d=sm.verts,
        blocks_2d=sm.blocks,
        patch_edges=sm.patch_edges(),
        sample_lines=sample_lines,
        inlet_height=1.0,
        outlet_heights=(1.0,),
        entry_length=L_in,
        outlet_lengths=(L_out,),
    )


def build_g9(params: Mapping[str, Any], level: int = 1) -> MeshSpec:
    """Build G9 (multi-stage pinch-relaxation array, 1->1) in star units (h1 = 1).

    ``n_stages`` periodic cosine pinches (throat_ratio) separated by relaxation
    cavities (cavity_ratio), with a final cosine relaxation back to h = 1 before
    the straight outlet.

    Parameters
    ----------
    params:
        throat_ratio : float in [0.55, 0.85]  (pinch height / h1; w_min/h1 >= 0.55)
        cavity_ratio : float in [1.15, 1.75] (cavity height / h1)
        n_stages : int in {2, 3, 4}
        L_pinch : float >= 1.0, optional (default 1.5)
        L_cavity : float >= 1.0, optional (default 2.0)
        L_in : float >= 5.0, optional (default 5.0)
        L_out_ratio : float >= 10.0, optional (default 10.0)
    level : int in {0, 1, 2, 3}
    """
    throat_ratio = float(params["throat_ratio"])
    cavity_ratio = float(params["cavity_ratio"])
    n_stages = int(params["n_stages"])
    L_pinch = float(params.get("L_pinch", 1.5))
    L_cavity = float(params.get("L_cavity", 2.0))
    L_in = float(params.get("L_in", 5.0))
    L_out_ratio = float(params.get("L_out_ratio", 10.0))
    if not (0.55 <= throat_ratio <= 0.85):
        raise ValueError(f"G9 throat_ratio must be in [0.55, 0.85], got {throat_ratio}")
    if not (1.15 <= cavity_ratio <= 1.75):
        raise ValueError(f"G9 cavity_ratio must be in [1.15, 1.75], got {cavity_ratio}")
    if n_stages not in (2, 3, 4):
        raise ValueError(f"G9 n_stages must be in {{2, 3, 4}}, got {n_stages}")
    if L_pinch < 1.0 or L_cavity < 1.0:
        raise ValueError(f"G9 L_pinch/L_cavity must be >= 1.0, got {L_pinch}/{L_cavity}")
    if L_in < 5.0:
        raise ValueError(f"G9 L_in must be >= 5.0, got {L_in}")
    if L_out_ratio < 10.0:
        raise ValueError(f"G9 L_out_ratio must be >= 10.0, got {L_out_ratio}")
    _read_fillet_r(params, "G9")

    L_relax = 1.5
    L_out = L_out_ratio * 1.0
    x_end = n_stages * (L_pinch + L_cavity) + L_relax + L_out

    def half_w(x: float) -> float:
        if x <= 0.0:
            return 0.5
        for s in range(n_stages):
            x0 = s * (L_pinch + L_cavity)
            if x <= x0 + L_pinch:
                return 0.5 * (1.0 + (throat_ratio - 1.0) * _ss((x - x0) / L_pinch))
            if x <= x0 + L_pinch + L_cavity:
                return 0.5 * (throat_ratio + (cavity_ratio - throat_ratio) * _ss((x - x0 - L_pinch) / L_cavity))
        if x <= n_stages * (L_pinch + L_cavity) + L_relax:
            return 0.5 * (cavity_ratio + (1.0 - cavity_ratio) * _ss((x - n_stages * (L_pinch + L_cavity)) / L_relax))
        return 0.5

    nseg = 6
    xs = [-L_in, 0.0]
    for s in range(n_stages):
        x0 = s * (L_pinch + L_cavity)
        xs += [x0 + L_pinch * (k + 1) / nseg for k in range(nseg)]
        xs += [x0 + L_pinch + L_cavity * (k + 1) / nseg for k in range(nseg)]
    xs += [n_stages * (L_pinch + L_cavity) + L_relax * (k + 1) / nseg for k in range(nseg)]
    xs += [n_stages * (L_pinch + L_cavity) + L_relax + L_out * (k + 1) / 4.0 for k in range(4)]

    sm = _StripMesh()
    for x in xs:
        w = half_w(x)
        sm.station(x, [0.5 - w, 0.5 + w])
    sm.finish_inlet()
    ny = _scale_cells(_base_cells_for_length(1.0, 12.0, minimum=8), level)
    for i in range(len(xs) - 1):
        sm.strip_between(i, i + 1, [(0, 1, 0, 1)], _strip_nx(xs[i], xs[i + 1], 6.0, level), [ny],
                         outlet=("outlet" if i == len(xs) - 2 else None))

    x_pinch1 = 0.5 * L_pinch
    x_cav1 = L_pinch + 0.5 * L_cavity
    sample_lines = [
        SampleLine("inletProfile", "y", *_line_pts(-0.5, 0.0, -0.5, 1.0), n_points=80),
        SampleLine("pinchMid", "y", *_line_pts(x_pinch1, 0.5 - 0.5 * throat_ratio, x_pinch1, 0.5 + 0.5 * throat_ratio), n_points=80),
        SampleLine("cavityMid", "y", *_line_pts(x_cav1, 0.5 - 0.5 * cavity_ratio, x_cav1, 0.5 + 0.5 * cavity_ratio), n_points=100),
        SampleLine("centerline", "x", (-L_in + 0.02, 0.5, 0.5), (x_end - 0.02, 0.5, 0.5), n_points=200),
        SampleLine("outletProfile", "y", *_line_pts(x_end - 1.0, 0.0, x_end - 1.0, 1.0), n_points=80),
    ]
    return _extrude_2d_mesh(
        family="G9",
        level=level,
        params={"throat_ratio": throat_ratio, "cavity_ratio": cavity_ratio, "n_stages": n_stages,
                "L_pinch": L_pinch, "L_cavity": L_cavity, "L_in": L_in, "L_out_ratio": L_out_ratio},
        verts_2d=sm.verts,
        blocks_2d=sm.blocks,
        patch_edges=sm.patch_edges(),
        sample_lines=sample_lines,
        inlet_height=1.0,
        outlet_heights=(1.0,),
        entry_length=L_in,
        outlet_lengths=(L_out,),
    )


def build_g10(params: Mapping[str, Any], level: int = 1) -> MeshSpec:
    """Build G10 (one-sided vortex pocket, 1->1) in star units (h1 = 1).

    Flat bottom wall y* = 0; the top wall eases (C1 cosine ramp over L_ramp = 2.0)
    up to pocket_ratio, holds for L_pocket, eases back to 1, then a straight
    outlet of height 1.

    Parameters
    ----------
    params:
        pocket_ratio : float in [1.4, 2.8]  (pocket height / h1)
        L_pocket : float in [2.5, 7.0]
        L_ramp : float >= 1.0, optional (default 2.0)
        L_in : float >= 5.0, optional (default 5.0)
        L_out_ratio : float >= 10.0, optional (default 10.0)
    level : int in {0, 1, 2, 3}
    """
    pocket_ratio = float(params["pocket_ratio"])
    L_pocket = float(params["L_pocket"])
    L_ramp = float(params.get("L_ramp", 2.0))
    L_in = float(params.get("L_in", 5.0))
    L_out_ratio = float(params.get("L_out_ratio", 10.0))
    if not (1.4 <= pocket_ratio <= 2.8):
        raise ValueError(f"G10 pocket_ratio must be in [1.4, 2.8], got {pocket_ratio}")
    if not (2.5 <= L_pocket <= 7.0):
        raise ValueError(f"G10 L_pocket must be in [2.5, 7.0], got {L_pocket}")
    if L_ramp < 1.0:
        raise ValueError(f"G10 L_ramp must be >= 1.0, got {L_ramp}")
    if L_in < 5.0:
        raise ValueError(f"G10 L_in must be >= 5.0, got {L_in}")
    if L_out_ratio < 10.0:
        raise ValueError(f"G10 L_out_ratio must be >= 10.0, got {L_out_ratio}")
    _read_fillet_r(params, "G10")

    L_out = L_out_ratio * 1.0
    x_pocket_end = L_ramp + L_pocket
    x_ramp_end = x_pocket_end + L_ramp
    x_end = x_ramp_end + L_out

    def y_top(x: float) -> float:
        if x <= 0.0:
            return 1.0
        if x <= L_ramp:
            return 1.0 + (pocket_ratio - 1.0) * _ss(x / L_ramp)
        if x <= x_pocket_end:
            return pocket_ratio
        if x <= x_ramp_end:
            return pocket_ratio - (pocket_ratio - 1.0) * _ss((x - x_pocket_end) / L_ramp)
        return 1.0

    nseg = 8
    xs = [-L_in, 0.0]
    xs += [L_ramp * (k + 1) / nseg for k in range(nseg)]
    xs += [L_ramp + L_pocket * (k + 1) / 2.0 for k in range(2)]
    xs += [x_pocket_end + L_ramp * (k + 1) / nseg for k in range(nseg)]
    xs += [x_ramp_end + L_out * (k + 1) / 4.0 for k in range(4)]

    sm = _StripMesh()
    for x in xs:
        sm.station(x, [0.0, y_top(x)])
    sm.finish_inlet()
    ny = _scale_cells(_base_cells_for_length(1.0, 12.0, minimum=8), level)
    for i in range(len(xs) - 1):
        sm.strip_between(i, i + 1, [(0, 1, 0, 1)], _strip_nx(xs[i], xs[i + 1], 6.0, level), [ny],
                         outlet=("outlet" if i == len(xs) - 2 else None))

    x_pocket_mid = L_ramp + 0.5 * L_pocket
    sample_lines = [
        SampleLine("inletProfile", "y", *_line_pts(-0.5, 0.0, -0.5, 1.0), n_points=80),
        SampleLine("pocketMid", "y", *_line_pts(x_pocket_mid, 0.0, x_pocket_mid, pocket_ratio), n_points=100),
        SampleLine("nearWallBottom", "x", (0.02, 0.02, 0.5), (x_end - 0.02, 0.02, 0.5), n_points=200),
        SampleLine("outletProfile", "y", *_line_pts(x_end - 1.0, 0.0, x_end - 1.0, 1.0), n_points=80),
    ]
    return _extrude_2d_mesh(
        family="G10",
        level=level,
        params={"pocket_ratio": pocket_ratio, "L_pocket": L_pocket, "L_ramp": L_ramp,
                "L_in": L_in, "L_out_ratio": L_out_ratio},
        verts_2d=sm.verts,
        blocks_2d=sm.blocks,
        patch_edges=sm.patch_edges(),
        sample_lines=sample_lines,
        inlet_height=1.0,
        outlet_heights=(1.0,),
        entry_length=L_in,
        outlet_lengths=(L_out,),
    )


def build_g11(params: Mapping[str, Any], level: int = 1) -> MeshSpec:
    """Build G11 (split-and-recombine island loop, 1->2->1, genus 1) in star units.

    The inlet channel (h = 1) eases into a symmetric chamber of height H, runs
    around a smooth island (thickness 0.20, C1 cosine-tapered noses of length 0.5
    so the apex curvature radius stays >= 0.10 h1) of axial length L_loop centred
    at y_c, then eases back to h = 1 and a straight outlet.
    Lower branch height is exactly 0.50 h1; upper branch height is
    0.5*(1-branch_bias)/branch_bias (both >= 0.50 h1 for branch_bias in [0.35, 0.50]).
    The chamber entry/exit ramps have axial length L_r = max(1, (H-1)/tan(theta)).
    The island wall is a separate ``wallsIsland`` wall patch (FEniCSx tag WALL).

    Parameters
    ----------
    params:
        branch_bias : float in [0.35, 0.50]
        L_loop : float in [4.0, 9.0]  (island axial length incl. caps)
        theta_deg : float in [20.0, 50.0]  (sets the chamber ramp length)
        L_in : float >= 5.0, optional (default 5.0)
        L_out_ratio : float >= 10.0, optional (default 10.0)
    level : int in {0, 1, 2, 3}
    """
    branch_bias = float(params["branch_bias"])
    L_loop = float(params["L_loop"])
    theta_deg = float(params["theta_deg"])
    L_in = float(params.get("L_in", 5.0))
    L_out_ratio = float(params.get("L_out_ratio", 10.0))
    if not (0.35 <= branch_bias <= 0.50):
        raise ValueError(f"G11 branch_bias must be in [0.35, 0.50], got {branch_bias}")
    if not (4.0 <= L_loop <= 9.0):
        raise ValueError(f"G11 L_loop must be in [4.0, 9.0], got {L_loop}")
    if not (20.0 <= theta_deg <= 50.0):
        raise ValueError(f"G11 theta_deg must be in [20, 50], got {theta_deg}")
    if L_in < 5.0:
        raise ValueError(f"G11 L_in must be >= 5.0, got {L_in}")
    if L_out_ratio < 10.0:
        raise ValueError(f"G11 L_out_ratio must be >= 10.0, got {L_out_ratio}")
    _read_fillet_r(params, "G11")

    r = 0.10            # island cap radius r_f* = 0.10 (>= 0.10 guardrail)
    t_isl = 2.0 * r     # island thickness 0.20
    h_lo = 0.5
    h_up = 0.5 * (1.0 - branch_bias) / branch_bias
    H = h_lo + h_up + t_isl
    y_lo = 0.5 * (1.0 - H)
    y_top = y_lo + H
    y_c = y_lo + h_lo + r  # island centreline

    L_r = max(1.0, (H - 1.0) / math.tan(math.radians(theta_deg)))
    L_out = L_out_ratio * 1.0
    x_i0 = L_r + 1.0
    x_i1 = x_i0 + L_loop
    x_r_end = x_i1 + L_r
    x_end = x_r_end + L_out

    def walls(x: float) -> tuple[float, float, float]:
        """(y_bot, y_mid, y_top) of the channel at x (band split at y_mid)."""
        if x <= 0.0:
            return 0.0, 0.5, 1.0
        if x <= L_r:
            f = _ss(x / L_r)
            return _lerp(0.0, y_lo, f), _lerp(0.5, y_c, f), _lerp(1.0, y_top, f)
        if x <= x_i1:
            return y_lo, y_c, y_top
        if x <= x_r_end:
            f = _ss((x - x_i1) / L_r)
            return _lerp(y_lo, 0.0, f), _lerp(y_c, 0.5, f), _lerp(y_top, 1.0, f)
        return 0.0, 0.5, 1.0

    nseg = 8
    L_nose = 0.5
    nnose = 5
    if L_loop < 2.0 * L_nose + 0.5:
        raise ValueError(f"G11 L_loop must be >= {2.0 * L_nose + 0.5}, got {L_loop}")
    xs = [-L_in, 0.0]
    xs += [L_r * (k + 1) / nseg for k in range(nseg)]                    # entry ramp
    xs.append(x_i0)                                                        # island tip station (zero thickness)
    xs += [x_i0 + L_nose * k / nnose for k in range(1, nnose + 1)]        # entry nose (C1 cosine taper)
    xs.append(x_i0 + L_nose + 0.5 * (L_loop - 2.0 * L_nose))               # straight body mid
    xs.append(x_i1 - L_nose)                                               # exit nose start
    xs += [x_i1 - L_nose + L_nose * k / nnose for k in range(1, nnose)]    # exit nose
    xs.append(x_i1)                                                        # island end station (zero thickness)
    xs += [x_i1 + L_r * (k + 1) / nseg for k in range(nseg)]               # exit ramp
    xs += [x_r_end + L_out * (k + 1) / 4.0 for k in range(4)]              # outlet

    def is_island_station(x: float) -> bool:
        return x_i0 < x < x_i1

    sm = _StripMesh()
    for x in xs:
        yb, ym, yt = walls(x)
        if is_island_station(x):
            ht = _island_half_thickness(x, x_i0, x_i1, r, L_nose)
            sm.station(x, [yb, y_c - ht, y_c + ht, yt])
        else:
            sm.station(x, [yb, ym, yt])
    sm.finish_inlet()

    ny0 = _scale_cells(_base_cells_for_length(h_lo + r, 10.0, minimum=6), level)
    ny1 = _scale_cells(_base_cells_for_length(h_up + r, 10.0, minimum=6), level)
    for i in range(len(sm.xs) - 1):
        x0, x1 = sm.xs[i], sm.xs[i + 1]
        nx = _strip_nx(x0, x1, 6.0, level)
        isl0 = is_island_station(x0)
        isl1 = is_island_station(x1)
        if not isl0 and not isl1:
            bands = [(0, 1, 0, 1), (1, 2, 1, 2)]
            island = False
        elif isl0 and isl1:
            bands = [(0, 1, 0, 1), (2, 3, 2, 3)]
            island = True
        elif not isl0 and isl1:  # entering the island (left 3 levels, right 4)
            bands = [(0, 1, 0, 1), (1, 2, 2, 3)]
            island = True
        else:  # leaving the island (left 4 levels, right 3)
            bands = [(0, 1, 0, 1), (2, 3, 1, 2)]
            island = True
        sm.strip_between(i, i + 1, bands, nx, [ny0, ny1], island=island,
                         outlet=("outlet" if i == len(sm.xs) - 2 else None))

    x_mid = x_i0 + 0.5 * L_loop
    sample_lines = [
        SampleLine("inletProfile", "y", *_line_pts(-0.5, 0.0, -0.5, 1.0), n_points=80),
        SampleLine("lowerBranchMid", "y", *_line_pts(x_mid, y_lo, x_mid, y_c - r), n_points=80),
        SampleLine("upperBranchMid", "y", *_line_pts(x_mid, y_c + r, x_mid, y_top), n_points=80),
        SampleLine("centerline", "x", (-L_in + 0.02, 0.5, 0.5), (L_r - 0.02, 0.5, 0.5), n_points=120),
        SampleLine("outletProfile", "y", *_line_pts(x_end - 1.0, 0.0, x_end - 1.0, 1.0), n_points=80),
    ]
    return _extrude_2d_mesh(
        family="G11",
        level=level,
        params={"branch_bias": branch_bias, "L_loop": L_loop, "theta_deg": theta_deg,
                "L_in": L_in, "L_out_ratio": L_out_ratio},
        verts_2d=sm.verts,
        blocks_2d=sm.blocks,
        patch_edges=sm.patch_edges(),
        sample_lines=sample_lines,
        inlet_height=1.0,
        outlet_heights=(1.0,),
        entry_length=L_in,
        outlet_lengths=(L_out,),
    )


def build_g12(params: Mapping[str, Any], level: int = 1) -> MeshSpec:
    """Build G12 (sinusoidal meandering channel, 1->1, constant width 1) in star units.

    Centerline y_c(x) = 0.5 + (amp_ratio/2) sin(2 pi x / wavelength) with a C1
    half-wavelength entry/exit envelope; walls are the vertical offset y_c +- 0.5
    (normal width >= 1/sqrt(1+slope_max^2) >= 0.85 at the parameter corners).

    Parameters
    ----------
    params:
        amp_ratio : float in [0.15, 0.60]  (meander amplitude / h1)
        wavelength : float in [3.0, 7.0]
        n_waves : int in {2, 3, 4}
        L_in : float >= 5.0, optional (default 5.0)
        L_out_ratio : float >= 10.0, optional (default 10.0)
    level : int in {0, 1, 2, 3}
    """
    amp_ratio = float(params["amp_ratio"])
    wavelength = float(params["wavelength"])
    n_waves = int(params["n_waves"])
    L_in = float(params.get("L_in", 5.0))
    L_out_ratio = float(params.get("L_out_ratio", 10.0))
    if not (0.15 <= amp_ratio <= 0.60):
        raise ValueError(f"G12 amp_ratio must be in [0.15, 0.60], got {amp_ratio}")
    if not (3.0 <= wavelength <= 7.0):
        raise ValueError(f"G12 wavelength must be in [3.0, 7.0], got {wavelength}")
    if n_waves not in (2, 3, 4):
        raise ValueError(f"G12 n_waves must be in {{2, 3, 4}}, got {n_waves}")
    if L_in < 5.0:
        raise ValueError(f"G12 L_in must be >= 5.0, got {L_in}")
    if L_out_ratio < 10.0:
        raise ValueError(f"G12 L_out_ratio must be >= 10.0, got {L_out_ratio}")
    _read_fillet_r(params, "G12")

    L_m = n_waves * wavelength
    L_out = L_out_ratio * 1.0
    x_end = L_m + L_out

    def y_center(x: float) -> float:
        if x <= 0.0 or x >= L_m:
            return 0.5
        env = _ss(min(1.0, 2.0 * min(x, L_m - x) / wavelength))
        return 0.5 + 0.5 * amp_ratio * math.sin(2.0 * math.pi * x / wavelength) * env

    nseg_per_wave = 8
    xs = [-L_in, 0.0]
    xs += [L_m * (k + 1) / (nseg_per_wave * n_waves) for k in range(nseg_per_wave * n_waves)]
    xs += [L_m + L_out * (k + 1) / 4.0 for k in range(4)]

    sm = _StripMesh()
    for x in xs:
        yc = y_center(x)
        sm.station(x, [yc - 0.5, yc + 0.5])
    sm.finish_inlet()
    ny = _scale_cells(_base_cells_for_length(1.0, 12.0, minimum=8), level)
    for i in range(len(xs) - 1):
        sm.strip_between(i, i + 1, [(0, 1, 0, 1)], _strip_nx(xs[i], xs[i + 1], 6.0, level), [ny],
                         outlet=("outlet" if i == len(xs) - 2 else None))

    x_mid = 0.5 * L_m
    yc_mid = y_center(x_mid)
    sample_lines = [
        SampleLine("inletProfile", "y", *_line_pts(-0.5, 0.0, -0.5, 1.0), n_points=80),
        SampleLine("meanderMid", "y", *_line_pts(x_mid, yc_mid - 0.5, x_mid, yc_mid + 0.5), n_points=100),
        SampleLine("centerline", "x", (-L_in + 0.02, 0.5, 0.5), (x_end - 0.02, 0.5, 0.5), n_points=200),
        SampleLine("outletProfile", "y", *_line_pts(x_end - 1.0, 0.0, x_end - 1.0, 1.0), n_points=80),
    ]
    return _extrude_2d_mesh(
        family="G12",
        level=level,
        params={"amp_ratio": amp_ratio, "wavelength": wavelength, "n_waves": n_waves,
                "L_in": L_in, "L_out_ratio": L_out_ratio},
        verts_2d=sm.verts,
        blocks_2d=sm.blocks,
        patch_edges=sm.patch_edges(),
        sample_lines=sample_lines,
        inlet_height=1.0,
        outlet_heights=(1.0,),
        entry_length=L_in,
        outlet_lengths=(L_out,),
    )


def _branch_stations(
    x0: float,
    y0: float,
    phi_end: float,
    L_bend: float,
    n_bend: int,
    L_total: float,
    n_straight: int,
    h: float,
) -> list[tuple[Point2D, Point2D, float]]:
    """Stations along a cosine-angle bend then straight run.

    Returns per station ``(W_minus, W_plus, phi)`` wall points at +- h/2 from the
    centreline; the centreline starts at ``(x0, y0)`` with phi = 0.
    """
    n_int = 400
    ds_int = L_bend / n_int
    # integrate centreline on the fine grid
    cx = [x0]
    cy = [y0]
    x, y = x0, y0
    for k in range(n_int):
        s = (k + 0.5) * ds_int
        phi = phi_end * _ss(s / L_bend)
        x += ds_int * math.cos(phi)
        y += ds_int * math.sin(phi)
        cx.append(x)
        cy.append(y)

    def center_at(s: float) -> Point2D:
        if s <= L_bend:
            t = min(1.0, s / L_bend) * n_int
            k = min(n_int - 1, int(t))
            f = t - k
            return (_lerp(cx[k], cx[k + 1], f), _lerp(cy[k], cy[k + 1], f))
        phi = phi_end
        sb = L_bend
        return (cx[-1] + (s - sb) * math.cos(phi), cy[-1] + (s - sb) * math.sin(phi))

    stations: list[tuple[Point2D, Point2D, float]] = []
    ss = [L_bend * (k + 1) / n_bend for k in range(n_bend)]
    ss += [L_bend + (L_total - L_bend) * (k + 1) / n_straight for k in range(n_straight)]
    for s in ss:
        phi = phi_end * _ss(min(1.0, s / L_bend))
        c = center_at(s)
        nx_, ny_ = -math.sin(phi), math.cos(phi)
        stations.append(((c[0] - 0.5 * h * nx_, c[1] - 0.5 * h * ny_),
                         (c[0] + 0.5 * h * nx_, c[1] + 0.5 * h * ny_), phi))
    return stations


def build_g13(params: Mapping[str, Any], level: int = 1) -> MeshSpec:
    """Build G13 (symmetric trifurcation manifold, 1->3) in star units (h1 = 1).

    The inlet (h = 1) eases into a chamber of height H = 2*outer_ratio + 2*w_t +
    center_ratio (w_t = 0.20 wedge thickness).  Two stadium wedges (cap radius
    r_f* = 0.10, axial length L_isl = 3.0) split the chamber into three parallel
    branches of heights (outer_ratio + w_t/2, center_ratio + w_t, outer_ratio +
    w_t/2).  At the manifold end the outer branches bend by +-theta_deg over a
    C1 cosine bend (L_bend = 2.0) and run straight to their outlets; the centre
    branch continues straight.  Outlet patches: ``outlet1`` (lower), ``outlet2``
    (centre), ``outlet3`` (upper).  Wedge walls are the ``wallsIsland`` wall patch.

    Parameters
    ----------
    params:
        center_ratio : float in [0.50, 0.90]  (centre branch island-free height / h1)
        outer_ratio : float in [0.50, 0.80]   (outer branch island-free height / h1)
        theta_deg : float in [25.0, 60.0]     (outer branch angle; also sets ramp length)
        L_in : float >= 5.0, optional (default 5.0)
        L_out_ratio : float >= 10.0, optional (default 10.0)
    level : int in {0, 1, 2, 3}
    """
    center_ratio = float(params["center_ratio"])
    outer_ratio = float(params["outer_ratio"])
    theta_deg = float(params["theta_deg"])
    L_in = float(params.get("L_in", 5.0))
    L_out_ratio = float(params.get("L_out_ratio", 10.0))
    if not (0.50 <= center_ratio <= 0.90):
        raise ValueError(f"G13 center_ratio must be in [0.50, 0.90], got {center_ratio}")
    if not (0.50 <= outer_ratio <= 0.80):
        raise ValueError(f"G13 outer_ratio must be in [0.50, 0.80], got {outer_ratio}")
    if not (25.0 <= theta_deg <= 60.0):
        raise ValueError(f"G13 theta_deg must be in [25, 60], got {theta_deg}")
    if L_in < 5.0:
        raise ValueError(f"G13 L_in must be >= 5.0, got {L_in}")
    if L_out_ratio < 10.0:
        raise ValueError(f"G13 L_out_ratio must be >= 10.0, got {L_out_ratio}")
    _read_fillet_r(params, "G13")

    w_t = 0.20
    r = 0.10
    L_isl = 3.0
    L_bend = 2.0
    y0 = 0.5 * (1.0 - (2.0 * outer_ratio + 2.0 * w_t + center_ratio))
    y1 = y0 + outer_ratio
    y2 = y1 + w_t
    y3 = y2 + center_ratio
    y4 = y3 + w_t
    y5 = y4 + outer_ratio
    yc0 = 0.5 * (y1 + y2)
    yc1 = 0.5 * (y3 + y4)
    chamber = [y0, yc0, yc1, y5]
    inlet_levels = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]

    L_r = max(1.5, (y5 - y0 - 1.0) / math.tan(math.radians(theta_deg)))
    a = L_r + 0.75
    b = a + L_isl
    x_m = b + 0.75

    hB = (yc0 - y0, yc1 - yc0, y5 - yc1)  # branch (band) heights
    cB = (0.5 * (y0 + yc0), 0.5 * (yc0 + yc1), 0.5 * (yc1 + y5))

    def ramp_levels(x: float) -> list[float]:
        f = _ss(x / L_r)
        return [_lerp(a_, b_, f) for a_, b_ in zip(inlet_levels, chamber)]

    def is_island_station(x: float) -> bool:
        return a < x < b

    nseg = 8
    L_nose = 1.0  # both wedges taper over the same x-range, so the middle band narrows by 2r: longer nose = gentler slope
    nnose = 5
    xs = [-L_in, 0.0]
    xs += [L_r * (k + 1) / nseg for k in range(nseg)]
    xs.append(a)                                                           # wedge tip station (zero thickness)
    xs += [a + L_nose * k / nnose for k in range(1, nnose + 1)]            # entry nose (C1 cosine taper)
    xs.append(a + L_nose + 0.5 * (L_isl - 2.0 * L_nose))                   # straight body mid
    xs.append(b - L_nose)                                                  # exit nose start
    xs += [b - L_nose + L_nose * k / nnose for k in range(1, nnose)]       # exit nose
    xs.append(b)                                                           # wedge end station (zero thickness)
    xs.append(x_m)                                                         # manifold end / branch origin

    sm = _StripMesh()
    for x in xs:
        if x <= 0.0:
            sm.station(x, list(inlet_levels))
        elif x <= L_r:
            sm.station(x, ramp_levels(x))
        elif is_island_station(x):
            ht0 = _island_half_thickness(x, a, b, r, L_nose)
            ht1 = _island_half_thickness(x, a, b, r, L_nose)
            sm.station(x, [y0, yc0 - ht0, yc0 + ht0, yc1 - ht1, yc1 + ht1, y5])
        else:
            sm.station(x, list(chamber))
    sm.finish_inlet()

    nys_band = [_scale_cells(_base_cells_for_length(h, 10.0, minimum=6), level) for h in hB]
    for i in range(len(sm.xs) - 1):
        x0, x1 = sm.xs[i], sm.xs[i + 1]
        nx = _strip_nx(x0, x1, 6.0, level)
        isl0 = is_island_station(x0)
        isl1 = is_island_station(x1)
        if not isl0 and not isl1:
            bands = [(0, 1, 0, 1), (1, 2, 1, 2), (2, 3, 2, 3)]
            island = False
        elif isl0 and isl1:
            bands = [(0, 1, 0, 1), (2, 3, 2, 3), (4, 5, 4, 5)]
            island = True
        elif not isl0 and isl1:  # entering the wedges (left 4 levels, right 6)
            bands = [(0, 1, 0, 1), (1, 2, 2, 3), (2, 3, 4, 5)]
            island = True
        else:  # leaving the wedges (left 6 levels, right 4)
            bands = [(0, 1, 0, 1), (2, 3, 1, 2), (4, 5, 2, 3)]
            island = True
        sm.strip_between(i, i + 1, bands, nx, nys_band, island=island)

    # Branches: lower (k=0, -theta), centre (k=1, 0), upper (k=2, +theta).
    # The manifold end station (x_m, chamber levels) is the common branch origin;
    # each branch k starts from its band (band = (0,1)/(1,2)/(2,3)) of that station.
    theta = math.radians(theta_deg)
    branch_phi = (-theta, 0.0, theta)
    branch_outlet = ("outlet1", "outlet2", "outlet3")
    branch_line = ("branch1Mid", "branch2Mid", "branch3Mid")
    branch_band = ((0, 1), (1, 2), (2, 3))
    m_idx = len(sm.stations) - 1  # manifold end station at x_m

    branch_station_lists: list[list[tuple[Point2D, Point2D, float]]] = []
    for k in range(3):
        L_total = (L_bend if k != 1 else 0.0) + L_out_ratio * hB[k]
        n_straight = 4
        if k == 1:
            sts: list[tuple[Point2D, Point2D, float]] = []
            for j in range(1, n_straight + 1):
                s = L_total * j / n_straight
                sts.append(((x_m + s, yc0), (x_m + s, yc1), 0.0))
            branch_station_lists.append(sts)
        else:
            branch_station_lists.append(
                _branch_stations(x_m, cB[k], branch_phi[k], L_bend, 10, L_total, n_straight, hB[k])
            )

    for k in range(3):
        sts = branch_station_lists[k]
        band = branch_band[k]
        base = len(sm.stations)  # branch stations appended after the manifold end
        for (wm, wp, _phi) in sts:
            sm.station_pts([wm, wp])  # slanted cross-section: keep exact 2D wall points
        for j in range(len(sts)):
            i_left = m_idx if j == 0 else base + j - 1
            i_right = base + j
            if j == 0:
                bands = [(band[0], band[1], 0, 1)]
            else:
                bands = [(0, 1, 0, 1)]
            nx = _strip_nx(sm.xs[i_left], sm.xs[i_right], 6.0, level)
            sm.strip_between(i_left, i_right, bands, nx, [nys_band[k]],
                             outlet=(branch_outlet[k] if j == len(sts) - 1 else None))

    # sample lines: mid-branch cross-sections
    sample_lines = [
        SampleLine("inletProfile", "y", *_line_pts(-0.5, 0.0, -0.5, 1.0), n_points=80),
    ]
    for k in range(3):
        sts = branch_station_lists[k]
        mid = sts[len(sts) // 2]
        sample_lines.append(
            SampleLine(branch_line[k], "distance", *_line_pts(mid[0][0], mid[0][1], mid[1][0], mid[1][1]), n_points=80)
        )

    outlet_lengths = tuple(
        (L_bend if k != 1 else 0.0) + L_out_ratio * hB[k] for k in range(3)
    )
    return _extrude_2d_mesh(
        family="G13",
        level=level,
        params={"center_ratio": center_ratio, "outer_ratio": outer_ratio, "theta_deg": theta_deg,
                "L_in": L_in, "L_out_ratio": L_out_ratio},
        verts_2d=sm.verts,
        blocks_2d=sm.blocks,
        patch_edges=sm.patch_edges(),
        sample_lines=sample_lines,
        inlet_height=1.0,
        outlet_heights=tuple(hB),
        entry_length=L_in,
        outlet_lengths=outlet_lengths,
    )




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


# ---- G1 (from cfd/gen/m2b.py) + verified physical constants ----

# Verified primary-source physical constants
# Wu et al. 2015 §3 (PMC4521229, verified-primary):
WU2015_RHO_PLASMA = 1027.0       # kg/m^3
WU2015_RHO_RBC = 1093.0          # kg/m^3
WU2015_MU_PLASMA = 0.96e-3       # Pa.s (0.96 cP)
WU2015_DP_RBC = 8.0e-6           # m (8 um)
WU2015_RHO_RATIO = WU2015_RHO_RBC / WU2015_RHO_PLASMA  # 1.0642648...

# 2023 COMSOL slides (knowledge/13-2023-comsol-model.md, slide 6):
COMSOL2023_RHO_PLASMA = 1038.0   # kg/m^3 (most likely continuous phase, TODO: confirm)
COMSOL2023_RHO_RBC = 1098.0      # kg/m^3
COMSOL2023_MU_PLASMA = 4.0e-3    # Pa.s
COMSOL2023_DP_RBC = 8.0e-6       # m (8 um)
COMSOL2023_RHO_RATIO = COMSOL2023_RHO_RBC / COMSOL2023_RHO_PLASMA  # 1.0578034...


def build_g1(params: Mapping[str, Any], level: int = 1) -> MeshSpec:
    """Build G1 (2D planar straight microchannel) in star units (h1 = 1).

    Inlet sits on ``x = 0, y in [0, 1]`` and outlet sits on ``x = L_ratio, y in [0, 1]``.

    Parameters
    ----------
    params : Mapping[str, Any]
        Optional keys:
        - ``L_ratio`` : float in [10.0, 30.0], default 20.0 (``L / h1``)
    level : int in {0, 1, 2}
        Mesh refinement level (refinement ratio ``r = 1.5``).
    """
    L_ratio = float(params.get("L_ratio", 20.0))
    if not (10.0 <= L_ratio <= 30.0):
        raise ValueError(f"G1 L_ratio must be in [10.0, 30.0], got {L_ratio}")

    verts_2d = [
        (0.0, 0.0),      # 0
        (L_ratio, 0.0),  # 1
        (L_ratio, 1.0),  # 2
        (0.0, 1.0),      # 3
    ]

    ny = _scale_cells(16, level)
    nx = _scale_cells(_base_cells_for_length(L_ratio, 6.0, minimum=40), level)

    blocks_2d = [Block2D((0, 1, 2, 3), nx, ny)]
    patch_edges = [
        ("inlet", "patch", [(3, 0)]),
        ("outlet", "patch", [(1, 2)]),
        ("walls", "wall", [(0, 1), (2, 3)]),
    ]
    sample_lines = [
        SampleLine("inletProfile", "y", *_line_pts(0.5, 0.0, 0.5, 1.0), n_points=80),
        SampleLine(
            "midProfile",
            "y",
            *_line_pts(0.5 * L_ratio, 0.0, 0.5 * L_ratio, 1.0),
            n_points=80,
        ),
        SampleLine(
            "centerline",
            "x",
            (0.02, 0.5, 0.5),
            (L_ratio - 0.02, 0.5, 0.5),
            n_points=200,
        ),
        SampleLine(
            "outletProfile",
            "y",
            *_line_pts(L_ratio - 1.0, 0.0, L_ratio - 1.0, 1.0),
            n_points=100,
        ),
    ]

    return _extrude_2d_mesh(
        family="G1",
        level=level,
        params={"L_ratio": L_ratio},
        verts_2d=verts_2d,
        blocks_2d=blocks_2d,
        patch_edges=patch_edges,
        sample_lines=sample_lines,
        inlet_height=1.0,
        outlet_heights=(1.0,),
        entry_length=L_ratio,
        outlet_lengths=(L_ratio,),
    )


def build_geometry(family: str, params: Mapping[str, Any], level: int = 1) -> MeshSpec:
    """Dispatch geometry construction by family ID in {'G1'..'G13', 'KG'}."""
    builders = {
        "G1": build_g1,
        "G2": build_g2,
        "G3": build_g3,
        "G4": build_g4,
        "G5": build_g5,
        "G6": build_g6,
        "G7": build_g7,
        "G8": build_g8,
        "G9": build_g9,
        "G10": build_g10,
        "G11": build_g11,
        "G12": build_g12,
        "G13": build_g13,
        "KG": build_kg,
    }
    key = family.upper()
    if key not in builders:
        raise ValueError(f"Unsupported geometry family '{family}'; expected one of {sorted(builders)}")
    return builders[key](params, level=level)
