"""Unit tests for nondimensional OpenFOAM geometry & case generators (G2–G7) and M1 design."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from campaign.design_m1 import generate_design
from cfd.gen.case import write_case
from cfd.gen.geometry import MeshSpec, build_geometry, format_block_mesh_dict


def _signed_tet_volume(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
    d: tuple[float, float, float],
) -> float:
    """Signed volume of tetrahedron (a, b, c, d): det([b-a, c-a, d-a]) / 6."""
    mat = np.array(
        [
            [b[0] - a[0], c[0] - a[0], d[0] - a[0]],
            [b[1] - a[1], c[1] - a[1], d[1] - a[1]],
            [b[2] - a[2], c[2] - a[2], d[2] - a[2]],
        ],
        dtype=float,
    )
    return float(np.linalg.det(mat) / 6.0)


def _hex_corner_jacobians(
    verts: tuple[tuple[float, float, float], ...],
    hex_ids: tuple[int, int, int, int, int, int, int, int],
) -> list[float]:
    """Compute the 8 corner Jacobian determinants (6 * corner tet volume) of a hex block.

    OpenFOAM hex ordering: (0,1,2,3) on bottom face CCW, (4,5,6,7) on top face CCW.
    Every corner Jacobian must be strictly positive for a valid positive-volume hex.
    """
    p = [verts[i] for i in hex_ids]
    # Each tuple is (origin, +xi_neighbor, +eta_neighbor, +zeta_neighbor) in right-handed order.
    corners = (
        (0, 1, 3, 4),
        (1, 2, 0, 5),
        (2, 3, 1, 6),
        (3, 0, 2, 7),
        (4, 7, 5, 0),
        (5, 4, 6, 1),
        (6, 5, 7, 2),
        (7, 6, 4, 3),
    )
    return [6.0 * _signed_tet_volume(p[o], p[i], p[j], p[k]) for (o, i, j, k) in corners]


def _hex_signed_volume(
    verts: tuple[tuple[float, float, float], ...],
    hex_ids: tuple[int, int, int, int, int, int, int, int],
) -> float:
    """Total signed volume of a hex split into 6 positive-oriented tetrahedra."""
    p = [verts[i] for i in hex_ids]
    # Standard 6-tet decomposition along diagonal 0 -> 6 (valid for any convex hex).
    tets = (
        (0, 1, 2, 6),
        (0, 2, 3, 6),
        (0, 3, 7, 6),
        (0, 7, 4, 6),
        (0, 4, 5, 6),
        (0, 5, 1, 6),
    )
    return sum(_signed_tet_volume(p[a], p[b], p[c], p[d]) for (a, b, c, d) in tets)


def _canonical_face(face: tuple[int, int, int, int]) -> frozenset[int]:
    """Unordered vertex set identifying a quad face independently of cyclic shift/orientation."""
    return frozenset(face)


def _hex_faces_outward(
    h: tuple[int, int, int, int, int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    """Return the 6 outward-oriented faces of an OpenFOAM hex (0 1 2 3 4 5 6 7)."""
    v0, v1, v2, v3, v4, v5, v6, v7 = h
    return [
        (v0, v3, v2, v1),  # z_min (back)
        (v4, v5, v6, v7),  # z_max (front)
        (v0, v1, v5, v4),  # x2_min
        (v1, v2, v6, v5),  # x1_max
        (v2, v3, v7, v6),  # x2_max
        (v3, v0, v4, v7),  # x1_min
    ]


def _face_area_and_normal(
    verts: tuple[tuple[float, float, float], ...],
    face: tuple[int, int, int, int],
) -> tuple[float, np.ndarray]:
    """Compute quad face area and unit normal from right-hand rule."""
    pts = [np.array(verts[i], dtype=float) for i in face]
    n1 = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    n2 = np.cross(pts[2] - pts[0], pts[3] - pts[0])
    vec = 0.5 * (n1 + n2)
    area = float(np.linalg.norm(vec))
    return area, vec / area


def _assert_valid_mesh(mesh: MeshSpec) -> None:
    """Assert positive-volume blocks, exact boundary coverage, outward normals, and h_in = 1."""
    assert len(mesh.blocks) >= 1
    assert len(mesh.vertices) >= 8

    # 1. Every block has strictly positive volume and all 8 corner Jacobians > 0.
    for idx, (hex_ids, (nx, ny, nz), _grading) in enumerate(mesh.blocks):
        assert nz == 1, f"{mesh.family} block {idx} is not 2D (nz={nz})"
        assert nx >= 1 and ny >= 1
        vol = _hex_signed_volume(mesh.vertices, hex_ids)
        assert vol > 0.0, f"{mesh.family} block {idx} non-positive volume: {vol}"
        jacs = _hex_corner_jacobians(mesh.vertices, hex_ids)
        assert all(j > 0.0 for j in jacs), f"{mesh.family} block {idx} corner Jacobians: {jacs}"

    # 2. Boundary faces = faces appearing exactly once across all hex blocks.
    face_counts: Counter[frozenset[int]] = Counter()
    face_owner_centroid: dict[frozenset[int], np.ndarray] = {}
    for hex_ids, _cells, _grading in mesh.blocks:
        centroid = np.mean([mesh.vertices[i] for i in hex_ids], axis=0)
        for f in _hex_faces_outward(hex_ids):
            key = _canonical_face(f)
            face_counts[key] += 1
            face_owner_centroid[key] = centroid

    assert all(c in (1, 2) for c in face_counts.values()), "Non-manifold face detected"
    true_boundary = {k for k, c in face_counts.items() if c == 1}

    # 3. Patches must cover true_boundary EXACTLY ONCE with outward normals.
    patch_faces_seen: list[frozenset[int]] = []
    patch_by_name = {p.name: p for p in mesh.patches}
    assert "inlet" in patch_by_name
    assert "walls" in patch_by_name
    assert "frontAndBack" in patch_by_name
    assert patch_by_name["frontAndBack"].patch_type == "empty"
    assert any(name.startswith("outlet") for name in patch_by_name)

    for patch in mesh.patches:
        assert len(patch.faces) >= 1, f"Empty patch {patch.name}"
        for f in patch.faces:
            key = _canonical_face(f)
            assert key in true_boundary, f"Patch {patch.name} face {f} is not a boundary face"
            patch_faces_seen.append(key)

            # Outward normal check: (face_center - block_centroid) . normal > 0
            area, normal = _face_area_and_normal(mesh.vertices, f)
            assert area > 0.0
            f_center = np.mean([mesh.vertices[i] for i in f], axis=0)
            outward_dot = float(np.dot(f_center - face_owner_centroid[key], normal))
            assert outward_dot > 0.0, (
                f"{mesh.family} patch {patch.name} face {f} points inward (dot={outward_dot:.3e})"
            )

    assert len(patch_faces_seen) == len(set(patch_faces_seen)), "Duplicate face across patches"
    assert set(patch_faces_seen) == true_boundary, "Patches do not cover boundary exactly once"

    # 4. Inlet height is exact 1.0 (y in [0, 1], z thickness = 1.0, area = 1.0, normal = -x).
    inlet_patch = patch_by_name["inlet"]
    inlet_area = 0.0
    inlet_ys: list[float] = []
    inlet_xs: list[float] = []
    for f in inlet_patch.faces:
        area, normal = _face_area_and_normal(mesh.vertices, f)
        inlet_area += area
        assert np.allclose(normal, [-1.0, 0.0, 0.0], atol=1e-9), (
            f"{mesh.family} inlet normal {normal} != (-1, 0, 0)"
        )
        for v_idx in f:
            inlet_xs.append(mesh.vertices[v_idx][0])
            inlet_ys.append(mesh.vertices[v_idx][1])

    assert math.isclose(inlet_area, 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(min(inlet_ys), 0.0, abs_tol=1e-9)
    assert math.isclose(max(inlet_ys), 1.0, abs_tol=1e-9)
    assert math.isclose(max(inlet_xs) - min(inlet_xs), 0.0, abs_tol=1e-9)
    assert math.isclose(mesh.inlet_height, 1.0, abs_tol=1e-9)

    # 5. Entry length >= 5h and every outlet length >= 10 h_out.
    assert mesh.entry_length >= 5.0 - 1e-9
    for h_out, l_out in zip(mesh.outlet_heights, mesh.outlet_lengths, strict=True):
        assert l_out >= 10.0 * h_out - 1e-9


# Corner and interior test cases for every family G2–G7.
FAMILY_CASES: list[tuple[str, dict[str, Any]]] = [
    ("G2", {"E": 1.5}),
    ("G2", {"E": 2.0}),
    ("G2", {"E": 4.0}),
    ("G3", {"throat_ratio": 0.3, "L_t": 1.0}),
    ("G3", {"throat_ratio": 0.5, "L_t": 3.0}),
    ("G3", {"throat_ratio": 0.8, "L_t": 5.0}),
    ("G4", {"theta_deg": 20.0, "daughter_ratio": 0.5}),
    ("G4", {"theta_deg": 20.0, "daughter_ratio": 1.0}),
    ("G4", {"theta_deg": 45.0, "daughter_ratio": 0.7}),
    ("G4", {"theta_deg": 60.0, "daughter_ratio": 0.55}),
    ("G4", {"theta_deg": 90.0, "daughter_ratio": 0.5}),
    ("G4", {"theta_deg": 90.0, "daughter_ratio": 1.0}),
    ("G5", {"width_ratio": 0.5, "theta_deg": 30.0}),
    ("G5", {"width_ratio": 0.75, "theta_deg": 60.0}),
    ("G5", {"width_ratio": 0.5, "theta_deg": 90.0}),
    ("G5", {"width_ratio": 1.0, "theta_deg": 90.0}),
    ("G6", {"Rc_ratio": 1.5, "n_bends": 1}),
    ("G6", {"Rc_ratio": 3.0, "n_bends": 2}),
    ("G6", {"Rc_ratio": 4.5, "n_bends": 3}),
    ("G6", {"Rc_ratio": 6.0, "n_bends": 4}),
    ("G7", {"E": 1.5, "theta_deg": 20.0}),
    ("G7", {"E": 2.5, "theta_deg": 45.0}),
    ("G7", {"E": 4.0, "theta_deg": 90.0}),
]


@pytest.mark.parametrize(("family", "params"), FAMILY_CASES)
def test_geometry_validity_all_families(family: str, params: dict[str, Any]) -> None:
    """Every family & parameter corner produces positive-volume blocks, exact boundary coverage, and h1=1."""
    mesh = build_geometry(family, params, level=1)
    _assert_valid_mesh(mesh)
    bmd = format_block_mesh_dict(mesh)
    assert "scale 1;" in bmd
    assert "frontAndBack" in bmd
    assert "type empty;" in bmd
    if family == "G6":
        assert "arc " in bmd


@pytest.mark.parametrize(
    ("family", "params"),
    [
        ("G2", {"E": 2.0}),
        ("G3", {"throat_ratio": 0.5, "L_t": 2.0}),
        ("G4", {"theta_deg": 45.0, "daughter_ratio": 0.75}),
        ("G5", {"width_ratio": 0.6, "theta_deg": 60.0}),
        ("G6", {"Rc_ratio": 2.5, "n_bends": 2}),
        ("G7", {"E": 2.0, "theta_deg": 45.0}),
    ],
)
def test_mesh_refinement_ratio_gci(family: str, params: dict[str, Any]) -> None:
    """Mesh levels 0, 1, 2 refine both in-plane directions by r ≈ 1.5 (cell count ~2.25x per level)."""
    meshes = [build_geometry(family, params, level=lvl) for lvl in (0, 1, 2)]
    for m in meshes:
        _assert_valid_mesh(m)

    for lvl in (0, 1):
        for (_, (nx0, ny0, nz0), _), (_, (nx1, ny1, nz1), _) in zip(
            meshes[lvl].blocks, meshes[lvl + 1].blocks, strict=True
        ):
            assert nz0 == 1 and nz1 == 1
            rx = nx1 / nx0
            ry = ny1 / ny0
            assert 1.35 <= rx <= 1.67, f"{family} lvl {lvl}->{lvl+1} rx={rx}"
            assert 1.35 <= ry <= 1.67, f"{family} lvl {lvl}->{lvl+1} ry={ry}"


@pytest.mark.parametrize("rheology", ["Newtonian", "BirdCarreau"])
def test_write_case_complete_and_nondimensional(tmp_path: Path, rheology: str) -> None:
    """write_case writes all required OpenFOAM v2306 files with exact star-unit parameters."""
    Re = 10.0
    beta = 16.0
    Cu = 1000.0
    n_val = 0.3568
    params: dict[str, Any] = {
        "run_id": f"test_{rheology}",
        "family": "G4",
        "rheology": rheology,
        "Re": Re,
        "beta": beta if rheology == "BirdCarreau" else None,
        "Cu": Cu if rheology == "BirdCarreau" else None,
        "n": n_val if rheology == "BirdCarreau" else None,
        "a": 2.0 if rheology == "BirdCarreau" else None,
        "geometry": {"theta_deg": 45.0, "daughter_ratio": 0.7},
    }
    case_dir = tmp_path / f"case_{rheology}"
    write_case(case_dir, params, level=0)

    expected_files = [
        "system/blockMeshDict",
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
        "0/U",
        "0/p",
        "constant/transportProperties",
        "constant/turbulenceProperties",
        "run_case.sh",
        "params.json",
    ]
    for rel in expected_files:
        assert (case_dir / rel).is_file(), f"Missing {rel}"

    u_txt = (case_dir / "0" / "U").read_text(encoding="utf-8")
    tp_txt = (case_dir / "constant" / "transportProperties").read_text(encoding="utf-8")
    fvs_txt = (case_dir / "system" / "fvSolution").read_text(encoding="utf-8")
    fvsch_txt = (case_dir / "system" / "fvSchemes").read_text(encoding="utf-8")
    cd_txt = (case_dir / "system" / "controlDict").read_text(encoding="utf-8")

    assert "residualControl" in fvs_txt
    assert "1.0e-05" in fvs_txt
    assert "nNonOrthogonalCorrectors 0;" in fvs_txt
    assert "cellLimited" not in fvsch_txt
    assert "default         Gauss linear;" in fvsch_txt
    assert "inletPressure" in cd_txt
    assert "qoiProbes" in cd_txt
    assert "writeCellCentres" in cd_txt
    assert "libsampling.so" in cd_txt

    if rheology == "Newtonian":
        assert "exprFixedValue" in u_txt
        assert "6.0 * pos().y() * (1.0 - pos().y())" in u_txt
        assert "transportModel  Newtonian;" in tp_txt
        assert "0.1;" in tp_txt  # 1 / Re = 0.1
    else:
        assert "value           uniform (1 0 0);" in u_txt
        assert "transportModel  BirdCarreau;" in tp_txt
        assert "nu0 [0 2 -1 0 0 0 0] 1.6;" in tp_txt   # beta / Re = 1.6
        assert "nuInf [0 2 -1 0 0 0 0] 0.1;" in tp_txt  # 1 / Re = 0.1
        assert "k [0 0 1 0 0 0 0] 1000;" in tp_txt


def test_design_m1_sobol_coverage_and_valid_meshes() -> None:
    """Sobol design produces 64 points/family, ~10% Newtonian, deterministic, and all valid meshes."""
    records_a = generate_design(n_per_family=64, seed=20261124)
    records_b = generate_design(n_per_family=64, seed=20261124)
    assert records_a == records_b
    assert len(records_a) == 6 * 64

    by_family: dict[str, list[dict[str, Any]]] = {}
    for rec in records_a:
        by_family.setdefault(rec["family"], []).append(rec)

    assert sorted(by_family) == ["G2", "G3", "G4", "G5", "G6", "G7"]

    for fam, recs in by_family.items():
        assert len(recs) == 64
        n_newt = sum(1 for r in recs if r["rheology"] == "Newtonian")
        assert n_newt == 6  # round(0.10 * 64)
        for r in recs:
            assert 0.5 <= r["Re"] <= 25.0
            if r["rheology"] == "BirdCarreau":
                assert 5.0 <= r["beta"] <= 30.0
                assert 1.0 <= r["Cu"] <= 1.0e5
                assert 0.25 <= r["n"] <= 0.60
                assert r["a"] == 2.0
            # Verify EVERY single generated design point yields a valid positive-volume mesh!
            mesh = build_geometry(fam, r["geometry"], level=0)
            _assert_valid_mesh(mesh)
            # Ensure JSON serializability
            assert json.loads(json.dumps(r))["run_id"] == r["run_id"]
