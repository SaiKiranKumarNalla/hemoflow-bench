"""Unit tests for T2: batch runner, star-unit field extraction, GCI tooling, and KG wedge geometry."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from campaign.design_gci import generate_gci_cases, generate_gci_design
from campaign.gci import (
    analyze_case_gci,
    format_summary_table,
    roache_gci,
    run_gci_from_manifest,
)
from campaign.run_batch import (
    FAMILY_GRID_SHAPES,
    build_fluid_mask,
    compute_sha256,
    evaluate_convergence,
    extract_case,
    load_manifest,
    parse_openfoam_field,
    recompute_window_fraction,
    run_batch,
    strain_rate_magnitude,
)
from cfd.gen.case import _outlet_centreline_probes, write_case
from cfd.gen.geometry import build_geometry, build_kg, format_block_mesh_dict
from rbc.postprocess import trapezoid_weights, window_fraction


# ---------------------------------------------------------------------------
# Helpers to synthesize realistic OpenFOAM v2306 output files (no OpenFOAM binary needed)
# ---------------------------------------------------------------------------

def _write_foam_scalar_field(
    path: Path,
    values: np.ndarray,
    *,
    obj_name: str,
    foam_class: str = "volScalarField",
    internal_keyword: str = "internalField",
    boundary_block: str = "",
) -> None:
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        f"    class       {foam_class};",
        f"    object      {obj_name};",
        "}",
        "dimensions      [0 0 0 0 0 0 0];",
        f"{internal_keyword}   nonuniform List<scalar>",
        f"{len(values)}",
        "(",
    ]
    lines.extend(f"{v:.10g}" for v in values)
    lines.append(");")
    if boundary_block:
        lines.append(boundary_block)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_foam_vector_field(path: Path, values: np.ndarray, *, obj_name: str) -> None:
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       volVectorField;",
        f"    object      {obj_name};",
        "}",
        "dimensions      [0 1 -1 0 0 0 0];",
        "internalField   nonuniform List<vector>",
        f"{len(values)}",
        "(",
    ]
    lines.extend(f"({v[0]:.10g} {v[1]:.10g} {v[2]:.10g})" for v in values)
    lines.append(");")
    lines.append("boundaryField { }")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_foam_tensor_field(
    path: Path,
    values: np.ndarray,
    *,
    obj_name: str,
    walls_values: np.ndarray | None = None,
) -> None:
    lines = [
        "FoamFile",
        "{",
        "    version     2.0;",
        "    format      ascii;",
        "    class       volTensorField;",
        f"    object      {obj_name};",
        "}",
        "dimensions      [0 0 -1 0 0 0 0];",
        "internalField   nonuniform List<tensor>",
        f"{len(values)}",
        "(",
    ]
    for t in values:
        flat = " ".join(f"{x:.10g}" for x in t.reshape(9))
        lines.append(f"({flat})")
    lines.append(");")
    lines.append("boundaryField")
    lines.append("{")
    if walls_values is not None:
        lines.append("    walls")
        lines.append("    {")
        lines.append("        type            calculated;")
        lines.append("        value           nonuniform List<tensor>")
        lines.append(f"{len(walls_values)}")
        lines.append("(")
        for t in walls_values:
            flat = " ".join(f"{x:.10g}" for x in t.reshape(9))
            lines.append(f"({flat})")
        lines.append(");")
        lines.append("    }")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _populate_fake_g2_run(
    case_dir: Path,
    *,
    E: float = 2.0,
    Re: float = 10.0,
    L_in: float = 5.0,
    L_out_ratio: float = 10.0,
    xr_target: float = 1.25,
    dp_grad_star: float = 12.0,
    wall_gdot_max: float = 9.5,
) -> None:
    """Write synthetic OpenFOAM v2306 solution + logs into an already-written G2 case_dir."""
    t_dir = case_dir / "250"
    t_dir.mkdir(parents=True, exist_ok=True)
    s = 0.5 * (E - 1.0)
    L_out = L_out_ratio * E

    # Build structured cell centers covering inlet [-L_in, 0] x [0, 1] and outlet [0, L_out] x [-s, 1+s].
    xs_in = np.linspace(-L_in + 0.25, -0.25, 10)
    ys_in = np.linspace(0.05, 0.95, 10)
    xx_i, yy_i = np.meshgrid(xs_in, ys_in)

    xs_out = np.linspace(0.25, L_out - 0.25, 20)
    ys_out = np.linspace(-s + 0.05, 1.0 + s - 0.05, 20)
    xx_o, yy_o = np.meshgrid(xs_out, ys_out)

    x_c = np.concatenate([xx_i.ravel(), xx_o.ravel()])
    y_c = np.concatenate([yy_i.ravel(), yy_o.ravel()])
    z_c = np.full_like(x_c, 0.5)
    n_cells = x_c.size

    # Linear viscous star pressure p*(x) = dp_grad_star * (L_out - x) => p_OF = p* / Re.
    p_star_exact = dp_grad_star * (L_out - x_c)
    p_of = p_star_exact / Re

    # Simple smooth Poiseuille-like velocity in the expanded outlet [-s, 1+s] (height E):
    # u(y) = (6 / E) * eta * (1 - eta), eta = (y + s) / E, mean velocity = 1 / E (mass-conserving).
    eta = np.clip((y_c + s) / E, 0.0, 1.0)
    ux = (6.0 / E) * eta * (1.0 - eta)
    uy = np.zeros_like(ux)
    uz = np.zeros_like(ux)
    dux_dy = (6.0 / (E * E)) * (1.0 - 2.0 * eta)

    grad_u = np.zeros((n_cells, 3, 3), dtype=np.float64)
    grad_u[:, 0, 1] = dux_dy  # dUx/dy -> |gdot| = |dux_dy|

    # Walls boundary grad(U) with known max shear rate wall_gdot_max.
    walls_grad = np.zeros((4, 3, 3), dtype=np.float64)
    walls_grad[0, 0, 1] = 3.0
    walls_grad[1, 0, 1] = -wall_gdot_max
    walls_grad[2, 0, 1] = 4.2
    walls_grad[3, 0, 1] = 1.5

    _write_foam_vector_field(t_dir / "C", np.column_stack([x_c, y_c, z_c]), obj_name="C")
    _write_foam_scalar_field(
        t_dir / "V",
        np.full(n_cells, 0.05, dtype=np.float64),
        obj_name="V",
        foam_class="volScalarField::Internal",
        internal_keyword="value",
    )
    _write_foam_vector_field(t_dir / "U", np.column_stack([ux, uy, uz]), obj_name="U")
    _write_foam_scalar_field(t_dir / "p", p_of, obj_name="p", boundary_block="boundaryField { }")
    _write_foam_tensor_field(t_dir / "grad(U)", grad_u, obj_name="grad(U)", walls_values=walls_grad)

    # Sample lines: nearWallLower (zero crossing at x = xr_target) and outletProfile.
    samp_dir = case_dir / "postProcessing" / "sample" / "250"
    samp_dir.mkdir(parents=True, exist_ok=True)

    x_nw = np.linspace(0.02, L_out - 0.02, 200)
    ux_nw = x_nw - xr_target  # negative for x < xr_target, positive for x > xr_target
    nw_mat = np.column_stack([x_nw, np.zeros_like(x_nw), ux_nw, np.zeros_like(x_nw), np.zeros_like(x_nw)])
    np.savetxt(samp_dir / "nearWallLower_p_U.xy", nw_mat)

    y_out = np.linspace(-s + 0.02, 1.0 + s - 0.02, 100)
    eta_o = (y_out + s) / E
    ux_out = (6.0 / E) * eta_o * (1.0 - eta_o)
    out_mat = np.column_stack(
        [y_out, np.zeros_like(y_out), ux_out, np.zeros_like(y_out), np.zeros_like(y_out)]
    )
    np.savetxt(samp_dir / "outletProfile_p_U.xy", out_mat)

    (case_dir / "log.checkMesh").write_text(
        f"Mesh stats\n    cells:            {n_cells}\nMesh OK.\n",
        encoding="utf-8",
    )
    (case_dir / "log.simpleFoam").write_text(
        "Build  : _a1b2c3-2306\n"
        "Time = 249\n"
        "smoothSolver:  Solving for Ux, Initial residual = 9.1e-07, Final residual = 1.2e-08\n"
        "smoothSolver:  Solving for Uy, Initial residual = 8.4e-07, Final residual = 1.1e-08\n"
        "GAMG:  Solving for p, Initial residual = 7.5e-07, Final residual = 3.0e-09\n"
        "Time = 250\n"
        "SIMPLE solution converged in 250 iterations\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 1. Wedge geometry (KG) validity tests
# ---------------------------------------------------------------------------

def _signed_tet_vol(a: tuple[float, float, float], b: tuple[float, float, float],
                    c: tuple[float, float, float], d: tuple[float, float, float]) -> float:
    mat = np.array(
        [
            [b[0] - a[0], c[0] - a[0], d[0] - a[0]],
            [b[1] - a[1], c[1] - a[1], d[1] - a[1]],
            [b[2] - a[2], c[2] - a[2], d[2] - a[2]],
        ],
        dtype=float,
    )
    return float(np.linalg.det(mat) / 6.0)


def _hex_vol_6tet(verts: tuple[tuple[float, float, float], ...], h: tuple[int, ...]) -> float:
    p = [verts[i] for i in h]
    tets = (
        (0, 1, 2, 6),
        (0, 2, 3, 6),
        (0, 3, 7, 6),
        (0, 7, 4, 6),
        (0, 4, 5, 6),
        (0, 5, 1, 6),
    )
    return sum(_signed_tet_vol(p[a], p[b], p[c], p[d]) for a, b, c, d in tets)


@pytest.mark.parametrize("E", [3.25, 252.0 / 75.5])
@pytest.mark.parametrize("level", [0, 1, 2])
def test_kg_wedge_geometry_validity(E: float, level: int) -> None:
    """KG axisymmetric wedge has positive volumes, symmetric <=5 deg wedge planes, axis patch, d1*=1."""
    mesh = build_kg({"E": E}, level=level)
    assert mesh.family == "KG"
    assert math.isclose(mesh.inlet_height, 1.0, abs_tol=1e-12)
    assert math.isclose(mesh.outlet_heights[0], E, abs_tol=1e-12)
    assert mesh.entry_length >= 5.0
    assert mesh.outlet_lengths[0] >= 10.0

    # Every block has strictly positive 3D volume (including collapsed axis prisms).
    for idx, (hex_ids, (nx, ny, nz), _grading) in enumerate(mesh.blocks):
        assert nz == 1 and nx >= 1 and ny >= 1
        vol = _hex_vol_6tet(mesh.vertices, hex_ids)
        assert vol > 0.0, f"KG block {idx} volume={vol}"

    # Verify patch types and exact wedge symmetry about z = 0.
    by_name = {p.name: p for p in mesh.patches}
    assert set(by_name) == {"inlet", "outlet", "walls", "axis", "wedgeBack", "wedgeFront"}
    assert by_name["axis"].patch_type == "empty"
    assert by_name["wedgeBack"].patch_type == "wedge"
    assert by_name["wedgeFront"].patch_type == "wedge"
    assert by_name["walls"].patch_type == "wall"

    # Compute normals on wedgeFront and wedgeBack and check symmetry + outward orientation.
    def _face_vec(face: tuple[int, int, int, int]) -> np.ndarray:
        pts = [np.array(mesh.vertices[i], dtype=float) for i in face]
        return 0.5 * (np.cross(pts[1] - pts[0], pts[2] - pts[0]) + np.cross(pts[2] - pts[0], pts[3] - pts[0]))

    n_front = _face_vec(by_name["wedgeFront"].faces[0])
    n_back = _face_vec(by_name["wedgeBack"].faces[0])
    u_front = n_front / np.linalg.norm(n_front)
    u_back = n_back / np.linalg.norm(n_back)
    assert math.isclose(u_front[0], 0.0, abs_tol=1e-12)
    assert math.isclose(u_back[0], 0.0, abs_tol=1e-12)
    assert math.isclose(u_front[1], u_back[1], abs_tol=1e-12)
    assert math.isclose(u_front[2], -u_back[2], abs_tol=1e-12)
    assert u_front[2] > 0.0 and u_back[2] < 0.0
    # Included wedge angle = 180 deg - angle(u_front, u_back)
    cos_between = float(np.clip(np.dot(u_front, u_back), -1.0, 1.0))
    wedge_angle = 180.0 - math.degrees(math.acos(cos_between))
    assert math.isclose(wedge_angle, 5.0, abs_tol=1e-6)

    # Non-degenerate inlet/outlet/wall normals point outward.
    inlet_n = _face_vec(by_name["inlet"].faces[0])
    assert inlet_n[0] < 0.0 and math.isclose(inlet_n[1], 0.0, abs_tol=1e-12)
    for f in by_name["outlet"].faces:
        out_n = _face_vec(f)
        assert out_n[0] > 0.0

    # Axis faces lie strictly on y = 0, z = 0 (collapsed zero-area edges).
    for f in by_name["axis"].faces:
        for v_idx in f:
            _, yv, zv = mesh.vertices[v_idx]
            assert math.isclose(yv, 0.0, abs_tol=1e-12)
            assert math.isclose(zv, 0.0, abs_tol=1e-12)

    bmd = format_block_mesh_dict(mesh)
    assert "type wedge;" in bmd
    assert "axis" in bmd


@pytest.mark.parametrize("Re", [12.2, 37.8])
def test_kg_write_case_newtonian(tmp_path: Path, Re: float) -> None:
    """write_case for KG writes wedge BCs, Hagen-Poiseuille inlet, writeCellVolumes, and gradU."""
    case_dir = tmp_path / f"kg_Re{Re}"
    write_case(
        case_dir,
        {
            "run_id": f"KG_Re{Re}",
            "family": "KG",
            "rheology": "Newtonian",
            "Re": Re,
            "geometry": {"E": 252.0 / 75.5},
        },
        level=1,
    )
    u_txt = (case_dir / "0" / "U").read_text(encoding="utf-8")
    p_txt = (case_dir / "0" / "p").read_text(encoding="utf-8")
    cd_txt = (case_dir / "system" / "controlDict").read_text(encoding="utf-8")
    tp_txt = (case_dir / "constant" / "transportProperties").read_text(encoding="utf-8")

    assert "type            wedge;" in u_txt
    assert "type            wedge;" in p_txt
    assert "2.0 * max(0.0, 1.0 - 4.0 *" in u_txt
    assert "writeCellVolumes" in cd_txt
    assert "type            grad;" in cd_txt
    assert f"{1.0 / Re:.6g}"[:6] in tp_txt


# ---------------------------------------------------------------------------
# 2. Field extraction & NPZ schema tests
# ---------------------------------------------------------------------------

def test_extract_case_npz_schema_and_scalars(tmp_path: Path) -> None:
    """extract_case produces exact viscous p*, gdot*, max wall tau*, x_r/h1, grid mask, and outlet profile."""
    Re = 10.0
    E = 2.0
    L_in = 5.0
    L_out_ratio = 10.0
    dp_grad = 12.0
    xr_target = 1.25
    wall_gdot_max = 9.5

    record: dict[str, Any] = {
        "run_id": "TEST_G2_000",
        "case_id": "TEST_G2",
        "family": "G2",
        "rheology": "Newtonian",
        "Re": Re,
        "beta": None,
        "Cu": None,
        "n": None,
        "a": None,
        "geometry": {"E": E, "L_in": L_in, "L_out_ratio": L_out_ratio},
    }
    case_dir = tmp_path / "case"
    write_case(case_dir, record, level=0)
    _populate_fake_g2_run(
        case_dir,
        E=E,
        Re=Re,
        L_in=L_in,
        L_out_ratio=L_out_ratio,
        xr_target=xr_target,
        dp_grad_star=dp_grad,
        wall_gdot_max=wall_gdot_max,
    )

    out_npz = tmp_path / "TEST_G2_000.npz"
    res = extract_case(
        case_dir=case_dir,
        design_record=record,
        level=0,
        wall_seconds=12.34,
        out_npz_path=out_npz,
        tau_lo=0.5,
        tau_hi=1.2,
    )

    # Verify scalars in star units.
    expected_dp_star = dp_grad * (L_in + L_out_ratio * E)  # exact linear extrapolation to x = -L_in
    assert math.isclose(res["scalars"]["dp_star"], expected_dp_star, rel_tol=1e-6)
    assert math.isclose(res["scalars"]["xr_over_h1"], xr_target, abs_tol=1e-4)
    assert math.isclose(res["scalars"]["max_wall_tau_star"], wall_gdot_max, rel_tol=1e-12)
    assert 0.0 < res["scalars"]["f_window"] < 1.0

    # Verify NPZ schema and metadata JSON.
    with np.load(out_npz) as data:
        required_keys = {
            "x_star",
            "y_star",
            "cell_volumes",
            "U_star",
            "p_star",
            "gdot_star",
            "tau_star",
            "grid_x",
            "grid_y",
            "grid_mask",
            "grid_ux",
            "grid_uy",
            "grid_p",
            "grid_gdot",
            "grid_tau",
            "outlet_s",
            "outlet_u_n",
            "outlet_gdot",
            "outlet_tau",
            "outlet_weights",
            "outlet_branch_id",
            "params",
        }
        assert required_keys.issubset(set(data.files))

        nx, ny = FAMILY_GRID_SHAPES["G2"]
        assert data["grid_mask"].shape == (ny, nx)
        assert data["grid_ux"].shape == (ny, nx)
        # Points outside fluid (e.g. step recess x < 0, y < 0) must be masked False and 0.0.
        xx, yy = np.meshgrid(data["grid_x"], data["grid_y"])
        outside_step = (xx < -0.5) & (yy < -0.1)
        assert np.any(outside_step)
        assert not np.any(data["grid_mask"][outside_step])
        assert np.all(data["grid_ux"][~data["grid_mask"]] == 0.0)

        meta = json.loads(str(data["params"]))
        assert meta["run_id"] == "TEST_G2_000"
        assert meta["level"] == 0
        assert meta["n_cells"] == 500
        assert meta["iterations"] == 250
        assert meta["converged"] is True
        assert meta["openfoam_version"] == "_a1b2c3-2306"
        assert math.isclose(meta["final_residuals"]["p"], 7.5e-7)

        # Verify outlet profile recomputation matches f_window and responds to new windows.
        f_recomp = recompute_window_fraction(out_npz, 0.5, 1.2)
        assert math.isclose(f_recomp, res["scalars"]["f_window"], rel_tol=1e-12)
        f_wide = recompute_window_fraction(out_npz, 0.0, 100.0)
        assert math.isclose(f_wide, 1.0, rel_tol=1e-12)
        f_empty = recompute_window_fraction(out_npz, 50.0, 60.0)
        assert math.isclose(f_empty, 0.0, abs_tol=1e-12)

        # Verify against rbc.postprocess.window_fraction directly.
        w_check = trapezoid_weights(data["outlet_s"])
        f_direct = window_fraction(data["outlet_tau"], data["outlet_u_n"], w_check, lo=0.5, hi=1.2)
        assert math.isclose(f_direct, res["scalars"]["f_window"], rel_tol=1e-12)


def test_strain_rate_magnitude_transpose_invariant() -> None:
    """gdot* = sqrt(2 D*:D*) is invariant to grad(U) transpose convention."""
    rng = np.random.default_rng(42)
    g = rng.normal(size=(25, 3, 3))
    gdot1 = strain_rate_magnitude(g)
    gdot2 = strain_rate_magnitude(np.swapaxes(g, -1, -2))
    assert np.allclose(gdot1, gdot2, rtol=1e-14, atol=1e-14)
    # Pure simple shear du_x/dy = gamma => D_xy = D_yx = gamma/2 => 2 D:D = gamma^2 => gdot = |gamma|
    shear = np.zeros((1, 3, 3))
    shear[0, 0, 1] = 6.0
    assert math.isclose(float(strain_rate_magnitude(shear)[0]), 6.0, rel_tol=1e-14)


def test_parse_openfoam_uniform_and_comments(tmp_path: Path) -> None:
    """OpenFOAM parser handles comments and uniform fields cleanly."""
    p = tmp_path / "U"
    p.write_text(
        "// leading comment\n"
        "FoamFile { version 2.0; format ascii; class volVectorField; object U; }\n"
        "/* block comment */\n"
        "internalField uniform (1 2 3);\n"
        "boundaryField { walls { type noSlip; value uniform (0 0 0); } }\n",
        encoding="utf-8",
    )
    parsed = parse_openfoam_field(p)
    kind, arr = parsed["internal"]
    assert kind == "uniform"
    assert np.allclose(arr, [1.0, 2.0, 3.0])
    assert parsed["boundary"]["walls"]["type"] == "noSlip"


# ---------------------------------------------------------------------------
# 3. Manifest & resume logic tests on fake runs
# ---------------------------------------------------------------------------

def test_run_batch_manifest_cleanup_and_resume(tmp_path: Path) -> None:
    """Batch runner writes manifest + sha256, deletes ok cases, keeps failed cases, and resumes."""
    design_path = tmp_path / "design.jsonl"
    recs = [
        {
            "run_id": "FAKE_G2_000",
            "case_id": "FAKE_G2_A",
            "family": "G2",
            "rheology": "Newtonian",
            "Re": 10.0,
            "beta": None,
            "Cu": None,
            "n": None,
            "a": None,
            "level": 0,
            "geometry": {"E": 2.0, "L_in": 5.0, "L_out_ratio": 10.0},
        },
        {
            "run_id": "FAKE_G2_001",
            "case_id": "FAKE_G2_B",
            "family": "G2",
            "rheology": "BirdCarreau",
            "Re": 12.0,
            "beta": 16.0,
            "Cu": 50.0,
            "n": 0.35,
            "a": 2.0,
            "level": 0,
            "geometry": {"E": 2.5, "L_in": 5.0, "L_out_ratio": 10.0},
        },
    ]
    design_path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")

    # Fake GCS bucket backed by a local directory.
    fake_bucket = tmp_path / "fake_gcs"
    fake_bucket.mkdir()

    def fake_uri_to_path(uri: str) -> Path:
        rel = uri.replace("gs://<corpus-bucket>/", "")
        return fake_bucket / rel

    def fake_upload(local_path: Path, uri: str) -> None:
        dst = fake_uri_to_path(uri)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dst)

    def fake_download(uri: str, local_path: Path) -> bool:
        src = fake_uri_to_path(uri)
        if not src.is_file():
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)
        return True

    invocations: list[str] = []
    fail_second = {"enabled": True}

    def fake_runner(case_dir: Path, _timeout: float) -> tuple[bool, str | None]:
        meta = json.loads((case_dir / "params.json").read_text(encoding="utf-8"))
        run_id = meta["run_id"]
        invocations.append(run_id)
        if run_id == "FAKE_G2_001" and fail_second["enabled"]:
            return False, "simulated solver divergence"
        _populate_fake_g2_run(
            case_dir,
            E=float(meta["geometry"]["E"]),
            Re=float(meta["Re"]),
        )
        return True, None

    # Pass 1: run 0 succeeds, run 1 fails.
    work1 = tmp_path / "work1"
    s1 = run_batch(
        design_path,
        work_dir=work1,
        workers=2,
        upload=True,
        runner_fn=fake_runner,
        uploader_fn=fake_upload,
        downloader_fn=fake_download,
    )
    assert s1["succeeded"] == 1
    assert s1["failed"] == 1
    assert s1["skipped"] == 0

    # Successful case dir deleted; failed case dir preserved for inspection.
    assert not (work1 / "cases" / "G2" / "FAKE_G2_000").exists()
    assert (work1 / "cases" / "G2" / "FAKE_G2_001").is_dir()

    # Uploaded NPZ exists and SHA-256 matches manifest.
    uploaded_npz = fake_bucket / "corpus/v1/runs/m1/G2/FAKE_G2_000.npz"
    uploaded_manifest = fake_bucket / "corpus/v1/manifest.jsonl"
    assert uploaded_npz.is_file()
    assert uploaded_manifest.is_file()
    rows1 = load_manifest(uploaded_manifest)
    ok_rows = [r for r in rows1 if r["run_id"] == "FAKE_G2_000"]
    fail_rows = [r for r in rows1 if r["run_id"] == "FAKE_G2_001"]
    assert len(ok_rows) == 1 and ok_rows[0]["status"] == "ok"
    assert ok_rows[0]["sha256"] == compute_sha256(uploaded_npz)
    assert len(fail_rows) == 1 and fail_rows[0]["status"] == "failed"
    assert "simulated solver divergence" in fail_rows[0]["reason"]

    # Pass 2 (resume on fresh work_dir): run 0 MUST be skipped; run 1 retried and succeeds.
    invocations.clear()
    fail_second["enabled"] = False
    work2 = tmp_path / "work2"
    s2 = run_batch(
        design_path,
        work_dir=work2,
        workers=2,
        upload=True,
        runner_fn=fake_runner,
        uploader_fn=fake_upload,
        downloader_fn=fake_download,
    )
    assert s2["skipped"] == 1
    assert s2["attempted"] == 1
    assert s2["succeeded"] == 1
    assert invocations == ["FAKE_G2_001"]
    assert (fake_bucket / "corpus/v1/runs/m1/G2/FAKE_G2_001.npz").is_file()


# ---------------------------------------------------------------------------
# 4. Roache GCI analytic unit tests & manifest integration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p_true", [1.0, 1.5, 2.0])
def test_roache_gci_exact_analytic_order(p_true: float) -> None:
    """Roache 3-level GCI recovers exact observed order p, f_exact, GCI_fine, and AR ~ 1."""
    r = 1.5
    fs = 1.25
    f_exact = 250.0
    c = 12.0
    h_fine = 0.05
    f_fine = f_exact + c * (h_fine**p_true)
    f_med = f_exact + c * ((r * h_fine) ** p_true)
    f_coarse = f_exact + c * ((r * r * h_fine) ** p_true)

    res = roache_gci(f_coarse, f_med, f_fine, r=r, fs=fs, threshold=0.02)
    assert res["convergence"] == "monotonic"
    assert math.isclose(res["p"], p_true, rel_tol=1e-11)
    assert math.isclose(res["f_ext"], f_exact, rel_tol=1e-11)

    expected_ea21 = abs((f_med - f_fine) / f_fine)
    expected_gci_fine = fs * expected_ea21 / (r**p_true - 1.0)
    assert math.isclose(res["gci_fine"], expected_gci_fine, rel_tol=1e-12)
    assert 0.95 <= res["asymptotic_ratio"] <= 1.05


def test_roache_gci_oscillatory_and_divergent() -> None:
    """Oscillatory and divergent sequences are handled cleanly without exceptions."""
    osc = roache_gci(10.4, 9.8, 10.05, r=1.5, fs=1.25)
    assert osc["convergence"] == "oscillatory"
    assert osc["p"] is not None and osc["gci_fine"] is not None

    div = roache_gci(10.0, 10.1, 10.5, r=1.5, fs=1.25, threshold=0.02)
    assert div["convergence"] == "divergent"
    assert div["p"] is None
    assert div["pass_threshold"] is False


def test_gci_from_manifest_json_and_table(tmp_path: Path) -> None:
    """run_gci_from_manifest writes corpus/v1/gci/<family>/<case>.json and a Markdown table."""
    r = 1.5
    h = 0.04
    manifest = tmp_path / "manifest.jsonl"
    entries = []
    for lvl, mult in ((0, r * r), (1, r), (2, 1.0)):
        entries.append(
            {
                "run_id": f"GCI_G2_inertial_L{lvl}",
                "case_id": "GCI_G2_inertial",
                "family": "G2",
                "level": lvl,
                "status": "ok",
                "params": {"n_cells": int(round(2000 * (2.25**lvl)))},
                "scalars": {
                    "dp_star": 180.0 + 4.0 * (h * mult) ** 2,
                    "xr_over_h1": 2.40 - 0.5 * (h * mult) ** 2,
                    "max_wall_tau_star": 8.5 + 0.4 * (h * mult) ** 2,
                    "f_window": 0.32 + 0.08 * (h * mult) ** 2,
                },
            }
        )
    manifest.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    # Direct case analysis
    rep = analyze_case_gci(entries)
    assert rep["all_passed"] is True
    assert math.isclose(rep["scalars"]["dp_star"]["p"], 2.0, rel_tol=1e-10)
    assert math.isclose(rep["scalars"]["xr_over_h1"]["f_ext"], 2.40, rel_tol=1e-10)

    # End-to-end file & summary output
    out_dir = tmp_path / "gci_out"
    reports, table = run_gci_from_manifest(manifest, out_dir)
    assert len(reports) == 1
    out_json = out_dir / "corpus" / "v1" / "gci" / "G2" / "GCI_G2_inertial.json"
    assert out_json.is_file()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["case_id"] == "GCI_G2_inertial"
    assert "GCI_G2_inertial" in table
    assert "PASS" in format_summary_table(reports)


# ---------------------------------------------------------------------------
# 5. GCI design generator (10 cases x 3 levels = 30 runs)
# ---------------------------------------------------------------------------

def test_gci_design_10_cases_30_runs_valid_meshes() -> None:
    """design_gci generates 10 extreme-corner cases (2/family G2-G6) x 3 levels = 30 valid meshes."""
    cases = generate_gci_cases()
    assert len(cases) == 10
    fams = [c["family"] for c in cases]
    for fam in ("G2", "G3", "G4", "G5", "G6"):
        assert fams.count(fam) == 2

    records = generate_gci_design()
    assert len(records) == 30
    assert len({r["run_id"] for r in records}) == 30

    for rec in records:
        if rec["corner"] == "inertial_max_Re":
            assert rec["Re"] == 25.0
            assert rec["rheology"] == "Newtonian"
        else:
            assert rec["rheology"] == "BirdCarreau"
            assert rec["beta"] == 30.0
            assert rec["n"] == 0.25
            assert 1.0 <= rec["Cu"] <= 10.0
        mesh = build_geometry(rec["family"], rec["geometry"], level=rec["level"])
        assert len(mesh.blocks) >= 1


def test_extract_case_g6_odd_bend_dp_and_outlet_normal(tmp_path: Path) -> None:
    """G6 with n_bends=1 isolates inlet dp* from the -x outlet leg and flips outlet normal to -x."""
    Rc = 2.0
    Re = 5.0
    record: dict[str, Any] = {
        "run_id": "TEST_G6_odd",
        "family": "G6",
        "rheology": "Newtonian",
        "Re": Re,
        "beta": None,
        "Cu": None,
        "n": None,
        "a": None,
        "geometry": {"Rc_ratio": Rc, "n_bends": 1, "L_in": 5.0, "L_out_ratio": 10.0},
    }
    case_dir = tmp_path / "case_g6"
    write_case(case_dir, record, level=0)

    t_dir = case_dir / "100"
    t_dir.mkdir(parents=True, exist_ok=True)
    # Inlet cells on x in [-5, 0], y in [0, 1] with p* = 200 - 10*(x + 5) (so p*(x=-5) = 200).
    xs_in = np.linspace(-4.75, -0.25, 10)
    ys_in = np.linspace(0.1, 0.9, 5)
    xx_i, yy_i = np.meshgrid(xs_in, ys_in)
    p_in = 200.0 - 10.0 * (xx_i.ravel() + 5.0)

    # Outlet leg on x in [-10, 0], y in [2*Rc, 2*Rc + 1] with p* = 0 near x = -10 and ux < 0!
    xs_out = np.linspace(-9.75, -0.25, 10)
    ys_out = np.linspace(2.0 * Rc + 0.1, 2.0 * Rc + 0.9, 5)
    xx_o, yy_o = np.meshgrid(xs_out, ys_out)
    p_out = 5.0 * (xx_o.ravel() + 10.0)

    x_c = np.concatenate([xx_i.ravel(), xx_o.ravel()])
    y_c = np.concatenate([yy_i.ravel(), yy_o.ravel()])
    z_c = np.full_like(x_c, 0.5)
    p_star = np.concatenate([p_in, p_out])
    ux = np.concatenate([np.ones(xx_i.size), -np.ones(xx_o.size)])
    n_cells = x_c.size

    grad_u = np.zeros((n_cells, 3, 3), dtype=np.float64)
    grad_u[:, 0, 1] = 2.5

    _write_foam_vector_field(t_dir / "C", np.column_stack([x_c, y_c, z_c]), obj_name="C")
    _write_foam_scalar_field(
        t_dir / "V",
        np.full(n_cells, 0.1),
        obj_name="V",
        foam_class="volScalarField::Internal",
        internal_keyword="value",
    )
    _write_foam_vector_field(
        t_dir / "U", np.column_stack([ux, np.zeros(n_cells), np.zeros(n_cells)]), obj_name="U"
    )
    _write_foam_scalar_field(t_dir / "p", p_star / Re, obj_name="p", boundary_block="boundaryField { }")
    _write_foam_tensor_field(t_dir / "grad(U)", grad_u, obj_name="grad(U)")

    samp_dir = case_dir / "postProcessing" / "sample" / "100"
    samp_dir.mkdir(parents=True, exist_ok=True)
    y_prof = np.linspace(2.0 * Rc + 0.02, 2.0 * Rc + 0.98, 40)
    # Outlet flows in -x, so Ux is negative; outward normal (-1, 0) yields positive u_n!
    ux_prof = -6.0 * (y_prof - 2.0 * Rc) * (1.0 - (y_prof - 2.0 * Rc))
    np.savetxt(
        samp_dir / "outletProfile_p_U.xy",
        np.column_stack([y_prof, np.zeros_like(y_prof), ux_prof, np.zeros_like(y_prof), np.zeros_like(y_prof)]),
    )
    (case_dir / "log.checkMesh").write_text(f"cells: {n_cells}\nMesh OK.\n", encoding="utf-8")
    (case_dir / "log.simpleFoam").write_text(
        "Build : v2306\nTime = 100\nSIMPLE solution converged in 100 iterations\n",
        encoding="utf-8",
    )

    res = extract_case(
        case_dir,
        record,
        level=0,
        wall_seconds=1.0,
        out_npz_path=tmp_path / "g6.npz",
        tau_lo=2.0,
        tau_hi=3.0,
    )
    assert math.isclose(res["scalars"]["dp_star"], 200.0, rel_tol=1e-6)
    assert res["scalars"]["xr_over_h1"] is None
    assert math.isclose(res["scalars"]["f_window"], 1.0, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# 6. T5 regressions: run_case.sh crash guard, non-ortho hook, QoI convergence
# ---------------------------------------------------------------------------

_REAL_OF2306_BANNER = """\
/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2306                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
Build  : _fbf00d6b-20230626 OPENFOAM=2306 version=2306
Arch   : "LSB;label=32;scalar=64"
Exec   : simpleFoam
Date   : Nov 24 2026
Time   : 15:20:21
Host   : rbc-team-2
PID    : 12345
I/O    : uncollated
Case   : /home/user/smoke/G4
nProcs : 1
trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).
fileModificationChecking : Monitoring run-time modified files using timeStampMaster (fileModificationSkew 5, maxFileModificationPolls 20)
allowSystemOperations : Allowing user-supplied system call operations
"""

_RUN_CASE_SH = Path(__file__).resolve().parents[1] / "cfd" / "gen" / "run_case.sh"


def test_run_case_crash_guard_passes_real_banner(tmp_path: Path) -> None:
    """Crash guard in run_case.sh must PASS (exit 0) on OpenFOAM v2306 trapFpe banner."""
    log_ok = tmp_path / "log.simpleFoam"
    log_ok.write_text(
        _REAL_OF2306_BANNER
        + "\nTime = 104\n"
        + "smoothSolver:  Solving for Ux, Initial residual = 6.2e-06, Final residual = 4.1e-07, No Iterations 2\n"
        + "smoothSolver:  Solving for Uy, Initial residual = 7.4e-06, Final residual = 5.0e-07, No Iterations 2\n"
        + "GAMG:  Solving for p, Initial residual = 8.9e-06, Final residual = 7.2e-08, No Iterations 5\n"
        + "time step continuity errors : sum local = 1.0e-09, global = -1.2e-11, cumulative = 3.4e-08\n"
        + "SIMPLE solution converged in 104 iterations\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["bash", str(_RUN_CASE_SH), "--check-log", str(log_ok)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Expected exit 0 on real banner, got {proc.returncode}: {proc.stderr}"


@pytest.mark.parametrize(
    ("label", "crash_snippet"),
    [
        (
            "sigFpe_traceback",
            "#0  Foam::error::printStack(Foam::Ostream&) at ??:?\n"
            "#1  Foam::sigFpe::sigHandler(int) at ??:?\n",
        ),
        (
            "core_dumped_fpe",
            "Floating point exception (core dumped)\n",
        ),
        (
            "foam_fatal",
            "--> FOAM FATAL ERROR: (openfoam-2306)\n    Maximum number of iterations exceeded\n",
        ),
        (
            "nan_in_residual",
            "smoothSolver:  Solving for Ux, Initial residual = -nan, Final residual = nan, No Iterations 0\n",
        ),
        (
            "inf_in_continuity",
            "time step continuity errors : sum local = inf, global = nan, cumulative = nan\n",
        ),
    ],
)
def test_run_case_crash_guard_fails_on_real_crashes(
    tmp_path: Path, label: str, crash_snippet: str
) -> None:
    """Crash guard in run_case.sh must FAIL (exit 4) on real FPE, FOAM FATAL, or residual NaN/Inf."""
    log_bad = tmp_path / f"log_{label}.simpleFoam"
    log_bad.write_text(_REAL_OF2306_BANNER + "\nTime = 12\n" + crash_snippet, encoding="utf-8")
    proc = subprocess.run(
        ["bash", str(_RUN_CASE_SH), "--check-log", str(log_bad)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 4, f"{label}: expected exit 4, got {proc.returncode}"


@pytest.mark.parametrize(
    ("family", "max_non_ortho", "expected_corr"),
    [
        ("G2", 52.0, 0),   # orthogonal family stays 0 even if synthetic checkMesh > 40
        ("G4", 35.0, 0),   # bifurcation <= 40 deg stays 0
        ("G4", 48.5, 1),   # bifurcation > 40 deg switches to 1
        ("G6", 42.1, 1),   # serpentine > 40 deg switches to 1
        ("G7", 39.9, 0),   # network cell <= 40 deg stays 0
    ],
)
def test_configure_non_ortho_correctors_threshold(
    tmp_path: Path, family: str, max_non_ortho: float, expected_corr: int
) -> None:
    """run_case.sh sets nNonOrthogonalCorrectors 1 only for G4-G7 when checkMesh non-ortho > 40."""
    geom_map: dict[str, dict[str, Any]] = {
        "G2": {"E": 2.0},
        "G4": {"theta_deg": 45.0, "daughter_ratio": 0.75},
        "G6": {"Rc_ratio": 2.5, "n_bends": 2},
        "G7": {"E": 2.0, "theta_deg": 45.0},
    }
    case_dir = tmp_path / f"case_{family}_{max_non_ortho}"
    write_case(
        case_dir,
        {"run_id": "t", "family": family, "rheology": "Newtonian", "Re": 5.0, "geometry": geom_map[family]},
        level=0,
    )
    (case_dir / "log.checkMesh").write_text(
        f"Checking geometry...\n    Mesh non-orthogonality Max: {max_non_ortho} average: 5.0\nMesh OK.\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["bash", str(_RUN_CASE_SH), "--configure-non-ortho", str(case_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    fv_sol = (case_dir / "system" / "fvSolution").read_text(encoding="utf-8")
    assert f"nNonOrthogonalCorrectors {expected_corr};" in fv_sol


def test_outlet_centreline_probes_inside_fluid_all_families() -> None:
    """Every family G2-G7 & KG places all QoI outlet-centreline probe(s) strictly inside the fluid."""
    specs = [
        ("G2", {"E": 2.0}),
        ("G3", {"throat_ratio": 0.5, "L_t": 2.0}),
        ("G4", {"theta_deg": 45.0, "daughter_ratio": 0.75}),
        ("G5", {"width_ratio": 0.65, "theta_deg": 60.0}),
        ("G6", {"Rc_ratio": 2.5, "n_bends": 1}),
        ("G6", {"Rc_ratio": 2.5, "n_bends": 2}),
        ("G7", {"E": 2.0, "theta_deg": 45.0}),
        ("KG", {"E": 252.0 / 75.5}),
    ]
    for fam, gparams in specs:
        mesh = build_geometry(fam, gparams, level=0)
        probes = _outlet_centreline_probes(mesh)
        expected_n = 2 if fam in ("G4", "G5", "G7") else 1
        assert len(probes) == expected_n
        for px, py, _pz in probes:
            mask = build_fluid_mask(mesh, np.array([px]), np.array([py]))
            assert bool(mask[0, 0]), f"{fam} probe ({px}, {py}) lies outside fluid mask"


def _write_qoi_monitors(
    case_dir: Path,
    iters: np.ndarray,
    dp_vals: np.ndarray,
    u_mags: np.ndarray,
) -> None:
    """Write synthetic postProcessing/{inletPressure,qoiProbes} files matching OpenFOAM v2306 format."""
    p_dir = case_dir / "postProcessing" / "inletPressure" / "0"
    p_dir.mkdir(parents=True, exist_ok=True)
    p_lines = ["# Region : patch inlet", "# Time  areaAverage(p)"]
    p_lines.extend(f"{int(it)}  {val:.10e}" for it, val in zip(iters, dp_vals, strict=True))
    (p_dir / "surfaceFieldValue.dat").write_text("\n".join(p_lines) + "\n", encoding="utf-8")

    u_dir = case_dir / "postProcessing" / "qoiProbes" / "0"
    u_dir.mkdir(parents=True, exist_ok=True)
    u_lines = ["# Probe 0 (19 0.5 0.5)", "# Time  0"]
    u_lines.extend(f"{int(it)}  ({val:.10e} 0 0)" for it, val in zip(iters, u_mags, strict=True))
    (u_dir / "U").write_text("\n".join(u_lines) + "\n", encoding="utf-8")


def test_evaluate_convergence_residual_qoi_and_not_converged(tmp_path: Path) -> None:
    """evaluate_convergence distinguishes residual (<1e-5), qoi plateau, and not_converged."""
    # 1. Criterion (a): residuals < 1e-5 -> converged_by == "residual"
    c_res = tmp_path / "c_res"
    c_res.mkdir()
    (c_res / "log.simpleFoam").write_text(
        _REAL_OF2306_BANNER
        + "\nTime = 320\n"
        + "smoothSolver:  Solving for Ux, Initial residual = 4.2e-06, Final residual = 3.1e-07\n"
        + "smoothSolver:  Solving for Uy, Initial residual = 8.1e-06, Final residual = 5.4e-07\n"
        + "GAMG:  Solving for p, Initial residual = 9.5e-06, Final residual = 9.0e-08\n"
        + "time step continuity errors : sum local = 2.0e-09, global = 1e-11, cumulative = 1e-08\n"
        + "SIMPLE solution converged in 320 iterations\n",
        encoding="utf-8",
    )
    ev_res = evaluate_convergence(c_res)
    assert ev_res["converged"] is True
    assert ev_res["converged_by"] == "residual"

    # 2. Criterion (b): residuals ~4e-4 (<1e-3), continuity 3e-9 (<1e-8), QoI steady <1e-5 over last 500
    c_qoi = tmp_path / "c_qoi"
    c_qoi.mkdir()
    (c_qoi / "log.simpleFoam").write_text(
        _REAL_OF2306_BANNER
        + "\nTime = 5000\n"
        + "smoothSolver:  Solving for Ux, Initial residual = 2.1e-04, Final residual = 2.0e-05\n"
        + "smoothSolver:  Solving for Uy, Initial residual = 4.5e-04, Final residual = 4.0e-05\n"
        + "GAMG:  Solving for p, Initial residual = 3.8e-04, Final residual = 3.0e-06\n"
        + "time step continuity errors : sum local = 3.2e-09, global = -1.1e-11, cumulative = 5.0e-08\n",
        encoding="utf-8",
    )
    iters = np.arange(4400, 5001, dtype=float)
    dp_steady = 25.0 * (1.0 + 2.0e-6 * np.exp(-(iters - 4500.0) / 200.0))
    u_steady = 0.75 * (1.0 - 1.5e-6 * np.exp(-(iters - 4500.0) / 200.0))
    _write_qoi_monitors(c_qoi, iters, dp_steady, u_steady)
    ev_qoi = evaluate_convergence(c_qoi)
    assert ev_qoi["converged"] is True
    assert ev_qoi["converged_by"] == "qoi"
    assert ev_qoi["dp_rel_change_500"] < 1e-5
    assert ev_qoi["u_out_rel_change_500"] < 1e-5

    # 3. Failing criterion (b) due to drifting QoI (> 1e-5) -> not converged
    c_drift = tmp_path / "c_drift"
    shutil.copytree(c_qoi, c_drift)
    dp_drift = 25.0 * (1.0 + 5.0e-4 * (iters - 4500.0) / 500.0)
    _write_qoi_monitors(c_drift, iters, dp_drift, u_steady)
    ev_drift = evaluate_convergence(c_drift)
    assert ev_drift["converged"] is False
    assert ev_drift["converged_by"] is None

    # 4. Failing criterion (b) due to high residual (>= 1e-3) -> not converged
    c_hi_res = tmp_path / "c_hi_res"
    shutil.copytree(c_qoi, c_hi_res)
    (c_hi_res / "log.simpleFoam").write_text(
        _REAL_OF2306_BANNER
        + "\nTime = 5000\n"
        + "smoothSolver:  Solving for Ux, Initial residual = 9.4e-04, Final residual = 8.0e-05\n"
        + "smoothSolver:  Solving for Uy, Initial residual = 4.3e-03, Final residual = 3.0e-04\n"
        + "GAMG:  Solving for p, Initial residual = 2.6e-03, Final residual = 2.0e-05\n"
        + "time step continuity errors : sum local = 3.2e-09, global = -1.1e-11, cumulative = 5.0e-08\n",
        encoding="utf-8",
    )
    ev_hi_res = evaluate_convergence(c_hi_res)
    assert ev_hi_res["converged"] is False

    # 5. Failing criterion (b) due to continuity error (>= 1e-8) -> not converged
    c_hi_cont = tmp_path / "c_hi_cont"
    shutil.copytree(c_qoi, c_hi_cont)
    (c_hi_cont / "log.simpleFoam").write_text(
        _REAL_OF2306_BANNER
        + "\nTime = 5000\n"
        + "smoothSolver:  Solving for Ux, Initial residual = 2.1e-04, Final residual = 2.0e-05\n"
        + "smoothSolver:  Solving for Uy, Initial residual = 4.5e-04, Final residual = 4.0e-05\n"
        + "GAMG:  Solving for p, Initial residual = 3.8e-04, Final residual = 3.0e-06\n"
        + "time step continuity errors : sum local = 5.0e-08, global = -1.1e-10, cumulative = 5.0e-07\n",
        encoding="utf-8",
    )
    ev_hi_cont = evaluate_convergence(c_hi_cont)
    assert ev_hi_cont["converged"] is False


def test_run_batch_records_converged_by_and_rejects_not_converged(tmp_path: Path) -> None:
    """run_batch writes converged_by=residual|qoi for ok runs and status=failed, reason=not_converged otherwise."""
    design_path = tmp_path / "design.jsonl"
    base_geom = {"E": 2.0, "L_in": 5.0, "L_out_ratio": 10.0}
    recs = [
        {
            "run_id": "CONV_RES",
            "family": "G2",
            "rheology": "Newtonian",
            "Re": 10.0,
            "level": 0,
            "geometry": base_geom,
        },
        {
            "run_id": "CONV_QOI",
            "family": "G2",
            "rheology": "BirdCarreau",
            "Re": 8.0,
            "beta": 16.0,
            "Cu": 500.0,
            "n": 0.35,
            "a": 2.0,
            "level": 0,
            "geometry": base_geom,
        },
        {
            "run_id": "CONV_FAIL",
            "family": "G2",
            "rheology": "BirdCarreau",
            "Re": 8.0,
            "beta": 16.0,
            "Cu": 500.0,
            "n": 0.35,
            "a": 2.0,
            "level": 0,
            "geometry": base_geom,
        },
    ]
    design_path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")

    iters = np.arange(4400, 5001, dtype=float)
    dp_steady = np.full_like(iters, 30.0) + 1e-7 * np.sin(iters)
    u_steady = np.full_like(iters, 0.75) + 1e-7 * np.cos(iters)
    dp_drifting = 30.0 + 0.05 * (iters - 4400.0) / 600.0

    def fake_runner(case_dir: Path, _timeout: float) -> tuple[bool, str | None]:
        meta = json.loads((case_dir / "params.json").read_text(encoding="utf-8"))
        run_id = meta["run_id"]
        _populate_fake_g2_run(case_dir, E=2.0, Re=float(meta["Re"]))
        if run_id == "CONV_QOI":
            (case_dir / "log.simpleFoam").write_text(
                _REAL_OF2306_BANNER
                + "\nTime = 5000\n"
                + "smoothSolver:  Solving for Ux, Initial residual = 3.0e-04, Final residual = 2.0e-05\n"
                + "smoothSolver:  Solving for Uy, Initial residual = 4.0e-04, Final residual = 3.0e-05\n"
                + "GAMG:  Solving for p, Initial residual = 5.0e-04, Final residual = 4.0e-06\n"
                + "time step continuity errors : sum local = 2.5e-09, global = 1e-11, cumulative = 1e-08\n",
                encoding="utf-8",
            )
            _write_qoi_monitors(case_dir, iters, dp_steady, u_steady)
        elif run_id == "CONV_FAIL":
            (case_dir / "log.simpleFoam").write_text(
                _REAL_OF2306_BANNER
                + "\nTime = 5000\n"
                + "smoothSolver:  Solving for Ux, Initial residual = 3.0e-04, Final residual = 2.0e-05\n"
                + "smoothSolver:  Solving for Uy, Initial residual = 4.0e-04, Final residual = 3.0e-05\n"
                + "GAMG:  Solving for p, Initial residual = 5.0e-04, Final residual = 4.0e-06\n"
                + "time step continuity errors : sum local = 2.5e-09, global = 1e-11, cumulative = 1e-08\n",
                encoding="utf-8",
            )
            _write_qoi_monitors(case_dir, iters, dp_drifting, u_steady)
        return True, None

    work = tmp_path / "work"
    summary = run_batch(
        design_path,
        work_dir=work,
        workers=2,
        upload=False,
        runner_fn=fake_runner,
    )
    assert summary["succeeded"] == 2
    assert summary["failed"] == 1

    rows = {r["run_id"]: r for r in load_manifest(work / "manifest.jsonl")}
    assert rows["CONV_RES"]["status"] == "ok"
    assert rows["CONV_RES"]["converged_by"] == "residual"
    assert rows["CONV_QOI"]["status"] == "ok"
    assert rows["CONV_QOI"]["converged_by"] == "qoi"
    assert rows["CONV_FAIL"]["status"] == "failed"
    assert rows["CONV_FAIL"]["reason"] == "not_converged"
    assert rows["CONV_FAIL"]["converged_by"] is None
    assert rows["CONV_FAIL"]["sha256"] is None
    # Failed case dir kept locally; converged case dirs cleaned up.
    assert (work / "cases" / "G2" / "CONV_FAIL").is_dir()
    assert not (work / "cases" / "G2" / "CONV_RES").exists()
    assert not (work / "cases" / "G2" / "CONV_QOI").exists()
