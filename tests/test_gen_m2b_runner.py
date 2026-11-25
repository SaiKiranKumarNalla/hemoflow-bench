"""Unit tests for T6: M2b-A/M2b-B run_batch extension, CFL/phi-weighted extraction, and M2b-B lift generator."""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from campaign.design_m2b import generate_m2b_design
from campaign.run_batch import (
    FAMILY_GRID_SHAPES,
    compute_cfl_thickness_star,
    evaluate_m2b_convergence,
    extract_case,
    load_manifest,
    parse_openfoam_field,
    read_m2b_sample_xy,
    recompute_window_fraction,
    run_batch,
)
from cfd.gen.m2b import write_m2b_case
from cfd.gen.m2b_lift import (
    K_LEGENDRE_MAGNAUDET,
    K_SAFFMAN_SPHERE,
    K_WU_EQ10,
    SUPPORTED_LIFT_MODELS,
    generate_m2b_lift_design,
    legendre_magnaudet_cl_star,
    saffman_constant_cl,
    write_m2b_lift_case,
    write_m2b_lift_design_jsonl,
)
from rbc.postprocess import trapezoid_weights, window_fraction


# ---------------------------------------------------------------------------
# Helpers for OpenFOAM dictionary verification & synthetic M2b solution output
# ---------------------------------------------------------------------------

EXPECTED_M2B_LIFT_FILES = (
    "system/blockMeshDict",
    "system/controlDict",
    "system/fvSchemes",
    "system/fvSolution",
    "constant/phaseProperties",
    "constant/thermophysicalProperties.plasma",
    "constant/thermophysicalProperties.RBC",
    "constant/turbulenceProperties.plasma",
    "constant/turbulenceProperties.RBC",
    "constant/g",
    "0/alpha.RBC",
    "0/alpha.plasma",
    "0/U.RBC",
    "0/U.plasma",
    "0/p_rgh",
    "0/p",
    "0/T.RBC",
    "0/T.plasma",
    "run_case.sh",
    "params.json",
)


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _assert_balanced_foam_dict(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    assert "FoamFile" in text, f"Missing FoamFile header in {path}"
    clean = _strip_comments(text)
    clean_no_quotes = re.sub(r'"[^"]*"', '""', clean)
    for open_ch, close_ch in (("{", "}"), ("(", ")")):
        depth = 0
        for ch in clean_no_quotes:
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                assert depth >= 0, f"Unbalanced '{close_ch}' in {path}"
        assert depth == 0, f"Unclosed '{open_ch}' in {path} (depth={depth})"
    return clean


def _write_scalar(
    path: Path,
    values: np.ndarray,
    *,
    obj_name: str,
    foam_class: str = "volScalarField",
    internal_keyword: str = "internalField",
    boundary_block: str = "boundaryField { }",
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


def _write_vector(path: Path, values: np.ndarray, *, obj_name: str) -> None:
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


def _write_tensor(
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


def _write_surface_series(case_dir: Path, func_name: str, times: np.ndarray, vals: np.ndarray) -> None:
    d = case_dir / "postProcessing" / func_name / "0"
    d.mkdir(parents=True, exist_ok=True)
    lines = [f"# {func_name}", "# Time  value"]
    lines.extend(f"{t:.6g}  {v:.10e}" for t, v in zip(times, vals, strict=True))
    (d / "surfaceFieldValue.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _populate_fake_m2b_run(
    case_dir: Path,
    *,
    family: str,
    Re: float,
    phi_inlet: float,
    solver: str = "multiphaseEulerFoam",
    end_time: float = 60.0,
    cfl_width: float = 0.12,
    xr_target: float = 1.5,
    dp_grad_star: float = 12.0,
    dp_drift: float = 0.0,
    phi_drift: float = 0.0,
    xr_drift: float = 0.0,
    reached_end: bool = True,
    geometry: dict[str, Any] | None = None,
) -> None:
    """Write synthetic ESI v2306 multiphase fields, logs, monitors, and raw .xy samples."""
    fam = family.upper()
    geom = geometry or {}
    t_str = f"{end_time:g}"
    t_dir = case_dir / t_str
    t_dir.mkdir(parents=True, exist_ok=True)

    if fam == "G1":
        L_ratio = float(geom.get("L_ratio", 20.0))
        xs = np.linspace(0.5, L_ratio - 0.5, 20)
        ys = np.linspace(0.025, 0.975, 20)
        xx, yy = np.meshgrid(xs, ys)
        x_c = xx.ravel()
        y_c = yy.ravel()
        z_c = np.full_like(x_c, 0.5)
        x_out = L_ratio
        y_lo, y_hi = 0.0, 1.0
        eta = np.clip((y_c - y_lo) / (y_hi - y_lo), 0.0, 1.0)
        ux_pla = 6.0 * eta * (1.0 - eta)
        dux_dy = 6.0 * (1.0 - 2.0 * eta)
    elif fam == "G2":
        E = float(geom.get("E", 2.0))
        L_in = float(geom.get("L_in", 5.0))
        L_out = float(geom.get("L_out_ratio", 10.0)) * E
        s = 0.5 * (E - 1.0)
        xs_i = np.linspace(-L_in + 0.25, -0.25, 10)
        ys_i = np.linspace(0.05, 0.95, 10)
        xx_i, yy_i = np.meshgrid(xs_i, ys_i)
        xs_o = np.linspace(0.25, L_out - 0.25, 20)
        ys_o = np.linspace(-s + 0.05, 1.0 + s - 0.05, 20)
        xx_o, yy_o = np.meshgrid(xs_o, ys_o)
        x_c = np.concatenate([xx_i.ravel(), xx_o.ravel()])
        y_c = np.concatenate([yy_i.ravel(), yy_o.ravel()])
        z_c = np.full_like(x_c, 0.5)
        x_out = L_out
        y_lo, y_hi = -s, 1.0 + s
        eta = np.clip((y_c - y_lo) / E, 0.0, 1.0)
        ux_pla = (6.0 / E) * eta * (1.0 - eta)
        dux_dy = (6.0 / (E * E)) * (1.0 - 2.0 * eta)
    elif fam == "KG":
        E = float(geom.get("E", 252.0 / 75.5))
        L_in = float(geom.get("L_in", 1000.0 / 151.0))
        L_out = float(geom.get("L_out", 5000.0 / 151.0))
        r2 = 0.5 * E
        xs_i = np.linspace(-L_in + 0.3, -0.3, 10)
        rs_i = np.linspace(0.05, 0.45, 8)
        xx_i, rr_i = np.meshgrid(xs_i, rs_i)
        xs_o = np.linspace(0.5, L_out - 0.5, 20)
        rs_o = np.linspace(0.05, r2 - 0.05, 16)
        xx_o, rr_o = np.meshgrid(xs_o, rs_o)
        x_c = np.concatenate([xx_i.ravel(), xx_o.ravel()])
        y_c = np.concatenate([rr_i.ravel(), rr_o.ravel()])
        z_c = np.zeros_like(x_c)
        x_out = L_out
        y_lo, y_hi = 0.0, r2
        ux_pla = (2.0 / (E * E)) * np.maximum(0.0, 1.0 - (y_c / r2) ** 2)
        dux_dy = -(4.0 / (E * E)) * (y_c / (r2 * r2))
    else:
        raise ValueError(fam)

    n_cells = x_c.size
    p_star_exact = dp_grad_star * (x_out - x_c)
    p_of = p_star_exact / Re

    def _phi_profile(coord: np.ndarray) -> np.ndarray:
        if cfl_width <= 0.0:
            return np.full_like(coord, phi_inlet)
        if fam == "KG":
            dist = np.maximum(0.0, y_hi - coord)
        else:
            dist = np.minimum(np.maximum(0.0, coord - y_lo), np.maximum(0.0, y_hi - coord))
        # Smooth linear ramp across [0, 2*cfl_width] so 0.5*phi_inlet crossing is exact at dist == cfl_width!
        return np.where(dist >= 2.0 * cfl_width, phi_inlet, phi_inlet * (dist / (2.0 * cfl_width)))

    phi_cell = _phi_profile(y_c)
    u_pla = np.column_stack([ux_pla, np.zeros(n_cells), np.zeros(n_cells)])
    u_rbc = np.column_stack([0.98 * ux_pla, np.zeros(n_cells), np.zeros(n_cells)])
    u_mix = phi_cell[:, None] * u_rbc + (1.0 - phi_cell[:, None]) * u_pla

    grad_p = np.zeros((n_cells, 3, 3), dtype=np.float64)
    grad_p[:, 0, 1] = dux_dy
    grad_r = 0.98 * grad_p

    walls_g = np.zeros((4, 3, 3), dtype=np.float64)
    walls_g[:, 0, 1] = [2.0, -7.5, 3.0, 1.5]

    _write_vector(t_dir / "C", np.column_stack([x_c, y_c, z_c]), obj_name="C")
    _write_scalar(
        t_dir / "V",
        np.full(n_cells, 0.05),
        obj_name="V",
        foam_class="volScalarField::Internal",
        internal_keyword="value",
        boundary_block="",
    )
    _write_scalar(t_dir / "alpha.RBC", phi_cell, obj_name="alpha.RBC")
    _write_scalar(t_dir / "alpha.plasma", 1.0 - phi_cell, obj_name="alpha.plasma")
    _write_vector(t_dir / "U.plasma", u_pla, obj_name="U.plasma")
    _write_vector(t_dir / "U.RBC", u_rbc, obj_name="U.RBC")
    if solver == "multiphaseEulerFoam":
        # Stock multiphaseEulerFoam writes mixture U; reactingTwoPhaseEulerFoam does NOT.
        _write_vector(t_dir / "U", u_mix, obj_name="U")
    _write_scalar(
        t_dir / "p",
        p_of,
        obj_name="p",
        boundary_block="boundaryField { outlet { type fixedValue; value uniform 0; } }",
    )
    _write_tensor(t_dir / "grad(U.plasma)", grad_p, obj_name="grad(U.plasma)", walls_values=walls_g)
    _write_tensor(t_dir / "grad(U.RBC)", grad_r, obj_name="grad(U.RBC)", walls_values=walls_g)

    # Write final sample files (split scalar + vector files matching ESI v2306 raw writer).
    samp_dir = case_dir / "postProcessing" / "sample" / t_str
    samp_dir.mkdir(parents=True, exist_ok=True)
    s_out = np.linspace(y_lo + 0.01, y_hi - 0.01, 100)
    phi_out = _phi_profile(s_out)
    if fam == "KG":
        E = float(geom.get("E", 252.0 / 75.5))
        r2 = 0.5 * E
        ux_o_p = (2.0 / (E * E)) * np.maximum(0.0, 1.0 - (s_out / r2) ** 2)
    else:
        H = y_hi - y_lo
        eta_o = np.clip((s_out - y_lo) / H, 0.0, 1.0)
        ux_o_p = (6.0 / H) * eta_o * (1.0 - eta_o)
    ux_o_r = 0.98 * ux_o_p

    np.savetxt(
        samp_dir / "outletProfile_alpha.RBC_alpha.plasma_p.xy",
        np.column_stack([s_out, phi_out, 1.0 - phi_out, np.zeros_like(s_out)]),
    )
    np.savetxt(
        samp_dir / "outletProfile_U.plasma.xy",
        np.column_stack([s_out, ux_o_p, np.zeros_like(s_out), np.zeros_like(s_out)]),
    )
    np.savetxt(
        samp_dir / "outletProfile_U.RBC.xy",
        np.column_stack([s_out, ux_o_r, np.zeros_like(s_out), np.zeros_like(s_out)]),
    )

    # Near-wall line for expansion families (final + steadySample history over last 12 %).
    steady_times = np.linspace(0.88 * end_time, end_time, 7)
    if fam in ("G2", "KG"):
        nw_name = "nearWallLower" if fam == "G2" else "nearWallOuter"
        x_nw = np.linspace(0.02, x_out - 0.02, 160)
        ux_nw = x_nw - xr_target
        np.savetxt(
            samp_dir / f"{nw_name}_U.plasma.xy",
            np.column_stack([x_nw, ux_nw, np.zeros_like(x_nw), np.zeros_like(x_nw)]),
        )
        np.savetxt(
            samp_dir / f"{nw_name}_U.RBC.xy",
            np.column_stack([x_nw, ux_nw, np.zeros_like(x_nw), np.zeros_like(x_nw)]),
        )
        np.savetxt(
            samp_dir / f"{nw_name}_alpha.RBC_p.xy",
            np.column_stack([x_nw, np.full_like(x_nw, phi_inlet), np.zeros_like(x_nw)]),
        )

        for idx, t_snap in enumerate(steady_times):
            frac_prog = idx / max(1, len(steady_times) - 1)
            xr_t = xr_target * (1.0 - xr_drift * (1.0 - frac_prog))
            ss_dir = case_dir / "postProcessing" / "steadySample" / f"{t_snap:.6g}"
            ss_dir.mkdir(parents=True, exist_ok=True)
            ux_t = x_nw - xr_t
            np.savetxt(
                ss_dir / f"{nw_name}_U.plasma.xy",
                np.column_stack([x_nw, ux_t, np.zeros_like(x_nw), np.zeros_like(x_nw)]),
            )
            np.savetxt(
                ss_dir / "outletProfile_alpha.RBC_p.xy",
                np.column_stack([s_out, phi_out, np.zeros_like(s_out)]),
            )

    # Time-series monitors: inletPressure & outletPhi across [0, end_time]
    t_mon = np.linspace(0.0, end_time, 101)
    # Linear drift over last 10 % controlled by dp_drift and phi_drift
    p_in_base = dp_grad_star * (x_out - (0.0 if fam == "G1" else -float(geom.get("L_in", 5.0)))) / Re
    ramp = np.clip((t_mon - 0.9 * end_time) / (0.1 * end_time), 0.0, 1.0)
    dp_series = p_in_base * (1.0 - dp_drift * (1.0 - ramp))
    phi_series = phi_inlet * (1.0 - phi_drift * (1.0 - ramp))
    _write_surface_series(case_dir, "inletPressure", t_mon, dp_series)
    _write_surface_series(case_dir, "outletPhi", t_mon, phi_series)

    (case_dir / "log.checkMesh").write_text(
        f"Mesh stats\n    cells:            {n_cells}\nMesh OK.\n", encoding="utf-8"
    )
    log_lines = [
        "Build  : _fbf00d6b-20230626 OPENFOAM=2306",
        "trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).",
        f"Time = {0.5 * end_time:g}",
        "time step continuity errors : sum local = 1.2e-10, global = 0, cumulative = 1e-09",
        f"Time = {end_time:g}",
        "time step continuity errors : sum local = 8.0e-11, global = 0, cumulative = 2e-09",
    ]
    if reached_end:
        log_lines.append("End")
    (case_dir / f"log.{solver}").write_text("\n".join(log_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. M2b-B (reactingTwoPhaseEulerFoam + lift) case generation & formula tests
# ---------------------------------------------------------------------------

def test_legendre_magnaudet_reduces_to_saffman_scaling_at_low_rep() -> None:
    """Verify ESI v2306 LegendreMagnaudet::Cl() recovers K_LM / sqrt(Re_G) when Sr >> 0.2 Re_p."""
    Re = 4.0
    d_p_star = 0.16
    gdot_star = 6.0
    re_g = Re * (d_p_star ** 2) * gdot_star  # 0.6144
    cl_lm = legendre_magnaudet_cl_star(
        Re=Re, d_p_star=d_p_star, gdot_star=gdot_star, Re_p=1e-4, residual_re=1e-4
    )
    cl_saff_low = K_LEGENDRE_MAGNAUDET / math.sqrt(re_g)
    cl_high = 0.5 * (1e-4 + 16.0) / (1e-4 + 29.0)
    expected = math.sqrt(cl_saff_low ** 2 + cl_high ** 2)
    assert cl_lm == pytest.approx(expected, rel=1e-5)
    # Exact Saffman prefactor relationship: K_LM / K_SAFFMAN_SPHERE == 4/9 (Legendre & Magnaudet 1998)
    assert K_LEGENDRE_MAGNAUDET / K_SAFFMAN_SPHERE == pytest.approx(4.0 / 9.0, rel=1e-3)
    # K_WU_EQ10 == K_SAFFMAN_SPHERE / 4
    assert K_WU_EQ10 == pytest.approx(0.25 * K_SAFFMAN_SPHERE, rel=1e-12)


def test_saffman_constant_cl_star_unit_scaling() -> None:
    """Verify constantCoefficient Saffman-matched Cl* = K / (d_p* * sqrt(Re * gdot*))."""
    cl1 = saffman_constant_cl(Re=4.0, d_p_star=0.16, gdot_star_nom=4.0, prefactor=K_WU_EQ10)
    cl2 = saffman_constant_cl(Re=16.0, d_p_star=0.16, gdot_star_nom=4.0, prefactor=K_WU_EQ10)
    cl3 = saffman_constant_cl(Re=4.0, d_p_star=0.08, gdot_star_nom=4.0, prefactor=K_WU_EQ10)
    assert cl2 == pytest.approx(0.5 * cl1, rel=1e-12)  # ~ 1/sqrt(Re)
    assert cl3 == pytest.approx(2.0 * cl1, rel=1e-12)  # ~ 1/d_p*


@pytest.mark.parametrize("lift_model", SUPPORTED_LIFT_MODELS)
@pytest.mark.parametrize(
    "case_params",
    [
        {
            "run_id": "M2BL_G1_test",
            "family": "G1",
            "Re": 4.0,
            "phi": 0.30,
            "d_p_star": 0.16,
            "geometry": {"L_ratio": 20.0},
        },
        {
            "run_id": "M2BL_G2_test",
            "family": "G2",
            "Re": 8.3,
            "phi": 0.20,
            "d_p_star": 0.08,
            "geometry": {"E": 2.0, "L_in": 10.0, "L_out_ratio": 10.0},
        },
        {
            "run_id": "M2BL_KG_test",
            "family": "KG",
            "Re": 12.2,
            "phi": 0.01,
            "d_p_star": 8.0 / 151.0,
            "inlet_profile": "parabolic",
            "geometry": {
                "E": 252.0 / 75.5,
                "L_in": 1000.0 / 151.0,
                "L_out": 5000.0 / 151.0,
                "wedge_angle_deg": 5.0,
            },
        },
    ],
)
def test_write_m2b_lift_case_files_parse_and_star_units(
    tmp_path: Path, lift_model: str, case_params: dict[str, Any]
) -> None:
    """M2b-B cases generate valid ESI v2306 reactingTwoPhaseEulerFoam dicts & exact star units."""
    rec = {**case_params, "lift_model": lift_model}
    cdir = tmp_path / f"{rec['run_id']}_{lift_model}"
    write_m2b_lift_case(cdir, rec, level=0)

    for rel in EXPECTED_M2B_LIFT_FILES:
        assert (cdir / rel).is_file(), f"Missing {rel}"
        if rel not in ("run_case.sh", "params.json"):
            _assert_balanced_foam_dict(cdir / rel)

    # bash syntax check & trapFpe-safe crash guard
    syn = subprocess.run(["bash", "-n", str(cdir / "run_case.sh")], capture_output=True, text=True)
    assert syn.returncode == 0, syn.stderr

    # Verify alpha sum == 1 and BCs via OpenFOAM field parser
    a_rbc = parse_openfoam_field(cdir / "0" / "alpha.RBC")
    a_pla = parse_openfoam_field(cdir / "0" / "alpha.plasma")
    p_rgh = parse_openfoam_field(cdir / "0" / "p_rgh")
    p_fld = parse_openfoam_field(cdir / "0" / "p")
    t_pla = parse_openfoam_field(cdir / "0" / "T.plasma")

    phi_val = float(rec["phi"])
    assert float(a_rbc["internal"][1]) == pytest.approx(phi_val, abs=1e-12)
    assert float(a_pla["internal"][1]) == pytest.approx(1.0 - phi_val, abs=1e-12)
    assert p_rgh["boundary"]["outlet"]["type"] == "fixedValue"
    assert p_fld["boundary"]["outlet"]["type"] == "calculated"
    assert float(t_pla["internal"][1]) == pytest.approx(300.0)

    # Verify exact star-unit dynamic viscosities & densities in thermophysicalProperties.*
    tp_pla = (cdir / "constant" / "thermophysicalProperties.plasma").read_text(encoding="utf-8")
    tp_rbc = (cdir / "constant" / "thermophysicalProperties.RBC").read_text(encoding="utf-8")
    for txt in (tp_pla, tp_rbc):
        assert "equationOfState rhoConst;" in txt
        assert "transport       const;" in txt
        assert "thermo          eConst;" in txt

    m_rho_p = re.search(r"\brho\s+([0-9.eE+-]+)\s*;", tp_pla)
    m_mu_p = re.search(r"\bmu\s+([0-9.eE+-]+)\s*;", tp_pla)
    m_rho_r = re.search(r"\brho\s+([0-9.eE+-]+)\s*;", tp_rbc)
    m_mu_r = re.search(r"\bmu\s+([0-9.eE+-]+)\s*;", tp_rbc)
    assert m_rho_p and m_mu_p and m_rho_r and m_mu_r
    assert float(m_rho_p.group(1)) == pytest.approx(1.0, rel=1e-12)
    assert float(m_mu_p.group(1)) == pytest.approx(1.0 / float(rec["Re"]), rel=1e-8)

    meta = json.loads((cdir / "params.json").read_text(encoding="utf-8"))
    assert float(m_rho_r.group(1)) == pytest.approx(meta["rho_ratio"], rel=1e-8)
    assert float(m_mu_r.group(1)) == pytest.approx(meta["mu_ratio"] / float(rec["Re"]), rel=1e-8)

    # Verify ESI v2306 phaseProperties entries
    pp = (cdir / "constant" / "phaseProperties").read_text(encoding="utf-8")
    for tok in (
        "type            basicTwoPhaseSystem;",
        "phases          (RBC plasma);",
        "type            none;",
        "continuousPhase plasma;",
        "(RBC in plasma)",
        "type            WenYu;",
        "swarmCorrection",
        f"type            {lift_model};",
        "pMin            -1e12;",
    ):
        assert tok in pp, f"Missing '{tok}' in phaseProperties"
    if lift_model == "constantCoefficient":
        assert re.search(r"\bCl\s+[0-9.eE+-]+\s*;", pp) is not None

    # Verify isothermal nEnergyCorrectors 0 in fvSolution & steady monitors in controlDict
    fv_sol = (cdir / "system" / "fvSolution").read_text(encoding="utf-8")
    assert "nEnergyCorrectors           0;" in fv_sol
    cd = (cdir / "system" / "controlDict").read_text(encoding="utf-8")
    for tok in ("application     reactingTwoPhaseEulerFoam;", "inletPressure", "outletPhi", "steadySample"):
        assert tok in cd


def test_m2b_lift_design_20_anchors_matches_committed_jsonl(tmp_path: Path) -> None:
    """campaign/m2b_lift_design.jsonl contains the exact 20 M2b anchors configured for M2b-B."""
    base = generate_m2b_design(level=1)
    lift = generate_m2b_lift_design(level=1)
    assert len(lift) == 20
    assert len({r["run_id"] for r in lift}) == 20
    for b, l in zip(base, lift, strict=True):
        assert l["family"] == b["family"]
        assert l["Re"] == b["Re"]
        assert l["phi"] == b["phi"]
        assert l["d_p_star"] == b["d_p_star"]
        assert l["mu_ratio"] == b["mu_ratio"]
        assert l["geometry"] == b["geometry"]
        assert l["solver"] == "reactingTwoPhaseEulerFoam"
        assert l["tier"] == "m2b-b"
        assert l["lift_model"] == "LegendreMagnaudet"

    jsonl_path = Path(__file__).resolve().parents[1] / "campaign" / "m2b_lift_design.jsonl"
    assert jsonl_path.is_file(), "campaign/m2b_lift_design.jsonl not generated yet"
    rows = [json.loads(ln) for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert rows == lift

    # Generate all 20 M2b-B cases at level 0 and check alpha sum == 1
    out = write_m2b_lift_design_jsonl(tmp_path / "copy.jsonl", lift)
    assert out.is_file()
    for rec in lift:
        cdir = tmp_path / rec["run_id"]
        write_m2b_lift_case(cdir, rec, level=0)
        a_r = float(parse_openfoam_field(cdir / "0" / "alpha.RBC")["boundary"]["inlet"]["value"][1])
        a_p = float(parse_openfoam_field(cdir / "0" / "alpha.plasma")["boundary"]["inlet"]["value"][1])
        assert a_r + a_p == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. CFL thickness metric & M2b sample parser tests
# ---------------------------------------------------------------------------

def test_compute_cfl_thickness_planar_and_axisymmetric() -> None:
    """CFL thickness returns 0 for uniform phi and exact crossing distance for wall-depleted profiles."""
    # Planar G1 [0, 1]: uniform phi -> 0.0
    y = np.linspace(0.0, 1.0, 201)
    cfl_uni = compute_cfl_thickness_star(y, np.full_like(y, 0.30), phi_inlet=0.30, family="G1")
    assert cfl_uni["cfl_thickness_star"] == pytest.approx(0.0, abs=1e-12)

    # Planar G1 [0, 1]: linear ramp from 0 at wall to 0.30 at dist = 2*delta => 50% crossing at delta = 0.08
    delta = 0.08
    dist = np.minimum(y, 1.0 - y)
    phi_ramp = np.where(dist >= 2.0 * delta, 0.30, 0.30 * dist / (2.0 * delta))
    cfl_g1 = compute_cfl_thickness_star(
        y, phi_ramp, phi_inlet=0.30, family="G1", wall_lo=0.0, wall_hi=1.0
    )
    assert cfl_g1["cfl_thickness_star"] == pytest.approx(delta, abs=1e-6)
    assert cfl_g1["cfl_bot_star"] == pytest.approx(delta, abs=1e-6)
    assert cfl_g1["cfl_top_star"] == pytest.approx(delta, abs=1e-6)

    # Axisymmetric KG r in [0, 1.66887]: single outer wall at r_hi
    r_hi = 0.5 * (252.0 / 75.5)
    r = np.linspace(0.0, r_hi, 201)
    delta_kg = 0.15
    phi_kg = np.where(r_hi - r >= 2.0 * delta_kg, 0.20, 0.20 * (r_hi - r) / (2.0 * delta_kg))
    cfl_kg = compute_cfl_thickness_star(
        r, phi_kg, phi_inlet=0.20, family="KG", wall_lo=0.0, wall_hi=r_hi
    )
    assert cfl_kg["cfl_thickness_star"] == pytest.approx(delta_kg, abs=1e-6)
    assert cfl_kg["cfl_bot_star"] is None


# ---------------------------------------------------------------------------
# 3. M2b multiphase NPZ extraction, phi-weighted f_window, and steady-state check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("family", "solver", "tier", "geom", "cfl_width"),
    [
        ("G1", "multiphaseEulerFoam", "m2b", {"L_ratio": 20.0}, 0.0),
        ("G1", "reactingTwoPhaseEulerFoam", "m2b-b", {"L_ratio": 20.0}, 0.10),
        ("G2", "multiphaseEulerFoam", "m2b", {"E": 2.0, "L_in": 5.0, "L_out_ratio": 10.0}, 0.12),
        (
            "KG",
            "reactingTwoPhaseEulerFoam",
            "m2b-b",
            {"E": 252.0 / 75.5, "L_in": 1000.0 / 151.0, "L_out": 5000.0 / 151.0, "wedge_angle_deg": 5.0},
            0.14,
        ),
    ],
)
def test_extract_m2b_case_schema_cfl_and_phi_weighted_window(
    tmp_path: Path,
    family: str,
    solver: str,
    tier: str,
    geom: dict[str, Any],
    cfl_width: float,
) -> None:
    """extract_case dispatches M2b-A/M2b-B, extracts full multiphase NPZ schema, CFL, and phi-weighted f_window."""
    rec: dict[str, Any] = {
        "run_id": f"TEST_{tier}_{family}",
        "case_id": f"TEST_{family}",
        "tier": tier,
        "solver": solver,
        "family": family,
        "level": 0,
        "Re": 10.0,
        "phi": 0.30,
        "rho_ratio": 1.064265,
        "mu_ratio": 2.0,
        "d_p_star": 0.16 if family == "G1" else 0.08,
        "drag_model": "WenYu",
        "lift_model": "none" if solver == "multiphaseEulerFoam" else "LegendreMagnaudet",
        "end_time": 60.0,
        "geometry": geom,
    }
    case_dir = tmp_path / rec["run_id"]
    if solver == "reactingTwoPhaseEulerFoam":
        write_m2b_lift_case(case_dir, rec, level=0)
    else:
        write_m2b_case(case_dir, rec, level=0)

    _populate_fake_m2b_run(
        case_dir,
        family=family,
        Re=10.0,
        phi_inlet=0.30,
        solver=solver,
        end_time=60.0,
        cfl_width=cfl_width,
        xr_target=1.45,
        dp_grad_star=12.0,
        geometry=geom,
    )

    out_npz = tmp_path / f"{rec['run_id']}.npz"
    res = extract_case(
        case_dir=case_dir,
        design_record=rec,
        level=0,
        wall_seconds=9.5,
        out_npz_path=out_npz,
        tau_lo=0.1,
        tau_hi=4.0,
    )

    # Verify CFL metric & reattachment length
    assert res["scalars"]["cfl_thickness_star"] == pytest.approx(cfl_width, abs=5e-3)
    if family == "G1":
        assert res["scalars"]["xr_over_h1"] is None
    else:
        assert res["scalars"]["xr_over_h1"] == pytest.approx(1.45, abs=5e-3)
    assert res["scalars"]["dp_star"] > 0.0

    # Verify full multiphase NPZ schema
    with np.load(out_npz) as data:
        required_m2b_keys = {
            "x_star",
            "y_star",
            "cell_volumes",
            "U_star",
            "U_plasma_star",
            "U_rbc_star",
            "phi_star",
            "p_star",
            "gdot_star",
            "gdot_plasma_star",
            "gdot_rbc_star",
            "tau_star",
            "tau_plasma_star",
            "grid_x",
            "grid_y",
            "grid_mask",
            "grid_ux",
            "grid_uy",
            "grid_u_plasma_x",
            "grid_u_plasma_y",
            "grid_u_rbc_x",
            "grid_u_rbc_y",
            "grid_phi",
            "grid_p",
            "grid_gdot",
            "grid_tau",
            "outlet_s",
            "outlet_u_n",
            "outlet_u_plasma_n",
            "outlet_phi",
            "outlet_gdot",
            "outlet_tau",
            "outlet_weights",
            "outlet_branch_id",
            "params",
        }
        assert required_m2b_keys.issubset(set(data.files))

        nx, ny = FAMILY_GRID_SHAPES[family]
        assert data["grid_phi"].shape == (ny, nx)
        assert data["grid_u_rbc_x"].shape == (ny, nx)
        assert np.all(data["grid_phi"][~data["grid_mask"]] == 0.0)

        meta = json.loads(str(data["params"]))
        assert meta["solver"] == solver
        assert meta["converged"] is True
        assert meta["converged_by"] == "steady_10pct"
        assert meta["window_weighting"] == "rbc_flux (alpha.RBC * u_RBC_n * dA)"

        # Verify phi-weighted window_fraction matches recompute_window_fraction and direct call
        f_recomp = recompute_window_fraction(out_npz, 0.1, 4.0)
        assert f_recomp == pytest.approx(res["scalars"]["f_window"], rel=1e-12)
        f_direct = window_fraction(
            data["outlet_tau"],
            data["outlet_u_n"],
            data["outlet_weights"],
            phi=data["outlet_phi"],
            lo=0.1,
            hi=4.0,
        )
        assert f_direct == pytest.approx(res["scalars"]["f_window"], rel=1e-12)
        assert recompute_window_fraction(out_npz, 0.0, 1e4) == pytest.approx(1.0, rel=1e-12)

        # For wall-depleted G1, verify phi-weighted f_window differs from unweighted in high-shear near-wall band
        if family == "G1" and cfl_width > 0.0:
            f_phi_wall = window_fraction(
                data["outlet_tau"], data["outlet_u_n"], data["outlet_weights"],
                phi=data["outlet_phi"], lo=3.5, hi=10.0,
            )
            f_unw_wall = window_fraction(
                data["outlet_tau"], data["outlet_u_n"], data["outlet_weights"],
                phi=None, lo=3.5, hi=10.0,
            )
            # Wall depletion removes RBCs from the high-shear wall layer -> lower phi-weighted fraction!
            assert f_phi_wall < f_unw_wall


def test_evaluate_m2b_convergence_and_run_batch_not_steady(tmp_path: Path) -> None:
    """Transient M2b convergence passes <1e-4 over last 10 % and fails with reason=not_steady otherwise."""
    design_path = tmp_path / "m2b_test_design.jsonl"
    g1_geom = {"L_ratio": 20.0}
    g2_geom = {"E": 2.0, "L_in": 5.0, "L_out_ratio": 10.0}
    recs = [
        {
            "run_id": "M2B_G1_STEADY",
            "tier": "m2b",
            "solver": "multiphaseEulerFoam",
            "family": "G1",
            "level": 0,
            "Re": 4.0,
            "phi": 0.30,
            "rho_ratio": 1.064265,
            "mu_ratio": 2.0,
            "d_p_star": 0.16,
            "end_time": 60.0,
            "geometry": g1_geom,
        },
        {
            "run_id": "M2BL_G2_STEADY",
            "tier": "m2b-b",
            "solver": "reactingTwoPhaseEulerFoam",
            "family": "G2",
            "level": 0,
            "Re": 8.3,
            "phi": 0.20,
            "rho_ratio": 1.064265,
            "mu_ratio": 2.5,
            "d_p_star": 0.08,
            "lift_model": "LegendreMagnaudet",
            "end_time": 80.0,
            "geometry": g2_geom,
        },
        {
            "run_id": "M2B_G2_DRIFT_DP",
            "tier": "m2b",
            "solver": "multiphaseEulerFoam",
            "family": "G2",
            "level": 0,
            "Re": 8.3,
            "phi": 0.20,
            "rho_ratio": 1.064265,
            "mu_ratio": 2.5,
            "d_p_star": 0.08,
            "end_time": 80.0,
            "geometry": g2_geom,
        },
        {
            "run_id": "M2BL_G2_DRIFT_PHI",
            "tier": "m2b-b",
            "solver": "reactingTwoPhaseEulerFoam",
            "family": "G2",
            "level": 0,
            "Re": 8.3,
            "phi": 0.20,
            "rho_ratio": 1.064265,
            "mu_ratio": 2.5,
            "d_p_star": 0.08,
            "lift_model": "LegendreMagnaudet",
            "end_time": 80.0,
            "geometry": g2_geom,
        },
        {
            "run_id": "M2B_G2_DRIFT_XR",
            "tier": "m2b",
            "solver": "multiphaseEulerFoam",
            "family": "G2",
            "level": 0,
            "Re": 8.3,
            "phi": 0.20,
            "rho_ratio": 1.064265,
            "mu_ratio": 2.5,
            "d_p_star": 0.08,
            "end_time": 80.0,
            "geometry": g2_geom,
        },
    ]
    design_path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")

    def fake_runner(case_dir: Path, _timeout: float) -> tuple[bool, str | None]:
        meta = json.loads((case_dir / "params.json").read_text(encoding="utf-8"))
        rid = meta["run_id"]
        solver = meta["solver"]
        # Verify correct case generator was dispatched by run_batch
        if solver == "reactingTwoPhaseEulerFoam":
            assert (case_dir / "constant" / "phaseProperties").is_file()
        else:
            assert (case_dir / "constant" / "transportProperties").is_file()

        _populate_fake_m2b_run(
            case_dir,
            family=meta["family"],
            Re=float(meta["Re"]),
            phi_inlet=float(meta["phi"]),
            solver=solver,
            end_time=float(meta["end_time"]),
            cfl_width=0.0 if solver == "multiphaseEulerFoam" else 0.09,
            dp_drift=5e-4 if rid == "M2B_G2_DRIFT_DP" else 2e-5,
            phi_drift=5e-4 if rid == "M2BL_G2_DRIFT_PHI" else 2e-5,
            xr_drift=5e-4 if rid == "M2B_G2_DRIFT_XR" else 2e-5,
            geometry=meta["geometry"],
        )
        return True, None

    work = tmp_path / "work_m2b"
    summary = run_batch(
        design_path,
        work_dir=work,
        workers=2,
        tier="m2b",
        upload=False,
        runner_fn=fake_runner,
    )
    assert summary["succeeded"] == 2
    assert summary["failed"] == 3

    rows = {r["run_id"]: r for r in load_manifest(work / "manifest.jsonl")}
    assert rows["M2B_G1_STEADY"]["status"] == "ok"
    assert rows["M2B_G1_STEADY"]["converged_by"] == "steady_10pct"
    assert rows["M2B_G1_STEADY"]["scalars"]["cfl_thickness_star"] == pytest.approx(0.0, abs=1e-6)
    assert rows["M2BL_G2_STEADY"]["status"] == "ok"
    assert rows["M2BL_G2_STEADY"]["converged_by"] == "steady_10pct"
    assert rows["M2BL_G2_STEADY"]["scalars"]["cfl_thickness_star"] == pytest.approx(0.09, abs=5e-3)

    for bad_id in ("M2B_G2_DRIFT_DP", "M2BL_G2_DRIFT_PHI", "M2B_G2_DRIFT_XR"):
        assert rows[bad_id]["status"] == "failed"
        assert rows[bad_id]["reason"] == "not_steady"
        assert rows[bad_id]["converged_by"] is None
        assert rows[bad_id]["sha256"] is None
        # Failed cases kept locally for inspection
        assert (work / "cases" / "G2" / bad_id).is_dir()

    # Converged cases cleaned up
    assert not (work / "cases" / "G1" / "M2B_G1_STEADY").exists()
    assert not (work / "cases" / "G2" / "M2BL_G2_STEADY").exists()


def test_m2b_lift_run_case_log_guard_and_missing_end_not_steady(tmp_path: Path) -> None:
    """M2b-B run_case.sh allows trapFpe banner, catches real crashes, and missing End fails steady check."""
    cdir = tmp_path / "m2bl_guard"
    write_m2b_lift_case(
        cdir,
        {"family": "G1", "Re": 4.0, "phi": 0.3, "d_p_star": 0.16, "geometry": {"L_ratio": 20.0}},
        level=0,
    )
    script = str(cdir / "run_case.sh")
    good_log = cdir / "good.log"
    good_log.write_text(
        "Build : v2306\n"
        "trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).\n"
        "Time = 60\n"
        "Phase-sum volume fraction, min, max = 1 1 1\n"
        "End\n",
        encoding="utf-8",
    )
    assert subprocess.run(["bash", script, "--check-log", str(good_log)], capture_output=True).returncode == 0

    bad_log = cdir / "bad.log"
    bad_log.write_text(
        "trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).\n"
        "Phase-sum volume fraction, min, max = nan 0 1\n",
        encoding="utf-8",
    )
    assert subprocess.run(["bash", script, "--check-log", str(bad_log)], capture_output=True).returncode == 4

    # Missing End in log -> evaluate_m2b_convergence returns converged=False
    _populate_fake_m2b_run(
        cdir,
        family="G1",
        Re=4.0,
        phi_inlet=0.3,
        solver="reactingTwoPhaseEulerFoam",
        end_time=60.0,
        reached_end=False,
        geometry={"L_ratio": 20.0},
    )
    ev = evaluate_m2b_convergence(cdir, family="G1", solver="reactingTwoPhaseEulerFoam")
    assert ev["converged"] is False
    assert ev["reached_end"] is False

