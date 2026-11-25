"""Unit tests for M2b Euler–Euler (multiphaseEulerFoam) generator & 20-anchor design."""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

import pytest

from campaign.design_m2b import generate_m2b_design, write_m2b_design_jsonl
from campaign.run_batch import parse_openfoam_field
from cfd.gen.m2b import (
    COMSOL2023_DP_RBC,
    COMSOL2023_MU_PLASMA,
    COMSOL2023_RHO_PLASMA,
    COMSOL2023_RHO_RBC,
    SUPPORTED_DRAG_MODELS,
    WU2015_DP_RBC,
    WU2015_MU_PLASMA,
    WU2015_RHO_PLASMA,
    WU2015_RHO_RBC,
    build_g1,
    build_m2b_geometry,
    dimensional_to_star,
    star_to_dimensional,
    write_m2b_case,
    wu2015_mu_ratio,
    wu2015_rbc_viscosity_cP,
)

EXPECTED_CASE_FILES = (
    "system/blockMeshDict",
    "system/controlDict",
    "system/fvSchemes",
    "system/fvSolution",
    "constant/transportProperties",
    "constant/turbulenceProperties",
    "constant/g",
    "0/alpha.RBC",
    "0/alpha.plasma",
    "0/U.RBC",
    "0/U.plasma",
    "0/p_rgh",
    "run_case.sh",
    "params.json",
)


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _assert_balanced_foam_dict(path: Path) -> str:
    """Assert FoamFile header and balanced {} / () delimiters in an OpenFOAM ASCII file."""
    text = path.read_text(encoding="utf-8")
    assert "FoamFile" in text, f"Missing FoamFile header in {path}"
    clean = _strip_comments(text)
    # Remove quoted strings before counting delimiters (e.g. regex divSchemes keys)
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


def _parse_phase_scalar(transport_text: str, phase_name: str, key: str) -> float:
    """Extract a numeric scalar entry from a phase block in constant/transportProperties."""
    m_phase = re.search(rf"\b{re.escape(phase_name)}\s*\{{", transport_text)
    assert m_phase is not None, f"Phase '{phase_name}' not found in transportProperties"
    start = m_phase.end() - 1
    depth = 0
    end = start
    for idx in range(start, len(transport_text)):
        if transport_text[idx] == "{":
            depth += 1
        elif transport_text[idx] == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break
    block = transport_text[start : end + 1]
    m_val = re.search(rf"\b{re.escape(key)}\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*;", block)
    assert m_val is not None, f"Key '{key}' not found in phase '{phase_name}' block"
    return float(m_val.group(1))


# ---------------------------------------------------------------------------
# 1. Geometry & mesh tests (G1, G2, KG)
# ---------------------------------------------------------------------------

def test_g1_mesh_and_refinement() -> None:
    m0 = build_g1({"L_ratio": 20.0}, level=0)
    m1 = build_g1({"L_ratio": 20.0}, level=1)
    m2 = build_g1({"L_ratio": 20.0}, level=2)

    assert m0.family == "G1"
    assert m0.inlet_height == pytest.approx(1.0)
    assert m0.outlet_heights == (1.0,)
    assert {p.name for p in m0.patches} == {"inlet", "outlet", "walls", "frontAndBack"}

    n0 = sum(nx * ny * nz for _, (nx, ny, nz), _ in m0.blocks)
    n1 = sum(nx * ny * nz for _, (nx, ny, nz), _ in m1.blocks)
    n2 = sum(nx * ny * nz for _, (nx, ny, nz), _ in m2.blocks)
    assert n1 / n0 == pytest.approx(2.25, rel=0.05)
    assert n2 / n1 == pytest.approx(2.25, rel=0.05)


def test_build_m2b_geometry_rejects_unsupported_family() -> None:
    with pytest.raises(ValueError, match="M2b supports families"):
        build_m2b_geometry("G4", {"theta_deg": 45.0, "daughter_ratio": 0.7})


# ---------------------------------------------------------------------------
# 2. Case file generation, parsing & phase-fraction sum == 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case_params",
    [
        {
            "run_id": "test_G1",
            "family": "G1",
            "Re": 4.0,
            "phi": 0.30,
            "d_p_star": 0.16,
            "geometry": {"L_ratio": 20.0},
        },
        {
            "run_id": "test_G2",
            "family": "G2",
            "Re": 8.3,
            "phi": 0.20,
            "d_p_star": 0.08,
            "geometry": {"E": 2.0, "L_in": 10.0, "L_out_ratio": 10.0},
        },
        {
            "run_id": "test_KG",
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
def test_write_m2b_case_files_parse_and_alpha_sums_to_one(
    tmp_path: Path, case_params: dict
) -> None:
    case_dir = tmp_path / case_params["run_id"]
    write_m2b_case(case_dir, case_params, level=0)

    # All expected OpenFOAM + metadata files exist
    for rel in EXPECTED_CASE_FILES:
        assert (case_dir / rel).is_file(), f"Missing {rel}"

    # Every OpenFOAM dictionary/field has valid FoamFile header and balanced delimiters
    for rel in EXPECTED_CASE_FILES:
        if rel not in ("run_case.sh", "params.json"):
            _assert_balanced_foam_dict(case_dir / rel)

    # run_case.sh passes bash syntax check
    proc = subprocess.run(["bash", "-n", str(case_dir / "run_case.sh")], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    # Parse field files with the project's OpenFOAM parser
    a_rbc = parse_openfoam_field(case_dir / "0" / "alpha.RBC")
    a_pla = parse_openfoam_field(case_dir / "0" / "alpha.plasma")
    u_rbc = parse_openfoam_field(case_dir / "0" / "U.RBC")
    u_pla = parse_openfoam_field(case_dir / "0" / "U.plasma")
    p_rgh = parse_openfoam_field(case_dir / "0" / "p_rgh")

    # Phase fractions sum to exactly 1.0 in internalField and at inlet
    phi_expected = float(case_params["phi"])
    assert float(a_rbc["internal"][1]) == pytest.approx(phi_expected, abs=1e-12)
    assert float(a_pla["internal"][1]) == pytest.approx(1.0 - phi_expected, abs=1e-12)
    assert float(a_rbc["internal"][1]) + float(a_pla["internal"][1]) == pytest.approx(1.0, abs=1e-12)

    inlet_rbc = float(a_rbc["boundary"]["inlet"]["value"][1])
    inlet_pla = float(a_pla["boundary"]["inlet"]["value"][1])
    assert inlet_rbc == pytest.approx(phi_expected, abs=1e-12)
    assert inlet_pla == pytest.approx(1.0 - phi_expected, abs=1e-12)
    assert inlet_rbc + inlet_pla == pytest.approx(1.0, abs=1e-12)

    # Velocity & pressure BCs match Wu 2015 Table 1 / 2023 COMSOL
    assert u_rbc["boundary"]["walls"]["type"] == "fixedValue"
    assert u_pla["boundary"]["walls"]["type"] == "fixedValue"
    assert p_rgh["boundary"]["inlet"]["type"] == "fixedFluxPressure"
    assert p_rgh["boundary"]["walls"]["type"] == "fixedFluxPressure"
    assert p_rgh["boundary"]["outlet"]["type"] == "fixedValue"

    if case_params["family"] == "KG":
        assert a_rbc["boundary"]["wedgeFront"]["type"] == "wedge"
        assert a_rbc["boundary"]["wedgeBack"]["type"] == "wedge"
        assert a_rbc["boundary"]["axis"]["type"] == "empty"
    else:
        assert a_rbc["boundary"]["frontAndBack"]["type"] == "empty"

    # Verify required ESI v2306 multiphaseEulerFoam entries in transportProperties & controlDict
    tp = (case_dir / "constant" / "transportProperties").read_text(encoding="utf-8")
    for token in (
        "phases",
        "RBC",
        "plasma",
        "sigmas",
        "interfaceCompression",
        "virtualMass",
        "drag",
        "(RBC plasma)",
        "WenYu",
        "residualPhaseFraction",
        "residualSlip",
        "kappa",
        "Cp",
        "diameterModel",
    ):
        assert token in tp, f"Missing '{token}' in transportProperties"

    cd = (case_dir / "system" / "controlDict").read_text(encoding="utf-8")
    for token in (
        "application     multiphaseEulerFoam;",
        "adjustTimeStep  yes;",
        "alpha.RBC",
        "U.plasma",
        "U.RBC",
        " p)",
    ):
        assert token in cd, f"Missing '{token}' in controlDict"


@pytest.mark.parametrize("drag_model", SUPPORTED_DRAG_MODELS)
def test_supported_drag_models(tmp_path: Path, drag_model: str) -> None:
    case_dir = tmp_path / f"drag_{drag_model}"
    write_m2b_case(
        case_dir,
        {
            "family": "G1",
            "Re": 5.0,
            "phi": 0.25,
            "d_p_star": 0.16,
            "drag_model": drag_model,
            "geometry": {"L_ratio": 15.0},
        },
        level=0,
    )
    tp = (case_dir / "constant" / "transportProperties").read_text(encoding="utf-8")
    assert f"type                    {drag_model};" in tp


# ---------------------------------------------------------------------------
# 3. Star-unit <-> dimensional scaling reproduces 2023 COMSOL & Wu 2015 cases
# ---------------------------------------------------------------------------

def test_star_units_reproduce_dimensional_2023_comsol_case(tmp_path: Path) -> None:
    """Verify star-unit parameters written to OpenFOAM scale back to the 2023 COMSOL case."""
    # 2023 COMSOL slide 6 & 7: 50 um channel, U = 0.25 m/s, Hct = 0.22,
    # plasma rho = 1038 kg/m^3, mu = 4e-3 Pa.s; RBC rho = 1098 kg/m^3, d_p = 8 um.
    h1_dim = 50.0e-6
    U_dim = 0.25
    phi_dim = 0.22
    mu_rbc_dim = 12.0e-3  # illustrative RBC effective viscosity (3x carrier)

    star = dimensional_to_star(
        h1=h1_dim,
        U_mean=U_dim,
        rho_plasma=COMSOL2023_RHO_PLASMA,
        mu_plasma=COMSOL2023_MU_PLASMA,
        rho_rbc=COMSOL2023_RHO_RBC,
        mu_rbc=mu_rbc_dim,
        d_p=COMSOL2023_DP_RBC,
        phi=phi_dim,
    )
    assert star["Re"] == pytest.approx(1038.0 * 0.25 * 50e-6 / 4e-3, rel=1e-12)
    assert star["rho_ratio"] == pytest.approx(1098.0 / 1038.0, rel=1e-12)
    assert star["mu_ratio"] == pytest.approx(3.0, rel=1e-12)
    assert star["d_p_star"] == pytest.approx(8.0 / 50.0, rel=1e-12)

    # Write an actual OpenFOAM case and read back values from constant/transportProperties & 0/alpha.RBC
    case_dir = tmp_path / "comsol2023_G1"
    write_m2b_case(
        case_dir,
        {
            "run_id": "COMSOL2023_50um",
            "family": "G1",
            "Re": star["Re"],
            "rho_ratio": star["rho_ratio"],
            "mu_ratio": star["mu_ratio"],
            "d_p_star": star["d_p_star"],
            "phi": star["phi"],
            "geometry": {"L_ratio": 20.0},
        },
        level=0,
    )

    tp = _strip_comments((case_dir / "constant" / "transportProperties").read_text(encoding="utf-8"))
    rho_pla_of = _parse_phase_scalar(tp, "plasma", "rho")
    nu_pla_of = _parse_phase_scalar(tp, "plasma", "nu")
    rho_rbc_of = _parse_phase_scalar(tp, "RBC", "rho")
    nu_rbc_of = _parse_phase_scalar(tp, "RBC", "nu")
    d_rbc_of = _parse_phase_scalar(tp, "RBC", "d")
    phi_of = float(parse_openfoam_field(case_dir / "0" / "alpha.RBC")["boundary"]["inlet"]["value"][1])

    # Scale directly from OpenFOAM dictionary entries back to SI units
    assert rho_pla_of == pytest.approx(1.0, rel=1e-12)
    reconstructed = star_to_dimensional(
        star={
            "Re": 1.0 / nu_pla_of,
            "rho_ratio": rho_rbc_of / rho_pla_of,
            "mu_ratio": (rho_rbc_of * nu_rbc_of) / (rho_pla_of * nu_pla_of),
            "d_p_star": d_rbc_of,
            "phi": phi_of,
            "nu_plasma_star": nu_pla_of,
            "nu_rbc_star": nu_rbc_of,
        },
        h1=h1_dim,
        U_mean=U_dim,
        rho_plasma=COMSOL2023_RHO_PLASMA,
    )

    assert reconstructed["rho_plasma"] == pytest.approx(COMSOL2023_RHO_PLASMA, rel=1e-9)
    assert reconstructed["mu_plasma"] == pytest.approx(COMSOL2023_MU_PLASMA, rel=1e-8)
    assert reconstructed["rho_rbc"] == pytest.approx(COMSOL2023_RHO_RBC, rel=1e-8)
    assert reconstructed["mu_rbc"] == pytest.approx(mu_rbc_dim, rel=1e-8)
    assert reconstructed["nu_plasma"] == pytest.approx(COMSOL2023_MU_PLASMA / COMSOL2023_RHO_PLASMA, rel=1e-8)
    assert reconstructed["nu_rbc"] == pytest.approx(mu_rbc_dim / COMSOL2023_RHO_RBC, rel=1e-8)
    assert reconstructed["d_p"] == pytest.approx(COMSOL2023_DP_RBC, rel=1e-8)
    assert reconstructed["phi"] == pytest.approx(phi_dim, rel=1e-12)

    # Also verify 2023 slide-7 sudden expansion (d1 = 150 um, Hct = 0.22, Re = 12.2)
    d1_2023 = 150.0e-6
    U_exp = 12.2 * COMSOL2023_MU_PLASMA / (COMSOL2023_RHO_PLASMA * d1_2023)
    star_exp = dimensional_to_star(
        h1=d1_2023,
        U_mean=U_exp,
        rho_plasma=COMSOL2023_RHO_PLASMA,
        mu_plasma=COMSOL2023_MU_PLASMA,
        rho_rbc=COMSOL2023_RHO_RBC,
        mu_rbc=mu_rbc_dim,
        d_p=COMSOL2023_DP_RBC,
        phi=0.22,
    )
    back_exp = star_to_dimensional(
        star=star_exp, h1=d1_2023, U_mean=U_exp, rho_plasma=COMSOL2023_RHO_PLASMA
    )
    assert star_exp["Re"] == pytest.approx(12.2, rel=1e-12)
    assert back_exp["d_p"] == pytest.approx(8.0e-6, rel=1e-12)
    assert back_exp["rho_rbc"] == pytest.approx(1098.0, rel=1e-12)
    assert back_exp["mu_plasma"] == pytest.approx(4.0e-3, rel=1e-12)


def test_star_units_reproduce_wu2015_kg_case() -> None:
    """Verify Wu 2015 §3.1 KG tubular expansion dimensional -> star -> dimensional round-trip."""
    d1 = 151.0e-6
    U = 0.0757
    star = dimensional_to_star(
        h1=d1,
        U_mean=U,
        rho_plasma=WU2015_RHO_PLASMA,
        mu_plasma=WU2015_MU_PLASMA,
        rho_rbc=WU2015_RHO_RBC,
        mu_rbc=WU2015_MU_PLASMA,
        d_p=WU2015_DP_RBC,
        phi=0.01,
    )
    # Wu 2015 §3.1 quotes Re = 12.2 (exact arithmetic 12.23)
    assert star["Re"] == pytest.approx(12.23, rel=2e-3)
    assert star["d_p_star"] == pytest.approx(8.0 / 151.0, rel=1e-12)
    back = star_to_dimensional(star=star, h1=d1, U_mean=U, rho_plasma=WU2015_RHO_PLASMA)
    assert back["rho_rbc"] == pytest.approx(1093.0, rel=1e-12)
    assert back["mu_plasma"] == pytest.approx(0.96e-3, rel=1e-12)
    assert back["d_p"] == pytest.approx(8.0e-6, rel=1e-12)


def test_wu2015_viscosity_equations_monotonic() -> None:
    # Check Eqs. (9a)-(9b) limits at phi = 0.45
    mu0_045 = wu2015_rbc_viscosity_cP(0.45, 0.0)
    mu_hi_045 = wu2015_rbc_viscosity_cP(0.45, 1.0e6)
    assert mu0_045 == pytest.approx(537.002 * 0.45**3 + 55.006 * 0.45**2 - 0.129 * 0.45, rel=1e-12)
    assert mu_hi_045 == pytest.approx(27.873 * 0.45**3 - 21.218 * 0.45**2 + 14.439 * 0.45, rel=1e-3)
    assert mu0_045 > wu2015_rbc_viscosity_cP(0.45, 10.0) > mu_hi_045 > 0.0
    assert wu2015_mu_ratio(0.45, 1.0e4) > 1.0


# ---------------------------------------------------------------------------
# 4. 20-anchor campaign/m2b_design.jsonl checks
# ---------------------------------------------------------------------------

def test_m2b_design_20_anchors_and_all_cases_generate(tmp_path: Path) -> None:
    records = generate_m2b_design(level=1)
    assert len(records) == 20
    assert len({r["run_id"] for r in records}) == 20

    # Verify exact anchor composition from plan/tasks/T4_gemini_m2b.md
    kg = [r for r in records if r["family"] == "KG"]
    g1 = [r for r in records if r["family"] == "G1"]
    g2_zhao = [r for r in records if r["anchor_group"] == "G2_Zhao2008_range"]
    g2_spread = [r for r in records if r["anchor_group"] == "G2_E_spread"]

    assert len(kg) == 2
    assert {r["Re"] for r in kg} == {12.2, 37.8}
    assert all(r["phi"] == pytest.approx(0.01) for r in kg)

    assert len(g2_zhao) == 4
    assert {(r["phi"], r["Re"]) for r in g2_zhao} == {
        (0.20, 8.3),
        (0.20, 41.7),
        (0.30, 8.3),
        (0.30, 41.7),
    }

    assert len(g1) == 9
    assert {r["phi"] for r in g1} == {0.20, 0.30, 0.45}
    for phi_val in (0.20, 0.30, 0.45):
        assert len([r for r in g1 if math.isclose(r["phi"], phi_val)]) == 3

    assert len(g2_spread) == 5
    assert {r["geometry"]["E"] for r in g2_spread} == {1.5, 3.0}

    # Check JSONL serialization + generate & validate every single anchor case (level 0 for speed)
    out_jsonl = write_m2b_design_jsonl(tmp_path / "m2b_design.jsonl", records)
    loaded = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(loaded) == 20

    for rec in loaded:
        cdir = tmp_path / "cases" / rec["run_id"]
        write_m2b_case(cdir, rec, level=0)
        a_rbc = float(parse_openfoam_field(cdir / "0" / "alpha.RBC")["boundary"]["inlet"]["value"][1])
        a_pla = float(parse_openfoam_field(cdir / "0" / "alpha.plasma")["boundary"]["inlet"]["value"][1])
        assert a_rbc + a_pla == pytest.approx(1.0, abs=1e-12)
        assert a_rbc == pytest.approx(rec["phi"], abs=1e-12)


def test_committed_m2b_design_jsonl_matches_generator() -> None:
    jsonl_path = Path(__file__).resolve().parents[1] / "campaign" / "m2b_design.jsonl"
    assert jsonl_path.is_file(), "campaign/m2b_design.jsonl has not been generated yet"
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows == generate_m2b_design(level=1)


def test_m2b_run_case_log_guard_allows_trapfpe_banner_and_catches_crashes(tmp_path: Path) -> None:
    """Verify M2b run_case.sh skips v2306 trapFpe banner while failing on real crashes/NaN."""
    cdir = tmp_path / "m2b_log_check"
    write_m2b_case(
        cdir,
        {"family": "G1", "Re": 4.0, "phi": 0.2, "d_p_star": 0.16, "geometry": {"L_ratio": 15.0}},
        level=0,
    )
    script = str(cdir / "run_case.sh")

    good_log = cdir / "good.log"
    good_log.write_text(
        "Build  : _c39a0664-20230612 OPENFOAM=2306\n"
        "trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).\n"
        "Time = 60\n"
        "RBC volume fraction, min, max = 0.2 0.19 0.21\n"
        "smoothSolver: Solving for U.plasma, Initial residual = 1e-06, Final residual = 1e-08\n"
        "End\n",
        encoding="utf-8",
    )
    ok = subprocess.run(["bash", script, "--check-log", str(good_log)], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr

    for name, bad_line in (
        ("fatal", "--> FOAM FATAL ERROR: maximum iterations exceeded\n"),
        ("sigfpe", "#1  Foam::sigFpe::sigHandler(int) at ??:?\n"),
        ("nan", "RBC volume fraction, min, max = nan 0 1\n"),
    ):
        bad_log = cdir / f"bad_{name}.log"
        bad_log.write_text(
            "trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).\n" + bad_line,
            encoding="utf-8",
        )
        res = subprocess.run(["bash", script, "--check-log", str(bad_log)], capture_output=True, text=True)
        assert res.returncode == 4, f"Expected exit 4 for {name}, got {res.returncode}"

