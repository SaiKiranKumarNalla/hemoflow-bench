"""T7 tests: Euler–Euler gate design and the stock wall-lubrication / virtual-mass / LTS options."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from campaign.design_ee_gates import (
    ANTAL_CW1,
    ANTAL_CW2,
    CVM,
    SEZER_ALPHA_R,
    SEZER_RE_100UL_MIN,
    generate_ee_gates_design,
    sezer_dp_star,
)
from cfd.gen.m2b_lift import (
    SUPPORTED_VIRTUAL_MASS_MODELS,
    SUPPORTED_WALL_LUBRICATION_MODELS,
    write_m2b_lift_case,
)


@pytest.fixture(scope="module")
def gates() -> list[dict]:
    return generate_ee_gates_design(level=0)


def test_gate_design_shape(gates: list[dict]) -> None:
    """4 KG anchors (2 Re x 2 variants) + 4 Sezer/Zhao alpha_R cases, all reactingTwoPhaseEulerFoam."""
    assert len(gates) == 8
    kg = [r for r in gates if r["family"] == "KG"]
    sezer = [r for r in gates if r["family"] == "G2"]
    assert len(kg) == 4 and len(sezer) == 4
    assert {r["variant"] for r in kg} == {"EE-DL", "EE-full"}
    assert all(r["solver"] == "reactingTwoPhaseEulerFoam" for r in gates)
    assert all(r["local_euler"] is True for r in gates)
    for r in kg:
        assert r["phi"] == 0.01 and r["mu_ratio"] == 1.0
        assert r["drag_model"] == "WenYu" and r["lift_model"] == "LegendreMagnaudet"
        assert r["target_rel_tol"] == 0.15
    assert [r["phi"] for r in sezer] == list(SEZER_ALPHA_R)
    for r in sezer:
        assert r["Re"] == SEZER_RE_100UL_MIN
        assert r["d_p_star"] == pytest.approx(0.08)
        assert r["drag_model"] == "GidaspowErgunWenYu"
        assert r["geometry"]["L_in"] == 10.0 and r["geometry"]["L_out_ratio"] == 20.0
        assert r["target_dp_star_sezer"] == pytest.approx(sezer_dp_star(r["phi"]))


def test_sezer_fit_minimum_near_03() -> None:
    """Sezer Eq. 23 has its minimum at alpha_R ~ 0.3 (knowledge/15, verified)."""
    values = {a: sezer_dp_star(a) for a in SEZER_ALPHA_R}
    assert min(values, key=values.get) == 0.3
    assert abs(-148.57 / (2 * 226.72)) == pytest.approx(0.3277, abs=1e-3)


def test_ee_full_writes_wall_lubrication_and_virtual_mass(tmp_path: Path, gates: list[dict]) -> None:
    """EE-full carries stock Antal wall lubrication + constantCoefficient virtual mass; EE-DL none."""
    for rec in gates:
        if rec["family"] != "KG":
            continue
        case_dir = tmp_path / rec["run_id"]
        write_m2b_lift_case(case_dir, rec, level=0)
        phase = (case_dir / "constant" / "phaseProperties").read_text()
        if rec["variant"] == "EE-full":
            assert "type            Antal;" in phase
            assert f"Cw1             {ANTAL_CW1:g};" in phase
            assert f"Cw2             {ANTAL_CW2:g};" in phase
            assert "type            constantCoefficient;" in phase
            assert f"Cvm             {CVM:g};" in phase
        else:
            assert "Antal" not in phase and "Cvm" not in phase


def test_local_euler_sets_lts_controls(tmp_path: Path, gates: list[dict]) -> None:
    """localEuler in fvSchemes plus maxCo/maxDeltaT/rDeltaTSmoothingCoeff in the PIMPLE dict."""
    rec = dict(gates[0])
    case_dir = tmp_path / "lts"
    write_m2b_lift_case(case_dir, rec, level=0)
    schemes = (case_dir / "system" / "fvSchemes").read_text()
    solution = (case_dir / "system" / "fvSolution").read_text()
    assert "default         localEuler;" in schemes
    assert "maxCo" in solution and "rDeltaTSmoothingCoeff" in solution
    assert "maxDeltaT" in solution

    rec_plain = {**rec, "local_euler": False}
    case_plain = tmp_path / "plain"
    write_m2b_lift_case(case_plain, rec_plain, level=0)
    assert "default         Euler;" in (case_plain / "system" / "fvSchemes").read_text()
    assert "maxCo" not in (case_plain / "system" / "fvSolution").read_text()


def test_missing_wall_lubrication_coefficients_raise(tmp_path: Path, gates: list[dict]) -> None:
    """No stock coefficient defaults exist: a model that needs coefficients refuses to write."""
    rec = next(r for r in gates if r["variant"] == "EE-full")
    broken = {k: v for k, v in rec.items() if k not in ("Cw1", "Cw2")}
    with pytest.raises(ValueError, match="Antal wall lubrication requires"):
        write_m2b_lift_case(tmp_path / "bad", broken, level=0)

    broken_vm = {**rec, "cvm": None}
    with pytest.raises(ValueError, match="Cvm"):
        write_m2b_lift_case(tmp_path / "bad2", broken_vm, level=0)

    unknown = {**rec, "wall_lubrication_model": "Lubchenko"}
    with pytest.raises(ValueError, match="Unsupported wall_lubrication_model"):
        write_m2b_lift_case(tmp_path / "bad3", unknown, level=0)


def test_supported_model_lists_match_v2306() -> None:
    """v2306 has exactly Antal/Frank/Tomiyama wall lubrication and 3 virtual-mass models."""
    assert set(SUPPORTED_WALL_LUBRICATION_MODELS) == {"none", "Antal", "Frank", "Tomiyama"}
    assert set(SUPPORTED_VIRTUAL_MASS_MODELS) == {"none", "constantCoefficient", "Lamb"}


def test_gate_design_jsonl_roundtrip(tmp_path: Path, gates: list[dict]) -> None:
    """Every gate record parses back from JSONL with its target attached."""
    path = tmp_path / "gates.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in gates) + "\n", encoding="utf-8")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 8
    assert all("gate" in r and "level" in r for r in rows)


def test_wall_lubrication_writes_wall_dist_method(tmp_path: Path, gates: list[dict]) -> None:
    """Wall-lubrication models call wallDist::New(mesh), so fvSchemes needs fvSchemes.wallDist.method."""
    full = next(r for r in gates if r["variant"] == "EE-full")
    dl = next(r for r in gates if r["variant"] == "EE-DL")
    write_m2b_lift_case(tmp_path / "full", full, level=0)
    write_m2b_lift_case(tmp_path / "dl", dl, level=0)
    schemes_full = (tmp_path / "full" / "system" / "fvSchemes").read_text()
    schemes_dl = (tmp_path / "dl" / "system" / "fvSchemes").read_text()
    assert "wallDist" in schemes_full and "method          meshWave;" in schemes_full
    assert "wallDist" not in schemes_dl
