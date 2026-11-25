"""T7 tests: paired 4-variant design (v2.3), same-mesh pairing, and the M1 retry design."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from campaign import blood_rheology as rheo
from campaign.design_m1_retry import build_retry_design, select_retry_ids
from campaign.design_paired import (
    D_RBC_UM,
    PHI_BLOOD_RANGE,
    PHI_DEVICE_RANGE,
    PINNED_PHI,
    RE_MAX,
    TAU_W_PA_RANGE,
    VARIANTS,
    _mesh_signature,
    generate_design,
    summarise,
    velocity_from_nominal_tau_w,
)
from cfd.gen.case import write_case


@pytest.fixture(scope="module")
def pilot() -> list[dict]:
    """The v2.1 pilot design (one design per family G2-G7)."""
    return generate_design(per_family=1)


@pytest.fixture(scope="module")
def main_corpus() -> list[dict]:
    """The v2.3 main corpus (12 device + 12 whole-blood + 3 pinned designs per family)."""
    return generate_design(per_family=12)


# ---------------------------------------------------------------------------
# 1. Pairing structure
# ---------------------------------------------------------------------------

def test_pilot_is_one_design_per_family() -> None:
    """v2.1 pilot: 1 design per family G2-G7, 4 variants = 24 runs, 12 of them OpenFOAM."""
    s = summarise(generate_design(per_family=1))
    assert s["runs"] == 24
    assert s["designs"] == {f"G{i}": 1 for i in range(2, 8)}
    assert s["per_variant"] == {v: 6 for v, _ in VARIANTS}
    assert s["openfoam_runs"] == 12 and s["fenicsx_runs"] == 12


def test_four_variants_share_design_pair_and_mesh(pilot: list[dict]) -> None:
    """Every design has exactly N/GN/EE-DL/EE-full with one pair_id and one mesh signature."""
    by_pair: dict[str, list[dict]] = {}
    for rec in pilot:
        by_pair.setdefault(rec["pair_id"], []).append(rec)
    assert len(by_pair) == 6
    for pair_id, recs in by_pair.items():
        assert len(recs) == 4, pair_id
        assert {r["variant"] for r in recs} == {"N", "GN", "EE-DL", "EE-full"}
        assert len({r["design_id"] for r in recs}) == 1
        # Same mesh for all four variants: same family, level and geometry => same blockMesh.
        assert len({_mesh_signature(r) for r in recs}) == 1
        # Same star scale: identical Re, h, U, tau_w_nominal_pa, carrier and phi.
        for key in ("Re", "h_um", "U_mps", "tau_w_nominal_pa", "mu_carrier", "phi", "tau_ref", "t_ref"):
            assert len({r[key] for r in recs}) == 1, key


def test_variants_share_the_same_blockmesh_dict(tmp_path: Path) -> None:
    """V1 and V2 records of one design generate byte-identical meshes and equal cell counts."""
    recs = [r for r in generate_design(per_family=1) if r["family"] == "G2"]
    written = {}
    for rec in recs:
        if rec["variant"] not in ("N", "GN"):
            continue
        case_dir = tmp_path / rec["variant"]
        write_case(case_dir, {**rec, "level": 0}, level=0)
        written[rec["variant"]] = (
            (case_dir / "system" / "blockMeshDict").read_text(),
            json.loads((case_dir / "params.json").read_text())["n_cells"],
        )
    assert written["N"][0] == written["GN"][0]
    assert written["N"][1] == written["GN"][1]


def test_solver_dispatch_matches_variant(pilot: list[dict]) -> None:
    """V1/V2 run in OpenFOAM simpleFoam; V3/V4 are FEniCSx (never executed by run_batch)."""
    for rec in pilot:
        if rec["variant"] in ("N", "GN"):
            assert rec["solver"] == "simpleFoam"
        else:
            assert rec["solver"] == "fenicsx"


# ---------------------------------------------------------------------------
# 2. Haematocrit, carrier and continuum limit (v2.3)
# ---------------------------------------------------------------------------

def test_phi_branches_and_continuum_limit(main_corpus: list[dict]) -> None:
    """12 designs per family in each phi branch, 3 pinned clinical, d_RBC/h <= 0.1 everywhere."""
    designs: dict[str, dict[str, dict]] = {}
    for rec in main_corpus:
        designs.setdefault(rec["family"], {})[rec["design_id"]] = rec
    for family, by_id in designs.items():
        if family == "G1":
            continue
        device = [r for k, r in by_id.items() if k.startswith("D")]
        blood = [r for k, r in by_id.items() if k.startswith("B")]
        pinned = [r for k, r in by_id.items() if k.startswith("C")]
        assert len(device) == 12 and len(blood) == 12 and len(pinned) == 3, family
        for r in device:
            assert PHI_DEVICE_RANGE[0] <= r["phi"] <= PHI_DEVICE_RANGE[1]
        for r in blood:
            assert PHI_BLOOD_RANGE[0] <= r["phi"] <= PHI_BLOOD_RANGE[1]
        assert [r["phi"] for r in pinned] == list(PINNED_PHI)
        for r in by_id.values():
            assert r["d_rbc_over_h"] == pytest.approx(D_RBC_UM / r["h_um"], rel=1e-6)
            assert r["d_rbc_over_h"] <= 0.1 + 1e-9
            assert 80.0 <= r["h_um"] <= 150.0
            assert r["U_mps"] > 0.0
            assert r["Re"] <= RE_MAX + 1e-9
            assert TAU_W_PA_RANGE[0] - 1e-6 <= r["tau_w_nominal_pa"] <= TAU_W_PA_RANGE[1] + 1e-6
            tau_w_calc = 6.0 * r["mu_eff"] * r["U_mps"] / (r["h_um"] * 1e-6)
            assert tau_w_calc == pytest.approx(r["tau_w_nominal_pa"], rel=1e-5)


def test_main_corpus_counts_pairing_and_tau_w_re_caps(main_corpus: list[dict]) -> None:
    """T10 canonical counts: 164 designs x 4 variants = 656 runs (328 OF + 328 FEniCSx), Re <= 60."""
    s = summarise(main_corpus)
    assert s["runs"] == 656
    assert s["n_designs"] == 164
    assert s["openfoam_runs"] == 328
    assert s["fenicsx_runs"] == 328
    assert s["designs"] == {"G2": 27, "G3": 27, "G4": 27, "G5": 27, "G6": 27, "G7": 27, "G1": 2}
    assert s["per_variant"] == {v: 164 for v, _ in VARIANTS}
    assert s["Re"]["max"] <= RE_MAX
    assert TAU_W_PA_RANGE[0] <= s["tau_w_nominal_pa"]["min"]
    assert s["tau_w_nominal_pa"]["max"] <= TAU_W_PA_RANGE[1]

    by_pair: dict[str, list[dict]] = {}
    for rec in main_corpus:
        by_pair.setdefault(rec["pair_id"], []).append(rec)
        assert rec["Re"] <= RE_MAX + 1e-9
        assert TAU_W_PA_RANGE[0] - 1e-6 <= rec["tau_w_nominal_pa"] <= TAU_W_PA_RANGE[1] + 1e-6
    assert len(by_pair) == 164
    for pair_id, recs in by_pair.items():
        assert len(recs) == 4, pair_id
        assert {r["variant"] for r in recs} == {"N", "GN", "EE-DL", "EE-full"}
        assert len({_mesh_signature(r) for r in recs}) == 1
        assert len({r["tau_w_nominal_pa"] for r in recs}) == 1


def test_velocity_from_nominal_tau_w_inverts_quemada() -> None:
    """velocity_from_nominal_tau_w solves 6 * mu_eff(phi, 3U/h) * U / h == tau_w accurately."""
    for phi, mu_c in ((0.01, 1.3e-3), (0.10, 1.5e-3), (0.30, rheo.MU_PLASMA), (0.55, rheo.MU_PLASMA), (0.60, rheo.MU_PLASMA)):
        for tau_w in (2.0, 20.0, 150.0):
            u = velocity_from_nominal_tau_w(tau_w, 100.0, phi, mu_c)
            h_m = 100.0e-6
            mu_eff = rheo.quemada_mu(phi, 3.0 * u / h_m, mu_c)
            assert 6.0 * mu_eff * u / h_m == pytest.approx(tau_w, rel=1e-9)


def test_g1_paired_anchor_writes_openfoam_case(tmp_path: Path, main_corpus: list[dict]) -> None:
    """G1 anchor V1/V2 paired records generate valid OpenFOAM cases with identical meshes."""
    g1_of = [r for r in main_corpus if r["family"] == "G1" and r["solver"] == "simpleFoam"]
    assert len(g1_of) == 4
    first_pair = [r for r in g1_of if r["design_id"] == "A000"]
    assert len(first_pair) == 2
    bmds = {}
    for rec in first_pair:
        cdir = tmp_path / rec["run_id"]
        write_case(cdir, {**rec, "level": 0}, level=0)
        bmds[rec["variant"]] = (cdir / "system" / "blockMeshDict").read_text()
    assert bmds["N"] == bmds["GN"]


def test_carrier_viscosity_rule(main_corpus: list[dict]) -> None:
    """Buffer carrier (1.1-1.7 mPa s) only for phi <= 0.1; plasma 0.96 mPa s for whole blood."""
    lo, hi = rheo.MU_BUFFER_RANGE
    for rec in main_corpus:
        if rec["phi"] > rheo.BUFFER_MAX_PHI:
            assert rec["mu_carrier"] == pytest.approx(rheo.MU_PLASMA)
        else:
            assert lo - 1e-12 <= rec["mu_carrier"] <= hi + 1e-12
    buffers = {r["mu_carrier"] for r in main_corpus if r["phi"] <= 0.1}
    assert len(buffers) > 50  # sampled per design, not a single fixed value


# ---------------------------------------------------------------------------
# 3. Physics of the two OpenFOAM variants
# ---------------------------------------------------------------------------

def test_v1_newtonian_matches_quemada_at_reference_shear(pilot: list[dict]) -> None:
    """V1 carries mu_eff(phi)/mu_carrier, V2 carries the derived Carreau levels and beta."""
    for rec in pilot:
        if rec["variant"] == "N":
            expected = rheo.quemada_mu(rec["phi"], rec["gdot_ref"], rec["mu_carrier"]) / rec["mu_carrier"]
            assert rec["viscosity_ratio"] == pytest.approx(expected, rel=1e-6)
            assert rec["rheology"] == "Newtonian" and rec["beta"] is None
        elif rec["variant"] == "GN":
            fit = rheo.carreau_params_from_quemada(rec["phi"], rec["mu_carrier"])
            assert rec["beta"] == pytest.approx(fit.beta, rel=1e-6)
            assert rec["n"] == pytest.approx(fit.n, rel=1e-6)
            assert rec["Cu"] == pytest.approx(fit.lam * rec["U_mps"] / (rec["h_um"] * 1e-6), rel=1e-6)
            assert rec["viscosity_ratio"] == pytest.approx(fit.mu_inf / rec["mu_carrier"], rel=1e-6)
            assert rec["carreau_fit_max_rel_err"] < 0.20


def test_reynolds_number_from_device_scale(pilot: list[dict]) -> None:
    """Re = rho U h / mu_carrier, identical across the four variants of a design."""
    for rec in pilot:
        expected = rec["rho"] * rec["U_mps"] * rec["h_um"] * 1e-6 / rec["mu_carrier"]
        assert rec["Re"] == pytest.approx(expected, rel=1e-6)
        assert rec["t_ref"] == pytest.approx(rec["h_um"] * 1e-6 / rec["U_mps"], rel=1e-6)
        assert rec["tau_ref"] == pytest.approx(rec["mu_carrier"] * rec["U_mps"] / (rec["h_um"] * 1e-6), rel=1e-6)


def test_quemada_guard_and_limits() -> None:
    """The Quemada law raises where its bracket closes, and thins monotonically elsewhere."""
    with pytest.raises(ValueError):
        rheo.quemada_mu(0.55, 1e-3)  # zero-shear singularity above phi ~ 0.46
    with pytest.raises(ValueError):
        rheo.quemada_mu(0.45, 0.0)
    mu_100 = rheo.quemada_mu(0.45, 100.0, 1.2e-3)
    mu_10 = rheo.quemada_mu(0.45, 10.0, 1.2e-3)
    assert mu_10 > mu_100
    # High-shear limit approached from below.
    assert rheo.quemada_mu(0.45, 1e7, 1.2e-3) == pytest.approx(rheo.quemada_high_shear_mu(0.45, 1.2e-3), rel=1e-2)


def test_design_is_deterministic() -> None:
    """Same seed => identical records; different seed => different records."""
    a = generate_design(per_family=4, seed=1234)
    b = generate_design(per_family=4, seed=1234)
    c = generate_design(per_family=4, seed=4321)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "code_version_note"} for r in rows]  # noqa: E731
    assert strip(a) == strip(b)
    assert strip(a) != strip(c)


# ---------------------------------------------------------------------------
# 4. M1 retry design (v2.3): relaxation only, honest labels
# ---------------------------------------------------------------------------

def _manifest_rows() -> list[dict]:
    ok = {
        "run_id": "M1_G2_000",
        "status": "ok",
        "reason": None,
        "scalars": {"dp_star": 1.0},
    }
    naive = {"run_id": "M1_G3_001", "status": "failed", "reason": "not_converged"}
    retried = {"run_id": "M1_G3_004", "status": "failed", "reason": "not_converged"}
    return [ok, naive, retried, retried]  # retry adds an append-only duplicate row


def test_select_retry_ids_dedupes_manifest() -> None:
    """Only the latest row per run_id counts: 2 unique not_converged ids, 1 ok skipped."""
    assert select_retry_ids(_manifest_rows()) == ["M1_G3_001", "M1_G3_004"]


def test_retry_design_changes_only_relaxation(tmp_path: Path) -> None:
    """Retry records copy the physics, set p 0.2 / U 0.5, and are labelled retry_of.

    The generated ``fvSolution`` must contain the new relaxation factors and the mesh must be
    unchanged, so a recovered run is honestly labelled with the settings that produced it.
    """
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in _manifest_rows()) + "\n", encoding="utf-8")
    design = tmp_path / "design.jsonl"
    original = {
        "run_id": "M1_G3_001",
        "family": "G3",
        "rheology": "BirdCarreau",
        "Re": 5.0,
        "beta": 8.0,
        "Cu": 1000.0,
        "n": 0.48,
        "a": 2.0,
        "level": 0,
        "geometry": {"L_in": 5.0, "L_out_ratio": 10.0, "throat_ratio": 0.66, "L_t": 3.7},
    }
    design.write_text(json.dumps(original) + "\n", encoding="utf-8")

    recs = build_retry_design(manifest, design)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["relax_p"] == 0.2 and rec["relax_u"] == 0.5
    assert rec["retry_of"] == "not_converged"
    for key in ("Re", "beta", "Cu", "n", "a", "geometry", "rheology"):
        assert rec[key] == original[key], key

    case_dir = tmp_path / "case"
    write_case(case_dir, rec, level=0)
    fv_solution = (case_dir / "system" / "fvSolution").read_text()
    assert "p               0.2;" in fv_solution
    assert "U               0.5;" in fv_solution
    meta = json.loads((case_dir / "params.json").read_text())
    assert meta["relax_p"] == pytest.approx(0.2)
    assert meta["relax_u"] == pytest.approx(0.5)


def test_retry_design_empty_manifest_raises(tmp_path: Path) -> None:
    """No retryable ids is an error, not an empty design file."""
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"run_id": "M1_G2_000", "status": "ok"}) + "\n", encoding="utf-8")
    design = tmp_path / "design.jsonl"
    design.write_text(json.dumps({"run_id": "M1_G2_000"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        build_retry_design(manifest, design)


def test_case_parsing_reads_back_paired_metadata(tmp_path: Path) -> None:
    """A generated paired case exposes the device scale in params.json (archive provenance)."""
    rec = next(r for r in generate_design(per_family=1) if r["variant"] == "GN")
    case_dir = tmp_path / "paired"
    write_case(case_dir, {**rec, "level": 0}, level=0)
    meta = json.loads((case_dir / "params.json").read_text())
    assert meta["rheology"] == "BirdCarreau"
    assert meta["viscosity_ratio"] == pytest.approx(rec["viscosity_ratio"])
    assert meta["Re"] == pytest.approx(rec["Re"])
    # BirdCarreau star coefficients carry the design's viscosity ratio.
    transport = (case_dir / "constant" / "transportProperties").read_text()
    nu_inf = float(next(line.split()[-1].rstrip(";") for line in transport.splitlines() if "nuInf" in line))
    assert nu_inf == pytest.approx(rec["viscosity_ratio"] / rec["Re"], rel=1e-8)
    assert math.isclose(meta["relax_p"], 0.3)
