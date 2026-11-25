"""Tests for T10 Part 3: level-3 mesh support, paired GCI design (72 runs), and L1-L3 Roache GCI."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from campaign.design_gci_paired import (
    GCI_PAIRED_FAMILIES,
    GCI_PAIRED_LEVELS,
    generate_gci_paired_design,
)
from campaign.gci import (
    analyze_case_gci,
    group_manifest_for_gci,
    re_strip_level_suffix,
    run_gci_from_manifest,
)
from cfd.gen.geometry import build_geometry


def _paired_design_path() -> Path:
    return Path(__file__).resolve().parent.parent / "campaign" / "paired_design.jsonl"


def test_gci_paired_design_72_runs_and_valid_l1_l2_l3_meshes() -> None:
    """72 runs = 6 families x 2 extremes (max Re, max phi) x 2 variants (N, GN) x 3 levels (1, 2, 3)."""
    records = generate_gci_paired_design(_paired_design_path())
    assert len(records) == 72
    assert len({r["run_id"] for r in records}) == 72
    assert len({r["case_id"] for r in records}) == 24
    assert sorted({r["level"] for r in records}) == list(GCI_PAIRED_LEVELS)
    assert {r["tier"] for r in records} == {"gci_paired"}
    assert {r["variant"] for r in records} == {"N", "GN"}

    # Verify 2 distinct designs per family G2-G7, exact extremes, and valid meshes at levels 1, 2, 3.
    paired_rows = [json.loads(line) for line in _paired_design_path().read_text().splitlines() if line.strip()]
    for fam in GCI_PAIRED_FAMILIES:
        fam_recs = [r for r in records if r["family"] == fam]
        assert len(fam_recs) == 12
        design_ids = {r["design_id"] for r in fam_recs}
        assert len(design_ids) == 2

        fam_all_gn = [r for r in paired_rows if r["family"] == fam and r["variant"] == "GN"]
        max_re = max(r["Re"] for r in fam_all_gn)
        max_phi = max(r["phi"] for r in fam_all_gn)
        rec_max_re = next(r for r in fam_recs if r["corner"] == "max_Re" and r["variant"] == "GN")
        rec_max_phi = next(r for r in fam_recs if r["corner"] == "max_phi" and r["variant"] == "GN")
        assert rec_max_re["Re"] == pytest.approx(max_re)
        assert rec_max_phi["phi"] == pytest.approx(max_phi)

    # Check mesh scaling ~2.25x per level across L1 -> L2 -> L3 for all 12 selected designs.
    seen_design_lvls: dict[tuple[str, str], dict[int, int]] = {}
    for rec in records:
        if rec["variant"] != "N":
            continue
        mesh = build_geometry(rec["family"], rec["geometry"], level=rec["level"])
        n_cells = sum(nx * ny * nz for _, (nx, ny, nz), _ in mesh.blocks)
        seen_design_lvls.setdefault((rec["family"], rec["design_id"]), {})[rec["level"]] = n_cells

    assert len(seen_design_lvls) == 12
    for key, counts in seen_design_lvls.items():
        assert set(counts) == {1, 2, 3}, key
        assert 2.15 <= counts[2] / counts[1] <= 2.35, (key, counts)
        assert 2.15 <= counts[3] / counts[2] <= 2.35, (key, counts)


def test_gci_levels_1_2_3_support(tmp_path: Path) -> None:
    """campaign.gci supports levels (1, 2, 3), _L3 suffix stripping, and dimensional window scalars."""
    assert re_strip_level_suffix("GCIP_G2_maxRe_N_L3") == "GCIP_G2_maxRe_N"

    r = 1.5
    h = 0.02
    entries = []
    for lvl, mult in ((1, r * r), (2, r), (3, 1.0)):
        entries.append(
            {
                "run_id": f"GCIP_G2_maxRe_N_L{lvl}",
                "case_id": "GCIP_G2_maxRe_N",
                "family": "G2",
                "level": lvl,
                "status": "ok",
                "params": {"n_cells": int(round(10000 * (2.25 ** (lvl - 1))))},
                "scalars": {
                    "dp_star": 120.0 + 3.0 * (h * mult) ** 2,
                    "xr_over_h1": 1.80 - 0.4 * (h * mult) ** 2,
                    "f_window_15_45": 0.40 + 0.05 * (h * mult) ** 2,
                    "f_window_30_60": 0.25 + 0.04 * (h * mult) ** 2,
                },
            }
        )

    grouped = group_manifest_for_gci(entries, levels=(1, 2, 3))
    assert "GCIP_G2_maxRe_N" in grouped
    rep = analyze_case_gci(
        grouped["GCIP_G2_maxRe_N"],
        levels=(1, 2, 3),
        scalar_names=("dp_star", "xr_over_h1", "f_window_15_45", "f_window_30_60"),
    )
    assert rep["levels"] == [1, 2, 3]
    assert math.isclose(rep["scalars"]["dp_star"]["p"], 2.0, rel_tol=1e-10)
    assert math.isclose(rep["scalars"]["f_window_15_45"]["f_ext"], 0.40, rel_tol=1e-10)
    assert rep["all_passed"] is True

    manifest = tmp_path / "manifest_gci_paired.jsonl"
    manifest.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    reports, table = run_gci_from_manifest(
        manifest,
        tmp_path / "out",
        levels=(1, 2, 3),
        scalar_names=("dp_star", "xr_over_h1", "f_window_15_45", "f_window_30_60"),
    )
    assert len(reports) == 1
    assert "f_window_15_45" in table
