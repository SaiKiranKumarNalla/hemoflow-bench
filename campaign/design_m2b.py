"""Generate the 20-anchor M2b Euler–Euler (``multiphaseEulerFoam``) design matrix.

Anchor allocation (``plan/tasks/T4_gemini_m2b.md`` & ``plan/CORPUS_PLAN.md`` §3, step 6):
1. **KG (Wu 2015 §3.1 / Karino & Goldsmith 1977)** — 2 anchors:
   - Axisymmetric tubular sudden expansion (``E = 252 / 75.5 = 3.337748``, ``d1 = 151 um``,
     ``L_in = 1000/151``, ``L_out = 5000/151``, ``d_p* = 8/151 = 0.052980``),
     ``phi = 0.01`` (Wu's 1 % hardened RBCs), ``Re in {12.2, 37.8}``.
2. **G2 planar (Zhao 2008 range, qualitative CFL)** — 4 anchors:
   - ``E = 2.0`` (``100 -> 200 um``, ``L_in = 10.0``, ``L_out_ratio = 10.0``, ``d_p* = 8/100 = 0.08``),
     ``phi in {0.20, 0.30} x Re in {8.3, 41.7}``.
3. **G1 straight channel** — 9 anchors:
   - ``L_ratio = 20.0`` (``h1 = 50 um`` baseline -> ``d_p* = 8/50 = 0.16``),
     ``phi in {0.20, 0.30, 0.45} x Re in {1.0, 4.0, 20.0}``.
4. **G2 planar spread across ``E in {1.5, 3.0}``** — 5 anchors:
   - Spans ``E in {1.5, 3.0}``, ``phi in {0.20, 0.30, 0.45}``, and ``Re in {3.8, 12.2, 25.0}``.

Total = 2 + 4 + 9 + 5 = **20 anchors**, written to ``campaign/m2b_design.jsonl``.

Phase properties follow Wu et al. (2015) (``knowledge/14-wu2015-closures.md``,
``verified-primary``, PMC4521229):
- ``rho_ratio = 1093 / 1027 = 1.064265``
- ``mu_ratio``: effective high-shear RBC-to-plasma dynamic viscosity ratio from Wu 2015
  Eqs. (7, 9a–9c) via ``wu2015_mu_ratio(phi, gdot_dim=1e4)`` (and ``1.0`` for KG hardened
  1 % RBCs where Wu §3.1 notes deformability effects on viscosity were ignored).
- ``drag_model = "WenYu"`` (closest stock ESI v2306 drag model with voidage hindrance).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cfd.gen.m2b import WU2015_RHO_RATIO, wu2015_mu_ratio


def _tag_num(val: float) -> str:
    """Format a numeric parameter cleanly for run_id tokens (e.g. 12.2 -> '12p2')."""
    return f"{val:g}".replace(".", "p")


def generate_m2b_design(level: int = 1) -> list[dict[str, Any]]:
    """Return the exact 20-anchor M2b Euler–Euler design records."""
    records: list[dict[str, Any]] = []
    rho_ratio = round(WU2015_RHO_RATIO, 6)

    # ------------------------------------------------------------------
    # 1. KG axisymmetric validation anchors (2 runs): Wu 2015 §3.1
    # ------------------------------------------------------------------
    kg_geom = {
        "E": 252.0 / 75.5,
        "L_in": 1000.0 / 151.0,
        "L_out": 5000.0 / 151.0,
        "wedge_angle_deg": 5.0,
    }
    for Re in (12.2, 37.8):
        phi = 0.01
        records.append(
            {
                "run_id": f"M2B_KG_Wu2015_Re{_tag_num(Re)}_phi0p01",
                "case_id": f"M2B_KG_Re{_tag_num(Re)}",
                "tier": "m2b",
                "anchor_group": "KG_Wu2015_hardened_1pct",
                "solver": "multiphaseEulerFoam",
                "family": "KG",
                "level": level,
                "Re": Re,
                "phi": phi,
                "rho_ratio": rho_ratio,
                # Wu 2015 §3.1: hardened 1% RBCs in water, deformability effect on viscosity ignored
                "mu_ratio": 1.0,
                "d_p_star": round(8.0 / 151.0, 6),
                "drag_model": "WenYu",
                "inlet_profile": "parabolic",
                "end_time": 100.0,
                "geometry": dict(kg_geom),
            }
        )

    # ------------------------------------------------------------------
    # 2. G2 planar Zhao 2008 range anchors (4 runs): E = 2.0
    #    phi in {0.20, 0.30} x Re in {8.3, 41.7}
    # ------------------------------------------------------------------
    zhao_geom = {"E": 2.0, "L_in": 10.0, "L_out_ratio": 10.0}
    for phi in (0.20, 0.30):
        mu_r = round(wu2015_mu_ratio(phi, gdot_dim=1.0e4), 6)
        for Re in (8.3, 41.7):
            records.append(
                {
                    "run_id": f"M2B_G2_Zhao_E2_Re{_tag_num(Re)}_phi{_tag_num(phi)}",
                    "case_id": f"M2B_G2_Zhao_Re{_tag_num(Re)}_phi{_tag_num(phi)}",
                    "tier": "m2b",
                    "anchor_group": "G2_Zhao2008_range",
                    "solver": "multiphaseEulerFoam",
                    "family": "G2",
                    "level": level,
                    "Re": Re,
                    "phi": phi,
                    "rho_ratio": rho_ratio,
                    "mu_ratio": mu_r,
                    "d_p_star": round(8.0 / 100.0, 6),  # Zhao h1 = 100 um, d_p = 8 um
                    "drag_model": "WenYu",
                    "inlet_profile": "uniform",
                    "end_time": 80.0,
                    "geometry": dict(zhao_geom),
                }
            )

    # ------------------------------------------------------------------
    # 3. G1 straight channel anchors (9 runs):
    #    phi in {0.20, 0.30, 0.45} x Re in {1.0, 4.0, 20.0}
    # ------------------------------------------------------------------
    g1_geom = {"L_ratio": 20.0}
    for phi in (0.20, 0.30, 0.45):
        mu_r = round(wu2015_mu_ratio(phi, gdot_dim=1.0e4), 6)
        for Re in (1.0, 4.0, 20.0):
            records.append(
                {
                    "run_id": f"M2B_G1_Re{_tag_num(Re)}_phi{_tag_num(phi)}",
                    "case_id": f"M2B_G1_Re{_tag_num(Re)}_phi{_tag_num(phi)}",
                    "tier": "m2b",
                    "anchor_group": "G1_straight_sweep",
                    "solver": "multiphaseEulerFoam",
                    "family": "G1",
                    "level": level,
                    "Re": Re,
                    "phi": phi,
                    "rho_ratio": rho_ratio,
                    "mu_ratio": mu_r,
                    "d_p_star": round(8.0 / 50.0, 6),   # 2023 h1 = 50 um, d_p = 8 um
                    "drag_model": "WenYu",
                    "inlet_profile": "uniform",
                    "end_time": 60.0,
                    "geometry": dict(g1_geom),
                }
            )

    # ------------------------------------------------------------------
    # 4. G2 expansion spread across E in {1.5, 3.0} (5 runs)
    # ------------------------------------------------------------------
    g2_spread = [
        # (E, Re, phi, d_p_star)
        (1.5, 3.8, 0.20, 8.0 / 50.0),
        (1.5, 12.2, 0.45, 8.0 / 50.0),
        (3.0, 3.8, 0.30, 8.0 / 50.0),
        (3.0, 12.2, 0.20, 8.0 / 100.0),
        (3.0, 25.0, 0.45, 8.0 / 100.0),
    ]
    for E_val, Re, phi, dp_s in g2_spread:
        mu_r = round(wu2015_mu_ratio(phi, gdot_dim=1.0e4), 6)
        geom = {"E": E_val, "L_in": 5.0, "L_out_ratio": 10.0}
        end_t = 60.0 if E_val <= 1.5 else 90.0
        records.append(
            {
                "run_id": f"M2B_G2_E{_tag_num(E_val)}_Re{_tag_num(Re)}_phi{_tag_num(phi)}",
                "case_id": f"M2B_G2_E{_tag_num(E_val)}_Re{_tag_num(Re)}_phi{_tag_num(phi)}",
                "tier": "m2b",
                "anchor_group": "G2_E_spread",
                "solver": "multiphaseEulerFoam",
                "family": "G2",
                "level": level,
                "Re": Re,
                "phi": phi,
                "rho_ratio": rho_ratio,
                "mu_ratio": mu_r,
                "d_p_star": round(dp_s, 6),
                "drag_model": "WenYu",
                "inlet_profile": "uniform",
                "end_time": end_t,
                "geometry": geom,
            }
        )

    assert len(records) == 20, f"Expected 20 M2b anchors, got {len(records)}"
    return records


def write_m2b_design_jsonl(output_path: Path | str, records: list[dict[str, Any]]) -> Path:
    """Write M2b design records to a JSONL file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=False) + "\n")
    return out


def main() -> None:
    """CLI entrypoint to generate ``campaign/m2b_design.jsonl``."""
    parser = argparse.ArgumentParser(description="Generate 20-anchor M2b Euler-Euler design JSONL.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "m2b_design.jsonl",
        help="Output JSONL path (default: campaign/m2b_design.jsonl)",
    )
    parser.add_argument("--level", type=int, default=1, choices=(0, 1, 2))
    args = parser.parse_args()

    records = generate_m2b_design(level=args.level)
    out = write_m2b_design_jsonl(args.out, records)
    print(f"Wrote {len(records)} M2b anchors (level={args.level}) to {out}")


if __name__ == "__main__":
    main()
