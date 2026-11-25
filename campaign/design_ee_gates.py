"""OpenFOAM Euler–Euler gate cases (T7 Part 2, AMENDMENT v2.2): KG anchors, Sezer/Zhao expansion.

Scope (``plan/tasks/T7_mistral_archive_m2b.md`` §2 step 4 + 4b, CORPUS_PLAN §6.1–6.2): OpenFOAM
Euler–Euler with **stock closures** is used for

1. **KG gate** — Karino & Goldsmith tubular expansion (Wu 2015 §3.1, axisymmetric wedge,
   ``phi = 0.01``) at ``Re = 12.2`` and ``37.8``, for both Euler–Euler variants:

   - ``EE-DL``: ``reactingTwoPhaseEulerFoam``, ``WenYu`` drag, stock ``LegendreMagnaudet`` lift,
     constant phase viscosities. Target ``x_r/d1 = 1.523 / 4.735`` within 15 %.
   - ``EE-full``: adds stock ``Antal`` wall lubrication and ``constantCoefficient`` virtual mass.

2. **Sezer/Zhao gate** — planar expansion ``h1 = 100 um -> h2 = 200 um``, ``L1 = 1000 um``,
   ``L2 = 2000 um`` (``L1/h1 = 10``, ``L2/h1 = 20``), ``d/h1 = 0.08``, ``alpha_R in
   {0.2, 0.3, 0.4, 0.5}`` at the 100 uL/min condition of Sezer 2026 (``Re = 17.8``, verified in
   ``knowledge/15``), with Sezer's own drag law (stock ``GidaspowErgunWenYu``) and
   ``LegendreMagnaudet`` lift. Target: the ``Delta P*(alpha_R)`` trend of Sezer 2026 Eq. 23
   (``226.72 a^2 - 148.57 a + 117.37``, minimum at ``alpha_R ~ 0.3``), realised as
   ``Delta P* = 2 * dp_star / Re`` in post-processing. Their domain is 3D quarter, ours 2D planar,
   so only the trend and the location of the minimum are compared (A12).

Closure choices and their provenance:

- ``WenYu`` drag for KG (Wu 2015's Stokes+hindrance has no stock equivalent, knowledge/14 §6).
- ``GidaspowErgunWenYu`` for the Sezer comparison (their Eqs. 14a/b are exactly Gidaspow's
  Ergun >= 0.2 / Wen-Yu < 0.2 split, knowledge/15).
- Wall lubrication ``Antal`` with ``Cw1 = -0.01``, ``Cw2 = 0.05``: the values "commonly used" in the
  literature (``secondary``; no ESI v2306 tutorial configures Antal, verified by source scan).
  Note v2306 declares both coefficients dimensionless.
- Virtual mass ``constantCoefficient`` with ``Cvm = 0.5`` (verified in v2306 tutorials; the isolated
  rigid-sphere value, ``secondary`` for the literature statement). In a converged steady state the
  virtual-mass force vanishes, so V4's physics difference comes from wall lubrication (state this
  in the article, CORPUS_PLAN §6.2).
- No per-phase shear-thinning viscosity exists in v2306 (verified source scan), so the RBC phase
  viscosity is constant: ``mu_ratio`` from Wu 2015 Eqs. (7, 9a-c) at the inlet ``phi`` and a nominal
  ``1e4 1/s`` shear rate — the same fallback as the M2b-A control. **FLAG.**
- ``localEuler`` (LTS) per v2.1: v2306 wires local time stepping into
  ``reactingTwoPhaseEulerFoam`` (``createRDeltaT.H``/``setRDeltaT.H``), with ``maxCo`` in the PIMPLE
  dict. ``multiphaseEulerFoam`` does not support it.

Run with ``campaign.run_batch --design campaign/ee_gates_design.jsonl --tier ee-gate``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cfd.gen.m2b import WU2015_RHO_RATIO, wu2015_mu_ratio

# Antal wall-lubrication coefficients (secondary: "commonly used" -0.01 / 0.05, no v2306 tutorial).
ANTAL_CW1 = -0.01
ANTAL_CW2 = 0.05
# Virtual mass coefficient for an isolated rigid sphere / bubble, per unit dispersed mass.
# Verified in ESI v2306 tutorials (bubbleColumn phaseProperties: Cvm 0.5).
CVM = 0.5

# Sezer 2026 geometry and flow condition (knowledge/15, verified).
SEZER_H1_UM = 100.0
SEZER_RE_100UL_MIN = 17.8
SEZER_ALPHA_R = (0.2, 0.3, 0.4, 0.5)
SEZER_DPSTAR_FIT = (226.72, -148.57, 117.37)  # Eq. 23, min at alpha_R ~ 0.3 (verified)
# Measured 2026-11-24 (T7, VM rbc-team-5): at end_time 120 the Sezer cases reach a p_in / phi_out
# drift of ~1.4e-4 / ~3e-4 over the last 10 % of time, i.e. just above the 1e-4 steady gate
# (smooth monotone drift, not an oscillation). end_time 400 gives ~3x the settling time.
SEZER_END_TIME = 400.0


def sezer_dp_star(alpha_r: float) -> float:
    """Sezer 2026 Eq. 23 fit for the normalised pressure drop ``Delta P*(alpha_R)``."""
    a, b, c = SEZER_DPSTAR_FIT
    return a * alpha_r**2 + b * alpha_r + c


def _kg_records(level: int, local_euler: bool) -> list[dict[str, Any]]:
    """KG anchors (phi = 0.01, Re 12.2 / 37.8) for EE-DL and EE-full."""
    geom = {
        "E": 252.0 / 75.5,
        "L_in": 1000.0 / 151.0,
        "L_out": 5000.0 / 151.0,
        "wedge_angle_deg": 5.0,
    }
    out: list[dict[str, Any]] = []
    for variant in ("EE-DL", "EE-full"):
        for Re in (12.2, 37.8):
            rec: dict[str, Any] = {
                "run_id": f"EEG_KG_{variant}_Re{Re:g}".replace(".", "p"),
                "family": "KG",
                "tier": "ee-gate",
                "gate": "KG_Wu2015",
                "variant": variant,
                "solver": "reactingTwoPhaseEulerFoam",
                "level": level,
                "Re": Re,
                "phi": 0.01,
                "rho_ratio": round(WU2015_RHO_RATIO, 6),
                # Wu 2015 §3.1: hardened 1 % RBCs, deformability effect on viscosity ignored.
                "mu_ratio": 1.0,
                "d_p_star": round(8.0 / 151.0, 6),
                "drag_model": "WenYu",
                "lift_model": "LegendreMagnaudet",
                "inlet_profile": "parabolic",
                "end_time": 100.0,
                "local_euler": local_euler,
                "geometry": dict(geom),
                # Validation target: Wu 2015 Fig. 3 measured reattachment, x_r / d1.
                "target_xr_over_d1": 1.5232 if Re < 20.0 else 4.7351,
                "target_rel_tol": 0.15,
            }
            if variant == "EE-full":
                rec.update(
                    {
                        "wall_lubrication_model": "Antal",
                        "Cw1": ANTAL_CW1,
                        "Cw2": ANTAL_CW2,
                        "virtual_mass_model": "constantCoefficient",
                        "cvm": CVM,
                    }
                )
            out.append(rec)
    return out


def _sezer_records(level: int, local_euler: bool) -> list[dict[str, Any]]:
    """Zhao/Sezer planar expansion at the 100 uL/min condition, alpha_R 0.2-0.5."""
    geom = {"E": 2.0, "L_in": 10.0, "L_out_ratio": 20.0}
    out: list[dict[str, Any]] = []
    for alpha_r in SEZER_ALPHA_R:
        out.append(
            {
                "run_id": f"EEG_G2_Sezer_Re{SEZER_RE_100UL_MIN:g}_aR{alpha_r:g}".replace(".", "p"),
                "family": "G2",
                "tier": "ee-gate",
                "gate": "Sezer2026_alphaR",
                "variant": "EE-DL",
                "solver": "reactingTwoPhaseEulerFoam",
                "level": level,
                "Re": SEZER_RE_100UL_MIN,
                "phi": alpha_r,
                "rho_ratio": round(WU2015_RHO_RATIO, 6),
                "mu_ratio": round(wu2015_mu_ratio(alpha_r, gdot_dim=1.0e4), 6),
                # Sezer uses the Gidaspow blend (Ergun >= 0.2, Wen-Yu < 0.2): stock GidaspowErgunWenYu.
                "drag_model": "GidaspowErgunWenYu",
                "lift_model": "LegendreMagnaudet",
                "inlet_profile": "uniform",
                "d_p_star": round(8.0 / SEZER_H1_UM, 6),  # d/h1 = 0.08
                "end_time": SEZER_END_TIME,
                "local_euler": local_euler,
                "geometry": dict(geom),
                "target_dp_star_sezer": round(sezer_dp_star(alpha_r), 6),
                "target_dp_star_note": "Delta P* = 2 * dp_star / Re (Sezer Eq. 22 normalisation)",
            }
        )
    return out


def generate_ee_gates_design(level: int = 2, local_euler: bool = True) -> list[dict[str, Any]]:
    """Return the 8 OpenFOAM Euler–Euler gate records (4 KG, 4 Sezer/Zhao)."""
    records = _kg_records(level, local_euler) + _sezer_records(level, local_euler)
    assert len(records) == 8, f"expected 8 gate records, got {len(records)}"
    return records


def write_design_jsonl(output_path: Path | str, records: list[dict[str, Any]]) -> Path:
    """Write design records to a JSONL file (one JSON object per line)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=False) + "\n")
    return out


def main() -> None:
    """CLI entrypoint to generate ``campaign/ee_gates_design.jsonl``."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate the OpenFOAM Euler-Euler gate design.")
    parser.add_argument("--out", type=Path, default=here / "ee_gates_design.jsonl")
    parser.add_argument("--level", type=int, default=2, choices=(0, 1, 2))
    parser.add_argument(
        "--no-lts",
        action="store_true",
        help="Use plain Euler time marching instead of localEuler (LTS)",
    )
    args = parser.parse_args()
    records = generate_ee_gates_design(level=args.level, local_euler=not args.no_lts)
    out = write_design_jsonl(args.out, records)
    print(f"Wrote {len(records)} Euler-Euler gate records (level={args.level}) to {out}")


if __name__ == "__main__":
    main()
