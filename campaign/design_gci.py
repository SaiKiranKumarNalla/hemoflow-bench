"""Generate the 10-case × 3-level (= 30 runs) GCI design for families G2–G6.

Per ``plan/CORPUS_PLAN.md`` §3 and ``plan/tasks/T2_gemini_batch_tooling.md``:
- 2 extreme corners per family G2–G6 (10 cases total):
  1. ``inertial``: max Re (= 25.0) with the most severe geometry in the family (Newtonian).
  2. ``shearthin``: strongest shear-thinning — max beta (= 30.0), min n (= 0.25), a = 2.0,
     Cu = 2.0 near the thinning transition (Cu * gdot* ~ O(1..10) across the channel),
     at moderate Re (= 10.0) with severe geometry.
- Each case is emitted at mesh levels 0, 1, 2 (r = 1.5 refinement ratio) → 30 runs total
  written to ``campaign/gci_design.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def generate_gci_cases() -> list[dict[str, Any]]:
    """Return the 10 extreme-corner GCI cases (2 per family G2–G6)."""
    severe_geometries: dict[str, dict[str, float | int]] = {
        "G2": {"E": 4.0, "L_in": 5.0, "L_out_ratio": 10.0},
        "G3": {"throat_ratio": 0.3, "L_t": 1.0, "L_in": 5.0, "L_out_ratio": 10.0},
        "G4": {"theta_deg": 90.0, "daughter_ratio": 0.5, "L_in": 5.0, "L_out_ratio": 10.0},
        "G5": {"width_ratio": 0.5, "theta_deg": 90.0, "L_in": 5.0, "L_out_ratio": 10.0},
        "G6": {"Rc_ratio": 1.5, "n_bends": 4, "L_in": 5.0, "L_out_ratio": 10.0},
    }

    cases: list[dict[str, Any]] = []
    for fam in ("G2", "G3", "G4", "G5", "G6"):
        geom = severe_geometries[fam]
        # Corner 1: max Re + most severe geometry (Newtonian)
        cases.append(
            {
                "case_id": f"GCI_{fam}_inertial",
                "corner": "inertial_max_Re",
                "family": fam,
                "rheology": "Newtonian",
                "Re": 25.0,
                "beta": None,
                "Cu": None,
                "n": None,
                "a": None,
                "geometry": dict(geom),
            }
        )
        # Corner 2: strongest shear-thinning (max beta, min n, Cu near thinning transition)
        # At star-unit wall gdot* ~ 6-15, Cu = 2.0 puts Cu*gdot* ~ 1-30 right across the
        # Bird-Carreau thinning knee (max viscosity gradient across the cross-section).
        cases.append(
            {
                "case_id": f"GCI_{fam}_shearthin",
                "corner": "strongest_shear_thinning",
                "family": fam,
                "rheology": "BirdCarreau",
                "Re": 10.0,
                "beta": 30.0,       # TODO: source (upper bound from CORPUS_PLAN)
                "Cu": 2.0,          # near shear-thinning transition (Cu * gdot* ~ O(1..10))
                "n": 0.25,          # TODO: source (lower bound from CORPUS_PLAN)
                "a": 2.0,           # Cho & Kensey 1991, TODO: verify
                "geometry": dict(geom),
            }
        )
    return cases


def generate_gci_design(levels: tuple[int, ...] = (0, 1, 2)) -> list[dict[str, Any]]:
    """Expand the 10 GCI cases across mesh levels 0, 1, 2 (30 runs total)."""
    records: list[dict[str, Any]] = []
    for case in generate_gci_cases():
        for lvl in levels:
            rec = dict(case)
            rec["level"] = int(lvl)
            rec["run_id"] = f"{case['case_id']}_L{lvl}"
            records.append(rec)
    return records


def write_gci_design_jsonl(output_path: Path | str, records: list[dict[str, Any]]) -> Path:
    """Write GCI design records to a JSONL file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=False) + "\n")
    return out


def main() -> None:
    """CLI entrypoint to write campaign/gci_design.jsonl."""
    parser = argparse.ArgumentParser(description="Generate 10-case x 3-level GCI design JSONL.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "gci_design.jsonl",
        help="Output JSONL path (default: campaign/gci_design.jsonl)",
    )
    args = parser.parse_args()

    records = generate_gci_design()
    out = write_gci_design_jsonl(args.out, records)
    print(f"Wrote {len(records)} GCI runs (10 cases x 3 levels) to {out}")


if __name__ == "__main__":
    main()
