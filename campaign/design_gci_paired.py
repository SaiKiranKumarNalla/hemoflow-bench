"""Generate the 72-run paired-design GCI study (families G2-G7 x 2 extremes x V1/V2 x levels 1,2,3).

Per ``plan/tasks/T10_gemini_single_phase_corpus.md`` Part 3 (steps 6-8):
- From the redesigned ``campaign/paired_design.jsonl`` (Re <= 60, tau_w in [2, 150] Pa),
  select per family G2-G7 two designs:
  (a) ``max_Re``: maximum Reynolds number (tie-break: higher phi, stronger shear-thinning).
  (b) ``max_phi``: maximum haematocrit (ties -> strongest shear-thinning: higher beta, lower n).
- For each of the 12 selected designs, emit both single-phase variants V1 (``N``) and V2 (``GN``)
  at mesh levels 1, 2, 3 (refinement ratio r = 1.5) -> 12 * 2 * 3 = 72 runs, tier ``gci_paired``.
- Output: ``campaign/gci_paired_design.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cfd.gen.geometry import build_geometry

GCI_PAIRED_FAMILIES: tuple[str, ...] = ("G2", "G3", "G4", "G5", "G6", "G7")
GCI_PAIRED_LEVELS: tuple[int, int, int] = (1, 2, 3)
GCI_PAIRED_VARIANTS: tuple[str, str] = ("N", "GN")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def select_paired_gci_designs(
    paired_records: list[dict[str, Any]],
    *,
    families: tuple[str, ...] = GCI_PAIRED_FAMILIES,
) -> list[tuple[str, str, dict[str, dict[str, Any]]]]:
    """Select (family, corner_tag, {variant: record}) for 2 extremes per family G2-G7.

    Returns 12 tuples: ``(family, "max_Re" | "max_phi", {"N": rec_n, "GN": rec_gn})``.
    """
    by_fam_design: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for rec in paired_records:
        fam = str(rec["family"]).upper()
        var = str(rec.get("variant") or "")
        if fam not in families or var not in GCI_PAIRED_VARIANTS:
            continue
        did = str(rec["design_id"])
        by_fam_design.setdefault(fam, {}).setdefault(did, {})[var] = rec

    selected: list[tuple[str, str, dict[str, dict[str, Any]]]] = []
    for fam in families:
        designs = by_fam_design.get(fam, {})
        if len(designs) < 2:
            raise ValueError(f"Family {fam} has fewer than 2 paired designs ({len(designs)})")

        def _re_key(did: str) -> tuple[float, float, float, float]:
            gn = designs[did]["GN"]
            return (
                float(gn["Re"]),
                float(gn["phi"]),
                float(gn.get("beta") or 0.0),
                -float(gn.get("n") or 1.0),
            )

        def _phi_key(did: str) -> tuple[float, float, float, float]:
            gn = designs[did]["GN"]
            return (
                float(gn["phi"]),
                float(gn.get("beta") or 0.0),
                -float(gn.get("n") or 1.0),
                float(gn["Re"]),
            )

        max_re_id = max(designs, key=_re_key)
        remaining = [did for did in designs if did != max_re_id]
        max_phi_id = max(remaining, key=_phi_key)

        selected.append((fam, "max_Re", designs[max_re_id]))
        selected.append((fam, "max_phi", designs[max_phi_id]))

    return selected


def generate_gci_paired_design(
    paired_design_path: Path | str,
    *,
    levels: tuple[int, int, int] = GCI_PAIRED_LEVELS,
) -> list[dict[str, Any]]:
    """Build the 72-run paired GCI design records from ``paired_design.jsonl``."""
    paired_records = _load_jsonl(Path(paired_design_path))
    selected = select_paired_gci_designs(paired_records)

    records: list[dict[str, Any]] = []
    for fam, corner, var_map in selected:
        corner_tag = "maxRe" if corner == "max_Re" else "maxPhi"
        for variant in GCI_PAIRED_VARIANTS:
            base = var_map[variant]
            case_id = f"GCIP_{fam}_{corner_tag}_{variant}"
            for lvl in levels:
                rec = dict(base)
                rec["tier"] = "gci_paired"
                rec["corner"] = corner
                rec["case_id"] = case_id
                rec["source_run_id"] = str(base["run_id"])
                rec["level"] = int(lvl)
                rec["run_id"] = f"{case_id}_L{lvl}"
                records.append(rec)
    return records


def write_gci_paired_design_jsonl(
    output_path: Path | str, records: list[dict[str, Any]]
) -> Path:
    """Write paired GCI design records to JSONL."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=False) + "\n")
    return out


def summarise_gci_paired(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise counts and L1/L2/L3 mesh cell counts for the paired GCI design."""
    cases: dict[str, dict[str, Any]] = {}
    for rec in records:
        cid = str(rec["case_id"])
        lvl = int(rec["level"])
        mesh = build_geometry(str(rec["family"]), rec["geometry"], level=lvl)
        n_cells = sum(nx * ny * nz for _, (nx, ny, nz), _ in mesh.blocks)
        entry = cases.setdefault(
            cid,
            {
                "family": rec["family"],
                "corner": rec["corner"],
                "design_id": rec["design_id"],
                "variant": rec["variant"],
                "Re": rec["Re"],
                "phi": rec["phi"],
                "tau_w_nominal_pa": rec["tau_w_nominal_pa"],
                "n_cells": {},
            },
        )
        entry["n_cells"][str(lvl)] = n_cells
    return {
        "n_runs": len(records),
        "n_cases": len(cases),
        "n_designs": len({(r["family"], r["design_id"]) for r in records}),
        "levels": sorted({int(r["level"]) for r in records}),
        "cases": cases,
    }


def main() -> None:
    """CLI entrypoint to generate ``campaign/gci_paired_design.jsonl``."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate 72-run paired GCI design JSONL (G2-G7 x 2 extremes x V1/V2 x L1-L3)."
    )
    parser.add_argument(
        "--paired-design",
        type=Path,
        default=here / "paired_design.jsonl",
        help="Input paired_design.jsonl path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=here / "gci_paired_design.jsonl",
        help="Output JSONL path (default: campaign/gci_paired_design.jsonl)",
    )
    args = parser.parse_args()

    records = generate_gci_paired_design(args.paired_design)
    out = write_gci_paired_design_jsonl(args.out, records)
    summary = summarise_gci_paired(records)
    print(f"Wrote {summary['n_runs']} paired GCI runs ({summary['n_cases']} cases, {summary['n_designs']} designs) to {out}")
    for cid, info in summary["cases"].items():
        if info["variant"] == "N":
            nc = info["n_cells"]
            print(
                f"  {info['family']} {info['corner']:7s} ({info['design_id']}): "
                f"Re={info['Re']:5.2f} phi={info['phi']:.4f} tw={info['tau_w_nominal_pa']:5.1f} Pa "
                f"cells L1/L2/L3={nc.get('1')}/{nc.get('2')}/{nc.get('3')}"
            )


if __name__ == "__main__":
    main()
