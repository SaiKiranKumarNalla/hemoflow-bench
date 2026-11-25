"""Roache 3-level Grid Convergence Index (GCI) analysis for CFD mesh independence.

Implements the standard 3-grid Richardson extrapolation & GCI procedure
(Roache 1994, 1998; Celik et al. 2008, ASME J. Fluids Eng. 130:078001) with
refinement ratio r = 1.5 and safety factor F_s = 1.25 (plan/CORPUS_PLAN.md §0).

Mesh level convention in this repo:
- level 0 = coarse (h_3 = r^2 * h_1)
- level 1 = medium (h_2 = r * h_1)
- level 2 = fine   (h_1)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_R = 1.5
DEFAULT_FS = 1.25

# Acceptance thresholds (plan/CORPUS_PLAN.md §0 & knowledge/08-validation-targets.md).
SCALAR_THRESHOLDS: dict[str, float] = {
    "dp_star": 0.02,
    "xr_over_h1": 0.02,
    "max_wall_tau_star": 0.02,
    "f_window": 0.05,
    "f_window_15_45": 0.05,
    "f_window_30_60": 0.05,
    "f_ge_15": 0.05,
    "f_ge_30": 0.05,
}


def roache_gci(
    f_coarse: float,
    f_medium: float,
    f_fine: float,
    *,
    r: float = DEFAULT_R,
    fs: float = DEFAULT_FS,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Compute 3-level Roache GCI metrics for one scalar quantity.

    Parameters
    ----------
    f_coarse, f_medium, f_fine : float
        Scalar values at mesh levels 0 (coarse), 1 (medium), and 2 (fine).
    r : float
        Constant linear grid refinement ratio h_coarse/h_medium = h_medium/h_fine (default 1.5).
    fs : float
        Roache safety factor for 3 grids (default 1.25).
    threshold : float | None
        Optional relative GCI_fine pass threshold (e.g. 0.02 for 2%).

    Returns
    -------
    dict[str, Any]
        Contains ``f_coarse``, ``f_medium``, ``f_fine``, ``r``, ``fs``, ``convergence``,
        ``p`` (observed order), ``f_ext`` (Richardson extrapolated value), ``e_a_21``,
        ``e_a_32``, ``gci_fine``, ``gci_medium``, ``asymptotic_ratio``, and ``pass_threshold``.
    """
    if r <= 1.0:
        raise ValueError(f"Refinement ratio r must be > 1.0, got {r}")
    if fs <= 0.0:
        raise ValueError(f"Safety factor fs must be > 0.0, got {fs}")

    f3 = float(f_coarse)
    f2 = float(f_medium)
    f1 = float(f_fine)

    eps21 = f2 - f1
    eps32 = f3 - f2

    base: dict[str, Any] = {
        "f_coarse": f3,
        "f_medium": f2,
        "f_fine": f1,
        "r": float(r),
        "fs": float(fs),
        "threshold": threshold,
    }

    if abs(eps21) < 1e-15 and abs(eps32) < 1e-15:
        return {
            **base,
            "convergence": "exact",
            "p": None,
            "f_ext": f1,
            "e_a_21": 0.0,
            "e_a_32": 0.0,
            "gci_fine": 0.0,
            "gci_medium": 0.0,
            "asymptotic_ratio": 1.0,
            "pass_threshold": True if threshold is not None else None,
        }

    if abs(eps32) < 1e-15:
        return {
            **base,
            "convergence": "divergent",
            "p": None,
            "f_ext": None,
            "e_a_21": abs(eps21 / f1) if abs(f1) > 1e-15 else None,
            "e_a_32": 0.0,
            "gci_fine": None,
            "gci_medium": None,
            "asymptotic_ratio": None,
            "pass_threshold": False if threshold is not None else None,
        }

    ratio = eps21 / eps32
    if abs(eps21) < 1e-15:
        return {
            **base,
            "convergence": "exact_fine",
            "p": None,
            "f_ext": f1,
            "e_a_21": 0.0,
            "e_a_32": abs(eps32 / f2) if abs(f2) > 1e-15 else None,
            "gci_fine": 0.0,
            "gci_medium": None,
            "asymptotic_ratio": None,
            "pass_threshold": True if threshold is not None else None,
        }

    if 0.0 < ratio < 1.0:
        convergence = "monotonic"
    elif -1.0 < ratio < 0.0:
        convergence = "oscillatory"
    else:
        return {
            **base,
            "convergence": "divergent",
            "p": None,
            "f_ext": None,
            "e_a_21": abs(eps21 / f1) if abs(f1) > 1e-15 else None,
            "e_a_32": abs(eps32 / f2) if abs(f2) > 1e-15 else None,
            "gci_fine": None,
            "gci_medium": None,
            "asymptotic_ratio": None,
            "pass_threshold": False if threshold is not None else None,
        }

    # Observed order of accuracy: p = ln(|eps32 / eps21|) / ln(r)
    p = math.log(abs(eps32 / eps21)) / math.log(r)
    rp = r**p
    denom = rp - 1.0

    # Richardson extrapolated value on the fine grid: f_ext = (r^p f1 - f2) / (r^p - 1)
    f_ext = (rp * f1 - f2) / denom

    e_a_21 = abs(eps21 / f1) if abs(f1) > 1e-15 else float("inf")
    e_a_32 = abs(eps32 / f2) if abs(f2) > 1e-15 else float("inf")

    gci_fine = fs * e_a_21 / denom
    gci_medium = fs * e_a_32 / denom
    asymptotic_ratio = (
        gci_medium / (rp * gci_fine)
        if math.isfinite(gci_fine) and gci_fine > 0.0 and math.isfinite(gci_medium)
        else None
    )

    passed: bool | None = None
    if threshold is not None:
        passed = bool(math.isfinite(gci_fine) and gci_fine <= threshold)

    return {
        **base,
        "convergence": convergence,
        "p": p,
        "f_ext": f_ext,
        "e_a_21": e_a_21,
        "e_a_32": e_a_32,
        "gci_fine": gci_fine,
        "gci_medium": gci_medium,
        "asymptotic_ratio": asymptotic_ratio,
        "pass_threshold": passed,
    }


def analyze_case_gci(
    entries: list[Mapping[str, Any]],
    *,
    r: float = DEFAULT_R,
    fs: float = DEFAULT_FS,
    levels: tuple[int, int, int] | None = None,
    scalar_names: tuple[str, ...] = ("dp_star", "xr_over_h1", "max_wall_tau_star", "f_window"),
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compute Roache GCI across all available scalars from 3 manifest entries (e.g. levels 0,1,2 or 1,2,3)."""
    by_level: dict[int, Mapping[str, Any]] = {}
    for entry in entries:
        if entry.get("status", "ok") != "ok":
            raise ValueError(f"Manifest entry {entry.get('run_id')} has status={entry.get('status')}")
        lvl = int(entry["level"])
        by_level[lvl] = entry

    if levels is None:
        if all(k in by_level for k in (0, 1, 2)):
            levels = (0, 1, 2)
        elif len(by_level) == 3:
            s_lvls = sorted(by_level)
            if s_lvls[1] - s_lvls[0] == 1 and s_lvls[2] - s_lvls[1] == 1:
                levels = (s_lvls[0], s_lvls[1], s_lvls[2])
            else:
                raise ValueError(f"Non-consecutive mesh levels {s_lvls} for GCI case")
        else:
            levels = (0, 1, 2)

    missing = [lvl for lvl in levels if lvl not in by_level]
    if missing:
        raise ValueError(f"Missing mesh level(s) {missing} for GCI case")

    l_coarse, l_med, l_fine = levels
    e0, e1, e2 = by_level[l_coarse], by_level[l_med], by_level[l_fine]
    family = str(e2["family"]).upper()
    case_id = str(
        e2.get("case_id")
        or re_strip_level_suffix(str(e2["run_id"]))
    )

    thresh_map = SCALAR_THRESHOLDS if thresholds is None else thresholds
    results: dict[str, Any] = {}
    for name in scalar_names:
        v0 = (e0.get("scalars") or {}).get(name)
        v1 = (e1.get("scalars") or {}).get(name)
        v2 = (e2.get("scalars") or {}).get(name)
        if v0 is None or v1 is None or v2 is None:
            continue
        results[name] = roache_gci(
            float(v0),
            float(v1),
            float(v2),
            r=r,
            fs=fs,
            threshold=thresh_map.get(name),
        )

    return {
        "case_id": case_id,
        "family": family,
        "levels": list(levels),
        "r": r,
        "fs": fs,
        "run_ids": {
            str(l_coarse): e0.get("run_id"),
            str(l_med): e1.get("run_id"),
            str(l_fine): e2.get("run_id"),
        },
        "n_cells": {
            str(l_coarse): (e0.get("params") or {}).get("n_cells"),
            str(l_med): (e1.get("params") or {}).get("n_cells"),
            str(l_fine): (e2.get("params") or {}).get("n_cells"),
        },
        "scalars": results,
        "all_passed": all(
            v["pass_threshold"] is not False for v in results.values()
        ),
    }


def re_strip_level_suffix(run_id: str) -> str:
    """Strip a trailing '_L0' / '_L1' / '_L2' / '_L3' level suffix from a run_id."""
    for suffix in ("_L0", "_L1", "_L2", "_L3"):
        if run_id.endswith(suffix):
            return run_id[: -len(suffix)]
    return run_id


def group_manifest_for_gci(
    manifest_rows: list[Mapping[str, Any]],
    *,
    levels: tuple[int, int, int] = (0, 1, 2),
) -> dict[str, list[Mapping[str, Any]]]:
    """Group successful manifest rows by ``case_id`` (keeping latest ok entry per level)."""
    grouped: dict[str, dict[int, Mapping[str, Any]]] = {}
    for row in manifest_rows:
        if row.get("status") != "ok":
            continue
        cid = str(row.get("case_id") or re_strip_level_suffix(str(row["run_id"])))
        lvl = int(row["level"])
        grouped.setdefault(cid, {})[lvl] = row
    l0, l1, l2 = levels
    return {
        cid: [lvls[l0], lvls[l1], lvls[l2]]
        for cid, lvls in sorted(grouped.items())
        if all(k in lvls for k in levels)
    }


def format_summary_table(reports: list[Mapping[str, Any]]) -> str:
    """Format a Markdown summary table across all GCI case reports."""
    lines = [
        "| Case | Family | Scalar | Conv | p | f_fine | f_ext | GCI_fine (%) | AR | Pass |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for rep in reports:
        cid = rep["case_id"]
        fam = rep["family"]
        for s_name, m in rep["scalars"].items():
            p_str = f"{m['p']:.3f}" if m["p"] is not None else "-"
            f_ext_str = f"{m['f_ext']:.5g}" if m["f_ext"] is not None else "-"
            gci_str = f"{100.0 * m['gci_fine']:.3f}" if m["gci_fine"] is not None else "-"
            ar_str = f"{m['asymptotic_ratio']:.4f}" if m["asymptotic_ratio"] is not None else "-"
            pass_str = "PASS" if m["pass_threshold"] else ("FAIL" if m["pass_threshold"] is False else "-")
            lines.append(
                f"| {cid} | {fam} | {s_name} | {m['convergence']} | {p_str} | "
                f"{m['f_fine']:.5g} | {f_ext_str} | {gci_str} | {ar_str} | {pass_str} |"
            )
    return "\n".join(lines) + "\n"


def run_gci_from_manifest(
    manifest_path: Path,
    out_dir: Path,
    *,
    r: float = DEFAULT_R,
    fs: float = DEFAULT_FS,
    levels: tuple[int, int, int] = (0, 1, 2),
    scalar_names: tuple[str, ...] = ("dp_star", "xr_over_h1", "max_wall_tau_star", "f_window"),
    upload_bucket: str | None = None,
    prefix: str = "corpus/v1",
    subdir: str = "gci",
) -> tuple[list[dict[str, Any]], str]:
    """Read manifest JSONL, write ``<out_dir>/<prefix>/<subdir>/<family>/<case>.json``, and return table."""
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                rows.append(json.loads(s))

    grouped = group_manifest_for_gci(rows, levels=levels)
    reports: list[dict[str, Any]] = []
    prefix_clean = prefix.strip("/")
    subdir_clean = subdir.strip("/")

    for cid, trio in grouped.items():
        rep = analyze_case_gci(trio, r=r, fs=fs, levels=levels, scalar_names=scalar_names)
        reports.append(rep)
        fam = rep["family"]
        dst = out_dir / prefix_clean / subdir_clean / fam / f"{cid}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")

        if upload_bucket:
            uri = f"{upload_bucket.rstrip('/')}/{prefix_clean}/{subdir_clean}/{fam}/{cid}.json"
            proc = subprocess.run(
                ["gcloud", "storage", "cp", str(dst), uri],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Failed to upload {uri}: {proc.stderr.strip()}")

    table = format_summary_table(reports)
    summary_path = out_dir / prefix_clean / subdir_clean / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(table, encoding="utf-8")
    if upload_bucket:
        uri = f"{upload_bucket.rstrip('/')}/{prefix_clean}/{subdir_clean}/summary.md"
        subprocess.run(["gcloud", "storage", "cp", str(summary_path), uri], check=True)

    return reports, table


def main() -> None:
    """CLI entrypoint for Roache 3-level GCI analysis."""
    parser = argparse.ArgumentParser(description="Compute Roache 3-level GCI from manifest.jsonl.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest.jsonl")
    parser.add_argument("--out-dir", type=Path, default=Path("work_gci"))
    parser.add_argument("--r", type=float, default=DEFAULT_R)
    parser.add_argument("--fs", type=float, default=DEFAULT_FS)
    parser.add_argument(
        "--levels",
        type=str,
        default="0,1,2",
        help="Comma-separated 3 mesh levels (e.g. '0,1,2' or '1,2,3')",
    )
    parser.add_argument(
        "--upload-bucket",
        type=str,
        default=None,
        help="Optional GCS bucket (e.g. gs://<corpus-bucket>) to upload gci JSONs",
    )
    parser.add_argument("--prefix", type=str, default="corpus/v1")
    parser.add_argument("--subdir", type=str, default="gci")
    args = parser.parse_args()

    lvl_parts = tuple(int(x.strip()) for x in args.levels.split(",") if x.strip())
    if len(lvl_parts) != 3:
        raise SystemExit(f"--levels must specify exactly 3 integers, got {args.levels!r}")

    reports, table = run_gci_from_manifest(
        args.manifest,
        args.out_dir,
        r=args.r,
        fs=args.fs,
        levels=lvl_parts,  # type: ignore[arg-type]
        upload_bucket=args.upload_bucket,
        prefix=args.prefix,
        subdir=args.subdir,
    )
    print(f"Analyzed {len(reports)} GCI cases (levels={lvl_parts}, r={args.r}, F_s={args.fs}):\n")
    print(table)


if __name__ == "__main__":
    main()
