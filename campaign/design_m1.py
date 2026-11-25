"""Scrambled Sobol experimental design for the M1 2D CFD corpus (families G2–G7).

Generates 64 nondimensional parameter points per family (384 runs total), with
10% Newtonian runs (6 per family, 36 total) and 90% BirdCarreau runs, written to
``campaign/m1_design.jsonl``.

Sampling rules (plan/tasks/T1_gemini_generator.md & plan/CORPUS_PLAN.md §1–§2):
- Re in [0.5, 25.0]: log-uniform
- beta = mu0 / muInf in [5.0, 30.0]: log-uniform  # TODO: source (patient range)
- Cu = lambda * U / h1 in [1.0, 1.0e5]: log-uniform  # TODO: source / confirm bounds
- n in [0.25, 0.60]: uniform  # TODO: source (patient range)
- a = 2.0: fixed Carreau-Yasuda exponent (Cho & Kensey 1991, TODO: verify)
- Geometry ratios: uniform on the ranges in plan/CORPUS_PLAN.md §2
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class FamilySpec:
    """Dimensionless parameter specification for a geometry family."""

    name: str
    # (param_name, low, high, is_discrete_int)
    geom_dims: tuple[tuple[str, float, float, bool], ...]


def _family_specs() -> tuple[FamilySpec, ...]:
    """Return immutable geometry parameter specifications for G2–G7."""
    return (
        FamilySpec("G2", (("E", 1.5, 4.0, False),)),
        FamilySpec("G3", (("throat_ratio", 0.3, 0.8, False), ("L_t", 1.0, 5.0, False))),
        FamilySpec("G4", (("theta_deg", 20.0, 90.0, False), ("daughter_ratio", 0.5, 1.0, False))),
        FamilySpec("G5", (("width_ratio", 0.5, 1.0, False), ("theta_deg", 30.0, 90.0, False))),
        FamilySpec("G6", (("Rc_ratio", 1.5, 6.0, False), ("n_bends", 1.0, 4.0, True))),
        FamilySpec("G7", (("E", 1.5, 4.0, False), ("theta_deg", 20.0, 90.0, False))),
    )


def _log_uniform(u: float, lo: float, hi: float) -> float:
    """Map u in [0, 1) to log-uniform on [lo, hi]."""
    return float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))


def _uniform(u: float, lo: float, hi: float) -> float:
    """Map u in [0, 1) to uniform on [lo, hi]."""
    return float(lo + u * (hi - lo))


def generate_design(
    n_per_family: int = 64,
    newtonian_fraction: float = 0.10,
    seed: int = 20261124,
    re_bounds: tuple[float, float] = (0.5, 25.0),
    beta_bounds: tuple[float, float] = (5.0, 30.0),       # TODO: source
    cu_bounds: tuple[float, float] = (1.0, 1.0e5),        # TODO: source
    n_bounds: tuple[float, float] = (0.25, 0.60),         # TODO: source
    carreau_a: float = 2.0,                               # Cho & Kensey 1991, TODO: verify
) -> list[dict[str, Any]]:
    """Generate scrambled Sobol M1 design records across families G2–G7.

    Parameters
    ----------
    n_per_family : int
        Number of Sobol points per family (must be a power of 2; default 64).
    newtonian_fraction : float
        Fraction of runs per family assigned Newtonian rheology (default 0.10 -> 6/64).
    seed : int
        Master RNG seed for deterministic scrambled Sobol sequences.

    Returns
    -------
    list[dict[str, Any]]
        One dictionary per simulation run, ready to serialize as JSONL.
    """
    if n_per_family <= 0 or (n_per_family & (n_per_family - 1)) != 0:
        raise ValueError(f"n_per_family must be a power of 2 for Sobol balance, got {n_per_family}")

    n_newtonian = round(newtonian_fraction * n_per_family)
    # Spread Newtonian indices evenly across the Sobol sequence so they remain well stratified.
    newtonian_indices = set(
        np.linspace(0, n_per_family - 1, num=n_newtonian, endpoint=False, dtype=int).tolist()
    )

    records: list[dict[str, Any]] = []
    for fam_offset, spec in enumerate(_family_specs()):
        # Dimensions: [Re, beta, Cu, n, *geom_dims]
        n_dims = 4 + len(spec.geom_dims)
        fam_seed = seed + fam_offset * 1009
        sampler = qmc.Sobol(d=n_dims, scramble=True, seed=fam_seed)
        samples = sampler.random(n=n_per_family)

        for idx in range(n_per_family):
            row = samples[idx]
            Re = _log_uniform(float(row[0]), re_bounds[0], re_bounds[1])
            is_newtonian = idx in newtonian_indices

            if is_newtonian:
                rheology = "Newtonian"
                beta: float | None = None
                Cu: float | None = None
                n_val: float | None = None
                a_val: float | None = None
            else:
                rheology = "BirdCarreau"
                beta = round(_log_uniform(float(row[1]), beta_bounds[0], beta_bounds[1]), 6)
                Cu = round(_log_uniform(float(row[2]), cu_bounds[0], cu_bounds[1]), 6)
                n_val = round(_uniform(float(row[3]), n_bounds[0], n_bounds[1]), 6)
                a_val = float(carreau_a)

            geom: dict[str, float | int] = {"L_in": 5.0, "L_out_ratio": 10.0}
            for g_i, (g_name, lo, hi, is_int) in enumerate(spec.geom_dims):
                u = float(row[4 + g_i])
                if is_int:
                    n_choices = int(hi - lo) + 1
                    val_int = min(int(hi), int(lo) + int(math.floor(u * n_choices)))
                    geom[g_name] = val_int
                else:
                    geom[g_name] = round(_uniform(u, lo, hi), 6)

            records.append(
                {
                    "run_id": f"M1_{spec.name}_{idx:03d}",
                    "family": spec.name,
                    "rheology": rheology,
                    "Re": round(Re, 6),
                    "beta": beta,
                    "Cu": Cu,
                    "n": n_val,
                    "a": a_val,
                    "geometry": geom,
                    "seed": fam_seed,
                }
            )

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
    """CLI entrypoint to generate campaign/m1_design.jsonl."""
    parser = argparse.ArgumentParser(description="Generate M1 scrambled Sobol design JSONL.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "m1_design.jsonl",
        help="Output JSONL path (default: campaign/m1_design.jsonl)",
    )
    parser.add_argument("--n-per-family", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()

    records = generate_design(n_per_family=args.n_per_family, seed=args.seed)
    out = write_design_jsonl(args.out, records)
    n_newt = sum(1 for r in records if r["rheology"] == "Newtonian")
    print(f"Wrote {len(records)} runs ({n_newt} Newtonian, {len(records) - n_newt} BirdCarreau) to {out}")


if __name__ == "__main__":
    main()
