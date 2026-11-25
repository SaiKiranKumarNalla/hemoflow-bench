"""Generate one tarball-ready nondimensional OpenFOAM example case per family (G2–G7)."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

from cfd.gen.case import write_case


def _example_specs() -> tuple[dict[str, Any], ...]:
    """One representative nondimensional case per family (mix of Newtonian & BirdCarreau)."""
    return (
        {
            "run_id": "EX_G2_expansion",
            "family": "G2",
            "rheology": "BirdCarreau",
            "Re": 5.0,
            "beta": 16.23,  # Cho & Kensey 1991 ratio 0.056 / 0.00345 (TODO: verify)
            "Cu": 1000.0,
            "n": 0.3568,
            "a": 2.0,
            "geometry": {"E": 2.0, "L_in": 5.0, "L_out_ratio": 10.0},
        },
        {
            "run_id": "EX_G3_stenosis",
            "family": "G3",
            "rheology": "BirdCarreau",
            "Re": 4.0,
            "beta": 16.23,
            "Cu": 500.0,
            "n": 0.3568,
            "a": 2.0,
            "geometry": {"throat_ratio": 0.5, "L_t": 2.0, "L_in": 5.0, "L_out_ratio": 10.0},
        },
        {
            "run_id": "EX_G4_sym_bifurc",
            "family": "G4",
            "rheology": "Newtonian",
            "Re": 4.0,
            "geometry": {"theta_deg": 45.0, "daughter_ratio": 0.75, "L_in": 5.0, "L_out_ratio": 10.0},
        },
        {
            "run_id": "EX_G5_asym_bifurc",
            "family": "G5",
            "rheology": "BirdCarreau",
            "Re": 6.0,
            "beta": 12.0,
            "Cu": 200.0,
            "n": 0.4,
            "a": 2.0,
            "geometry": {"width_ratio": 0.65, "theta_deg": 60.0, "L_in": 5.0, "L_out_ratio": 10.0},
        },
        {
            "run_id": "EX_G6_serpentine",
            "family": "G6",
            "rheology": "BirdCarreau",
            "Re": 3.0,
            "beta": 16.23,
            "Cu": 1500.0,
            "n": 0.3568,
            "a": 2.0,
            "geometry": {"Rc_ratio": 2.5, "n_bends": 2, "L_in": 5.0, "L_out_ratio": 10.0},
        },
        {
            "run_id": "EX_G7_network_cell",
            "family": "G7",
            "rheology": "BirdCarreau",
            "Re": 5.0,
            "beta": 16.23,
            "Cu": 800.0,
            "n": 0.3568,
            "a": 2.0,
            "geometry": {"E": 2.0, "theta_deg": 45.0, "L_in": 5.0, "L_out_ratio": 10.0},
        },
    )


def build_examples(out_root: Path, level: int = 0, make_tarball: bool = True) -> dict[str, int]:
    """Generate one example case per family G2–G7 and optionally pack examples.tar.gz."""
    out_root.mkdir(parents=True, exist_ok=True)
    cell_counts: dict[str, int] = {}

    for spec in _example_specs():
        fam = str(spec["family"])
        case_dir = out_root / fam
        mesh = write_case(case_dir, spec, level=level)
        cell_counts[fam] = sum(nx * ny * nz for _, (nx, ny, nz), _ in mesh.blocks)

    if make_tarball:
        tar_path = out_root / "m1_examples_g2_g7.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            for spec in _example_specs():
                fam = str(spec["family"])
                tf.add(out_root / fam, arcname=fam)

    return cell_counts


def main() -> None:
    """CLI entrypoint."""
    out_root = Path(__file__).resolve().parent / "examples"
    counts = build_examples(out_root, level=0, make_tarball=True)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
