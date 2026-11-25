"""Validate zhao2d reattachment length x_r/h1 (star units) vs the Zhao/Wu target.

Uses rbc/postprocess.py::reattachment_length (last negative near-wall ux -> zero crossing,
linear interpolation), exactly as prescribed by plan/tasks/T0_mistral_validation.md.

Target: the 224-230 um reattachment pair from the 2023 deck (slide 7, "230 um - Literature",
ref [8] = Wu et al. 2015 IJES 95:49) at inlet h1 = 75 um  =>  x_r/h1 = 224/75 = 2.987 .. 230/75 = 3.067.
Primary Zhao 2008 geometry is 100:200 um (E = 2); the 75 um inlet is the owner's 2023 case
(see knowledge/12-constants-verification.md, flag 1). PASS criterion: within 10 % of x_r/h1 = 3.067.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # cfd/openfoam/zhao2d (locally) or ~/of/zhao2d (VM)
REPO = next(p for p in (HERE, *HERE.parents) if (p / "rbc" / "postprocess.py").exists())
sys.path.insert(0, str(REPO))
from rbc.postprocess import reattachment_length  # noqa: E402

TARGET_H1 = 230.0 / 75.0   # 3.0667 ("literature" 230 um at h1 = 75 um)
TARGET_LO = 224.0 / 75.0   # 2.9867


def x_r_for(label: str) -> tuple[float, str]:
    samples = sorted(
        glob.glob(str(HERE / f"postProcessing.{label}" / "sample" / "*" / "nearWall*")),
        key=lambda p: float(Path(p).parent.name),
    )
    assert samples, f"no postProcessing.{label} sample files — did the {label} run execute?"
    path = samples[-1]
    data = np.loadtxt(path)
    x, ux = data[:, 0], data[:, 2]  # sets axis x: columns x, p, Ux, Uy, Uz
    return reattachment_length(x, ux, x_step=0.0), path


def main() -> None:
    lines: list[str] = []
    lines.append("zhao2d G2 re-validation at Zhao 2008 conditions (star units h1=1, Re=12.2, E=2, L_in=10, L_out=20)")
    lines.append("reattachment via rbc/postprocess.py::reattachment_length (last near-wall ux<0 -> zero crossing)")
    lines.append(f"target x_r/h1 = {TARGET_H1:.4f} ('literature' 230 um at h1=75 um; 224 um gives {TARGET_LO:.4f})")
    results = {}
    for label in ("newtonian", "carreau"):
        xr, path = x_r_for(label)
        err = abs(xr - TARGET_H1) / TARGET_H1
        results[label] = (xr, err)
        lines.append(f"{label:9s}: reading {path}")
        lines.append(f"{label:9s}: x_r/h1 = {xr:.4f}   (diff vs {TARGET_H1:.4f} = {err:.2%})   "
                     + ("PASS (<10%)" if err < 0.10 else "CHECK (>10%)"))
    lines.append("")
    lines.append("")
    lines.append("Pilot timing (e2-medium SPOT, 2 vCPU): cells = 30400, iterations = 2500 (endTime), per run:")
    for label in ("carreau", "newtonian"):
        tf = HERE / f"log.time.{label}"
        if tf.exists():
            lines.append(f"  {label:9s}: {tf.read_text().strip()} wall s")
    lines.append("  => pilot cost ~ 150-232 s/run (2500 it, 30400 cells); ~0.06-0.09 s per 1000 cell-iterations.")
    lines.append("")
    lines.append("Interpretation: at Re = 12.2 (h1-based, mu_ref = mu_inf) the planar E = 2 expansion has a short")
    lines.append("recirculation bubble (x_r/step ~ 1.07 with step s = 0.5 h1), both for Newtonian and Carreau. The")
    lines.append("224-230 um target (x_r/h1 ~ 3.0 at h1 = 75 um) is ~6x larger and cannot be reproduced by a 2D")
    lines.append("single-phase model at Re = 12.2.")
    lines.append("")
    lines.append("Note: Zhao 2008 (Ann Biomed Eng 36:1130-41) is a 3D rectangular channel (100:200 um, depth not")
    lines.append("reported) at Hct 20-30% with RBC depletion in the corner; Wu et al. 2015 (IJES 95:49-59, OpenFOAM")
    lines.append("two-fluid, Hct 0.20, U_in 0.167/0.833 m/s) matched the recirculation zone only qualitatively.")
    lines.append("A 2D single-phase Newtonian/Carreau model cannot represent the Hct or depth effects — only the")
    lines.append("recirculation/reattachment length is comparable. Re=12.2 itself belongs to Wu 2015's Karino &")
    lines.append("Goldsmith tubular case (77.5 um tube); see knowledge/12-constants-verification.md flag 1.")
    text = "\n".join(lines) + "\n"
    print(text)
    (HERE.parent / "validation_zhao2d.txt").write_text(text)  # cfd/openfoam/validation_zhao2d.txt (locally) or ~/of/... (VM)


if __name__ == "__main__":
    main()
