"""Validate 2D BirdCarreau sudden expansion reattachment length vs 224-230 um target."""
import glob
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
samples = sorted(
    glob.glob(str(HERE / "expansion2d" / "postProcessing" / "sample" / "*" / "nearWall*")),
    key=lambda p: float(Path(p).parent.name),
)
assert samples, "no expansion2d sample files — did the solver run?"
path = samples[-1]
print("reading", path)
data = np.loadtxt(path)
x_um, ux = data[:, 0] * 1e6, data[:, 2]
neg = np.flatnonzero(ux < 0)
assert len(neg) and neg[-1] < len(x_um) - 1, f"no recirculation zero-crossing found (min ux={ux.min():.4f})"
k = neg[-1]
lr_um = float(x_um[k] - ux[k] * (x_um[k + 1] - x_um[k]) / (ux[k + 1] - ux[k]))
err = abs(lr_um - 224.0) / 224.0
print(f"Reattachment length L_r = {lr_um:.2f} um (target 224-230 um, diff vs 224 = {err:.2%})")
print("PASS (<10%)" if err < 0.10 else "CHECK (>10%)")
