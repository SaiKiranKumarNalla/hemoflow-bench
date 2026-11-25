"""Validate the OpenFOAM 2D channel run against the exact Poiseuille profile.

Reads the latest sample line (x=9, across y in [0,1]) written by the sets functionObject
(axis y, raw format -> columns: y  p  Ux  Uy  Uz), compares Ux against
u(y) = 6 Uavg y (H-y) / H^2 (H=1, Uavg=2/3 -> Umax=1), and checks rel L2 < 2%.
"""
import glob
import os
from pathlib import Path
import numpy as np

UAVG = 2.0 / 3.0
H = 1.0

base = Path(__file__).resolve().parent / "channel2d" / "postProcessing" / "sample"
samples = sorted(
    glob.glob(str(base / "*" / "lineX9*")),
    key=lambda p: float(Path(p).parent.name),
)
assert samples, f"no sample files under {base} — did the solver run?"
path = samples[-1]
print("reading", path)
data = np.loadtxt(path)
y = data[:, 0]
p = data[:, 1]
ux = data[:, 2]  # columns (axis y, fields U p -> _p_U.xy): y  p  Ux  Uy  Uz
u_exact = 6.0 * UAVG * y * (H - y) / H**2
rel_l2 = float(np.sqrt(np.mean((ux - u_exact) ** 2) / np.mean(u_exact**2)))
p_exact = 8.0 * (10.0 - 9.0)  # G = 12*mu*Uavg/H^2 = 8, p(x=9) = G*(L-x) = 8.0
p_mean = float(np.mean(p))
print(f"rel L2 Ux vs Poiseuille = {rel_l2:.4%}")
print(f"centerline u_max = {ux.max():.6f} (exact 1.000000)")
print(f"pressure p(x=9)  = {p_mean:.4f} (exact {p_exact:.4f}, rel err {abs(p_mean - p_exact)/p_exact:.2%})")
print("PASS (<2%)" if rel_l2 < 0.02 else "FAIL (>=2%)")
