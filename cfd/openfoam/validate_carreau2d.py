"""Validate OpenFOAM BirdCarreau 2D channel against 1D Carreau shooting reference."""
import glob
from pathlib import Path
import numpy as np

RHO = 1060.0
HERE = Path(__file__).resolve().parent
ref = np.load(HERE / "carreau_1d_ref.npz")
samples = sorted(
    glob.glob(str(HERE / "carreau2d" / "postProcessing" / "sample" / "*" / "lineX9*")),
    key=lambda p: float(Path(p).parent.name),
)
assert samples, "no carreau2d sample files — did the solver run?"
path = samples[-1]
print("reading", path)
data = np.loadtxt(path)
y, p_kin, ux = data[:, 0], data[:, 1], data[:, 2]
u_exact = np.interp(y, ref["y"], ref["u"])
rel_l2 = float(np.sqrt(np.mean((ux - u_exact) ** 2) / np.mean(u_exact**2)))
G_num = float(np.mean(p_kin) * RHO / 50e-6)  # p(x=450um) over 50um remaining to outlet p=0
G_ref = float(ref["G"])
print(f"rel L2 Ux vs 1D Carreau = {rel_l2:.4%}")
print(f"u_max = {ux.max():.5f} m/s (1D ref {ref['u'].max():.5f})")
print(f"G     = {G_num:.4e} Pa/m (1D ref {G_ref:.4e}, err {abs(G_num - G_ref)/G_ref:.2%})")
print("PASS (<2%)" if rel_l2 < 0.02 else "FAIL (>=2%)")
