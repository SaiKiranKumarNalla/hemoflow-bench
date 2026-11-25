"""Benchmark F3c: Zhao/Sezer expansion, V4 (Wu per-phase + Saffman lift), solved with COMSOL's documented stationary
strategy (pseudo time stepping Eq. 3-60, CFL schedule Eq. 3-61, damped Newton) — knowledge/17 §5.
Usage: python3 bench_zhao_comsol.py [n_per_unit] [alpha] [max_iter]
"""
import sys
import time

import numpy as np
from dolfinx import fem

sys.path.insert(0, "/work/cfd/fenicsx")
import bench_zhao as bz  # noqa: E402
import twofluid as tf  # noqa: E402
from comsol_solver import ComsolStationary, cfl_comsol  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
max_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 120
msh, tags = bz.zhao_mesh(n)
W = tf.build_space(msh)
U, U_prev = fem.Function(W), fem.Function(W)
for sub in (0, 1):
    U.sub(sub).interpolate(lambda x: np.vstack([np.where(np.abs(x[1]) < 0.5, bz.channel_profile(np.clip(x[1], -0.5, 0.5)), 0.0),
                                                np.zeros_like(x[0])]))
U.sub(3).interpolate(lambda x: np.full_like(x[0], alpha))
U.x.scatter_forward()
bcs = tf.boundary_conditions(W, tags, alpha, inlet_profile=bz.channel_profile, pin_outlet_p=True)
for bc in bcs:
    bc.set(U.x.array)
# initial guess = converged V3 (COMSOL model) at the same conditions, as one would start a COMSOL study
bcs0 = tf.boundary_conditions(W, tags, alpha, inlet_profile=bz.channel_profile)
for re0 in (1.0, bz.RE_P):
    tf.solve(W, U, tf.residual(W, U, tf.TwoFluidParams(re_p=re0, d_hat=bz.D_HAT, phi_in=alpha, closure="comsol-solid")), bcs0)
print("V3 initial guess converged", flush=True)
import os  # noqa: E402
WL = os.environ.get("WALL_LIFT", "0") == "1"
prm = tf.TwoFluidParams(re_p=bz.RE_P, d_hat=bz.D_HAT, phi_in=alpha, form="wu-phase", lift=True, supg=True,
                        wall_lift=WL, phillips=WL, wall_lift_farfield=os.environ.get("FARFIELD", "0") == "1")
psi_w = tf.wall_psi(msh, tags) if WL else None
print(f"V4 wall_lift={WL} phillips={WL} farfield={prm.wall_lift_farfield} C_A={prm.wall_lift_C:.3f}", flush=True)
cfl = fem.Constant(msh, 1.0)
import ufl  # noqa: E402
ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
F = tf.residual(W, U, prm, ds=ds, U_old=U_prev, pseudo_cfl=cfl, psi_wall=psi_w)
solver = ComsolStationary(U, U_prev, F, bcs, cfl)
print(f"cells={msh.topology.index_map(2).size_local} alpha={alpha} Re_p={bz.RE_P:.2f} (COMSOL pseudo-time + damped Newton)",
      flush=True)
t0 = time.time()
for k in range(1, max_iter + 1):
    lam, ndx, nE = solver.step(k)
    done = k >= 50 and ndx < 1e-6
    if k % 5 == 0 or k <= 3 or done or k == max_iter:
        phi, p = U.sub(3).collapse(), U.sub(2).collapse()
        pin = bz.probe(msh, p, np.array([[-bz.L1, 0.0, 0.0]]))[0, 0]
        cup, prof_up = bz.cfl_thickness(msh, phi, -1.0, bz.H1)
        cdn, _ = bz.cfl_thickness(msh, phi, 0.66 * (bz.L1 + bz.L2) - bz.L1, bz.H2)
        X = phi.function_space.tabulate_dof_coordinates()
        bad = np.argsort(np.abs(phi.x.array - alpha))[-2:]
        print(f"k={k:3d} CFL={cfl_comsol(k):8.1f} lam={lam:.3f} |dU|={ndx:.2e} |E|={nE:.2e} wall={time.time() - t0:5.0f}s "
              f"dP*={pin * 2 / bz.RE_P:8.3f} phi=[{phi.x.array.min():.4f},{phi.x.array.max():.4f}] CFL*up={cup:.4f} "
              f"CFL*dn={cdn:.4f} wall/core_up={prof_up[-1]:.4f}/{prof_up[0]:.4f} "
              f"worst@{[(round(X[i, 0], 1), round(X[i, 1], 2)) for i in bad]}", flush=True)
    if done:
        print("CONVERGED (steady)", flush=True)
        break


# SAVE_FIELDS: write 2D fields for plotting (same grid as bench_expansion_compare)
import bench_expansion_compare as bec  # noqa: E402
phi_f, w_f = U.sub(3).collapse(), U.sub(1).collapse()
xs, ys, phig, _ = bec.grid(msh, phi_f)
_, _, ux, uy = bec.grid(msh, w_f)
tag = "v4wl" if WL else "v4"
np.savez(f"/work/data/validation/expansion_{tag}_a{alpha:g}.npz", xg=xs, yg=ys, phi=phig, ux=ux, uy=uy, re=bz.RE_P)
print("saved fields", tag, flush=True)
