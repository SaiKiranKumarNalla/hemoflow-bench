"""Benchmark F3b: Zhao/Sezer expansion, V4 (Wu per-phase form + Saffman lift), backward-Euler time marching to
t* = T (Wu 2015 reports Zhao at steady state t = 0.15 s -> t* = t U/h1 = 250 at 100 uL/min). Coarse dev mesh.
Usage: python3 bench_zhao_transient.py [n_per_unit] [alpha] [dt] [T]
"""
import sys
import time

import numpy as np
from dolfinx import fem

sys.path.insert(0, "/work/cfd/fenicsx")
import bench_zhao as bz  # noqa: E402
import twofluid as tf  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
dt = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
T = float(sys.argv[4]) if len(sys.argv) > 4 else 250.0
msh, tags = bz.zhao_mesh(n)
W = tf.build_space(msh)
U, U_old = fem.Function(W), fem.Function(W)
for sub in (0, 1):
    U.sub(sub).interpolate(lambda x: np.vstack([np.where(np.abs(x[1]) < 0.5, bz.channel_profile(np.clip(x[1], -0.5, 0.5)), 0.0),
                                                np.zeros_like(x[0])]))
U.sub(3).interpolate(lambda x: np.full_like(x[0], alpha))
U.x.scatter_forward()
bcs = tf.boundary_conditions(W, tags, alpha, inlet_profile=bz.channel_profile, pin_outlet_p=True)
prm = tf.TwoFluidParams(re_p=bz.RE_P, d_hat=bz.D_HAT, phi_in=alpha, form="wu-phase", lift=True, supg=True)
dt_c = fem.Constant(msh, dt)
F = tf.residual(W, U, prm, U_old=U_old, dt=dt_c)
print(f"cells={msh.topology.index_map(2).size_local} alpha={alpha} dt*={dt} T*={T} Re_p={bz.RE_P:.2f}", flush=True)
t, k, t0 = 0.0, 0, time.time()
DT_MAX = float(sys.argv[5]) if len(sys.argv) > 5 else 50.0
# pseudo time stepping with an automatic step (COMSOL-equivalent CFL ramp): grow x1.5 on easy steps, halve on failure
while t < T - 1e-9:
    U_old.x.array[:] = U.x.array
    dt_c.value = min(dt, T - t)
    try:
        n_it, _ = tf.solve(W, U, F, bcs, max_it=12, linesearch="bt")
    except RuntimeError:
        U.x.array[:] = U_old.x.array
        dt *= 0.5
        print(f"  step failed -> dt*={dt:.3g}", flush=True)
        continue
    t += float(dt_c.value)
    k += 1
    if n_it <= 4:
        dt = min(dt * 1.5, DT_MAX)
    phi, p = U.sub(3).collapse(), U.sub(2).collapse()
    change = np.abs(U.x.array - U_old.x.array).max() / float(dt_c.value)
    if k % 3 == 0 or t >= T - 1e-9:
        pin = bz.probe(msh, p, np.array([[-bz.L1, 0.0, 0.0]]))[0, 0]
        cup, prof_up = bz.cfl_thickness(msh, phi, -1.0, bz.H1)
        cdn, prof_dn = bz.cfl_thickness(msh, phi, 0.66 * (bz.L1 + bz.L2) - bz.L1, bz.H2)
        X = phi.function_space.tabulate_dof_coordinates()
        bad = np.argsort(np.abs(phi.x.array - alpha))[-2:]
        print(f"t*={t:7.1f} dt*={float(dt_c.value):6.2f} newton={n_it} wall={time.time() - t0:5.0f}s dP*={pin * 2 / bz.RE_P:8.3f} "
              f"phi=[{phi.x.array.min():.4f},{phi.x.array.max():.4f}] CFL*up={cup:.4f} CFL*dn={cdn:.4f} "
              f"phi_wall_up={prof_up[-1]:.4f} core_up={prof_up[0]:.4f} d/dt={change:.2e} "
              f"worst@{[(round(X[i, 0], 1), round(X[i, 1], 2)) for i in bad]}", flush=True)
