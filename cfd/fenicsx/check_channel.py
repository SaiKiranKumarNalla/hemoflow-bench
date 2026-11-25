"""Gate F1: straight channel, uniform inlet phi, developed inlet -> mixture Poiseuille (dp* = 12 mu_hat L, w = u)."""
import sys

import numpy as np
from dolfinx import fem
from mpi4py import MPI

sys.path.insert(0, "/work/cfd/fenicsx")
import twofluid as tf  # noqa: E402

LENGTH = 10.0
for phi_in in (0.01, 0.25, 0.45, 0.55):
    prm = tf.TwoFluidParams(re_p=10.0, d_hat=0.08, phi_in=phi_in, supg=False)
    msh, tags = tf.channel_mesh(LENGTH, 100, 20)
    W = tf.build_space(msh)
    U = fem.Function(W)
    tf.initial_guess(W, U, prm, LENGTH)
    bcs = tf.boundary_conditions(W, tags, phi_in)
    U.x.array[:] += 1e-3 * np.random.default_rng(0).standard_normal(U.x.array.size)  # perturb to exercise Newton
    for bc in bcs:
        bc.set(U.x.array)
    n_it, ok = tf.solve(W, U, tf.residual(W, U, prm, None), bcs)
    u, w, p, phi = (U.sub(i).collapse() for i in range(4))
    pts = np.array([[0.0, 0.5, 0.0], [LENGTH, 0.5, 0.0]])
    from dolfinx import geometry
    tree = geometry.bb_tree(msh, 2)
    cells = geometry.compute_colliding_cells(msh, geometry.compute_collisions_points(tree, pts), pts)
    pv = [p.eval(pts[i], cells.links(i)[0])[0] for i in range(2)]
    dp = pv[0] - pv[1]
    mu = float(tf.mixture_viscosity_hat(phi_in))
    slip = np.max(np.abs(w.x.array - u.x.array))
    print(f"phi_in={phi_in:.2f} newton={n_it} ok={ok} dp*={dp:.4f} exact={12*mu*LENGTH:.4f} "
          f"err={abs(dp/(12*mu*LENGTH)-1):.2e} max|w-u|={slip:.2e} phi range=[{phi.x.array.min():.4f},"
          f"{phi.x.array.max():.4f}]", flush=True)
