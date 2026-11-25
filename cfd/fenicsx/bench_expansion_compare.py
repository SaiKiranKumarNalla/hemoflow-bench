"""Comparison F6: mixture model vs Euler-Euler V3 on the Zhao/Sezer sudden expansion (knowledge/15, knowledge/18).

h1 = 100 um -> h2 = 200 um, L1 = 1 mm, L2 = 2 mm (bench_zhao.zhao_mesh); Ht 0.2 (Zhao 2008); d_RBC 8 um;
Wu densities 1027 / 1093 kg/m3, plasma 0.96 mPa s; U = Q / (100 um x 100 um) as in Sezer; gravity along -y.
Usage: python3 bench_expansion_compare.py <Q_uL_min> [n_per_unit]
"""
import sys
import time

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, geometry
from mpi4py import MPI

sys.path.insert(0, "/work/cfd/fenicsx")
import bench_zhao as bz  # noqa: E402
import twofluid as tf  # noqa: E402

H, PHI_IN, D = 100e-6, 0.2, 8e-6
MU, RHO_P, RHO_R, G = 0.96e-3, 1027.0, 1093.0, 9.81
PHI_MAX = 2.0 / 2.07


def scales(q):
    u = q * 1e-9 / 60.0 / (H * H)
    return u, RHO_P * u * H / MU, (RHO_R - RHO_P) * G * H**2 / (MU * u), (RHO_R - RHO_P) * G * D**2 / (18 * MU) / u


def inlet(x):
    return np.vstack([np.where(np.abs(x[1]) < 0.5, bz.channel_profile(np.clip(x[1], -0.5, 0.5)), 0.0), np.zeros_like(x[0])])


def mixture(msh, tags, re, buoy, vs, diff=1e-4):
    cell = msh.basix_cell()
    W = fem.functionspace(msh, basix.ufl.mixed_element([basix.ufl.element("Lagrange", cell, 2, shape=(2,)),
                                                        basix.ufl.element("Lagrange", cell, 1),
                                                        basix.ufl.element("Lagrange", cell, 1)]))
    U = fem.Function(W)
    u, p, phi = ufl.split(U)
    v, q, s = ufl.TestFunctions(W)
    gs = fem.Constant(msh, 0.0)
    rs = fem.Constant(msh, 1.0)
    ph = ufl.min_value(ufl.max_value(phi, 0.0), 0.999 * PHI_MAX)
    mu = (1.0 - ph / PHI_MAX) ** (-2.5 * PHI_MAX)
    gvec = ufl.as_vector((0.0, -1.0))
    flux = phi * u + gs * ph * (1.0 - ph) * (1.0 - ph / PHI_MAX) * vs * gvec
    n = ufl.FacetNormal(msh)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    F = (rs * re * ufl.inner(ufl.grad(u) * u, v) + 2.0 * mu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v)))
         - p * ufl.div(v) - gs * buoy * phi * ufl.dot(gvec, v) + q * ufl.div(u)
         - ufl.dot(flux, ufl.grad(s)) + diff * ufl.dot(ufl.grad(phi), ufl.grad(s))) * ufl.dx \
        + ufl.dot(flux, n) * s * ds(tf.OUTLET)
    msh.topology.create_connectivity(1, 2)
    V0, _ = W.sub(0).collapse()
    inflow, zero = fem.Function(V0), fem.Function(V0)
    inflow.interpolate(inlet)
    bcs = [fem.dirichletbc(inflow, fem.locate_dofs_topological((W.sub(0), V0), 1, tags.find(tf.INLET)), W.sub(0)),
           fem.dirichletbc(zero, fem.locate_dofs_topological((W.sub(0), V0), 1, tags.find(tf.WALL)), W.sub(0))]
    Q2, _ = W.sub(2).collapse()
    pin = fem.Function(Q2)
    pin.x.array[:] = PHI_IN
    bcs.append(fem.dirichletbc(pin, fem.locate_dofs_topological((W.sub(2), Q2), 1, tags.find(tf.INLET)), W.sub(2)))
    U.sub(0).interpolate(inlet)
    U.sub(2).interpolate(lambda x: np.full_like(x[0], PHI_IN))
    n_tot = 0
    for r_f, g_f in ((0.05, 0.0), (0.3, 0.0), (1.0, 0.0)) + (tuple((1.0, g) for g in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)) if buoy > 0.1 else ((1.0, 1.0),)):
        rs.value, gs.value = r_f, g_f
        n_it, _ = tf.solve(W, U, F, bcs, max_it=40)
        n_tot += n_it
    return U.sub(2).collapse(), U.sub(0).collapse(), n_tot


def euler_euler(msh, tags, re, buoy):
    W = tf.build_space(msh)
    U = fem.Function(W)
    for sub in (0, 1):
        U.sub(sub).interpolate(inlet)
    U.sub(3).interpolate(lambda x: np.full_like(x[0], PHI_IN))
    U.x.scatter_forward()
    bcs = tf.boundary_conditions(W, tags, PHI_IN, inlet_profile=bz.channel_profile)
    steps = [(r, 0.0) for r in (0.05, 0.3, 1.0)] + ([(1.0, g) for g in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)] if buoy > 0.1 else [(1.0, 1.0)])
    n_tot = 0
    for r_f, g_f in steps:
        prm = tf.TwoFluidParams(re_p=r_f * re, d_hat=D / H, phi_in=PHI_IN, closure="comsol-solid",
                                gravity=g_f * buoy, lift=False)
        n_it, _ = tf.solve(W, U, tf.residual(W, U, prm), bcs, max_it=40)
        n_tot += n_it
    return U.sub(3).collapse(), U.sub(1).collapse(), n_tot


def grid(msh, f, nx=500, ny=101):
    xs, ys = np.linspace(-bz.L1, bz.L2, nx), np.linspace(-0.999, 0.999, ny)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    tree = geometry.bb_tree(msh, 2)
    cells = geometry.compute_colliding_cells(msh, geometry.compute_collisions_points(tree, pts), pts)
    ncomp = f.function_space.value_size if hasattr(f.function_space, "value_size") else 1
    out = np.full((len(pts), 2), np.nan)
    for i in range(len(pts)):
        c = cells.links(i)
        if len(c):
            val = f.eval(pts[i], c[0])
            out[i, : len(val)] = val
    return xs, ys, out[:, 0].reshape(ny, nx), out[:, 1].reshape(ny, nx)


if __name__ == "__main__":
    qv = float(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    u_ref, re, buoy, vs = scales(qv)
    msh, tags = bz.zhao_mesh(n)
    print(f"Q={qv} uL/min U={u_ref:.4g} m/s Re={re:.3g} buoyancy={buoy:.3g} Vs/U={vs:.3g} cells={msh.topology.index_map(2).size_local}",
          flush=True)
    for name, fn in (("mixture", lambda: mixture(msh, tags, re, buoy, vs)), ("euler-euler", lambda: euler_euler(msh, tags, re, buoy))):
        t0 = time.time()
        phi, vel, n_it = fn()
        xs, ys, phig, _ = grid(msh, phi)
        _, _, ux, uy = grid(msh, vel)
        print(f"{name:12s} newton={n_it} {time.time() - t0:5.0f}s phi range [{np.nanmin(phig):.4f},{np.nanmax(phig):.4f}]",
              flush=True)
        np.savez(f"/work/data/validation/expansion_{name}_q{qv:g}.npz", xg=xs, yg=ys, phi=phig, ux=ux, uy=uy,
                 re=re, q=qv)
