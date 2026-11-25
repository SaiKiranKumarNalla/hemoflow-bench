"""Comparison F5: Piergiovanni et al. 2020 (Ann Biomed Eng 48:236) straight 50-um channel, mixture model vs our V3.

Planar 2D in the vertical plane (gravity along -y), channel S: L = 10 mm, h = 50 um; Ht 5 %; d_RBC 8 um; Wu densities
(1027 / 1093 kg/m3); carrier mu 1 mPa s (PBS, assumed); U = Q / h^2 (square section). Paper: RBC deposition near the outlet
at 0.25 uL/min, none above 10 uL/min (Fig. 1d-e).
(a) MIXTURE: one momentum eq. for the mixture + phi transport with algebraic Stokes settling slip (hindered by
    (1-phi)(1-phi/phi_max)), Krieger viscosity, buoyancy phi*(drho) g in the mixture momentum.
(b) EULER-EULER V3: tf.residual closure comsol-solid + reduced gravity on the RBC phase.
Usage: python3 bench_piergiovanni.py <Q_uL_min>
"""
import sys
import time

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, geometry
from mpi4py import MPI

sys.path.insert(0, "/work/cfd/fenicsx")
import twofluid as tf  # noqa: E402

H, LEN_MM, PHI_IN, D = 50e-6, 10.0, 0.05, 8e-6
MU, RHO_P, RHO_R, G = 1.0e-3, 1027.0, 1093.0, 9.81
L = LEN_MM * 1e-3 / H


def scales(q_ul_min):
    u = q_ul_min * 1e-9 / 60.0 / H**2
    re = RHO_P * u * H / MU
    buoy = (RHO_R - RHO_P) * G * H**2 / (MU * u)          # reduced-gravity number
    vs = (RHO_R - RHO_P) * G * D**2 / (18.0 * MU) / u      # Stokes settling / U
    return u, re, buoy, vs


def mixture(msh, tags, re, buoy, vs, phi_max=2.0 / 2.07, diff=1e-4):
    gs = fem.Constant(msh, 0.0)  # gravity continuation factor 0 -> 1
    cell = msh.basix_cell()
    W = fem.functionspace(msh, basix.ufl.mixed_element([basix.ufl.element("Lagrange", cell, 2, shape=(2,)),
                                                        basix.ufl.element("Lagrange", cell, 1),
                                                        basix.ufl.element("Lagrange", cell, 1)]))
    U = fem.Function(W)
    u, p, phi = ufl.split(U)
    v, q, s = ufl.TestFunctions(W)
    ph = ufl.min_value(ufl.max_value(phi, 0.0), 0.999 * phi_max)
    mu = (1.0 - ph / phi_max) ** (-2.5 * phi_max)
    gvec = ufl.as_vector((0.0, -1.0))
    flux = phi * u + gs * ph * (1.0 - ph) * (1.0 - ph / phi_max) * vs * gvec
    n = ufl.FacetNormal(msh)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    F = (re * ufl.inner(ufl.grad(u) * u, v) + 2.0 * mu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v)))
         - p * ufl.div(v) - gs * buoy * phi * ufl.dot(gvec, v) + q * ufl.div(u)
         - ufl.dot(flux, ufl.grad(s)) + diff * ufl.dot(ufl.grad(phi), ufl.grad(s))) * ufl.dx \
        + ufl.dot(flux, n) * s * ds(tf.OUTLET)
    fdim = 1
    msh.topology.create_connectivity(fdim, 2)
    V0, _ = W.sub(0).collapse()
    inflow, zero = fem.Function(V0), fem.Function(V0)
    inflow.interpolate(lambda x: np.vstack([tf.poiseuille(x[1]), np.zeros_like(x[0])]))
    bcs = [fem.dirichletbc(inflow, fem.locate_dofs_topological((W.sub(0), V0), fdim, tags.find(tf.INLET)), W.sub(0)),
           fem.dirichletbc(zero, fem.locate_dofs_topological((W.sub(0), V0), fdim, tags.find(tf.WALL)), W.sub(0))]
    Q2, _ = W.sub(2).collapse()
    pin = fem.Function(Q2)
    pin.x.array[:] = PHI_IN
    bcs.append(fem.dirichletbc(pin, fem.locate_dofs_topological((W.sub(2), Q2), fdim, tags.find(tf.INLET)), W.sub(2)))
    U.sub(0).interpolate(lambda x: np.vstack([tf.poiseuille(x[1]), np.zeros_like(x[0])]))
    U.sub(2).interpolate(lambda x: np.full_like(x[0], PHI_IN))
    n_tot = 0
    for g in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        gs.value = g
        n_it, _ = tf.solve(W, U, F, bcs, max_it=40)
        n_tot += n_it
    return U.sub(2).collapse(), U.sub(0).collapse(), n_tot


def euler_euler(msh, tags, re, buoy):
    W = tf.build_space(msh)
    U = fem.Function(W)
    tf.initial_guess(W, U, tf.TwoFluidParams(re_p=re, d_hat=D / H, phi_in=PHI_IN), L)
    bcs = tf.boundary_conditions(W, tags, PHI_IN)
    n_tot = 0
    for g in ((0.0, 0.1, 0.25, 0.5, 0.75, 1.0) if buoy > 0.1 else (1.0,)):
        prm = tf.TwoFluidParams(re_p=re, d_hat=D / H, phi_in=PHI_IN, closure="comsol-solid", gravity=g * buoy, lift=False)
        n_it, _ = tf.solve(W, U, tf.residual(W, U, prm), bcs, max_it=40)
        n_tot += n_it
    return U.sub(3).collapse(), U.sub(1).collapse(), n_tot


def profile(msh, f, x, comp=None):
    ys = np.linspace(0.001, 0.999, 401)
    pts = np.column_stack([np.full_like(ys, x), ys, np.zeros_like(ys)])
    tree = geometry.bb_tree(msh, 2)
    cells = geometry.compute_colliding_cells(msh, geometry.compute_collisions_points(tree, pts), pts)
    vals = np.array([f.eval(pts[i], cells.links(i)[0]) for i in range(len(ys))])
    return ys, (vals[:, 0] if comp is None else vals[:, comp])


def field_grid(msh, f, nx=600, ny=41, comp=None):
    xs, ys = np.linspace(0.0, L, nx), np.linspace(0.0005, 0.9995, ny)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    tree = geometry.bb_tree(msh, 2)
    cells = geometry.compute_colliding_cells(msh, geometry.compute_collisions_points(tree, pts), pts)
    vals = np.array([f.eval(pts[i], cells.links(i)[0]) for i in range(len(pts))])
    v = np.linalg.norm(vals, axis=1) if comp == "mag" else vals[:, 0]
    return xs, ys, v.reshape(ny, nx)


if __name__ == "__main__":
    q = float(sys.argv[1])
    u_ref, re, buoy, vs = scales(q)
    msh, tags = tf.channel_mesh(L, int(4 * L), 20)
    print(f"Q={q} uL/min U={u_ref:.4g} m/s Re={re:.3g} buoyancy={buoy:.3g} Vs/U={vs:.3g} "
          f"settling over L ~ {vs * L * 50:.1f} um, cells={msh.topology.index_map(2).size_local}", flush=True)
    for name, fn in (("MIXTURE", lambda: mixture(msh, tags, re, buoy, vs)), ("EULER-EULER V3", lambda: euler_euler(msh, tags, re, buoy))):
        t0 = time.time()
        phi, vel, n_it = fn()
        for xf in (0.1, 0.5, 0.95):
            ys, pv = profile(msh, phi, xf * L)
            print(f"{name:15s} newton={n_it} {time.time() - t0:5.0f}s x/L={xf:.2f}: phi bottom/centre/top = "
                  f"{pv[2]:.4f}/{pv[200]:.4f}/{pv[-3]:.4f}  min/max {pv.min():.4f}/{pv.max():.4f}", flush=True)
        xs, yg, phig = field_grid(msh, phi)
        _, _, velg = field_grid(msh, vel, comp="mag")
        np.savez(f"/work/data/validation/piergiovanni_{name.split()[0].lower()}_q{q:g}.npz", y=ys, phi_outlet=pv,
                 xg=xs, yg=yg, phi_grid=phig, vel_grid=velg)
