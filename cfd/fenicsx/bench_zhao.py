"""Benchmark F3: Zhao et al. 2008 / Sezer et al. 2026 planar sudden expansion, 2D (knowledge/15, knowledge/18).

h1 = 100 um -> 200 um (E = 2), L1 = 1000 um, L2 = 2000 um; d_RBC = 8 um; Q = 100 uL/min.
Sezer is a 3D quarter domain of a square 100x100 um inlet; this is planar 2D -> compare TRENDS (Delta P*(alpha),
minimum location, CFL), not absolute values. Delta P* = Delta P / (0.5 rho_p u_i^2) (Sezer Eq. 22).
Usage: python3 bench_zhao.py [cells_per_unit] [variant]
"""
import os
import sys
import time

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, geometry, mesh
from mpi4py import MPI

sys.path.insert(0, "/work/cfd/fenicsx")
import twofluid as tf  # noqa: E402

H1, H2, L1, L2 = 1.0, 2.0, 10.0, 20.0
# Sezer 3D: u_i = Q / (h1 W) with W = 100 um -> 0.1667 m/s; Re_p = rho_p u_i h1 / mu_p
U_I = 100e-9 / 60.0 / (100e-6 * 100e-6)
RE_P = 1027.0 * U_I * 100e-6 / 0.96e-3
D_HAT = 8.0 / 100.0
FORM = os.environ.get("FORM", "comsol")
PHILLIPS = os.environ.get("PHILLIPS", "0") == "1"
ALPHAS = tuple(float(a) for a in os.environ.get("ALPHAS", "0.01,0.1,0.2,0.3,0.4,0.5").split(","))


def sezer_dpstar(alpha):
    return 226.72 * alpha**2 - 148.57 * alpha + 117.37


def zhao_mesh(n: int):
    xs = np.unique(np.concatenate([np.linspace(-L1, 0.0, int(L1 * n) + 1), np.linspace(0.0, L2, int(L2 * n) + 1)]))
    ys = np.unique(np.concatenate([np.linspace(-H2 / 2, -H1 / 2, int((H2 - H1) / 2 * n) + 1),
                                   np.linspace(-H1 / 2, H1 / 2, int(H1 * n) + 1),
                                   np.linspace(H1 / 2, H2 / 2, int((H2 - H1) / 2 * n) + 1)]))
    nx, ny = len(xs), len(ys)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    nid = np.arange(nx * ny).reshape(nx, ny)
    cells = [[nid[i, j], nid[i + 1, j], nid[i, j + 1], nid[i + 1, j + 1]]
             for i in range(nx - 1) for j in range(ny - 1)
             if not (0.5 * (xs[i] + xs[i + 1]) < 0.0 and abs(0.5 * (ys[j] + ys[j + 1])) > H1 / 2)]
    cells = np.array(cells, dtype=np.int64)
    used = np.unique(cells)
    remap = -np.ones(len(nodes), dtype=np.int64)
    remap[used] = np.arange(len(used))
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,)))
    msh = mesh.create_mesh(MPI.COMM_WORLD, remap[cells], nodes[used], domain)
    msh.topology.create_connectivity(1, 2)
    bf = mesh.exterior_facet_indices(msh.topology)
    mid = mesh.compute_midpoints(msh, 1, bf)
    tag = np.full(len(bf), tf.WALL, dtype=np.int32)
    tag[np.isclose(mid[:, 0], -L1)] = tf.INLET
    tag[np.isclose(mid[:, 0], L2)] = tf.OUTLET
    order = np.argsort(bf)
    return msh, mesh.meshtags(msh, 1, bf[order], tag[order])


def channel_profile(y):
    return tf.poiseuille(y + 0.5)  # inlet y in [-0.5, 0.5]


def probe(msh, f, pts):
    tree = geometry.bb_tree(msh, 2)
    cells = geometry.compute_colliding_cells(msh, geometry.compute_collisions_points(tree, pts), pts)
    return np.array([f.eval(pts[i], cells.links(i)[0]) for i in range(len(pts))])


def cfl_thickness(msh, phi, x, h_local):
    """Sezer Eq. 21: first inflection of phi(y) moving away from the upper wall, excluding |y/h| > 0.48."""
    ys = np.linspace(0.0, 0.48 * h_local, 400)
    vals = probe(msh, phi, np.column_stack([np.full_like(ys, x), ys, np.zeros_like(ys)]))[:, 0]
    d2 = np.gradient(np.gradient(vals, ys), ys)
    sign = np.sign(d2)
    for k in range(len(ys) - 2, 0, -1):  # from the wall inward
        if sign[k] != sign[k - 1] and sign[k] != 0:
            return (0.5 * h_local - ys[k]) / h_local, vals
    return float("nan"), vals


def main(n: int, variant: str):
    msh, tags = zhao_mesh(n)
    W = tf.build_space(msh)
    U = fem.Function(W)
    for sub in (0, 1):
        U.sub(sub).interpolate(lambda x: np.vstack([np.where(np.abs(x[1]) < 0.5, channel_profile(np.clip(x[1], -0.5, 0.5)), 0.0),
                                                    np.zeros_like(x[0])]))
    print(f"Re_p={RE_P:.3f} d_hat={D_HAT} cells={msh.topology.index_map(2).size_local} "
          f"dofs={W.dofmap.index_map.size_global * W.dofmap.index_map_bs}", flush=True)
    rows = []
    phi_prev = 0.01
    for alpha in ALPHAS:
        U.sub(3).interpolate(lambda x, a=phi_prev: np.full_like(x[0], a))
        U.x.scatter_forward()
        bcs = tf.boundary_conditions(W, tags, alpha, inlet_profile=channel_profile, pin_outlet_p=(FORM == "wu-phase"))
        t0 = time.time()
        for re in ((1.0, RE_P) if alpha == ALPHAS[0] else (RE_P,)):
            prm = tf.TwoFluidParams(re_p=re, d_hat=D_HAT, phi_in=alpha, lift=(variant != "drag-only"), form=FORM, phillips=PHILLIPS)
            n_it, ok = tf.solve(W, U, tf.residual(W, U, prm), bcs)
        p, phi = U.sub(2).collapse(), U.sub(3).collapse()
        pin, pout = probe(msh, p, np.array([[-L1, 0.0, 0.0], [L2, 0.0, 0.0]]))[:, 0]
        dp_star = (pin - pout) * 2.0 / RE_P
        cfl_up, _ = cfl_thickness(msh, phi, -1.0, H1)
        cfl_dn, _ = cfl_thickness(msh, phi, 0.66 * (L1 + L2) - L1, H2)
        ref = sezer_dpstar(alpha) if alpha >= 0.2 else float("nan")
        rows.append((alpha, dp_star, ref))
        print(f"alpha={alpha:.2f} newton={n_it} ok={ok} {time.time() - t0:5.1f}s  dP*={dp_star:8.3f}  Sezer3D={ref:8.3f}  "
              f"CFL*(x=-1)={cfl_up:.4f} CFL*(z/L=.66)={cfl_dn:.4f}  phi=[{phi.x.array.min():.4f},{phi.x.array.max():.4f}]",
              flush=True)
        phi_prev = alpha
    a = np.array([r[0] for r in rows if r[0] >= 0.2])
    if len(a) < 3:
        return
    d = np.array([r[1] for r in rows if r[0] >= 0.2])
    c = np.polyfit(a, d, 2)
    print(f"RESULT quadratic fit ours: {c[0]:.2f} a^2 + {c[1]:.2f} a + {c[2]:.2f}; argmin={-c[1] / (2 * c[0]):.3f} "
          f"| Sezer: 226.72 a^2 - 148.57 a + 117.37, argmin=0.328")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20, sys.argv[2] if len(sys.argv) > 2 else "EE-DL")
