"""Benchmark F2: Karino & Goldsmith (1977) tube expansion as modelled by Wu et al. 2015 §3.1 (knowledge/14 §4).

Axisymmetric (x, r), star units h1 = d1 = 151 um. R1 = 0.5, R2 = 0.5 * 3.3377; L_in = 1 mm, L_out = 2 mm.
phi_in = 0.01, d_RBC = 8 um. Targets x_r / d1 = 1.523 (Re 12.2) and 4.735 (Re 37.8).
Usage (container): python3 bench_kg.py [cells_per_unit]
"""
import sys
import time

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, geometry, mesh
from mpi4py import MPI

sys.path.insert(0, "/work/cfd/fenicsx")
import twofluid as tf  # noqa: E402

AXIS = 4
import os
SUPG = os.environ.get("SUPG", "1") == "1"
D1_UM = 151.0
R1, R2 = 0.5, 0.5 * 3.3377
L_IN, L_OUT = 1000.0 / D1_UM, 2000.0 / D1_UM
TARGET = {12.2: 1.523, 37.8: 4.735}
# Re_p from Wu's inlet velocities with plasma properties: rho 1027, mu 0.96 cP, d1 151 um
RE_P = {12.2: 1027 * 0.0757 * 151e-6 / 0.96e-3, 37.8: 1027 * 0.233 * 151e-6 / 0.96e-3}


def kg_mesh(n_per_unit: int):
    xs = np.unique(np.concatenate([np.linspace(-L_IN, 0.0, int(L_IN * n_per_unit) + 1),
                                   np.linspace(0.0, L_OUT, int(L_OUT * n_per_unit) + 1)]))
    rs = np.unique(np.concatenate([np.linspace(0.0, R1, int(R1 * n_per_unit) + 1),
                                   np.linspace(R1, R2, int((R2 - R1) * n_per_unit) + 1)]))
    nx, nr = len(xs), len(rs)
    X, Rr = np.meshgrid(xs, rs, indexing="ij")
    nodes = np.column_stack([X.ravel(), Rr.ravel()])
    nid = np.arange(nx * nr).reshape(nx, nr)
    cells = []
    for i in range(nx - 1):
        for j in range(nr - 1):
            xc, rc = 0.5 * (xs[i] + xs[i + 1]), 0.5 * (rs[j] + rs[j + 1])
            if xc < 0.0 and rc > R1:
                continue
            cells.append([nid[i, j], nid[i + 1, j], nid[i, j + 1], nid[i + 1, j + 1]])
    cells = np.array(cells, dtype=np.int64)
    used = np.unique(cells)
    remap = -np.ones(len(nodes), dtype=np.int64)
    remap[used] = np.arange(len(used))
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,)))
    msh = mesh.create_mesh(MPI.COMM_WORLD, remap[cells], nodes[used], domain)
    fdim = 1
    msh.topology.create_connectivity(fdim, 2)
    bfacets = mesh.exterior_facet_indices(msh.topology)
    mid = mesh.compute_midpoints(msh, fdim, bfacets)
    tag = np.full(len(bfacets), tf.WALL, dtype=np.int32)
    tag[np.isclose(mid[:, 0], -L_IN)] = tf.INLET
    tag[np.isclose(mid[:, 0], L_OUT)] = tf.OUTLET
    tag[np.isclose(mid[:, 1], 0.0)] = AXIS
    order = np.argsort(bfacets)
    return msh, mesh.meshtags(msh, fdim, bfacets[order], tag[order])


def tube_profile(r):
    return 2.0 * (1.0 - (r / R1) ** 2)


def bcs_kg(W, tags, phi_in):
    bcs = tf.boundary_conditions(W, tags, phi_in, inlet_profile=tube_profile)
    for sub in (0, 1):
        Vr, _ = W.sub(sub).sub(1).collapse()
        zero = fem.Function(Vr)
        dofs = fem.locate_dofs_topological((W.sub(sub).sub(1), Vr), 1, tags.find(AXIS))
        bcs.append(fem.dirichletbc(zero, dofs, W.sub(sub).sub(1)))
    return bcs


def reattachment(msh, u, delta=0.02):
    xs = np.linspace(0.01, L_OUT - 0.5, 4000)
    pts = np.column_stack([xs, np.full_like(xs, R2 - delta), np.zeros_like(xs)])
    tree = geometry.bb_tree(msh, 2)
    cells = geometry.compute_colliding_cells(msh, geometry.compute_collisions_points(tree, pts), pts)
    ux = np.array([u.eval(pts[i], cells.links(i)[0])[0] for i in range(len(xs))])
    neg = np.where(ux < 0)[0]
    if len(neg) == 0:
        return 0.0
    k = neg[-1]  # last point of the corner vortex
    return float(xs[k] + (xs[k + 1] - xs[k]) * (-ux[k]) / (ux[k + 1] - ux[k]))


def main(n_per_unit: int, viscosity: str = "wu-high-shear"):
    msh, tags = kg_mesh(n_per_unit)
    phi_in = 0.01
    W = tf.build_space(msh)
    U = fem.Function(W)
    for sub in (0, 1):
        U.sub(sub).interpolate(lambda x: np.vstack([np.where(x[1] < R1, tube_profile(np.minimum(x[1], R1)), 0.0),
                                                    np.zeros_like(x[0])]))
    U.sub(3).interpolate(lambda x: np.full_like(x[0], phi_in))
    U.x.scatter_forward()
    bcs = bcs_kg(W, tags, phi_in)
    print(f"viscosity={viscosity} mesh n/unit={n_per_unit} cells={msh.topology.index_map(2).size_local} dofs={W.dofmap.index_map.size_global * W.dofmap.index_map_bs}", flush=True)
    results = {}
    for re in (1.0, 6.0, RE_P[12.2], 25.0, RE_P[37.8]):
        prm = tf.TwoFluidParams(re_p=re, d_hat=8.0 / D1_UM, phi_in=phi_in, axisym=True, viscosity=viscosity, supg=SUPG)
        t0 = time.time()
        n_it, ok = tf.solve(W, U, tf.residual(W, U, prm), bcs)
        u, w, phi = U.sub(0).collapse(), U.sub(1).collapse(), U.sub(3).collapse()
        xr = reattachment(msh, u)
        slip = np.max(np.abs(w.x.array - u.x.array))
        print(f"Re_p={re:7.3f} newton={n_it} ok={ok} {time.time() - t0:6.1f}s  x_r/d1={xr:.4f}  max|w-u|={slip:.2e} "
              f"phi=[{phi.x.array.min():.5f},{phi.x.array.max():.5f}]", flush=True)
        for key, val in RE_P.items():
            if np.isclose(re, val):
                results[key] = xr
    for key, xr in results.items():
        print(f"RESULT Re {key}: x_r/d1 = {xr:.4f}  target {TARGET[key]}  error {100 * (xr / TARGET[key] - 1):+.2f} %")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20, sys.argv[2] if len(sys.argv) > 2 else "wu-high-shear")
