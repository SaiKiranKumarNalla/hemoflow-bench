"""Validation F4: Gracka et al. 2022 (CMPB 226:107117) hyperbolic-contraction chips, CFL 300 um downstream.

Planar 2D in the 400-um width plane (real chip depth 50 um: Hele-Shaw-type approximation, knowledge/19).
Star units: length 400 um, U = Q / (400 um x 50 um) at 10 uL/min, mu_ref = Dx40 5.2 mPa s, rho 1042 kg/m3.
Chip A: throat 54 um, contraction length 128 um; chip B: throat 20 um, contraction length 382 um.
Wall shape: 1/w linear in x (constant extension rate). Experiment: CFL = 5 um (A), 35 um (B) (phi > 2 % edge).
Usage: python3 bench_gracka.py A|B [variant: v4wl|v4|v4wl_ff] [max_iter] [refine factor, default 1]
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
from comsol_solver import ComsolStationary, cfl_comsol  # noqa: E402

L_REF = 400e-6
CHIPS = {"A": (54.0, 128.0, 5.0), "B": (20.0, 382.0, 35.0)}  # throat um, contraction length um, measured CFL um
MU, RHO = 5.2e-3, 1042.0
U_REF = 10e-9 / 60.0 / (400e-6 * 50e-6)
RE = RHO * U_REF * L_REF / MU
D_HAT = 8e-6 / L_REF
PHI_IN = 0.05


def grade(n, beta):
    s = np.linspace(-1.0, 1.0, n + 1)
    return 0.5 * (1.0 + np.tanh(beta * s) / np.tanh(beta))


def gracka_mesh(chip, nt=28, no=36, nup=40, nc=50, nd=110, bt=2.2, bo=2.6):
    w_um, lc_um, _ = CHIPS[chip]
    wt, lc, lup, ld = w_um / 400.0, lc_um / 400.0, 1.0, 4.0
    y_t = -wt / 2 + wt * grade(nt, bt)                 # throat nodes, clustered at both throat walls
    y_o = wt / 2 + (0.5 - wt / 2) * grade(no, bo)      # outer block, clustered at both ends
    y_d = np.unique(np.round(np.concatenate([-y_o[::-1], y_t, y_o]), 12))
    s = y_t / (wt / 2)                                  # normalised positions for the converging part
    x_up = np.linspace(-lup, 0.0, nup + 1)
    x_c = lc * (1.0 - (1.0 - np.linspace(0, 1, nc + 1)) ** 1.5)       # cluster toward the throat end
    x_d = lc + ld * np.linspace(0, 1, nd + 1) ** 1.6                   # cluster just behind the expansion

    def half_w(x):
        if x <= 0.0:
            return 0.5
        return 0.5 / (1.0 + (1.0 / wt - 1.0) * x / lc)

    nodes, key, cells = [], {}, []

    def nid(x, y):
        k = (round(x, 10), round(y, 10))
        if k not in key:
            key[k] = len(nodes)
            nodes.append((x, y))
        return key[k]

    xs1 = np.unique(np.concatenate([x_up, x_c]))
    for i in range(len(xs1) - 1):
        a, b = xs1[i], xs1[i + 1]
        for j in range(len(s) - 1):
            cells.append([nid(a, s[j] * half_w(a)), nid(b, s[j] * half_w(b)),
                          nid(a, s[j + 1] * half_w(a)), nid(b, s[j + 1] * half_w(b))])
    for i in range(len(x_d) - 1):
        a, b = x_d[i], x_d[i + 1]
        for j in range(len(y_d) - 1):
            cells.append([nid(a, y_d[j]), nid(b, y_d[j]), nid(a, y_d[j + 1]), nid(b, y_d[j + 1])])
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,)))
    msh = mesh.create_mesh(MPI.COMM_WORLD, np.array(cells, dtype=np.int64), np.array(nodes), domain)
    msh.topology.create_connectivity(1, 2)
    bf = mesh.exterior_facet_indices(msh.topology)
    mid = mesh.compute_midpoints(msh, 1, bf)
    tag = np.full(len(bf), tf.WALL, dtype=np.int32)
    tag[np.isclose(mid[:, 0], -lup)] = tf.INLET
    tag[np.isclose(mid[:, 0], lc + ld)] = tf.OUTLET
    order = np.argsort(bf)
    return msh, mesh.meshtags(msh, 1, bf[order], tag[order]), lc


def inlet_profile(y):
    return 6.0 * (y + 0.5) * (0.5 - y)


def cfl_edge(msh, phi, x, thr=0.02):
    ys = np.linspace(-0.4999, 0.4999, 4001)
    pts = np.column_stack([np.full_like(ys, x), ys, np.zeros_like(ys)])
    tree = geometry.bb_tree(msh, 2)
    cells = geometry.compute_colliding_cells(msh, geometry.compute_collisions_points(tree, pts), pts)
    ok = np.array([len(cells.links(i)) > 0 for i in range(len(ys))])
    vals = np.full(len(ys), np.nan)
    vals[ok] = [phi.eval(pts[i], cells.links(i)[0])[0] for i in np.where(ok)[0]]
    above = np.where(vals > thr)[0]
    if len(above) == 0:
        return float("nan"), float("nan"), float("nan"), vals
    lo = (ys[above[0]] + 0.5) * 400.0
    hi = (0.5 - ys[above[-1]]) * 400.0
    core = np.nanmean(vals[(np.abs(ys) < 0.3)])
    return lo, hi, core, vals


def main(chip, variant, max_iter, refine=1.0):
    r = refine
    msh, tags, lc = gracka_mesh(chip, nt=int(28 * r), no=int(36 * r), nup=int(40 * r), nc=int(50 * r), nd=int(110 * r))
    hmin = msh.h(2, np.arange(msh.topology.index_map(2).size_local)).min() * 400.0
    W = tf.build_space(msh)
    U, U_prev = fem.Function(W), fem.Function(W)
    U.sub(3).interpolate(lambda x: np.full_like(x[0], PHI_IN))
    U.x.scatter_forward()
    bcs0 = tf.boundary_conditions(W, tags, PHI_IN, inlet_profile=inlet_profile)
    tf.solve(W, U, tf.residual(W, U, tf.TwoFluidParams(re_p=RE, d_hat=D_HAT, phi_in=PHI_IN, closure="comsol-solid")),
             bcs0)
    wl = variant.startswith("v4wl")
    prm = tf.TwoFluidParams(re_p=RE, d_hat=D_HAT, phi_in=PHI_IN, form="wu-phase", lift=True, supg=True,
                            wall_lift=wl, phillips=wl, wall_lift_farfield=variant.endswith("ff"))
    psi_w = tf.wall_psi(msh, tags) if wl else None
    bcs = tf.boundary_conditions(W, tags, PHI_IN, inlet_profile=inlet_profile, pin_outlet_p=True)
    for bc in bcs:
        bc.set(U.x.array)
    cfl = fem.Constant(msh, 1.0)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    F = tf.residual(W, U, prm, ds=ds, U_old=U_prev, pseudo_cfl=cfl, psi_wall=psi_w)
    solver = ComsolStationary(U, U_prev, F, bcs, cfl)
    x_meas = lc + 300.0 / 400.0
    print(f"chip {chip} variant {variant} Re={RE:.3f} d_hat={D_HAT} cells={msh.topology.index_map(2).size_local} "
          f"h_min={hmin:.2f}um measured CFL={CHIPS[chip][2]}um", flush=True)
    t0 = time.time()
    for k in range(1, max_iter + 1):
        lam, ndx, nE = solver.step(k)
        done = k >= 50 and ndx < 1e-6
        if k % 10 == 0 or done or k == max_iter:
            phi = U.sub(3).collapse()
            lo, hi, core, _ = cfl_edge(msh, phi, x_meas)
            print(f"k={k:3d} CFL={cfl_comsol(k):7.1f} lam={lam:.2f} |dU|={ndx:.1e} wall={time.time() - t0:5.0f}s "
                  f"phi=[{phi.x.array.min():.4f},{phi.x.array.max():.4f}] CFL300um lower/upper={lo:.1f}/{hi:.1f}um "
                  f"core_phi={core:.4f}", flush=True)
        if done:
            print("CONVERGED", flush=True)
            break
    phi = U.sub(3).collapse()
    lo, hi, core, _ = cfl_edge(msh, phi, x_meas)
    ys = np.linspace(-0.4999, 0.4999, 4001)
    out = f"/work/data/validation/gracka_{chip}_{variant}_r{r:g}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    _, _, _, prof = cfl_edge(msh, phi, x_meas)
    np.savez(out + ".npz", y_um=(ys + 0.5) * 400.0, phi=prof, cfl_lower_um=lo, cfl_upper_um=hi, core_phi=core,
             measured_um=CHIPS[chip][2], iterations=k, converged=bool(done))
    print(f"RESULT chip {chip} {variant} r={r:g}: CFL = {0.5 * (lo + hi):.1f} um (lower {lo:.1f}, upper {hi:.1f}) vs experiment "
          f"{CHIPS[chip][2]} um; core phi {core:.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "v4wl", int(sys.argv[3]) if len(sys.argv) > 3 else 120,
         float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
