"""Steady 2D two-fluid (Euler–Euler) blood solver in FEniCSx — formulation: knowledge/17-fenicsx-two-fluid.md.

Star units: h1 = 1, U = mixture mean velocity = 1, mu_ref = plasma viscosity, rho_ref = plasma density.
Unknowns: u (plasma velocity), w (RBC velocity), p (shared pressure), phi (RBC volume fraction).
Run inside the dolfinx/dolfinx:v0.9.0 container (see run_docker.sh).
"""
from __future__ import annotations

from dataclasses import dataclass

import basix.ufl
import dolfinx
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py import PETSc

INLET, OUTLET, WALL = 1, 2, 3

# Wu et al. 2015 (knowledge/14): plasma 0.96 cP, rho 1027 / 1093 kg/m3, d = 8 um, k = 11 s.
MU_PLASMA_CP = 0.96
RHO_RATIO = 1093.0 / 1027.0
SAFFMAN_C = 3.0 * 6.46 / (4.0 * np.pi)
PHI_RES = 1.0e-4
K_C, K_MU = 0.41, 0.62  # Phillips 1992 (knowledge/11 §5.2, secondary)
PHI_MAX = 0.95  # cap for the Phillips core (TODO: source RBC packing limit)
GDOT_EPS = 1.0e-3
CROSSWIND_C = 0.7  # shock-capturing constant (Codina-type; COMSOL exact constant not read)
SUPG_W_FLOOR = 0.1  # star units (mean velocity = 1)


@dataclass(frozen=True)
class TwoFluidParams:
    re_p: float            # rho_p U h1 / mu_p
    d_hat: float           # d_RBC / h1
    phi_in: float          # inlet haematocrit
    variant: str = "EE-DL"  # "EE-DL" (V3) or "EE-full" (V4, not implemented yet)
    lift: bool = True
    supg: bool = False  # SUPG off: tau=h/2|w| corrupts phi at walls; Galerkin + phi_diff is exact in the channel gate
    phi_diff: float = 1.0e-4  # numerical phi diffusion (star units, Pe ~ 1e4); fixes phi at no-slip wall nodes
    axisym: bool = False      # True: (x, r) axisymmetric, axis at r = 0 (x[1])
    viscosity: str = "wu-high-shear"  # or "plasma" (mu_hat = 1, diagnostic)
    phillips: bool = False  # Phillips et al. 1992 shear-induced migration flux in the phi equation (knowledge/11 §5.2)
    closure: str = "wu"   # "wu": Stokes + Rusche-Issa hindrance, Wu viscosity; "comsol-solid": COMSOL 2023 setup =
    #                       solid particles, Schiller-Naumann drag (COMSOL 09.158 img 26/30/31), Krieger mixture
    #                       viscosity (img 18) for both phases (img 17)
    gravity: float = 0.0  # reduced gravity on the RBC phase, (rho_r/rho_p - 1) rho_p g h^2/(mu U), acting along -y
    phi_max: float = 2.0 / 2.07  # Krieger packing limit for RBCs = 2/k_inf (Quemada high shear, k_inf 2.07, Mimouni 2016
    #                              via knowledge/16) = 0.966; owner decision 2026-11-24. COMSOL solid default 0.62 = comparison only
    wall_lift: bool = False      # Abkarian & Viallat 2005 Eq. 4 deformability lift (knowledge/19)
    wall_lift_C: float = 3.0 * 2.0 / (4.0 * np.pi)  # 3 f(nu)/(4 pi), f = 2 for RBC nu = 0.7 (Abkarian 2005 Discussion)
    wall_lift_farfield: bool = False  # extra R/y decay for y_gap > R (Olla-type, recalled)
    form: str = "comsol"  # "comsol": divided by phi, shared mixture viscosity (COMSOL default);
    #                       "wu-phase": Wu 2015 Eqs. 12-13, per-phase stresses 2 mu_p (1-phi) D(u) and mu_inf(phi) D(w)


def mu_inf_cp(phi):
    """Wu 2015 Eq. 9b, high-shear RBC contribution (cP)."""
    return 27.873 * phi**3 - 21.218 * phi**2 + 14.439 * phi


def mixture_viscosity_hat(phi):
    """Newtonian high-shear mixture viscosity / mu_p (Wu Eq. 8 with Eq. 9b; printed stress factor 1/2, knowledge/14)."""
    return (1.0 - phi) + 0.5 * mu_inf_cp(phi) / MU_PLASMA_CP


def hindrance(phi):
    """Rusche & Issa hindrance used by Wu 2015: f = exp(2.68 phi) + phi^0.43."""
    return ufl.exp(2.68 * phi) + phi**0.43


def grad3(v, axisym):
    """Velocity gradient; axisymmetric (x, r) adds the hoop component v_r / r."""
    if not axisym:
        return ufl.grad(v)
    r = ufl.SpatialCoordinate(ufl.domain.extract_unique_domain(v))[1]
    return ufl.as_tensor([[v[0].dx(0), v[0].dx(1), 0.0], [v[1].dx(0), v[1].dx(1), 0.0], [0.0, 0.0, v[1] / r]])


def div_(v, axisym):
    if not axisym:
        return ufl.div(v)
    r = ufl.SpatialCoordinate(ufl.domain.extract_unique_domain(v))[1]
    return v[0].dx(0) + v[1].dx(1) + v[1] / r


def sym_grad(v, axisym=False):
    return ufl.sym(grad3(v, axisym))


def build_space(msh):
    cell = msh.basix_cell()
    vec2 = basix.ufl.element("Lagrange", cell, 2, shape=(2,))
    sca1 = basix.ufl.element("Lagrange", cell, 1)
    return fem.functionspace(msh, basix.ufl.mixed_element([vec2, vec2, sca1, sca1]))


def phillips_flux(u, w, phi, prm):
    """N = -a^2 [K_c phi grad(gdot phi) + K_mu phi^2 gdot dln(mu)/dphi grad(phi)], star units, a = d_hat/2."""
    ax = prm.axisym
    a = 0.5 * prm.d_hat
    phi_c = ufl.min_value(ufl.max_value(phi, 0.0), PHI_MAX)
    j = phi_c * w + (1.0 - phi_c) * u
    dj = sym_grad(j, ax)
    gdot = ufl.sqrt(2.0 * ufl.inner(dj, dj) + GDOT_EPS**2)
    pv = ufl.variable(phi_c)
    dlnmu = ufl.diff(ufl.ln(mixture_viscosity_hat(pv)), pv)
    return -a**2 * (K_C * phi_c * ufl.grad(gdot * phi_c) + K_MU * phi_c**2 * gdot * dlnmu * ufl.grad(phi))


def residual(W, U, prm: TwoFluidParams, ds=None, U_old=None, dt=None, pseudo_cfl=None, psi_wall=None):
    u, w, p, phi = ufl.split(U)
    v, z, q, s = ufl.TestFunctions(W)
    ax = prm.axisym
    jac = ufl.SpatialCoordinate(W.mesh)[1] if ax else 1.0  # r dr dx weight
    phi_reg = ufl.min_value(ufl.max_value(phi, 0.0), 1.0)
    phi_pos = ufl.max_value(phi, PHI_RES)
    phic_pos = ufl.max_value(1.0 - phi, PHI_RES)

    mu = mixture_viscosity_hat(phi_reg) if prm.viscosity == "wu-high-shear" else 1.0
    slip = w - u
    if prm.closure == "comsol-solid":
        # Krieger (solids): mu_mix = mu_c (1 - phi/phi_max)^(-2.5 phi_max); capped just below phi_max
        phi_k = ufl.min_value(phi_reg, 0.999 * prm.phi_max)
        mu = (1.0 - phi_k / prm.phi_max) ** (-2.5 * prm.phi_max)
        # Schiller-Naumann (solids): beta = 3/(4d) phi_d C_D rho_c |slip|, C_D = 24/Re_p (1 + 0.15 Re_p^0.687),
        # Re_p = rho_c d |slip| / mu_c  ->  star: beta* = 18 phi (1 + 0.15 Re_d^0.687) / d_hat^2, Re_d = Re_p* d_hat |slip*|
        re_d = prm.re_p * prm.d_hat * ufl.sqrt(ufl.dot(slip, slip) + 1.0e-20)
        k_drag = 18.0 * phi_reg * (1.0 + 0.15 * re_d**0.687) / prm.d_hat**2
    else:
        k_drag = 18.0 * phi_reg * hindrance(phi_pos) / prm.d_hat**2  # phi_pos: d(phi^0.43)/dphi = inf at 0 -> NaN
    force = k_drag * slip
    if prm.lift:
        d_u = sym_grad(u, ax)
        gdot = ufl.sqrt(2.0 * ufl.inner(d_u, d_u) + GDOT_EPS**2)
        d_u2 = ufl.as_tensor([[d_u[i, j] for j in range(2)] for i in range(2)])  # in-plane block
        force = force + SAFFMAN_C * phi_reg * np.sqrt(prm.re_p) / prm.d_hat * gdot**-0.5 * ufl.dot(d_u2, slip)

    if prm.wall_lift:
        # f_wl = C_A phi mu_c gdot / y_gap  along n_w (away from the nearest wall); interphase: the reaction acts on
        # the plasma automatically through `force` (RBC eq. has +force, plasma -force).
        gpsi = ufl.grad(psi_wall)
        gnorm = ufl.sqrt(ufl.dot(gpsi, gpsi) + 1.0e-12)
        dist = ufl.sqrt(ufl.dot(gpsi, gpsi) + 2.0 * ufl.max_value(psi_wall, 0.0)) - gnorm  # Spalding wall distance
        r_hat = 0.5 * prm.d_hat
        y_gap = ufl.max_value(dist - r_hat, 0.1 * r_hat)
        du_l = sym_grad(u, prm.axisym)
        gdot_l = ufl.sqrt(2.0 * ufl.inner(du_l, du_l) + GDOT_EPS**2)
        mag = prm.wall_lift_C * phi_reg * gdot_l / y_gap
        if prm.wall_lift_farfield:
            mag = mag * ufl.min_value(1.0, r_hat / y_gap)
        force = force - mag * gpsi / gnorm
    re = prm.re_p
    if prm.form == "wu-phase":
        # Wu 2015 Eqs. 12-13 with the standard shared-pressure terms -(1-phi) grad p, -phi grad p (knowledge/14 note);
        # RBC stress as printed (no factor 2), high-shear mu_inf(phi) (Eq. 9b) in units of mu_p.
        mu_r = mu_inf_cp(phi_reg) / MU_PLASMA_CP
        phic = 1.0 - phi_reg
        f_plasma = (re * phic * ufl.inner(ufl.grad(u) * u, v)
                    + 2.0 * phic * ufl.inner(sym_grad(u, ax), sym_grad(v, ax))
                    + phic * ufl.inner(ufl.grad(p), v) - ufl.inner(force, v)) * jac * ufl.dx
        f_rbc = (re * RHO_RATIO * phi_pos * ufl.inner(ufl.grad(w) * w, z)
                 + (mu_r + PHI_RES) * ufl.inner(sym_grad(w, ax), sym_grad(z, ax))
                 + phi_pos * ufl.inner(ufl.grad(p), z) + ufl.inner(force, z)) * jac * ufl.dx
        f_mix = q * div_(phi * w + (1.0 - phi) * u, ax) * jac * ufl.dx
        r_phi = div_(phi * w, ax)
        if ds is not None:
            # conservative weak form: -phi w.grad(s) + outflow flux phi (w.n) s on OUTLET (inlet phi is Dirichlet,
            # walls have w = 0); fixes undetermined phi at outlet-wall corners (2026-11-24)
            n_out = ufl.FacetNormal(W.mesh)
            f_phi = (-phi * ufl.dot(w, ufl.grad(s)) + prm.phi_diff * ufl.dot(ufl.grad(phi), ufl.grad(s))) * jac * ufl.dx
            f_phi += phi * ufl.max_value(ufl.dot(w, n_out), 0.0) * s * jac * ds(OUTLET)
        else:
            f_phi = (s * r_phi + prm.phi_diff * ufl.dot(ufl.grad(phi), ufl.grad(s))) * jac * ufl.dx
        if prm.phillips:
            f_phi += -ufl.dot(phillips_flux(u, w, phi, prm), ufl.grad(s)) * jac * ufl.dx
        if U_old is not None:
            # (a) backward Euler in star time (dt given), or
            # (b) COMSOL pseudo time stepping (pseudo_cfl given; CFD Module Eq. 3-60): U_old = previous nonlinear
            #     iterate ("nojac"), element-local step dt~ = CFL_loc h / |u| per phase.
            u0, w0, _, phi0 = ufl.split(U_old)
            h = ufl.CellDiameter(W.mesh)
            if pseudo_cfl is not None:
                dt_u = pseudo_cfl * h / (ufl.sqrt(ufl.dot(u0, u0)) + 1.0e-3)
                dt_w = pseudo_cfl * h / (ufl.sqrt(ufl.dot(w0, w0)) + 1.0e-3)
            else:
                dt_u = dt_w = dt
            f_plasma += re * phic * ufl.inner((u - u0) / dt_u, v) * jac * ufl.dx
            f_rbc += re * RHO_RATIO * phi_pos * ufl.inner((w - w0) / dt_w, z) * jac * ufl.dx
            f_phi += s * (phi - phi0) / dt_w * jac * ufl.dx
            if prm.supg:  # SUPG with time-scale term: 2/dt bounds tau where |w| -> 0 at walls
                wmag2 = ufl.dot(w, w) + 1.0e-12
                tau = ((2.0 / dt_w) ** 2 + 4.0 * wmag2 / h**2 + (4.0 * prm.phi_diff / h**2) ** 2) ** -0.5
                res = (phi - phi0) / dt_w + r_phi
                f_phi += tau * ufl.dot(w, ufl.grad(s)) * res * jac * ufl.dx
                # crosswind / shock-capturing diffusion (COMSOL-equivalent, Codina-type), lagged on U_old:
                # nu_c = 0.5 C h |R(phi_old)| / |grad phi_old|, acting across the streamlines only
                w0n = ufl.sqrt(ufl.dot(w0, w0) + 1.0e-12)
                r_old = div_(phi0 * w0, ax)
                gphi0 = ufl.sqrt(ufl.dot(ufl.grad(phi0), ufl.grad(phi0)) + 1.0e-8)
                nu_c = 0.5 * CROSSWIND_C * h * abs(r_old) / gphi0
                dim = 2
                pcw = ufl.Identity(dim) - ufl.outer(w0, w0) / w0n**2
                f_phi += nu_c * ufl.dot(pcw * ufl.grad(phi), ufl.grad(s)) * jac * ufl.dx
        return f_plasma + f_rbc + f_mix + f_phi
    f_plasma = (re * ufl.inner(ufl.grad(u) * u, v) + 2.0 * mu * ufl.inner(sym_grad(u, ax), sym_grad(v, ax))
                - p * div_(v, ax) - ufl.inner(force / phic_pos, v)) * jac * ufl.dx
    f_rbc = (re * RHO_RATIO * ufl.inner(ufl.grad(w) * w, z) + 2.0 * mu * ufl.inner(sym_grad(w, ax), sym_grad(z, ax))
             - p * div_(z, ax) + ufl.inner(force / phi_pos, z)) * jac * ufl.dx
    if prm.gravity:  # buoyancy-reduced gravity on the dispersed phase (plasma hydrostatics absorbed in p)
        f_rbc += prm.gravity * z[1] * jac * ufl.dx
    f_mix = q * div_(phi * w + (1.0 - phi) * u, ax) * jac * ufl.dx
    r_phi = div_(phi * w, ax)
    f_phi = (s * r_phi + prm.phi_diff * ufl.dot(ufl.grad(phi), ufl.grad(s))) * jac * ufl.dx
    if prm.phillips:
        f_phi += -ufl.dot(phillips_flux(u, w, phi, prm), ufl.grad(s)) * jac * ufl.dx
    if prm.supg:
        h = ufl.CellDiameter(W.mesh)
        # velocity floor: tau = h/(2|w|) diverges at no-slip walls (found 2026-11-24, channel gate)
        wmag = ufl.sqrt(ufl.dot(w, w) + SUPG_W_FLOOR**2)
        tau = h / (2.0 * wmag)
        f_phi += tau * ufl.dot(w, ufl.grad(s)) * r_phi * jac * ufl.dx
    return f_plasma + f_rbc + f_mix + f_phi


def poiseuille(y):
    """Developed plane-Poiseuille profile, channel y in [0, 1], mean 1."""
    return 6.0 * y * (1.0 - y)


def channel_mesh(length: float, nx: int, ny: int):
    msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([length, 1.0])], [nx, ny],
                                mesh.CellType.quadrilateral)
    fdim = msh.topology.dim - 1
    locators = {INLET: lambda x: np.isclose(x[0], 0.0), OUTLET: lambda x: np.isclose(x[0], length),
                WALL: lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)}
    idx, val = [], []
    for tag, loc in locators.items():
        f = mesh.locate_entities_boundary(msh, fdim, loc)
        idx.append(f)
        val.append(np.full_like(f, tag))
    idx = np.concatenate(idx)
    order = np.argsort(idx)
    tags = mesh.meshtags(msh, fdim, idx[order], np.concatenate(val)[order])
    return msh, tags


def boundary_conditions(W, tags, phi_in: float, inlet_profile=poiseuille, pin_outlet_p: bool = False):
    msh = W.mesh
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    bcs = []
    for sub in (0, 1):
        V, _ = W.sub(sub).collapse()
        inflow = fem.Function(V)
        inflow.interpolate(lambda x: np.vstack([inlet_profile(x[1]), np.zeros_like(x[0])]))
        zero = fem.Function(V)
        for func, tag in ((inflow, INLET), (zero, WALL)):
            dofs = fem.locate_dofs_topological((W.sub(sub), V), fdim, tags.find(tag))
            bcs.append(fem.dirichletbc(func, dofs, W.sub(sub)))
    Q, _ = W.sub(3).collapse()
    phi_bc = fem.Function(Q)
    phi_bc.x.array[:] = phi_in
    dofs = fem.locate_dofs_topological((W.sub(3), Q), fdim, tags.find(INLET))
    bcs.append(fem.dirichletbc(phi_bc, dofs, W.sub(3)))
    if pin_outlet_p:  # needed for form="wu-phase" (grad p kept in strong form -> no natural outlet pressure)
        P, _ = W.sub(2).collapse()
        p0 = fem.Function(P)
        dofs = fem.locate_dofs_topological((W.sub(2), P), fdim, tags.find(OUTLET))
        bcs.append(fem.dirichletbc(p0, dofs, W.sub(2)))
    return bcs


def initial_guess(W, U, prm: TwoFluidParams, length: float):
    mu = float(mixture_viscosity_hat(prm.phi_in))
    for sub in (0, 1):
        U.sub(sub).interpolate(lambda x: np.vstack([poiseuille(x[1]), np.zeros_like(x[0])]))
    U.sub(2).interpolate(lambda x: 12.0 * mu * (length - x[0]))
    U.sub(3).interpolate(lambda x: np.full_like(x[0], prm.phi_in))
    U.x.scatter_forward()


def solve(W, U, F, bcs, rtol=1e-9, max_it=30, linesearch="none"):
    if int(dolfinx.__version__.split(".")[1]) >= 10:  # dolfinx >= 0.10: SNES-based NonlinearProblem
        opts = {"snes_type": "newtonls", "snes_linesearch_type": linesearch, "snes_rtol": rtol, "snes_atol": 1e-11,
                "snes_stol": 0.0, "snes_max_it": max_it, "ksp_type": "preonly", "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"}
        problem = NonlinearProblem(F, U, bcs=bcs, petsc_options_prefix="tf_", petsc_options=opts)
        problem.solve()
        snes = problem.solver
        if snes.getConvergedReason() <= 0:
            raise RuntimeError(f"Newton (SNES) did not converge, reason {snes.getConvergedReason()}")
        U.x.scatter_forward()
        return snes.getIterationNumber(), True
    problem = NonlinearProblem(F, U, bcs=bcs)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.rtol, solver.atol, solver.max_it = rtol, 1e-11, max_it
    solver.convergence_criterion = "residual"
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    prefix = ksp.getOptionsPrefix()
    opts[f"{prefix}ksp_type"] = "preonly"
    opts[f"{prefix}pc_type"] = "lu"
    opts[f"{prefix}pc_factor_mat_solver_type"] = "mumps"
    ksp.setFromOptions()
    n_it, converged = solver.solve(U)
    U.x.scatter_forward()
    return n_it, converged


def developed_phillips_profile(phi_bulk: float, n: int = 401):
    """Fully developed planar channel y in [0,1] with Phillips migration (comsol form, u = w).

    N_y = 0 and the linear shear stress tau ~ |y - 1/2| give phi * tau * mu(phi)^(K_mu/K_c - 1) = C (algebraic),
    capped at PHI_MAX; C is set so that the flux-weighted mean phi equals phi_bulk. Returns (y, u, phi) with mean u = 1.
    """
    def brentq(g, lo, hi, tol=1e-12):  # plain bisection (no scipy in the dolfinx image)
        glo = g(lo)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            gm = g(mid)
            if (gm > 0) == (glo > 0):
                lo, glo = mid, gm
            else:
                hi = mid
            if hi - lo < tol:
                break
        return 0.5 * (lo + hi)

    y = np.linspace(0.0, 1.0, n)
    tau = np.abs(y - 0.5) + 1e-9
    expo = K_MU / K_C - 1.0

    def phi_of(c):
        out = np.empty_like(y)
        for i, t in enumerate(tau):
            g = lambda f: f * t * mixture_viscosity_hat(f) ** expo - c  # noqa: E731
            out[i] = PHI_MAX if g(PHI_MAX) <= 0 else brentq(g, 0.0, PHI_MAX)
        return out

    def profile(c):
        phi = phi_of(c)
        mu = mixture_viscosity_hat(phi)
        dudy = -(y - 0.5) / mu  # tau = -G (y - 1/2), G scaled later
        u = np.concatenate([[0.0], np.cumsum(0.5 * (dudy[1:] + dudy[:-1]) * np.diff(y))])
        u /= np.trapezoid(u, y)
        return phi, u

    def mismatch(c):
        phi, u = profile(c)
        return np.trapezoid(phi * u, y) - phi_bulk

    c = brentq(mismatch, 1e-8, 10.0)
    phi, u = profile(c)
    return y, u, phi


def wall_psi(msh, tags):
    """Poisson 'wall potential' for the Spalding wall distance: -lap(psi) = 1, psi = 0 on WALL (natural elsewhere)."""
    import ufl as _u
    from dolfinx.fem.petsc import LinearProblem
    V = fem.functionspace(msh, ("Lagrange", 2))
    psi, q = _u.TrialFunction(V), _u.TestFunction(V)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    bc = fem.dirichletbc(0.0, fem.locate_dofs_topological(V, fdim, tags.find(WALL)), V)
    prob = LinearProblem(_u.inner(_u.grad(psi), _u.grad(q)) * _u.dx, fem.Constant(msh, 1.0) * q * _u.dx, bcs=[bc],
                         petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    return prob.solve()
