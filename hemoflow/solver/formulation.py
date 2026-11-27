"""Modular UFL residual builder for the 2D two-phase Euler-Euler model (lazy dolfinx/ufl).

Faithful to ``cfd/fenicsx/twofluid.py`` (``wu-phase`` and ``comsol`` forms, COMSOL
pseudo-time stepping, SUPG + Codina crosswind, conservative outlet flux, lagged
positivity/jamming stabilisation), but the viscosity / drag / lift / migration terms are
supplied by the registered plug-ins resolved from :class:`EulerEulerConfig`, so users can
plug in/out closures.

This module imports ``ufl``/``dolfinx``/``basix`` **lazily inside functions only**; importing
it (and ``hemoflow``) on a host without dolfinx is safe.
"""
from __future__ import annotations

from typing import Any

from hemoflow.solver.config import EulerEulerConfig
from hemoflow.closures.constants import (
    CROSSWIND_C, GDOT_EPS, K_C, K_MU, PHI_JAM, PHI_RES, RHO_RATIO, SUPG_W_FLOOR,
)

__all__ = ["build_space", "build_residual", "boundary_conditions", "initial_guess", "wall_psi", "closure_name"]


def _deps():
    import basix.ufl
    import ufl
    from dolfinx import fem
    return basix.ufl, ufl, fem


def closure_name(closure) -> str | None:
    """Best-effort registered name of a closure (str passthrough; callable -> registry lookup)."""
    from hemoflow.registry import _REGISTRY
    if isinstance(closure, str):
        return closure
    for kind in ("viscosity", "drag", "lift", "migration", "geometry", "bc"):
        for name, obj in _REGISTRY[kind].items():
            if obj is closure:
                return name
    return None


def build_space(msh):
    """Mixed (u, w, p, phi) function space: P2-P2-P1-P1 (verbatim twofluid.build_space)."""
    basix_ufl, ufl, fem = _deps()
    cell = msh.basix_cell()
    dim = msh.topology.dim
    vec2 = basix_ufl.element("Lagrange", cell, 2, shape=(dim,))
    sca1 = basix_ufl.element("Lagrange", cell, 1)
    return fem.functionspace(msh, basix_ufl.mixed_element([vec2, vec2, sca1, sca1]))


def _sym_grad(ufl, v, axisym=False):
    g = ufl.grad(v)
    if axisym:
        r = ufl.SpatialCoordinate(v.ufl_domain())[1]
        g = ufl.as_tensor([[g[0, 0], g[0, 1], 0.0], [g[1, 0], g[1, 1], 0.0], [0.0, 0.0, v[1] / r]])
    return ufl.sym(g)


def _div(ufl, v, axisym=False):
    if not axisym:
        return ufl.div(v)
    r = ufl.SpatialCoordinate(v.ufl_domain())[1]
    return v[0].dx(0) + v[1].dx(1) + v[1] / r


def _gdot_of(ufl, v, eps, axisym=False):
    D = _sym_grad(ufl, v, axisym=axisym)
    return ufl.sqrt(2.0 * ufl.inner(D, D) + eps**2)


def _eval_viscosity(visc, vname: str | None, phi_arg, gdot_u, gdot_w, cfg: EulerEulerConfig):
    """Invoke a built-in or custom viscosity plugin uniformly, returning (mu_mix, mu_rbc_hat)."""
    if vname == "wu_shear_thinning":
        mu_r = visc(phi_arg, gdot_w, cfg.k_star)
        return (1.0 - phi_arg) + 0.5 * mu_r, mu_r
    if vname == "quemada":
        mu = visc(phi_arg, gdot_u)
        return mu, mu
    if vname == "bird_carreau":
        mu = visc(gdot_u, mu_0=5.0, mu_inf=1.0)
        return mu, mu
    if vname == "krieger":
        mu = visc(phi_arg, phi_max=cfg.phi_max)
        return mu, mu
    mu = visc(phi_arg)
    return mu, mu


def build_residual(W, U, cfg: EulerEulerConfig, ds=None, U_old=None, dt=None,
                   pseudo_cfl=None, psi_wall=None):
    """Assemble the UFL residual ``F`` for the configured two-phase model.

    Returns a ufl form.  ``ds`` is the facet measure (for the conservative outlet flux);
    ``U_old``/``dt``/``pseudo_cfl`` enable COMSOL pseudo-time stepping; ``psi_wall`` is the
    Spalding wall-distance potential needed by the wall-lift plug-ins.
    """
    basix_ufl, ufl, fem = _deps()
    from hemoflow.geometries.mesher import INLET, OUTLET, WALL
    closures = cfg.resolve()
    visc, drag, lift, migr = (closures["viscosity"], closures["drag"],
                              closures["lift"], closures["migration"])
    vname = closure_name(cfg.viscosity)
    lname = closure_name(cfg.lift)
    mname = closure_name(cfg.migration)

    u, w, p, phi = ufl.split(U)
    v, z, q, s = ufl.TestFunctions(W)
    ax = cfg.axisym
    jac = ufl.SpatialCoordinate(W.mesh)[1] if ax else 1.0
    dx = ufl.dx(metadata={"quadrature_degree": int(cfg.qdeg)}) if cfg.qdeg else ufl.dx
    dim = W.mesh.topology.dim

    phi_reg = ufl.min_value(ufl.max_value(phi, 0.0), 1.0)
    phi_pos = ufl.max_value(phi, PHI_RES)
    phic_pos = ufl.max_value(1.0 - phi, PHI_RES)
    slip = w - u
    slip_mag = ufl.sqrt(ufl.dot(slip, slip) + 1.0e-20)

    # --- viscosity ---
    gdot_u = _gdot_of(ufl, u, cfg.gdot_eps, axisym=ax)
    gdot_w = _gdot_of(ufl, w, cfg.gdot_eps, axisym=ax)
    mu, mu_r = _eval_viscosity(visc, vname, phi_reg, gdot_u, gdot_w, cfg)

    # --- drag + lift -> interphase force ---
    if closure_name(cfg.drag) == "schiller_naumann":
        k_drag = drag(phi_reg, cfg.d_hat, cfg.re_p, slip_mag)
    else:
        k_drag = drag(phi, cfg.d_hat)
    force = k_drag * slip
    if lname != "none" and cfg.lift is not None and not (isinstance(cfg.lift, str) and cfg.lift == "none"):
        D_u = _sym_grad(ufl, u, ax)
        D_u2 = ufl.as_tensor([[D_u[i, j] for j in range(dim)] for i in range(dim)])
        if lname == "saffman_plus_twowall":
            force = force + lift(phi, cfg.re_p, cfg.d_hat, gdot_u, D_u2, slip,
                                 ufl.grad(psi_wall), psi_wall, cfg.wall_lift_C, cfg.d_hat,
                                 cfg.wall_lift_law, cfg.wall_lift_farfield)
        elif lname == "saffman":
            from hemoflow.closures.lift import saffman
            force = force + saffman(phi, cfg.re_p, cfg.d_hat, gdot_u, D_u2, slip)
        elif lname in ("abkarian_wall", "abkarian_twowall"):
            force = force + lift(phi, gdot_u, ufl.grad(psi_wall), psi_wall,
                                 cfg.wall_lift_C, cfg.d_hat, cfg.wall_lift_law, cfg.wall_lift_farfield)
        else:
            force = force + lift(phi, cfg.re_p, cfg.d_hat, gdot_u, D_u2, slip)

    # --- migration flux N ---
    a = 0.5 * cfg.d_hat
    if mname != "none" and not (isinstance(cfg.migration, str) and cfg.migration == "none"):
        j = phi_reg * w + (1.0 - phi_reg) * u
        dj = _sym_grad(ufl, j, ax)
        g2 = 2.0 * ufl.inner(dj, dj) + cfg.gdot_eps**2
        if cfg.depth_hat > 0.0:
            g2 = g2 + (12.0 / cfg.depth_hat**2) * ufl.dot(j, j)
        if mname == "phillips_nonlocal":
            g2 = g2 + (cfg.phillips_nl_c * a) ** 2 * ufl.dot(j, j)
        gdot_j = ufl.sqrt(g2)
        phi_c = ufl.min_value(ufl.max_value(phi, 0.0), cfg.phi_jam)
        pv = ufl.variable(phi_c)
        mu_pv, _ = _eval_viscosity(visc, vname, pv, gdot_u, gdot_w, cfg)
        dlnmu = ufl.diff(ufl.ln(mu_pv), pv)
        n_ph = migr(phi, gdot_j, ufl.grad(gdot_j), ufl.grad(phi), dlnmu, a)
    else:
        n_ph = ufl.as_vector([0.0] * dim)

    re = cfg.re_p
    d_hz = cfg.depth_hat

    # ============ wu-phase form (Wu 2015 Eqs. 12-13) ============
    if cfg.form == "wu-phase":
        if vname in ("wu_high_shear", None) or vname == "wu_shear_thinning":
            mu_r_eff = (mu_r + PHI_RES) if vname == "wu_shear_thinning" else (mu_inf_hat(ufl, phi_reg) + PHI_RES)
        else:
            mu_r_eff = mu_r + PHI_RES
        f_plasma = (re * phic_pos * ufl.inner(ufl.grad(u) * u, v)
                    + 2.0 * phic_pos * ufl.inner(_sym_grad(ufl, u, ax), _sym_grad(ufl, v, ax))
                    + phic_pos * ufl.inner(ufl.grad(p), v) - ufl.inner(force, v)) * jac * dx
        f_rbc = (re * RHO_RATIO * phi_pos * ufl.inner(ufl.grad(w) * w, z)
                 + mu_r_eff * ufl.inner(_sym_grad(ufl, w, ax), _sym_grad(ufl, z, ax))
                 + phi_pos * ufl.inner(ufl.grad(p), z) + ufl.inner(force, z)) * jac * dx
        if d_hz > 0.0:
            f_plasma += (12.0 * phic_pos / d_hz**2) * ufl.inner(u, v) * jac * dx
            f_rbc += (6.0 * mu_r_eff / d_hz**2) * ufl.inner(w, z) * jac * dx
        if ds is not None:
            n_out = ufl.FacetNormal(W.mesh)
            f_plasma += -phic_pos * ufl.inner(ufl.dot(ufl.grad(u).T, n_out), v) * jac * ds(OUTLET)
            f_rbc += -0.5 * mu_r_eff * ufl.inner(ufl.dot(ufl.grad(w).T, n_out), z) * jac * ds(OUTLET)
        f_mix = q * _div(ufl, phi_reg * w + (1.0 - phi_reg) * u + n_ph, ax) * jac * dx
        r_phi = _div(ufl, phi * w + n_ph, ax)
        if U_old is not None and cfg.lag_phillips_stab:
            _uo, _wo, _po, _phio = ufl.split(U_old)
            r_phi = _div(ufl, phi * w, ax) + _div(
                ufl, _migration_flux_ufl(ufl, migr, visc, vname, mname, _uo, _wo, _phio, cfg, a), ax
            )
        phi_diff_eff = cfg.phi_diff
        if cfg.phi_neg_diff > 0.0 and U_old is not None:
            _phi_prev = ufl.min_value(ufl.max_value(ufl.split(U_old)[3], -0.05), 0.05)
            _phi_hi = ufl.min_value(ufl.max_value(ufl.split(U_old)[3] - 0.62, -0.05), 0.05)
            phi_diff_eff = (cfg.phi_diff
                            + cfg.phi_neg_diff * 0.5 * (1.0 - ufl.tanh(_phi_prev / 0.005))
                            + cfg.phi_neg_diff * 0.5 * (1.0 + ufl.tanh(_phi_hi / 0.008)))
        if ds is not None:
            n_out = ufl.FacetNormal(W.mesh)
            f_phi = (-phi * ufl.dot(w, ufl.grad(s)) + phi_diff_eff * ufl.dot(ufl.grad(phi), ufl.grad(s))) * jac * dx
            f_phi += phi * ufl.max_value(ufl.dot(w, n_out), 0.0) * s * jac * ds(OUTLET)
        else:
            f_phi = (s * r_phi + phi_diff_eff * ufl.dot(ufl.grad(phi), ufl.grad(s))) * jac * dx
        f_phi += -ufl.dot(n_ph, ufl.grad(s)) * jac * dx
        if U_old is not None:
            u0, w0, _, phi0 = ufl.split(U_old)
            h = ufl.CellDiameter(W.mesh)
            if pseudo_cfl is not None:
                dt_u = pseudo_cfl * h / (ufl.sqrt(ufl.dot(u0, u0)) + 1.0e-3)
                dt_w = pseudo_cfl * h / (ufl.sqrt(ufl.dot(w0, w0)) + 1.0e-3)
            else:
                dt_u = dt_w = dt
            f_plasma += re * phic_pos * ufl.inner((u - u0) / dt_u, v) * jac * dx
            f_rbc += re * RHO_RATIO * phi_pos * ufl.inner((w - w0) / dt_w, z) * jac * dx
            f_phi += s * (phi - phi0) / dt_w * jac * dx
            if cfg.supg:
                wmag2 = ufl.dot(w, w) + 1.0e-12
                tau = ((2.0 / dt_w) ** 2 + 4.0 * wmag2 / h**2 + (4.0 * cfg.phi_diff / h**2) ** 2) ** -0.5
                f_phi += tau * ufl.dot(w, ufl.grad(s)) * ((phi - phi0) / dt_w + r_phi) * jac * dx
                w0n = ufl.sqrt(ufl.dot(w0, w0) + 1.0e-12)
                n_old = _migration_flux_ufl(ufl, migr, visc, vname, mname, u0, w0, phi0, cfg, a) if mname != "none" else ufl.as_vector([0.0] * dim)
                r_old = _div(ufl, phi0 * w0 + n_old, ax)
                gphi0 = ufl.sqrt(ufl.dot(ufl.grad(phi0), ufl.grad(phi0)) + 1.0e-8)
                nu_c = 0.5 * cfg.crosswind_C * h * ufl.abs(r_old) / gphi0
                pcw = ufl.Identity(dim) - ufl.outer(w0, w0) / w0n**2
                f_phi += nu_c * ufl.dot(pcw * ufl.grad(phi), ufl.grad(s)) * jac * dx
        return f_plasma + f_rbc + f_mix + f_phi

    # ============ comsol form (shared mixture viscosity, divided by phase fraction) ============
    f_plasma = (re * ufl.inner(ufl.grad(u) * u, v) + 2.0 * mu * ufl.inner(_sym_grad(ufl, u, ax), _sym_grad(ufl, v, ax))
                - p * _div(ufl, v, ax) - ufl.inner(force / phic_pos, v)) * jac * dx
    f_rbc = (re * RHO_RATIO * ufl.inner(ufl.grad(w) * w, z) + 2.0 * mu * ufl.inner(_sym_grad(ufl, w, ax), _sym_grad(ufl, z, ax))
             - p * _div(ufl, z, ax) + ufl.inner(force / phi_pos, z)) * jac * dx
    if d_hz > 0.0:
        f_plasma += (12.0 * mu / d_hz**2) * ufl.inner(u, v) * jac * dx
        f_rbc += (12.0 * mu / d_hz**2) * ufl.inner(w, z) * jac * dx
    if cfg.gravity:
        f_rbc += cfg.gravity * z[1] * jac * dx
    f_mix = q * _div(ufl, phi * w + (1.0 - phi) * u + n_ph, ax) * jac * dx
    r_phi = _div(ufl, phi * w + n_ph, ax)
    f_phi = (s * r_phi + cfg.phi_diff * ufl.dot(ufl.grad(phi), ufl.grad(s))) * jac * dx
    f_phi += -ufl.dot(n_ph, ufl.grad(s)) * jac * dx
    if cfg.supg:
        h = ufl.CellDiameter(W.mesh)
        wmag = ufl.sqrt(ufl.dot(w, w) + cfg.supg_w_floor**2)
        tau = h / (2.0 * wmag)
        f_phi += tau * ufl.dot(w, ufl.grad(s)) * r_phi * jac * dx
    return f_plasma + f_rbc + f_mix + f_phi


def mu_inf_hat(ufl, phi_reg):
    from hemoflow.closures.viscosity import mu_inf_cp
    from hemoflow.closures.constants import MU_PLASMA_CP
    return mu_inf_cp(phi_reg) / MU_PLASMA_CP


def _migration_flux_ufl(ufl, migr, visc, vname, mname, u, w, phi, cfg, a):
    if mname != "none" and not (isinstance(cfg.migration, str) and cfg.migration == "none"):
        ax = cfg.axisym
        phi_reg = ufl.min_value(ufl.max_value(phi, 0.0), cfg.phi_jam)
        j = phi_reg * w + (1.0 - phi_reg) * u
        dj = _sym_grad(ufl, j, ax)
        g2 = 2.0 * ufl.inner(dj, dj) + cfg.gdot_eps**2
        if cfg.depth_hat > 0.0:
            g2 = g2 + (12.0 / cfg.depth_hat**2) * ufl.dot(j, j)
        if mname == "phillips_nonlocal":
            g2 = g2 + (cfg.phillips_nl_c * a) ** 2 * ufl.dot(j, j)
        gdot_j = ufl.sqrt(g2)
        pv = ufl.variable(phi_reg)
        gdot_u = _gdot_of(ufl, u, cfg.gdot_eps, axisym=ax)
        gdot_w = _gdot_of(ufl, w, cfg.gdot_eps, axisym=ax)
        mu_pv, _ = _eval_viscosity(visc, vname, pv, gdot_u, gdot_w, cfg)
        dlnmu = ufl.diff(ufl.ln(mu_pv), pv)
        return migr(phi, gdot_j, ufl.grad(gdot_j), ufl.grad(phi), dlnmu, a)
    return ufl.as_vector([0.0] * u.ufl_shape[0])


def boundary_conditions(W, tags, phi_in: float, inlet_profile=None, pin_outlet_p: bool = False):
    """Dirichlet BCs: parabolic inlet on u/w, phi = phi_in on inlet, no-slip walls (verbatim twofluid)."""
    basix_ufl, ufl, fem = _deps()
    import numpy as np
    from hemoflow.geometries.mesher import INLET, WALL
    from hemoflow.solver.analytic import poiseuille
    if inlet_profile is None:
        inlet_profile = poiseuille
    msh = W.mesh
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    bcs = []
    for sub in (0, 1):
        V, _ = W.sub(sub).collapse()
        inflow = fem.Function(V)
        _dim = msh.topology.dim
        inflow.interpolate(lambda x: np.vstack([inlet_profile(x[1])] + [np.zeros_like(x[0])] * (_dim - 1)))
        zero = fem.Function(V)
        for func, tag in ((inflow, INLET), (zero, WALL)):
            dofs = fem.locate_dofs_topological((W.sub(sub), V), fdim, tags.find(tag))
            bcs.append(fem.dirichletbc(func, dofs, W.sub(sub)))
    Q, _ = W.sub(3).collapse()
    phi_bc = fem.Function(Q)
    phi_bc.x.array[:] = phi_in
    dofs = fem.locate_dofs_topological((W.sub(3), Q), fdim, tags.find(INLET))
    bcs.append(fem.dirichletbc(phi_bc, dofs, W.sub(3)))
    if pin_outlet_p:
        P, _ = W.sub(2).collapse()
        p0 = fem.Function(P)
        dofs = fem.locate_dofs_topological((W.sub(2), P), fdim, tags.find(OUTLET := 2))
        bcs.append(fem.dirichletbc(p0, dofs, W.sub(2)))
    return bcs


def initial_guess(W, U, cfg: EulerEulerConfig, length: float):
    """Poiseuille inlet guess for u/w, linear p drop, phi = phi_in (verbatim twofluid)."""
    import numpy as np
    from hemoflow.solver.analytic import poiseuille, mixture_viscosity_hat
    mu = float(mixture_viscosity_hat(cfg.phi_in))
    for sub in (0, 1):
        _dim = W.mesh.topology.dim
        U.sub(sub).interpolate(lambda x: np.vstack([poiseuille(x[1])] + [np.zeros_like(x[0])] * (_dim - 1)))
    U.sub(2).interpolate(lambda x: 12.0 * mu * (length - x[0]))
    U.sub(3).interpolate(lambda x: np.full_like(x[0], cfg.phi_in))
    U.x.scatter_forward()


def wall_psi(msh, tags):
    """Spalding wall-distance potential: -lap(psi) = 1, psi = 0 on WALL (verbatim twofluid)."""
    basix_ufl, ufl, fem = _deps()
    from dolfinx.fem.petsc import LinearProblem
    from hemoflow.geometries.mesher import WALL
    V = fem.functionspace(msh, ("Lagrange", 2))
    psi, q = ufl.TrialFunction(V), ufl.TestFunction(V)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    bc = fem.dirichletbc(0.0, fem.locate_dofs_topological(V, fdim, tags.find(WALL)), V)
    prob = LinearProblem(ufl.inner(ufl.grad(psi), ufl.grad(q)) * ufl.dx, fem.Constant(msh, 1.0) * q * ufl.dx,
                         bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                   "pc_factor_mat_solver_type": "mumps"})
    return prob.solve()
