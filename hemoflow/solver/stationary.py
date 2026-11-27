"""COMSOL-documented stationary solver, ported from ``cfd/fenicsx/comsol_solver.py``.

- :func:`cfl_comsol` / :func:`comsol_scaled_error` are **host-safe** (NumPy only) so unit
  tests can check them without dolfinx.
- :class:`ComsolStationary` and :class:`EulerEulerSolver` import dolfinx/ufl/petsc4py lazily.

Pseudo time stepping (CFD Module Eq. 3-60) + CFL schedule (Eq. 3-61) + damped Newton with
trust region & phi cap (Reference Manual "About the Stationary Solver").
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["cfl_comsol", "comsol_scaled_error", "ComsolStationary", "EulerEulerSolver",
           "FLOOR_U", "FLOOR_W", "FLOOR_PHI", "FIELD_NAMES"]

FLOOR_U = 1.0
FLOOR_W = 1.0
FLOOR_PHI = 0.05
FIELD_NAMES = ("u", "w", "p", "phi")


def cfl_comsol(k: int) -> float:
    """CFD Module Eq. 3-61 (k = nonlinear iteration number, starting at 1)."""
    c = 1.3 ** min(k, 9)
    if k > 20:
        c += 9.0 * 1.3 ** min(k - 20, 9)
    if k > 40:
        c += 90.0 * 1.3 ** min(k - 40, 9)
    return c


def comsol_scaled_error(delta_fields, ref_fields, p_absmax: float) -> dict[str, float]:
    """Per-field scaled relative error of an update (COMSOL automatic scaling), verbatim.

    ``e_f = ||dU_f|| / max(||U_f||, s_f)`` with floors s_u = s_w = 1 (star velocity),
    s_p = max|p| at the current iterate, s_phi = 0.05; ``E = RMS(e_u, e_w, e_p, e_phi)``.
    Returns ``{"u","w","p","phi","E"}`` (inf on non-finite input).
    """
    floors = (FLOOR_U, FLOOR_W, max(float(p_absmax), 1.0e-12), FLOOR_PHI)
    errs: list[float] = []
    for d, r, s in zip(delta_fields, ref_fields, floors):
        nd = float(np.linalg.norm(np.asarray(d, dtype=np.float64)))
        nr = float(np.linalg.norm(np.asarray(r, dtype=np.float64)))
        if not (np.isfinite(nd) and np.isfinite(nr)):
            out = {name: float("inf") for name in FIELD_NAMES}
            out["E"] = float("inf")
            return out
        errs.append(nd / max(nr, s))
    out = dict(zip(FIELD_NAMES, errs))
    out["E"] = float(np.sqrt(np.mean(np.square(errs))))
    return out


def _deps():
    import ufl
    from dolfinx import fem
    from dolfinx.fem import petsc as fp
    from petsc4py import PETSc
    return ufl, fem, fp, PETSc


class ComsolStationary:
    """Damped Newton with COMSOL pseudo-time CFL schedule, trust region and phi cap (verbatim)."""

    def __init__(self, U, U_prev, F, bcs, cfl_const, lam_min=1e-4, cfl_scale=1.0, phi_cap=0.68):
        ufl, fem, fp, PETSc = _deps()
        self.U, self.U_prev, self.bcs, self.cfl = U, U_prev, bcs, cfl_const
        self.cfl_scale = float(cfl_scale)
        self.phi_cap = float(phi_cap)
        self.Ff = fem.form(F)
        self.Jf = fem.form(ufl.derivative(F, U))
        self.A = fp.create_matrix(self.Jf)
        self.b = fp.create_vector(self.Ff)
        self.dx, self.E = self.b.duplicate(), self.b.duplicate()
        self.ksp = PETSc.KSP().create(U.function_space.mesh.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType("preonly")
        pc = self.ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        self.lam_min = lam_min
        W = U.function_space
        n_arr = int(U.x.array.size)
        self.field_dofs = []
        for i in range(len(FIELD_NAMES)):
            _, cmap = W.sub(i).collapse()
            idx = np.asarray(cmap, dtype=np.int64)
            self.field_dofs.append(idx[idx < n_arr])
        self.last_field_errors: dict[str, float] = {name: float("nan") for name in FIELD_NAMES}
        self.last_field_errors["E"] = float("nan")

    def _assemble_F(self):
        _, _, fp, PETSc = _deps()
        with self.b.localForm() as bl:
            bl.set(0.0)
        fp.assemble_vector(self.b, self.Ff)
        fp.apply_lifting(self.b, [self.Jf], bcs=[self.bcs], x0=[self.U.x.petsc_vec], alpha=-1.0)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fp.set_bc(self.b, self.bcs, self.U.x.petsc_vec, -1.0)

    def step(self, k: int):
        """One COMSOL nonlinear iteration k.  Returns ``(lam, ndx, nE, E)``."""
        self.cfl.value = cfl_comsol(k) * self.cfl_scale
        self.U_prev.x.array[:] = self.U.x.array
        self.A.zeroEntries()
        _, _, fp, _ = _deps()
        fp.assemble_matrix(self.A, self.Jf, bcs=self.bcs)
        self.A.assemble()
        self._assemble_F()
        self.ksp.solve(self.b, self.dx)
        u0 = self.U.x.array.copy()
        nU = max(np.linalg.norm(u0), 1e-12)
        ndx = self.dx.norm() / nU
        if ndx > 0.25:
            self.dx.array[:] *= 0.25 / ndx
            ndx = 0.25
        lam = 1.0
        while True:
            self.U.x.array[:] = u0 - lam * self.dx.array
            phi_idx = self.field_dofs[3]
            if phi_idx.size:
                self.U.x.array[phi_idx] = np.clip(self.U.x.array[phi_idx], 0.0, self.phi_cap)
            self.U.x.scatter_forward()
            self._assemble_F()
            self.ksp.solve(self.b, self.E)
            nE = self.E.norm() / nU
            if not np.isfinite(nE):
                nE = np.inf
            if nE < ndx or lam <= self.lam_min:
                u1 = self.U.x.array
                dU = lam * self.dx.array
                p_idx = self.field_dofs[2]
                p_absmax = float(np.max(np.abs(u1[p_idx]))) if p_idx.size else 0.0
                errs = comsol_scaled_error([dU[idx] for idx in self.field_dofs],
                                           [u1[idx] for idx in self.field_dofs], p_absmax)
                self.last_field_errors = errs
                return lam, ndx, nE, errs["E"]
            lam *= 0.5


class EulerEulerSolver:
    """High-level solver: ``EulerEulerSolver(config, bc, solver_cfg).solve(geometry_or_mesh) -> FlowResult``.

    ``geometry_or_mesh`` may be a family string ("G2"), a ``MeshSpec``, a ``(msh, tags)``
    tuple (from :func:`hemoflow.geometries.mesher.custom_mesh` / ``from_gmsh``), or a dolfinx
    ``Mesh``.  Requires the dolfinx Docker/conda environment.
    """

    def __init__(self, config, bc=None, solver_cfg=None):
        from hemoflow.solver.config import BoundaryConfig, SolverConfig
        self.config = config
        self.bc = bc if bc is not None else BoundaryConfig()
        self.solver_cfg = solver_cfg if solver_cfg is not None else SolverConfig()

    def _mesh_and_tags(self, geometry_or_mesh):
        from hemoflow.geometries import mesher, builders
        if isinstance(geometry_or_mesh, str):
            fam = geometry_or_mesh.upper()
            params = getattr(self, "geometry_params", {}) or {}
            msh, tags, spec = mesher.build_dolfinx_mesh(fam, params, refine=1.0)
            return msh, tags, spec
        if isinstance(geometry_or_mesh, builders.MeshSpec):
            spec = geometry_or_mesh
            msh, tags, _ = mesher.build_dolfinx_mesh(spec.family, spec.params, refine=1.0)
            return msh, tags, spec
        if isinstance(geometry_or_mesh, tuple):
            msh, tags = geometry_or_mesh
            return msh, tags, None
        # assume a dolfinx Mesh
        return geometry_or_mesh, None, None

    def solve(self, geometry_or_mesh, geometry_params: dict | None = None):
        """Run the stationary solve and return a :class:`FlowResult` (requires dolfinx)."""
        from dolfinx import fem
        from dolfinx.fem.petsc import NonlinearProblem  # noqa: F401
        import ufl
        from hemoflow.solver.formulation import (build_space, build_residual,
                                                 boundary_conditions, initial_guess, wall_psi)
        from hemoflow.solver.result import FlowResult
        if geometry_params is not None:
            self.geometry_params = geometry_params
        msh, tags, spec = self._mesh_and_tags(geometry_or_mesh)
        W = build_space(msh)
        U = fem.Function(W)
        length = float(spec.params.get("L_out_ratio", 20.0)) if spec else 20.0
        initial_guess(W, U, self.config, length)
        bcs = boundary_conditions(W, tags, self.config.phi_in, self.bc.inlet_function(), self.bc.pin_outlet_p)
        psi = wall_psi(msh, tags) if (isinstance(self.config.lift, str) and "wall" in self.config.lift) else None
        cfl = fem.Constant(msh, self.solver_cfg.cfl_const)
        ds = ufl.Measure("ds", domain=msh)
        U_prev = fem.Function(W)
        F = build_residual(W, U, self.config, ds=ds, U_old=U_prev, pseudo_cfl=cfl, psi_wall=psi)
        solver = ComsolStationary(U, U_prev, F, bcs, cfl,
                                  lam_min=self.solver_cfg.lam_min, cfl_scale=self.solver_cfg.cfl_scale,
                                  phi_cap=self.solver_cfg.phi_cap)
        E = np.inf
        for k in range(1, self.solver_cfg.max_outer + 1):
            lam, ndx, nE, E = solver.step(k)
            if E < self.solver_cfg.tol_E:
                break
        U.x.scatter_forward()
        return self._to_result(msh, W, U, spec, k, E)

    def _to_result(self, msh, W, U, spec, n_it, E):
        """Project the mixed solution onto a resampled grid -> FlowResult."""
        import numpy as np
        from dolfinx import fem as _fem, geometry as _geo
        from hemoflow.solver.result import FlowResult
        u, _w, p, phi = U.sub(0), U.sub(1), U.sub(2), U.sub(3)
        bb = msh.geometry.x
        x0, x1 = float(bb[:, 0].min()), float(bb[:, 0].max())
        y0, y1 = float(bb[:, 1].min()), float(bb[:, 1].max())
        nx, ny = 256, 128
        gx = np.linspace(x0, x1, nx)
        gy = np.linspace(y0, y1, ny)
        X, Y = np.meshgrid(gx, gy)
        pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])

        bb_tree = _geo.bb_tree(msh, msh.topology.dim)
        cell_candidates = _geo.compute_collisions_points(bb_tree, pts)
        colliding = _geo.compute_colliding_cells(msh, cell_candidates, pts)
        cell_ids = np.full(X.size, -1, dtype=np.int32)
        for k in range(X.size):
            links = colliding.links(k)
            if len(links) > 0:
                cell_ids[k] = links[0]
        valid = cell_ids >= 0
        grid_mask = valid.reshape(ny, nx)

        grid_ux = np.zeros((ny, nx), dtype=np.float64)
        grid_uy = np.zeros((ny, nx), dtype=np.float64)
        grid_p = np.zeros((ny, nx), dtype=np.float64)
        grid_phi = np.zeros((ny, nx), dtype=np.float64)

        if valid.any():
            Vv = _fem.functionspace(msh, ("Lagrange", 1, (msh.topology.dim,)))
            V1 = _fem.functionspace(msh, ("Lagrange", 1))
            uv = _fem.Function(Vv); uv.interpolate(u)
            pv = _fem.Function(V1); pv.interpolate(p)
            fv = _fem.Function(V1); fv.interpolate(phi)
            vpts = pts[valid]
            vcells = cell_ids[valid]
            uvals = uv.eval(vpts, vcells)
            grid_ux.ravel()[valid] = uvals[:, 0]
            grid_uy.ravel()[valid] = uvals[:, 1]
            grid_p.ravel()[valid] = pv.eval(vpts, vcells)[:, 0]
            grid_phi.ravel()[valid] = fv.eval(vpts, vcells)[:, 0]

        meta = {"family": spec.family if spec else None, "n_it": n_it, "E": float(E),
                "re_p": self.config.re_p, "d_hat": self.config.d_hat, "phi_in": self.config.phi_in}
        return FlowResult(grid_x=gx, grid_y=gy, grid_mask=grid_mask,
                          grid_ux=grid_ux, grid_uy=grid_uy, grid_p=grid_p, grid_phi=grid_phi,
                          params=dict(self.config.__dict__), metadata=meta)
