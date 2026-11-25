"""COMSOL-documented stationary solver strategy, re-implemented for FEniCSx (knowledge/17 §5).

- Pseudo time stepping (CFD Module Eq. 3-60): one pseudo step per nonlinear iteration, previous iterate as "nojac".
- CFL schedule (CFD Module Eq. 3-61, default manual expression):
  CFL = 1.3^min(k,9) + [k>20] 9*1.3^min(k-20,9) + [k>40] 90*1.3^min(k-40,9).
- Damped Newton (Reference Manual, "About the Stationary Solver"): step dU from J dU = -f(U0); trial U1 = U0 + l*dU;
  error estimate E from J E = -f(U1) with the same Jacobian; reduce l while |E| >= |dU| (affine-invariant test).
"""
from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, create_matrix, create_vector, set_bc
from petsc4py import PETSc


def cfl_comsol(k: int) -> float:
    """CFD Module Eq. 3-61 (k = nonlinear iteration number, starting at 1)."""
    c = 1.3 ** min(k, 9)
    if k > 20:
        c += 9.0 * 1.3 ** min(k - 20, 9)
    if k > 40:
        c += 90.0 * 1.3 ** min(k - 40, 9)
    return c


class ComsolStationary:
    def __init__(self, U, U_prev, F, bcs, cfl_const, lam_min=1e-4):
        self.U, self.U_prev, self.bcs, self.cfl = U, U_prev, bcs, cfl_const
        self.Ff = fem.form(F)
        self.Jf = fem.form(ufl.derivative(F, U))
        self.A = create_matrix(self.Jf)
        self.b = create_vector(self.Ff)
        self.dx, self.E = self.b.duplicate(), self.b.duplicate()
        self.ksp = PETSc.KSP().create(U.function_space.mesh.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType("preonly")
        pc = self.ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        self.lam_min = lam_min

    def _assemble_F(self):
        with self.b.localForm() as bl:
            bl.set(0.0)
        assemble_vector(self.b, self.Ff)
        apply_lifting(self.b, [self.Jf], bcs=[self.bcs], x0=[self.U.x.petsc_vec], alpha=-1.0)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b, self.bcs, self.U.x.petsc_vec, -1.0)

    def step(self, k: int):
        """One COMSOL nonlinear iteration k: returns (lambda, |dU|/|U|, |E|/|U|)."""
        self.cfl.value = cfl_comsol(k)
        self.U_prev.x.array[:] = self.U.x.array
        self.A.zeroEntries()
        assemble_matrix(self.A, self.Jf, bcs=self.bcs)
        self.A.assemble()
        self._assemble_F()
        self.ksp.solve(self.b, self.dx)  # U_new = U - dx (dolfinx sign convention)
        u0 = self.U.x.array.copy()
        nU = max(np.linalg.norm(u0), 1e-12)
        ndx = self.dx.norm() / nU
        lam = 1.0
        while True:
            self.U.x.array[:] = u0 - lam * self.dx.array
            self.U.x.scatter_forward()
            self._assemble_F()
            self.ksp.solve(self.b, self.E)  # same factorised Jacobian
            nE = self.E.norm() / nU
            if not np.isfinite(nE):
                nE = np.inf
            if nE < ndx or lam <= self.lam_min:
                return lam, ndx, nE
            lam *= 0.5
