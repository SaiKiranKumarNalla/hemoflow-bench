"""hemoflow solver subpackage: config, formulation (lazy UFL), stationary, analytic, result.

Importing this subpackage is host-safe (no dolfinx/ufl at module import time); the FEniCSx
backend is imported lazily inside :mod:`.formulation` / :mod:`.stationary` functions.
"""
from .config import EulerEulerConfig, BoundaryConfig, SolverConfig  # noqa: F401
from .analytic import poiseuille, brinkman_inlet, developed_phillips_profile  # noqa: F401
from .result import FlowResult  # noqa: F401
from . import formulation, stationary  # noqa: F401  (host-safe module import; lazy dolfinx inside)
from .stationary import cfl_comsol, comsol_scaled_error, ComsolStationary, EulerEulerSolver  # noqa: F401
