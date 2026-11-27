"""hemoflow: plug-in/plug-out 2D Euler-Euler blood-flow library (MIT).

Top-level import is **host-safe**: it never imports dolfinx/ufl/basix/petsc4py/mpi4py.
Those are imported lazily only inside the FEniCSx solve/mesh entry points
(:mod:`hemoflow.solver.formulation`, :mod:`hemoflow.solver.stationary`,
:func:`hemoflow.geometries.mesher.build_dolfinx_mesh` / ``from_gmsh``).

Quick start::

    import hemoflow
    hemoflow.list_plugins()                       # all built-in closures + geometries
    cfg = hemoflow.EulerEulerConfig.v4_wu(re_p=10.0, d_hat=0.063, phi_in=0.30, k_star=2092.0)
    # custom closure plug-in:
    @hemoflow.register_viscosity("my_mu")
    def my_mu(phi): return 1.0 + 2.0 * phi
    cfg2 = hemoflow.EulerEulerConfig(re_p=10.0, d_hat=0.063, phi_in=0.3, viscosity="my_mu")
"""
from __future__ import annotations

__version__ = "0.2.0"

from . import registry  # noqa: F401
from .registry import (  # noqa: F401
    register_plugin, register_viscosity, register_drag, register_lift,
    register_migration, register_geometry, register_bc, get_plugin, list_plugins,
    geometry_metadata, PluginNotFoundError,
)
from . import closures  # noqa: F401  (registers built-in closure plugins)
from . import geometries  # noqa: F401  (registers built-in geometry plugins)
from . import solver  # noqa: F401
from .solver import EulerEulerConfig, BoundaryConfig, SolverConfig, FlowResult  # noqa: F401
from .solver.stationary import cfl_comsol, comsol_scaled_error  # noqa: F401

__all__ = [
    "__version__",
    "register_plugin", "register_viscosity", "register_drag", "register_lift",
    "register_migration", "register_geometry", "register_bc", "get_plugin", "list_plugins",
    "geometry_metadata", "PluginNotFoundError",
    "EulerEulerConfig", "BoundaryConfig", "SolverConfig", "FlowResult",
    "cfl_comsol", "comsol_scaled_error",
    "registry", "closures", "geometries", "solver",
]
