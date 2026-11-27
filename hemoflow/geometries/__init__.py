"""hemoflow geometries: exact 2D families G1-G13 + KG (builders) + graded mesher + metadata.

Importing this package registers every family in :mod:`hemoflow.registry` under
``@register_geometry`` with its topology/parameter metadata (see :mod:`.families`).
"""
from __future__ import annotations

from hemoflow.registry import register_geometry
from . import builders, mesher, families  # noqa: F401
from .builders import (  # noqa: F401
    MeshSpec, Block2D, PatchSpec, SampleLine, Point2D,
    build_geometry, build_g1, format_block_mesh_dict,
)
from .mesher import (  # noqa: F401
    INLET, OUTLET, WALL, AXIS,
    graded_mesh_arrays, graded_mesh_arrays_from_spec, graded_block_schedule,
    cell_signed_areas, edge_use_counts, wall_normal_thicknesses,
    build_dolfinx_mesh, custom_mesh, from_gmsh, build_mesh_spec, expected_domain_area,
)
from .families import FAMILIES, FLOW_PARAMS, VARIANTS  # noqa: F401


def _make_builder(fam: str):
    def builder(params, level: int = 1):
        return build_geometry(fam, params, level=level)
    builder.__name__ = f"build_{fam.lower()}"
    return builder


def register_all_geometries() -> None:
    for fam, meta in FAMILIES.items():
        register_geometry(
            fam,
            name=meta["name"],
            topology=meta["topology"],
            description=meta["description"],
            params=meta["params"],
        )(_make_builder(fam))


register_all_geometries()
