"""Nondimensional OpenFOAM case generator for 2D microfluidic families G2–G7."""

from cfd.gen.case import write_case
from cfd.gen.geometry import (
    ArcEdge,
    Block2D,
    MeshSpec,
    SampleLine,
    build_g2,
    build_g3,
    build_g4,
    build_g5,
    build_g6,
    build_g7,
    build_geometry,
    build_kg,
    format_block_mesh_dict,
)

__all__ = [
    "ArcEdge",
    "Block2D",
    "MeshSpec",
    "SampleLine",
    "build_g2",
    "build_g3",
    "build_g4",
    "build_g5",
    "build_g6",
    "build_g7",
    "build_geometry",
    "build_kg",
    "format_block_mesh_dict",
    "write_case",
]
