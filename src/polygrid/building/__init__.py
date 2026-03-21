"""Grid construction, Goldberg topology, composites, and assembly."""

from .builders import (
    build_pure_hex_grid,
    build_pentagon_centered_grid,
    hex_face_count,
    validate_pentagon_topology,
)
from .goldberg_topology import (
    build_goldberg_grid,
    goldberg_topology,
    goldberg_face_count,
    goldberg_embed_tutte,
    goldberg_optimise,
    fix_face_winding,
)
from .composite import CompositeGrid, StitchSpec, stitch_grids, join_grids, split_composite
from .assembly import AssemblyPlan, pent_hex_assembly, translate_grid, rotate_grid, scale_grid

__all__ = [
    "build_pure_hex_grid", "build_pentagon_centered_grid",
    "hex_face_count", "validate_pentagon_topology",
    "build_goldberg_grid", "goldberg_topology", "goldberg_face_count",
    "goldberg_embed_tutte", "goldberg_optimise", "fix_face_winding",
    "CompositeGrid", "StitchSpec", "stitch_grids", "join_grids", "split_composite",
    "AssemblyPlan", "pent_hex_assembly", "translate_grid", "rotate_grid", "scale_grid",
]
