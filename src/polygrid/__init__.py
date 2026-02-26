"""PolyGrid - pentagon-hex grid toolkit."""

from .models import Vertex, Edge, Face, MacroEdge
from .polygrid import PolyGrid
from .algorithms import build_face_adjacency, ring_faces
from .io import load_json, save_json
from .render import render_png
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
from .transforms import Overlay, OverlayPoint, OverlaySegment, OverlayRegion, apply_voronoi, apply_partition
from .visualize import (
    render_assembly_panels,
    render_single_panel,
    render_exploded,
    render_stitched,
    render_stitched_with_overlay,
    render_unstitched_with_overlay,
)
from .diagnostics import (
    ring_diagnostics,
    summarize_ring_stats,
    min_face_signed_area,
    has_edge_crossings,
    ring_quality_gates,
    diagnostics_report,
    ring_angle_spec,
)

__all__ = [
    "Vertex",
    "Edge",
    "Face",
    "MacroEdge",
    "PolyGrid",
    "build_face_adjacency",
    "ring_faces",
    "load_json",
    "save_json",
    "render_png",
    "build_pure_hex_grid",
    "build_pentagon_centered_grid",
    "build_goldberg_grid",
    "goldberg_topology",
    "goldberg_face_count",
    "goldberg_embed_tutte",
    "goldberg_optimise",
    "fix_face_winding",
    "hex_face_count",
    "validate_pentagon_topology",
    "CompositeGrid",
    "StitchSpec",
    "stitch_grids",
    "join_grids",
    "split_composite",
    "AssemblyPlan",
    "pent_hex_assembly",
    "translate_grid",
    "rotate_grid",
    "scale_grid",
    "Overlay",
    "OverlayPoint",
    "OverlaySegment",
    "OverlayRegion",
    "apply_voronoi",
    "apply_partition",
    "render_assembly_panels",
    "render_single_panel",
    "render_exploded",
    "render_stitched",
    "render_stitched_with_overlay",
    "render_unstitched_with_overlay",
    "ring_diagnostics",
    "summarize_ring_stats",
    "min_face_signed_area",
    "has_edge_crossings",
    "ring_quality_gates",
    "diagnostics_report",
    "ring_angle_spec",
]
