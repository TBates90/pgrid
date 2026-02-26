"""PolyGrid — topology-first polygon grid toolkit.

Public API is organised into layers:

- **Core** — models, container, algorithms, I/O
- **Building** — grid constructors, Goldberg topology, stitching, assembly
- **Transforms** — overlay model and transform functions
- **Rendering** — visualisation (requires matplotlib)
- **Diagnostics** — quality checks and reports
"""

# ── Core ────────────────────────────────────────────────────────────
from .models import Vertex, Edge, Face, MacroEdge
from .polygrid import PolyGrid
from .algorithms import build_face_adjacency, ring_faces
from .io import load_json, save_json

# ── Building ────────────────────────────────────────────────────────
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

# ── Transforms ──────────────────────────────────────────────────────
from .transforms import (
    Overlay,
    OverlayPoint,
    OverlaySegment,
    OverlayRegion,
    apply_voronoi,
    apply_partition,
)

# ── Rendering (requires matplotlib) ────────────────────────────────
from .visualize import (
    render_png,
    render_single_panel,
    render_assembly_panels,
    render_exploded,
    render_stitched,
    render_stitched_with_overlay,
    render_unstitched_with_overlay,
)

# ── Diagnostics ─────────────────────────────────────────────────────
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
    # Core
    "Vertex",
    "Edge",
    "Face",
    "MacroEdge",
    "PolyGrid",
    "build_face_adjacency",
    "ring_faces",
    "load_json",
    "save_json",
    # Building
    "build_pure_hex_grid",
    "build_pentagon_centered_grid",
    "hex_face_count",
    "validate_pentagon_topology",
    "build_goldberg_grid",
    "goldberg_topology",
    "goldberg_face_count",
    "goldberg_embed_tutte",
    "goldberg_optimise",
    "fix_face_winding",
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
    # Transforms
    "Overlay",
    "OverlayPoint",
    "OverlaySegment",
    "OverlayRegion",
    "apply_voronoi",
    "apply_partition",
    # Rendering
    "render_png",
    "render_single_panel",
    "render_assembly_panels",
    "render_exploded",
    "render_stitched",
    "render_stitched_with_overlay",
    "render_unstitched_with_overlay",
    # Diagnostics
    "ring_diagnostics",
    "summarize_ring_stats",
    "min_face_signed_area",
    "has_edge_crossings",
    "ring_quality_gates",
    "diagnostics_report",
    "ring_angle_spec",
]
