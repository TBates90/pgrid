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
from .algorithms import build_face_adjacency, get_face_adjacency, ring_faces
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

# ── Tile Data ───────────────────────────────────────────────────────
from .tile_data import (
    FieldDef,
    TileSchema,
    TileData,
    TileDataStore,
    save_tile_data,
    load_tile_data,
)

# ── Regions (Terrain Partitioning) ──────────────────────────────────
from .regions import (
    Region,
    RegionMap,
    RegionValidation,
    partition_angular,
    partition_flood_fill,
    partition_voronoi,
    partition_noise,
    assign_field,
    assign_biome,
    regions_to_overlay,
    validate_region_map,
)

# ── Terrain Generation (Phase 7) ────────────────────────────────────
from .noise import (
    fbm,
    ridged_noise,
    domain_warp,
    gradient_mask,
    terrace,
    normalize as noise_normalize,
    remap,
    fbm_3d,
    ridged_noise_3d,
)
from .heightmap import (
    sample_noise_field,
    sample_noise_field_region,
    sample_noise_field_3d,
    smooth_field,
    blend_fields,
    clamp_field,
    normalize_field,
)
from .mountains import (
    MountainConfig,
    generate_mountains,
    MOUNTAIN_RANGE,
    ALPINE_PEAKS,
    ROLLING_HILLS,
    MESA_PLATEAU,
)
from .terrain_render import (
    elevation_to_overlay,
    hillshade,
    render_terrain,
)
from .rivers import (
    RiverSegment,
    RiverNetwork,
    RiverConfig,
    steepest_descent_path,
    find_drainage_basins,
    fill_depressions,
    flow_accumulation,
    generate_rivers,
    carve_river_valleys,
    assign_river_data,
    river_to_overlay,
)
from .pipeline import (
    TerrainStep,
    StepResult,
    TerrainPipeline,
    PipelineResult,
    MountainStep,
    RiverStep,
    CustomStep,
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

# ── Globe (requires models library) ────────────────────────────────
try:
    from .globe import build_globe_grid, GlobeGrid
except ImportError:
    pass  # models library not installed — globe features unavailable

try:
    from .globe_mesh import (
        terrain_colors_for_layout,
        terrain_colors_from_tile_colours,
        build_terrain_layout_mesh,
        build_terrain_face_meshes,
        build_terrain_tile_meshes,
        build_terrain_edge_mesh,
    )
except ImportError:
    pass  # models library not installed — globe mesh features unavailable

try:
    from .globe_export import (
        export_globe_payload,
        export_globe_json,
        validate_globe_payload,
    )
except ImportError:
    pass  # models library not installed — globe export features unavailable

try:
    from .globe_renderer import (
        build_coloured_globe_mesh,
        build_coloured_globe_mesh_from_export,
        build_edge_mesh_for_frequency,
        prepare_terrain_scene,
        render_terrain_globe_opengl,
    )
except ImportError:
    pass  # models library not installed — globe renderer features unavailable

# ── Detail grids ────────────────────────────────────────────────────
from .detail_grid import (
    build_detail_grid,
    detail_face_count,
    generate_detail_terrain,
    render_detail_texture,
    build_texture_atlas,
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
    "get_face_adjacency",
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
    # Tile Data
    "FieldDef",
    "TileSchema",
    "TileData",
    "TileDataStore",
    "save_tile_data",
    "load_tile_data",
    # Regions
    "Region",
    "RegionMap",
    "RegionValidation",
    "partition_angular",
    "partition_flood_fill",
    "partition_voronoi",
    "partition_noise",
    "assign_field",
    "assign_biome",
    "regions_to_overlay",
    "validate_region_map",
    # Terrain Generation (Phase 7)
    # -- Noise primitives
    "fbm",
    "ridged_noise",
    "domain_warp",
    "gradient_mask",
    "terrace",
    "noise_normalize",
    "remap",
    "fbm_3d",
    "ridged_noise_3d",
    # -- Heightmap bridge
    "sample_noise_field",
    "sample_noise_field_region",
    "sample_noise_field_3d",
    "smooth_field",
    "blend_fields",
    "clamp_field",
    "normalize_field",
    # -- Mountains
    "MountainConfig",
    "generate_mountains",
    "MOUNTAIN_RANGE",
    "ALPINE_PEAKS",
    "ROLLING_HILLS",
    "MESA_PLATEAU",
    # -- Terrain rendering
    "elevation_to_overlay",
    "hillshade",
    "render_terrain",
    # -- Rivers
    "RiverSegment",
    "RiverNetwork",
    "RiverConfig",
    "steepest_descent_path",
    "find_drainage_basins",
    "fill_depressions",
    "flow_accumulation",
    "generate_rivers",
    "carve_river_valleys",
    "assign_river_data",
    "river_to_overlay",
    # -- Pipeline
    "TerrainStep",
    "StepResult",
    "TerrainPipeline",
    "PipelineResult",
    "MountainStep",
    "RiverStep",
    "CustomStep",
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
    # Globe (optional — requires models library)
    "build_globe_grid",
    "GlobeGrid",
    # Globe mesh bridge (optional — requires models library)
    "terrain_colors_for_layout",
    "terrain_colors_from_tile_colours",
    "build_terrain_layout_mesh",
    "build_terrain_face_meshes",
    "build_terrain_tile_meshes",
    "build_terrain_edge_mesh",
    # Globe export
    "export_globe_payload",
    "export_globe_json",
    # Globe renderer (optional — requires models library)
    "build_coloured_globe_mesh",
    "build_coloured_globe_mesh_from_export",
    "build_edge_mesh_for_frequency",
    "prepare_terrain_scene",
    "render_terrain_globe_opengl",
    # Detail grids
    "build_detail_grid",
    "detail_face_count",
    "generate_detail_terrain",
    "render_detail_texture",
    "build_texture_atlas",
]
