"""PolyGrid — topology-first polygon grid toolkit.

Public API is organised into layers:

- **Core** — models, container, algorithms, geometry
- **Building** — grid constructors, Goldberg topology, stitching, assembly
- **Transforms** — overlay model and transform functions
- **Terrain** — noise, heightmaps, mountains, regions
- **Globe** — globe generation, export (requires *models* library)
- **Detail** — detail grids, tile detail, terrain, rendering

Specialised subsystems (globe_renderer_v2, uv_texture, tile_uv_align,
atlas_utils) are **not** re-exported here — import them directly::

    from polygrid.globe_renderer_v2 import render_globe_v2
    from polygrid.tile_uv_align import build_polygon_cut_atlas
"""

import warnings as _warnings

# ── Core ────────────────────────────────────────────────────────────
from .models import Vertex, Edge, Face, MacroEdge, Region
from .polygrid import PolyGrid
from .algorithms import build_face_adjacency, get_face_adjacency, ring_faces

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

# ── Regions ─────────────────────────────────────────────────────────
from .regions import (
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

# ── Terrain Generation ──────────────────────────────────────────────
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

# ── Globe (requires models library) ────────────────────────────────
try:
    from .globe import build_globe_grid, GlobeGrid
except ImportError:
    _warnings.warn(
        "polygrid.globe requires the 'models' library.  "
        "Install with: pip install polygrid[globe]",
        ImportWarning,
        stacklevel=2,
    )

try:
    from .globe_export import (
        export_globe_payload,
        export_globe_json,
        validate_globe_payload,
        globe_to_colour_map,
    )
except ImportError:
    pass  # globe_export depends on globe → already warned above

# ── Detail grids ────────────────────────────────────────────────────
from .detail_grid import (
    build_detail_grid,
    deform_grid_to_uv_shape,
    detail_face_count,
    generate_detail_terrain,
)
from .tile_detail import (
    TileDetailSpec,
    build_all_detail_grids,
    DetailGridCollection,
    find_polygon_corners,
    build_tile_with_neighbours,
)
from .detail_terrain import (
    compute_boundary_elevations,
    classify_detail_faces,
    generate_detail_terrain_bounded,
    generate_all_detail_terrain,
)
from .detail_render import (
    BiomeConfig,
    detail_elevation_to_colour,
    render_detail_texture_enhanced,
)

__all__ = [
    # Core
    "Vertex", "Edge", "Face", "MacroEdge", "Region",
    "PolyGrid",
    "build_face_adjacency", "get_face_adjacency", "ring_faces",
    # Building
    "build_pure_hex_grid", "build_pentagon_centered_grid",
    "hex_face_count", "validate_pentagon_topology",
    "build_goldberg_grid", "goldberg_topology", "goldberg_face_count",
    "goldberg_embed_tutte", "goldberg_optimise", "fix_face_winding",
    "CompositeGrid", "StitchSpec", "stitch_grids", "join_grids", "split_composite",
    "AssemblyPlan", "pent_hex_assembly", "translate_grid", "rotate_grid", "scale_grid",
    # Transforms
    "Overlay", "OverlayPoint", "OverlaySegment", "OverlayRegion",
    "apply_voronoi",
    # Tile Data
    "FieldDef", "TileSchema", "TileData", "TileDataStore",
    "save_tile_data", "load_tile_data",
    # Regions
    "RegionMap", "RegionValidation",
    "partition_angular", "partition_flood_fill", "partition_voronoi", "partition_noise",
    "assign_field", "assign_biome", "regions_to_overlay", "validate_region_map",
    # Terrain
    "fbm", "ridged_noise", "domain_warp", "gradient_mask", "terrace",
    "noise_normalize", "remap", "fbm_3d", "ridged_noise_3d",
    "sample_noise_field", "sample_noise_field_region", "sample_noise_field_3d",
    "smooth_field", "blend_fields", "clamp_field", "normalize_field",
    "MountainConfig", "generate_mountains",
    "MOUNTAIN_RANGE", "ALPINE_PEAKS", "ROLLING_HILLS", "MESA_PLATEAU",
    # Globe (available when models library is installed)
    "build_globe_grid", "GlobeGrid",
    "export_globe_payload", "export_globe_json", "validate_globe_payload",
    "globe_to_colour_map",
    # Detail grids
    "build_detail_grid", "deform_grid_to_uv_shape",
    "detail_face_count", "generate_detail_terrain",
    "TileDetailSpec", "build_all_detail_grids", "DetailGridCollection",
    "find_polygon_corners", "build_tile_with_neighbours",
    "compute_boundary_elevations", "classify_detail_faces",
    "generate_detail_terrain_bounded", "generate_all_detail_terrain",
    "BiomeConfig", "detail_elevation_to_colour", "render_detail_texture_enhanced",
]
