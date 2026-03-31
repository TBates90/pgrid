"""PolyGrid — topology-first polygon grid toolkit.

Public API is organised into sub-packages:

- **core/** — models, container, algorithms, geometry
- **building/** — grid constructors, Goldberg topology, stitching, assembly
- **data/** — tile data store, overlay model, transform functions
- **terrain/** — noise, heightmaps, mountains, regions
- **globe/** — globe generation, export (requires *models* library)
- **detail/** — detail grids, tile detail, terrain, rendering
- **rendering/** — UV mapping, texture rendering, globe visualization

The rendering sub-package (globe_renderer_v2, uv_texture, tile_uv_align,
atlas_utils) is **not** re-exported here — import directly::

    from polygrid.rendering.globe_renderer_v2 import render_globe_v2
    from polygrid.rendering.tile_uv_align import build_polygon_cut_atlas

Backward-compat shims at the old flat paths still work::

    from polygrid.globe_renderer_v2 import render_globe_v2  # also OK
"""

import warnings as _warnings

# ── Core ────────────────────────────────────────────────────────────
from .core.models import Vertex, Edge, Face, MacroEdge, Region
from .core.polygrid import PolyGrid
from .core.algorithms import build_face_adjacency, get_face_adjacency, ring_faces

# ── Building ────────────────────────────────────────────────────────
from .building.builders import (
    build_pure_hex_grid,
    build_pentagon_centered_grid,
    hex_face_count,
    validate_pentagon_topology,
)
from .building.goldberg_topology import (
    build_goldberg_grid,
    goldberg_topology,
    goldberg_face_count,
    goldberg_embed_tutte,
    goldberg_optimise,
    fix_face_winding,
)
from .building.composite import CompositeGrid, StitchSpec, stitch_grids, join_grids, split_composite
from .building.assembly import AssemblyPlan, pent_hex_assembly, translate_grid, rotate_grid, scale_grid

# ── Transforms ──────────────────────────────────────────────────────
from .data.transforms import (
    Overlay,
    OverlayPoint,
    OverlaySegment,
    OverlayRegion,
    apply_voronoi,
)

# ── Tile Data ───────────────────────────────────────────────────────
from .data.tile_data import (
    FieldDef,
    TileSchema,
    TileData,
    TileDataStore,
    save_tile_data,
    load_tile_data,
)

# ── Regions ─────────────────────────────────────────────────────────
from .terrain.regions import (
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
from .terrain.noise import (
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
from .terrain.heightmap import (
    sample_noise_field,
    sample_noise_field_region,
    sample_noise_field_3d,
    smooth_field,
    blend_fields,
    clamp_field,
    normalize_field,
)
from .terrain.mountains import (
    MountainConfig,
    generate_mountains,
    MOUNTAIN_RANGE,
    ALPINE_PEAKS,
    ROLLING_HILLS,
    MESA_PLATEAU,
)
from .terrain.temperature import (
    compute_temperature,
    generate_temperature_field,
    LATITUDE_WEIGHT,
    LAPSE_RATE,
)
from .terrain.moisture import (
    compute_ocean_distance,
    generate_moisture_field,
    OCEAN_PROXIMITY_WEIGHT,
    ELEVATION_PENALTY,
    MAX_OCEAN_DISTANCE,
)
from .terrain.classification import (
    classify_tile,
    generate_terrain_field,
    TERRAIN_TYPES,
    OCEAN,
    SNOW,
    TUNDRA,
    MOUNTAINS,
    DESERT,
    WETLAND,
    HILLS,
    PLAINS,
)
from .terrain.features import (
    detect_coast,
    detect_lakes,
    place_forests,
    generate_features,
    get_features,
    add_feature,
    COAST,
    LAKE,
    FOREST,
    FEATURE_TYPES,
)

# ── Integration API ─────────────────────────────────────────────────
try:
    from .integration import (
        PlanetParams,
        RegionParams,
        TileResult,
        GenerationResult,
        generate_planet,
        parse_layout,
    )
except ImportError:
    pass  # integration depends on globe → warned below

try:
    from .integration_atlas import (
        PlanetAtlasResult,
        generate_planet_atlas,
    )
except ImportError:
    pass  # atlas generation has heavier deps (scipy, matplotlib)

# ── Globe (requires models library) ────────────────────────────────
try:
    from .globe.globe import build_globe_grid, GlobeGrid
except ImportError:
    _warnings.warn(
        "polygrid.globe requires the 'models' library.  "
        "Install with: pip install polygrid[globe]",
        ImportWarning,
        stacklevel=2,
    )

try:
    from .globe.globe_export import (
        export_globe_payload,
        export_globe_json,
        validate_globe_payload,
        globe_to_colour_map,
    )
except ImportError:
    pass  # globe_export depends on globe → already warned above

# ── Detail grids ────────────────────────────────────────────────────
from .detail.detail_grid import (
    build_detail_grid,
    deform_grid_to_uv_shape,
    detail_face_count,
    generate_detail_terrain,
)
from .detail.column import (
    build_hex_prism,
    extrude_polygrid_column,
)
from .detail.tile_detail import (
    TileDetailSpec,
    build_all_detail_grids,
    DetailGridCollection,
    find_polygon_corners,
    build_tile_with_neighbours,
)
from .detail.detail_terrain import (
    compute_boundary_elevations,
    classify_detail_faces,
    generate_detail_terrain_bounded,
    generate_all_detail_terrain,
)
from .detail.detail_render import (
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
    # Temperature & Moisture
    "compute_temperature", "generate_temperature_field",
    "LATITUDE_WEIGHT", "LAPSE_RATE",
    "compute_ocean_distance", "generate_moisture_field",
    "OCEAN_PROXIMITY_WEIGHT", "ELEVATION_PENALTY", "MAX_OCEAN_DISTANCE",
    # Terrain Classification
    "classify_tile", "generate_terrain_field", "TERRAIN_TYPES",
    "OCEAN", "SNOW", "TUNDRA", "MOUNTAINS", "DESERT", "WETLAND", "HILLS", "PLAINS",
    # Features
    "detect_coast", "detect_lakes", "place_forests", "generate_features",
    "get_features", "add_feature",
    "COAST", "LAKE", "FOREST", "FEATURE_TYPES",
    # Integration API
    "PlanetParams", "RegionParams", "TileResult", "GenerationResult",
    "generate_planet", "parse_layout",
    "PlanetAtlasResult", "generate_planet_atlas",
    # Globe (available when models library is installed)
    "build_globe_grid", "GlobeGrid",
    "export_globe_payload", "export_globe_json", "validate_globe_payload",
    "globe_to_colour_map",
    # Detail grids
    "build_hex_prism", "extrude_polygrid_column",
    "build_detail_grid", "deform_grid_to_uv_shape",
    "detail_face_count", "generate_detail_terrain",
    "TileDetailSpec", "build_all_detail_grids", "DetailGridCollection",
    "find_polygon_corners", "build_tile_with_neighbours",
    "compute_boundary_elevations", "classify_detail_faces",
    "generate_detail_terrain_bounded", "generate_all_detail_terrain",
    "BiomeConfig", "detail_elevation_to_colour", "render_detail_texture_enhanced",
]
