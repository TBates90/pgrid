"""PolyGrid — topology-first polygon grid toolkit.

Public API is organised into layers:

- **Core** — models, container, algorithms, geometry
- **Building** — grid constructors, Goldberg topology, stitching, assembly
- **Transforms** — overlay model and transform functions
- **Terrain** — noise, heightmaps, mountains, regions
- **Globe** — globe generation, export, rendering (requires *models* library)
- **Detail** — detail grids, tile detail, terrain, rendering, UV alignment
"""

# ── Core ────────────────────────────────────────────────────────────
from .models import Vertex, Edge, Face, MacroEdge
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

# ── Regions ─────────────────────────────────────────────────────────
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
    pass

try:
    from .globe_export import (
        export_globe_payload,
        export_globe_json,
        validate_globe_payload,
        globe_to_colour_map,
    )
except ImportError:
    pass

try:
    from .globe_renderer_v2 import (
        flood_fill_tile_texture,
        flood_fill_atlas,
        subdivide_tile_mesh,
        build_batched_globe_mesh,
        render_globe_v2,
        compute_uv_polygon_inset,
        clamp_uv_to_polygon,
        blend_biome_configs,
        compute_neighbour_average_colours,
        harmonise_tile_colours,
        encode_normal_to_rgb,
        decode_rgb_to_normal,
        build_normal_map_atlas,
        get_pbr_shader_sources,
        classify_water_tiles,
        compute_water_depth,
        DEFAULT_WATER_LEVEL,
        build_atmosphere_shell,
        build_background_quad,
        compute_bloom_threshold,
        get_atmosphere_shader_sources,
        get_background_shader_sources,
        get_bloom_shader_sources,
        ATMOSPHERE_SCALE,
        ATMOSPHERE_COLOR,
        BLOOM_THRESHOLD,
        BLOOM_INTENSITY,
        BG_CENTER_COLOR,
        BG_EDGE_COLOR,
        select_lod_level,
        estimate_tile_screen_fraction,
        is_tile_backfacing,
        stitch_lod_boundary,
        build_lod_batched_globe_mesh,
        LOD_LEVELS,
        LOD_THRESHOLDS,
        BACKFACE_THRESHOLD,
    )
except ImportError:
    pass

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
    NeighbourBorderFace,
    find_polygon_corners,
    get_neighbour_border_faces,
    get_neighbour_border_grid,
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

# ── UV / Atlas ──────────────────────────────────────────────────────
from .atlas_utils import fill_gutter, compute_atlas_layout
from .uv_texture import (
    UVTransform,
    compute_tile_basis,
    get_goldberg_tiles,
    get_tile_uv_vertices,
    project_point_to_tile_uv,
    compute_tile_uv_bounds,
    project_and_normalize,
    compute_detail_to_uv_transform,
)
from .tile_uv_align import (
    compute_polygon_corners_px,
    compute_tile_view_limits,
    mask_to_polygon,
    uv_polygon_px,
    mask_warped_to_uv_polygon,
    compute_grid_to_uv_affine,
    compute_grid_to_px_affine,
    warp_tile_to_uv,
    build_polygon_cut_atlas,
)

__all__ = [
    # Core
    "Vertex", "Edge", "Face", "MacroEdge",
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
    "apply_voronoi", "apply_partition",
    # Tile Data
    "FieldDef", "TileSchema", "TileData", "TileDataStore",
    "save_tile_data", "load_tile_data",
    # Regions
    "Region", "RegionMap", "RegionValidation",
    "partition_angular", "partition_flood_fill", "partition_voronoi", "partition_noise",
    "assign_field", "assign_biome", "regions_to_overlay", "validate_region_map",
    # Terrain
    "fbm", "ridged_noise", "domain_warp", "gradient_mask", "terrace",
    "noise_normalize", "remap", "fbm_3d", "ridged_noise_3d",
    "sample_noise_field", "sample_noise_field_region", "sample_noise_field_3d",
    "smooth_field", "blend_fields", "clamp_field", "normalize_field",
    "MountainConfig", "generate_mountains",
    "MOUNTAIN_RANGE", "ALPINE_PEAKS", "ROLLING_HILLS", "MESA_PLATEAU",
    # Globe
    "build_globe_grid", "GlobeGrid",
    "export_globe_payload", "export_globe_json", "validate_globe_payload",
    "globe_to_colour_map",
    "flood_fill_tile_texture", "flood_fill_atlas",
    "subdivide_tile_mesh", "build_batched_globe_mesh", "render_globe_v2",
    "compute_uv_polygon_inset", "clamp_uv_to_polygon",
    "blend_biome_configs", "compute_neighbour_average_colours", "harmonise_tile_colours",
    "encode_normal_to_rgb", "decode_rgb_to_normal", "build_normal_map_atlas",
    "get_pbr_shader_sources",
    "classify_water_tiles", "compute_water_depth", "DEFAULT_WATER_LEVEL",
    "build_atmosphere_shell", "build_background_quad", "compute_bloom_threshold",
    "get_atmosphere_shader_sources", "get_background_shader_sources",
    "get_bloom_shader_sources",
    "ATMOSPHERE_SCALE", "ATMOSPHERE_COLOR", "BLOOM_THRESHOLD", "BLOOM_INTENSITY",
    "BG_CENTER_COLOR", "BG_EDGE_COLOR",
    "select_lod_level", "estimate_tile_screen_fraction", "is_tile_backfacing",
    "stitch_lod_boundary", "build_lod_batched_globe_mesh",
    "LOD_LEVELS", "LOD_THRESHOLDS", "BACKFACE_THRESHOLD",
    # Detail grids
    "build_detail_grid", "deform_grid_to_uv_shape",
    "detail_face_count", "generate_detail_terrain",
    "TileDetailSpec", "build_all_detail_grids", "DetailGridCollection",
    "NeighbourBorderFace", "find_polygon_corners",
    "get_neighbour_border_faces", "get_neighbour_border_grid",
    "build_tile_with_neighbours",
    "compute_boundary_elevations", "classify_detail_faces",
    "generate_detail_terrain_bounded", "generate_all_detail_terrain",
    "BiomeConfig", "detail_elevation_to_colour", "render_detail_texture_enhanced",
    # UV / Atlas
    "fill_gutter", "compute_atlas_layout",
    "UVTransform", "compute_tile_basis", "get_goldberg_tiles",
    "get_tile_uv_vertices", "project_point_to_tile_uv",
    "compute_tile_uv_bounds", "project_and_normalize",
    "compute_detail_to_uv_transform",
    "compute_polygon_corners_px", "compute_tile_view_limits",
    "mask_to_polygon", "uv_polygon_px", "mask_warped_to_uv_polygon",
    "compute_grid_to_uv_affine", "compute_grid_to_px_affine",
    "warp_tile_to_uv", "build_polygon_cut_atlas",
]
