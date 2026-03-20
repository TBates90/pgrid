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
        render_textured_globe_opengl,
    )
except ImportError:
    pass  # models library not installed — globe renderer features unavailable

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
    pass  # models library not installed — v2 renderer features unavailable

# ── Detail grids ────────────────────────────────────────────────────
from .detail_grid import (
    build_detail_grid,
    deform_grid_to_uv_shape,
    detail_face_count,
    generate_detail_terrain,
    render_detail_texture,
    build_texture_atlas,
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
from .apron_grid import (
    ApronResult,
    EdgeSubfaceMapping,
    build_all_apron_grids,
    build_apron_grid,
    boundary_subface_ids,
    classify_boundary_subfaces,
    compute_edge_subface_mapping,
    propagate_apron_terrain,
)
from .apron_texture import (
    render_detail_texture_apron,
    build_apron_atlas,
    build_apron_feature_atlas,
)
from .detail_terrain_3d import (
    Terrain3DSpec,
    compute_subface_3d_position,
    precompute_3d_positions,
    precompute_all_3d_positions,
    generate_detail_terrain_3d,
    generate_all_detail_terrain_3d,
)
from .terrain_patches import (
    TerrainPatch,
    TerrainDistribution,
    TERRAIN_PRESETS,
    EARTHLIKE,
    MOUNTAINOUS,
    ARCHIPELAGO,
    PANGAEA,
    FOREST_WORLD,
    DEEP_FOREST,
    OCEAN_WORLD,
    generate_terrain_patches,
    apply_terrain_patches,
    generate_patched_terrain,
)
from .biome_scatter import (
    FeatureInstance,
    poisson_disk_sample,
    scatter_features_on_tile,
    scatter_features_fullslot,
    compute_density_field,
    collect_margin_features,
)
from .biome_render import (
    ForestFeatureConfig,
    FOREST_PRESETS,
    TEMPERATE_FOREST,
    TROPICAL_FOREST,
    BOREAL_FOREST,
    SPARSE_WOODLAND,
    render_canopy,
    render_undergrowth,
    render_forest_tile,
    render_forest_on_ground,
    render_forest_on_ground_fullslot,
)
from .biome_pipeline import (
    BiomeRenderer,
    ForestRenderer,
    OceanRenderer,
    identify_forest_tiles,
    build_feature_atlas,
)
from .biome_continuity import (
    build_biome_density_map,
    get_tile_margin_features,
    compute_biome_transition_mask,
    stitch_feature_boundary,
)
from .ocean_render import (
    OceanFeatureConfig,
    TROPICAL_OCEAN,
    TEMPERATE_OCEAN,
    ARCTIC_OCEAN,
    DEEP_OCEAN,
    OCEAN_PRESETS,
    identify_ocean_tiles,
    compute_ocean_depth_map,
    compute_coast_direction,
    render_ocean_depth_gradient,
    render_wave_pattern,
    render_coastal_features,
    render_deep_ocean_features,
    render_ocean_tile,
)
from .biome_topology import (
    SubfaceTree,
    SubfaceOceanProps,
    scatter_trees_on_grid,
    render_topology_forest,
    compute_subface_ocean_depth,
    identify_coastal_subfaces,
    compute_ocean_subface_props,
    render_topology_ocean,
    render_hybrid_biome,
    TopologyForestRenderer,
    TopologyOceanRenderer,
)
from .region_stitch import (
    FaceMapping,
    stitch_detail_grids,
    generate_terrain_on_stitched,
    split_terrain_to_tiles,
    generate_stitched_patch_terrain,
)
from .globe_terrain import (
    MountainConfig3D,
    GLOBE_MOUNTAIN_RANGE,
    GLOBE_VOLCANIC_CHAIN,
    GLOBE_CONTINENTAL_DIVIDE,
    MOUNTAIN_3D_PRESETS,
    generate_mountains_3d,
    generate_rivers_on_stitched,
    ErosionConfig,
    erode_terrain,
)
from .detail_render import (
    BiomeConfig,
    detail_elevation_to_colour,
    render_detail_texture_enhanced,
)
from .render_enhanced import (
    OCEAN_BIOME,
    VEGETATION_BIOME,
    MOUNTAIN_BIOME,
    DESERT_BIOME,
    SNOW_BIOME,
    BIOME_PRESETS,
    assign_biome,
    assign_all_biomes,
    compute_normal_map,
    compute_all_normal_maps,
    render_seamless_texture,
)

try:
    from .texture_pipeline import (
        build_detail_atlas,
        compute_tile_uvs,
        build_textured_tile_mesh,
        build_textured_globe_meshes,
    )
except ImportError:
    pass  # models library not installed — texture pipeline unavailable

# ── Performance (Phase 10F) ─────────────────────────────────────────
from .detail_perf import (
    generate_all_detail_terrain_parallel,
    render_detail_texture_fast,
    build_detail_atlas_fast,
    build_detail_atlas_fullslot,
    DetailCache,
    generate_all_detail_terrain_cached,
    benchmark_pipeline,
)

# ── Tile texture — full-slot rendering (Phase 16A) ──────────────────
from .tile_texture import (
    build_face_lookup,
    interpolate_at_pixel,
    render_detail_texture_fullslot,
    compute_tile_blend_mask,
    apply_blend_mask_to_atlas,
    # Phase 16D — hex shape softening
    jitter_polygon_vertices,
    apply_noise_overlay,
    apply_colour_dithering,
)

# ── Texture export (Phase 18D) ──────────────────────────────────────
from .texture_export import (
    next_power_of_two,
    atlas_pot_size,
    resize_atlas_pot,
    compute_mip_levels,
    generate_atlas_mipmaps,
    export_atlas_ktx2,
    validate_ktx2_header,
    build_orm_atlas,
    build_material_set,
    export_globe_gltf,
)

# ── Visual cohesion validation (Phase 18E) ──────────────────────────
from .visual_cohesion import (
    sample_boundary_pixels,
    sample_interior_pixels,
    measure_seam_visibility,
    verify_topology_features,
    run_full_pipeline,
    benchmark_apron_pipeline,
)
from .coastline import (
    CoastlineConfig,
    GENTLE_COAST,
    RUGGED_COAST,
    ARCHIPELAGO_COAST,
    COASTLINE_PRESETS,
    TileBiomeContext,
    classify_tile_biome_context,
    classify_all_tiles,
    compute_edge_direction,
    compute_coastline_mask,
    CoastlineMask,
    build_coastline_mask,
    blend_biome_images,
    render_coastal_strip,
)
from .uv_texture import (
    UVTransform,
    compute_tile_basis,
    get_goldberg_tiles,
    get_tile_uv_vertices,
    project_point_to_tile_uv,
    compute_tile_uv_bounds,
    project_and_normalize,
    compute_detail_to_uv_transform,
    render_tile_uv_aligned,
    render_tile_uv_aligned_full,
    build_uv_aligned_atlas,
)

# ── Polygon-Cut UV Alignment (Phase 21) ─────────────────────────────
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
    "validate_globe_payload",
    # Globe renderer (optional — requires models library)
    "build_coloured_globe_mesh",
    "build_coloured_globe_mesh_from_export",
    "build_edge_mesh_for_frequency",
    "prepare_terrain_scene",
    "render_terrain_globe_opengl",
    "render_textured_globe_opengl",
    # Globe renderer v2 (Phase 12)
    "flood_fill_tile_texture",
    "flood_fill_atlas",
    "subdivide_tile_mesh",
    "build_batched_globe_mesh",
    "render_globe_v2",
    # UV clamping (Phase 13C)
    "compute_uv_polygon_inset",
    "clamp_uv_to_polygon",
    # Colour harmonisation (Phase 13D)
    "blend_biome_configs",
    "compute_neighbour_average_colours",
    "harmonise_tile_colours",
    # Normal-mapped lighting (Phase 13E)
    "encode_normal_to_rgb",
    "decode_rgb_to_normal",
    "build_normal_map_atlas",
    "get_pbr_shader_sources",
    # Detail grids
    "build_detail_grid",
    "deform_grid_to_uv_shape",
    "detail_face_count",
    "generate_detail_terrain",
    "render_detail_texture",
    "build_texture_atlas",
    # Tile detail (Phase 10A)
    "TileDetailSpec",
    "build_all_detail_grids",
    "DetailGridCollection",
    "NeighbourBorderFace",
    "find_polygon_corners",
    "get_neighbour_border_faces",
    "get_neighbour_border_grid",
    "build_tile_with_neighbours",
    # Detail terrain (Phase 10B)
    "compute_boundary_elevations",
    "classify_detail_faces",
    "generate_detail_terrain_bounded",
    "generate_all_detail_terrain",
    # Apron grid (Phase 18A)
    "ApronResult",
    "EdgeSubfaceMapping",
    "build_all_apron_grids",
    "build_apron_grid",
    "boundary_subface_ids",
    "classify_boundary_subfaces",
    "compute_edge_subface_mapping",
    "propagate_apron_terrain",
    # Apron texture (Phase 18B)
    "render_detail_texture_apron",
    "build_apron_atlas",
    "build_apron_feature_atlas",
    # Detail render (Phase 10C)
    "BiomeConfig",
    "detail_elevation_to_colour",
    "render_detail_texture_enhanced",
    # Detail terrain 3D (Phase 11A)
    "Terrain3DSpec",
    "compute_subface_3d_position",
    "precompute_3d_positions",
    "precompute_all_3d_positions",
    "generate_detail_terrain_3d",
    "generate_all_detail_terrain_3d",
    # Terrain patches (Phase 11B)
    "TerrainPatch",
    "TerrainDistribution",
    "TERRAIN_PRESETS",
    "EARTHLIKE",
    "MOUNTAINOUS",
    "ARCHIPELAGO",
    "PANGAEA",
    "FOREST_WORLD",
    "DEEP_FOREST",
    "OCEAN_WORLD",
    "generate_terrain_patches",
    "apply_terrain_patches",
    "generate_patched_terrain",
    # Biome scatter (Phase 14A)
    "FeatureInstance",
    "poisson_disk_sample",
    "scatter_features_on_tile",
    "scatter_features_fullslot",
    "compute_density_field",
    "collect_margin_features",
    # Biome render (Phase 14B)
    "ForestFeatureConfig",
    "FOREST_PRESETS",
    "TEMPERATE_FOREST",
    "TROPICAL_FOREST",
    "BOREAL_FOREST",
    "SPARSE_WOODLAND",
    "render_canopy",
    "render_undergrowth",
    "render_forest_tile",
    "render_forest_on_ground",
    "render_forest_on_ground_fullslot",
    # Biome continuity (Phase 14C)
    "build_biome_density_map",
    "get_tile_margin_features",
    "compute_biome_transition_mask",
    "stitch_feature_boundary",
    # Ocean biome (Phase 17A)
    "OceanFeatureConfig",
    "TROPICAL_OCEAN",
    "TEMPERATE_OCEAN",
    "ARCTIC_OCEAN",
    "DEEP_OCEAN",
    "OCEAN_PRESETS",
    "identify_ocean_tiles",
    "compute_ocean_depth_map",
    "compute_coast_direction",
    "render_ocean_depth_gradient",
    "render_wave_pattern",
    "render_coastal_features",
    "render_deep_ocean_features",
    "render_ocean_tile",
    # Biome topology (Phase 18C)
    "SubfaceTree",
    "SubfaceOceanProps",
    "scatter_trees_on_grid",
    "render_topology_forest",
    "compute_subface_ocean_depth",
    "identify_coastal_subfaces",
    "compute_ocean_subface_props",
    "render_topology_ocean",
    "render_hybrid_biome",
    "TopologyForestRenderer",
    "TopologyOceanRenderer",
    # Biome pipeline (Phase 14D)
    "BiomeRenderer",
    "ForestRenderer",
    "OceanRenderer",
    "identify_forest_tiles",
    "build_feature_atlas",
    # Region stitch (Phase 11C)
    "FaceMapping",
    "stitch_detail_grids",
    "generate_terrain_on_stitched",
    "split_terrain_to_tiles",
    "generate_stitched_patch_terrain",
    # Globe terrain enhancements (Phase 11D)
    "MountainConfig3D",
    "GLOBE_MOUNTAIN_RANGE",
    "GLOBE_VOLCANIC_CHAIN",
    "GLOBE_CONTINENTAL_DIVIDE",
    "MOUNTAIN_3D_PRESETS",
    "generate_mountains_3d",
    "generate_rivers_on_stitched",
    "ErosionConfig",
    "erode_terrain",
    # Texture pipeline (Phase 10D)
    "build_detail_atlas",
    "compute_tile_uvs",
    "build_textured_tile_mesh",
    "build_textured_globe_meshes",
    # Performance (Phase 10F)
    "generate_all_detail_terrain_parallel",
    "render_detail_texture_fast",
    "build_detail_atlas_fast",
    "build_detail_atlas_fullslot",
    "DetailCache",
    "generate_all_detail_terrain_cached",
    "benchmark_pipeline",
    # Tile texture — full-slot rendering (Phase 16A/B)
    "build_face_lookup",
    "interpolate_at_pixel",
    "render_detail_texture_fullslot",
    "compute_tile_blend_mask",
    "apply_blend_mask_to_atlas",
    # Hex shape softening (Phase 16D)
    "jitter_polygon_vertices",
    "apply_noise_overlay",
    "apply_colour_dithering",
    # Texture export (Phase 18D)
    "next_power_of_two",
    "atlas_pot_size",
    "resize_atlas_pot",
    "compute_mip_levels",
    "generate_atlas_mipmaps",
    "export_atlas_ktx2",
    "validate_ktx2_header",
    "build_orm_atlas",
    "build_material_set",
    "export_globe_gltf",
    # Visual cohesion (Phase 18E)
    "sample_boundary_pixels",
    "sample_interior_pixels",
    "measure_seam_visibility",
    "verify_topology_features",
    "run_full_pipeline",
    "benchmark_apron_pipeline",
    # Coastline transitions (Phase 19)
    "CoastlineConfig",
    "GENTLE_COAST",
    "RUGGED_COAST",
    "ARCHIPELAGO_COAST",
    "COASTLINE_PRESETS",
    "TileBiomeContext",
    "classify_tile_biome_context",
    "classify_all_tiles",
    "compute_edge_direction",
    "compute_coastline_mask",
    "CoastlineMask",
    "build_coastline_mask",
    "blend_biome_images",
    "render_coastal_strip",
    # Render enhanced (Phase 11E)
    "OCEAN_BIOME",
    "VEGETATION_BIOME",
    "MOUNTAIN_BIOME",
    "DESERT_BIOME",
    "SNOW_BIOME",
    "BIOME_PRESETS",
    "assign_biome",
    "assign_all_biomes",
    "compute_normal_map",
    "compute_all_normal_maps",
    "render_seamless_texture",
]
