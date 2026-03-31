"""Detail grid construction, terrain, and rendering for globe tiles."""

from .detail_grid import (
    build_detail_grid,
    deform_grid_to_uv_shape,
    detail_face_count,
    generate_detail_terrain,
)
from .column import (
    build_hex_prism,
    extrude_polygrid_column,
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
    compute_neighbor_edge_mapping,
    generate_detail_terrain_bounded,
    generate_all_detail_terrain,
)
from .detail_render import (
    BiomeConfig,
    detail_elevation_to_colour,
    render_detail_texture_enhanced,
    _detail_hillshade,
)

__all__ = [
    "build_detail_grid", "deform_grid_to_uv_shape", "detail_face_count",
    "build_hex_prism", "extrude_polygrid_column",
    "generate_detail_terrain",
    "TileDetailSpec", "build_all_detail_grids", "DetailGridCollection",
    "find_polygon_corners", "build_tile_with_neighbours",
    "compute_boundary_elevations", "classify_detail_faces",
    "compute_neighbor_edge_mapping",
    "generate_detail_terrain_bounded", "generate_all_detail_terrain",
    "BiomeConfig", "detail_elevation_to_colour",
    "render_detail_texture_enhanced", "_detail_hillshade",
]
