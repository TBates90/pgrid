"""UV mapping, texture rendering, and globe visualization."""

from .atlas_utils import fill_gutter, compute_atlas_layout
from .uv_texture import (
    UVTransform,
    compute_detail_to_uv_transform,
    compute_tile_basis,
    compute_tile_uv_bounds,
    get_goldberg_tiles,
    get_tile_uv_vertices,
    project_and_normalize,
    project_point_to_tile_uv,
)
from .tile_uv_align import (
    build_polygon_cut_atlas,
    compute_pg_to_macro_edge_map,
    compute_tile_view_limits,
    compute_uniform_half_span,
    get_macro_edge_corners,
    match_grid_corners_to_uv,
)

__all__ = [
    "fill_gutter", "compute_atlas_layout",
    "UVTransform", "compute_detail_to_uv_transform", "compute_tile_basis",
    "compute_tile_uv_bounds", "get_goldberg_tiles", "get_tile_uv_vertices",
    "project_and_normalize", "project_point_to_tile_uv",
    "build_polygon_cut_atlas", "compute_pg_to_macro_edge_map",
    "compute_tile_view_limits", "compute_uniform_half_span",
    "get_macro_edge_corners", "match_grid_corners_to_uv",
]
