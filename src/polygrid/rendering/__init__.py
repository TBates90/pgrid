"""UV mapping, texture rendering, and globe visualization."""

from .atlas_utils import fill_gutter, compute_atlas_layout
from .column_debug import plot_polygrid_column
from .detail_centers import (
    build_slug_keyed_detail_centers,
    compute_all_detail_centers,
    compute_detail_cell_centers_3d,
)
from .detail_cells_audit import audit_detail_cells_payload
from .detail_topology import (
    DetailCellAddress,
    build_detail_cell_addresses,
    build_detail_ring_positions,
)
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
    "plot_polygrid_column",
    "fill_gutter",
    "compute_atlas_layout",
    "compute_detail_cell_centers_3d",
    "compute_all_detail_centers",
    "build_slug_keyed_detail_centers",
    "audit_detail_cells_payload",
    "DetailCellAddress",
    "build_detail_ring_positions",
    "build_detail_cell_addresses",
    "UVTransform",
    "compute_detail_to_uv_transform",
    "compute_tile_basis",
    "compute_tile_uv_bounds",
    "get_goldberg_tiles",
    "get_tile_uv_vertices",
    "project_and_normalize", "project_point_to_tile_uv",
    "build_polygon_cut_atlas", "compute_pg_to_macro_edge_map",
    "compute_tile_view_limits", "compute_uniform_half_span",
    "get_macro_edge_corners", "match_grid_corners_to_uv",
]
