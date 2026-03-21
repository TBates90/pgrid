# Backward-compat shim — canonical module is polygrid.rendering.uv_texture
from .rendering.uv_texture import *  # noqa: F401,F403
from .rendering.uv_texture import (  # noqa: F401
    _find_polygon_corners,
    _normalize_vec,
    _compute_tile_basis_from_grid,
    _match_tile_to_face,
    _find_shared_edges,
    _HAS_MODELS,
)
