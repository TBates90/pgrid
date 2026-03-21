# Backward-compat shim — canonical module is polygrid.rendering.tile_uv_align
from .rendering.tile_uv_align import *  # noqa: F401,F403
from .rendering.tile_uv_align import (  # noqa: F401
    _stitch_atlas_seams,
    _build_sector_affines,
    _blend_corner_junctions,
    _find_vertex_junctions,
)
