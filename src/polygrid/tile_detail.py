# Backward-compat shim — canonical module is polygrid.detail.tile_detail
from .detail.tile_detail import *  # noqa: F401,F403
from .detail.tile_detail import _macro_edge_overlap_ok, _find_closest_macro_edge_pair  # noqa: F401
