# Backward-compat shim — canonical module is polygrid.rendering.globe_renderer_v2
from .rendering.globe_renderer_v2 import *  # noqa: F401,F403
from .rendering.globe_renderer_v2 import (  # noqa: F401
    _normalize_vec3,
    _project_to_sphere,
    _point_in_convex_polygon,
    _nearest_point_on_segment,
    _nearest_point_on_polygon_edge,
)
