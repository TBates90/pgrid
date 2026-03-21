# Backward-compat shim — canonical module is polygrid.detail.detail_render
from .detail.detail_render import *  # noqa: F401,F403
from .detail.detail_render import _detail_hillshade, _lerp_ramp, _RAMP_DETAIL_SATELLITE, _RAMPS  # noqa: F401
