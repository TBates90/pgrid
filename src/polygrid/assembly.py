# Backward-compatibility shim — canonical location: polygrid.building.assembly
from .building.assembly import *  # noqa: F401,F403
from .building.assembly import _position_hex_for_stitch, _snap_hex_hex_boundaries  # noqa: F401 — private, used by tests
