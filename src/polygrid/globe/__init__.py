"""Globe grid generation and export."""

import warnings as _warnings

try:
    from .globe import build_globe_grid, GlobeGrid, _HAS_MODELS
except ImportError:
    _HAS_MODELS = False
    _warnings.warn(
        "polygrid.globe requires the 'models' library.  "
        "Install with: pip install polygrid[globe]",
        ImportWarning,
        stacklevel=2,
    )

try:
    from .globe_export import (
        export_globe_payload,
        export_globe_json,
        validate_globe_payload,
        globe_to_colour_map,
    )
except ImportError:
    pass

__all__ = [
    "build_globe_grid", "GlobeGrid", "_HAS_MODELS",
    "export_globe_payload", "export_globe_json",
    "validate_globe_payload", "globe_to_colour_map",
]
