"""Legacy single-grid renderer — **deprecated**.

All rendering is now in :mod:`polygrid.visualize`.  This module
re-exports :func:`render_png` so that existing imports continue to
work, but emits a :class:`DeprecationWarning` on first import.
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "polygrid.render is deprecated — use polygrid.visualize.render_png instead",
    DeprecationWarning,
    stacklevel=2,
)

from .visualize import render_png  # noqa: E402, F401

__all__ = ["render_png"]
