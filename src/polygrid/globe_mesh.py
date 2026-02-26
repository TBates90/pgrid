"""Globe mesh bridge — terrain-coloured meshes for the ``models`` renderer.

Converts :func:`globe_to_tile_colours` output into ``models``-compatible
meshes where each Goldberg tile is coloured by its terrain data.

This module is the hand-off between **polygrid** (terrain generation) and
**models** (OpenGL rendering).  It requires the ``models`` library.

Functions
---------
- :func:`terrain_colors_for_layout` — convert colour map → ``Color`` sequence
- :func:`build_terrain_layout_mesh` — single mesh with terrain colours
- :func:`build_terrain_face_meshes` — per-face meshes with terrain colours
- :func:`build_terrain_tile_meshes` — per-tile meshes with model matrices
- :func:`build_terrain_edge_mesh` — wireframe edge mesh
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

try:
    from models.core import Color, ShapeMesh
    from models.objects.goldberg.layout import GoldbergLayout, layout_for_frequency
    from models.objects.goldberg.mesh import (
        DEFAULT_LAYOUT_COLORS,
        FaceMesh,
        TileMeshInstance,
        build_layout_edge_mesh,
        build_layout_face_meshes,
        build_layout_mesh,
        build_layout_tile_meshes,
    )

    _HAS_MODELS = True
except ImportError:  # pragma: no cover
    _HAS_MODELS = False

from .globe import GlobeGrid
from .tile_data import TileDataStore


def _require_models() -> None:
    if not _HAS_MODELS:
        raise ImportError(
            "The 'models' library is required for globe meshes.  "
            "Install it with: pip install models  "
            "or: pip install polygrid[globe]"
        )


# ═══════════════════════════════════════════════════════════════════
# Colour mapping
# ═══════════════════════════════════════════════════════════════════

def terrain_colors_for_layout(
    globe_grid: GlobeGrid,
    colour_map: Dict[str, Tuple[float, float, float]],
    layout: "GoldbergLayout",
    *,
    fallback: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> Tuple["Color", ...]:
    """Convert a polygrid colour map into a ``Color`` sequence for the models layout.

    The returned sequence is aligned with ``layout.polygons`` — one
    ``Color`` per polygon, ordered by ``polygon.index``.

    Parameters
    ----------
    globe_grid : GlobeGrid
        The globe grid that produced the colour map.
    colour_map : dict
        ``{face_id: (r, g, b)}`` as returned by
        :func:`globe_render.globe_to_colour_map`.
    layout : GoldbergLayout
        The models layout for the same frequency.
    fallback : tuple
        RGB colour for tiles not present in the colour map.

    Returns
    -------
    tuple of Color
        One ``Color`` per layout polygon.
    """
    _require_models()

    colors: list["Color"] = []
    for polygon in layout.polygons:
        face_id = f"t{polygon.index}"
        rgb = colour_map.get(face_id, fallback)
        colors.append(Color(rgb[0], rgb[1], rgb[2], 1.0))
    return tuple(colors)


def terrain_colors_from_tile_colours(
    tile_colours: Dict[str, dict],
    layout: "GoldbergLayout",
    *,
    fallback: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> Tuple["Color", ...]:
    """Convert a ``globe_to_tile_colours`` JSON payload into ``Color`` objects.

    Parameters
    ----------
    tile_colours : dict
        ``{face_id: {"color": [r, g, b], ...}}`` as returned by
        :func:`globe_render.globe_to_tile_colours`.
    layout : GoldbergLayout
        The models layout.
    fallback : tuple
        Default colour.

    Returns
    -------
    tuple of Color
    """
    _require_models()

    colors: list["Color"] = []
    for polygon in layout.polygons:
        face_id = f"t{polygon.index}"
        entry = tile_colours.get(face_id, {})
        rgb = entry.get("color", list(fallback))
        colors.append(Color(rgb[0], rgb[1], rgb[2], 1.0))
    return tuple(colors)


# ═══════════════════════════════════════════════════════════════════
# Mesh builders — thin wrappers over models.objects.goldberg.mesh
# ═══════════════════════════════════════════════════════════════════

def build_terrain_layout_mesh(
    globe_grid: GlobeGrid,
    colour_map: Dict[str, Tuple[float, float, float]],
    *,
    radius: float = 1.0,
) -> "ShapeMesh":
    """Build a single triangle-fan mesh with terrain colours.

    This is the most efficient representation for static rendering —
    all tiles in one draw call.

    Parameters
    ----------
    globe_grid : GlobeGrid
    colour_map : dict
        ``{face_id: (r, g, b)}``.
    radius : float
        Mesh radius.

    Returns
    -------
    ShapeMesh
    """
    _require_models()

    layout = layout_for_frequency(globe_grid.frequency)
    colors = terrain_colors_for_layout(globe_grid, colour_map, layout)
    return build_layout_mesh(layout, radius=radius, colors=colors)


def build_terrain_face_meshes(
    globe_grid: GlobeGrid,
    colour_map: Dict[str, Tuple[float, float, float]],
    *,
    radius: float = 1.0,
) -> Tuple["FaceMesh", ...]:
    """Build per-face meshes with terrain colours.

    Useful when individual tiles need to be selected, highlighted,
    or have their colour updated at runtime.

    Parameters
    ----------
    globe_grid : GlobeGrid
    colour_map : dict
    radius : float

    Returns
    -------
    tuple of FaceMesh
    """
    _require_models()

    layout = layout_for_frequency(globe_grid.frequency)
    colors = terrain_colors_for_layout(globe_grid, colour_map, layout)
    return build_layout_face_meshes(layout, radius=radius, colors=colors)


def build_terrain_tile_meshes(
    globe_grid: GlobeGrid,
    colour_map: Dict[str, Tuple[float, float, float]],
    *,
    radius: float = 1.0,
) -> Tuple["TileMeshInstance", ...]:
    """Build per-tile meshes with model matrices and terrain colours.

    Each tile has local-coordinate geometry + a model matrix for
    instanced rendering.

    Parameters
    ----------
    globe_grid : GlobeGrid
    colour_map : dict
    radius : float

    Returns
    -------
    tuple of TileMeshInstance
    """
    _require_models()

    layout = layout_for_frequency(globe_grid.frequency, radius=radius)
    colors = terrain_colors_for_layout(globe_grid, colour_map, layout)
    return build_layout_tile_meshes(layout, radius=radius, colors=colors)


def build_terrain_edge_mesh(
    globe_grid: GlobeGrid,
    *,
    radius: float = 1.0,
    color: Optional["Color"] = None,
    segments: int = 1,
) -> "ShapeMesh":
    """Build an edge wireframe mesh (useful for overlaying on terrain).

    Parameters
    ----------
    globe_grid : GlobeGrid
    radius : float
    color : Color, optional
        Wire colour (default: light grey).
    segments : int
        Sub-segments per edge for smoother arcs.

    Returns
    -------
    ShapeMesh
    """
    _require_models()

    layout = layout_for_frequency(globe_grid.frequency)
    return build_layout_edge_mesh(layout, radius=radius, color=color, segments=segments)
