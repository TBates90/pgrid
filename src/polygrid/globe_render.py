"""Globe rendering — 2-D flat map and 3-D matplotlib views of a GlobeGrid.

Functions
---------
- :func:`globe_to_colour_map` — map elevation/terrain data to per-tile RGB
- :func:`render_globe_flat` — equirectangular 2-D projection (PNG)
- :func:`render_globe_3d` — matplotlib Poly3DCollection view (PNG)
- :func:`globe_to_tile_colours` — export colour map as JSON-ready dict
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from .globe import GlobeGrid

# ═══════════════════════════════════════════════════════════════════
# Colour ramps
# ═══════════════════════════════════════════════════════════════════

def _lerp_colour(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    t: float,
) -> Tuple[float, float, float]:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def _ramp_satellite(elevation: float) -> Tuple[float, float, float]:
    """Earth-like colour ramp: deep blue → green → brown → snow."""
    if elevation < 0.15:
        return _lerp_colour((0.05, 0.10, 0.40), (0.10, 0.25, 0.55), elevation / 0.15)
    elif elevation < 0.35:
        t = (elevation - 0.15) / 0.20
        return _lerp_colour((0.10, 0.40, 0.15), (0.30, 0.55, 0.20), t)
    elif elevation < 0.60:
        t = (elevation - 0.35) / 0.25
        return _lerp_colour((0.45, 0.40, 0.25), (0.55, 0.45, 0.30), t)
    elif elevation < 0.85:
        t = (elevation - 0.60) / 0.25
        return _lerp_colour((0.55, 0.45, 0.30), (0.70, 0.65, 0.60), t)
    else:
        t = (elevation - 0.85) / 0.15
        return _lerp_colour((0.80, 0.80, 0.80), (1.0, 1.0, 1.0), min(t, 1.0))


def _ramp_topo(elevation: float) -> Tuple[float, float, float]:
    """Topographic colour ramp: blue → cyan → green → yellow → red → white."""
    stops = [
        (0.00, (0.0, 0.0, 0.5)),
        (0.20, (0.0, 0.4, 0.8)),
        (0.35, (0.1, 0.6, 0.3)),
        (0.50, (0.5, 0.7, 0.2)),
        (0.65, (0.8, 0.7, 0.1)),
        (0.80, (0.8, 0.3, 0.1)),
        (1.00, (1.0, 1.0, 1.0)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if elevation <= t1:
            t = (elevation - t0) / (t1 - t0) if t1 > t0 else 0.0
            return _lerp_colour(c0, c1, t)
    return stops[-1][1]


_RAMPS: Dict[str, Callable[[float], Tuple[float, float, float]]] = {
    "satellite": _ramp_satellite,
    "topo": _ramp_topo,
}


# ═══════════════════════════════════════════════════════════════════
# Colour map
# ═══════════════════════════════════════════════════════════════════

def globe_to_colour_map(
    globe_grid: GlobeGrid,
    store: "TileDataStore",
    *,
    field_name: str = "elevation",
    ramp: str = "satellite",
) -> Dict[str, Tuple[float, float, float]]:
    """Map per-tile data to RGB colours.

    Returns ``{face_id: (r, g, b)}`` where each component is in ``[0, 1]``.
    """
    ramp_fn = _RAMPS.get(ramp, _ramp_satellite)
    colours: Dict[str, Tuple[float, float, float]] = {}
    for fid in globe_grid.faces:
        val = store.get(fid, field_name)
        val = max(0.0, min(1.0, float(val)))
        colours[fid] = ramp_fn(val)
    return colours


def globe_to_tile_colours(
    globe_grid: GlobeGrid,
    store: "TileDataStore",
    *,
    field_name: str = "elevation",
    ramp: str = "satellite",
) -> Dict[str, dict]:
    """Export per-tile colours as a JSON-serialisable dict.

    Returns ``{face_id: {"color": [r, g, b], "elevation": float, ...}}``.
    """
    colour_map = globe_to_colour_map(globe_grid, store, field_name=field_name, ramp=ramp)
    payload: Dict[str, dict] = {}
    for fid, (r, g, b) in colour_map.items():
        entry: dict = {
            "color": [round(r, 4), round(g, 4), round(b, 4)],
            "elevation": round(float(store.get(fid, field_name)), 6),
        }
        ll = globe_grid.tile_lat_lon(fid)
        if ll:
            entry["latitude_deg"] = round(ll[0], 4)
            entry["longitude_deg"] = round(ll[1], 4)
        mid = globe_grid.tile_models_id(fid)
        if mid is not None:
            entry["models_tile_id"] = mid
        payload[fid] = entry
    return payload


# ═══════════════════════════════════════════════════════════════════
# 2-D flat render (equirectangular projection)
# ═══════════════════════════════════════════════════════════════════

def render_globe_flat(
    globe_grid: GlobeGrid,
    store: "TileDataStore",
    out_path: Union[str, Path],
    *,
    field_name: str = "elevation",
    ramp: str = "satellite",
    figsize: Tuple[float, float] = (14, 7),
    dpi: int = 150,
) -> Path:
    """Render an equirectangular flat projection of the globe.

    Each tile is plotted as a filled polygon at its lat/lon position.
    Returns the output file path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import RegularPolygon
    from matplotlib.collections import PatchCollection
    import matplotlib.patches as mpatches

    colour_map = globe_to_colour_map(globe_grid, store, field_name=field_name, ramp=ramp)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.set_facecolor((0.05, 0.05, 0.15))

    for fid, face in globe_grid.faces.items():
        ll = globe_grid.tile_lat_lon(fid)
        if ll is None:
            continue
        lat, lon = ll
        colour = colour_map.get(fid, (0.5, 0.5, 0.5))
        n_sides = 5 if face.face_type == "pent" else 6
        # Approximate tile as a regular polygon in lon/lat space
        radius = 360.0 / (10.0 * globe_grid.frequency) * 0.6
        patch = RegularPolygon(
            (lon, lat), n_sides, radius=radius,
            facecolor=colour, edgecolor=(0.2, 0.2, 0.2, 0.3),
            linewidth=0.5,
        )
        ax.add_patch(patch)

    ax.set_xlim(-10, 370)
    ax.set_ylim(-100, 100)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(f"Globe Grid — frequency {globe_grid.frequency} ({len(globe_grid.faces)} tiles)")
    ax.set_aspect("equal")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════
# 3-D render (matplotlib Poly3DCollection)
# ═══════════════════════════════════════════════════════════════════

def render_globe_3d(
    globe_grid: GlobeGrid,
    store: "TileDataStore",
    out_path: Union[str, Path],
    *,
    field_name: str = "elevation",
    ramp: str = "satellite",
    figsize: Tuple[float, float] = (10, 10),
    dpi: int = 150,
    elev: float = 20.0,
    azim: float = -60.0,
) -> Path:
    """Render the Goldberg polyhedron in 3-D with terrain colours.

    Uses matplotlib's ``Poly3DCollection`` — no OpenGL required.
    Returns the output file path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    colour_map = globe_to_colour_map(globe_grid, store, field_name=field_name, ramp=ramp)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    polygons: list = []
    colours: list = []

    for fid, face in globe_grid.faces.items():
        # Get 3D vertices for this face
        verts_3d = []
        for vid in face.vertex_ids:
            v = globe_grid.vertices[vid]
            if v.has_position_3d():
                verts_3d.append((v.x, v.y, v.z))
        if len(verts_3d) >= 3:
            polygons.append(verts_3d)
            colours.append(colour_map.get(fid, (0.5, 0.5, 0.5)))

    collection = Poly3DCollection(
        polygons,
        facecolors=colours,
        edgecolors=[(0.15, 0.15, 0.15, 0.4)] * len(polygons),
        linewidths=0.3,
    )
    ax.add_collection3d(collection)

    # Set axis limits based on radius
    r = globe_grid.radius * 1.2
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(
        f"Goldberg Polyhedron — freq {globe_grid.frequency} "
        f"({len(globe_grid.faces)} tiles)",
        fontsize=12,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
