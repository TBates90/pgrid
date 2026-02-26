"""Elevation-aware terrain rendering — Phase 7D.

Converts per-face elevation data from a :class:`~tile_data.TileDataStore`
into coloured overlays and hillshaded imagery using the existing
:mod:`visualize` overlay system.

Functions
---------
- :func:`elevation_to_overlay` — colour each face by a colour ramp
- :func:`hillshade` — per-face brightness from elevation differences
- :func:`render_terrain` — convenience end-to-end terrain PNG render
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .algorithms import build_face_adjacency, get_face_adjacency
from .geometry import face_center
from .polygrid import PolyGrid
from .tile_data import TileDataStore
from .transforms import Overlay, OverlayRegion


# ═══════════════════════════════════════════════════════════════════
# Colour ramps
# ═══════════════════════════════════════════════════════════════════

# Each ramp is a list of (t, R, G, B) control points (t in [0, 1]).
# Colours are linearly interpolated between them.

_RAMP_TERRAIN: List[Tuple[float, int, int, int]] = [
    (0.00, 30, 80, 160),    # deep water
    (0.10, 50, 120, 190),   # shallow water
    (0.20, 100, 180, 100),  # lowland green
    (0.40, 140, 190, 80),   # grassland
    (0.55, 180, 170, 80),   # dry grassland
    (0.70, 150, 120, 70),   # highland brown
    (0.85, 120, 100, 80),   # rocky grey-brown
    (0.95, 200, 200, 210),  # snow line
    (1.00, 250, 250, 255),  # snow cap
]

_RAMP_GREYSCALE: List[Tuple[float, int, int, int]] = [
    (0.0, 0, 0, 0),
    (1.0, 255, 255, 255),
]

_RAMP_SATELLITE: List[Tuple[float, int, int, int]] = [
    (0.00, 30, 70, 140),    # ocean blue
    (0.08, 40, 100, 160),   # shallow water
    (0.15, 60, 140, 80),    # coastal green
    (0.25, 100, 170, 60),   # lowland yellow-green
    (0.40, 140, 170, 50),   # grassland olive
    (0.55, 160, 150, 60),   # dry highland
    (0.70, 140, 110, 60),   # mountain brown
    (0.82, 110, 95, 75),    # rocky grey
    (0.92, 180, 180, 190),  # snow line
    (1.00, 240, 245, 250),  # snow white
]

_RAMPS: Dict[str, List[Tuple[float, int, int, int]]] = {
    "terrain": _RAMP_TERRAIN,
    "greyscale": _RAMP_GREYSCALE,
    "satellite": _RAMP_SATELLITE,
}


def _lerp_ramp(
    ramp: List[Tuple[float, int, int, int]], t: float,
) -> Tuple[float, float, float]:
    """Linearly interpolate a colour ramp at position *t* ∈ [0, 1].

    Returns an ``(R, G, B)`` tuple with components in ``[0, 1]``.
    """
    t = max(0.0, min(1.0, t))
    if len(ramp) == 0:
        return (0.5, 0.5, 0.5)
    if t <= ramp[0][0]:
        return (ramp[0][1] / 255.0, ramp[0][2] / 255.0, ramp[0][3] / 255.0)
    if t >= ramp[-1][0]:
        return (ramp[-1][1] / 255.0, ramp[-1][2] / 255.0, ramp[-1][3] / 255.0)

    for i in range(len(ramp) - 1):
        t0, r0, g0, b0 = ramp[i]
        t1, r1, g1, b1 = ramp[i + 1]
        if t0 <= t <= t1:
            frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            r = (r0 + (r1 - r0) * frac) / 255.0
            g = (g0 + (g1 - g0) * frac) / 255.0
            b = (b0 + (b1 - b0) * frac) / 255.0
            return (r, g, b)

    # Fallback
    return (ramp[-1][1] / 255.0, ramp[-1][2] / 255.0, ramp[-1][3] / 255.0)


# ═══════════════════════════════════════════════════════════════════
# Hillshade
# ═══════════════════════════════════════════════════════════════════

def hillshade(
    grid: PolyGrid,
    store: TileDataStore,
    field_name: str = "elevation",
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> Dict[str, float]:
    """Compute per-face hillshade brightness from elevation differences.

    Simulates directional sunlight by computing a simplified surface
    normal from the elevation gradient between a face and its neighbours,
    then dotting it with the light direction.

    Parameters
    ----------
    grid : PolyGrid
    store : TileDataStore
    field_name : str
        Name of the elevation field.
    azimuth : float
        Sun direction in degrees (0 = north, 90 = east, 315 = NW).
    altitude : float
        Sun altitude in degrees above the horizon.

    Returns
    -------
    dict
        ``{face_id: brightness}`` where brightness ∈ ``[0, 1]``.
    """
    adj = get_face_adjacency(grid)
    centroids = {
        fid: face_center(grid.vertices, face)
        for fid, face in grid.faces.items()
    }

    # Light direction vector
    az_rad = math.radians(azimuth)
    alt_rad = math.radians(altitude)
    lx = math.cos(alt_rad) * math.sin(az_rad)
    ly = math.cos(alt_rad) * math.cos(az_rad)
    lz = math.sin(alt_rad)

    result: Dict[str, float] = {}

    for fid in grid.faces:
        c = centroids.get(fid)
        if c is None:
            result[fid] = 0.5
            continue

        cx, cy = c
        elev = store.get(fid, field_name)

        # Estimate gradient from neighbours
        dzdx = 0.0
        dzdy = 0.0
        count = 0
        for nid in adj.get(fid, []):
            nc = centroids.get(nid)
            if nc is None:
                continue
            nx_, ny_ = nc
            ne = store.get(nid, field_name)
            dx = nx_ - cx
            dy = ny_ - cy
            dist = math.hypot(dx, dy)
            if dist < 1e-12:
                continue
            de = ne - elev
            # Accumulate gradient components
            dzdx += (de / dist) * (dx / dist)
            dzdy += (de / dist) * (dy / dist)
            count += 1

        if count > 0:
            dzdx /= count
            dzdy /= count

        # Surface normal from gradient: n = (-dzdx, -dzdy, 1) normalized
        nx_n = -dzdx
        ny_n = -dzdy
        nz_n = 1.0
        norm = math.sqrt(nx_n * nx_n + ny_n * ny_n + nz_n * nz_n) or 1.0
        nx_n /= norm
        ny_n /= norm
        nz_n /= norm

        # Dot product with light direction
        shade = max(0.0, nx_n * lx + ny_n * ly + nz_n * lz)
        result[fid] = shade

    return result


# ═══════════════════════════════════════════════════════════════════
# Elevation → Overlay
# ═══════════════════════════════════════════════════════════════════

def elevation_to_overlay(
    grid: PolyGrid,
    store: TileDataStore,
    field_name: str = "elevation",
    *,
    ramp: str = "terrain",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    shade: Optional[Dict[str, float]] = None,
    shade_strength: float = 0.5,
) -> Overlay:
    """Convert an elevation TileData field into a coloured :class:`Overlay`.

    Each face becomes an :class:`OverlayRegion` whose metadata stores
    an RGB colour tuple derived from a colour ramp, optionally modulated
    by hillshade brightness.

    Parameters
    ----------
    grid : PolyGrid
    store : TileDataStore
    field_name : str
        Elevation field name.
    ramp : str
        Colour ramp name: ``"terrain"``, ``"greyscale"``, or ``"satellite"``.
    vmin, vmax : float, optional
        Value range for colour mapping.  If *None*, auto-detected from
        the data.
    shade : dict, optional
        Per-face hillshade from :func:`hillshade`.  If given, modulates
        the base colour to simulate lighting.
    shade_strength : float
        How much hillshade affects the colour (0 = none, 1 = full).

    Returns
    -------
    Overlay
        An overlay with ``kind="terrain"`` and per-face coloured regions.
        Each region's metadata includes ``"color"`` as an ``(R, G, B)``
        tuple with components in ``[0, 1]``.
    """
    ramp_data = _RAMPS.get(ramp)
    if ramp_data is None:
        raise ValueError(f"Unknown ramp: {ramp!r}. Available: {list(_RAMPS.keys())}")

    # Determine value range
    all_vals = [store.get(fid, field_name) for fid in grid.faces]
    if vmin is None:
        vmin = min(all_vals) if all_vals else 0.0
    if vmax is None:
        vmax = max(all_vals) if all_vals else 1.0

    overlay = Overlay(kind="terrain")
    overlay.metadata["ramp"] = ramp
    overlay.metadata["vmin"] = vmin
    overlay.metadata["vmax"] = vmax

    for face in grid.faces.values():
        pts: List[Tuple[float, float]] = []
        for vid in face.vertex_ids:
            v = grid.vertices[vid]
            if v.has_position():
                pts.append((v.x, v.y))
        if len(pts) < 3:
            continue

        elev = store.get(face.id, field_name)
        t = (elev - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        t = max(0.0, min(1.0, t))

        r, g, b = _lerp_ramp(ramp_data, t)

        # Apply hillshade
        if shade is not None and face.id in shade:
            s = shade[face.id]
            # Blend: colour * (1 − strength + strength × shade)
            factor = 1.0 - shade_strength + shade_strength * s
            r = max(0.0, min(1.0, r * factor))
            g = max(0.0, min(1.0, g * factor))
            b = max(0.0, min(1.0, b * factor))

        region = OverlayRegion(
            id=f"ter_{face.id}",
            points=pts,
            source_vertex_id=face.id,
        )
        region.id = f"ter_{face.id}"
        # Store colour in overlay metadata keyed by region id
        overlay.metadata[f"color_{face.id}"] = (r, g, b)
        overlay.regions.append(region)

    return overlay


# ═══════════════════════════════════════════════════════════════════
# Render terrain (end-to-end convenience)
# ═══════════════════════════════════════════════════════════════════

def render_terrain(
    grid: PolyGrid,
    store: TileDataStore,
    output_path: str,
    *,
    field_name: str = "elevation",
    ramp: str = "satellite",
    hillshade_enabled: bool = True,
    hillshade_azimuth: float = 315.0,
    hillshade_altitude: float = 45.0,
    shade_strength: float = 0.5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 200,
    title: Optional[str] = None,
) -> None:
    """Render elevation data to a terrain-coloured PNG.

    This is a convenience function that:
    1. Computes hillshade (optional)
    2. Builds a terrain overlay
    3. Draws it using matplotlib

    Parameters
    ----------
    grid : PolyGrid
    store : TileDataStore
    output_path : str
        Path for the output PNG.
    field_name : str
        Elevation field name.
    ramp : str
        Colour ramp name.
    hillshade_enabled : bool
        Whether to compute and apply hillshade.
    hillshade_azimuth, hillshade_altitude : float
        Sun direction for hillshade.
    shade_strength : float
        Hillshade modulation strength.
    vmin, vmax : float, optional
        Manual elevation range for colour mapping.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Output resolution.
    title : str, optional
        Plot title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    shade = None
    if hillshade_enabled:
        shade = hillshade(grid, store, field_name, azimuth=hillshade_azimuth, altitude=hillshade_altitude)

    overlay = elevation_to_overlay(
        grid, store, field_name,
        ramp=ramp, vmin=vmin, vmax=vmax,
        shade=shade, shade_strength=shade_strength,
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    for region in overlay.regions:
        if len(region.points) < 3:
            continue
        fid = region.source_vertex_id
        color = overlay.metadata.get(f"color_{fid}", (0.5, 0.5, 0.5))
        poly = Polygon(
            region.points,
            closed=True,
            facecolor=color,
            edgecolor=color,  # match face to hide seams
            linewidth=0.3,
            zorder=1,
        )
        ax.add_patch(poly)

    ax.autoscale_view()
    if title:
        ax.set_title(title, fontsize=14)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
