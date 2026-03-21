"""Enhanced colour ramps and biome rendering for sub-tile detail grids.

This module extends the basic :mod:`terrain_render` colour system with
richer, multi-tonal satellite-style colouring that accounts for
vegetation, rock exposure, snow, and hillshade — producing
high-quality per-tile textures.

Functions
---------
- :class:`BiomeConfig` — per-biome rendering parameters
- :func:`detail_elevation_to_colour` — enhanced per-face colour
- :func:`render_detail_texture_enhanced` — high-quality PNG render
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core.algorithms import get_face_adjacency
from ..core.geometry import face_center
from ..terrain.noise import fbm
from ..core.polygrid import PolyGrid
from ..data.tile_data import TileDataStore


# ═══════════════════════════════════════════════════════════════════
# 10C.1 — BiomeConfig
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BiomeConfig:
    """Per-biome rendering parameters.

    Controls how elevation values are mapped to visual colours,
    accounting for vegetation, rock, snow, and water.

    Parameters
    ----------
    base_ramp : str
        Colour ramp name (``"detail_satellite"``, ``"satellite"``,
        ``"terrain"``).  Default uses the detailed ramp.
    vegetation_density : float
        How much green appears at low/mid elevations (0–1).
    rock_exposure : float
        How much bare rock shows at high elevations (0–1).
    snow_line : float
        Elevation above which snow appears (0–1).
    water_level : float
        Elevation below which water appears (0–1).
    moisture : float
        Affects green vs brown balance at mid-elevations (0–1).
        Higher moisture = more green.
    hillshade_strength : float
        Multiplier for hillshade darkening (0 = off, 1 = full).
    azimuth : float
        Sun direction in degrees (0 = north, 90 = east, 315 = NW).
    altitude : float
        Sun altitude above horizon in degrees.
    """

    base_ramp: str = "detail_satellite"
    vegetation_density: float = 0.6
    rock_exposure: float = 0.4
    snow_line: float = 0.85
    water_level: float = 0.12
    moisture: float = 0.5
    hillshade_strength: float = 0.5
    azimuth: float = 315.0
    altitude: float = 45.0


# ═══════════════════════════════════════════════════════════════════
# 10C.2 — Detailed satellite colour ramp
# ═══════════════════════════════════════════════════════════════════

# 14 control points for smooth gradients
_RAMP_DETAIL_SATELLITE: List[Tuple[float, int, int, int]] = [
    (0.00, 20, 55, 120),    # deep ocean
    (0.04, 30, 70, 145),    # mid ocean
    (0.08, 40, 95, 160),    # shallow water
    (0.12, 55, 135, 100),   # coastal wetland / sand-green
    (0.17, 80, 155, 75),    # lowland green
    (0.25, 110, 175, 55),   # lush vegetation
    (0.35, 140, 175, 50),   # grassland olive
    (0.45, 160, 165, 55),   # dry grass / savanna
    (0.55, 165, 145, 65),   # dry highland
    (0.65, 145, 115, 65),   # mountain brown
    (0.75, 120, 100, 80),   # exposed rock
    (0.85, 155, 155, 165),  # grey scree / snow line start
    (0.93, 210, 215, 220),  # snow field
    (1.00, 245, 248, 252),  # bright snow
]

_RAMPS: Dict[str, List[Tuple[float, int, int, int]]] = {
    "detail_satellite": _RAMP_DETAIL_SATELLITE,
}


def _lerp_ramp(
    ramp: List[Tuple[float, int, int, int]], t: float,
) -> Tuple[float, float, float]:
    """Linearly interpolate a colour ramp at position *t* ∈ [0, 1].

    Returns an ``(R, G, B)`` tuple with components in ``[0, 1]``.
    """
    t = max(0.0, min(1.0, t))
    if not ramp:
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

    return (ramp[-1][1] / 255.0, ramp[-1][2] / 255.0, ramp[-1][3] / 255.0)


# ═══════════════════════════════════════════════════════════════════
# Hillshade for detail grids
# ═══════════════════════════════════════════════════════════════════

def _detail_hillshade(
    grid: PolyGrid,
    store: TileDataStore,
    field_name: str = "elevation",
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> Dict[str, float]:
    """Compute per-face hillshade brightness (0–1) for a detail grid."""
    adj = get_face_adjacency(grid)
    centroids: Dict[str, Optional[Tuple[float, float]]] = {}
    for fid, face in grid.faces.items():
        centroids[fid] = face_center(grid.vertices, face)

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
        own_elev = store.get(fid, field_name)
        neighbours = adj.get(fid, [])
        if not neighbours:
            result[fid] = 0.5
            continue

        dx_sum, dy_sum, count = 0.0, 0.0, 0
        for nid in neighbours:
            nc = centroids.get(nid)
            if nc is None:
                continue
            n_elev = store.get(nid, field_name)
            ddx = nc[0] - c[0]
            ddy = nc[1] - c[1]
            dist = math.sqrt(ddx * ddx + ddy * ddy)
            if dist < 1e-12:
                continue
            slope = (n_elev - own_elev) / dist
            dx_sum += slope * ddx / dist
            dy_sum += slope * ddy / dist
            count += 1

        if count == 0:
            result[fid] = 0.5
            continue

        dx_sum /= count
        dy_sum /= count

        # Surface normal from gradient
        nm = math.sqrt(dx_sum * dx_sum + dy_sum * dy_sum + 1.0)
        nx = -dx_sum / nm
        ny = -dy_sum / nm
        nz = 1.0 / nm

        brightness = max(0.0, min(1.0, nx * lx + ny * ly + nz * lz))
        result[fid] = brightness

    return result


# ═══════════════════════════════════════════════════════════════════
# 10C.3 — Enhanced per-face colour function
# ═══════════════════════════════════════════════════════════════════

def detail_elevation_to_colour(
    elevation: float,
    biome: BiomeConfig,
    *,
    hillshade_val: float = 0.5,
    noise_x: float = 0.0,
    noise_y: float = 0.0,
    noise_seed: int = 0,
) -> Tuple[float, float, float]:
    """Compute an RGB colour for a detail-grid face.

    Combines base ramp colour with vegetation noise, rock exposure,
    snow, and hillshade.

    Parameters
    ----------
    elevation : float
        Normalised elevation in [0, 1].
    biome : BiomeConfig
    hillshade_val : float
        Pre-computed hillshade brightness in [0, 1].
    noise_x, noise_y : float
        Face centroid for vegetation/rock noise sampling.
    noise_seed : int
        Seed for the overlay noise.

    Returns
    -------
    tuple
        ``(r, g, b)`` in ``[0, 1]``.
    """
    ramp = _RAMPS.get(biome.base_ramp, _RAMP_DETAIL_SATELLITE)
    r, g, b = _lerp_ramp(ramp, elevation)

    # ── Water ───────────────────────────────────────────────────
    if elevation < biome.water_level:
        # Pure water colour from the ramp, slightly darkened by depth
        depth_factor = max(0.6, elevation / max(biome.water_level, 0.01))
        r *= depth_factor
        g *= depth_factor
        b *= depth_factor
        # Apply hillshade subtly on water (reflections)
        shade = 0.85 + 0.15 * hillshade_val
        return (
            max(0.0, min(1.0, r * shade)),
            max(0.0, min(1.0, g * shade)),
            max(0.0, min(1.0, b * shade)),
        )

    # ── Vegetation overlay at low/mid elevations ────────────────
    veg_zone = 1.0 - max(0.0, min(1.0,
        (elevation - biome.water_level) / max(biome.snow_line - biome.water_level, 0.01)
    ))
    # Stronger in lower half of elevation range
    veg_zone = veg_zone ** 0.5  # softer falloff

    if biome.vegetation_density > 0 and veg_zone > 0.1:
        veg_noise = fbm(
            noise_x * 12.0, noise_y * 12.0,
            octaves=3, frequency=1.0, seed=noise_seed,
        )
        veg_noise = (veg_noise + 1.0) * 0.5  # remap [-1,1] → [0,1]
        veg_strength = biome.vegetation_density * veg_zone * biome.moisture

        # Green shift
        green_tint = veg_strength * veg_noise * 0.15
        r = r * (1.0 - green_tint * 0.5)
        g = g + green_tint
        b = b * (1.0 - green_tint * 0.3)

    # ── Rock exposure at high elevations ────────────────────────
    rock_zone = max(0.0, min(1.0,
        (elevation - 0.5) / max(biome.snow_line - 0.5, 0.01)
    ))
    if biome.rock_exposure > 0 and rock_zone > 0.1:
        rock_noise = fbm(
            noise_x * 8.0, noise_y * 8.0,
            octaves=2, frequency=2.0, seed=noise_seed + 500,
        )
        rock_noise = (rock_noise + 1.0) * 0.5
        rock_strength = biome.rock_exposure * rock_zone * rock_noise

        # Desaturate toward grey-brown
        grey = 0.3 * r + 0.5 * g + 0.2 * b
        r = r * (1.0 - rock_strength) + grey * rock_strength * 0.9
        g = g * (1.0 - rock_strength) + grey * rock_strength * 0.85
        b = b * (1.0 - rock_strength) + grey * rock_strength * 0.8

    # ── Snow with fractal edge ──────────────────────────────────
    if elevation > biome.snow_line * 0.9:
        snow_noise = fbm(
            noise_x * 6.0, noise_y * 6.0,
            octaves=3, frequency=1.5, seed=noise_seed + 1000,
        )
        # Fractal snow line: actual threshold varies with noise
        snow_threshold = biome.snow_line + snow_noise * 0.05
        if elevation > snow_threshold:
            snow_t = min(1.0, (elevation - snow_threshold) / 0.1)
            snow_r, snow_g, snow_b = 0.94, 0.96, 0.98
            r = r * (1.0 - snow_t) + snow_r * snow_t
            g = g * (1.0 - snow_t) + snow_g * snow_t
            b = b * (1.0 - snow_t) + snow_b * snow_t

    # ── Hillshade ───────────────────────────────────────────────
    if biome.hillshade_strength > 0:
        # Map hillshade 0..1 to a darkening factor
        shade = 0.5 + 0.5 * hillshade_val  # → [0.5, 1.0]
        shade = 1.0 - biome.hillshade_strength * (1.0 - shade)
        r *= shade
        g *= shade
        b *= shade

    return (
        max(0.0, min(1.0, r)),
        max(0.0, min(1.0, g)),
        max(0.0, min(1.0, b)),
    )


# ═══════════════════════════════════════════════════════════════════
# 10C.4 — Enhanced detail texture renderer
# ═══════════════════════════════════════════════════════════════════

def render_detail_texture_enhanced(
    detail_grid: PolyGrid,
    store: TileDataStore,
    output_path: Path | str,
    biome: Optional[BiomeConfig] = None,
    *,
    tile_size: int = 256,
    elevation_field: str = "elevation",
    noise_seed: int = 0,
) -> Path:
    """Render a detail grid to a high-quality PNG tile texture.

    Uses the enhanced colour system with hillshade, vegetation noise,
    rock exposure, and fractal snow lines.

    Parameters
    ----------
    detail_grid : PolyGrid
    store : TileDataStore
    output_path : Path or str
    biome : BiomeConfig, optional
        Uses defaults if not given.
    tile_size : int
        Output image size in pixels (square).
    elevation_field : str
        Name of the elevation field in *store*.
    noise_seed : int
        Seed for overlay noise (vegetation, rock, snow).

    Returns
    -------
    Path
        The output file path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    if biome is None:
        biome = BiomeConfig()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute hillshade
    hs = _detail_hillshade(
        detail_grid, store, elevation_field,
        azimuth=biome.azimuth, altitude=biome.altitude,
    )

    # Build patches
    patches = []
    colours = []
    for fid, face in detail_grid.faces.items():
        verts = []
        for vid in face.vertex_ids:
            v = detail_grid.vertices.get(vid)
            if v is None or not v.has_position():
                break
            verts.append((v.x, v.y))
        else:
            if len(verts) >= 3:
                elev = store.get(fid, elevation_field)
                c = face_center(detail_grid.vertices, face)
                cx, cy = c if c else (0.0, 0.0)
                colour = detail_elevation_to_colour(
                    elev, biome,
                    hillshade_val=hs.get(fid, 0.5),
                    noise_x=cx, noise_y=cy,
                    noise_seed=noise_seed,
                )
                patches.append(MplPolygon(verts, closed=True))
                colours.append(colour)

    if not patches:
        # Empty grid — create a tiny placeholder
        fig, ax = plt.subplots(1, 1, figsize=(1, 1))
        fig.savefig(str(output_path), dpi=tile_size)
        plt.close(fig)
        return output_path

    # Compute the average tile colour for the background.
    # This ensures pixels outside the polygon are terrain-coloured
    # instead of black, eliminating seams in the 3D atlas texture.
    avg_r = sum(c[0] for c in colours) / len(colours)
    avg_g = sum(c[1] for c in colours) / len(colours)
    avg_b = sum(c[2] for c in colours) / len(colours)
    bg_colour = (avg_r, avg_g, avg_b)

    # Render
    dpi = 100
    fig_size = tile_size / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    pc = PatchCollection(patches, facecolors=colours, edgecolors="none", linewidths=0)
    ax.add_collection(pc)
    ax.autoscale_view()

    fig.savefig(
        str(output_path), dpi=dpi, bbox_inches="tight", pad_inches=0,
        facecolor=bg_colour,
    )
    plt.close(fig)

    return output_path
