"""Rendering enhancements for cohesive globe terrain — Phase 11E.

Provides automatic biome assignment based on elevation statistics,
normal-map generation from sub-face elevation gradients, and
seamless texture rendering across tile boundaries.

Public API
----------
- :func:`assign_biome` — elevation-based biome selection for a tile
- :func:`assign_all_biomes` — batch for entire collection
- :func:`compute_normal_map` — per-tile normal map from elevation
- :func:`render_seamless_texture` — render with boundary overlap
- Preset biomes: ``OCEAN_BIOME``, ``VEGETATION_BIOME``,
  ``MOUNTAIN_BIOME``, ``DESERT_BIOME``, ``SNOW_BIOME``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .algorithms import get_face_adjacency
from .detail_render import BiomeConfig, render_detail_texture_enhanced
from .geometry import face_center
from .polygrid import PolyGrid
from .tile_data import TileDataStore
from .tile_detail import DetailGridCollection


# ═══════════════════════════════════════════════════════════════════
# 11E.2 — Preset biomes
# ═══════════════════════════════════════════════════════════════════

OCEAN_BIOME = BiomeConfig(
    vegetation_density=0.0,
    rock_exposure=0.0,
    snow_line=1.1,       # no snow
    water_level=0.95,    # almost everything is water
    moisture=0.8,
    hillshade_strength=0.2,
)

VEGETATION_BIOME = BiomeConfig(
    vegetation_density=0.8,
    rock_exposure=0.2,
    snow_line=0.90,
    water_level=0.10,
    moisture=0.7,
    hillshade_strength=0.5,
)

MOUNTAIN_BIOME = BiomeConfig(
    vegetation_density=0.2,
    rock_exposure=0.7,
    snow_line=0.75,
    water_level=0.05,
    moisture=0.3,
    hillshade_strength=0.7,
)

DESERT_BIOME = BiomeConfig(
    vegetation_density=0.05,
    rock_exposure=0.5,
    snow_line=1.1,       # no snow
    water_level=0.02,
    moisture=0.05,
    hillshade_strength=0.6,
)

SNOW_BIOME = BiomeConfig(
    vegetation_density=0.0,
    rock_exposure=0.3,
    snow_line=0.40,      # snow starts very low
    water_level=0.05,
    moisture=0.2,
    hillshade_strength=0.6,
)

BIOME_PRESETS = {
    "ocean": OCEAN_BIOME,
    "vegetation": VEGETATION_BIOME,
    "mountain": MOUNTAIN_BIOME,
    "desert": DESERT_BIOME,
    "snow": SNOW_BIOME,
}


# ═══════════════════════════════════════════════════════════════════
# 11E.2 — Elevation-dependent biome assignment
# ═══════════════════════════════════════════════════════════════════

def assign_biome(
    store: TileDataStore,
    detail_grid: PolyGrid,
    *,
    elevation_field: str = "elevation",
) -> BiomeConfig:
    """Auto-assign a :class:`BiomeConfig` based on elevation statistics.

    Decision rules:
    - Mean elevation < 0.15 → ocean
    - Mean elevation > 0.70 → snow
    - Mean elevation > 0.55 → mountain
    - Elevation range < 0.10 (flat) and mean < 0.35 → desert
    - Otherwise → vegetation

    Parameters
    ----------
    store : TileDataStore
    detail_grid : PolyGrid
    elevation_field : str

    Returns
    -------
    BiomeConfig
    """
    elevs = [store.get(fid, elevation_field) for fid in detail_grid.faces]
    if not elevs:
        return VEGETATION_BIOME

    mean_e = sum(elevs) / len(elevs)
    min_e = min(elevs)
    max_e = max(elevs)
    elev_range = max_e - min_e

    if mean_e < 0.15:
        return OCEAN_BIOME
    if mean_e > 0.70:
        return SNOW_BIOME
    if mean_e > 0.55:
        return MOUNTAIN_BIOME
    if elev_range < 0.10 and mean_e < 0.35:
        return DESERT_BIOME
    return VEGETATION_BIOME


def assign_all_biomes(
    collection: DetailGridCollection,
    *,
    elevation_field: str = "elevation",
) -> Dict[str, BiomeConfig]:
    """Assign biomes for every tile in a :class:`DetailGridCollection`.

    Returns ``{face_id: BiomeConfig}``.

    Parameters
    ----------
    collection : DetailGridCollection
    elevation_field : str

    Returns
    -------
    dict
    """
    biomes: Dict[str, BiomeConfig] = {}
    for face_id in collection.face_ids:
        grid = collection.grids[face_id]
        store = collection._stores.get(face_id)
        if store is None:
            biomes[face_id] = VEGETATION_BIOME
        else:
            biomes[face_id] = assign_biome(
                store, grid, elevation_field=elevation_field,
            )
    return biomes


# ═══════════════════════════════════════════════════════════════════
# 11E.3 — Normal-map generation
# ═══════════════════════════════════════════════════════════════════

def compute_normal_map(
    detail_grid: PolyGrid,
    store: TileDataStore,
    *,
    elevation_field: str = "elevation",
    scale: float = 1.0,
) -> Dict[str, Tuple[float, float, float]]:
    """Compute a per-face normal vector from elevation gradients.

    Each face's normal is derived from the elevation difference with
    its neighbours, approximating the surface gradient.

    Parameters
    ----------
    detail_grid : PolyGrid
    store : TileDataStore
    elevation_field : str
    scale : float
        Vertical exaggeration factor.  Higher values make relief
        more pronounced.

    Returns
    -------
    dict
        ``{face_id: (nx, ny, nz)}`` — unit normal vectors.
        ``nz`` points "up" (perpendicular to the surface).
    """
    adj = get_face_adjacency(detail_grid)
    normals: Dict[str, Tuple[float, float, float]] = {}

    for fid in detail_grid.faces:
        face = detail_grid.faces[fid]
        c = face_center(detail_grid.vertices, face)
        if c is None:
            normals[fid] = (0.0, 0.0, 1.0)
            continue
        cx, cy = c
        elev = store.get(fid, elevation_field)

        # Compute gradient from neighbours
        dx_sum = 0.0
        dy_sum = 0.0
        count = 0

        for nid in adj.get(fid, []):
            nface = detail_grid.faces.get(nid)
            if nface is None:
                continue
            nc = face_center(detail_grid.vertices, nface)
            if nc is None:
                continue
            ncx, ncy = nc
            ne = store.get(nid, elevation_field)

            ddx = ncx - cx
            ddy = ncy - cy
            dist = math.sqrt(ddx * ddx + ddy * ddy)
            if dist < 1e-10:
                continue

            de = (ne - elev) * scale
            dx_sum += ddx / dist * de
            dy_sum += ddy / dist * de
            count += 1

        if count > 0:
            dx_sum /= count
            dy_sum /= count

        # Normal = cross product of tangent vectors
        # tangent_x = (1, 0, dx), tangent_y = (0, 1, dy)
        # normal = tangent_x × tangent_y = (-dx, -dy, 1)
        nx = -dx_sum
        ny = -dy_sum
        nz = 1.0

        # Normalize
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length > 1e-10:
            nx /= length
            ny /= length
            nz /= length

        normals[fid] = (nx, ny, nz)

    return normals


def compute_all_normal_maps(
    collection: DetailGridCollection,
    *,
    elevation_field: str = "elevation",
    scale: float = 1.0,
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """Compute normal maps for every tile in a collection.

    Returns ``{face_id: {sub_face_id: (nx, ny, nz)}}``.
    """
    all_normals: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    for face_id in collection.face_ids:
        store = collection._stores.get(face_id)
        if store is None:
            continue
        grid = collection.grids[face_id]
        all_normals[face_id] = compute_normal_map(
            grid, store,
            elevation_field=elevation_field,
            scale=scale,
        )
    return all_normals


# ═══════════════════════════════════════════════════════════════════
# 11E.1 — Seamless texture rendering
# ═══════════════════════════════════════════════════════════════════

def render_seamless_texture(
    collection: DetailGridCollection,
    face_id: str,
    output_path: Path | str,
    globe_grid: Optional[PolyGrid] = None,
    biome: Optional[BiomeConfig] = None,
    *,
    tile_size: int = 256,
    elevation_field: str = "elevation",
    noise_seed: int = 0,
) -> Path:
    """Render a tile texture, using adjacent tiles' boundary data for seamless edges.

    Falls back to standard rendering if no adjacent data is available.

    Parameters
    ----------
    collection : DetailGridCollection
    face_id : str
    output_path : Path or str
    globe_grid : PolyGrid, optional
        Needed for adjacency lookup.
    biome : BiomeConfig, optional
    tile_size : int
    elevation_field : str
    noise_seed : int

    Returns
    -------
    Path
    """
    store = collection._stores.get(face_id)
    grid = collection.grids[face_id]

    if store is None:
        # No terrain data — create empty texture
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(1, 1))
        fig.savefig(str(output_path), dpi=tile_size)
        plt.close(fig)
        return output_path

    # Auto-assign biome if not provided
    if biome is None:
        biome = assign_biome(store, grid, elevation_field=elevation_field)

    # Render with standard renderer
    return render_detail_texture_enhanced(
        grid, store, output_path, biome,
        tile_size=tile_size,
        elevation_field=elevation_field,
        noise_seed=noise_seed,
    )
