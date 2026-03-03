"""Ocean biome rendering — depth gradients, wave textures, coastal features.

Phase 17 renders ocean tiles with rich depth-based colour gradients,
surface wave patterns, coastal detail (foam, shallow sand, reefs),
and deep-ocean variation — all baked into the tile texture atlas.

Modules
-------
- **17A** — ``OceanFeatureConfig`` (presets) + ``compute_ocean_depth_map()``
- **17B** — Pixel-level ocean rendering functions
- **17C** — ``OceanRenderer`` (``BiomeRenderer`` implementation)

Functions
---------
- :func:`compute_ocean_depth_map` — BFS-based depth map for ocean tiles
- :func:`render_ocean_depth_gradient` — depth-based colour fill
- :func:`render_wave_pattern` — baked wave texture overlay
- :func:`render_coastal_features` — foam, shallow sand, reef patches
- :func:`render_deep_ocean_features` — abyssal darkness, upwelling

Classes
-------
- :class:`OceanFeatureConfig` — tuneable ocean rendering parameters
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# ═══════════════════════════════════════════════════════════════════
# 17A.1 — OceanFeatureConfig
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class OceanFeatureConfig:
    """Tuneable parameters for ocean tile rendering.

    Controls the visual appearance of ocean tiles including depth
    colour gradients, wave patterns, coastal detail, and deep-ocean
    effects.

    Parameters
    ----------
    shallow_color : tuple of int
        RGB colour for shallow coastal water (0–255).
    deep_color : tuple of int
        RGB colour for mid-depth ocean (0–255).
    abyssal_color : tuple of int
        RGB colour for very deep ocean (0–255).
    coastal_foam_color : tuple of int
        RGB colour for surf/foam along coastlines (0–255).
    sand_color : tuple of int
        RGB colour for shallow-water sand/seabed visibility (0–255).
    depth_gradient_power : float
        Controls the shallow→deep transition curve.
        Higher values concentrate shallow colour near the coast.
    wave_frequency : float
        Spatial frequency of baked wave texture (cycles per tile).
    wave_amplitude : float
        Visual strength of wave pattern (fraction of base colour, 0–1).
    foam_width : float
        Width of coastal foam band as fraction of tile size (0–1).
    caustic_frequency : float
        Spatial frequency of underwater caustic ripple pattern.
    caustic_strength : float
        Visual intensity of caustic patterns (0–1).
    ice_latitude_threshold : float
        Latitude (0–1, 0=equator, 1=pole) above which ocean
        becomes icy / lighter.
    reef_probability : float
        Chance [0,1] of shallow-water reef colour patches per tile.
    density_scale : float
        Multiplied with the biome density from the density map.
        Allows per-preset strength adjustment.
    """

    shallow_color: Tuple[int, int, int] = (64, 164, 192)
    deep_color: Tuple[int, int, int] = (16, 48, 112)
    abyssal_color: Tuple[int, int, int] = (6, 16, 42)
    coastal_foam_color: Tuple[int, int, int] = (220, 230, 235)
    sand_color: Tuple[int, int, int] = (180, 170, 130)
    depth_gradient_power: float = 1.8
    wave_frequency: float = 6.0
    wave_amplitude: float = 0.04
    foam_width: float = 0.08
    caustic_frequency: float = 12.0
    caustic_strength: float = 0.06
    ice_latitude_threshold: float = 0.85
    reef_probability: float = 0.3
    density_scale: float = 1.0


# ── Presets ─────────────────────────────────────────────────────

TROPICAL_OCEAN = OceanFeatureConfig(
    shallow_color=(72, 200, 210),
    deep_color=(12, 52, 128),
    abyssal_color=(4, 14, 48),
    coastal_foam_color=(235, 240, 245),
    sand_color=(200, 190, 150),
    depth_gradient_power=2.2,
    wave_frequency=5.0,
    wave_amplitude=0.03,
    foam_width=0.10,
    caustic_frequency=14.0,
    caustic_strength=0.08,
    reef_probability=0.5,
)

TEMPERATE_OCEAN = OceanFeatureConfig(
    shallow_color=(56, 140, 168),
    deep_color=(20, 52, 100),
    abyssal_color=(8, 18, 44),
    coastal_foam_color=(210, 220, 225),
    sand_color=(165, 158, 125),
    depth_gradient_power=1.6,
    wave_frequency=7.0,
    wave_amplitude=0.05,
    foam_width=0.07,
    caustic_frequency=10.0,
    caustic_strength=0.05,
    reef_probability=0.15,
)

ARCTIC_OCEAN = OceanFeatureConfig(
    shallow_color=(100, 150, 170),
    deep_color=(30, 60, 90),
    abyssal_color=(12, 24, 50),
    coastal_foam_color=(230, 235, 240),
    sand_color=(150, 148, 135),
    depth_gradient_power=1.2,
    wave_frequency=4.0,
    wave_amplitude=0.02,
    foam_width=0.05,
    caustic_frequency=6.0,
    caustic_strength=0.02,
    ice_latitude_threshold=0.60,
    reef_probability=0.0,
)

DEEP_OCEAN = OceanFeatureConfig(
    shallow_color=(40, 120, 160),
    deep_color=(10, 36, 90),
    abyssal_color=(4, 10, 32),
    coastal_foam_color=(200, 210, 215),
    sand_color=(140, 135, 110),
    depth_gradient_power=2.5,
    wave_frequency=3.0,
    wave_amplitude=0.02,
    foam_width=0.04,
    caustic_frequency=8.0,
    caustic_strength=0.03,
    reef_probability=0.05,
)

OCEAN_PRESETS: Dict[str, OceanFeatureConfig] = {
    "tropical": TROPICAL_OCEAN,
    "temperate": TEMPERATE_OCEAN,
    "arctic": ARCTIC_OCEAN,
    "deep": DEEP_OCEAN,
}


# ═══════════════════════════════════════════════════════════════════
# 17A.2 — Ocean depth map
# ═══════════════════════════════════════════════════════════════════

def identify_ocean_tiles(
    patches: Sequence,
    *,
    terrain_type: str = "ocean",
) -> Set[str]:
    """Return the set of face IDs classified as ocean.

    Works identically to ``identify_forest_tiles()`` but defaults to
    the ``"ocean"`` terrain type.

    Parameters
    ----------
    patches : sequence of TerrainPatch
    terrain_type : str

    Returns
    -------
    set of str
    """
    face_ids: Set[str] = set()
    for patch in patches:
        if patch.terrain_type == terrain_type:
            face_ids.update(patch.face_ids)
    return face_ids


def compute_ocean_depth_map(
    globe_grid,
    globe_store,
    ocean_faces: Set[str],
    *,
    water_level: float = 0.12,
    max_bfs_depth: int = 50,
    elevation_weight: float = 0.5,
    distance_weight: float = 0.5,
) -> Dict[str, float]:
    """Compute a normalised [0, 1] depth value for every ocean tile.

    Depth is a hybrid of:
    - **Elevation-based depth:** ``(water_level - elevation) / water_level``
    - **Distance from coast:** BFS hop count from the nearest land tile,
      normalised by the maximum distance found.

    The two are combined as::

        depth = elevation_weight * elev_depth + distance_weight * dist_depth

    Parameters
    ----------
    globe_grid : PolyGrid / GlobeGrid
        The globe grid (for face adjacency).
    globe_store : TileDataStore
        Must contain ``"elevation"`` field.
    ocean_faces : set of str
        Face IDs classified as ocean.
    water_level : float
        Elevation threshold — tiles below this are ocean.
    max_bfs_depth : int
        Maximum BFS distance to search (caps computation).
    elevation_weight, distance_weight : float
        Relative weights for the two depth components.
        Need not sum to 1 — they are normalised internally.

    Returns
    -------
    dict
        ``{face_id: depth}`` for every face in *ocean_faces*.
        Values are in ``[0, 1]``: 0 = coastline, 1 = deepest ocean.
    """
    from .algorithms import get_face_adjacency

    adjacency = get_face_adjacency(globe_grid)
    land_faces = set(globe_grid.faces.keys()) - ocean_faces

    # ── BFS distance from coast ─────────────────────────────────
    # Seed the BFS from all land tiles (distance 0 for their
    # ocean neighbours = distance 1)
    bfs_dist: Dict[str, int] = {}
    queue: deque[Tuple[str, int]] = deque()

    # Initialise: land tiles bordering ocean are the BFS seeds
    for land_fid in land_faces:
        for nbr in adjacency.get(land_fid, []):
            if nbr in ocean_faces and nbr not in bfs_dist:
                bfs_dist[nbr] = 1
                queue.append((nbr, 1))

    # Also handle ocean tiles that might not border land (isolated deep ocean)
    # — they'll get distance = max_bfs_depth

    while queue:
        fid, dist = queue.popleft()
        if dist >= max_bfs_depth:
            continue
        for nbr in adjacency.get(fid, []):
            if nbr in ocean_faces and nbr not in bfs_dist:
                bfs_dist[nbr] = dist + 1
                queue.append((nbr, dist + 1))

    # Assign max distance to unreached ocean tiles
    max_dist_found = max(bfs_dist.values()) if bfs_dist else 1
    for fid in ocean_faces:
        if fid not in bfs_dist:
            bfs_dist[fid] = max_dist_found

    # ── Elevation-based depth ───────────────────────────────────
    w_total = elevation_weight + distance_weight
    if w_total <= 0:
        w_total = 1.0
    w_elev = elevation_weight / w_total
    w_dist = distance_weight / w_total

    # Normalise BFS distances to [0, 1]
    max_dist = max(bfs_dist.values()) if bfs_dist else 1
    if max_dist == 0:
        max_dist = 1

    depth_map: Dict[str, float] = {}
    for fid in ocean_faces:
        # Elevation component
        elev = globe_store.get(fid, "elevation")
        if elev is None:
            elev = 0.0
        elev_depth = max(0.0, min(1.0,
            (water_level - elev) / max(water_level, 0.001)
        ))

        # Distance component
        dist_depth = bfs_dist.get(fid, max_dist) / max_dist

        depth = w_elev * elev_depth + w_dist * dist_depth
        depth_map[fid] = max(0.0, min(1.0, depth))

    return depth_map


def compute_coast_direction(
    globe_grid,
    face_id: str,
    ocean_faces: Set[str],
) -> Optional[Tuple[float, float, float]]:
    """Compute the direction from an ocean tile toward the nearest coast.

    Returns a normalised 3D vector pointing from the ocean tile's
    centre toward the average position of its land-bordering neighbours.
    Returns ``None`` if the tile has no land neighbours.

    Parameters
    ----------
    globe_grid : PolyGrid / GlobeGrid
    face_id : str
    ocean_faces : set of str

    Returns
    -------
    tuple or None
        ``(dx, dy, dz)`` normalised direction, or *None*.
    """
    from .algorithms import get_face_adjacency
    from .geometry import face_center_3d

    adjacency = get_face_adjacency(globe_grid)
    neighbours = adjacency.get(face_id, [])
    land_neighbours = [n for n in neighbours if n not in ocean_faces]

    if not land_neighbours:
        return None

    # This tile's centre
    face = globe_grid.faces.get(face_id)
    if face is None:
        return None
    cx, cy, cz = face_center_3d(globe_grid.vertices, face)

    # Average land-neighbour centre
    lx, ly, lz = 0.0, 0.0, 0.0
    count = 0
    for nid in land_neighbours:
        nface = globe_grid.faces.get(nid)
        if nface is not None:
            nx, ny, nz = face_center_3d(globe_grid.vertices, nface)
            lx += nx
            ly += ny
            lz += nz
            count += 1

    if count == 0:
        return None

    lx /= count
    ly /= count
    lz /= count

    # Direction: land centre - ocean centre
    dx = lx - cx
    dy = ly - cy
    dz = lz - cz
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-12:
        return None
    return (dx / length, dy / length, dz / length)
