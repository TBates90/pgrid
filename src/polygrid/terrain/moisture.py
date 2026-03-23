"""Moisture field generation for globe tiles.

Computes per-tile moisture from four factors:

1. **Base moisture** — planet ``water_abundance`` (0–1) scaled by the
   region's ``humidity_modifier`` (default 1.0).
2. **Ocean proximity** — tiles closer to ocean are wetter.  Implemented
   as a flood-fill distance from ocean tiles, inverted and normalised.
3. **Elevation penalty** — higher ground is drier (orographic rain shadow
   on the lee side is too complex for this pass; we approximate with a
   simple altitude penalty).
4. **Noise variation** — optional small-scale noise to break up uniform
   moisture bands.

The final value is clamped to ``[0, 1]``.

Functions
---------
- :func:`compute_ocean_distance` — BFS distance from ocean tiles.
- :func:`generate_moisture_field` — write moisture for all tiles.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Iterable, Optional, Set

from ..core.algorithms import get_face_adjacency
from ..core.polygrid import PolyGrid
from ..data.tile_data import TileDataStore


# ── Constants ───────────────────────────────────────────────────────

#: Weight given to the ocean-proximity signal.
OCEAN_PROXIMITY_WEIGHT: float = 0.35

#: How much elevation reduces moisture (per unit normalised elevation).
ELEVATION_PENALTY: float = 0.25

#: Maximum BFS hops used when computing ocean distance.  Tiles farther
#: than this are treated as maximally inland.
MAX_OCEAN_DISTANCE: int = 20


# ── Ocean distance (BFS) ────────────────────────────────────────────

def compute_ocean_distance(
    grid: PolyGrid,
    store: TileDataStore,
    elevation_field: str = "elevation",
    water_level: float = 0.0,
    max_distance: int = MAX_OCEAN_DISTANCE,
) -> Dict[str, int]:
    """Return BFS hop-distance from the nearest ocean tile for each face.

    Ocean tiles themselves have distance 0.  Land tiles unreachable
    within *max_distance* hops are assigned *max_distance*.

    Parameters
    ----------
    grid : PolyGrid
    store : TileDataStore
        Must contain *elevation_field*.
    elevation_field : str
    water_level : float
        Tiles with elevation ≤ this are ocean.
    max_distance : int
        BFS cutoff.

    Returns
    -------
    dict[str, int]
        ``{face_id: distance}`` for every face in the grid.
    """
    adj = get_face_adjacency(grid)

    # Identify ocean seeds
    ocean: Set[str] = set()
    for fid in grid.faces:
        if store.get(fid, elevation_field) <= water_level:
            ocean.add(fid)

    dist: Dict[str, int] = {fid: max_distance for fid in grid.faces}
    queue: deque[str] = deque()
    for fid in ocean:
        dist[fid] = 0
        queue.append(fid)

    while queue:
        fid = queue.popleft()
        d = dist[fid]
        if d >= max_distance:
            continue
        for nid in adj.get(fid, []):
            if dist[nid] > d + 1:
                dist[nid] = d + 1
                queue.append(nid)

    return dist


# ── Grid-wide generation ────────────────────────────────────────────

def generate_moisture_field(
    grid: PolyGrid,
    store: TileDataStore,
    water_abundance: float,
    *,
    elevation_field: str = "elevation",
    moisture_field: str = "moisture",
    water_level: float = 0.0,
    ocean_proximity_weight: float = OCEAN_PROXIMITY_WEIGHT,
    elevation_penalty: float = ELEVATION_PENALTY,
    max_ocean_distance: int = MAX_OCEAN_DISTANCE,
    region_humidity: Optional[Dict[str, float]] = None,
    region_field: Optional[str] = None,
    face_ids: Optional[Iterable[str]] = None,
) -> None:
    """Compute moisture for every tile and write to *store*.

    Parameters
    ----------
    grid : PolyGrid
    store : TileDataStore
        Must have *elevation_field* and *moisture_field* in schema.
    water_abundance : float
        Planet-wide water level (0.0–1.0).
    elevation_field : str
        Field name for elevation.
    moisture_field : str
        Field name to write moisture into.
    water_level : float
        Elevation threshold for ocean.
    ocean_proximity_weight : float
        Weight of the distance-to-ocean signal.
    elevation_penalty : float
        Moisture reduction per unit elevation.
    max_ocean_distance : int
        BFS cutoff for ocean distance.
    region_humidity : dict[str, float], optional
        ``{region_id: humidity_modifier}`` for per-region scaling.
    region_field : str, optional
        Field name in *store* that holds the region id string for each
        tile (needed when *region_humidity* is provided).
    face_ids : iterable of str, optional
        Restrict to these faces.
    """
    # Pre-compute ocean distance for the whole grid.
    ocean_dist = compute_ocean_distance(
        grid, store,
        elevation_field=elevation_field,
        water_level=water_level,
        max_distance=max_ocean_distance,
    )

    targets = face_ids if face_ids is not None else grid.faces.keys()

    for fid in targets:
        face = grid.faces.get(fid)
        if face is None:
            continue

        elev = store.get(fid, elevation_field)
        is_ocean = elev <= water_level

        # Ocean tiles get full moisture.
        if is_ocean:
            store.set(fid, moisture_field, 1.0)
            continue

        # Base moisture from planet water_abundance.
        base = water_abundance

        # Region humidity modifier.
        if region_humidity and region_field:
            try:
                rid = store.get(fid, region_field)
                modifier = region_humidity.get(str(rid), 1.0)
            except (KeyError, TypeError):
                modifier = 1.0
            base *= modifier

        # Ocean proximity boost: distance 0 → 1.0, max_distance → 0.0.
        d = ocean_dist.get(fid, max_ocean_distance)
        proximity = 1.0 - (d / max_ocean_distance)

        # Elevation penalty: higher = drier.
        elev_factor = elevation_penalty * max(0.0, elev)

        moisture = (
            base * (1.0 - ocean_proximity_weight)
            + proximity * ocean_proximity_weight
            - elev_factor
        )
        store.set(fid, moisture_field, max(0.0, min(1.0, moisture)))
