"""Feature generation — secondary tags applied after terrain classification.

Each feature pass runs independently and appends tags to a per-tile
features string stored in ``TileDataStore``.  Features are stored as
a **comma-separated string** (the store only supports scalar types).

Feature passes, in order:
    1. Coast — land tiles adjacent to ocean.
    2. Lake — interior low-elevation depressions (simplified).
    3. Forest — noise-based density on suitable terrain.

Rivers are referenced in the Biome Bible but depend on flow-accumulation
logic that lives in a future ``rivers.py`` module.  A hook is provided
but not yet implemented.

Functions
---------
- :func:`detect_coast` — mark land tiles neighbouring ocean.
- :func:`detect_lakes` — find inland depression clusters.
- :func:`place_forests` — noise-based forest placement.
- :func:`generate_features` — run all feature passes in order.
- :func:`add_feature` / :func:`get_features` — helpers for the
  comma-separated features string.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, Optional, Set

from ..core.algorithms import get_face_adjacency
from ..core.polygrid import PolyGrid
from ..data.tile_data import TileDataStore
from .classification import OCEAN, PLAINS, HILLS


# ── Feature tag constants ───────────────────────────────────────────

COAST: str = "coast"
LAKE: str = "lake"
FOREST: str = "forest"

FEATURE_TYPES: tuple[str, ...] = (COAST, LAKE, FOREST)


# ── Comma-separated feature helpers ─────────────────────────────────

def get_features(store: TileDataStore, face_id: str, field: str = "features") -> list[str]:
    """Return the list of feature tags for a tile (may be empty)."""
    raw = store.get(face_id, field)
    if not raw or raw == "":
        return []
    return [f.strip() for f in raw.split(",") if f.strip()]


def add_feature(
    store: TileDataStore,
    face_id: str,
    feature: str,
    field: str = "features",
) -> None:
    """Append a feature tag to a tile's comma-separated features string."""
    existing = get_features(store, face_id, field)
    if feature not in existing:
        existing.append(feature)
    store.set(face_id, field, ",".join(existing))


# ── Coast detection ─────────────────────────────────────────────────

def detect_coast(
    grid: PolyGrid,
    store: TileDataStore,
    *,
    terrain_field: str = "terrain",
    features_field: str = "features",
    face_ids: Optional[Iterable[str]] = None,
) -> set[str]:
    """Tag land tiles adjacent to ocean with the ``"coast"`` feature.

    Returns the set of face IDs that were tagged.
    """
    adj = get_face_adjacency(grid)
    targets = set(face_ids) if face_ids is not None else set(grid.faces.keys())

    # Identify all ocean tiles.
    ocean_tiles: Set[str] = set()
    for fid in grid.faces:
        if store.get(fid, terrain_field) == OCEAN:
            ocean_tiles.add(fid)

    tagged: set[str] = set()
    for fid in targets:
        if fid in ocean_tiles:
            continue  # ocean tiles don't get coast tag
        neighbours = adj.get(fid, [])
        if any(nid in ocean_tiles for nid in neighbours):
            add_feature(store, fid, COAST, features_field)
            tagged.add(fid)

    return tagged


# ── Lake detection (simplified) ─────────────────────────────────────

def detect_lakes(
    grid: PolyGrid,
    store: TileDataStore,
    *,
    elevation_field: str = "elevation",
    terrain_field: str = "terrain",
    features_field: str = "features",
    min_lake_size: int = 3,
    depression_threshold: float = 0.05,
    face_ids: Optional[Iterable[str]] = None,
) -> set[str]:
    """Find inland depressions and tag them as lakes.

    A simplified algorithm: find land tiles that are local minima
    (lower than all neighbours) and flood-fill outward up to
    *depression_threshold* above the minimum.  Clusters smaller than
    *min_lake_size* are discarded.

    Returns the set of face IDs tagged as lake.
    """
    adj = get_face_adjacency(grid)
    targets = set(face_ids) if face_ids is not None else set(grid.faces.keys())

    # Only consider non-ocean land tiles.
    land: Set[str] = set()
    for fid in targets:
        if store.get(fid, terrain_field) != OCEAN:
            land.add(fid)

    # Find local minima among land tiles.
    minima: Set[str] = set()
    for fid in land:
        elev = store.get(fid, elevation_field)
        neighbours = adj.get(fid, [])
        land_neighbours = [n for n in neighbours if n in land]
        if not land_neighbours:
            continue
        if all(store.get(n, elevation_field) > elev for n in land_neighbours):
            minima.add(fid)

    # Flood-fill from each minimum up to depression_threshold.
    all_lake_tiles: set[str] = set()
    visited: Set[str] = set()

    for seed in minima:
        if seed in visited:
            continue
        min_elev = store.get(seed, elevation_field)
        threshold = min_elev + depression_threshold

        cluster: set[str] = set()
        queue: deque[str] = deque([seed])
        while queue:
            fid = queue.popleft()
            if fid in cluster or fid not in land:
                continue
            if store.get(fid, elevation_field) > threshold:
                continue
            cluster.add(fid)
            for nid in adj.get(fid, []):
                if nid not in cluster and nid in land:
                    queue.append(nid)

        if len(cluster) >= min_lake_size:
            for fid in cluster:
                add_feature(store, fid, LAKE, features_field)
                all_lake_tiles.add(fid)
        visited |= cluster

    return all_lake_tiles


# ── Forest placement ────────────────────────────────────────────────

def _hash_noise_simple(x: float, y: float, z: float, seed: int = 0) -> float:
    """Cheap deterministic hash-based pseudo-noise in [0, 1]."""
    h = hash((round(x * 1000), round(y * 1000), round(z * 1000), seed))
    return (h & 0xFFFFFF) / 0xFFFFFF


def place_forests(
    grid: PolyGrid,
    store: TileDataStore,
    *,
    terrain_field: str = "terrain",
    moisture_field: str = "moisture",
    temperature_field: str = "temperature",
    features_field: str = "features",
    seed: int = 42,
    density_threshold: float = 0.45,
    min_moisture: float = 0.4,
    min_temperature: float = 0.2,
    max_temperature: float = 0.8,
    forest_weight: float = 1.0,
    face_ids: Optional[Iterable[str]] = None,
) -> set[str]:
    """Place forests on suitable tiles using a noise-based density field.

    Forests only appear on Plains or Hills tiles with sufficient
    moisture and moderate temperature.

    Parameters
    ----------
    density_threshold : float
        Noise values above this produce forest.  Lower = more forest.
    min_moisture, min_temperature, max_temperature : float
        Suitability limits.
    forest_weight : float
        Multiplier on the density score.  >1 = more forest, <1 = less.
        Maps to region ``feature_weights["forests"]``.
    seed : int
        Deterministic noise seed.

    Returns the set of face IDs tagged as forest.
    """
    suitable_terrains = {PLAINS, HILLS}
    targets = set(face_ids) if face_ids is not None else set(grid.faces.keys())

    tagged: set[str] = set()
    for fid in targets:
        terrain = store.get(fid, terrain_field)
        if terrain not in suitable_terrains:
            continue

        moisture = store.get(fid, moisture_field)
        temperature = store.get(fid, temperature_field)
        if moisture < min_moisture:
            continue
        if temperature < min_temperature or temperature > max_temperature:
            continue

        # 3D noise for density, avoiding projection seams.
        center = grid.faces[fid].metadata.get("center_3d")
        if center is None:
            continue
        cx, cy, cz = center
        noise_val = _hash_noise_simple(cx, cy, cz, seed)

        # Scale by forest_weight and moisture (wetter → denser forest).
        score = noise_val * forest_weight * (0.5 + moisture * 0.5)
        if score > density_threshold:
            add_feature(store, fid, FOREST, features_field)
            tagged.add(fid)

    return tagged


# ── Orchestrator ────────────────────────────────────────────────────

def generate_features(
    grid: PolyGrid,
    store: TileDataStore,
    *,
    seed: int = 42,
    forest_weight: float = 1.0,
    face_ids: Optional[Iterable[str]] = None,
) -> Dict[str, set[str]]:
    """Run all feature passes in order and return tagged face IDs per feature.

    Returns
    -------
    dict[str, set[str]]
        ``{feature_name: set_of_face_ids}`` for each feature type.
    """
    kwargs = {}
    if face_ids is not None:
        kwargs["face_ids"] = face_ids

    coast = detect_coast(grid, store, **kwargs)
    lakes = detect_lakes(grid, store, **kwargs)
    forests = place_forests(grid, store, seed=seed, forest_weight=forest_weight, **kwargs)

    return {
        COAST: coast,
        LAKE: lakes,
        FOREST: forests,
    }
