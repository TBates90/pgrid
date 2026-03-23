"""Terrain classification — assign a terrain type to every tile.

Uses the priority table from the Biome Bible to classify each tile
based on its elevation, temperature, and moisture values.  The table is
evaluated top-to-bottom; the first matching rule wins.

Priority order (highest → lowest):
    Ocean → Snow → Tundra → Mountains → Desert → Wetland → Hills → Plains

A tile that doesn't match any specific rule falls through to ``"plains"``
as the catch-all default.

Functions
---------
- :func:`classify_tile` — classify a single tile (pure function).
- :func:`generate_terrain_field` — classify all tiles in a grid, writing
  results to a :class:`~polygrid.data.tile_data.TileDataStore`.
"""

from __future__ import annotations

from typing import Iterable, Optional

from ..core.polygrid import PolyGrid
from ..data.tile_data import TileDataStore


# ── Terrain type constants ──────────────────────────────────────────

OCEAN: str = "ocean"
SNOW: str = "snow"
TUNDRA: str = "tundra"
MOUNTAINS: str = "mountains"
DESERT: str = "desert"
WETLAND: str = "wetland"
HILLS: str = "hills"
PLAINS: str = "plains"

#: All terrain types in priority order.
TERRAIN_TYPES: tuple[str, ...] = (
    OCEAN, SNOW, TUNDRA, MOUNTAINS, DESERT, WETLAND, HILLS, PLAINS,
)


# ── Default thresholds ──────────────────────────────────────────────

#: Elevation above which a tile is classified as mountains (if not snow).
MOUNTAIN_THRESHOLD: float = 0.65

#: Elevation above which a cold mountain becomes snow.
SNOW_ELEVATION: float = 0.75

#: Temperature below which snow replaces mountains at high elevation.
SNOW_TEMPERATURE: float = 0.2

#: Temperature below which any land tile is tundra.
TUNDRA_TEMPERATURE: float = 0.15

#: Minimum temperature for desert.
DESERT_TEMPERATURE: float = 0.6

#: Maximum moisture for desert.
DESERT_MOISTURE: float = 0.2

#: How far above water_level a tile can be and still qualify as wetland.
WETLAND_ELEVATION_MARGIN: float = 0.05

#: Minimum moisture for wetland.
WETLAND_MOISTURE: float = 0.6

#: Minimum temperature for wetland.
WETLAND_TEMPERATURE: float = 0.3

#: Elevation range for hills.
HILLS_ELEVATION_LOW: float = 0.40
HILLS_ELEVATION_HIGH: float = 0.65


# ── Single-tile classification ──────────────────────────────────────

def classify_tile(
    elevation: float,
    temperature: float,
    moisture: float,
    water_level: float = 0.0,
) -> str:
    """Return the terrain type for one tile.

    Parameters
    ----------
    elevation : float
        Normalised elevation (0.0–1.0).
    temperature : float
        Temperature (0.0–1.0).
    moisture : float
        Moisture (0.0–1.0).
    water_level : float
        Global water-level threshold.

    Returns
    -------
    str
        One of the ``TERRAIN_TYPES`` constants.
    """
    # --- Priority 1: Ocean ---
    if elevation <= water_level:
        return OCEAN

    # --- Priority 2: Snow (high peak + cold) ---
    if elevation > SNOW_ELEVATION and temperature < SNOW_TEMPERATURE:
        return SNOW

    # --- Priority 3: Tundra (freezing land) ---
    if temperature < TUNDRA_TEMPERATURE:
        return TUNDRA

    # --- Priority 4: Mountains (high elevation) ---
    if elevation > MOUNTAIN_THRESHOLD:
        return MOUNTAINS

    # --- Priority 5: Desert (hot + dry) ---
    if temperature > DESERT_TEMPERATURE and moisture < DESERT_MOISTURE:
        return DESERT

    # --- Priority 6: Wetland (just above water, warm, wet) ---
    if (
        elevation <= water_level + WETLAND_ELEVATION_MARGIN
        and moisture > WETLAND_MOISTURE
        and temperature > WETLAND_TEMPERATURE
    ):
        return WETLAND

    # --- Priority 7: Hills (moderate-high elevation) ---
    if HILLS_ELEVATION_LOW <= elevation <= HILLS_ELEVATION_HIGH:
        return HILLS

    # --- Priority 8: Plains (catch-all land) ---
    return PLAINS


# ── Grid-wide generation ────────────────────────────────────────────

def generate_terrain_field(
    grid: PolyGrid,
    store: TileDataStore,
    *,
    elevation_field: str = "elevation",
    temperature_field: str = "temperature",
    moisture_field: str = "moisture",
    terrain_field: str = "terrain",
    water_level: float = 0.0,
    face_ids: Optional[Iterable[str]] = None,
) -> None:
    """Classify every tile and write the terrain type to *store*.

    Parameters
    ----------
    grid : PolyGrid
    store : TileDataStore
        Must have *elevation_field*, *temperature_field*,
        *moisture_field*, and *terrain_field* in its schema.
    elevation_field, temperature_field, moisture_field : str
        Names of the input fields.
    terrain_field : str
        Name of the field to write terrain type into.
    water_level : float
        Global water-level threshold.
    face_ids : iterable of str, optional
        Restrict to these faces.
    """
    targets = face_ids if face_ids is not None else grid.faces.keys()

    for fid in targets:
        if fid not in grid.faces:
            continue

        elev = store.get(fid, elevation_field)
        temp = store.get(fid, temperature_field)
        moist = store.get(fid, moisture_field)

        terrain = classify_tile(elev, temp, moist, water_level=water_level)
        store.set(fid, terrain_field, terrain)
