"""Temperature field generation for globe tiles.

Computes per-tile temperature from three additive components:

1. **Planet baseline** — a uniform offset representing the planet's
   overall thermal budget (0.0 = frozen world, 1.0 = scorching).
2. **Latitude gradient** — equator is warmest, poles are coldest.
   Modelled as ``cos(latitude)`` so the drop-off is smooth and
   physically plausible.
3. **Elevation lapse rate** — higher ground is colder.  Uses a
   tuneable rate (default ``LAPSE_RATE = 0.6``) applied to the
   normalised elevation (0–1).

The final value is clamped to ``[0, 1]``.

Functions
---------
- :func:`compute_temperature` — pure function for a single tile.
- :func:`generate_temperature_field` — write temperatures for all
  tiles in a :class:`~polygrid.globe.globe.GlobeGrid` into a
  :class:`~polygrid.data.tile_data.TileDataStore`.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

from ..core.polygrid import PolyGrid
from ..data.tile_data import TileDataStore


# ── Constants ───────────────────────────────────────────────────────

#: Default weight of the latitude gradient in the temperature formula.
#: 1.0 means the full cosine curve applies; 0.0 would disable it.
LATITUDE_WEIGHT: float = 0.5

#: Default lapse-rate coefficient.  Elevation is normalised [0, 1],
#: so a rate of 0.6 means a tile at max elevation is 0.6 cooler than
#: an equivalent tile at sea level.
LAPSE_RATE: float = 0.6


# ── Single-tile computation ─────────────────────────────────────────

def compute_temperature(
    latitude_deg: float,
    elevation: float,
    planet_temperature: float,
    *,
    latitude_weight: float = LATITUDE_WEIGHT,
    lapse_rate: float = LAPSE_RATE,
) -> float:
    """Return the temperature for one tile, clamped to ``[0, 1]``.

    Parameters
    ----------
    latitude_deg : float
        Latitude in degrees (−90 … +90).
    elevation : float
        Normalised elevation (0.0 = sea level, 1.0 = highest peak).
        Tiles below sea-level (ocean) should be passed as 0.0 since
        ocean temperature is governed by the surface, not the seabed.
    planet_temperature : float
        Planet-wide baseline (0.0–1.0).
    latitude_weight : float
        How strongly latitude affects temperature.  Default 0.5.
    lapse_rate : float
        Temperature reduction per unit of normalised elevation.
        Default 0.6.

    Returns
    -------
    float
        Temperature in ``[0.0, 1.0]``.
    """
    # Latitude component: cos(lat) gives 1.0 at equator, 0.0 at poles.
    lat_factor = math.cos(math.radians(latitude_deg))

    # Blend: baseline + latitude contribution − elevation penalty.
    temp = (
        planet_temperature * (1.0 - latitude_weight)
        + lat_factor * latitude_weight
        - lapse_rate * elevation
    )
    return max(0.0, min(1.0, temp))


# ── Grid-wide generation ────────────────────────────────────────────

def generate_temperature_field(
    grid: PolyGrid,
    store: TileDataStore,
    planet_temperature: float,
    *,
    elevation_field: str = "elevation",
    temperature_field: str = "temperature",
    latitude_weight: float = LATITUDE_WEIGHT,
    lapse_rate: float = LAPSE_RATE,
    water_level: float = 0.0,
    face_ids: Optional[Iterable[str]] = None,
) -> None:
    """Compute temperature for every tile and write to *store*.

    Reads ``latitude_deg`` from face metadata and the elevation value
    from *store* (defaulting to 0.0 for tiles below *water_level*).

    Parameters
    ----------
    grid : PolyGrid
        Globe grid — faces must carry ``latitude_deg`` in metadata.
    store : TileDataStore
        Must have both *elevation_field* and *temperature_field* in its
        schema.
    planet_temperature : float
        Planet-wide baseline (0.0–1.0).
    elevation_field : str
        Name of the elevation field to read.
    temperature_field : str
        Name of the field to write temperature into.
    latitude_weight : float
        Strength of the latitude gradient.
    lapse_rate : float
        Elevation lapse-rate coefficient.
    water_level : float
        Elevation threshold below which a tile is ocean.  Ocean tiles
        use elevation = 0.0 for the lapse-rate calculation (ocean
        surface temperature is not affected by seabed depth).
    face_ids : iterable of str, optional
        Restrict to these faces.  Defaults to all faces.
    """
    targets = face_ids if face_ids is not None else grid.faces.keys()

    for fid in targets:
        face = grid.faces.get(fid)
        if face is None:
            continue

        lat = face.metadata.get("latitude_deg")
        if lat is None:
            continue  # skip faces without lat (non-globe grids)

        raw_elev = store.get(fid, elevation_field)
        # Ocean tiles: treat depth as 0 for lapse-rate purposes.
        elev = max(0.0, raw_elev) if raw_elev <= water_level else raw_elev

        temp = compute_temperature(
            latitude_deg=lat,
            elevation=elev,
            planet_temperature=planet_temperature,
            latitude_weight=latitude_weight,
            lapse_rate=lapse_rate,
        )
        store.set(fid, temperature_field, temp)
