"""Public API — single entry point for planet terrain generation.

This module hides the internal pipeline complexity behind a clean
function that external consumers (e.g. the playground app) call with
a simple parameter set and receive a fully populated result.

Classes
-------
- :class:`PlanetParams` — input configuration from the consumer.
- :class:`RegionParams` — per-region modifiers.
- :class:`TileResult` — per-tile output.
- :class:`GenerationResult` — full result from a generation run.

Functions
---------
- :func:`generate_planet` — orchestrate the full terrain pipeline.
- :func:`parse_layout` — extract Goldberg frequency from a layout string.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .data.tile_data import FieldDef, TileDataStore, TileSchema
from .globe.globe import GlobeGrid, build_globe_grid
from .globe.globe_export import _ramp_satellite
from .terrain.temperature import generate_temperature_field
from .terrain.moisture import generate_moisture_field
from .terrain.classification import generate_terrain_field
from .terrain.features import generate_features, get_features
from .terrain.heightmap import (
    normalize_field,
    sample_noise_field_3d,
)
from .terrain.noise import fbm_3d, ridged_noise_3d


# ═══════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RegionParams:
    """Per-region modifier set.

    Attributes
    ----------
    region_id : str
        Unique identifier for the region (typically a UUID from the
        consumer application).
    tile_slugs : list[str]
        Goldberg tile slugs assigned to this region (e.g.
        ``["2:f1:2-0-0", "2:f1:2-0-1"]``).  These are resolved to
        pgrid face IDs internally.
    elevation_modifier : float
        Multiplicative scaling for elevation (default 1.0).
    humidity_modifier : float
        Multiplicative scaling for moisture (default 1.0).
    feature_weights : dict[str, float]
        Per-feature density scaling (e.g. ``{"forests": 1.5}``).
    biome_hint : str | None
        Optional consumer-side classification slug.  Not used by pgrid
        but round-tripped in the result for the consumer's convenience.
    """

    region_id: str
    tile_slugs: list[str] = field(default_factory=list)
    elevation_modifier: float = 1.0
    humidity_modifier: float = 1.0
    feature_weights: dict[str, float] = field(default_factory=dict)
    biome_hint: str | None = None


@dataclass(frozen=True)
class PlanetParams:
    """Parameters the consumer sends to pgrid for planet generation.

    Attributes
    ----------
    frequency : int
        Goldberg polyhedron subdivision frequency (≥ 1).
    seed : int
        Deterministic generation seed.
    water_abundance : float
        Planet-wide water level (0.0–1.0).  Higher values mean more
        ocean tiles and wetter land.
    roughness : float
        Terrain roughness (0.0–1.0).  Controls noise octaves and ridge
        mixing.
    temperature : float
        Planet-wide temperature baseline (0.0–1.0).
    minerals : list[str]
        Mineral resource tags (e.g. ``["iron", "crystal"]``).
        Not used by the terrain pipeline but round-tripped in metadata.
    regions : list[RegionParams]
        One entry per painted region.  Tiles not assigned to any region
        use default modifiers (1.0).
    """

    frequency: int = 3
    seed: int = 42
    water_abundance: float = 0.5
    roughness: float = 0.5
    temperature: float = 0.5
    minerals: list[str] = field(default_factory=list)
    regions: list[RegionParams] = field(default_factory=list)


@dataclass(frozen=True)
class TileResult:
    """Per-tile output from the terrain pipeline.

    All numeric fields are normalised to ``[0, 1]`` unless stated
    otherwise.
    """

    tile_slug: str
    face_id: str
    face_type: str  # "pent" or "hex"
    elevation: float
    temperature: float
    moisture: float
    terrain: str
    features: list[str]
    color: tuple[float, float, float]
    region_id: str | None


@dataclass(frozen=True)
class GenerationResult:
    """Full result from a generation run."""

    tiles: list[TileResult]
    metadata: dict[str, Any]


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def parse_layout(layout: str) -> int:
    """Extract the Goldberg frequency from a layout string.

    Supported formats::

        "gb3"  → 3
        "gb10" → 10
        "3"    → 3      (bare integer)

    Parameters
    ----------
    layout : str
        Layout string such as ``"gb3"``.

    Returns
    -------
    int
        Goldberg subdivision frequency (≥ 1).

    Raises
    ------
    ValueError
        If the string cannot be parsed.
    """
    token = layout.strip().lower()
    if token.startswith("gb"):
        token = token[2:]
    try:
        freq = int(token)
    except ValueError:
        raise ValueError(
            f"Cannot parse layout string {layout!r} — expected 'gbN' or 'N'"
        ) from None
    if freq < 1:
        raise ValueError(f"Frequency must be ≥ 1, got {freq}")
    return freq


def _build_schema() -> TileSchema:
    """Build the TileSchema used by the generation pipeline."""
    return TileSchema([
        FieldDef("elevation", float, 0.0),
        FieldDef("temperature", float, 0.0),
        FieldDef("moisture", float, 0.0),
        FieldDef("terrain", str, ""),
        FieldDef("features", str, ""),
        FieldDef("region_id", str, ""),
    ])


def _compute_water_level(water_abundance: float) -> float:
    """Map water_abundance (0–1) to an elevation water-level threshold.

    A water_abundance of 0.5 puts the waterline at elevation 0.30,
    meaning roughly 30% of tiles (by elevation) become ocean.  The
    relationship is linear: ``water_level = water_abundance * 0.6``.
    """
    return water_abundance * 0.6


# ═══════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════

def generate_planet(params: PlanetParams) -> GenerationResult:
    """Run the full terrain generation pipeline for a planet.

    This is the single entry point that external consumers call.

    Pipeline stages
    ---------------
    1. Parse layout → build :class:`GlobeGrid`.
    2. Resolve regions (tile slugs → face IDs).
    3. Generate base elevation (3-D FBM + ridged noise, blended by
       roughness).
    4. Apply per-region elevation modifiers.
    5. Normalise elevation to ``[0, 1]``.
    6. Compute water level from ``water_abundance``.
    7. Generate temperature field.
    8. Generate moisture field (with per-region humidity modifiers).
    9. Classify terrain types.
    10. Detect features (coast, lakes, forests).
    11. Compute colours from the satellite ramp.
    12. Collect :class:`TileResult` objects.

    Parameters
    ----------
    params : PlanetParams
        All generation parameters.

    Returns
    -------
    GenerationResult
        Per-tile data and generation metadata.
    """
    t_start = time.monotonic()

    # ── 1. Build globe grid ─────────────────────────────────────────
    frequency = params.frequency
    grid = build_globe_grid(frequency)
    store = TileDataStore(grid, schema=_build_schema())

    # ── 2. Resolve regions ──────────────────────────────────────────
    slug_to_face = grid.build_reverse_slug_lookup()
    face_to_slug = grid.build_slug_lookup()

    # Build region_id → RegionParams and face_id → region_id maps.
    region_lookup: Dict[str, RegionParams] = {}
    face_region: Dict[str, str] = {}
    for rp in params.regions:
        region_lookup[rp.region_id] = rp
        for slug in rp.tile_slugs:
            fid = slug_to_face.get(slug)
            if fid is not None:
                face_region[fid] = rp.region_id
                store.set(fid, "region_id", rp.region_id)

    # ── 3. Generate base elevation ──────────────────────────────────
    #
    # Blend FBM (smooth rolling terrain) with ridged noise (mountains)
    # controlled by the roughness parameter.  Higher roughness means
    # more ridged noise and more octaves.
    fbm_octaves = 4 + int(params.roughness * 4)  # 4–8
    ridge_mix = params.roughness  # 0.0 = pure FBM, 1.0 = pure ridged

    def elevation_noise(x: float, y: float, z: float) -> float:
        base = fbm_3d(
            x, y, z,
            octaves=fbm_octaves,
            frequency=2.0,
            seed=params.seed,
        )
        ridge = ridged_noise_3d(
            x, y, z,
            octaves=max(3, fbm_octaves - 1),
            frequency=2.5,
            seed=params.seed + 1,
        )
        # Blend: (1 − ridge_mix) × base + ridge_mix × ridge
        # base is in ~[-1, 1], ridge is in [0, 1]; shift base to [0, 1].
        base_01 = (base + 1.0) * 0.5
        return (1.0 - ridge_mix) * base_01 + ridge_mix * ridge

    sample_noise_field_3d(grid, store, "elevation", elevation_noise)

    # ── 4. Apply per-region elevation modifiers ─────────────────────
    for fid in grid.faces:
        rid = face_region.get(fid)
        if rid is not None:
            rp = region_lookup[rid]
            if rp.elevation_modifier != 1.0:
                raw = store.get(fid, "elevation")
                store.set(fid, "elevation", raw * rp.elevation_modifier)

    # ── 5. Normalise elevation to [0, 1] ────────────────────────────
    normalize_field(store, "elevation")

    # ── 6. Water level ──────────────────────────────────────────────
    water_level = _compute_water_level(params.water_abundance)

    # ── 7. Temperature ──────────────────────────────────────────────
    generate_temperature_field(grid, store, params.temperature)

    # ── 8. Moisture ─────────────────────────────────────────────────
    # Build per-region humidity map.
    region_humidity: Dict[str, float] | None = None
    if params.regions:
        region_humidity = {
            rp.region_id: rp.humidity_modifier
            for rp in params.regions
        }

    generate_moisture_field(
        grid,
        store,
        params.water_abundance,
        water_level=water_level,
        region_humidity=region_humidity,
        region_field="region_id" if region_humidity else None,
    )

    # ── 9. Terrain classification ───────────────────────────────────
    generate_terrain_field(grid, store, water_level=water_level)

    # ── 10. Features ────────────────────────────────────────────────
    # Use the max forest_weight from any region, or default 1.0.
    forest_weight = 1.0
    for rp in params.regions:
        fw = rp.feature_weights.get("forests", 1.0)
        if fw > forest_weight:
            forest_weight = fw

    generate_features(grid, store, seed=params.seed, forest_weight=forest_weight)

    # ── 11. Colours ─────────────────────────────────────────────────
    # Use the satellite ramp from globe_export for consistency.
    colour_map: Dict[str, Tuple[float, float, float]] = {}
    for fid in grid.faces:
        elev = store.get(fid, "elevation")
        colour_map[fid] = _ramp_satellite(max(0.0, min(1.0, float(elev))))

    # ── 12. Collect results ─────────────────────────────────────────
    tiles: List[TileResult] = []
    for fid in sorted(grid.faces.keys()):
        slug = face_to_slug.get(fid, fid)
        face = grid.faces[fid]
        features_raw = get_features(store, fid)
        rgb = colour_map.get(fid, (0.5, 0.5, 0.5))
        rid = face_region.get(fid)

        tiles.append(TileResult(
            tile_slug=slug,
            face_id=fid,
            face_type=face.face_type,
            elevation=round(store.get(fid, "elevation"), 6),
            temperature=round(store.get(fid, "temperature"), 6),
            moisture=round(store.get(fid, "moisture"), 6),
            terrain=store.get(fid, "terrain"),
            features=features_raw,
            color=(round(rgb[0], 4), round(rgb[1], 4), round(rgb[2], 4)),
            region_id=rid,
        ))

    t_elapsed = time.monotonic() - t_start

    metadata: Dict[str, Any] = {
        "seed": params.seed,
        "frequency": frequency,
        "water_abundance": params.water_abundance,
        "water_level": round(water_level, 4),
        "roughness": params.roughness,
        "temperature": params.temperature,
        "minerals": list(params.minerals),
        "tile_count": len(tiles),
        "pentagon_count": sum(1 for t in tiles if t.face_type == "pent"),
        "hexagon_count": sum(1 for t in tiles if t.face_type == "hex"),
        "region_count": len(params.regions),
        "generation_time_s": round(t_elapsed, 3),
    }

    return GenerationResult(tiles=tiles, metadata=metadata)
