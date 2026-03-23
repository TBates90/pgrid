"""Tests for terrain/classification.py — Phase 2."""

from __future__ import annotations

import pytest

from polygrid.globe.globe import build_globe_grid
from polygrid.terrain.classification import (
    classify_tile,
    generate_terrain_field,
    TERRAIN_TYPES,
    OCEAN, SNOW, TUNDRA, MOUNTAINS, DESERT, WETLAND, HILLS, PLAINS,
    MOUNTAIN_THRESHOLD, SNOW_ELEVATION, SNOW_TEMPERATURE,
    TUNDRA_TEMPERATURE, DESERT_TEMPERATURE, DESERT_MOISTURE,
    WETLAND_ELEVATION_MARGIN, WETLAND_MOISTURE, WETLAND_TEMPERATURE,
    HILLS_ELEVATION_LOW, HILLS_ELEVATION_HIGH,
)
from polygrid.terrain.temperature import generate_temperature_field
from polygrid.terrain.moisture import generate_moisture_field
from polygrid.terrain.heightmap import sample_noise_field_3d, normalize_field
from polygrid.terrain.noise import fbm_3d
from polygrid.tile_data import FieldDef, TileSchema, TileDataStore


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def globe():
    return build_globe_grid(2)


@pytest.fixture
def schema():
    return TileSchema([
        FieldDef("elevation", float, 0.0),
        FieldDef("temperature", float, 0.0),
        FieldDef("moisture", float, 0.0),
        FieldDef("terrain", str, "unset"),
    ])


@pytest.fixture
def store(globe, schema):
    s = TileDataStore(globe, schema=schema)
    s.initialise_all()
    return s


# ═══════════════════════════════════════════════════════════════════
# classify_tile (pure function) — one test per terrain type
# ═══════════════════════════════════════════════════════════════════


class TestClassifyTile:
    """Verify each terrain type can be triggered and the priority order works."""

    def test_ocean(self):
        assert classify_tile(elevation=-0.1, temperature=0.5, moisture=0.5, water_level=0.0) == OCEAN

    def test_ocean_at_water_level(self):
        """Tiles at exactly water_level are ocean (≤ threshold)."""
        assert classify_tile(elevation=0.0, temperature=0.5, moisture=0.5, water_level=0.0) == OCEAN

    def test_snow(self):
        assert classify_tile(elevation=0.8, temperature=0.1, moisture=0.5) == SNOW

    def test_snow_requires_high_elevation(self):
        """Cold but not high enough → tundra, not snow."""
        assert classify_tile(elevation=0.5, temperature=0.1, moisture=0.5) != SNOW

    def test_tundra(self):
        assert classify_tile(elevation=0.3, temperature=0.1, moisture=0.5) == TUNDRA

    def test_tundra_any_elevation(self):
        """Tundra applies at any land elevation when cold enough."""
        # Low elevation.
        assert classify_tile(elevation=0.2, temperature=0.05, moisture=0.3) == TUNDRA
        # Mid elevation (below mountain threshold but above hills).
        assert classify_tile(elevation=0.5, temperature=0.1, moisture=0.5) == TUNDRA

    def test_mountains(self):
        assert classify_tile(elevation=0.7, temperature=0.5, moisture=0.5) == MOUNTAINS

    def test_mountains_priority_over_hills(self):
        """Elevation in both hills and mountains range → mountains wins."""
        assert classify_tile(elevation=0.66, temperature=0.5, moisture=0.5) == MOUNTAINS

    def test_desert(self):
        assert classify_tile(elevation=0.3, temperature=0.8, moisture=0.1) == DESERT

    def test_desert_requires_hot_and_dry(self):
        """Hot but wet → not desert."""
        assert classify_tile(elevation=0.3, temperature=0.8, moisture=0.5) != DESERT

    def test_wetland(self):
        assert classify_tile(elevation=0.03, temperature=0.5, moisture=0.8, water_level=0.0) == WETLAND

    def test_wetland_requires_near_water_level(self):
        """High elevation wet + warm → not wetland."""
        assert classify_tile(elevation=0.3, temperature=0.5, moisture=0.8) != WETLAND

    def test_hills(self):
        assert classify_tile(elevation=0.5, temperature=0.5, moisture=0.5) == HILLS

    def test_plains(self):
        assert classify_tile(elevation=0.3, temperature=0.5, moisture=0.5) == PLAINS

    def test_plains_catchall(self):
        """Low elevation, moderate temp/moisture → plains."""
        assert classify_tile(elevation=0.2, temperature=0.4, moisture=0.4) == PLAINS

    def test_all_terrain_types_reachable(self):
        """Every terrain type in TERRAIN_TYPES should be producible."""
        cases = {
            OCEAN: (-0.1, 0.5, 0.5, 0.0),
            SNOW: (0.8, 0.1, 0.5, 0.0),
            TUNDRA: (0.3, 0.1, 0.5, 0.0),
            MOUNTAINS: (0.7, 0.5, 0.5, 0.0),
            DESERT: (0.3, 0.8, 0.1, 0.0),
            WETLAND: (0.03, 0.5, 0.8, 0.0),
            HILLS: (0.5, 0.5, 0.5, 0.0),
            PLAINS: (0.3, 0.5, 0.5, 0.0),
        }
        produced = set()
        for terrain, (e, t, m, wl) in cases.items():
            result = classify_tile(e, t, m, water_level=wl)
            produced.add(result)
            assert result == terrain, f"Expected {terrain}, got {result}"
        assert produced == set(TERRAIN_TYPES)


# ═══════════════════════════════════════════════════════════════════
# Priority order tests
# ═══════════════════════════════════════════════════════════════════


class TestPriorityOrder:
    """Verify that higher-priority terrain types win over lower ones."""

    def test_ocean_beats_everything(self):
        """Below water level is always ocean regardless of temp/moisture."""
        assert classify_tile(-0.1, 0.01, 0.01, water_level=0.0) == OCEAN
        assert classify_tile(-0.1, 0.99, 0.99, water_level=0.0) == OCEAN

    def test_snow_beats_tundra(self):
        """High elevation + cold → snow, not tundra."""
        # Both snow (elev > 0.75, temp < 0.2) and tundra (temp < 0.15) qualify.
        result = classify_tile(0.8, 0.1, 0.5)
        assert result == SNOW

    def test_snow_beats_mountains(self):
        """High elevation + cold → snow, not mountains."""
        result = classify_tile(0.8, 0.1, 0.5)
        assert result == SNOW

    def test_tundra_beats_desert(self):
        """Very cold land → tundra even if hypothetically dry."""
        result = classify_tile(0.3, 0.1, 0.1)
        assert result == TUNDRA

    def test_mountains_beat_hills(self):
        """High elevation → mountains, not hills."""
        result = classify_tile(0.66, 0.5, 0.5)
        assert result == MOUNTAINS

    def test_desert_beats_plains(self):
        """Hot + dry at plains elevation → desert, not plains."""
        result = classify_tile(0.3, 0.8, 0.1)
        assert result == DESERT


# ═══════════════════════════════════════════════════════════════════
# generate_terrain_field (grid-level)
# ═══════════════════════════════════════════════════════════════════


class TestGenerateTerrainField:
    """Tests for writing terrain classification to a TileDataStore."""

    def test_all_tiles_classified(self, globe, store):
        # Set uniform moderate values so every tile hits the classifier.
        for fid in globe.faces:
            store.set(fid, "elevation", 0.3)
            store.set(fid, "temperature", 0.5)
            store.set(fid, "moisture", 0.5)

        generate_terrain_field(globe, store)
        for fid in globe.faces:
            t = store.get(fid, "terrain")
            assert t in TERRAIN_TYPES

    def test_uniform_ocean(self, globe, store):
        """All tiles below water level → all ocean."""
        for fid in globe.faces:
            store.set(fid, "elevation", -0.5)
            store.set(fid, "temperature", 0.5)
            store.set(fid, "moisture", 0.5)

        generate_terrain_field(globe, store, water_level=0.0)
        for fid in globe.faces:
            assert store.get(fid, "terrain") == OCEAN

    def test_multiple_terrain_types_produced(self, globe, store):
        """With varied inputs, more than one terrain type should appear."""
        # Generate realistic elevation, temperature, moisture.
        noise_fn = lambda x, y, z: fbm_3d(x, y, z, seed=42, octaves=4)
        sample_noise_field_3d(globe, store, "elevation", noise_fn)
        normalize_field(store, "elevation", lo=0.0, hi=1.0)

        generate_temperature_field(globe, store, 0.5)
        generate_moisture_field(globe, store, 0.5)
        generate_terrain_field(globe, store, water_level=0.3)

        terrains = {store.get(fid, "terrain") for fid in globe.faces}
        assert len(terrains) > 1, f"Only one terrain type: {terrains}"

    def test_restricted_face_ids(self, globe, store):
        """Only specified faces should be classified."""
        for fid in globe.faces:
            store.set(fid, "elevation", 0.3)
            store.set(fid, "temperature", 0.5)
            store.set(fid, "moisture", 0.5)

        all_fids = list(globe.faces.keys())
        target = all_fids[:5]
        rest = all_fids[5:]

        generate_terrain_field(globe, store, face_ids=target)

        for fid in target:
            assert store.get(fid, "terrain") in TERRAIN_TYPES
        for fid in rest:
            assert store.get(fid, "terrain") == "unset"  # default

    def test_full_pipeline_integration(self, globe, store):
        """End-to-end: elevation → temperature → moisture → terrain."""
        noise_fn = lambda x, y, z: fbm_3d(x, y, z, seed=99, octaves=4)
        sample_noise_field_3d(globe, store, "elevation", noise_fn)
        normalize_field(store, "elevation", lo=0.0, hi=1.0)

        generate_temperature_field(globe, store, 0.5)
        generate_moisture_field(globe, store, 0.5, water_level=0.3)
        generate_terrain_field(globe, store, water_level=0.3)

        # Every tile should have a valid terrain type.
        for fid in globe.faces:
            terrain = store.get(fid, "terrain")
            assert terrain in TERRAIN_TYPES, f"Unknown terrain: {terrain}"

        # Ocean tiles should have elevation ≤ water_level.
        for fid in globe.faces:
            if store.get(fid, "terrain") == OCEAN:
                assert store.get(fid, "elevation") <= 0.3
