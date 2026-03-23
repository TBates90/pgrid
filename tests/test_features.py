"""Tests for terrain/features.py — Phase 3."""

from __future__ import annotations

import pytest

from polygrid.globe.globe import build_globe_grid
from polygrid.terrain.features import (
    detect_coast,
    detect_lakes,
    place_forests,
    generate_features,
    get_features,
    add_feature,
    COAST, LAKE, FOREST, FEATURE_TYPES,
)
from polygrid.terrain.classification import (
    generate_terrain_field,
    OCEAN, PLAINS, HILLS, MOUNTAINS, TUNDRA,
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
        FieldDef("features", str, ""),
    ])


@pytest.fixture
def store(globe, schema):
    s = TileDataStore(globe, schema=schema)
    s.initialise_all()
    return s


@pytest.fixture
def populated_store(globe, store):
    """Store with full pipeline: elevation → temperature → moisture → terrain."""
    noise_fn = lambda x, y, z: fbm_3d(x, y, z, seed=42, octaves=4)
    sample_noise_field_3d(globe, store, "elevation", noise_fn)
    normalize_field(store, "elevation", lo=0.0, hi=1.0)

    generate_temperature_field(globe, store, 0.5)
    generate_moisture_field(globe, store, 0.5, water_level=0.3)
    generate_terrain_field(globe, store, water_level=0.3)
    return store


# ═══════════════════════════════════════════════════════════════════
# Feature string helpers
# ═══════════════════════════════════════════════════════════════════


class TestFeatureHelpers:
    """Tests for get_features / add_feature comma-separated logic."""

    def test_empty_features(self, globe, store):
        fid = list(globe.faces.keys())[0]
        assert get_features(store, fid) == []

    def test_add_one_feature(self, globe, store):
        fid = list(globe.faces.keys())[0]
        add_feature(store, fid, "coast")
        assert get_features(store, fid) == ["coast"]

    def test_add_multiple_features(self, globe, store):
        fid = list(globe.faces.keys())[0]
        add_feature(store, fid, "coast")
        add_feature(store, fid, "forest")
        assert get_features(store, fid) == ["coast", "forest"]

    def test_no_duplicates(self, globe, store):
        fid = list(globe.faces.keys())[0]
        add_feature(store, fid, "coast")
        add_feature(store, fid, "coast")
        assert get_features(store, fid) == ["coast"]


# ═══════════════════════════════════════════════════════════════════
# Coast detection
# ═══════════════════════════════════════════════════════════════════


class TestCoastDetection:
    """Tests for detect_coast."""

    def test_coast_only_on_land_adjacent_to_ocean(self, globe, populated_store):
        tagged = detect_coast(globe, populated_store)

        for fid in tagged:
            # Must not be ocean.
            assert populated_store.get(fid, "terrain") != OCEAN
            # Must have at least one ocean neighbour.
            neighbours = globe.faces[fid].neighbor_ids
            ocean_neighbours = [
                n for n in neighbours
                if populated_store.get(n, "terrain") == OCEAN
            ]
            assert len(ocean_neighbours) > 0

    def test_no_coast_on_all_land(self, globe, store):
        """If there's no ocean, no coast should be detected."""
        for fid in globe.faces:
            store.set(fid, "elevation", 0.5)
            store.set(fid, "temperature", 0.5)
            store.set(fid, "moisture", 0.5)
        generate_terrain_field(globe, store, water_level=0.0)

        tagged = detect_coast(globe, store)
        assert len(tagged) == 0

    def test_coast_on_all_ocean_border(self, globe, store):
        """All land tiles next to ocean should get coast."""
        fids = list(globe.faces.keys())
        # Make first tile ocean, rest land.
        for fid in fids:
            store.set(fid, "elevation", 0.5)
            store.set(fid, "temperature", 0.5)
            store.set(fid, "moisture", 0.5)
        store.set(fids[0], "elevation", -0.1)
        generate_terrain_field(globe, store, water_level=0.0)

        tagged = detect_coast(globe, store)
        ocean_neighbours = set(globe.faces[fids[0]].neighbor_ids)
        assert tagged == ocean_neighbours

    def test_ocean_tile_not_tagged(self, globe, populated_store):
        """Ocean tiles themselves should not be tagged as coast."""
        detect_coast(globe, populated_store)
        for fid in globe.faces:
            if populated_store.get(fid, "terrain") == OCEAN:
                assert COAST not in get_features(populated_store, fid)


# ═══════════════════════════════════════════════════════════════════
# Lake detection
# ═══════════════════════════════════════════════════════════════════


class TestLakeDetection:
    """Tests for detect_lakes."""

    def test_no_lakes_on_uniform_elevation(self, globe, store):
        """Uniform elevation has no local minima → no lakes."""
        for fid in globe.faces:
            store.set(fid, "elevation", 0.5)
            store.set(fid, "temperature", 0.5)
            store.set(fid, "moisture", 0.5)
        generate_terrain_field(globe, store, water_level=0.0)

        tagged = detect_lakes(globe, store)
        assert len(tagged) == 0

    def test_no_lakes_on_all_ocean(self, globe, store):
        """All ocean → no land → no lakes."""
        for fid in globe.faces:
            store.set(fid, "elevation", -0.1)
            store.set(fid, "temperature", 0.5)
            store.set(fid, "moisture", 0.5)
        generate_terrain_field(globe, store, water_level=0.0)

        tagged = detect_lakes(globe, store)
        assert len(tagged) == 0

    def test_lake_tiles_get_feature_tag(self, globe, populated_store):
        """Any tiles detected as lakes should have the 'lake' feature."""
        tagged = detect_lakes(globe, populated_store)
        for fid in tagged:
            assert LAKE in get_features(populated_store, fid)


# ═══════════════════════════════════════════════════════════════════
# Forest placement
# ═══════════════════════════════════════════════════════════════════


class TestForestPlacement:
    """Tests for place_forests."""

    def test_forests_only_on_suitable_terrain(self, globe, populated_store):
        """Forests should only appear on Plains or Hills."""
        tagged = place_forests(globe, populated_store, seed=42)
        for fid in tagged:
            terrain = populated_store.get(fid, "terrain")
            assert terrain in (PLAINS, HILLS), f"Forest on {terrain}"

    def test_forests_respect_moisture_threshold(self, globe, populated_store):
        """Forest tiles should have moisture ≥ min_moisture (default 0.4)."""
        tagged = place_forests(globe, populated_store, seed=42)
        for fid in tagged:
            assert populated_store.get(fid, "moisture") >= 0.4

    def test_forests_respect_temperature_range(self, globe, populated_store):
        """Forest tiles should have temperature in [0.2, 0.8]."""
        tagged = place_forests(globe, populated_store, seed=42)
        for fid in tagged:
            temp = populated_store.get(fid, "temperature")
            assert 0.2 <= temp <= 0.8

    def test_no_forest_on_desert(self, globe, store):
        """Desert tiles should never get forest."""
        for fid in globe.faces:
            store.set(fid, "elevation", 0.3)
            store.set(fid, "temperature", 0.8)
            store.set(fid, "moisture", 0.1)
            store.set(fid, "terrain", "desert")

        tagged = place_forests(globe, store, seed=42)
        assert len(tagged) == 0

    def test_forest_weight_affects_density(self, globe, populated_store):
        """Higher forest_weight should produce more (or equal) forest tiles."""
        tagged_low = place_forests(globe, populated_store, seed=42, forest_weight=0.3)
        # Reset features.
        for fid in globe.faces:
            populated_store.set(fid, "features", "")
        tagged_high = place_forests(globe, populated_store, seed=42, forest_weight=2.0)
        assert len(tagged_high) >= len(tagged_low)

    def test_deterministic(self, globe, populated_store):
        """Same seed → same forest placement."""
        tagged1 = place_forests(globe, populated_store, seed=99)
        for fid in globe.faces:
            populated_store.set(fid, "features", "")
        tagged2 = place_forests(globe, populated_store, seed=99)
        assert tagged1 == tagged2


# ═══════════════════════════════════════════════════════════════════
# generate_features (orchestrator)
# ═══════════════════════════════════════════════════════════════════


class TestGenerateFeatures:
    """Tests for the combined feature generation pass."""

    def test_returns_all_feature_types(self, globe, populated_store):
        result = generate_features(globe, populated_store, seed=42)
        assert set(result.keys()) == set(FEATURE_TYPES)

    def test_no_crash_on_all_ocean(self, globe, store):
        """Feature generation shouldn't crash when everything is ocean."""
        for fid in globe.faces:
            store.set(fid, "elevation", -0.5)
            store.set(fid, "temperature", 0.5)
            store.set(fid, "moisture", 1.0)
        generate_terrain_field(globe, store, water_level=0.0)

        result = generate_features(globe, store, seed=42)
        assert len(result[COAST]) == 0
        assert len(result[FOREST]) == 0

    def test_features_stored_correctly(self, globe, populated_store):
        """After generation, tiles should have their features accessible."""
        generate_features(globe, populated_store, seed=42)

        for fid in globe.faces:
            features = get_features(populated_store, fid)
            for f in features:
                assert f in FEATURE_TYPES, f"Unknown feature: {f}"

    def test_full_pipeline_integration(self, globe, store):
        """End-to-end: elevation → temperature → moisture → terrain → features."""
        noise_fn = lambda x, y, z: fbm_3d(x, y, z, seed=7, octaves=4)
        sample_noise_field_3d(globe, store, "elevation", noise_fn)
        normalize_field(store, "elevation", lo=0.0, hi=1.0)

        generate_temperature_field(globe, store, 0.5)
        generate_moisture_field(globe, store, 0.5, water_level=0.3)
        generate_terrain_field(globe, store, water_level=0.3)
        result = generate_features(globe, store, seed=7)

        # Should have at least some coast tiles (mixed ocean/land).
        has_ocean = any(store.get(fid, "terrain") == OCEAN for fid in globe.faces)
        has_land = any(store.get(fid, "terrain") != OCEAN for fid in globe.faces)
        if has_ocean and has_land:
            assert len(result[COAST]) > 0
