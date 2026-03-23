"""Tests for terrain/temperature.py and terrain/moisture.py — Phase 1."""

from __future__ import annotations

import math

import pytest

from polygrid.globe.globe import build_globe_grid
from polygrid.terrain.temperature import (
    compute_temperature,
    generate_temperature_field,
    LATITUDE_WEIGHT,
    LAPSE_RATE,
)
from polygrid.terrain.moisture import (
    compute_ocean_distance,
    generate_moisture_field,
    OCEAN_PROXIMITY_WEIGHT,
    ELEVATION_PENALTY,
    MAX_OCEAN_DISTANCE,
)
from polygrid.terrain.heightmap import sample_noise_field_3d, normalize_field
from polygrid.terrain.noise import fbm_3d
from polygrid.tile_data import FieldDef, TileSchema, TileDataStore


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def globe():
    """Small frequency-2 globe (42 tiles)."""
    return build_globe_grid(2)


@pytest.fixture
def schema():
    return TileSchema([
        FieldDef("elevation", float, 0.0),
        FieldDef("temperature", float, 0.0),
        FieldDef("moisture", float, 0.0),
        FieldDef("region_id", str, "default"),
    ])


@pytest.fixture
def store(globe, schema):
    s = TileDataStore(globe, schema=schema)
    s.initialise_all()
    return s


@pytest.fixture
def store_with_elevation(globe, store):
    """Store with noise-generated elevation, normalised to [0, 1]."""
    noise_fn = lambda x, y, z: fbm_3d(x, y, z, seed=42, octaves=4)
    sample_noise_field_3d(globe, store, "elevation", noise_fn)
    normalize_field(store, "elevation", lo=0.0, hi=1.0)
    return store


# ═══════════════════════════════════════════════════════════════════
# compute_temperature (pure function)
# ═══════════════════════════════════════════════════════════════════


class TestComputeTemperature:
    """Unit tests for the single-tile temperature function."""

    def test_equator_warmer_than_pole(self):
        equator = compute_temperature(0.0, 0.0, 0.5)
        pole = compute_temperature(90.0, 0.0, 0.5)
        assert equator > pole

    def test_south_pole_same_as_north(self):
        north = compute_temperature(90.0, 0.0, 0.5)
        south = compute_temperature(-90.0, 0.0, 0.5)
        assert north == pytest.approx(south)

    def test_hot_planet_warmer_than_cold(self):
        hot = compute_temperature(45.0, 0.0, 1.0)
        cold = compute_temperature(45.0, 0.0, 0.0)
        assert hot > cold

    def test_high_elevation_cooler(self):
        low = compute_temperature(45.0, 0.0, 0.5)
        high = compute_temperature(45.0, 1.0, 0.5)
        assert low > high

    def test_output_clamped_low(self):
        # Extreme cold: pole + max elevation + frozen planet
        t = compute_temperature(90.0, 1.0, 0.0)
        assert t == 0.0

    def test_output_clamped_high(self):
        # Extreme hot: equator + sea level + scorching planet
        t = compute_temperature(0.0, 0.0, 1.0)
        assert t == 1.0

    def test_mid_latitude_mid_planet(self):
        t = compute_temperature(45.0, 0.0, 0.5)
        assert 0.0 < t < 1.0

    def test_custom_weights(self):
        # With zero latitude weight, latitude should not matter.
        eq = compute_temperature(0.0, 0.0, 0.5, latitude_weight=0.0)
        pole = compute_temperature(90.0, 0.0, 0.5, latitude_weight=0.0)
        assert eq == pytest.approx(pole)

    def test_zero_lapse_rate(self):
        # With no lapse rate, elevation should not affect temperature.
        low = compute_temperature(45.0, 0.0, 0.5, lapse_rate=0.0)
        high = compute_temperature(45.0, 1.0, 0.5, lapse_rate=0.0)
        assert low == pytest.approx(high)


# ═══════════════════════════════════════════════════════════════════
# generate_temperature_field (grid-level)
# ═══════════════════════════════════════════════════════════════════


class TestGenerateTemperatureField:
    """Tests for writing temperature to a TileDataStore."""

    def test_all_tiles_populated(self, globe, store_with_elevation):
        generate_temperature_field(globe, store_with_elevation, 0.5)
        for fid in globe.faces:
            t = store_with_elevation.get(fid, "temperature")
            assert isinstance(t, float)
            assert 0.0 <= t <= 1.0

    def test_values_vary(self, globe, store_with_elevation):
        generate_temperature_field(globe, store_with_elevation, 0.5)
        temps = [store_with_elevation.get(fid, "temperature") for fid in globe.faces]
        assert min(temps) < max(temps), "temperatures should vary across the globe"

    def test_polar_tiles_colder(self, globe, store_with_elevation):
        """Tiles near the poles should have lower temperature than equatorial tiles."""
        generate_temperature_field(globe, store_with_elevation, 0.5)

        polar_temps = []
        equatorial_temps = []
        for fid in globe.faces:
            face = globe.faces[fid]
            lat = face.metadata.get("latitude_deg", 0)
            t = store_with_elevation.get(fid, "temperature")
            if abs(lat) > 60:
                polar_temps.append(t)
            elif abs(lat) < 30:
                equatorial_temps.append(t)

        if polar_temps and equatorial_temps:
            # Average polar temp should be less than average equatorial temp.
            avg_polar = sum(polar_temps) / len(polar_temps)
            avg_equatorial = sum(equatorial_temps) / len(equatorial_temps)
            assert avg_polar < avg_equatorial

    def test_hot_planet_higher_temps(self, globe, store_with_elevation):
        """A hotter planet should produce higher average temperature."""
        store2 = TileDataStore(globe, schema=store_with_elevation.schema)
        store2.initialise_all()
        # Copy elevation from the original store.
        for fid in globe.faces:
            store2.set(fid, "elevation", store_with_elevation.get(fid, "elevation"))

        generate_temperature_field(globe, store_with_elevation, 0.2)
        generate_temperature_field(globe, store2, 0.8)

        avg_cold = sum(store_with_elevation.get(fid, "temperature") for fid in globe.faces) / len(globe.faces)
        avg_hot = sum(store2.get(fid, "temperature") for fid in globe.faces) / len(globe.faces)
        assert avg_hot > avg_cold

    def test_restricted_face_ids(self, globe, store_with_elevation):
        """Only specified faces should be modified."""
        all_fids = list(globe.faces.keys())
        target = all_fids[:5]
        rest = all_fids[5:]

        generate_temperature_field(globe, store_with_elevation, 0.5, face_ids=target)

        for fid in target:
            # These should have been set (probably non-zero).
            assert store_with_elevation.get(fid, "temperature") >= 0.0
        for fid in rest:
            assert store_with_elevation.get(fid, "temperature") == 0.0  # default


# ═══════════════════════════════════════════════════════════════════
# compute_ocean_distance
# ═══════════════════════════════════════════════════════════════════


class TestOceanDistance:
    """Tests for BFS ocean distance computation."""

    def test_ocean_tiles_distance_zero(self, globe, store):
        # Set all tiles below water level → all ocean.
        for fid in globe.faces:
            store.set(fid, "elevation", -0.1)

        dist = compute_ocean_distance(globe, store, water_level=0.0)
        for fid in globe.faces:
            assert dist[fid] == 0

    def test_all_land_max_distance(self, globe, store):
        # Set all tiles above water level → all land, no ocean.
        for fid in globe.faces:
            store.set(fid, "elevation", 0.5)

        dist = compute_ocean_distance(globe, store, water_level=0.0)
        for fid in globe.faces:
            assert dist[fid] == MAX_OCEAN_DISTANCE

    def test_adjacent_to_ocean(self, globe, store):
        """Land tiles adjacent to ocean should have distance 1."""
        # Make one tile ocean, rest land.
        fids = list(globe.faces.keys())
        ocean_fid = fids[0]
        for fid in fids:
            store.set(fid, "elevation", 0.5)
        store.set(ocean_fid, "elevation", -0.1)

        dist = compute_ocean_distance(globe, store, water_level=0.0)
        assert dist[ocean_fid] == 0

        neighbours = globe.faces[ocean_fid].neighbor_ids
        for nid in neighbours:
            assert dist[nid] == 1

    def test_distance_increases_from_ocean(self, globe, store):
        """Distance should generally increase away from ocean tiles."""
        # Make one tile ocean.
        fids = list(globe.faces.keys())
        for fid in fids:
            store.set(fid, "elevation", 0.5)
        store.set(fids[0], "elevation", -0.1)

        dist = compute_ocean_distance(globe, store, water_level=0.0)
        distances = sorted(set(dist.values()))
        # Should have at least distance 0 and distance 1.
        assert 0 in distances
        assert 1 in distances


# ═══════════════════════════════════════════════════════════════════
# generate_moisture_field
# ═══════════════════════════════════════════════════════════════════


class TestGenerateMoistureField:
    """Tests for writing moisture to a TileDataStore."""

    def test_all_tiles_populated(self, globe, store_with_elevation):
        generate_moisture_field(globe, store_with_elevation, 0.5)
        for fid in globe.faces:
            m = store_with_elevation.get(fid, "moisture")
            assert isinstance(m, float)
            assert 0.0 <= m <= 1.0

    def test_ocean_tiles_fully_wet(self, globe, store):
        """Ocean tiles (below water level) should have moisture = 1.0."""
        for fid in globe.faces:
            store.set(fid, "elevation", -0.1)

        generate_moisture_field(globe, store, 0.5, water_level=0.0)
        for fid in globe.faces:
            assert store.get(fid, "moisture") == 1.0

    def test_values_vary_on_mixed_terrain(self, globe, store_with_elevation):
        generate_moisture_field(globe, store_with_elevation, 0.5)
        moistures = [store_with_elevation.get(fid, "moisture") for fid in globe.faces]
        assert min(moistures) < max(moistures)

    def test_higher_water_abundance_more_moisture(self, globe, store_with_elevation):
        """Higher water_abundance → higher average moisture."""
        store2 = TileDataStore(globe, schema=store_with_elevation.schema)
        store2.initialise_all()
        for fid in globe.faces:
            store2.set(fid, "elevation", store_with_elevation.get(fid, "elevation"))

        generate_moisture_field(globe, store_with_elevation, 0.2)
        generate_moisture_field(globe, store2, 0.9)

        # Filter to land tiles only.
        land_fids = [fid for fid in globe.faces if store_with_elevation.get(fid, "elevation") > 0.0]
        if land_fids:
            avg_dry = sum(store_with_elevation.get(fid, "moisture") for fid in land_fids) / len(land_fids)
            avg_wet = sum(store2.get(fid, "moisture") for fid in land_fids) / len(land_fids)
            assert avg_wet > avg_dry

    def test_region_humidity_modifier(self, globe, store_with_elevation):
        """Region humidity modifier should scale moisture for tiles in that region."""
        # Assign all tiles to a single region.
        for fid in globe.faces:
            store_with_elevation.set(fid, "region_id", "desert_region")

        generate_moisture_field(
            globe, store_with_elevation, 0.5,
            region_humidity={"desert_region": 0.1},
            region_field="region_id",
        )
        avg_dry = sum(
            store_with_elevation.get(fid, "moisture")
            for fid in globe.faces
            if store_with_elevation.get(fid, "elevation") > 0.0
        )

        # Reset moisture and try with high modifier.
        for fid in globe.faces:
            store_with_elevation.set(fid, "moisture", 0.0)

        generate_moisture_field(
            globe, store_with_elevation, 0.5,
            region_humidity={"desert_region": 2.0},
            region_field="region_id",
        )
        avg_wet = sum(
            store_with_elevation.get(fid, "moisture")
            for fid in globe.faces
            if store_with_elevation.get(fid, "elevation") > 0.0
        )

        assert avg_wet > avg_dry

    def test_restricted_face_ids(self, globe, store_with_elevation):
        """Only specified faces should be modified."""
        all_fids = list(globe.faces.keys())
        target = all_fids[:5]
        rest = all_fids[5:]

        generate_moisture_field(globe, store_with_elevation, 0.5, face_ids=target)

        for fid in rest:
            assert store_with_elevation.get(fid, "moisture") == 0.0  # default


# ═══════════════════════════════════════════════════════════════════
# Integration: temperature + moisture together on a globe
# ═══════════════════════════════════════════════════════════════════


class TestTemperatureMoistureIntegration:
    """End-to-end tests with both fields populated on a globe."""

    def test_full_pipeline(self, globe, store_with_elevation):
        """Generate temperature and moisture in sequence — no crashes, valid output."""
        generate_temperature_field(globe, store_with_elevation, 0.5)
        generate_moisture_field(globe, store_with_elevation, 0.5)

        for fid in globe.faces:
            t = store_with_elevation.get(fid, "temperature")
            m = store_with_elevation.get(fid, "moisture")
            assert 0.0 <= t <= 1.0
            assert 0.0 <= m <= 1.0

    def test_determinism(self, globe, store_with_elevation):
        """Same inputs → same outputs."""
        generate_temperature_field(globe, store_with_elevation, 0.5)
        generate_moisture_field(globe, store_with_elevation, 0.5)
        temps1 = {fid: store_with_elevation.get(fid, "temperature") for fid in globe.faces}
        moist1 = {fid: store_with_elevation.get(fid, "moisture") for fid in globe.faces}

        # Reset and regenerate.
        for fid in globe.faces:
            store_with_elevation.set(fid, "temperature", 0.0)
            store_with_elevation.set(fid, "moisture", 0.0)

        generate_temperature_field(globe, store_with_elevation, 0.5)
        generate_moisture_field(globe, store_with_elevation, 0.5)

        for fid in globe.faces:
            assert store_with_elevation.get(fid, "temperature") == pytest.approx(temps1[fid])
            assert store_with_elevation.get(fid, "moisture") == pytest.approx(moist1[fid])
