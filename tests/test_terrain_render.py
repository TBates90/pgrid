"""Tests for terrain_render.py — Phase 7D: Elevation-aware rendering."""

from __future__ import annotations

import os
import tempfile

import pytest

from polygrid import build_pure_hex_grid
from polygrid.mountains import generate_mountains, ROLLING_HILLS
from polygrid.terrain_render import (
    elevation_to_overlay,
    hillshade,
    render_terrain,
    _lerp_ramp,
    _RAMP_TERRAIN,
    _RAMPS,
)
from polygrid.tile_data import FieldDef, TileSchema, TileDataStore


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def grid():
    return build_pure_hex_grid(rings=1)


@pytest.fixture
def store_with_elevation(grid):
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid, schema=schema)
    store.initialise_all()
    generate_mountains(grid, store, ROLLING_HILLS)
    return store


# ═══════════════════════════════════════════════════════════════════
# Colour ramp
# ═══════════════════════════════════════════════════════════════════


class TestColourRamp:
    def test_lerp_at_zero(self):
        r, g, b = _lerp_ramp(_RAMP_TERRAIN, 0.0)
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

    def test_lerp_at_one(self):
        r, g, b = _lerp_ramp(_RAMP_TERRAIN, 1.0)
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

    def test_lerp_mid(self):
        r, g, b = _lerp_ramp(_RAMP_TERRAIN, 0.5)
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

    def test_lerp_clamped_below(self):
        r, g, b = _lerp_ramp(_RAMP_TERRAIN, -1.0)
        r0, g0, b0 = _lerp_ramp(_RAMP_TERRAIN, 0.0)
        assert (r, g, b) == (r0, g0, b0)

    def test_lerp_clamped_above(self):
        r, g, b = _lerp_ramp(_RAMP_TERRAIN, 2.0)
        r1, g1, b1 = _lerp_ramp(_RAMP_TERRAIN, 1.0)
        assert (r, g, b) == (r1, g1, b1)

    def test_all_ramps_exist(self):
        assert "terrain" in _RAMPS
        assert "greyscale" in _RAMPS
        assert "satellite" in _RAMPS

    @pytest.mark.parametrize("ramp_name", list(_RAMPS.keys()))
    def test_all_ramps_produce_valid_colors(self, ramp_name):
        ramp = _RAMPS[ramp_name]
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            r, g, b = _lerp_ramp(ramp, t)
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0


# ═══════════════════════════════════════════════════════════════════
# Hillshade
# ═══════════════════════════════════════════════════════════════════


class TestHillshade:
    def test_returns_all_faces(self, grid, store_with_elevation):
        shade = hillshade(grid, store_with_elevation)
        assert set(shade.keys()) == set(grid.faces.keys())

    def test_values_in_range(self, grid, store_with_elevation):
        shade = hillshade(grid, store_with_elevation)
        for v in shade.values():
            assert 0.0 <= v <= 1.0

    def test_flat_terrain_uniform_shade(self, grid):
        """Flat elevation → all faces get the same shade."""
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid, schema=schema)
        store.initialise_all()
        store.bulk_set(grid.faces.keys(), "elevation", 0.5)

        shade = hillshade(grid, store)
        vals = list(shade.values())
        # All should be the same (or very close)
        assert max(vals) - min(vals) < 0.01

    def test_different_azimuth(self, grid, store_with_elevation):
        shade_a = hillshade(grid, store_with_elevation, azimuth=0.0)
        shade_b = hillshade(grid, store_with_elevation, azimuth=180.0)
        # Different sun direction → different shade pattern
        vals_a = list(shade_a.values())
        vals_b = list(shade_b.values())
        assert vals_a != vals_b


# ═══════════════════════════════════════════════════════════════════
# Elevation to overlay
# ═══════════════════════════════════════════════════════════════════


class TestElevationToOverlay:
    def test_overlay_has_correct_face_count(self, grid, store_with_elevation):
        overlay = elevation_to_overlay(grid, store_with_elevation)
        assert len(overlay.regions) == len(grid.faces)

    def test_overlay_kind(self, grid, store_with_elevation):
        overlay = elevation_to_overlay(grid, store_with_elevation)
        assert overlay.kind == "terrain"

    def test_colors_are_valid_rgb(self, grid, store_with_elevation):
        overlay = elevation_to_overlay(grid, store_with_elevation)
        for region in overlay.regions:
            fid = region.source_vertex_id
            color = overlay.metadata.get(f"color_{fid}")
            assert color is not None
            r, g, b = color
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0

    def test_with_hillshade(self, grid, store_with_elevation):
        shade = hillshade(grid, store_with_elevation)
        overlay = elevation_to_overlay(grid, store_with_elevation, shade=shade)
        assert len(overlay.regions) == len(grid.faces)

    @pytest.mark.parametrize("ramp_name", ["terrain", "greyscale", "satellite"])
    def test_different_ramps(self, grid, store_with_elevation, ramp_name):
        overlay = elevation_to_overlay(grid, store_with_elevation, ramp=ramp_name)
        assert len(overlay.regions) > 0

    def test_unknown_ramp_raises(self, grid, store_with_elevation):
        with pytest.raises(ValueError, match="Unknown ramp"):
            elevation_to_overlay(grid, store_with_elevation, ramp="nonexistent")


# ═══════════════════════════════════════════════════════════════════
# render_terrain (full PNG output)
# ═══════════════════════════════════════════════════════════════════


class TestRenderTerrain:
    def test_produces_png(self, grid, store_with_elevation):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            render_terrain(grid, store_with_elevation, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 1000  # not empty
        finally:
            os.unlink(path)

    def test_without_hillshade(self, grid, store_with_elevation):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            render_terrain(grid, store_with_elevation, path, hillshade_enabled=False)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_with_title(self, grid, store_with_elevation):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            render_terrain(grid, store_with_elevation, path, title="Test Mountain")
            assert os.path.exists(path)
        finally:
            os.unlink(path)
