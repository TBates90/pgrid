"""Tests for Phase 10C — Enhanced colour ramps & biome rendering (detail_render.py).

Covers:
- BiomeConfig defaults and custom values
- detail_elevation_to_colour returns valid RGB for all elevation values
- Hillshade brightness values in [0, 1]
- Water colour appears below water_level
- Snow appears above snow_line
- Vegetation noise varies between faces (not uniform)
- render_detail_texture_enhanced produces a PNG file
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from polygrid.tile_data import FieldDef, TileDataStore, TileSchema

from polygrid.detail_render import (
    BiomeConfig,
    detail_elevation_to_colour,
    _detail_hillshade,
    _lerp_ramp,
    _RAMP_DETAIL_SATELLITE,
    render_detail_texture_enhanced,
)

try:
    from polygrid.globe import build_globe_grid, _HAS_MODELS
    _skip = not _HAS_MODELS
except ImportError:
    _skip = True

needs_models = pytest.mark.skipif(_skip, reason="models library not installed")


def _make_detail_grid_with_terrain():
    """Build one detail grid with terrain for testing rendering."""
    from polygrid.globe import build_globe_grid
    from polygrid.mountains import MountainConfig, generate_mountains
    from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
    from polygrid.detail_terrain import generate_all_detail_terrain

    grid = build_globe_grid(3)
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)
    generate_mountains(grid, store, MountainConfig(seed=42))

    spec = TileDetailSpec(detail_rings=2)
    coll = DetailGridCollection.build(grid, spec)
    generate_all_detail_terrain(coll, grid, store, spec, seed=42)

    fid = coll.face_ids[0]
    detail_grid, detail_store = coll.get(fid)
    return detail_grid, detail_store


# ═══════════════════════════════════════════════════════════════════
# BiomeConfig
# ═══════════════════════════════════════════════════════════════════

class TestBiomeConfig:
    def test_defaults(self):
        b = BiomeConfig()
        assert b.base_ramp == "detail_satellite"
        assert 0 <= b.vegetation_density <= 1
        assert 0 <= b.rock_exposure <= 1
        assert 0 <= b.snow_line <= 1
        assert 0 <= b.water_level <= 1
        assert 0 <= b.moisture <= 1
        assert 0 <= b.hillshade_strength <= 1

    def test_custom(self):
        b = BiomeConfig(snow_line=0.9, water_level=0.2, moisture=0.8)
        assert b.snow_line == 0.9
        assert b.water_level == 0.2
        assert b.moisture == 0.8

    def test_frozen(self):
        b = BiomeConfig()
        with pytest.raises(AttributeError):
            b.snow_line = 0.5  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════
# Colour ramp interpolation
# ═══════════════════════════════════════════════════════════════════

class TestLerpRamp:
    def test_at_start(self):
        r, g, b = _lerp_ramp(_RAMP_DETAIL_SATELLITE, 0.0)
        assert 0 <= r <= 1
        assert 0 <= g <= 1
        assert 0 <= b <= 1

    def test_at_end(self):
        r, g, b = _lerp_ramp(_RAMP_DETAIL_SATELLITE, 1.0)
        assert 0 <= r <= 1
        assert 0 <= g <= 1
        assert 0 <= b <= 1

    def test_midpoint(self):
        r, g, b = _lerp_ramp(_RAMP_DETAIL_SATELLITE, 0.5)
        assert 0 <= r <= 1
        assert 0 <= g <= 1
        assert 0 <= b <= 1

    def test_clamped_below_zero(self):
        r, g, b = _lerp_ramp(_RAMP_DETAIL_SATELLITE, -0.5)
        assert 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1

    def test_clamped_above_one(self):
        r, g, b = _lerp_ramp(_RAMP_DETAIL_SATELLITE, 1.5)
        assert 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1


# ═══════════════════════════════════════════════════════════════════
# detail_elevation_to_colour
# ═══════════════════════════════════════════════════════════════════

class TestDetailElevationToColour:
    @pytest.mark.parametrize("elev", [0.0, 0.05, 0.12, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0])
    def test_valid_rgb_at_various_elevations(self, elev):
        biome = BiomeConfig()
        r, g, b = detail_elevation_to_colour(elev, biome)
        assert 0 <= r <= 1, f"r={r} out of range at elev={elev}"
        assert 0 <= g <= 1, f"g={g} out of range at elev={elev}"
        assert 0 <= b <= 1, f"b={b} out of range at elev={elev}"

    def test_water_colour_below_water_level(self):
        biome = BiomeConfig(water_level=0.2)
        r_water, g_water, b_water = detail_elevation_to_colour(0.05, biome)
        r_land, g_land, b_land = detail_elevation_to_colour(0.4, biome)
        # Water should be more blue than land (higher b relative to r, g)
        assert b_water > r_water, "Water should be bluish"

    def test_snow_above_snow_line(self):
        biome = BiomeConfig(snow_line=0.8)
        r, g, b = detail_elevation_to_colour(0.95, biome)
        # Snow should be bright (high r, g, b)
        assert r > 0.7, f"Snow r={r} not bright enough"
        assert g > 0.7, f"Snow g={g} not bright enough"
        assert b > 0.7, f"Snow b={b} not bright enough"

    def test_hillshade_darkens(self):
        biome = BiomeConfig(hillshade_strength=1.0)
        r_bright, g_bright, b_bright = detail_elevation_to_colour(
            0.5, biome, hillshade_val=1.0,
        )
        r_dark, g_dark, b_dark = detail_elevation_to_colour(
            0.5, biome, hillshade_val=0.0,
        )
        # Lower hillshade should produce darker colours
        assert r_dark < r_bright or g_dark < g_bright or b_dark < b_bright

    def test_vegetation_varies_with_position(self):
        biome = BiomeConfig(vegetation_density=0.8, moisture=0.9)
        c1 = detail_elevation_to_colour(
            0.3, biome, noise_x=0.0, noise_y=0.0, noise_seed=42,
        )
        c2 = detail_elevation_to_colour(
            0.3, biome, noise_x=5.0, noise_y=5.0, noise_seed=42,
        )
        # Different positions should produce at least slightly different colours
        assert c1 != c2, "Vegetation noise should vary by position"

    def test_no_hillshade_effect_when_strength_zero(self):
        biome = BiomeConfig(hillshade_strength=0.0)
        r1, g1, b1 = detail_elevation_to_colour(0.5, biome, hillshade_val=0.0)
        r2, g2, b2 = detail_elevation_to_colour(0.5, biome, hillshade_val=1.0)
        assert abs(r1 - r2) < 1e-10
        assert abs(g1 - g2) < 1e-10
        assert abs(b1 - b2) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# _detail_hillshade
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestDetailHillshade:
    def test_values_in_range(self):
        detail_grid, detail_store = _make_detail_grid_with_terrain()
        hs = _detail_hillshade(detail_grid, detail_store)
        for fid, val in hs.items():
            assert 0 <= val <= 1, f"Hillshade {val} out of range for {fid}"

    def test_all_faces_have_value(self):
        detail_grid, detail_store = _make_detail_grid_with_terrain()
        hs = _detail_hillshade(detail_grid, detail_store)
        assert set(hs.keys()) == set(detail_grid.faces.keys())


# ═══════════════════════════════════════════════════════════════════
# render_detail_texture_enhanced
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestRenderDetailTextureEnhanced:
    def test_produces_png(self, tmp_path):
        detail_grid, detail_store = _make_detail_grid_with_terrain()
        out = tmp_path / "test_tile.png"
        result = render_detail_texture_enhanced(
            detail_grid, detail_store, out,
            tile_size=64,
        )
        assert result.exists()
        assert result.suffix == ".png"
        assert result.stat().st_size > 0

    def test_custom_biome_produces_png(self, tmp_path):
        detail_grid, detail_store = _make_detail_grid_with_terrain()
        out = tmp_path / "test_biome.png"
        biome = BiomeConfig(snow_line=0.7, water_level=0.15)
        result = render_detail_texture_enhanced(
            detail_grid, detail_store, out, biome,
            tile_size=64,
        )
        assert result.exists()
        assert result.stat().st_size > 0
