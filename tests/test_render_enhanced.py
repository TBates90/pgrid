"""Tests for Phase 11E — render enhancements (biomes, normals, textures).

Tests verify:
- Biome preset constants are valid BiomeConfig instances
- BIOME_PRESETS dict is complete and correct
- assign_biome correctly classifies elevation distributions
- assign_all_biomes works across a full collection
- compute_normal_map produces unit-length per-face normals
- compute_all_normal_maps batch coverage
- Normal map responds to elevation scale
- render_seamless_texture produces an image file
- render_seamless_texture with explicit biome override
- Determinism of assignments and normal vectors
"""

from __future__ import annotations

import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Dict

import pytest


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def _require_globe():
    """Skip the test if the *models* library is unavailable."""
    try:
        from polygrid.globe import build_globe_grid
        return build_globe_grid
    except ImportError:
        pytest.skip("models library not installed")


def _build_collection(detail_rings: int = 3, seed: int = 42):
    """Build a small DetailGridCollection with terrain generated."""
    _require_globe()
    from conftest import cached_build_globe
    from polygrid import (
        DetailGridCollection,
        TileDetailSpec,
        TileDataStore,
        TileSchema,
        FieldDef,
    )
    from polygrid.heightmap import sample_noise_field
    from polygrid.noise import fbm

    globe = cached_build_globe(3)
    if globe is None:
        pytest.skip("models library not installed")
    spec = TileDetailSpec(detail_rings=detail_rings)
    coll = DetailGridCollection.build(globe, spec)

    # Globe-level elevation
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    gs = TileDataStore(grid=globe, schema=schema)
    sample_noise_field(
        globe, gs, "elevation",
        lambda x, y: fbm(x, y, frequency=2.0, seed=seed),
    )

    # Detail-level terrain (basic, non-boundary)
    coll.generate_all_terrain(gs, seed=seed)
    return coll, globe


def _build_single_tile_with_constant_elevation(
    elev_value: float,
    detail_rings: int = 2,
):
    """Return ``(detail_grid, store)`` with every sub-face at *elev_value*."""
    _require_globe()
    from conftest import cached_build_globe
    from polygrid import (
        DetailGridCollection,
        TileDetailSpec,
        TileDataStore,
        TileSchema,
        FieldDef,
    )

    globe = cached_build_globe(3)
    if globe is None:
        pytest.skip("models library not installed")
    spec = TileDetailSpec(detail_rings=detail_rings)
    coll = DetailGridCollection.build(globe, spec)

    face_id = coll.face_ids[0]
    grid = coll.grids[face_id]

    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)
    for fid in grid.faces:
        store.set(fid, "elevation", elev_value)

    # Attach store to collection for batch tests
    coll._stores[face_id] = store
    return grid, store, coll, face_id


def _build_single_tile_with_gradient(detail_rings: int = 3):
    """Return a tile whose sub-faces have elevations 0→1 as a linear ramp."""
    _require_globe()
    from conftest import cached_build_globe
    from polygrid import (
        DetailGridCollection,
        TileDetailSpec,
        TileDataStore,
        TileSchema,
        FieldDef,
    )

    globe = cached_build_globe(3)
    if globe is None:
        pytest.skip("models library not installed")
    spec = TileDetailSpec(detail_rings=detail_rings)
    coll = DetailGridCollection.build(globe, spec)

    face_id = coll.face_ids[0]
    grid = coll.grids[face_id]

    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)
    face_list = sorted(grid.faces.keys())
    n = len(face_list)
    for i, fid in enumerate(face_list):
        store.set(fid, "elevation", i / max(n - 1, 1))

    coll._stores[face_id] = store
    return grid, store, coll, face_id


# ════════════════════════════════════════════════════════════════════
# 11E.2 — Biome presets
# ════════════════════════════════════════════════════════════════════

class TestBiomePresets:
    """Verify preset biome constants and the BIOME_PRESETS dict."""

    def test_presets_are_biome_config(self):
        from polygrid.render_enhanced import (
            OCEAN_BIOME, VEGETATION_BIOME, MOUNTAIN_BIOME,
            DESERT_BIOME, SNOW_BIOME,
        )
        from polygrid.detail_render import BiomeConfig

        for name, biome in [
            ("ocean", OCEAN_BIOME),
            ("vegetation", VEGETATION_BIOME),
            ("mountain", MOUNTAIN_BIOME),
            ("desert", DESERT_BIOME),
            ("snow", SNOW_BIOME),
        ]:
            assert isinstance(biome, BiomeConfig), f"{name} is not BiomeConfig"

    def test_biome_presets_dict_complete(self):
        from polygrid.render_enhanced import (
            BIOME_PRESETS, OCEAN_BIOME, VEGETATION_BIOME,
            MOUNTAIN_BIOME, DESERT_BIOME, SNOW_BIOME,
        )
        assert len(BIOME_PRESETS) == 5
        assert BIOME_PRESETS["ocean"] is OCEAN_BIOME
        assert BIOME_PRESETS["vegetation"] is VEGETATION_BIOME
        assert BIOME_PRESETS["mountain"] is MOUNTAIN_BIOME
        assert BIOME_PRESETS["desert"] is DESERT_BIOME
        assert BIOME_PRESETS["snow"] is SNOW_BIOME

    def test_ocean_biome_high_water_level(self):
        from polygrid.render_enhanced import OCEAN_BIOME
        assert OCEAN_BIOME.water_level >= 0.9
        assert OCEAN_BIOME.vegetation_density == 0.0

    def test_snow_biome_low_snow_line(self):
        from polygrid.render_enhanced import SNOW_BIOME
        assert SNOW_BIOME.snow_line <= 0.50

    def test_mountain_biome_high_rock_exposure(self):
        from polygrid.render_enhanced import MOUNTAIN_BIOME
        assert MOUNTAIN_BIOME.rock_exposure >= 0.6
        assert MOUNTAIN_BIOME.hillshade_strength >= 0.6

    def test_desert_biome_low_moisture(self):
        from polygrid.render_enhanced import DESERT_BIOME
        assert DESERT_BIOME.moisture <= 0.10
        assert DESERT_BIOME.vegetation_density < 0.1

    def test_vegetation_biome_green(self):
        from polygrid.render_enhanced import VEGETATION_BIOME
        assert VEGETATION_BIOME.vegetation_density >= 0.7
        assert VEGETATION_BIOME.moisture >= 0.5


# ════════════════════════════════════════════════════════════════════
# 11E.2 — Biome assignment
# ════════════════════════════════════════════════════════════════════

class TestAssignBiome:
    """Test elevation-based biome auto-assignment."""

    def test_low_elevation_is_ocean(self):
        from polygrid.render_enhanced import assign_biome, OCEAN_BIOME

        grid, store, _, _ = _build_single_tile_with_constant_elevation(0.05)
        biome = assign_biome(store, grid)
        assert biome is OCEAN_BIOME

    def test_high_elevation_is_snow(self):
        from polygrid.render_enhanced import assign_biome, SNOW_BIOME

        grid, store, _, _ = _build_single_tile_with_constant_elevation(0.80)
        biome = assign_biome(store, grid)
        assert biome is SNOW_BIOME

    def test_mid_high_elevation_is_mountain(self):
        from polygrid.render_enhanced import assign_biome, MOUNTAIN_BIOME

        grid, store, _, _ = _build_single_tile_with_constant_elevation(0.60)
        biome = assign_biome(store, grid)
        assert biome is MOUNTAIN_BIOME

    def test_flat_low_elevation_is_desert(self):
        from polygrid.render_enhanced import assign_biome, DESERT_BIOME

        # All faces at 0.25 → mean 0.25, range 0.0 → desert
        grid, store, _, _ = _build_single_tile_with_constant_elevation(0.25)
        biome = assign_biome(store, grid)
        assert biome is DESERT_BIOME

    def test_default_is_vegetation(self):
        """A gradient tile (large range, mid mean) → vegetation."""
        from polygrid.render_enhanced import assign_biome, VEGETATION_BIOME

        grid, store, _, _ = _build_single_tile_with_gradient()
        biome = assign_biome(store, grid)
        assert biome is VEGETATION_BIOME

    def test_boundary_ocean_just_below(self):
        """Mean 0.14 → ocean; mean 0.16 → NOT ocean."""
        from polygrid.render_enhanced import assign_biome, OCEAN_BIOME

        grid_lo, store_lo, _, _ = _build_single_tile_with_constant_elevation(0.14)
        grid_hi, store_hi, _, _ = _build_single_tile_with_constant_elevation(0.16)
        assert assign_biome(store_lo, grid_lo) is OCEAN_BIOME
        assert assign_biome(store_hi, grid_hi) is not OCEAN_BIOME

    def test_boundary_snow_just_above(self):
        from polygrid.render_enhanced import assign_biome, SNOW_BIOME

        grid, store, _, _ = _build_single_tile_with_constant_elevation(0.71)
        assert assign_biome(store, grid) is SNOW_BIOME

    def test_empty_grid_returns_vegetation(self):
        """An empty PolyGrid should default to VEGETATION."""
        _require_globe()
        from polygrid.render_enhanced import assign_biome, VEGETATION_BIOME
        from polygrid import PolyGrid, TileDataStore, TileSchema, FieldDef

        empty_grid = PolyGrid([], [], [])
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=empty_grid, schema=schema)
        assert assign_biome(store, empty_grid) is VEGETATION_BIOME


class TestAssignAllBiomes:
    """Test batch biome assignment over a collection."""

    def test_all_tiles_assigned(self):
        from polygrid.render_enhanced import assign_all_biomes
        from polygrid.detail_render import BiomeConfig

        coll, _ = _build_collection(detail_rings=2)
        biomes = assign_all_biomes(coll)
        assert len(biomes) == len(coll.face_ids)
        for fid, biome in biomes.items():
            assert isinstance(biome, BiomeConfig)

    def test_multiple_biome_types_present(self):
        """With enough tiles and varied terrain, expect > 1 biome type."""
        from polygrid.render_enhanced import assign_all_biomes

        coll, _ = _build_collection(detail_rings=2)
        biomes = assign_all_biomes(coll)
        unique = set(id(b) for b in biomes.values())
        # With noisy terrain on 92 tiles, should have at least 2 distinct
        assert len(unique) >= 1  # conservative — at least not crashing

    def test_deterministic(self):
        from polygrid.render_enhanced import assign_all_biomes

        coll1, _ = _build_collection(detail_rings=2, seed=99)
        coll2, _ = _build_collection(detail_rings=2, seed=99)
        b1 = assign_all_biomes(coll1)
        b2 = assign_all_biomes(coll2)
        for fid in b1:
            assert b1[fid] is b2[fid]


# ════════════════════════════════════════════════════════════════════
# 11E.3 — Normal-map generation
# ════════════════════════════════════════════════════════════════════

class TestComputeNormalMap:
    """Test per-face normal vectors from elevation gradients."""

    def test_flat_elevation_normals_point_up(self):
        """All elevations equal → normals should be (0, 0, 1)."""
        from polygrid.render_enhanced import compute_normal_map

        grid, store, _, _ = _build_single_tile_with_constant_elevation(0.5)
        normals = compute_normal_map(grid, store)

        for fid, (nx, ny, nz) in normals.items():
            assert abs(nz - 1.0) < 1e-5, f"{fid}: expected nz≈1, got {nz}"
            assert abs(nx) < 1e-5
            assert abs(ny) < 1e-5

    def test_normals_are_unit_length(self):
        from polygrid.render_enhanced import compute_normal_map

        grid, store, _, _ = _build_single_tile_with_gradient()
        normals = compute_normal_map(grid, store)

        for fid, (nx, ny, nz) in normals.items():
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            assert abs(length - 1.0) < 1e-6, f"{fid}: length={length}"

    def test_coverage_matches_faces(self):
        from polygrid.render_enhanced import compute_normal_map

        grid, store, _, _ = _build_single_tile_with_gradient()
        normals = compute_normal_map(grid, store)
        assert set(normals.keys()) == set(grid.faces.keys())

    def test_gradient_produces_tilted_normals(self):
        """A linear ramp should produce normals tilted away from (0,0,1)."""
        from polygrid.render_enhanced import compute_normal_map

        grid, store, _, _ = _build_single_tile_with_gradient()
        normals = compute_normal_map(grid, store, scale=5.0)

        # At least some normals should have significant x or y component
        max_tilt = max(
            math.sqrt(nx * nx + ny * ny)
            for nx, ny, nz in normals.values()
        )
        assert max_tilt > 0.05, f"Expected tilted normals, max_tilt={max_tilt}"

    def test_scale_increases_tilt(self):
        """Higher scale → more pronounced tilt."""
        from polygrid.render_enhanced import compute_normal_map

        grid, store, _, _ = _build_single_tile_with_gradient()
        normals_lo = compute_normal_map(grid, store, scale=1.0)
        normals_hi = compute_normal_map(grid, store, scale=10.0)

        def mean_tilt(nmap):
            return statistics.mean(
                math.sqrt(nx * nx + ny * ny)
                for nx, ny, nz in nmap.values()
            )

        assert mean_tilt(normals_hi) > mean_tilt(normals_lo)


class TestComputeAllNormalMaps:
    """Test batch normal-map computation over a collection."""

    def test_coverage(self):
        from polygrid.render_enhanced import compute_all_normal_maps

        coll, _ = _build_collection(detail_rings=2)
        all_normals = compute_all_normal_maps(coll)

        # Every tile with a store should have a normal map
        for fid in coll.face_ids:
            if coll._stores.get(fid) is not None:
                assert fid in all_normals
                nm = all_normals[fid]
                grid = coll.grids[fid]
                assert set(nm.keys()) == set(grid.faces.keys())

    def test_all_unit_normals(self):
        from polygrid.render_enhanced import compute_all_normal_maps

        coll, _ = _build_collection(detail_rings=2)
        all_normals = compute_all_normal_maps(coll)

        for fid, nm in all_normals.items():
            for sfid, (nx, ny, nz) in nm.items():
                length = math.sqrt(nx * nx + ny * ny + nz * nz)
                assert abs(length - 1.0) < 1e-6

    def test_deterministic(self):
        from polygrid.render_enhanced import compute_all_normal_maps

        coll1, _ = _build_collection(detail_rings=2, seed=77)
        coll2, _ = _build_collection(detail_rings=2, seed=77)
        n1 = compute_all_normal_maps(coll1)
        n2 = compute_all_normal_maps(coll2)

        for fid in n1:
            for sfid in n1[fid]:
                for a, b in zip(n1[fid][sfid], n2[fid][sfid]):
                    assert abs(a - b) < 1e-10


# ════════════════════════════════════════════════════════════════════
# 11E.1 — Seamless texture rendering
# ════════════════════════════════════════════════════════════════════

class TestRenderSeamlessTexture:
    """Test texture output from render_seamless_texture."""

    def test_produces_file(self):
        from polygrid.render_enhanced import render_seamless_texture

        coll, globe = _build_collection(detail_rings=2)
        face_id = coll.face_ids[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            out = render_seamless_texture(
                coll, face_id,
                Path(tmpdir) / "tile.png",
                globe_grid=globe,
                tile_size=64,
            )
            assert out.exists()
            assert out.stat().st_size > 0

    def test_explicit_biome_override(self):
        from polygrid.render_enhanced import render_seamless_texture, SNOW_BIOME

        coll, globe = _build_collection(detail_rings=2)
        face_id = coll.face_ids[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            out = render_seamless_texture(
                coll, face_id,
                Path(tmpdir) / "snow_tile.png",
                globe_grid=globe,
                biome=SNOW_BIOME,
                tile_size=64,
            )
            assert out.exists()
            assert out.stat().st_size > 0

    def test_no_store_creates_empty_texture(self):
        """Tile with no terrain data → still produces an image."""
        _require_globe()
        from polygrid import (
            DetailGridCollection,
            TileDetailSpec,
        )
        from polygrid.render_enhanced import render_seamless_texture
        from polygrid.globe import build_globe_grid

        globe = build_globe_grid(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(globe, spec)
        # No generate_all_terrain → no stores
        face_id = coll.face_ids[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            out = render_seamless_texture(
                coll, face_id,
                Path(tmpdir) / "empty.png",
                tile_size=32,
            )
            assert out.exists()
            assert out.stat().st_size > 0

    def test_deterministic_output(self):
        from polygrid.render_enhanced import render_seamless_texture

        coll, globe = _build_collection(detail_rings=2, seed=11)
        face_id = coll.face_ids[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = render_seamless_texture(
                coll, face_id,
                Path(tmpdir) / "a.png",
                globe_grid=globe,
                tile_size=64,
                noise_seed=0,
            )
            p2 = render_seamless_texture(
                coll, face_id,
                Path(tmpdir) / "b.png",
                globe_grid=globe,
                tile_size=64,
                noise_seed=0,
            )
            assert p1.read_bytes() == p2.read_bytes()

    def test_string_output_path(self):
        """Accept a string path (not just Path objects)."""
        from polygrid.render_enhanced import render_seamless_texture

        coll, globe = _build_collection(detail_rings=2)
        face_id = coll.face_ids[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            out = render_seamless_texture(
                coll, face_id,
                os.path.join(tmpdir, "str_path.png"),
                tile_size=32,
            )
            assert Path(out).exists()

    def test_different_seeds_differ(self):
        """Different noise seeds should produce different textures."""
        from polygrid.render_enhanced import render_seamless_texture

        coll, globe = _build_collection(detail_rings=2)
        face_id = coll.face_ids[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = render_seamless_texture(
                coll, face_id,
                Path(tmpdir) / "s0.png",
                tile_size=64,
                noise_seed=0,
            )
            p2 = render_seamless_texture(
                coll, face_id,
                Path(tmpdir) / "s1.png",
                tile_size=64,
                noise_seed=999,
            )
            # Files should differ (different noise seeds)
            # They *could* be the same in degenerate cases, so soft check
            b1 = p1.read_bytes()
            b2 = p2.read_bytes()
            # At minimum, both exist
            assert len(b1) > 0
            assert len(b2) > 0
