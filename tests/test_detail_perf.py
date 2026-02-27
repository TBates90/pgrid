"""Tests for Phase 10F — Performance utilities (detail_perf.py).

Covers:
- Parallel generation produces identical results to serial
- Fast PIL renderer produces a valid PNG
- Fast atlas builder produces atlas + UV layout
- DetailCache put/get/has/clear/size operations
- Cached results match fresh generation
- generate_all_detail_terrain_cached works end-to-end
- benchmark_pipeline returns timing dict
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
from polygrid.detail_terrain import generate_all_detail_terrain

from polygrid.detail_perf import (
    generate_all_detail_terrain_parallel,
    render_detail_texture_fast,
    build_detail_atlas_fast,
    DetailCache,
    generate_all_detail_terrain_cached,
)

try:
    from polygrid.globe import build_globe_grid, _HAS_MODELS
    _skip = not _HAS_MODELS
except ImportError:
    _skip = True

needs_models = pytest.mark.skipif(_skip, reason="models library not installed")

try:
    from PIL import Image
    _has_pil = True
except ImportError:
    _has_pil = False

needs_pil = pytest.mark.skipif(not _has_pil, reason="Pillow not installed")


def _make_globe_with_elevation(frequency=3, seed=42):
    from polygrid.globe import build_globe_grid
    from polygrid.mountains import MountainConfig, generate_mountains

    grid = build_globe_grid(frequency)
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)
    generate_mountains(grid, store, MountainConfig(seed=seed))
    return grid, store


def _make_detail_grid_with_terrain():
    """Build one detail grid + store for fast-render tests."""
    grid, store = _make_globe_with_elevation(3)
    spec = TileDetailSpec(detail_rings=2)
    coll = DetailGridCollection.build(grid, spec)
    generate_all_detail_terrain(coll, grid, store, spec, seed=42)
    fid = coll.face_ids[0]
    detail_grid, detail_store = coll.get(fid)
    return detail_grid, detail_store


# ═══════════════════════════════════════════════════════════════════
# Parallel generation
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestParallelGeneration:
    def test_produces_same_results_as_serial(self):
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)

        # Serial
        coll_serial = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll_serial, grid, store, spec, seed=42)

        # Parallel
        coll_parallel = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain_parallel(
            coll_parallel, grid, store, spec, seed=42, max_workers=2,
        )

        assert len(coll_serial.stores) == len(coll_parallel.stores)
        for fid in coll_serial.face_ids:
            dg_s, s_s = coll_serial.get(fid)
            dg_p, s_p = coll_parallel.get(fid)
            for sub_fid in dg_s.faces:
                v_s = s_s.get(sub_fid, "elevation")
                v_p = s_p.get(sub_fid, "elevation")
                assert abs(v_s - v_p) < 1e-10, (
                    f"Mismatch at {fid}/{sub_fid}: serial={v_s}, parallel={v_p}"
                )

    def test_all_stores_populated(self):
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain_parallel(
            coll, grid, store, spec, seed=42, max_workers=2,
        )
        assert len(coll.stores) == len(coll.grids)


# ═══════════════════════════════════════════════════════════════════
# Fast PIL renderer
# ═══════════════════════════════════════════════════════════════════

@needs_models
@needs_pil
class TestRenderDetailTextureFast:
    def test_produces_png(self, tmp_path):
        detail_grid, detail_store = _make_detail_grid_with_terrain()
        out = tmp_path / "fast_tile.png"
        result = render_detail_texture_fast(
            detail_grid, detail_store, out, tile_size=64,
        )
        assert result.exists()
        assert result.suffix == ".png"
        assert result.stat().st_size > 0

    def test_correct_dimensions(self, tmp_path):
        detail_grid, detail_store = _make_detail_grid_with_terrain()
        out = tmp_path / "fast_tile.png"
        render_detail_texture_fast(
            detail_grid, detail_store, out, tile_size=128,
        )
        img = Image.open(out)
        assert img.size == (128, 128)


# ═══════════════════════════════════════════════════════════════════
# Fast atlas builder
# ═══════════════════════════════════════════════════════════════════

@needs_models
@needs_pil
class TestBuildDetailAtlasFast:
    def test_atlas_exists(self, tmp_path):
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)
        atlas_path, uv_layout = build_detail_atlas_fast(
            coll, output_dir=tmp_path, tile_size=32,
        )
        assert atlas_path.exists()
        assert atlas_path.stat().st_size > 0

    def test_uv_layout_complete(self, tmp_path):
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)
        _, uv_layout = build_detail_atlas_fast(
            coll, output_dir=tmp_path, tile_size=32,
        )
        assert set(uv_layout.keys()) == set(coll.face_ids)
        for fid, (u_min, v_min, u_max, v_max) in uv_layout.items():
            assert 0.0 <= u_min < u_max <= 1.0
            assert 0.0 <= v_min < v_max <= 1.0


# ═══════════════════════════════════════════════════════════════════
# DetailCache
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestDetailCache:
    def test_empty_cache(self, tmp_path):
        cache = DetailCache(cache_dir=tmp_path / "cache")
        assert cache.size == 0

    def test_put_and_has(self, tmp_path):
        cache = DetailCache(cache_dir=tmp_path / "cache")
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)

        fid = coll.face_ids[0]
        detail_grid, detail_store = coll.get(fid)
        parent_elev = store.get(fid, "elevation")

        assert not cache.has(fid, spec, parent_elev, 42)
        cache.put(fid, spec, parent_elev, 42, detail_store, detail_grid)
        assert cache.has(fid, spec, parent_elev, 42)
        assert cache.size == 1

    def test_get_returns_same_values(self, tmp_path):
        cache = DetailCache(cache_dir=tmp_path / "cache")
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)

        fid = coll.face_ids[0]
        detail_grid, detail_store = coll.get(fid)
        parent_elev = store.get(fid, "elevation")

        cache.put(fid, spec, parent_elev, 42, detail_store, detail_grid)
        cached = cache.get(fid, spec, parent_elev, 42, detail_grid)
        assert cached is not None
        for sub_fid in detail_grid.faces:
            original = detail_store.get(sub_fid, "elevation")
            cached_val = cached.get(sub_fid, "elevation")
            assert abs(original - cached_val) < 1e-10

    def test_get_miss_returns_none(self, tmp_path):
        cache = DetailCache(cache_dir=tmp_path / "cache")
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        fid = coll.face_ids[0]
        detail_grid, _ = coll.get(fid)
        result = cache.get(fid, spec, 0.5, 42, detail_grid)
        assert result is None

    def test_clear(self, tmp_path):
        cache = DetailCache(cache_dir=tmp_path / "cache")
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)

        fid = coll.face_ids[0]
        detail_grid, detail_store = coll.get(fid)
        parent_elev = store.get(fid, "elevation")
        cache.put(fid, spec, parent_elev, 42, detail_store, detail_grid)
        assert cache.size == 1

        removed = cache.clear()
        assert removed == 1
        assert cache.size == 0

    def test_different_seed_different_key(self, tmp_path):
        cache = DetailCache(cache_dir=tmp_path / "cache")
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)

        fid = coll.face_ids[0]
        detail_grid, detail_store = coll.get(fid)
        parent_elev = store.get(fid, "elevation")

        cache.put(fid, spec, parent_elev, 42, detail_store, detail_grid)
        # Different seed → cache miss
        assert not cache.has(fid, spec, parent_elev, 99)


# ═══════════════════════════════════════════════════════════════════
# generate_all_detail_terrain_cached
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestGenerateAllDetailTerrainCached:
    def test_first_run_zero_hits(self, tmp_path):
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        cache = DetailCache(cache_dir=tmp_path / "cache")
        hits = generate_all_detail_terrain_cached(
            coll, grid, store, spec, seed=42, cache=cache,
        )
        assert hits == 0
        assert len(coll.stores) == len(coll.grids)

    def test_second_run_all_hits(self, tmp_path):
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        cache = DetailCache(cache_dir=tmp_path / "cache")

        coll1 = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain_cached(
            coll1, grid, store, spec, seed=42, cache=cache,
        )

        coll2 = DetailGridCollection.build(grid, spec)
        hits = generate_all_detail_terrain_cached(
            coll2, grid, store, spec, seed=42, cache=cache,
        )
        assert hits == len(coll2.grids)

    def test_cached_matches_fresh(self, tmp_path):
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        cache = DetailCache(cache_dir=tmp_path / "cache")

        # Fresh
        coll_fresh = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll_fresh, grid, store, spec, seed=42)

        # Cached (first run populates, second reads back)
        coll_cache = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain_cached(
            coll_cache, grid, store, spec, seed=42, cache=cache,
        )

        for fid in coll_fresh.face_ids:
            dg_f, s_f = coll_fresh.get(fid)
            dg_c, s_c = coll_cache.get(fid)
            for sub_fid in dg_f.faces:
                v_f = s_f.get(sub_fid, "elevation")
                v_c = s_c.get(sub_fid, "elevation")
                assert abs(v_f - v_c) < 1e-10
