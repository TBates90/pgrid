"""Tests for Phase 10D — Texture atlas & UV mapping (texture_pipeline.py).

Covers:
- build_detail_atlas produces atlas image and UV layout
- Atlas UV layout covers all tiles
- UVs within [0, 1] range
- compute_tile_uvs maps into atlas slot correctly
- build_textured_tile_mesh (models-dependent)
- build_textured_globe_meshes (models-dependent)
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
from polygrid.detail_terrain import generate_all_detail_terrain
from polygrid.texture_pipeline import compute_tile_uvs

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


def _make_collection_with_terrain(frequency=3, detail_rings=2, seed=42):
    from conftest import cached_build_globe
    from polygrid.mountains import MountainConfig, generate_mountains

    grid = cached_build_globe(frequency)
    if grid is None:
        pytest.skip("models library not installed")
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)
    generate_mountains(grid, store, MountainConfig(seed=seed))

    spec = TileDetailSpec(detail_rings=detail_rings)
    coll = DetailGridCollection.build(grid, spec)
    generate_all_detail_terrain(coll, grid, store, spec, seed=seed)
    return grid, store, coll


# ═══════════════════════════════════════════════════════════════════
# compute_tile_uvs
# ═══════════════════════════════════════════════════════════════════

class TestComputeTileUvs:
    def test_maps_into_atlas_slot(self):
        tile_uvs = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        slot = (0.25, 0.25, 0.75, 0.75)
        mapped = compute_tile_uvs(tile_uvs, slot)
        assert len(mapped) == 3

        # (0,0) maps to (0.25, 0.25)
        assert abs(mapped[0][0] - 0.25) < 1e-10
        assert abs(mapped[0][1] - 0.25) < 1e-10

        # (1,0) maps to (0.75, 0.25)
        assert abs(mapped[1][0] - 0.75) < 1e-10
        assert abs(mapped[1][1] - 0.25) < 1e-10

        # (0.5, 1.0) maps to (0.5, 0.75)
        assert abs(mapped[2][0] - 0.5) < 1e-10
        assert abs(mapped[2][1] - 0.75) < 1e-10

    def test_uvs_within_slot_range(self):
        tile_uvs = [(0.0, 0.0), (1.0, 1.0), (0.3, 0.7)]
        slot = (0.1, 0.2, 0.9, 0.8)
        mapped = compute_tile_uvs(tile_uvs, slot)
        for u, v in mapped:
            assert 0.1 - 1e-10 <= u <= 0.9 + 1e-10
            assert 0.2 - 1e-10 <= v <= 0.8 + 1e-10

    def test_clamps_out_of_range_input(self):
        tile_uvs = [(-0.1, 1.2)]
        slot = (0.0, 0.0, 1.0, 1.0)
        mapped = compute_tile_uvs(tile_uvs, slot)
        u, v = mapped[0]
        assert 0.0 <= u <= 1.0
        assert 0.0 <= v <= 1.0

    def test_full_unit_slot(self):
        tile_uvs = [(0.5, 0.5)]
        slot = (0.0, 0.0, 1.0, 1.0)
        mapped = compute_tile_uvs(tile_uvs, slot)
        assert abs(mapped[0][0] - 0.5) < 1e-10
        assert abs(mapped[0][1] - 0.5) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# build_detail_atlas
# ═══════════════════════════════════════════════════════════════════

@needs_models
@needs_pil
class TestBuildDetailAtlas:
    def test_atlas_exists_and_has_size(self, tmp_path):
        from polygrid.texture_pipeline import build_detail_atlas

        _, _, coll = _make_collection_with_terrain(3, detail_rings=2)
        atlas_path, uv_layout = build_detail_atlas(
            coll, output_dir=tmp_path, tile_size=32,
        )
        assert atlas_path.exists()
        assert atlas_path.stat().st_size > 0

    def test_uv_layout_covers_all_tiles(self, tmp_path):
        from polygrid.texture_pipeline import build_detail_atlas

        _, _, coll = _make_collection_with_terrain(3, detail_rings=2)
        _, uv_layout = build_detail_atlas(
            coll, output_dir=tmp_path, tile_size=32,
        )
        assert set(uv_layout.keys()) == set(coll.face_ids)

    def test_uvs_in_valid_range(self, tmp_path):
        from polygrid.texture_pipeline import build_detail_atlas

        _, _, coll = _make_collection_with_terrain(3, detail_rings=2)
        _, uv_layout = build_detail_atlas(
            coll, output_dir=tmp_path, tile_size=32,
        )
        for fid, (u_min, v_min, u_max, v_max) in uv_layout.items():
            assert 0.0 <= u_min <= 1.0, f"u_min={u_min} for {fid}"
            assert 0.0 <= v_min <= 1.0, f"v_min={v_min} for {fid}"
            assert 0.0 <= u_max <= 1.0, f"u_max={u_max} for {fid}"
            assert 0.0 <= v_max <= 1.0, f"v_max={v_max} for {fid}"
            assert u_max > u_min
            assert v_max > v_min

    def test_atlas_image_dimensions(self, tmp_path):
        from polygrid.texture_pipeline import build_detail_atlas

        _, _, coll = _make_collection_with_terrain(3, detail_rings=2)
        tile_size = 32
        gutter = 4  # default gutter
        atlas_path, _ = build_detail_atlas(
            coll, output_dir=tmp_path, tile_size=tile_size,
        )
        img = Image.open(atlas_path)
        w, h = img.size
        # Atlas should be a multiple of slot_size (tile_size + 2*gutter)
        slot_size = tile_size + 2 * gutter
        assert w % slot_size == 0
        assert h % slot_size == 0
        # Should have enough slots for all tiles
        slots = (w // slot_size) * (h // slot_size)
        assert slots >= len(coll.face_ids)

    def test_individual_tiles_created(self, tmp_path):
        from polygrid.texture_pipeline import build_detail_atlas

        _, _, coll = _make_collection_with_terrain(3, detail_rings=2)
        build_detail_atlas(coll, output_dir=tmp_path, tile_size=32)
        # Individual tile PNGs should exist
        tile_files = list(tmp_path.glob("tile_*.png"))
        assert len(tile_files) == len(coll.face_ids)
