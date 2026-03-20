# TODO REMOVE — Tests dead Phase 13 code (texture_pipeline.py / render_enhanced.py).
"""Tests for Phase 13A (full-coverage textures) and 13B (atlas gutters).

Verifies that:
- Tile textures have no black corners (13A)
- Atlas slots have gutter pixels filled with edge colours (13B)
- UV layout accounts for gutter inset (13B)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

from PIL import Image


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _make_collection_with_terrain(frequency: int = 3, detail_rings: int = 2):
    """Build a globe + detail collection with terrain data."""
    from conftest import cached_build_globe
    from polygrid.mountains import MountainConfig, generate_mountains
    from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
    from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
    from polygrid.detail_terrain import generate_all_detail_terrain

    grid = cached_build_globe(frequency)
    if grid is None:
        import pytest
        pytest.skip("models library not installed")
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)
    config = MountainConfig(seed=42)
    generate_mountains(grid, store, config)

    spec = TileDetailSpec(detail_rings=detail_rings)
    coll = DetailGridCollection.build(grid, spec)
    generate_all_detail_terrain(coll, grid, store, spec, seed=42)
    return grid, store, coll


# ═══════════════════════════════════════════════════════════════════
# 13A — Full-coverage tile textures
# ═══════════════════════════════════════════════════════════════════

class TestFullCoverageTileTextures:
    """13A: Tile textures should have no black corners."""

    @pytest.fixture(scope="class")
    def rendered_tile(self, tmp_path_factory):
        """Render a single tile texture and return the image array."""
        tmp = tmp_path_factory.mktemp("tile13a")
        _, _, coll = _make_collection_with_terrain(3, detail_rings=2)
        fid = coll.face_ids[0]
        grid, store = coll.get(fid)

        from polygrid.detail_render import render_detail_texture_enhanced
        path = tmp / "tile.png"
        render_detail_texture_enhanced(grid, store, path, tile_size=64)
        img = Image.open(str(path)).convert("RGB")
        return np.array(img.resize((64, 64), Image.LANCZOS))

    @pytest.fixture(scope="class")
    def rendered_tile_fast(self, tmp_path_factory):
        """Render a single tile using the fast PIL renderer."""
        tmp = tmp_path_factory.mktemp("tile13a_fast")
        _, _, coll = _make_collection_with_terrain(3, detail_rings=2)
        fid = coll.face_ids[0]
        grid, store = coll.get(fid)

        from polygrid.detail_perf import render_detail_texture_fast
        path = tmp / "tile_fast.png"
        render_detail_texture_fast(grid, store, path, tile_size=64)
        return np.array(Image.open(str(path)).convert("RGB"))

    def test_no_black_corners_matplotlib(self, rendered_tile):
        """Matplotlib renderer: corner pixels should NOT be black."""
        arr = rendered_tile
        # Check all four corners (5x5 blocks)
        corners = [
            arr[0:5, 0:5],      # top-left
            arr[0:5, -5:],      # top-right
            arr[-5:, 0:5],      # bottom-left
            arr[-5:, -5:],      # bottom-right
        ]
        for i, corner in enumerate(corners):
            mean_brightness = corner.astype(float).sum(axis=2).mean()
            assert mean_brightness > 20, (
                f"Corner {i} is black (brightness={mean_brightness:.1f}). "
                "Background should be terrain-coloured."
            )

    def test_no_black_corners_pil(self, rendered_tile_fast):
        """PIL renderer: corner pixels should NOT be black."""
        arr = rendered_tile_fast
        corners = [
            arr[0:5, 0:5],
            arr[0:5, -5:],
            arr[-5:, 0:5],
            arr[-5:, -5:],
        ]
        for i, corner in enumerate(corners):
            mean_brightness = corner.astype(float).sum(axis=2).mean()
            assert mean_brightness > 20, (
                f"Corner {i} is black (brightness={mean_brightness:.1f}). "
                "Background should be terrain-coloured."
            )

    def test_low_black_percentage_matplotlib(self, rendered_tile):
        """Matplotlib: less than 5% of pixels should be very dark."""
        arr = rendered_tile.astype(float)
        brightness = arr.sum(axis=2)
        dark_count = np.sum(brightness < 15)
        total = arr.shape[0] * arr.shape[1]
        pct = 100 * dark_count / total
        assert pct < 5.0, (
            f"{pct:.1f}% of tile pixels are black (expected <5%)"
        )

    def test_low_black_percentage_pil(self, rendered_tile_fast):
        """PIL: less than 5% of pixels should be very dark."""
        arr = rendered_tile_fast.astype(float)
        brightness = arr.sum(axis=2)
        dark_count = np.sum(brightness < 15)
        total = arr.shape[0] * arr.shape[1]
        pct = 100 * dark_count / total
        assert pct < 5.0, (
            f"{pct:.1f}% of tile pixels are black (expected <5%)"
        )

    def test_background_matches_average_colour_matplotlib(self, rendered_tile):
        """Corner colour should roughly match the tile's average colour."""
        arr = rendered_tile.astype(float)
        # Average colour of the whole tile
        avg = arr.mean(axis=(0, 1))
        # Corner average
        corner = arr[0:3, 0:3].mean(axis=(0, 1))
        # They should be in the same ballpark (within 80 per channel)
        diff = np.abs(avg - corner).max()
        assert diff < 80, (
            f"Corner colour {corner} too far from average {avg} (diff={diff:.0f})"
        )


# ═══════════════════════════════════════════════════════════════════
# 13B — Atlas gutter system
# ═══════════════════════════════════════════════════════════════════

class TestAtlasGutter:
    """13B: Atlas should have gutter pixels that prevent bleed."""

    @pytest.fixture(scope="class")
    def atlas_data(self, tmp_path_factory):
        """Build an atlas with gutters and return (atlas_arr, uv_layout, tile_size, gutter)."""
        tmp = tmp_path_factory.mktemp("atlas13b")
        _, _, coll = _make_collection_with_terrain(3, detail_rings=2)

        from polygrid.texture_pipeline import build_detail_atlas
        tile_size = 32
        gutter = 4
        atlas_path, uv_layout = build_detail_atlas(
            coll, output_dir=tmp, tile_size=tile_size, gutter=gutter,
        )
        arr = np.array(Image.open(str(atlas_path)).convert("RGB"))
        return arr, uv_layout, tile_size, gutter

    def test_atlas_no_black_slots(self, atlas_data):
        """No atlas slot should be predominantly black."""
        arr, uv_layout, tile_size, gutter = atlas_data
        slot_size = tile_size + 2 * gutter
        atlas_h = arr.shape[0]
        atlas_w = arr.shape[1]
        cols = atlas_w // slot_size
        rows = atlas_h // slot_size

        for idx, (fid, (u_min, v_min, u_max, v_max)) in enumerate(uv_layout.items()):
            col = idx % cols
            row = idx // cols
            slot = arr[
                row * slot_size:(row + 1) * slot_size,
                col * slot_size:(col + 1) * slot_size,
            ]
            brightness = slot.astype(float).sum(axis=2)
            black_pct = 100 * np.sum(brightness < 15) / slot.shape[0] / slot.shape[1]
            assert black_pct < 10, (
                f"Slot {fid} is {black_pct:.1f}% black (expected <10%)"
            )

    def test_gutter_pixels_not_black(self, atlas_data):
        """Gutter pixels around the first slot should be terrain-coloured."""
        arr, uv_layout, tile_size, gutter = atlas_data
        slot_size = tile_size + 2 * gutter

        # Check top gutter of first slot
        top_gutter = arr[0:gutter, gutter:gutter + tile_size]
        mean_brightness = top_gutter.astype(float).sum(axis=2).mean()
        assert mean_brightness > 20, (
            f"Top gutter is too dark (brightness={mean_brightness:.1f})"
        )

        # Check left gutter of first slot
        left_gutter = arr[0:slot_size, 0:gutter]
        mean_brightness = left_gutter.astype(float).sum(axis=2).mean()
        assert mean_brightness > 20, (
            f"Left gutter is too dark (brightness={mean_brightness:.1f})"
        )

    def test_gutter_matches_tile_edge(self, atlas_data):
        """Gutter should be filled with the adjacent tile edge colour."""
        arr, uv_layout, tile_size, gutter = atlas_data
        slot_size = tile_size + 2 * gutter

        # Top row of the actual tile (just below gutter)
        tile_top_row = arr[gutter, gutter:gutter + tile_size].astype(float)
        # Top gutter row (just above tile)
        gutter_row = arr[gutter - 1, gutter:gutter + tile_size].astype(float)

        # They should be identical (gutter repeats tile edge)
        np.testing.assert_array_equal(
            tile_top_row.astype(np.uint8),
            gutter_row.astype(np.uint8),
            err_msg="Gutter top row should match tile top edge"
        )

    def test_uv_layout_inset(self, atlas_data):
        """UV coordinates should map to the inner tile region, not the gutter."""
        arr, uv_layout, tile_size, gutter = atlas_data
        atlas_h, atlas_w = arr.shape[:2]
        slot_size = tile_size + 2 * gutter

        fid = list(uv_layout.keys())[0]
        u_min, v_min, u_max, v_max = uv_layout[fid]

        # UV range should correspond to tile_size, not slot_size
        u_span_px = (u_max - u_min) * atlas_w
        v_span_px = (v_max - v_min) * atlas_h

        assert abs(u_span_px - tile_size) < 1.0, (
            f"UV u-span is {u_span_px:.1f}px, expected {tile_size}px"
        )
        assert abs(v_span_px - tile_size) < 1.0, (
            f"UV v-span is {v_span_px:.1f}px, expected {tile_size}px"
        )

    def test_uv_not_at_slot_boundary(self, atlas_data):
        """UV min should be offset from the slot boundary by the gutter."""
        arr, uv_layout, tile_size, gutter = atlas_data
        atlas_h, atlas_w = arr.shape[:2]

        fid = list(uv_layout.keys())[0]
        u_min, v_min, u_max, v_max = uv_layout[fid]

        # First slot starts at pixel (gutter, gutter), not (0, 0)
        expected_u_min = gutter / atlas_w
        assert abs(u_min - expected_u_min) < 1e-6, (
            f"u_min={u_min:.6f}, expected {expected_u_min:.6f} (gutter offset)"
        )

    def test_atlas_size_accounts_for_gutter(self, atlas_data):
        """Atlas size should be columns * (tile_size + 2*gutter)."""
        arr, uv_layout, tile_size, gutter = atlas_data
        atlas_h, atlas_w = arr.shape[:2]
        slot_size = tile_size + 2 * gutter

        assert atlas_w % slot_size == 0
        assert atlas_h % slot_size == 0

    def test_zero_gutter_backwards_compatible(self, tmp_path):
        """With gutter=0, atlas should work as before."""
        _, _, coll = _make_collection_with_terrain(3, detail_rings=2)

        from polygrid.texture_pipeline import build_detail_atlas
        tile_size = 32
        atlas_path, uv_layout = build_detail_atlas(
            coll, output_dir=tmp_path, tile_size=tile_size, gutter=0,
        )
        arr = np.array(Image.open(str(atlas_path)).convert("RGB"))
        atlas_w = arr.shape[1]

        # With gutter=0, atlas_w should be a multiple of tile_size
        assert atlas_w % tile_size == 0

        # UV should start at 0.0
        fid = list(uv_layout.keys())[0]
        u_min = uv_layout[fid][0]
        assert abs(u_min) < 1e-6


# ═══════════════════════════════════════════════════════════════════
# 13B — Fast atlas gutter
# ═══════════════════════════════════════════════════════════════════

class TestFastAtlasGutter:
    """13B: The fast (PIL) atlas builder should also support gutters."""

    def test_fast_atlas_has_gutters(self, tmp_path):
        _, _, coll = _make_collection_with_terrain(3, detail_rings=2)

        from polygrid.detail_perf import build_detail_atlas_fast
        tile_size = 32
        gutter = 4
        atlas_path, uv_layout = build_detail_atlas_fast(
            coll, output_dir=tmp_path, tile_size=tile_size, gutter=gutter,
        )
        arr = np.array(Image.open(str(atlas_path)).convert("RGB"))
        slot_size = tile_size + 2 * gutter

        assert arr.shape[1] % slot_size == 0
        assert arr.shape[0] % slot_size == 0

        # UV should be inset
        fid = list(uv_layout.keys())[0]
        u_min = uv_layout[fid][0]
        expected = gutter / arr.shape[1]
        assert abs(u_min - expected) < 1e-6

    def test_fast_atlas_no_black_slots(self, tmp_path):
        _, _, coll = _make_collection_with_terrain(3, detail_rings=2)

        from polygrid.detail_perf import build_detail_atlas_fast
        tile_size = 32
        atlas_path, uv_layout = build_detail_atlas_fast(
            coll, output_dir=tmp_path, tile_size=tile_size,
        )
        arr = np.array(Image.open(str(atlas_path)).convert("RGB"))
        brightness = arr.astype(float).sum(axis=2)
        black_pct = 100 * np.sum(brightness < 15) / brightness.size
        # Only empty slots at the end should be dark
        n_empty = (arr.shape[0] // (tile_size + 8)) * (arr.shape[1] // (tile_size + 8)) - len(uv_layout)
        # Allow generous tolerance for empty trailing slots
        max_allowed = max(15, n_empty * 100 / max(1, brightness.size // ((tile_size + 8) ** 2)))
        assert black_pct < max_allowed + 15


# ═══════════════════════════════════════════════════════════════════
# Integration: end-to-end atlas → mesh → no black in texture
# ═══════════════════════════════════════════════════════════════════

class TestEndToEnd:
    """Integration: full pipeline from terrain to atlas should be seam-free."""

    def test_atlas_mostly_coloured(self, tmp_path):
        """The full atlas should have minimal dark pixels in tile slots."""
        _, _, coll = _make_collection_with_terrain(3, detail_rings=2)

        from polygrid.texture_pipeline import build_detail_atlas
        tile_size = 64
        gutter = 4
        atlas_path, uv_layout = build_detail_atlas(
            coll, output_dir=tmp_path, tile_size=tile_size, gutter=gutter,
        )
        arr = np.array(Image.open(str(atlas_path)).convert("RGB"))
        slot_size = tile_size + 2 * gutter
        cols = arr.shape[1] // slot_size
        rows = arr.shape[0] // slot_size

        # Check occupied slots only
        dark_slots = 0
        for idx, fid in enumerate(uv_layout):
            col = idx % cols
            row = idx // cols
            slot = arr[
                row * slot_size:(row + 1) * slot_size,
                col * slot_size:(col + 1) * slot_size,
            ]
            brightness = slot.astype(float).sum(axis=2)
            dark_pct = 100 * np.sum(brightness < 15) / slot.size * 3  # 3 channels
            if dark_pct > 5:
                dark_slots += 1

        assert dark_slots == 0, (
            f"{dark_slots} tile slots have >5% dark pixels"
        )
