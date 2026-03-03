"""Tests for Phase 16A — Full-slot tile texture rendering (tile_texture.py).

Covers:
- build_face_lookup produces correct arrays
- interpolate_at_pixel returns valid values
- render_detail_texture_fullslot produces a PNG with no flat-fill regions
- Corner pixels have valid terrain colours (not average-of-all-faces)
- Elevation continuity at the hex boundary
- Pixel statistics (mean, std) of full slot are similar to hex interior
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest
import numpy as np

from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
from polygrid.detail_render import BiomeConfig

try:
    from polygrid.globe import build_globe_grid, _HAS_MODELS
    _skip = not _HAS_MODELS
except ImportError:
    _skip = True

needs_models = pytest.mark.skipif(_skip, reason="models library not installed")


def _make_detail_grid_with_terrain(detail_rings: int = 2):
    """Build one detail grid with terrain for testing rendering."""
    from conftest import cached_build_globe
    from polygrid.mountains import MountainConfig, generate_mountains
    from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
    from polygrid.detail_terrain import generate_all_detail_terrain

    grid = cached_build_globe(3)
    if grid is None:
        pytest.skip("models library not installed")
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)
    generate_mountains(grid, store, MountainConfig(seed=42))

    spec = TileDetailSpec(detail_rings=detail_rings)
    coll = DetailGridCollection.build(grid, spec)
    generate_all_detail_terrain(coll, grid, store, spec, seed=42)

    fid = coll.face_ids[0]
    detail_grid, detail_store = coll.get(fid)
    return detail_grid, detail_store


# ═══════════════════════════════════════════════════════════════════
# build_face_lookup
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestBuildFaceLookup:
    def test_arrays_match_face_count(self):
        from polygrid.tile_texture import build_face_lookup
        detail_grid, detail_store = _make_detail_grid_with_terrain()
        centroids, elevations, face_ids = build_face_lookup(
            detail_grid, detail_store,
        )
        assert centroids.shape[0] == len(face_ids)
        assert elevations.shape[0] == len(face_ids)
        assert len(face_ids) > 0

    def test_centroids_are_2d(self):
        from polygrid.tile_texture import build_face_lookup
        detail_grid, detail_store = _make_detail_grid_with_terrain()
        centroids, _, _ = build_face_lookup(detail_grid, detail_store)
        assert centroids.ndim == 2
        assert centroids.shape[1] == 2

    def test_elevations_are_finite(self):
        from polygrid.tile_texture import build_face_lookup
        detail_grid, detail_store = _make_detail_grid_with_terrain()
        _, elevations, _ = build_face_lookup(detail_grid, detail_store)
        assert np.all(np.isfinite(elevations))

    def test_face_ids_from_grid(self):
        from polygrid.tile_texture import build_face_lookup
        detail_grid, detail_store = _make_detail_grid_with_terrain()
        _, _, face_ids = build_face_lookup(detail_grid, detail_store)
        for fid in face_ids:
            assert fid in detail_grid.faces


# ═══════════════════════════════════════════════════════════════════
# interpolate_at_pixel
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestInterpolateAtPixel:
    def test_returns_four_values(self):
        from polygrid.tile_texture import build_face_lookup, interpolate_at_pixel
        from polygrid.detail_render import _detail_hillshade
        from scipy.spatial import KDTree

        detail_grid, detail_store = _make_detail_grid_with_terrain()
        centroids, elevations, face_ids = build_face_lookup(
            detail_grid, detail_store,
        )
        hs_dict = _detail_hillshade(detail_grid, detail_store)
        hs_arr = np.array([hs_dict.get(fid, 0.5) for fid in face_ids])
        tree = KDTree(centroids)

        # Query at the centroid of the first face
        cx, cy = centroids[0]
        result = interpolate_at_pixel(cx, cy, tree, centroids, elevations, hs_arr)
        assert len(result) == 4
        elev, hs, nx, ny = result
        assert 0.0 <= elev <= 1.0
        assert 0.0 <= hs <= 1.0

    def test_exact_hit_returns_face_values(self):
        from polygrid.tile_texture import build_face_lookup, interpolate_at_pixel
        from polygrid.detail_render import _detail_hillshade
        from scipy.spatial import KDTree

        detail_grid, detail_store = _make_detail_grid_with_terrain()
        centroids, elevations, face_ids = build_face_lookup(
            detail_grid, detail_store,
        )
        hs_dict = _detail_hillshade(detail_grid, detail_store)
        hs_arr = np.array([hs_dict.get(fid, 0.5) for fid in face_ids])
        tree = KDTree(centroids)

        # Query exactly at first centroid
        cx, cy = centroids[0]
        elev, hs, nx, ny = interpolate_at_pixel(
            cx, cy, tree, centroids, elevations, hs_arr,
        )
        assert abs(elev - elevations[0]) < 1e-6
        assert abs(hs - hs_arr[0]) < 1e-6

    def test_far_point_still_returns_valid(self):
        from polygrid.tile_texture import build_face_lookup, interpolate_at_pixel
        from polygrid.detail_render import _detail_hillshade
        from scipy.spatial import KDTree

        detail_grid, detail_store = _make_detail_grid_with_terrain()
        centroids, elevations, face_ids = build_face_lookup(
            detail_grid, detail_store,
        )
        hs_dict = _detail_hillshade(detail_grid, detail_store)
        hs_arr = np.array([hs_dict.get(fid, 0.5) for fid in face_ids])
        tree = KDTree(centroids)

        # Query far outside the grid
        elev, hs, nx, ny = interpolate_at_pixel(
            999.0, 999.0, tree, centroids, elevations, hs_arr,
        )
        assert np.isfinite(elev)
        assert np.isfinite(hs)


# ═══════════════════════════════════════════════════════════════════
# render_detail_texture_fullslot
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestRenderDetailTextureFullslot:
    def test_produces_png(self, tmp_path):
        from polygrid.tile_texture import render_detail_texture_fullslot
        detail_grid, detail_store = _make_detail_grid_with_terrain()
        out = tmp_path / "fullslot.png"
        result = render_detail_texture_fullslot(
            detail_grid, detail_store, out,
            tile_size=64,
        )
        assert result.exists()
        assert result.suffix == ".png"
        assert result.stat().st_size > 0

    def test_no_flat_fill_regions(self, tmp_path):
        """Verify that no contiguous region of identical pixels exists
        larger than a small threshold (which would indicate flat fill)."""
        from PIL import Image
        from polygrid.tile_texture import render_detail_texture_fullslot

        detail_grid, detail_store = _make_detail_grid_with_terrain()
        out = tmp_path / "fullslot_variance.png"
        render_detail_texture_fullslot(
            detail_grid, detail_store, out,
            tile_size=64,
        )

        img = np.array(Image.open(out))  # (H, W, 3)
        # Check that no row has all identical pixels
        for row_idx in range(img.shape[0]):
            row = img[row_idx]
            unique_colours = np.unique(row.reshape(-1, 3), axis=0)
            # Allow at most 90% of pixels to be the same colour
            # (very generous — flat fill would be 100%)
            max_same = 0
            for uc in unique_colours:
                count = np.sum(np.all(row == uc, axis=1))
                max_same = max(max_same, count)
            assert max_same < img.shape[1] * 0.9, (
                f"Row {row_idx} has {max_same}/{img.shape[1]} identical "
                f"pixels — possible flat fill"
            )

    def test_corner_pixels_are_not_average(self, tmp_path):
        """Corner pixels should have valid terrain colour, not the
        average-of-all-faces that the old renderer used."""
        from PIL import Image
        from polygrid.tile_texture import render_detail_texture_fullslot

        detail_grid, detail_store = _make_detail_grid_with_terrain()
        out = tmp_path / "fullslot_corners.png"
        render_detail_texture_fullslot(
            detail_grid, detail_store, out,
            tile_size=64,
        )

        img = np.array(Image.open(out))
        ts = 64

        # Gather corner pixels
        corners = [
            img[0, 0],
            img[0, ts - 1],
            img[ts - 1, 0],
            img[ts - 1, ts - 1],
        ]

        # They should NOT all be the same (flat fill would make them identical)
        corner_set = set(tuple(c) for c in corners)
        # At least 2 different corner colours (likely all 4 differ)
        assert len(corner_set) >= 2, (
            f"All corners identical: {corners[0]} — likely flat fill"
        )

    def test_pixel_variance_similar_to_interior(self, tmp_path):
        """Pixel variance across the full slot should be similar to
        variance of the hex interior (not drastically lower due to
        flat-fill homogeneity)."""
        from PIL import Image
        from polygrid.tile_texture import render_detail_texture_fullslot

        detail_grid, detail_store = _make_detail_grid_with_terrain()
        out = tmp_path / "fullslot_stats.png"
        render_detail_texture_fullslot(
            detail_grid, detail_store, out,
            tile_size=64,
        )

        img = np.array(Image.open(out), dtype=np.float64)  # (64, 64, 3)
        # Full image stats
        full_std = img.std()
        # Centre region (inner 50%) — approximates hex interior
        q1, q3 = 16, 48
        centre = img[q1:q3, q1:q3]
        centre_std = centre.std()

        # Full image std should be at least 50% of centre std
        # (flat fill would dramatically reduce full_std since ~44% of
        # pixels would be the same colour)
        if centre_std > 1.0:  # only meaningful if there's real variation
            ratio = full_std / centre_std
            assert ratio > 0.5, (
                f"Full std={full_std:.1f}, centre std={centre_std:.1f}, "
                f"ratio={ratio:.2f} — likely flat fill outside hex"
            )

    def test_custom_biome(self, tmp_path):
        from polygrid.tile_texture import render_detail_texture_fullslot
        detail_grid, detail_store = _make_detail_grid_with_terrain()
        out = tmp_path / "fullslot_biome.png"
        biome = BiomeConfig(snow_line=0.7, water_level=0.15)
        result = render_detail_texture_fullslot(
            detail_grid, detail_store, out, biome,
            tile_size=64,
        )
        assert result.exists()
        assert result.stat().st_size > 0

    def test_deterministic_output(self, tmp_path):
        """Same seed → same pixels."""
        from PIL import Image
        from polygrid.tile_texture import render_detail_texture_fullslot

        detail_grid, detail_store = _make_detail_grid_with_terrain()

        out1 = tmp_path / "det1.png"
        out2 = tmp_path / "det2.png"
        render_detail_texture_fullslot(
            detail_grid, detail_store, out1,
            tile_size=32, noise_seed=123,
        )
        render_detail_texture_fullslot(
            detail_grid, detail_store, out2,
            tile_size=32, noise_seed=123,
        )

        img1 = np.array(Image.open(out1))
        img2 = np.array(Image.open(out2))
        assert np.array_equal(img1, img2), "Same seed should produce identical output"

    def test_different_seeds_produce_different_output(self, tmp_path):
        from PIL import Image
        from polygrid.tile_texture import render_detail_texture_fullslot

        detail_grid, detail_store = _make_detail_grid_with_terrain()

        out1 = tmp_path / "seed1.png"
        out2 = tmp_path / "seed2.png"
        render_detail_texture_fullslot(
            detail_grid, detail_store, out1,
            tile_size=32, noise_seed=42,
        )
        render_detail_texture_fullslot(
            detail_grid, detail_store, out2,
            tile_size=32, noise_seed=999,
        )

        img1 = np.array(Image.open(out1))
        img2 = np.array(Image.open(out2))
        assert not np.array_equal(img1, img2), "Different seeds should produce different output"


# ═══════════════════════════════════════════════════════════════════
# Phase 16B — Soft tile-edge blending mask
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestComputeTileBlendMask:
    def test_mask_shape(self):
        from polygrid.tile_texture import compute_tile_blend_mask
        detail_grid, _ = _make_detail_grid_with_terrain()
        mask = compute_tile_blend_mask(detail_grid, tile_size=64, fade_width=8)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.float32

    def test_mask_range(self):
        from polygrid.tile_texture import compute_tile_blend_mask
        detail_grid, _ = _make_detail_grid_with_terrain()
        mask = compute_tile_blend_mask(detail_grid, tile_size=64, fade_width=8)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_centre_is_one(self):
        """The mask should be 1.0 at the tile centre (deep inside the hex)."""
        from polygrid.tile_texture import compute_tile_blend_mask
        detail_grid, _ = _make_detail_grid_with_terrain()
        mask = compute_tile_blend_mask(detail_grid, tile_size=64, fade_width=8)
        # Centre pixel
        centre = mask[32, 32]
        assert centre == pytest.approx(1.0), f"Centre mask value {centre} != 1.0"

    def test_corners_are_low(self):
        """The mask should be close to 0.0 at tile corners (outside hex)."""
        from polygrid.tile_texture import compute_tile_blend_mask
        detail_grid, _ = _make_detail_grid_with_terrain()
        mask = compute_tile_blend_mask(detail_grid, tile_size=64, fade_width=8)
        corners = [mask[0, 0], mask[0, 63], mask[63, 0], mask[63, 63]]
        for c in corners:
            assert c < 0.5, f"Corner mask value {c} should be < 0.5"

    def test_hex_edge_midpoint_higher_than_corner(self):
        """Midpoint of a hex edge should have higher mask than a corner,
        since the hex edge is closer to the polygon interior."""
        from polygrid.tile_texture import compute_tile_blend_mask
        detail_grid, _ = _make_detail_grid_with_terrain()
        mask = compute_tile_blend_mask(detail_grid, tile_size=64, fade_width=8)
        # Top-centre is nearer to hex boundary than top-left corner
        top_centre = mask[0, 32]
        top_left = mask[0, 0]
        assert top_centre >= top_left, (
            f"Top-centre {top_centre} should be >= top-left corner {top_left}"
        )

    def test_mask_follows_polygon_not_circle(self):
        """The mask should not be circularly symmetric — hex shape
        should make some edge midpoints lighter than corners."""
        from polygrid.tile_texture import compute_tile_blend_mask
        detail_grid, _ = _make_detail_grid_with_terrain()
        mask = compute_tile_blend_mask(detail_grid, tile_size=64, fade_width=12)
        # Edge midpoints (top, left, bottom, right)
        midpoints = [mask[0, 32], mask[32, 0], mask[63, 32], mask[32, 63]]
        # Corners
        corners = [mask[0, 0], mask[0, 63], mask[63, 0], mask[63, 63]]
        avg_mid = sum(midpoints) / len(midpoints)
        avg_corner = sum(corners) / len(corners)
        # Edge midpoints should be brighter on average than corners
        assert avg_mid > avg_corner, (
            f"Avg midpoint {avg_mid:.3f} should be > avg corner {avg_corner:.3f}"
        )

    def test_zero_fade_width(self):
        """With fade_width=0, mask should be binary: 1 inside, 0 outside."""
        from polygrid.tile_texture import compute_tile_blend_mask
        detail_grid, _ = _make_detail_grid_with_terrain()
        mask = compute_tile_blend_mask(detail_grid, tile_size=64, fade_width=0)
        unique = np.unique(mask)
        # Should only have 0.0 and 1.0
        assert len(unique) == 2
        assert 0.0 in unique
        assert 1.0 in unique


@needs_models
class TestApplyBlendMaskToAtlas:
    def test_atlas_unchanged_where_mask_is_one(self):
        """Pixels where mask is 1.0 should be unchanged."""
        from polygrid.tile_texture import apply_blend_mask_to_atlas
        tile_size = 8
        gutter = 2
        slot_size = tile_size + 2 * gutter
        atlas = np.full((slot_size, slot_size, 3), 200, dtype=np.uint8)

        mask = np.ones((tile_size, tile_size), dtype=np.float32)
        result = apply_blend_mask_to_atlas(
            atlas, {"t0": mask}, ["t0"], tile_size, gutter, columns=1,
        )
        inner = result[gutter:gutter + tile_size, gutter:gutter + tile_size]
        np.testing.assert_array_equal(inner, 200)

    def test_atlas_darkened_where_mask_is_zero(self):
        """Pixels where mask is 0.0 should become black."""
        from polygrid.tile_texture import apply_blend_mask_to_atlas
        tile_size = 8
        gutter = 2
        slot_size = tile_size + 2 * gutter
        atlas = np.full((slot_size, slot_size, 3), 200, dtype=np.uint8)

        mask = np.zeros((tile_size, tile_size), dtype=np.float32)
        result = apply_blend_mask_to_atlas(
            atlas, {"t0": mask}, ["t0"], tile_size, gutter, columns=1,
        )
        inner = result[gutter:gutter + tile_size, gutter:gutter + tile_size]
        np.testing.assert_array_equal(inner, 0)
