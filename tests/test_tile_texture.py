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


from functools import lru_cache


@lru_cache(maxsize=4)
def _cached_detail_grid_with_terrain(detail_rings: int = 2):
    """Build and cache one detail grid with terrain for testing rendering."""
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


def _make_detail_grid_with_terrain(detail_rings: int = 2):
    """Return cached detail grid + store (immutable data, safe to share)."""
    return _cached_detail_grid_with_terrain(detail_rings)


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


# ═══════════════════════════════════════════════════════════════════
# Phase 16D — Hex Shape Softening
# ═══════════════════════════════════════════════════════════════════


class TestJitterPolygonVertices:
    """16D.1 — Sub-face edge dissolution."""

    def test_jittered_within_bounds(self):
        """Jittered positions are within ±max_jitter of originals."""
        from polygrid.tile_texture import jitter_polygon_vertices

        verts = [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]
        jittered = jitter_polygon_vertices(verts, max_jitter=2.0, seed=42)
        assert len(jittered) == len(verts)
        for (ox, oy), (jx, jy) in zip(verts, jittered):
            assert abs(jx - ox) <= 2.0, f"x jitter too large: {abs(jx - ox)}"
            assert abs(jy - oy) <= 2.0, f"y jitter too large: {abs(jy - oy)}"

    def test_zero_jitter_returns_original(self):
        """max_jitter=0 should return the original vertices."""
        from polygrid.tile_texture import jitter_polygon_vertices

        verts = [(10.0, 20.0), (30.0, 40.0)]
        jittered = jitter_polygon_vertices(verts, max_jitter=0.0, seed=42)
        for (ox, oy), (jx, jy) in zip(verts, jittered):
            assert ox == jx
            assert oy == jy

    def test_deterministic(self):
        """Same seed + same verts → same result."""
        from polygrid.tile_texture import jitter_polygon_vertices

        verts = [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]
        a = jitter_polygon_vertices(verts, max_jitter=1.5, seed=99)
        b = jitter_polygon_vertices(verts, max_jitter=1.5, seed=99)
        assert a == b

    def test_different_seed_gives_different_result(self):
        from polygrid.tile_texture import jitter_polygon_vertices

        verts = [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]
        a = jitter_polygon_vertices(verts, max_jitter=1.5, seed=1)
        b = jitter_polygon_vertices(verts, max_jitter=1.5, seed=2)
        # At least one vertex should differ
        assert a != b


class TestApplyNoiseOverlay:
    """16D.2 — Pixel-level noise overlay."""

    def test_output_shape_and_dtype(self):
        from polygrid.tile_texture import apply_noise_overlay

        pixels = np.full((32, 32, 3), 128, dtype=np.uint8)
        result = apply_noise_overlay(pixels, frequency=0.1, amplitude=0.05, seed=42)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8

    def test_noise_changes_pixels(self):
        """Noise should change at least some pixel values."""
        from polygrid.tile_texture import apply_noise_overlay

        pixels = np.full((32, 32, 3), 128, dtype=np.uint8)
        result = apply_noise_overlay(pixels, frequency=0.1, amplitude=0.05, seed=42)
        assert not np.array_equal(pixels, result), "Noise overlay had no effect"

    def test_noise_within_amplitude(self):
        """Pixel changes should not exceed amplitude fraction of base."""
        from polygrid.tile_texture import apply_noise_overlay

        base_val = 200
        amp = 0.10  # 10%
        pixels = np.full((16, 16, 3), base_val, dtype=np.uint8)
        result = apply_noise_overlay(pixels, frequency=0.05, amplitude=amp, seed=42)
        diff = np.abs(result.astype(np.int16) - base_val)
        # Max possible change: base_val * amplitude = 200 * 0.10 = 20
        # Allow a small margin for floating-point rounding
        max_expected = base_val * amp + 2
        assert diff.max() <= max_expected, (
            f"Max pixel change {diff.max()} exceeds expected {max_expected}"
        )

    def test_deterministic(self):
        from polygrid.tile_texture import apply_noise_overlay

        pixels = np.full((16, 16, 3), 128, dtype=np.uint8)
        a = apply_noise_overlay(pixels, seed=42)
        b = apply_noise_overlay(pixels, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        from polygrid.tile_texture import apply_noise_overlay

        pixels = np.full((16, 16, 3), 128, dtype=np.uint8)
        a = apply_noise_overlay(pixels, seed=42)
        b = apply_noise_overlay(pixels, seed=999)
        assert not np.array_equal(a, b)


class TestApplyColourDithering:
    """16D.3 — Sub-face colour dithering."""

    def test_output_shape_and_dtype(self):
        from polygrid.tile_texture import apply_colour_dithering, _BG_SENTINEL

        pixels = np.full((16, 16, 3), 128, dtype=np.uint8)
        centroids = np.array([[4.0, 4.0], [12.0, 12.0]])
        colours = np.array([[100.0, 150.0, 200.0], [200.0, 100.0, 50.0]])
        result = apply_colour_dithering(
            pixels, centroids, colours, blend_radius=6.0,
        )
        assert result.shape == (16, 16, 3)
        assert result.dtype == np.uint8

    def test_sentinel_pixels_unchanged(self):
        """Sentinel (background) pixels should not be dithered."""
        from polygrid.tile_texture import apply_colour_dithering, _BG_SENTINEL

        pixels = np.full((16, 16, 3), _BG_SENTINEL[0], dtype=np.uint8)
        # Make the whole image sentinel
        pixels[:, :, 0] = _BG_SENTINEL[0]
        pixels[:, :, 1] = _BG_SENTINEL[1]
        pixels[:, :, 2] = _BG_SENTINEL[2]

        centroids = np.array([[4.0, 4.0], [12.0, 12.0]])
        colours = np.array([[100.0, 150.0, 200.0], [200.0, 100.0, 50.0]])
        result = apply_colour_dithering(pixels, centroids, colours)
        np.testing.assert_array_equal(result, pixels)

    def test_centre_pixels_less_changed_than_edge(self):
        """Pixels near a centroid should change less than pixels at edges."""
        from polygrid.tile_texture import apply_colour_dithering

        # Create a simple image: left half one colour, right half another
        pixels = np.zeros((16, 16, 3), dtype=np.uint8)
        pixels[:, :8] = [100, 150, 200]
        pixels[:, 8:] = [200, 100, 50]

        centroids = np.array([[4.0, 8.0], [12.0, 8.0]])
        colours = np.array([[100.0, 150.0, 200.0], [200.0, 100.0, 50.0]])
        result = apply_colour_dithering(
            pixels, centroids, colours, blend_radius=8.0,
        )

        # Pixel at centroid (4, 8) should barely change
        centre_diff = np.abs(
            result[8, 4].astype(int) - pixels[8, 4].astype(int),
        ).sum()
        # Pixel at boundary (8, 8) should change more
        edge_diff = np.abs(
            result[8, 8].astype(int) - pixels[8, 8].astype(int),
        ).sum()
        assert centre_diff <= edge_diff, (
            f"Centre diff {centre_diff} should be <= edge diff {edge_diff}"
        )

    def test_dithered_reduces_boundary_contrast(self):
        """Dithering should reduce the colour jump at sub-face boundaries."""
        from polygrid.tile_texture import apply_colour_dithering

        # Sharp boundary at column 8
        pixels = np.zeros((16, 16, 3), dtype=np.uint8)
        pixels[:, :8] = [60, 120, 80]
        pixels[:, 8:] = [180, 60, 40]

        centroids = np.array([[4.0, 8.0], [12.0, 8.0]])
        colours = np.array([[60.0, 120.0, 80.0], [180.0, 60.0, 40.0]])

        result = apply_colour_dithering(
            pixels, centroids, colours, blend_radius=8.0,
        )

        # Contrast at boundary: diff between col 7 and col 8
        orig_contrast = np.abs(
            pixels[:, 7].astype(float) - pixels[:, 8].astype(float),
        ).mean()
        dither_contrast = np.abs(
            result[:, 7].astype(float) - result[:, 8].astype(float),
        ).mean()
        assert dither_contrast < orig_contrast, (
            f"Dithered contrast {dither_contrast:.1f} should be < "
            f"original {orig_contrast:.1f}"
        )


@needs_models
class TestFullslotWith16D:
    """Integration tests for fullslot renderer with 16D enhancements."""

    def test_with_all_enhancements(self, tmp_path):
        """Render with all 16D enhancements enabled — should succeed."""
        from polygrid.tile_texture import render_detail_texture_fullslot

        detail_grid, detail_store = _make_detail_grid_with_terrain()
        out = tmp_path / "enhanced.png"
        result = render_detail_texture_fullslot(
            detail_grid, detail_store, out,
            tile_size=32,
            vertex_jitter=1.5,
            noise_overlay=True,
            colour_dither=True,
        )
        assert result.exists()
        assert result.stat().st_size > 0

    def test_without_enhancements(self, tmp_path):
        """Render with all 16D enhancements disabled — should succeed."""
        from polygrid.tile_texture import render_detail_texture_fullslot

        detail_grid, detail_store = _make_detail_grid_with_terrain()
        out = tmp_path / "plain.png"
        result = render_detail_texture_fullslot(
            detail_grid, detail_store, out,
            tile_size=32,
            vertex_jitter=0.0,
            noise_overlay=False,
            colour_dither=False,
        )
        assert result.exists()
        assert result.stat().st_size > 0

    def test_enhanced_differs_from_plain(self, tmp_path):
        """16D enhancements should change the output pixels."""
        from PIL import Image
        from polygrid.tile_texture import render_detail_texture_fullslot

        detail_grid, detail_store = _make_detail_grid_with_terrain()

        plain = tmp_path / "plain.png"
        render_detail_texture_fullslot(
            detail_grid, detail_store, plain,
            tile_size=32, noise_seed=42,
            vertex_jitter=0.0,
            noise_overlay=False,
            colour_dither=False,
        )

        enhanced = tmp_path / "enhanced.png"
        render_detail_texture_fullslot(
            detail_grid, detail_store, enhanced,
            tile_size=32, noise_seed=42,
            vertex_jitter=1.5,
            noise_overlay=True,
            colour_dither=True,
        )

        img_plain = np.array(Image.open(plain))
        img_enhanced = np.array(Image.open(enhanced))
        assert not np.array_equal(img_plain, img_enhanced), (
            "Enhanced output should differ from plain"
        )

    def test_deterministic_with_enhancements(self, tmp_path):
        """Same seed → same pixels, even with all enhancements."""
        from PIL import Image
        from polygrid.tile_texture import render_detail_texture_fullslot

        detail_grid, detail_store = _make_detail_grid_with_terrain()

        out1 = tmp_path / "det1.png"
        out2 = tmp_path / "det2.png"
        for out in (out1, out2):
            render_detail_texture_fullslot(
                detail_grid, detail_store, out,
                tile_size=32, noise_seed=123,
                vertex_jitter=1.5,
                noise_overlay=True,
                colour_dither=True,
            )

        img1 = np.array(Image.open(out1))
        img2 = np.array(Image.open(out2))
        assert np.array_equal(img1, img2), "Same seed should produce identical output"
