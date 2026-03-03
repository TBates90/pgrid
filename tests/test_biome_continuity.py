"""Tests for Phase 14C — biome_continuity.py (cross-tile continuity)."""

from __future__ import annotations

import pytest

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    from polygrid.globe import build_globe_grid, _HAS_MODELS
    _skip = not _HAS_MODELS
except ImportError:
    _skip = True

needs_models = pytest.mark.skipif(_skip, reason="models library not installed")
needs_pil = pytest.mark.skipif(not _HAS_PIL, reason="PIL/Pillow not installed")


# ═══════════════════════════════════════════════════════════════════
# build_biome_density_map
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestBuildBiomeDensityMap:
    """Density map with neighbour transitions."""

    def _get_grid(self):
        from conftest import cached_build_globe
        return cached_build_globe(1)

    def test_all_biome_returns_positive(self):
        from polygrid.biome_continuity import build_biome_density_map
        grid = self._get_grid()
        fids = list(grid.faces.keys())
        dm = build_biome_density_map(grid, fids, seed=42)
        assert len(dm) == len(fids)
        # All tiles are biome → all should be > 0
        for fid in fids:
            assert dm[fid] > 0.0

    def test_no_biome_returns_zero(self):
        from polygrid.biome_continuity import build_biome_density_map
        grid = self._get_grid()
        fids = list(grid.faces.keys())
        dm = build_biome_density_map(
            grid, fids, biome_faces=set(), seed=42
        )
        # Empty biome set — everything should be zero
        for fid in fids:
            assert dm[fid] == 0.0

    def test_partial_biome_has_transition_tiles(self):
        from polygrid.biome_continuity import build_biome_density_map
        grid = self._get_grid()
        fids = list(grid.faces.keys())
        # Pick a small biome subset (just the first 3 faces)
        biome = set(fids[:3])
        dm = build_biome_density_map(
            grid, fids, biome_faces=biome,
            neighbour_transition=0.3, seed=42,
        )
        # Biome tiles should have high density
        for fid in biome:
            assert dm[fid] > 0.3, f"{fid} should be biome interior"

        # Some non-biome tiles adjacent to biome should have transition density
        transition_tiles = [
            fid for fid in fids
            if fid not in biome and dm[fid] > 0.01
        ]
        assert len(transition_tiles) > 0, "No transition tiles found"
        for fid in transition_tiles:
            assert dm[fid] <= 0.35  # neighbour_transition = 0.3

    def test_values_in_unit_range(self):
        from polygrid.biome_continuity import build_biome_density_map
        grid = self._get_grid()
        fids = list(grid.faces.keys())
        dm = build_biome_density_map(grid, fids, seed=99)
        for d in dm.values():
            assert 0.0 <= d <= 1.0

    def test_deterministic(self):
        from polygrid.biome_continuity import build_biome_density_map
        grid = self._get_grid()
        fids = list(grid.faces.keys())
        dm1 = build_biome_density_map(grid, fids, seed=42)
        dm2 = build_biome_density_map(grid, fids, seed=42)
        assert dm1 == dm2


# ═══════════════════════════════════════════════════════════════════
# get_tile_margin_features
# ═══════════════════════════════════════════════════════════════════

class TestGetTileMarginFeatures:
    """Margin features from neighbouring tiles."""

    def _make_instance(self, px, py, radius=4.0):
        from polygrid.biome_scatter import FeatureInstance
        return FeatureInstance(
            px=px, py=py, radius=radius,
            color=(50, 130, 40),
            shadow_color=(15, 35, 10),
            species_id=0,
            depth=py,
        )

    def test_no_neighbours_returns_empty(self):
        from polygrid.biome_continuity import get_tile_margin_features
        result = get_tile_margin_features(
            "t0", own_scatter=[], neighbour_scatters={},
            tile_size=64, margin=8.0,
        )
        assert result == []

    def test_far_interior_not_collected(self):
        from polygrid.biome_continuity import get_tile_margin_features
        # Feature at centre of neighbour — should NOT be collected
        inst = self._make_instance(32.0, 32.0, radius=4.0)
        result = get_tile_margin_features(
            "t0", own_scatter=[],
            neighbour_scatters={"t1": [inst]},
            tile_size=64, margin=8.0,
        )
        assert len(result) == 0

    def test_near_right_edge_collected(self):
        from polygrid.biome_continuity import get_tile_margin_features
        # Feature near right edge of neighbour → overlaps our left
        inst = self._make_instance(60.0, 32.0, radius=5.0)
        result = get_tile_margin_features(
            "t0", own_scatter=[],
            neighbour_scatters={"t1": [inst]},
            tile_size=64, margin=8.0,
        )
        assert len(result) == 1
        # Translated: 60 - 64 = -4  (just outside our left edge)
        assert result[0].px == pytest.approx(-4.0)

    def test_near_bottom_edge_collected(self):
        from polygrid.biome_continuity import get_tile_margin_features
        inst = self._make_instance(32.0, 59.0, radius=5.0)
        result = get_tile_margin_features(
            "t0", own_scatter=[],
            neighbour_scatters={"t1": [inst]},
            tile_size=64, margin=8.0,
        )
        assert len(result) == 1
        assert result[0].py == pytest.approx(-5.0)

    def test_multiple_neighbours(self):
        from polygrid.biome_continuity import get_tile_margin_features
        inst_right = self._make_instance(60.0, 32.0, radius=5.0)
        inst_top = self._make_instance(32.0, 3.0, radius=5.0)
        result = get_tile_margin_features(
            "t0", own_scatter=[],
            neighbour_scatters={
                "t1": [inst_right],
                "t2": [inst_top],
            },
            tile_size=64, margin=8.0,
        )
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════
# compute_biome_transition_mask
# ═══════════════════════════════════════════════════════════════════

class TestComputeBiomeTransitionMask:
    """2-D gradient mask for biome transitions."""

    def test_zero_density_all_zero(self):
        from polygrid.biome_continuity import compute_biome_transition_mask
        mask = compute_biome_transition_mask(0.0, {}, tile_size=32)
        for row in mask:
            for v in row:
                assert v == 0.0

    def test_uniform_density_no_feather(self):
        from polygrid.biome_continuity import compute_biome_transition_mask
        mask = compute_biome_transition_mask(
            0.8, {}, tile_size=32
        )
        # No neighbours with lower density → uniform at tile_density
        for row in mask:
            for v in row:
                assert v == pytest.approx(0.8, abs=0.01)

    def test_left_neighbour_lower_creates_gradient(self):
        from polygrid.biome_continuity import compute_biome_transition_mask
        mask = compute_biome_transition_mask(
            0.8, {"left": 0.0}, tile_size=32, feather_width=0.25,
        )
        # Left edge (x=0) should be lower than interior
        left_col = [mask[y][0] for y in range(32)]
        interior_col = [mask[y][16] for y in range(32)]
        assert sum(left_col) < sum(interior_col)

    def test_gradient_is_smooth(self):
        from polygrid.biome_continuity import compute_biome_transition_mask
        mask = compute_biome_transition_mask(
            0.8, {"right": 0.0}, tile_size=64, feather_width=0.2,
        )
        # Check no pixel-to-pixel jump exceeds a reasonable threshold
        max_jump = 0.0
        for y in range(64):
            for x in range(1, 64):
                jump = abs(mask[y][x] - mask[y][x - 1])
                max_jump = max(max_jump, jump)
        assert max_jump < 0.15, f"Max gradient jump {max_jump} too large"

    def test_values_in_range(self):
        from polygrid.biome_continuity import compute_biome_transition_mask
        mask = compute_biome_transition_mask(
            0.9, {"left": 0.1, "bottom": 0.2}, tile_size=32,
        )
        for row in mask:
            for v in row:
                assert 0.0 <= v <= 1.0

    def test_all_neighbours_lower(self):
        from polygrid.biome_continuity import compute_biome_transition_mask
        mask = compute_biome_transition_mask(
            0.8,
            {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0},
            tile_size=32, feather_width=0.3,
        )
        # Centre should be at full density, edges lower
        centre = mask[16][16]
        edge = mask[0][0]
        assert centre > edge


# ═══════════════════════════════════════════════════════════════════
# stitch_feature_boundary
# ═══════════════════════════════════════════════════════════════════

@needs_pil
class TestStitchFeatureBoundary:
    """Feather-blending of boundary pixel strips."""

    def test_right_edge_blending_reduces_discontinuity(self):
        from polygrid.biome_continuity import stitch_feature_boundary
        # Two images with very different colours at the boundary
        a = Image.new("RGB", (32, 32), (255, 0, 0))
        b = Image.new("RGB", (32, 32), (0, 0, 255))

        a2, b2 = stitch_feature_boundary(a, b, edge="right", feather_pixels=4)

        # After stitching, the boundary strip should be blended
        # a's right edge should have some blue, b's left edge some red
        pa = a2.getpixel((31, 16))  # a's rightmost pixel
        pb = b2.getpixel((0, 16))   # b's leftmost pixel

        # Both should be closer to purple-ish than pure red/blue
        assert pa[2] > 50, f"a's right should have some blue: {pa}"
        assert pb[0] > 10, f"b's left should have some red: {pb}"

    def test_bottom_edge_blending(self):
        from polygrid.biome_continuity import stitch_feature_boundary
        a = Image.new("RGB", (32, 32), (200, 200, 200))
        b = Image.new("RGB", (32, 32), (50, 50, 50))

        a2, b2 = stitch_feature_boundary(a, b, edge="bottom", feather_pixels=4)

        pa = a2.getpixel((16, 31))
        pb = b2.getpixel((16, 0))

        # Should be blended toward the other tile's colour
        assert pa[0] < 200
        assert pb[0] > 50

    def test_no_change_away_from_boundary(self):
        from polygrid.biome_continuity import stitch_feature_boundary
        a = Image.new("RGB", (32, 32), (255, 0, 0))
        b = Image.new("RGB", (32, 32), (0, 0, 255))

        a2, b2 = stitch_feature_boundary(a, b, edge="right", feather_pixels=4)

        # Far from the boundary → unchanged
        assert a2.getpixel((0, 16)) == (255, 0, 0)
        assert b2.getpixel((31, 16)) == (0, 0, 255)

    def test_identical_images_unchanged(self):
        from polygrid.biome_continuity import stitch_feature_boundary
        a = Image.new("RGB", (32, 32), (100, 100, 100))
        b = Image.new("RGB", (32, 32), (100, 100, 100))

        a2, b2 = stitch_feature_boundary(a, b, edge="right", feather_pixels=4)

        # Same colour → blending should keep them the same
        for x in range(32):
            for y in range(32):
                assert a2.getpixel((x, y)) == (100, 100, 100)
                assert b2.getpixel((x, y)) == (100, 100, 100)

    def test_returns_copies(self):
        from polygrid.biome_continuity import stitch_feature_boundary
        a = Image.new("RGB", (32, 32), (255, 0, 0))
        b = Image.new("RGB", (32, 32), (0, 0, 255))

        a2, b2 = stitch_feature_boundary(a, b, edge="right", feather_pixels=4)

        # Original images should be unmodified
        assert a.getpixel((31, 16)) == (255, 0, 0)
        assert b.getpixel((0, 16)) == (0, 0, 255)
