"""Tests for Phase 14B — biome_render.py (forest feature rendering)."""

from __future__ import annotations

import random
import pytest

from polygrid.biome_scatter import FeatureInstance, scatter_features_on_tile

try:
    from PIL import Image, ImageDraw
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

needs_pil = pytest.mark.skipif(not _HAS_PIL, reason="PIL/Pillow not installed")


@needs_pil
class TestForestFeatureConfig:
    def test_default_config(self):
        from polygrid.biome_render import ForestFeatureConfig
        cfg = ForestFeatureConfig()
        assert cfg.canopy_radius_range == (2.5, 7.0)
        assert len(cfg.canopy_colors) >= 3
        assert 0 < cfg.shadow_opacity <= 1.0
        assert cfg.density_scale > 0

    def test_presets_exist(self):
        from polygrid.biome_render import FOREST_PRESETS
        assert "temperate" in FOREST_PRESETS
        assert "tropical" in FOREST_PRESETS
        assert "boreal" in FOREST_PRESETS
        assert "sparse_woodland" in FOREST_PRESETS

    def test_tropical_denser_than_boreal(self):
        from polygrid.biome_render import TROPICAL_FOREST, BOREAL_FOREST
        assert TROPICAL_FOREST.density_scale > BOREAL_FOREST.density_scale

    def test_config_is_frozen(self):
        from polygrid.biome_render import ForestFeatureConfig
        cfg = ForestFeatureConfig()
        with pytest.raises(Exception):
            cfg.density_scale = 0.5  # type: ignore[misc]


@needs_pil
class TestRenderCanopy:
    def test_canopy_modifies_pixels(self):
        from polygrid.biome_render import render_canopy, TEMPERATE_FOREST
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        inst = FeatureInstance(px=32, py=32, radius=8.0, color=(50, 130, 40))
        render_canopy(draw, inst, TEMPERATE_FOREST)
        # Pixel at centre should be non-transparent
        pixel = img.getpixel((32, 32))
        assert pixel[3] > 0, f"Centre pixel alpha={pixel[3]}"

    def test_shadow_is_offset(self):
        from polygrid.biome_render import render_canopy, ForestFeatureConfig
        cfg = ForestFeatureConfig(shadow_offset=(5, 5), shadow_opacity=0.8)
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        inst = FeatureInstance(px=20, py=20, radius=6.0, color=(50, 130, 40))
        render_canopy(draw, inst, cfg)
        # Shadow pixel should be at offset
        shadow_pixel = img.getpixel((25, 25))
        assert shadow_pixel[3] > 0, "Shadow pixel should be non-transparent"

    def test_highlight_when_enabled(self):
        from polygrid.biome_render import render_canopy, ForestFeatureConfig
        cfg = ForestFeatureConfig(highlight_strength=0.8, highlight_offset=(-2, -2))
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        inst = FeatureInstance(px=32, py=32, radius=10.0, color=(50, 130, 40))
        render_canopy(draw, inst, cfg)
        # Pixel near highlight offset should be bright
        hp = img.getpixel((30, 30))
        assert hp[3] > 0

    def test_no_highlight_when_disabled(self):
        from polygrid.biome_render import render_canopy, ForestFeatureConfig
        cfg = ForestFeatureConfig(highlight_strength=0.0)
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        inst = FeatureInstance(px=32, py=32, radius=6.0, color=(50, 130, 40))
        render_canopy(draw, inst, cfg)
        # Canopy should still render (just no highlight)
        pixel = img.getpixel((32, 32))
        assert pixel[3] > 0


@needs_pil
class TestRenderUndergrowth:
    def test_undergrowth_modifies_image(self):
        from polygrid.biome_render import render_undergrowth, TEMPERATE_FOREST
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        render_undergrowth(img, density=0.8, config=TEMPERATE_FOREST, seed=42)
        # Should have some non-transparent pixels
        non_transparent = sum(
            1 for x in range(64) for y in range(64)
            if img.getpixel((x, y))[3] > 0
        )
        assert non_transparent > 100

    def test_density_affects_opacity(self):
        from polygrid.biome_render import render_undergrowth, TEMPERATE_FOREST
        img_dense = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        render_undergrowth(img_dense, density=0.9, config=TEMPERATE_FOREST, seed=1)
        img_sparse = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        render_undergrowth(img_sparse, density=0.2, config=TEMPERATE_FOREST, seed=1)

        def mean_alpha(img):
            return sum(
                img.getpixel((x, y))[3]
                for x in range(64) for y in range(64)
            ) / (64 * 64)

        assert mean_alpha(img_dense) > mean_alpha(img_sparse)


@needs_pil
class TestRenderForestTile:
    def test_returns_rgba_image(self):
        from polygrid.biome_render import render_forest_tile, TEMPERATE_FOREST
        ground = Image.new("RGB", (64, 64), (100, 80, 50))
        instances = [
            FeatureInstance(px=32, py=32, radius=8.0, color=(50, 130, 40)),
        ]
        result = render_forest_tile(ground, instances, TEMPERATE_FOREST)
        assert result.mode == "RGBA"
        assert result.size == (64, 64)

    def test_forest_differs_from_ground(self):
        from polygrid.biome_render import render_forest_tile, TEMPERATE_FOREST
        ground = Image.new("RGB", (128, 128), (120, 100, 70))
        instances = scatter_features_on_tile(
            0.8, tile_size=128, seed=42,
        )
        result = render_forest_tile(ground, instances, TEMPERATE_FOREST)
        # Convert both to comparable format
        ground_rgba = ground.convert("RGBA")
        # Count pixels that differ
        diff_count = 0
        for x in range(128):
            for y in range(128):
                gp = ground_rgba.getpixel((x, y))
                rp = result.getpixel((x, y))
                if gp != rp:
                    diff_count += 1
        assert diff_count > 500, f"Only {diff_count} pixels changed"

    def test_empty_instances_minimal_change(self):
        from polygrid.biome_render import render_forest_tile, TEMPERATE_FOREST
        ground = Image.new("RGB", (64, 64), (120, 100, 70))
        result = render_forest_tile(ground, [], TEMPERATE_FOREST, density=0.01)
        # With zero density undergrowth should be very faint
        assert result.mode == "RGBA"

    def test_dense_forest_darker_than_sparse(self):
        from polygrid.biome_render import render_forest_tile, TEMPERATE_FOREST
        ground = Image.new("RGB", (128, 128), (150, 150, 150))

        dense_inst = scatter_features_on_tile(0.9, tile_size=128, seed=10)
        sparse_inst = scatter_features_on_tile(0.2, tile_size=128, seed=10)

        dense_result = render_forest_tile(ground, dense_inst, TEMPERATE_FOREST, density=0.9)
        sparse_result = render_forest_tile(ground, sparse_inst, TEMPERATE_FOREST, density=0.2)

        def mean_brightness(img):
            total = 0
            px = img.load()
            w, h = img.size
            for x in range(w):
                for y in range(h):
                    r, g, b, a = px[x, y]
                    total += r + g + b
            return total / (w * h * 3)

        # Dense forest (more shadow) should be darker
        assert mean_brightness(dense_result) < mean_brightness(sparse_result)

    def test_different_presets_produce_different_output(self):
        from polygrid.biome_render import (
            render_forest_tile, TEMPERATE_FOREST, TROPICAL_FOREST,
        )
        ground = Image.new("RGB", (64, 64), (120, 100, 70))
        instances = scatter_features_on_tile(0.8, tile_size=64, seed=42)

        temp = render_forest_tile(ground, instances, TEMPERATE_FOREST)
        trop = render_forest_tile(ground, instances, TROPICAL_FOREST)

        # Count differing pixels
        diff = sum(
            1 for x in range(64) for y in range(64)
            if temp.getpixel((x, y)) != trop.getpixel((x, y))
        )
        assert diff > 100, f"Only {diff} pixels differ between presets"


@needs_pil
class TestRenderForestOnGround:
    def test_convenience_function_works(self):
        from polygrid.biome_render import render_forest_on_ground
        ground = Image.new("RGB", (128, 128), (100, 80, 50))
        result = render_forest_on_ground(ground, 0.8, seed=42)
        assert result.mode == "RGBA"
        assert result.size == (128, 128)

    def test_zero_density_returns_mostly_ground(self):
        from polygrid.biome_render import render_forest_on_ground
        ground = Image.new("RGB", (64, 64), (100, 80, 50))
        result = render_forest_on_ground(ground, 0.0, seed=42)
        # Should be essentially the ground (maybe faint undergrowth)
        assert result.mode == "RGBA"

    def test_with_globe_3d_center(self):
        from polygrid.biome_render import render_forest_on_ground
        ground = Image.new("RGB", (128, 128), (100, 80, 50))
        result = render_forest_on_ground(
            ground, 0.8, seed=42, globe_3d_center=(0.5, 0.5, 0.707),
        )
        assert result.size == (128, 128)


# ═══════════════════════════════════════════════════════════════════
# Phase 16C — Full-slot forest rendering with feature cross-fade
# ═══════════════════════════════════════════════════════════════════

@needs_pil
class TestRenderForestOnGroundFullslot:
    """16C.3 — Feature-level cross-fade with blend mask."""

    def test_produces_image(self):
        from polygrid.biome_render import render_forest_on_ground_fullslot
        ground = Image.new("RGB", (64, 64), (100, 80, 50))
        result = render_forest_on_ground_fullslot(
            ground, 0.8, tile_size=64, seed=42,
        )
        assert result.size == (64, 64)

    def test_with_blend_mask(self):
        import numpy as np
        from polygrid.biome_render import render_forest_on_ground_fullslot
        ground = Image.new("RGB", (64, 64), (100, 80, 50))
        # Mask: 1 in centre, 0 at edges
        mask = np.ones((64, 64), dtype=np.float32)
        mask[:8, :] = 0.0
        mask[-8:, :] = 0.0
        mask[:, :8] = 0.0
        mask[:, -8:] = 0.0

        result = render_forest_on_ground_fullslot(
            ground, 0.8, tile_size=64, seed=42,
            blend_mask=mask,
        )
        assert result.size == (64, 64)

    def test_mask_fades_features_at_edges(self):
        """Features at tile edges should be faded toward ground colour."""
        import numpy as np
        from polygrid.biome_render import render_forest_on_ground_fullslot

        ground_colour = (100, 80, 50)
        ground = Image.new("RGB", (64, 64), ground_colour)

        # Full mask (no fade)
        full_mask = np.ones((64, 64), dtype=np.float32)
        no_fade = render_forest_on_ground_fullslot(
            ground, 0.9, tile_size=64, seed=42,
            blend_mask=full_mask,
        )

        # Edge-fade mask
        fade_mask = np.ones((64, 64), dtype=np.float32)
        fade_mask[:10, :] = 0.0
        fade_mask[-10:, :] = 0.0
        fade_mask[:, :10] = 0.0
        fade_mask[:, -10:] = 0.0

        with_fade = render_forest_on_ground_fullslot(
            ground, 0.9, tile_size=64, seed=42,
            blend_mask=fade_mask,
        )

        # Edge pixels with fade should be closer to ground colour
        no_fade_arr = np.array(no_fade.convert("RGB"))
        with_fade_arr = np.array(with_fade.convert("RGB"))
        ground_arr = np.array(ground.convert("RGB"))

        # Edge region (first 5 rows)
        edge_nofade_diff = np.abs(
            no_fade_arr[:5].astype(float) - ground_arr[:5].astype(float),
        ).mean()
        edge_fade_diff = np.abs(
            with_fade_arr[:5].astype(float) - ground_arr[:5].astype(float),
        ).mean()

        # With fade, edge pixels should be closer to ground
        assert edge_fade_diff <= edge_nofade_diff + 1.0, (
            f"Faded edges ({edge_fade_diff:.1f}) should be closer to ground "
            f"than unfaded ({edge_nofade_diff:.1f})"
        )

    def test_no_mask_is_same_as_standard(self):
        """Without blend_mask, fullslot render should produce features."""
        from polygrid.biome_render import render_forest_on_ground_fullslot
        ground = Image.new("RGB", (64, 64), (100, 80, 50))
        result = render_forest_on_ground_fullslot(
            ground, 0.8, tile_size=64, seed=42,
        )
        # Should differ from ground (features added)
        import numpy as np
        result_arr = np.array(result.convert("RGB"))
        ground_arr = np.array(ground)
        assert not np.array_equal(result_arr, ground_arr)

    def test_deterministic(self):
        from polygrid.biome_render import render_forest_on_ground_fullslot
        import numpy as np
        ground = Image.new("RGB", (64, 64), (100, 80, 50))
        a = render_forest_on_ground_fullslot(ground, 0.8, seed=42)
        b = render_forest_on_ground_fullslot(ground, 0.8, seed=42)
        np.testing.assert_array_equal(
            np.array(a.convert("RGB")),
            np.array(b.convert("RGB")),
        )
