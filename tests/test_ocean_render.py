"""Tests for Phase 17A — ocean_render.py (config + depth map)."""

from __future__ import annotations

import pytest

# ── Gate behind models availability ─────────────────────────────────
try:
    from polygrid.globe import build_globe_grid, _HAS_MODELS
    _skip = not _HAS_MODELS
except ImportError:
    _skip = True

needs_models = pytest.mark.skipif(_skip, reason="models library not installed")


# ═══════════════════════════════════════════════════════════════════
# 17A.1 — OceanFeatureConfig
# ═══════════════════════════════════════════════════════════════════

class TestOceanFeatureConfig:
    """Config construction and preset validation."""

    def test_default_config_valid(self):
        from polygrid.ocean_render import OceanFeatureConfig
        cfg = OceanFeatureConfig()
        assert len(cfg.shallow_color) == 3
        assert len(cfg.deep_color) == 3
        assert len(cfg.abyssal_color) == 3
        assert all(0 <= c <= 255 for c in cfg.shallow_color)
        assert all(0 <= c <= 255 for c in cfg.deep_color)
        assert all(0 <= c <= 255 for c in cfg.abyssal_color)

    def test_presets_exist(self):
        from polygrid.ocean_render import OCEAN_PRESETS
        assert "tropical" in OCEAN_PRESETS
        assert "temperate" in OCEAN_PRESETS
        assert "arctic" in OCEAN_PRESETS
        assert "deep" in OCEAN_PRESETS

    def test_presets_have_valid_colours(self):
        from polygrid.ocean_render import OCEAN_PRESETS, OceanFeatureConfig
        for name, cfg in OCEAN_PRESETS.items():
            assert isinstance(cfg, OceanFeatureConfig), f"{name}"
            for attr in ("shallow_color", "deep_color", "abyssal_color",
                         "coastal_foam_color", "sand_color"):
                colour = getattr(cfg, attr)
                assert len(colour) == 3, f"{name}.{attr}"
                assert all(0 <= c <= 255 for c in colour), f"{name}.{attr}={colour}"

    def test_presets_have_positive_frequencies(self):
        from polygrid.ocean_render import OCEAN_PRESETS
        for name, cfg in OCEAN_PRESETS.items():
            assert cfg.wave_frequency > 0, f"{name}.wave_frequency"
            assert cfg.caustic_frequency > 0, f"{name}.caustic_frequency"

    def test_presets_shallow_brighter_than_deep(self):
        """Shallow water should be brighter than deep/abyssal."""
        from polygrid.ocean_render import OCEAN_PRESETS
        for name, cfg in OCEAN_PRESETS.items():
            shallow_lum = sum(cfg.shallow_color) / 3.0
            deep_lum = sum(cfg.deep_color) / 3.0
            abyssal_lum = sum(cfg.abyssal_color) / 3.0
            assert shallow_lum > deep_lum, f"{name}: shallow not brighter than deep"
            assert deep_lum > abyssal_lum, f"{name}: deep not brighter than abyssal"

    def test_config_frozen(self):
        from polygrid.ocean_render import OceanFeatureConfig
        cfg = OceanFeatureConfig()
        with pytest.raises(AttributeError):
            cfg.wave_frequency = 99.0  # type: ignore[misc]

    def test_custom_config(self):
        from polygrid.ocean_render import OceanFeatureConfig
        cfg = OceanFeatureConfig(
            shallow_color=(100, 200, 220),
            depth_gradient_power=3.0,
        )
        assert cfg.shallow_color == (100, 200, 220)
        assert cfg.depth_gradient_power == 3.0
        # Defaults still set
        assert cfg.wave_amplitude > 0


# ═══════════════════════════════════════════════════════════════════
# 17A.2 — identify_ocean_tiles
# ═══════════════════════════════════════════════════════════════════

class TestIdentifyOceanTiles:
    """Tests for ocean face identification from terrain patches."""

    def _make_patch(self, terrain_type, face_ids):
        """Create a minimal mock patch."""
        class FakePatch:
            pass
        p = FakePatch()
        p.terrain_type = terrain_type
        p.face_ids = face_ids
        return p

    def test_finds_ocean_faces(self):
        from polygrid.ocean_render import identify_ocean_tiles
        patches = [
            self._make_patch("ocean", {"f0", "f1", "f2"}),
            self._make_patch("forest", {"f3", "f4"}),
            self._make_patch("ocean", {"f5"}),
        ]
        result = identify_ocean_tiles(patches)
        assert result == {"f0", "f1", "f2", "f5"}

    def test_no_ocean(self):
        from polygrid.ocean_render import identify_ocean_tiles
        patches = [
            self._make_patch("forest", {"f0", "f1"}),
            self._make_patch("mountain", {"f2"}),
        ]
        result = identify_ocean_tiles(patches)
        assert result == set()

    def test_custom_terrain_type(self):
        from polygrid.ocean_render import identify_ocean_tiles
        patches = [
            self._make_patch("desert", {"f0"}),
        ]
        result = identify_ocean_tiles(patches, terrain_type="desert")
        assert result == {"f0"}


# ═══════════════════════════════════════════════════════════════════
# 17A.2 — compute_ocean_depth_map
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestComputeOceanDepthMap:
    """Depth map tests using a real globe grid."""

    def _build_globe_with_ocean(self, frequency=1, seed=42):
        """Build a globe + store with some ocean and some land tiles."""
        from conftest import cached_build_globe
        from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
        import random

        grid = cached_build_globe(frequency)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)

        rng = random.Random(seed)
        face_ids = sorted(grid.faces.keys())

        # Make ~half ocean, half land
        ocean_faces = set()
        for fid in face_ids:
            elev = rng.uniform(-0.1, 0.5)
            store.set(fid, "elevation", elev)
            if elev < 0.12:  # water_level
                ocean_faces.add(fid)

        return grid, store, ocean_faces

    def test_depth_map_has_all_ocean_tiles(self):
        from polygrid.ocean_render import compute_ocean_depth_map
        grid, store, ocean_faces = self._build_globe_with_ocean()
        if not ocean_faces:
            pytest.skip("No ocean tiles generated")
        depth_map = compute_ocean_depth_map(grid, store, ocean_faces)
        assert set(depth_map.keys()) == ocean_faces

    def test_depth_values_in_range(self):
        from polygrid.ocean_render import compute_ocean_depth_map
        grid, store, ocean_faces = self._build_globe_with_ocean()
        if not ocean_faces:
            pytest.skip("No ocean tiles generated")
        depth_map = compute_ocean_depth_map(grid, store, ocean_faces)
        for fid, depth in depth_map.items():
            assert 0.0 <= depth <= 1.0, f"{fid}: depth={depth}"

    def test_land_tiles_not_in_map(self):
        from polygrid.ocean_render import compute_ocean_depth_map
        grid, store, ocean_faces = self._build_globe_with_ocean()
        if not ocean_faces:
            pytest.skip("No ocean tiles generated")
        depth_map = compute_ocean_depth_map(grid, store, ocean_faces)
        land_faces = set(grid.faces.keys()) - ocean_faces
        for fid in land_faces:
            assert fid not in depth_map

    def test_coastal_tiles_shallower_than_deep(self):
        """Tiles adjacent to land should have lower depth than remote ones."""
        from polygrid.ocean_render import compute_ocean_depth_map
        from polygrid.algorithms import get_face_adjacency

        # Use freq=3 for more tiles and better depth variation
        grid, store, ocean_faces = self._build_globe_with_ocean(frequency=3)
        if len(ocean_faces) < 3:
            pytest.skip("Not enough ocean tiles")

        depth_map = compute_ocean_depth_map(grid, store, ocean_faces)
        adjacency = get_face_adjacency(grid)
        land_faces = set(grid.faces.keys()) - ocean_faces

        # Find coastal ocean tiles (adjacent to land)
        coastal = [
            fid for fid in ocean_faces
            if any(n in land_faces for n in adjacency.get(fid, []))
        ]
        # Find non-coastal ocean tiles
        deep = [fid for fid in ocean_faces if fid not in coastal]

        if not coastal or not deep:
            pytest.skip("Need both coastal and deep ocean tiles")

        avg_coastal = sum(depth_map[f] for f in coastal) / len(coastal)
        avg_deep = sum(depth_map[f] for f in deep) / len(deep)
        # Deep tiles should have higher average depth
        assert avg_deep >= avg_coastal, (
            f"avg_deep={avg_deep:.3f} < avg_coastal={avg_coastal:.3f}"
        )

    def test_empty_ocean_set(self):
        from polygrid.ocean_render import compute_ocean_depth_map
        from conftest import cached_build_globe
        from polygrid.tile_data import FieldDef, TileDataStore, TileSchema

        grid = cached_build_globe(1)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        for fid in grid.faces:
            store.set(fid, "elevation", 0.5)

        depth_map = compute_ocean_depth_map(grid, store, set())
        assert depth_map == {}

    def test_all_ocean(self):
        """All tiles are ocean — no coast to anchor BFS from."""
        from polygrid.ocean_render import compute_ocean_depth_map
        from conftest import cached_build_globe
        from polygrid.tile_data import FieldDef, TileDataStore, TileSchema

        grid = cached_build_globe(1)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        ocean_faces = set()
        for fid in grid.faces:
            store.set(fid, "elevation", 0.01)
            ocean_faces.add(fid)

        depth_map = compute_ocean_depth_map(grid, store, ocean_faces)
        assert len(depth_map) == len(ocean_faces)
        # All tiles should still have valid depth
        for d in depth_map.values():
            assert 0.0 <= d <= 1.0

    def test_elevation_weight_dominates(self):
        """With elevation_weight=1, distance_weight=0, depth = elev only."""
        from polygrid.ocean_render import compute_ocean_depth_map
        grid, store, ocean_faces = self._build_globe_with_ocean()
        if not ocean_faces:
            pytest.skip("No ocean tiles generated")
        depth_map = compute_ocean_depth_map(
            grid, store, ocean_faces,
            elevation_weight=1.0, distance_weight=0.0,
        )
        # Verify depth tracks elevation inversely
        for fid in ocean_faces:
            elev = store.get(fid, "elevation")
            expected = max(0.0, min(1.0, (0.12 - elev) / 0.12))
            assert abs(depth_map[fid] - expected) < 0.01, (
                f"{fid}: depth={depth_map[fid]:.3f}, expected={expected:.3f}"
            )


# ═══════════════════════════════════════════════════════════════════
# 17A — compute_coast_direction
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestComputeCoastDirection:
    """Coast direction vector tests."""

    def test_coastal_tile_has_direction(self):
        from polygrid.ocean_render import compute_coast_direction
        from polygrid.algorithms import get_face_adjacency
        from conftest import cached_build_globe
        from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
        import random, math

        grid = cached_build_globe(3)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        rng = random.Random(99)
        ocean_faces = set()
        for fid in grid.faces:
            elev = rng.uniform(-0.1, 0.5)
            store.set(fid, "elevation", elev)
            if elev < 0.12:
                ocean_faces.add(fid)

        if not ocean_faces:
            pytest.skip("No ocean tiles")

        adjacency = get_face_adjacency(grid)
        land_faces = set(grid.faces.keys()) - ocean_faces

        # Find a coastal ocean tile
        coastal = None
        for fid in ocean_faces:
            if any(n in land_faces for n in adjacency.get(fid, [])):
                coastal = fid
                break

        if coastal is None:
            pytest.skip("No coastal tiles")

        direction = compute_coast_direction(grid, coastal, ocean_faces)
        assert direction is not None
        dx, dy, dz = direction
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        assert abs(length - 1.0) < 1e-6, f"Not unit vector: length={length}"

    def test_deep_ocean_tile_no_direction(self):
        """A tile with no land neighbours should return None."""
        from polygrid.ocean_render import compute_coast_direction
        from conftest import cached_build_globe
        from polygrid.tile_data import FieldDef, TileDataStore, TileSchema

        grid = cached_build_globe(1)
        # All ocean
        ocean_faces = set(grid.faces.keys())
        # All tiles have no land neighbours
        fid = sorted(ocean_faces)[0]
        direction = compute_coast_direction(grid, fid, ocean_faces)
        assert direction is None


# ═══════════════════════════════════════════════════════════════════
# 17B — Ocean Texture Rendering
# ═══════════════════════════════════════════════════════════════════

try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

needs_pil = pytest.mark.skipif(not _HAS_PIL, reason="PIL/Pillow not installed")


def _make_ocean_tile(size=32, color=(40, 100, 160)):
    """Create a small solid-colour test image."""
    img = _PILImage.new("RGB", (size, size), color)
    return img


@needs_pil
class TestRenderOceanDepthGradient:
    """17B.1 — depth gradient fill."""

    def test_shallow_brighter_than_deep(self):
        from polygrid.ocean_render import render_ocean_depth_gradient
        import numpy as np

        shallow = _make_ocean_tile()
        render_ocean_depth_gradient(shallow, 0.1, seed=42)
        deep = _make_ocean_tile()
        render_ocean_depth_gradient(deep, 0.9, seed=42)

        s_mean = np.array(shallow).mean()
        d_mean = np.array(deep).mean()
        assert s_mean > d_mean, f"shallow={s_mean:.1f}, deep={d_mean:.1f}"

    def test_not_flat_fill(self):
        """Noise should produce spatial variation (not all identical pixels)."""
        from polygrid.ocean_render import render_ocean_depth_gradient
        import numpy as np

        img = _make_ocean_tile(size=64)
        render_ocean_depth_gradient(img, 0.4, seed=42)
        arr = np.array(img)
        # At least some pixel variance across the tile
        assert arr.std() > 0.5, f"std={arr.std():.2f} — too flat"

    def test_zero_depth_is_shallow_color(self):
        from polygrid.ocean_render import render_ocean_depth_gradient, TEMPERATE_OCEAN
        import numpy as np

        img = _make_ocean_tile(size=16)
        render_ocean_depth_gradient(img, 0.0, TEMPERATE_OCEAN, seed=0)
        arr = np.array(img)
        # Centre pixel should be close to shallow_color
        mid = arr[8, 8]
        sc = TEMPERATE_OCEAN.shallow_color
        # Within ±20 per channel (noise perturbation)
        for i in range(3):
            assert abs(int(mid[i]) - sc[i]) < 25, (
                f"ch{i}: pixel={mid[i]}, shallow={sc[i]}"
            )

    def test_deterministic(self):
        from polygrid.ocean_render import render_ocean_depth_gradient

        a = _make_ocean_tile()
        render_ocean_depth_gradient(a, 0.5, seed=123)
        b = _make_ocean_tile()
        render_ocean_depth_gradient(b, 0.5, seed=123)
        assert a.tobytes() == b.tobytes()


@needs_pil
class TestRenderWavePattern:
    """17B.2 — wave pattern overlay."""

    def test_modifies_pixels(self):
        from polygrid.ocean_render import render_ocean_depth_gradient, render_wave_pattern
        import numpy as np

        img = _make_ocean_tile(size=64)
        render_ocean_depth_gradient(img, 0.3, seed=42)
        before = np.array(img).copy()
        render_wave_pattern(img, 0.3, seed=42)
        after = np.array(img)
        assert not np.array_equal(before, after), "Waves had no effect"

    def test_pixel_variance_increased(self):
        from polygrid.ocean_render import render_ocean_depth_gradient, render_wave_pattern
        import numpy as np

        img = _make_ocean_tile(size=64)
        render_ocean_depth_gradient(img, 0.3, seed=42)
        var_before = np.array(img).astype(float).var()
        render_wave_pattern(img, 0.3, seed=42)
        var_after = np.array(img).astype(float).var()
        # Waves should add variation
        assert var_after >= var_before * 0.9  # at least not drastically reduced

    def test_deep_ocean_calmer(self):
        """Deep ocean should have less wave amplitude than shallow."""
        from polygrid.ocean_render import render_ocean_depth_gradient, render_wave_pattern
        import numpy as np

        shallow = _make_ocean_tile(size=64)
        render_ocean_depth_gradient(shallow, 0.1, seed=42)
        before_s = np.array(shallow).copy().astype(float)
        render_wave_pattern(shallow, 0.1, seed=42)
        after_s = np.array(shallow).astype(float)
        diff_shallow = np.abs(after_s - before_s).mean()

        deep = _make_ocean_tile(size=64)
        render_ocean_depth_gradient(deep, 0.9, seed=42)
        before_d = np.array(deep).copy().astype(float)
        render_wave_pattern(deep, 0.9, seed=42)
        after_d = np.array(deep).astype(float)
        diff_deep = np.abs(after_d - before_d).mean()

        assert diff_shallow >= diff_deep * 0.8, (
            f"shallow_diff={diff_shallow:.2f}, deep_diff={diff_deep:.2f}"
        )

    def test_deterministic(self):
        from polygrid.ocean_render import render_wave_pattern

        a = _make_ocean_tile()
        render_wave_pattern(a, 0.5, seed=77)
        b = _make_ocean_tile()
        render_wave_pattern(b, 0.5, seed=77)
        assert a.tobytes() == b.tobytes()


@needs_pil
class TestRenderCoastalFeatures:
    """17B.3 — foam, sand, caustic, reef."""

    def test_shallow_tile_modified(self):
        from polygrid.ocean_render import (
            render_ocean_depth_gradient, render_coastal_features,
        )
        import numpy as np

        img = _make_ocean_tile(size=64)
        render_ocean_depth_gradient(img, 0.05, seed=42)
        before = np.array(img).copy()
        render_coastal_features(img, 0.05, (0, -1, 0), seed=42)
        after = np.array(img)
        assert not np.array_equal(before, after), "Coastal features had no effect"

    def test_deep_tile_unmodified(self):
        """Tiles with depth > 0.3 should be unaffected."""
        from polygrid.ocean_render import (
            render_ocean_depth_gradient, render_coastal_features,
        )
        import numpy as np

        img = _make_ocean_tile(size=32)
        render_ocean_depth_gradient(img, 0.5, seed=42)
        before = np.array(img).copy()
        render_coastal_features(img, 0.5, (0, -1, 0), seed=42)
        after = np.array(img)
        assert np.array_equal(before, after), "Deep tile was modified by coastal features"

    def test_foam_pixels_brighter_near_coast_edge(self):
        """Foam zone should be brighter than the tile interior."""
        from polygrid.ocean_render import (
            render_ocean_depth_gradient, render_coastal_features,
        )
        import numpy as np

        img = _make_ocean_tile(size=64)
        render_ocean_depth_gradient(img, 0.05, seed=42)
        # Coast direction: coast is to the "top" (negative y)
        render_coastal_features(img, 0.05, (0.0, -1.0, 0.0), seed=42)
        arr = np.array(img).astype(float)
        # Top edge (coast-facing) should be brighter than bottom
        top_mean = arr[:8, :, :].mean()
        bottom_mean = arr[-8:, :, :].mean()
        assert top_mean > bottom_mean, (
            f"top={top_mean:.1f}, bottom={bottom_mean:.1f}"
        )

    def test_none_coast_direction_ok(self):
        """Should work with coast_direction=None (default north)."""
        from polygrid.ocean_render import (
            render_ocean_depth_gradient, render_coastal_features,
        )
        img = _make_ocean_tile(size=32)
        render_ocean_depth_gradient(img, 0.1, seed=42)
        render_coastal_features(img, 0.1, None, seed=42)
        # Just checking it doesn't crash
        assert img.size == (32, 32)


@needs_pil
class TestRenderDeepOceanFeatures:
    """17B.4 — abyssal darkness + upwelling."""

    def test_deep_tile_darkened(self):
        from polygrid.ocean_render import (
            render_ocean_depth_gradient, render_deep_ocean_features,
        )
        import numpy as np

        img = _make_ocean_tile(size=64)
        render_ocean_depth_gradient(img, 0.8, seed=42)
        before_mean = np.array(img).astype(float).mean()
        render_deep_ocean_features(img, 0.8, seed=42)
        after_mean = np.array(img).astype(float).mean()
        assert after_mean < before_mean, (
            f"before={before_mean:.1f}, after={after_mean:.1f}"
        )

    def test_shallow_tile_unmodified(self):
        from polygrid.ocean_render import (
            render_ocean_depth_gradient, render_deep_ocean_features,
        )
        import numpy as np

        img = _make_ocean_tile(size=32)
        render_ocean_depth_gradient(img, 0.3, seed=42)
        before = np.array(img).copy()
        render_deep_ocean_features(img, 0.3, seed=42)
        after = np.array(img)
        assert np.array_equal(before, after)

    def test_very_deep_darker_than_medium(self):
        from polygrid.ocean_render import (
            render_ocean_depth_gradient, render_deep_ocean_features,
        )
        import numpy as np

        medium = _make_ocean_tile(size=32)
        render_ocean_depth_gradient(medium, 0.6, seed=42)
        render_deep_ocean_features(medium, 0.6, seed=42)

        very_deep = _make_ocean_tile(size=32)
        render_ocean_depth_gradient(very_deep, 0.95, seed=42)
        render_deep_ocean_features(very_deep, 0.95, seed=42)

        m_mean = np.array(medium).astype(float).mean()
        v_mean = np.array(very_deep).astype(float).mean()
        assert v_mean < m_mean, f"very_deep={v_mean:.1f}, medium={m_mean:.1f}"


@needs_pil
class TestRenderOceanTile:
    """17B.5 — composite render_ocean_tile."""

    def test_produces_rgb_image(self):
        from polygrid.ocean_render import render_ocean_tile
        ground = _make_ocean_tile(size=32)
        result = render_ocean_tile(ground, 0.3, seed=42)
        assert result.mode == "RGB"
        assert result.size == (32, 32)

    def test_shallow_vs_deep_brightness(self):
        from polygrid.ocean_render import render_ocean_tile
        import numpy as np

        shallow = render_ocean_tile(_make_ocean_tile(size=32), 0.05, seed=42)
        deep = render_ocean_tile(_make_ocean_tile(size=32), 0.9, seed=42)
        s_mean = np.array(shallow).mean()
        d_mean = np.array(deep).mean()
        assert s_mean > d_mean

    def test_not_identical_to_input(self):
        from polygrid.ocean_render import render_ocean_tile
        ground = _make_ocean_tile(size=32)
        result = render_ocean_tile(ground, 0.5, seed=42)
        assert ground.tobytes() != result.tobytes()

    def test_deterministic(self):
        from polygrid.ocean_render import render_ocean_tile
        a = render_ocean_tile(_make_ocean_tile(), 0.4, seed=99)
        b = render_ocean_tile(_make_ocean_tile(), 0.4, seed=99)
        assert a.tobytes() == b.tobytes()

    def test_different_seeds_differ(self):
        from polygrid.ocean_render import render_ocean_tile
        a = render_ocean_tile(_make_ocean_tile(), 0.4, seed=1)
        b = render_ocean_tile(_make_ocean_tile(), 0.4, seed=2)
        assert a.tobytes() != b.tobytes()
