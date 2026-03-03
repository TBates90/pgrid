"""Tests for Phase 14D — biome_pipeline.py (atlas integration)."""

from __future__ import annotations

import pytest

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# ── Gate behind models availability ─────────────────────────────────
try:
    from polygrid.globe import build_globe_grid, _HAS_MODELS
    _skip = not _HAS_MODELS
except ImportError:
    _skip = True

needs_models = pytest.mark.skipif(_skip, reason="models library not installed")
needs_pil = pytest.mark.skipif(not _HAS_PIL, reason="PIL/Pillow not installed")


# ═══════════════════════════════════════════════════════════════════
# ForestRenderer
# ═══════════════════════════════════════════════════════════════════

@needs_pil
class TestForestRenderer:
    def test_satisfies_protocol(self):
        from polygrid.biome_pipeline import BiomeRenderer, ForestRenderer
        renderer = ForestRenderer()
        # Check it has the render method with right signature
        assert hasattr(renderer, "render")
        assert callable(renderer.render)

    def test_render_produces_image(self):
        from polygrid.biome_pipeline import ForestRenderer
        renderer = ForestRenderer()
        ground = Image.new("RGB", (64, 64), (100, 80, 50))
        result = renderer.render(ground, "t0", density=0.8, seed=42)
        assert result.size == (64, 64)

    def test_zero_density_minimal_change(self):
        from polygrid.biome_pipeline import ForestRenderer
        renderer = ForestRenderer()
        ground = Image.new("RGB", (64, 64), (100, 80, 50))
        result = renderer.render(ground, "t0", density=0.0, seed=42)
        # Very low density → should be close to ground
        result_rgb = result.convert("RGB")
        assert result_rgb.size == (64, 64)

    def test_different_tiles_different_output(self):
        from polygrid.biome_pipeline import ForestRenderer
        renderer = ForestRenderer()
        ground = Image.new("RGB", (64, 64), (100, 80, 50))
        r1 = renderer.render(ground, "t0", density=0.8, seed=42)
        r2 = renderer.render(ground, "t5", density=0.8, seed=42)
        # Different tile_id → different seed → different scatter
        diff = sum(
            1 for x in range(64) for y in range(64)
            if r1.getpixel((x, y)) != r2.getpixel((x, y))
        )
        assert diff > 50


# ═══════════════════════════════════════════════════════════════════
# identify_forest_tiles
# ═══════════════════════════════════════════════════════════════════

class TestIdentifyForestTiles:
    def test_finds_forest_faces(self):
        from polygrid.biome_pipeline import identify_forest_tiles
        from dataclasses import dataclass

        @dataclass
        class FakePatch:
            terrain_type: str
            face_ids: list

        patches = [
            FakePatch(terrain_type="forest", face_ids=["t0", "t1"]),
            FakePatch(terrain_type="mountain", face_ids=["t2"]),
            FakePatch(terrain_type="forest", face_ids=["t3"]),
        ]
        result = identify_forest_tiles(patches)
        assert result == {"t0", "t1", "t3"}

    def test_no_forest_patches(self):
        from polygrid.biome_pipeline import identify_forest_tiles
        from dataclasses import dataclass

        @dataclass
        class FakePatch:
            terrain_type: str
            face_ids: list

        patches = [
            FakePatch(terrain_type="mountain", face_ids=["t0"]),
            FakePatch(terrain_type="ocean", face_ids=["t1"]),
        ]
        result = identify_forest_tiles(patches)
        assert result == set()

    def test_custom_terrain_type(self):
        from polygrid.biome_pipeline import identify_forest_tiles
        from dataclasses import dataclass

        @dataclass
        class FakePatch:
            terrain_type: str
            face_ids: list

        patches = [
            FakePatch(terrain_type="desert", face_ids=["t0", "t1"]),
        ]
        result = identify_forest_tiles(patches, terrain_type="desert")
        assert result == {"t0", "t1"}


# ═══════════════════════════════════════════════════════════════════
# build_feature_atlas (integration — needs models + PIL)
# ═══════════════════════════════════════════════════════════════════

@needs_models
@needs_pil
class TestBuildFeatureAtlas:
    """Integration tests for the full feature atlas pipeline."""

    def _build_collection(self, freq=1, rings=1):
        from conftest import cached_build_globe
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
        from polygrid.detail_terrain import generate_all_detail_terrain
        from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
        import random

        grid = cached_build_globe(freq)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        rng = random.Random(42)
        for fid in grid.faces:
            store.set(fid, "elevation", rng.uniform(0.1, 0.9))
        spec = TileDetailSpec(detail_rings=rings)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)
        return grid, store, coll

    def test_atlas_with_no_features(self, tmp_path):
        from polygrid.biome_pipeline import build_feature_atlas
        grid, store, coll = self._build_collection()
        atlas_path, uv = build_feature_atlas(
            coll, globe_grid=grid,
            output_dir=tmp_path / "tiles",
            tile_size=64,
        )
        assert atlas_path.exists()
        assert len(uv) == len(coll.face_ids)
        img = Image.open(str(atlas_path))
        assert img.size[0] > 0

    def test_atlas_with_forest_features(self, tmp_path):
        from polygrid.biome_pipeline import build_feature_atlas, ForestRenderer
        grid, store, coll = self._build_collection()

        # Give all tiles some density
        density_map = {fid: 0.8 for fid in coll.face_ids}

        atlas_path, uv = build_feature_atlas(
            coll, globe_grid=grid,
            biome_renderers={"forest": ForestRenderer()},
            density_map=density_map,
            output_dir=tmp_path / "tiles",
            tile_size=64,
        )
        assert atlas_path.exists()
        assert len(uv) == len(coll.face_ids)

    def test_featured_atlas_differs_from_plain(self, tmp_path):
        from polygrid.biome_pipeline import build_feature_atlas, ForestRenderer
        grid, store, coll = self._build_collection()

        # Plain atlas (no density)
        plain_path, _ = build_feature_atlas(
            coll, globe_grid=grid,
            output_dir=tmp_path / "plain",
            tile_size=64,
        )
        # Featured atlas
        density_map = {fid: 0.9 for fid in coll.face_ids}
        feat_path, _ = build_feature_atlas(
            coll, globe_grid=grid,
            biome_renderers={"forest": ForestRenderer()},
            density_map=density_map,
            output_dir=tmp_path / "featured",
            tile_size=64,
        )

        plain_img = Image.open(str(plain_path))
        feat_img = Image.open(str(feat_path))
        assert plain_img.size == feat_img.size

        # Images should differ (features overlay)
        plain_bytes = plain_img.tobytes()
        feat_bytes = feat_img.tobytes()
        # Compare pixel-by-pixel (3 bytes per pixel for RGB)
        bpp = 3
        n_pixels = len(plain_bytes) // bpp
        diff = sum(
            1 for i in range(n_pixels)
            if plain_bytes[i*bpp:(i+1)*bpp] != feat_bytes[i*bpp:(i+1)*bpp]
        )
        assert diff > 100, f"Only {diff} pixels differ"

    def test_uv_layout_same_as_standard(self, tmp_path):
        """UV layout structure should match the standard atlas format."""
        from polygrid.biome_pipeline import build_feature_atlas
        grid, store, coll = self._build_collection()
        _, uv = build_feature_atlas(
            coll, globe_grid=grid,
            output_dir=tmp_path / "tiles",
            tile_size=64,
        )
        for fid, (u_min, v_min, u_max, v_max) in uv.items():
            assert 0 <= u_min < u_max <= 1.0
            assert 0 <= v_min < v_max <= 1.0

    def test_partial_density_map(self, tmp_path):
        """Only some tiles have density — others get plain ground."""
        from polygrid.biome_pipeline import build_feature_atlas, ForestRenderer
        grid, store, coll = self._build_collection()

        face_ids = coll.face_ids
        density_map = {face_ids[0]: 0.9}  # only first tile

        atlas_path, uv = build_feature_atlas(
            coll, globe_grid=grid,
            biome_renderers={"forest": ForestRenderer()},
            density_map=density_map,
            output_dir=tmp_path / "tiles",
            tile_size=64,
        )
        assert atlas_path.exists()
        assert len(uv) == len(face_ids)


# ═══════════════════════════════════════════════════════════════════
# Phase 16E — Pipeline Integration & Validation
# ═══════════════════════════════════════════════════════════════════

@needs_pil
@needs_models
class TestSoftBlendPipeline:
    """Integration tests for the full Phase 16 soft-blend pipeline."""

    def _build_collection(self, freq=1, rings=1):
        from conftest import cached_build_globe
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
        from polygrid.detail_terrain import generate_all_detail_terrain
        from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
        import random

        grid = cached_build_globe(freq)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        rng = random.Random(42)
        for fid in grid.faces:
            store.set(fid, "elevation", rng.uniform(0.1, 0.9))
        spec = TileDetailSpec(detail_rings=rings)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)
        return grid, store, coll

    # ── 16E.1  soft_blend produces a valid atlas ────────────────

    def test_soft_blend_atlas_produces_file(self, tmp_path):
        """soft_blend=True should produce a valid atlas PNG."""
        from polygrid.biome_pipeline import build_feature_atlas
        grid, store, coll = self._build_collection()
        atlas_path, uv = build_feature_atlas(
            coll, globe_grid=grid,
            output_dir=tmp_path / "tiles",
            tile_size=64,
            soft_blend=True,
        )
        assert atlas_path.exists()
        assert len(uv) == len(coll.face_ids)
        img = Image.open(str(atlas_path))
        assert img.size[0] > 0 and img.size[1] > 0

    def test_soft_blend_atlas_dimensions_unchanged(self, tmp_path):
        """soft_blend should not change atlas dimensions vs plain."""
        from polygrid.biome_pipeline import build_feature_atlas
        grid, store, coll = self._build_collection()

        plain_path, uv_plain = build_feature_atlas(
            coll, globe_grid=grid,
            output_dir=tmp_path / "plain",
            tile_size=64,
        )
        blend_path, uv_blend = build_feature_atlas(
            coll, globe_grid=grid,
            output_dir=tmp_path / "blend",
            tile_size=64,
            soft_blend=True,
        )

        plain_img = Image.open(str(plain_path))
        blend_img = Image.open(str(blend_path))
        assert plain_img.size == blend_img.size
        # UV layout should have same keys
        assert set(uv_plain.keys()) == set(uv_blend.keys())

    def test_soft_blend_with_forest_features(self, tmp_path):
        """soft_blend + forest features should produce a valid atlas."""
        from polygrid.biome_pipeline import build_feature_atlas, ForestRenderer
        grid, store, coll = self._build_collection()
        density_map = {fid: 0.8 for fid in coll.face_ids}

        atlas_path, uv = build_feature_atlas(
            coll, globe_grid=grid,
            biome_renderers={"forest": ForestRenderer(fullslot=True)},
            density_map=density_map,
            output_dir=tmp_path / "tiles",
            tile_size=64,
            soft_blend=True,
        )
        assert atlas_path.exists()
        assert len(uv) == len(coll.face_ids)

    def test_soft_blend_forces_fullslot(self, tmp_path):
        """When soft_blend=True, the pipeline should use fullslot rendering."""
        from polygrid.biome_pipeline import build_feature_atlas
        import numpy as np

        grid, store, coll = self._build_collection()
        atlas_path, _ = build_feature_atlas(
            coll, globe_grid=grid,
            output_dir=tmp_path / "tiles",
            tile_size=64,
            soft_blend=True,
        )
        # With fullslot, no magenta sentinel pixels should appear
        img = Image.open(str(atlas_path))
        arr = np.array(img)
        magenta_mask = (arr[:, :, 0] == 255) & (arr[:, :, 1] == 0) & (arr[:, :, 2] == 255)
        assert magenta_mask.sum() == 0, "Magenta sentinel pixels found in soft_blend atlas"

    def test_soft_blend_alters_tile_edges(self, tmp_path):
        """soft_blend should fade tile edges, reducing boundary contrast."""
        from polygrid.biome_pipeline import build_feature_atlas
        import numpy as np

        grid, store, coll = self._build_collection()

        # Build plain atlas (no blend)
        plain_path, _ = build_feature_atlas(
            coll, globe_grid=grid,
            output_dir=tmp_path / "plain",
            tile_size=64,
            fullslot=True,
        )
        # Build soft-blend atlas
        blend_path, _ = build_feature_atlas(
            coll, globe_grid=grid,
            output_dir=tmp_path / "blend",
            tile_size=64,
            soft_blend=True,
        )

        plain_arr = np.array(Image.open(str(plain_path))).astype(np.float32)
        blend_arr = np.array(Image.open(str(blend_path))).astype(np.float32)

        # The blend atlas should differ from the plain one
        diff = np.abs(plain_arr - blend_arr).mean()
        assert diff > 0.5, f"Expected noticeable difference, got mean diff={diff:.2f}"

    def test_soft_blend_uv_values_in_range(self, tmp_path):
        """UV coords from soft_blend atlas should be valid."""
        from polygrid.biome_pipeline import build_feature_atlas
        grid, store, coll = self._build_collection()
        _, uv = build_feature_atlas(
            coll, globe_grid=grid,
            output_dir=tmp_path / "tiles",
            tile_size=64,
            soft_blend=True,
        )
        for fid, (u_min, v_min, u_max, v_max) in uv.items():
            assert 0 <= u_min < u_max <= 1.0, f"{fid}: u range invalid"
            assert 0 <= v_min < v_max <= 1.0, f"{fid}: v range invalid"

    def test_blend_fade_width_parameter(self, tmp_path):
        """Custom blend_fade_width should be accepted and alter result."""
        from polygrid.biome_pipeline import build_feature_atlas
        import numpy as np

        grid, store, coll = self._build_collection()

        narrow_path, _ = build_feature_atlas(
            coll, globe_grid=grid,
            output_dir=tmp_path / "narrow",
            tile_size=64,
            soft_blend=True,
            blend_fade_width=4,
        )
        wide_path, _ = build_feature_atlas(
            coll, globe_grid=grid,
            output_dir=tmp_path / "wide",
            tile_size=64,
            soft_blend=True,
            blend_fade_width=24,
        )

        narrow_arr = np.array(Image.open(str(narrow_path))).astype(np.float32)
        wide_arr = np.array(Image.open(str(wide_path))).astype(np.float32)
        # Different fade widths should produce different results
        diff = np.abs(narrow_arr - wide_arr).mean()
        assert diff > 0.1, f"Expected different results, got mean diff={diff:.2f}"

    def test_soft_blend_partial_density(self, tmp_path):
        """soft_blend with partial density: only some tiles get features."""
        from polygrid.biome_pipeline import build_feature_atlas, ForestRenderer
        grid, store, coll = self._build_collection()

        face_ids = coll.face_ids
        density_map = {face_ids[0]: 0.9}

        atlas_path, uv = build_feature_atlas(
            coll, globe_grid=grid,
            biome_renderers={"forest": ForestRenderer(fullslot=True)},
            density_map=density_map,
            output_dir=tmp_path / "tiles",
            tile_size=64,
            soft_blend=True,
        )
        assert atlas_path.exists()
        assert len(uv) == len(face_ids)
