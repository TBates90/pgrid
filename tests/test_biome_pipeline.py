"""Tests for Phase 14D — biome_pipeline.py (atlas integration).

Performance-optimised: the expensive globe + detail-terrain build is
done **once per module** and shared across all integration classes via
``_shared_build_collection()``.  Each call returns a fresh
``DetailGridCollection`` wrapper so tests can mutate ``_stores``
safely.
"""

from __future__ import annotations

import random
from functools import lru_cache

import numpy as np
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
# Shared, cached collection builder  (runs once per (freq, rings))
# ═══════════════════════════════════════════════════════════════════

@lru_cache(maxsize=4)
def _cached_collection_internals(freq: int = 1, rings: int = 1):
    """Build and cache the expensive parts: globe, grids, spec, globe_store.

    Returns (globe, grids_dict, spec, globe_store) — all immutable /
    read-only so safe to cache.
    """
    from conftest import cached_build_globe
    from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
    from polygrid.detail_terrain import generate_all_detail_terrain
    from polygrid.tile_data import FieldDef, TileDataStore, TileSchema

    grid = cached_build_globe(freq)
    if grid is None:
        return None

    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)
    rng = random.Random(42)
    for fid in grid.faces:
        store.set(fid, "elevation", rng.uniform(0.1, 0.9))

    spec = TileDetailSpec(detail_rings=rings)
    coll = DetailGridCollection.build(grid, spec)
    generate_all_detail_terrain(coll, grid, store, spec, seed=42)

    # Snapshot the stores so we can copy them into fresh collections
    return grid, store, coll._grids, coll._stores, spec


def _shared_build_collection(freq: int = 1, rings: int = 1):
    """Return (grid, globe_store, collection) with a *fresh* wrapper.

    The heavy grid/terrain work is cached; each call gets its own
    ``DetailGridCollection`` with a **copy** of ``_stores`` so tests
    can't cross-contaminate.
    """
    result = _cached_collection_internals(freq, rings)
    if result is None:
        pytest.skip("models library not installed")

    grid, store, grids_dict, stores_dict, spec = result
    from polygrid.tile_detail import DetailGridCollection
    coll = DetailGridCollection(grid, spec, grids_dict)
    # Copy the stores dict (shallow — store objects are not mutated)
    coll._stores = dict(stores_dict)
    return grid, store, coll


# ═══════════════════════════════════════════════════════════════════
# ForestRenderer
# ═══════════════════════════════════════════════════════════════════

@needs_pil
class TestForestRenderer:
    def test_satisfies_protocol(self):
        from polygrid.biome_pipeline import BiomeRenderer, ForestRenderer
        renderer = ForestRenderer()
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
        result_rgb = result.convert("RGB")
        assert result_rgb.size == (64, 64)

    def test_different_tiles_different_output(self):
        from polygrid.biome_pipeline import ForestRenderer
        renderer = ForestRenderer()
        ground = Image.new("RGB", (64, 64), (100, 80, 50))
        r1 = renderer.render(ground, "t0", density=0.8, seed=42)
        r2 = renderer.render(ground, "t5", density=0.8, seed=42)
        # Fast numpy comparison instead of pixel-by-pixel Python loop
        diff = (np.array(r1) != np.array(r2)).any(axis=-1).sum()
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
    """Integration tests for the full feature atlas pipeline.

    Uses a class-scoped fixture so the atlas is built **once** and
    shared across simple validation tests (UV ranges, file exists, etc).
    """

    @pytest.fixture(scope="class")
    def shared_atlas(self, tmp_path_factory):
        """Build a plain atlas once for the whole class."""
        from polygrid.biome_pipeline import build_feature_atlas
        grid, _store, coll = _shared_build_collection()
        out = tmp_path_factory.mktemp("atlas_shared")
        atlas_path, uv = build_feature_atlas(
            coll, globe_grid=grid,
            output_dir=out / "tiles",
            tile_size=64,
        )
        return grid, coll, atlas_path, uv

    def test_atlas_with_no_features(self, shared_atlas):
        _grid, coll, atlas_path, uv = shared_atlas
        assert atlas_path.exists()
        assert len(uv) == len(coll.face_ids)
        img = Image.open(str(atlas_path))
        assert img.size[0] > 0

    def test_uv_layout_same_as_standard(self, shared_atlas):
        """UV layout structure should match the standard atlas format."""
        _grid, _coll, _path, uv = shared_atlas
        for fid, (u_min, v_min, u_max, v_max) in uv.items():
            assert 0 <= u_min < u_max <= 1.0
            assert 0 <= v_min < v_max <= 1.0

    def test_atlas_with_forest_features(self, tmp_path):
        from polygrid.biome_pipeline import build_feature_atlas, ForestRenderer
        grid, _store, coll = _shared_build_collection()
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
        grid, _store, coll = _shared_build_collection()

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

        plain_arr = np.array(Image.open(str(plain_path)))
        feat_arr = np.array(Image.open(str(feat_path)))
        assert plain_arr.shape == feat_arr.shape
        diff = (plain_arr != feat_arr).any(axis=-1).sum()
        assert diff > 100, f"Only {diff} pixels differ"

    def test_partial_density_map(self, tmp_path):
        """Only some tiles have density — others get plain ground."""
        from polygrid.biome_pipeline import build_feature_atlas, ForestRenderer
        grid, _store, coll = _shared_build_collection()

        face_ids = coll.face_ids
        density_map = {face_ids[0]: 0.9}

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
    """Integration tests for the full Phase 16 soft-blend pipeline.

    A class-scoped fixture builds the plain + soft-blend atlases once;
    individual tests inspect the cached results.
    """

    @pytest.fixture(scope="class")
    def blend_atlases(self, tmp_path_factory):
        """Build plain and soft-blend atlases once for the whole class."""
        from polygrid.biome_pipeline import build_feature_atlas
        grid, _store, coll = _shared_build_collection()

        out = tmp_path_factory.mktemp("blend")

        plain_path, uv_plain = build_feature_atlas(
            coll, globe_grid=grid,
            output_dir=out / "plain",
            tile_size=64,
            fullslot=True,
        )
        blend_path, uv_blend = build_feature_atlas(
            coll, globe_grid=grid,
            output_dir=out / "blend",
            tile_size=64,
            soft_blend=True,
        )
        return {
            "grid": grid, "coll": coll,
            "plain_path": plain_path, "uv_plain": uv_plain,
            "blend_path": blend_path, "uv_blend": uv_blend,
        }

    # ── 16E.1  soft_blend produces a valid atlas ────────────────

    def test_soft_blend_atlas_produces_file(self, blend_atlases):
        """soft_blend=True should produce a valid atlas PNG."""
        atlas_path = blend_atlases["blend_path"]
        uv = blend_atlases["uv_blend"]
        coll = blend_atlases["coll"]
        assert atlas_path.exists()
        assert len(uv) == len(coll.face_ids)
        img = Image.open(str(atlas_path))
        assert img.size[0] > 0 and img.size[1] > 0

    def test_soft_blend_atlas_dimensions_unchanged(self, blend_atlases):
        """soft_blend should not change atlas dimensions vs plain."""
        plain_img = Image.open(str(blend_atlases["plain_path"]))
        blend_img = Image.open(str(blend_atlases["blend_path"]))
        assert plain_img.size == blend_img.size
        assert set(blend_atlases["uv_plain"].keys()) == set(blend_atlases["uv_blend"].keys())

    def test_soft_blend_forces_fullslot(self, blend_atlases):
        """When soft_blend=True, no magenta sentinel pixels should appear."""
        img = Image.open(str(blend_atlases["blend_path"]))
        arr = np.array(img)
        magenta_mask = (arr[:, :, 0] == 255) & (arr[:, :, 1] == 0) & (arr[:, :, 2] == 255)
        assert magenta_mask.sum() == 0, "Magenta sentinel pixels found in soft_blend atlas"

    def test_soft_blend_alters_tile_edges(self, blend_atlases):
        """soft_blend should fade tile edges, reducing boundary contrast."""
        plain_arr = np.array(Image.open(str(blend_atlases["plain_path"]))).astype(np.float32)
        blend_arr = np.array(Image.open(str(blend_atlases["blend_path"]))).astype(np.float32)
        diff = np.abs(plain_arr - blend_arr).mean()
        assert diff > 0.5, f"Expected noticeable difference, got mean diff={diff:.2f}"

    def test_soft_blend_uv_values_in_range(self, blend_atlases):
        """UV coords from soft_blend atlas should be valid."""
        for fid, (u_min, v_min, u_max, v_max) in blend_atlases["uv_blend"].items():
            assert 0 <= u_min < u_max <= 1.0, f"{fid}: u range invalid"
            assert 0 <= v_min < v_max <= 1.0, f"{fid}: v range invalid"

    def test_soft_blend_with_forest_features(self, tmp_path):
        """soft_blend + forest features should produce a valid atlas."""
        from polygrid.biome_pipeline import build_feature_atlas, ForestRenderer
        grid, _store, coll = _shared_build_collection()
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

    def test_blend_fade_width_parameter(self, tmp_path):
        """Custom blend_fade_width should be accepted and alter result."""
        from polygrid.biome_pipeline import build_feature_atlas
        grid, _store, coll = _shared_build_collection()

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
        diff = np.abs(narrow_arr - wide_arr).mean()
        assert diff > 0.1, f"Expected different results, got mean diff={diff:.2f}"

    def test_soft_blend_partial_density(self, tmp_path):
        """soft_blend with partial density: only some tiles get features."""
        from polygrid.biome_pipeline import build_feature_atlas, ForestRenderer
        grid, _store, coll = _shared_build_collection()

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


# ═══════════════════════════════════════════════════════════════════
# Phase 17C — Ocean Pipeline Integration
# ═══════════════════════════════════════════════════════════════════

@needs_pil
@needs_models
class TestOceanRenderer:
    """Tests for OceanRenderer class."""

    def test_satisfies_protocol(self):
        from polygrid.biome_pipeline import OceanRenderer
        import inspect
        renderer = OceanRenderer()
        assert hasattr(renderer, "render")
        sig = inspect.signature(renderer.render)
        assert "ground_image" in sig.parameters
        assert "tile_id" in sig.parameters
        assert "density" in sig.parameters

    def test_render_produces_image(self):
        from polygrid.biome_pipeline import OceanRenderer
        renderer = OceanRenderer()
        ground = Image.new("RGB", (32, 32), (40, 100, 160))
        result = renderer.render(ground, "f0", 0.8, seed=42)
        assert result.size == (32, 32)
        assert result.mode == "RGB"

    def test_render_changes_pixels(self):
        from polygrid.biome_pipeline import OceanRenderer
        renderer = OceanRenderer()
        ground = Image.new("RGB", (32, 32), (40, 100, 160))
        result = renderer.render(ground, "f0", 0.8, seed=42)
        assert ground.tobytes() != result.tobytes()

    def test_depth_map_affects_output(self):
        from polygrid.biome_pipeline import OceanRenderer

        shallow_renderer = OceanRenderer(ocean_depth_map={"f0": 0.05})
        deep_renderer = OceanRenderer(ocean_depth_map={"f0": 0.95})
        ground = Image.new("RGB", (32, 32), (40, 100, 160))
        shallow = shallow_renderer.render(ground, "f0", 1.0, seed=42)
        deep = deep_renderer.render(ground, "f0", 1.0, seed=42)

        s_mean = np.array(shallow).mean()
        d_mean = np.array(deep).mean()
        assert s_mean > d_mean, f"shallow={s_mean:.1f} not brighter than deep={d_mean:.1f}"


@needs_pil
@needs_models
class TestOceanPipelineIntegration:
    """Integration tests: ocean + forest in the same atlas."""

    def test_ocean_only_atlas(self, tmp_path):
        """Atlas with only ocean tiles should render correctly."""
        from polygrid.biome_pipeline import build_feature_atlas, OceanRenderer
        grid, _store, coll = _shared_build_collection()
        face_ids = coll.face_ids

        density_map = {fid: 1.0 for fid in face_ids}
        depth_map = {fid: 0.5 for fid in face_ids}
        ocean_faces = set(face_ids)

        renderer = OceanRenderer(
            ocean_depth_map=depth_map,
            ocean_faces=ocean_faces,
            globe_grid=grid,
        )

        atlas_path, uv = build_feature_atlas(
            coll, globe_grid=grid,
            biome_renderers={"ocean": renderer},
            density_map=density_map,
            output_dir=tmp_path / "tiles",
            tile_size=64,
        )
        assert atlas_path.exists()
        assert len(uv) == len(face_ids)

    def test_mixed_forest_ocean_atlas(self, tmp_path):
        """Atlas with both forest and ocean tiles using biome_type_map."""
        from polygrid.biome_pipeline import (
            build_feature_atlas, ForestRenderer, OceanRenderer,
        )
        grid, _store, coll = _shared_build_collection()
        face_ids = coll.face_ids

        mid = len(face_ids) // 2
        forest_ids = face_ids[:mid]
        ocean_ids = face_ids[mid:]

        density_map = {fid: 0.8 for fid in face_ids}
        depth_map = {fid: 0.5 for fid in ocean_ids}
        biome_type_map = {
            **{fid: "forest" for fid in forest_ids},
            **{fid: "ocean" for fid in ocean_ids},
        }

        renderers = {
            "forest": ForestRenderer(),
            "ocean": OceanRenderer(
                ocean_depth_map=depth_map,
                ocean_faces=set(ocean_ids),
                globe_grid=grid,
            ),
        }

        atlas_path, uv = build_feature_atlas(
            coll, globe_grid=grid,
            biome_renderers=renderers,
            density_map=density_map,
            biome_type_map=biome_type_map,
            output_dir=tmp_path / "tiles",
            tile_size=64,
        )
        assert atlas_path.exists()
        assert len(uv) == len(face_ids)

    def test_ocean_tiles_differ_from_forest(self, tmp_path):
        """Ocean and forest renderers should produce visibly different output."""
        from polygrid.biome_pipeline import (
            build_feature_atlas, ForestRenderer, OceanRenderer,
        )
        grid, _store, coll = _shared_build_collection()
        face_ids = coll.face_ids
        density_map = {fid: 0.9 for fid in face_ids}

        # All-forest atlas
        forest_path, _ = build_feature_atlas(
            coll, globe_grid=grid,
            biome_renderers={"forest": ForestRenderer()},
            density_map=density_map,
            output_dir=tmp_path / "forest",
            tile_size=64,
        )

        # All-ocean atlas
        depth_map = {fid: 0.5 for fid in face_ids}
        ocean_path, _ = build_feature_atlas(
            coll, globe_grid=grid,
            biome_renderers={"ocean": OceanRenderer(ocean_depth_map=depth_map)},
            density_map=density_map,
            output_dir=tmp_path / "ocean",
            tile_size=64,
        )

        f_arr = np.array(Image.open(str(forest_path))).astype(float)
        o_arr = np.array(Image.open(str(ocean_path))).astype(float)
        diff = np.abs(f_arr - o_arr).mean()
        assert diff > 5, f"Forest and ocean too similar: mean_diff={diff:.1f}"

    def test_biome_type_map_routes_correctly(self, tmp_path):
        """biome_type_map should route tiles to the correct renderer."""
        from polygrid.biome_pipeline import (
            build_feature_atlas, ForestRenderer, OceanRenderer,
        )
        grid, _store, coll = _shared_build_collection()
        face_ids = coll.face_ids
        if len(face_ids) < 2:
            pytest.skip("Need at least 2 tiles")

        density_map = {fid: 0.9 for fid in face_ids}
        depth_map = {face_ids[0]: 0.2}
        biome_type_map = {face_ids[0]: "ocean"}

        renderers = {
            "forest": ForestRenderer(),
            "ocean": OceanRenderer(
                ocean_depth_map=depth_map,
                ocean_faces={face_ids[0]},
                globe_grid=grid,
            ),
        }

        atlas_path, uv = build_feature_atlas(
            coll, globe_grid=grid,
            biome_renderers=renderers,
            density_map=density_map,
            biome_type_map=biome_type_map,
            output_dir=tmp_path / "tiles",
            tile_size=64,
        )
        assert atlas_path.exists()
        assert len(uv) == len(face_ids)

    def test_land_tiles_unaffected_by_ocean(self, tmp_path):
        """Tiles without ocean density should not be ocean-rendered."""
        from polygrid.biome_pipeline import build_feature_atlas, OceanRenderer
        grid, _store, coll = _shared_build_collection()
        face_ids = coll.face_ids

        # No density → no features applied
        atlas_no_features, _ = build_feature_atlas(
            coll, globe_grid=grid,
            biome_renderers={"ocean": OceanRenderer()},
            output_dir=tmp_path / "no_feat",
            tile_size=64,
        )
        # Same with empty density map
        atlas_empty, _ = build_feature_atlas(
            coll, globe_grid=grid,
            biome_renderers={"ocean": OceanRenderer()},
            density_map={},
            output_dir=tmp_path / "empty",
            tile_size=64,
        )

        a = np.array(Image.open(str(atlas_no_features)))
        b = np.array(Image.open(str(atlas_empty)))
        assert np.array_equal(a, b)
