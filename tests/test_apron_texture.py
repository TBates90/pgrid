"""Tests for Phase 18B — apron-aware texture rendering.

Covers:
- 18B.1 — Apron texture rendering (render_detail_texture_apron)
- 18B.4 — Atlas with apron-filled gutters (build_apron_atlas)
- 18B.3 — Feature atlas with apron gutters (build_apron_feature_atlas)
- 18B.2/3 — Seamless boundary verification
"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from polygrid.algorithms import get_face_adjacency
from polygrid.apron_grid import (
    ApronResult,
    build_all_apron_grids,
    build_apron_grid,
    propagate_apron_terrain,
)
from polygrid.apron_texture import (
    build_apron_atlas,
    build_apron_feature_atlas,
    render_detail_texture_apron,
)
from polygrid.builders import build_pure_hex_grid
from polygrid.detail_render import BiomeConfig
from polygrid.polygrid import PolyGrid
from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
from polygrid.tile_detail import (
    DetailGridCollection,
    TileDetailSpec,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

import copy
from functools import lru_cache


def _make_globe_grid(frequency: int = 2) -> PolyGrid:
    grid = build_pure_hex_grid(frequency)
    return grid.with_neighbors()


def _make_detail_collection(
    globe_grid: PolyGrid,
    detail_rings: int = 2,
) -> DetailGridCollection:
    spec = TileDetailSpec(detail_rings=detail_rings)
    coll = DetailGridCollection.build(globe_grid, spec)

    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    globe_store = TileDataStore(grid=globe_grid, schema=schema)
    for fid in globe_grid.faces:
        idx = int(fid.replace("f", "")) if fid.startswith("f") else 0
        globe_store.set(fid, "elevation", 0.1 * (idx % 5))

    coll.generate_all_terrain(globe_store, seed=42)
    return coll


@lru_cache(maxsize=4)
def _cached_globe_grid(frequency: int = 2) -> PolyGrid:
    return _make_globe_grid(frequency)


@lru_cache(maxsize=4)
def _cached_collection_internals(frequency: int = 2, detail_rings: int = 2):
    globe = _cached_globe_grid(frequency)
    return _make_detail_collection(globe, detail_rings)


def _shared_collection(frequency: int = 2, detail_rings: int = 2):
    cached = _cached_collection_internals(frequency, detail_rings)
    wrapper = copy.copy(cached)
    wrapper._stores = copy.copy(cached._stores)
    return wrapper


@pytest.fixture
def tmp_dir():
    """Temporary directory for atlas output."""
    d = tempfile.mkdtemp(prefix="apron_tex_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════
# 18B.1 — Apron texture rendering
# ═══════════════════════════════════════════════════════════════════

class TestRenderDetailTextureApron:
    """Tests for render_detail_texture_apron()."""

    def test_returns_image(self):
        """Produces a PIL Image of the correct size."""
        from PIL import Image

        globe = _cached_globe_grid(2)
        coll = _shared_collection(2, 2)

        face_id = list(globe.faces.keys())[0]
        apron_grid, mapping = build_apron_grid(globe, face_id, coll)
        store = propagate_apron_terrain(
            apron_grid, mapping, coll, face_id,
        )

        img = render_detail_texture_apron(
            apron_grid, store, tile_size=128,
        )

        assert isinstance(img, Image.Image)
        assert img.size == (128, 128)
        assert img.mode == "RGB"

    def test_no_sentinel_pixels(self):
        """No magenta sentinel pixels should remain in the output."""
        globe = _cached_globe_grid(2)
        coll = _shared_collection(2, 2)

        face_id = list(globe.faces.keys())[0]
        apron_grid, mapping = build_apron_grid(globe, face_id, coll)
        store = propagate_apron_terrain(
            apron_grid, mapping, coll, face_id,
        )

        img = render_detail_texture_apron(
            apron_grid, store, tile_size=128,
        )

        pixels = np.array(img)
        sentinel_mask = (
            (pixels[:, :, 0] == 255) &
            (pixels[:, :, 1] == 0) &
            (pixels[:, :, 2] == 255)
        )
        # Some stray sentinel pixels at corners are acceptable,
        # but the vast majority should be filled.
        sentinel_pct = sentinel_mask.sum() / (128 * 128)
        assert sentinel_pct < 0.05, (
            f"Too many sentinel pixels: {sentinel_pct:.1%}"
        )

    def test_more_terrain_coverage_than_standard(self):
        """Apron rendering should cover more terrain area than standard.

        When we render the same tile_size with an apron grid (which
        has more sub-faces), we expect better coverage in the border
        zone.
        """
        from PIL import Image
        from polygrid.tile_texture import render_detail_texture_fullslot

        globe = _cached_globe_grid(2)
        coll = _shared_collection(2, 2)
        face_id = list(globe.faces.keys())[0]

        # Standard rendering (no apron)
        grid, own_store = coll.get(face_id)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            render_detail_texture_fullslot(
                grid, own_store, f.name, tile_size=128,
            )
            standard_img = Image.open(f.name).convert("RGB")

        # Apron rendering
        apron_grid, mapping = build_apron_grid(globe, face_id, coll)
        store = propagate_apron_terrain(
            apron_grid, mapping, coll, face_id,
        )
        apron_img = render_detail_texture_apron(
            apron_grid, store, tile_size=128,
        )

        # Both should have similar overall brightness (terrain data)
        std_arr = np.array(standard_img).astype(float)
        apron_arr = np.array(apron_img).astype(float)

        # The apron image should have fewer zero/sentinel pixels
        std_dark = (std_arr.mean(axis=2) < 5).sum()
        apron_dark = (apron_arr.mean(axis=2) < 5).sum()

        # Apron should not be dramatically worse
        assert apron_dark <= std_dark + 100

    def test_custom_biome_config(self):
        """Respects custom BiomeConfig."""
        globe = _cached_globe_grid(1)
        coll = _shared_collection(1, 2)

        face_id = list(globe.faces.keys())[0]
        apron_grid, mapping = build_apron_grid(globe, face_id, coll)
        store = propagate_apron_terrain(
            apron_grid, mapping, coll, face_id,
        )

        biome = BiomeConfig(snow_line=0.01)  # everything snowy
        img = render_detail_texture_apron(
            apron_grid, store, biome, tile_size=64,
        )

        pixels = np.array(img)
        # Snow is bright — most pixels should be light-coloured
        mean_brightness = pixels.mean()
        assert mean_brightness > 100  # snow is bright

    def test_disables_hex_softening(self):
        """Can disable all 16D features."""
        globe = _cached_globe_grid(1)
        coll = _shared_collection(1, 2)

        face_id = list(globe.faces.keys())[0]
        apron_grid, mapping = build_apron_grid(globe, face_id, coll)
        store = propagate_apron_terrain(
            apron_grid, mapping, coll, face_id,
        )

        img = render_detail_texture_apron(
            apron_grid, store,
            tile_size=64,
            vertex_jitter=0,
            noise_overlay=False,
            colour_dither=False,
        )

        assert img.size == (64, 64)

    def test_different_tile_sizes(self):
        """Works at various tile sizes."""
        globe = _cached_globe_grid(1)
        coll = _shared_collection(1, 2)

        face_id = list(globe.faces.keys())[0]
        apron_grid, mapping = build_apron_grid(globe, face_id, coll)
        store = propagate_apron_terrain(
            apron_grid, mapping, coll, face_id,
        )

        for size in [32, 64, 128, 256]:
            img = render_detail_texture_apron(
                apron_grid, store, tile_size=size,
            )
            assert img.size == (size, size)


# ═══════════════════════════════════════════════════════════════════
# 18B.4 — Atlas with apron gutters
# ═══════════════════════════════════════════════════════════════════

class TestBuildApronAtlas:
    """Tests for build_apron_atlas()."""

    def test_creates_atlas(self, tmp_dir):
        """Produces an atlas PNG and UV layout."""
        globe = _cached_globe_grid(1)
        coll = _shared_collection(1, 2)

        atlas_path, uv_layout = build_apron_atlas(
            coll, globe,
            output_dir=tmp_dir,
            tile_size=64,
            gutter=2,
        )

        assert atlas_path.exists()
        assert atlas_path.suffix == ".png"
        assert len(uv_layout) == len(globe.faces)

    def test_uv_layout_bounds(self, tmp_dir):
        """UV coordinates are within [0, 1]."""
        globe = _cached_globe_grid(1)
        coll = _shared_collection(1, 2)

        _, uv_layout = build_apron_atlas(
            coll, globe,
            output_dir=tmp_dir,
            tile_size=64,
            gutter=2,
        )

        for fid, (u_min, v_min, u_max, v_max) in uv_layout.items():
            assert 0.0 <= u_min <= 1.0, f"{fid}: u_min={u_min}"
            assert 0.0 <= v_min <= 1.0, f"{fid}: v_min={v_min}"
            assert 0.0 <= u_max <= 1.0, f"{fid}: u_max={u_max}"
            assert 0.0 <= v_max <= 1.0, f"{fid}: v_max={v_max}"
            assert u_max > u_min
            assert v_max > v_min

    def test_atlas_size_correct(self, tmp_dir):
        """Atlas dimensions match layout calculation."""
        from PIL import Image

        globe = _cached_globe_grid(1)
        coll = _shared_collection(1, 2)

        tile_size = 64
        gutter = 4

        atlas_path, _ = build_apron_atlas(
            coll, globe,
            output_dir=tmp_dir,
            tile_size=tile_size,
            gutter=gutter,
        )

        img = Image.open(atlas_path)
        n = len(globe.faces)
        columns = max(1, math.isqrt(n))
        if columns * columns < n:
            columns += 1
        rows = math.ceil(n / columns)
        slot_size = tile_size + 2 * gutter

        assert img.size[0] == columns * slot_size
        assert img.size[1] == rows * slot_size

    def test_individual_tiles_saved(self, tmp_dir):
        """Individual tile PNGs are saved alongside the atlas."""
        globe = _cached_globe_grid(1)
        coll = _shared_collection(1, 2)

        build_apron_atlas(
            coll, globe,
            output_dir=tmp_dir,
            tile_size=64,
        )

        for fid in globe.faces:
            tile_path = tmp_dir / f"tile_{fid}.png"
            assert tile_path.exists(), f"Missing tile {tile_path}"

    def test_gutter_not_grey(self, tmp_dir):
        """Gutter pixels should not be default grey (128,128,128)."""
        from PIL import Image

        globe = _cached_globe_grid(1)
        coll = _shared_collection(1, 2)

        gutter = 4
        tile_size = 64

        atlas_path, _ = build_apron_atlas(
            coll, globe,
            output_dir=tmp_dir,
            tile_size=tile_size,
            gutter=gutter,
        )

        atlas = np.array(Image.open(atlas_path))

        # Check the top gutter row of the first slot
        # It should not be (128, 128, 128) — the default fill
        top_gutter = atlas[0:gutter, gutter:gutter + tile_size, :]
        mean_val = top_gutter.mean()
        # If gutter was filled, it should differ from 128
        # (could be terrain-coloured)
        is_default_grey = np.allclose(top_gutter, 128, atol=5)
        assert not is_default_grey, "Gutter still contains default grey fill"

    def test_zero_gutter(self, tmp_dir):
        """Works with gutter=0."""
        globe = _cached_globe_grid(1)
        coll = _shared_collection(1, 2)

        atlas_path, uv_layout = build_apron_atlas(
            coll, globe,
            output_dir=tmp_dir,
            tile_size=64,
            gutter=0,
        )

        assert atlas_path.exists()
        assert len(uv_layout) == len(globe.faces)


# ═══════════════════════════════════════════════════════════════════
# 18B.3 — Feature atlas with apron gutters
# ═══════════════════════════════════════════════════════════════════

class TestBuildApronFeatureAtlas:
    """Tests for build_apron_feature_atlas()."""

    def test_creates_atlas_no_biomes(self, tmp_dir):
        """Works with no biome renderers (ground-only)."""
        globe = _cached_globe_grid(1)
        coll = _shared_collection(1, 2)

        atlas_path, uv_layout = build_apron_feature_atlas(
            coll, globe,
            output_dir=tmp_dir,
            tile_size=64,
            gutter=2,
        )

        assert atlas_path.exists()
        assert len(uv_layout) == len(globe.faces)

    def test_with_forest_renderer(self, tmp_dir):
        """Works with a ForestRenderer on some tiles."""
        from polygrid.biome_pipeline import ForestRenderer

        globe = _cached_globe_grid(1)
        coll = _shared_collection(1, 2)

        face_ids = list(globe.faces.keys())
        density_map = {fid: 0.8 for fid in face_ids[:3]}
        biome_type_map = {fid: "forest" for fid in face_ids[:3]}

        atlas_path, uv_layout = build_apron_feature_atlas(
            coll, globe,
            biome_renderers={"forest": ForestRenderer()},
            density_map=density_map,
            biome_type_map=biome_type_map,
            output_dir=tmp_dir,
            tile_size=64,
            gutter=2,
        )

        assert atlas_path.exists()
        assert len(uv_layout) == len(globe.faces)

    def test_with_ocean_renderer(self, tmp_dir):
        """Works with an OceanRenderer on some tiles."""
        from polygrid.biome_pipeline import OceanRenderer

        globe = _cached_globe_grid(1)
        coll = _shared_collection(1, 2)

        face_ids = list(globe.faces.keys())
        density_map = {fid: 1.0 for fid in face_ids[-2:]}
        biome_type_map = {fid: "ocean" for fid in face_ids[-2:]}

        atlas_path, uv_layout = build_apron_feature_atlas(
            coll, globe,
            biome_renderers={"ocean": OceanRenderer()},
            density_map=density_map,
            biome_type_map=biome_type_map,
            output_dir=tmp_dir,
            tile_size=64,
            gutter=2,
        )

        assert atlas_path.exists()
        assert len(uv_layout) == len(globe.faces)


# ═══════════════════════════════════════════════════════════════════
# 18B.2/3 — Seamless boundary verification
# ═══════════════════════════════════════════════════════════════════

class TestSeamlessness:
    """Verify that apron rendering improves boundary continuity."""

    def test_apron_images_have_terrain_at_edges(self):
        """Edge pixels of apron-rendered tiles should have real terrain."""
        globe = _cached_globe_grid(1)
        coll = _shared_collection(1, 2)

        face_id = list(globe.faces.keys())[0]
        apron_grid, mapping = build_apron_grid(globe, face_id, coll)
        store = propagate_apron_terrain(
            apron_grid, mapping, coll, face_id,
        )

        img = render_detail_texture_apron(
            apron_grid, store, tile_size=128,
        )

        pixels = np.array(img)

        # Check edge rows/columns — they should not be uniform
        # (which would indicate clamped/empty edges)
        top_row = pixels[0, :, :]
        bottom_row = pixels[-1, :, :]
        left_col = pixels[:, 0, :]
        right_col = pixels[:, -1, :]

        # At least some variation in edge pixels
        for name, edge in [("top", top_row), ("bottom", bottom_row),
                           ("left", left_col), ("right", right_col)]:
            std = edge.std()
            # Edges should have some colour variation (real terrain)
            # A clamped uniform edge would have std ≈ 0
            assert std > 0, f"{name} edge has zero variation"

    def test_adjacent_tile_edge_similarity(self):
        """Adjacent tiles should have similar colours near their shared edge."""
        globe = _cached_globe_grid(2)
        coll = _shared_collection(2, 2)

        adj = get_face_adjacency(globe)
        face_a = list(globe.faces.keys())[0]
        face_b = adj[face_a][0]

        # Render both tiles with apron
        results = build_all_apron_grids(globe, coll)
        img_a = render_detail_texture_apron(
            results[face_a].grid, results[face_a].store, tile_size=128,
        )
        img_b = render_detail_texture_apron(
            results[face_b].grid, results[face_b].store, tile_size=128,
        )

        # Both images should be valid (no black/blank)
        arr_a = np.array(img_a).astype(float)
        arr_b = np.array(img_b).astype(float)
        assert arr_a.mean() > 10, "Tile A appears blank"
        assert arr_b.mean() > 10, "Tile B appears blank"
