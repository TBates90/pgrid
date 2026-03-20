# TODO REMOVE — Tests dead module biome_topology.py.
"""Tests for Phase 18C — topology-aware biome feature rendering."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict

import pytest

from polygrid import PolyGrid
from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
from polygrid.biome_topology import (
    SubfaceTree,
    SubfaceOceanProps,
    scatter_trees_on_grid,
    render_topology_forest,
    compute_subface_ocean_depth,
    identify_coastal_subfaces,
    compute_ocean_subface_props,
    render_topology_ocean,
    render_hybrid_biome,
    TopologyForestRenderer,
    TopologyOceanRenderer,
    _polygon_area_2d,
)

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

np = pytest.importorskip("numpy")


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def hex_grid():
    """Build a small pure-hex detail grid for testing."""
    from polygrid.builders import build_pure_hex_grid
    return build_pure_hex_grid(2)


@pytest.fixture
def hex_store(hex_grid):
    """TileDataStore with elevation for the hex grid."""
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=hex_grid, schema=schema)
    rng = random.Random(42)
    for fid in hex_grid.faces:
        store.set(fid, "elevation", rng.uniform(0.0, 0.3))
    return store


@pytest.fixture
def ocean_store(hex_grid):
    """TileDataStore with low (underwater) elevation for ocean testing."""
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=hex_grid, schema=schema)
    rng = random.Random(42)
    for fid in hex_grid.faces:
        store.set(fid, "elevation", rng.uniform(-0.1, 0.1))
    return store


@pytest.fixture
def mixed_store(hex_grid):
    """Store where roughly half the sub-faces are above and half below water."""
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=hex_grid, schema=schema)
    rng = random.Random(123)
    face_ids = sorted(hex_grid.faces.keys())
    for i, fid in enumerate(face_ids):
        if i < len(face_ids) // 2:
            store.set(fid, "elevation", rng.uniform(0.15, 0.4))  # land
        else:
            store.set(fid, "elevation", rng.uniform(-0.05, 0.08))  # ocean
    return store


@pytest.fixture
def ground_image():
    """A simple 128×128 ground texture."""
    return Image.new("RGB", (128, 128), (100, 130, 80))


# ═══════════════════════════════════════════════════════════════════
# 18C.1 — Forest scatter tests
# ═══════════════════════════════════════════════════════════════════

class TestScatterTreesOnGrid:
    """Tests for topology-aware tree placement."""

    def test_trees_placed_at_subface_centroids(self, hex_grid, hex_store):
        """Trees should be placed near sub-face centroids."""
        trees = scatter_trees_on_grid(
            hex_grid, hex_store,
            tile_size=128,
            density=1.0,
            seed=42,
        )
        assert len(trees) > 0
        # Every tree should reference a valid face
        face_ids_with_trees = {t.face_id for t in trees}
        for fid in face_ids_with_trees:
            assert fid in hex_grid.faces

    def test_deterministic_placement(self, hex_grid, hex_store):
        """Same seed should produce same trees."""
        trees_a = scatter_trees_on_grid(hex_grid, hex_store, seed=99, density=0.8)
        trees_b = scatter_trees_on_grid(hex_grid, hex_store, seed=99, density=0.8)
        assert len(trees_a) == len(trees_b)
        for a, b in zip(trees_a, trees_b):
            assert a.face_id == b.face_id
            assert abs(a.px - b.px) < 0.01
            assert abs(a.py - b.py) < 0.01

    def test_density_controls_count(self, hex_grid, hex_store):
        """Higher density should produce more trees."""
        low = scatter_trees_on_grid(hex_grid, hex_store, density=0.2, seed=42)
        high = scatter_trees_on_grid(hex_grid, hex_store, density=0.9, seed=42)
        assert len(high) > len(low)

    def test_zero_density_no_trees(self, hex_grid, hex_store):
        """Zero density should produce no trees."""
        trees = scatter_trees_on_grid(hex_grid, hex_store, density=0.0, seed=42)
        assert len(trees) == 0

    def test_tree_radius_proportional_to_area(self, hex_grid, hex_store):
        """Trees should have radius related to sub-face area."""
        trees = scatter_trees_on_grid(
            hex_grid, hex_store,
            density=1.0,
            min_radius=2.0,
            max_radius=10.0,
            seed=42,
        )
        assert len(trees) > 0
        for t in trees:
            assert 2.0 <= t.radius <= 10.0

    def test_trees_sorted_by_depth(self, hex_grid, hex_store):
        """Trees should be sorted by depth (back to front)."""
        trees = scatter_trees_on_grid(hex_grid, hex_store, density=0.8, seed=42)
        if len(trees) > 1:
            for i in range(len(trees) - 1):
                assert trees[i].depth <= trees[i + 1].depth

    def test_alpine_thinning(self, hex_grid):
        """High-elevation faces should have fewer trees."""
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=hex_grid, schema=schema)
        # Set all faces to high elevation
        for fid in hex_grid.faces:
            store.set(fid, "elevation", 0.95)
        trees = scatter_trees_on_grid(
            hex_grid, store,
            density=0.8,
            alpine_threshold=0.5,
            elevation_range=(0.0, 1.0),
            seed=42,
        )
        # Should have very few or no trees at 95% elevation
        low_store = TileDataStore(grid=hex_grid, schema=schema)
        for fid in hex_grid.faces:
            low_store.set(fid, "elevation", 0.1)
        low_trees = scatter_trees_on_grid(
            hex_grid, low_store,
            density=0.8,
            alpine_threshold=0.5,
            elevation_range=(0.0, 1.0),
            seed=42,
        )
        assert len(trees) <= len(low_trees)


class TestRenderTopologyForest:
    """Tests for topology-driven forest rendering."""

    def test_produces_valid_image(self, hex_grid, hex_store, ground_image):
        trees = scatter_trees_on_grid(
            hex_grid, hex_store, tile_size=128, density=0.8, seed=42,
        )
        result = render_topology_forest(ground_image, trees, seed=42)
        assert result.size == (128, 128)
        assert result.mode == "RGBA"

    def test_forest_modifies_image(self, hex_grid, hex_store, ground_image):
        """Forest rendering should change the ground image."""
        trees = scatter_trees_on_grid(
            hex_grid, hex_store, tile_size=128, density=0.8, seed=42,
        )
        result = render_topology_forest(ground_image, trees, seed=42)
        orig_arr = np.array(ground_image.convert("RGBA"))
        result_arr = np.array(result)
        assert not np.array_equal(orig_arr, result_arr)

    def test_no_trees_minimal_change(self, ground_image):
        """No trees should only add undergrowth (density-dependent)."""
        result = render_topology_forest(ground_image, [], density=0.0, seed=42)
        assert result.size == (128, 128)


# ═══════════════════════════════════════════════════════════════════
# 18C.2 — Ocean sub-face tests
# ═══════════════════════════════════════════════════════════════════

class TestSubfaceOceanDepth:
    """Tests for per-sub-face ocean depth computation."""

    def test_depth_from_elevation(self, hex_grid, ocean_store):
        depths = compute_subface_ocean_depth(hex_grid, ocean_store)
        assert len(depths) == len(hex_grid.faces)
        for fid, d in depths.items():
            assert 0.0 <= d <= 1.0

    def test_above_water_depth_zero(self, hex_grid):
        """Sub-faces above water level should have depth 0."""
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=hex_grid, schema=schema)
        for fid in hex_grid.faces:
            store.set(fid, "elevation", 0.5)  # above water
        depths = compute_subface_ocean_depth(hex_grid, store, water_level=0.12)
        for d in depths.values():
            assert d == 0.0

    def test_below_water_depth_positive(self, hex_grid):
        """Sub-faces below water level should have positive depth."""
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=hex_grid, schema=schema)
        for fid in hex_grid.faces:
            store.set(fid, "elevation", 0.0)
        depths = compute_subface_ocean_depth(hex_grid, store, water_level=0.12)
        for d in depths.values():
            assert d > 0.0

    def test_per_subface_variation(self, hex_grid, ocean_store):
        """Different sub-faces should have different depths."""
        depths = compute_subface_ocean_depth(hex_grid, ocean_store)
        values = list(depths.values())
        assert len(set(round(v, 6) for v in values)) > 1


class TestCoastalSubfaces:
    """Tests for coastal sub-face identification."""

    def test_no_land_no_coast(self, hex_grid, ocean_store):
        """All-ocean grid should have no coastal faces (or only boundary)."""
        # Set all to underwater
        for fid in hex_grid.faces:
            ocean_store.set(fid, "elevation", 0.0)
        depths = compute_subface_ocean_depth(hex_grid, ocean_store)
        coastal = identify_coastal_subfaces(hex_grid, depths)
        # All underwater — no land neighbours within the grid
        assert len(coastal) == 0

    def test_mixed_has_coastal(self, hex_grid, mixed_store):
        """Mixed land/ocean should produce coastal sub-faces."""
        depths = compute_subface_ocean_depth(hex_grid, mixed_store)
        coastal = identify_coastal_subfaces(hex_grid, depths)
        # Should identify some coastal faces at the boundary
        assert len(coastal) > 0

    def test_coastal_are_underwater(self, hex_grid, mixed_store):
        """Coastal sub-faces should be underwater."""
        depths = compute_subface_ocean_depth(hex_grid, mixed_store)
        coastal = identify_coastal_subfaces(hex_grid, depths)
        for fid in coastal:
            assert depths[fid] > 0.05  # above land_threshold


class TestComputeOceanSubfaceProps:
    """Tests for full ocean property computation."""

    def test_props_for_all_faces(self, hex_grid, ocean_store):
        props = compute_ocean_subface_props(hex_grid, ocean_store, tile_size=128)
        # Should have props for faces with valid geometry
        assert len(props) > 0
        for p in props:
            assert isinstance(p, SubfaceOceanProps)
            assert p.face_id in hex_grid.faces
            assert 0.0 <= p.depth <= 1.0
            assert p.area > 0


class TestRenderTopologyOcean:
    """Tests for topology-driven ocean rendering."""

    def test_produces_valid_image(self, hex_grid, ocean_store, ground_image):
        props = compute_ocean_subface_props(
            hex_grid, ocean_store, tile_size=128,
        )
        result = render_topology_ocean(ground_image, props, seed=42)
        assert result.size == (128, 128)
        assert result.mode == "RGB"

    def test_ocean_modifies_image(self, hex_grid, ocean_store, ground_image):
        props = compute_ocean_subface_props(
            hex_grid, ocean_store, tile_size=128,
        )
        result = render_topology_ocean(ground_image, props, seed=42)
        orig_arr = np.array(ground_image)
        result_arr = np.array(result)
        assert not np.array_equal(orig_arr, result_arr)

    def test_coastal_foam(self, hex_grid, mixed_store, ground_image):
        """Coastal sub-faces should get foam overlay."""
        props = compute_ocean_subface_props(
            hex_grid, mixed_store, tile_size=128,
        )
        has_coastal = any(p.is_coastal for p in props)
        # Mixed store should produce some coastal sub-faces
        assert has_coastal
        result = render_topology_ocean(ground_image, props, seed=42)
        assert result.size == (128, 128)


# ═══════════════════════════════════════════════════════════════════
# 18C.3 — Hybrid rendering tests
# ═══════════════════════════════════════════════════════════════════

class TestHybridRendering:
    """Tests for hybrid topology + pixel rendering."""

    def test_forest_hybrid(self, hex_grid, hex_store, ground_image):
        result = render_hybrid_biome(
            ground_image, hex_grid, hex_store,
            biome_type="forest",
            tile_size=128,
            density=0.7,
            seed=42,
        )
        assert result.size == (128, 128)

    def test_ocean_hybrid(self, hex_grid, ocean_store, ground_image):
        result = render_hybrid_biome(
            ground_image, hex_grid, ocean_store,
            biome_type="ocean",
            tile_size=128,
            density=0.8,
            seed=42,
        )
        assert result.size == (128, 128)

    def test_noise_overlay_applied(self, hex_grid, hex_store, ground_image):
        """Noise overlay should add micro-detail variation."""
        no_noise = render_hybrid_biome(
            ground_image, hex_grid, hex_store,
            biome_type="forest",
            density=0.5,
            noise_overlay=False,
            seed=42,
        )
        with_noise = render_hybrid_biome(
            ground_image, hex_grid, hex_store,
            biome_type="forest",
            density=0.5,
            noise_overlay=True,
            seed=42,
        )
        # They should differ due to noise
        a = np.array(no_noise.convert("RGB"))
        b = np.array(with_noise.convert("RGB"))
        assert not np.array_equal(a, b)


# ═══════════════════════════════════════════════════════════════════
# Renderer class tests
# ═══════════════════════════════════════════════════════════════════

class TestTopologyForestRenderer:
    """Tests for the TopologyForestRenderer class."""

    def test_with_grid_context(self, hex_grid, hex_store, ground_image):
        renderer = TopologyForestRenderer()
        renderer.set_grid_context(hex_grid, hex_store)
        result = renderer.render(
            ground_image, "test_tile", 0.8, seed=42,
        )
        assert result.size == (128, 128)

    def test_fallback_without_context(self, ground_image):
        renderer = TopologyForestRenderer()
        # No grid context — should fall back to pixel-based
        result = renderer.render(
            ground_image, "test_tile", 0.8, seed=42,
        )
        assert result.size == (128, 128)

    def test_satisfies_biome_renderer_protocol(self, ground_image):
        """Should work as a BiomeRenderer drop-in."""
        renderer = TopologyForestRenderer()
        # BiomeRenderer protocol requires: render(ground, tile_id, density, ...)
        result = renderer.render(
            ground_image, "face_0", 0.5, seed=42,
        )
        assert result is not None


class TestTopologyOceanRenderer:
    """Tests for the TopologyOceanRenderer class."""

    def test_with_grid_context(self, hex_grid, ocean_store, ground_image):
        renderer = TopologyOceanRenderer()
        renderer.set_grid_context(hex_grid, ocean_store)
        result = renderer.render(
            ground_image, "test_tile", 0.8, seed=42,
        )
        assert result.size == (128, 128)

    def test_fallback_without_context(self, ground_image):
        renderer = TopologyOceanRenderer()
        result = renderer.render(
            ground_image, "test_tile", 0.8, seed=42,
        )
        assert result.size == (128, 128)

    def test_satisfies_biome_renderer_protocol(self, ground_image):
        renderer = TopologyOceanRenderer()
        result = renderer.render(
            ground_image, "face_0", 0.5, seed=42,
        )
        assert result is not None


# ═══════════════════════════════════════════════════════════════════
# Utility tests
# ═══════════════════════════════════════════════════════════════════

class TestPolygonArea:
    def test_unit_square(self):
        verts = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert abs(_polygon_area_2d(verts) - 1.0) < 1e-6

    def test_triangle(self):
        verts = [(0, 0), (4, 0), (0, 3)]
        assert abs(_polygon_area_2d(verts) - 6.0) < 1e-6

    def test_degenerate(self):
        assert _polygon_area_2d([(0, 0), (1, 1)]) == 0.0
        assert _polygon_area_2d([]) == 0.0
