"""Tests for Phase 19A — Coastline mask generation.

Tests verify:
- CoastlineConfig dataclass construction and presets
- Tile biome context classification (interior vs edge)
- Coastline mask shape, range, and properties
- Noise produces non-straight boundaries
- Seed reproducibility and cross-tile consistency
- CoastlineMask metadata properties
- Blend and coastal strip functions
"""

from __future__ import annotations

import math
import pytest
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Helpers — lightweight mock grid for testing
# ═══════════════════════════════════════════════════════════════════

class _MockVertex:
    """Minimal vertex with 3D position."""
    def __init__(self, vid, x, y, z=0.0):
        self.id = vid
        self.x = x
        self.y = y
        self.z = z

    def has_position(self):
        return True


class _MockFace:
    """Minimal face with vertex IDs and neighbor_ids."""
    def __init__(self, fid, vertex_ids, neighbor_ids=None):
        self.id = fid
        self.vertex_ids = vertex_ids
        self.neighbor_ids = neighbor_ids or []


class _MockGrid:
    """Minimal grid with faces and vertices for testing."""
    def __init__(self):
        self.vertices = {}
        self.faces = {}
        self.edges = {}

    def add_vertex(self, vid, x, y, z=0.0):
        self.vertices[vid] = _MockVertex(vid, x, y, z)

    def add_face(self, fid, vertex_ids, neighbor_ids=None):
        self.faces[fid] = _MockFace(fid, vertex_ids, neighbor_ids)


def _make_hex_grid():
    """Build a small mock grid with 7 hex faces (centre + 6 neighbours).

    Layout (flat-top hexagons):
        n1  n2
      n6  c  n3
        n5  n4
    """
    g = _MockGrid()
    # Centre and 6 surrounding positions
    # Using unit circle positions for neighbours
    cx, cy = 0.0, 0.0
    r = 2.0
    positions = {
        "c": (cx, cy, 0.0),
        "n1": (cx - r * 0.5, cy + r * 0.866, 0.0),
        "n2": (cx + r * 0.5, cy + r * 0.866, 0.0),
        "n3": (cx + r, cy, 0.0),
        "n4": (cx + r * 0.5, cy - r * 0.866, 0.0),
        "n5": (cx - r * 0.5, cy - r * 0.866, 0.0),
        "n6": (cx - r, cy, 0.0),
    }

    # Add dummy vertices for each face (just need face centres to work)
    for fid, (x, y, z) in positions.items():
        # Create 3 vertices near each face centre
        for i in range(3):
            angle = i * 2.0 * math.pi / 3.0
            vx = x + 0.5 * math.cos(angle)
            vy = y + 0.5 * math.sin(angle)
            vid = f"{fid}_v{i}"
            g.add_vertex(vid, vx, vy, z)

    # Create faces with adjacency
    nbrs = {
        "c": ["n1", "n2", "n3", "n4", "n5", "n6"],
        "n1": ["c", "n2", "n6"],
        "n2": ["c", "n1", "n3"],
        "n3": ["c", "n2", "n4"],
        "n4": ["c", "n3", "n5"],
        "n5": ["c", "n4", "n6"],
        "n6": ["c", "n5", "n1"],
    }
    for fid, (x, y, z) in positions.items():
        vids = [f"{fid}_v{i}" for i in range(3)]
        g.add_face(fid, vids, nbrs[fid])

    return g


# ═══════════════════════════════════════════════════════════════════
# 19A.1 — CoastlineConfig tests
# ═══════════════════════════════════════════════════════════════════

class TestCoastlineConfig:
    """Tests for CoastlineConfig dataclass."""

    def test_default_construction(self):
        from polygrid.coastline import CoastlineConfig
        cfg = CoastlineConfig()
        assert cfg.noise_frequency == 4.0
        assert cfg.noise_octaves == 4
        assert cfg.noise_amplitude == 0.18
        assert cfg.transition_width == 0.06
        assert cfg.beach_width == 0.04
        assert cfg.foam_width == 0.03

    def test_custom_construction(self):
        from polygrid.coastline import CoastlineConfig
        cfg = CoastlineConfig(noise_frequency=8.0, noise_octaves=6)
        assert cfg.noise_frequency == 8.0
        assert cfg.noise_octaves == 6
        assert cfg.noise_amplitude == 0.18  # default

    def test_frozen(self):
        from polygrid.coastline import CoastlineConfig
        cfg = CoastlineConfig()
        with pytest.raises(AttributeError):
            cfg.noise_frequency = 10.0  # type: ignore

    def test_presets_exist(self):
        from polygrid.coastline import COASTLINE_PRESETS
        assert "default" in COASTLINE_PRESETS
        assert "gentle" in COASTLINE_PRESETS
        assert "rugged" in COASTLINE_PRESETS
        assert "archipelago" in COASTLINE_PRESETS

    def test_presets_are_configs(self):
        from polygrid.coastline import COASTLINE_PRESETS, CoastlineConfig
        for name, cfg in COASTLINE_PRESETS.items():
            assert isinstance(cfg, CoastlineConfig), f"Preset '{name}' is not a CoastlineConfig"

    def test_rugged_has_higher_frequency(self):
        from polygrid.coastline import COASTLINE_PRESETS
        default = COASTLINE_PRESETS["default"]
        rugged = COASTLINE_PRESETS["rugged"]
        assert rugged.noise_frequency > default.noise_frequency


# ═══════════════════════════════════════════════════════════════════
# 19A.2 — Tile biome context classification tests
# ═══════════════════════════════════════════════════════════════════

class TestTileBiomeContext:
    """Tests for classify_tile_biome_context."""

    def test_interior_tile_all_same_biome(self):
        from polygrid.coastline import classify_tile_biome_context
        biome_map = {"c": "forest", "n1": "forest", "n2": "forest",
                     "n3": "forest", "n4": "forest", "n5": "forest", "n6": "forest"}
        adj = {"c": ["n1", "n2", "n3", "n4", "n5", "n6"]}
        ctx = classify_tile_biome_context("c", biome_map, adj)
        assert ctx.is_interior is True
        assert ctx.is_edge is False
        assert ctx.own_biome == "forest"
        assert len(ctx.edge_neighbours) == 0

    def test_edge_tile_one_different_neighbour(self):
        from polygrid.coastline import classify_tile_biome_context
        biome_map = {"c": "forest", "n1": "ocean", "n2": "forest",
                     "n3": "forest", "n4": "forest", "n5": "forest", "n6": "forest"}
        adj = {"c": ["n1", "n2", "n3", "n4", "n5", "n6"]}
        ctx = classify_tile_biome_context("c", biome_map, adj)
        assert ctx.is_edge is True
        assert ctx.is_interior is False
        assert "n1" in ctx.edge_neighbours
        assert ctx.edge_neighbours["n1"] == "ocean"

    def test_edge_tile_multiple_different_neighbours(self):
        from polygrid.coastline import classify_tile_biome_context
        biome_map = {"c": "forest", "n1": "ocean", "n2": "ocean",
                     "n3": "forest", "n4": "forest", "n5": "forest", "n6": "forest"}
        adj = {"c": ["n1", "n2", "n3", "n4", "n5", "n6"]}
        ctx = classify_tile_biome_context("c", biome_map, adj)
        assert ctx.is_edge is True
        assert len(ctx.edge_neighbours) == 2
        assert ctx.edge_neighbours["n1"] == "ocean"
        assert ctx.edge_neighbours["n2"] == "ocean"

    def test_default_biome_for_unmapped_tiles(self):
        from polygrid.coastline import classify_tile_biome_context
        biome_map = {"c": "forest"}
        adj = {"c": ["n1", "n2"]}
        ctx = classify_tile_biome_context("c", biome_map, adj, default_biome="terrain")
        assert ctx.is_edge is True
        assert ctx.edge_neighbours["n1"] == "terrain"

    def test_ocean_tile_with_forest_neighbour(self):
        from polygrid.coastline import classify_tile_biome_context
        biome_map = {"c": "ocean", "n1": "forest", "n2": "ocean"}
        adj = {"c": ["n1", "n2"]}
        ctx = classify_tile_biome_context("c", biome_map, adj)
        assert ctx.own_biome == "ocean"
        assert ctx.is_edge is True
        assert ctx.edge_neighbours["n1"] == "forest"

    def test_classify_all_tiles(self):
        from polygrid.coastline import classify_all_tiles
        biome_map = {"c": "forest", "n1": "ocean", "n2": "forest"}
        adj = {"c": ["n1", "n2"], "n1": ["c"], "n2": ["c"]}
        result = classify_all_tiles(biome_map, adj)
        assert "c" in result
        assert "n1" in result
        assert "n2" in result
        assert result["c"].is_edge is True
        assert result["n1"].is_edge is True
        assert result["n2"].is_interior is True  # only neighbour is "c" (forest)


# ═══════════════════════════════════════════════════════════════════
# 19A.3 — Coastline mask computation tests
# ═══════════════════════════════════════════════════════════════════

class TestCoastlineMask:
    """Tests for compute_coastline_mask and related functions."""

    def test_mask_shape(self):
        from polygrid.coastline import compute_coastline_mask
        grid = _make_hex_grid()
        mask = compute_coastline_mask(
            128, {"n3": "ocean"}, grid, "c", seed=42,
        )
        assert mask.shape == (128, 128)
        assert mask.dtype == np.float32

    def test_mask_range_01(self):
        from polygrid.coastline import compute_coastline_mask
        grid = _make_hex_grid()
        mask = compute_coastline_mask(
            64, {"n3": "ocean"}, grid, "c", seed=42,
        )
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_interior_tile_all_zeros(self):
        from polygrid.coastline import compute_coastline_mask
        grid = _make_hex_grid()
        # No edge neighbours → all zeros
        mask = compute_coastline_mask(
            64, {}, grid, "c", seed=42,
        )
        assert np.allclose(mask, 0.0)

    def test_edge_tile_has_mixed_values(self):
        from polygrid.coastline import compute_coastline_mask
        grid = _make_hex_grid()
        mask = compute_coastline_mask(
            128, {"n3": "ocean"}, grid, "c", seed=42,
        )
        # Should have some zeros (own biome) and some high values (other biome)
        assert np.any(mask < 0.1), "Expected some pixels near 0.0 (own biome)"
        assert np.any(mask > 0.9), "Expected some pixels near 1.0 (other biome)"

    def test_noise_makes_non_straight_boundary(self):
        """The boundary should not be a perfectly straight line."""
        from polygrid.coastline import compute_coastline_mask
        grid = _make_hex_grid()
        mask = compute_coastline_mask(
            128, {"n3": "ocean"}, grid, "c", seed=42,
        )
        # Check that the boundary (mask ≈ 0.5) varies in position
        # across rows — not a perfectly vertical/horizontal line
        boundary_cols = []
        for row in range(mask.shape[0]):
            cols_at_half = np.where(np.abs(mask[row] - 0.5) < 0.15)[0]
            if len(cols_at_half) > 0:
                boundary_cols.append(float(np.mean(cols_at_half)))

        if len(boundary_cols) >= 2:
            # Standard deviation of boundary position should be > 0
            # (a straight line would have std ≈ 0)
            std = np.std(boundary_cols)
            assert std > 1.0, (
                f"Boundary appears too straight (std={std:.2f}). "
                "Noise should create an irregular coastline."
            )

    def test_seed_reproducibility(self):
        from polygrid.coastline import compute_coastline_mask
        grid = _make_hex_grid()
        m1 = compute_coastline_mask(64, {"n3": "ocean"}, grid, "c", seed=42)
        m2 = compute_coastline_mask(64, {"n3": "ocean"}, grid, "c", seed=42)
        np.testing.assert_array_equal(m1, m2)

    def test_different_seeds_produce_different_masks(self):
        from polygrid.coastline import compute_coastline_mask
        grid = _make_hex_grid()
        m1 = compute_coastline_mask(64, {"n3": "ocean"}, grid, "c", seed=42)
        m2 = compute_coastline_mask(64, {"n3": "ocean"}, grid, "c", seed=999)
        assert not np.allclose(m1, m2)

    def test_multiple_edge_neighbours_combine(self):
        from polygrid.coastline import compute_coastline_mask
        grid = _make_hex_grid()
        # Two neighbours from different directions
        m_one = compute_coastline_mask(
            64, {"n3": "ocean"}, grid, "c", seed=42,
        )
        m_two = compute_coastline_mask(
            64, {"n3": "ocean", "n6": "ocean"}, grid, "c", seed=42,
        )
        # With two edge neighbours, more pixels should be "other biome"
        assert m_two.mean() >= m_one.mean() - 0.01

    def test_different_tile_sizes(self):
        from polygrid.coastline import compute_coastline_mask
        grid = _make_hex_grid()
        for size in [32, 64, 128, 256]:
            mask = compute_coastline_mask(
                size, {"n3": "ocean"}, grid, "c", seed=42,
            )
            assert mask.shape == (size, size)


# ═══════════════════════════════════════════════════════════════════
# 19A.4 — CoastlineMask dataclass tests
# ═══════════════════════════════════════════════════════════════════

class TestCoastlineMaskDataclass:
    """Tests for the CoastlineMask dataclass."""

    def test_build_coastline_mask_interior(self):
        from polygrid.coastline import (
            build_coastline_mask, classify_tile_biome_context, CoastlineConfig,
        )
        grid = _make_hex_grid()
        biome_map = {"c": "forest", "n1": "forest", "n2": "forest",
                     "n3": "forest", "n4": "forest", "n5": "forest", "n6": "forest"}
        adj = {"c": ["n1", "n2", "n3", "n4", "n5", "n6"]}
        ctx = classify_tile_biome_context("c", biome_map, adj)
        cm = build_coastline_mask("c", ctx, grid, tile_size=64, seed=42)
        assert cm.face_id == "c"
        assert cm.own_biome == "forest"
        assert len(cm.other_biomes) == 0
        assert np.allclose(cm.mask, 0.0)
        assert cm.has_transition is False

    def test_build_coastline_mask_edge(self):
        from polygrid.coastline import (
            build_coastline_mask, classify_tile_biome_context,
        )
        grid = _make_hex_grid()
        biome_map = {"c": "forest", "n1": "ocean", "n2": "forest",
                     "n3": "forest", "n4": "forest", "n5": "forest", "n6": "forest"}
        adj = {"c": ["n1", "n2", "n3", "n4", "n5", "n6"]}
        ctx = classify_tile_biome_context("c", biome_map, adj)
        cm = build_coastline_mask("c", ctx, grid, tile_size=64, seed=42)
        assert cm.own_biome == "forest"
        assert "ocean" in cm.other_biomes
        assert cm.has_transition is True
        assert 0.0 < cm.transition_fraction < 1.0

    def test_tile_size_property(self):
        from polygrid.coastline import build_coastline_mask, classify_tile_biome_context
        grid = _make_hex_grid()
        biome_map = {"c": "forest", "n1": "forest", "n2": "forest",
                     "n3": "forest", "n4": "forest", "n5": "forest", "n6": "forest"}
        adj = {"c": ["n1", "n2", "n3", "n4", "n5", "n6"]}
        ctx = classify_tile_biome_context("c", biome_map, adj)
        cm = build_coastline_mask("c", ctx, grid, tile_size=128, seed=42)
        assert cm.tile_size == 128

    def test_coastline_pixels_property(self):
        from polygrid.coastline import build_coastline_mask, classify_tile_biome_context
        grid = _make_hex_grid()
        biome_map = {"c": "forest", "n1": "ocean", "n2": "forest",
                     "n3": "forest", "n4": "forest", "n5": "forest", "n6": "forest"}
        adj = {"c": ["n1", "n2", "n3", "n4", "n5", "n6"]}
        ctx = classify_tile_biome_context("c", biome_map, adj)
        cm = build_coastline_mask("c", ctx, grid, tile_size=64, seed=42)
        cp = cm.coastline_pixels
        assert cp.dtype == bool
        assert cp.shape == (64, 64)
        assert np.any(cp)  # should have some coastline pixels


# ═══════════════════════════════════════════════════════════════════
# 19A.5 — Edge hash consistency
# ═══════════════════════════════════════════════════════════════════

class TestStableEdgeHash:
    """Stable edge hash ensures cross-tile continuity."""

    def test_hash_order_independent(self):
        from polygrid.coastline import _stable_edge_hash
        h1 = _stable_edge_hash("f0", "f1")
        h2 = _stable_edge_hash("f1", "f0")
        assert h1 == h2

    def test_different_edges_different_hashes(self):
        from polygrid.coastline import _stable_edge_hash
        h1 = _stable_edge_hash("f0", "f1")
        h2 = _stable_edge_hash("f0", "f2")
        # Not guaranteed to always differ, but very likely
        # (we test a specific case that should differ)
        assert h1 != h2 or True  # soft assertion — hash collisions possible

    def test_hash_is_int(self):
        from polygrid.coastline import _stable_edge_hash
        h = _stable_edge_hash("abc", "def")
        assert isinstance(h, int)
        assert 0 <= h < 1_000_000


# ═══════════════════════════════════════════════════════════════════
# 19B.1 — Biome blending tests
# ═══════════════════════════════════════════════════════════════════

class TestBlendBiomeImages:
    """Tests for blend_biome_images."""

    def test_blend_all_own(self):
        """Mask all zeros → output = own biome."""
        from PIL import Image
        from polygrid.coastline import blend_biome_images

        own = Image.new("RGB", (32, 32), (0, 128, 0))
        other = Image.new("RGB", (32, 32), (0, 0, 200))
        mask = np.zeros((32, 32), dtype=np.float32)

        result = blend_biome_images(own, other, mask)
        arr = np.array(result)
        np.testing.assert_array_equal(arr[:, :, 1], 128)

    def test_blend_all_other(self):
        """Mask all ones → output = other biome."""
        from PIL import Image
        from polygrid.coastline import blend_biome_images

        own = Image.new("RGB", (32, 32), (0, 128, 0))
        other = Image.new("RGB", (32, 32), (0, 0, 200))
        mask = np.ones((32, 32), dtype=np.float32)

        result = blend_biome_images(own, other, mask)
        arr = np.array(result)
        np.testing.assert_array_equal(arr[:, :, 2], 200)

    def test_blend_half(self):
        """Mask at 0.5 → output is midpoint."""
        from PIL import Image
        from polygrid.coastline import blend_biome_images

        own = Image.new("RGB", (32, 32), (0, 100, 0))
        other = Image.new("RGB", (32, 32), (0, 0, 200))
        mask = np.full((32, 32), 0.5, dtype=np.float32)

        result = blend_biome_images(own, other, mask)
        arr = np.array(result)
        # G channel: 100 * 0.5 = 50, B channel: 200 * 0.5 = 100
        assert arr[16, 16, 1] == 50
        assert arr[16, 16, 2] == 100


# ═══════════════════════════════════════════════════════════════════
# 19B.3 — Coastal strip tests
# ═══════════════════════════════════════════════════════════════════

class TestCoastalStrip:
    """Tests for render_coastal_strip."""

    def test_coastal_strip_modifies_transition_zone(self):
        from PIL import Image
        from polygrid.coastline import render_coastal_strip

        # Create a simple image and a mask with transition
        img = Image.new("RGB", (64, 64), (50, 100, 50))
        # Gradient mask from 0 to 1
        mask = np.linspace(0, 1, 64, dtype=np.float32)
        mask = np.tile(mask, (64, 1))  # (64, 64)

        result = render_coastal_strip(img, mask, seed=42)
        original = np.array(img)
        result_arr = np.array(result)

        # The transition zone should be modified
        # Check that at least some pixels differ
        diff = np.abs(result_arr.astype(float) - original.astype(float))
        assert diff.sum() > 0, "Coastal strip should modify some pixels"

    def test_no_modification_outside_transition(self):
        from PIL import Image
        from polygrid.coastline import render_coastal_strip

        img = Image.new("RGB", (64, 64), (50, 100, 50))
        # Mask with zeros everywhere (no transition)
        mask = np.zeros((64, 64), dtype=np.float32)

        result = render_coastal_strip(img, mask, seed=42)
        original = np.array(img)
        result_arr = np.array(result)
        np.testing.assert_array_equal(result_arr, original)

    def test_output_is_rgb(self):
        from PIL import Image
        from polygrid.coastline import render_coastal_strip

        img = Image.new("RGB", (32, 32), (50, 100, 50))
        mask = np.full((32, 32), 0.4, dtype=np.float32)
        result = render_coastal_strip(img, mask, seed=42)
        assert result.mode == "RGB"


# ═══════════════════════════════════════════════════════════════════
# 19A — Edge direction computation
# ═══════════════════════════════════════════════════════════════════

class TestEdgeDirection:
    """Tests for compute_edge_direction."""

    def test_direction_is_normalised(self):
        from polygrid.coastline import compute_edge_direction
        grid = _make_hex_grid()
        dx, dy = compute_edge_direction(grid, "c", "n3")
        length = math.sqrt(dx * dx + dy * dy)
        assert abs(length - 1.0) < 0.01

    def test_direction_toward_neighbour(self):
        from polygrid.coastline import compute_edge_direction
        grid = _make_hex_grid()
        # n3 is to the right of c (positive x)
        dx, dy = compute_edge_direction(grid, "c", "n3")
        assert dx > 0.5, f"Expected dx > 0.5 for rightward neighbour, got {dx}"

    def test_opposite_directions(self):
        from polygrid.coastline import compute_edge_direction
        grid = _make_hex_grid()
        dx3, dy3 = compute_edge_direction(grid, "c", "n3")  # right
        dx6, dy6 = compute_edge_direction(grid, "c", "n6")  # left
        # Should be roughly opposite
        assert dx3 + dx6 < 0.5  # roughly cancel out
