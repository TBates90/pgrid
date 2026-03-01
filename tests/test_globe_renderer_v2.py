"""Tests for globe_renderer_v2 — Phase 12 rendering improvements.

Tests the three main subsystems:
- 12A: Texture flood-fill (black border removal)
- 12B: Sphere subdivision
- 12C: Batched globe mesh builder
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from polygrid.globe_renderer_v2 import (
    flood_fill_tile_texture,
    flood_fill_atlas,
    subdivide_tile_mesh,
    build_batched_globe_mesh,
    _normalize_vec3,
    _project_to_sphere,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def hex_tile_data():
    """A regular hexagonal tile for testing."""
    center = (0.0, 0.0, 1.0)
    n_sides = 6
    angle_step = 2 * math.pi / n_sides
    r = 0.1
    vertices = [
        (r * math.cos(i * angle_step),
         r * math.sin(i * angle_step),
         1.0)
        for i in range(n_sides)
    ]
    center_uv = (0.5, 0.5)
    vertex_uvs = [
        (0.5 + 0.4 * math.cos(i * angle_step),
         0.5 + 0.4 * math.sin(i * angle_step))
        for i in range(n_sides)
    ]
    return center, vertices, center_uv, vertex_uvs


@pytest.fixture
def pent_tile_data():
    """A regular pentagonal tile for testing."""
    center = (1.0, 0.0, 0.0)
    n_sides = 5
    angle_step = 2 * math.pi / n_sides
    r = 0.12
    vertices = [
        (1.0,
         r * math.cos(i * angle_step),
         r * math.sin(i * angle_step))
        for i in range(n_sides)
    ]
    center_uv = (0.5, 0.5)
    vertex_uvs = [
        (0.5 + 0.4 * math.cos(i * angle_step),
         0.5 + 0.4 * math.sin(i * angle_step))
        for i in range(n_sides)
    ]
    return center, vertices, center_uv, vertex_uvs


@pytest.fixture
def black_bordered_image(tmp_path):
    """Create a 64x64 test image with black border and coloured center."""
    from PIL import Image
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    # Paint a 40x40 coloured region in the center
    pixels = img.load()
    for x in range(12, 52):
        for y in range(12, 52):
            pixels[x, y] = (100, 150, 80)
    path = tmp_path / "test_tile.png"
    img.save(str(path))
    return path


# ═══════════════════════════════════════════════════════════════════
# 12A — Flood fill tests
# ═══════════════════════════════════════════════════════════════════

class TestFloodFill:
    """Tests for black border removal via flood fill."""

    def test_flood_fill_reduces_black_pixels(self, black_bordered_image):
        """After flood-fill, fewer pixels should be black."""
        from PIL import Image

        original = np.array(Image.open(str(black_bordered_image)))
        black_before = np.sum(original.sum(axis=2) <= 10)

        output = black_bordered_image.parent / "filled.png"
        flood_fill_tile_texture(black_bordered_image, output, iterations=8)

        result = np.array(Image.open(str(output)))
        black_after = np.sum(result.sum(axis=2) <= 10)

        assert black_after < black_before, (
            f"Expected fewer black pixels after flood fill: "
            f"before={black_before}, after={black_after}"
        )

    def test_flood_fill_preserves_coloured_center(self, black_bordered_image):
        """Flood fill should not alter the existing coloured pixels."""
        from PIL import Image

        original = np.array(Image.open(str(black_bordered_image)))
        output = black_bordered_image.parent / "filled.png"
        flood_fill_tile_texture(black_bordered_image, output, iterations=8)
        result = np.array(Image.open(str(output)))

        # Check center region is preserved
        center_orig = original[20:44, 20:44]
        center_result = result[20:44, 20:44]
        diff = np.abs(center_orig.astype(float) - center_result.astype(float)).max()
        assert diff < 2.0, f"Centre pixels changed by {diff}"

    def test_flood_fill_overwrites_in_place(self, black_bordered_image):
        """When no output_path given, overwrites the input."""
        from PIL import Image

        original = np.array(Image.open(str(black_bordered_image)))
        flood_fill_tile_texture(black_bordered_image, iterations=4)
        result = np.array(Image.open(str(black_bordered_image)))

        # Should have changed
        assert not np.array_equal(original, result)

    def test_flood_fill_edge_pixels_get_colour(self, black_bordered_image):
        """Pixels immediately adjacent to the coloured region should be filled."""
        from PIL import Image

        output = black_bordered_image.parent / "filled.png"
        flood_fill_tile_texture(black_bordered_image, output, iterations=4)
        result = np.array(Image.open(str(output)))

        # Pixel at (11, 30) was black, adjacent to coloured region at (12, 30)
        pixel = result[30, 11]  # (y, x) indexing for numpy
        brightness = int(pixel[0]) + int(pixel[1]) + int(pixel[2])
        assert brightness > 30, (
            f"Adjacent pixel should be filled: {pixel}"
        )

    def test_flood_fill_atlas_alias(self, black_bordered_image):
        """flood_fill_atlas should work the same as flood_fill_tile_texture."""
        from PIL import Image

        output = black_bordered_image.parent / "atlas_filled.png"
        flood_fill_atlas(black_bordered_image, output, iterations=4)
        assert output.exists()
        result = np.array(Image.open(str(output)))
        black_count = np.sum(result.sum(axis=2) <= 10)
        total = result.shape[0] * result.shape[1]
        # Most black should be gone (only corners far from centre remain)
        assert black_count < total * 0.5

    def test_flood_fill_zero_iterations(self, black_bordered_image):
        """With 0 iterations, output should match input."""
        from PIL import Image

        output = black_bordered_image.parent / "zero.png"
        flood_fill_tile_texture(black_bordered_image, output, iterations=0)
        original = np.array(Image.open(str(black_bordered_image)))
        result = np.array(Image.open(str(output)))
        np.testing.assert_array_equal(original, result)

    def test_flood_fill_all_coloured(self, tmp_path):
        """Image with no black pixels should be unchanged."""
        from PIL import Image

        img = Image.new("RGB", (32, 32), (80, 120, 60))
        path = tmp_path / "allcolour.png"
        img.save(str(path))

        output = tmp_path / "out.png"
        flood_fill_tile_texture(path, output, iterations=4)
        original = np.array(Image.open(str(path)))
        result = np.array(Image.open(str(output)))
        np.testing.assert_array_equal(original, result)


# ═══════════════════════════════════════════════════════════════════
# 12B — Subdivision tests
# ═══════════════════════════════════════════════════════════════════

class TestSubdivision:
    """Tests for tile triangle fan subdivision and sphere projection."""

    def test_subdivide_returns_correct_shape(self, hex_tile_data):
        """Output arrays should have correct shapes."""
        center, verts, cuv, vuvs = hex_tile_data
        vdata, idata = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (1.0, 1.0, 1.0),
            radius=1.0, subdivisions=2,
        )
        assert vdata.ndim == 2
        assert vdata.shape[1] == 8  # pos(3) + col(3) + uv(2)
        assert idata.ndim == 2
        assert idata.shape[1] == 3  # triangles

    def test_subdivide_increases_geometry(self, hex_tile_data):
        """Higher subdivisions should produce more triangles."""
        center, verts, cuv, vuvs = hex_tile_data
        _, idata_low = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (1.0, 1.0, 1.0),
            radius=1.0, subdivisions=1,
        )
        _, idata_high = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (1.0, 1.0, 1.0),
            radius=1.0, subdivisions=3,
        )
        assert len(idata_high) > len(idata_low)

    def test_subdivide_s1_preserves_tri_count(self, hex_tile_data):
        """With subdivisions=1, we should get exactly n_sides triangles."""
        center, verts, cuv, vuvs = hex_tile_data
        _, idata = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (1.0, 1.0, 1.0),
            radius=1.0, subdivisions=1,
        )
        assert len(idata) == 6  # one triangle per hex side

    def test_subdivide_s2_triangle_count(self, hex_tile_data):
        """With s=2, each original triangle → 4 sub-triangles."""
        center, verts, cuv, vuvs = hex_tile_data
        _, idata = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (1.0, 1.0, 1.0),
            radius=1.0, subdivisions=2,
        )
        # 6 sides × 4 sub-tris = 24
        assert len(idata) == 24

    def test_subdivide_pent_s2(self, pent_tile_data):
        """Pentagon with s=2 → 5×4 = 20 triangles."""
        center, verts, cuv, vuvs = pent_tile_data
        _, idata = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (1.0, 1.0, 1.0),
            radius=1.0, subdivisions=2,
        )
        assert len(idata) == 20

    def test_vertices_on_sphere(self, hex_tile_data):
        """All subdivided vertices should lie on the sphere surface."""
        center, verts, cuv, vuvs = hex_tile_data
        radius = 1.0
        vdata, _ = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (1.0, 1.0, 1.0),
            radius=radius, subdivisions=3,
        )
        positions = vdata[:, :3]
        norms = np.linalg.norm(positions, axis=1)
        np.testing.assert_allclose(norms, radius, atol=1e-6)

    def test_vertices_on_sphere_larger_radius(self, hex_tile_data):
        """Sphere projection should work with non-unit radius."""
        center, verts, cuv, vuvs = hex_tile_data
        # Scale center and vertices to radius 2
        center2 = tuple(c * 2 for c in center)
        verts2 = [tuple(v * 2 for v in vert) for vert in verts]
        radius = 2.0
        vdata, _ = subdivide_tile_mesh(
            center2, verts2, cuv, vuvs, (1.0, 1.0, 1.0),
            radius=radius, subdivisions=2,
        )
        norms = np.linalg.norm(vdata[:, :3], axis=1)
        np.testing.assert_allclose(norms, radius, atol=1e-6)

    def test_uvs_in_range(self, hex_tile_data):
        """UV coordinates should stay within [0, 1]."""
        center, verts, cuv, vuvs = hex_tile_data
        vdata, _ = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (1.0, 1.0, 1.0),
            radius=1.0, subdivisions=3,
        )
        uvs = vdata[:, 6:8]
        assert uvs.min() >= -0.01
        assert uvs.max() <= 1.01

    def test_colour_propagated(self, hex_tile_data):
        """Vertex colours should match the input colour."""
        center, verts, cuv, vuvs = hex_tile_data
        color = (0.3, 0.7, 0.2)
        vdata, _ = subdivide_tile_mesh(
            center, verts, cuv, vuvs, color,
            radius=1.0, subdivisions=2,
        )
        colours = vdata[:, 3:6]
        expected = np.tile(np.array(color, dtype=np.float32), (len(vdata), 1))
        np.testing.assert_allclose(colours, expected, atol=1e-6)

    def test_no_degenerate_triangles(self, hex_tile_data):
        """All triangles should have positive area (no degenerate tris)."""
        center, verts, cuv, vuvs = hex_tile_data
        vdata, idata = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (1.0, 1.0, 1.0),
            radius=1.0, subdivisions=3,
        )
        positions = vdata[:, :3]
        for tri in idata:
            p0 = positions[tri[0]]
            p1 = positions[tri[1]]
            p2 = positions[tri[2]]
            cross = np.cross(p1 - p0, p2 - p0)
            area = np.linalg.norm(cross) * 0.5
            assert area > 1e-10, f"Degenerate triangle: {tri}"

    def test_indices_within_bounds(self, hex_tile_data):
        """All indices should reference valid vertices."""
        center, verts, cuv, vuvs = hex_tile_data
        vdata, idata = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (1.0, 1.0, 1.0),
            radius=1.0, subdivisions=3,
        )
        assert idata.max() < len(vdata)
        assert idata.min() >= 0

    def test_vertex_deduplication(self, hex_tile_data):
        """Shared edges between triangles should produce deduplicated vertices."""
        center, verts, cuv, vuvs = hex_tile_data
        vdata, _ = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (1.0, 1.0, 1.0),
            radius=1.0, subdivisions=2,
        )
        # Without dedup: 6 tris × (2+1)(2+2)/2 = 6 × 6 = 36 vertices
        # With dedup: significantly fewer
        positions = vdata[:, :3]
        unique_positions = np.unique(np.round(positions, 7), axis=0)
        assert len(unique_positions) == len(positions), (
            "Vertices should be already deduplicated"
        )
        # Should be fewer than non-deduplicated count
        assert len(positions) < 36

    def test_subdivide_s3_triangle_count_hex(self, hex_tile_data):
        """s=3: each original tri → 9 sub-tris, hex = 6×9 = 54."""
        center, verts, cuv, vuvs = hex_tile_data
        _, idata = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (1.0, 1.0, 1.0),
            radius=1.0, subdivisions=3,
        )
        assert len(idata) == 54


# ═══════════════════════════════════════════════════════════════════
# 12B — Helper function tests
# ═══════════════════════════════════════════════════════════════════

class TestHelpers:
    """Tests for utility functions."""

    def test_normalize_vec3_unit(self):
        v = np.array([3.0, 4.0, 0.0])
        result = _normalize_vec3(v)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-10)

    def test_normalize_vec3_zero(self):
        v = np.array([0.0, 0.0, 0.0])
        result = _normalize_vec3(v)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    def test_project_to_sphere(self):
        v = np.array([2.0, 0.0, 0.0])
        result = _project_to_sphere(v, 3.0)
        np.testing.assert_allclose(result, [3.0, 0.0, 0.0], atol=1e-10)

    def test_project_to_sphere_diagonal(self):
        v = np.array([1.0, 1.0, 1.0])
        result = _project_to_sphere(v, 1.0)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════
# 12C — Batched mesh tests (require models library)
# ═══════════════════════════════════════════════════════════════════

try:
    from models.objects.goldberg import generate_goldberg_tiles
    _HAS_MODELS = True
except ImportError:
    _HAS_MODELS = False


@pytest.mark.skipif(not _HAS_MODELS, reason="models library required")
class TestBatchedMesh:
    """Tests for the batched globe mesh builder."""

    @pytest.fixture
    def uv_layout_f3(self):
        """Minimal UV layout for freq=3 tiles."""
        from models.objects.goldberg import generate_goldberg_tiles
        tiles = generate_goldberg_tiles(frequency=3, radius=1.0)
        n = len(tiles)
        # Grid layout: ceil(sqrt(n)) columns
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        layout = {}
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            u_min = c / cols
            v_min = r / rows
            u_max = (c + 1) / cols
            v_max = (r + 1) / rows
            layout[f"t{tile.index}"] = (u_min, v_min, u_max, v_max)
        return layout

    def test_batched_mesh_output_shapes(self, uv_layout_f3):
        """Mesh should have correct array shapes."""
        vdata, idata = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1,
        )
        assert vdata.ndim == 2
        assert vdata.shape[1] == 8
        assert idata.ndim == 2
        assert idata.shape[1] == 3

    def test_batched_mesh_nonzero(self, uv_layout_f3):
        """Should produce a non-empty mesh."""
        vdata, idata = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1,
        )
        assert len(vdata) > 0
        assert len(idata) > 0

    def test_batched_mesh_all_on_sphere(self, uv_layout_f3):
        """All vertices should lie on the sphere surface."""
        radius = 1.0
        vdata, _ = build_batched_globe_mesh(
            3, uv_layout_f3, radius=radius, subdivisions=2,
        )
        norms = np.linalg.norm(vdata[:, :3], axis=1)
        np.testing.assert_allclose(norms, radius, atol=1e-5)

    def test_batched_mesh_indices_valid(self, uv_layout_f3):
        """All indices should be within vertex array bounds."""
        vdata, idata = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=2,
        )
        assert idata.max() < len(vdata)
        assert idata.min() >= 0

    def test_batched_mesh_more_tris_with_higher_subdiv(self, uv_layout_f3):
        """Higher subdivisions should produce more triangles."""
        _, idata_1 = build_batched_globe_mesh(3, uv_layout_f3, subdivisions=1)
        _, idata_3 = build_batched_globe_mesh(3, uv_layout_f3, subdivisions=3)
        assert len(idata_3) > len(idata_1)

    def test_batched_mesh_uses_colour_map(self, uv_layout_f3):
        """Tile colour map should be reflected in vertex colours."""
        colours = {0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0)}
        vdata, _ = build_batched_globe_mesh(
            3, uv_layout_f3,
            tile_colour_map=colours,
            subdivisions=1,
        )
        # At least some vertices should have red or green
        has_red = np.any(vdata[:, 3] > 0.9)
        has_green = np.any(vdata[:, 4] > 0.9)
        assert has_red or has_green

    def test_batched_mesh_empty_layout(self):
        """Empty UV layout should produce empty mesh."""
        vdata, idata = build_batched_globe_mesh(3, {}, subdivisions=1)
        assert len(vdata) == 0
        assert len(idata) == 0

    def test_batched_mesh_tile_count_f3(self, uv_layout_f3):
        """f=3 Goldberg has 92 tiles, mesh should cover all."""
        # With s=1, each hex → 6 tris, each pent → 5 tris
        # Total: 12 * 5 + 80 * 6 = 60 + 480 = 540 triangles
        _, idata = build_batched_globe_mesh(3, uv_layout_f3, subdivisions=1)
        assert len(idata) == 540

    def test_batched_mesh_custom_radius(self, uv_layout_f3):
        """Mesh at radius=2 should have vertices at r≈2."""
        radius = 2.0
        vdata, _ = build_batched_globe_mesh(
            3, uv_layout_f3, radius=radius, subdivisions=1,
        )
        norms = np.linalg.norm(vdata[:, :3], axis=1)
        np.testing.assert_allclose(norms, radius, atol=1e-5)
