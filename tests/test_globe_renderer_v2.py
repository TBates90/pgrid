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
    compute_uv_polygon_inset,
    clamp_uv_to_polygon,
    _point_in_convex_polygon,
    _nearest_point_on_segment,
    _nearest_point_on_polygon_edge,
    blend_biome_configs,
    compute_neighbour_average_colours,
    harmonise_tile_colours,
    encode_normal_to_rgb,
    decode_rgb_to_normal,
    build_normal_map_atlas,
    get_pbr_shader_sources,
    get_v2_shader_sources,
    classify_water_tiles,
    compute_water_depth,
    DEFAULT_WATER_LEVEL,
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


# ═══════════════════════════════════════════════════════════════════
# 13C — UV polygon inset & clamping tests
# ═══════════════════════════════════════════════════════════════════

class TestPointInConvexPolygon:
    """Tests for the convex-polygon containment check."""

    @pytest.fixture
    def unit_square(self):
        return [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    @pytest.fixture
    def hex_uv_poly(self):
        """Regular hexagon centred at (0.5, 0.5), radius ~0.4."""
        n = 6
        return [
            (0.5 + 0.4 * math.cos(2 * math.pi * i / n),
             0.5 + 0.4 * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]

    def test_center_is_inside_square(self, unit_square):
        assert _point_in_convex_polygon(0.5, 0.5, unit_square) is True

    def test_corner_is_inside_square(self, unit_square):
        # On the boundary → should be True (edge-inclusive)
        assert _point_in_convex_polygon(0.0, 0.0, unit_square) is True

    def test_outside_square(self, unit_square):
        assert _point_in_convex_polygon(-0.1, 0.5, unit_square) is False
        assert _point_in_convex_polygon(1.1, 0.5, unit_square) is False

    def test_center_is_inside_hex(self, hex_uv_poly):
        assert _point_in_convex_polygon(0.5, 0.5, hex_uv_poly) is True

    def test_outside_hex_corner(self, hex_uv_poly):
        # Outside the hex but inside its bounding box
        assert _point_in_convex_polygon(0.1, 0.1, hex_uv_poly) is False


class TestNearestPointOnSegment:
    """Tests for point-to-segment projection."""

    def test_projection_mid_segment(self):
        # Project (0.5, 1.0) onto horizontal segment (0,0)→(1,0)
        px, py = _nearest_point_on_segment(0.5, 1.0, 0.0, 0.0, 1.0, 0.0)
        assert abs(px - 0.5) < 1e-9
        assert abs(py - 0.0) < 1e-9

    def test_clamp_to_start(self):
        # Project (-1, 0) onto segment (0,0)→(1,0) → clamped to start
        px, py = _nearest_point_on_segment(-1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        assert abs(px) < 1e-9
        assert abs(py) < 1e-9

    def test_clamp_to_end(self):
        # Project (2, 0) onto segment (0,0)→(1,0) → clamped to end
        px, py = _nearest_point_on_segment(2.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        assert abs(px - 1.0) < 1e-9
        assert abs(py) < 1e-9


class TestNearestPointOnPolygonEdge:
    """Tests for closest point on any polygon edge."""

    def test_projects_to_nearest_edge(self):
        square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        # Point just outside the top edge
        bx, by = _nearest_point_on_polygon_edge(0.5, 1.1, square)
        assert abs(bx - 0.5) < 1e-9
        assert abs(by - 1.0) < 1e-9


class TestUVPolygonInset:
    """Tests for compute_uv_polygon_inset."""

    @pytest.fixture
    def hex_uv_poly(self):
        """Regular hexagon centred at (0.5, 0.5) with radius 0.4 in UV."""
        n = 6
        return [
            (0.5 + 0.4 * math.cos(2 * math.pi * i / n),
             0.5 + 0.4 * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]

    def test_inset_produces_smaller_polygon(self, hex_uv_poly):
        """Inset polygon should be strictly smaller than original."""
        inset = compute_uv_polygon_inset(
            hex_uv_poly, inset_px=2.0, atlas_size=256,
        )
        # Same number of vertices
        assert len(inset) == len(hex_uv_poly)
        # All inset vertices closer to centroid than originals
        cx = sum(u for u, v in hex_uv_poly) / len(hex_uv_poly)
        cy = sum(v for u, v in hex_uv_poly) / len(hex_uv_poly)
        for (ou, ov), (iu, iv) in zip(hex_uv_poly, inset):
            orig_dist = math.hypot(ou - cx, ov - cy)
            inset_dist = math.hypot(iu - cx, iv - cy)
            assert inset_dist < orig_dist

    def test_inset_preserves_centroid(self, hex_uv_poly):
        """Centroid should remain the same."""
        inset = compute_uv_polygon_inset(
            hex_uv_poly, inset_px=2.0, atlas_size=256,
        )
        cx_orig = sum(u for u, v in hex_uv_poly) / len(hex_uv_poly)
        cy_orig = sum(v for u, v in hex_uv_poly) / len(hex_uv_poly)
        cx_inset = sum(u for u, v in inset) / len(inset)
        cy_inset = sum(v for u, v in inset) / len(inset)
        assert abs(cx_orig - cx_inset) < 1e-9
        assert abs(cy_orig - cy_inset) < 1e-9

    def test_zero_inset_returns_original(self, hex_uv_poly):
        """Inset of 0 pixels should return the original polygon."""
        inset = compute_uv_polygon_inset(
            hex_uv_poly, inset_px=0.0, atlas_size=256,
        )
        for (ou, ov), (iu, iv) in zip(hex_uv_poly, inset):
            assert abs(ou - iu) < 1e-12
            assert abs(ov - iv) < 1e-12

    def test_inset_amount_scales_with_atlas_size(self, hex_uv_poly):
        """Larger atlas → smaller UV-space inset for the same pixel count."""
        inset_small = compute_uv_polygon_inset(
            hex_uv_poly, inset_px=2.0, atlas_size=128,
        )
        inset_large = compute_uv_polygon_inset(
            hex_uv_poly, inset_px=2.0, atlas_size=512,
        )
        # With a larger atlas, inset vertices should be closer to the
        # original (less UV movement).
        cx = sum(u for u, v in hex_uv_poly) / len(hex_uv_poly)
        cy = sum(v for u, v in hex_uv_poly) / len(hex_uv_poly)
        for (su, sv), (lu, lv), (ou, ov) in zip(
            inset_small, inset_large, hex_uv_poly
        ):
            shrink_small = math.hypot(ou - su, ov - sv)
            shrink_large = math.hypot(ou - lu, ov - lv)
            assert shrink_large < shrink_small


class TestClampUVToPolygon:
    """Tests for clamping a UV point to a convex polygon."""

    @pytest.fixture
    def hex_uv_poly(self):
        n = 6
        return [
            (0.5 + 0.4 * math.cos(2 * math.pi * i / n),
             0.5 + 0.4 * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]

    def test_interior_point_unchanged(self, hex_uv_poly):
        """A point already inside should be returned unchanged."""
        cu, cv = clamp_uv_to_polygon(0.5, 0.5, hex_uv_poly)
        assert abs(cu - 0.5) < 1e-12
        assert abs(cv - 0.5) < 1e-12

    def test_exterior_point_clamped_inside(self, hex_uv_poly):
        """A point outside should be moved to the polygon boundary."""
        # Well outside the hex
        cu, cv = clamp_uv_to_polygon(0.0, 0.0, hex_uv_poly)
        # Should now be on or very close to the polygon edge
        assert _point_in_convex_polygon(cu, cv, hex_uv_poly)

    def test_clamped_point_near_polygon_edge(self, hex_uv_poly):
        """Clamped point should be on the nearest edge, not just the centroid."""
        # Point just outside the right edge of the hex
        cu, cv = clamp_uv_to_polygon(0.95, 0.5, hex_uv_poly)
        # The right-most hex vertex is at (0.9, 0.5)
        assert cu <= 0.9 + 1e-6
        assert abs(cv - 0.5) < 0.05  # Roughly on the horizontal axis


class TestSubdivideMeshWithClamping:
    """Tests that UV clamping in subdivide_tile_mesh works correctly."""

    @pytest.fixture
    def hex_tile_data(self):
        """Hex tile centred at (0,0,1), UVs at (0.5,0.5)."""
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

    def test_clamped_uvs_inside_polygon(self, hex_tile_data):
        """All UVs from clamped subdivision must lie inside the clamp polygon."""
        center, vertices, center_uv, vertex_uvs = hex_tile_data
        # Build a slightly inset polygon for clamping
        inset = compute_uv_polygon_inset(
            vertex_uvs, inset_px=2.0, atlas_size=256,
        )
        vdata, _ = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 1.0, 1.0), radius=1.0, subdivisions=3,
            uv_clamp_polygon=inset,
        )
        for row_idx in range(len(vdata)):
            u, v = float(vdata[row_idx, 6]), float(vdata[row_idx, 7])
            assert _point_in_convex_polygon(u, v, inset), (
                f"UV ({u:.6f}, {v:.6f}) at vertex {row_idx} is "
                f"outside the inset polygon"
            )

    def test_no_clamp_polygon_backward_compat(self, hex_tile_data):
        """Without clamp polygon, subdivide_tile_mesh should work unchanged."""
        center, vertices, center_uv, vertex_uvs = hex_tile_data
        vdata_new, idata_new = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 1.0, 1.0), radius=1.0, subdivisions=2,
            uv_clamp_polygon=None,
        )
        vdata_old, idata_old = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 1.0, 1.0), radius=1.0, subdivisions=2,
        )
        np.testing.assert_array_equal(vdata_new, vdata_old)
        np.testing.assert_array_equal(idata_new, idata_old)

    def test_clamping_does_not_change_vertex_count(self, hex_tile_data):
        """Clamping UVs should not change the mesh topology."""
        center, vertices, center_uv, vertex_uvs = hex_tile_data
        inset = compute_uv_polygon_inset(
            vertex_uvs, inset_px=2.0, atlas_size=256,
        )
        vdata_clamped, idata_clamped = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 1.0, 1.0), radius=1.0, subdivisions=3,
            uv_clamp_polygon=inset,
        )
        vdata_plain, idata_plain = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 1.0, 1.0), radius=1.0, subdivisions=3,
        )
        assert vdata_clamped.shape[0] == vdata_plain.shape[0]
        assert idata_clamped.shape[0] == idata_plain.shape[0]

    def test_clamping_preserves_positions(self, hex_tile_data):
        """Clamping should only change UVs, not 3D positions or colours."""
        center, vertices, center_uv, vertex_uvs = hex_tile_data
        inset = compute_uv_polygon_inset(
            vertex_uvs, inset_px=2.0, atlas_size=256,
        )
        vdata_clamped, _ = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 1.0, 1.0), radius=1.0, subdivisions=3,
            uv_clamp_polygon=inset,
        )
        vdata_plain, _ = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 1.0, 1.0), radius=1.0, subdivisions=3,
        )
        # 3D positions (cols 0:3) and colours (cols 3:6) should match
        np.testing.assert_allclose(
            vdata_clamped[:, :6], vdata_plain[:, :6], atol=1e-6,
        )


@pytest.mark.skipif(not _HAS_MODELS, reason="models library required")
class TestBatchedMeshWithClamping:
    """Tests for UV clamping in the batched globe mesh builder."""

    @pytest.fixture
    def uv_layout_f3(self):
        """Minimal UV layout for freq=3 tiles."""
        from models.objects.goldberg import generate_goldberg_tiles
        tiles = generate_goldberg_tiles(frequency=3, radius=1.0)
        n = len(tiles)
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

    def test_batched_mesh_with_inset_produces_output(self, uv_layout_f3):
        """Batched mesh with UV inset should produce valid output."""
        vdata, idata = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=2,
            uv_inset_px=1.5, atlas_size=1024,
        )
        assert len(vdata) > 0
        assert len(idata) > 0
        assert vdata.shape[1] == 8

    def test_batched_mesh_inset_same_topology(self, uv_layout_f3):
        """UV inset should not change the number of vertices or triangles."""
        vdata_plain, idata_plain = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1,
        )
        vdata_inset, idata_inset = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1,
            uv_inset_px=1.5, atlas_size=1024,
        )
        assert vdata_inset.shape[0] == vdata_plain.shape[0]
        assert idata_inset.shape[0] == idata_plain.shape[0]

    def test_batched_mesh_inset_positions_unchanged(self, uv_layout_f3):
        """UV inset should not change 3D positions."""
        vdata_plain, _ = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1,
        )
        vdata_inset, _ = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1,
            uv_inset_px=1.5, atlas_size=1024,
        )
        np.testing.assert_allclose(
            vdata_inset[:, :3], vdata_plain[:, :3], atol=1e-6,
        )

    def test_batched_mesh_requires_atlas_size_with_inset(self, uv_layout_f3):
        """Passing uv_inset_px > 0 without atlas_size should raise."""
        with pytest.raises(ValueError, match="atlas_size"):
            build_batched_globe_mesh(
                3, uv_layout_f3, subdivisions=1,
                uv_inset_px=1.5,
            )

    def test_batched_mesh_zero_inset_matches_no_inset(self, uv_layout_f3):
        """uv_inset_px=0 should produce identical results to no inset."""
        vdata_none, idata_none = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1,
        )
        vdata_zero, idata_zero = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1,
            uv_inset_px=0.0, atlas_size=1024,
        )
        np.testing.assert_array_equal(vdata_none, vdata_zero)
        np.testing.assert_array_equal(idata_none, idata_zero)


# ═══════════════════════════════════════════════════════════════════
# 13D — Cross-tile colour harmonisation tests
# ═══════════════════════════════════════════════════════════════════

class TestBlendBiomeConfigs:
    """Tests for BiomeConfig interpolation."""

    def test_blend_zero_returns_first(self):
        from polygrid.detail_render import BiomeConfig
        a = BiomeConfig(vegetation_density=0.8, rock_exposure=0.2)
        b = BiomeConfig(vegetation_density=0.1, rock_exposure=0.9)
        result = blend_biome_configs(a, b, 0.0)
        assert abs(result.vegetation_density - 0.8) < 1e-9
        assert abs(result.rock_exposure - 0.2) < 1e-9

    def test_blend_one_returns_second(self):
        from polygrid.detail_render import BiomeConfig
        a = BiomeConfig(vegetation_density=0.8, rock_exposure=0.2)
        b = BiomeConfig(vegetation_density=0.1, rock_exposure=0.9)
        result = blend_biome_configs(a, b, 1.0)
        assert abs(result.vegetation_density - 0.1) < 1e-9
        assert abs(result.rock_exposure - 0.9) < 1e-9

    def test_blend_half_interpolates(self):
        from polygrid.detail_render import BiomeConfig
        a = BiomeConfig(snow_line=0.4, moisture=0.2)
        b = BiomeConfig(snow_line=0.8, moisture=0.6)
        result = blend_biome_configs(a, b, 0.5)
        assert abs(result.snow_line - 0.6) < 1e-9
        assert abs(result.moisture - 0.4) < 1e-9

    def test_blend_clamps_weight(self):
        from polygrid.detail_render import BiomeConfig
        a = BiomeConfig(water_level=0.0)
        b = BiomeConfig(water_level=1.0)
        # t > 1 → clamped to 1
        result = blend_biome_configs(a, b, 5.0)
        assert abs(result.water_level - 1.0) < 1e-9
        # t < 0 → clamped to 0
        result2 = blend_biome_configs(a, b, -2.0)
        assert abs(result2.water_level - 0.0) < 1e-9

    def test_blend_preserves_base_ramp(self):
        from polygrid.detail_render import BiomeConfig
        a = BiomeConfig(base_ramp="detail_satellite")
        b = BiomeConfig(base_ramp="other_ramp")
        result = blend_biome_configs(a, b, 0.5)
        assert result.base_ramp == "detail_satellite"


class _FakeTile:
    """Minimal mock for GoldbergTile in colour tests."""

    def __init__(self, index: int, neighbor_indices: Tuple[int, ...] = ()):
        self.index = index
        self.neighbor_indices = neighbor_indices


class TestComputeNeighbourAverageColours:
    """Tests for neighbour average colour computation."""

    def test_single_tile_no_neighbours(self):
        tile = _FakeTile(0, ())
        colour_map = {0: (1.0, 0.0, 0.0)}
        result = compute_neighbour_average_colours(colour_map, [tile])
        assert result[0] == (1.0, 0.0, 0.0)

    def test_two_adjacent_tiles(self):
        tiles = [
            _FakeTile(0, (1,)),
            _FakeTile(1, (0,)),
        ]
        colour_map = {0: (1.0, 0.0, 0.0), 1: (0.0, 0.0, 1.0)}
        result = compute_neighbour_average_colours(colour_map, tiles)
        # Tile 0's neighbour avg = colour of tile 1
        assert result[0] == (0.0, 0.0, 1.0)
        # Tile 1's neighbour avg = colour of tile 0
        assert result[1] == (1.0, 0.0, 0.0)

    def test_three_tiles_triangle(self):
        tiles = [
            _FakeTile(0, (1, 2)),
            _FakeTile(1, (0, 2)),
            _FakeTile(2, (0, 1)),
        ]
        colour_map = {
            0: (1.0, 0.0, 0.0),
            1: (0.0, 1.0, 0.0),
            2: (0.0, 0.0, 1.0),
        }
        result = compute_neighbour_average_colours(colour_map, tiles)
        # Tile 0's avg = mean(green, blue) = (0, 0.5, 0.5)
        assert abs(result[0][0] - 0.0) < 1e-9
        assert abs(result[0][1] - 0.5) < 1e-9
        assert abs(result[0][2] - 0.5) < 1e-9

    def test_missing_neighbour_in_colour_map(self):
        tiles = [
            _FakeTile(0, (1, 2)),
            _FakeTile(1, (0,)),
        ]
        colour_map = {0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0)}
        # Tile 0 has neighbour 2, which is not in colour_map — skip it
        result = compute_neighbour_average_colours(colour_map, tiles)
        # Only neighbour 1 contributes to tile 0's average
        assert result[0] == (0.0, 1.0, 0.0)


class TestHarmoniseTileColours:
    """Tests for the full harmonisation (colour smoothing)."""

    def test_zero_strength_returns_original(self):
        tiles = [
            _FakeTile(0, (1,)),
            _FakeTile(1, (0,)),
        ]
        colour_map = {0: (1.0, 0.0, 0.0), 1: (0.0, 0.0, 1.0)}
        result = harmonise_tile_colours(colour_map, tiles, strength=0.0)
        assert result[0] == (1.0, 0.0, 0.0)
        assert result[1] == (0.0, 0.0, 1.0)

    def test_full_strength_replaces_with_neighbour_avg(self):
        tiles = [
            _FakeTile(0, (1,)),
            _FakeTile(1, (0,)),
        ]
        colour_map = {0: (1.0, 0.0, 0.0), 1: (0.0, 0.0, 1.0)}
        result = harmonise_tile_colours(colour_map, tiles, strength=1.0)
        # Fully replaced: tile 0 gets tile 1's colour
        assert abs(result[0][0] - 0.0) < 1e-9
        assert abs(result[0][2] - 1.0) < 1e-9

    def test_half_strength_blends(self):
        tiles = [
            _FakeTile(0, (1,)),
            _FakeTile(1, (0,)),
        ]
        colour_map = {0: (1.0, 0.0, 0.0), 1: (0.0, 0.0, 1.0)}
        result = harmonise_tile_colours(colour_map, tiles, strength=0.5)
        # Tile 0: 0.5*(1,0,0) + 0.5*(0,0,1) = (0.5, 0, 0.5)
        assert abs(result[0][0] - 0.5) < 1e-9
        assert abs(result[0][2] - 0.5) < 1e-9

    def test_does_not_mutate_original(self):
        tiles = [_FakeTile(0, (1,)), _FakeTile(1, (0,))]
        colour_map = {0: (1.0, 0.0, 0.0), 1: (0.0, 0.0, 1.0)}
        original_copy = dict(colour_map)
        harmonise_tile_colours(colour_map, tiles, strength=0.5)
        assert colour_map == original_copy

    def test_reduces_colour_difference_between_neighbours(self):
        tiles = [
            _FakeTile(0, (1,)),
            _FakeTile(1, (0,)),
        ]
        colour_map = {0: (1.0, 0.0, 0.0), 1: (0.0, 0.0, 1.0)}
        result = harmonise_tile_colours(colour_map, tiles, strength=0.5)
        # Original colour distance
        orig_dist = sum((a - b) ** 2 for a, b in zip(colour_map[0], colour_map[1])) ** 0.5
        # Harmonised colour distance
        harm_dist = sum((a - b) ** 2 for a, b in zip(result[0], result[1])) ** 0.5
        assert harm_dist < orig_dist


class TestSubdivideWithEdgeColour:
    """Tests for per-vertex colour blending in subdivide_tile_mesh."""

    @pytest.fixture
    def hex_tile_data(self):
        """Hex tile centred at (0,0,1), UVs at (0.5,0.5)."""
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

    def test_no_edge_colour_gives_uniform_colour(self, hex_tile_data):
        """Without edge_color, all vertices have the same colour."""
        center, vertices, center_uv, vertex_uvs = hex_tile_data
        vdata, _ = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(0.5, 0.5, 0.5), subdivisions=2,
        )
        colours = vdata[:, 3:6]
        np.testing.assert_allclose(colours, 0.5, atol=1e-6)

    def test_edge_colour_center_vertex_keeps_center_colour(self, hex_tile_data):
        """The center vertex (b0=1) should have exactly the center colour."""
        center, vertices, center_uv, vertex_uvs = hex_tile_data
        vdata, _ = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 0.0, 0.0), subdivisions=3,
            edge_color=(0.0, 0.0, 1.0),
        )
        # The center vertex is at position (0,0,1) projected to sphere
        # Find it: the vertex closest to (0, 0, radius)
        dists = np.linalg.norm(vdata[:, :3] - np.array([0, 0, 1]), axis=1)
        center_idx = np.argmin(dists)
        center_col = vdata[center_idx, 3:6]
        np.testing.assert_allclose(center_col, [1.0, 0.0, 0.0], atol=1e-5)

    def test_edge_colour_boundary_vertices_blend_toward_edge(self, hex_tile_data):
        """Boundary vertices (b0=0) should have exactly the edge colour."""
        center, vertices, center_uv, vertex_uvs = hex_tile_data
        vdata, _ = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 0.0, 0.0), subdivisions=3,
            edge_color=(0.0, 0.0, 1.0),
        )
        # Boundary vertices are furthest from center
        dists = np.linalg.norm(vdata[:, :3] - np.array([0, 0, 1]), axis=1)
        max_dist = dists.max()
        # Vertices at max distance are on the boundary (b0=0)
        boundary_mask = dists > max_dist - 1e-5
        boundary_colours = vdata[boundary_mask, 3:6]
        # Should be pure edge colour (0, 0, 1)
        np.testing.assert_allclose(
            boundary_colours[:, 0], 0.0, atol=1e-5,
        )
        np.testing.assert_allclose(
            boundary_colours[:, 2], 1.0, atol=1e-5,
        )

    def test_edge_colour_produces_gradient(self, hex_tile_data):
        """Intermediate vertices should have interpolated colours."""
        center, vertices, center_uv, vertex_uvs = hex_tile_data
        vdata, _ = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 0.0, 0.0), subdivisions=4,
            edge_color=(0.0, 1.0, 0.0),
        )
        # Check that not all vertices have the same colour
        colours = vdata[:, 3:6]
        assert colours.std() > 0.01, "Expected colour variation across vertices"

    def test_edge_colour_same_as_center_gives_uniform(self, hex_tile_data):
        """If edge_color == color, result should be uniform colour."""
        center, vertices, center_uv, vertex_uvs = hex_tile_data
        col = (0.3, 0.6, 0.9)
        vdata, _ = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=col, subdivisions=2,
            edge_color=col,
        )
        colours = vdata[:, 3:6]
        np.testing.assert_allclose(colours[:, 0], 0.3, atol=1e-5)
        np.testing.assert_allclose(colours[:, 1], 0.6, atol=1e-5)
        np.testing.assert_allclose(colours[:, 2], 0.9, atol=1e-5)

    def test_edge_colour_preserves_topology(self, hex_tile_data):
        """edge_color should not change vertex count or triangle count."""
        center, vertices, center_uv, vertex_uvs = hex_tile_data
        vdata_plain, idata_plain = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(0.5, 0.5, 0.5), subdivisions=3,
        )
        vdata_blend, idata_blend = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(0.5, 0.5, 0.5), subdivisions=3,
            edge_color=(0.0, 1.0, 0.0),
        )
        assert vdata_blend.shape == vdata_plain.shape
        assert idata_blend.shape == idata_plain.shape

    def test_edge_colour_preserves_positions_and_uvs(self, hex_tile_data):
        """edge_color should not change 3D positions or UVs."""
        center, vertices, center_uv, vertex_uvs = hex_tile_data
        vdata_plain, _ = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(0.5, 0.5, 0.5), subdivisions=3,
        )
        vdata_blend, _ = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(0.5, 0.5, 0.5), subdivisions=3,
            edge_color=(0.0, 1.0, 0.0),
        )
        # Positions (0:3) should match
        np.testing.assert_allclose(
            vdata_blend[:, :3], vdata_plain[:, :3], atol=1e-6,
        )
        # UVs (6:8) should match
        np.testing.assert_allclose(
            vdata_blend[:, 6:8], vdata_plain[:, 6:8], atol=1e-6,
        )


@pytest.mark.skipif(not _HAS_MODELS, reason="models library required")
class TestBatchedMeshWithEdgeBlend:
    """Tests for edge_blend in the batched globe mesh builder."""

    @pytest.fixture
    def uv_layout_f3(self):
        """Minimal UV layout for freq=3 tiles."""
        from models.objects.goldberg import generate_goldberg_tiles
        tiles = generate_goldberg_tiles(frequency=3, radius=1.0)
        n = len(tiles)
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

    @pytest.fixture
    def colour_map_f3(self):
        """Colour map with distinct colours for first few tiles."""
        return {
            i: ((i * 37 % 256) / 255.0,
                (i * 73 % 256) / 255.0,
                (i * 113 % 256) / 255.0)
            for i in range(92)
        }

    def test_edge_blend_zero_matches_no_blend(self, uv_layout_f3, colour_map_f3):
        """edge_blend=0 should produce identical results to default."""
        vdata_none, idata_none = build_batched_globe_mesh(
            3, uv_layout_f3, colour_map_f3, subdivisions=1,
        )
        vdata_zero, idata_zero = build_batched_globe_mesh(
            3, uv_layout_f3, colour_map_f3, subdivisions=1,
            edge_blend=0.0,
        )
        np.testing.assert_array_equal(vdata_none, vdata_zero)
        np.testing.assert_array_equal(idata_none, idata_zero)

    def test_edge_blend_produces_output(self, uv_layout_f3, colour_map_f3):
        """Batched mesh with edge blend should produce valid output."""
        vdata, idata = build_batched_globe_mesh(
            3, uv_layout_f3, colour_map_f3, subdivisions=2,
            edge_blend=0.5,
        )
        assert len(vdata) > 0
        assert len(idata) > 0

    def test_edge_blend_same_topology(self, uv_layout_f3, colour_map_f3):
        """Edge blend should not change topology."""
        vdata_plain, idata_plain = build_batched_globe_mesh(
            3, uv_layout_f3, colour_map_f3, subdivisions=1,
        )
        vdata_blend, idata_blend = build_batched_globe_mesh(
            3, uv_layout_f3, colour_map_f3, subdivisions=1,
            edge_blend=0.5,
        )
        assert vdata_blend.shape[0] == vdata_plain.shape[0]
        assert idata_blend.shape[0] == idata_plain.shape[0]

    def test_edge_blend_changes_colours(self, uv_layout_f3, colour_map_f3):
        """With distinct tile colours, edge blend should change some vertex colours."""
        vdata_plain, _ = build_batched_globe_mesh(
            3, uv_layout_f3, colour_map_f3, subdivisions=2,
        )
        vdata_blend, _ = build_batched_globe_mesh(
            3, uv_layout_f3, colour_map_f3, subdivisions=2,
            edge_blend=0.5,
        )
        # Positions should match
        np.testing.assert_allclose(
            vdata_blend[:, :3], vdata_plain[:, :3], atol=1e-6,
        )
        # Some colours should differ
        colour_diff = np.abs(vdata_blend[:, 3:6] - vdata_plain[:, 3:6])
        assert colour_diff.max() > 0.01, (
            "Expected some colour difference with edge_blend=0.5"
        )

    def test_edge_blend_without_colour_map_has_no_effect(self, uv_layout_f3):
        """Without a tile colour map, edge blend should have no effect."""
        vdata_none, idata_none = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1,
        )
        vdata_blend, idata_blend = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1,
            edge_blend=0.5,
        )
        np.testing.assert_array_equal(vdata_none, vdata_blend)


# ═══════════════════════════════════════════════════════════════════
# Phase 13E — Normal-mapped lighting tests
# ═══════════════════════════════════════════════════════════════════

class TestEncodeNormalToRgb:
    """Tests for normal vector ↔ RGB encoding/decoding."""

    def test_flat_normal_is_blue(self):
        """A flat-up normal (0, 0, 1) should encode as ~(128, 128, 255)."""
        r, g, b = encode_normal_to_rgb(0.0, 0.0, 1.0)
        assert r == 128
        assert g == 128
        assert b == 255

    def test_positive_x_normal(self):
        """Normal (1, 0, 0) should encode as (255, 128, 128)."""
        r, g, b = encode_normal_to_rgb(1.0, 0.0, 0.0)
        assert r == 255
        assert g == 128
        assert b == 128

    def test_negative_x_normal(self):
        """Normal (-1, 0, 0) should encode as (0, 128, 128)."""
        r, g, b = encode_normal_to_rgb(-1.0, 0.0, 0.0)
        assert r == 0
        assert g == 128
        assert b == 128

    def test_clamping(self):
        """Values outside [-1, 1] should be clamped."""
        r, g, b = encode_normal_to_rgb(2.0, -2.0, 0.5)
        assert 0 <= r <= 255
        assert 0 <= g <= 255
        assert 0 <= b <= 255

    def test_round_trip(self):
        """encode → decode should approximately recover the original."""
        for nx, ny, nz in [(0, 0, 1), (1, 0, 0), (0, 1, 0),
                           (0.577, 0.577, 0.577)]:
            length = math.sqrt(nx*nx + ny*ny + nz*nz)
            nx, ny, nz = nx/length, ny/length, nz/length
            r, g, b = encode_normal_to_rgb(nx, ny, nz)
            dnx, dny, dnz = decode_rgb_to_normal(r, g, b)
            assert abs(dnx - nx) < 0.02, f"nx mismatch: {dnx} vs {nx}"
            assert abs(dny - ny) < 0.02, f"ny mismatch: {dny} vs {ny}"
            assert abs(dnz - nz) < 0.02, f"nz mismatch: {dnz} vs {nz}"

    def test_decode_normalises(self):
        """Decoded normal should be unit length."""
        r, g, b = encode_normal_to_rgb(0.6, 0.8, 0.0)
        dnx, dny, dnz = decode_rgb_to_normal(r, g, b)
        length = math.sqrt(dnx*dnx + dny*dny + dnz*dnz)
        assert abs(length - 1.0) < 0.02


class TestSubdivideWithTangent:
    """Tests for tangent/bitangent in subdivide_tile_mesh."""

    @pytest.fixture
    def hex_data(self, hex_tile_data):
        center, vertices, center_uv, vertex_uvs = hex_tile_data
        return center, vertices, center_uv, vertex_uvs

    def test_without_tangent_gives_8_columns(self, hex_data):
        """Without tangent/bitangent, vertex data should have 8 columns."""
        center, vertices, center_uv, vertex_uvs = hex_data
        vdata, idata = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 0.0, 0.0), subdivisions=2,
        )
        assert vdata.shape[1] == 8

    def test_with_tangent_gives_14_columns(self, hex_data):
        """With tangent/bitangent, vertex data should have 14 columns."""
        center, vertices, center_uv, vertex_uvs = hex_data
        vdata, idata = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 0.0, 0.0), subdivisions=2,
            tangent=(1.0, 0.0, 0.0),
            bitangent=(0.0, 1.0, 0.0),
        )
        assert vdata.shape[1] == 14

    def test_tangent_columns_are_unit_length(self, hex_data):
        """Tangent and bitangent columns should be approximately unit length."""
        center, vertices, center_uv, vertex_uvs = hex_data
        vdata, idata = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 0.0, 0.0), subdivisions=2,
            tangent=(1.0, 0.0, 0.0),
            bitangent=(0.0, 1.0, 0.0),
        )
        tangents = vdata[:, 8:11]
        bitangents = vdata[:, 11:14]
        t_lengths = np.linalg.norm(tangents, axis=1)
        b_lengths = np.linalg.norm(bitangents, axis=1)
        np.testing.assert_allclose(t_lengths, 1.0, atol=0.01)
        np.testing.assert_allclose(b_lengths, 1.0, atol=0.01)

    def test_tangent_orthogonal_to_sphere_normal(self, hex_data):
        """Tangent should be approximately orthogonal to the sphere normal."""
        center, vertices, center_uv, vertex_uvs = hex_data
        vdata, idata = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 0.0, 0.0), subdivisions=2,
            tangent=(1.0, 0.0, 0.0),
            bitangent=(0.0, 1.0, 0.0),
        )
        positions = vdata[:, 0:3]
        tangents = vdata[:, 8:11]
        # Sphere normal = normalised position
        norms = positions / np.linalg.norm(positions, axis=1, keepdims=True)
        dots = np.abs(np.sum(norms * tangents, axis=1))
        assert dots.max() < 0.15, f"Tangent not orthogonal to normal, max dot={dots.max()}"

    def test_positions_unchanged_by_tangent(self, hex_data):
        """Positions and UVs should not change when tangent is added."""
        center, vertices, center_uv, vertex_uvs = hex_data
        vdata_8, idata_8 = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 0.0, 0.0), subdivisions=2,
        )
        vdata_14, idata_14 = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 0.0, 0.0), subdivisions=2,
            tangent=(1.0, 0.0, 0.0),
            bitangent=(0.0, 1.0, 0.0),
        )
        # First 8 columns should match
        np.testing.assert_allclose(vdata_14[:, :8], vdata_8[:, :8], atol=1e-6)
        np.testing.assert_array_equal(idata_14, idata_8)

    def test_only_tangent_without_bitangent_gives_8_columns(self, hex_data):
        """If only tangent is provided (no bitangent), revert to 8 columns."""
        center, vertices, center_uv, vertex_uvs = hex_data
        vdata, idata = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 0.0, 0.0), subdivisions=2,
            tangent=(1.0, 0.0, 0.0),
            bitangent=None,
        )
        assert vdata.shape[1] == 8


class TestBatchedMeshNormalMapped:
    """Tests for build_batched_globe_mesh with normal_mapped=True."""

    @pytest.fixture
    def uv_layout_f3(self):
        from models.objects.goldberg import generate_goldberg_tiles
        tiles = generate_goldberg_tiles(frequency=3)
        layout = {}
        n = len(tiles)
        cols = max(1, math.isqrt(n))
        if cols * cols < n:
            cols += 1
        for i, tile in enumerate(tiles):
            fid = f"t{tile.index}"
            c = i % cols
            r = i // cols
            u_min = c / cols
            u_max = (c + 1) / cols
            v_min = r / cols
            v_max = (r + 1) / cols
            layout[fid] = (u_min, v_min, u_max, v_max)
        return layout

    @pytest.fixture
    def colour_map_f3(self):
        from models.objects.goldberg import generate_goldberg_tiles
        tiles = generate_goldberg_tiles(frequency=3)
        return {
            tile.index: (
                (tile.index * 37 % 256) / 255.0,
                (tile.index * 73 % 256) / 255.0,
                (tile.index * 113 % 256) / 255.0,
            )
            for tile in tiles
        }

    def test_normal_mapped_false_gives_8_columns(self, uv_layout_f3, colour_map_f3):
        """normal_mapped=False (default) should give 8-column vertex data."""
        vdata, idata = build_batched_globe_mesh(
            3, uv_layout_f3, colour_map_f3, subdivisions=1,
        )
        assert vdata.shape[1] == 8

    def test_normal_mapped_true_gives_14_columns(self, uv_layout_f3, colour_map_f3):
        """normal_mapped=True should give 14-column vertex data."""
        vdata, idata = build_batched_globe_mesh(
            3, uv_layout_f3, colour_map_f3, subdivisions=1,
            normal_mapped=True,
        )
        assert vdata.shape[1] == 14

    def test_normal_mapped_same_positions(self, uv_layout_f3, colour_map_f3):
        """Positions should be the same with or without normal mapping."""
        vdata_8, idata_8 = build_batched_globe_mesh(
            3, uv_layout_f3, colour_map_f3, subdivisions=1,
        )
        vdata_14, idata_14 = build_batched_globe_mesh(
            3, uv_layout_f3, colour_map_f3, subdivisions=1,
            normal_mapped=True,
        )
        np.testing.assert_allclose(vdata_14[:, :3], vdata_8[:, :3], atol=1e-6)

    def test_normal_mapped_tangent_unit_length(self, uv_layout_f3, colour_map_f3):
        """All tangent vectors should be approximately unit length."""
        vdata, idata = build_batched_globe_mesh(
            3, uv_layout_f3, colour_map_f3, subdivisions=1,
            normal_mapped=True,
        )
        tangents = vdata[:, 8:11]
        lengths = np.linalg.norm(tangents, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=0.02)

    def test_normal_mapped_bitangent_unit_length(self, uv_layout_f3, colour_map_f3):
        """All bitangent vectors should be approximately unit length."""
        vdata, idata = build_batched_globe_mesh(
            3, uv_layout_f3, colour_map_f3, subdivisions=1,
            normal_mapped=True,
        )
        bitangents = vdata[:, 11:14]
        lengths = np.linalg.norm(bitangents, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=0.02)

    def test_normal_mapped_topology_unchanged(self, uv_layout_f3, colour_map_f3):
        """Normal mapping should not change vertex or triangle counts."""
        vdata_8, idata_8 = build_batched_globe_mesh(
            3, uv_layout_f3, colour_map_f3, subdivisions=1,
        )
        vdata_14, idata_14 = build_batched_globe_mesh(
            3, uv_layout_f3, colour_map_f3, subdivisions=1,
            normal_mapped=True,
        )
        assert vdata_14.shape[0] == vdata_8.shape[0]
        assert idata_14.shape[0] == idata_8.shape[0]


class TestBuildNormalMapAtlas:
    """Tests for normal map atlas building."""

    @pytest.fixture
    def _small_collection(self, tmp_path):
        """Build a tiny 2-tile detail grid collection mock for testing."""
        from polygrid.polygrid import PolyGrid, Vertex, Face
        from polygrid.tile_data import TileDataStore, TileSchema, FieldDef

        # Build a minimal collection with 2 face IDs
        face_ids_list = ["t0", "t1"]
        grids_dict = {}
        stores_dict = {}

        for fid in face_ids_list:
            vertices = [
                Vertex("v0", 0.0, 0.0),
                Vertex("v1", 1.0, 0.0),
                Vertex("v2", 1.0, 1.0),
                Vertex("v3", 0.0, 1.0),
                Vertex("v4", 0.5, 0.5),
            ]
            faces = [
                Face(f"{fid}_f0", "triangle", ("v0", "v1", "v4")),
                Face(f"{fid}_f1", "triangle", ("v1", "v2", "v4")),
                Face(f"{fid}_f2", "triangle", ("v2", "v3", "v4")),
                Face(f"{fid}_f3", "triangle", ("v3", "v0", "v4")),
            ]
            grid = PolyGrid(vertices=vertices, edges=[], faces=faces)
            schema = TileSchema([FieldDef("elevation", float, 0.0)])
            store = TileDataStore(grid=grid, schema=schema)
            for f in faces:
                store.set(f.id, "elevation", 0.5)
            grids_dict[fid] = grid
            stores_dict[fid] = store

        class _MockCollection:
            """Lightweight mock of DetailGridCollection."""
            def __init__(self, face_ids, grids, stores):
                self._face_ids = face_ids
                self._grids = grids
                self._stores = stores

            @property
            def face_ids(self):
                return sorted(self._grids.keys())

            @property
            def grids(self):
                return dict(self._grids)

        collection = _MockCollection(face_ids_list, grids_dict, stores_dict)
        return collection, stores_dict

    def test_atlas_is_rgb_image(self, _small_collection):
        """The atlas should be an RGB PIL Image."""
        from PIL import Image
        collection, stores = _small_collection

        normal_maps = {}
        for fid in collection.face_ids:
            grid = collection.grids[fid]
            normal_maps[fid] = {
                sfid: (0.0, 0.0, 1.0) for sfid in grid.faces
            }

        atlas, uv_layout = build_normal_map_atlas(
            normal_maps, collection, tile_size=32, gutter=2,
        )
        assert isinstance(atlas, Image.Image)
        assert atlas.mode == "RGB"

    def test_atlas_uv_layout_covers_all_faces(self, _small_collection):
        """UV layout should have an entry for every face ID."""
        collection, stores = _small_collection

        normal_maps = {}
        for fid in collection.face_ids:
            grid = collection.grids[fid]
            normal_maps[fid] = {
                sfid: (0.0, 0.0, 1.0) for sfid in grid.faces
            }

        atlas, uv_layout = build_normal_map_atlas(
            normal_maps, collection, tile_size=32, gutter=2,
        )
        for fid in collection.face_ids:
            assert fid in uv_layout

    def test_atlas_uv_values_in_range(self, _small_collection):
        """All UV values should be in [0, 1]."""
        collection, stores = _small_collection

        normal_maps = {
            fid: {sfid: (0.0, 0.0, 1.0) for sfid in collection.grids[fid].faces}
            for fid in collection.face_ids
        }

        atlas, uv_layout = build_normal_map_atlas(
            normal_maps, collection, tile_size=32, gutter=2,
        )
        for fid, (u_min, v_min, u_max, v_max) in uv_layout.items():
            assert 0.0 <= u_min <= 1.0
            assert 0.0 <= v_min <= 1.0
            assert 0.0 <= u_max <= 1.0
            assert 0.0 <= v_max <= 1.0
            assert u_min < u_max
            assert v_min < v_max

    def test_flat_normals_produce_blue_atlas(self, _small_collection):
        """Flat normals (0,0,1) should produce a predominantly blue atlas."""
        collection, stores = _small_collection

        normal_maps = {
            fid: {sfid: (0.0, 0.0, 1.0) for sfid in collection.grids[fid].faces}
            for fid in collection.face_ids
        }

        atlas, uv_layout = build_normal_map_atlas(
            normal_maps, collection, tile_size=32, gutter=0,
        )
        import numpy as np_
        pixels = np_.array(atlas)
        # Blue channel should dominate (flat normal blue = 255)
        avg_b = pixels[:, :, 2].mean()
        avg_r = pixels[:, :, 0].mean()
        assert avg_b > avg_r + 50, (
            f"Expected blue-dominant atlas, got avg R={avg_r:.1f}, B={avg_b:.1f}"
        )

    def test_atlas_gutter_fills(self, _small_collection):
        """Atlas with gutter > 0 should be larger than without."""
        collection, stores = _small_collection
        normal_maps = {
            fid: {sfid: (0.0, 0.0, 1.0) for sfid in collection.grids[fid].faces}
            for fid in collection.face_ids
        }

        atlas_no_gutter, _ = build_normal_map_atlas(
            normal_maps, collection, tile_size=32, gutter=0,
        )
        atlas_with_gutter, _ = build_normal_map_atlas(
            normal_maps, collection, tile_size=32, gutter=4,
        )
        # With gutter, atlas should be larger
        assert atlas_with_gutter.size[0] >= atlas_no_gutter.size[0]
        assert atlas_with_gutter.size[1] >= atlas_no_gutter.size[1]


class TestPBRShaderSources:
    """Tests for PBR shader source retrieval."""

    def test_get_pbr_shader_sources_returns_strings(self):
        """Should return two non-empty strings."""
        vs, fs = get_pbr_shader_sources()
        assert isinstance(vs, str) and len(vs) > 100
        assert isinstance(fs, str) and len(fs) > 100

    def test_get_v2_shader_sources_returns_strings(self):
        """Should return two non-empty strings for legacy shaders."""
        vs, fs = get_v2_shader_sources()
        assert isinstance(vs, str) and len(vs) > 50
        assert isinstance(fs, str) and len(fs) > 50

    def test_pbr_vertex_shader_has_tangent_attribute(self):
        """PBR vertex shader should declare tangent/bitangent inputs."""
        vs, _ = get_pbr_shader_sources()
        assert "in vec3 tangent" in vs
        assert "in vec3 bitangent" in vs

    def test_pbr_fragment_shader_has_normal_map(self):
        """PBR fragment shader should sample a normal map."""
        _, fs = get_pbr_shader_sources()
        assert "u_normal_map" in fs
        assert "u_use_normal_map" in fs

    def test_pbr_fragment_has_specular(self):
        """PBR fragment shader should have Blinn-Phong specular."""
        _, fs = get_pbr_shader_sources()
        assert "specular" in fs.lower() or "spec" in fs

    def test_pbr_fragment_has_fresnel(self):
        """PBR fragment shader should have Fresnel rim lighting."""
        _, fs = get_pbr_shader_sources()
        assert "fresnel" in fs.lower() or "FRESNEL" in fs

    def test_pbr_fragment_has_hemisphere_ambient(self):
        """PBR fragment shader should have hemisphere ambient (sky/ground)."""
        _, fs = get_pbr_shader_sources()
        assert "SKY_AMB" in fs
        assert "GND_AMB" in fs

    def test_pbr_fragment_has_fill_light(self):
        """PBR fragment shader should have a fill light."""
        _, fs = get_pbr_shader_sources()
        assert "u_fill_dir" in fs
        assert "FILL_COLOR" in fs

    def test_pbr_vertex_has_tbn_matrix(self):
        """PBR vertex shader should output a TBN matrix."""
        vs, _ = get_pbr_shader_sources()
        assert "v_tbn" in vs

    def test_pbr_shaders_are_glsl_330(self):
        """Both PBR shaders should be GLSL 330 core."""
        vs, fs = get_pbr_shader_sources()
        assert "#version 330 core" in vs
        assert "#version 330 core" in fs

    def test_pbr_fragment_has_tone_mapping(self):
        """PBR fragment shader should include tone mapping."""
        _, fs = get_pbr_shader_sources()
        # Reinhard tone mapping: color / (color + 1)
        assert "color + vec3(1.0)" in fs or "Reinhard" in fs


# ═══════════════════════════════════════════════════════════════════
# 13H — Water rendering tests
# ═══════════════════════════════════════════════════════════════════


class TestClassifyWaterTiles:
    """Tests for the water tile classification function."""

    def test_deep_blue_is_water(self):
        """A tile with dominant blue should be classified as water."""
        colours = {0: (0.05, 0.08, 0.30)}
        result = classify_water_tiles(colours)
        assert result[0] is True

    def test_green_is_land(self):
        """A green tile should be classified as land."""
        colours = {0: (0.2, 0.6, 0.15)}
        result = classify_water_tiles(colours)
        assert result[0] is False

    def test_red_is_land(self):
        """A reddish/brown tile should be classified as land."""
        colours = {0: (0.5, 0.3, 0.1)}
        result = classify_water_tiles(colours)
        assert result[0] is False

    def test_white_is_land(self):
        """White (snow) should be classified as land."""
        colours = {0: (0.9, 0.9, 0.9)}
        result = classify_water_tiles(colours)
        assert result[0] is False

    def test_borderline_below_threshold(self):
        """A tile with blue excess just below the threshold → land."""
        # blue - max(r, g) = 0.30 - 0.19 = 0.11 < 0.12
        colours = {0: (0.19, 0.10, 0.30)}
        result = classify_water_tiles(colours)
        assert result[0] is False

    def test_borderline_at_threshold(self):
        """A tile with blue excess exactly at the threshold → water."""
        # blue - max(r, g) = 0.30 - 0.18 = 0.12 >= 0.12
        colours = {0: (0.18, 0.10, 0.30)}
        result = classify_water_tiles(colours)
        assert result[0] is True

    def test_multiple_tiles(self):
        """Multiple tiles should be classified independently."""
        colours = {
            0: (0.05, 0.08, 0.30),  # water
            1: (0.5, 0.6, 0.2),     # land
            2: (0.03, 0.05, 0.25),  # water
            3: (0.8, 0.8, 0.8),     # land (snow)
        }
        result = classify_water_tiles(colours)
        assert result[0] is True
        assert result[1] is False
        assert result[2] is True
        assert result[3] is False

    def test_custom_water_level(self):
        """Custom water_level threshold should be respected."""
        colours = {0: (0.10, 0.10, 0.30)}
        # blue excess = 0.20, with threshold 0.25 → land
        result = classify_water_tiles(colours, water_level=0.25)
        assert result[0] is False
        # with threshold 0.15 → water
        result = classify_water_tiles(colours, water_level=0.15)
        assert result[0] is True

    def test_empty_map(self):
        """Empty colour map should return empty dict."""
        result = classify_water_tiles({})
        assert result == {}

    def test_default_water_level_matches_biome_config(self):
        """DEFAULT_WATER_LEVEL should match BiomeConfig.water_level."""
        from polygrid.detail_render import BiomeConfig
        assert DEFAULT_WATER_LEVEL == BiomeConfig().water_level


class TestComputeWaterDepth:
    """Tests for the water depth computation."""

    def test_land_returns_zero(self):
        """Land tiles should have zero depth."""
        assert compute_water_depth(0.5, 0.6, 0.2) == 0.0

    def test_deep_ocean_near_one(self):
        """Deep blue ocean should have high depth."""
        depth = compute_water_depth(0.02, 0.03, 0.90)
        assert depth > 0.7

    def test_shallow_water_near_zero(self):
        """Shallow water (just above threshold) should have low depth."""
        # blue excess = 0.30 - 0.18 = 0.12 = water_level → depth ≈ 0
        depth = compute_water_depth(0.18, 0.10, 0.30)
        assert depth < 0.1

    def test_depth_range_zero_to_one(self):
        """Depth should always be in [0, 1]."""
        for r, g, b in [
            (0.0, 0.0, 1.0),
            (0.05, 0.08, 0.5),
            (0.1, 0.1, 0.3),
        ]:
            d = compute_water_depth(r, g, b)
            assert 0.0 <= d <= 1.0

    def test_custom_water_level(self):
        """Custom threshold should affect depth computation."""
        # blue excess = 0.20, threshold = 0.10 → water, nonzero depth
        d1 = compute_water_depth(0.10, 0.10, 0.30, water_level=0.10)
        assert d1 > 0.0
        # threshold = 0.25 → land, zero depth
        d2 = compute_water_depth(0.10, 0.10, 0.30, water_level=0.25)
        assert d2 == 0.0


class TestSubdivideWithWaterFlag:
    """Tests for the water_flag parameter in subdivide_tile_mesh."""

    def test_water_flag_none_default_stride_8(self, hex_tile_data):
        """With water_flag=None (default), stride should remain 8."""
        center, verts, cuv, vuvs = hex_tile_data
        vdata, _ = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (0.5, 0.5, 0.5),
            subdivisions=1,
        )
        assert vdata.shape[1] == 8

    def test_water_flag_one_stride_9(self, hex_tile_data):
        """With water_flag=1.0, stride should expand to 9."""
        center, verts, cuv, vuvs = hex_tile_data
        vdata, _ = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (0.0, 0.1, 0.8),
            subdivisions=1,
            water_flag=1.0,
        )
        assert vdata.shape[1] == 9

    def test_water_flag_zero_stride_9(self, hex_tile_data):
        """With water_flag=0.0 (explicit), stride should still be 9."""
        center, verts, cuv, vuvs = hex_tile_data
        vdata, _ = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (0.5, 0.5, 0.5),
            subdivisions=1,
            water_flag=0.0,
        )
        assert vdata.shape[1] == 9

    def test_water_flag_stored_in_column_8(self, hex_tile_data):
        """Water flag should be stored in the last column (index 8)."""
        center, verts, cuv, vuvs = hex_tile_data
        vdata, _ = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (0.0, 0.1, 0.8),
            subdivisions=1,
            water_flag=1.0,
        )
        np.testing.assert_allclose(vdata[:, 8], 1.0, atol=1e-6)

    def test_land_flag_zero_in_column_8(self, hex_tile_data):
        """Land tiles (water_flag=0.0) should have 0.0 in column 8."""
        center, verts, cuv, vuvs = hex_tile_data
        vdata, _ = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (0.5, 0.5, 0.2),
            subdivisions=1,
            water_flag=0.0,
        )
        assert vdata.shape[1] == 9
        np.testing.assert_allclose(vdata[:, 8], 0.0, atol=1e-6)

    def test_tbn_plus_water_stride_15(self, hex_tile_data):
        """With TBN and water_flag, stride should be 15."""
        center, verts, cuv, vuvs = hex_tile_data
        vdata, _ = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (0.0, 0.1, 0.8),
            subdivisions=1,
            tangent=(1.0, 0.0, 0.0),
            bitangent=(0.0, 1.0, 0.0),
            water_flag=1.0,
        )
        assert vdata.shape[1] == 15

    def test_tbn_water_flag_in_column_14(self, hex_tile_data):
        """With TBN, water flag should be at column 14."""
        center, verts, cuv, vuvs = hex_tile_data
        vdata, _ = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (0.0, 0.1, 0.8),
            subdivisions=1,
            tangent=(1.0, 0.0, 0.0),
            bitangent=(0.0, 1.0, 0.0),
            water_flag=1.0,
        )
        np.testing.assert_allclose(vdata[:, 14], 1.0, atol=1e-6)

    def test_tbn_no_water_stride_14(self, hex_tile_data):
        """TBN without water (None) should keep stride=14."""
        center, verts, cuv, vuvs = hex_tile_data
        vdata, _ = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (0.5, 0.5, 0.5),
            subdivisions=1,
            tangent=(1.0, 0.0, 0.0),
            bitangent=(0.0, 1.0, 0.0),
        )
        assert vdata.shape[1] == 14

    def test_positions_unchanged_by_water_flag(self, hex_tile_data):
        """Water flag should not alter vertex positions or UVs."""
        center, verts, cuv, vuvs = hex_tile_data
        vdata_land, _ = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (0.5, 0.5, 0.5),
            subdivisions=2,
        )
        vdata_water, _ = subdivide_tile_mesh(
            center, verts, cuv, vuvs, (0.5, 0.5, 0.5),
            subdivisions=2,
            water_flag=1.0,
        )
        # Positions (cols 0-2) and UVs (cols 6-7) should match
        np.testing.assert_allclose(
            vdata_land[:, :3], vdata_water[:, :3], atol=1e-6,
        )
        np.testing.assert_allclose(
            vdata_land[:, 6:8], vdata_water[:, 6:8], atol=1e-6,
        )


class TestBatchedMeshWithWater:
    """Tests for water_tiles parameter in build_batched_globe_mesh."""

    @pytest.fixture
    def uv_layout_f3(self):
        """Minimal UV layout for freq=3 tiles."""
        from models.objects.goldberg import generate_goldberg_tiles
        tiles = generate_goldberg_tiles(frequency=3, radius=1.0)
        n = len(tiles)
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

    def test_no_water_tiles_stride_8(self, uv_layout_f3):
        """Without water_tiles, stride should be 8."""
        vdata, _ = build_batched_globe_mesh(3, uv_layout_f3, subdivisions=1)
        assert vdata.shape[1] == 8

    def test_all_false_water_stride_8(self, uv_layout_f3):
        """When all tiles are land, stride should stay 8."""
        wt = {i: False for i in range(92)}
        vdata, _ = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1, water_tiles=wt,
        )
        assert vdata.shape[1] == 8

    def test_some_water_stride_9(self, uv_layout_f3):
        """When some tiles are water, stride should expand to 9."""
        wt = {i: (i < 10) for i in range(92)}
        vdata, _ = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1, water_tiles=wt,
        )
        assert vdata.shape[1] == 9

    def test_water_flag_values(self, uv_layout_f3):
        """Water tiles should have flag=1.0, land tiles flag=0.0."""
        wt = {i: (i < 5) for i in range(92)}
        vdata, _ = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1, water_tiles=wt,
        )
        # The water column is at index 8
        water_col = vdata[:, 8]
        # Should have both 0 and 1 values
        assert np.any(water_col > 0.5)
        assert np.any(water_col < 0.5)

    def test_water_normal_mapped_stride_15(self, uv_layout_f3):
        """Water + normal_mapped should produce stride=15."""
        wt = {i: (i < 10) for i in range(92)}
        vdata, _ = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1,
            normal_mapped=True, water_tiles=wt,
        )
        assert vdata.shape[1] == 15

    def test_no_water_normal_mapped_stride_14(self, uv_layout_f3):
        """Normal-mapped without water should stay at stride=14."""
        vdata, _ = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1, normal_mapped=True,
        )
        assert vdata.shape[1] == 14

    def test_triangle_count_unchanged(self, uv_layout_f3):
        """Water flag should not change the number of triangles."""
        _, idata_no = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1,
        )
        wt = {i: (i % 2 == 0) for i in range(92)}
        _, idata_water = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1, water_tiles=wt,
        )
        assert len(idata_no) == len(idata_water)

    def test_vertex_count_unchanged(self, uv_layout_f3):
        """Water flag should not change the number of vertices."""
        vdata_no, _ = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1,
        )
        wt = {i: (i % 2 == 0) for i in range(92)}
        vdata_water, _ = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1, water_tiles=wt,
        )
        assert len(vdata_no) == len(vdata_water)

    def test_empty_water_dict_no_effect(self, uv_layout_f3):
        """Empty water_tiles dict should produce stride=8."""
        vdata, _ = build_batched_globe_mesh(
            3, uv_layout_f3, subdivisions=1, water_tiles={},
        )
        assert vdata.shape[1] == 8


class TestPBRShaderWaterFeatures:
    """Tests for 13H water rendering features in PBR shaders."""

    def test_vertex_shader_has_water_flag_input(self):
        """PBR vertex shader should accept water_flag attribute."""
        vs, _ = get_pbr_shader_sources()
        assert "water_flag" in vs

    def test_vertex_shader_outputs_v_water(self):
        """PBR vertex shader should output v_water varying."""
        vs, _ = get_pbr_shader_sources()
        assert "v_water" in vs

    def test_fragment_shader_has_u_time(self):
        """PBR fragment shader should have u_time uniform for waves."""
        _, fs = get_pbr_shader_sources()
        assert "u_time" in fs

    def test_fragment_shader_has_water_shallow_deep(self):
        """Fragment shader should have depth-based water colours."""
        _, fs = get_pbr_shader_sources()
        assert "WATER_SHALLOW" in fs
        assert "WATER_DEEP" in fs

    def test_fragment_shader_has_wave_animation(self):
        """Fragment shader should have wave animation constants."""
        _, fs = get_pbr_shader_sources()
        assert "WAVE_SPEED" in fs
        assert "WAVE_SCALE" in fs
        assert "WAVE_AMPLITUDE" in fs

    def test_fragment_shader_has_coastline(self):
        """Fragment shader should have coastline emphasis."""
        _, fs = get_pbr_shader_sources()
        assert "COAST_COLOR" in fs
        assert "coast_factor" in fs

    def test_fragment_shader_uses_dFdx(self):
        """Coastline detection should use screen-space derivatives."""
        _, fs = get_pbr_shader_sources()
        assert "dFdx" in fs
        assert "dFdy" in fs

    def test_fragment_shader_water_hint_from_flag(self):
        """Water detection should use the per-vertex flag."""
        _, fs = get_pbr_shader_sources()
        assert "v_water" in fs

    def test_vertex_shader_water_location_5(self):
        """Water flag attribute should be at location 5."""
        vs, _ = get_pbr_shader_sources()
        assert "layout(location = 5) in float water_flag" in vs

    def test_fragment_shader_still_has_pbr_features(self):
        """Water additions should not break existing PBR features."""
        _, fs = get_pbr_shader_sources()
        assert "KEY_COLOR" in fs
        assert "FILL_COLOR" in fs
        assert "ROUGHNESS_TERRAIN" in fs
        assert "ROUGHNESS_WATER" in fs
        assert "FRESNEL_POWER" in fs
        assert "u_normal_map" in fs
        assert "Reinhard" in fs or "color + vec3(1.0)" in fs
