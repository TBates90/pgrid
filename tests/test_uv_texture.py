"""Tests for Phase 20A — UV-aligned texture rendering.

Covers:
- 20A.1 — Tile basis and UV projection
- 20A.2 — Affine transform from detail-2D to tile-UV
- 20A.3 — UV-aligned tile rasteriser
- 20B   — UV-aligned atlas builder
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pytest

from polygrid.models import Edge, Face, Vertex
from polygrid.polygrid import PolyGrid
from polygrid.uv_texture import (
    UVTransform,
    _normalize_vec,
    compute_detail_to_uv_transform,
    compute_tile_basis,
    compute_tile_uv_bounds,
    project_and_normalize,
    project_point_to_tile_uv,
    render_tile_uv_aligned,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers — build a mock globe grid with 3D hex/pent vertices
# ═══════════════════════════════════════════════════════════════════

def _make_hex_globe_tile(
    face_id: str = "t0",
    center_3d: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    radius: float = 0.1,
) -> PolyGrid:
    """Create a minimal PolyGrid with a single hexagonal face in 3D.

    The hex is arranged on the tangent plane perpendicular to the
    radial direction from the origin.
    """
    cx, cy, cz = center_3d
    c = np.array(center_3d, dtype=np.float64)
    # Normal points outward from origin
    normal = c / np.linalg.norm(c)

    # Build a tangent/bitangent basis
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    tangent = np.cross(up, normal)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    bitangent /= np.linalg.norm(bitangent)

    # 6 hex vertices on tangent plane
    verts = {}
    vids = []
    for i in range(6):
        angle = math.pi / 6 + 2 * math.pi * i / 6  # flat-top hex
        pt = c + radius * (math.cos(angle) * tangent + math.sin(angle) * bitangent)
        vid = f"v{i}"
        verts[vid] = Vertex(vid, float(pt[0]), float(pt[1]), float(pt[2]))
        vids.append(vid)

    face = Face(id=face_id, face_type="hex", vertex_ids=tuple(vids))
    edges = []
    for i in range(6):
        a, b = vids[i], vids[(i + 1) % 6]
        edges.append(Edge(id=f"e{i}", vertex_ids=(a, b), face_ids=(face_id,)))

    return PolyGrid(verts.values(), edges, [face])


def _make_pent_globe_tile(
    face_id: str = "t0",
    center_3d: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    radius: float = 0.1,
) -> PolyGrid:
    """Create a PolyGrid with a single pentagonal face in 3D."""
    c = np.array(center_3d, dtype=np.float64)
    normal = c / np.linalg.norm(c)

    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, up))) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    tangent = np.cross(up, normal)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    bitangent /= np.linalg.norm(bitangent)

    verts = {}
    vids = []
    for i in range(5):
        angle = math.pi / 2 + 2 * math.pi * i / 5
        pt = c + radius * (math.cos(angle) * tangent + math.sin(angle) * bitangent)
        vid = f"v{i}"
        verts[vid] = Vertex(vid, float(pt[0]), float(pt[1]), float(pt[2]))
        vids.append(vid)

    face = Face(id=face_id, face_type="pent", vertex_ids=tuple(vids))
    edges = []
    for i in range(5):
        a, b = vids[i], vids[(i + 1) % 5]
        edges.append(Edge(id=f"e{i}", vertex_ids=(a, b), face_ids=(face_id,)))

    return PolyGrid(verts.values(), edges, [face])


def _make_detail_grid_2d(n_sides: int = 6, rings: int = 2) -> PolyGrid:
    """Create a small 2D detail grid matching hex or pent shape."""
    from polygrid.builders import build_pure_hex_grid, build_pentagon_centered_grid

    if n_sides == 5:
        return build_pentagon_centered_grid(rings)
    else:
        return build_pure_hex_grid(rings)


# ═══════════════════════════════════════════════════════════════════
# 20A.1 — Tile basis and UV projection
# ═══════════════════════════════════════════════════════════════════

class TestNormalizeVec:
    def test_unit_vector_unchanged(self):
        v = np.array([1.0, 0.0, 0.0])
        result = _normalize_vec(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-10

    def test_zero_vector(self):
        v = np.array([0.0, 0.0, 0.0])
        result = _normalize_vec(v)
        assert np.linalg.norm(result) < 1e-10

    def test_arbitrary_vector(self):
        v = np.array([3.0, 4.0, 0.0])
        result = _normalize_vec(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-10
        np.testing.assert_allclose(result, [0.6, 0.8, 0.0], atol=1e-10)


class TestComputeTileBasis:
    def test_hex_basis_orthogonal(self):
        globe = _make_hex_globe_tile("t0", (1.0, 0.0, 0.0))
        center, normal, tangent, bitangent = compute_tile_basis(globe, "t0")

        # All unit vectors
        assert abs(np.linalg.norm(normal) - 1.0) < 1e-8
        assert abs(np.linalg.norm(tangent) - 1.0) < 1e-8
        assert abs(np.linalg.norm(bitangent) - 1.0) < 1e-8

        # Mutually orthogonal
        assert abs(float(np.dot(normal, tangent))) < 1e-8
        assert abs(float(np.dot(normal, bitangent))) < 1e-8
        assert abs(float(np.dot(tangent, bitangent))) < 1e-8

    def test_pent_basis_orthogonal(self):
        globe = _make_pent_globe_tile("t0", (0.0, 1.0, 0.0))
        center, normal, tangent, bitangent = compute_tile_basis(globe, "t0")

        assert abs(np.linalg.norm(normal) - 1.0) < 1e-8
        assert abs(np.linalg.norm(tangent) - 1.0) < 1e-8
        assert abs(np.linalg.norm(bitangent) - 1.0) < 1e-8
        assert abs(float(np.dot(normal, tangent))) < 1e-8

    def test_center_is_mean_of_vertices(self):
        globe = _make_hex_globe_tile("t0", (1.0, 0.0, 0.0))
        center, _, _, _ = compute_tile_basis(globe, "t0")
        face = globe.faces["t0"]
        verts = [globe.vertices[vid] for vid in face.vertex_ids]
        expected = np.mean(
            [[v.x, v.y, v.z] for v in verts], axis=0,
        )
        np.testing.assert_allclose(center, expected, atol=1e-10)

    def test_different_center_positions(self):
        """Tile basis works for tiles at various positions on the unit sphere."""
        for center_3d in [(1, 0, 0), (0, 0, 1), (0.577, 0.577, 0.577)]:
            globe = _make_hex_globe_tile("t0", center_3d, radius=0.08)
            center, normal, tangent, bitangent = compute_tile_basis(globe, "t0")
            # Should always produce orthogonal basis
            assert abs(float(np.dot(tangent, bitangent))) < 1e-8


class TestProjectPointToTileUV:
    def test_center_projects_to_zero(self):
        globe = _make_hex_globe_tile("t0", (1.0, 0.0, 0.0))
        center, _, tangent, bitangent = compute_tile_basis(globe, "t0")
        u, v = project_point_to_tile_uv(center, center, tangent, bitangent)
        assert abs(u) < 1e-10
        assert abs(v) < 1e-10

    def test_tangent_offset(self):
        center = np.array([1.0, 0.0, 0.0])
        tangent = np.array([0.0, 1.0, 0.0])
        bitangent = np.array([0.0, 0.0, 1.0])

        pt = center + 0.5 * tangent
        u, v = project_point_to_tile_uv(pt, center, tangent, bitangent)
        assert abs(u - 0.5) < 1e-10
        assert abs(v) < 1e-10

    def test_bitangent_offset(self):
        center = np.array([1.0, 0.0, 0.0])
        tangent = np.array([0.0, 1.0, 0.0])
        bitangent = np.array([0.0, 0.0, 1.0])

        pt = center + 0.3 * bitangent
        u, v = project_point_to_tile_uv(pt, center, tangent, bitangent)
        assert abs(u) < 1e-10
        assert abs(v - 0.3) < 1e-10


class TestComputeTileUVBounds:
    def test_bounds_symmetric_hex(self):
        globe = _make_hex_globe_tile("t0", (1.0, 0.0, 0.0), radius=0.1)
        center, _, tangent, bitangent = compute_tile_basis(globe, "t0")
        u_min, v_min, u_max, v_max = compute_tile_uv_bounds(
            globe, "t0", center, tangent, bitangent,
        )
        # Non-degenerate bounds
        assert u_max > u_min
        assert v_max > v_min
        # Approximately symmetric around zero (center is the mean)
        assert abs((u_max + u_min) / 2) < 1e-6
        assert abs((v_max + v_min) / 2) < 1e-6


class TestProjectAndNormalize:
    def test_polygon_vertices_in_01(self):
        """All polygon vertices should project to [0,1] after normalisation."""
        globe = _make_hex_globe_tile("t0", (1.0, 0.0, 0.0), radius=0.1)
        center, _, tangent, bitangent = compute_tile_basis(globe, "t0")
        uv_bounds = compute_tile_uv_bounds(globe, "t0", center, tangent, bitangent)

        face = globe.faces["t0"]
        for vid in face.vertex_ids:
            v = globe.vertices[vid]
            pt = np.array([v.x, v.y, v.z])
            u, vv = project_and_normalize(pt, center, tangent, bitangent, uv_bounds)
            assert -1e-8 <= u <= 1.0 + 1e-8, f"u={u} out of range"
            assert -1e-8 <= vv <= 1.0 + 1e-8, f"v={vv} out of range"

    def test_extremes_are_0_and_1(self):
        """At least one vertex maps to u=0, one to u=1, etc."""
        globe = _make_hex_globe_tile("t0", (1.0, 0.0, 0.0), radius=0.1)
        center, _, tangent, bitangent = compute_tile_basis(globe, "t0")
        uv_bounds = compute_tile_uv_bounds(globe, "t0", center, tangent, bitangent)

        face = globe.faces["t0"]
        us, vs = [], []
        for vid in face.vertex_ids:
            v = globe.vertices[vid]
            pt = np.array([v.x, v.y, v.z])
            u, vv = project_and_normalize(pt, center, tangent, bitangent, uv_bounds)
            us.append(u)
            vs.append(vv)
        assert abs(min(us)) < 1e-8
        assert abs(max(us) - 1.0) < 1e-8
        assert abs(min(vs)) < 1e-8
        assert abs(max(vs) - 1.0) < 1e-8

    def test_center_projects_to_mid(self):
        """The tile center should project approximately to (0.5, 0.5)."""
        globe = _make_hex_globe_tile("t0", (1.0, 0.0, 0.0), radius=0.1)
        center, _, tangent, bitangent = compute_tile_basis(globe, "t0")
        uv_bounds = compute_tile_uv_bounds(globe, "t0", center, tangent, bitangent)

        u, v = project_and_normalize(center, center, tangent, bitangent, uv_bounds)
        # For a regular hex the center is equidistant from extremes
        # so normalised center should be (0.5, 0.5) ± some tolerance
        # depending on vertex arrangement.
        assert 0.3 < u < 0.7
        assert 0.3 < v < 0.7


# ═══════════════════════════════════════════════════════════════════
# 20A.2 — UVTransform
# ═══════════════════════════════════════════════════════════════════

class TestUVTransform:
    def test_identity_transform(self):
        xf = UVTransform(A=np.eye(2), t=np.zeros(2))
        u, v = xf.apply(3.0, 4.0)
        assert abs(u - 3.0) < 1e-10
        assert abs(v - 4.0) < 1e-10

    def test_translation_only(self):
        xf = UVTransform(A=np.eye(2), t=np.array([1.0, 2.0]))
        u, v = xf.apply(0.0, 0.0)
        assert abs(u - 1.0) < 1e-10
        assert abs(v - 2.0) < 1e-10

    def test_scale_transform(self):
        xf = UVTransform(A=np.array([[2.0, 0.0], [0.0, 3.0]]), t=np.zeros(2))
        u, v = xf.apply(1.0, 1.0)
        assert abs(u - 2.0) < 1e-10
        assert abs(v - 3.0) < 1e-10

    def test_rotation_90(self):
        # 90-degree rotation: [[0, -1], [1, 0]]
        xf = UVTransform(A=np.array([[0.0, -1.0], [1.0, 0.0]]), t=np.zeros(2))
        u, v = xf.apply(1.0, 0.0)
        assert abs(u) < 1e-10
        assert abs(v - 1.0) < 1e-10

    def test_apply_array(self):
        xf = UVTransform(A=np.array([[2.0, 0.0], [0.0, 3.0]]), t=np.array([1.0, 2.0]))
        pts = np.array([[1.0, 1.0], [0.0, 0.0], [0.5, 0.5]])
        result = xf.apply_array(pts)
        assert result.shape == (3, 2)
        np.testing.assert_allclose(result[0], [3.0, 5.0], atol=1e-10)
        np.testing.assert_allclose(result[1], [1.0, 2.0], atol=1e-10)
        np.testing.assert_allclose(result[2], [2.0, 3.5], atol=1e-10)

    def test_apply_array_matches_apply(self):
        A = np.array([[1.5, -0.3], [0.3, 1.5]])
        t = np.array([0.1, 0.2])
        xf = UVTransform(A=A, t=t)
        pts = np.array([[1.0, 2.0], [-1.0, 3.0], [0.5, -0.5]])
        result = xf.apply_array(pts)
        for i in range(3):
            u, v = xf.apply(pts[i, 0], pts[i, 1])
            np.testing.assert_allclose(result[i], [u, v], atol=1e-10)


class TestComputeDetailToUVTransform:
    def test_hex_transform_maps_center_near_uv_center(self):
        globe = _make_hex_globe_tile("t0", (1.0, 0.0, 0.0), radius=0.1)
        center, normal, tangent, bitangent = compute_tile_basis(globe, "t0")
        uv_bounds = compute_tile_uv_bounds(globe, "t0", center, tangent, bitangent)

        detail = _make_detail_grid_2d(n_sides=6, rings=2)
        xf = compute_detail_to_uv_transform(
            globe, "t0", detail, center, tangent, bitangent, uv_bounds,
        )

        # The detail grid's centroid (near 0,0) should map near the UV centroid
        u, v = xf.apply(0.0, 0.0)
        assert 0.2 < u < 0.8, f"u={u} too far from center"
        assert 0.2 < v < 0.8, f"v={v} too far from center"

    def test_pent_transform_maps_center_near_uv_center(self):
        globe = _make_pent_globe_tile("t0", (0.0, 1.0, 0.0), radius=0.1)
        center, normal, tangent, bitangent = compute_tile_basis(globe, "t0")
        uv_bounds = compute_tile_uv_bounds(globe, "t0", center, tangent, bitangent)

        detail = _make_detail_grid_2d(n_sides=5, rings=2)
        xf = compute_detail_to_uv_transform(
            globe, "t0", detail, center, tangent, bitangent, uv_bounds,
        )

        u, v = xf.apply(0.0, 0.0)
        assert 0.1 < u < 0.9, f"u={u} too far from center"
        assert 0.1 < v < 0.9, f"v={v} too far from center"

    def test_transform_is_similarity(self):
        """The fitted transform should preserve angles (similarity)."""
        globe = _make_hex_globe_tile("t0", (1.0, 0.0, 0.0), radius=0.1)
        center, normal, tangent, bitangent = compute_tile_basis(globe, "t0")
        uv_bounds = compute_tile_uv_bounds(globe, "t0", center, tangent, bitangent)

        detail = _make_detail_grid_2d(n_sides=6, rings=2)
        xf = compute_detail_to_uv_transform(
            globe, "t0", detail, center, tangent, bitangent, uv_bounds,
        )

        # A similarity matrix has the form [[a, -b], [b, a]]
        A = xf.A
        assert abs(A[0, 0] - A[1, 1]) < 1e-6, "A[0,0] != A[1,1]"
        assert abs(A[0, 1] + A[1, 0]) < 1e-6, "A[0,1] != -A[1,0]"

    def test_scale_is_positive(self):
        """Transform should have a positive scale factor."""
        globe = _make_hex_globe_tile("t0", (1.0, 0.0, 0.0), radius=0.1)
        center, normal, tangent, bitangent = compute_tile_basis(globe, "t0")
        uv_bounds = compute_tile_uv_bounds(globe, "t0", center, tangent, bitangent)

        detail = _make_detail_grid_2d(n_sides=6, rings=3)
        xf = compute_detail_to_uv_transform(
            globe, "t0", detail, center, tangent, bitangent, uv_bounds,
        )

        # Scale = sqrt(a² + b²)
        a, b = xf.A[0, 0], xf.A[1, 0]
        scale = math.sqrt(a * a + b * b)
        assert scale > 1e-6, f"Scale too small: {scale}"

    def test_boundary_vertices_map_inside_01(self):
        """Outermost detail grid vertices should map roughly within [0,1]²."""
        globe = _make_hex_globe_tile("t0", (1.0, 0.0, 0.0), radius=0.1)
        center, normal, tangent, bitangent = compute_tile_basis(globe, "t0")
        uv_bounds = compute_tile_uv_bounds(globe, "t0", center, tangent, bitangent)

        detail = _make_detail_grid_2d(n_sides=6, rings=2)
        xf = compute_detail_to_uv_transform(
            globe, "t0", detail, center, tangent, bitangent, uv_bounds,
        )

        # Collect all detail vertices
        all_verts = []
        for v in detail.vertices.values():
            if v.has_position():
                all_verts.append((v.x, v.y))

        # Transform through xf
        uvs = xf.apply_array(np.array(all_verts))
        # At least the interior vertices should land in [0,1] ± margin
        # Boundary vertices might be very close to 0 or 1
        assert uvs[:, 0].min() > -0.3, f"u_min = {uvs[:, 0].min()}"
        assert uvs[:, 0].max() < 1.3, f"u_max = {uvs[:, 0].max()}"
        assert uvs[:, 1].min() > -0.3, f"v_min = {uvs[:, 1].min()}"
        assert uvs[:, 1].max() < 1.3, f"v_max = {uvs[:, 1].max()}"


# ═══════════════════════════════════════════════════════════════════
# 20A.3 — UV-aligned tile rasteriser
# ═══════════════════════════════════════════════════════════════════

class TestRenderTileUVAligned:
    @pytest.fixture()
    def hex_render_args(self):
        """Set up everything needed to render a hex tile with UV alignment."""
        from polygrid.builders import build_pure_hex_grid
        from polygrid.tile_data import FieldDef, TileDataStore, TileSchema

        # Globe tile
        globe = _make_hex_globe_tile("t0", (1.0, 0.0, 0.0), radius=0.1)
        center, normal, tangent, bitangent = compute_tile_basis(globe, "t0")
        uv_bounds = compute_tile_uv_bounds(globe, "t0", center, tangent, bitangent)

        # Detail grid (2D)
        detail = build_pure_hex_grid(2)

        # UV transform
        xf = compute_detail_to_uv_transform(
            globe, "t0", detail, center, tangent, bitangent, uv_bounds,
        )

        # Store with elevation data
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=detail, schema=schema)
        for fid in detail.faces:
            idx = int(fid.replace("f", "")) if fid.startswith("f") else 0
            store.set(fid, "elevation", 0.1 + 0.05 * (idx % 5))

        return detail, store, xf

    def test_renders_rgb_image(self, hex_render_args):
        grid, store, xf = hex_render_args
        img = render_tile_uv_aligned(grid, store, xf, tile_size=64)
        assert img.mode == "RGB"
        assert img.size == (64, 64)

    def test_non_black_pixels_present(self, hex_render_args):
        grid, store, xf = hex_render_args
        img = render_tile_uv_aligned(grid, store, xf, tile_size=64)
        arr = np.array(img)
        # At least some pixels should be non-black
        assert arr.max() > 0, "Image is completely black"

    def test_no_magenta_sentinel_in_output(self, hex_render_args):
        """The sentinel colour (255, 0, 255) should not dominate the output."""
        grid, store, xf = hex_render_args
        img = render_tile_uv_aligned(
            grid, store, xf, tile_size=64, gutter_pixels=2,
        )
        arr = np.array(img)
        sentinel_mask = (
            (arr[:, :, 0] == 255) &
            (arr[:, :, 1] == 0) &
            (arr[:, :, 2] == 255)
        )
        sentinel_frac = sentinel_mask.sum() / (arr.shape[0] * arr.shape[1])
        # Very few (if any) sentinel pixels should remain in the final image.
        # The IDW fill should have covered background pixels.
        assert sentinel_frac < 0.05, (
            f"Sentinel pixels are {sentinel_frac:.1%} of image"
        )

    def test_tile_size_respected(self, hex_render_args):
        grid, store, xf = hex_render_args
        for ts in [32, 128]:
            img = render_tile_uv_aligned(grid, store, xf, tile_size=ts)
            assert img.size == (ts, ts)

    def test_gutter_zero_still_works(self, hex_render_args):
        grid, store, xf = hex_render_args
        img = render_tile_uv_aligned(
            grid, store, xf, tile_size=64, gutter_pixels=0,
        )
        assert img.size == (64, 64)

    def test_pentagon_tile(self):
        """Render works for pentagon tiles too."""
        from polygrid.builders import build_pentagon_centered_grid
        from polygrid.tile_data import FieldDef, TileDataStore, TileSchema

        globe = _make_pent_globe_tile("t0", (0.0, 1.0, 0.0), radius=0.1)
        center, normal, tangent, bitangent = compute_tile_basis(globe, "t0")
        uv_bounds = compute_tile_uv_bounds(globe, "t0", center, tangent, bitangent)

        detail = build_pentagon_centered_grid(2)
        xf = compute_detail_to_uv_transform(
            globe, "t0", detail, center, tangent, bitangent, uv_bounds,
        )

        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=detail, schema=schema)
        for fid in detail.faces:
            store.set(fid, "elevation", 0.2)

        img = render_tile_uv_aligned(detail, store, xf, tile_size=64)
        assert img.mode == "RGB"
        assert img.size == (64, 64)
        # Should have some non-black content
        assert np.array(img).max() > 0


# ═══════════════════════════════════════════════════════════════════
# Integration — matches models library UV computation
# ═══════════════════════════════════════════════════════════════════

class TestUVMatchesModelsLibrary:
    """Verify our projection replicates the models library's
    GoldbergTile.uv_vertices computation."""

    def test_projection_matches_models_derive_basis(self):
        """Tangent is computed the same way as models' derive_basis_from_vertices."""
        globe = _make_hex_globe_tile("t0", (1.0, 0.0, 0.0), radius=0.1)
        center, normal, tangent, bitangent = compute_tile_basis(globe, "t0")

        face = globe.faces["t0"]
        verts_3d = []
        for vid in face.vertex_ids:
            v = globe.vertices[vid]
            verts_3d.append(np.array([v.x, v.y, v.z], dtype=np.float64))

        # Reproduce models library logic:
        # normal = normalize(cross(edge0, edge1))
        a = verts_3d[1] - verts_3d[0]
        b = verts_3d[2] - verts_3d[0]
        expected_normal = np.cross(a, b)
        expected_normal /= np.linalg.norm(expected_normal)

        # tangent = normalize(edge0 - dot(edge0, normal)*normal)
        edge = verts_3d[1] - verts_3d[0]
        edge_proj = edge - expected_normal * np.dot(edge, expected_normal)
        expected_tangent = edge_proj / np.linalg.norm(edge_proj)

        expected_bitangent = np.cross(expected_normal, expected_tangent)
        expected_bitangent /= np.linalg.norm(expected_bitangent)

        # Our implementation should match
        np.testing.assert_allclose(normal, expected_normal, atol=1e-8)
        np.testing.assert_allclose(tangent, expected_tangent, atol=1e-8)
        np.testing.assert_allclose(bitangent, expected_bitangent, atol=1e-8)

    def test_normalized_uvs_match_bounding_box_normalization(self):
        """project_and_normalize should replicate normalize_uvs behaviour."""
        globe = _make_hex_globe_tile("t0", (1.0, 0.0, 0.0), radius=0.1)
        center, _, tangent, bitangent = compute_tile_basis(globe, "t0")
        uv_bounds = compute_tile_uv_bounds(globe, "t0", center, tangent, bitangent)

        face = globe.faces["t0"]
        uvs = []
        for vid in face.vertex_ids:
            v = globe.vertices[vid]
            pt = np.array([v.x, v.y, v.z])
            u, vv = project_and_normalize(pt, center, tangent, bitangent, uv_bounds)
            uvs.append((u, vv))

        # Also compute via raw projection + manual bbox normalization
        raw_uvs = []
        for vid in face.vertex_ids:
            v = globe.vertices[vid]
            pt = np.array([v.x, v.y, v.z])
            u, vv = project_point_to_tile_uv(pt, center, tangent, bitangent)
            raw_uvs.append((u, vv))

        raw_arr = np.array(raw_uvs)
        u_min, v_min = raw_arr.min(axis=0)
        u_max, v_max = raw_arr.max(axis=0)
        manual_norm = (raw_arr - [u_min, v_min]) / [u_max - u_min, v_max - v_min]

        np.testing.assert_allclose(np.array(uvs), manual_norm, atol=1e-10)
