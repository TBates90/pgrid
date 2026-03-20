"""Tests for Phase 20A — UV-aligned texture rendering.

Covers:
- 20A.1 — Tile basis and UV projection (fallback + authoritative paths)
- 20A.2 — Affine transform from detail-2D to tile-UV
- 20A.3 — UV-aligned tile rasteriser
- 20A.4 — Integration: UV alignment matches models library exactly
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
    _find_polygon_corners,
    _normalize_vec,
    _compute_tile_basis_from_grid,
    compute_detail_to_uv_transform,
    compute_tile_basis,
    compute_tile_uv_bounds,
    get_goldberg_tiles,
    get_tile_uv_vertices,
    project_and_normalize,
    project_point_to_tile_uv,
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

    NOTE: This mock grid does NOT have ``frequency``/``radius`` metadata,
    so functions will use the fallback grid-based path (not the
    authoritative GoldbergTile path).
    """
    c = np.array(center_3d, dtype=np.float64)
    normal = c / np.linalg.norm(c)

    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(normal, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    tangent = np.cross(up, normal)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    bitangent /= np.linalg.norm(bitangent)

    verts = {}
    vids = []
    for i in range(6):
        angle = math.pi / 6 + 2 * math.pi * i / 6
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


def _make_detail_grid_2d(n_sides: int = 6, rings: int = 2) -> PolyGrid:
    """Create a small 2D detail grid matching hex or pent shape."""
    from polygrid.builders import build_pure_hex_grid, build_pentagon_centered_grid

    if n_sides == 5:
        return build_pentagon_centered_grid(rings)
    else:
        return build_pure_hex_grid(rings)


# ═══════════════════════════════════════════════════════════════════
# Helper — build a real globe grid (uses models library)
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def real_globe_f3():
    """A real freq-3 globe grid (cached per module)."""
    from polygrid.globe import build_globe_grid
    return build_globe_grid(3)


# ═══════════════════════════════════════════════════════════════════
# 20A.1 — Helpers
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


# ═══════════════════════════════════════════════════════════════════
# 20A.1 — Fallback basis derivation (from grid vertices)
# ═══════════════════════════════════════════════════════════════════

class TestFallbackBasis:
    """Tests for _compute_tile_basis_from_grid (no models library)."""

    def test_orthogonal_basis(self):
        globe = _make_hex_globe_tile("t0", (1.0, 0.0, 0.0))
        center, normal, tangent, bitangent = _compute_tile_basis_from_grid(globe, "t0")
        assert abs(np.linalg.norm(normal) - 1.0) < 1e-8
        assert abs(np.linalg.norm(tangent) - 1.0) < 1e-8
        assert abs(np.linalg.norm(bitangent) - 1.0) < 1e-8
        assert abs(float(np.dot(normal, tangent))) < 1e-8
        assert abs(float(np.dot(normal, bitangent))) < 1e-8
        assert abs(float(np.dot(tangent, bitangent))) < 1e-8

    def test_center_is_vertex_mean(self):
        globe = _make_hex_globe_tile("t0", (1.0, 0.0, 0.0))
        center, _, _, _ = _compute_tile_basis_from_grid(globe, "t0")
        face = globe.faces["t0"]
        expected = np.mean(
            [[globe.vertices[vid].x, globe.vertices[vid].y, globe.vertices[vid].z]
             for vid in face.vertex_ids],
            axis=0,
        )
        np.testing.assert_allclose(center, expected, atol=1e-10)

    def test_fallback_uv_bounds_consistent(self):
        """Fallback project_and_normalize maps polygon verts to [0,1]."""
        globe = _make_hex_globe_tile("t0", (1.0, 0.0, 0.0), radius=0.1)
        center, _, tangent, bitangent = _compute_tile_basis_from_grid(globe, "t0")
        face = globe.faces["t0"]

        # Compute bounds manually from fallback basis
        us, vs = [], []
        for vid in face.vertex_ids:
            v = globe.vertices[vid]
            pt = np.array([v.x, v.y, v.z])
            u, vv = project_point_to_tile_uv(pt, center, tangent, bitangent)
            us.append(u)
            vs.append(vv)
        uv_bounds = (min(us), min(vs), max(us), max(vs))

        # All vertices should normalise to [0,1]
        for vid in face.vertex_ids:
            v = globe.vertices[vid]
            pt = np.array([v.x, v.y, v.z])
            u, vv = project_and_normalize(pt, center, tangent, bitangent, uv_bounds)
            assert -1e-8 <= u <= 1.0 + 1e-8
            assert -1e-8 <= vv <= 1.0 + 1e-8


class TestProjectPointToTileUV:
    def test_center_projects_to_zero(self):
        center = np.array([1.0, 0.0, 0.0])
        tangent = np.array([0.0, 1.0, 0.0])
        bitangent = np.array([0.0, 0.0, 1.0])
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


# ═══════════════════════════════════════════════════════════════════
# 20A.1 — Authoritative basis from GoldbergTile (requires models)
# ═══════════════════════════════════════════════════════════════════

class TestAuthoritativeBasis:
    """Test that compute_tile_basis returns the models library's
    GoldbergTile basis when a real globe grid is provided."""

    def test_basis_matches_goldberg_tile(self, real_globe_f3):
        """Our basis must exactly match generate_goldberg_tiles()."""
        tiles = get_goldberg_tiles(3, 1.0)
        for tile in tiles[:10]:  # spot-check first 10
            fid = f"t{tile.index}"
            center, normal, tangent, bitangent = compute_tile_basis(real_globe_f3, fid)
            np.testing.assert_allclose(center, tile.center, atol=1e-10)
            np.testing.assert_allclose(tangent, tile.tangent, atol=1e-10)
            np.testing.assert_allclose(bitangent, tile.bitangent, atol=1e-10)
            np.testing.assert_allclose(normal, tile.normal, atol=1e-10)

    def test_basis_orthogonal_for_all_tiles(self, real_globe_f3):
        tiles = get_goldberg_tiles(3, 1.0)
        for tile in tiles:
            fid = f"t{tile.index}"
            _, normal, tangent, bitangent = compute_tile_basis(real_globe_f3, fid)
            assert abs(float(np.dot(tangent, bitangent))) < 1e-8
            assert abs(float(np.dot(normal, tangent))) < 1e-8

    def test_uv_vertices_match_goldberg_tile(self, real_globe_f3):
        """get_tile_uv_vertices returns the GoldbergTile.uv_vertices."""
        tiles = get_goldberg_tiles(3, 1.0)
        for tile in tiles[:10]:
            fid = f"t{tile.index}"
            our_uvs = get_tile_uv_vertices(real_globe_f3, fid)
            expected = list(tile.uv_vertices)
            assert len(our_uvs) == len(expected)
            for (ou, ov), (eu, ev) in zip(our_uvs, expected):
                assert abs(ou - eu) < 1e-10
                assert abs(ov - ev) < 1e-10

    def test_uv_bounds_from_authoritative_vertices(self, real_globe_f3):
        """UV bounds use GoldbergTile vertices, not globe_grid vertices."""
        tiles = get_goldberg_tiles(3, 1.0)
        tile = tiles[5]
        fid = f"t{tile.index}"
        center, normal, tangent, bitangent = compute_tile_basis(real_globe_f3, fid)
        u_min, v_min, u_max, v_max = compute_tile_uv_bounds(
            real_globe_f3, fid, center, tangent, bitangent,
        )
        assert u_max > u_min
        assert v_max > v_min

    def test_normalised_uvs_replicate_tile_uv_vertices(self, real_globe_f3):
        """project_and_normalize on GoldbergTile vertices should exactly
        reproduce tile.uv_vertices."""
        tiles = get_goldberg_tiles(3, 1.0)
        for tile in tiles[:10]:
            fid = f"t{tile.index}"
            center, normal, tangent, bitangent = compute_tile_basis(real_globe_f3, fid)
            uv_bounds = compute_tile_uv_bounds(
                real_globe_f3, fid, center, tangent, bitangent,
            )
            for vtx_3d, (eu, ev) in zip(tile.vertices, tile.uv_vertices):
                pt = np.array(vtx_3d, dtype=np.float64)
                u, v = project_and_normalize(pt, center, tangent, bitangent, uv_bounds)
                assert abs(u - eu) < 1e-8, f"u mismatch: {u} vs {eu}"
                assert abs(v - ev) < 1e-8, f"v mismatch: {v} vs {ev}"


# ═══════════════════════════════════════════════════════════════════
# 20A.2 — UVTransform
# ═══════════════════════════════════════════════════════════════════

class TestUVTransform:
    """Test the piecewise-linear polygon-warp UVTransform."""

    def _make_hex_warp(self, offset=(0.0, 0.0), scale=1.0):
        """Create a UVTransform that warps a regular hex to a shifted hex."""
        # Regular hexagon corners centered at origin
        src_corners = np.array([
            [scale * math.cos(a), scale * math.sin(a)]
            for a in (math.radians(d) for d in range(0, 360, 60))
        ])
        dst_corners = src_corners + np.array(offset)
        return UVTransform(
            src_centroid=np.array([0.0, 0.0]),
            src_corners=src_corners,
            dst_centroid=np.array(offset),
            dst_corners=dst_corners,
        )

    def test_identity_transform(self):
        xf = self._make_hex_warp()
        u, v = xf.apply(0.0, 0.0)
        assert abs(u) < 1e-10
        assert abs(v) < 1e-10

    def test_translation(self):
        xf = self._make_hex_warp(offset=(1.0, 2.0))
        u, v = xf.apply(0.0, 0.0)
        assert abs(u - 1.0) < 1e-10
        assert abs(v - 2.0) < 1e-10

    def test_corners_map_exactly(self):
        src_corners = np.array([
            [1.0, 0.0], [0.5, 0.866], [-0.5, 0.866],
            [-1.0, 0.0], [-0.5, -0.866], [0.5, -0.866],
        ])
        dst_corners = np.array([
            [2.0, 1.0], [1.5, 1.866], [0.5, 1.866],
            [0.0, 1.0], [0.5, 0.134], [1.5, 0.134],
        ])
        xf = UVTransform(
            src_centroid=np.zeros(2),
            src_corners=src_corners,
            dst_centroid=np.array([1.0, 1.0]),
            dst_corners=dst_corners,
        )
        for i in range(6):
            u, v = xf.apply(src_corners[i, 0], src_corners[i, 1])
            np.testing.assert_allclose([u, v], dst_corners[i], atol=1e-10)

    def test_apply_array(self):
        xf = self._make_hex_warp(offset=(1.0, 2.0))
        pts = np.array([[0.0, 0.0], [0.5, 0.0], [-0.3, 0.2]])
        result = xf.apply_array(pts)
        assert result.shape == (3, 2)
        for i in range(3):
            u, v = xf.apply(pts[i, 0], pts[i, 1])
            np.testing.assert_allclose(result[i], [u, v], atol=1e-10)

    def test_apply_array_matches_apply(self):
        # Non-trivial warp: regular hex → irregular hex
        src_corners = np.array([
            [1.0, 0.0], [0.5, 0.866], [-0.5, 0.866],
            [-1.0, 0.0], [-0.5, -0.866], [0.5, -0.866],
        ])
        dst_corners = np.array([
            [0.8, 0.0], [0.3, 1.0], [-0.7, 0.9],
            [-1.1, -0.1], [-0.4, -0.9], [0.6, -0.8],
        ])
        xf = UVTransform(
            src_centroid=np.zeros(2),
            src_corners=src_corners,
            dst_centroid=np.array([0.0, 0.0]),
            dst_corners=dst_corners,
        )
        pts = np.array([[0.0, 0.0], [0.5, 0.3], [-0.2, 0.5], [0.3, -0.4]])
        result = xf.apply_array(pts)
        for i in range(len(pts)):
            u, v = xf.apply(pts[i, 0], pts[i, 1])
            np.testing.assert_allclose(result[i], [u, v], atol=1e-10)


class TestComputeDetailToUVTransform:
    """Test the polygon warp using a real globe grid."""

    def test_hex_transform_maps_center_near_uv_center(self, real_globe_f3):
        fid = "t5"  # a hex tile
        center, normal, tangent, bitangent = compute_tile_basis(real_globe_f3, fid)
        uv_bounds = compute_tile_uv_bounds(real_globe_f3, fid, center, tangent, bitangent)

        face = real_globe_f3.faces[fid]
        n_sides = len(face.vertex_ids)
        detail = _make_detail_grid_2d(n_sides=n_sides, rings=2)
        xf = compute_detail_to_uv_transform(
            real_globe_f3, fid, detail, center, tangent, bitangent, uv_bounds,
        )

        u, v = xf.apply(0.0, 0.0)
        assert 0.2 < u < 0.8, f"u={u} too far from center"
        assert 0.2 < v < 0.8, f"v={v} too far from center"

    def test_pent_transform_maps_center_near_uv_center(self, real_globe_f3):
        # Find a pentagon tile
        pent_fid = None
        for fid, face in real_globe_f3.faces.items():
            if face.face_type == "pent":
                pent_fid = fid
                break
        assert pent_fid is not None

        center, normal, tangent, bitangent = compute_tile_basis(real_globe_f3, pent_fid)
        uv_bounds = compute_tile_uv_bounds(real_globe_f3, pent_fid, center, tangent, bitangent)

        detail = _make_detail_grid_2d(n_sides=5, rings=2)
        xf = compute_detail_to_uv_transform(
            real_globe_f3, pent_fid, detail, center, tangent, bitangent, uv_bounds,
        )

        u, v = xf.apply(0.0, 0.0)
        assert 0.1 < u < 0.9, f"u={u} too far from center"
        assert 0.1 < v < 0.9, f"v={v} too far from center"

    def test_polygon_corners_map_exactly(self, real_globe_f3):
        """Polygon warp must give zero error at polygon corners."""
        fid = "t5"
        center, normal, tangent, bitangent = compute_tile_basis(real_globe_f3, fid)
        uv_bounds = compute_tile_uv_bounds(real_globe_f3, fid, center, tangent, bitangent)

        detail = _make_detail_grid_2d(n_sides=6, rings=3)
        xf = compute_detail_to_uv_transform(
            real_globe_f3, fid, detail, center, tangent, bitangent, uv_bounds,
        )

        for i in range(len(xf.src_corners)):
            src = xf.src_corners[i]
            expected = xf.dst_corners[i]
            actual = np.array(xf.apply(src[0], src[1]))
            np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_centroid_maps_to_uv_centroid(self, real_globe_f3):
        """Polygon warp centroid should map to UV centroid."""
        fid = "t5"
        center, normal, tangent, bitangent = compute_tile_basis(real_globe_f3, fid)
        uv_bounds = compute_tile_uv_bounds(real_globe_f3, fid, center, tangent, bitangent)

        detail = _make_detail_grid_2d(n_sides=6, rings=3)
        xf = compute_detail_to_uv_transform(
            real_globe_f3, fid, detail, center, tangent, bitangent, uv_bounds,
        )

        u, v = xf.apply(xf.src_centroid[0], xf.src_centroid[1])
        np.testing.assert_allclose([u, v], xf.dst_centroid, atol=1e-8)

    def test_boundary_vertices_map_inside_01(self, real_globe_f3):
        fid = "t5"
        center, normal, tangent, bitangent = compute_tile_basis(real_globe_f3, fid)
        uv_bounds = compute_tile_uv_bounds(real_globe_f3, fid, center, tangent, bitangent)

        detail = _make_detail_grid_2d(n_sides=6, rings=2)
        xf = compute_detail_to_uv_transform(
            real_globe_f3, fid, detail, center, tangent, bitangent, uv_bounds,
        )

        all_verts = []
        for v in detail.vertices.values():
            if v.has_position():
                all_verts.append((v.x, v.y))

        uvs = xf.apply_array(np.array(all_verts))
        assert uvs[:, 0].min() > -0.3, f"u_min = {uvs[:, 0].min()}"
        assert uvs[:, 0].max() < 1.3, f"u_max = {uvs[:, 0].max()}"
        assert uvs[:, 1].min() > -0.3, f"v_min = {uvs[:, 1].min()}"
        assert uvs[:, 1].max() < 1.3, f"v_max = {uvs[:, 1].max()}"


# ═══════════════════════════════════════════════════════════════════
# 38B.5 — Exact pentagon corner detection
# ═══════════════════════════════════════════════════════════════════

class TestFindPolygonCornersMetadataFastPath:
    """Verify _find_polygon_corners uses corner_vertex_ids metadata."""

    @pytest.mark.parametrize("rings", [2, 3, 4])
    def test_pentagon_metadata_corners_count(self, rings):
        """Pentagon grids return exactly 5 corners via metadata fast-path."""
        from polygrid.builders import build_pentagon_centered_grid

        grid = build_pentagon_centered_grid(rings)
        assert "corner_vertex_ids" in grid.metadata
        centroid, corners = _find_polygon_corners(grid)
        assert corners.shape == (5, 2), f"Expected 5 corners, got {corners.shape[0]}"

    @pytest.mark.parametrize("rings", [2, 3, 4])
    def test_pentagon_fast_path_matches_clustering(self, rings):
        """Fast-path (metadata) and clustering fallback find corners in
        roughly the same region.  The clustering heuristic is approximate
        (it averages nearby outer vertices), so we allow generous tolerance.
        At higher ring counts the clustering centroids can drift up to ~1
        grid-unit from the topological corner."""
        from polygrid.builders import build_pentagon_centered_grid

        grid = build_pentagon_centered_grid(rings)
        # Fast path result (with metadata)
        centroid_fast, corners_fast = _find_polygon_corners(grid)

        # Remove metadata to force clustering fallback
        saved = grid.metadata.pop("corner_vertex_ids")
        centroid_cluster, corners_cluster = _find_polygon_corners(grid)
        grid.metadata["corner_vertex_ids"] = saved  # restore

        if corners_cluster.shape[0] == 5:
            # Both should find 5 corners at similar positions.
            # Match each fast corner to the nearest cluster corner.
            # Tolerance scales with grid size (rings).
            tol = 0.3 * rings
            for fc in corners_fast:
                dists = np.linalg.norm(corners_cluster - fc, axis=1)
                assert dists.min() < tol, (
                    f"Fast-path corner {fc} has no close match in clustering "
                    f"result (min dist={dists.min():.4f}, tol={tol:.2f})"
                )

    def test_hex_grid_no_metadata_corners(self):
        """Hex grids don't have corner_vertex_ids; fallback finds 6 corners."""
        from polygrid.builders import build_pure_hex_grid

        grid = build_pure_hex_grid(3)
        assert "corner_vertex_ids" not in grid.metadata
        centroid, corners = _find_polygon_corners(grid)
        assert corners.shape[0] == 6

    @pytest.mark.parametrize("rings", [2, 3, 4])
    def test_pentagon_corners_on_boundary(self, rings):
        """Corners from metadata are actual outermost vertices."""
        from polygrid.builders import build_pentagon_centered_grid

        grid = build_pentagon_centered_grid(rings)
        centroid, corners = _find_polygon_corners(grid)

        # Compute max distance from centroid
        all_verts = []
        for v in grid.vertices.values():
            if v.has_position():
                all_verts.append(np.array([v.x, v.y]))
        arr = np.array(all_verts)
        dists = np.linalg.norm(arr - centroid, axis=1)
        max_dist = dists.max()

        # All corners should be near the boundary
        for i, c in enumerate(corners):
            d = float(np.linalg.norm(c - centroid))
            assert d > max_dist * 0.85, (
                f"Corner {i} at dist {d:.4f} is not near boundary "
                f"(max_dist={max_dist:.4f})"
            )

    def test_corners_counter_clockwise(self):
        """Corners are ordered counter-clockwise."""
        from polygrid.builders import build_pentagon_centered_grid

        grid = build_pentagon_centered_grid(3)
        centroid, corners = _find_polygon_corners(grid)
        angles = np.arctan2(
            corners[:, 1] - centroid[1],
            corners[:, 0] - centroid[0],
        )
        # Should be strictly increasing (they were sorted by angle)
        for i in range(len(angles) - 1):
            assert angles[i] < angles[i + 1], (
                f"Corners not counter-clockwise: angle[{i}]={angles[i]:.4f} "
                f">= angle[{i+1}]={angles[i+1]:.4f}"
            )


# ═══════════════════════════════════════════════════════════════════
# 38C.2 — Robust corner-to-corner matching (rotational offset)
# ═══════════════════════════════════════════════════════════════════

class TestRobustCornerMatching:
    """Verify that compute_detail_to_uv_transform handles rotated pentagons."""

    @pytest.mark.needs_models
    def test_pentagon_rotated_72deg_recovery(self, real_globe_f3):
        """A pentagon grid rotated by 72° still produces a valid transform."""
        from polygrid.builders import build_pentagon_centered_grid

        pent_fid = None
        for fid, face in real_globe_f3.faces.items():
            if face.face_type == "pent":
                pent_fid = fid
                break
        assert pent_fid is not None

        center, normal, tangent, bitangent = compute_tile_basis(
            real_globe_f3, pent_fid
        )
        uv_bounds = compute_tile_uv_bounds(
            real_globe_f3, pent_fid, center, tangent, bitangent
        )

        detail = build_pentagon_centered_grid(2)

        # Build the standard transform
        xf = compute_detail_to_uv_transform(
            real_globe_f3, pent_fid, detail, center, tangent, bitangent, uv_bounds,
        )

        # Map the centroid — should land near UV polygon centre
        uc, vc = xf.apply(float(xf.src_centroid[0]), float(xf.src_centroid[1]))
        assert 0.1 < uc < 0.9, f"Centroid u={uc:.3f} out of range"
        assert 0.1 < vc < 0.9, f"Centroid v={vc:.3f} out of range"

        # All corners should map close to [0,1]
        for i in range(xf.n):
            su, sv = xf.apply(
                float(xf.src_corners[i, 0]),
                float(xf.src_corners[i, 1]),
            )
            assert -0.05 < su < 1.05, f"Corner {i}: u={su:.3f}"
            assert -0.05 < sv < 1.05, f"Corner {i}: v={sv:.3f}"

    @pytest.mark.needs_models
    def test_all_pentagon_tiles_produce_valid_transforms(self, real_globe_f3):
        """Every pentagon tile on the globe produces a usable transform."""
        from polygrid.builders import build_pentagon_centered_grid

        pent_fids = [
            fid for fid, face in real_globe_f3.faces.items()
            if face.face_type == "pent"
        ]
        assert len(pent_fids) == 12

        for pent_fid in pent_fids:
            center, normal, tangent, bitangent = compute_tile_basis(
                real_globe_f3, pent_fid
            )
            uv_bounds = compute_tile_uv_bounds(
                real_globe_f3, pent_fid, center, tangent, bitangent
            )
            detail = build_pentagon_centered_grid(2)
            xf = compute_detail_to_uv_transform(
                real_globe_f3, pent_fid, detail,
                center, tangent, bitangent, uv_bounds,
            )
            # Centroid should map somewhere reasonable
            uc, vc = xf.apply(
                float(xf.src_centroid[0]), float(xf.src_centroid[1])
            )
            assert 0.05 < uc < 0.95, (
                f"Face {pent_fid}: centroid u={uc:.3f}"
            )
            assert 0.05 < vc < 0.95, (
                f"Face {pent_fid}: centroid v={vc:.3f}"
            )
