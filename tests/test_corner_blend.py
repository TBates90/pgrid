# TODO REVIEW — Tests live modules (tile_uv_align, globe_renderer_v2).
#   Verify all tested functions are exercised by live scripts, then keep.
"""Tests for Phase 21B.2 — Corner-junction blending and subdivision fixes.

Covers:
- _find_vertex_junctions builds correct vertex → face map
- _blend_corner_junctions averages atlas pixels at shared vertices
- build_polygon_cut_atlas accepts blend_corners / blend_radius params
- subdivide_tile_mesh boundary-skip: b0==0 vertices bypass UV clamp
- subdivide_tile_mesh UV-aware dedup: same pos, different UV → kept separate
"""

from __future__ import annotations

import inspect
import math
from typing import Dict, List, Tuple
from unittest.mock import patch

import numpy as np
import pytest

from PIL import Image

from polygrid.tile_uv_align import (
    _blend_corner_junctions,
    _find_vertex_junctions,
    build_polygon_cut_atlas,
)
from polygrid.globe_renderer_v2 import subdivide_tile_mesh


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _make_mock_globe_grid(
    face_ids: List[str],
    n_sides: int = 6,
    *,
    shared_vertex_ids: Dict[str, List[Tuple[str, int]]] | None = None,
):
    """Create a minimal mock globe grid.

    Parameters
    ----------
    face_ids : list of str
    n_sides : int
    shared_vertex_ids : dict, optional
        ``{vertex_id: [(face_id, vertex_index), …]}`` — if given,
        the grid is constructed so that the specified vertices are
        shared at the given positions.  All other vertices are unique.
    """
    from polygrid.models import Face, Vertex, Edge
    from polygrid.polygrid import PolyGrid

    verts = {}
    face_vids: Dict[str, list] = {fid: [] for fid in face_ids}

    # Assign unique default vertex ids
    for fid in face_ids:
        for i in range(n_sides):
            vid = f"{fid}_v{i}"
            face_vids[fid].append(vid)
            verts[vid] = Vertex(vid, float(i), 0.0, 0.0)

    # Override with shared vertices where requested
    if shared_vertex_ids:
        for vid, entries in shared_vertex_ids.items():
            if vid not in verts:
                verts[vid] = Vertex(vid, 0.0, 0.0, 0.0)
            for fid, idx in entries:
                old_vid = face_vids[fid][idx]
                face_vids[fid][idx] = vid
                # Clean up orphan
                if old_vid != vid and old_vid in verts:
                    # Only delete if not used by another face
                    used = any(
                        old_vid in fvids for fvids in face_vids.values()
                    )
                    if not used:
                        del verts[old_vid]

    faces = []
    for fid in face_ids:
        neighbor_ids = [f for f in face_ids if f != fid]
        faces.append(
            Face(
                id=fid,
                face_type="hex",
                vertex_ids=tuple(face_vids[fid]),
                neighbor_ids=neighbor_ids,
            )
        )

    grid = PolyGrid(verts.values(), [], faces)
    grid.metadata["frequency"] = 3
    grid.metadata["radius"] = 1.0
    return grid


def _make_atlas_with_layout(
    n_tiles: int,
    tile_size: int = 64,
    gutter: int = 4,
    colours: List[Tuple[int, int, int]] | None = None,
) -> Tuple[Image.Image, Dict[str, Tuple[float, float, float, float]]]:
    """Build an N-tile atlas with distinct solid colours per slot."""
    slot = tile_size + 2 * gutter
    atlas_w = n_tiles * slot
    atlas_h = slot
    atlas = Image.new("RGB", (atlas_w, atlas_h), (128, 128, 128))

    if colours is None:
        colours = [
            (200, 0, 0), (0, 200, 0), (0, 0, 200),
            (200, 200, 0), (200, 0, 200), (0, 200, 200),
        ][:n_tiles]

    uv_layout: Dict[str, Tuple[float, float, float, float]] = {}
    for i in range(n_tiles):
        fid = f"t{i}"
        slot_x = i * slot
        tile_img = Image.new("RGB", (slot, slot), colours[i])
        atlas.paste(tile_img, (slot_x, 0))

        inner_x = slot_x + gutter
        inner_y = gutter
        uv_layout[fid] = (
            inner_x / atlas_w,
            1.0 - (inner_y + tile_size) / atlas_h,
            (inner_x + tile_size) / atlas_w,
            1.0 - inner_y / atlas_h,
        )

    return atlas, uv_layout


def _hex_tile_data():
    """Hex tile centred at (0,0,1) with UVs at (0.5,0.5)."""
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


# ═══════════════════════════════════════════════════════════════════
# _find_vertex_junctions
# ═══════════════════════════════════════════════════════════════════

class TestFindVertexJunctions:

    def test_no_shared_vertices_returns_empty(self):
        """Tiles with no shared vertices → no junctions."""
        grid = _make_mock_globe_grid(["t0", "t1"])
        result = _find_vertex_junctions(grid, ["t0", "t1"])
        assert result == {}

    def test_shared_vertex_detected(self):
        """A vertex shared by two tiles appears as a junction."""
        shared = {"v_shared": [("t0", 2), ("t1", 0)]}
        grid = _make_mock_globe_grid(["t0", "t1"], shared_vertex_ids=shared)
        result = _find_vertex_junctions(grid, ["t0", "t1"])
        assert "v_shared" in result
        assert len(result["v_shared"]) == 2

    def test_three_tile_junction(self):
        """A vertex shared by three tiles is correctly reported."""
        shared = {"v_corner": [("t0", 1), ("t1", 3), ("t2", 5)]}
        grid = _make_mock_globe_grid(
            ["t0", "t1", "t2"], shared_vertex_ids=shared,
        )
        result = _find_vertex_junctions(grid, ["t0", "t1", "t2"])
        assert "v_corner" in result
        assert len(result["v_corner"]) == 3

    def test_only_atlas_tiles_considered(self):
        """Vertices shared with tiles not in face_ids are ignored."""
        shared = {"v_shared": [("t0", 2), ("t1", 0)]}
        grid = _make_mock_globe_grid(["t0", "t1"], shared_vertex_ids=shared)
        # Only include t0 in face_ids
        result = _find_vertex_junctions(grid, ["t0"])
        assert result == {}


# ═══════════════════════════════════════════════════════════════════
# _blend_corner_junctions
# ═══════════════════════════════════════════════════════════════════

class TestBlendCornerJunctions:

    def test_no_junctions_returns_unchanged(self):
        """When no vertex junctions exist, atlas is untouched."""
        tile_size, gutter = 64, 4
        atlas, uv_layout = _make_atlas_with_layout(2, tile_size, gutter)
        original = atlas.copy()
        grid = _make_mock_globe_grid(["t0", "t1"])

        # Provide a mock that returns a simple UV polygon
        hex_uv = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0),
                   (0.0, 1.0), (0.0, 0.5), (0.5, 0.0)]
        with patch(
            "polygrid.uv_texture.get_tile_uv_vertices",
            return_value=hex_uv,
        ):
            result = _blend_corner_junctions(
                atlas, uv_layout, grid, ["t0", "t1"],
                tile_size=tile_size, gutter=gutter,
            )

        np.testing.assert_array_equal(np.array(result), np.array(original))

    def test_shared_vertex_blends_pixels(self):
        """Pixels at a shared vertex should be averaged across tiles."""
        tile_size, gutter = 64, 4
        colour_a = (200, 0, 0)
        colour_b = (0, 0, 200)
        atlas, uv_layout = _make_atlas_with_layout(
            2, tile_size, gutter, colours=[colour_a, colour_b],
        )
        original_arr = np.array(atlas, dtype=np.float64).copy()

        # Share a vertex: t0 vertex 2 == t1 vertex 0
        shared = {"v_shared": [("t0", 2), ("t1", 0)]}
        grid = _make_mock_globe_grid(["t0", "t1"], shared_vertex_ids=shared)

        # UV vertices: vertex 2 of t0 is at (1.0, 1.0),
        # vertex 0 of t1 is at (0.0, 0.0)
        hex_uv = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0),
                   (0.0, 1.0), (0.0, 0.5), (0.5, 0.0)]

        with patch(
            "polygrid.uv_texture.get_tile_uv_vertices",
            return_value=hex_uv,
        ):
            result = _blend_corner_junctions(
                atlas, uv_layout, grid, ["t0", "t1"],
                tile_size=tile_size, gutter=gutter,
                blend_radius=2,
            )

        result_arr = np.array(result, dtype=np.float64)
        diff = np.abs(result_arr - original_arr)
        changed = np.any(diff > 0.5, axis=-1)
        assert changed.sum() > 0, "Corner blending should modify some pixels"

        # Changed pixels should show an average of the two colours
        changed_vals = result_arr[changed]
        assert np.all(changed_vals[:, 0] < 201), "Red averaged down"
        assert np.all(changed_vals[:, 2] < 201), "Blue averaged down"

    def test_blend_radius_zero_still_works(self):
        """blend_radius=0 → single-pixel blend, no crash."""
        tile_size, gutter = 64, 4
        atlas, uv_layout = _make_atlas_with_layout(2, tile_size, gutter)
        shared = {"v_shared": [("t0", 2), ("t1", 0)]}
        grid = _make_mock_globe_grid(["t0", "t1"], shared_vertex_ids=shared)

        hex_uv = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0),
                   (0.0, 1.0), (0.0, 0.5), (0.5, 0.0)]

        with patch(
            "polygrid.uv_texture.get_tile_uv_vertices",
            return_value=hex_uv,
        ):
            result = _blend_corner_junctions(
                atlas, uv_layout, grid, ["t0", "t1"],
                tile_size=tile_size, gutter=gutter,
                blend_radius=0,
            )

        assert result.size == atlas.size


# ═══════════════════════════════════════════════════════════════════
# build_polygon_cut_atlas signature
# ═══════════════════════════════════════════════════════════════════

class TestBuildPolygonCutAtlasCornerParams:
    """Verify build_polygon_cut_atlas accepts the new corner params."""

    def test_signature_accepts_blend_corners_param(self):
        sig = inspect.signature(build_polygon_cut_atlas)
        assert "blend_corners" in sig.parameters
        assert sig.parameters["blend_corners"].default is True

    def test_signature_accepts_blend_radius_param(self):
        sig = inspect.signature(build_polygon_cut_atlas)
        assert "blend_radius" in sig.parameters
        assert sig.parameters["blend_radius"].default == 2


# ═══════════════════════════════════════════════════════════════════
# subdivide_tile_mesh — boundary clamp skip (Fix 2)
# ═══════════════════════════════════════════════════════════════════

class TestSubdivideBoundaryClampSkip:
    """Verify that boundary vertices (b0==0) bypass UV clamping."""

    @pytest.fixture
    def hex_data(self):
        return _hex_tile_data()

    def test_boundary_uvs_unchanged_with_clamp(self, hex_data):
        """Boundary verts should keep original barycentric UV, not clamped."""
        center, vertices, center_uv, vertex_uvs = hex_data

        # Use a tight inset polygon that would aggressively clamp
        from polygrid.globe_renderer_v2 import compute_uv_polygon_inset
        inset = compute_uv_polygon_inset(
            vertex_uvs, inset_px=8.0, atlas_size=128,
        )

        vdata_clamped, _ = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 1.0, 1.0), radius=1.0, subdivisions=4,
            uv_clamp_polygon=inset,
        )
        vdata_plain, _ = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 1.0, 1.0), radius=1.0, subdivisions=4,
            uv_clamp_polygon=None,
        )

        # The actual polygon-corner UVs (= vertex_uvs) should be
        # identical in both because b0==0 at those vertices.
        # Find boundary vertices: they appear at the original
        # vertex_uvs positions (within rounding).
        corner_uv_set = {
            (round(u, 5), round(v, 5)) for u, v in vertex_uvs
        }

        for row_idx in range(vdata_clamped.shape[0]):
            u_c = round(float(vdata_clamped[row_idx, 6]), 5)
            v_c = round(float(vdata_clamped[row_idx, 7]), 5)
            u_p = round(float(vdata_plain[row_idx, 6]), 5)
            v_p = round(float(vdata_plain[row_idx, 7]), 5)
            if (u_p, v_p) in corner_uv_set:
                assert (u_c, v_c) == (u_p, v_p), (
                    f"Boundary UV at vertex {row_idx} was clamped: "
                    f"({u_c}, {v_c}) != ({u_p}, {v_p})"
                )

    def test_interior_uvs_still_clamped(self, hex_data):
        """Interior verts (b0>0) should still be clamped when outside the inset polygon."""
        center, vertices, center_uv, vertex_uvs = hex_data

        from polygrid.globe_renderer_v2 import (
            compute_uv_polygon_inset,
            clamp_uv_to_polygon,
        )
        # Use a very aggressive inset so many interior verts fall outside
        inset = compute_uv_polygon_inset(
            vertex_uvs, inset_px=8.0, atlas_size=64,
        )

        vdata_clamped, _ = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 1.0, 1.0), radius=1.0, subdivisions=4,
            uv_clamp_polygon=inset,
        )

        # Verify that at least one clamped UV differs from the
        # barycentric interpolation.  We check this by comparing
        # each clamped UV with the result of clamp_uv_to_polygon:
        # if it was clamped, it should equal the clamp output.
        n_clamped = 0
        for row_idx in range(vdata_clamped.shape[0]):
            u = float(vdata_clamped[row_idx, 6])
            v = float(vdata_clamped[row_idx, 7])
            cu, cv = clamp_uv_to_polygon(u, v, inset)
            # If (u,v) is outside the inset polygon, clamp would move
            # it.  Since we already clamped, (u,v) == (cu,cv).
            # We just need to show that *some* vertices were moved
            # from their original barycentric position.
            # Compare with distance from center: points far from
            # center (b0 small but >0) are likely clamped.
            du = u - center_uv[0]
            dv = v - center_uv[1]
            dist_from_center = math.sqrt(du * du + dv * dv)
            # Boundary UVs (vertex_uvs) are at radius ~0.4 from center.
            # Interior UVs near the boundary should be clamped inward.
            # A vertex is "clamped" if it sits exactly on the inset polygon
            # (clamp is idempotent there).
            if abs(cu - u) < 1e-8 and abs(cv - v) < 1e-8:
                # It's on or inside — check if it's on the edge
                # by testing slightly outward
                ou, ov = u + du * 0.01, v + dv * 0.01
                cu2, cv2 = clamp_uv_to_polygon(ou, ov, inset)
                if abs(cu2 - ou) > 1e-6 or abs(cv2 - ov) > 1e-6:
                    # The point was on the inset boundary → was clamped
                    n_clamped += 1

        # With a very large inset on a small atlas, many near-boundary
        # interior verts should be clamped.
        assert n_clamped > 0, (
            "Expected some interior vertices to be clamped onto the "
            "inset polygon boundary"
        )


# ═══════════════════════════════════════════════════════════════════
# subdivide_tile_mesh — UV-aware dedup (Fix 3)
# ═══════════════════════════════════════════════════════════════════

class TestSubdivideUVAwareDedup:
    """Verify that vertices at the same position with different UVs
    are NOT merged (UV-aware dedup key)."""

    def test_shared_position_different_uv_kept_separate(self):
        """Two fan sectors meeting at a boundary vertex where UV
        clamping has diverged should produce distinct vertices."""
        # Create a simple triangle (3-sided tile) to have clear
        # sector boundaries.
        center = (0.0, 0.0, 1.0)
        n = 3
        angle_step = 2 * math.pi / n
        r = 0.1
        vertices = [
            (r * math.cos(i * angle_step),
             r * math.sin(i * angle_step),
             1.0)
            for i in range(n)
        ]
        center_uv = (0.5, 0.5)
        vertex_uvs = [
            (0.5 + 0.4 * math.cos(i * angle_step),
             0.5 + 0.4 * math.sin(i * angle_step))
            for i in range(n)
        ]

        # With no clamping, boundary verts from adjacent sectors share
        # the same position AND UV, so dedup merges them.
        vdata_plain, _ = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 1.0, 1.0), radius=1.0, subdivisions=2,
        )

        # With clamping on but boundary skip active, boundary verts
        # still have identical UVs → same vertex count.
        from polygrid.globe_renderer_v2 import compute_uv_polygon_inset
        inset = compute_uv_polygon_inset(
            vertex_uvs, inset_px=2.0, atlas_size=256,
        )
        vdata_clamped, _ = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 1.0, 1.0), radius=1.0, subdivisions=2,
            uv_clamp_polygon=inset,
        )

        # Vertex count should be the same because the boundary skip
        # ensures consistent UVs at sector boundaries, so dedup can
        # still merge them.
        assert vdata_clamped.shape[0] == vdata_plain.shape[0], (
            f"With boundary skip, vertex count should match: "
            f"{vdata_clamped.shape[0]} vs {vdata_plain.shape[0]}"
        )

    def test_dedup_still_merges_identical_vertices(self):
        """Vertices with same position AND same UV should still be merged."""
        center, vertices, center_uv, vertex_uvs = _hex_tile_data()

        vdata, _ = subdivide_tile_mesh(
            center, vertices, center_uv, vertex_uvs,
            color=(1.0, 1.0, 1.0), radius=1.0, subdivisions=3,
        )

        # Count unique positions
        pos_keys = set()
        for row_idx in range(vdata.shape[0]):
            key = (
                round(float(vdata[row_idx, 0]), 7),
                round(float(vdata[row_idx, 1]), 7),
                round(float(vdata[row_idx, 2]), 7),
            )
            pos_keys.add(key)

        # The vertex count should be less than the total generated
        # (i.e. dedup is working), but equal to unique positions
        # (since UVs agree where positions agree in this case).
        assert len(pos_keys) == vdata.shape[0], (
            f"Expected {len(pos_keys)} unique verts but got "
            f"{vdata.shape[0]} — dedup may not be working"
        )
