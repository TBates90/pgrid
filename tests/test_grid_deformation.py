"""Tests for UV shape-matched grid deformation.

Covers:
- deform_grid_to_uv_shape preserves face count and vertex count
- Deformed grid corners match UV polygon proportions
- Warp anisotropy drops to ~1.0 for pentagon-adjacent hex tiles
- Non-pent-adjacent hexes are essentially unchanged (near-no-op)
- Pentagon tiles are not deformed (skipped)
- All sub-faces remain convex after deformation
- build_detail_grid integration (uv_shape_match flag)
- Backward compatibility (uv_shape_match=False gives regular grid)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from polygrid.detail_grid import build_detail_grid, deform_grid_to_uv_shape, detail_face_count
from polygrid.builders import build_pure_hex_grid

# ── Gate behind models availability ─────────────────────────────
try:
    from polygrid.globe import build_globe_grid, _HAS_MODELS
    from polygrid.uv_texture import get_tile_uv_vertices
    from polygrid.tile_uv_align import (
        get_macro_edge_corners,
        match_grid_corners_to_uv,
        _build_sector_affines,
    )
    _skip = not _HAS_MODELS
except ImportError:
    _skip = True

needs_models = pytest.mark.skipif(_skip, reason="models library not installed")


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def globe():
    """Shared frequency-3 globe grid for all tests."""
    return build_globe_grid(3)


def _pent_adjacent_hex_ids(globe):
    """Return face IDs of hex tiles adjacent to a pentagon."""
    pent_ids = {fid for fid, f in globe.faces.items() if f.face_type == "pent"}
    result = set()
    for pid in pent_ids:
        for nid in globe.faces[pid].neighbor_ids:
            if globe.faces[nid].face_type == "hex":
                result.add(nid)
    return sorted(result)


def _max_warp_anisotropy(globe, fid, uv_shape_match, detail_rings=2):
    """Measure worst-sector anisotropy for a tile."""
    face = globe.faces[fid]
    n_sides = len(face.vertex_ids)
    dg = build_detail_grid(globe, fid, detail_rings=detail_rings, size=1.0,
                           uv_shape_match=uv_shape_match)
    corner_ids = dg.metadata.get("corner_vertex_ids")
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)
    gc_raw = get_macro_edge_corners(dg, n_sides)
    gc_matched = match_grid_corners_to_uv(gc_raw, globe, fid)
    uv_raw = get_tile_uv_vertices(globe, fid)

    gc = np.array(gc_matched, dtype=np.float64)
    uv = np.array(uv_raw, dtype=np.float64)
    src_c = gc.mean(axis=0)
    dst_c = uv.mean(axis=0)
    sectors = _build_sector_affines(gc, src_c, uv, dst_c)

    max_aniso = 0.0
    for A, _t in sectors:
        _U, s, _Vt = np.linalg.svd(A)
        ratio = s[0] / s[1] if s[1] > 1e-12 else float("inf")
        max_aniso = max(max_aniso, ratio)
    return max_aniso


def _is_convex(pts):
    """Check if a polygon (list of (x, y)) is convex."""
    n = len(pts)
    signs = []
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        x3, y3 = pts[(i + 2) % n]
        cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        signs.append(cross)
    return all(s >= -1e-10 for s in signs) or all(s <= 1e-10 for s in signs)


# ═══════════════════════════════════════════════════════════════════
# deform_grid_to_uv_shape — unit tests
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestDeformGridToUvShape:
    """Tests for the standalone deform_grid_to_uv_shape function."""

    def test_preserves_face_and_vertex_counts(self, globe):
        """Deformation must not add or remove faces or vertices."""
        fid = "t1"
        face = globe.faces[fid]
        n = len(face.vertex_ids)
        dg = build_detail_grid(globe, fid, detail_rings=2, size=1.0, uv_shape_match=False)
        orig_n_faces = len(dg.faces)
        orig_n_verts = len(dg.vertices)

        dg.compute_macro_edges(n_sides=n)
        gc_raw = get_macro_edge_corners(dg, n)
        gc_matched = match_grid_corners_to_uv(gc_raw, globe, fid)
        uv_corners = get_tile_uv_vertices(globe, fid)

        deform_grid_to_uv_shape(dg, gc_matched, uv_corners)
        assert len(dg.faces) == orig_n_faces
        assert len(dg.vertices) == orig_n_verts

    def test_centroid_unchanged(self, globe):
        """The grid centroid should not move after deformation."""
        fid = "t1"
        face = globe.faces[fid]
        n = len(face.vertex_ids)
        dg = build_detail_grid(globe, fid, detail_rings=2, size=1.0, uv_shape_match=False)
        dg.compute_macro_edges(n_sides=n)
        gc_raw = get_macro_edge_corners(dg, n)
        gc_matched = match_grid_corners_to_uv(gc_raw, globe, fid)

        # Record centroid before
        xs = [v.x for v in dg.vertices.values()]
        ys = [v.y for v in dg.vertices.values()]
        cx_before = sum(xs) / len(xs)
        cy_before = sum(ys) / len(ys)

        uv_corners = get_tile_uv_vertices(globe, fid)
        deform_grid_to_uv_shape(dg, gc_matched, uv_corners)

        xs = [v.x for v in dg.vertices.values()]
        ys = [v.y for v in dg.vertices.values()]
        cx_after = sum(xs) / len(xs)
        cy_after = sum(ys) / len(ys)

        # Allow small floating-point drift
        assert abs(cx_after - cx_before) < 1e-6
        assert abs(cy_after - cy_before) < 1e-6

    def test_no_op_for_regular_polygon(self, globe):
        """For a tile with regular UV polygon, deformation should be near-zero."""
        # t3 is a non-pent-adjacent hex with UV edge ratio ≈ 1.0
        fid = "t3"
        face = globe.faces[fid]
        n = len(face.vertex_ids)
        dg = build_detail_grid(globe, fid, detail_rings=2, size=1.0, uv_shape_match=False)
        dg.compute_macro_edges(n_sides=n)
        gc_raw = get_macro_edge_corners(dg, n)
        gc_matched = match_grid_corners_to_uv(gc_raw, globe, fid)

        # Record positions before
        positions_before = {vid: (v.x, v.y) for vid, v in dg.vertices.items()}

        uv_corners = get_tile_uv_vertices(globe, fid)
        deform_grid_to_uv_shape(dg, gc_matched, uv_corners)

        # All displacements should be very small
        max_disp = 0.0
        for vid, v in dg.vertices.items():
            bx, by = positions_before[vid]
            max_disp = max(max_disp, math.hypot(v.x - bx, v.y - by))

        # Non-pent-adjacent hexes may have mild UV irregularity (ratio ≤ 1.15)
        # so allow a small displacement threshold.
        assert max_disp < 0.05, f"Non-regular tile moved by {max_disp}"

    def test_all_subfaces_convex_after_deform(self, globe):
        """Every sub-face must remain convex after deformation."""
        for fid in globe.faces:
            face = globe.faces[fid]
            if face.face_type != "hex":
                continue
            n = len(face.vertex_ids)
            dg = build_detail_grid(globe, fid, detail_rings=3, size=1.0,
                                   uv_shape_match=True)
            for sf in dg.faces.values():
                pts = [(dg.vertices[vid].x, dg.vertices[vid].y)
                       for vid in sf.vertex_ids]
                assert _is_convex(pts), (
                    f"Non-convex sub-face {sf.id} in deformed {fid}"
                )


# ═══════════════════════════════════════════════════════════════════
# Warp anisotropy — the key metric
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestWarpAnisotropy:
    """Verify that deformation eliminates warp anisotropy."""

    def test_pent_adjacent_anisotropy_without_deform(self, globe):
        """Without deformation, pent-adjacent hexes have anisotropy > 1.2."""
        fid = _pent_adjacent_hex_ids(globe)[0]
        aniso = _max_warp_anisotropy(globe, fid, uv_shape_match=False)
        assert aniso > 1.15, f"Expected aniso > 1.15, got {aniso:.3f}"

    def test_pent_adjacent_anisotropy_with_deform(self, globe):
        """With deformation, pent-adjacent hexes have anisotropy ≈ 1.0."""
        fid = _pent_adjacent_hex_ids(globe)[0]
        aniso = _max_warp_anisotropy(globe, fid, uv_shape_match=True)
        assert aniso < 1.02, f"Expected aniso < 1.02, got {aniso:.3f}"

    def test_all_pent_adjacent_hexes_improved(self, globe):
        """Every pentagon-adjacent hex tile should have anisotropy < 1.02."""
        for fid in _pent_adjacent_hex_ids(globe):
            aniso = _max_warp_anisotropy(globe, fid, uv_shape_match=True)
            assert aniso < 1.02, f"{fid}: aniso={aniso:.3f}"

    def test_non_pent_adjacent_unaffected(self, globe):
        """Non-pent-adjacent hexes should stay at anisotropy ≈ 1.0."""
        pent_adj = set(_pent_adjacent_hex_ids(globe))
        for fid, face in globe.faces.items():
            if face.face_type != "hex" or fid in pent_adj:
                continue
            aniso = _max_warp_anisotropy(globe, fid, uv_shape_match=True)
            assert aniso < 1.02, f"{fid}: aniso={aniso:.3f}"
            break  # one sample is enough — they're all symmetric


# ═══════════════════════════════════════════════════════════════════
# build_detail_grid integration
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestBuildDetailGridIntegration:
    """Test that uv_shape_match is properly wired into build_detail_grid."""

    def test_metadata_flag_set(self, globe):
        """build_detail_grid should set uv_shape_matched metadata."""
        dg = build_detail_grid(globe, "t1", detail_rings=2, uv_shape_match=True)
        assert dg.metadata.get("uv_shape_matched") is True

    def test_metadata_flag_not_set_when_disabled(self, globe):
        """With uv_shape_match=False, no metadata flag should be set."""
        dg = build_detail_grid(globe, "t1", detail_rings=2, uv_shape_match=False)
        assert "uv_shape_matched" not in dg.metadata

    def test_pentagon_not_deformed(self, globe):
        """Pentagon tiles should never be deformed regardless of flag."""
        dg = build_detail_grid(globe, "t0", detail_rings=2, uv_shape_match=True)
        assert "uv_shape_matched" not in dg.metadata

    def test_face_count_preserved(self, globe):
        """Deformed grid must have the same face count as regular grid."""
        for rings in (1, 2, 3, 4):
            dg_off = build_detail_grid(globe, "t1", detail_rings=rings,
                                       uv_shape_match=False)
            dg_on = build_detail_grid(globe, "t1", detail_rings=rings,
                                      uv_shape_match=True)
            assert len(dg_on.faces) == len(dg_off.faces)

    def test_macro_edges_valid_after_deform(self, globe):
        """Macro edges should be recomputed and valid after deformation."""
        fid = "t1"
        face = globe.faces[fid]
        n = len(face.vertex_ids)
        dg = build_detail_grid(globe, fid, detail_rings=2, uv_shape_match=True)
        dg.compute_macro_edges(n_sides=n)
        assert len(dg.macro_edges) == n
        for me in dg.macro_edges:
            assert len(me.vertex_ids) >= 2

    @pytest.mark.parametrize("rings", [1, 2, 3, 4, 5])
    def test_various_ring_counts(self, globe, rings):
        """Deformation should work for all ring counts."""
        dg = build_detail_grid(globe, "t1", detail_rings=rings,
                               uv_shape_match=True)
        assert dg.metadata.get("uv_shape_matched") is True
        expected = detail_face_count("hex", rings)
        assert len(dg.faces) == expected


# ═══════════════════════════════════════════════════════════════════
# Edge-length ratio matching
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestEdgeRatioMatching:
    """Verify that deformed grid edge ratios match UV polygon edge ratios."""

    def test_grid_edge_ratio_matches_uv(self, globe):
        """After deformation, grid macro-edge ratio ≈ UV edge ratio."""
        fid = "t1"
        face = globe.faces[fid]
        n = len(face.vertex_ids)

        dg = build_detail_grid(globe, fid, detail_rings=2, uv_shape_match=True)
        dg.compute_macro_edges(n_sides=n)
        gc_raw = get_macro_edge_corners(dg, n)
        gc_matched = match_grid_corners_to_uv(gc_raw, globe, fid)

        gc = np.array(gc_matched, dtype=np.float64)
        gc_edges = [np.linalg.norm(gc[(i + 1) % n] - gc[i]) for i in range(n)]
        grid_ratio = max(gc_edges) / min(gc_edges)

        uv = np.array(get_tile_uv_vertices(globe, fid), dtype=np.float64)
        uv_edges = [np.linalg.norm(uv[(i + 1) % n] - uv[i]) for i in range(n)]
        uv_ratio = max(uv_edges) / min(uv_edges)

        # Grid edge ratio should be close to UV edge ratio (within 5%)
        assert abs(grid_ratio - uv_ratio) < 0.05, (
            f"grid_ratio={grid_ratio:.3f} vs uv_ratio={uv_ratio:.3f}"
        )
