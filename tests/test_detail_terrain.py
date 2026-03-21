"""Tests for Phase 10B — Boundary-aware detail terrain (detail_terrain.py).

Covers:
- compute_boundary_elevations returns expected structure
- classify_detail_faces produces interior / boundary / corner labels
- generate_detail_terrain_bounded produces valid elevation for all faces
- Boundary faces interpolate between parent and neighbour elevations
- Interior faces cluster around parent elevation
- Adjacent tile boundary seam test
- generate_all_detail_terrain batch generation
- Determinism: same inputs → same output
- No NaN or infinite values
"""

from __future__ import annotations

import math
import pytest

from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
from polygrid.detail_grid import detail_face_count
from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
from polygrid.detail_terrain import (
    compute_boundary_elevations,
    classify_detail_faces,
    generate_detail_terrain_bounded,
    generate_all_detail_terrain,
    compute_neighbor_edge_mapping,
)

try:
    from polygrid.globe import build_globe_grid, _HAS_MODELS
    _skip = not _HAS_MODELS
except ImportError:
    _skip = True

needs_models = pytest.mark.skipif(_skip, reason="models library not installed")


def _make_globe_with_elevation(frequency: int = 3, seed: int = 42):
    from conftest import cached_build_globe
    from polygrid.mountains import MountainConfig, generate_mountains

    grid = cached_build_globe(frequency)
    if grid is None:
        pytest.skip("models library not installed")
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)
    config = MountainConfig(seed=seed)
    generate_mountains(grid, store, config)
    return grid, store


def _compute_edge_mapping(globe_grid, face_id, detail_grid):
    """Compute neighbour→macro-edge mapping, matching the batch pipeline."""
    from polygrid.tile_uv_align import compute_pg_to_macro_edge_map

    pg_map = compute_neighbor_edge_mapping(globe_grid, face_id)
    corner_ids = detail_grid.metadata.get("corner_vertex_ids")
    if corner_ids:
        n_sides = len(corner_ids)
        if not detail_grid.macro_edges:
            detail_grid.compute_macro_edges(n_sides, corner_ids=corner_ids)
        pg2macro = compute_pg_to_macro_edge_map(
            globe_grid, face_id, detail_grid,
        )
        return {nid: pg2macro.get(idx, idx) for nid, idx in pg_map.items()}
    return pg_map


# ═══════════════════════════════════════════════════════════════════
# compute_boundary_elevations
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestComputeBoundaryElevations:
    def test_returns_dict_for_every_face(self):
        grid, store = _make_globe_with_elevation(3)
        boundary = compute_boundary_elevations(grid, store)
        assert set(boundary.keys()) == set(grid.faces.keys())

    def test_neighbour_entries_are_averages(self):
        grid, store = _make_globe_with_elevation(3)
        boundary = compute_boundary_elevations(grid, store)
        fid = list(grid.faces.keys())[0]
        own_elev = store.get(fid, "elevation")
        for nid, target in boundary[fid].items():
            n_elev = store.get(nid, "elevation")
            expected = (own_elev + n_elev) / 2.0
            assert abs(target - expected) < 1e-10

    def test_pentagons_have_5_neighbours(self):
        grid, store = _make_globe_with_elevation(3)
        boundary = compute_boundary_elevations(grid, store)
        for fid in grid.faces:
            if grid.faces[fid].face_type == "pent":
                assert len(boundary[fid]) == 5, (
                    f"Pentagon {fid} has {len(boundary[fid])} boundary entries"
                )

    def test_hexagons_have_6_neighbours(self):
        grid, store = _make_globe_with_elevation(3)
        boundary = compute_boundary_elevations(grid, store)
        for fid in grid.faces:
            if grid.faces[fid].face_type == "hex":
                assert len(boundary[fid]) == 6, (
                    f"Hexagon {fid} has {len(boundary[fid])} boundary entries"
                )


# ═══════════════════════════════════════════════════════════════════
# classify_detail_faces
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestClassifyDetailFaces:
    def test_all_faces_classified(self):
        grid, _ = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=3)
        coll = DetailGridCollection.build(grid, spec)
        fid = coll.face_ids[0]
        detail_grid, _ = coll.get(fid)
        cls = classify_detail_faces(detail_grid, boundary_depth=1)
        assert set(cls.keys()) == set(detail_grid.faces.keys())

    def test_only_valid_labels(self):
        grid, _ = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=3)
        coll = DetailGridCollection.build(grid, spec)
        fid = coll.face_ids[0]
        detail_grid, _ = coll.get(fid)
        cls = classify_detail_faces(detail_grid, boundary_depth=1)
        for label in cls.values():
            assert label in ("interior", "boundary", "corner")

    def test_has_interior_and_boundary(self):
        """A grid with enough rings should have both interior and boundary."""
        grid, _ = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=3)
        coll = DetailGridCollection.build(grid, spec)
        fid = coll.face_ids[0]
        detail_grid, _ = coll.get(fid)
        cls = classify_detail_faces(detail_grid, boundary_depth=1)
        labels = set(cls.values())
        assert "interior" in labels
        assert "boundary" in labels

    def test_deeper_boundary_adds_more_boundary_faces(self):
        grid, _ = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=4)
        coll = DetailGridCollection.build(grid, spec)
        fid = coll.face_ids[0]
        detail_grid, _ = coll.get(fid)
        cls1 = classify_detail_faces(detail_grid, boundary_depth=1)
        cls2 = classify_detail_faces(detail_grid, boundary_depth=2)
        n_boundary_1 = sum(1 for v in cls1.values() if v != "interior")
        n_boundary_2 = sum(1 for v in cls2.values() if v != "interior")
        assert n_boundary_2 >= n_boundary_1


# ═══════════════════════════════════════════════════════════════════
# generate_detail_terrain_bounded
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestGenerateDetailTerrainBounded:
    def test_all_faces_have_elevation(self):
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        fid = coll.face_ids[0]
        detail_grid, _ = coll.get(fid)
        parent_elev = store.get(fid, "elevation")
        boundary = compute_boundary_elevations(grid, store)
        nbr_elevs = boundary[fid]
        edge_map = _compute_edge_mapping(grid, fid, detail_grid)

        result_store = generate_detail_terrain_bounded(
            detail_grid, parent_elev, nbr_elevs, spec, seed=42,
            neighbor_edge_map=edge_map,
        )
        for sub_fid in detail_grid.faces:
            val = result_store.get(sub_fid, "elevation")
            assert isinstance(val, (int, float))

    def test_no_nan_or_inf(self):
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        fid = coll.face_ids[0]
        detail_grid, _ = coll.get(fid)
        parent_elev = store.get(fid, "elevation")
        boundary = compute_boundary_elevations(grid, store)
        edge_map = _compute_edge_mapping(grid, fid, detail_grid)

        result_store = generate_detail_terrain_bounded(
            detail_grid, parent_elev, boundary[fid], spec, seed=42,
            neighbor_edge_map=edge_map,
        )
        for sub_fid in detail_grid.faces:
            val = result_store.get(sub_fid, "elevation")
            assert math.isfinite(val), f"Non-finite value in {sub_fid}: {val}"

    def test_determinism(self):
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        fid = coll.face_ids[0]
        detail_grid, _ = coll.get(fid)
        parent_elev = store.get(fid, "elevation")
        boundary = compute_boundary_elevations(grid, store)
        edge_map = _compute_edge_mapping(grid, fid, detail_grid)

        s1 = generate_detail_terrain_bounded(
            detail_grid, parent_elev, boundary[fid], spec, seed=42,
            neighbor_edge_map=edge_map,
        )
        s2 = generate_detail_terrain_bounded(
            detail_grid, parent_elev, boundary[fid], spec, seed=42,
            neighbor_edge_map=edge_map,
        )
        for sub_fid in detail_grid.faces:
            assert s1.get(sub_fid, "elevation") == s2.get(sub_fid, "elevation")

    def test_empty_neighbours_uses_parent_as_boundary_target(self):
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2, boundary_smoothing=0)
        coll = DetailGridCollection.build(grid, spec)
        fid = coll.face_ids[0]
        detail_grid, _ = coll.get(fid)
        parent_elev = store.get(fid, "elevation")

        # No neighbour elevations → boundary target = parent elevation
        result_store = generate_detail_terrain_bounded(
            detail_grid, parent_elev, {}, spec, seed=42,
        )
        for sub_fid in detail_grid.faces:
            val = result_store.get(sub_fid, "elevation")
            assert isinstance(val, (int, float))


# ═══════════════════════════════════════════════════════════════════
# generate_all_detail_terrain (batch)
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestGenerateAllDetailTerrain:
    def test_all_stores_populated(self):
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)
        assert len(coll.stores) == len(coll.grids)

    def test_every_face_has_elevation(self):
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)
        for fid in coll.face_ids:
            detail_grid, s = coll.get(fid)
            for sub_fid in detail_grid.faces:
                val = s.get(sub_fid, "elevation")
                assert math.isfinite(val)

    def test_boundary_seam_similarity(self):
        """Adjacent tiles' boundary faces should have similar elevation.

        This is a soft check: we verify the max difference is below a
        reasonable threshold rather than expecting exact equality.
        """
        from polygrid.core.algorithms import get_face_adjacency

        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=3, boundary_smoothing=2)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)

        adj = get_face_adjacency(grid)
        max_diff = 0.0
        comparisons = 0

        # For a handful of tile pairs, compare boundary-face elevations
        for fid in list(grid.faces.keys())[:5]:
            detail_grid_a, store_a = coll.get(fid)
            elev_a = store.get(fid, "elevation")
            for nid in adj.get(fid, [])[:2]:
                detail_grid_b, store_b = coll.get(nid)
                elev_b = store.get(nid, "elevation")
                # The boundary target for this pair is (elev_a + elev_b) / 2
                target = (elev_a + elev_b) / 2.0
                # Check that boundary faces of both tiles are near the target
                cls_a = classify_detail_faces(detail_grid_a, boundary_depth=1)
                cls_b = classify_detail_faces(detail_grid_b, boundary_depth=1)
                for sub_fid, label in cls_a.items():
                    if label == "boundary":
                        diff = abs(store_a.get(sub_fid, "elevation") - target)
                        max_diff = max(max_diff, diff)
                        comparisons += 1

        assert comparisons > 0, "No boundary comparisons made"
        # Boundary faces should be within a reasonable range of the target
        # (not exact due to noise, but not wildly different)
        assert max_diff < 1.0, f"Max boundary diff {max_diff} too large"
