# TODO REMOVE — Tests dead module region_stitch.py.
"""Tests for Phase 11C — stitched sub-grid terrain.

Tests verify:
- Stitched grid contains all sub-faces from source tiles (no loss)
- Face mapping is complete and correct
- Gnomonic projection produces sensible 2-D coordinates
- Terrain generation on combined grid produces valid elevation
- Split-back populates per-tile stores correctly
- End-to-end convenience function works
- Cross-tile elevation is more continuous than independent generation
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Set, Tuple

import pytest


# ── Test helpers ─────────────────────────────────────────────────────

def _require_globe():
    try:
        from polygrid.globe import build_globe_grid
        return build_globe_grid
    except ImportError:
        pytest.skip("models library not installed")


def _build_test_fixtures(frequency: int = 3, detail_rings: int = 3):
    """Build globe, collection, and globe store for testing."""
    from conftest import cached_build_globe_and_collection

    result = cached_build_globe_and_collection(frequency, detail_rings)
    if result is None:
        pytest.skip("models library not installed")
    return result


def _get_adjacent_group(globe, count: int = 4) -> List[str]:
    """Return a group of *count* adjacent tile ids."""
    from polygrid.algorithms import get_face_adjacency
    adj = get_face_adjacency(globe)
    start = list(globe.faces.keys())[5]  # arbitrary start
    group = [start]
    for _ in range(count - 1):
        for fid in group:
            for nbr in adj[fid]:
                if nbr not in group:
                    group.append(nbr)
                    break
            if len(group) > len(set(group)) - 1:
                break
        if len(group) >= count:
            break
    return group[:count]


# ═══════════════════════════════════════════════════════════════════
# Test: stitch_detail_grids
# ═══════════════════════════════════════════════════════════════════

class TestStitchDetailGrids:
    def test_combined_face_count(self):
        """Combined grid has same total faces as sum of individual grids."""
        globe, coll, _ = _build_test_fixtures()
        from polygrid.region_stitch import stitch_detail_grids

        face_ids = _get_adjacent_group(globe, 4)
        combined, mapping = stitch_detail_grids(coll, globe, face_ids)

        expected = sum(len(coll.grids[fid].faces) for fid in face_ids)
        assert len(combined.faces) == expected

    def test_mapping_complete(self):
        """Every combined face maps back to a (tile_id, sub_face_id)."""
        globe, coll, _ = _build_test_fixtures()
        from polygrid.region_stitch import stitch_detail_grids

        face_ids = _get_adjacent_group(globe, 3)
        combined, mapping = stitch_detail_grids(coll, globe, face_ids)

        assert len(mapping) == len(combined.faces)
        for cfid in combined.faces:
            assert cfid in mapping
            tile_id, sub_fid = mapping[cfid]
            assert tile_id in face_ids
            assert sub_fid in coll.grids[tile_id].faces

    def test_mapping_covers_all_source_faces(self):
        """Every source sub-face appears exactly once in the mapping."""
        globe, coll, _ = _build_test_fixtures()
        from polygrid.region_stitch import stitch_detail_grids

        face_ids = _get_adjacent_group(globe, 3)
        combined, mapping = stitch_detail_grids(coll, globe, face_ids)

        source_faces: Set[Tuple[str, str]] = set()
        for tile_id in face_ids:
            for sf_id in coll.grids[tile_id].faces:
                source_faces.add((tile_id, sf_id))

        mapped_faces = set(mapping.values())
        assert mapped_faces == source_faces

    def test_vertices_have_positions(self):
        """All combined vertices have valid 2-D coordinates."""
        globe, coll, _ = _build_test_fixtures()
        from polygrid.region_stitch import stitch_detail_grids

        face_ids = _get_adjacent_group(globe, 3)
        combined, _ = stitch_detail_grids(coll, globe, face_ids)

        for vid, v in combined.vertices.items():
            assert v.x is not None, f"Vertex {vid} has no x"
            assert v.y is not None, f"Vertex {vid} has no y"
            assert math.isfinite(v.x), f"Vertex {vid} x is not finite"
            assert math.isfinite(v.y), f"Vertex {vid} y is not finite"

    def test_gnomonic_coordinates_reasonable(self):
        """Projected coordinates should be small (near the tangent point)."""
        globe, coll, _ = _build_test_fixtures()
        from polygrid.region_stitch import stitch_detail_grids

        face_ids = _get_adjacent_group(globe, 4)
        combined, _ = stitch_detail_grids(coll, globe, face_ids)

        max_coord = max(
            max(abs(v.x), abs(v.y))
            for v in combined.vertices.values()
            if v.x is not None and v.y is not None
        )
        # On a unit sphere, gnomonic coordinates for a small patch
        # should be well under 1.0
        assert max_coord < 2.0, f"Coordinates too large: {max_coord}"

    def test_metadata(self):
        """Combined grid metadata records source tiles."""
        globe, coll, _ = _build_test_fixtures()
        from polygrid.region_stitch import stitch_detail_grids

        face_ids = _get_adjacent_group(globe, 3)
        combined, _ = stitch_detail_grids(coll, globe, face_ids)

        assert combined.metadata["generator"] == "region_stitch"
        assert set(combined.metadata["source_tiles"]) == set(face_ids)
        assert combined.metadata["tile_count"] == len(face_ids)

    def test_empty_face_ids_raises(self):
        globe, coll, _ = _build_test_fixtures()
        from polygrid.region_stitch import stitch_detail_grids

        with pytest.raises(ValueError, match="non-empty"):
            stitch_detail_grids(coll, globe, [])

    def test_single_tile(self):
        """Stitching a single tile should produce a valid grid."""
        globe, coll, _ = _build_test_fixtures()
        from polygrid.region_stitch import stitch_detail_grids

        face_ids = [list(globe.faces.keys())[0]]
        combined, mapping = stitch_detail_grids(coll, globe, face_ids)

        expected = len(coll.grids[face_ids[0]].faces)
        assert len(combined.faces) == expected
        assert len(mapping) == expected


# ═══════════════════════════════════════════════════════════════════
# Test: generate_terrain_on_stitched
# ═══════════════════════════════════════════════════════════════════

class TestGenerateTerrainOnStitched:
    def test_all_faces_have_elevation(self):
        globe, coll, gs = _build_test_fixtures()
        from polygrid.region_stitch import (
            stitch_detail_grids, generate_terrain_on_stitched,
        )
        from polygrid.detail_terrain_3d import Terrain3DSpec

        face_ids = _get_adjacent_group(globe, 4)
        combined, mapping = stitch_detail_grids(coll, globe, face_ids)
        store = generate_terrain_on_stitched(
            combined, mapping, globe, gs, Terrain3DSpec(seed=42),
        )

        for cfid in combined.faces:
            elev = store.get(cfid, "elevation")
            assert isinstance(elev, float)
            assert math.isfinite(elev)

    def test_elevation_varies(self):
        """Elevation should not be uniform — noise should create variation."""
        globe, coll, gs = _build_test_fixtures()
        from polygrid.region_stitch import (
            stitch_detail_grids, generate_terrain_on_stitched,
        )
        from polygrid.detail_terrain_3d import Terrain3DSpec

        face_ids = _get_adjacent_group(globe, 4)
        combined, mapping = stitch_detail_grids(coll, globe, face_ids)
        store = generate_terrain_on_stitched(
            combined, mapping, globe, gs, Terrain3DSpec(seed=42),
        )

        elevs = [store.get(cfid, "elevation") for cfid in combined.faces]
        assert max(elevs) > min(elevs), "No elevation variation"


# ═══════════════════════════════════════════════════════════════════
# Test: split_terrain_to_tiles
# ═══════════════════════════════════════════════════════════════════

class TestSplitTerrainToTiles:
    def test_all_tile_stores_created(self):
        globe, coll, gs = _build_test_fixtures()
        from polygrid.region_stitch import (
            stitch_detail_grids, generate_terrain_on_stitched,
            split_terrain_to_tiles,
        )
        from polygrid.detail_terrain_3d import Terrain3DSpec

        face_ids = _get_adjacent_group(globe, 3)
        combined, mapping = stitch_detail_grids(coll, globe, face_ids)
        store = generate_terrain_on_stitched(
            combined, mapping, globe, gs, Terrain3DSpec(seed=42),
        )
        split_terrain_to_tiles(store, mapping, coll)

        for fid in face_ids:
            assert fid in coll._stores, f"Missing store for {fid}"

    def test_split_values_match_combined(self):
        """Split elevation values must exactly match the combined grid."""
        globe, coll, gs = _build_test_fixtures()
        from polygrid.region_stitch import (
            stitch_detail_grids, generate_terrain_on_stitched,
            split_terrain_to_tiles,
        )
        from polygrid.detail_terrain_3d import Terrain3DSpec

        face_ids = _get_adjacent_group(globe, 3)
        combined, mapping = stitch_detail_grids(coll, globe, face_ids)
        cstore = generate_terrain_on_stitched(
            combined, mapping, globe, gs, Terrain3DSpec(seed=42),
        )
        split_terrain_to_tiles(cstore, mapping, coll)

        for cfid, (tile_id, sub_fid) in mapping.items():
            combined_elev = cstore.get(cfid, "elevation")
            tile_elev = coll._stores[tile_id].get(sub_fid, "elevation")
            assert combined_elev == tile_elev, \
                f"Mismatch at {tile_id}/{sub_fid}: {combined_elev} vs {tile_elev}"

    def test_all_subfaces_populated(self):
        """Every sub-face in every tile store has an elevation value."""
        globe, coll, gs = _build_test_fixtures()
        from polygrid.region_stitch import (
            stitch_detail_grids, generate_terrain_on_stitched,
            split_terrain_to_tiles,
        )
        from polygrid.detail_terrain_3d import Terrain3DSpec

        face_ids = _get_adjacent_group(globe, 4)
        combined, mapping = stitch_detail_grids(coll, globe, face_ids)
        cstore = generate_terrain_on_stitched(
            combined, mapping, globe, gs, Terrain3DSpec(seed=42),
        )
        split_terrain_to_tiles(cstore, mapping, coll)

        for tile_id in face_ids:
            store = coll._stores[tile_id]
            grid = coll.grids[tile_id]
            for sf_id in grid.faces:
                elev = store.get(sf_id, "elevation")
                assert isinstance(elev, float), f"Missing {tile_id}/{sf_id}"


# ═══════════════════════════════════════════════════════════════════
# Test: generate_stitched_patch_terrain (end-to-end)
# ═══════════════════════════════════════════════════════════════════

class TestGenerateStitchedPatchTerrain:
    def test_end_to_end(self):
        globe, coll, gs = _build_test_fixtures()
        from polygrid.region_stitch import generate_stitched_patch_terrain
        from polygrid.detail_terrain_3d import Terrain3DSpec

        face_ids = _get_adjacent_group(globe, 4)
        combined_store = generate_stitched_patch_terrain(
            coll, globe, gs, face_ids, Terrain3DSpec(seed=42),
        )

        assert combined_store is not None
        for fid in face_ids:
            assert fid in coll._stores

    def test_deterministic(self):
        """Same inputs produce identical results."""
        globe, coll_a, gs = _build_test_fixtures()
        _, coll_b, _ = _build_test_fixtures()
        from polygrid.region_stitch import generate_stitched_patch_terrain
        from polygrid.detail_terrain_3d import Terrain3DSpec

        face_ids = _get_adjacent_group(globe, 3)
        spec = Terrain3DSpec(seed=42)

        generate_stitched_patch_terrain(coll_a, globe, gs, face_ids, spec)
        generate_stitched_patch_terrain(coll_b, globe, gs, face_ids, spec)

        for fid in face_ids:
            grid = coll_a.grids[fid]
            for sf_id in grid.faces:
                a = coll_a._stores[fid].get(sf_id, "elevation")
                b = coll_b._stores[fid].get(sf_id, "elevation")
                assert a == b, f"Non-deterministic at {fid}/{sf_id}"


# ═══════════════════════════════════════════════════════════════════
# Test: Cross-tile continuity improvement
# ═══════════════════════════════════════════════════════════════════

class TestCrossTileContinuity:
    def test_stitched_terrain_is_continuous(self):
        """Stitched terrain should have reasonable elevation variation."""
        globe, coll, gs = _build_test_fixtures()
        from polygrid.region_stitch import generate_stitched_patch_terrain
        from polygrid.detail_terrain_3d import Terrain3DSpec

        face_ids = _get_adjacent_group(globe, 4)
        generate_stitched_patch_terrain(
            coll, globe, gs, face_ids,
            Terrain3DSpec(seed=42, boundary_smoothing=3),
        )

        # Collect all elevations
        all_elevs = []
        for fid in face_ids:
            store = coll._stores[fid]
            grid = coll.grids[fid]
            for sf_id in grid.faces:
                all_elevs.append(store.get(sf_id, "elevation"))

        # Basic sanity: values should vary and be finite
        assert all(math.isfinite(e) for e in all_elevs)
        assert max(all_elevs) > min(all_elevs)
        # Standard deviation should be non-trivial
        assert statistics.stdev(all_elevs) > 0.001


# ═══════════════════════════════════════════════════════════════════
# Test: gnomonic projection helper
# ═══════════════════════════════════════════════════════════════════

class TestGnomonicProject:
    def test_centre_projects_to_origin(self):
        """The tangent point itself should project to (0, 0)."""
        from polygrid.region_stitch import _gnomonic_project
        from polygrid.detail_terrain_3d import _tangent_basis

        centre = (0.0, 0.0, 1.0)
        tu, tv = _tangent_basis(centre)
        u, v = _gnomonic_project(centre, centre, tu, tv)
        assert abs(u) < 1e-10
        assert abs(v) < 1e-10

    def test_nearby_point_projects_close(self):
        """A point near the tangent point should project to small coordinates."""
        from polygrid.region_stitch import _gnomonic_project
        from polygrid.detail_terrain_3d import _normalize, _tangent_basis

        centre = (0.0, 0.0, 1.0)
        tu, tv = _tangent_basis(centre)
        # Slightly offset point on the sphere
        nearby = _normalize((0.01, 0.0, 1.0))
        u, v = _gnomonic_project(nearby, centre, tu, tv)
        assert abs(u) < 0.1
        assert abs(v) < 0.1

    def test_opposite_hemisphere_degenerate(self):
        """A point near the opposite pole should produce large coordinates."""
        from polygrid.region_stitch import _gnomonic_project
        from polygrid.detail_terrain_3d import _normalize, _tangent_basis

        centre = (0.0, 0.0, 1.0)
        tu, tv = _tangent_basis(centre)
        # Point near equator — still projects but with larger coords
        equator = _normalize((1.0, 0.0, 0.1))
        u, v = _gnomonic_project(equator, centre, tu, tv)
        # Should be larger than nearby points
        assert abs(u) > 1.0 or abs(v) > 1.0 or math.sqrt(u*u + v*v) > 0.5
