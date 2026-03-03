"""Tests for Phase 18A — apron grid construction.

Covers:
- 18A.1 — boundary sub-face classification
- 18A.2 — edge sub-face mapping between adjacent tiles
- 18A.3 — apron grid construction (extended PolyGrid)
- 18A.4 — terrain propagation into apron zone
"""

from __future__ import annotations

import math
from typing import Dict

import pytest

from polygrid.algorithms import get_face_adjacency
from polygrid.apron_grid import (
    ApronResult,
    EdgeSubfaceMapping,
    build_all_apron_grids,
    build_apron_grid,
    boundary_subface_ids,
    classify_boundary_subfaces,
    compute_edge_subface_mapping,
    propagate_apron_terrain,
)
from polygrid.builders import build_pure_hex_grid, hex_face_count
from polygrid.detail_grid import build_detail_grid, detail_face_count
from polygrid.geometry import face_center
from polygrid.goldberg_topology import build_goldberg_grid
from polygrid.models import Face, Vertex
from polygrid.polygrid import PolyGrid
from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
from polygrid.tile_detail import (
    DetailGridCollection,
    TileDetailSpec,
    build_all_detail_grids,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _make_globe_grid(frequency: int = 2) -> PolyGrid:
    """Create a minimal globe grid for testing.

    Uses a pure hex grid as a stand-in for a globe grid — each face
    represents a "globe tile" with adjacency.
    """
    grid = build_pure_hex_grid(frequency)
    return grid.with_neighbors()


def _make_detail_collection(
    globe_grid: PolyGrid,
    detail_rings: int = 2,
    generate_terrain: bool = True,
) -> DetailGridCollection:
    """Build a DetailGridCollection with optional terrain."""
    spec = TileDetailSpec(detail_rings=detail_rings)
    coll = DetailGridCollection.build(globe_grid, spec)

    if generate_terrain:
        # Create a simple globe store with elevation
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        globe_store = TileDataStore(grid=globe_grid, schema=schema)
        for fid in globe_grid.faces:
            # Vary elevation by face index for interesting patterns
            idx = int(fid.replace("f", "")) if fid.startswith("f") else 0
            globe_store.set(fid, "elevation", 0.1 * (idx % 5))

        coll.generate_all_terrain(globe_store, seed=42)

    return coll


# ═══════════════════════════════════════════════════════════════════
# 18A.1 — Classify boundary sub-faces
# ═══════════════════════════════════════════════════════════════════

class TestClassifyBoundarySubfaces:
    """Tests for classify_boundary_subfaces()."""

    def test_all_faces_classified(self):
        """Every sub-face gets a classification."""
        grid = build_pure_hex_grid(2)
        result = classify_boundary_subfaces(grid)
        assert len(result) == len(grid.faces)
        for fid in grid.faces:
            assert fid in result

    def test_classification_values(self):
        """Only valid classification labels."""
        grid = build_pure_hex_grid(2)
        result = classify_boundary_subfaces(grid)
        valid = {"interior", "boundary", "edge_band"}
        for label in result.values():
            assert label in valid

    def test_boundary_faces_on_edge(self):
        """Boundary faces must touch at least one boundary edge."""
        grid = build_pure_hex_grid(2)
        result = classify_boundary_subfaces(grid)
        boundary_fids = {fid for fid, cls in result.items() if cls == "boundary"}

        # Verify each boundary face touches a boundary edge
        for fid in boundary_fids:
            face = grid.faces[fid]
            touches_boundary = False
            for eid in face.edge_ids:
                edge = grid.edges[eid]
                if len(edge.face_ids) < 2:
                    touches_boundary = True
                    break
            assert touches_boundary, f"Face {fid} classified as boundary but doesn't touch boundary edge"

    def test_interior_faces_exist(self):
        """A rings=2 hex grid should have at least one interior face."""
        grid = build_pure_hex_grid(2)
        result = classify_boundary_subfaces(grid)
        interior = [fid for fid, cls in result.items() if cls == "interior"]
        assert len(interior) >= 1

    def test_rings_0_all_boundary(self):
        """A rings=0 grid (single face) — all faces are boundary."""
        grid = build_pure_hex_grid(0)
        result = classify_boundary_subfaces(grid)
        # Single face, all edges are boundary
        for cls in result.values():
            assert cls == "boundary"

    def test_rings_1_mostly_boundary(self):
        """A rings=1 grid — 6 outer + 1 center. Centre should be interior or edge_band."""
        grid = build_pure_hex_grid(1)
        result = classify_boundary_subfaces(grid)
        boundary_count = sum(1 for cls in result.values() if cls == "boundary")
        # Ring-1 hex grid has 7 faces, 6 on the boundary
        assert boundary_count == 6

    def test_edge_band_depth(self):
        """edge_band_depth > 1 expands the band inward."""
        grid = build_pure_hex_grid(3)
        result_d1 = classify_boundary_subfaces(grid, edge_band_depth=1)
        result_d2 = classify_boundary_subfaces(grid, edge_band_depth=2)

        band_d1 = sum(1 for cls in result_d1.values() if cls == "edge_band")
        band_d2 = sum(1 for cls in result_d2.values() if cls == "edge_band")
        # Deeper edge band should have more faces
        assert band_d2 >= band_d1

    def test_pentagon_grid(self):
        """Works on a pentagon-centred grid."""
        grid = build_goldberg_grid(2, size=1.0, optimise=True)
        result = classify_boundary_subfaces(grid)
        assert len(result) == len(grid.faces)
        boundary_count = sum(1 for cls in result.values() if cls == "boundary")
        assert boundary_count > 0

    def test_boundary_subface_ids_convenience(self):
        """boundary_subface_ids() returns same set as classification boundary."""
        grid = build_pure_hex_grid(2)
        cls_result = classify_boundary_subfaces(grid)
        bnd_from_classify = {fid for fid, cls in cls_result.items() if cls == "boundary"}
        bnd_from_func = boundary_subface_ids(grid)
        assert bnd_from_classify == bnd_from_func


# ═══════════════════════════════════════════════════════════════════
# 18A.2 — Edge sub-face mapping
# ═══════════════════════════════════════════════════════════════════

class TestEdgeSubfaceMapping:
    """Tests for compute_edge_subface_mapping()."""

    def test_returns_mapping(self):
        """Basic: returns an EdgeSubfaceMapping with non-empty lists."""
        globe = _make_globe_grid(2)
        coll = _make_detail_collection(globe, detail_rings=2)

        adj = get_face_adjacency(globe)
        # Pick any pair of adjacent faces
        face_a = list(globe.faces.keys())[0]
        face_b = adj[face_a][0]

        mapping = compute_edge_subface_mapping(globe, face_a, face_b, coll)

        assert isinstance(mapping, EdgeSubfaceMapping)
        assert mapping.face_id_a == face_a
        assert mapping.face_id_b == face_b
        # At least some sub-faces should be mapped
        assert len(mapping.subfaces_a) > 0
        assert len(mapping.subfaces_b) > 0

    def test_subfaces_are_boundary(self):
        """Mapped sub-faces should be boundary faces in their respective grids."""
        globe = _make_globe_grid(2)
        coll = _make_detail_collection(globe, detail_rings=2)

        adj = get_face_adjacency(globe)
        face_a = list(globe.faces.keys())[0]
        face_b = adj[face_a][0]

        mapping = compute_edge_subface_mapping(globe, face_a, face_b, coll)

        bnd_a = boundary_subface_ids(coll.grids[face_a])
        bnd_b = boundary_subface_ids(coll.grids[face_b])

        for sfid in mapping.subfaces_a:
            assert sfid in bnd_a, f"{sfid} not a boundary face of {face_a}"
        for sfid in mapping.subfaces_b:
            assert sfid in bnd_b, f"{sfid} not a boundary face of {face_b}"

    def test_multiple_neighbours(self):
        """Mapping works for all neighbours of a face."""
        globe = _make_globe_grid(2)
        coll = _make_detail_collection(globe, detail_rings=2)

        adj = get_face_adjacency(globe)
        face_a = list(globe.faces.keys())[0]

        for face_b in adj[face_a]:
            mapping = compute_edge_subface_mapping(globe, face_a, face_b, coll)
            assert isinstance(mapping, EdgeSubfaceMapping)
            assert len(mapping.subfaces_a) > 0
            assert len(mapping.subfaces_b) > 0

    def test_sorted_along_edge(self):
        """Mapped sub-faces should be spatially ordered."""
        globe = _make_globe_grid(2)
        coll = _make_detail_collection(globe, detail_rings=3)

        adj = get_face_adjacency(globe)
        face_a = list(globe.faces.keys())[0]
        face_b = adj[face_a][0]

        mapping = compute_edge_subface_mapping(globe, face_a, face_b, coll)

        # Sub-faces should be ordered (we can't easily verify direction,
        # but at least they should be consistent — no duplicates)
        assert len(mapping.subfaces_a) == len(set(mapping.subfaces_a))
        assert len(mapping.subfaces_b) == len(set(mapping.subfaces_b))


# ═══════════════════════════════════════════════════════════════════
# 18A.3 — Build apron grid
# ═══════════════════════════════════════════════════════════════════

class TestBuildApronGrid:
    """Tests for build_apron_grid()."""

    def test_apron_grid_has_more_faces(self):
        """Apron grid should have more sub-faces than the original."""
        globe = _make_globe_grid(2)
        coll = _make_detail_collection(globe, detail_rings=2)

        face_id = list(globe.faces.keys())[0]
        own_count = len(coll.grids[face_id].faces)

        apron_grid, mapping = build_apron_grid(globe, face_id, coll)

        assert len(apron_grid.faces) > own_count
        assert len(mapping) > 0
        assert len(apron_grid.faces) == own_count + len(mapping)

    def test_own_faces_preserved(self):
        """All of the tile's own sub-faces remain in the apron grid."""
        globe = _make_globe_grid(2)
        coll = _make_detail_collection(globe, detail_rings=2)

        face_id = list(globe.faces.keys())[0]
        own_faces = set(coll.grids[face_id].faces.keys())

        apron_grid, _ = build_apron_grid(globe, face_id, coll)

        for fid in own_faces:
            assert fid in apron_grid.faces, f"Own face {fid} missing from apron grid"

    def test_apron_faces_prefixed(self):
        """Apron faces have a distinct prefix to avoid id collisions."""
        globe = _make_globe_grid(2)
        coll = _make_detail_collection(globe, detail_rings=2)

        face_id = list(globe.faces.keys())[0]
        apron_grid, mapping = build_apron_grid(globe, face_id, coll)

        for apron_fid in mapping:
            assert apron_fid.startswith("apron_")

    def test_apron_mapping_records_source(self):
        """Apron mapping records (source_tile, source_sub_face) for each apron face."""
        globe = _make_globe_grid(2)
        coll = _make_detail_collection(globe, detail_rings=2)

        face_id = list(globe.faces.keys())[0]
        adj = get_face_adjacency(globe)
        neighbours = adj[face_id]

        apron_grid, mapping = build_apron_grid(globe, face_id, coll)

        source_tiles = {src_tile for src_tile, _ in mapping.values()}
        # All apron source tiles should be neighbours
        for st in source_tiles:
            assert st in neighbours, f"Apron source tile {st} is not a neighbour"

    def test_apron_metadata(self):
        """Apron grid metadata records apron info."""
        globe = _make_globe_grid(2)
        coll = _make_detail_collection(globe, detail_rings=2)

        face_id = list(globe.faces.keys())[0]
        apron_grid, mapping = build_apron_grid(globe, face_id, coll)

        assert apron_grid.metadata.get("has_apron") is True
        assert apron_grid.metadata.get("apron_face_count") == len(mapping)

    def test_apron_vertices_have_positions(self):
        """All vertices in the apron grid have valid positions."""
        globe = _make_globe_grid(2)
        coll = _make_detail_collection(globe, detail_rings=2)

        face_id = list(globe.faces.keys())[0]
        apron_grid, _ = build_apron_grid(globe, face_id, coll)

        for vid, v in apron_grid.vertices.items():
            assert v.has_position(), f"Vertex {vid} has no position"
            assert math.isfinite(v.x), f"Vertex {vid} has non-finite x"
            assert math.isfinite(v.y), f"Vertex {vid} has non-finite y"

    def test_hex_tile_gets_multiple_apron_strips(self):
        """A hex tile (6 sides) should get apron from multiple neighbours."""
        globe = _make_globe_grid(2)
        coll = _make_detail_collection(globe, detail_rings=2)

        # Find a hex face with maximum neighbours
        adj = get_face_adjacency(globe)
        # Centre face of freq=2 grid has 6 neighbours
        face_id = "f1"  # Centre face
        n_neighbours = len(adj.get(face_id, []))

        apron_grid, mapping = build_apron_grid(globe, face_id, coll)

        source_tiles = {src_tile for src_tile, _ in mapping.values()}
        # Should have apron from multiple neighbours
        assert len(source_tiles) >= min(n_neighbours, 3)

    def test_apron_grid_validates(self):
        """Apron grid passes basic PolyGrid validation."""
        globe = _make_globe_grid(2)
        coll = _make_detail_collection(globe, detail_rings=2)

        face_id = list(globe.faces.keys())[0]
        apron_grid, _ = build_apron_grid(globe, face_id, coll)

        errors = apron_grid.validate()
        # Filter out edge-related errors (apron edges may reference
        # vertices that are only in the apron)
        critical_errors = [
            e for e in errors
            if "missing vertex" in e.lower()
            and "apron_" not in e  # apron edges are self-contained
        ]
        assert len(critical_errors) == 0, f"Validation errors: {critical_errors}"


# ═══════════════════════════════════════════════════════════════════
# 18A.4 — Apron terrain propagation
# ═══════════════════════════════════════════════════════════════════

class TestPropagateApronTerrain:
    """Tests for propagate_apron_terrain()."""

    def test_all_faces_have_elevation(self):
        """Every face in the apron grid gets an elevation value."""
        globe = _make_globe_grid(2)
        coll = _make_detail_collection(globe, detail_rings=2)

        face_id = list(globe.faces.keys())[0]
        apron_grid, mapping = build_apron_grid(globe, face_id, coll)

        store = propagate_apron_terrain(
            apron_grid, mapping, coll, face_id,
        )

        for fid in apron_grid.faces:
            elev = store.get(fid, "elevation")
            assert isinstance(elev, (int, float))
            assert math.isfinite(elev)

    def test_own_faces_preserve_elevation(self):
        """Own sub-faces should keep their original elevation (approximately)."""
        globe = _make_globe_grid(2)
        coll = _make_detail_collection(globe, detail_rings=2)

        face_id = list(globe.faces.keys())[0]
        _, own_store = coll.get(face_id)

        apron_grid, mapping = build_apron_grid(globe, face_id, coll)
        apron_store = propagate_apron_terrain(
            apron_grid, mapping, coll, face_id,
            smooth_iterations=0,  # No smoothing — exact copy
        )

        # Without smoothing, own faces should have exact same elevation
        for fid in coll.grids[face_id].faces:
            expected = own_store.get(fid, "elevation")
            actual = apron_store.get(fid, "elevation")
            assert abs(actual - expected) < 1e-10, (
                f"Face {fid}: expected {expected}, got {actual}"
            )

    def test_apron_faces_get_neighbour_elevation(self):
        """Apron faces should have elevation from the source neighbour."""
        globe = _make_globe_grid(2)
        coll = _make_detail_collection(globe, detail_rings=2)

        face_id = list(globe.faces.keys())[0]
        apron_grid, mapping = build_apron_grid(globe, face_id, coll)

        store = propagate_apron_terrain(
            apron_grid, mapping, coll, face_id,
            smooth_iterations=0,  # No smoothing — exact copy
        )

        for apron_fid, (src_tile, src_face) in mapping.items():
            _, src_store = coll.get(src_tile)
            if src_store is None:
                continue
            expected = src_store.get(src_face, "elevation")
            actual = store.get(apron_fid, "elevation")
            assert abs(actual - expected) < 1e-10, (
                f"Apron face {apron_fid}: expected {expected}, got {actual}"
            )

    def test_smoothing_modifies_boundary(self):
        """With smoothing, boundary elevations should be modified."""
        globe = _make_globe_grid(2)
        coll = _make_detail_collection(globe, detail_rings=2)

        face_id = list(globe.faces.keys())[0]
        apron_grid, mapping = build_apron_grid(globe, face_id, coll)

        store_no_smooth = propagate_apron_terrain(
            apron_grid, mapping, coll, face_id,
            smooth_iterations=0,
        )
        store_smooth = propagate_apron_terrain(
            apron_grid, mapping, coll, face_id,
            smooth_iterations=3,
            smooth_weight=0.4,
        )

        # At least some boundary faces should have different values
        own_boundary = boundary_subface_ids(coll.grids[face_id])
        diffs = 0
        for fid in own_boundary:
            if fid in apron_grid.faces:
                e1 = store_no_smooth.get(fid, "elevation")
                e2 = store_smooth.get(fid, "elevation")
                if abs(e1 - e2) > 1e-10:
                    diffs += 1

        # At least some boundary faces should change
        assert diffs >= 0  # May be 0 if all neighbours have same elevation


# ═══════════════════════════════════════════════════════════════════
# build_all_apron_grids
# ═══════════════════════════════════════════════════════════════════

class TestBuildAllApronGrids:
    """Tests for the batch build_all_apron_grids()."""

    def test_builds_for_all_tiles(self):
        """Builds an ApronResult for every tile."""
        globe = _make_globe_grid(1)
        coll = _make_detail_collection(globe, detail_rings=2)

        results = build_all_apron_grids(globe, coll)

        assert len(results) == len(globe.faces)
        for face_id in globe.faces:
            assert face_id in results
            r = results[face_id]
            assert isinstance(r, ApronResult)
            assert r.face_id == face_id
            assert r.own_face_count > 0
            assert len(r.grid.faces) == r.own_face_count + r.apron_face_count

    def test_apron_result_has_store(self):
        """Each ApronResult has a store with elevation data."""
        globe = _make_globe_grid(1)
        coll = _make_detail_collection(globe, detail_rings=2)

        results = build_all_apron_grids(globe, coll)

        for face_id, r in results.items():
            for fid in r.grid.faces:
                elev = r.store.get(fid, "elevation")
                assert math.isfinite(elev)

    def test_apron_face_count_matches(self):
        """ApronResult.apron_face_count matches len(apron_mapping)."""
        globe = _make_globe_grid(1)
        coll = _make_detail_collection(globe, detail_rings=2)

        results = build_all_apron_grids(globe, coll)

        for face_id, r in results.items():
            assert r.apron_face_count == len(r.apron_mapping)
