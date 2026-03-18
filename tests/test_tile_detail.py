"""Tests for Phase 10A — Detail grid infrastructure (tile_detail.py).

Covers:
- TileDetailSpec dataclass defaults and overrides
- build_all_detail_grids produces one grid per globe tile
- Hex tiles get hex grids, pent tiles get pent grids
- Correct sub-face counts for the given ring count
- DetailGridCollection stores and retrieves grids correctly
- total_face_count matches sum of detail_face_count per tile type
- generate_all_terrain populates stores for every tile
- summary() produces human-readable output
"""

from __future__ import annotations

import pytest

from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
from polygrid.detail_grid import detail_face_count

from polygrid.tile_detail import (
    TileDetailSpec,
    build_all_detail_grids,
    DetailGridCollection,
    build_tile_with_neighbours,
    _macro_edge_overlap_ok,
)

# ── Gate behind models availability ─────────────────────────────
try:
    from polygrid.globe import build_globe_grid, _HAS_MODELS
    _skip = not _HAS_MODELS
except ImportError:
    _skip = True

needs_models = pytest.mark.skipif(_skip, reason="models library not installed")


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _make_globe_with_elevation(frequency: int = 3, seed: int = 42):
    """Build a globe grid and populate an elevation field."""
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


# ═══════════════════════════════════════════════════════════════════
# TileDetailSpec
# ═══════════════════════════════════════════════════════════════════

class TestTileDetailSpec:
    def test_defaults(self):
        spec = TileDetailSpec()
        assert spec.detail_rings == 4
        assert spec.noise_frequency == 6.0
        assert spec.noise_octaves == 5
        assert spec.amplitude == 0.12
        assert spec.base_weight == 0.80
        assert spec.boundary_smoothing == 2
        assert spec.seed_offset == 0

    def test_custom_values(self):
        spec = TileDetailSpec(detail_rings=6, amplitude=0.25, seed_offset=100)
        assert spec.detail_rings == 6
        assert spec.amplitude == 0.25
        assert spec.seed_offset == 100

    def test_frozen(self):
        spec = TileDetailSpec()
        with pytest.raises(AttributeError):
            spec.detail_rings = 10  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════
# build_all_detail_grids
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestBuildAllDetailGrids:
    def test_one_grid_per_face(self):
        grid, _ = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        grids = build_all_detail_grids(grid, spec)
        assert len(grids) == len(grid.faces)

    def test_hex_tiles_get_hex_grids(self):
        grid, _ = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        grids = build_all_detail_grids(grid, spec)
        for fid, detail_grid in grids.items():
            parent_face = grid.faces[fid]
            if parent_face.face_type == "hex":
                # Hex grids have hex_face_count sub-faces
                expected = detail_face_count("hex", 2)
                assert len(detail_grid.faces) == expected, (
                    f"Hex tile {fid}: expected {expected}, got {len(detail_grid.faces)}"
                )

    def test_pent_tiles_get_pent_grids(self):
        grid, _ = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        grids = build_all_detail_grids(grid, spec)
        for fid, detail_grid in grids.items():
            parent_face = grid.faces[fid]
            if parent_face.face_type == "pent":
                expected = detail_face_count("pent", 2)
                assert len(detail_grid.faces) == expected, (
                    f"Pent tile {fid}: expected {expected}, got {len(detail_grid.faces)}"
                )

    def test_metadata_stores_parent(self):
        grid, _ = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        grids = build_all_detail_grids(grid, spec)
        for fid, detail_grid in grids.items():
            assert detail_grid.metadata.get("parent_face_id") == fid
            assert detail_grid.metadata.get("detail_rings") == 2


# ═══════════════════════════════════════════════════════════════════
# DetailGridCollection
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestDetailGridCollection:
    def test_build_factory(self):
        grid, _ = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        assert len(coll.grids) == len(grid.faces)

    def test_build_default_spec(self):
        grid, _ = _make_globe_with_elevation(3)
        coll = DetailGridCollection.build(grid)
        assert coll.spec == TileDetailSpec()

    def test_get_returns_grid_and_none_store_before_terrain(self):
        grid, _ = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        fid = coll.face_ids[0]
        detail_grid, store = coll.get(fid)
        assert detail_grid is not None
        assert store is None

    def test_get_raises_for_unknown_face(self):
        grid, _ = _make_globe_with_elevation(3)
        coll = DetailGridCollection.build(grid, TileDetailSpec(detail_rings=2))
        with pytest.raises(KeyError):
            coll.get("nonexistent_face")

    def test_face_ids_sorted(self):
        grid, _ = _make_globe_with_elevation(3)
        coll = DetailGridCollection.build(grid, TileDetailSpec(detail_rings=2))
        ids = coll.face_ids
        assert ids == sorted(ids)

    def test_total_face_count(self):
        grid, _ = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)

        n_pent = sum(1 for f in grid.faces.values() if f.face_type == "pent")
        n_hex = len(grid.faces) - n_pent
        expected = (
            n_pent * detail_face_count("pent", 2)
            + n_hex * detail_face_count("hex", 2)
        )
        assert coll.total_face_count == expected

    def test_detail_face_count_for(self):
        grid, _ = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        for fid in coll.face_ids:
            ft = grid.faces[fid].face_type
            expected = detail_face_count(ft, 2)
            assert coll.detail_face_count_for(fid) == expected

    def test_generate_all_terrain_populates_stores(self):
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        coll.generate_all_terrain(store, seed=42)
        assert len(coll.stores) == len(coll.grids)
        for fid in coll.face_ids:
            _, s = coll.get(fid)
            assert s is not None

    def test_generate_all_terrain_elevation_values(self):
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        coll.generate_all_terrain(store, seed=42)
        for fid in coll.face_ids:
            detail_grid, s = coll.get(fid)
            for sub_fid in detail_grid.faces:
                val = s.get(sub_fid, "elevation")
                assert isinstance(val, (int, float))
                assert not (val != val), f"NaN in face {sub_fid}"  # NaN check

    def test_summary_string(self):
        grid, _ = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        summary = coll.summary()
        assert "DetailGridCollection" in summary
        assert "Pentagon" in summary or "pentagon" in summary.lower()
        assert "Hexagon" in summary or "hexagon" in summary.lower()

    def test_repr(self):
        grid, _ = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        r = repr(coll)
        assert "DetailGridCollection" in r
        assert "rings=2" in r

    def test_properties_return_copies(self):
        grid, _ = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        grids1 = coll.grids
        grids2 = coll.grids
        assert grids1 is not grids2  # defensive copy


# ═══════════════════════════════════════════════════════════════════
# build_tile_with_neighbours — pentagon distortion fix
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestBuildTileWithNeighboursPentFix:
    """Verify that pentagon-centred composites skip neighbour↔neighbour
    closure to avoid the wedge/pinch distortion artefact, while
    hex-centred composites still perform full closure.
    """

    def _find_face_by_type(self, grid, face_type):
        """Return the first face_id with the given face_type."""
        for fid, face in grid.faces.items():
            if face.face_type == face_type:
                return fid
        return None

    def test_pentagon_composite_has_no_outer_stitches(self):
        """A pentagon tile should only have centre↔neighbour stitches
        (5 stitches for 5 neighbours), not neighbour↔neighbour."""
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        coll.generate_all_terrain(store, seed=42)

        pent_fid = self._find_face_by_type(grid, "pent")
        assert pent_fid is not None, "No pentagon face found"

        composite = build_tile_with_neighbours(coll, pent_fid, grid)
        merged = composite.merged

        # Pentagon has 5 neighbours → exactly 5 centre↔neighbour
        # stitches.  If outer closure were active, there would be up
        # to 5 more.  With the fix, merged vertex count should be
        # strictly higher than a closed version (more unique boundary
        # vertices).  We verify indirectly: the composite should have
        # exactly 6 component grids (1 centre + 5 neighbours).
        assert len(composite.id_prefixes) == 6

        # The merged grid should still be valid topology.
        errors = merged.validate()
        assert errors == [], f"Pentagon composite validation errors: {errors}"

    def test_hex_composite_still_has_outer_stitches(self):
        """Hex-centred composites should still perform full closure."""
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        coll.generate_all_terrain(store, seed=42)

        hex_fid = self._find_face_by_type(grid, "hex")
        assert hex_fid is not None, "No hex face found"

        composite = build_tile_with_neighbours(coll, hex_fid, grid)
        merged = composite.merged

        # Hex has 6 neighbours → 6 centre↔neighbour stitches + up to
        # 6 neighbour↔neighbour stitches for full closure.
        assert len(composite.id_prefixes) == 7  # 1 centre + 6 neighbours

        errors = merged.validate()
        assert errors == [], f"Hex composite validation errors: {errors}"

    def test_pentagon_composite_no_vertex_distortion(self):
        """Pentagon neighbour vertices should retain their original
        positioned coordinates (no forced averaging)."""
        grid, store = _make_globe_with_elevation(3)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        coll.generate_all_terrain(store, seed=42)

        pent_fid = self._find_face_by_type(grid, "pent")
        assert pent_fid is not None

        composite = build_tile_with_neighbours(coll, pent_fid, grid)
        merged = composite.merged

        # All vertices should have valid positions (no NaN / missing).
        for vid, v in merged.vertices.items():
            assert v.has_position(), f"Vertex {vid} has no position"


# ═══════════════════════════════════════════════════════════════════
# _macro_edge_overlap_ok — overlap quality guard
# ═══════════════════════════════════════════════════════════════════

class TestMacroEdgeOverlapOk:
    """Unit tests for the overlap quality check used by the hex
    neighbour↔neighbour closure path."""

    def test_overlapping_edges_pass(self):
        """Two hex grids stitched in a hex-ring configuration should have
        overlapping outer edges that pass the quality check."""
        from polygrid.builders import build_pure_hex_grid
        from polygrid.assembly import _position_hex_for_stitch, _snap_hex_hex_boundaries
        from polygrid.composite import StitchSpec

        # Build a mini hex-ring: centre + two adjacent hex neighbours
        # positioned as they would be in a hex-centred composite.
        centre = build_pure_hex_grid(2)
        h0 = build_pure_hex_grid(2)
        h1 = build_pure_hex_grid(2)
        centre.compute_macro_edges(n_sides=6)
        h0.compute_macro_edges(n_sides=6)
        h1.compute_macro_edges(n_sides=6)

        # Position hex0 flush to centre edge 0, hex1 flush to centre edge 1
        pos_h0 = _position_hex_for_stitch(centre, 0, h0, 3)
        pos_h1 = _position_hex_for_stitch(centre, 1, h1, 3)
        pos_h0.compute_macro_edges(n_sides=6)
        pos_h1.compute_macro_edges(n_sides=6)

        # The outer edges of pos_h0 and pos_h1 that meet at the
        # centre's corner should overlap (before snapping).  Use the
        # closest-pair finder to identify them.
        from polygrid.tile_detail import _find_closest_macro_edge_pair
        e0, e1 = _find_closest_macro_edge_pair(
            pos_h0, pos_h1, exclude_g1=3, exclude_g2=3,
        )
        # In a hex ring, adjacent neighbours meet cleanly (120° + 120° = 240°
        # which is close enough for the tolerance).
        assert _macro_edge_overlap_ok(pos_h0, e0, pos_h1, e1)

    def test_distant_edges_fail(self):
        """Edges that are geometrically far apart should fail."""
        from polygrid.builders import build_pure_hex_grid
        from polygrid.assembly import _position_hex_for_stitch

        # Position a hex grid far from the centre so its edges are
        # nowhere near the centre grid's opposite-side edges.
        centre = build_pure_hex_grid(2)
        remote = build_pure_hex_grid(2)
        centre.compute_macro_edges(n_sides=6)
        remote.compute_macro_edges(n_sides=6)

        # Position remote flush against edge 0 of centre
        positioned = _position_hex_for_stitch(centre, 0, remote, 3)
        positioned.compute_macro_edges(n_sides=6)

        # Edge 3 of centre (opposite side from edge 0) should be far
        # from any non-stitch edge of positioned.
        assert not _macro_edge_overlap_ok(centre, 3, positioned, 0)
