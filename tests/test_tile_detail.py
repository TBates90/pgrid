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
