"""Tests for the tile data layer (Phase 5)."""

import json
import math

import pytest

from polygrid.builders import build_pure_hex_grid, build_pentagon_centered_grid
from polygrid.tile_data import (
    FieldDef,
    TileSchema,
    TileData,
    TileDataStore,
    save_tile_data,
    load_tile_data,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def terrain_schema():
    return TileSchema([
        FieldDef("elevation", float, 0.0),
        FieldDef("biome", str, "none"),
        FieldDef("moisture", float, 0.5),
    ])


@pytest.fixture
def hex_grid():
    return build_pure_hex_grid(2)


@pytest.fixture
def pent_grid():
    return build_pentagon_centered_grid(2)


@pytest.fixture
def store(hex_grid, terrain_schema):
    return TileDataStore(hex_grid, schema=terrain_schema)


# ═══════════════════════════════════════════════════════════════════
# TileSchema tests
# ═══════════════════════════════════════════════════════════════════

class TestTileSchema:

    def test_field_names(self, terrain_schema):
        assert terrain_schema.field_names == ["elevation", "biome", "moisture"]

    def test_has_field(self, terrain_schema):
        assert terrain_schema.has_field("elevation")
        assert not terrain_schema.has_field("temperature")

    def test_get_field(self, terrain_schema):
        fd = terrain_schema.get_field("biome")
        assert fd.name == "biome"
        assert fd.dtype is str
        assert fd.default == "none"

    def test_get_field_missing_raises(self, terrain_schema):
        with pytest.raises(KeyError):
            terrain_schema.get_field("nonexistent")

    def test_validate_value_ok(self, terrain_schema):
        terrain_schema.validate_value("elevation", 1.5)
        terrain_schema.validate_value("biome", "forest")

    def test_validate_value_wrong_type(self, terrain_schema):
        with pytest.raises(TypeError, match="expects float"):
            terrain_schema.validate_value("elevation", "high")

    def test_validate_value_unknown_field(self, terrain_schema):
        with pytest.raises(KeyError):
            terrain_schema.validate_value("nonexistent", 42)

    def test_default_record(self, terrain_schema):
        defaults = terrain_schema.default_record()
        assert defaults == {"elevation": 0.0, "biome": "none", "moisture": 0.5}

    def test_duplicate_field_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            TileSchema([FieldDef("x", float, 0.0), FieldDef("x", float, 1.0)])

    def test_unsupported_dtype_raises(self):
        with pytest.raises(TypeError, match="Unsupported dtype"):
            FieldDef("bad", list)

    def test_default_type_mismatch_raises(self):
        with pytest.raises(TypeError, match="must be float"):
            FieldDef("x", float, "oops")

    def test_field_without_default(self):
        fd = FieldDef("required_field", int)
        assert fd.default is None

    def test_schema_round_trip(self, terrain_schema):
        d = terrain_schema.to_dict()
        restored = TileSchema.from_dict(d)
        assert restored.field_names == terrain_schema.field_names
        for name in terrain_schema.field_names:
            orig = terrain_schema.get_field(name)
            rest = restored.get_field(name)
            assert orig.name == rest.name
            assert orig.dtype == rest.dtype
            assert orig.default == rest.default

    def test_repr(self, terrain_schema):
        r = repr(terrain_schema)
        assert "elevation" in r
        assert "TileSchema" in r


# ═══════════════════════════════════════════════════════════════════
# TileData tests
# ═══════════════════════════════════════════════════════════════════

class TestTileData:

    def test_set_and_get(self, terrain_schema):
        td = TileData(terrain_schema)
        td.set("f1", "elevation", 10.0)
        assert td.get("f1", "elevation") == 10.0

    def test_get_default(self, terrain_schema):
        td = TileData(terrain_schema)
        assert td.get("f1", "elevation") == 0.0

    def test_get_no_default_raises(self):
        schema = TileSchema([FieldDef("required", int)])
        td = TileData(schema)
        with pytest.raises(KeyError, match="no value.*no default"):
            td.get("f1", "required")

    def test_set_wrong_type_raises(self, terrain_schema):
        td = TileData(terrain_schema)
        with pytest.raises(TypeError):
            td.set("f1", "elevation", "high")

    def test_set_unknown_field_raises(self, terrain_schema):
        td = TileData(terrain_schema)
        with pytest.raises(KeyError):
            td.set("f1", "temperature", 20.0)

    def test_bulk_set(self, terrain_schema):
        td = TileData(terrain_schema)
        td.bulk_set(["f1", "f2", "f3"], "biome", "desert")
        assert td.get("f1", "biome") == "desert"
        assert td.get("f2", "biome") == "desert"
        assert td.get("f3", "biome") == "desert"

    def test_bulk_set_wrong_type_raises(self, terrain_schema):
        td = TileData(terrain_schema)
        with pytest.raises(TypeError):
            td.bulk_set(["f1"], "elevation", "high")

    def test_has(self, terrain_schema):
        td = TileData(terrain_schema)
        assert not td.has("f1", "elevation")
        td.set("f1", "elevation", 5.0)
        assert td.has("f1", "elevation")

    def test_clear_field(self, terrain_schema):
        td = TileData(terrain_schema)
        td.set("f1", "elevation", 5.0)
        td.set("f1", "biome", "forest")
        td.clear("f1", "elevation")
        assert not td.has("f1", "elevation")
        assert td.has("f1", "biome")

    def test_clear_all_fields(self, terrain_schema):
        td = TileData(terrain_schema)
        td.set("f1", "elevation", 5.0)
        td.set("f1", "biome", "forest")
        td.clear("f1")
        assert "f1" not in td.face_ids

    def test_clear_nonexistent_is_noop(self, terrain_schema):
        td = TileData(terrain_schema)
        td.clear("f999")  # should not raise
        td.clear("f999", "elevation")

    def test_face_ids(self, terrain_schema):
        td = TileData(terrain_schema)
        td.set("f1", "elevation", 1.0)
        td.set("f2", "elevation", 2.0)
        assert td.face_ids == {"f1", "f2"}

    def test_json_round_trip(self, terrain_schema):
        td = TileData(terrain_schema)
        td.set("f1", "elevation", 10.5)
        td.set("f1", "biome", "forest")
        td.set("f2", "moisture", 0.8)

        json_str = td.to_json()
        restored = TileData.from_json(json_str)

        assert restored.get("f1", "elevation") == 10.5
        assert restored.get("f1", "biome") == "forest"
        assert restored.get("f2", "moisture") == 0.8
        # Defaults still work
        assert restored.get("f2", "elevation") == 0.0

    def test_dict_round_trip(self, terrain_schema):
        td = TileData(terrain_schema)
        td.set("f1", "elevation", 3.0)

        d = td.to_dict()
        restored = TileData.from_dict(d)
        assert restored.get("f1", "elevation") == 3.0

    def test_json_coerces_int_to_float(self):
        """JSON doesn't distinguish int/float for whole numbers."""
        schema = TileSchema([FieldDef("height", float, 0.0)])
        td = TileData(schema)
        td.set("f1", "height", 5.0)

        # Simulate JSON round-trip (5.0 may become 5 in JSON)
        raw = json.loads(td.to_json())
        raw["tiles"]["f1"]["height"] = 5  # force int
        restored = TileData.from_dict(raw)
        assert isinstance(restored.get("f1", "height"), float)
        assert restored.get("f1", "height") == 5.0

    def test_repr(self, terrain_schema):
        td = TileData(terrain_schema)
        td.set("f1", "elevation", 1.0)
        r = repr(td)
        assert "TileData" in r
        assert "faces=1" in r


# ═══════════════════════════════════════════════════════════════════
# TileDataStore tests
# ═══════════════════════════════════════════════════════════════════

class TestTileDataStore:

    def test_create_with_schema(self, hex_grid, terrain_schema):
        store = TileDataStore(hex_grid, schema=terrain_schema)
        assert store.schema is terrain_schema

    def test_create_with_tile_data(self, hex_grid, terrain_schema):
        td = TileData(terrain_schema)
        td.set("f1", "elevation", 5.0)
        store = TileDataStore(hex_grid, tile_data=td)
        assert store.get("f1", "elevation") == 5.0

    def test_create_requires_schema_or_data(self, hex_grid):
        with pytest.raises(ValueError, match="Either"):
            TileDataStore(hex_grid)

    def test_get_set(self, store, hex_grid):
        fid = next(iter(hex_grid.faces))
        store.set(fid, "elevation", 42.0)
        assert store.get(fid, "elevation") == 42.0

    def test_bulk_set(self, store, hex_grid):
        fids = list(hex_grid.faces.keys())[:3]
        store.bulk_set(fids, "biome", "tundra")
        for fid in fids:
            assert store.get(fid, "biome") == "tundra"

    def test_initialise_all(self, store, hex_grid):
        store.initialise_all()
        for fid in hex_grid.faces:
            assert store.get(fid, "elevation") == 0.0
            assert store.get(fid, "biome") == "none"
            assert store.get(fid, "moisture") == 0.5

    def test_initialise_all_preserves_existing(self, store, hex_grid):
        fid = next(iter(hex_grid.faces))
        store.set(fid, "elevation", 99.0)
        store.initialise_all()
        # Existing value should NOT be overwritten
        assert store.get(fid, "elevation") == 99.0

    def test_initialise_all_raises_if_no_default(self, hex_grid):
        schema = TileSchema([FieldDef("required", int)])
        store = TileDataStore(hex_grid, schema=schema)
        with pytest.raises(ValueError, match="Cannot initialise"):
            store.initialise_all()

    # ── neighbour queries ───────────────────────────────────────────

    def test_get_neighbors_data(self, hex_grid, terrain_schema):
        store = TileDataStore(hex_grid, schema=terrain_schema)
        store.initialise_all()

        # Pick a face and set its elevation
        fid = next(iter(hex_grid.faces))
        store.set(fid, "elevation", 100.0)

        # Pick a neighbour and query
        adj = hex_grid.compute_face_neighbors()
        neighbor_id = adj[fid][0]
        neighbors_data = store.get_neighbors_data(neighbor_id, "elevation")

        # The original face should appear in the neighbour list
        neighbor_ids = [nid for nid, _ in neighbors_data]
        assert fid in neighbor_ids

        # And its value should be 100.0
        for nid, val in neighbors_data:
            if nid == fid:
                assert val == 100.0

    def test_get_neighbors_data_returns_defaults(self, hex_grid, terrain_schema):
        store = TileDataStore(hex_grid, schema=terrain_schema)
        fid = next(iter(hex_grid.faces))
        neighbors_data = store.get_neighbors_data(fid, "elevation")
        # All should be the default 0.0
        for _, val in neighbors_data:
            assert val == 0.0

    # ── ring queries ────────────────────────────────────────────────

    def test_get_ring_data(self, hex_grid, terrain_schema):
        store = TileDataStore(hex_grid, schema=terrain_schema)
        store.initialise_all()

        fid = next(iter(hex_grid.faces))
        store.set(fid, "elevation", 50.0)

        ring_data = store.get_ring_data(fid, radius=1, key="elevation")

        # Ring 0 is the centre face
        assert len(ring_data[0]) == 1
        assert ring_data[0][0] == (fid, 50.0)

        # Ring 1 should have neighbours
        assert len(ring_data[1]) > 0

    def test_get_ring_data_radius_zero(self, store, hex_grid):
        fid = next(iter(hex_grid.faces))
        store.set(fid, "elevation", 7.0)
        ring_data = store.get_ring_data(fid, radius=0, key="elevation")
        assert 0 in ring_data
        assert ring_data[0] == [(fid, 7.0)]

    # ── bulk operations ─────────────────────────────────────────────

    def test_apply_to_all(self, store, hex_grid):
        store.initialise_all()
        store.apply_to_all("elevation", lambda fid, val: val + 10.0)
        for fid in hex_grid.faces:
            assert store.get(fid, "elevation") == 10.0

    def test_apply_to_ring(self, store, hex_grid):
        store.initialise_all()
        center = next(iter(hex_grid.faces))
        store.apply_to_ring(center, radius=1, key="elevation",
                            fn=lambda fid, val: 99.0)

        # Centre + ring-1 should be 99.0
        adj = hex_grid.compute_face_neighbors()
        assert store.get(center, "elevation") == 99.0
        for nid in adj[center]:
            assert store.get(nid, "elevation") == 99.0

    def test_apply_to_faces(self, store, hex_grid):
        store.initialise_all()
        fids = list(hex_grid.faces.keys())[:3]
        store.apply_to_faces(fids, "biome", lambda fid, val: "mountain")
        for fid in fids:
            assert store.get(fid, "biome") == "mountain"

    # ── serialisation ───────────────────────────────────────────────

    def test_store_json_round_trip(self, hex_grid, terrain_schema):
        store = TileDataStore(hex_grid, schema=terrain_schema)
        store.initialise_all()
        fid = next(iter(hex_grid.faces))
        store.set(fid, "elevation", 25.0)

        json_str = store.to_json()
        restored = TileDataStore.from_json(hex_grid, json_str)

        assert restored.get(fid, "elevation") == 25.0
        assert restored.schema.field_names == terrain_schema.field_names

    def test_store_dict_round_trip(self, hex_grid, terrain_schema):
        store = TileDataStore(hex_grid, schema=terrain_schema)
        store.initialise_all()

        d = store.to_dict()
        restored = TileDataStore.from_dict(hex_grid, d)

        for fid in hex_grid.faces:
            assert restored.get(fid, "elevation") == store.get(fid, "elevation")

    def test_repr(self, store):
        r = repr(store)
        assert "TileDataStore" in r


# ═══════════════════════════════════════════════════════════════════
# File I/O tests
# ═══════════════════════════════════════════════════════════════════

class TestFileIO:

    def test_save_and_load(self, tmp_path, terrain_schema):
        td = TileData(terrain_schema)
        td.set("f1", "elevation", 15.5)
        td.set("f1", "biome", "jungle")

        path = tmp_path / "tile_data.json"
        save_tile_data(td, path)

        loaded = load_tile_data(path)
        assert loaded.get("f1", "elevation") == 15.5
        assert loaded.get("f1", "biome") == "jungle"
        assert loaded.get("f1", "moisture") == 0.5  # default

    def test_saved_file_is_valid_json(self, tmp_path, terrain_schema):
        td = TileData(terrain_schema)
        td.set("f1", "elevation", 1.0)

        path = tmp_path / "tile_data.json"
        save_tile_data(td, path)

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert "schema" in raw
        assert "tiles" in raw


# ═══════════════════════════════════════════════════════════════════
# Integration: works with pentagon-centred grid too
# ═══════════════════════════════════════════════════════════════════

class TestPentGridIntegration:

    def test_store_on_pent_grid(self, pent_grid, terrain_schema):
        store = TileDataStore(pent_grid, schema=terrain_schema)
        store.initialise_all()

        # Every face should be accessible
        for fid in pent_grid.faces:
            assert store.get(fid, "elevation") == 0.0

        # Neighbour queries work
        fid = next(iter(pent_grid.faces))
        neighbors = store.get_neighbors_data(fid, "biome")
        assert len(neighbors) > 0

    def test_ring_query_on_pent_grid(self, pent_grid, terrain_schema):
        store = TileDataStore(pent_grid, schema=terrain_schema)
        store.initialise_all()

        fid = next(iter(pent_grid.faces))
        ring_data = store.get_ring_data(fid, radius=2, key="elevation")
        assert 0 in ring_data
        # For a 2-ring pent grid, should reach ring 2
        if len(pent_grid.faces) > 1:
            assert len(ring_data) >= 2
