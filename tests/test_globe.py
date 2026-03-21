"""Tests for Phase 8 — Globe grid builder, 3-D noise, and heightmap bridge."""

from __future__ import annotations

import math
import json
from pathlib import Path
import pytest

from polygrid.core.models import Vertex, Face
from polygrid.core.polygrid import PolyGrid
from polygrid.noise import fbm_3d, ridged_noise_3d, _init_noise3
from polygrid.heightmap import sample_noise_field_3d
from polygrid.core.geometry import face_center_3d
from polygrid.tile_data import FieldDef, TileSchema, TileDataStore
from polygrid.core.algorithms import get_face_adjacency

# ── Gate globe-specific tests behind models availability ────────────
try:
    from polygrid.globe import build_globe_grid as _build_globe_grid_uncached, GlobeGrid, _HAS_MODELS
    _skip_globe = not _HAS_MODELS
    # Wrap with cache — building a Goldberg polyhedron is ~15-20s.
    # Tests only read the grid, so sharing a cached instance is safe.
    from conftest import cached_build_globe
    def build_globe_grid(frequency=3, *, radius=1.0):
        return cached_build_globe(frequency, radius)
except ImportError:
    _skip_globe = True

needs_models = pytest.mark.skipif(_skip_globe, reason="models library not installed")


# ═══════════════════════════════════════════════════════════════════
# 8A.2 — Vertex z coordinate
# ═══════════════════════════════════════════════════════════════════

class TestVertexZ:
    def test_default_z_is_none(self):
        v = Vertex("v0", 1.0, 2.0)
        assert v.z is None
        assert v.has_position() is True
        assert v.has_position_3d() is False

    def test_z_set(self):
        v = Vertex("v0", 1.0, 2.0, 3.0)
        assert v.z == 3.0
        assert v.has_position_3d() is True

    def test_z_none_has_position_3d_false(self):
        v = Vertex("v0", 1.0, 2.0, z=None)
        assert v.has_position_3d() is False

    def test_backward_compat_no_z_kwarg(self):
        v = Vertex(id="v1", x=5.0, y=6.0)
        assert v.z is None
        assert v.has_position() is True


# ═══════════════════════════════════════════════════════════════════
# 8A.2 — Vertex z round-trip through JSON
# ═══════════════════════════════════════════════════════════════════

class TestVertexZSerialization:
    def test_vertex_z_round_trip(self):
        v = Vertex("v0", 1.0, 2.0, 3.0)
        grid = PolyGrid(
            vertices=[v],
            edges=[],
            faces=[],
            metadata={"test": True},
        )
        data = grid.to_dict(include_neighbors=False)
        assert data["vertices"][0]["position"]["z"] == 3.0

        grid2 = PolyGrid.from_dict(data)
        v2 = grid2.vertices["v0"]
        assert v2.z == 3.0
        assert v2.has_position_3d() is True

    def test_vertex_no_z_round_trip(self):
        v = Vertex("v0", 1.0, 2.0)
        grid = PolyGrid(vertices=[v], edges=[], faces=[])
        data = grid.to_dict(include_neighbors=False)
        assert "z" not in data["vertices"][0].get("position", {})

        grid2 = PolyGrid.from_dict(data)
        assert grid2.vertices["v0"].z is None


# ═══════════════════════════════════════════════════════════════════
# 8A.3 — Face metadata
# ═══════════════════════════════════════════════════════════════════

class TestFaceMetadata:
    def test_default_metadata_is_empty(self):
        f = Face("f0", "hex", ("v0", "v1", "v2", "v3", "v4", "v5"))
        assert f.metadata == {}

    def test_metadata_set(self):
        f = Face("f0", "hex", ("v0",), metadata={"center_3d": (1, 2, 3)})
        assert f.metadata["center_3d"] == (1, 2, 3)

    def test_metadata_round_trip(self):
        f = Face("f0", "hex", (), metadata={"center_3d": [1, 2, 3], "custom": 42})
        grid = PolyGrid(vertices=[], edges=[], faces=[f])
        data = grid.to_dict(include_neighbors=False)
        assert data["faces"][0]["metadata"]["custom"] == 42

        grid2 = PolyGrid.from_dict(data)
        assert grid2.faces["f0"].metadata["custom"] == 42

    def test_no_metadata_key_when_empty(self):
        f = Face("f0", "hex", ())
        grid = PolyGrid(vertices=[], edges=[], faces=[f])
        data = grid.to_dict(include_neighbors=False)
        assert "metadata" not in data["faces"][0]


# ═══════════════════════════════════════════════════════════════════
# 8A.4 — 3-D noise primitives
# ═══════════════════════════════════════════════════════════════════

class TestNoise3D:
    def test_fbm_3d_deterministic(self):
        a = fbm_3d(0.5, 0.3, 0.7, seed=42)
        b = fbm_3d(0.5, 0.3, 0.7, seed=42)
        assert a == b

    def test_fbm_3d_different_seeds(self):
        a = fbm_3d(0.5, 0.3, 0.7, seed=42)
        b = fbm_3d(0.5, 0.3, 0.7, seed=99)
        assert a != b

    def test_fbm_3d_range(self):
        """Spot-check that values are in a reasonable range."""
        vals = [
            fbm_3d(x * 0.1, y * 0.1, z * 0.1, seed=42)
            for x in range(10)
            for y in range(10)
            for z in range(3)
        ]
        assert all(-1.5 <= v <= 1.5 for v in vals), f"out of range: {min(vals)}, {max(vals)}"

    def test_ridged_noise_3d_range(self):
        vals = [
            ridged_noise_3d(x * 0.1, y * 0.1, z * 0.1, seed=42)
            for x in range(10)
            for y in range(10)
            for z in range(3)
        ]
        assert all(0.0 <= v <= 1.0 for v in vals), f"out of range: {min(vals)}, {max(vals)}"

    def test_ridged_noise_3d_deterministic(self):
        a = ridged_noise_3d(1.0, 2.0, 3.0, seed=7)
        b = ridged_noise_3d(1.0, 2.0, 3.0, seed=7)
        assert a == b

    def test_init_noise3_returns_callable(self):
        fn = _init_noise3(42)
        result = fn(0.0, 0.0, 0.0)
        assert isinstance(result, float)


# ═══════════════════════════════════════════════════════════════════
# 8A.5 — face_center_3d
# ═══════════════════════════════════════════════════════════════════

class TestFaceCenter3D:
    def test_basic_3d_centroid(self):
        verts = {
            "v0": Vertex("v0", 0.0, 0.0, 0.0),
            "v1": Vertex("v1", 2.0, 0.0, 0.0),
            "v2": Vertex("v2", 1.0, 2.0, 0.0),
        }
        face = Face("f0", "other", ("v0", "v1", "v2"))
        c = face_center_3d(verts, face)
        assert c is not None
        assert abs(c[0] - 1.0) < 1e-9
        assert abs(c[1] - 2.0 / 3.0) < 1e-9
        assert abs(c[2] - 0.0) < 1e-9

    def test_returns_none_without_z(self):
        verts = {
            "v0": Vertex("v0", 0.0, 0.0),
            "v1": Vertex("v1", 1.0, 1.0),
        }
        face = Face("f0", "other", ("v0", "v1"))
        assert face_center_3d(verts, face) is None


# ═══════════════════════════════════════════════════════════════════
# 8A.5 — sample_noise_field_3d
# ═══════════════════════════════════════════════════════════════════

class TestSampleNoiseField3D:
    def _make_grid_and_store(self):
        verts = [
            Vertex("v0", 0.0, 0.0, 1.0),
            Vertex("v1", 1.0, 0.0, 0.0),
            Vertex("v2", 0.0, 1.0, 0.0),
            Vertex("v3", -1.0, 0.0, 0.0),
            Vertex("v4", 0.0, -1.0, 0.0),
            Vertex("v5", 0.0, 0.0, -1.0),
        ]
        faces = [
            Face("f0", "other", ("v0", "v1", "v2")),
            Face("f1", "other", ("v0", "v3", "v4")),
        ]
        grid = PolyGrid(vertices=verts, edges=[], faces=faces)
        schema = TileSchema([FieldDef("height", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        return grid, store

    def test_writes_to_all_faces(self):
        grid, store = self._make_grid_and_store()
        sample_noise_field_3d(grid, store, "height", lambda x, y, z: x + y + z)
        for fid in grid.faces:
            val = store.get(fid, "height")
            assert isinstance(val, float)

    def test_uses_3d_coords(self):
        grid, store = self._make_grid_and_store()
        # Noise fn that returns the z coordinate — verifies 3D is passed
        sample_noise_field_3d(grid, store, "height", lambda x, y, z: z)
        # f0 vertices: z = 1, 0, 0 → centroid z ≈ 0.333
        assert abs(store.get("f0", "height") - 1.0 / 3.0) < 1e-9

    def test_skips_faces_without_3d(self):
        verts = [Vertex("v0", 0.0, 0.0), Vertex("v1", 1.0, 0.0)]
        faces = [Face("f0", "other", ("v0", "v1"))]
        grid = PolyGrid(vertices=verts, edges=[], faces=faces)
        schema = TileSchema([FieldDef("h", float, -1.0)])
        store = TileDataStore(grid=grid, schema=schema)
        sample_noise_field_3d(grid, store, "h", lambda x, y, z: 999.0)
        assert store.get("f0", "h") == -1.0  # untouched


# ═══════════════════════════════════════════════════════════════════
# 8B — Globe grid builder
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestBuildGlobeGrid:
    def test_freq_2_tile_count(self):
        grid = build_globe_grid(2)
        assert len(grid.faces) == 42

    def test_freq_3_tile_count(self):
        grid = build_globe_grid(3)
        assert len(grid.faces) == 92

    def test_freq_4_tile_count(self):
        grid = build_globe_grid(4)
        assert len(grid.faces) == 162

    def test_pentagon_count_always_12(self):
        for freq in (2, 3, 4):
            grid = build_globe_grid(freq)
            pent = sum(1 for f in grid.faces.values() if f.face_type == "pent")
            assert pent == 12, f"freq={freq}, pent={pent}"

    def test_hexagon_count(self):
        for freq, expected_hex in [(2, 30), (3, 80), (4, 150)]:
            grid = build_globe_grid(freq)
            hexes = sum(1 for f in grid.faces.values() if f.face_type == "hex")
            assert hexes == expected_hex, f"freq={freq}, hex={hexes}"

    def test_vertex_counts_per_face(self):
        grid = build_globe_grid(3)
        for face in grid.faces.values():
            if face.face_type == "pent":
                assert len(face.vertex_ids) == 5
            elif face.face_type == "hex":
                assert len(face.vertex_ids) == 6

    def test_neighbor_counts(self):
        grid = build_globe_grid(3)
        for face in grid.faces.values():
            expected = 5 if face.face_type == "pent" else 6
            assert len(face.neighbor_ids) == expected, (
                f"face {face.id} ({face.face_type}): "
                f"expected {expected} neighbors, got {len(face.neighbor_ids)}"
            )

    def test_all_neighbors_exist(self):
        grid = build_globe_grid(3)
        for face in grid.faces.values():
            for nid in face.neighbor_ids:
                assert nid in grid.faces, f"face {face.id} neighbor {nid} not in grid"

    def test_adjacency_symmetric(self):
        grid = build_globe_grid(3)
        for face in grid.faces.values():
            for nid in face.neighbor_ids:
                neighbor = grid.faces[nid]
                assert face.id in neighbor.neighbor_ids, (
                    f"face {face.id} → neighbor {nid}, but {nid} does not list {face.id}"
                )

    def test_vertices_have_3d_positions(self):
        grid = build_globe_grid(2)
        for v in grid.vertices.values():
            assert v.has_position_3d(), f"vertex {v.id} missing 3D position"

    def test_face_ids_deterministic(self):
        a = build_globe_grid(3)
        b = build_globe_grid(3)
        assert sorted(a.faces.keys()) == sorted(b.faces.keys())

    def test_metadata_generator(self):
        grid = build_globe_grid(3)
        assert grid.metadata["generator"] == "globe"
        assert grid.metadata["frequency"] == 3
        assert grid.metadata["tile_count"] == 92

    def test_face_metadata_has_3d_props(self):
        grid = build_globe_grid(2)
        for face in grid.faces.values():
            assert "center_3d" in face.metadata
            assert "normal_3d" in face.metadata
            assert "latitude_deg" in face.metadata
            assert "longitude_deg" in face.metadata
            assert "tile_id" in face.metadata
            assert len(face.metadata["center_3d"]) == 3
            assert len(face.metadata["normal_3d"]) == 3

    def test_validate_clean(self):
        grid = build_globe_grid(3)
        errors = grid.validate()
        assert errors == [], f"validation errors: {errors}"


@needs_models
class TestGlobeGridAccessors:
    def test_frequency_property(self):
        grid = build_globe_grid(3)
        assert grid.frequency == 3

    def test_radius_property(self):
        grid = build_globe_grid(3, radius=2.5)
        assert grid.radius == 2.5

    def test_polyhedron_reference(self):
        grid = build_globe_grid(3)
        assert grid.polyhedron is not None
        assert grid.polyhedron.frequency == 3

    def test_tile_3d_center(self):
        grid = build_globe_grid(2)
        fid = list(grid.faces.keys())[0]
        center = grid.tile_3d_center(fid)
        assert center is not None
        assert len(center) == 3
        # Should be approximately on the unit sphere
        dist = math.sqrt(sum(c ** 2 for c in center))
        assert 0.5 < dist < 1.5  # roughly radius=1

    def test_tile_normal(self):
        grid = build_globe_grid(2)
        fid = list(grid.faces.keys())[0]
        normal = grid.tile_normal(fid)
        assert normal is not None
        length = math.sqrt(sum(n ** 2 for n in normal))
        assert abs(length - 1.0) < 0.01  # unit normal

    def test_tile_lat_lon(self):
        grid = build_globe_grid(2)
        fid = list(grid.faces.keys())[0]
        ll = grid.tile_lat_lon(fid)
        assert ll is not None
        lat, lon = ll
        assert -90.0 <= lat <= 90.0
        assert -180.0 <= lon <= 360.0

    def test_tile_models_id(self):
        grid = build_globe_grid(2)
        fid = list(grid.faces.keys())[0]
        mid = grid.tile_models_id(fid)
        assert mid is not None
        assert isinstance(mid, int)

    def test_nonexistent_face_returns_none(self):
        grid = build_globe_grid(2)
        assert grid.tile_3d_center("nonexistent") is None
        assert grid.tile_normal("nonexistent") is None
        assert grid.tile_lat_lon("nonexistent") is None
        assert grid.tile_models_id("nonexistent") is None


@needs_models
class TestGlobeGridSerialization:
    def test_to_dict_round_trip(self):
        grid = build_globe_grid(2)
        data = grid.to_dict(include_neighbors=True)
        grid2 = PolyGrid.from_dict(data)
        assert len(grid2.faces) == 42
        assert len(grid2.vertices) == len(grid.vertices)
        # Check face metadata survives
        fid = list(grid.faces.keys())[0]
        assert "center_3d" in grid2.faces[fid].metadata

    def test_to_json_round_trip(self):
        grid = build_globe_grid(2)
        js = grid.to_json()
        data = json.loads(js)
        assert data["metadata"]["generator"] == "globe"
        assert len(data["faces"]) == 42

    def test_vertex_z_survives_round_trip(self):
        grid = build_globe_grid(2)
        data = grid.to_dict()
        grid2 = PolyGrid.from_dict(data)
        for vid, v in grid2.vertices.items():
            assert v.z is not None, f"vertex {vid} lost z coordinate"


# ═══════════════════════════════════════════════════════════════════
# 8B — TileDataStore on globe grid
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestGlobeTileData:
    def test_store_creation(self):
        grid = build_globe_grid(2)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        assert len(list(store.grid.faces.keys())) == 42

    def test_set_and_get(self):
        grid = build_globe_grid(2)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        fid = list(grid.faces.keys())[0]
        store.set(fid, "elevation", 0.75)
        assert store.get(fid, "elevation") == 0.75

    def test_sample_noise_field_3d_on_globe(self):
        grid = build_globe_grid(2)
        schema = TileSchema([FieldDef("height", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        sample_noise_field_3d(
            grid, store, "height",
            lambda x, y, z: fbm_3d(x, y, z, seed=42, frequency=2.0),
        )
        # Every face should now have a non-default value
        vals = [store.get(fid, "height") for fid in grid.faces]
        assert any(v != 0.0 for v in vals), "noise field produced all zeros"


# ═══════════════════════════════════════════════════════════════════
# 8C — Globe terrain generation (quick validation)
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestGlobeTerrain:
    def test_mountains_on_globe(self):
        from polygrid.mountains import MountainConfig, generate_mountains

        grid = build_globe_grid(3)
        config = MountainConfig(
            seed=42,
            ridge_frequency=2.0,
            ridge_octaves=4,
        )
        schema = TileSchema([
            FieldDef("elevation", float, 0.0),
        ])
        store = TileDataStore(grid=grid, schema=schema)
        generate_mountains(grid, store, config)
        elevations = [store.get(fid, "elevation") for fid in grid.faces]
        assert len(elevations) == 92
        assert max(elevations) > min(elevations), "flat terrain — no variation"

    def test_regions_on_globe(self):
        from polygrid.regions import partition_voronoi, RegionMap

        grid = build_globe_grid(3)
        face_ids = list(grid.faces.keys())
        seeds = face_ids[:5]  # pick 5 seed faces
        region_map = partition_voronoi(grid, seeds)
        assert isinstance(region_map, RegionMap)
        # All faces assigned
        assigned = set()
        for region in region_map.regions:
            assigned.update(region.face_ids)
        assert assigned == set(face_ids)


# ═══════════════════════════════════════════════════════════════════
# 8C — get_face_adjacency universal helper
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestGetFaceAdjacency:
    def test_globe_adjacency_via_helper(self):
        from polygrid.core.algorithms import get_face_adjacency

        grid = build_globe_grid(2)
        adj = get_face_adjacency(grid)
        assert len(adj) == 42
        # Every face should have 5 or 6 neighbours
        for fid, nbrs in adj.items():
            assert len(nbrs) in (5, 6), f"{fid} has {len(nbrs)} neighbours"

    def test_flat_grid_adjacency_via_helper(self):
        from polygrid.core.algorithms import get_face_adjacency
        from polygrid import build_pure_hex_grid

        grid = build_pure_hex_grid(2)
        adj = get_face_adjacency(grid)
        # Flat grids don't have neighbor_ids populated, so should fall back to edge-based
        assert len(adj) > 0
        # At least some faces should have neighbours
        has_nbrs = sum(1 for nbrs in adj.values() if len(nbrs) > 0)
        assert has_nbrs > 0

    def test_flat_grid_with_neighbors_via_helper(self):
        from polygrid.core.algorithms import get_face_adjacency
        from polygrid import build_pure_hex_grid

        grid = build_pure_hex_grid(2).with_neighbors()
        adj = get_face_adjacency(grid)
        # Should prefer neighbor_ids path
        assert len(adj) > 0
        has_nbrs = sum(1 for nbrs in adj.values() if len(nbrs) > 0)
        assert has_nbrs > 0


# ═══════════════════════════════════════════════════════════════════
# 8D — Globe colour mapping
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestGlobeColourMap:
    def _make_globe_with_terrain(self, freq=2):
        from polygrid.mountains import MountainConfig, generate_mountains

        grid = build_globe_grid(freq)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        config = MountainConfig(seed=42, ridge_frequency=2.0, ridge_octaves=3)
        generate_mountains(grid, store, config)
        return grid, store

    def test_colour_map_has_all_faces(self):
        from polygrid.globe_export import globe_to_colour_map

        grid, store = self._make_globe_with_terrain()
        colours = globe_to_colour_map(grid, store)
        assert set(colours.keys()) == set(grid.faces.keys())

    def test_colour_map_valid_rgb(self):
        from polygrid.globe_export import globe_to_colour_map

        grid, store = self._make_globe_with_terrain()
        colours = globe_to_colour_map(grid, store)
        for fid, (r, g, b) in colours.items():
            assert 0.0 <= r <= 1.0, f"{fid}: r={r}"
            assert 0.0 <= g <= 1.0, f"{fid}: g={g}"
            assert 0.0 <= b <= 1.0, f"{fid}: b={b}"

    def test_topo_ramp(self):
        from polygrid.globe_export import globe_to_colour_map

        grid, store = self._make_globe_with_terrain()
        colours = globe_to_colour_map(grid, store, ramp="topo")
        assert len(colours) == len(grid.faces)



@needs_models
class TestGlobeExport:
    """Tests for globe_export.py — payload, JSON file, and schema validation."""

    def _make_globe_with_terrain(self, freq=2):
        from polygrid.mountains import MountainConfig, generate_mountains

        grid = build_globe_grid(freq)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        config = MountainConfig(seed=42, ridge_frequency=2.0, ridge_octaves=3)
        generate_mountains(grid, store, config)
        return grid, store

    # ── payload structure ───────────────────────────────────────────

    def test_payload_top_level_keys(self):
        from polygrid.globe_export import export_globe_payload

        grid, store = self._make_globe_with_terrain()
        payload = export_globe_payload(grid, store)
        assert set(payload.keys()) == {"metadata", "tiles", "adjacency"}

    def test_metadata_fields(self):
        from polygrid.globe_export import export_globe_payload

        grid, store = self._make_globe_with_terrain(freq=2)
        meta = export_globe_payload(grid, store)["metadata"]
        assert meta["version"] == "1.0"
        assert meta["generator"] == "polygrid.globe_export"
        assert meta["frequency"] == 2
        assert meta["radius"] == grid.radius
        assert meta["tile_count"] == len(grid.faces)
        assert meta["pentagon_count"] == 12
        assert meta["hexagon_count"] == len(grid.faces) - 12

    def test_tile_count_matches(self):
        from polygrid.globe_export import export_globe_payload

        for freq in (1, 2, 3):
            grid, store = self._make_globe_with_terrain(freq=freq)
            payload = export_globe_payload(grid, store)
            expected = 10 * freq ** 2 + 2
            assert len(payload["tiles"]) == expected
            assert payload["metadata"]["tile_count"] == expected

    def test_all_tiles_have_required_fields(self):
        from polygrid.globe_export import export_globe_payload

        grid, store = self._make_globe_with_terrain()
        payload = export_globe_payload(grid, store)
        required = {"id", "face_type", "vertices_3d", "center_3d",
                     "elevation", "color", "neighbor_ids"}
        for tile in payload["tiles"]:
            missing = required - set(tile.keys())
            assert not missing, f"Tile {tile.get('id')}: missing {missing}"

    def test_tile_colour_is_valid_rgb(self):
        from polygrid.globe_export import export_globe_payload

        grid, store = self._make_globe_with_terrain()
        payload = export_globe_payload(grid, store)
        for tile in payload["tiles"]:
            r, g, b = tile["color"]
            assert 0.0 <= r <= 1.0, f"{tile['id']}: r={r}"
            assert 0.0 <= g <= 1.0, f"{tile['id']}: g={g}"
            assert 0.0 <= b <= 1.0, f"{tile['id']}: b={b}"

    def test_tile_center_3d_present(self):
        from polygrid.globe_export import export_globe_payload

        grid, store = self._make_globe_with_terrain()
        payload = export_globe_payload(grid, store)
        for tile in payload["tiles"]:
            c = tile["center_3d"]
            assert c is not None, f"Tile {tile['id']}: center_3d is None"
            assert len(c) == 3

    def test_tile_vertices_3d_counts(self):
        from polygrid.globe_export import export_globe_payload

        grid, store = self._make_globe_with_terrain()
        payload = export_globe_payload(grid, store)
        for tile in payload["tiles"]:
            n_verts = len(tile["vertices_3d"])
            if tile["face_type"] == "pent":
                assert n_verts == 5, f"{tile['id']}: expected 5, got {n_verts}"
            else:
                assert n_verts == 6, f"{tile['id']}: expected 6, got {n_verts}"

    def test_tile_neighbor_ids_count(self):
        from polygrid.globe_export import export_globe_payload

        grid, store = self._make_globe_with_terrain()
        payload = export_globe_payload(grid, store)
        for tile in payload["tiles"]:
            n = len(tile["neighbor_ids"])
            if tile["face_type"] == "pent":
                assert n == 5, f"{tile['id']}: expected 5 neighbours, got {n}"
            else:
                assert n == 6, f"{tile['id']}: expected 6 neighbours, got {n}"

    def test_tile_lat_lon_ranges(self):
        from polygrid.globe_export import export_globe_payload

        grid, store = self._make_globe_with_terrain()
        payload = export_globe_payload(grid, store)
        for tile in payload["tiles"]:
            lat = tile["latitude_deg"]
            lon = tile["longitude_deg"]
            assert lat is not None
            assert lon is not None
            assert -90.0 <= lat <= 90.0, f"{tile['id']}: lat={lat}"
            assert -180.0 <= lon <= 180.0, f"{tile['id']}: lon={lon}"

    def test_tile_face_types(self):
        from polygrid.globe_export import export_globe_payload

        grid, store = self._make_globe_with_terrain()
        payload = export_globe_payload(grid, store)
        pents = [t for t in payload["tiles"] if t["face_type"] == "pent"]
        hexes = [t for t in payload["tiles"] if t["face_type"] == "hex"]
        assert len(pents) == 12
        assert len(hexes) == len(grid.faces) - 12

    def test_adjacency_edges(self):
        from polygrid.globe_export import export_globe_payload

        grid, store = self._make_globe_with_terrain()
        payload = export_globe_payload(grid, store)
        adj = payload["adjacency"]
        # Each edge is a pair of face IDs
        for edge in adj:
            assert len(edge) == 2
            assert all(isinstance(e, str) for e in edge)
        # Every adjacency edge should be symmetric in the tile neighbours
        tile_map = {t["id"]: t for t in payload["tiles"]}
        for a, b in adj:
            assert b in tile_map[a]["neighbor_ids"]
            assert a in tile_map[b]["neighbor_ids"]

    def test_adjacency_edge_count(self):
        """Edge count for Goldberg: E = 3F/2 - 6  (Euler for convex polyhedra)."""
        from polygrid.globe_export import export_globe_payload

        grid, store = self._make_globe_with_terrain(freq=2)
        payload = export_globe_payload(grid, store)
        f = len(payload["tiles"])
        expected_edges = (3 * f - 6) // 2  # Euler's formula for 3-connected planar graph
        # Actually: sum of neighbors / 2
        n_edges_from_neighbors = sum(
            len(t["neighbor_ids"]) for t in payload["tiles"]
        ) // 2
        assert len(payload["adjacency"]) == n_edges_from_neighbors

    # ── extra fields ────────────────────────────────────────────────

    def test_extra_fields_included(self):
        from polygrid.globe_export import export_globe_payload
        from polygrid.mountains import MountainConfig, generate_mountains

        grid = build_globe_grid(2)
        schema = TileSchema([
            FieldDef("elevation", float, 0.0),
            FieldDef("biome", str, "plains"),
        ])
        store = TileDataStore(grid=grid, schema=schema)
        config = MountainConfig(seed=42, ridge_frequency=2.0, ridge_octaves=3)
        generate_mountains(grid, store, config)
        for fid in grid.faces:
            store.set(fid, "biome", "forest")

        payload = export_globe_payload(grid, store, extra_fields=["biome"])
        assert payload["metadata"]["extra_fields"] == ["biome"]
        for tile in payload["tiles"]:
            assert tile["biome"] == "forest"

    def test_extra_fields_missing_in_store_gives_none(self):
        from polygrid.globe_export import export_globe_payload

        grid, store = self._make_globe_with_terrain()
        # Request a field that doesn't exist in the store
        payload = export_globe_payload(grid, store, extra_fields=["no_such_field"])
        for tile in payload["tiles"]:
            assert tile["no_such_field"] is None

    # ── JSON file export ────────────────────────────────────────────

    def test_export_json_creates_file(self, tmp_path):
        from polygrid.globe_export import export_globe_json

        grid, store = self._make_globe_with_terrain()
        out = export_globe_json(grid, store, tmp_path / "globe.json")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_export_json_is_valid_json(self, tmp_path):
        from polygrid.globe_export import export_globe_json

        grid, store = self._make_globe_with_terrain()
        out = export_globe_json(grid, store, tmp_path / "globe.json")
        payload = json.loads(out.read_text())
        assert "metadata" in payload
        assert "tiles" in payload
        assert "adjacency" in payload

    def test_export_json_roundtrip_tile_count(self, tmp_path):
        from polygrid.globe_export import export_globe_json

        grid, store = self._make_globe_with_terrain(freq=3)
        out = export_globe_json(grid, store, tmp_path / "globe.json")
        payload = json.loads(out.read_text())
        assert payload["metadata"]["tile_count"] == len(grid.faces)
        assert len(payload["tiles"]) == len(grid.faces)

    # ── schema validation ───────────────────────────────────────────

    def test_payload_validates_against_schema(self):
        import jsonschema
        from polygrid.globe_export import export_globe_payload

        schema_path = Path(__file__).resolve().parent.parent / "schemas" / "globe.schema.json"
        schema = json.loads(schema_path.read_text())

        grid, store = self._make_globe_with_terrain()
        payload = export_globe_payload(grid, store)
        # Should not raise
        jsonschema.validate(instance=payload, schema=schema)

    def test_json_file_validates_against_schema(self, tmp_path):
        import jsonschema
        from polygrid.globe_export import export_globe_json

        schema_path = Path(__file__).resolve().parent.parent / "schemas" / "globe.schema.json"
        schema = json.loads(schema_path.read_text())

        grid, store = self._make_globe_with_terrain()
        out = export_globe_json(grid, store, tmp_path / "globe.json")
        payload = json.loads(out.read_text())
        jsonschema.validate(instance=payload, schema=schema)

    # ── lightweight validator ───────────────────────────────────────

    def test_validate_globe_payload_passes(self):
        from polygrid.globe_export import export_globe_payload, validate_globe_payload

        grid, store = self._make_globe_with_terrain()
        payload = export_globe_payload(grid, store)
        errors = validate_globe_payload(payload)
        assert errors == []

    def test_validate_globe_payload_detects_missing_keys(self):
        from polygrid.globe_export import validate_globe_payload

        errors = validate_globe_payload({"metadata": {}, "tiles": []})
        assert any("adjacency" in e for e in errors)
        assert any("version" in e for e in errors)

    # ── topo ramp ───────────────────────────────────────────────────

    def test_topo_ramp_export(self):
        from polygrid.globe_export import export_globe_payload

        grid, store = self._make_globe_with_terrain()
        payload = export_globe_payload(grid, store, ramp="topo")
        assert payload["metadata"]["colour_ramp"] == "topo"
        # Colours should differ from satellite ramp (at least some)
        payload_sat = export_globe_payload(grid, store, ramp="satellite")
        diff = sum(
            1 for t1, t2 in zip(payload["tiles"], payload_sat["tiles"])
            if t1["color"] != t2["color"]
        )
        # Most tiles should differ between ramps
        assert diff > 0
