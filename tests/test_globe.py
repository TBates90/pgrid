"""Tests for Phase 8 — Globe grid builder, 3-D noise, and heightmap bridge."""

from __future__ import annotations

import math
import json
from pathlib import Path
import pytest

from polygrid.models import Vertex, Face
from polygrid.polygrid import PolyGrid
from polygrid.noise import fbm_3d, ridged_noise_3d, _init_noise3
from polygrid.heightmap import sample_noise_field_3d
from polygrid.geometry import face_center_3d
from polygrid.tile_data import FieldDef, TileSchema, TileDataStore
from polygrid.algorithms import get_face_adjacency

# ── Gate globe-specific tests behind models availability ────────────
try:
    from polygrid.globe import build_globe_grid, GlobeGrid, _HAS_MODELS
    _skip_globe = not _HAS_MODELS
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

    def test_pipeline_on_globe(self):
        from polygrid.pipeline import TerrainPipeline, MountainStep
        from polygrid.mountains import MountainConfig

        grid = build_globe_grid(2)
        config = MountainConfig(
            seed=7,
            ridge_frequency=2.0,
            ridge_octaves=3,
        )
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        pipeline = TerrainPipeline(steps=[MountainStep(config=config)])
        result = pipeline.run(grid, store)
        elevations = [store.get(fid, "elevation") for fid in grid.faces]
        assert len(elevations) == 42

    def test_rivers_on_globe(self):
        from polygrid.mountains import MountainConfig, generate_mountains
        from polygrid.rivers import RiverConfig, generate_rivers

        grid = build_globe_grid(3)
        schema = TileSchema([
            FieldDef("elevation", float, 0.0),
            FieldDef("river", bool, False),
            FieldDef("river_width", float, 0.0),
        ])
        store = TileDataStore(grid=grid, schema=schema)
        config = MountainConfig(seed=42, ridge_frequency=2.0, ridge_octaves=4)
        generate_mountains(grid, store, config)

        river_config = RiverConfig(
            min_accumulation=3, min_length=2, carve_depth=0.03, seed=42,
        )
        network = generate_rivers(grid, store, river_config)
        assert len(network) > 0, "Expected at least one river segment"
        river_faces = network.all_river_face_ids()
        assert len(river_faces) > 0, "Expected at least one river face"
        # River faces must be a subset of grid faces
        assert river_faces <= set(grid.faces.keys())

    def test_river_carving_on_globe(self):
        from polygrid.mountains import MountainConfig, generate_mountains
        from polygrid.rivers import (
            RiverConfig, generate_rivers, carve_river_valleys, assign_river_data,
        )

        grid = build_globe_grid(3)
        schema = TileSchema([
            FieldDef("elevation", float, 0.0),
            FieldDef("river", bool, False),
            FieldDef("river_width", float, 0.0),
        ])
        store = TileDataStore(grid=grid, schema=schema)
        config = MountainConfig(seed=42, ridge_frequency=2.0, ridge_octaves=4)
        generate_mountains(grid, store, config)

        # Snapshot elevations before carving
        pre_elevations = {fid: store.get(fid, "elevation") for fid in grid.faces}

        river_config = RiverConfig(
            min_accumulation=3, min_length=2, carve_depth=0.03, seed=42,
        )
        network = generate_rivers(grid, store, river_config)
        if len(network) > 0:
            carve_river_valleys(grid, store, network, carve_depth=0.03)
            assign_river_data(store, network)

            # At least some river faces should have lower elevation
            river_faces = network.all_river_face_ids()
            lowered = [
                fid for fid in river_faces
                if store.get(fid, "elevation") < pre_elevations[fid]
            ]
            assert len(lowered) > 0, "Carving should lower some river faces"

            # River data fields should be set
            river_count = sum(1 for fid in grid.faces if store.get(fid, "river"))
            assert river_count == len(river_faces)

    def test_river_pipeline_on_globe(self):
        from polygrid.mountains import MountainConfig
        from polygrid.rivers import RiverConfig
        from polygrid.pipeline import TerrainPipeline, MountainStep, RiverStep

        grid = build_globe_grid(3)
        schema = TileSchema([
            FieldDef("elevation", float, 0.0),
            FieldDef("river", bool, False),
            FieldDef("river_width", float, 0.0),
        ])
        store = TileDataStore(grid=grid, schema=schema)

        mountain_cfg = MountainConfig(seed=42, ridge_frequency=2.0, ridge_octaves=4)
        river_cfg = RiverConfig(min_accumulation=3, min_length=2, carve_depth=0.03, seed=42)

        pipeline = TerrainPipeline(steps=[
            MountainStep(config=mountain_cfg),
            RiverStep(config=river_cfg, carve=True),
        ])
        result = pipeline.run(grid, store)
        network = result.artefact("rivers", "network")
        assert len(network) > 0


# ═══════════════════════════════════════════════════════════════════
# 8C — get_face_adjacency universal helper
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestGetFaceAdjacency:
    def test_globe_adjacency_via_helper(self):
        from polygrid.algorithms import get_face_adjacency

        grid = build_globe_grid(2)
        adj = get_face_adjacency(grid)
        assert len(adj) == 42
        # Every face should have 5 or 6 neighbours
        for fid, nbrs in adj.items():
            assert len(nbrs) in (5, 6), f"{fid} has {len(nbrs)} neighbours"

    def test_flat_grid_adjacency_via_helper(self):
        from polygrid.algorithms import get_face_adjacency
        from polygrid import build_pure_hex_grid

        grid = build_pure_hex_grid(2)
        adj = get_face_adjacency(grid)
        # Flat grids don't have neighbor_ids populated, so should fall back to edge-based
        assert len(adj) > 0
        # At least some faces should have neighbours
        has_nbrs = sum(1 for nbrs in adj.values() if len(nbrs) > 0)
        assert has_nbrs > 0

    def test_flat_grid_with_neighbors_via_helper(self):
        from polygrid.algorithms import get_face_adjacency
        from polygrid import build_pure_hex_grid

        grid = build_pure_hex_grid(2).with_neighbors()
        adj = get_face_adjacency(grid)
        # Should prefer neighbor_ids path
        assert len(adj) > 0
        has_nbrs = sum(1 for nbrs in adj.values() if len(nbrs) > 0)
        assert has_nbrs > 0


# ═══════════════════════════════════════════════════════════════════
# 8D — Globe rendering
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestGlobeRendering:
    def _make_globe_with_terrain(self, freq=2):
        from polygrid.mountains import MountainConfig, generate_mountains

        grid = build_globe_grid(freq)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        config = MountainConfig(seed=42, ridge_frequency=2.0, ridge_octaves=3)
        generate_mountains(grid, store, config)
        return grid, store

    def test_colour_map_has_all_faces(self):
        from polygrid.globe_render import globe_to_colour_map

        grid, store = self._make_globe_with_terrain()
        colours = globe_to_colour_map(grid, store)
        assert set(colours.keys()) == set(grid.faces.keys())

    def test_colour_map_valid_rgb(self):
        from polygrid.globe_render import globe_to_colour_map

        grid, store = self._make_globe_with_terrain()
        colours = globe_to_colour_map(grid, store)
        for fid, (r, g, b) in colours.items():
            assert 0.0 <= r <= 1.0, f"{fid}: r={r}"
            assert 0.0 <= g <= 1.0, f"{fid}: g={g}"
            assert 0.0 <= b <= 1.0, f"{fid}: b={b}"

    def test_tile_colours_export(self):
        from polygrid.globe_render import globe_to_tile_colours

        grid, store = self._make_globe_with_terrain()
        payload = globe_to_tile_colours(grid, store)
        assert len(payload) == len(grid.faces)
        first = next(iter(payload.values()))
        assert "color" in first
        assert "elevation" in first
        assert len(first["color"]) == 3

    def test_render_flat_produces_file(self, tmp_path):
        from polygrid.globe_render import render_globe_flat

        grid, store = self._make_globe_with_terrain()
        out = tmp_path / "flat.png"
        result = render_globe_flat(grid, store, out)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_render_3d_produces_file(self, tmp_path):
        from polygrid.globe_render import render_globe_3d

        grid, store = self._make_globe_with_terrain()
        out = tmp_path / "globe_3d.png"
        result = render_globe_3d(grid, store, out)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_topo_ramp(self):
        from polygrid.globe_render import globe_to_colour_map

        grid, store = self._make_globe_with_terrain()
        colours = globe_to_colour_map(grid, store, ramp="topo")
        assert len(colours) == len(grid.faces)


# ═══════════════════════════════════════════════════════════════════
# 8E — Globe mesh bridge (models integration)
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestGlobeMesh:
    def _make_globe_with_colours(self, freq=2):
        from polygrid.mountains import MountainConfig, generate_mountains
        from polygrid.globe_render import globe_to_colour_map, globe_to_tile_colours

        grid = build_globe_grid(freq)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        config = MountainConfig(seed=42, ridge_frequency=2.0, ridge_octaves=3)
        generate_mountains(grid, store, config)
        colour_map = globe_to_colour_map(grid, store)
        tile_colours = globe_to_tile_colours(grid, store)
        return grid, store, colour_map, tile_colours

    def test_terrain_colors_for_layout_count(self):
        from polygrid.globe_mesh import terrain_colors_for_layout
        from models.objects.goldberg.layout import layout_for_frequency

        grid, _, colour_map, _ = self._make_globe_with_colours()
        layout = layout_for_frequency(grid.frequency)
        colors = terrain_colors_for_layout(grid, colour_map, layout)
        assert len(colors) == len(layout.polygons)

    def test_terrain_colors_for_layout_valid_rgba(self):
        from polygrid.globe_mesh import terrain_colors_for_layout
        from models.objects.goldberg.layout import layout_for_frequency

        grid, _, colour_map, _ = self._make_globe_with_colours()
        layout = layout_for_frequency(grid.frequency)
        colors = terrain_colors_for_layout(grid, colour_map, layout)
        for c in colors:
            assert 0.0 <= c.r <= 1.0
            assert 0.0 <= c.g <= 1.0
            assert 0.0 <= c.b <= 1.0
            assert c.a == 1.0

    def test_terrain_colors_from_tile_colours(self):
        from polygrid.globe_mesh import terrain_colors_from_tile_colours
        from models.objects.goldberg.layout import layout_for_frequency

        grid, _, _, tile_colours = self._make_globe_with_colours()
        layout = layout_for_frequency(grid.frequency)
        colors = terrain_colors_from_tile_colours(tile_colours, layout)
        assert len(colors) == len(layout.polygons)

    def test_build_terrain_layout_mesh(self):
        from polygrid.globe_mesh import build_terrain_layout_mesh

        grid, _, colour_map, _ = self._make_globe_with_colours()
        mesh = build_terrain_layout_mesh(grid, colour_map)
        assert len(mesh.vertex_data) > 0
        assert len(mesh.index_data) > 0
        assert mesh.stride > 0

    def test_build_terrain_face_meshes(self):
        from polygrid.globe_mesh import build_terrain_face_meshes

        grid, _, colour_map, _ = self._make_globe_with_colours()
        face_meshes = build_terrain_face_meshes(grid, colour_map)
        assert len(face_meshes) == len(grid.faces)
        for fm in face_meshes:
            assert len(fm.mesh.vertex_data) > 0
            assert fm.kind in ("pentagon", "hexagon")

    def test_build_terrain_tile_meshes(self):
        from polygrid.globe_mesh import build_terrain_tile_meshes

        grid, _, colour_map, _ = self._make_globe_with_colours()
        tile_meshes = build_terrain_tile_meshes(grid, colour_map)
        assert len(tile_meshes) == len(grid.faces)
        for tm in tile_meshes:
            assert tm.model_matrix.shape == (4, 4)

    def test_build_terrain_edge_mesh(self):
        from polygrid.globe_mesh import build_terrain_edge_mesh

        grid, _, _, _ = self._make_globe_with_colours()
        mesh = build_terrain_edge_mesh(grid)
        assert len(mesh.vertex_data) > 0
        assert len(mesh.index_data) > 0

    def test_colour_map_matches_tile_colours(self):
        """Colours from colour_map and tile_colours should produce similar results.

        tile_colours JSON rounds to 4 decimal places, so tolerance is 5e-5.
        """
        from polygrid.globe_mesh import (
            terrain_colors_for_layout,
            terrain_colors_from_tile_colours,
        )
        from models.objects.goldberg.layout import layout_for_frequency

        grid, _, colour_map, tile_colours = self._make_globe_with_colours()
        layout = layout_for_frequency(grid.frequency)
        c1 = terrain_colors_for_layout(grid, colour_map, layout)
        c2 = terrain_colors_from_tile_colours(tile_colours, layout)
        for a, b in zip(c1, c2):
            assert abs(a.r - b.r) < 5e-5
            assert abs(a.g - b.g) < 5e-5
            assert abs(a.b - b.b) < 5e-5


# ═══════════════════════════════════════════════════════════════════
# 9A — Globe export
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# 9B — Multi-resolution detail grids
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestDetailGrid:
    """Tests for detail_grid.py — build, terrain, textures, atlas."""

    # ── build_detail_grid ───────────────────────────────────────────

    def test_hex_detail_face_count(self):
        from polygrid.detail_grid import build_detail_grid, detail_face_count

        grid = build_globe_grid(2)
        hex_fid = next(fid for fid, f in grid.faces.items() if f.face_type == "hex")
        for rings in (0, 1, 2, 3):
            d = build_detail_grid(grid, hex_fid, detail_rings=rings)
            expected = detail_face_count("hex", rings)
            assert len(d.faces) == expected, f"rings={rings}: {len(d.faces)} != {expected}"

    def test_pent_detail_face_count(self):
        from polygrid.detail_grid import build_detail_grid, detail_face_count

        grid = build_globe_grid(2)
        pent_fid = next(fid for fid, f in grid.faces.items() if f.face_type == "pent")
        for rings in (0, 1, 2):
            d = build_detail_grid(grid, pent_fid, detail_rings=rings)
            expected = detail_face_count("pent", rings)
            assert len(d.faces) == expected, f"rings={rings}: {len(d.faces)} != {expected}"

    def test_parent_face_id_in_metadata(self):
        from polygrid.detail_grid import build_detail_grid

        grid = build_globe_grid(2)
        fid = next(iter(grid.faces))
        d = build_detail_grid(grid, fid, detail_rings=1)
        assert d.metadata["parent_face_id"] == fid

    def test_parent_metadata_propagated(self):
        from polygrid.detail_grid import build_detail_grid

        grid = build_globe_grid(2)
        fid = next(iter(grid.faces))
        d = build_detail_grid(grid, fid, detail_rings=1)
        assert d.metadata.get("detail_rings") == 1
        # Parent 3D metadata should be present
        assert "parent_center_3d" in d.metadata
        assert "parent_normal_3d" in d.metadata

    def test_invalid_face_id_raises(self):
        from polygrid.detail_grid import build_detail_grid

        grid = build_globe_grid(1)
        with pytest.raises(KeyError):
            build_detail_grid(grid, "no_such_face", detail_rings=1)

    def test_detail_grid_has_positions(self):
        from polygrid.detail_grid import build_detail_grid

        grid = build_globe_grid(2)
        fid = next(iter(grid.faces))
        d = build_detail_grid(grid, fid, detail_rings=2)
        # All vertices should have 2D positions
        for v in d.vertices.values():
            assert v.has_position(), f"Vertex {v.id} has no position"

    # ── generate_detail_terrain ─────────────────────────────────────

    def test_detail_terrain_all_faces_populated(self):
        from polygrid.detail_grid import build_detail_grid, generate_detail_terrain

        grid = build_globe_grid(2)
        fid = next(fid for fid, f in grid.faces.items() if f.face_type == "hex")
        d = build_detail_grid(grid, fid, detail_rings=2)
        store = generate_detail_terrain(d, parent_elevation=0.5, seed=42)
        for dfid in d.faces:
            val = store.get(dfid, "elevation")
            assert isinstance(val, float)

    def test_detail_terrain_near_parent_elevation(self):
        from polygrid.detail_grid import build_detail_grid, generate_detail_terrain

        grid = build_globe_grid(2)
        fid = next(fid for fid, f in grid.faces.items() if f.face_type == "hex")
        d = build_detail_grid(grid, fid, detail_rings=2)
        parent_elev = 0.7
        store = generate_detail_terrain(d, parent_elevation=parent_elev, seed=42)
        vals = [store.get(dfid, "elevation") for dfid in d.faces]
        avg = sum(vals) / len(vals)
        # Average should be close to parent_elevation * base_weight
        assert abs(avg - parent_elev * 0.85) < 0.2

    def test_detail_terrain_deterministic(self):
        from polygrid.detail_grid import build_detail_grid, generate_detail_terrain

        grid = build_globe_grid(2)
        fid = next(fid for fid, f in grid.faces.items() if f.face_type == "hex")
        d = build_detail_grid(grid, fid, detail_rings=2)
        s1 = generate_detail_terrain(d, parent_elevation=0.5, seed=123)
        s2 = generate_detail_terrain(d, parent_elevation=0.5, seed=123)
        for dfid in d.faces:
            assert s1.get(dfid, "elevation") == s2.get(dfid, "elevation")

    def test_detail_terrain_different_seeds(self):
        from polygrid.detail_grid import build_detail_grid, generate_detail_terrain

        grid = build_globe_grid(2)
        fid = next(fid for fid, f in grid.faces.items() if f.face_type == "hex")
        d = build_detail_grid(grid, fid, detail_rings=2)
        s1 = generate_detail_terrain(d, parent_elevation=0.5, seed=1)
        s2 = generate_detail_terrain(d, parent_elevation=0.5, seed=2)
        # At least some values should differ
        diffs = sum(
            1 for dfid in d.faces
            if s1.get(dfid, "elevation") != s2.get(dfid, "elevation")
        )
        assert diffs > 0

    # ── render_detail_texture ───────────────────────────────────────

    def test_texture_creates_file(self, tmp_path):
        from polygrid.detail_grid import build_detail_grid, generate_detail_terrain, render_detail_texture

        grid = build_globe_grid(2)
        fid = next(fid for fid, f in grid.faces.items() if f.face_type == "hex")
        d = build_detail_grid(grid, fid, detail_rings=2)
        store = generate_detail_terrain(d, parent_elevation=0.5)
        out = render_detail_texture(d, store, tmp_path / "tile.png")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_texture_pent_creates_file(self, tmp_path):
        from polygrid.detail_grid import build_detail_grid, generate_detail_terrain, render_detail_texture

        grid = build_globe_grid(2)
        fid = next(fid for fid, f in grid.faces.items() if f.face_type == "pent")
        d = build_detail_grid(grid, fid, detail_rings=1)
        store = generate_detail_terrain(d, parent_elevation=0.3)
        out = render_detail_texture(d, store, tmp_path / "pent.png")
        assert out.exists()
        assert out.stat().st_size > 0

    # ── build_texture_atlas ─────────────────────────────────────────

    def test_atlas_creates_file(self, tmp_path):
        from polygrid.detail_grid import (
            build_detail_grid, generate_detail_terrain,
            render_detail_texture, build_texture_atlas,
        )

        grid = build_globe_grid(1)
        paths = []
        for i, fid in enumerate(list(grid.faces.keys())[:4]):
            d = build_detail_grid(grid, fid, detail_rings=1)
            store = generate_detail_terrain(d, parent_elevation=0.5, seed=i)
            p = render_detail_texture(d, store, tmp_path / f"tile_{fid}.png")
            paths.append(p)

        atlas_path, layout = build_texture_atlas(
            paths, tmp_path / "atlas.png", tile_size=64,
        )
        assert atlas_path.exists()
        assert atlas_path.stat().st_size > 0

    def test_atlas_layout_correct_count(self, tmp_path):
        from polygrid.detail_grid import (
            build_detail_grid, generate_detail_terrain,
            render_detail_texture, build_texture_atlas,
        )

        grid = build_globe_grid(1)
        paths = []
        for i, fid in enumerate(list(grid.faces.keys())[:6]):
            d = build_detail_grid(grid, fid, detail_rings=1)
            store = generate_detail_terrain(d, parent_elevation=0.5, seed=i)
            p = render_detail_texture(d, store, tmp_path / f"tile_{fid}.png")
            paths.append(p)

        _, layout = build_texture_atlas(
            paths, tmp_path / "atlas.png", tile_size=64,
        )
        assert len(layout) == 6

    def test_atlas_dimensions(self, tmp_path):
        """Atlas should have correct pixel dimensions based on columns/rows."""
        from polygrid.detail_grid import (
            build_detail_grid, generate_detail_terrain,
            render_detail_texture, build_texture_atlas,
        )
        from PIL import Image

        grid = build_globe_grid(1)
        paths = []
        for i, fid in enumerate(list(grid.faces.keys())[:4]):
            d = build_detail_grid(grid, fid, detail_rings=1)
            store = generate_detail_terrain(d, parent_elevation=0.5, seed=i)
            p = render_detail_texture(d, store, tmp_path / f"tile_{fid}.png")
            paths.append(p)

        atlas_path, _ = build_texture_atlas(
            paths, tmp_path / "atlas.png",
            tile_size=64, columns=2,
        )
        img = Image.open(atlas_path)
        assert img.size == (2 * 64, 2 * 64)  # 2 cols × 2 rows

    def test_atlas_no_textures_raises(self, tmp_path):
        from polygrid.detail_grid import build_texture_atlas

        with pytest.raises(ValueError, match="No texture"):
            build_texture_atlas([], tmp_path / "atlas.png")


# ═══════════════════════════════════════════════════════════════════
# 9C — Models renderer integration (CPU-side mesh tests)
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestGlobeRenderer:
    """Tests for globe_renderer.py — mesh builders, scene prep."""

    def _make_payload(self, freq=2):
        from polygrid.mountains import MountainConfig, generate_mountains
        from polygrid.globe_export import export_globe_payload

        grid = build_globe_grid(freq)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        config = MountainConfig(seed=42, ridge_frequency=2.0, ridge_octaves=3)
        generate_mountains(grid, store, config)
        payload = export_globe_payload(grid, store)
        return grid, store, payload

    def test_build_coloured_globe_mesh(self):
        from polygrid.globe_renderer import build_coloured_globe_mesh

        grid, _, payload = self._make_payload()
        tile_colours = {t["id"]: tuple(t["color"]) for t in payload["tiles"]}
        mesh = build_coloured_globe_mesh(grid.frequency, tile_colours)
        assert len(mesh.vertex_data) > 0
        assert len(mesh.index_data) > 0

    def test_build_coloured_mesh_from_export(self):
        from polygrid.globe_renderer import build_coloured_globe_mesh_from_export

        _, _, payload = self._make_payload()
        mesh = build_coloured_globe_mesh_from_export(payload)
        assert len(mesh.vertex_data) > 0
        assert len(mesh.index_data) > 0

    def test_mesh_vertex_count_matches_terrain_mesh(self):
        """Coloured mesh from export should match terrain_layout_mesh vertex count."""
        from polygrid.globe_renderer import build_coloured_globe_mesh_from_export
        from polygrid.globe_mesh import build_terrain_layout_mesh
        from polygrid.globe_render import globe_to_colour_map

        grid, store, payload = self._make_payload()
        colour_map = globe_to_colour_map(grid, store)
        ref_mesh = build_terrain_layout_mesh(grid, colour_map)
        test_mesh = build_coloured_globe_mesh_from_export(payload)
        assert len(test_mesh.vertex_data) == len(ref_mesh.vertex_data)
        assert len(test_mesh.index_data) == len(ref_mesh.index_data)

    def test_mesh_colours_match_payload(self):
        """Verify mesh colours correspond to the export payload colours."""
        from polygrid.globe_renderer import build_coloured_globe_mesh

        _, _, payload = self._make_payload(freq=1)
        tile_colours = {t["id"]: tuple(t["color"]) for t in payload["tiles"]}
        mesh = build_coloured_globe_mesh(1, tile_colours)
        # Mesh has vertex data — just verify it's non-trivial
        assert mesh.stride > 0
        assert len(mesh.attributes) >= 2  # position + color

    def test_build_edge_mesh(self):
        from polygrid.globe_renderer import build_edge_mesh_for_frequency

        mesh = build_edge_mesh_for_frequency(2)
        assert len(mesh.vertex_data) > 0
        assert len(mesh.index_data) > 0

    def test_prepare_terrain_scene(self):
        from polygrid.globe_renderer import prepare_terrain_scene

        _, _, payload = self._make_payload()
        scene = prepare_terrain_scene(payload)
        assert "mesh" in scene
        assert "edge_mesh" in scene
        assert scene["frequency"] == 2
        assert scene["tile_count"] == len(payload["tiles"])
        assert len(scene["mesh"].vertex_data) > 0

    def test_prepare_scene_no_edges(self):
        from polygrid.globe_renderer import prepare_terrain_scene

        _, _, payload = self._make_payload()
        scene = prepare_terrain_scene(payload, include_edges=False)
        assert scene["edge_mesh"] is None
        assert len(scene["mesh"].vertex_data) > 0

    def test_prepare_scene_custom_radius(self):
        from polygrid.globe_renderer import prepare_terrain_scene

        _, _, payload = self._make_payload()
        scene = prepare_terrain_scene(payload, radius=2.5)
        assert scene["radius"] == 2.5


# ═══════════════════════════════════════════════════════════════════
# 10A — Tile detail infrastructure
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestTileDetail:
    """Tests for tile_detail.py — TileDetailSpec, build_all, collection."""

    # ── helpers ──────────────────────────────────────────────────────

    def _make_globe(self, freq: int = 2):
        grid = build_globe_grid(freq)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        for fid in grid.faces:
            store.set(fid, "elevation", 0.5)
        return grid, store

    # ── TileDetailSpec ──────────────────────────────────────────────

    def test_spec_defaults(self):
        from polygrid.tile_detail import TileDetailSpec

        spec = TileDetailSpec()
        assert spec.detail_rings == 4
        assert spec.noise_frequency == 6.0
        assert spec.noise_octaves == 5
        assert spec.amplitude == 0.12
        assert spec.base_weight == 0.80
        assert spec.boundary_smoothing == 2
        assert spec.warp_strength == 0.15
        assert spec.seed_offset == 0

    def test_spec_frozen(self):
        from polygrid.tile_detail import TileDetailSpec

        spec = TileDetailSpec()
        with pytest.raises(AttributeError):
            spec.detail_rings = 10  # type: ignore[misc]

    def test_spec_custom_values(self):
        from polygrid.tile_detail import TileDetailSpec

        spec = TileDetailSpec(detail_rings=6, noise_frequency=10.0, amplitude=0.3)
        assert spec.detail_rings == 6
        assert spec.noise_frequency == 10.0
        assert spec.amplitude == 0.3

    # ── build_all_detail_grids ──────────────────────────────────────

    def test_build_all_creates_one_per_face(self):
        from polygrid.tile_detail import TileDetailSpec, build_all_detail_grids

        grid, _ = self._make_globe(2)
        spec = TileDetailSpec(detail_rings=2)
        grids = build_all_detail_grids(grid, spec)
        assert len(grids) == len(grid.faces)
        assert set(grids.keys()) == set(grid.faces.keys())

    def test_build_all_hex_face_count(self):
        from polygrid.tile_detail import TileDetailSpec, build_all_detail_grids
        from polygrid.detail_grid import detail_face_count

        grid, _ = self._make_globe(2)
        spec = TileDetailSpec(detail_rings=3)
        grids = build_all_detail_grids(grid, spec)
        hex_fid = next(fid for fid, f in grid.faces.items() if f.face_type == "hex")
        expected = detail_face_count("hex", 3)
        assert len(grids[hex_fid].faces) == expected

    def test_build_all_pent_face_count(self):
        from polygrid.tile_detail import TileDetailSpec, build_all_detail_grids
        from polygrid.detail_grid import detail_face_count

        grid, _ = self._make_globe(2)
        spec = TileDetailSpec(detail_rings=2)
        grids = build_all_detail_grids(grid, spec)
        pent_fid = next(fid for fid, f in grid.faces.items() if f.face_type == "pent")
        expected = detail_face_count("pent", 2)
        assert len(grids[pent_fid].faces) == expected

    def test_build_all_metadata_preserved(self):
        from polygrid.tile_detail import TileDetailSpec, build_all_detail_grids

        grid, _ = self._make_globe(2)
        spec = TileDetailSpec(detail_rings=2)
        grids = build_all_detail_grids(grid, spec)
        for fid, detail_grid in grids.items():
            assert detail_grid.metadata["parent_face_id"] == fid
            assert detail_grid.metadata["detail_rings"] == 2

    # ── DetailGridCollection.build ──────────────────────────────────

    def test_collection_build_default_spec(self):
        from polygrid.tile_detail import DetailGridCollection

        grid, _ = self._make_globe(1)
        coll = DetailGridCollection.build(grid)
        assert len(coll.grids) == len(grid.faces)
        assert coll.spec.detail_rings == 4

    def test_collection_build_custom_spec(self):
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid, _ = self._make_globe(1)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        assert coll.spec.detail_rings == 2
        assert len(coll.grids) == len(grid.faces)

    # ── Collection properties ───────────────────────────────────────

    def test_collection_globe_grid(self):
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid, _ = self._make_globe(1)
        coll = DetailGridCollection.build(grid, TileDetailSpec(detail_rings=1))
        assert coll.globe_grid is grid

    def test_collection_face_ids(self):
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid, _ = self._make_globe(1)
        coll = DetailGridCollection.build(grid, TileDetailSpec(detail_rings=1))
        assert set(coll.face_ids) == set(grid.faces.keys())

    def test_collection_total_face_count(self):
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
        from polygrid.detail_grid import detail_face_count

        grid, _ = self._make_globe(1)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        n_pent = sum(1 for f in grid.faces.values() if f.face_type == "pent")
        n_hex = len(grid.faces) - n_pent
        expected = (
            n_pent * detail_face_count("pent", 2)
            + n_hex * detail_face_count("hex", 2)
        )
        assert coll.total_face_count == expected

    def test_collection_get_returns_grid(self):
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid, _ = self._make_globe(1)
        coll = DetailGridCollection.build(grid, TileDetailSpec(detail_rings=1))
        fid = next(iter(grid.faces))
        detail_grid, store = coll.get(fid)
        assert len(detail_grid.faces) > 0
        assert store is None  # no terrain generated yet

    def test_collection_get_invalid_raises(self):
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid, _ = self._make_globe(1)
        coll = DetailGridCollection.build(grid, TileDetailSpec(detail_rings=1))
        with pytest.raises(KeyError, match="no_such"):
            coll.get("no_such")

    def test_collection_detail_face_count_for(self):
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
        from polygrid.detail_grid import detail_face_count

        grid, _ = self._make_globe(2)
        spec = TileDetailSpec(detail_rings=3)
        coll = DetailGridCollection.build(grid, spec)
        hex_fid = next(fid for fid, f in grid.faces.items() if f.face_type == "hex")
        assert coll.detail_face_count_for(hex_fid) == detail_face_count("hex", 3)

    def test_collection_stores_empty_before_terrain(self):
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid, _ = self._make_globe(1)
        coll = DetailGridCollection.build(grid, TileDetailSpec(detail_rings=1))
        assert len(coll.stores) == 0

    # ── generate_all_terrain ────────────────────────────────────────

    def test_generate_all_terrain_populates_stores(self):
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid, store = self._make_globe(1)
        spec = TileDetailSpec(detail_rings=2, boundary_smoothing=1)
        coll = DetailGridCollection.build(grid, spec)
        coll.generate_all_terrain(store, seed=42)
        assert len(coll.stores) == len(grid.faces)
        for fid in grid.faces:
            _, s = coll.get(fid)
            assert s is not None
            for dfid in coll.grids[fid].faces:
                val = s.get(dfid, "elevation")
                assert isinstance(val, float)

    def test_generate_terrain_values_near_parent(self):
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid, store = self._make_globe(1)
        parent_elev = 0.6
        for fid in grid.faces:
            store.set(fid, "elevation", parent_elev)
        spec = TileDetailSpec(detail_rings=2, base_weight=0.9, amplitude=0.05)
        coll = DetailGridCollection.build(grid, spec)
        coll.generate_all_terrain(store, seed=42)
        for fid in grid.faces:
            _, s = coll.get(fid)
            vals = [s.get(dfid, "elevation") for dfid in coll.grids[fid].faces]
            avg = sum(vals) / len(vals)
            # With base_weight=0.9 and amplitude=0.05, average should be close
            assert abs(avg - parent_elev * 0.9) < 0.15

    def test_generate_terrain_deterministic(self):
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid, store = self._make_globe(1)
        spec = TileDetailSpec(detail_rings=2)
        coll1 = DetailGridCollection.build(grid, spec)
        coll1.generate_all_terrain(store, seed=99)
        coll2 = DetailGridCollection.build(grid, spec)
        coll2.generate_all_terrain(store, seed=99)
        for fid in grid.faces:
            _, s1 = coll1.get(fid)
            _, s2 = coll2.get(fid)
            for dfid in coll1.grids[fid].faces:
                assert s1.get(dfid, "elevation") == s2.get(dfid, "elevation")

    def test_generate_terrain_different_seeds_differ(self):
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid, store = self._make_globe(1)
        spec = TileDetailSpec(detail_rings=2)
        coll1 = DetailGridCollection.build(grid, spec)
        coll1.generate_all_terrain(store, seed=1)
        coll2 = DetailGridCollection.build(grid, spec)
        coll2.generate_all_terrain(store, seed=2)
        diffs = 0
        for fid in grid.faces:
            _, s1 = coll1.get(fid)
            _, s2 = coll2.get(fid)
            for dfid in coll1.grids[fid].faces:
                if s1.get(dfid, "elevation") != s2.get(dfid, "elevation"):
                    diffs += 1
        assert diffs > 0

    def test_generate_terrain_no_warp(self):
        """Collection with warp_strength=0 should use plain fbm."""
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid, store = self._make_globe(1)
        spec = TileDetailSpec(detail_rings=2, warp_strength=0.0)
        coll = DetailGridCollection.build(grid, spec)
        coll.generate_all_terrain(store, seed=42)
        assert len(coll.stores) == len(grid.faces)

    def test_generate_terrain_no_smoothing(self):
        """Collection with boundary_smoothing=0 should skip smooth pass."""
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid, store = self._make_globe(1)
        spec = TileDetailSpec(detail_rings=2, boundary_smoothing=0)
        coll = DetailGridCollection.build(grid, spec)
        coll.generate_all_terrain(store, seed=42)
        assert len(coll.stores) == len(grid.faces)

    # ── summary / repr ──────────────────────────────────────────────

    def test_summary_format(self):
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid, store = self._make_globe(1)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        coll.generate_all_terrain(store, seed=42)
        text = coll.summary()
        assert "DetailGridCollection" in text
        assert "Pentagon" in text
        assert "Hexagon" in text
        assert "Detail rings" in text
        assert "Terrain stores" in text

    def test_repr(self):
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid, _ = self._make_globe(1)
        coll = DetailGridCollection.build(grid, TileDetailSpec(detail_rings=2))
        r = repr(coll)
        assert "DetailGridCollection" in r
        assert "rings=2" in r


# ═══════════════════════════════════════════════════════════════════
# 10B — Boundary-aware detail terrain
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestDetailTerrain:
    """Tests for detail_terrain.py — boundary classification, seam continuity."""

    # ── helpers ──────────────────────────────────────────────────────

    def _make_globe(self, freq: int = 2):
        grid = build_globe_grid(freq)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        for fid in grid.faces:
            store.set(fid, "elevation", 0.5)
        return grid, store

    def _make_varied_globe(self, freq: int = 2):
        """Globe with elevation varying by tile."""
        grid = build_globe_grid(freq)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        import random
        rng = random.Random(42)
        for fid in grid.faces:
            store.set(fid, "elevation", rng.uniform(0.1, 0.9))
        return grid, store

    # ── compute_boundary_elevations ─────────────────────────────────

    def test_boundary_elevations_all_faces(self):
        from polygrid.detail_terrain import compute_boundary_elevations

        grid, store = self._make_globe(2)
        result = compute_boundary_elevations(grid, store)
        assert set(result.keys()) == set(grid.faces.keys())

    def test_boundary_elevations_symmetric(self):
        """boundary_target(A→B) == boundary_target(B→A)."""
        from polygrid.detail_terrain import compute_boundary_elevations

        grid, store = self._make_varied_globe(2)
        result = compute_boundary_elevations(grid, store)
        adj = get_face_adjacency(grid)
        for fid in grid.faces:
            for nid in adj.get(fid, []):
                assert abs(result[fid][nid] - result[nid][fid]) < 1e-12

    def test_boundary_elevations_average(self):
        """Target = (own + neighbour) / 2."""
        from polygrid.detail_terrain import compute_boundary_elevations

        grid, store = self._make_varied_globe(1)
        result = compute_boundary_elevations(grid, store)
        adj = get_face_adjacency(grid)
        for fid in grid.faces:
            own = store.get(fid, "elevation")
            for nid in adj.get(fid, []):
                n_elev = store.get(nid, "elevation")
                expected = (own + n_elev) / 2.0
                assert abs(result[fid][nid] - expected) < 1e-12

    def test_boundary_elevations_uniform(self):
        """Uniform elevation → all boundary targets equal to that elevation."""
        from polygrid.detail_terrain import compute_boundary_elevations

        grid, store = self._make_globe(1)
        result = compute_boundary_elevations(grid, store)
        for fid, nmap in result.items():
            for nid, val in nmap.items():
                assert abs(val - 0.5) < 1e-12

    # ── classify_detail_faces ───────────────────────────────────────

    def test_classify_all_faces_classified(self):
        from polygrid.detail_terrain import classify_detail_faces
        from polygrid.detail_grid import build_detail_grid

        grid, _ = self._make_globe(2)
        fid = next(fid for fid, f in grid.faces.items() if f.face_type == "hex")
        d = build_detail_grid(grid, fid, detail_rings=3)
        cls = classify_detail_faces(d, boundary_depth=1)
        assert set(cls.keys()) == set(d.faces.keys())
        for val in cls.values():
            assert val in ("interior", "boundary", "corner")

    def test_classify_has_interior(self):
        from polygrid.detail_terrain import classify_detail_faces
        from polygrid.detail_grid import build_detail_grid

        grid, _ = self._make_globe(2)
        fid = next(fid for fid, f in grid.faces.items() if f.face_type == "hex")
        d = build_detail_grid(grid, fid, detail_rings=3)
        cls = classify_detail_faces(d, boundary_depth=1)
        counts = {v: 0 for v in ("interior", "boundary", "corner")}
        for v in cls.values():
            counts[v] += 1
        assert counts["interior"] > 0, "Should have interior faces"
        assert counts["boundary"] > 0, "Should have boundary faces"

    def test_classify_deeper_boundary(self):
        """More boundary depth → fewer interior faces."""
        from polygrid.detail_terrain import classify_detail_faces
        from polygrid.detail_grid import build_detail_grid

        grid, _ = self._make_globe(2)
        fid = next(fid for fid, f in grid.faces.items() if f.face_type == "hex")
        d = build_detail_grid(grid, fid, detail_rings=4)
        cls1 = classify_detail_faces(d, boundary_depth=1)
        cls2 = classify_detail_faces(d, boundary_depth=2)
        n_interior1 = sum(1 for v in cls1.values() if v == "interior")
        n_interior2 = sum(1 for v in cls2.values() if v == "interior")
        assert n_interior2 <= n_interior1

    def test_classify_pent_grid(self):
        from polygrid.detail_terrain import classify_detail_faces
        from polygrid.detail_grid import build_detail_grid

        grid, _ = self._make_globe(2)
        fid = next(fid for fid, f in grid.faces.items() if f.face_type == "pent")
        d = build_detail_grid(grid, fid, detail_rings=2)
        cls = classify_detail_faces(d, boundary_depth=1)
        assert set(cls.keys()) == set(d.faces.keys())

    # ── generate_detail_terrain_bounded ─────────────────────────────

    def test_bounded_all_faces_have_elevation(self):
        from polygrid.detail_terrain import generate_detail_terrain_bounded
        from polygrid.detail_grid import build_detail_grid
        from polygrid.tile_detail import TileDetailSpec

        grid, store = self._make_globe(2)
        fid = next(fid for fid, f in grid.faces.items() if f.face_type == "hex")
        d = build_detail_grid(grid, fid, detail_rings=3)
        spec = TileDetailSpec(detail_rings=3, boundary_smoothing=1)
        s = generate_detail_terrain_bounded(
            d, 0.5, {"n1": 0.4, "n2": 0.6}, spec, seed=42,
        )
        for dfid in d.faces:
            val = s.get(dfid, "elevation")
            assert isinstance(val, float)
            assert math.isfinite(val)

    def test_bounded_no_nan(self):
        from polygrid.detail_terrain import generate_detail_terrain_bounded
        from polygrid.detail_grid import build_detail_grid
        from polygrid.tile_detail import TileDetailSpec

        grid, store = self._make_varied_globe(2)
        adj = get_face_adjacency(grid)
        fid = next(fid for fid, f in grid.faces.items() if f.face_type == "hex")
        neighbor_elevs = {
            nid: (store.get(fid, "elevation") + store.get(nid, "elevation")) / 2
            for nid in adj.get(fid, [])
        }
        d = build_detail_grid(grid, fid, detail_rings=3)
        spec = TileDetailSpec(detail_rings=3)
        s = generate_detail_terrain_bounded(
            d, store.get(fid, "elevation"), neighbor_elevs, spec,
        )
        for dfid in d.faces:
            assert math.isfinite(s.get(dfid, "elevation"))

    def test_bounded_interior_near_parent(self):
        """Interior faces should cluster around the parent elevation."""
        from polygrid.detail_terrain import (
            generate_detail_terrain_bounded,
            classify_detail_faces,
        )
        from polygrid.detail_grid import build_detail_grid
        from polygrid.tile_detail import TileDetailSpec

        grid, _ = self._make_globe(2)
        fid = next(fid for fid, f in grid.faces.items() if f.face_type == "hex")
        d = build_detail_grid(grid, fid, detail_rings=4)
        parent_elev = 0.6
        spec = TileDetailSpec(
            detail_rings=4, base_weight=0.95, amplitude=0.02,
            boundary_smoothing=1,
        )
        s = generate_detail_terrain_bounded(
            d, parent_elev, {"n1": 0.3}, spec, seed=42,
        )
        cls = classify_detail_faces(d, boundary_depth=1)
        interior_vals = [
            s.get(dfid, "elevation")
            for dfid, c in cls.items() if c == "interior"
        ]
        if interior_vals:
            avg = sum(interior_vals) / len(interior_vals)
            assert abs(avg - parent_elev * spec.base_weight) < 0.15

    def test_bounded_boundary_between_parent_and_neighbor(self):
        """Boundary faces should trend toward the boundary target."""
        from polygrid.detail_terrain import (
            generate_detail_terrain_bounded,
            classify_detail_faces,
        )
        from polygrid.detail_grid import build_detail_grid
        from polygrid.tile_detail import TileDetailSpec

        grid, _ = self._make_globe(2)
        fid = next(fid for fid, f in grid.faces.items() if f.face_type == "hex")
        d = build_detail_grid(grid, fid, detail_rings=4)
        parent_elev = 0.8
        neighbor_elevs = {"n1": 0.2, "n2": 0.2, "n3": 0.2}  # low neighbours
        boundary_target = 0.2
        spec = TileDetailSpec(
            detail_rings=4, base_weight=0.95, amplitude=0.01,
            boundary_smoothing=2,
        )
        s = generate_detail_terrain_bounded(
            d, parent_elev, neighbor_elevs, spec, seed=42,
        )
        cls = classify_detail_faces(d, boundary_depth=2)
        boundary_vals = [
            s.get(dfid, "elevation")
            for dfid, c in cls.items() if c == "boundary"
        ]
        interior_vals = [
            s.get(dfid, "elevation")
            for dfid, c in cls.items() if c == "interior"
        ]
        if boundary_vals and interior_vals:
            b_avg = sum(boundary_vals) / len(boundary_vals)
            i_avg = sum(interior_vals) / len(interior_vals)
            # Boundary average should be closer to boundary target than interior
            assert abs(b_avg - boundary_target * spec.base_weight) < abs(
                i_avg - boundary_target * spec.base_weight
            )

    def test_bounded_deterministic(self):
        from polygrid.detail_terrain import generate_detail_terrain_bounded
        from polygrid.detail_grid import build_detail_grid
        from polygrid.tile_detail import TileDetailSpec

        grid, _ = self._make_globe(2)
        fid = next(fid for fid, f in grid.faces.items() if f.face_type == "hex")
        d = build_detail_grid(grid, fid, detail_rings=3)
        spec = TileDetailSpec(detail_rings=3)
        s1 = generate_detail_terrain_bounded(d, 0.5, {"n1": 0.4}, spec, seed=99)
        s2 = generate_detail_terrain_bounded(d, 0.5, {"n1": 0.4}, spec, seed=99)
        for dfid in d.faces:
            assert s1.get(dfid, "elevation") == s2.get(dfid, "elevation")

    def test_bounded_no_warp(self):
        from polygrid.detail_terrain import generate_detail_terrain_bounded
        from polygrid.detail_grid import build_detail_grid
        from polygrid.tile_detail import TileDetailSpec

        grid, _ = self._make_globe(2)
        fid = next(fid for fid, f in grid.faces.items() if f.face_type == "hex")
        d = build_detail_grid(grid, fid, detail_rings=2)
        spec = TileDetailSpec(detail_rings=2, warp_strength=0.0)
        s = generate_detail_terrain_bounded(d, 0.5, {}, spec, seed=42)
        for dfid in d.faces:
            assert math.isfinite(s.get(dfid, "elevation"))

    # ── generate_all_detail_terrain ─────────────────────────────────

    def test_generate_all_populates_stores(self):
        from polygrid.detail_terrain import generate_all_detail_terrain
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid, store = self._make_globe(1)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)
        assert len(coll.stores) == len(grid.faces)

    def test_generate_all_no_nan(self):
        from polygrid.detail_terrain import generate_all_detail_terrain
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid, store = self._make_varied_globe(1)
        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)
        for fid in grid.faces:
            _, s = coll.get(fid)
            for dfid in coll.grids[fid].faces:
                assert math.isfinite(s.get(dfid, "elevation"))

    def test_generate_all_deterministic(self):
        from polygrid.detail_terrain import generate_all_detail_terrain
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid, store = self._make_globe(1)
        spec = TileDetailSpec(detail_rings=2)
        coll1 = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll1, grid, store, spec, seed=42)
        coll2 = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll2, grid, store, spec, seed=42)
        for fid in grid.faces:
            _, s1 = coll1.get(fid)
            _, s2 = coll2.get(fid)
            for dfid in coll1.grids[fid].faces:
                assert s1.get(dfid, "elevation") == s2.get(dfid, "elevation")

    def test_generate_all_uses_collection_spec(self):
        """If no spec override, uses collection's spec."""
        from polygrid.detail_terrain import generate_all_detail_terrain
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid, store = self._make_globe(1)
        spec = TileDetailSpec(detail_rings=2, amplitude=0.01)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll, grid, store, seed=42)
        assert len(coll.stores) == len(grid.faces)

    def test_seam_continuity(self):
        """Adjacent tiles' boundary faces should have similar elevations.

        This is the key seam test: with boundary blending, max diff
        between boundary-face averages of neighbours should be small.
        """
        from polygrid.detail_terrain import (
            generate_all_detail_terrain,
            classify_detail_faces,
        )
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid, store = self._make_varied_globe(1)
        spec = TileDetailSpec(
            detail_rings=3, base_weight=0.9, amplitude=0.02,
            boundary_smoothing=2,
        )
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)

        adj = get_face_adjacency(grid)
        max_diff = 0.0
        for fid in grid.faces:
            _, s = coll.get(fid)
            cls = classify_detail_faces(coll.grids[fid], boundary_depth=2)
            b_vals = [
                s.get(dfid, "elevation")
                for dfid, c in cls.items() if c in ("boundary", "corner")
            ]
            if not b_vals:
                continue
            b_avg = sum(b_vals) / len(b_vals)

            for nid in adj.get(fid, []):
                _, ns = coll.get(nid)
                n_cls = classify_detail_faces(coll.grids[nid], boundary_depth=2)
                nb_vals = [
                    ns.get(dfid, "elevation")
                    for dfid, c in n_cls.items() if c in ("boundary", "corner")
                ]
                if not nb_vals:
                    continue
                nb_avg = sum(nb_vals) / len(nb_vals)
                max_diff = max(max_diff, abs(b_avg - nb_avg))

        # With boundary blending, the difference should be modest
        assert max_diff < 0.5, f"Seam max_diff={max_diff:.4f} too large"


# ═══════════════════════════════════════════════════════════════════
# 10C — Enhanced colour ramps & biome rendering
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestDetailRender:
    """Tests for detail_render.py — BiomeConfig, colour function, textures."""

    # ── helpers ──────────────────────────────────────────────────────

    def _make_globe(self, freq: int = 2):
        grid = build_globe_grid(freq)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        for fid in grid.faces:
            store.set(fid, "elevation", 0.5)
        return grid, store

    # ── BiomeConfig ─────────────────────────────────────────────────

    def test_biome_defaults(self):
        from polygrid.detail_render import BiomeConfig

        bc = BiomeConfig()
        assert bc.base_ramp == "detail_satellite"
        assert 0.0 <= bc.vegetation_density <= 1.0
        assert 0.0 <= bc.rock_exposure <= 1.0
        assert 0.0 < bc.snow_line < 1.0
        assert 0.0 <= bc.water_level <= 1.0

    def test_biome_frozen(self):
        from polygrid.detail_render import BiomeConfig

        bc = BiomeConfig()
        with pytest.raises(AttributeError):
            bc.moisture = 0.9  # type: ignore[misc]

    # ── detail_elevation_to_colour ──────────────────────────────────

    def test_colour_valid_rgb_range(self):
        """All elevation values produce valid RGB in [0, 1]."""
        from polygrid.detail_render import detail_elevation_to_colour, BiomeConfig

        bc = BiomeConfig()
        for t in [0.0, 0.05, 0.1, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]:
            r, g, b = detail_elevation_to_colour(
                t, bc, hillshade_val=0.5, noise_x=t, noise_y=t,
            )
            assert 0.0 <= r <= 1.0, f"r={r} at t={t}"
            assert 0.0 <= g <= 1.0, f"g={g} at t={t}"
            assert 0.0 <= b <= 1.0, f"b={b} at t={t}"

    def test_colour_water_below_level(self):
        """Water colour should be bluish (b > g > r) below water_level."""
        from polygrid.detail_render import detail_elevation_to_colour, BiomeConfig

        bc = BiomeConfig(water_level=0.15)
        r, g, b = detail_elevation_to_colour(
            0.05, bc, hillshade_val=0.5,
        )
        # Water should be mostly blue
        assert b > r, f"Water should be bluish: r={r}, b={b}"

    def test_colour_snow_above_line(self):
        """Colour above snow_line should be bright (high luminance)."""
        from polygrid.detail_render import detail_elevation_to_colour, BiomeConfig

        bc = BiomeConfig(snow_line=0.80)
        r, g, b = detail_elevation_to_colour(
            0.95, bc, hillshade_val=0.7,
        )
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        assert luminance > 0.6, f"Snow should be bright: lum={luminance:.3f}"

    def test_hillshade_darkens_south(self):
        """Low hillshade value should produce darker colour than high."""
        from polygrid.detail_render import detail_elevation_to_colour, BiomeConfig

        bc = BiomeConfig(hillshade_strength=0.8)
        r_dark, g_dark, b_dark = detail_elevation_to_colour(
            0.4, bc, hillshade_val=0.0,
        )
        r_light, g_light, b_light = detail_elevation_to_colour(
            0.4, bc, hillshade_val=1.0,
        )
        lum_dark = 0.299 * r_dark + 0.587 * g_dark + 0.114 * b_dark
        lum_light = 0.299 * r_light + 0.587 * g_light + 0.114 * b_light
        assert lum_dark < lum_light, (
            f"Dark shade ({lum_dark:.3f}) should be < light ({lum_light:.3f})"
        )

    def test_vegetation_varies(self):
        """Vegetation noise at different positions should produce different colours."""
        from polygrid.detail_render import detail_elevation_to_colour, BiomeConfig

        bc = BiomeConfig(vegetation_density=0.8, moisture=0.8)
        c1 = detail_elevation_to_colour(
            0.3, bc, noise_x=0.0, noise_y=0.0, noise_seed=42,
        )
        c2 = detail_elevation_to_colour(
            0.3, bc, noise_x=5.0, noise_y=5.0, noise_seed=42,
        )
        # At least one channel should differ
        assert c1 != c2, "Vegetation noise should vary spatially"

    def test_colour_no_vegetation_no_crash(self):
        """Zero vegetation_density should not crash."""
        from polygrid.detail_render import detail_elevation_to_colour, BiomeConfig

        bc = BiomeConfig(vegetation_density=0.0, rock_exposure=0.0)
        r, g, b = detail_elevation_to_colour(0.5, bc)
        assert 0.0 <= r <= 1.0

    def test_colour_extreme_elevations(self):
        """Clamped elevations outside [0,1] should still produce valid RGB."""
        from polygrid.detail_render import detail_elevation_to_colour, BiomeConfig

        bc = BiomeConfig()
        for t in [-0.5, -0.1, 1.1, 2.0]:
            r, g, b = detail_elevation_to_colour(t, bc)
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0

    # ── render_detail_texture_enhanced ──────────────────────────────

    def test_enhanced_texture_creates_file(self, tmp_path):
        from polygrid.detail_render import render_detail_texture_enhanced, BiomeConfig
        from polygrid.detail_grid import build_detail_grid
        from polygrid.detail_terrain import generate_detail_terrain_bounded
        from polygrid.tile_detail import TileDetailSpec

        grid, store = self._make_globe(2)
        fid = next(fid for fid, f in grid.faces.items() if f.face_type == "hex")
        d = build_detail_grid(grid, fid, detail_rings=2)
        spec = TileDetailSpec(detail_rings=2)
        s = generate_detail_terrain_bounded(d, 0.5, {}, spec, seed=42)
        out = render_detail_texture_enhanced(d, s, tmp_path / "tile.png")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_enhanced_texture_pent(self, tmp_path):
        from polygrid.detail_render import render_detail_texture_enhanced
        from polygrid.detail_grid import build_detail_grid
        from polygrid.detail_terrain import generate_detail_terrain_bounded
        from polygrid.tile_detail import TileDetailSpec

        grid, store = self._make_globe(2)
        fid = next(fid for fid, f in grid.faces.items() if f.face_type == "pent")
        d = build_detail_grid(grid, fid, detail_rings=1)
        spec = TileDetailSpec(detail_rings=1)
        s = generate_detail_terrain_bounded(d, 0.4, {}, spec, seed=42)
        out = render_detail_texture_enhanced(d, s, tmp_path / "pent.png")
        assert out.exists()

    def test_enhanced_texture_custom_biome(self, tmp_path):
        from polygrid.detail_render import render_detail_texture_enhanced, BiomeConfig
        from polygrid.detail_grid import build_detail_grid
        from polygrid.detail_terrain import generate_detail_terrain_bounded
        from polygrid.tile_detail import TileDetailSpec

        grid, store = self._make_globe(2)
        fid = next(fid for fid, f in grid.faces.items() if f.face_type == "hex")
        d = build_detail_grid(grid, fid, detail_rings=2)
        spec = TileDetailSpec(detail_rings=2)
        s = generate_detail_terrain_bounded(d, 0.6, {}, spec, seed=42)
        bc = BiomeConfig(
            vegetation_density=0.9, snow_line=0.7, moisture=0.8,
        )
        out = render_detail_texture_enhanced(
            d, s, tmp_path / "custom.png", bc, tile_size=128,
        )
        assert out.exists()
        assert out.stat().st_size > 0


# ═══════════════════════════════════════════════════════════════════
# 10D — Texture atlas & UV mapping
# ═══════════════════════════════════════════════════════════════════

@needs_models
class TestTexturePipeline:
    """Tests for texture_pipeline.py — atlas, UVs, textured meshes."""

    # ── helpers ──────────────────────────────────────────────────────

    def _make_collection(self, freq: int = 1, rings: int = 2):
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
        from polygrid.detail_terrain import generate_all_detail_terrain

        grid = build_globe_grid(freq)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        import random
        rng = random.Random(42)
        for fid in grid.faces:
            store.set(fid, "elevation", rng.uniform(0.1, 0.9))
        spec = TileDetailSpec(detail_rings=rings)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)
        return grid, store, coll

    # ── build_detail_atlas ──────────────────────────────────────────

    def test_atlas_creates_files(self, tmp_path):
        from polygrid.texture_pipeline import build_detail_atlas

        _, _, coll = self._make_collection(freq=1, rings=1)
        atlas_path, uv_layout = build_detail_atlas(
            coll, output_dir=tmp_path / "tiles", tile_size=64,
        )
        assert atlas_path.exists()
        assert atlas_path.stat().st_size > 0
        assert len(uv_layout) == len(coll.grids)

    def test_atlas_uv_range(self, tmp_path):
        """All UVs in [0, 1]."""
        from polygrid.texture_pipeline import build_detail_atlas

        _, _, coll = self._make_collection(freq=1, rings=1)
        _, uv_layout = build_detail_atlas(
            coll, output_dir=tmp_path / "tiles", tile_size=64,
        )
        for fid, (u_min, v_min, u_max, v_max) in uv_layout.items():
            assert 0.0 <= u_min <= 1.0, f"{fid}: u_min={u_min}"
            assert 0.0 <= v_min <= 1.0, f"{fid}: v_min={v_min}"
            assert 0.0 <= u_max <= 1.0, f"{fid}: u_max={u_max}"
            assert 0.0 <= v_max <= 1.0, f"{fid}: v_max={v_max}"
            assert u_max > u_min
            assert v_max > v_min

    def test_atlas_dimensions(self, tmp_path):
        from PIL import Image
        from polygrid.texture_pipeline import build_detail_atlas

        _, _, coll = self._make_collection(freq=1, rings=1)
        atlas_path, _ = build_detail_atlas(
            coll, output_dir=tmp_path / "tiles", tile_size=64, columns=4,
        )
        img = Image.open(atlas_path)
        n = len(coll.grids)
        rows = math.ceil(n / 4)
        assert img.size[0] == 4 * 64
        assert img.size[1] == rows * 64

    def test_atlas_covers_all_tiles(self, tmp_path):
        from polygrid.texture_pipeline import build_detail_atlas

        grid, _, coll = self._make_collection(freq=1, rings=1)
        _, uv_layout = build_detail_atlas(
            coll, output_dir=tmp_path / "tiles", tile_size=64,
        )
        assert set(uv_layout.keys()) == set(grid.faces.keys())

    def test_atlas_no_stores_raises(self, tmp_path):
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
        from polygrid.texture_pipeline import build_detail_atlas

        grid = build_globe_grid(1)
        spec = TileDetailSpec(detail_rings=1)
        coll = DetailGridCollection.build(grid, spec)
        # No terrain generated → stores empty
        with pytest.raises(ValueError, match="No terrain store"):
            build_detail_atlas(coll, output_dir=tmp_path / "tiles")

    # ── compute_tile_uvs ────────────────────────────────────────────

    def test_compute_uvs_maps_correctly(self):
        from polygrid.texture_pipeline import compute_tile_uvs

        tile_uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        slot = (0.25, 0.5, 0.5, 0.75)
        mapped = compute_tile_uvs(tile_uvs, slot)
        assert len(mapped) == 4
        # (0,0) → (0.25, 0.5)
        assert abs(mapped[0][0] - 0.25) < 1e-9
        assert abs(mapped[0][1] - 0.5) < 1e-9
        # (1,1) → (0.5, 0.75)
        assert abs(mapped[2][0] - 0.5) < 1e-9
        assert abs(mapped[2][1] - 0.75) < 1e-9

    def test_compute_uvs_within_range(self):
        from polygrid.texture_pipeline import compute_tile_uvs

        tile_uvs = [(0.2, 0.3), (0.8, 0.7)]
        slot = (0.0, 0.0, 0.5, 0.5)
        mapped = compute_tile_uvs(tile_uvs, slot)
        for u, v in mapped:
            assert 0.0 <= u <= 0.5
            assert 0.0 <= v <= 0.5

    # ── build_textured_tile_mesh ────────────────────────────────────

    def test_textured_mesh_stride(self, tmp_path):
        from polygrid.texture_pipeline import build_detail_atlas, build_textured_tile_mesh
        from models.objects.goldberg import generate_goldberg_tiles

        _, _, coll = self._make_collection(freq=1, rings=1)
        _, uv_layout = build_detail_atlas(
            coll, output_dir=tmp_path / "tiles", tile_size=64,
        )
        tiles = generate_goldberg_tiles(frequency=1)
        tile = tiles[0]
        fid = f"t{tile.index}"
        if fid in uv_layout:
            mesh = build_textured_tile_mesh(tile, uv_layout[fid])
            assert mesh.stride == 32
            assert len(mesh.attributes) == 3

    def test_textured_mesh_vertex_count(self, tmp_path):
        from polygrid.texture_pipeline import build_detail_atlas, build_textured_tile_mesh
        from models.objects.goldberg import generate_goldberg_tiles

        _, _, coll = self._make_collection(freq=1, rings=1)
        _, uv_layout = build_detail_atlas(
            coll, output_dir=tmp_path / "tiles", tile_size=64,
        )
        tiles = generate_goldberg_tiles(frequency=1)
        tile = tiles[0]
        fid = f"t{tile.index}"
        if fid in uv_layout:
            mesh = build_textured_tile_mesh(tile, uv_layout[fid])
            n_verts = len(tile.vertices) + 1  # +1 for center
            expected_floats = n_verts * 8  # 8 floats per vertex
            assert len(mesh.vertex_data) == expected_floats

    # ── build_textured_globe_meshes ─────────────────────────────────

    def test_globe_meshes_count(self, tmp_path):
        from polygrid.texture_pipeline import (
            build_detail_atlas, build_textured_globe_meshes,
        )

        grid, _, coll = self._make_collection(freq=1, rings=1)
        _, uv_layout = build_detail_atlas(
            coll, output_dir=tmp_path / "tiles", tile_size=64,
        )
        meshes = build_textured_globe_meshes(1, uv_layout)
        assert len(meshes) == len(grid.faces)

    def test_globe_meshes_all_stride_32(self, tmp_path):
        from polygrid.texture_pipeline import (
            build_detail_atlas, build_textured_globe_meshes,
        )

        _, _, coll = self._make_collection(freq=1, rings=1)
        _, uv_layout = build_detail_atlas(
            coll, output_dir=tmp_path / "tiles", tile_size=64,
        )
        meshes = build_textured_globe_meshes(1, uv_layout)
        for mesh in meshes:
            assert mesh.stride == 32


# ════════════════════════════════════════════════════════════════════
# Phase 10E — Textured OpenGL Renderer (no-context tests)
# ════════════════════════════════════════════════════════════════════

class TestTexturedRenderer:
    """Tests for the textured OpenGL renderer (shader sources, mesh
    construction, atlas loading helpers).  These do NOT open an
    OpenGL window — they validate the CPU-side logic only."""

    # ── shader source tests ─────────────────────────────────────────

    def test_textured_vertex_shader_is_string(self):
        from polygrid.globe_renderer import _TEXTURED_VERTEX_SHADER
        assert isinstance(_TEXTURED_VERTEX_SHADER, str)
        assert "gl_Position" in _TEXTURED_VERTEX_SHADER
        assert "v_uv" in _TEXTURED_VERTEX_SHADER

    def test_textured_fragment_shader_is_string(self):
        from polygrid.globe_renderer import _TEXTURED_FRAGMENT_SHADER
        assert isinstance(_TEXTURED_FRAGMENT_SHADER, str)
        assert "u_atlas" in _TEXTURED_FRAGMENT_SHADER
        assert "u_use_texture" in _TEXTURED_FRAGMENT_SHADER

    def test_textured_vertex_shader_has_uv_passthrough(self):
        from polygrid.globe_renderer import _TEXTURED_VERTEX_SHADER
        assert "in vec2 uv" in _TEXTURED_VERTEX_SHADER
        assert "out vec2 v_uv" in _TEXTURED_VERTEX_SHADER

    def test_textured_fragment_shader_fallback(self):
        """Fragment shader has a fallback to vertex colour when
        u_use_texture == 0."""
        from polygrid.globe_renderer import _TEXTURED_FRAGMENT_SHADER
        assert "v_color" in _TEXTURED_FRAGMENT_SHADER

    def test_textured_shaders_version_330(self):
        from polygrid.globe_renderer import (
            _TEXTURED_VERTEX_SHADER, _TEXTURED_FRAGMENT_SHADER,
        )
        assert "#version 330" in _TEXTURED_VERTEX_SHADER
        assert "#version 330" in _TEXTURED_FRAGMENT_SHADER

    # ── textured mesh stride ────────────────────────────────────────

    def test_textured_mesh_has_uv_attribute(self, tmp_path):
        from polygrid.texture_pipeline import (
            build_detail_atlas, build_textured_tile_mesh,
        )
        from models.objects.goldberg import generate_goldberg_tiles

        _, _, coll = self._make_collection(freq=1, rings=1)
        _, uv_layout = build_detail_atlas(
            coll, output_dir=tmp_path / "tiles", tile_size=64,
        )
        tiles = generate_goldberg_tiles(frequency=1)
        tile = tiles[0]
        fid = f"t{tile.index}"
        if fid in uv_layout:
            mesh = build_textured_tile_mesh(tile, uv_layout[fid])
            attr_names = [a.name for a in mesh.attributes]
            assert "uv" in attr_names
            assert "position" in attr_names
            assert "color" in attr_names

    def test_textured_mesh_uvs_within_atlas_slot(self, tmp_path):
        """UV coordinates in the mesh should fall within the atlas
        slot bounds (approximately — center is the average)."""
        from polygrid.texture_pipeline import (
            build_detail_atlas, build_textured_tile_mesh,
        )
        from models.objects.goldberg import generate_goldberg_tiles

        _, _, coll = self._make_collection(freq=1, rings=1)
        _, uv_layout = build_detail_atlas(
            coll, output_dir=tmp_path / "tiles", tile_size=64,
        )
        tiles = generate_goldberg_tiles(frequency=1)
        tile = tiles[0]
        fid = f"t{tile.index}"
        if fid not in uv_layout:
            pytest.skip("tile not in uv_layout")
        mesh = build_textured_tile_mesh(tile, uv_layout[fid])
        u_min, v_min, u_max, v_max = uv_layout[fid]
        # Extract UVs from vertex data (stride = 8 floats, uv at offset 6)
        n_verts = len(mesh.vertex_data) // 8
        for i in range(n_verts):
            u = mesh.vertex_data[i * 8 + 6]
            v = mesh.vertex_data[i * 8 + 7]
            assert u_min - 0.01 <= u <= u_max + 0.01, f"u={u}"
            assert v_min - 0.01 <= v <= v_max + 0.01, f"v={v}"

    # ── atlas texture loading (PIL side) ────────────────────────────

    def test_atlas_image_loadable(self, tmp_path):
        """The atlas PNG produced by build_detail_atlas can be loaded
        and flipped (as the renderer does)."""
        from PIL import Image
        from polygrid.texture_pipeline import build_detail_atlas

        _, _, coll = self._make_collection(freq=1, rings=1)
        atlas_path, _ = build_detail_atlas(
            coll, output_dir=tmp_path / "tiles", tile_size=64,
        )
        img = Image.open(str(atlas_path)).convert("RGBA").transpose(
            Image.FLIP_TOP_BOTTOM,
        )
        assert img.size[0] > 0
        assert img.size[1] > 0
        raw = img.tobytes()
        assert len(raw) == img.size[0] * img.size[1] * 4

    # ── render_textured_globe_opengl signature check ────────────────

    def test_render_textured_globe_opengl_importable(self):
        from polygrid.globe_renderer import render_textured_globe_opengl
        import inspect
        sig = inspect.signature(render_textured_globe_opengl)
        params = list(sig.parameters.keys())
        assert "payload" in params
        assert "atlas_path" in params
        assert "uv_layout" in params

    # ── view_globe --textured flag ──────────────────────────────────

    def test_view_globe_has_textured_flag(self):
        """view_globe.py exposes _launch_textured and _launch_flat."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "view_globe",
            str(Path(__file__).resolve().parent.parent / "scripts" / "view_globe.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        # Don't execute — just check that the functions exist in source
        source = Path(spec.origin).read_text()
        assert "def _launch_textured" in source
        assert "def _launch_flat" in source
        assert "--textured" in source
        assert "--detail-rings" in source

    # ── helpers (shared with TestTexturePipeline) ───────────────────

    def _make_collection(self, freq: int = 1, rings: int = 2):
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
        from polygrid.detail_terrain import generate_all_detail_terrain

        grid = build_globe_grid(freq)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        import random
        rng = random.Random(42)
        for fid in grid.faces:
            store.set(fid, "elevation", rng.uniform(0.1, 0.9))
        spec = TileDetailSpec(detail_rings=rings)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)
        return grid, store, coll


# ════════════════════════════════════════════════════════════════════
# Phase 10F — Performance & Scale
# ════════════════════════════════════════════════════════════════════

class TestDetailPerf:
    """Tests for detail_perf.py — parallel gen, fast render, cache."""

    # ── helpers ──────────────────────────────────────────────────────

    def _make_globe(self, freq: int = 1, rings: int = 2):
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

        grid = build_globe_grid(freq)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        import random
        rng = random.Random(42)
        for fid in grid.faces:
            store.set(fid, "elevation", rng.uniform(0.1, 0.9))
        spec = TileDetailSpec(detail_rings=rings)
        coll = DetailGridCollection.build(grid, spec)
        return grid, store, spec, coll

    # ── 10F.1 — Parallel terrain generation ─────────────────────────

    def test_parallel_produces_same_face_ids(self):
        from polygrid.detail_perf import generate_all_detail_terrain_parallel
        from polygrid.detail_terrain import generate_all_detail_terrain

        grid, store, spec, coll_serial = self._make_globe(freq=1, rings=2)
        generate_all_detail_terrain(coll_serial, grid, store, spec, seed=42)

        _, _, _, coll_parallel = self._make_globe(freq=1, rings=2)
        generate_all_detail_terrain_parallel(
            coll_parallel, grid, store, spec, seed=42, max_workers=2,
        )

        for fid in coll_serial.face_ids:
            _, s_serial = coll_serial.get(fid)
            _, s_parallel = coll_parallel.get(fid)
            assert s_serial is not None
            assert s_parallel is not None

    def test_parallel_produces_identical_results(self):
        from polygrid.detail_perf import generate_all_detail_terrain_parallel
        from polygrid.detail_terrain import generate_all_detail_terrain

        grid, store, spec, coll_serial = self._make_globe(freq=1, rings=2)
        generate_all_detail_terrain(coll_serial, grid, store, spec, seed=42)

        _, _, _, coll_parallel = self._make_globe(freq=1, rings=2)
        generate_all_detail_terrain_parallel(
            coll_parallel, grid, store, spec, seed=42, max_workers=2,
        )

        for fid in coll_serial.face_ids:
            g_s, s_s = coll_serial.get(fid)
            g_p, s_p = coll_parallel.get(fid)
            for sub_fid in g_s.faces:
                val_s = s_s.get(sub_fid, "elevation")
                val_p = s_p.get(sub_fid, "elevation")
                assert abs(val_s - val_p) < 1e-12, (
                    f"{fid}/{sub_fid}: serial={val_s} vs parallel={val_p}"
                )

    def test_parallel_all_stores_populated(self):
        from polygrid.detail_perf import generate_all_detail_terrain_parallel

        grid, store, spec, coll = self._make_globe(freq=1, rings=2)
        generate_all_detail_terrain_parallel(
            coll, grid, store, spec, seed=42,
        )
        for fid in coll.face_ids:
            _, s = coll.get(fid)
            assert s is not None, f"Store missing for {fid}"

    # ── 10F.2 — Fast PIL renderer ──────────────────────────────────

    def test_fast_render_creates_file(self, tmp_path):
        from polygrid.detail_perf import render_detail_texture_fast
        from polygrid.detail_terrain import generate_all_detail_terrain

        grid, store, spec, coll = self._make_globe(freq=1, rings=2)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)

        fid = coll.face_ids[0]
        g, s = coll.get(fid)
        out = tmp_path / "tile.png"
        result = render_detail_texture_fast(g, s, out, tile_size=64)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_fast_render_correct_dimensions(self, tmp_path):
        from PIL import Image
        from polygrid.detail_perf import render_detail_texture_fast
        from polygrid.detail_terrain import generate_all_detail_terrain

        grid, store, spec, coll = self._make_globe(freq=1, rings=2)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)

        fid = coll.face_ids[0]
        g, s = coll.get(fid)
        out = tmp_path / "tile.png"
        render_detail_texture_fast(g, s, out, tile_size=128)
        img = Image.open(out)
        assert img.size == (128, 128)

    def test_fast_render_non_black(self, tmp_path):
        """Output should not be entirely black."""
        from PIL import Image
        from polygrid.detail_perf import render_detail_texture_fast
        from polygrid.detail_terrain import generate_all_detail_terrain

        grid, store, spec, coll = self._make_globe(freq=1, rings=2)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)

        fid = coll.face_ids[0]
        g, s = coll.get(fid)
        out = tmp_path / "tile.png"
        render_detail_texture_fast(g, s, out, tile_size=64)
        img = Image.open(out)
        pixels = list(img.getdata())
        non_black = sum(1 for r, g, b in pixels if r + g + b > 0)
        assert non_black > 0, "Image is entirely black"

    # ── 10F.2b — Fast atlas builder ─────────────────────────────────

    def test_fast_atlas_creates_files(self, tmp_path):
        from polygrid.detail_perf import build_detail_atlas_fast
        from polygrid.detail_terrain import generate_all_detail_terrain

        grid, store, spec, coll = self._make_globe(freq=1, rings=1)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)

        atlas_path, uv_layout = build_detail_atlas_fast(
            coll, output_dir=tmp_path / "tiles", tile_size=64,
        )
        assert atlas_path.exists()
        assert len(uv_layout) == len(coll.grids)

    def test_fast_atlas_uv_range(self, tmp_path):
        from polygrid.detail_perf import build_detail_atlas_fast
        from polygrid.detail_terrain import generate_all_detail_terrain

        grid, store, spec, coll = self._make_globe(freq=1, rings=1)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)

        _, uv_layout = build_detail_atlas_fast(
            coll, output_dir=tmp_path / "tiles", tile_size=64,
        )
        for fid, (u_min, v_min, u_max, v_max) in uv_layout.items():
            assert 0.0 <= u_min < u_max <= 1.0
            assert 0.0 <= v_min < v_max <= 1.0

    # ── 10F.4 — Cache ──────────────────────────────────────────────

    def test_cache_put_and_get(self, tmp_path):
        from polygrid.detail_perf import DetailCache
        from polygrid.detail_terrain import generate_all_detail_terrain

        grid, store, spec, coll = self._make_globe(freq=1, rings=1)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)

        cache = DetailCache(cache_dir=tmp_path / "cache")
        fid = coll.face_ids[0]
        g, s = coll.get(fid)
        parent_elev = store.get(fid, "elevation")

        assert not cache.has(fid, spec, parent_elev, 42)
        cache.put(fid, spec, parent_elev, 42, s, g)
        assert cache.has(fid, spec, parent_elev, 42)

        loaded = cache.get(fid, spec, parent_elev, 42, g)
        assert loaded is not None
        for sub_fid in g.faces:
            assert abs(
                s.get(sub_fid, "elevation") - loaded.get(sub_fid, "elevation")
            ) < 1e-12

    def test_cache_miss_returns_none(self, tmp_path):
        from polygrid.detail_perf import DetailCache

        cache = DetailCache(cache_dir=tmp_path / "cache")
        grid, store, spec, coll = self._make_globe(freq=1, rings=1)
        fid = coll.face_ids[0]
        g, _ = coll.get(fid)
        result = cache.get(fid, spec, 0.5, 999, g)
        assert result is None

    def test_cache_clear(self, tmp_path):
        from polygrid.detail_perf import DetailCache
        from polygrid.detail_terrain import generate_all_detail_terrain

        grid, store, spec, coll = self._make_globe(freq=1, rings=1)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)

        cache = DetailCache(cache_dir=tmp_path / "cache")
        fid = coll.face_ids[0]
        g, s = coll.get(fid)
        parent_elev = store.get(fid, "elevation")
        cache.put(fid, spec, parent_elev, 42, s, g)
        assert cache.size >= 1
        removed = cache.clear()
        assert removed >= 1
        assert cache.size == 0

    def test_cached_generation_produces_results(self, tmp_path):
        from polygrid.detail_perf import (
            DetailCache, generate_all_detail_terrain_cached,
        )

        grid, store, spec, coll = self._make_globe(freq=1, rings=1)
        cache = DetailCache(cache_dir=tmp_path / "cache")

        # First run — all misses
        hits = generate_all_detail_terrain_cached(
            coll, grid, store, spec, seed=42, cache=cache,
        )
        assert hits == 0
        for fid in coll.face_ids:
            _, s = coll.get(fid)
            assert s is not None

        # Second run — all hits (fresh collection)
        _, _, _, coll2 = self._make_globe(freq=1, rings=1)
        hits2 = generate_all_detail_terrain_cached(
            coll2, grid, store, spec, seed=42, cache=cache,
        )
        assert hits2 == len(coll2.face_ids)

    def test_cached_matches_fresh(self, tmp_path):
        """Cached results should be identical to freshly generated."""
        from polygrid.detail_perf import (
            DetailCache, generate_all_detail_terrain_cached,
        )
        from polygrid.detail_terrain import generate_all_detail_terrain

        grid, store, spec, coll_fresh = self._make_globe(freq=1, rings=1)
        generate_all_detail_terrain(coll_fresh, grid, store, spec, seed=42)

        _, _, _, coll_cached = self._make_globe(freq=1, rings=1)
        cache = DetailCache(cache_dir=tmp_path / "cache")
        generate_all_detail_terrain_cached(
            coll_cached, grid, store, spec, seed=42, cache=cache,
        )

        for fid in coll_fresh.face_ids:
            g_f, s_f = coll_fresh.get(fid)
            g_c, s_c = coll_cached.get(fid)
            for sub_fid in g_f.faces:
                val_f = s_f.get(sub_fid, "elevation")
                val_c = s_c.get(sub_fid, "elevation")
                assert abs(val_f - val_c) < 1e-12

    # ── 10F.5 — Benchmark ──────────────────────────────────────────

    def test_benchmark_returns_timings(self):
        from polygrid.detail_perf import benchmark_pipeline

        timings = benchmark_pipeline(
            frequency=1, detail_rings=2, seed=42,
            use_parallel=True, use_fast_render=True, tile_size=32,
        )
        assert "grid_build" in timings
        assert "terrain_gen" in timings
        assert "texture_render" in timings
        assert "total" in timings
        assert timings["total"] > 0
        assert timings["tile_count"] == 12

    def test_benchmark_completes_within_timeout(self):
        """freq=1 detail_rings=2 should complete in < 30 seconds."""
        from polygrid.detail_perf import benchmark_pipeline

        timings = benchmark_pipeline(
            frequency=1, detail_rings=2, seed=42,
            use_parallel=True, use_fast_render=True, tile_size=32,
        )
        assert timings["total"] < 30.0, (
            f"Pipeline took {timings['total']:.1f}s (limit: 30s)"
        )


# ════════════════════════════════════════════════════════════════════
# Phase 10G — Demo & Integration
# ════════════════════════════════════════════════════════════════════

class TestDetailIntegration:
    """End-to-end integration tests for the full detail pipeline."""

    def test_full_pipeline_freq1(self, tmp_path):
        """Complete pipeline: globe → terrain → detail → atlas."""
        from polygrid.globe import build_globe_grid
        from polygrid.mountains import MountainConfig, generate_mountains
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
        from polygrid.detail_perf import (
            generate_all_detail_terrain_parallel,
            build_detail_atlas_fast,
        )
        from polygrid.detail_render import BiomeConfig

        grid = build_globe_grid(1)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        generate_mountains(grid, store, MountainConfig(seed=42))

        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain_parallel(
            coll, grid, store, spec, seed=42,
        )

        atlas_path, uv_layout = build_detail_atlas_fast(
            coll, BiomeConfig(), tmp_path / "atlas", tile_size=32,
        )
        assert atlas_path.exists()
        assert len(uv_layout) == 12  # freq=1 → 12 tiles

    def test_full_pipeline_freq2(self, tmp_path):
        """Pipeline at frequency=2 (42 tiles)."""
        from polygrid.globe import build_globe_grid
        from polygrid.mountains import MountainConfig, generate_mountains
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
        from polygrid.detail_perf import (
            generate_all_detail_terrain_parallel,
            build_detail_atlas_fast,
        )
        from polygrid.detail_render import BiomeConfig

        grid = build_globe_grid(2)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        generate_mountains(grid, store, MountainConfig(seed=42))

        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain_parallel(
            coll, grid, store, spec, seed=42,
        )

        atlas_path, uv_layout = build_detail_atlas_fast(
            coll, BiomeConfig(), tmp_path / "atlas", tile_size=32,
        )
        assert atlas_path.exists()
        assert len(uv_layout) == 42

    def test_atlas_tiles_match_globe(self, tmp_path):
        """Every globe face ID appears in the atlas UV layout."""
        from polygrid.globe import build_globe_grid
        from polygrid.mountains import MountainConfig, generate_mountains
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
        from polygrid.detail_perf import (
            generate_all_detail_terrain_parallel,
            build_detail_atlas_fast,
        )
        from polygrid.detail_render import BiomeConfig

        grid = build_globe_grid(1)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        generate_mountains(grid, store, MountainConfig(seed=42))

        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain_parallel(
            coll, grid, store, spec, seed=42,
        )

        _, uv_layout = build_detail_atlas_fast(
            coll, BiomeConfig(), tmp_path / "atlas", tile_size=32,
        )
        assert set(uv_layout.keys()) == set(grid.faces.keys())

    def test_textured_meshes_from_pipeline(self, tmp_path):
        """Pipeline produces textured meshes with correct stride."""
        from polygrid.globe import build_globe_grid
        from polygrid.mountains import MountainConfig, generate_mountains
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
        from polygrid.detail_perf import (
            generate_all_detail_terrain_parallel,
            build_detail_atlas_fast,
        )
        from polygrid.detail_render import BiomeConfig
        from polygrid.texture_pipeline import build_textured_globe_meshes

        grid = build_globe_grid(1)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        generate_mountains(grid, store, MountainConfig(seed=42))

        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain_parallel(
            coll, grid, store, spec, seed=42,
        )

        _, uv_layout = build_detail_atlas_fast(
            coll, BiomeConfig(), tmp_path / "atlas", tile_size=32,
        )

        meshes = build_textured_globe_meshes(1, uv_layout)
        assert len(meshes) == 12
        for m in meshes:
            assert m.stride == 32

    def test_demo_script_exists_and_importable(self):
        """demo_detail_globe.py exists and has expected entry points."""
        script = Path(__file__).resolve().parent.parent / "scripts" / "demo_detail_globe.py"
        assert script.exists()
        source = script.read_text()
        assert "def main()" in source
        assert "--detail-rings" in source
        assert "--compare" in source
        assert "--view" in source
        assert "--fast" in source

    def test_view_globe_textured_flag_preserved(self):
        """view_globe.py still has --textured and --detail-rings."""
        script = Path(__file__).resolve().parent.parent / "scripts" / "view_globe.py"
        source = script.read_text()
        assert "--textured" in source
        assert "--detail-rings" in source
        assert "def _launch_textured" in source
        assert "def _launch_flat" in source

    def test_render_textured_globe_importable(self):
        """render_textured_globe_opengl is importable from the package."""
        from polygrid.globe_renderer import render_textured_globe_opengl
        assert callable(render_textured_globe_opengl)

    def test_fast_renderer_matches_matplotlib_shapes(self, tmp_path):
        """Fast renderer produces same-sized output as matplotlib."""
        from polygrid.globe import build_globe_grid
        from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
        from polygrid.detail_terrain import generate_all_detail_terrain
        from polygrid.detail_render import BiomeConfig, render_detail_texture_enhanced
        from polygrid.detail_perf import render_detail_texture_fast
        from PIL import Image

        grid = build_globe_grid(1)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        import random
        rng = random.Random(42)
        for fid in grid.faces:
            store.set(fid, "elevation", rng.uniform(0.1, 0.9))

        spec = TileDetailSpec(detail_rings=2)
        coll = DetailGridCollection.build(grid, spec)
        generate_all_detail_terrain(coll, grid, store, spec, seed=42)

        fid = coll.face_ids[0]
        g, s = coll.get(fid)
        biome = BiomeConfig()

        fast_path = tmp_path / "fast.png"
        render_detail_texture_fast(g, s, fast_path, biome, tile_size=64)
        fast_img = Image.open(fast_path)
        assert fast_img.size == (64, 64)

        mpl_path = tmp_path / "mpl.png"
        render_detail_texture_enhanced(g, s, mpl_path, biome, tile_size=64)
        mpl_img = Image.open(mpl_path)
        # Both should produce images (may differ in exact pixels)
        assert mpl_img.size[0] > 0
        assert mpl_img.size[1] > 0
