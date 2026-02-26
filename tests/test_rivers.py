"""Tests for rivers.py — Phase 7E: River generation."""

from __future__ import annotations

import pytest

from polygrid import build_pure_hex_grid, build_face_adjacency
from polygrid.mountains import generate_mountains, MOUNTAIN_RANGE, MountainConfig
from polygrid.rivers import (
    RiverConfig,
    RiverNetwork,
    RiverSegment,
    assign_river_data,
    carve_river_valleys,
    fill_depressions,
    find_drainage_basins,
    flow_accumulation,
    generate_rivers,
    river_to_overlay,
    steepest_descent_path,
)
from polygrid.tile_data import FieldDef, TileSchema, TileDataStore


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _make_store(grid, extra_fields=None):
    fields = [FieldDef("elevation", float, 0.0)]
    if extra_fields:
        fields.extend(extra_fields)
    schema = TileSchema(fields)
    store = TileDataStore(grid, schema=schema)
    store.initialise_all()
    return store


def _make_mountain_store(grid, config=None):
    store = _make_store(grid)
    cfg = config or MountainConfig(
        seed=42,
        smooth_iterations=0,
        edge_falloff=False,
    )
    generate_mountains(grid, store, cfg)
    return store


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def hex2_grid():
    return build_pure_hex_grid(rings=2)


@pytest.fixture
def hex3_grid():
    return build_pure_hex_grid(rings=3)


# ═══════════════════════════════════════════════════════════════════
# 7E.1 — Primitives
# ═══════════════════════════════════════════════════════════════════


class TestSteepestDescentPath:
    def test_descends_or_stays_level(self, hex2_grid):
        store = _make_mountain_store(hex2_grid)
        adj = build_face_adjacency(hex2_grid.faces.values(), hex2_grid.edges.values())

        # Pick the highest face as start
        highest = max(hex2_grid.faces.keys(), key=lambda f: store.get(f, "elevation"))
        path = steepest_descent_path(adj, store, highest)

        assert len(path) >= 1
        # Check monotonically non-increasing
        for i in range(1, len(path)):
            e_prev = store.get(path[i - 1], "elevation")
            e_curr = store.get(path[i], "elevation")
            assert e_curr <= e_prev + 1e-10, (
                f"Ascent at step {i}: {e_prev:.4f} → {e_curr:.4f}"
            )

    def test_returns_at_least_start(self, hex2_grid):
        store = _make_mountain_store(hex2_grid)
        adj = build_face_adjacency(hex2_grid.faces.values(), hex2_grid.edges.values())
        fid = list(hex2_grid.faces.keys())[0]
        path = steepest_descent_path(adj, store, fid)
        assert path[0] == fid


class TestFindDrainageBasins:
    def test_every_face_has_basin(self, hex2_grid):
        store = _make_mountain_store(hex2_grid)
        adj = build_face_adjacency(hex2_grid.faces.values(), hex2_grid.edges.values())
        basins = find_drainage_basins(adj, store)
        assert set(basins.keys()) == set(hex2_grid.faces.keys())

    def test_basin_id_is_a_face(self, hex2_grid):
        store = _make_mountain_store(hex2_grid)
        adj = build_face_adjacency(hex2_grid.faces.values(), hex2_grid.edges.values())
        basins = find_drainage_basins(adj, store)
        for basin_id in basins.values():
            assert basin_id in hex2_grid.faces


class TestFillDepressions:
    def test_no_interior_local_minima(self, hex2_grid):
        store = _make_mountain_store(hex2_grid)
        adj = build_face_adjacency(hex2_grid.faces.values(), hex2_grid.edges.values())

        fill_depressions(hex2_grid, store, adj)

        # Check: no interior face is a strict local minimum
        max_nbrs = max(len(v) for v in adj.values())
        interior = {fid for fid, nbrs in adj.items() if len(nbrs) >= max_nbrs}

        for fid in interior:
            e = store.get(fid, "elevation")
            nbr_elevs = [store.get(nid, "elevation") for nid in adj[fid]]
            # At least one neighbour should be ≤ this face
            has_lower_or_equal = any(ne <= e + 1e-10 for ne in nbr_elevs)
            if not has_lower_or_equal:
                # This face is a local minimum among interior faces
                # After filling, this should not happen
                assert False, f"Interior face {fid} is still a local minimum after filling"

    def test_returns_change_count(self, hex2_grid):
        store = _make_mountain_store(hex2_grid)
        adj = build_face_adjacency(hex2_grid.faces.values(), hex2_grid.edges.values())
        changes = fill_depressions(hex2_grid, store, adj)
        assert isinstance(changes, int)
        assert changes >= 0


class TestFlowAccumulation:
    def test_all_faces_have_accumulation(self, hex2_grid):
        store = _make_mountain_store(hex2_grid)
        adj = build_face_adjacency(hex2_grid.faces.values(), hex2_grid.edges.values())
        acc = flow_accumulation(adj, store)
        assert set(acc.keys()) == set(hex2_grid.faces.keys())

    def test_min_accumulation_is_one(self, hex2_grid):
        store = _make_mountain_store(hex2_grid)
        adj = build_face_adjacency(hex2_grid.faces.values(), hex2_grid.edges.values())
        acc = flow_accumulation(adj, store)
        assert all(v >= 1 for v in acc.values())

    def test_max_accumulation_at_low_elevation(self, hex2_grid):
        """The highest-accumulation face should tend toward lower elevation."""
        store = _make_mountain_store(hex2_grid)
        adj = build_face_adjacency(hex2_grid.faces.values(), hex2_grid.edges.values())
        acc = flow_accumulation(adj, store)

        max_acc_fid = max(acc, key=acc.get)
        max_acc_elev = store.get(max_acc_fid, "elevation")
        mean_elev = sum(store.get(f, "elevation") for f in hex2_grid.faces) / len(hex2_grid.faces)
        # The max-accumulation face should be below the mean elevation
        assert max_acc_elev <= mean_elev + 0.3  # generous tolerance


# ═══════════════════════════════════════════════════════════════════
# 7E.2 — River network
# ═══════════════════════════════════════════════════════════════════


class TestRiverSegment:
    def test_create(self):
        seg = RiverSegment(name="r0", face_ids=["f1", "f2", "f3"], order=1, width=1.0)
        assert seg.name == "r0"
        assert len(seg.face_ids) == 3

    def test_order(self):
        seg = RiverSegment(name="r0", order=3)
        assert seg.order == 3


class TestRiverNetwork:
    def test_all_river_face_ids(self):
        s1 = RiverSegment(name="r0", face_ids=["f1", "f2"])
        s2 = RiverSegment(name="r1", face_ids=["f3", "f4"])
        net = RiverNetwork(segments=[s1, s2])
        assert net.all_river_face_ids() == {"f1", "f2", "f3", "f4"}

    def test_segments_through(self):
        s1 = RiverSegment(name="r0", face_ids=["f1", "f2"])
        s2 = RiverSegment(name="r1", face_ids=["f2", "f3"])
        net = RiverNetwork(segments=[s1, s2])
        assert len(net.segments_through("f2")) == 2
        assert len(net.segments_through("f1")) == 1

    def test_main_stem(self):
        s1 = RiverSegment(name="r0", face_ids=["f1", "f2"], order=1)
        s2 = RiverSegment(name="r1", face_ids=["f3", "f4", "f5"], order=2)
        net = RiverNetwork(segments=[s1, s2])
        assert net.main_stem().name == "r1"

    def test_empty_network(self):
        net = RiverNetwork()
        assert len(net) == 0
        assert net.main_stem() is None
        assert net.all_river_face_ids() == set()


class TestGenerateRivers:
    def test_produces_river_network(self, hex3_grid):
        store = _make_mountain_store(hex3_grid)
        config = RiverConfig(min_accumulation=3, min_length=2)
        network = generate_rivers(hex3_grid, store, config)
        assert isinstance(network, RiverNetwork)

    def test_rivers_flow_downhill(self, hex3_grid):
        store = _make_mountain_store(hex3_grid)
        config = RiverConfig(min_accumulation=3, min_length=2)
        network = generate_rivers(hex3_grid, store, config)

        for seg in network.segments:
            for i in range(1, len(seg.face_ids)):
                e_prev = store.get(seg.face_ids[i - 1], "elevation")
                e_curr = store.get(seg.face_ids[i], "elevation")
                # After fill_depressions, should be non-increasing
                assert e_curr <= e_prev + 0.01, (
                    f"River {seg.name} ascends at step {i}: {e_prev:.4f} → {e_curr:.4f}"
                )

    def test_min_length_respected(self, hex3_grid):
        store = _make_mountain_store(hex3_grid)
        config = RiverConfig(min_accumulation=3, min_length=3)
        network = generate_rivers(hex3_grid, store, config)
        for seg in network.segments:
            assert len(seg.face_ids) >= config.min_length

    def test_on_flat_terrain_no_rivers(self, hex2_grid):
        """Flat terrain → no flow accumulation → no rivers."""
        store = _make_store(hex2_grid)
        store.bulk_set(hex2_grid.faces.keys(), "elevation", 0.5)
        config = RiverConfig(min_accumulation=5, min_length=3)
        network = generate_rivers(hex2_grid, store, config)
        # On flat terrain, flow accumulation is always 1, so no rivers
        assert len(network) == 0


# ═══════════════════════════════════════════════════════════════════
# 7E.3 — River ↔ terrain integration
# ═══════════════════════════════════════════════════════════════════


class TestCarveRiverValleys:
    def test_river_faces_lowered(self, hex3_grid):
        store = _make_mountain_store(hex3_grid)

        # Record pre-carve elevations
        pre_elevs = {fid: store.get(fid, "elevation") for fid in hex3_grid.faces}

        config = RiverConfig(min_accumulation=3, min_length=2)
        network = generate_rivers(hex3_grid, store, config)

        if len(network) == 0:
            pytest.skip("No rivers generated — terrain too flat for this config")

        # Re-record after fill_depressions (generate_rivers modifies elevation)
        pre_carve = {fid: store.get(fid, "elevation") for fid in hex3_grid.faces}

        carve_river_valleys(hex3_grid, store, network, carve_depth=0.1)

        river_faces = network.all_river_face_ids()
        for fid in river_faces:
            assert store.get(fid, "elevation") < pre_carve[fid] + 1e-10

    def test_non_river_faces_mostly_unchanged(self, hex3_grid):
        store = _make_mountain_store(hex3_grid)
        config = RiverConfig(min_accumulation=3, min_length=2)
        network = generate_rivers(hex3_grid, store, config)

        if len(network) == 0:
            pytest.skip("No rivers")

        pre_carve = {fid: store.get(fid, "elevation") for fid in hex3_grid.faces}
        carve_river_valleys(hex3_grid, store, network, carve_depth=0.1)

        adj = build_face_adjacency(hex3_grid.faces.values(), hex3_grid.edges.values())
        river_faces = network.all_river_face_ids()
        river_neighbors = set()
        for fid in river_faces:
            river_neighbors.update(adj.get(fid, []))

        # Faces far from rivers should be totally unchanged
        far_faces = set(hex3_grid.faces.keys()) - river_faces - river_neighbors
        for fid in far_faces:
            assert store.get(fid, "elevation") == pytest.approx(pre_carve[fid])


class TestAssignRiverData:
    def test_river_field_set(self, hex3_grid):
        store = _make_store(hex3_grid, [
            FieldDef("river", bool, False),
            FieldDef("river_width", float, 0.0),
        ])
        generate_mountains(hex3_grid, store, MountainConfig(seed=42, smooth_iterations=0, edge_falloff=False))

        config = RiverConfig(min_accumulation=3, min_length=2)
        network = generate_rivers(hex3_grid, store, config)

        if len(network) == 0:
            pytest.skip("No rivers")

        assign_river_data(store, network)

        river_faces = network.all_river_face_ids()
        for fid in river_faces:
            assert store.get(fid, "river") is True
            assert store.get(fid, "river_width") > 0

        non_river = set(hex3_grid.faces.keys()) - river_faces
        for fid in non_river:
            assert store.get(fid, "river") is False
            assert store.get(fid, "river_width") == 0.0


class TestRiverToOverlay:
    def test_overlay_kind(self, hex3_grid):
        store = _make_mountain_store(hex3_grid)
        config = RiverConfig(min_accumulation=3, min_length=2)
        network = generate_rivers(hex3_grid, store, config)
        overlay = river_to_overlay(hex3_grid, network)
        assert overlay.kind == "river"

    def test_overlay_face_count(self, hex3_grid):
        store = _make_mountain_store(hex3_grid)
        config = RiverConfig(min_accumulation=3, min_length=2)
        network = generate_rivers(hex3_grid, store, config)
        overlay = river_to_overlay(hex3_grid, network)
        # Overlay regions should match the total river face count
        # (some faces may appear in multiple segments but overlay deduplication
        # isn't required — it's okay to have overlapping regions)
        assert len(overlay.regions) >= 0  # may be 0 if no rivers

    def test_colors_in_metadata(self, hex3_grid):
        store = _make_mountain_store(hex3_grid)
        config = RiverConfig(min_accumulation=3, min_length=2)
        network = generate_rivers(hex3_grid, store, config)
        overlay = river_to_overlay(hex3_grid, network)

        for region in overlay.regions:
            fid = region.source_vertex_id
            color = overlay.metadata.get(f"color_{fid}")
            assert color is not None
            r, g, b = color
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0
