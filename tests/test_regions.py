"""Tests for the regions module — Phase 6: Terrain Partitioning."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Set

import pytest

from polygrid import PolyGrid, build_face_adjacency
from polygrid.regions import (
    Region,
    RegionMap,
    RegionValidation,
    assign_biome,
    assign_field,
    partition_angular,
    partition_flood_fill,
    partition_noise,
    partition_voronoi,
    regions_to_overlay,
    validate_region_map,
)
from polygrid.tile_data import FieldDef, TileData, TileDataStore, TileSchema


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def hex_grid() -> PolyGrid:
    """A small hex grid with ring=1 (7 faces)."""
    from polygrid import build_pure_hex_grid

    return build_pure_hex_grid(rings=1)


@pytest.fixture
def pent_grid() -> PolyGrid:
    """A pentagon-centered grid with ring=1."""
    from polygrid import build_pentagon_centered_grid

    return build_pentagon_centered_grid(rings=1)


@pytest.fixture
def large_hex_grid() -> PolyGrid:
    """A larger hex grid with ring=2 (19 faces)."""
    from polygrid import build_pure_hex_grid

    return build_pure_hex_grid(rings=2)


@pytest.fixture
def tile_store(hex_grid: PolyGrid) -> TileDataStore:
    """A TileDataStore on the hex grid with biome + elevation fields."""
    schema = TileSchema([
        FieldDef("biome", str, "none"),
        FieldDef("elevation", float, 0.0),
    ])
    store = TileDataStore(hex_grid, schema=schema)
    store.initialise_all()
    return store


# ═══════════════════════════════════════════════════════════════════
# Region model
# ═══════════════════════════════════════════════════════════════════


class TestRegion:
    def test_create_region(self) -> None:
        r = Region(name="continent", face_ids=frozenset({"f1", "f2", "f3"}))
        assert r.name == "continent"
        assert r.size == 3
        assert "f1" in r
        assert "f99" not in r

    def test_metadata(self) -> None:
        r = Region(name="ocean", face_ids=frozenset({"f1"}), metadata={"biome": "water"})
        assert r.metadata["biome"] == "water"

    def test_repr(self) -> None:
        r = Region(name="test", face_ids=frozenset({"a", "b"}))
        assert "test" in repr(r)
        assert "2" in repr(r)


class TestRegionMap:
    def test_create(self) -> None:
        r1 = Region(name="a", face_ids=frozenset({"f1", "f2"}))
        r2 = Region(name="b", face_ids=frozenset({"f3"}))
        rm = RegionMap(
            regions=[r1, r2],
            grid_face_ids=frozenset({"f1", "f2", "f3"}),
        )
        assert len(rm) == 2
        assert rm.region_names == ["a", "b"]

    def test_get_region(self) -> None:
        r1 = Region(name="land", face_ids=frozenset({"f1"}))
        rm = RegionMap(regions=[r1], grid_face_ids=frozenset({"f1"}))
        assert rm.get_region("land") is r1
        with pytest.raises(KeyError):
            rm.get_region("water")

    def test_face_to_region(self) -> None:
        r1 = Region(name="a", face_ids=frozenset({"f1", "f2"}))
        r2 = Region(name="b", face_ids=frozenset({"f3"}))
        rm = RegionMap(
            regions=[r1, r2],
            grid_face_ids=frozenset({"f1", "f2", "f3"}),
        )
        f2r = rm.face_to_region()
        assert f2r["f1"] == "a"
        assert f2r["f3"] == "b"

    def test_region_for_face(self) -> None:
        r1 = Region(name="a", face_ids=frozenset({"f1"}))
        rm = RegionMap(regions=[r1], grid_face_ids=frozenset({"f1"}))
        assert rm.region_for_face("f1") == "a"
        with pytest.raises(KeyError):
            rm.region_for_face("f99")

    def test_repr(self) -> None:
        r1 = Region(name="x", face_ids=frozenset({"f1", "f2"}))
        rm = RegionMap(regions=[r1], grid_face_ids=frozenset({"f1", "f2"}))
        assert "x=2" in repr(rm)


# ═══════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════


class TestValidation:
    def test_valid_map(self) -> None:
        r1 = Region(name="a", face_ids=frozenset({"f1", "f2"}))
        r2 = Region(name="b", face_ids=frozenset({"f3"}))
        rm = RegionMap(
            regions=[r1, r2],
            grid_face_ids=frozenset({"f1", "f2", "f3"}),
        )
        result = validate_region_map(rm)
        assert result.ok
        assert not result.errors

    def test_missing_faces(self) -> None:
        r1 = Region(name="a", face_ids=frozenset({"f1"}))
        rm = RegionMap(
            regions=[r1],
            grid_face_ids=frozenset({"f1", "f2", "f3"}),
        )
        result = validate_region_map(rm)
        assert not result.ok
        assert any("Unassigned" in e for e in result.errors)

    def test_overlapping_faces(self) -> None:
        r1 = Region(name="a", face_ids=frozenset({"f1", "f2"}))
        r2 = Region(name="b", face_ids=frozenset({"f2", "f3"}))
        rm = RegionMap(
            regions=[r1, r2],
            grid_face_ids=frozenset({"f1", "f2", "f3"}),
        )
        result = validate_region_map(rm)
        assert not result.ok
        assert any("f2" in e for e in result.errors)

    def test_extra_faces(self) -> None:
        r1 = Region(name="a", face_ids=frozenset({"f1", "f99"}))
        rm = RegionMap(
            regions=[r1],
            grid_face_ids=frozenset({"f1"}),
        )
        result = validate_region_map(rm)
        assert not result.ok
        assert any("Extra" in e for e in result.errors)

    def test_duplicate_names(self) -> None:
        r1 = Region(name="a", face_ids=frozenset({"f1"}))
        r2 = Region(name="a", face_ids=frozenset({"f2"}))
        rm = RegionMap(
            regions=[r1, r2],
            grid_face_ids=frozenset({"f1", "f2"}),
        )
        result = validate_region_map(rm)
        assert not result.ok
        assert any("Duplicate" in e for e in result.errors)

    def test_min_region_size(self) -> None:
        r1 = Region(name="a", face_ids=frozenset({"f1"}))
        r2 = Region(name="b", face_ids=frozenset({"f2", "f3", "f4"}))
        rm = RegionMap(
            regions=[r1, r2],
            grid_face_ids=frozenset({"f1", "f2", "f3", "f4"}),
        )
        result = validate_region_map(rm, min_region_size=2)
        assert not result.ok
        assert any("min required" in e for e in result.errors)

    def test_max_region_count(self) -> None:
        r1 = Region(name="a", face_ids=frozenset({"f1"}))
        r2 = Region(name="b", face_ids=frozenset({"f2"}))
        r3 = Region(name="c", face_ids=frozenset({"f3"}))
        rm = RegionMap(
            regions=[r1, r2, r3],
            grid_face_ids=frozenset({"f1", "f2", "f3"}),
        )
        result = validate_region_map(rm, max_region_count=2)
        assert not result.ok
        assert any("Too many" in e for e in result.errors)

    def test_required_adjacency_satisfied(self, hex_grid: PolyGrid) -> None:
        # Partition into 2 regions, check they're adjacent
        rm = partition_angular(hex_grid, n_sections=2)
        adj = build_face_adjacency(hex_grid.faces.values(), hex_grid.edges.values())
        names = rm.region_names
        # Require that sector_0 touches sector_1
        result = validate_region_map(
            rm,
            adjacency=adj,
            required_adjacency={names[0]: names[1]},
        )
        assert result.ok

    def test_bool_coercion(self) -> None:
        v = RegionValidation(ok=True)
        assert v
        v2 = RegionValidation(ok=False, errors=["bad"])
        assert not v2


# ═══════════════════════════════════════════════════════════════════
# Angular partition
# ═══════════════════════════════════════════════════════════════════


class TestPartitionAngular:
    def test_full_coverage(self, hex_grid: PolyGrid) -> None:
        rm = partition_angular(hex_grid, n_sections=3)
        result = validate_region_map(rm)
        assert result.ok, result.errors

    def test_section_count(self, hex_grid: PolyGrid) -> None:
        rm = partition_angular(hex_grid, n_sections=3)
        assert len(rm) == 3

    def test_no_gaps_no_overlaps(self, hex_grid: PolyGrid) -> None:
        rm = partition_angular(hex_grid, n_sections=4)
        all_assigned: Set[str] = set()
        for r in rm.regions:
            overlap = all_assigned & r.face_ids
            assert not overlap, f"Overlap: {overlap}"
            all_assigned |= r.face_ids
        assert all_assigned == set(hex_grid.faces.keys())

    def test_single_section(self, hex_grid: PolyGrid) -> None:
        rm = partition_angular(hex_grid, n_sections=1)
        assert len(rm) == 1
        assert rm.regions[0].size == len(hex_grid.faces)

    def test_custom_prefix(self, hex_grid: PolyGrid) -> None:
        rm = partition_angular(hex_grid, n_sections=2, name_prefix="zone")
        assert rm.regions[0].name.startswith("zone_")

    def test_invalid_sections(self, hex_grid: PolyGrid) -> None:
        with pytest.raises(ValueError):
            partition_angular(hex_grid, n_sections=0)

    def test_many_sections(self, hex_grid: PolyGrid) -> None:
        """More sections than faces — some will be empty but should still be valid."""
        rm = partition_angular(hex_grid, n_sections=20)
        result = validate_region_map(rm)
        assert result.ok, result.errors
        # Total faces assigned = grid faces
        total = sum(r.size for r in rm.regions)
        assert total == len(hex_grid.faces)

    def test_pent_grid(self, pent_grid: PolyGrid) -> None:
        rm = partition_angular(pent_grid, n_sections=5)
        result = validate_region_map(rm)
        assert result.ok, result.errors


# ═══════════════════════════════════════════════════════════════════
# Flood-fill partition
# ═══════════════════════════════════════════════════════════════════


class TestPartitionFloodFill:
    def _get_seed_faces(self, grid: PolyGrid, n: int) -> List[str]:
        """Pick n well-spaced seed faces."""
        face_ids = sorted(grid.faces.keys())
        step = max(1, len(face_ids) // n)
        return [face_ids[i * step] for i in range(n)]

    def test_full_coverage(self, hex_grid: PolyGrid) -> None:
        seeds = self._get_seed_faces(hex_grid, 2)
        rm = partition_flood_fill(hex_grid, seeds)
        result = validate_region_map(rm)
        assert result.ok, result.errors

    def test_region_count(self, hex_grid: PolyGrid) -> None:
        seeds = self._get_seed_faces(hex_grid, 3)
        rm = partition_flood_fill(hex_grid, seeds, names=["a", "b", "c"])
        assert len(rm) == 3

    def test_custom_names(self, hex_grid: PolyGrid) -> None:
        seeds = self._get_seed_faces(hex_grid, 2)
        rm = partition_flood_fill(hex_grid, seeds, names=["land", "sea"])
        assert rm.region_names == ["land", "sea"]

    def test_single_seed(self, hex_grid: PolyGrid) -> None:
        """One seed → one region with all faces."""
        seeds = [sorted(hex_grid.faces.keys())[0]]
        rm = partition_flood_fill(hex_grid, seeds)
        assert len(rm) == 1
        assert rm.regions[0].size == len(hex_grid.faces)

    def test_no_gaps_no_overlaps(self, hex_grid: PolyGrid) -> None:
        seeds = self._get_seed_faces(hex_grid, 3)
        rm = partition_flood_fill(hex_grid, seeds)
        all_faces: Set[str] = set()
        for r in rm.regions:
            overlap = all_faces & r.face_ids
            assert not overlap
            all_faces |= r.face_ids
        assert all_faces == set(hex_grid.faces.keys())

    def test_deterministic_with_seed(self, hex_grid: PolyGrid) -> None:
        seeds = self._get_seed_faces(hex_grid, 3)
        rng1 = random.Random(123)
        rng2 = random.Random(123)
        rm1 = partition_flood_fill(hex_grid, seeds, rng=rng1)
        rm2 = partition_flood_fill(hex_grid, seeds, rng=rng2)
        for r1, r2 in zip(rm1.regions, rm2.regions):
            assert r1.face_ids == r2.face_ids

    def test_invalid_seed(self, hex_grid: PolyGrid) -> None:
        with pytest.raises(KeyError):
            partition_flood_fill(hex_grid, ["nonexistent_face"])

    def test_empty_seeds(self, hex_grid: PolyGrid) -> None:
        with pytest.raises(ValueError):
            partition_flood_fill(hex_grid, [])

    def test_names_length_mismatch(self, hex_grid: PolyGrid) -> None:
        seeds = self._get_seed_faces(hex_grid, 2)
        with pytest.raises(ValueError):
            partition_flood_fill(hex_grid, seeds, names=["only_one"])

    def test_seed_faces_in_regions(self, hex_grid: PolyGrid) -> None:
        """Each seed must end up in its own region."""
        seeds = self._get_seed_faces(hex_grid, 3)
        rm = partition_flood_fill(hex_grid, seeds, names=["r0", "r1", "r2"])
        for i, sid in enumerate(seeds):
            assert sid in rm.regions[i].face_ids

    def test_large_grid(self, large_hex_grid: PolyGrid) -> None:
        seeds = sorted(large_hex_grid.faces.keys())[:4]
        rm = partition_flood_fill(large_hex_grid, seeds)
        result = validate_region_map(rm)
        assert result.ok, result.errors


# ═══════════════════════════════════════════════════════════════════
# Voronoi partition
# ═══════════════════════════════════════════════════════════════════


class TestPartitionVoronoi:
    def _get_seed_faces(self, grid: PolyGrid, n: int) -> List[str]:
        face_ids = sorted(grid.faces.keys())
        step = max(1, len(face_ids) // n)
        return [face_ids[i * step] for i in range(n)]

    def test_full_coverage(self, hex_grid: PolyGrid) -> None:
        seeds = self._get_seed_faces(hex_grid, 3)
        rm = partition_voronoi(hex_grid, seeds)
        result = validate_region_map(rm)
        assert result.ok, result.errors

    def test_no_gaps_no_overlaps(self, hex_grid: PolyGrid) -> None:
        seeds = self._get_seed_faces(hex_grid, 2)
        rm = partition_voronoi(hex_grid, seeds)
        all_faces: Set[str] = set()
        for r in rm.regions:
            overlap = all_faces & r.face_ids
            assert not overlap
            all_faces |= r.face_ids
        assert all_faces == set(hex_grid.faces.keys())

    def test_seed_in_own_region(self, hex_grid: PolyGrid) -> None:
        seeds = self._get_seed_faces(hex_grid, 2)
        rm = partition_voronoi(hex_grid, seeds, names=["a", "b"])
        for i, sid in enumerate(seeds):
            assert sid in rm.regions[i].face_ids

    def test_deterministic(self, hex_grid: PolyGrid) -> None:
        seeds = self._get_seed_faces(hex_grid, 3)
        rm1 = partition_voronoi(hex_grid, seeds)
        rm2 = partition_voronoi(hex_grid, seeds)
        for r1, r2 in zip(rm1.regions, rm2.regions):
            assert r1.face_ids == r2.face_ids

    def test_single_seed(self, hex_grid: PolyGrid) -> None:
        seeds = [sorted(hex_grid.faces.keys())[0]]
        rm = partition_voronoi(hex_grid, seeds)
        assert rm.regions[0].size == len(hex_grid.faces)

    def test_invalid_seed(self, hex_grid: PolyGrid) -> None:
        with pytest.raises(KeyError):
            partition_voronoi(hex_grid, ["bad_face"])

    def test_large_grid(self, large_hex_grid: PolyGrid) -> None:
        seeds = sorted(large_hex_grid.faces.keys())[:5]
        rm = partition_voronoi(large_hex_grid, seeds, names=[f"r{i}" for i in range(5)])
        result = validate_region_map(rm)
        assert result.ok, result.errors


# ═══════════════════════════════════════════════════════════════════
# Noise-based partition
# ═══════════════════════════════════════════════════════════════════


class TestPartitionNoise:
    def _get_seed_faces(self, grid: PolyGrid, n: int) -> List[str]:
        face_ids = sorted(grid.faces.keys())
        step = max(1, len(face_ids) // n)
        return [face_ids[i * step] for i in range(n)]

    def test_full_coverage(self, hex_grid: PolyGrid) -> None:
        seeds = self._get_seed_faces(hex_grid, 3)
        rm = partition_noise(hex_grid, seeds)
        result = validate_region_map(rm)
        assert result.ok, result.errors

    def test_no_gaps_no_overlaps(self, hex_grid: PolyGrid) -> None:
        seeds = self._get_seed_faces(hex_grid, 2)
        rm = partition_noise(hex_grid, seeds, noise_weight=0.3)
        all_faces: Set[str] = set()
        for r in rm.regions:
            overlap = all_faces & r.face_ids
            assert not overlap
            all_faces |= r.face_ids
        assert all_faces == set(hex_grid.faces.keys())

    def test_deterministic_with_same_seed(self, hex_grid: PolyGrid) -> None:
        seeds = self._get_seed_faces(hex_grid, 3)
        rm1 = partition_noise(hex_grid, seeds, seed=99)
        rm2 = partition_noise(hex_grid, seeds, seed=99)
        for r1, r2 in zip(rm1.regions, rm2.regions):
            assert r1.face_ids == r2.face_ids

    def test_zero_noise_matches_voronoi(self, hex_grid: PolyGrid) -> None:
        """With noise_weight=0, should produce same result as pure Voronoi."""
        seeds = self._get_seed_faces(hex_grid, 2)
        rm_noise = partition_noise(hex_grid, seeds, noise_weight=0.0)
        rm_voronoi = partition_voronoi(hex_grid, seeds)
        for rn, rv in zip(rm_noise.regions, rm_voronoi.regions):
            assert rn.face_ids == rv.face_ids

    def test_invalid_seed(self, hex_grid: PolyGrid) -> None:
        with pytest.raises(KeyError):
            partition_noise(hex_grid, ["bad"])

    def test_large_grid(self, large_hex_grid: PolyGrid) -> None:
        seeds = sorted(large_hex_grid.faces.keys())[:4]
        rm = partition_noise(large_hex_grid, seeds, noise_scale=2.0, noise_weight=0.4)
        result = validate_region_map(rm)
        assert result.ok, result.errors


# ═══════════════════════════════════════════════════════════════════
# TileData integration
# ═══════════════════════════════════════════════════════════════════


class TestTileDataIntegration:
    def test_assign_field(self, hex_grid: PolyGrid, tile_store: TileDataStore) -> None:
        rm = partition_angular(hex_grid, n_sections=2)
        r0 = rm.regions[0]
        assign_field(r0, tile_store, "biome", "forest")
        for fid in r0.face_ids:
            assert tile_store.get(fid, "biome") == "forest"

    def test_assign_biome(self, hex_grid: PolyGrid, tile_store: TileDataStore) -> None:
        rm = partition_angular(hex_grid, n_sections=2)
        r0 = rm.regions[0]
        assign_biome(r0, tile_store, "desert")
        for fid in r0.face_ids:
            assert tile_store.get(fid, "biome") == "desert"
        assert r0.metadata["biome"] == "desert"

    def test_assign_all_regions(self, hex_grid: PolyGrid, tile_store: TileDataStore) -> None:
        rm = partition_angular(hex_grid, n_sections=3)
        biomes = ["forest", "desert", "ocean"]
        for r, b in zip(rm.regions, biomes):
            assign_biome(r, tile_store, b)
        for r, b in zip(rm.regions, biomes):
            for fid in r.face_ids:
                assert tile_store.get(fid, "biome") == b

    def test_assign_field_with_tile_data_directly(self, hex_grid: PolyGrid) -> None:
        """assign_field works with raw TileData too, not just TileDataStore."""
        schema = TileSchema([FieldDef("tag", str, "x")])
        td = TileData(schema)
        r = Region(name="r", face_ids=frozenset(hex_grid.faces.keys()))
        assign_field(r, td, "tag", "hello")
        for fid in r.face_ids:
            assert td.get(fid, "tag") == "hello"


# ═══════════════════════════════════════════════════════════════════
# Overlay conversion
# ═══════════════════════════════════════════════════════════════════


class TestOverlay:
    def test_overlay_kind(self, hex_grid: PolyGrid) -> None:
        rm = partition_angular(hex_grid, n_sections=3)
        ov = regions_to_overlay(rm, hex_grid)
        assert ov.kind == "partition"

    def test_overlay_region_count(self, hex_grid: PolyGrid) -> None:
        rm = partition_angular(hex_grid, n_sections=3)
        ov = regions_to_overlay(rm, hex_grid)
        # One overlay region per face
        assert len(ov.regions) == len(hex_grid.faces)

    def test_overlay_section_indices(self, hex_grid: PolyGrid) -> None:
        rm = partition_angular(hex_grid, n_sections=3)
        ov = regions_to_overlay(rm, hex_grid)
        indices = {int(r.source_vertex_id) for r in ov.regions}
        assert indices == {0, 1, 2}

    def test_overlay_metadata(self, hex_grid: PolyGrid) -> None:
        rm = partition_angular(hex_grid, n_sections=4)
        ov = regions_to_overlay(rm, hex_grid)
        assert ov.metadata["n_sections"] == 4
        assert len(ov.metadata["region_names"]) == 4

    def test_voronoi_partition_overlay(self, hex_grid: PolyGrid) -> None:
        seeds = sorted(hex_grid.faces.keys())[:2]
        rm = partition_voronoi(hex_grid, seeds)
        ov = regions_to_overlay(rm, hex_grid)
        assert len(ov.regions) == len(hex_grid.faces)


# ═══════════════════════════════════════════════════════════════════
# Region adjacency
# ═══════════════════════════════════════════════════════════════════


class TestRegionAdjacency:
    def test_region_adjacency(self, hex_grid: PolyGrid) -> None:
        rm = partition_angular(hex_grid, n_sections=3)
        adj = build_face_adjacency(hex_grid.faces.values(), hex_grid.edges.values())
        radj = rm.region_adjacency(adj)
        # With 3 sectors, each should be adjacent to the other two
        for name in rm.region_names:
            others = set(rm.region_names) - {name}
            assert radj[name] == others, f"{name} should touch {others}, got {radj[name]}"

    def test_two_region_adjacency(self, hex_grid: PolyGrid) -> None:
        rm = partition_angular(hex_grid, n_sections=2)
        adj = build_face_adjacency(hex_grid.faces.values(), hex_grid.edges.values())
        radj = rm.region_adjacency(adj)
        names = rm.region_names
        assert names[1] in radj[names[0]]
        assert names[0] in radj[names[1]]


# ═══════════════════════════════════════════════════════════════════
# Cross-algorithm consistency
# ═══════════════════════════════════════════════════════════════════


class TestCrossAlgorithm:
    """All algorithms must produce valid, full-coverage RegionMaps."""

    def _get_seeds(self, grid: PolyGrid, n: int) -> List[str]:
        face_ids = sorted(grid.faces.keys())
        step = max(1, len(face_ids) // n)
        return [face_ids[i * step] for i in range(n)]

    @pytest.mark.parametrize("n_sections", [2, 3, 5, 6])
    def test_angular_valid(self, hex_grid: PolyGrid, n_sections: int) -> None:
        rm = partition_angular(hex_grid, n_sections=n_sections)
        result = validate_region_map(rm)
        assert result.ok, result.errors

    @pytest.mark.parametrize("n_seeds", [1, 2, 3, 5])
    def test_flood_fill_valid(self, large_hex_grid: PolyGrid, n_seeds: int) -> None:
        seeds = self._get_seeds(large_hex_grid, n_seeds)
        rm = partition_flood_fill(large_hex_grid, seeds)
        result = validate_region_map(rm)
        assert result.ok, result.errors

    @pytest.mark.parametrize("n_seeds", [1, 2, 3, 5])
    def test_voronoi_valid(self, large_hex_grid: PolyGrid, n_seeds: int) -> None:
        seeds = self._get_seeds(large_hex_grid, n_seeds)
        rm = partition_voronoi(large_hex_grid, seeds)
        result = validate_region_map(rm)
        assert result.ok, result.errors

    @pytest.mark.parametrize("n_seeds", [1, 2, 3])
    def test_noise_valid(self, large_hex_grid: PolyGrid, n_seeds: int) -> None:
        seeds = self._get_seeds(large_hex_grid, n_seeds)
        rm = partition_noise(large_hex_grid, seeds)
        result = validate_region_map(rm)
        assert result.ok, result.errors

    def test_pent_grid_all_algorithms(self, pent_grid: PolyGrid) -> None:
        """All algorithms should work on pentagon-centered grids too."""
        # Angular
        rm = partition_angular(pent_grid, n_sections=3)
        assert validate_region_map(rm).ok

        # Flood-fill
        seeds = sorted(pent_grid.faces.keys())[:2]
        rm = partition_flood_fill(pent_grid, seeds)
        assert validate_region_map(rm).ok

        # Voronoi
        rm = partition_voronoi(pent_grid, seeds)
        assert validate_region_map(rm).ok

        # Noise
        rm = partition_noise(pent_grid, seeds)
        assert validate_region_map(rm).ok
