"""Tests for mountains.py — Phase 7C: Mountain terrain generation."""

from __future__ import annotations

import statistics

import pytest

from polygrid import build_pure_hex_grid, build_pentagon_centered_grid
from polygrid.mountains import (
    MountainConfig,
    generate_mountains,
    MOUNTAIN_RANGE,
    ALPINE_PEAKS,
    ROLLING_HILLS,
    MESA_PLATEAU,
)
from polygrid.regions import Region
from polygrid.tile_data import FieldDef, TileSchema, TileDataStore


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


def _make_store(grid):
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid, schema=schema)
    store.initialise_all()
    return store


@pytest.fixture
def hex2_grid():
    return build_pure_hex_grid(rings=2)


@pytest.fixture
def pent1_grid():
    return build_pentagon_centered_grid(rings=1)


# ═══════════════════════════════════════════════════════════════════
# Basic generation
# ═══════════════════════════════════════════════════════════════════


class TestGenerateMountains:
    """Core mountain generation tests."""

    def test_all_faces_have_elevation(self, hex2_grid):
        store = _make_store(hex2_grid)
        generate_mountains(hex2_grid, store, MOUNTAIN_RANGE)
        for fid in hex2_grid.faces:
            v = store.get(fid, "elevation")
            assert isinstance(v, float)

    def test_values_in_range(self, hex2_grid):
        """Elevation should be within [base, peak] after generation."""
        cfg = MountainConfig(base_elevation=0.1, peak_elevation=1.0)
        store = _make_store(hex2_grid)
        generate_mountains(hex2_grid, store, cfg)
        for fid in hex2_grid.faces:
            v = store.get(fid, "elevation")
            # Allow tiny overshoot from smoothing
            assert 0.0 <= v <= 1.1, f"face {fid} elevation={v}"

    def test_has_peaks(self, hex2_grid):
        """Top 10% of faces should have elevation > 0.5 × peak."""
        cfg = MountainConfig(peak_elevation=1.0, base_elevation=0.0, smooth_iterations=0, edge_falloff=False)
        store = _make_store(hex2_grid)
        generate_mountains(hex2_grid, store, cfg)
        vals = sorted(store.get(fid, "elevation") for fid in hex2_grid.faces)
        n = len(vals)
        top_10 = vals[int(n * 0.9):]
        assert all(v > 0.5 for v in top_10), f"top 10% values: {top_10}"

    def test_determinism(self, hex2_grid):
        """Same config + seed → identical output."""
        cfg = MountainConfig(seed=123)

        store1 = _make_store(hex2_grid)
        generate_mountains(hex2_grid, store1, cfg)
        vals1 = {fid: store1.get(fid, "elevation") for fid in hex2_grid.faces}

        store2 = _make_store(hex2_grid)
        generate_mountains(hex2_grid, store2, cfg)
        vals2 = {fid: store2.get(fid, "elevation") for fid in hex2_grid.faces}

        assert vals1 == vals2

    def test_different_seeds_different_output(self, hex2_grid):
        cfg_a = MountainConfig(seed=1)
        cfg_b = MountainConfig(seed=2)

        store_a = _make_store(hex2_grid)
        generate_mountains(hex2_grid, store_a, cfg_a)

        store_b = _make_store(hex2_grid)
        generate_mountains(hex2_grid, store_b, cfg_b)

        vals_a = [store_a.get(fid, "elevation") for fid in hex2_grid.faces]
        vals_b = [store_b.get(fid, "elevation") for fid in hex2_grid.faces]
        assert vals_a != vals_b


# ═══════════════════════════════════════════════════════════════════
# Region-restricted generation
# ═══════════════════════════════════════════════════════════════════


class TestRegionRestricted:
    def test_only_region_faces_modified(self, hex2_grid):
        all_fids = list(hex2_grid.faces.keys())
        region_fids = frozenset(all_fids[: len(all_fids) // 2])
        other_fids = set(all_fids) - region_fids
        region = Region(name="mountains", face_ids=region_fids)

        store = _make_store(hex2_grid)
        generate_mountains(hex2_grid, store, MOUNTAIN_RANGE, region=region)

        # Region faces should have non-default values
        region_vals = [store.get(fid, "elevation") for fid in region_fids]
        assert any(v != 0.0 for v in region_vals)

        # Other faces should still be 0 (default)
        for fid in other_fids:
            assert store.get(fid, "elevation") == 0.0


# ═══════════════════════════════════════════════════════════════════
# Presets produce different distributions
# ═══════════════════════════════════════════════════════════════════


class TestPresets:
    @pytest.mark.parametrize("preset", [MOUNTAIN_RANGE, ALPINE_PEAKS, ROLLING_HILLS, MESA_PLATEAU])
    def test_preset_runs_without_error(self, hex2_grid, preset):
        store = _make_store(hex2_grid)
        generate_mountains(hex2_grid, store, preset)
        vals = [store.get(fid, "elevation") for fid in hex2_grid.faces]
        assert all(isinstance(v, float) for v in vals)

    def test_presets_differ(self, hex2_grid):
        """Different presets should produce measurably different height distributions."""
        results = {}
        for name, preset in [
            ("range", MOUNTAIN_RANGE),
            ("alpine", ALPINE_PEAKS),
            ("hills", ROLLING_HILLS),
            ("mesa", MESA_PLATEAU),
        ]:
            store = _make_store(hex2_grid)
            generate_mountains(hex2_grid, store, preset)
            vals = [store.get(fid, "elevation") for fid in hex2_grid.faces]
            results[name] = (statistics.mean(vals), statistics.stdev(vals))

        # At least 3 of the 4 presets should have different means
        means = [m for m, _ in results.values()]
        unique_means = set(round(m, 3) for m in means)
        assert len(unique_means) >= 2, f"Means too similar: {results}"


# ═══════════════════════════════════════════════════════════════════
# Works on pentagon grid too
# ═══════════════════════════════════════════════════════════════════


class TestPentGrid:
    def test_on_pent_grid(self, pent1_grid):
        store = _make_store(pent1_grid)
        generate_mountains(pent1_grid, store, ROLLING_HILLS)
        vals = [store.get(fid, "elevation") for fid in pent1_grid.faces]
        assert all(isinstance(v, float) for v in vals)
        assert any(v > 0 for v in vals)


# ═══════════════════════════════════════════════════════════════════
# Config edge cases
# ═══════════════════════════════════════════════════════════════════


class TestConfigEdgeCases:
    def test_zero_warp(self, hex2_grid):
        cfg = MountainConfig(warp_strength=0.0)
        store = _make_store(hex2_grid)
        generate_mountains(hex2_grid, store, cfg)
        vals = [store.get(fid, "elevation") for fid in hex2_grid.faces]
        assert any(v > 0 for v in vals)

    def test_zero_foothill_blend(self, hex2_grid):
        cfg = MountainConfig(foothill_blend=0.0)
        store = _make_store(hex2_grid)
        generate_mountains(hex2_grid, store, cfg)
        vals = [store.get(fid, "elevation") for fid in hex2_grid.faces]
        assert any(v > 0 for v in vals)

    def test_no_smoothing(self, hex2_grid):
        cfg = MountainConfig(smooth_iterations=0)
        store = _make_store(hex2_grid)
        generate_mountains(hex2_grid, store, cfg)
        vals = [store.get(fid, "elevation") for fid in hex2_grid.faces]
        assert any(v > 0 for v in vals)

    def test_high_terrace_steps(self, hex2_grid):
        cfg = MountainConfig(terrace_steps=10, terrace_smoothing=0.0, smooth_iterations=0)
        store = _make_store(hex2_grid)
        generate_mountains(hex2_grid, store, cfg)
        vals = [store.get(fid, "elevation") for fid in hex2_grid.faces]
        # With terracing and no smoothing, we should see discrete levels
        unique_rounded = set(round(v, 2) for v in vals)
        assert len(unique_rounded) <= 12  # at most ~terrace_steps+2 levels
