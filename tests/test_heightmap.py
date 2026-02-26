"""Tests for heightmap.py — Phase 7B: Grid-noise bridge."""

from __future__ import annotations

import math
from typing import Set

import pytest

from polygrid import build_pure_hex_grid, build_face_adjacency
from polygrid.noise import fbm, ridged_noise
from polygrid.tile_data import FieldDef, TileSchema, TileDataStore
from polygrid.heightmap import (
    sample_noise_field,
    sample_noise_field_region,
    smooth_field,
    blend_fields,
    clamp_field,
    normalize_field,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def grid():
    return build_pure_hex_grid(rings=1)


@pytest.fixture
def schema():
    return TileSchema([
        FieldDef("elevation", float, 0.0),
        FieldDef("temp", float, 0.0),
        FieldDef("blend_out", float, 0.0),
    ])


@pytest.fixture
def store(grid, schema):
    s = TileDataStore(grid, schema=schema)
    s.initialise_all()
    return s


# ═══════════════════════════════════════════════════════════════════
# sample_noise_field
# ═══════════════════════════════════════════════════════════════════


class TestSampleNoiseField:
    """Tests for sampling noise onto a grid."""

    def test_all_faces_get_values(self, grid, store):
        noise_fn = lambda x, y: fbm(x, y, seed=42)
        sample_noise_field(grid, store, "elevation", noise_fn)
        for fid in grid.faces:
            val = store.get(fid, "elevation")
            assert isinstance(val, float)

    def test_values_not_all_zero(self, grid, store):
        """Noise should produce non-trivial values (not all defaults)."""
        noise_fn = lambda x, y: fbm(x, y, seed=42)
        sample_noise_field(grid, store, "elevation", noise_fn)
        vals = [store.get(fid, "elevation") for fid in grid.faces]
        assert any(v != 0.0 for v in vals)

    def test_values_in_expected_range(self, grid, store):
        """FBM values should be roughly in [−1, 1]."""
        noise_fn = lambda x, y: fbm(x, y, seed=42)
        sample_noise_field(grid, store, "elevation", noise_fn)
        for fid in grid.faces:
            v = store.get(fid, "elevation")
            assert -1.5 <= v <= 1.5  # generous bounds

    def test_restricted_face_ids(self, grid, store):
        """Only specified faces should be modified."""
        all_fids = list(grid.faces.keys())
        target = all_fids[:3]
        rest = all_fids[3:]

        noise_fn = lambda x, y: 99.0  # constant so we can detect it
        sample_noise_field(grid, store, "elevation", noise_fn, face_ids=target)

        for fid in target:
            assert store.get(fid, "elevation") == 99.0
        for fid in rest:
            assert store.get(fid, "elevation") == 0.0  # default

    def test_determinism(self, grid, store):
        noise_fn = lambda x, y: fbm(x, y, seed=7)
        sample_noise_field(grid, store, "elevation", noise_fn)
        vals1 = {fid: store.get(fid, "elevation") for fid in grid.faces}

        # Reset and resample
        store.initialise_all()
        sample_noise_field(grid, store, "elevation", noise_fn)
        vals2 = {fid: store.get(fid, "elevation") for fid in grid.faces}

        assert vals1 == vals2


# ═══════════════════════════════════════════════════════════════════
# sample_noise_field_region
# ═══════════════════════════════════════════════════════════════════


class TestSampleNoiseFieldRegion:
    def test_only_region_faces_modified(self, grid, store):
        all_fids = list(grid.faces.keys())
        region_fids = set(all_fids[:3])
        other_fids = set(all_fids[3:])

        noise_fn = lambda x, y: 42.0
        sample_noise_field_region(grid, store, "elevation", noise_fn, region_fids)

        for fid in region_fids:
            assert store.get(fid, "elevation") == 42.0
        for fid in other_fids:
            assert store.get(fid, "elevation") == 0.0


# ═══════════════════════════════════════════════════════════════════
# smooth_field
# ═══════════════════════════════════════════════════════════════════


class TestSmoothField:
    def test_smoothing_reduces_variance(self, grid, store):
        """After smoothing, variance should decrease."""
        noise_fn = lambda x, y: fbm(x, y, seed=42, frequency=5.0)
        sample_noise_field(grid, store, "elevation", noise_fn)

        vals_before = [store.get(fid, "elevation") for fid in grid.faces]
        var_before = _variance(vals_before)

        smooth_field(grid, store, "elevation", iterations=3)

        vals_after = [store.get(fid, "elevation") for fid in grid.faces]
        var_after = _variance(vals_after)

        assert var_after <= var_before + 1e-10

    def test_uniform_field_unchanged(self, grid, store):
        """Smoothing a uniform field should leave it unchanged."""
        store.bulk_set(grid.faces.keys(), "elevation", 5.0)
        smooth_field(grid, store, "elevation", iterations=3)
        for fid in grid.faces:
            assert store.get(fid, "elevation") == pytest.approx(5.0, abs=1e-10)

    def test_respects_face_ids(self, grid, store):
        """Only specified faces should be smoothed."""
        all_fids = list(grid.faces.keys())

        # Set a spike on the first face
        store.bulk_set(grid.faces.keys(), "elevation", 0.0)
        store.set(all_fids[0], "elevation", 100.0)

        # Smooth only a subset that excludes the spike
        others = all_fids[1:]
        smooth_field(grid, store, "elevation", face_ids=others, iterations=1)

        # Spike should still be 100
        assert store.get(all_fids[0], "elevation") == 100.0

    def test_multiple_iterations(self, grid, store):
        """More iterations → more smoothing."""
        noise_fn = lambda x, y: fbm(x, y, seed=42, frequency=5.0)
        sample_noise_field(grid, store, "elevation", noise_fn)

        vals_0 = [store.get(fid, "elevation") for fid in grid.faces]
        var_0 = _variance(vals_0)

        smooth_field(grid, store, "elevation", iterations=1)
        vals_1 = [store.get(fid, "elevation") for fid in grid.faces]
        var_1 = _variance(vals_1)

        smooth_field(grid, store, "elevation", iterations=5)
        vals_5 = [store.get(fid, "elevation") for fid in grid.faces]
        var_5 = _variance(vals_5)

        assert var_5 <= var_1 + 1e-10 <= var_0 + 1e-10


# ═══════════════════════════════════════════════════════════════════
# blend_fields
# ═══════════════════════════════════════════════════════════════════


class TestBlendFields:
    def test_multiply(self, grid, store):
        """Blend via multiplication."""
        store.bulk_set(grid.faces.keys(), "elevation", 2.0)
        store.bulk_set(grid.faces.keys(), "temp", 3.0)
        blend_fields(store, "elevation", "temp", "blend_out", lambda a, b: a * b)
        for fid in grid.faces:
            assert store.get(fid, "blend_out") == pytest.approx(6.0)

    def test_add(self, grid, store):
        store.bulk_set(grid.faces.keys(), "elevation", 1.0)
        store.bulk_set(grid.faces.keys(), "temp", 0.5)
        blend_fields(store, "elevation", "temp", "blend_out", lambda a, b: a + b)
        for fid in grid.faces:
            assert store.get(fid, "blend_out") == pytest.approx(1.5)

    def test_in_place_blend(self, grid, store):
        """Output field can be the same as input field."""
        store.bulk_set(grid.faces.keys(), "elevation", 4.0)
        store.bulk_set(grid.faces.keys(), "temp", 2.0)
        blend_fields(store, "elevation", "temp", "elevation", lambda a, b: a / b)
        for fid in grid.faces:
            assert store.get(fid, "elevation") == pytest.approx(2.0)


# ═══════════════════════════════════════════════════════════════════
# clamp_field
# ═══════════════════════════════════════════════════════════════════


class TestClampField:
    def test_clamps_high(self, grid, store):
        store.bulk_set(grid.faces.keys(), "elevation", 5.0)
        clamp_field(store, "elevation", lo=0.0, hi=1.0)
        for fid in grid.faces:
            assert store.get(fid, "elevation") == 1.0

    def test_clamps_low(self, grid, store):
        store.bulk_set(grid.faces.keys(), "elevation", -5.0)
        clamp_field(store, "elevation", lo=0.0, hi=1.0)
        for fid in grid.faces:
            assert store.get(fid, "elevation") == 0.0

    def test_in_range_unchanged(self, grid, store):
        store.bulk_set(grid.faces.keys(), "elevation", 0.5)
        clamp_field(store, "elevation", lo=0.0, hi=1.0)
        for fid in grid.faces:
            assert store.get(fid, "elevation") == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════
# normalize_field
# ═══════════════════════════════════════════════════════════════════


class TestNormalizeField:
    def test_normalizes_to_range(self, grid, store):
        """After normalization, min=lo and max=hi."""
        noise_fn = lambda x, y: fbm(x, y, seed=42)
        sample_noise_field(grid, store, "elevation", noise_fn)
        normalize_field(store, "elevation", lo=0.0, hi=1.0)

        vals = [store.get(fid, "elevation") for fid in grid.faces]
        assert min(vals) == pytest.approx(0.0, abs=1e-10)
        assert max(vals) == pytest.approx(1.0, abs=1e-10)

    def test_all_in_range(self, grid, store):
        noise_fn = lambda x, y: fbm(x, y, seed=42)
        sample_noise_field(grid, store, "elevation", noise_fn)
        normalize_field(store, "elevation", lo=10.0, hi=20.0)

        for fid in grid.faces:
            v = store.get(fid, "elevation")
            assert 10.0 - 1e-10 <= v <= 20.0 + 1e-10

    def test_uniform_values(self, grid, store):
        """If all values are the same, normalize to midpoint."""
        store.bulk_set(grid.faces.keys(), "elevation", 5.0)
        normalize_field(store, "elevation", lo=0.0, hi=1.0)
        for fid in grid.faces:
            assert store.get(fid, "elevation") == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _variance(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    return sum((v - mean) ** 2 for v in vals) / n
