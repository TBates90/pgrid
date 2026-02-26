"""Tests for diagnostics module."""

from polygrid.builders import build_pentagon_centered_grid, build_pure_hex_grid
from polygrid.diagnostics import (
    ring_diagnostics,
    min_face_signed_area,
    has_edge_crossings,
    ring_quality_gates,
    diagnostics_report,
    ring_angle_spec,
)


def test_ring_diagnostics_smoke():
    grid = build_pentagon_centered_grid(1, embed=True, embed_mode="tutte+optimise")
    stats = ring_diagnostics(grid, max_ring=1)
    assert 1 in stats
    assert stats[1].protruding_lengths is not None
    assert stats[1].pointy_lengths is not None


def test_quality_gate_diagnostics_smoke():
    grid = build_pentagon_centered_grid(1, embed=True, embed_mode="tutte+optimise")
    assert min_face_signed_area(grid) != 0.0
    assert isinstance(has_edge_crossings(grid), bool)
    grid_no_embed = build_pentagon_centered_grid(1, embed=False)
    assert has_edge_crossings(grid_no_embed) is False


def test_ring_quality_gates_smoke():
    grid = build_pentagon_centered_grid(1, embed=True, embed_mode="tutte+optimise")
    stats = ring_diagnostics(grid, max_ring=1)
    quality = ring_quality_gates(stats[1])
    assert "passed" in quality


def test_diagnostics_report_smoke():
    grid = build_pentagon_centered_grid(1, embed=True, embed_mode="tutte+optimise")
    report = diagnostics_report(grid, max_ring=1)
    assert "min_face_signed_area" in report
    assert "edge_crossings" in report
    assert "rings" in report


def test_embedding_quality_gates():
    pent_grid = build_pentagon_centered_grid(1, embed=True, embed_mode="tutte+optimise")
    assert abs(min_face_signed_area(pent_grid)) > 0.0
    hex_grid = build_pure_hex_grid(1)
    assert abs(min_face_signed_area(hex_grid)) > 0.0
    assert has_edge_crossings(hex_grid) is False


def test_ring_angle_spec():
    import pytest
    spec1 = ring_angle_spec(1)
    assert spec1.inner_angle_deg == pytest.approx(126.0)
    assert spec1.outer_angle_deg == pytest.approx(117.0)

    spec2 = ring_angle_spec(2)
    assert spec2.inner_angle_deg == pytest.approx(123.0)
    assert spec2.outer_angle_deg == pytest.approx(118.5)
