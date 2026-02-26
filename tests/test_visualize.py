"""Tests for the visualize module (rendering to PNG)."""

import os
import tempfile
from pathlib import Path

import pytest

from polygrid.assembly import pent_hex_assembly
from polygrid.builders import build_pure_hex_grid
from polygrid.transforms import apply_voronoi
from polygrid.visualize import (
    render_assembly_panels,
    render_single_panel,
)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestRenderSinglePanel:
    def test_renders_grid(self, tmp_dir):
        grid = build_pure_hex_grid(2)
        out = tmp_dir / "single.png"
        render_single_panel(grid, out, title="Test grid")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_renders_with_overlay(self, tmp_dir):
        grid = build_pure_hex_grid(2)
        overlay = apply_voronoi(grid)
        out = tmp_dir / "single_overlay.png"
        render_single_panel(grid, out, overlay=overlay, title="Grid + Voronoi")
        assert out.exists()
        assert out.stat().st_size > 0


class TestRenderAssemblyPanels:
    def test_produces_png(self, tmp_dir):
        plan = pent_hex_assembly(rings=1)
        composite = plan.build()
        overlay = apply_voronoi(composite.merged)
        out = tmp_dir / "panels.png"
        render_assembly_panels(plan, out, overlay=overlay)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_without_overlay(self, tmp_dir):
        plan = pent_hex_assembly(rings=1)
        out = tmp_dir / "panels_no_overlay.png"
        render_assembly_panels(plan, out)
        assert out.exists()
        assert out.stat().st_size > 0
