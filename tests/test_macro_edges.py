"""Tests for macro-edges and boundary detection."""

import pytest
from polygrid.builders import build_pure_hex_grid, build_pentagon_centered_grid
from polygrid.models import MacroEdge


class TestHexMacroEdges:
    def test_hex_grid_has_6_macro_edges(self):
        grid = build_pure_hex_grid(3)
        macro = grid.compute_macro_edges(n_sides=6)
        assert len(macro) == 6

    def test_hex_macro_edge_vertex_counts(self):
        """Each macro-edge of a rings=3 hex grid should have 2*3+2=8 vertices."""
        grid = build_pure_hex_grid(3)
        macro = grid.compute_macro_edges(n_sides=6)
        for me in macro:
            assert len(me.vertex_ids) == 2 * 3 + 2  # 8 vertices per side

    def test_hex_macro_edge_edge_counts(self):
        """Each macro-edge should have one fewer edge than vertices."""
        grid = build_pure_hex_grid(3)
        macro = grid.compute_macro_edges(n_sides=6)
        for me in macro:
            assert len(me.edge_ids) == len(me.vertex_ids) - 1

    def test_hex_corners_shared(self):
        """Adjacent macro-edges share corner vertices."""
        grid = build_pure_hex_grid(3)
        macro = grid.compute_macro_edges(n_sides=6)
        for i in range(6):
            nxt = (i + 1) % 6
            assert macro[i].corner_end == macro[nxt].corner_start

    def test_macro_edges_cover_boundary(self):
        """All boundary vertices should appear in macro-edges."""
        grid = build_pure_hex_grid(2)
        cycle = grid.boundary_vertex_cycle()
        macro = grid.compute_macro_edges(n_sides=6)
        macro_verts = set()
        for me in macro:
            macro_verts.update(me.vertex_ids)
        assert set(cycle) == macro_verts

    def test_stored_in_grid(self):
        """compute_macro_edges stores result in grid.macro_edges."""
        grid = build_pure_hex_grid(2)
        assert grid.macro_edges == []
        grid.compute_macro_edges(n_sides=6)
        assert len(grid.macro_edges) == 6


class TestPentMacroEdges:
    def test_pent_grid_has_5_macro_edges(self):
        grid = build_pentagon_centered_grid(3, embed=True, embed_mode="tutte+optimise")
        macro = grid.compute_macro_edges(n_sides=5)
        assert len(macro) == 5

    def test_pent_macro_edge_vertex_counts(self):
        """Each macro-edge of a rings=3 pent grid should have 2*3+2=8 vertices."""
        grid = build_pentagon_centered_grid(3, embed=True, embed_mode="tutte+optimise")
        macro = grid.compute_macro_edges(n_sides=5)
        for me in macro:
            assert len(me.vertex_ids) == 2 * 3 + 2

    def test_pent_corners_shared(self):
        grid = build_pentagon_centered_grid(3, embed=True, embed_mode="tutte+optimise")
        macro = grid.compute_macro_edges(n_sides=5)
        for i in range(5):
            nxt = (i + 1) % 5
            assert macro[i].corner_end == macro[nxt].corner_start


class TestMacroEdgeSerialization:
    def test_round_trip(self):
        grid = build_pure_hex_grid(2)
        grid.compute_macro_edges(n_sides=6)
        json_str = grid.to_json()
        from polygrid.polygrid import PolyGrid
        loaded = PolyGrid.from_json(json_str)
        assert len(loaded.macro_edges) == 6
        for orig, loaded_me in zip(grid.macro_edges, loaded.macro_edges):
            assert orig.id == loaded_me.id
            assert orig.vertex_ids == loaded_me.vertex_ids
            assert orig.corner_start == loaded_me.corner_start
            assert orig.corner_end == loaded_me.corner_end
