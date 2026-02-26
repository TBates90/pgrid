"""Tests for grid stitching / composite assembly."""

import pytest
from polygrid.builders import build_pure_hex_grid, build_pentagon_centered_grid
from polygrid.composite import stitch_grids, join_grids, split_composite, StitchSpec


class TestJoinGrids:
    def test_join_and_split(self):
        grid_a = build_pure_hex_grid(0)
        grid_b = build_pure_hex_grid(0)
        composite = join_grids({"a": grid_a, "b": grid_b})
        assert len(composite.merged.faces) == 2
        assert set(split_composite(composite).keys()) == {"a", "b"}


class TestStitchHexGrids:
    def test_stitch_two_hex_grids(self):
        """Stitch two small hex grids along one edge."""
        g1 = build_pure_hex_grid(2)
        g2 = build_pure_hex_grid(2)
        g1.compute_macro_edges(n_sides=6)
        g2.compute_macro_edges(n_sides=6)

        # Stitch edge 0 of g1 to edge 3 of g2
        composite = stitch_grids(
            {"g1": g1, "g2": g2},
            stitches=[StitchSpec(grid_a="g1", edge_a=0, grid_b="g2", edge_b=3)],
        )

        merged = composite.merged
        # Should have fewer vertices than 2x individual (shared boundary)
        total_verts = len(g1.vertices) + len(g2.vertices)
        assert len(merged.vertices) < total_verts

        # Faces should be the sum of both
        assert len(merged.faces) == len(g1.faces) + len(g2.faces)

    def test_stitch_preserves_topology(self):
        """Stitched grid should still validate."""
        g1 = build_pure_hex_grid(1)
        g2 = build_pure_hex_grid(1)
        g1.compute_macro_edges(n_sides=6)
        g2.compute_macro_edges(n_sides=6)

        composite = stitch_grids(
            {"g1": g1, "g2": g2},
            stitches=[StitchSpec(grid_a="g1", edge_a=0, grid_b="g2", edge_b=3)],
        )
        errors = composite.merged.validate()
        assert errors == []

    def test_stitch_edge_length_mismatch_raises(self):
        """Stitching grids with different ring counts should fail."""
        g1 = build_pure_hex_grid(1)
        g2 = build_pure_hex_grid(2)
        g1.compute_macro_edges(n_sides=6)
        g2.compute_macro_edges(n_sides=6)

        with pytest.raises(ValueError, match="length mismatch"):
            stitch_grids(
                {"g1": g1, "g2": g2},
                stitches=[StitchSpec(grid_a="g1", edge_a=0, grid_b="g2", edge_b=0)],
            )


class TestStitchMixed:
    def test_stitch_hex_and_pent_same_rings(self):
        """Hex and pent grids with same ring count have same boundary edge count per side."""
        g_hex = build_pure_hex_grid(3)
        g_pent = build_pentagon_centered_grid(3, embed=True, embed_mode="tutte+optimise")
        g_hex.compute_macro_edges(n_sides=6)
        g_pent.compute_macro_edges(n_sides=5)

        # Both should have the same number of vertices per macro-edge
        hex_verts = len(g_hex.macro_edges[0].vertex_ids)
        pent_verts = len(g_pent.macro_edges[0].vertex_ids)
        assert hex_verts == pent_verts

        # Stitching should work
        composite = stitch_grids(
            {"hex": g_hex, "pent": g_pent},
            stitches=[StitchSpec(grid_a="hex", edge_a=0, grid_b="pent", edge_b=0)],
        )
        assert len(composite.merged.faces) == len(g_hex.faces) + len(g_pent.faces)
