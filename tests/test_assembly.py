"""Tests for the assembly module."""

import math
import pytest

from polygrid.assembly import (
    AssemblyPlan,
    pent_hex_assembly,
    translate_grid,
    rotate_grid,
)
from polygrid.builders import build_pure_hex_grid, build_pentagon_centered_grid
from polygrid.composite import StitchSpec


class TestTranslateGrid:
    def test_shifts_vertices(self):
        grid = build_pure_hex_grid(1)
        shifted = translate_grid(grid, 10.0, 20.0)
        for vid in grid.vertices:
            orig = grid.vertices[vid]
            new = shifted.vertices[vid]
            assert abs(new.x - (orig.x + 10.0)) < 1e-10
            assert abs(new.y - (orig.y + 20.0)) < 1e-10

    def test_preserves_topology(self):
        grid = build_pure_hex_grid(1)
        shifted = translate_grid(grid, 5.0, 5.0)
        assert len(shifted.faces) == len(grid.faces)
        assert len(shifted.edges) == len(grid.edges)
        assert shifted.validate() == []


class TestRotateGrid:
    def test_90_degree_rotation(self):
        grid = build_pure_hex_grid(1)
        rotated = rotate_grid(grid, math.pi / 2)
        # A point at (1, 0) should become (0, 1)
        for vid in grid.vertices:
            orig = grid.vertices[vid]
            new = rotated.vertices[vid]
            expected_x = -orig.y
            expected_y = orig.x
            assert abs(new.x - expected_x) < 1e-10
            assert abs(new.y - expected_y) < 1e-10

    def test_preserves_topology(self):
        grid = build_pure_hex_grid(1)
        rotated = rotate_grid(grid, 0.5)
        assert len(rotated.faces) == len(grid.faces)
        assert rotated.validate() == []


class TestAssemblyPlan:
    def test_empty_plan_builds(self):
        plan = AssemblyPlan()
        grid = build_pure_hex_grid(0)
        plan.components["a"] = grid
        composite = plan.build()
        assert len(composite.merged.faces) == 1

    def test_plan_with_stitch(self):
        g1 = build_pure_hex_grid(1)
        g2 = build_pure_hex_grid(1)
        g1.compute_macro_edges(n_sides=6)
        g2.compute_macro_edges(n_sides=6)
        g2 = translate_grid(g2, 10.0, 0.0)
        g2.compute_macro_edges(n_sides=6)

        plan = AssemblyPlan(
            components={"g1": g1, "g2": g2},
            stitches=[StitchSpec(grid_a="g1", edge_a=0, grid_b="g2", edge_b=3)],
        )
        composite = plan.build()
        assert len(composite.merged.faces) == len(g1.faces) + len(g2.faces)


class TestPentHexAssembly:
    def test_has_6_components(self):
        plan = pent_hex_assembly(rings=2)
        assert len(plan.components) == 6
        assert "pent" in plan.components
        for i in range(5):
            assert f"hex{i}" in plan.components

    def test_has_10_stitches(self):
        """5 pent-hex + 5 hex-hex stitches."""
        plan = pent_hex_assembly(rings=2)
        assert len(plan.stitches) == 10
        pent_hex = [s for s in plan.stitches if s.grid_a == "pent"]
        hex_hex = [s for s in plan.stitches if s.grid_a.startswith("hex")]
        assert len(pent_hex) == 5
        assert len(hex_hex) == 5

    def test_stitch_specs_reference_valid_components(self):
        plan = pent_hex_assembly(rings=2)
        for spec in plan.stitches:
            assert spec.grid_a in plan.components
            assert spec.grid_b in plan.components

    def test_all_components_have_macro_edges(self):
        plan = pent_hex_assembly(rings=2)
        for name, grid in plan.components.items():
            assert len(grid.macro_edges) > 0, f"{name} has no macro-edges"

    def test_macro_edge_compatibility(self):
        """Stitched edges should have the same vertex count."""
        plan = pent_hex_assembly(rings=2)
        for spec in plan.stitches:
            me_a = next(m for m in plan.components[spec.grid_a].macro_edges
                        if m.id == spec.edge_a)
            me_b = next(m for m in plan.components[spec.grid_b].macro_edges
                        if m.id == spec.edge_b)
            assert len(me_a.vertex_ids) == len(me_b.vertex_ids)

    def test_builds_composite(self):
        plan = pent_hex_assembly(rings=2)
        composite = plan.build()
        # 1 pent grid + 5 hex grids, all with rings=2
        pent_faces = len(plan.components["pent"].faces)
        hex_faces = sum(len(plan.components[f"hex{i}"].faces) for i in range(5))
        assert len(composite.merged.faces) == pent_faces + hex_faces

    def test_composite_validates(self):
        plan = pent_hex_assembly(rings=1)
        composite = plan.build()
        errors = composite.merged.validate()
        assert errors == []

    def test_different_ring_counts(self):
        for rings in [1, 2, 3]:
            plan = pent_hex_assembly(rings=rings)
            composite = plan.build()
            assert len(composite.merged.faces) > 0

    def test_hex_hex_boundaries_aligned(self):
        """Adjacent hex grids must have perfectly overlapping boundary edges."""
        plan = pent_hex_assembly(rings=2)
        for spec in plan.stitches:
            if not spec.grid_a.startswith("hex"):
                continue
            ga = plan.components[spec.grid_a]
            gb = plan.components[spec.grid_b]
            me_a = next(m for m in ga.macro_edges if m.id == spec.edge_a)
            me_b = next(m for m in gb.macro_edges if m.id == spec.edge_b)
            vids_a = list(me_a.vertex_ids)
            vids_b = list(me_b.vertex_ids)[::-1]
            for va_id, vb_id in zip(vids_a, vids_b):
                a = ga.vertices[va_id]
                b = gb.vertices[vb_id]
                assert math.hypot(a.x - b.x, a.y - b.y) < 1e-10

    def test_hex_grids_on_outside_of_pent(self):
        """The farthest hex corner (corner 0, opposite the shared edge)
        should be farther from the pent centre than the shared edge,
        confirming the hex body is on the outside."""
        plan = pent_hex_assembly(rings=2)
        pent = plan.components["pent"]
        pcx = sum(v.x for v in pent.vertices.values() if v.has_position()) / \
              sum(1 for v in pent.vertices.values() if v.has_position())
        pcy = sum(v.y for v in pent.vertices.values() if v.has_position()) / \
              sum(1 for v in pent.vertices.values() if v.has_position())

        for i in range(5):
            h = plan.components[f"hex{i}"]
            # Corner 0 is opposite to the shared edge (edge 3)
            me0 = next(m for m in sorted(h.macro_edges, key=lambda m: m.id)
                       if m.id == 0)
            c0 = h.vertices[me0.vertex_ids[0]]
            r_far = math.hypot(c0.x - pcx, c0.y - pcy)

            # Shared edge midpoint
            me3 = next(m for m in pent.macro_edges if m.id == i)
            mx = sum(pent.vertices[vid].x for vid in me3.vertex_ids) / len(me3.vertex_ids)
            my = sum(pent.vertices[vid].y for vid in me3.vertex_ids) / len(me3.vertex_ids)
            r_edge = math.hypot(mx - pcx, my - pcy)

            assert r_far > r_edge, f"hex{i} corner 0 should be farther than pent edge"
