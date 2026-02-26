"""Tests for the transforms module (Voronoi, partition, etc.)."""

import pytest

from polygrid.builders import build_pure_hex_grid, build_pentagon_centered_grid
from polygrid.transforms import Overlay, apply_voronoi, apply_partition


class TestVoronoiBasic:
    def test_returns_overlay(self):
        grid = build_pure_hex_grid(1)
        overlay = apply_voronoi(grid)
        assert isinstance(overlay, Overlay)
        assert overlay.kind == "voronoi"

    def test_sites_equal_face_count(self):
        grid = build_pure_hex_grid(2)
        overlay = apply_voronoi(grid)
        assert len(overlay.points) == len(grid.faces)

    def test_segments_nonempty(self):
        grid = build_pure_hex_grid(2)
        overlay = apply_voronoi(grid)
        assert len(overlay.segments) > 0

    def test_regions_nonempty(self):
        grid = build_pure_hex_grid(2)
        overlay = apply_voronoi(grid)
        assert len(overlay.regions) > 0

    def test_metadata_populated(self):
        grid = build_pure_hex_grid(1)
        overlay = apply_voronoi(grid)
        assert overlay.metadata["n_sites"] == len(grid.faces)
        assert overlay.metadata["n_segments"] > 0


class TestVoronoiHex:
    def test_hex_grid_segments(self):
        """Every internal edge should produce a dual segment."""
        grid = build_pure_hex_grid(2)
        overlay = apply_voronoi(grid)
        # Internal edges have 2 faces; boundary edges have 1.
        internal_edges = [e for e in grid.edges.values() if len(e.face_ids) == 2]
        boundary_edges = [e for e in grid.edges.values() if len(e.face_ids) == 1]
        # Each internal edge → 1 segment, each boundary edge → 1 segment to midpoint
        assert len(overlay.segments) == len(internal_edges) + len(boundary_edges)

    def test_all_sites_inside_grid(self):
        """Sites should be face centroids, i.e. inside the grid bbox."""
        grid = build_pure_hex_grid(2)
        overlay = apply_voronoi(grid)
        xs = [v.x for v in grid.vertices.values() if v.has_position()]
        ys = [v.y for v in grid.vertices.values() if v.has_position()]
        for pt in overlay.points:
            assert min(xs) - 1e-6 <= pt.x <= max(xs) + 1e-6
            assert min(ys) - 1e-6 <= pt.y <= max(ys) + 1e-6


class TestVoronoiPent:
    def test_pent_grid_sites(self):
        grid = build_pentagon_centered_grid(2, embed=True, embed_mode="tutte+optimise")
        overlay = apply_voronoi(grid)
        assert len(overlay.points) == len(grid.faces)

    def test_pent_grid_segments(self):
        grid = build_pentagon_centered_grid(2, embed=True, embed_mode="tutte+optimise")
        overlay = apply_voronoi(grid)
        assert len(overlay.segments) > 0


class TestVoronoiComposite:
    def test_voronoi_on_stitched(self):
        """Voronoi should work on a stitched composite grid."""
        from polygrid.assembly import pent_hex_assembly

        plan = pent_hex_assembly(rings=1)
        composite = plan.build()
        overlay = apply_voronoi(composite.merged)
        assert len(overlay.points) == len(composite.merged.faces)
        assert len(overlay.segments) > 0


class TestPartitionBasic:
    def test_returns_overlay(self):
        grid = build_pure_hex_grid(2)
        overlay = apply_partition(grid, n_sections=4)
        assert isinstance(overlay, Overlay)
        assert overlay.kind == "partition"

    def test_regions_equal_face_count(self):
        grid = build_pure_hex_grid(2)
        overlay = apply_partition(grid, n_sections=4)
        assert len(overlay.regions) == len(grid.faces)

    def test_metadata(self):
        grid = build_pure_hex_grid(2)
        overlay = apply_partition(grid, n_sections=6)
        assert overlay.metadata["n_sections"] == 6
        assert len(overlay.metadata["section_assignments"]) == len(grid.faces)

    def test_all_sections_used_hex(self):
        """With enough faces, most sections should get at least one face."""
        grid = build_pure_hex_grid(3)
        overlay = apply_partition(grid, n_sections=4)
        used = set(overlay.metadata["section_assignments"].values())
        assert len(used) >= 2  # at least 2 sections used

    def test_partition_on_composite(self):
        """Partition should work on a stitched composite grid."""
        from polygrid.assembly import pent_hex_assembly

        plan = pent_hex_assembly(rings=2)
        composite = plan.build()
        overlay = apply_partition(composite.merged, n_sections=8)
        assert len(overlay.regions) == len(composite.merged.faces)
        assert overlay.metadata["n_sections"] == 8
