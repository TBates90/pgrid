"""Tests for the Goldberg topology and embedding pipeline."""

import math
import pytest

from polygrid.goldberg_topology import (
    build_goldberg_grid,
    goldberg_face_count,
    goldberg_topology,
    goldberg_embed_tutte,
    goldberg_optimise,
    fix_face_winding,
    _edges_from_faces,
)
from polygrid.diagnostics import min_face_signed_area, has_edge_crossings


class TestGoldbergFaceCount:
    def test_rings_0(self):
        assert goldberg_face_count(0) == 1

    def test_rings_1(self):
        assert goldberg_face_count(1) == 6

    def test_rings_2(self):
        assert goldberg_face_count(2) == 16

    def test_rings_3(self):
        assert goldberg_face_count(3) == 31

    def test_rings_12(self):
        assert goldberg_face_count(12) == 391

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            goldberg_face_count(-1)


class TestGoldbergTopology:
    """Verify combinatorial invariants of the dualized triangulation."""

    @pytest.mark.parametrize("rings", range(8))
    def test_face_count(self, rings):
        verts, edges, faces = goldberg_topology(rings)
        expected = goldberg_face_count(rings)
        assert len(faces) == expected

    @pytest.mark.parametrize("rings", range(8))
    def test_pentagon_count(self, rings):
        _, _, faces = goldberg_topology(rings)
        pentagons = [f for f in faces if f.face_type == "pent"]
        assert len(pentagons) == 1

    @pytest.mark.parametrize("rings", range(8))
    def test_face_vertex_counts(self, rings):
        _, _, faces = goldberg_topology(rings)
        for face in faces:
            if face.face_type == "pent":
                assert len(face.vertex_ids) == 5, f"{face.id} pent has {len(face.vertex_ids)} verts"
            else:
                assert len(face.vertex_ids) == 6, f"{face.id} hex has {len(face.vertex_ids)} verts"

    @pytest.mark.parametrize("rings", range(8))
    def test_interior_vertex_degree(self, rings):
        """All interior vertices must have degree 3."""
        _, edges, _ = goldberg_topology(rings)
        # Interior = vertices incident to ≥2-face edges only (non-boundary)
        boundary_vids = set()
        for e in edges:
            if len(e.face_ids) < 2:
                boundary_vids.update(e.vertex_ids)

        degree = {}
        for e in edges:
            for vid in e.vertex_ids:
                degree[vid] = degree.get(vid, 0) + 1

        for vid, deg in degree.items():
            if vid not in boundary_vids:
                assert deg == 3, f"Interior vertex {vid} has degree {deg}"

    @pytest.mark.parametrize("rings", [1, 2, 3, 4])
    def test_boundary_vertex_count(self, rings):
        """Boundary should have 5*(2R+1) vertices."""
        _, edges, _ = goldberg_topology(rings)
        boundary_vids = set()
        for e in edges:
            if len(e.face_ids) < 2:
                boundary_vids.update(e.vertex_ids)
        assert len(boundary_vids) == 5 * (2 * rings + 1)

    def test_no_duplicate_vertex_ids_in_faces(self):
        for rings in range(6):
            _, _, faces = goldberg_topology(rings)
            for face in faces:
                assert len(set(face.vertex_ids)) == len(face.vertex_ids), (
                    f"rings={rings} {face.id}: duplicate vids"
                )


class TestGoldbergEmbedding:
    """Verify embedding quality for small ring counts."""

    @pytest.mark.parametrize("rings", range(6))
    def test_no_crossings(self, rings):
        grid = build_goldberg_grid(rings)
        assert not has_edge_crossings(grid), f"rings={rings}: edge crossings detected"

    @pytest.mark.parametrize("rings", range(6))
    def test_all_positive_areas(self, rings):
        grid = build_goldberg_grid(rings)
        area = min_face_signed_area(grid)
        assert area > 0, f"rings={rings}: min signed area = {area}"

    @pytest.mark.parametrize("rings", range(6))
    def test_all_vertices_positioned(self, rings):
        grid = build_goldberg_grid(rings)
        for vid, v in grid.vertices.items():
            assert v.has_position(), f"rings={rings}: vertex {vid} has no position"

    @pytest.mark.parametrize("rings", range(6))
    def test_grid_validates(self, rings):
        grid = build_goldberg_grid(rings)
        errors = grid.validate(strict=True)
        assert not errors, f"rings={rings}: {errors}"


class TestBuildGoldbergGrid:
    """Integration tests via the high-level builder."""

    def test_rings_0_single_pentagon(self):
        grid = build_goldberg_grid(0)
        assert len(grid.faces) == 1
        face = list(grid.faces.values())[0]
        assert face.face_type == "pent"
        assert len(face.vertex_ids) == 5

    def test_metadata(self):
        grid = build_goldberg_grid(3)
        assert grid.metadata.get("generator") == "goldberg"
        assert grid.metadata.get("rings") == 3

    def test_no_optimise(self):
        """Even without optimise, Tutte embedding should be valid."""
        grid = build_goldberg_grid(3, optimise=False)
        assert not has_edge_crossings(grid)
        assert min_face_signed_area(grid) > 0

    def test_serialization_roundtrip(self):
        from polygrid.polygrid import PolyGrid

        grid = build_goldberg_grid(2)
        data = grid.to_dict()
        grid2 = PolyGrid.from_dict(data)
        assert len(grid2.faces) == len(grid.faces)
        assert len(grid2.vertices) == len(grid.vertices)
