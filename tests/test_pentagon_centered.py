"""Tests for pentagon-centred grid builder and topology validation."""

from polygrid.builders import build_pentagon_centered_grid, validate_pentagon_topology
from polygrid.algorithms import build_face_adjacency, ring_faces
from polygrid.geometry import face_vertex_cycle


def test_pentagon_centered_grid_center_face():
    grid = build_pentagon_centered_grid(1, embed=False)
    pent_faces = [face for face in grid.faces.values() if face.face_type == "pent"]
    assert len(pent_faces) == 1
    assert len(pent_faces[0].vertex_ids) == 5
    assert all(len(face.vertex_ids) in (5, 6) for face in grid.faces.values())


def test_pentagon_topology_validation():
    grid = build_pentagon_centered_grid(2, embed=False)
    assert validate_pentagon_topology(grid, rings=2) == []


def test_pentagon_strict_validation():
    grid = build_pentagon_centered_grid(2, embed=False)
    assert grid.validate(strict=True) == []


def test_face_vertex_cycle_ordering():
    grid = build_pentagon_centered_grid(2, embed=False)
    for face in grid.faces.values():
        cycle = face_vertex_cycle(face, grid.edges.values())
        assert set(cycle) == set(face.vertex_ids)
        assert len(cycle) == len(face.vertex_ids)


def test_global_optimizer_smoke():
    grid = build_pentagon_centered_grid(2, embed=True, embed_mode="tutte+optimise")
    pent_face = next(face for face in grid.faces.values() if face.face_type == "pent")
    assert len(pent_face.vertex_ids) == 5


def test_metadata_has_sides():
    grid = build_pentagon_centered_grid(2, embed=False)
    assert grid.metadata.get("generator") == "goldberg"
