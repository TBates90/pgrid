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


def test_metadata_has_corner_vertex_ids():
    """build_goldberg_grid stores the 5 corner vertex IDs in metadata."""
    grid = build_pentagon_centered_grid(2, embed=True)
    corner_ids = grid.metadata.get("corner_vertex_ids")
    assert corner_ids is not None
    assert len(corner_ids) == 5
    # All corner IDs must be actual vertices in the grid
    for vid in corner_ids:
        assert vid in grid.vertices, f"corner {vid} not in grid vertices"


def test_corner_vertex_ids_at_boundary():
    """Corner vertices should be among the outermost vertices."""
    import numpy as np

    grid = build_pentagon_centered_grid(3, embed=True)
    corner_ids = grid.metadata["corner_vertex_ids"]

    all_pos = []
    for v in grid.vertices.values():
        if v.has_position():
            all_pos.append(np.array([v.x, v.y]))
    arr = np.array(all_pos)
    centroid = arr.mean(axis=0)
    dists = np.linalg.norm(arr - centroid, axis=1)
    max_dist = dists.max()

    for vid in corner_ids:
        v = grid.vertices[vid]
        d = np.linalg.norm(np.array([v.x, v.y]) - centroid)
        assert d > max_dist * 0.85, (
            f"Corner {vid} at dist {d:.4f} is not near boundary "
            f"(max_dist={max_dist:.4f})"
        )


def test_rings_zero_corner_vertex_ids():
    """Rings=0 (single pentagon) should still have corner_vertex_ids."""
    grid = build_pentagon_centered_grid(0, embed=True)
    corner_ids = grid.metadata.get("corner_vertex_ids")
    assert corner_ids is not None
    assert len(corner_ids) == 5
