from polygrid.builders import build_pure_hex_grid
from polygrid.algorithms import ring_faces


def test_ring_faces_bfs():
    grid = build_pure_hex_grid(1)
    adjacency = grid.compute_face_neighbors()
    center_face = min(
        grid.faces.values(),
        key=lambda face: (
            sum(grid.vertices[vid].x for vid in face.vertex_ids) / len(face.vertex_ids)
        ) ** 2
        + (
            sum(grid.vertices[vid].y for vid in face.vertex_ids) / len(face.vertex_ids)
        ) ** 2,
    ).id

    rings = ring_faces(adjacency, center_face, max_depth=2)

    assert rings[0] == [center_face]
    assert len(rings[1]) == 6
