from polygrid.models import Edge, Face, Vertex
from polygrid.polygrid import PolyGrid


def test_face_adjacency():
    vertices = [
        Vertex("v1"),
        Vertex("v2"),
        Vertex("v3"),
        Vertex("v4"),
        Vertex("v5"),
        Vertex("v6"),
    ]
    edges = [
        Edge("e1", ("v1", "v2"), ("f1",)),
        Edge("e2", ("v2", "v3"), ("f1", "f2")),
        Edge("e3", ("v3", "v4"), ("f2",)),
        Edge("e4", ("v4", "v5"), ("f2",)),
        Edge("e5", ("v5", "v6"), ("f2",)),
        Edge("e6", ("v6", "v1"), ("f1",)),
    ]
    faces = [
        Face("f1", "other", ("v1", "v2", "v3", "v6"), ("e1", "e2", "e6")),
        Face("f2", "other", ("v2", "v3", "v4", "v5", "v6"), ("e2", "e3", "e4", "e5")),
    ]

    grid = PolyGrid(vertices, edges, faces)
    adjacency = grid.compute_face_neighbors()

    assert adjacency["f1"] == ["f2"]
    assert adjacency["f2"] == ["f1"]
