from polygrid.models import Edge, Face, Vertex
from polygrid.polygrid import PolyGrid


def test_round_trip_json():
    vertices = [
        Vertex("v1", 0, 0),
        Vertex("v2", 1, 0),
        Vertex("v3", 1, 1),
        Vertex("v4", 0, 1),
    ]
    edges = [
        Edge("e1", ("v1", "v2"), ("f1",)),
        Edge("e2", ("v2", "v3"), ("f1",)),
        Edge("e3", ("v3", "v4"), ("f1",)),
        Edge("e4", ("v4", "v1"), ("f1",)),
    ]
    faces = [
        Face("f1", "other", ("v1", "v2", "v3", "v4"), ("e1", "e2", "e3", "e4"))
    ]
    grid = PolyGrid(vertices, edges, faces)

    json_data = grid.to_json()
    loaded = PolyGrid.from_json(json_data)

    assert loaded.vertices.keys() == grid.vertices.keys()
    assert loaded.edges.keys() == grid.edges.keys()
    assert loaded.faces.keys() == grid.faces.keys()
