"""Phase 1–4 core topology smoke-tests.

Consolidates the single-test files that each exercised one aspect
of the PolyGrid data-model:

  • test_adjacency.py     → TestFaceAdjacency
  • test_build_hex.py     → TestBuildHex
  • test_composite.py     → TestComposite
  • test_hex_shape.py     → TestHexShape
  • test_rings.py         → TestRingFaces
  • test_serialization.py → TestSerialization

All originals deleted — see Phase 15D in TASKLIST.md.
"""

from polygrid.core.models import Edge, Face, Vertex
from polygrid.core.polygrid import PolyGrid
from polygrid.builders import build_pure_hex_grid, hex_face_count
from polygrid.core.algorithms import ring_faces
from polygrid.composite import join_grids, split_composite


# ═══════════════════════════════════════════════════════════════════
# Face adjacency (was test_adjacency.py)
# ═══════════════════════════════════════════════════════════════════

class TestFaceAdjacency:
    def test_face_adjacency(self):
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


# ═══════════════════════════════════════════════════════════════════
# Build hex grid (was test_build_hex.py)
# ═══════════════════════════════════════════════════════════════════

class TestBuildHex:
    def test_build_pure_hex_grid_counts(self):
        grid = build_pure_hex_grid(1)
        assert len(grid.faces) == 7
        assert len(grid.edges) > 0
        assert len(grid.vertices) > 0


# ═══════════════════════════════════════════════════════════════════
# Composite grids (was test_composite.py)
# ═══════════════════════════════════════════════════════════════════

class TestComposite:
    def test_join_and_split_composite(self):
        grid_a = build_pure_hex_grid(0)
        grid_b = build_pure_hex_grid(0)

        composite = join_grids({"a": grid_a, "b": grid_b})

        assert len(composite.merged.faces) == 2
        assert set(split_composite(composite).keys()) == {"a", "b"}


# ═══════════════════════════════════════════════════════════════════
# Hex face count formula (was test_hex_shape.py)
# ═══════════════════════════════════════════════════════════════════

class TestHexShape:
    def test_hex_face_count_matches_formula(self):
        for rings in range(4):
            grid = build_pure_hex_grid(rings)
            assert len(grid.faces) == hex_face_count(rings)


# ═══════════════════════════════════════════════════════════════════
# Ring BFS (was test_rings.py)
# ═══════════════════════════════════════════════════════════════════

class TestRingFaces:
    def test_ring_faces_bfs(self):
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


# ═══════════════════════════════════════════════════════════════════
# JSON round-trip serialization (was test_serialization.py)
# ═══════════════════════════════════════════════════════════════════

class TestSerialization:
    def test_round_trip_json(self):
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
