from polygrid import PolyGrid
from polygrid.builders import build_pure_hex_grid
from polygrid.detail import build_hex_prism, extrude_polygrid_column


class TestHexPrism:
    def test_build_hex_prism_counts_and_roles(self):
        grid = build_hex_prism(radius=1.25, height=0.3)

        assert len(grid.vertices) == 12
        assert len(grid.edges) == 18
        assert len(grid.faces) == 8
        assert grid.metadata["generator"] == "hex-prism"
        assert grid.metadata["tile_like"] is True

        roles = [face.metadata.get("surface_role") for face in grid.faces.values()]
        assert roles.count("top") == 1
        assert roles.count("side") == 6
        assert roles.count("base") == 1

        top_z = {grid.vertices[vid].z for vid in grid.faces["footprint"].vertex_ids}
        assert top_z == {0.3}

    def test_hex_prism_round_trip_preserves_3d_positions(self):
        grid = build_hex_prism(height=0.4)

        loaded = PolyGrid.from_dict(grid.to_dict())

        assert loaded.metadata["generator"] == "hex-prism"
        assert all(vertex.z is not None for vertex in loaded.vertices.values())
        assert loaded.faces["footprint"].metadata["surface_role"] == "top"


class TestPolygridColumn:
    def test_extrude_polygrid_column_uses_only_boundary_walls(self):
        source = build_pure_hex_grid(1)

        grid = extrude_polygrid_column(source, 0.45)

        assert len(grid.vertices) == len(source.vertices) * 2
        assert len(grid.metadata["top_face_ids"]) == len(source.faces)
        assert len(grid.metadata["side_face_ids"]) == len(source.boundary_edges())
        assert grid.metadata["base_face_id"] == "base"

        side_faces = [face for face in grid.faces.values() if face.metadata.get("surface_role") == "side"]
        assert len(side_faces) == len(source.boundary_edges())

    def test_extrude_polygrid_column_requires_positioned_vertices(self):
        grid = PolyGrid([], [], [])

        try:
            extrude_polygrid_column(grid, 0.5)
        except ValueError as exc:
            assert "simple outer boundary" in str(exc) or "positioned 2D vertices" in str(exc)
        else:
            raise AssertionError("expected ValueError for invalid source grid")