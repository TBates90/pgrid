from polygrid.builders import build_pure_hex_grid


def test_build_pure_hex_grid_counts():
    grid = build_pure_hex_grid(1)
    assert len(grid.faces) == 7
    # Ensure edges and vertices were created
    assert len(grid.edges) > 0
    assert len(grid.vertices) > 0
