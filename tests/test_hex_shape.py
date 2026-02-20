from polygrid.builders import build_pure_hex_grid, hex_face_count


def test_hex_face_count_matches_formula():
    for rings in range(4):
        grid = build_pure_hex_grid(rings)
        assert len(grid.faces) == hex_face_count(rings)
