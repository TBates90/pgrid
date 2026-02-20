from polygrid.builders import build_pure_hex_grid
from polygrid.composite import join_grids, split_composite


def test_join_and_split_composite():
    grid_a = build_pure_hex_grid(0)
    grid_b = build_pure_hex_grid(0)

    composite = join_grids({"a": grid_a, "b": grid_b})

    assert len(composite.merged.faces) == 2
    assert set(split_composite(composite).keys()) == {"a", "b"}
