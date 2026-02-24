from polygrid.builders import build_pentagon_centered_grid


def test_pentagon_grid_json_determinism():
    grid_a = build_pentagon_centered_grid(1, embed=True, embed_mode="tutte+optimise")
    grid_b = build_pentagon_centered_grid(1, embed=True, embed_mode="tutte+optimise")
    assert grid_a.to_json() == grid_b.to_json()


def test_pentagon_grid_json_determinism_no_embed():
    grid_a = build_pentagon_centered_grid(1, embed=False)
    grid_b = build_pentagon_centered_grid(1, embed=False)
    assert grid_a.to_json() == grid_b.to_json()
