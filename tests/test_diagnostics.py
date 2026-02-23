from polygrid.builders import build_pentagon_centered_grid
from polygrid.diagnostics import ring_diagnostics


def test_ring_diagnostics_smoke():
    grid = build_pentagon_centered_grid(1, embed=True, embed_mode="angle")
    stats = ring_diagnostics(grid, max_ring=1)
    assert 1 in stats
    assert stats[1].protruding_lengths is not None
    assert stats[1].pointy_lengths is not None
