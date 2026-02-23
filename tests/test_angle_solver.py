import math

import pytest

from polygrid.angle_solver import ring_angle_spec, solve_ring_hex_lengths
from polygrid import angle_solver


def test_ring_angle_spec_math():
    ring1 = ring_angle_spec(1)
    assert ring1.inner_angle_deg == pytest.approx(126.0)
    assert ring1.outer_angle_deg == pytest.approx(117.0)

    ring2 = ring_angle_spec(2)
    assert ring2.inner_angle_deg == pytest.approx(121.5)
    assert ring2.outer_angle_deg == pytest.approx(119.25)


def test_ring_length_solver_closure():
    result = solve_ring_hex_lengths(1, inner_edge_length=1.0)
    assert result["inner"] > 0
    assert result["protrude"] > 0
    assert result["outer"] > 0
    assert result["residual"] < 0.2
