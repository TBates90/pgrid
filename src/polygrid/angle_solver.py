from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover
    least_squares = None


@dataclass(frozen=True)
class RingAngleSpec:
    ring: int
    inner_angle_deg: float
    outer_angle_deg: float


DEFAULT_RING_SPECS: Dict[int, RingAngleSpec] = {
    1: RingAngleSpec(ring=1, inner_angle_deg=126.0, outer_angle_deg=117.0),
    2: RingAngleSpec(ring=2, inner_angle_deg=121.5, outer_angle_deg=119.25),
}


def ring_angle_spec(ring: int) -> RingAngleSpec:
    """Return the angle specification for a given ring.

    The first two rings are fixed based on pentagon (108°) and ring-1 (117°) propagation.
    For larger rings, we approximate by propagating the previous ring's outer angle.
    """
    if ring < 1:
        raise ValueError("ring must be >= 1")
    if ring in DEFAULT_RING_SPECS:
        return DEFAULT_RING_SPECS[ring]

    prev = ring_angle_spec(ring - 1)
    inner = (360.0 - prev.outer_angle_deg) / 2.0
    outer = (720.0 - inner * 2.0) / 4.0
    return RingAngleSpec(ring=ring, inner_angle_deg=inner, outer_angle_deg=outer)


def solve_ring_hex_lengths(
    ring: int,
    inner_edge_length: float | None = None,
    max_iter: int = 200,
) -> Dict[str, float]:
    """Solve symmetric hex edge lengths for a ring given fixed angles.

    We assume a symmetric edge pattern [d, a, c, c, c, a] where:
    - d = inner edge length (shared with inner ring)
    - a = protruding edge length
    - c = outer edge length
    Returns lengths and the closure residual.
    """
    reg_weight = 0.12 if ring == 1 else 0.05
    min_ratio = 0.5 if ring == 1 else 0.2

    if least_squares is None:
        from math import sqrt

        spec = ring_angle_spec(ring)
        inner = spec.inner_angle_deg
        outer = spec.outer_angle_deg
        angles = [inner, outer, outer, outer, outer, inner]
        base_inner = inner_edge_length if inner_edge_length is not None else 1.0
        # Simple fallback optimizer when SciPy is not available.
        # Do a coarse grid search over reasonable ranges for protrude and outer
        # then perform a small local coordinate-descent refinement. This avoids
        # pathological shrinking to the lower bound and produces a usable
        # closure with modest residuals.
        def residual_for(p, c):
            x, y = _polygon_closure([base_inner, p, c, c, c, p], angles)
            residual = math.hypot(x, y)
            # penalize shrinking too far below the inner length to avoid collapsed rings
            penalty = reg_weight * (max(0.0, base_inner - p) + max(0.0, base_inner - c))
            return residual + penalty

        # coarse search ranges
        min_val = 1e-2
        low, high = min_ratio * base_inner, 3.0 * base_inner
        best_p, best_c = base_inner, base_inner
        best_err = residual_for(best_p, best_c)

        steps = 30
        for i in range(steps):
            p = low + (high - low) * i / (steps - 1)
            for j in range(steps):
                c = low + (high - low) * j / (steps - 1)
                err = residual_for(p, c)
                if err < best_err:
                    best_err = err
                    best_p, best_c = p, c

        # small local refinement via coordinate descent
        for _ in range(40):
            improved = False
            for delta in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
                for candidate in ((best_p * (1 - delta), best_c), (best_p * (1 + delta), best_c),
                                  (best_p, best_c * (1 - delta)), (best_p, best_c * (1 + delta))):
                    err = residual_for(*candidate)
                    if err + 1e-12 < best_err:
                        best_err = err
                        best_p, best_c = candidate
                        improved = True
                if improved:
                    break
            if not improved:
                break

        # compute closure residual without regularization for reporting
        x, y = _polygon_closure([base_inner, best_p, best_c, best_c, best_c, best_p], angles)
        residual = math.hypot(x, y)
        return {
            "inner": float(base_inner),
            "protrude": float(best_p),
            "outer": float(best_c),
            "residual": float(residual),
        }

    spec = ring_angle_spec(ring)
    inner = spec.inner_angle_deg
    outer = spec.outer_angle_deg
    angles = [inner, outer, outer, outer, outer, inner]

    base_inner = inner_edge_length if inner_edge_length is not None else 1.0

    def closure(params: Tuple[float, float]) -> List[float]:
        protrude, outer_edge = params
        lengths = [base_inner, protrude, outer_edge, outer_edge, outer_edge, protrude]
        x, y = _polygon_closure(lengths, angles)
        penalty = reg_weight * max(0.0, base_inner - protrude)
        penalty2 = reg_weight * max(0.0, base_inner - outer_edge)
        return [x, y, penalty, penalty2]

    lower_bound = min_ratio * base_inner
    result = least_squares(
        closure,
        x0=[base_inner, base_inner],
        bounds=([lower_bound, lower_bound], [float("inf"), float("inf")]),
        max_nfev=max_iter,
    )
    protrude, outer_edge = result.x
    x, y = _polygon_closure(
        [base_inner, protrude, outer_edge, outer_edge, outer_edge, protrude],
        angles,
    )
    return {
        "inner": float(base_inner),
        "protrude": float(protrude),
        "outer": float(outer_edge),
        "residual": float(math.hypot(x, y)),
    }


def solve_ring_hex_outer_length(
    ring: int,
    inner_edge_length: float,
    protrude_length: float,
    max_iter: int = 200,
) -> Dict[str, float]:
    """Solve for outer edge length when inner and protrude lengths are fixed."""
    if inner_edge_length <= 0 or protrude_length <= 0:
        raise ValueError("inner_edge_length and protrude_length must be positive")

    spec = ring_angle_spec(ring)
    angles = [
        spec.inner_angle_deg,
        spec.outer_angle_deg,
        spec.outer_angle_deg,
        spec.outer_angle_deg,
        spec.outer_angle_deg,
        spec.inner_angle_deg,
    ]

    min_ratio = 0.5 if ring == 1 else 0.2
    lower_bound = min_ratio * inner_edge_length

    def closure(outer_edge: float) -> tuple[float, float]:
        lengths = [
            inner_edge_length,
            protrude_length,
            outer_edge,
            outer_edge,
            outer_edge,
            protrude_length,
        ]
        return _polygon_closure(lengths, angles)

    if least_squares is None:
        best_outer = max(lower_bound, inner_edge_length)
        best_res = float("inf")
        low, high = lower_bound, 3.0 * inner_edge_length
        steps = 40
        for i in range(steps):
            outer_edge = low + (high - low) * i / (steps - 1)
            x, y = closure(outer_edge)
            res = math.hypot(x, y)
            if res < best_res:
                best_res = res
                best_outer = outer_edge
        return {
            "inner": float(inner_edge_length),
            "protrude": float(protrude_length),
            "outer": float(best_outer),
            "residual": float(best_res),
        }

    def closure_vec(params: Tuple[float]) -> List[float]:
        outer_edge = params[0]
        x, y = closure(outer_edge)
        return [x, y]

    result = least_squares(
        closure_vec,
        x0=[max(lower_bound, inner_edge_length)],
        bounds=([lower_bound], [float("inf")]),
        max_nfev=max_iter,
    )
    outer_edge = float(result.x[0])
    x, y = closure(outer_edge)
    return {
        "inner": float(inner_edge_length),
        "protrude": float(protrude_length),
        "outer": float(outer_edge),
        "residual": float(math.hypot(x, y)),
    }


def _polygon_closure(lengths: List[float], angles_deg: List[float]) -> Tuple[float, float]:
    angle = 0.0
    x = 0.0
    y = 0.0
    for length, interior in zip(lengths, angles_deg):
        x += length * math.cos(angle)
        y += length * math.sin(angle)
        turn = math.radians(180.0 - interior)
        angle += turn
    return x, y
