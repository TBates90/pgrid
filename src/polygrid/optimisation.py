from __future__ import annotations

import math
from typing import Dict, Iterable

from .algorithms import build_face_adjacency
from .angle_solver import ring_angle_spec
from .geometry import _face_signed_area, _face_vertex_cycle, _interior_angle, _ordered_face_vertices
from .grid_utils import _boundary_vertex_ids, _collect_face_vertices
from .models import Edge, Face, Vertex
from .polygrid import PolyGrid

try:  # pragma: no cover - optional dependency
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover
    least_squares = None


def optimise_positions_to_edge_targets(
    grid: PolyGrid,
    initial_positions: Dict[str, Vertex],
    edge_ratio_targets: Dict[str, float],
    fixed_positions: Dict[str, tuple[float, float]],
    iterations: int = 40,
) -> Dict[str, Vertex]:
    """Scale ratio targets to absolute lengths using current geometry, then relax."""
    current_lengths = {}
    ratios = []
    lengths = []
    for edge in grid.edges.values():
        a, b = edge.vertex_ids
        va = initial_positions[a]
        vb = initial_positions[b]
        dist = math.hypot(vb.x - va.x, vb.y - va.y)
        if edge.id in edge_ratio_targets:
            ratios.append(edge_ratio_targets[edge.id])
            lengths.append(dist)
            current_lengths[edge.id] = dist

    if not ratios:
        return initial_positions

    mean_ratio = sum(ratios) / len(ratios)
    mean_length = sum(lengths) / len(lengths)
    scale = mean_length / mean_ratio if mean_ratio > 0 else 1.0

    edge_abs_targets: Dict[str, float] = {}
    for eid, ratio in edge_ratio_targets.items():
        edge_abs_targets[eid] = ratio * scale

    return _edge_length_relax_to_targets(
        initial_positions, grid.edges.values(), edge_abs_targets, fixed_positions, iterations=iterations
    )


def _edge_length_relax_to_targets(
    vertices: Dict[str, Vertex],
    edges: Iterable[Edge],
    edge_targets: Dict[str, float],
    fixed_positions: Dict[str, tuple[float, float]],
    iterations: int,
    strength: float = 0.1,
) -> Dict[str, Vertex]:
    fixed_set = set(fixed_positions.keys())
    current = {vid: Vertex(vid, v.x, v.y) for vid, v in vertices.items()}

    for _ in range(iterations):
        deltas: Dict[str, list[float]] = {vid: [0.0, 0.0, 0.0] for vid in current}
        for edge in edges:
            target = edge_targets.get(edge.id)
            if target is None or target <= 0:
                continue
            a, b = edge.vertex_ids
            va = current[a]
            vb = current[b]
            dx = vb.x - va.x
            dy = vb.y - va.y
            dist = math.hypot(dx, dy) or 1.0
            diff = (dist - target) / dist
            ux = dx * diff
            uy = dy * diff
            deltas[a][0] += ux
            deltas[a][1] += uy
            deltas[a][2] += 1.0
            deltas[b][0] -= ux
            deltas[b][1] -= uy
            deltas[b][2] += 1.0

        next_state: Dict[str, Vertex] = {}
        for vid, vertex in current.items():
            if vid in fixed_set:
                x, y = fixed_positions[vid]
                next_state[vid] = Vertex(vid, x, y)
                continue
            total_x, total_y, count = deltas[vid]
            if count == 0:
                next_state[vid] = vertex
                continue
            step_x = strength * (total_x / count)
            step_y = strength * (total_y / count)
            next_state[vid] = Vertex(vid, vertex.x - step_x, vertex.y - step_y)

        current = next_state

    return current


def _global_optimize_positions(
    grid: PolyGrid,
    initial_positions: Dict[str, Vertex],
    fixed_positions: Dict[str, tuple[float, float]],
    edge_ratio_targets: Dict[str, float],
    rings_map: dict,
    iterations: int = 80,
    angle_weight: float = 0.6,
    area_weight: float = 5.0,
    area_epsilon: float = 1e-3,
    boundary_weight: float = 0.2,
) -> Dict[str, Vertex]:
    """Global least-squares optimisation for edge and angle targets with inversion barrier."""
    if least_squares is None:
        return optimise_positions_to_edge_targets(
            grid, initial_positions, edge_ratio_targets, fixed_positions, iterations=iterations
        )

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - optional dependency
        return optimise_positions_to_edge_targets(
            grid, initial_positions, edge_ratio_targets, fixed_positions, iterations=iterations
        )

    edge_abs_targets = _scale_edge_targets(grid, initial_positions, edge_ratio_targets)
    angle_targets = _compute_angle_targets(grid, rings_map)

    fixed_set = set(fixed_positions.keys())
    movable = [vid for vid in grid.vertices.keys() if vid not in fixed_set]
    boundary_ids = set(_boundary_vertex_ids(grid))
    if 1 in rings_map:
        angle_weight = max(angle_weight, 0.8)
    if boundary_ids:
        center = _positions_center(initial_positions)
        boundary_radius = _mean_radius_from_positions(initial_positions, boundary_ids, center)
    else:
        center = (0.0, 0.0)
        boundary_radius = 0.0

    x0 = []
    for vid in movable:
        v = initial_positions[vid]
        x0.extend([v.x, v.y])
    x0 = np.array(x0, dtype=float)

    def unpack_positions(x: np.ndarray) -> Dict[str, Vertex]:
        pos = {vid: Vertex(vid, v.x, v.y) for vid, v in initial_positions.items()}
        for i, vid in enumerate(movable):
            pos[vid] = Vertex(vid, float(x[2 * i]), float(x[2 * i + 1]))
        for vid, (fx, fy) in fixed_positions.items():
            pos[vid] = Vertex(vid, fx, fy)
        return pos

    def residuals(x: np.ndarray) -> np.ndarray:
        positions = unpack_positions(x)
        res: list[float] = []

        for edge_id, target in edge_abs_targets.items():
            edge = grid.edges.get(edge_id)
            if edge is None or target <= 0:
                res.append(0.0)
                continue
            a, b = edge.vertex_ids
            va = positions[a]
            vb = positions[b]
            dist = math.hypot(vb.x - va.x, vb.y - va.y)
            res.append((dist - target) / target)

        for (face_id, vid), target_angle in angle_targets.items():
            face = grid.faces.get(face_id)
            if face is None:
                res.append(0.0)
                continue
            ordered = _face_vertex_cycle(face, grid.edges.values())
            if len(ordered) != len(face.vertex_ids):
                ordered = _ordered_face_vertices(positions, face)
            if vid not in ordered:
                res.append(0.0)
                continue
            idx = ordered.index(vid)
            prev_vid = ordered[(idx - 1) % len(ordered)]
            next_vid = ordered[(idx + 1) % len(ordered)]
            angle = _interior_angle(positions, prev_vid, vid, next_vid)
            res.append((angle - target_angle) * angle_weight)

        for face in grid.faces.values():
            area = _face_signed_area(positions, face, grid.edges.values())
            if area is None:
                res.append(0.0)
                continue
            barrier = max(0.0, area_epsilon - area)
            res.append(barrier * area_weight)

        if boundary_ids and boundary_radius > 0:
            for vid in boundary_ids:
                if vid in fixed_set:
                    continue
                v = positions[vid]
                dist = math.hypot(v.x - center[0], v.y - center[1])
                res.append((dist - boundary_radius) * boundary_weight)

        return np.array(res, dtype=float)

    result = least_squares(residuals, x0=x0, max_nfev=iterations)
    return unpack_positions(result.x)


def optimize_outer_rings(
    grid: PolyGrid,
    initial_positions: Dict[str, Vertex],
    edge_targets: Dict[str, float],
    rings_map: dict,
    fixed_base: Dict[str, tuple[float, float]],
    start_ring: int = 2,
    iterations: int = 80,
    angle_weight: float = 1.0,
    area_weight: float = 6.0,
    area_epsilon: float = 1e-3,
    anchor_weight: float = 0.05,
    edge_weight: float = 4.0,
) -> Dict[str, Vertex]:
    """Optimize each outer ring with inner rings fixed to their current positions."""
    if least_squares is None:
        return initial_positions

    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return initial_positions

    from .diagnostics import has_edge_crossings

    current = {vid: Vertex(vid, v.x, v.y) for vid, v in initial_positions.items()}
    angle_targets = _compute_angle_targets(grid, rings_map)

    max_ring = max(rings_map.keys()) if rings_map else 0
    for ring_idx in range(start_ring, max_ring + 1):
        ring_faces = rings_map.get(ring_idx, [])
        if not ring_faces:
            continue
        ring_face_set = set(ring_faces)
        inner_vertices = set(
            vid
            for depth in range(0, ring_idx)
            for fid in rings_map.get(depth, [])
            for vid in grid.faces[fid].vertex_ids
        )
        fixed_positions = {**fixed_base}
        for vid in inner_vertices:
            v = current.get(vid)
            if v and v.has_position():
                fixed_positions[vid] = (v.x, v.y)

        movable = [
            vid
            for vid in grid.vertices.keys()
            if vid not in fixed_positions and vid in _collect_face_vertices(grid, ring_faces)
        ]
        if not movable:
            continue

        ring_edges = [
            edge for edge in grid.edges.values() if any(fid in ring_face_set for fid in edge.face_ids)
        ]
        ring_angle_targets = {
            (fid, vid): target
            for (fid, vid), target in angle_targets.items()
            if fid in ring_face_set
        }

        x0 = []
        for vid in movable:
            v = current[vid]
            x0.extend([v.x, v.y])
        x0 = np.array(x0, dtype=float)

        def unpack_positions(x: np.ndarray) -> Dict[str, Vertex]:
            pos = {vid: Vertex(vid, v.x, v.y) for vid, v in current.items()}
            for i, vid in enumerate(movable):
                pos[vid] = Vertex(vid, float(x[2 * i]), float(x[2 * i + 1]))
            for vid, (fx, fy) in fixed_positions.items():
                pos[vid] = Vertex(vid, fx, fy)
            return pos

        def residuals(x: np.ndarray) -> np.ndarray:
            positions = unpack_positions(x)
            res: list[float] = []

            for edge in ring_edges:
                target = edge_targets.get(edge.id)
                if target is None or target <= 0:
                    res.append(0.0)
                    continue
                a, b = edge.vertex_ids
                va = positions[a]
                vb = positions[b]
                dist = math.hypot(vb.x - va.x, vb.y - va.y)
                res.append(((dist - target) / target) * edge_weight)

            for (fid, vid), target_angle in ring_angle_targets.items():
                face = grid.faces.get(fid)
                if face is None:
                    res.append(0.0)
                    continue
                ordered = _face_vertex_cycle(face, grid.edges.values())
                if len(ordered) != len(face.vertex_ids):
                    ordered = _ordered_face_vertices(positions, face)
                if vid not in ordered:
                    res.append(0.0)
                    continue
                idx = ordered.index(vid)
                prev_vid = ordered[(idx - 1) % len(ordered)]
                next_vid = ordered[(idx + 1) % len(ordered)]
                angle = _interior_angle(positions, prev_vid, vid, next_vid)
                res.append((angle - target_angle) * angle_weight)

            for fid in ring_faces:
                face = grid.faces.get(fid)
                if face is None:
                    continue
                area = _face_signed_area(positions, face, grid.edges.values())
                if area is None:
                    res.append(0.0)
                    continue
                barrier = max(0.0, area_epsilon - area)
                res.append(barrier * area_weight)

            for i, vid in enumerate(movable):
                v0 = current[vid]
                res.append((positions[vid].x - v0.x) * anchor_weight)
                res.append((positions[vid].y - v0.y) * anchor_weight)

            return np.array(res, dtype=float)

        result = least_squares(residuals, x0=x0, max_nfev=iterations)
        candidate = unpack_positions(result.x)
        candidate_grid = PolyGrid(candidate.values(), grid.edges.values(), grid.faces.values())
        if has_edge_crossings(candidate_grid):
            blended = None
            for alpha in (0.5, 0.25, 0.125, 0.0625):
                temp = {vid: Vertex(vid, v.x, v.y) for vid, v in current.items()}
                for vid in movable:
                    v0 = current[vid]
                    v1 = candidate[vid]
                    temp[vid] = Vertex(
                        vid,
                        v0.x + (v1.x - v0.x) * alpha,
                        v0.y + (v1.y - v0.y) * alpha,
                    )
                temp_grid = PolyGrid(temp.values(), grid.edges.values(), grid.faces.values())
                if not has_edge_crossings(temp_grid):
                    blended = temp
                    break
            if blended is None:
                continue
            current = blended
            continue
        current = candidate

    return current


def optimize_outer_rings_constrained(
    grid: PolyGrid,
    initial_positions: Dict[str, Vertex],
    edge_targets: Dict[str, float],
    rings_map: dict,
    fixed_base: Dict[str, tuple[float, float]],
    start_ring: int = 2,
    iterations: int = 200,
    angle_weight: float = 1.0,
    anchor_weight: float = 0.05,
    area_epsilon: float = 1e-3,
) -> Dict[str, Vertex]:
    """Constrained optimizer that enforces edge lengths exactly per ring."""
    try:
        import numpy as np
        from scipy.optimize import NonlinearConstraint, minimize
    except ImportError:  # pragma: no cover
        return initial_positions

    current = {vid: Vertex(vid, v.x, v.y) for vid, v in initial_positions.items()}
    angle_targets = _compute_angle_targets(grid, rings_map)
    max_ring = max(rings_map.keys()) if rings_map else 0

    for ring_idx in range(start_ring, max_ring + 1):
        ring_faces = rings_map.get(ring_idx, [])
        if not ring_faces:
            continue
        ring_face_set = set(ring_faces)
        ring_vertices = set(_collect_face_vertices(grid, ring_faces))
        inner_vertices = set(
            vid
            for depth in range(0, ring_idx)
            for fid in rings_map.get(depth, [])
            for vid in grid.faces[fid].vertex_ids
        )
        fixed_positions = {**fixed_base}
        for vid in inner_vertices:
            v = current.get(vid)
            if v and v.has_position():
                fixed_positions[vid] = (v.x, v.y)

        inner_radius = 0.0
        if inner_vertices:
            cx, cy = _positions_center({vid: current[vid] for vid in inner_vertices if vid in current})
            inner_radius = max(
                math.hypot(current[vid].x - cx, current[vid].y - cy)
                for vid in inner_vertices
                if current[vid].has_position()
            )
        else:
            cx, cy = _positions_center(current)

        movable = [vid for vid in ring_vertices if vid not in fixed_positions]
        if not movable:
            continue

        ring_edges = [
            edge
            for edge in grid.edges.values()
            if edge.id in edge_targets and any(fid in ring_face_set for fid in edge.face_ids)
        ]
        if not ring_edges:
            continue

        x0 = np.array([coord for vid in movable for coord in (current[vid].x, current[vid].y)])

        def unpack_positions(x: np.ndarray) -> Dict[str, Vertex]:
            pos = {vid: Vertex(vid, v.x, v.y) for vid, v in current.items()}
            for i, vid in enumerate(movable):
                pos[vid] = Vertex(vid, float(x[2 * i]), float(x[2 * i + 1]))
            for vid, (fx, fy) in fixed_positions.items():
                pos[vid] = Vertex(vid, fx, fy)
            return pos

        def objective(x: np.ndarray) -> float:
            positions = unpack_positions(x)
            total = 0.0
            for (fid, vid), target_angle in angle_targets.items():
                if fid not in ring_face_set:
                    continue
                face = grid.faces.get(fid)
                if face is None:
                    continue
                ordered = _face_vertex_cycle(face, grid.edges.values())
                if len(ordered) != len(face.vertex_ids):
                    ordered = _ordered_face_vertices(positions, face)
                if vid not in ordered:
                    continue
                idx = ordered.index(vid)
                prev_vid = ordered[(idx - 1) % len(ordered)]
                next_vid = ordered[(idx + 1) % len(ordered)]
                angle = _interior_angle(positions, prev_vid, vid, next_vid)
                total += ((angle - target_angle) * angle_weight) ** 2

            for vid in movable:
                v0 = current[vid]
                v1 = positions[vid]
                total += ((v1.x - v0.x) * anchor_weight) ** 2
                total += ((v1.y - v0.y) * anchor_weight) ** 2

            return total

        def edge_constraints(x: np.ndarray) -> np.ndarray:
            positions = unpack_positions(x)
            values = []
            for edge in ring_edges:
                target = edge_targets[edge.id]
                a, b = edge.vertex_ids
                va = positions[a]
                vb = positions[b]
                values.append(math.hypot(vb.x - va.x, vb.y - va.y) - target)
            return np.array(values, dtype=float)

        def area_constraints(x: np.ndarray) -> np.ndarray:
            positions = unpack_positions(x)
            values = []
            for fid in ring_faces:
                face = grid.faces.get(fid)
                if face is None:
                    values.append(0.0)
                    continue
                area = _face_signed_area(positions, face, grid.edges.values())
                values.append(area if area is not None else 0.0)
            return np.array(values, dtype=float)

        def radial_constraints(x: np.ndarray) -> np.ndarray:
            positions = unpack_positions(x)
            values = []
            for vid in movable:
                v = positions[vid]
                values.append(math.hypot(v.x - cx, v.y - cy) - (inner_radius + 1e-3))
            return np.array(values, dtype=float)

        nlc = NonlinearConstraint(edge_constraints, 0.0, 0.0)
        area_c = NonlinearConstraint(area_constraints, area_epsilon, float("inf"))
        radial_c = NonlinearConstraint(radial_constraints, 0.0, float("inf"))
        result = minimize(
            objective,
            x0,
            method="trust-constr",
            constraints=[nlc, area_c, radial_c],
            options={"maxiter": iterations},
        )
        current = unpack_positions(result.x)

    return current


def _scale_edge_targets(
    grid: PolyGrid,
    positions: Dict[str, Vertex],
    edge_ratio_targets: Dict[str, float],
) -> Dict[str, float]:
    ratios = []
    lengths = []
    for edge in grid.edges.values():
        if edge.id not in edge_ratio_targets:
            continue
        a, b = edge.vertex_ids
        va = positions[a]
        vb = positions[b]
        dist = math.hypot(vb.x - va.x, vb.y - va.y)
        ratios.append(edge_ratio_targets[edge.id])
        lengths.append(dist)

    if not ratios:
        return {}
    mean_ratio = sum(ratios) / len(ratios)
    mean_length = sum(lengths) / len(lengths)
    scale = mean_length / mean_ratio if mean_ratio > 0 else 1.0
    return {eid: ratio * scale for eid, ratio in edge_ratio_targets.items()}


def _compute_angle_targets(grid: PolyGrid, rings_map: dict) -> Dict[tuple[str, str], float]:
    targets: Dict[tuple[str, str], float] = {}
    for ring_idx, face_ids in rings_map.items():
        if ring_idx == 0:
            continue
        spec = ring_angle_spec(ring_idx)
        inner_angle = math.radians(spec.inner_angle_deg)
        if ring_idx == 1:
            outer_angle = math.radians(120.0)
        else:
            outer_angle = math.radians(spec.outer_angle_deg)
        inner_vertices = set(
            vid for fid in rings_map.get(ring_idx - 1, []) for vid in grid.faces[fid].vertex_ids
        )
        for fid in face_ids:
            face = grid.faces.get(fid)
            if face is None or len(face.vertex_ids) != 6:
                continue
            for vid in face.vertex_ids:
                target = inner_angle if vid in inner_vertices else outer_angle
                targets[(fid, vid)] = target

    return targets


def _positions_center(positions: Dict[str, Vertex]) -> tuple[float, float]:
    xs = [v.x for v in positions.values() if v.x is not None]
    ys = [v.y for v in positions.values() if v.y is not None]
    return (sum(xs) / len(xs), sum(ys) / len(ys)) if xs and ys else (0.0, 0.0)


def _mean_radius_from_positions(
    positions: Dict[str, Vertex],
    vertex_ids: Iterable[str],
    center: tuple[float, float],
) -> float:
    radii = []
    for vid in vertex_ids:
        v = positions.get(vid)
        if v is None or v.x is None or v.y is None:
            continue
        radii.append(math.hypot(v.x - center[0], v.y - center[1]))
    return sum(radii) / len(radii) if radii else 0.0
