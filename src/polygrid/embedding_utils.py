from __future__ import annotations

import math
from typing import Dict, Iterable, List

from .algorithms import build_face_adjacency, ring_faces
from .angle_solver import ring_angle_spec, solve_ring_hex_lengths
from .geometry import _face_vertex_cycle, _ordered_face_vertices
from .grid_utils import (
    _angle_at_vertex,
    _circle_intersections,
    _collect_face_vertices,
    _edge_length,
    _face_center,
    _find_pentagon_face,
    _grid_center,
    _mean_value,
)
from .models import Edge, Face, Vertex
from .polygrid import PolyGrid


def _laplacian_relax(
    vertices: Dict[str, Vertex],
    edges: Iterable[Edge],
    fixed_positions: Dict[str, tuple[float, float]],
    iterations: int = 20,
    alpha: float = 0.5,
) -> Dict[str, Vertex]:
    neighbors: Dict[str, List[str]] = {vid: [] for vid in vertices}
    for edge in edges:
        a, b = edge.vertex_ids
        neighbors[a].append(b)
        neighbors[b].append(a)

    fixed_set = set(fixed_positions.keys())
    current = {vid: Vertex(vid, v.x, v.y) for vid, v in vertices.items()}

    for _ in range(iterations):
        updated: Dict[str, Vertex] = {}
        for vid, vertex in current.items():
            if vid in fixed_set:
                x, y = fixed_positions[vid]
                updated[vid] = Vertex(vid, x, y)
                continue
            nbrs = neighbors.get(vid, [])
            if not nbrs:
                updated[vid] = vertex
                continue
            xs = [current[n].x for n in nbrs if current[n].x is not None]
            ys = [current[n].y for n in nbrs if current[n].y is not None]
            if not xs or not ys:
                updated[vid] = vertex
                continue
            mean_x = sum(xs) / len(xs)
            mean_y = sum(ys) / len(ys)
            new_x = vertex.x + alpha * (mean_x - vertex.x)
            new_y = vertex.y + alpha * (mean_y - vertex.y)
            updated[vid] = Vertex(vid, new_x, new_y)
        current = updated

    return current


def _ring1_symmetry_relax(
    grid: PolyGrid,
    vertices: Dict[str, Vertex],
    fixed_positions: Dict[str, tuple[float, float]],
    iterations: int,
    strength: float,
) -> Dict[str, Vertex]:
    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return vertices

    pent_vertices = set(pent_face.vertex_ids)
    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    ring1_faces = adjacency.get(pent_face.id, [])

    ring1_hexes = [grid.faces[fid] for fid in ring1_faces if fid in grid.faces]
    pairs = []
    for face in ring1_hexes:
        shared = [vid for vid in face.vertex_ids if vid in pent_vertices]
        if len(shared) == 2:
            pairs.append((face.id, tuple(shared)))

    if not pairs:
        return vertices

    fixed_set = set(fixed_positions.keys())
    current = {vid: Vertex(vid, v.x, v.y) for vid, v in vertices.items()}

    pent_center = _face_center(grid, pent_face)
    if pent_center is None:
        return vertices

    for _ in range(iterations):
        next_state = dict(current)
        for face_id, (v1, v2) in pairs:
            if v1 in fixed_set and v2 in fixed_set:
                continue
            face = grid.faces.get(face_id)
            if face is None:
                continue
            hex_center = _face_center(grid, face)
            if hex_center is None:
                continue
            ref_angle = math.atan2(hex_center[1] - pent_center[1], hex_center[0] - pent_center[0])

            p1 = current[v1]
            p2 = current[v2]
            r1 = math.hypot(p1.x - hex_center[0], p1.y - hex_center[1])
            r2 = math.hypot(p2.x - hex_center[0], p2.y - hex_center[1])
            a1 = math.atan2(p1.y - hex_center[1], p1.x - hex_center[0])
            a2 = math.atan2(p2.y - hex_center[1], p2.x - hex_center[0])

            d1 = ((a1 - ref_angle + math.pi) % (2 * math.pi)) - math.pi
            d2 = ((a2 - ref_angle + math.pi) % (2 * math.pi)) - math.pi
            spread = (abs(d1) + abs(d2)) / 2 or 0.01

            target1 = ref_angle + spread
            target2 = ref_angle - spread

            if v1 not in fixed_set:
                tx = hex_center[0] + math.cos(target1) * r1
                ty = hex_center[1] + math.sin(target1) * r1
                next_state[v1] = Vertex(
                    v1,
                    p1.x + (tx - p1.x) * strength,
                    p1.y + (ty - p1.y) * strength,
                )
            if v2 not in fixed_set:
                tx = hex_center[0] + math.cos(target2) * r2
                ty = hex_center[1] + math.sin(target2) * r2
                next_state[v2] = Vertex(
                    v2,
                    p2.x + (tx - p2.x) * strength,
                    p2.y + (ty - p2.y) * strength,
                )

        current = next_state

    return current


def _ring1_symmetry_snap(
    grid: PolyGrid,
    vertices: Dict[str, Vertex],
    fixed_positions: Dict[str, tuple[float, float]],
) -> Dict[str, Vertex]:
    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return vertices

    pent_vertices = set(pent_face.vertex_ids)
    pent_center = _face_center(grid, pent_face)
    if pent_center is None:
        return vertices

    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    ring1_faces = adjacency.get(pent_face.id, [])
    ring1_hexes = [grid.faces[fid] for fid in ring1_faces if fid in grid.faces]

    fixed_set = set(fixed_positions.keys())
    current = dict(vertices)

    for face in ring1_hexes:
        shared = [vid for vid in face.vertex_ids if vid in pent_vertices]
        if len(shared) != 2:
            continue
        v1, v2 = shared
        if v1 in fixed_set and v2 in fixed_set:
            continue
        hex_center = _face_center(grid, face)
        if hex_center is None:
            continue
        line_point, line_dir = _pent_edge_normal_line(grid, pent_face, v1, v2, pent_center)
        perp_dir = (-line_dir[1], line_dir[0])

        p1 = current[v1]
        p2 = current[v2]

        v1x = p1.x - line_point[0]
        v1y = p1.y - line_point[1]
        v2x = p2.x - line_point[0]
        v2y = p2.y - line_point[1]

        proj1 = v1x * line_dir[0] + v1y * line_dir[1]
        proj2 = v2x * line_dir[0] + v2y * line_dir[1]
        perp1 = v1x * perp_dir[0] + v1y * perp_dir[1]
        perp2 = v2x * perp_dir[0] + v2y * perp_dir[1]

        target_proj = (proj1 + proj2) / 2
        target_perp = (abs(perp1) + abs(perp2)) / 2 or 0.01

        if v1 not in fixed_set:
            current[v1] = Vertex(
                v1,
                line_point[0] + line_dir[0] * target_proj + perp_dir[0] * target_perp,
                line_point[1] + line_dir[1] * target_proj + perp_dir[1] * target_perp,
            )
        if v2 not in fixed_set:
            current[v2] = Vertex(
                v2,
                line_point[0] + line_dir[0] * target_proj - perp_dir[0] * target_perp,
                line_point[1] + line_dir[1] * target_proj - perp_dir[1] * target_perp,
            )

    return current


def _ring1_pent_angle_snap(
    grid: PolyGrid,
    vertices: Dict[str, Vertex],
    fixed_positions: Dict[str, tuple[float, float]],
    target_angle_deg: float,
    strength: float,
) -> Dict[str, Vertex]:
    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return vertices

    pent_vertices = set(pent_face.vertex_ids)
    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    ring1_faces = adjacency.get(pent_face.id, [])
    ring1_hexes = [grid.faces[fid] for fid in ring1_faces if fid in grid.faces]

    fixed_set = set(fixed_positions.keys())
    current = dict(vertices)
    targets: Dict[str, list[float]] = {}
    counts: Dict[str, int] = {}

    target_angle = math.radians(target_angle_deg)

    for face in ring1_hexes:
        verts = _ordered_face_vertices(current, face)
        n = len(verts)
        for idx, vid in enumerate(verts):
            if vid not in pent_vertices:
                continue
            prev_vid = verts[(idx - 1) % n]
            next_vid = verts[(idx + 1) % n]

            if prev_vid in pent_vertices and next_vid not in pent_vertices:
                pent_neighbor = prev_vid
                other = next_vid
            elif next_vid in pent_vertices and prev_vid not in pent_vertices:
                pent_neighbor = next_vid
                other = prev_vid
            else:
                continue

            if other in fixed_set:
                continue

            v = current[vid]
            p = current[pent_neighbor]
            o = current[other]

            base_angle = math.atan2(p.y - v.y, p.x - v.x)
            dist = math.hypot(o.x - v.x, o.y - v.y) or 1.0
            ux, uy = p.x - v.x, p.y - v.y
            vx, vy = o.x - v.x, o.y - v.y
            cross = ux * vy - uy * vx
            sign = 1.0 if cross >= 0 else -1.0
            target = base_angle + sign * target_angle
            tx = v.x + math.cos(target) * dist
            ty = v.y + math.sin(target) * dist

            if other not in targets:
                targets[other] = [0.0, 0.0]
                counts[other] = 0
            targets[other][0] += tx
            targets[other][1] += ty
            counts[other] += 1

    for vid, (sx, sy) in targets.items():
        if vid in fixed_set:
            continue
        avg_x = sx / counts[vid]
        avg_y = sy / counts[vid]
        v = current[vid]
        current[vid] = Vertex(
            vid,
            v.x + (avg_x - v.x) * strength,
            v.y + (avg_y - v.y) * strength,
        )

    return current


def _ring_constraints_snap(
    grid: PolyGrid,
    vertices: Dict[str, Vertex],
    fixed_positions: Dict[str, tuple[float, float]],
    max_ring: int,
    iterations: int = 1,
) -> Dict[str, Vertex]:
    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return vertices

    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings = ring_faces(adjacency, pent_face.id, max_depth=max_ring)
    if len(rings) <= 1:
        return vertices

    center = _grid_center(grid)
    fixed_set = set(pent_face.vertex_ids)
    current = {vid: Vertex(vid, v.x, v.y) for vid, v in vertices.items()}

    for _ in range(iterations):
        for ring_idx in range(1, max(rings.keys()) + 1):
            ring_face_ids = rings.get(ring_idx, [])
            if not ring_face_ids:
                continue
            ring_faces_list = [grid.faces[fid] for fid in ring_face_ids if fid in grid.faces]
            ring_vertices = set(_collect_face_vertices(grid, ring_face_ids))
            inner_faces = rings.get(ring_idx - 1, [])
            inner_vertices = set(_collect_face_vertices(grid, inner_faces))
            ring_outer = ring_vertices - inner_vertices
            if not ring_vertices or not inner_vertices:
                continue
            inner_neighbor_counts: Dict[str, int] = {vid: 0 for vid in ring_vertices}
            for edge in grid.edges.values():
                a, b = edge.vertex_ids
                if a in ring_vertices and b in inner_vertices:
                    inner_neighbor_counts[a] += 1
                elif b in ring_vertices and a in inner_vertices:
                    inner_neighbor_counts[b] += 1

            angle_constraints: list[tuple[str, str, str]] = []
            for face in ring_faces_list:
                ordered = _ordered_face_vertices(current, face)
                n = len(ordered)
                for idx, vid in enumerate(ordered):
                    if vid not in ring_outer:
                        continue
                    prev_vid = ordered[(idx - 1) % n]
                    next_vid = ordered[(idx + 1) % n]
                    if (prev_vid in inner_vertices) ^ (next_vid in inner_vertices):
                        inner = prev_vid if prev_vid in inner_vertices else next_vid
                        outer = next_vid if inner == prev_vid else prev_vid
                        angle_constraints.append((vid, inner, outer))

            spec = ring_angle_spec(ring_idx)
            target_outer_angle = math.radians(spec.inner_angle_deg)
            if target_outer_angle > 0:
                targets: Dict[str, list[float]] = {}
                counts: Dict[str, int] = {}
                for v_id, inner_id, outer_id in angle_constraints:
                    if outer_id in fixed_set or outer_id not in ring_outer:
                        continue
                    v = current[v_id]
                    inner = current[inner_id]
                    outer = current[outer_id]
                    base_angle = math.atan2(inner.y - v.y, inner.x - v.x)
                    ux, uy = inner.x - v.x, inner.y - v.y
                    vx, vy = outer.x - v.x, outer.y - v.y
                    cross = ux * vy - uy * vx
                    sign = 1.0 if cross >= 0 else -1.0
                    dist = math.hypot(outer.x - v.x, outer.y - v.y) or 1.0
                    target = base_angle + sign * target_outer_angle
                    tx = v.x + math.cos(target) * dist
                    ty = v.y + math.sin(target) * dist
                    targets.setdefault(outer_id, [0.0, 0.0])
                    counts[outer_id] = counts.get(outer_id, 0) + 1
                    targets[outer_id][0] += tx
                    targets[outer_id][1] += ty
                for vid, (sx, sy) in targets.items():
                    if vid in fixed_set:
                        continue
                    current[vid] = Vertex(vid, sx / counts[vid], sy / counts[vid])

            pointy_constraints: list[tuple[str, str, str]] = []
            for face in ring_faces_list:
                ordered = _ordered_face_vertices(current, face)
                if not ordered:
                    continue
                candidates = [
                    vid
                    for vid in ordered
                    if vid in ring_outer and inner_neighbor_counts.get(vid, 0) == 0
                ]
                if not candidates:
                    continue
                pointy = max(
                    candidates,
                    key=lambda vid: math.hypot(current[vid].x - center[0], current[vid].y - center[1]),
                )
                idx = ordered.index(pointy)
                prev_vid = ordered[(idx - 1) % len(ordered)]
                next_vid = ordered[(idx + 1) % len(ordered)]
                pointy_constraints.append((pointy, prev_vid, next_vid))

            target_pointy_angle = math.radians(spec.outer_angle_deg)
            if target_pointy_angle > 0:
                for pointy, prev_vid, next_vid in pointy_constraints:
                    if pointy in fixed_set:
                        continue
                    prev_v = current[prev_vid]
                    next_v = current[next_vid]
                    dx = next_v.x - prev_v.x
                    dy = next_v.y - prev_v.y
                    d = math.hypot(dx, dy)
                    if d == 0:
                        continue
                    half_angle = target_pointy_angle / 2.0
                    sin_half = math.sin(half_angle)
                    if sin_half == 0:
                        continue
                    radius = d / (2 * sin_half)
                    h_sq = radius * radius - (d / 2) ** 2
                    if h_sq < 0:
                        continue
                    h = math.sqrt(h_sq)
                    mx = (prev_v.x + next_v.x) / 2
                    my = (prev_v.y + next_v.y) / 2
                    ux = -dy / d
                    uy = dx / d
                    candidate1 = (mx + ux * h, my + uy * h)
                    candidate2 = (mx - ux * h, my - uy * h)
                    chosen = max(
                        (candidate1, candidate2),
                        key=lambda pt: math.hypot(pt[0] - center[0], pt[1] - center[1]),
                    )
                    current[pointy] = Vertex(pointy, chosen[0], chosen[1])

            protruding_map: Dict[str, list[str]] = {}
            for edge in grid.edges.values():
                a, b = edge.vertex_ids
                if (a in ring_vertices and b in inner_vertices) or (b in ring_vertices and a in inner_vertices):
                    inner = a if a in inner_vertices else b
                    outer = b if inner == a else a
                    protruding_map.setdefault(outer, []).append(inner)

            lengths_solution = solve_ring_hex_lengths(ring_idx, inner_edge_length=1.0)
            target_protrude = lengths_solution["protrude"]
            if target_protrude > 0:
                for outer, inners in protruding_map.items():
                    if outer in fixed_set:
                        continue
                    vo = current[outer]
                    if len(inners) >= 2:
                        p1 = current[inners[0]]
                        p2 = current[inners[1]]
                        intersections = _circle_intersections(
                            (p1.x, p1.y),
                            (p2.x, p2.y),
                            target_protrude,
                        )
                        if intersections:
                            chosen = max(
                                intersections,
                                key=lambda pt: math.hypot(pt[0] - center[0], pt[1] - center[1]),
                            )
                            current[outer] = Vertex(outer, chosen[0], chosen[1])
                            continue
                    if len(inners) != 1:
                        continue
                    inner = inners[0]
                    vi = current[inner]
                    dx = vo.x - vi.x
                    dy = vo.y - vi.y
                    dist = math.hypot(dx, dy) or 1.0
                    tx = vi.x + dx / dist * target_protrude
                    ty = vi.y + dy / dist * target_protrude
                    current[outer] = Vertex(outer, tx, ty)

    return current


def _ring_protruding_edge_snap(
    grid: PolyGrid,
    vertices: Dict[str, Vertex],
    fixed_positions: Dict[str, tuple[float, float]],
    max_ring: int,
) -> Dict[str, Vertex]:
    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return vertices

    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings = ring_faces(adjacency, pent_face.id, max_depth=max_ring)
    if len(rings) <= 1:
        return vertices

    center = _grid_center(grid)
    fixed_set = set(pent_face.vertex_ids)
    current = {vid: Vertex(vid, v.x, v.y) for vid, v in vertices.items()}

    for ring_idx in range(1, max(rings.keys()) + 1):
        ring_face_ids = rings.get(ring_idx, [])
        if not ring_face_ids:
            continue
        ring_vertices = set(_collect_face_vertices(grid, ring_face_ids))
        inner_vertices = set(_collect_face_vertices(grid, rings.get(ring_idx - 1, [])))
        ring_outer = ring_vertices - inner_vertices
        if not ring_vertices or not inner_vertices:
            continue

        protruding_map: Dict[str, list[str]] = {}
        for edge in grid.edges.values():
            a, b = edge.vertex_ids
            if (a in ring_vertices and b in inner_vertices) or (b in ring_vertices and a in inner_vertices):
                inner = a if a in inner_vertices else b
                outer = b if inner == a else a
                if outer not in ring_outer:
                    continue
                protruding_map.setdefault(outer, []).append(inner)

        lengths = [
            _edge_length(current, outer, inners[0])
            for outer, inners in protruding_map.items()
            if len(inners) == 1 and outer not in fixed_set
        ]
        target_len = _mean_value(lengths)
        if target_len <= 0:
            continue

        for outer, inners in protruding_map.items():
            if outer in fixed_set or len(inners) != 1:
                continue
            inner = inners[0]
            vi = current[inner]
            vo = current[outer]
            dx = vo.x - vi.x
            dy = vo.y - vi.y
            dist = math.hypot(dx, dy) or 1.0
            tx = vi.x + dx / dist * target_len
            ty = vi.y + dy / dist * target_len
            current[outer] = Vertex(outer, tx, ty)

    return current


def _ring_pointy_edge_snap(
    grid: PolyGrid,
    vertices: Dict[str, Vertex],
    fixed_positions: Dict[str, tuple[float, float]],
    max_ring: int,
) -> Dict[str, Vertex]:
    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return vertices

    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings = ring_faces(adjacency, pent_face.id, max_depth=max_ring)
    if len(rings) <= 1:
        return vertices

    center = _grid_center(grid)
    fixed_set = set(pent_face.vertex_ids)
    current = {vid: Vertex(vid, v.x, v.y) for vid, v in vertices.items()}

    for ring_idx in range(1, max(rings.keys()) + 1):
        ring_face_ids = rings.get(ring_idx, [])
        if not ring_face_ids:
            continue
        ring_faces_list = [grid.faces[fid] for fid in ring_face_ids if fid in grid.faces]
        ring_vertices = set(_collect_face_vertices(grid, ring_face_ids))
        inner_vertices = set(_collect_face_vertices(grid, rings.get(ring_idx - 1, [])))
        ring_outer = ring_vertices - inner_vertices
        if not ring_outer:
            continue
        inner_neighbor_counts: Dict[str, int] = {vid: 0 for vid in ring_vertices}
        for edge in grid.edges.values():
            a, b = edge.vertex_ids
            if a in ring_vertices and b in inner_vertices:
                inner_neighbor_counts[a] += 1
            elif b in ring_vertices and a in inner_vertices:
                inner_neighbor_counts[b] += 1

        pointy_constraints: list[tuple[str, str, str]] = []
        pointy_lengths: list[float] = []
        min_lengths: list[float] = []
        for face in ring_faces_list:
            ordered = _ordered_face_vertices(current, face)
            if not ordered:
                continue
            candidates = [
                vid
                for vid in ordered
                if vid in ring_outer and inner_neighbor_counts.get(vid, 0) == 0
            ]
            if not candidates:
                continue
            pointy = max(
                candidates,
                key=lambda vid: math.hypot(current[vid].x - center[0], current[vid].y - center[1]),
            )
            idx = ordered.index(pointy)
            prev_vid = ordered[(idx - 1) % len(ordered)]
            next_vid = ordered[(idx + 1) % len(ordered)]
            pointy_constraints.append((pointy, prev_vid, next_vid))
            pointy_lengths.append(_edge_length(current, pointy, prev_vid))
            pointy_lengths.append(_edge_length(current, pointy, next_vid))
            prev_v = current[prev_vid]
            next_v = current[next_vid]
            min_lengths.append(math.hypot(next_v.x - prev_v.x, next_v.y - prev_v.y) / 2)

        lengths_solution = solve_ring_hex_lengths(ring_idx, inner_edge_length=1.0)
        target_len = max(lengths_solution["outer"], max(min_lengths) if min_lengths else 0.0)
        if target_len <= 0:
            continue

        for pointy, prev_vid, next_vid in pointy_constraints:
            if pointy in fixed_set:
                continue
            prev_v = current[prev_vid]
            next_v = current[next_vid]
            intersections = _circle_intersections(
                (prev_v.x, prev_v.y),
                (next_v.x, next_v.y),
                target_len,
            )
            if intersections:
                chosen = max(
                    intersections,
                    key=lambda pt: math.hypot(pt[0] - center[0], pt[1] - center[1]),
                )
                current[pointy] = Vertex(pointy, chosen[0], chosen[1])
                continue
            v = current[pointy]
            dx1 = v.x - prev_v.x
            dy1 = v.y - prev_v.y
            dist1 = math.hypot(dx1, dy1) or 1.0
            tx1 = prev_v.x + dx1 / dist1 * target_len
            ty1 = prev_v.y + dy1 / dist1 * target_len
            dx2 = v.x - next_v.x
            dy2 = v.y - next_v.y
            dist2 = math.hypot(dx2, dy2) or 1.0
            tx2 = next_v.x + dx2 / dist2 * target_len
            ty2 = next_v.y + dy2 / dist2 * target_len
            current[pointy] = Vertex(pointy, (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0)

    return current


def _fivefold_symmetry_snap(
    grid: PolyGrid,
    positions: Dict[str, Vertex],
    ring_vertices: Iterable[str],
) -> Dict[str, Vertex]:
    """Snap ring vertices to fivefold symmetric angles around the center."""
    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return positions

    ordered_pent = _face_vertex_cycle(pent_face, grid.edges.values())
    if len(ordered_pent) != 5:
        ordered_pent = _ordered_face_vertices(positions, pent_face)
    if len(ordered_pent) != 5:
        return positions

    center = (
        sum(positions[vid].x for vid in ordered_pent) / 5,
        sum(positions[vid].y for vid in ordered_pent) / 5,
    )
    base_angle = math.atan2(
        positions[ordered_pent[0]].y - center[1],
        positions[ordered_pent[0]].x - center[0],
    )
    step = 2 * math.pi / 5

    samples: list[tuple[str, float, float]] = []
    for vid in ring_vertices:
        v = positions.get(vid)
        if v is None or v.x is None or v.y is None:
            continue
        angle = math.atan2(v.y - center[1], v.x - center[0])
        rel = (angle - base_angle) % (2 * math.pi)
        radius = math.hypot(v.x - center[0], v.y - center[1])
        samples.append((vid, rel, radius))

    if not samples or len(samples) % 5 != 0:
        return positions

    samples.sort(key=lambda item: item[1])
    per_sector = len(samples) // 5
    template = samples[:per_sector]

    updated = dict(positions)
    for idx in range(5):
        sector = samples[idx * per_sector : (idx + 1) * per_sector]
        for j, (vid, _, _) in enumerate(sector):
            rel_angle = template[j][1]
            radius = template[j][2]
            angle = base_angle + idx * step + rel_angle
            updated[vid] = Vertex(
                vid,
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
            )

    return updated


# Local helpers

def _pent_edge_normal_line(
    grid: PolyGrid,
    pent_face: Face,
    v1: str,
    v2: str,
    pent_center: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    p1 = grid.vertices[v1]
    p2 = grid.vertices[v2]
    mx = (p1.x + p2.x) / 2
    my = (p1.y + p2.y) / 2
    dx = mx - pent_center[0]
    dy = my - pent_center[1]
    norm = math.hypot(dx, dy) or 1.0
    return (mx, my), (dx / norm, dy / norm)
