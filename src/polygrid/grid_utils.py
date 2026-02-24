from __future__ import annotations

import math
from typing import Dict, Iterable, List

from .models import Edge, Face, Vertex
from .polygrid import PolyGrid


def _face_center(grid: PolyGrid, face: Face) -> tuple[float, float] | None:
    coords = [grid.vertices[vid] for vid in face.vertex_ids]
    if not coords or not all(v.has_position() for v in coords):
        return None
    cx = sum(v.x for v in coords if v.x is not None) / len(coords)
    cy = sum(v.y for v in coords if v.y is not None) / len(coords)
    return (cx, cy)


def _boundary_vertex_ids(grid: PolyGrid) -> List[str]:
    degree = {vid: 0 for vid in grid.vertices}
    for edge in grid.edges.values():
        degree[edge.vertex_ids[0]] += 1
        degree[edge.vertex_ids[1]] += 1
    boundary = [vid for vid, deg in degree.items() if deg < 3]
    if not boundary:
        return []

    cycle = _boundary_vertex_cycle(grid)
    if cycle:
        return cycle

    cx = sum(grid.vertices[vid].x for vid in boundary if grid.vertices[vid].x is not None) / len(boundary)
    cy = sum(grid.vertices[vid].y for vid in boundary if grid.vertices[vid].y is not None) / len(boundary)
    return sorted(boundary, key=lambda vid: math.atan2(grid.vertices[vid].y - cy, grid.vertices[vid].x - cx))


def _boundary_vertex_cycle(grid: PolyGrid) -> List[str]:
    boundary_edges = [edge for edge in grid.edges.values() if len(edge.face_ids) < 2]
    if not boundary_edges:
        return []

    neighbors: Dict[str, List[str]] = {}
    for edge in boundary_edges:
        a, b = edge.vertex_ids
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)

    if any(len(nbrs) != 2 for nbrs in neighbors.values()):
        return []

    start = sorted(neighbors.keys())[0]
    cycle = [start]
    prev = None
    current = start
    while True:
        nbrs = neighbors.get(current, [])
        if not nbrs:
            break
        nxt = nbrs[0] if nbrs[0] != prev else (nbrs[1] if len(nbrs) > 1 else None)
        if nxt is None or nxt == start:
            break
        if nxt in cycle:
            break
        cycle.append(nxt)
        prev, current = current, nxt
        if len(cycle) > len(neighbors) + 1:
            break

    return cycle


def _grid_center(grid: PolyGrid) -> tuple[float, float]:
    xs = [v.x for v in grid.vertices.values() if v.x is not None]
    ys = [v.y for v in grid.vertices.values() if v.y is not None]
    return (sum(xs) / len(xs), sum(ys) / len(ys)) if xs and ys else (0.0, 0.0)


def _find_pentagon_face(grid: PolyGrid) -> Face | None:
    for face in grid.faces.values():
        if face.face_type == "pent" or len(face.vertex_ids) == 5:
            return face
    return None


def _collect_face_vertices(grid: PolyGrid, face_ids: Iterable[str]) -> List[str]:
    vertex_ids = []
    for fid in face_ids:
        face = grid.faces.get(fid)
        if not face:
            continue
        vertex_ids.extend(face.vertex_ids)
    return list(dict.fromkeys(vertex_ids))


def _edge_length(vertices: Dict[str, Vertex], a: str, b: str) -> float:
    va = vertices[a]
    vb = vertices[b]
    return math.hypot(vb.x - va.x, vb.y - va.y)


def _angle_at_vertex(vertices: Dict[str, Vertex], a: str, b: str, c: str) -> float:
    va = vertices[a]
    vb = vertices[b]
    vc = vertices[c]
    v1x = va.x - vb.x
    v1y = va.y - vb.y
    v2x = vc.x - vb.x
    v2y = vc.y - vb.y
    denom = (math.hypot(v1x, v1y) * math.hypot(v2x, v2y)) or 1.0
    dot = (v1x * v2x + v1y * v2y) / denom
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)


def _mean_value(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _circle_intersections(
    c1: tuple[float, float],
    c2: tuple[float, float],
    r: float,
) -> list[tuple[float, float]]:
    x1, y1 = c1
    x2, y2 = c2
    dx = x2 - x1
    dy = y2 - y1
    d = math.hypot(dx, dy)
    if d == 0 or d > 2 * r:
        return []
    a = d / 2
    h_sq = r * r - a * a
    if h_sq < 0:
        return []
    h = math.sqrt(h_sq)
    mx = x1 + dx / 2
    my = y1 + dy / 2
    rx = -dy / d
    ry = dx / d
    return [(mx + rx * h, my + ry * h), (mx - rx * h, my - ry * h)]
