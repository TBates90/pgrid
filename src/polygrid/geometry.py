from __future__ import annotations

import math
from typing import Dict, Iterable, List

from .models import Edge, Face, Vertex


def _ordered_face_vertices(vertices: Dict[str, Vertex], face: Face) -> List[str]:
    coords = [vertices[vid] for vid in face.vertex_ids]
    if not coords or not all(v.has_position() for v in coords):
        return list(face.vertex_ids)
    cx = sum(v.x for v in coords if v.x is not None) / len(coords)
    cy = sum(v.y for v in coords if v.y is not None) / len(coords)

    def angle(vid: str) -> float:
        v = vertices[vid]
        return math.atan2(v.y - cy, v.x - cx)

    return sorted(face.vertex_ids, key=angle)


def _face_vertex_cycle(face: Face, edges: Iterable[Edge]) -> List[str]:
    neighbors: Dict[str, List[str]] = {}
    for edge in edges:
        if face.id not in edge.face_ids:
            continue
        a, b = edge.vertex_ids
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)

    if not neighbors:
        return list(face.vertex_ids)

    if any(len(nbrs) != 2 for nbrs in neighbors.values()):
        return list(face.vertex_ids)

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


def _interior_angle(vertices: Dict[str, Vertex], a: str, b: str, c: str) -> float:
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
    angle = math.acos(dot)
    return max(angle, math.pi - angle)


def _face_signed_area(
    positions: Dict[str, Vertex],
    face: Face,
    edges: Iterable[Edge],
) -> float | None:
    ordered = _face_vertex_cycle(face, edges)
    if len(ordered) != len(face.vertex_ids):
        ordered = _ordered_face_vertices(positions, face)
    coords = [positions[vid] for vid in ordered]
    if not coords or not all(v.has_position() for v in coords):
        return None
    area = 0.0
    for i in range(len(coords)):
        x1, y1 = coords[i].x, coords[i].y
        x2, y2 = coords[(i + 1) % len(coords)].x, coords[(i + 1) % len(coords)].y
        area += x1 * y2 - x2 * y1
    return area / 2.0
