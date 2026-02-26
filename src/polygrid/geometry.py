"""Geometry helper functions used across the package."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List

from .models import Edge, Face, Vertex


def ordered_face_vertices(vertices: Dict[str, Vertex], face: Face) -> List[str]:
    """Return face vertex ids sorted by angle around face centroid."""
    coords = [vertices[vid] for vid in face.vertex_ids]
    if not coords or not all(v.has_position() for v in coords):
        return list(face.vertex_ids)
    cx = sum(v.x for v in coords if v.x is not None) / len(coords)
    cy = sum(v.y for v in coords if v.y is not None) / len(coords)

    def angle(vid: str) -> float:
        v = vertices[vid]
        return math.atan2(v.y - cy, v.x - cx)

    return sorted(face.vertex_ids, key=angle)


def face_vertex_cycle(face: Face, edges: Iterable[Edge]) -> List[str]:
    """Walk edges of face to produce an ordered vertex cycle."""
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


def interior_angle(vertices: Dict[str, Vertex], a: str, b: str, c: str) -> float:
    """Interior angle at vertex *b* in the path a→b→c, in radians.

    Always returns the larger of the two possible angles (i.e. the
    interior angle of the polygon, not the reflex angle).
    """
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


def face_signed_area(
    positions: Dict[str, Vertex],
    face: Face,
    edges: Iterable[Edge] | None = None,
) -> float | None:
    """Signed area of *face* via the shoelace formula.

    Returns a positive value if vertices are wound counter-clockwise,
    negative if clockwise, or ``None`` if any vertex lacks a position.
    """
    ordered = list(face.vertex_ids)
    coords = [positions.get(vid) for vid in ordered]
    if not coords or not all(v is not None and v.has_position() for v in coords):
        return None
    area = 0.0
    for i in range(len(coords)):
        x1, y1 = coords[i].x, coords[i].y
        x2, y2 = coords[(i + 1) % len(coords)].x, coords[(i + 1) % len(coords)].y
        area += x1 * y2 - x2 * y1
    return area / 2.0


def edge_length(vertices: Dict[str, Vertex], a: str, b: str) -> float:
    """Euclidean distance between two positioned vertices."""
    va = vertices[a]
    vb = vertices[b]
    return math.hypot(vb.x - va.x, vb.y - va.y)


def face_center(vertices: Dict[str, Vertex], face: Face) -> tuple[float, float] | None:
    """Centroid of a face (2D, using x/y coordinates)."""
    coords = [vertices[vid] for vid in face.vertex_ids]
    if not coords or not all(v.has_position() for v in coords):
        return None
    cx = sum(v.x for v in coords if v.x is not None) / len(coords)
    cy = sum(v.y for v in coords if v.y is not None) / len(coords)
    return (cx, cy)


def face_center_3d(
    vertices: Dict[str, Vertex], face: Face,
) -> tuple[float, float, float] | None:
    """Centroid of a face in 3D (x, y, z).  Returns *None* if any vertex lacks a z coordinate."""
    coords = [vertices[vid] for vid in face.vertex_ids]
    if not coords or not all(v.has_position_3d() for v in coords):
        return None
    n = len(coords)
    cx = sum(v.x for v in coords if v.x is not None) / n
    cy = sum(v.y for v in coords if v.y is not None) / n
    cz = sum(v.z for v in coords if v.z is not None) / n
    return (cx, cy, cz)


def grid_center(vertices: Dict[str, Vertex]) -> tuple[float, float]:
    """Mean of all positioned vertex coordinates."""
    xs = [v.x for v in vertices.values() if v.x is not None]
    ys = [v.y for v in vertices.values() if v.y is not None]
    return (sum(xs) / len(xs), sum(ys) / len(ys)) if xs and ys else (0.0, 0.0)


def find_pentagon_face(faces: Dict[str, Face]) -> Face | None:
    """Return the first pentagon face found, or None."""
    for face in faces.values():
        if face.face_type == "pent" or len(face.vertex_ids) == 5:
            return face
    return None


def collect_face_vertices(
    faces: Dict[str, Face], face_ids: Iterable[str]
) -> List[str]:
    """Return unique vertex ids from the given faces, preserving first-seen order."""
    vertex_ids: list[str] = []
    for fid in face_ids:
        face = faces.get(fid)
        if not face:
            continue
        vertex_ids.extend(face.vertex_ids)
    return list(dict.fromkeys(vertex_ids))


def boundary_vertex_cycle(edges: Iterable[Edge]) -> List[str]:
    """Walk boundary edges (those with < 2 faces) to produce an ordered vertex cycle.

    Returns an empty list if the boundary is not a simple closed loop.
    """
    boundary_edges = [e for e in edges if len(e.face_ids) < 2]
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
