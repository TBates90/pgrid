from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List

from .models import Edge, Face, Vertex
from .polygrid import PolyGrid
from .algorithms import build_face_adjacency, ring_faces


@dataclass(frozen=True)
class RingStats:
    ring: int
    protruding_lengths: list[float]
    pointy_lengths: list[float]
    inner_angles: list[float]
    pointy_angles: list[float]


def ring_diagnostics(grid: PolyGrid, max_ring: int) -> Dict[int, RingStats]:
    """Compute per-ring diagnostics for lengths and angles.

    Angles are returned in degrees. Lists may be empty if a ring has no samples.
    """
    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return {}

    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings = ring_faces(adjacency, pent_face.id, max_depth=max_ring)
    stats: Dict[int, RingStats] = {}

    center = _grid_center(grid)
    for ring_idx in range(1, max(rings.keys()) + 1):
        ring_face_ids = rings.get(ring_idx, [])
        if not ring_face_ids:
            continue
        ring_faces_list = [grid.faces[fid] for fid in ring_face_ids if fid in grid.faces]
        ring_vertices = set(_collect_face_vertices(grid, ring_face_ids))
        inner_vertices = set(_collect_face_vertices(grid, rings.get(ring_idx - 1, [])))
        ring_outer = ring_vertices - inner_vertices
        if not ring_vertices:
            continue

        inner_neighbor_counts = _inner_neighbor_counts(grid.edges.values(), ring_vertices, inner_vertices)

        protruding_lengths: list[float] = []
        for edge in grid.edges.values():
            a, b = edge.vertex_ids
            if (a in ring_vertices and b in inner_vertices) or (b in ring_vertices and a in inner_vertices):
                inner = a if a in inner_vertices else b
                outer = b if inner == a else a
                if inner_neighbor_counts.get(outer, 0) != 1:
                    continue
                protruding_lengths.append(_edge_length(grid.vertices, outer, inner))

        pointy_lengths: list[float] = []
        inner_angles: list[float] = []
        pointy_angles: list[float] = []
        for face in ring_faces_list:
            ordered = _face_vertex_cycle(face, grid.edges.values())
            if len(ordered) != len(face.vertex_ids):
                ordered = _ordered_face_vertices(grid.vertices, face)
            if not ordered:
                continue

            # Inner angles live on the inner-ring vertices of this face.
            inner_idxs = [idx for idx, vid in enumerate(ordered) if vid in inner_vertices]
            for idx in inner_idxs:
                prev_vid = ordered[(idx - 1) % len(ordered)]
                next_vid = ordered[(idx + 1) % len(ordered)]
                inner_angles.append(_interior_angle(grid.vertices, prev_vid, ordered[idx], next_vid))

            # Pointy vertex is the outermost vertex among this face's ring vertices.
            candidates = [vid for vid in ordered if vid in ring_outer]
            if candidates:
                pointy = max(
                    candidates,
                    key=lambda vid: math.hypot(
                        grid.vertices[vid].x - center[0],
                        grid.vertices[vid].y - center[1],
                    ),
                )
                idx = ordered.index(pointy)
                prev_vid = ordered[(idx - 1) % len(ordered)]
                next_vid = ordered[(idx + 1) % len(ordered)]
                pointy_lengths.append(_edge_length(grid.vertices, pointy, prev_vid))
                pointy_lengths.append(_edge_length(grid.vertices, pointy, next_vid))
                pointy_angles.append(_interior_angle(grid.vertices, prev_vid, pointy, next_vid))

        stats[ring_idx] = RingStats(
            ring=ring_idx,
            protruding_lengths=protruding_lengths,
            pointy_lengths=pointy_lengths,
            inner_angles=[math.degrees(a) for a in inner_angles],
            pointy_angles=[math.degrees(a) for a in pointy_angles],
        )

    return stats


def summarize_ring_stats(stats: RingStats) -> Dict[str, float]:
    return {
        "protruding_min": _min(stats.protruding_lengths),
        "protruding_max": _max(stats.protruding_lengths),
        "protruding_mean": _mean(stats.protruding_lengths),
        "pointy_min": _min(stats.pointy_lengths),
        "pointy_max": _max(stats.pointy_lengths),
        "pointy_mean": _mean(stats.pointy_lengths),
        "inner_angle_min": _min(stats.inner_angles),
        "inner_angle_max": _max(stats.inner_angles),
        "inner_angle_mean": _mean(stats.inner_angles),
        "pointy_angle_min": _min(stats.pointy_angles),
        "pointy_angle_max": _max(stats.pointy_angles),
        "pointy_angle_mean": _mean(stats.pointy_angles),
    }


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
        cycle.append(nxt)
        prev, current = current, nxt
        if len(cycle) > len(neighbors) + 1:
            break

    return cycle


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


def _interior_angle(vertices: Dict[str, Vertex], a: str, b: str, c: str) -> float:
    angle = _angle_at_vertex(vertices, a, b, c)
    return max(angle, math.pi - angle)


def _edge_length(vertices: Dict[str, Vertex], a: str, b: str) -> float:
    va = vertices[a]
    vb = vertices[b]
    return math.hypot(vb.x - va.x, vb.y - va.y)


def _inner_neighbor_counts(
    edges: Iterable[Edge],
    ring_vertices: Iterable[str],
    inner_vertices: Iterable[str],
) -> Dict[str, int]:
    ring_set = set(ring_vertices)
    inner_set = set(inner_vertices)
    counts = {vid: 0 for vid in ring_set}
    for edge in edges:
        a, b = edge.vertex_ids
        if a in ring_set and b in inner_set:
            counts[a] += 1
        elif b in ring_set and a in inner_set:
            counts[b] += 1
    return counts


def _grid_center(grid: PolyGrid) -> tuple[float, float]:
    xs = [v.x for v in grid.vertices.values() if v.x is not None]
    ys = [v.y for v in grid.vertices.values() if v.y is not None]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _collect_face_vertices(grid: PolyGrid, face_ids: Iterable[str]) -> List[str]:
    vertex_ids = []
    for fid in face_ids:
        face = grid.faces.get(fid)
        if not face:
            continue
        vertex_ids.extend(face.vertex_ids)
    return list(dict.fromkeys(vertex_ids))


def _find_pentagon_face(grid: PolyGrid) -> Face | None:
    for face in grid.faces.values():
        if face.face_type == "pent" or len(face.vertex_ids) == 5:
            return face
    return None


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _min(values: List[float]) -> float:
    return min(values) if values else 0.0


def _max(values: List[float]) -> float:
    return max(values) if values else 0.0
