from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List

from .models import Edge, Face, Vertex
from .polygrid import PolyGrid
from .algorithms import build_face_adjacency, ring_faces
from .angle_solver import ring_angle_spec
from .geometry import _face_signed_area, _face_vertex_cycle, _interior_angle, _ordered_face_vertices
from .grid_utils import _collect_face_vertices, _edge_length, _find_pentagon_face, _grid_center


@dataclass(frozen=True)
class RingStats:
    ring: int
    protruding_lengths: list[float]
    pointy_lengths: list[float]
    inner_angles: list[float]
    pointy_angles: list[float]


def min_face_signed_area(grid: PolyGrid) -> float:
    areas = []
    for face in grid.faces.values():
        area = _face_signed_area(grid.vertices, face, grid.edges.values())
        if area is not None:
            areas.append(area)
    return min(areas) if areas else 0.0


def has_edge_crossings(grid: PolyGrid) -> bool:
    edges = list(grid.edges.values())
    for i, edge_a in enumerate(edges):
        a1, a2 = edge_a.vertex_ids
        va1 = grid.vertices[a1]
        va2 = grid.vertices[a2]
        if not (va1.has_position() and va2.has_position()):
            continue
        for edge_b in edges[i + 1 :]:
            b1, b2 = edge_b.vertex_ids
            if len({a1, a2, b1, b2}) < 4:
                continue
            vb1 = grid.vertices[b1]
            vb2 = grid.vertices[b2]
            if not (vb1.has_position() and vb2.has_position()):
                continue
            if _segments_intersect((va1.x, va1.y), (va2.x, va2.y), (vb1.x, vb1.y), (vb2.x, vb2.y)):
                return True
    return False


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


def ring_quality_gates(
    stats: RingStats,
    angle_tol_deg: float = 3.0,
    protrude_rel_tol: float = 0.15,
) -> Dict[str, float | bool]:
    """Return per-ring quality gate metrics and pass/fail flags.

    Compares inner/pointy angles to ring specs and evaluates protruding length variance.
    """
    spec = ring_angle_spec(stats.ring)
    inner_target = spec.inner_angle_deg
    pointy_target = spec.outer_angle_deg

    inner_mean = _mean(stats.inner_angles)
    pointy_mean = _mean(stats.pointy_angles)
    protrude_mean = _mean(stats.protruding_lengths)
    protrude_range = _max(stats.protruding_lengths) - _min(stats.protruding_lengths)

    inner_ok = abs(inner_mean - inner_target) <= angle_tol_deg
    pointy_ok = abs(pointy_mean - pointy_target) <= angle_tol_deg
    protrude_ok = (
        protrude_mean == 0.0
        or (protrude_range / protrude_mean) <= protrude_rel_tol
    )

    return {
        "inner_angle_mean": inner_mean,
    "inner_angle_target": inner_target,
        "inner_angle_ok": inner_ok,
        "pointy_angle_mean": pointy_mean,
    "pointy_angle_target": pointy_target,
        "pointy_angle_ok": pointy_ok,
        "protrude_rel_range": (protrude_range / protrude_mean) if protrude_mean else 0.0,
        "protrude_ok": protrude_ok,
        "passed": inner_ok and pointy_ok and protrude_ok,
    }


def diagnostics_report(grid: PolyGrid, max_ring: int) -> Dict[str, object]:
    """Build a structured diagnostics report suitable for JSON export."""
    rings = ring_diagnostics(grid, max_ring=max_ring)
    ring_payload = {}
    for ring in sorted(rings.keys()):
        stats = rings[ring]
        ring_payload[str(ring)] = {
            "summary": summarize_ring_stats(stats),
            "quality": ring_quality_gates(stats),
        }

    return {
        "min_face_signed_area": min_face_signed_area(grid),
        "edge_crossings": has_edge_crossings(grid),
        "rings": ring_payload,
    }


def _segments_intersect(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    def orient(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> bool:
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)

    if o1 == 0 and on_segment(a1, b1, a2):
        return True
    if o2 == 0 and on_segment(a1, b2, a2):
        return True
    if o3 == 0 and on_segment(b1, a1, b2):
        return True
    if o4 == 0 and on_segment(b1, a2, b2):
        return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


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


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _min(values: List[float]) -> float:
    return min(values) if values else 0.0


def _max(values: List[float]) -> float:
    return max(values) if values else 0.0
