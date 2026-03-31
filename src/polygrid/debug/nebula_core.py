"""Frequency-1 PolyGrid wireframe data for debug visualizations.

This module keeps the geometry extraction logic in pgrid so playground and
other clients can reuse the same mesh source for animation experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, List, Sequence, Tuple

from polygrid.globe.globe import build_globe_grid

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class FrequencyOneWireframe:
    """Compact wireframe payload for a frequency-1 globe."""

    vertices: Tuple[Vec3, ...]
    edges: Tuple[Tuple[int, int], ...]
    edge_strengths: Tuple[float, ...]


def _normalize(vec: Sequence[float]) -> Vec3:
    x = float(vec[0])
    y = float(vec[1])
    z = float(vec[2])
    length = sqrt((x * x) + (y * y) + (z * z))
    if length <= 1e-9:
        return (0.0, 0.0, 1.0)
    inv = 1.0 / length
    return (x * inv, y * inv, z * inv)


def _vertex_key(vec: Sequence[float], precision: int = 6) -> Tuple[float, float, float]:
    return (
        round(float(vec[0]), precision),
        round(float(vec[1]), precision),
        round(float(vec[2]), precision),
    )


def _build_edges_from_faces(face_vertex_indices: Iterable[Tuple[int, ...]]) -> Tuple[Tuple[int, int], ...]:
    edges: set[Tuple[int, int]] = set()
    for indices in face_vertex_indices:
        count = len(indices)
        for idx in range(count):
            a = indices[idx]
            b = indices[(idx + 1) % count]
            lo = min(a, b)
            hi = max(a, b)
            edges.add((lo, hi))
    return tuple(sorted(edges))


def build_frequency_one_wireframe(*, radius: float = 1.0) -> FrequencyOneWireframe:
    """Build a reusable frequency-1 wireframe payload.

    Returns a deduplicated vertex list and undirected edge index list, plus a
    per-edge strength value (0.6-1.0) based on edge latitude for easy glow/line
    weighting in renderers.
    """

    grid = build_globe_grid(1, radius=radius)

    vertex_map: Dict[Tuple[float, float, float], int] = {}
    vertices: List[Vec3] = []
    face_indices: List[Tuple[int, ...]] = []

    for face in grid.faces.values():
        current_face: List[int] = []
        for vertex_id in face.vertex_ids:
            vertex = grid.vertices.get(vertex_id)
            if vertex is None:
                continue
            coords = _normalize((vertex.x, vertex.y, vertex.z))
            key = _vertex_key(coords)
            existing = vertex_map.get(key)
            if existing is None:
                existing = len(vertices)
                vertex_map[key] = existing
                vertices.append(coords)
            current_face.append(existing)
        if len(current_face) >= 3:
            face_indices.append(tuple(current_face))

    edges = _build_edges_from_faces(face_indices)
    strengths: List[float] = []
    for a, b in edges:
        va = vertices[a]
        vb = vertices[b]
        mid_y = (va[1] + vb[1]) * 0.5
        latitude_weight = 1.0 - min(1.0, abs(mid_y))
        strengths.append(0.6 + (latitude_weight * 0.4))

    return FrequencyOneWireframe(
        vertices=tuple(vertices),
        edges=edges,
        edge_strengths=tuple(strengths),
    )
