from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .models import Edge, Face, Vertex
from .polygrid import PolyGrid


@dataclass(frozen=True)
class AxialCoord:
    q: int
    r: int


def build_pure_hex_grid(rings: int, size: float = 1.0) -> PolyGrid:
    """Build a pure hex grid using axial coordinates.

    rings=0 produces a single hex. rings=1 produces 7 hexes, etc.
    """
    if rings < 0:
        raise ValueError("rings must be >= 0")

    coords = _hex_area(rings)
    vertex_map: Dict[str, Vertex] = {}
    edge_map: Dict[Tuple[str, str], Edge] = {}
    faces: List[Face] = []

    for idx, coord in enumerate(coords, start=1):
        center = _axial_to_pixel(coord, size)
        corners = _hex_corners(center, size)
        vertex_ids = [_get_vertex_id(vertex_map, corner) for corner in corners]
        edge_ids = _get_edge_ids(edge_map, vertex_ids, face_id=f"f{idx}")
        faces.append(
            Face(
                id=f"f{idx}",
                face_type="hex",
                vertex_ids=tuple(vertex_ids),
                edge_ids=tuple(edge_ids),
            )
        )

    return PolyGrid(list(vertex_map.values()), list(edge_map.values()), faces)


def hex_face_count(rings: int) -> int:
    if rings < 0:
        raise ValueError("rings must be >= 0")
    return 1 + 3 * rings * (rings + 1)


def build_pentagon_centered_grid(rings: int, size: float = 1.0) -> PolyGrid:
    """Build a pentagon-centered grid by collapsing one edge of the center hex.

    This is an experimental starter that preserves topology and provides
    positions for visualization.
    """
    grid = build_pure_hex_grid(rings, size)
    center_face = _find_center_face(grid)
    if center_face is None:
        return grid

    vertex_ids = list(center_face.vertex_ids)
    if len(vertex_ids) < 6:
        return grid

    keep_id = vertex_ids[0]
    drop_id = vertex_ids[1]

    keep_vertex = grid.vertices[keep_id]
    drop_vertex = grid.vertices[drop_id]
    merged_vertex = Vertex(
        keep_id,
        (keep_vertex.x + drop_vertex.x) / 2 if keep_vertex.x is not None else None,
        (keep_vertex.y + drop_vertex.y) / 2 if keep_vertex.y is not None else None,
    )

    vertices = {vid: vertex for vid, vertex in grid.vertices.items() if vid != drop_id}
    vertices[keep_id] = merged_vertex

    faces: List[Face] = []
    for face in grid.faces.values():
        new_ids = [keep_id if vid == drop_id else vid for vid in face.vertex_ids]
        compacted = _compact_vertices(new_ids)
        faces.append(
            Face(
                id=face.id,
                face_type="pent" if face.id == center_face.id else face.face_type,
                vertex_ids=tuple(compacted),
            )
        )

    edges, faces = _rebuild_edges_from_faces(faces)
    return PolyGrid(vertices.values(), edges, faces)


def _hex_area(rings: int) -> List[AxialCoord]:
    coords: List[AxialCoord] = []
    for q in range(-rings, rings + 1):
        r1 = max(-rings, -q - rings)
        r2 = min(rings, -q + rings)
        for r in range(r1, r2 + 1):
            coords.append(AxialCoord(q, r))
    return coords


def _axial_to_pixel(coord: AxialCoord, size: float) -> Tuple[float, float]:
    x = size * (1.5 * coord.q)
    y = size * (math.sqrt(3) * (coord.r + coord.q / 2))
    return x, y


def _hex_corners(center: Tuple[float, float], size: float) -> List[Tuple[float, float]]:
    cx, cy = center
    corners = []
    for i in range(6):
        angle = math.radians(60 * i)
        corners.append((cx + size * math.cos(angle), cy + size * math.sin(angle)))
    return corners


def _find_center_face(grid: PolyGrid) -> Face | None:
    best_face = None
    best_dist = float("inf")
    for face in grid.faces.values():
        coords = [grid.vertices[vid] for vid in face.vertex_ids]
        if not all(v.has_position() for v in coords):
            continue
        cx = sum(v.x for v in coords if v.x is not None) / len(coords)
        cy = sum(v.y for v in coords if v.y is not None) / len(coords)
        dist = (cx ** 2 + cy ** 2) ** 0.5
        if dist < best_dist:
            best_face = face
            best_dist = dist
    return best_face


def _compact_vertices(vertex_ids: List[str]) -> List[str]:
    compacted: List[str] = []
    for vid in vertex_ids:
        if not compacted or compacted[-1] != vid:
            compacted.append(vid)
    if len(compacted) > 1 and compacted[0] == compacted[-1]:
        compacted.pop()
    return compacted


def _rebuild_edges_from_faces(faces: Iterable[Face]) -> tuple[List[Edge], List[Face]]:
    edge_map: Dict[Tuple[str, str], Edge] = {}
    rebuilt_faces: List[Face] = []
    for face in faces:
        vertex_ids = list(face.vertex_ids)
        edge_ids: List[str] = []
        for i in range(len(vertex_ids)):
            a = vertex_ids[i]
            b = vertex_ids[(i + 1) % len(vertex_ids)]
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            edge = edge_map.get(key)
            if edge is None:
                edge = Edge(id=f"e{len(edge_map) + 1}", vertex_ids=key, face_ids=(face.id,))
                edge_map[key] = edge
            else:
                edge_map[key] = Edge(edge.id, edge.vertex_ids, edge.face_ids + (face.id,))
            edge_ids.append(edge_map[key].id)
        rebuilt_faces.append(
            Face(
                id=face.id,
                face_type=face.face_type,
                vertex_ids=face.vertex_ids,
                edge_ids=tuple(edge_ids),
            )
        )

    return list(edge_map.values()), rebuilt_faces


def _vertex_key(position: Tuple[float, float]) -> str:
    return f"{position[0]:.6f},{position[1]:.6f}"


def _get_vertex_id(vertex_map: Dict[str, Vertex], position: Tuple[float, float]) -> str:
    key = _vertex_key(position)
    if key not in vertex_map:
        vertex_map[key] = Vertex(f"v{len(vertex_map) + 1}", position[0], position[1])
    return vertex_map[key].id


def _get_edge_ids(
    edge_map: Dict[Tuple[str, str], Edge],
    vertex_ids: List[str],
    face_id: str,
) -> List[str]:
    edge_ids: List[str] = []
    count = len(vertex_ids)
    for i in range(count):
        a = vertex_ids[i]
        b = vertex_ids[(i + 1) % count]
        key = tuple(sorted((a, b)))
        edge = edge_map.get(key)
        if edge is None:
            edge = Edge(id=f"e{len(edge_map) + 1}", vertex_ids=key, face_ids=(face_id,))
            edge_map[key] = edge
        else:
            edge_map[key] = Edge(edge.id, edge.vertex_ids, edge.face_ids + (face_id,))
        edge_ids.append(edge_map[key].id)
    return edge_ids
