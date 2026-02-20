from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .models import Edge, Face, Vertex
from .polygrid import PolyGrid
from .embedding import tutte_embedding


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


def build_pentagon_centered_grid(
    rings: int,
    size: float = 1.0,
    embed: bool = True,
) -> PolyGrid:
    """Build a pentagon-centered grid via triangulation + wedge removal + dual.

    Steps:
    1) Build a triangular lattice in a hex-shaped region.
    2) Remove a 60° wedge and merge boundary rays (5-valent defect).
    3) Dualize the triangulation (triangle centroids -> vertices).
    4) Apply Tutte embedding for stable layout.
    """
    if rings < 0:
        raise ValueError("rings must be >= 0")

    tri_rings = rings + 2
    tri_vertices = _build_triangular_vertices(tri_rings, size)
    triangles = _build_triangles(tri_vertices)

    angle_min = 0.0
    angle_max = math.pi / 3
    angle_tol = math.radians(3)

    triangles = _remove_wedge(tri_vertices, triangles, angle_min, angle_max)
    merge_map = _build_ray_merge_map(tri_vertices, angle_min, angle_max, angle_tol)
    tri_vertices = _merge_vertices(tri_vertices, merge_map)
    triangles = _merge_triangles(triangles, merge_map)

    dual_vertices, dual_faces = _dualize(tri_vertices, triangles)

    edges, dual_faces = _rebuild_edges_from_faces(dual_faces)
    grid = PolyGrid(dual_vertices.values(), edges, dual_faces)

    boundary_ids = _boundary_vertex_ids(grid)
    if embed and boundary_ids:
        embedded = tutte_embedding(grid.vertices, grid.edges.values(), boundary_ids)
        grid = PolyGrid(embedded.values(), grid.edges.values(), grid.faces.values())

    return grid


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


def _collapse_single_face(grid: PolyGrid) -> PolyGrid:
    center_face = _find_center_face(grid)
    if center_face is None:
        return grid

    vertex_ids = list(center_face.vertex_ids)
    if len(vertex_ids) <= 5:
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
        ordered = _order_vertices_ccw(vertices, compacted)
        faces.append(
            Face(
                id=face.id,
                face_type="pent" if face.id == center_face.id else face.face_type,
                vertex_ids=tuple(ordered),
            )
        )

    edges, faces = _rebuild_edges_from_faces(faces)
    return PolyGrid(vertices.values(), edges, faces)


def _face_center(grid: PolyGrid, face: Face) -> Tuple[float, float] | None:
    coords = [grid.vertices[vid] for vid in face.vertex_ids]
    if not all(v.has_position() for v in coords):
        return None
    cx = sum(v.x for v in coords if v.x is not None) / len(coords)
    cy = sum(v.y for v in coords if v.y is not None) / len(coords)
    return cx, cy


def _compact_vertices(vertex_ids: List[str]) -> List[str]:
    compacted: List[str] = []
    for vid in vertex_ids:
        if not compacted or compacted[-1] != vid:
            compacted.append(vid)
    if len(compacted) > 1 and compacted[0] == compacted[-1]:
        compacted.pop()
    return compacted


def _angle_between(theta: float, angle_min: float, angle_max: float) -> bool:
    if angle_min <= angle_max:
        return angle_min <= theta <= angle_max
    return theta >= angle_min or theta <= angle_max


def _build_ray_merge_map(
    vertices: Dict[str, Vertex],
    angle_min: float,
    angle_max: float,
    angle_tol: float,
) -> Dict[str, str]:
    ray_min: Dict[str, Tuple[float, str]] = {}
    ray_max: Dict[str, Tuple[float, str]] = {}

    for vertex in vertices.values():
        if not vertex.has_position():
            continue
        theta = math.atan2(vertex.y, vertex.x)
        radius = math.hypot(vertex.x, vertex.y)
        key = f"{radius:.5f}"
        if abs(theta - angle_min) <= angle_tol:
            ray_min[key] = (radius, vertex.id)
        elif abs(theta - angle_max) <= angle_tol:
            ray_max[key] = (radius, vertex.id)

    merge: Dict[str, str] = {}
    for key, (_, vid_min) in ray_min.items():
        if key in ray_max:
            merge[vid_min] = ray_max[key][1]
    return merge


def _merge_vertices(vertices: Dict[str, Vertex], merge: Dict[str, str]) -> Dict[str, Vertex]:
    merged: Dict[str, Vertex] = {}
    for vid, vertex in vertices.items():
        target_id = merge.get(vid, vid)
        if target_id in merged:
            continue
        merged[target_id] = vertex if target_id == vid else Vertex(target_id, vertex.x, vertex.y)
    return merged


def _build_triangular_vertices(rings: int, size: float) -> Dict[str, Vertex]:
    coords = _hex_area(rings)
    vertices: Dict[str, Vertex] = {}
    for coord in coords:
        x = size * (coord.q + coord.r / 2)
        y = size * (math.sqrt(3) / 2 * coord.r)
        key = f"{coord.q},{coord.r}"
        vertices[key] = Vertex(key, x, y)
    return vertices


def _build_triangles(vertices: Dict[str, Vertex]) -> List[Tuple[str, str, str]]:
    triangles: List[Tuple[str, str, str]] = []
    coords = [tuple(map(int, vid.split(","))) for vid in vertices.keys()]
    coord_set = {coord: f"{coord[0]},{coord[1]}" for coord in coords}

    for q, r in coords:
        a = coord_set.get((q, r))
        b = coord_set.get((q + 1, r))
        c = coord_set.get((q, r + 1))
        d = coord_set.get((q + 1, r - 1))
        if a and b and c:
            triangles.append((a, b, c))
        if a and b and d:
            triangles.append((a, d, b))
    return triangles


def _remove_wedge(
    vertices: Dict[str, Vertex],
    triangles: List[Tuple[str, str, str]],
    angle_min: float,
    angle_max: float,
) -> List[Tuple[str, str, str]]:
    filtered: List[Tuple[str, str, str]] = []
    for tri in triangles:
        cx, cy = _triangle_centroid(vertices, tri)
        if abs(cx) < 1e-8 and abs(cy) < 1e-8:
            filtered.append(tri)
            continue
        theta = math.atan2(cy, cx)
        if _angle_between(theta, angle_min, angle_max):
            continue
        filtered.append(tri)
    return filtered


def _merge_triangles(
    triangles: List[Tuple[str, str, str]],
    merge_map: Dict[str, str],
) -> List[Tuple[str, str, str]]:
    merged: List[Tuple[str, str, str]] = []
    for a, b, c in triangles:
        new_ids = [merge_map.get(a, a), merge_map.get(b, b), merge_map.get(c, c)]
        if len(set(new_ids)) < 3:
            continue
        merged.append(tuple(new_ids))
    return merged


def _triangle_centroid(vertices: Dict[str, Vertex], tri: Tuple[str, str, str]) -> Tuple[float, float]:
    pts = [vertices[vid] for vid in tri]
    cx = sum(v.x for v in pts if v.x is not None) / 3
    cy = sum(v.y for v in pts if v.y is not None) / 3
    return cx, cy


def _dualize(
    vertices: Dict[str, Vertex],
    triangles: List[Tuple[str, str, str]],
) -> Tuple[Dict[str, Vertex], List[Face]]:
    tri_ids = []
    dual_vertices: Dict[str, Vertex] = {}
    incident: Dict[str, List[str]] = {}

    for idx, tri in enumerate(triangles, start=1):
        tri_id = f"t{idx}"
        tri_ids.append(tri_id)
        cx, cy = _triangle_centroid(vertices, tri)
        dual_vertices[tri_id] = Vertex(tri_id, cx, cy)
        for vid in tri:
            incident.setdefault(vid, []).append(tri_id)

    faces: List[Face] = []
    for vid, tri_list in incident.items():
        if len(tri_list) < 5:
            continue
        ordered = _order_triangles_around_vertex(vertices, dual_vertices, vid, tri_list)
        face_type = "pent" if len(ordered) == 5 else "hex"
        faces.append(Face(id=f"f_{vid}", face_type=face_type, vertex_ids=tuple(ordered)))

    return dual_vertices, faces


def _order_triangles_around_vertex(
    vertices: Dict[str, Vertex],
    dual_vertices: Dict[str, Vertex],
    vertex_id: str,
    triangle_ids: List[str],
) -> List[str]:
    center = vertices[vertex_id]
    if center.x is None or center.y is None:
        return triangle_ids

    def angle(tid: str) -> float:
        tri = dual_vertices[tid]
        return math.atan2(tri.y - center.y, tri.x - center.x)

    return sorted(triangle_ids, key=angle)


def _boundary_vertex_ids(grid: PolyGrid) -> List[str]:
    degree = {vid: 0 for vid in grid.vertices}
    for edge in grid.edges.values():
        degree[edge.vertex_ids[0]] += 1
        degree[edge.vertex_ids[1]] += 1
    boundary = [vid for vid, deg in degree.items() if deg < 3]
    if not boundary:
        return []
    cx = sum(grid.vertices[vid].x for vid in boundary if grid.vertices[vid].x is not None) / len(boundary)
    cy = sum(grid.vertices[vid].y for vid in boundary if grid.vertices[vid].y is not None) / len(boundary)
    return sorted(boundary, key=lambda vid: math.atan2(grid.vertices[vid].y - cy, grid.vertices[vid].x - cx))


def _order_vertices_ccw(vertices: Dict[str, Vertex], vertex_ids: List[str]) -> List[str]:
    coords = [vertices[vid] for vid in vertex_ids]
    if not all(v.has_position() for v in coords):
        return vertex_ids
    cx = sum(v.x for v in coords if v.x is not None) / len(coords)
    cy = sum(v.y for v in coords if v.y is not None) / len(coords)

    def angle(vid: str) -> float:
        vertex = vertices[vid]
        return math.atan2(vertex.y - cy, vertex.x - cx) if vertex.y is not None else 0.0

    return sorted(vertex_ids, key=angle)


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
