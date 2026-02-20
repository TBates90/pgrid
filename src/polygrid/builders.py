from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .models import Edge, Face, Vertex
from .polygrid import PolyGrid
from .embedding import tutte_embedding
from .algorithms import build_face_adjacency, ring_faces


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

    if embed:
        fixed_positions = _build_fixed_positions(grid)
        if fixed_positions:
            embedded = tutte_embedding(grid.vertices, grid.edges.values(), fixed_positions)
            relaxed = _laplacian_relax(
                embedded,
                grid.edges.values(),
                fixed_positions,
                iterations=40,
                alpha=0.25,
            )
            ring1_relaxed = _ring1_symmetry_relax(
                grid,
                relaxed,
                fixed_positions,
                iterations=120,
                strength=0.25,
            )
            snapped = _ring1_symmetry_snap(grid, ring1_relaxed, fixed_positions)
            angled = _ring1_pent_angle_snap(
                grid,
                snapped,
                fixed_positions,
                target_angle_deg=126.0,
                strength=1.0,
            )
            grid = PolyGrid(angled.values(), grid.edges.values(), grid.faces.values())

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


def _angle_between(theta: float, angle_min: float, angle_max: float) -> bool:
    if angle_min <= angle_max:
        return angle_min <= theta <= angle_max
    return theta >= angle_min or theta <= angle_max


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


def _face_center(grid: PolyGrid, face: Face) -> tuple[float, float] | None:
    coords = [grid.vertices[vid] for vid in face.vertex_ids]
    if not all(v.has_position() for v in coords):
        return None
    cx = sum(v.x for v in coords if v.x is not None) / len(coords)
    cy = sum(v.y for v in coords if v.y is not None) / len(coords)
    return cx, cy


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


def _pent_edge_normal_angle(
    grid: PolyGrid,
    pent_face: Face,
    v1: str,
    v2: str,
    pent_center: tuple[float, float],
) -> float:
    p1 = grid.vertices[v1]
    p2 = grid.vertices[v2]
    mx = (p1.x + p2.x) / 2
    my = (p1.y + p2.y) / 2
    dx = mx - pent_center[0]
    dy = my - pent_center[1]
    return math.atan2(dy, dx)


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


def _build_fixed_positions(grid: PolyGrid) -> Dict[str, tuple[float, float]]:
    boundary_ids = _boundary_vertex_ids(grid)
    if not boundary_ids:
        return {}

    center = _grid_center(grid)
    boundary_radius = _mean_radius(grid, boundary_ids, center)

    fixed: Dict[str, tuple[float, float]] = {}
    fixed.update(_uniform_circle_positions(grid, boundary_ids, boundary_radius, center))

    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return fixed

    pent_vertices = list(pent_face.vertex_ids)
    pent_radius = _mean_radius(grid, pent_vertices, center)
    fixed.update(_uniform_circle_positions(grid, pent_vertices, pent_radius, center))

    return fixed


def _grid_center(grid: PolyGrid) -> tuple[float, float]:
    xs = [v.x for v in grid.vertices.values() if v.x is not None]
    ys = [v.y for v in grid.vertices.values() if v.y is not None]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _mean_radius(grid: PolyGrid, vertex_ids: Iterable[str], center: tuple[float, float]) -> float:
    radii = []
    for vid in vertex_ids:
        v = grid.vertices[vid]
        if v.x is None or v.y is None:
            continue
        radii.append(math.hypot(v.x - center[0], v.y - center[1]))
    return sum(radii) / len(radii) if radii else 0.0


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


def _uniform_circle_positions(
    grid: PolyGrid,
    vertex_ids: List[str],
    radius: float,
    center: tuple[float, float],
) -> Dict[str, tuple[float, float]]:
    if not vertex_ids:
        return {}

    angles = []
    for vid in vertex_ids:
        v = grid.vertices[vid]
        angles.append(math.atan2(v.y - center[1], v.x - center[0]))

    ordered = [vid for _, vid in sorted(zip(angles, vertex_ids))]
    base_angle = sum(angles) / len(angles)
    step = 2 * math.pi / len(ordered)

    positions: Dict[str, tuple[float, float]] = {}
    for idx, vid in enumerate(ordered):
        angle = base_angle + idx * step
        positions[vid] = (
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
        )
    return positions


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


def _edge_length_relax_safe(
    vertices: Dict[str, Vertex],
    edges: Iterable[Edge],
    fixed_positions: Dict[str, tuple[float, float]],
    iterations: int,
    strength: float,
    max_step: float,
) -> Dict[str, Vertex]:
    fixed_set = set(fixed_positions.keys())
    current = {vid: Vertex(vid, v.x, v.y) for vid, v in vertices.items()}
    target = _mean_edge_length(current, edges)

    for _ in range(iterations):
        deltas: Dict[str, list[float]] = {vid: [0.0, 0.0, 0.0] for vid in current}
        for edge in edges:
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
            step_len = math.hypot(step_x, step_y)
            if step_len > max_step:
                scale = max_step / step_len
                step_x *= scale
                step_y *= scale
            next_state[vid] = Vertex(vid, vertex.x - step_x, vertex.y - step_y)

        current = next_state

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




def _edge_length_relax(
    vertices: Dict[str, Vertex],
    edges: Iterable[Edge],
    fixed_positions: Dict[str, tuple[float, float]],
    iterations: int,
    strength: float,
) -> Dict[str, Vertex]:
    fixed_set = set(fixed_positions.keys())
    current = {vid: Vertex(vid, v.x, v.y) for vid, v in vertices.items()}

    target = _mean_edge_length(current, edges)

    for _ in range(iterations):
        updates: Dict[str, list[float]] = {vid: [0.0, 0.0, 0.0] for vid in current}
        for edge in edges:
            a, b = edge.vertex_ids
            va = current[a]
            vb = current[b]
            dx = vb.x - va.x
            dy = vb.y - va.y
            dist = math.hypot(dx, dy) or 1.0
            diff = (dist - target) / dist
            ux = dx * diff
            uy = dy * diff
            updates[a][0] += ux
            updates[a][1] += uy
            updates[a][2] += 1.0
            updates[b][0] -= ux
            updates[b][1] -= uy
            updates[b][2] += 1.0

        next_state: Dict[str, Vertex] = {}
        for vid, vertex in current.items():
            if vid in fixed_set:
                x, y = fixed_positions[vid]
                next_state[vid] = Vertex(vid, x, y)
                continue
            total_x, total_y, count = updates[vid]
            if count == 0:
                next_state[vid] = vertex
                continue
            step_x = strength * (total_x / count)
            step_y = strength * (total_y / count)
            next_state[vid] = Vertex(vid, vertex.x - step_x, vertex.y - step_y)

        current = next_state

    return current


def _regularity_relax(
    vertices: Dict[str, Vertex],
    edges: Iterable[Edge],
    fixed_positions: Dict[str, tuple[float, float]],
    iterations: int,
    edge_weight: float,
    angle_weight: float,
    radial_bias: Dict[str, float],
    center: tuple[float, float],
    radial_weight: float = 0.1,
    max_step: float = 0.2,
) -> Dict[str, Vertex]:
    neighbors: Dict[str, List[str]] = {vid: [] for vid in vertices}
    for edge in edges:
        a, b = edge.vertex_ids
        neighbors[a].append(b)
        neighbors[b].append(a)

    fixed_set = set(fixed_positions.keys())
    current = {vid: Vertex(vid, v.x, v.y) for vid, v in vertices.items()}

    target_edge = _mean_edge_length(current, edges)

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

            # Edge length regularization
            edge_x = 0.0
            edge_y = 0.0
            for n in nbrs:
                vn = current[n]
                dx = vn.x - vertex.x
                dy = vn.y - vertex.y
                dist = math.hypot(dx, dy) or 1.0
                edge_x += dx / dist * (dist - target_edge)
                edge_y += dy / dist * (dist - target_edge)

            # Angle regularization (encourage 120° around a vertex)
            angle_x = 0.0
            angle_y = 0.0
            if len(nbrs) >= 2:
                angles = [math.atan2(current[n].y - vertex.y, current[n].x - vertex.x) for n in nbrs]
                angles.sort()
                ideal = 2 * math.pi / len(nbrs)
                for i in range(len(angles)):
                    a1 = angles[i]
                    a2 = angles[(i + 1) % len(angles)]
                    diff = (a2 - a1) % (2 * math.pi)
                    err = diff - ideal
                    angle_x += -math.cos(a1 + diff / 2) * err
                    angle_y += -math.sin(a1 + diff / 2) * err

            radial_x = 0.0
            radial_y = 0.0
            if vid in radial_bias:
                r_target = radial_bias[vid]
                dx = vertex.x - center[0]
                dy = vertex.y - center[1]
                dist = math.hypot(dx, dy) or 1.0
                radial_err = dist - r_target
                radial_x = (dx / dist) * radial_err
                radial_y = (dy / dist) * radial_err

            step_x = edge_weight * edge_x + angle_weight * angle_x + radial_weight * radial_x
            step_y = edge_weight * edge_y + angle_weight * angle_y + radial_weight * radial_y
            step_len = math.hypot(step_x, step_y)
            if step_len > max_step:
                scale = max_step / step_len
                step_x *= scale
                step_y *= scale

            new_x = vertex.x - step_x
            new_y = vertex.y - step_y
            updated[vid] = Vertex(vid, new_x, new_y)
        current = updated

    return current


def _mean_edge_length(vertices: Dict[str, Vertex], edges: Iterable[Edge]) -> float:
    lengths = []
    for edge in edges:
        v1 = vertices[edge.vertex_ids[0]]
        v2 = vertices[edge.vertex_ids[1]]
        if v1.x is None or v2.x is None:
            continue
        lengths.append(math.hypot(v1.x - v2.x, v1.y - v2.y))
    return sum(lengths) / len(lengths) if lengths else 1.0


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
