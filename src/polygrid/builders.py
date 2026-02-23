from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .models import Edge, Face, Vertex
from .polygrid import PolyGrid
from .embedding import tutte_embedding
from .angle_solver import ring_angle_spec, solve_ring_hex_lengths, solve_ring_hex_outer_length
from .algorithms import build_face_adjacency, ring_faces
try:  # pragma: no cover - optional dependency
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover
    least_squares = None


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
    embed_mode: str = "angle",
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
        if embed_mode == "angle":
            positioned = _apply_angle_first_layout(grid, rings, size)
            # Keep the angle-first layout exact; avoid post-relaxation that
            # can distort the target ring angles.
            grid = PolyGrid(positioned.values(), grid.edges.values(), grid.faces.values())
        else:
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
                constrained = _ring_constraints_snap(
                    grid,
                    snapped,
                    fixed_positions,
                    max_ring=len(grid.faces),
                    iterations=5,
                )
                protruded = _ring_protruding_edge_snap(
                    grid,
                    constrained,
                    fixed_positions,
                    max_ring=len(grid.faces),
                )
                angled = _ring1_pent_angle_snap(
                    grid,
                    protruded,
                    fixed_positions,
                    target_angle_deg=126.0,
                    strength=1.0,
                )
                pointy_final = _ring_pointy_edge_snap(
                    grid,
                    angled,
                    fixed_positions,
                    max_ring=len(grid.faces),
                )
                grid = PolyGrid(pointy_final.values(), grid.edges.values(), grid.faces.values())

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


def _apply_angle_first_layout(grid: PolyGrid, rings: int, size: float) -> Dict[str, Vertex]:
    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return grid.vertices

    pent_vertices = list(pent_face.vertex_ids)
    if len(pent_vertices) != 5:
        return grid.vertices

    side_length = size
    radius = side_length / (2 * math.sin(math.pi / 5))
    center = (0.0, 0.0)
    pent_points = [
        (
            radius * math.cos(2 * math.pi * i / 5),
            radius * math.sin(2 * math.pi * i / 5),
        )
        for i in range(5)
    ]

    ordered_pent = _face_vertex_cycle(pent_face, grid.edges.values())
    if len(ordered_pent) != 5:
        ordered_pent = _ordered_face_vertices(grid.vertices, pent_face)
    if len(ordered_pent) != 5:
        ordered_pent = pent_vertices

    current: Dict[str, Vertex] = {vid: Vertex(vid, v.x, v.y) for vid, v in grid.vertices.items()}
    for vid, (x, y) in zip(ordered_pent, pent_points):
        current[vid] = Vertex(vid, x, y)

    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings_map = ring_faces(adjacency, pent_face.id, max_depth=rings)

    prev_outer = side_length
    for ring_idx in range(1, rings + 1):
        ring_face_ids = rings_map.get(ring_idx, [])
        if not ring_face_ids:
            continue
        ring_faces_list = [grid.faces[fid] for fid in ring_face_ids if fid in grid.faces]

        region_faces = []
        for depth in range(0, ring_idx):
            region_faces.extend(rings_map.get(depth, []))
        inner_vertices = set(_collect_face_vertices(grid, region_faces))
        region_set = set(region_faces)
        ring_set = set(ring_face_ids)

        spec = ring_angle_spec(ring_idx)
        angles = [
            spec.inner_angle_deg,
            spec.outer_angle_deg,
            spec.outer_angle_deg,
            spec.outer_angle_deg,
            spec.outer_angle_deg,
            spec.inner_angle_deg,
        ]

        if ring_idx == 1:
            edge_lookup = {frozenset(edge.vertex_ids): edge for edge in grid.edges.values()}
            pent_index = {vid: idx for idx, vid in enumerate(ordered_pent)}
            ordered_faces = _order_ring_faces(current, ring_faces_list, inner_vertices, center)
            ring1_targets: Dict[str, List[float]] = {}
            ring1_counts: Dict[str, int] = {}
            for face in ordered_faces:
                ordered = _face_vertex_cycle(face, grid.edges.values())
                if len(ordered) != 6:
                    continue

                # identify the pent edge for this face to compute inner edge length
                pent_edge = None
                for edge_id in face.edge_ids:
                    edge = grid.edges.get(edge_id)
                    if edge is None:
                        continue
                    a, b = edge.vertex_ids
                    if a in inner_vertices and b in inner_vertices:
                        pent_edge = edge
                        break
                if pent_edge is None:
                    continue
                v0_id, v1_id = pent_edge.vertex_ids
                if v0_id in pent_index and v1_id in pent_index:
                    if (pent_index[v1_id] - pent_index[v0_id]) % len(ordered_pent) != 1:
                        v0_id, v1_id = v1_id, v0_id
                v0_inner = current[v0_id]
                v1_inner = current[v1_id]
                if v0_inner.x is None or v1_inner.x is None:
                    continue
                inner_edge_len = math.hypot(v1_inner.x - v0_inner.x, v1_inner.y - v0_inner.y)
                # build edge-length sequence based on topology instead of fixed order
                edge_types = []
                for i in range(len(ordered)):
                    a = ordered[i]
                    b = ordered[(i + 1) % len(ordered)]
                    edge = edge_lookup.get(frozenset((a, b)))
                    if edge is None:
                        edge_types.append("outer")
                        continue
                    if a in inner_vertices and b in inner_vertices:
                        edge_types.append("inner")
                    elif any(fid in ring_set and fid != face.id for fid in edge.face_ids):
                        edge_types.append("ring1")
                    else:
                        edge_types.append("outer")

                angles_seq = [
                    spec.inner_angle_deg if vid in inner_vertices else spec.outer_angle_deg
                    for vid in ordered
                ]

                inner_idx = next((i for i, t in enumerate(edge_types) if t == "inner"), None)
                if inner_idx is None:
                    continue
                ordered = _rotate_vertices(ordered, inner_idx)
                edge_types_rot = _rotate_list(edge_types, inner_idx)
                angles_vertex = _rotate_list(angles_seq, inner_idx)

                if not (ordered[0] == v0_id and ordered[1] == v1_id):
                    reversed_order = list(reversed(ordered))
                    rev_start = _find_edge_start_index(reversed_order, v0_id, v1_id)
                    if rev_start is not None:
                        ordered = _rotate_vertices(reversed_order, rev_start)
                        angles_vertex = _rotate_list(list(reversed(angles_vertex)), rev_start)

                # Recompute edge types based on the finalized ordering.
                edge_types_rot = []
                for i in range(len(ordered)):
                    a = ordered[i]
                    b = ordered[(i + 1) % len(ordered)]
                    edge = edge_lookup.get(frozenset((a, b)))
                    if edge is None:
                        edge_types_rot.append("outer")
                    elif a in inner_vertices and b in inner_vertices:
                        edge_types_rot.append("inner")
                    elif any(fid in ring_set and fid != face.id for fid in edge.face_ids):
                        edge_types_rot.append("ring1")
                    else:
                        edge_types_rot.append("outer")

                lengths = solve_ring_hex_lengths(ring_idx, inner_edge_length=inner_edge_len)

                lengths_by_type = {
                    "inner": lengths["inner"],
                    "ring1": lengths["protrude"],
                    "outer": lengths["outer"],
                }
                lengths_rot = [lengths_by_type[t] for t in edge_types_rot]
                angles_turn = _rotate_list(angles_vertex, 1)

                v0 = current[ordered[0]]
                v1 = current[ordered[1]]
                if v0.x is None or v1.x is None:
                    continue

                preferred_points = _hex_points_from_edge(
                    (v0.x, v0.y),
                    (v1.x, v1.y),
                    lengths_rot,
                    angles_turn,
                    center,
                )
                cw_points, ccw_points = _hex_points_from_edge_candidates(
                    (v0.x, v0.y),
                    (v1.x, v1.y),
                    lengths_rot,
                    angles_turn,
                )

                candidates = [
                    (ordered, cw_points),
                    (ordered, ccw_points),
                ]

                best_order = ordered
                best_points = preferred_points
                best_error = float("inf")
                for cand_order, cand_points in candidates:
                    error = 0.0
                    count = 0
                    for vid, (px, py) in zip(cand_order, cand_points):
                        v = current[vid]
                        if not v.has_position():
                            continue
                        error += (v.x - px) ** 2 + (v.y - py) ** 2
                        count += 1
                    if count == 0:
                        error = 0.0
                    if error < best_error - 1e-12:
                        best_error = error
                        best_order = cand_order
                        best_points = cand_points
                    elif abs(error - best_error) <= 1e-12:
                        if cand_points == preferred_points:
                            best_order = cand_order
                            best_points = cand_points

                for vid, (px, py) in zip(best_order, best_points):
                    if vid in inner_vertices:
                        continue
                    if vid not in ring1_targets:
                        ring1_targets[vid] = [0.0, 0.0]
                        ring1_counts[vid] = 0
                    ring1_targets[vid][0] += px
                    ring1_targets[vid][1] += py
                    ring1_counts[vid] += 1
            for vid, (sx, sy) in ring1_targets.items():
                count = ring1_counts[vid]
                if count <= 0:
                    continue
                current[vid] = Vertex(vid, sx / count, sy / count)
            # refine ring-1 vertex positions to satisfy shared edge lengths
            pent_lengths: list[float] = []
            for idx, vid in enumerate(ordered_pent):
                nxt = ordered_pent[(idx + 1) % len(ordered_pent)]
                v0 = current[vid]
                v1 = current[nxt]
                if v0.has_position() and v1.has_position():
                    pent_lengths.append(math.hypot(v1.x - v0.x, v1.y - v0.y))
            inner_edge_len = sum(pent_lengths) / len(pent_lengths) if pent_lengths else side_length
            current = _optimize_ring1_positions(
                grid,
                current,
                ring_faces_list,
                inner_vertices,
                inner_edge_len,
            )
            continue

        boundary_edges = _ordered_boundary_edges(grid, region_set, center)

        outer_lengths: list[float] = []
        for edge in boundary_edges:
            v0_id, v1_id = edge.vertex_ids
            if v0_id not in inner_vertices or v1_id not in inner_vertices:
                continue
            face_id = next((fid for fid in edge.face_ids if fid in ring_set), None)
            if face_id is None:
                continue
            face = grid.faces.get(face_id)
            if face is None:
                continue
            v0_inner = current[v0_id]
            v1_inner = current[v1_id]
            if v0_inner.x is None or v1_inner.x is None:
                continue
            inner_edge_len = math.hypot(v1_inner.x - v0_inner.x, v1_inner.y - v0_inner.y)
            lengths = solve_ring_hex_lengths(ring_idx, inner_edge_length=inner_edge_len)
            edge_lengths = [
                lengths["inner"],
                lengths["protrude"],
                lengths["outer"],
                lengths["outer"],
                lengths["outer"],
                lengths["protrude"],
            ]
            ordered = _face_vertex_cycle(face, grid.edges.values())
            if len(ordered) != 6:
                continue
            start_idx = _find_edge_start_index(ordered, v0_id, v1_id)
            if start_idx is None:
                continue
            ordered = _rotate_vertices(ordered, start_idx)
            lengths_rot = _rotate_list(edge_lengths, start_idx)
            angles_vertex = [
                spec.inner_angle_deg if vid in inner_vertices else spec.outer_angle_deg
                for vid in ordered
            ]
            angles_turn = _rotate_list(angles_vertex, 1)

            v0 = current[ordered[0]]
            v1 = current[ordered[1]]
            if v0.x is None or v1.x is None:
                continue
            hex_points = _hex_points_from_edge(
                (v0.x, v0.y),
                (v1.x, v1.y),
                lengths_rot,
                angles_turn,
                center,
            )

            for idx, vid in enumerate(ordered):
                if current[vid].has_position():
                    continue
                px, py = hex_points[idx]
                current[vid] = Vertex(vid, px, py)

            outer_lengths.append(lengths["outer"])

        if outer_lengths:
            prev_outer = sum(outer_lengths) / len(outer_lengths)

    return current


def _optimize_ring1_positions(
    grid: PolyGrid,
    vertices: Dict[str, Vertex],
    ring_faces_list: List[Face],
    inner_vertices: set[str],
    inner_edge_length: float,
    max_iter: int = 200,
) -> Dict[str, Vertex]:
    ring1_vertices = set(_collect_face_vertices(grid, [face.id for face in ring_faces_list]))
    ring1_vertices -= inner_vertices
    if not ring1_vertices:
        return vertices

    lengths = solve_ring_hex_lengths(1, inner_edge_length=inner_edge_length)
    initial_protrude = lengths["protrude"]
    initial_outer = lengths["outer"]

    ring1_faces = {face.id for face in ring_faces_list}
    edge_lookup = list(grid.edges.values())

    if least_squares is None:
        return vertices

    ring1_ids = sorted(ring1_vertices)
    index = {vid: i for i, vid in enumerate(ring1_ids)}

    x0 = []
    for vid in ring1_ids:
        v = vertices[vid]
        if v.has_position():
            x0.extend([v.x, v.y])
        else:
            x0.extend([0.0, 0.0])

    def get_pos(vid: str, params: List[float]) -> tuple[float, float]:
        if vid in index:
            idx = index[vid] * 2
            return params[idx], params[idx + 1]
        v = vertices[vid]
        return v.x, v.y

    def residuals(params: List[float]) -> List[float]:
        protrude = initial_protrude
        outer = initial_outer
        res: List[float] = []
        for edge in edge_lookup:
            a, b = edge.vertex_ids
            if a not in ring1_vertices and b not in ring1_vertices:
                continue
            if a in inner_vertices and b in inner_vertices:
                continue
            if a in inner_vertices or b in inner_vertices:
                target = protrude
            elif sum(1 for fid in edge.face_ids if fid in ring1_faces) > 1:
                target = protrude
            else:
                target = outer
            ax, ay = get_pos(a, params)
            bx, by = get_pos(b, params)
            res.append(math.hypot(bx - ax, by - ay) - target)
        return res

    result = least_squares(residuals, x0=x0, max_nfev=max_iter)
    params = result.x

    updated = dict(vertices)
    for vid in ring1_ids:
        idx = index[vid] * 2
        updated[vid] = Vertex(vid, float(params[idx]), float(params[idx + 1]))
    return updated


def _angle_first_ring_relax(
    grid: PolyGrid,
    vertices: Dict[str, Vertex],
    rings: int,
    iterations: int = 6,
    alpha: float = 0.25,
) -> Dict[str, Vertex]:
    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return vertices

    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings_map = ring_faces(adjacency, pent_face.id, max_depth=rings)
    if not rings_map:
        return vertices

    current = {vid: Vertex(vid, v.x, v.y) for vid, v in vertices.items()}
    neighbors: Dict[str, List[str]] = {vid: [] for vid in current}
    for edge in grid.edges.values():
        a, b = edge.vertex_ids
        neighbors[a].append(b)
        neighbors[b].append(a)

    for ring_idx in range(1, rings + 1):
        ring_face_ids = rings_map.get(ring_idx, [])
        if not ring_face_ids:
            continue
        ring_vertices = set(_collect_face_vertices(grid, ring_face_ids))
        inner_faces = []
        for depth in range(ring_idx):
            inner_faces.extend(rings_map.get(depth, []))
        fixed_vertices = set(_collect_face_vertices(grid, inner_faces))

        for _ in range(iterations):
            updates: Dict[str, Vertex] = {}
            for vid in ring_vertices:
                if vid in fixed_vertices:
                    continue
                nbrs = [n for n in neighbors.get(vid, []) if current[n].has_position()]
                if not nbrs:
                    continue
                mean_x = sum(current[n].x for n in nbrs) / len(nbrs)
                mean_y = sum(current[n].y for n in nbrs) / len(nbrs)
                v = current[vid]
                updates[vid] = Vertex(
                    vid,
                    v.x + alpha * (mean_x - v.x),
                    v.y + alpha * (mean_y - v.y),
                )
            current.update(updates)

    return current


def _order_ring_faces(
    vertices: Dict[str, Vertex],
    faces: List[Face],
    inner_vertices: set[str],
    center: tuple[float, float],
) -> List[Face]:
    scored: list[tuple[float, Face]] = []
    for face in faces:
        shared = [vid for vid in face.vertex_ids if vid in inner_vertices]
        if len(shared) != 2:
            continue
        v0 = vertices[shared[0]]
        v1 = vertices[shared[1]]
        if v0.x is None or v1.x is None:
            continue
        mx = (v0.x + v1.x) / 2
        my = (v0.y + v1.y) / 2
        angle = math.atan2(my - center[1], mx - center[0])
        scored.append((angle, face))
    scored.sort(key=lambda item: item[0])
    return [face for _, face in scored]


def _find_start_edge_index(
    ordered: List[str],
    vertices: Dict[str, Vertex],
    inner_vertices: set[str],
) -> int | None:
    n = len(ordered)
    preferred = None
    candidate = None
    for i in range(n):
        a = ordered[i]
        b = ordered[(i + 1) % n]
        va = vertices[a]
        vb = vertices[b]
        if not va.has_position() or not vb.has_position():
            continue
        if a in inner_vertices and b in inner_vertices:
            preferred = preferred if preferred is not None else i
        candidate = candidate if candidate is not None else i

    if preferred is not None:
        return preferred
    return candidate


def _rotate_list(values: List[float], start_idx: int) -> List[float]:
    return values[start_idx:] + values[:start_idx]


def _find_edge_start_index(ordered: List[str], v0: str, v1: str) -> int | None:
    n = len(ordered)
    for i in range(n):
        if ordered[i] == v0 and ordered[(i + 1) % n] == v1:
            return i
        if ordered[i] == v1 and ordered[(i + 1) % n] == v0:
            return i
    return None


def _ordered_boundary_edges(
    grid: PolyGrid,
    region_faces: set[str],
    center: tuple[float, float],
) -> List[Edge]:
    boundary: list[tuple[float, Edge]] = []
    for edge in grid.edges.values():
        count = sum(1 for fid in edge.face_ids if fid in region_faces)
        if count == 1:
            v0 = grid.vertices[edge.vertex_ids[0]]
            v1 = grid.vertices[edge.vertex_ids[1]]
            if v0.x is None or v1.x is None:
                continue
            mx = (v0.x + v1.x) / 2
            my = (v0.y + v1.y) / 2
            angle = math.atan2(my - center[1], mx - center[0])
            boundary.append((angle, edge))
    boundary.sort(key=lambda item: item[0])
    return [edge for _, edge in boundary]


def _find_face_by_edge(faces: List[Face], v0: str, v1: str) -> Face | None:
    for face in faces:
        if v0 in face.vertex_ids and v1 in face.vertex_ids:
            return face
    return None


def _snap_ring1_to_pent_normals(
    vertices: Dict[str, Vertex],
    ordered_pent: List[str],
    ring1_faces: List[Face],
) -> None:
    pent_points = [vertices[vid] for vid in ordered_pent]
    if not all(v.has_position() for v in pent_points):
        return
    cx = sum(v.x for v in pent_points) / len(pent_points)
    cy = sum(v.y for v in pent_points) / len(pent_points)

    targets: Dict[str, list[float]] = {}
    counts: Dict[str, int] = {}

    for idx in range(len(ordered_pent)):
        v0 = vertices[ordered_pent[idx]]
        v1 = vertices[ordered_pent[(idx + 1) % len(ordered_pent)]]
        face = _find_face_by_edge(ring1_faces, v0.id, v1.id)
        if face is None:
            continue
        mx = (v0.x + v1.x) / 2
        my = (v0.y + v1.y) / 2
        nx = mx - cx
        ny = my - cy
        norm = math.hypot(nx, ny) or 1.0
        nx /= norm
        ny /= norm

        # Find ring-1 vertices for this face and snap the two most outward to the
        # shared normal so they stay symmetric about the edge.
        candidates: List[tuple[str, Vertex, float]] = []
        for vid in face.vertex_ids:
            if vid in ordered_pent:
                continue
            v = vertices[vid]
            if not v.has_position():
                continue
            dx = v.x - mx
            dy = v.y - my
            proj = dx * nx + dy * ny
            candidates.append((vid, v, proj))
        # sort by projection distance (furthest outward first)
        candidates.sort(key=lambda tup: tup[2], reverse=True)
        if len(candidates) < 2:
            continue
        (id_a, v_a, proj_a), (id_b, v_b, proj_b) = candidates[0], candidates[1]
        avg_proj = (proj_a + proj_b) / 2
        new_x = mx + nx * avg_proj
        new_y = my + ny * avg_proj
        for vid in (id_a, id_b):
            if vid not in targets:
                targets[vid] = [0.0, 0.0]
                counts[vid] = 0
            targets[vid][0] += new_x
            targets[vid][1] += new_y
            counts[vid] += 1

    for vid, (sx, sy) in targets.items():
        avg_x = sx / counts[vid]
        avg_y = sy / counts[vid]
        vertices[vid] = Vertex(vid, avg_x, avg_y)


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


def _rotate_vertices(vertices: List[str], start_idx: int) -> List[str]:
    return vertices[start_idx:] + vertices[:start_idx]


def _hex_points_from_edge_candidates(
    p0: tuple[float, float],
    p1: tuple[float, float],
    lengths: List[float],
    angles: List[float],
) -> tuple[List[tuple[float, float]], List[tuple[float, float]]]:
    def build(sign: float) -> List[tuple[float, float]]:
        pts = [p0, p1]
        angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        current_angle = angle
        # Build the hex by turning at each subsequent vertex. The angle at v1
        # corresponds to angles[1], so we use angles[idx] for edge idx.
        for idx in range(1, 6):
            turn = math.radians(180.0 - angles[idx - 1]) * sign
            current_angle += turn
            length = lengths[idx]
            last = pts[-1]
            pts.append(
                (
                    last[0] + math.cos(current_angle) * length,
                    last[1] + math.sin(current_angle) * length,
                )
            )
        # Drop the closure point if it returns to the start.
        if len(pts) == 7 and math.hypot(pts[-1][0] - p0[0], pts[-1][1] - p0[1]) < 5e-3:
            pts = pts[:-1]
        return pts

    return build(1.0), build(-1.0)


def _hex_points_from_edge(
    p0: tuple[float, float],
    p1: tuple[float, float],
    lengths: List[float],
    angles: List[float],
    center: tuple[float, float],
) -> List[tuple[float, float]]:
    cw, ccw = _hex_points_from_edge_candidates(p0, p1, lengths, angles)

    mx = (p0[0] + p1[0]) / 2
    my = (p0[1] + p1[1]) / 2
    out_x = mx - center[0]
    out_y = my - center[1]
    out_len = math.hypot(out_x, out_y) or 1.0
    out_x /= out_len
    out_y /= out_len

    cw_center = _points_center(cw)
    ccw_center = _points_center(ccw)
    cw_dot = (cw_center[0] - mx) * out_x + (cw_center[1] - my) * out_y
    ccw_dot = (ccw_center[0] - mx) * out_x + (ccw_center[1] - my) * out_y
    if cw_dot >= ccw_dot:
        return cw
    return ccw


def _points_center(points: List[tuple[float, float]]) -> tuple[float, float]:
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


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

            # Inside angles at outer vertices (adjacent to inner ring)
            angle_constraints: list[tuple[str, str, str]] = []
            angle_values: list[float] = []
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
                        angle = _angle_at_vertex(current, inner, vid, outer)
                        angle_constraints.append((vid, inner, outer))
                        angle_values.append(angle)

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

            # Pointy outside angles (outermost vertex per face)
            pointy_constraints: list[tuple[str, str, str]] = []
            pointy_angles: list[float] = []
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
                angle = _angle_at_vertex(current, prev_vid, pointy, next_vid)
                pointy_constraints.append((pointy, prev_vid, next_vid))
                pointy_angles.append(angle)

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

            # Protruding edges (inner -> ring) enforced last
            protruding_map: Dict[str, list[str]] = {}
            protruding_lengths: list[float] = []
            for edge in grid.edges.values():
                a, b = edge.vertex_ids
                if (a in ring_vertices and b in inner_vertices) or (b in ring_vertices and a in inner_vertices):
                    inner = a if a in inner_vertices else b
                    outer = b if inner == a else a
                    protruding_map.setdefault(outer, []).append(inner)
                    protruding_lengths.append(_edge_length(current, a, b))

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
