from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .models import Edge, Face, Vertex
from .polygrid import PolyGrid
from .embedding import tutte_embedding
from .angle_solver import ring_angle_spec, solve_ring_hex_lengths, solve_ring_hex_outer_length
from .algorithms import build_face_adjacency, ring_faces
from .geometry import _face_vertex_cycle, _ordered_face_vertices
from .optimisation import optimise_positions_to_edge_targets, optimize_outer_rings_constrained
from .grid_utils import (
    _angle_at_vertex,
    _boundary_vertex_cycle,
    _boundary_vertex_ids,
    _circle_intersections,
    _collect_face_vertices,
    _edge_length,
    _face_center,
    _find_pentagon_face,
    _grid_center,
    _mean_value,
)
try:  # pragma: no cover - optional dependency
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover
    least_squares = None


@dataclass(frozen=True)
class AxialCoord:
    q: int
    r: int


def hex_face_count(rings: int) -> int:
    if rings < 0:
        raise ValueError("rings must be >= 0")
    return 1 + 3 * rings * (rings + 1)


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
        face_id = f"f{idx}"
        edge_ids = _get_edge_ids(edge_map, vertex_ids, face_id)
        faces.append(
            Face(
                id=face_id,
                face_type="hex",
                vertex_ids=tuple(vertex_ids),
                edge_ids=tuple(edge_ids),
            )
        )

    return PolyGrid(vertex_map.values(), edge_map.values(), faces)


def build_pentagon_centered_grid(
    rings: int,
    size: float = 1.0,
    embed: bool = True,
    embed_mode: str = "tutte+optimise",
    validate_topology: bool = False,
) -> PolyGrid:
    if rings < 0:
        raise ValueError("rings must be >= 0")

    # Use the new Goldberg topology pipeline (correct dualized triangulation)
    from .goldberg_topology import build_goldberg_grid

    optimise = embed and (embed_mode in ("tutte+optimise",))
    grid = build_goldberg_grid(rings, size=size, optimise=optimise)

    if validate_topology:
        errors = validate_pentagon_topology(grid, rings=rings)
        if errors:
            raise RuntimeError("Pentagon grid topology errors: " + "; ".join(errors))

    return grid


def _build_single_pentagon(size: float) -> PolyGrid:
    radius = size / (2 * math.sin(math.pi / 5))
    vertices: Dict[str, Vertex] = {}
    pent_ids: List[str] = []
    for idx in range(5):
        angle = 2 * math.pi * idx / 5
        vid = f"v{len(vertices) + 1}"
        pent_ids.append(vid)
        vertices[vid] = Vertex(vid, radius * math.cos(angle), radius * math.sin(angle))
    faces = [Face(id="f1", face_type="pent", vertex_ids=tuple(pent_ids))]
    edges, faces = _rebuild_edges_from_faces(faces)
    return PolyGrid(vertices.values(), edges, faces)


def _build_ring1_grid(size: float) -> PolyGrid:
    """Build pentagon + ring-1 hexes from explicit geometry."""
    pent_grid = _build_single_pentagon(size)
    pent_face = next(face for face in pent_grid.faces.values() if face.face_type == "pent")
    ordered_pent = _face_vertex_cycle(pent_face, pent_grid.edges.values())
    if len(ordered_pent) != 5:
        ordered_pent = _ordered_face_vertices(pent_grid.vertices, pent_face)

    center = (
        sum(pent_grid.vertices[vid].x for vid in ordered_pent) / 5,
        sum(pent_grid.vertices[vid].y for vid in ordered_pent) / 5,
    )

    edge_lengths = [
        math.hypot(
            pent_grid.vertices[a].x - pent_grid.vertices[b].x,
            pent_grid.vertices[a].y - pent_grid.vertices[b].y,
        )
        for a, b in zip(ordered_pent, ordered_pent[1:] + ordered_pent[:1])
    ]
    inner_edge_len = sum(edge_lengths) / len(edge_lengths) if edge_lengths else size
    lengths = solve_ring_hex_lengths(1, inner_edge_length=inner_edge_len)
    lengths_rot = [
        lengths["inner"],
        lengths["protrude"],
        lengths["outer"],
        lengths["outer"],
        lengths["outer"],
        lengths["protrude"],
    ]

    spec = ring_angle_spec(1)
    angles_vertex = [
        spec.inner_angle_deg,
        spec.inner_angle_deg,
        spec.outer_angle_deg,
        spec.outer_angle_deg,
        spec.outer_angle_deg,
        spec.outer_angle_deg,
    ]
    angles_turn = _rotate_list(angles_vertex, 1)

    vertex_map: Dict[str, Vertex] = {}
    faces: List[Face] = []
    # seed pentagon vertices
    for vid in ordered_pent:
        v = pent_grid.vertices[vid]
        vertex_map[vid] = Vertex(vid, v.x, v.y)

    for idx in range(5):
        v0_id = ordered_pent[idx]
        v1_id = ordered_pent[(idx + 1) % 5]
        v0 = vertex_map[v0_id]
        v1 = vertex_map[v1_id]
        candidates = _hex_points_from_edge_candidates(
            (v0.x, v0.y),
            (v1.x, v1.y),
            lengths_rot,
            angles_turn,
        )
        def score(points):
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            return math.hypot(cx - center[0], cy - center[1])

        points = max(candidates, key=score)
        hex_vertex_ids: List[str] = [v0_id, v1_id]
        for px, py in points[2:]:
            vid = _get_vertex_id(vertex_map, (px, py))
            hex_vertex_ids.append(vid)
        faces.append(
            Face(
                id=f"f{len(faces) + 2}",
                face_type="hex",
                vertex_ids=tuple(hex_vertex_ids),
            )
        )

    # add pent face last so edges rebuild correctly
    faces.append(
        Face(
            id="f1",
            face_type="pent",
            vertex_ids=tuple(ordered_pent),
        )
    )

    edges, faces = _rebuild_edges_from_faces(faces)
    return PolyGrid(vertex_map.values(), edges, faces)


def _add_hex_ring_geometry(
    vertex_map: Dict[str, Vertex],
    position_map: Dict[str, Vertex],
    faces: List[Face],
    ring_idx: int,
) -> None:
    """Append a new ring of hexes using boundary edges as inner edges."""
    edges, rebuilt_faces = _rebuild_edges_from_faces(faces)
    grid = PolyGrid(vertex_map.values(), edges, rebuilt_faces)
    region_faces = {face.id for face in faces}
    center = _grid_center(grid)
    boundary_edges = _ordered_boundary_edges(grid, region_faces, center)
    if not boundary_edges:
        return

    edge_lengths: List[Dict[str, float]] = []
    for v0_id, v1_id in boundary_edges:
        v0 = vertex_map[v0_id]
        v1 = vertex_map[v1_id]
        if v0.x is None or v1.x is None:
            edge_lengths.append({"inner": 0.0, "protrude": 0.0, "outer": 0.0})
            continue
        inner_edge_len = math.hypot(v1.x - v0.x, v1.y - v0.y)
        edge_lengths.append(solve_ring_hex_lengths(ring_idx, inner_edge_length=inner_edge_len))

    spec = ring_angle_spec(ring_idx)
    angles_vertex = [
        spec.inner_angle_deg,
        spec.inner_angle_deg,
        spec.outer_angle_deg,
        spec.outer_angle_deg,
        spec.outer_angle_deg,
        spec.outer_angle_deg,
    ]
    angles_turn = _rotate_list(angles_vertex, 1)

    hex_points: List[List[tuple[float, float]]] = []

    for edge_idx, (v0_id, v1_id) in enumerate(boundary_edges):
        v0 = vertex_map[v0_id]
        v1 = vertex_map[v1_id]
        if v0.x is None or v1.x is None:
            continue
        lengths = edge_lengths[edge_idx]
        lengths_rot = [
            lengths["inner"],
            lengths["protrude"],
            lengths["outer"],
            lengths["outer"],
            lengths["outer"],
            lengths["protrude"],
        ]

        candidates = _hex_points_from_edge_candidates(
            (v0.x, v0.y),
            (v1.x, v1.y),
            lengths_rot,
            angles_turn,
        )

        def score(points: List[tuple[float, float]]) -> float:
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            return math.hypot(cx - center[0], cy - center[1])

        points = max(candidates, key=score)
        hex_points.append(list(points))

    for (v0_id, v1_id), points in zip(boundary_edges, hex_points):
        hex_vertex_ids: List[str] = [v0_id, v1_id]
        for px, py in points[2:]:
            vid = _get_or_create_vertex_id(vertex_map, position_map, (px, py))
            hex_vertex_ids.append(vid)
        faces.append(
            Face(
                id=f"f{len(faces) + 1}",
                face_type="hex",
                vertex_ids=tuple(hex_vertex_ids),
            )
        )


def _apply_outer_ring_geometry(
    grid: PolyGrid,
    rings: int,
) -> Dict[str, Vertex]:
    """Compute vertex positions for rings>=2 while keeping existing positions fixed."""
    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return grid.vertices

    current: Dict[str, Vertex] = {vid: Vertex(vid, v.x, v.y) for vid, v in grid.vertices.items()}
    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings_map = ring_faces(adjacency, pent_face.id, max_depth=rings)
    center = _grid_center(PolyGrid(current.values(), grid.edges.values(), grid.faces.values()))

    for ring_idx in range(2, rings + 1):
        ring_face_ids = rings_map.get(ring_idx, [])
        if not ring_face_ids:
            continue
        ring_faces_list = [grid.faces[fid] for fid in ring_face_ids if fid in grid.faces]
        region_faces: list[str] = []
        for depth in range(0, ring_idx):
            region_faces.extend(rings_map.get(depth, []))
        inner_vertices = set(_collect_face_vertices(grid, region_faces))

        spec = ring_angle_spec(ring_idx)
        boundary_edges = _ordered_boundary_edges(grid, set(region_faces), center)
        if not boundary_edges:
            continue

        inner_edge_lengths = []
        for v0_id, v1_id in boundary_edges:
            v0 = current[v0_id]
            v1 = current[v1_id]
            if v0.has_position() and v1.has_position():
                inner_edge_lengths.append(math.hypot(v1.x - v0.x, v1.y - v0.y))
        inner_edge_len = sum(inner_edge_lengths) / len(inner_edge_lengths) if inner_edge_lengths else 1.0
        lengths = solve_ring_hex_lengths(ring_idx, inner_edge_length=inner_edge_len)
        lengths_rot = [
            lengths["inner"],
            lengths["protrude"],
            lengths["outer"],
            lengths["outer"],
            lengths["outer"],
            lengths["protrude"],
        ]

        protrude_map: Dict[str, tuple[float, float]] = {}
        first_outer_point: tuple[float, float] | None = None
        last_outer_point: tuple[float, float] | None = None

        for idx, (v0_id, v1_id) in enumerate(boundary_edges):
            v0 = current[v0_id]
            v1 = current[v1_id]
            if v0.x is None or v1.x is None:
                continue
            face = next(
                (
                    f
                    for f in ring_faces_list
                    if v0_id in f.vertex_ids and v1_id in f.vertex_ids
                ),
                None,
            )
            if face is None:
                continue
            ordered = _face_vertex_cycle(face, grid.edges.values())
            if len(ordered) != 6:
                ordered = _ordered_face_vertices(current, face)
            start_idx = _find_edge_start_index(ordered, v0_id, v1_id)
            if start_idx is None:
                continue
            ordered = _rotate_vertices(ordered, start_idx)

            angles_vertex = [
                spec.inner_angle_deg,
                spec.inner_angle_deg,
                spec.outer_angle_deg,
                spec.outer_angle_deg,
                spec.outer_angle_deg,
                spec.outer_angle_deg,
            ]
            angles_turn = _rotate_list(angles_vertex, 1)
            hex_points = _hex_points_from_edge(
                (v0.x, v0.y),
                (v1.x, v1.y),
                lengths_rot,
                angles_turn,
                center,
            )

            points = list(hex_points)
            if v0_id in protrude_map:
                points[5] = protrude_map[v0_id]
            if v1_id in protrude_map:
                points[2] = protrude_map[v1_id]
            if last_outer_point is not None:
                points[3] = last_outer_point
            if idx == 0:
                first_outer_point = points[3]
            if idx == len(boundary_edges) - 1 and first_outer_point is not None:
                points[4] = first_outer_point

            protrude_map[v0_id] = points[5]
            protrude_map[v1_id] = points[2]
            last_outer_point = points[4]

            for vid, (px, py) in zip(ordered, points):
                if vid in inner_vertices:
                    continue
                current[vid] = Vertex(vid, px, py)

    return current


def _scale_vertices_outward(
    vertex_map: Dict[str, Vertex],
    vertex_ids: Iterable[str],
    center: tuple[float, float],
    factor: float,
) -> None:
    cx, cy = center
    for vid in vertex_ids:
        vertex = vertex_map.get(vid)
        if not vertex or not vertex.has_position():
            continue
        nx = cx + (vertex.x - cx) * factor
        ny = cy + (vertex.y - cy) * factor
        vertex_map[vid] = Vertex(vid, nx, ny)


def _get_or_create_vertex_id(
    vertex_map: Dict[str, Vertex],
    position_map: Dict[str, Vertex],
    position: Tuple[float, float],
    tolerance: float = 1e-2,
) -> str:
    for existing in position_map.values():
        if existing.x is None or existing.y is None:
            continue
        if math.hypot(existing.x - position[0], existing.y - position[1]) < tolerance:
            return existing.id
    key = _vertex_key(position)
    existing = position_map.get(key)
    if existing is not None:
        return existing.id
    vid = f"v{len(vertex_map) + 1}"
    vertex = Vertex(vid, position[0], position[1])
    vertex_map[vid] = vertex
    position_map[key] = vertex
    return vid


def _add_hex_ring(
    vertices: Dict[str, Vertex],
    faces: List[Face],
    next_vertex_id: int,
    next_face_id: int,
) -> tuple[int, int]:
    edges, rebuilt_faces = _rebuild_edges_from_faces(faces)
    grid = PolyGrid(vertices.values(), edges, rebuilt_faces)
    boundary = _boundary_vertex_cycle(grid)
    if not boundary:
        return next_vertex_id, next_face_id

    inner_map: Dict[str, str] = {}
    outer_map: Dict[str, str] = {}
    for vid in boundary:
        inner_id = f"v{next_vertex_id}"
        next_vertex_id += 1
        outer_id = f"v{next_vertex_id}"
        next_vertex_id += 1
        inner_map[vid] = inner_id
        outer_map[vid] = outer_id
        vertices[inner_id] = Vertex(inner_id)
        vertices[outer_id] = Vertex(outer_id)

    ring_size = len(boundary)
    for idx, vid in enumerate(boundary):
        nxt = boundary[(idx + 1) % ring_size]
        face_vertices = [
            vid,
            nxt,
            inner_map[nxt],
            outer_map[nxt],
            outer_map[vid],
            inner_map[vid],
        ]
        faces.append(
            Face(
                id=f"f{next_face_id}",
                face_type="hex",
                vertex_ids=tuple(face_vertices),
            )
        )
        next_face_id += 1

    return next_vertex_id, next_face_id


def validate_pentagon_topology(grid: PolyGrid, rings: int) -> list[str]:
    errors: list[str] = []
    if rings < 0:
        errors.append("rings must be >= 0")
        return errors

    pent_faces = [face for face in grid.faces.values() if face.face_type == "pent"]
    if len(pent_faces) != 1:
        errors.append("expected exactly one pentagon face")
        return errors
    if len(pent_faces[0].vertex_ids) != 5:
        errors.append("pentagon face must have 5 vertices")

    if any(len(face.vertex_ids) not in (5, 6) for face in grid.faces.values()):
        errors.append("all faces must be pentagons or hexagons")

    if rings >= 1:
        adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
        rings_map = ring_faces(adjacency, pent_faces[0].id, max_depth=rings)
        if not rings_map.get(1):
            errors.append("expected at least one ring-1 face")

    return errors


def _build_fixed_positions(
    grid: PolyGrid,
    force_boundary: bool = False,
) -> Dict[str, tuple[float, float]]:
    fixed: Dict[str, tuple[float, float]] = {}
    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return fixed

    pent_vertices = list(pent_face.vertex_ids)
    coords = [grid.vertices[vid] for vid in pent_vertices]
    if coords and all(v.has_position() for v in coords):
        center = (
            sum(v.x for v in coords if v.x is not None) / len(coords),
            sum(v.y for v in coords if v.y is not None) / len(coords),
        )
        radius = _mean_radius(grid, pent_vertices, center)
    else:
        center = (0.0, 0.0)
        radius = 1.0

    fixed.update(_regular_pentagon_positions(grid, pent_vertices, radius, center))

    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings_map = ring_faces(adjacency, pent_face.id, max_depth=1)
    ring1_vertices = set(_collect_face_vertices(grid, rings_map.get(1, [])))
    if ring1_vertices:
        for vid in ring1_vertices:
            vertex = grid.vertices.get(vid)
            if vertex and vertex.has_position():
                fixed[vid] = (vertex.x, vertex.y)

    boundary = _boundary_vertex_cycle(grid)
    if boundary and (force_boundary or not ring1_vertices):
        boundary_vertices = [grid.vertices[vid] for vid in boundary]
        if all(v.has_position() for v in boundary_vertices):
            boundary_radius = max(
                math.hypot(v.x - center[0], v.y - center[1]) for v in boundary_vertices
            )
            base_angle = math.atan2(
                boundary_vertices[0].y - center[1],
                boundary_vertices[0].x - center[0],
            )
        else:
            boundary_radius = radius * 3.0
            base_angle = 0.0
        step = 2 * math.pi / len(boundary)
        for idx, vid in enumerate(boundary):
            angle = base_angle + idx * step
            fixed[vid] = (
                center[0] + boundary_radius * math.cos(angle),
                center[1] + boundary_radius * math.sin(angle),
            )
    return fixed


def _hex_area(rings: int) -> List[AxialCoord]:
    coords: List[AxialCoord] = []
    for q in range(-rings, rings + 1):
        for r in range(-rings, rings + 1):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) <= rings:
                coords.append(AxialCoord(q=q, r=r))
    return coords


def _axial_to_pixel(coord: AxialCoord, size: float) -> tuple[float, float]:
    # Pointy-top axial layout (matches _hex_corners orientation).
    x = size * (math.sqrt(3) * (coord.q + coord.r / 2))
    y = size * (3 / 2 * coord.r)
    return (x, y)


def _hex_corners(center: tuple[float, float], size: float) -> List[tuple[float, float]]:
    cx, cy = center
    corners = []
    for i in range(6):
        angle = math.radians(60 * i - 30)
        corners.append((cx + size * math.cos(angle), cy + size * math.sin(angle)))
    return corners


def _add_hex_ring(
    vertices: Dict[str, Vertex],
    faces: List[Face],
    next_vertex_id: int,
    next_face_id: int,
) -> tuple[int, int]:
    edges, rebuilt_faces = _rebuild_edges_from_faces(faces)
    grid = PolyGrid(vertices.values(), edges, rebuilt_faces)
    boundary = _boundary_vertex_cycle(grid)
    if not boundary:
        return next_vertex_id, next_face_id

    inner_map: Dict[str, str] = {}
    outer_map: Dict[str, str] = {}
    for vid in boundary:
        inner_id = f"v{next_vertex_id}"
        next_vertex_id += 1
        outer_id = f"v{next_vertex_id}"
        next_vertex_id += 1
        inner_map[vid] = inner_id
        outer_map[vid] = outer_id
        vertices[inner_id] = Vertex(inner_id)
        vertices[outer_id] = Vertex(outer_id)

    ring_size = len(boundary)
    for idx, vid in enumerate(boundary):
        nxt = boundary[(idx + 1) % ring_size]
        face_vertices = [
            vid,
            nxt,
            inner_map[nxt],
            outer_map[nxt],
            outer_map[vid],
            inner_map[vid],
        ]
        faces.append(
            Face(
                id=f"f{next_face_id}",
                face_type="hex",
                vertex_ids=tuple(face_vertices),
            )
        )
        next_face_id += 1

    return next_vertex_id, next_face_id


def _build_triangular_vertices(
    rings: int,
    size: float,
) -> tuple[Dict[str, Vertex], Dict[tuple[int, int], str]]:
    vertices: Dict[str, Vertex] = {}
    coords: Dict[tuple[int, int], str] = {}
    for q in range(-rings, rings + 1):
        for r in range(-rings, rings + 1):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) > rings:
                continue
            x = size * (q + r / 2)
            y = size * (math.sqrt(3) / 2 * r)
            vid = f"tv{len(vertices) + 1}"
            vertices[vid] = Vertex(vid, x, y)
            coords[(q, r)] = vid
    return vertices, coords


def _build_triangles(coords: Dict[tuple[int, int], str]) -> List[tuple[str, str, str]]:
    triangles: List[tuple[str, str, str]] = []
    for (q, r), vid in coords.items():
        up = [(q, r), (q + 1, r), (q, r + 1)]
        if all(pt in coords for pt in up):
            triangles.append(tuple(coords[pt] for pt in up))
        down = [(q, r), (q + 1, r - 1), (q + 1, r)]
        if all(pt in coords for pt in down):
            triangles.append(tuple(coords[pt] for pt in down))
    return triangles


def _remove_wedge(
    vertices: Dict[str, Vertex],
    triangles: List[tuple[str, str, str]],
    angle_min: float,
    angle_max: float,
    angle_tol: float,
) -> List[tuple[str, str, str]]:
    kept: List[tuple[str, str, str]] = []
    for tri in triangles:
        pts = [vertices[vid] for vid in tri]
        cx = sum(v.x for v in pts) / 3
        cy = sum(v.y for v in pts) / 3
        angle = math.atan2(cy, cx)
        if angle < 0:
            angle += 2 * math.pi
        if angle_min + angle_tol <= angle <= angle_max - angle_tol:
            continue
        kept.append(tri)
    return kept


def _build_ray_merge_map(
    vertices: Dict[str, Vertex],
    angle_min: float,
    angle_max: float,
    angle_tol: float,
) -> Dict[str, str]:
    ray_min: List[tuple[float, str]] = []
    ray_max: List[tuple[float, str]] = []
    for vid, vertex in vertices.items():
        if not vertex.has_position():
            continue
        radius = math.hypot(vertex.x, vertex.y)
        if radius < 1e-6:
            continue
        angle = math.atan2(vertex.y, vertex.x)
        if angle < 0:
            angle += 2 * math.pi
        if abs(angle - angle_min) <= angle_tol:
            ray_min.append((radius, vid))
        elif abs(angle - angle_max) <= angle_tol:
            ray_max.append((radius, vid))

    ray_min.sort(key=lambda item: item[0])
    ray_max.sort(key=lambda item: item[0])

    merge: Dict[str, str] = {}
    for (_, target), (_, source) in zip(ray_min, ray_max):
        merge[source] = target
    return merge


def _merge_vertices(
    vertices: Dict[str, Vertex],
    merge: Dict[str, str],
) -> Dict[str, Vertex]:
    updated: Dict[str, Vertex] = {vid: Vertex(vid, v.x, v.y) for vid, v in vertices.items()}
    for source, target in merge.items():
        if source not in updated or target not in updated:
            continue
        src = updated[source]
        tgt = updated[target]
        if src.has_position() and tgt.has_position():
            merged = Vertex(target, (src.x + tgt.x) / 2, (src.y + tgt.y) / 2)
        elif src.has_position():
            merged = Vertex(target, src.x, src.y)
        else:
            merged = tgt
        updated[target] = merged
        updated.pop(source, None)
    return updated


def _merge_triangles(
    triangles: List[tuple[str, str, str]],
    merge: Dict[str, str],
) -> List[tuple[str, str, str]]:
    updated: List[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for tri in triangles:
        mapped = tuple(merge.get(vid, vid) for vid in tri)
        if len(set(mapped)) < 3:
            continue
        key = tuple(sorted(mapped))
        if key in seen:
            continue
        seen.add(key)
        updated.append(mapped)
    return updated


def _dualize(
    vertices: Dict[str, Vertex],
    triangles: List[tuple[str, str, str]],
) -> tuple[Dict[str, Vertex], List[Face]]:
    dual_vertices: Dict[str, Vertex] = {}
    incident: Dict[str, List[str]] = {}

    for idx, tri in enumerate(triangles, start=1):
        tri_id = f"t{idx}"
        pts = [vertices[vid] for vid in tri]
        cx = sum(v.x for v in pts) / 3
        cy = sum(v.y for v in pts) / 3
        dual_vertices[tri_id] = Vertex(tri_id, cx, cy)
        for vid in tri:
            incident.setdefault(vid, []).append(tri_id)

    faces: List[Face] = []
    for vid, tri_ids in incident.items():
        if len(tri_ids) < 5:
            continue
        v = vertices[vid]
        ordered = sorted(
            tri_ids,
            key=lambda tid: math.atan2(
                dual_vertices[tid].y - v.y,
                dual_vertices[tid].x - v.x,
            ),
        )
        face_type = "pent" if len(ordered) == 5 else "hex"
        faces.append(
            Face(
                id=f"f{len(faces) + 1}",
                face_type=face_type,
                vertex_ids=tuple(ordered),
            )
        )

    return dual_vertices, faces


def _crop_to_rings(grid: PolyGrid, rings: int) -> PolyGrid:
    if rings <= 0:
        return grid
    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return grid
    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings_map = ring_faces(adjacency, pent_face.id, max_depth=rings)
    kept_faces = [grid.faces[fid] for r in range(rings + 1) for fid in rings_map.get(r, [])]
    if not kept_faces:
        return grid
    used_vertices = set()
    for face in kept_faces:
        used_vertices.update(face.vertex_ids)
    vertex_map = {vid: grid.vertices[vid] for vid in used_vertices if vid in grid.vertices}
    edges, faces = _rebuild_edges_from_faces(kept_faces)
    return PolyGrid(vertex_map.values(), edges, faces)


def _build_cone_dual_grid(rings: int, size: float) -> PolyGrid:
    """Build a pentagon-centered grid using wedge removal + dualization."""
    if rings <= 0:
        return _build_single_pentagon(size)
    return _build_cone_dual_grid_with_offset(rings, size, 0.0)


def _snap_cone_ring1_geometry(grid: PolyGrid, size: float) -> PolyGrid:
    """Replace ring-0/1 positions with the explicit pentagon+ring1 geometry."""
    ref = _build_ring1_grid(size)
    ref_pent = _find_pentagon_face(ref)
    cone_pent = _find_pentagon_face(grid)
    if ref_pent is None or cone_pent is None:
        return grid

    ref_center = _grid_center(ref)
    cone_center = _grid_center(grid)

    def ordered_pent_vertices(pent_face: Face, g: PolyGrid, center: tuple[float, float]) -> List[str]:
        ordered = _face_vertex_cycle(pent_face, g.edges.values())
        if len(ordered) != 5:
            ordered = _ordered_face_vertices(g.vertices, pent_face)
        return sorted(
            ordered,
            key=lambda vid: math.atan2(g.vertices[vid].y - center[1], g.vertices[vid].x - center[0]),
        )

    ref_pent_order = ordered_pent_vertices(ref_pent, ref, ref_center)
    cone_pent_order = ordered_pent_vertices(cone_pent, grid, cone_center)

    pent_map = {cone_vid: ref_vid for cone_vid, ref_vid in zip(cone_pent_order, ref_pent_order)}

    ref_adj = build_face_adjacency(ref.faces.values(), ref.edges.values())
    ref_ring1 = ref_adj.get(ref_pent.id, [])
    ref_ring1_faces = [ref.faces[fid] for fid in ref_ring1 if fid in ref.faces]

    cone_adj = build_face_adjacency(grid.faces.values(), grid.edges.values())
    cone_ring1 = cone_adj.get(cone_pent.id, [])
    cone_ring1_faces = [grid.faces[fid] for fid in cone_ring1 if fid in grid.faces]

    # Map each ring1 face by the pentagon edge it shares.
    ref_face_by_edge: Dict[Tuple[str, str], Face] = {}
    for face in ref_ring1_faces:
        shared = [vid for vid in face.vertex_ids if vid in ref_pent.vertex_ids]
        if len(shared) != 2:
            continue
        a, b = shared
        ref_face_by_edge[tuple(sorted((a, b)))] = face

    updated = {vid: Vertex(vid, v.x, v.y) for vid, v in grid.vertices.items()}

    for face in cone_ring1_faces:
        shared = [vid for vid in face.vertex_ids if vid in cone_pent.vertex_ids]
        if len(shared) != 2:
            continue
        a_cone, b_cone = shared
        a_ref = pent_map.get(a_cone)
        b_ref = pent_map.get(b_cone)
        if a_ref is None or b_ref is None:
            continue
        ref_face = ref_face_by_edge.get(tuple(sorted((a_ref, b_ref))))
        if ref_face is None:
            continue

        cone_order = _face_vertex_cycle(face, grid.edges.values())
        if len(cone_order) != len(face.vertex_ids):
            cone_order = _ordered_face_vertices(grid.vertices, face)
        ref_order = _face_vertex_cycle(ref_face, ref.edges.values())
        if len(ref_order) != len(ref_face.vertex_ids):
            ref_order = _ordered_face_vertices(ref.vertices, ref_face)

        if not cone_order or not ref_order:
            continue

        def rotate_to_edge(order: List[str], a: str, b: str) -> List[str]:
            if a not in order or b not in order:
                return order
            idx = order.index(a)
            rotated = order[idx:] + order[:idx]
            if len(rotated) > 1 and rotated[1] != b:
                rotated = [rotated[0]] + list(reversed(rotated[1:]))
            return rotated

        cone_order = rotate_to_edge(cone_order, a_cone, b_cone)
        ref_order = rotate_to_edge(ref_order, a_ref, b_ref)

        for cone_vid, ref_vid in zip(cone_order, ref_order):
            ref_vertex = ref.vertices.get(ref_vid)
            if ref_vertex is None or not ref_vertex.has_position():
                continue
            updated[cone_vid] = Vertex(cone_vid, ref_vertex.x, ref_vertex.y)

    # Snap pentagon vertices last to ensure exact regular pentagon.
    for cone_vid, ref_vid in pent_map.items():
        ref_vertex = ref.vertices.get(ref_vid)
        if ref_vertex is None or not ref_vertex.has_position():
            continue
        updated[cone_vid] = Vertex(cone_vid, ref_vertex.x, ref_vertex.y)

    return PolyGrid(updated.values(), grid.edges.values(), grid.faces.values(), grid.metadata)


def _snap_cone_pentagon_geometry(grid: PolyGrid, size: float) -> PolyGrid:
    """Snap only the central pentagon to the explicit regular geometry."""
    ref = _build_ring1_grid(size)
    ref_pent = _find_pentagon_face(ref)
    cone_pent = _find_pentagon_face(grid)
    if ref_pent is None or cone_pent is None:
        return grid

    ref_center = _grid_center(ref)
    cone_center = _grid_center(grid)

    def ordered_pent_vertices(pent_face: Face, g: PolyGrid, center: tuple[float, float]) -> List[str]:
        ordered = _face_vertex_cycle(pent_face, g.edges.values())
        if len(ordered) != 5:
            ordered = _ordered_face_vertices(g.vertices, pent_face)
        return sorted(
            ordered,
            key=lambda vid: math.atan2(g.vertices[vid].y - center[1], g.vertices[vid].x - center[0]),
        )

    ref_pent_order = ordered_pent_vertices(ref_pent, ref, ref_center)
    cone_pent_order = ordered_pent_vertices(cone_pent, grid, cone_center)

    updated = {vid: Vertex(vid, v.x, v.y) for vid, v in grid.vertices.items()}
    for cone_vid, ref_vid in zip(cone_pent_order, ref_pent_order):
        ref_vertex = ref.vertices.get(ref_vid)
        if ref_vertex is None or not ref_vertex.has_position():
            continue
        updated[cone_vid] = Vertex(cone_vid, ref_vertex.x, ref_vertex.y)

    return PolyGrid(updated.values(), grid.edges.values(), grid.faces.values(), grid.metadata)


def _cone_exact_outer_optimize(grid: PolyGrid, rings: int) -> PolyGrid:
    """Constrained optimisation for outer rings to keep ring1 exact and ring2 targeted."""
    if rings < 2:
        return grid

    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return grid

    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings_map = ring_faces(adjacency, pent_face.id, max_depth=rings)
    edge_targets = compute_edge_target_lengths_by_ring(grid, rings_map, grid.vertices)

    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return grid
    fixed_base = {
        vid: (grid.vertices[vid].x, grid.vertices[vid].y)
        for vid in pent_face.vertex_ids
        if grid.vertices[vid].has_position()
    }
    if not fixed_base:
        return grid

    positions = optimize_outer_rings_constrained(
        grid,
        grid.vertices,
        edge_targets,
        rings_map,
        fixed_base,
        start_ring=1,
        iterations=200,
        angle_weight=1.6,
        anchor_weight=0.02,
    )
    candidate = PolyGrid(positions.values(), grid.edges.values(), grid.faces.values(), grid.metadata)

    from .diagnostics import has_edge_crossings, min_face_signed_area

    if min_face_signed_area(candidate) <= 0:
        return grid
    if has_edge_crossings(candidate):
        return grid
    return candidate


def _build_hybrid_cone_grid(rings: int, size: float) -> PolyGrid:
    """Hybrid strategy: exact ring-1/2 geometry, then Tutte+optimise outer rings."""
    if rings <= 0:
        return _build_single_pentagon(size)

    grid = _build_ring1_grid(size)
    if rings == 1:
        return grid

    vertices = {vid: Vertex(vid, v.x, v.y) for vid, v in grid.vertices.items()}
    faces = list(grid.faces.values())
    position_map = {
        _vertex_key((v.x, v.y)): Vertex(vid, v.x, v.y)
        for vid, v in vertices.items()
        if v.has_position()
    }

    # Build ring-2 with explicit geometry.
    _add_hex_ring_geometry(vertices, position_map, faces, ring_idx=2)
    edges, faces = _rebuild_edges_from_faces(faces)
    grid = PolyGrid(vertices.values(), edges, faces)
    grid = _exact_ring2_geometry(grid, rings=2)

    if rings == 2:
        return _ensure_positions_with_tutte(grid, max_ring=2)

    # Expand remaining rings topologically.
    next_vid = 1 + max(int(vid[1:]) for vid in vertices if vid.startswith("v"))
    next_fid = 1 + max(int(face.id[1:]) for face in faces if face.id.startswith("f"))
    for _ in range(2, rings):
        next_vid, next_fid = _add_hex_ring(vertices, faces, next_vid, next_fid)
    edges, faces = _rebuild_edges_from_faces(faces)
    grid = PolyGrid(vertices.values(), edges, faces)

    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return grid

    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings_map = ring_faces(adjacency, pent_face.id, max_depth=rings)

    fixed_positions = _build_fixed_positions(grid, force_boundary=True)
    ring2_vertices = set(_collect_face_vertices(grid, rings_map.get(2, [])))
    for vid in ring2_vertices:
        v = grid.vertices.get(vid)
        if v and v.has_position():
            fixed_positions[vid] = (v.x, v.y)

    embedded = tutte_embedding(grid.vertices, grid.edges.values(), fixed_positions)
    grid = PolyGrid(embedded.values(), grid.edges.values(), grid.faces.values())

    edge_targets = compute_edge_target_ratios_by_class(grid, rings_map)
    optimised = optimise_positions_to_edge_targets(
        grid,
        grid.vertices,
        edge_targets,
        fixed_positions,
        iterations=80,
    )
    return PolyGrid(optimised.values(), grid.edges.values(), grid.faces.values())


def _ensure_positions_with_tutte(grid: PolyGrid, max_ring: int) -> PolyGrid:
    """Fill missing positions using Tutte embedding with fixed inner rings."""
    if all(v.has_position() for v in grid.vertices.values()):
        return grid

    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return grid

    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings_map = ring_faces(adjacency, pent_face.id, max_depth=max_ring)
    fixed_positions = _build_fixed_positions(grid, force_boundary=True)
    ring_vertices = set(
        vid
        for depth in range(1, max_ring + 1)
        for fid in rings_map.get(depth, [])
        for vid in grid.faces[fid].vertex_ids
    )
    for vid in ring_vertices:
        v = grid.vertices.get(vid)
        if v and v.has_position():
            fixed_positions[vid] = (v.x, v.y)

    embedded = tutte_embedding(grid.vertices, grid.edges.values(), fixed_positions)
    return PolyGrid(embedded.values(), grid.edges.values(), grid.faces.values())


def _exact_ring2_geometry(grid: PolyGrid, rings: int) -> PolyGrid:
    """Constrained optimisation to enforce exact ring-2 geometry with ring-1 fixed."""
    if rings < 2:
        return grid

    grid = _ensure_positions_with_tutte(grid, max_ring=2)

    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return grid

    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings_map = ring_faces(adjacency, pent_face.id, max_depth=rings)
    ring1_faces = rings_map.get(1, [])
    ring1_vertices = set(_collect_face_vertices(grid, ring1_faces))

    fixed_base = {
        vid: (grid.vertices[vid].x, grid.vertices[vid].y)
        for vid in ring1_vertices
        if grid.vertices[vid].has_position()
    }
    if not fixed_base:
        return grid

    edge_targets = compute_edge_target_lengths_by_ring(grid, rings_map, grid.vertices)
    positions = optimize_outer_rings_constrained(
        grid,
        grid.vertices,
        edge_targets,
        rings_map,
        fixed_base,
        start_ring=2,
        iterations=200,
        angle_weight=1.6,
        anchor_weight=0.02,
    )
    candidate = PolyGrid(positions.values(), grid.edges.values(), grid.faces.values(), grid.metadata)

    from .diagnostics import has_edge_crossings, min_face_signed_area

    if min_face_signed_area(candidate) <= 0:
        return grid
    if has_edge_crossings(candidate):
        return grid
    return candidate


def _apply_cone_unwrap(grid: PolyGrid, defect_angle: float) -> PolyGrid:
    """Project planar coordinates onto an unwrapped cone by compressing angles and cutting seams."""
    if defect_angle <= 0:
        return grid
    scale = (2 * math.pi - defect_angle) / (2 * math.pi)
    if scale <= 0:
        return grid

    twopi_scaled = 2 * math.pi * scale
    base_theta: Dict[str, float] = {}
    radius: Dict[str, float] = {}
    for vid, v in grid.vertices.items():
        if not v.has_position():
            continue
        r = math.hypot(v.x, v.y)
        theta = math.atan2(v.y, v.x)
        if theta < 0:
            theta += 2 * math.pi
        base_theta[vid] = theta * scale
        radius[vid] = r

    def wrap_key(vid: str, wrap: int) -> str:
        return f"{vid}_w{wrap}" if wrap != 0 else vid

    new_vertices: Dict[str, Vertex] = {}
    new_faces: List[Face] = []

    for face in grid.faces.values():
        ordered = _face_vertex_cycle(face, grid.edges.values())
        if len(ordered) != len(face.vertex_ids):
            ordered = _ordered_face_vertices(grid.vertices, face)
        if not ordered:
            new_faces.append(face)
            continue

        wraps: Dict[str, int] = {}
        first_vid = ordered[0]
        wraps[first_vid] = 0
        prev_theta = base_theta.get(first_vid, 0.0)

        for vid in ordered[1:]:
            theta = base_theta.get(vid, 0.0)
            best_wrap = 0
            best_delta = None
            for k in (-1, 0, 1):
                candidate = theta + k * twopi_scaled
                delta = abs(candidate - prev_theta)
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_wrap = k
            wraps[vid] = wraps[ordered[ordered.index(vid) - 1]] + best_wrap
            prev_theta = theta + best_wrap * twopi_scaled

        face_vertex_ids: List[str] = []
        for vid in ordered:
            wrap = wraps.get(vid, 0)
            key = wrap_key(vid, wrap)
            face_vertex_ids.append(key)
            if key in new_vertices:
                continue
            if vid in base_theta and vid in radius:
                theta = base_theta[vid] + wrap * twopi_scaled
                r = radius[vid]
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                new_vertices[key] = Vertex(key, x, y)
            else:
                v = grid.vertices.get(vid)
                new_vertices[key] = Vertex(key, v.x, v.y) if v else Vertex(key)

        new_faces.append(
            Face(
                id=face.id,
                face_type=face.face_type,
                vertex_ids=tuple(face_vertex_ids),
            )
        )

    edges, rebuilt_faces = _rebuild_edges_from_faces(new_faces)
    return PolyGrid(new_vertices.values(), edges, rebuilt_faces, grid.metadata)


def _snap_cone_ring2_geometry(
    grid: PolyGrid,
    size: float,
    strength: float = 0.4,
) -> PolyGrid:
    """Gently pull ring-2 vertices toward geometry-first targets for pentagon symmetry."""
    if strength <= 0:
        return grid

    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return grid

    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings_map = ring_faces(adjacency, pent_face.id, max_depth=2)
    ring2_faces = rings_map.get(2, [])
    if not ring2_faces:
        return grid

    ring2_vertices = set(_collect_face_vertices(grid, ring2_faces))
    target_positions = _apply_outer_ring_geometry(grid, rings=2)

    def blended_grid(alpha: float) -> PolyGrid:
        updated = {vid: Vertex(vid, v.x, v.y) for vid, v in grid.vertices.items()}
        for vid in ring2_vertices:
            current = grid.vertices.get(vid)
            target = target_positions.get(vid)
            if current is None or target is None:
                continue
            if not (current.has_position() and target.has_position()):
                continue
            nx = current.x + (target.x - current.x) * alpha
            ny = current.y + (target.y - current.y) * alpha
            updated[vid] = Vertex(vid, nx, ny)
        return PolyGrid(updated.values(), grid.edges.values(), grid.faces.values(), grid.metadata)

    from .diagnostics import has_edge_crossings, min_face_signed_area

    candidates = [strength, strength * 0.7, strength * 0.5, strength * 0.35, 0.2, 0.1, 0.05]
    for alpha in candidates:
        if alpha <= 0:
            continue
        candidate = blended_grid(alpha)
        if min_face_signed_area(candidate) <= 0:
            continue
        if has_edge_crossings(candidate):
            continue
        return candidate

    return grid


def _relax_cone_outer_geometry(grid: PolyGrid, rings: int) -> PolyGrid:
    """Relax outer rings while keeping ring-1 fixed to avoid inversions/crossings."""
    pent_face = _find_pentagon_face(grid)
    if pent_face is None or rings < 2:
        return grid

    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings_map = ring_faces(adjacency, pent_face.id, max_depth=1)
    ring1_faces = rings_map.get(1, [])
    ring1_vertices = set(_collect_face_vertices(grid, ring1_faces))

    fixed_positions = {
        vid: (grid.vertices[vid].x, grid.vertices[vid].y)
        for vid in ring1_vertices
        if grid.vertices[vid].has_position()
    }
    if not fixed_positions:
        return grid

    from .diagnostics import has_edge_crossings, min_face_signed_area

    attempts = [
        (40, 0.25, 0.12),
        (60, 0.2, 0.1),
        (80, 0.15, 0.08),
        (120, 0.1, 0.05),
    ]
    for iterations, strength, max_step in attempts:
        relaxed = _edge_length_relax_safe(
            grid.vertices,
            grid.edges.values(),
            fixed_positions,
            iterations=iterations,
            strength=strength,
            max_step=max_step,
        )
        candidate = PolyGrid(relaxed.values(), grid.edges.values(), grid.faces.values(), grid.metadata)
        if min_face_signed_area(candidate) <= 0:
            continue
        if has_edge_crossings(candidate):
            continue
        return candidate

    return grid


def _apply_cone_outer_geometry(grid: PolyGrid, rings: int) -> PolyGrid:
    """Replace outer ring positions using geometry-first placement from ring-1 boundary."""
    if rings < 2:
        return grid

    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return grid

    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings_map = ring_faces(adjacency, pent_face.id, max_depth=rings)
    ring1_faces = rings_map.get(1, [])
    ring1_vertices = set(_collect_face_vertices(grid, ring1_faces))

    target_positions = _apply_outer_ring_geometry(grid, rings=rings)
    updated = {vid: Vertex(vid, v.x, v.y) for vid, v in grid.vertices.items()}
    for vid, target in target_positions.items():
        if vid in ring1_vertices:
            continue
        if not target.has_position():
            continue
        updated[vid] = Vertex(vid, target.x, target.y)

    return PolyGrid(updated.values(), grid.edges.values(), grid.faces.values(), grid.metadata)


def _tutte_relax_cone(grid: PolyGrid) -> PolyGrid:
    """Re-embed non-fixed vertices with Tutte using fixed pentagon/ring1/boundary."""
    fixed_positions = _build_fixed_positions(grid, force_boundary=True)
    if not fixed_positions:
        return grid
    embedded = tutte_embedding(grid.vertices, grid.edges.values(), fixed_positions)
    return PolyGrid(embedded.values(), grid.edges.values(), grid.faces.values(), grid.metadata)


def _build_cone_dual_grid_with_offset(rings: int, size: float, offset: float) -> PolyGrid:
    padding = 4
    tri_vertices, coords = _build_triangular_vertices(rings + padding, size)
    triangles = _build_triangles(coords)
    angle_tol = 0.015
    wedge_angle = math.pi / 3
    seam_count = 5
    micro_width = wedge_angle / seam_count

    # Remove triangles inside micro-wedges contained within the 60° wedge.
    kept: List[tuple[str, str, str]] = []
    for tri in triangles:
        pts = [tri_vertices[vid] for vid in tri]
        cx = sum(v.x for v in pts) / 3
        cy = sum(v.y for v in pts) / 3
        ang = math.atan2(cy, cx)
        if ang < 0:
            ang += 2 * math.pi
        inside = False
        for i in range(seam_count):
            a0 = (offset + i * micro_width) % (2 * math.pi)
            a1 = (a0 + micro_width) % (2 * math.pi)
            if a0 < a1:
                if a0 + angle_tol <= ang <= a1 - angle_tol:
                    inside = True
                    break
            else:
                if ang >= a0 + angle_tol or ang <= a1 - angle_tol:
                    inside = True
                    break
        if not inside:
            kept.append(tri)

    triangles = kept

    # Merge each micro-wedge boundary ray pair.
    merge_total: Dict[str, str] = {}
    for i in range(seam_count):
        a0 = (offset + i * micro_width) % (2 * math.pi)
        a1 = (a0 + micro_width) % (2 * math.pi)
        seam_merge = _build_ray_merge_map(tri_vertices, a0, a1, angle_tol)
        for src, tgt in seam_merge.items():
            merge_total.setdefault(src, tgt)

    tri_vertices = _merge_vertices(tri_vertices, merge_total)
    triangles = _merge_triangles(triangles, merge_total)
    dual_vertices, faces = _dualize(tri_vertices, triangles)
    edges, faces = _rebuild_edges_from_faces(faces)
    grid = PolyGrid(dual_vertices.values(), edges, faces)
    return _crop_to_rings(grid, rings)


def _boundary_radius_variance(grid: PolyGrid) -> float:
    boundary = _boundary_vertex_cycle(grid)
    if not boundary:
        return 0.0
    center = _grid_center(grid)
    radii = []
    for vid in boundary:
        v = grid.vertices.get(vid)
        if v is None or not v.has_position():
            continue
        radii.append(math.hypot(v.x - center[0], v.y - center[1]))
    if not radii:
        return 0.0
    mean = sum(radii) / len(radii)
    return sum((r - mean) ** 2 for r in radii) / len(radii)


def _ring1_spacing_variance(grid: PolyGrid) -> float:
    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return 0.0
    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings_map = ring_faces(adjacency, pent_face.id, max_depth=1)
    ring1_faces = rings_map.get(1, [])
    if len(ring1_faces) < 5:
        return 0.0

    center = _grid_center(grid)
    angles = []
    for fid in ring1_faces:
        face = grid.faces.get(fid)
        if face is None:
            continue
        coords = [grid.vertices[vid] for vid in face.vertex_ids]
        if not coords or not all(v.has_position() for v in coords):
            continue
        cx = sum(v.x for v in coords) / len(coords)
        cy = sum(v.y for v in coords) / len(coords)
        angle = math.atan2(cy - center[1], cx - center[0])
        angles.append(angle)
    if len(angles) < 5:
        return 0.0
    angles = sorted(angles)
    gaps = []
    for i in range(len(angles)):
        a0 = angles[i]
        a1 = angles[(i + 1) % len(angles)]
        if i == len(angles) - 1:
            a1 += 2 * math.pi
        gaps.append(a1 - a0)
    target = 2 * math.pi / 5
    return sum((gap - target) ** 2 for gap in gaps) / len(gaps)




def _mean_radius(grid: PolyGrid, vertex_ids: Iterable[str], center: tuple[float, float]) -> float:
    radii = []
    for vid in vertex_ids:
        v = grid.vertices[vid]
        if v.x is None or v.y is None:
            continue
        radii.append(math.hypot(v.x - center[0], v.y - center[1]))
    return sum(radii) / len(radii) if radii else 0.0



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


def _regular_pentagon_positions(
    grid: PolyGrid,
    vertex_ids: List[str],
    radius: float,
    center: tuple[float, float],
) -> Dict[str, tuple[float, float]]:
    if len(vertex_ids) != 5:
        return _uniform_circle_positions(grid, vertex_ids, radius, center)

    # order pentagon vertices by angle around center, then assign equal spacing
    angles = []
    for vid in vertex_ids:
        v = grid.vertices[vid]
        angles.append(math.atan2(v.y - center[1], v.x - center[0]))
    ordered = [vid for _, vid in sorted(zip(angles, vertex_ids))]
    base_angle = sum(angles) / len(angles)
    step = 2 * math.pi / 5

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
) -> List[tuple[str, str]]:
    boundary_cycle = _boundary_vertex_cycle(grid)
    if not boundary_cycle:
        return []
    edge_lookup = {
        tuple(sorted(edge.vertex_ids)): edge for edge in grid.edges.values()
        if sum(1 for fid in edge.face_ids if fid in region_faces) == 1
    }
    ordered_edges: List[tuple[str, str]] = []
    for idx in range(len(boundary_cycle)):
        a = boundary_cycle[idx]
        b = boundary_cycle[(idx + 1) % len(boundary_cycle)]
        if tuple(sorted((a, b))) in edge_lookup:
            ordered_edges.append((a, b))
    return ordered_edges


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
    face_data: List[tuple[Face, List[str]]] = []
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
        face_data.append((face, edge_ids))

    edges = list(edge_map.values())
    rebuilt_faces: List[Face] = []
    for face, edge_ids in face_data:
        ordered = _face_vertex_cycle(face, edges)
        if len(ordered) == len(face.vertex_ids) and set(ordered) == set(face.vertex_ids):
            vertex_ids = tuple(ordered)
        else:
            vertex_ids = face.vertex_ids
        rebuilt_faces.append(
            Face(
                id=face.id,
                face_type=face.face_type,
                vertex_ids=vertex_ids,
                edge_ids=tuple(edge_ids),
            )
        )

    return edges, rebuilt_faces


def _vertex_key(position: Tuple[float, float]) -> str:
    x, y = position
    if abs(x) < 1e-7:
        x = 0.0
    if abs(y) < 1e-7:
        y = 0.0
    return f"{x:.6f},{y:.6f}"


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


def classify_edges(grid: PolyGrid, rings_map: dict) -> Dict[str, str]:
    """Classify edges into pent-inner, ring-radial, ring-tangential, outer-boundary.

    rings_map: mapping ring->face ids (as from ring_faces)
    Returns edge_id -> class_name
    """
    face_to_ring: Dict[str, int] = {}
    for r, fids in rings_map.items():
        for fid in fids:
            face_to_ring[fid] = r

    edge_class: Dict[str, str] = {}
    for edge in grid.edges.values():
        if len(edge.face_ids) == 0:
            edge_class[edge.id] = "outer-boundary"
            continue
        if any(grid.faces.get(fid) and grid.faces[fid].face_type == "pent" for fid in edge.face_ids):
            edge_class[edge.id] = "pent-inner"
            continue
        # classify by ring adjacency
        rings = [face_to_ring.get(fid, -1) for fid in edge.face_ids]
        rings = [r for r in rings if r >= 0]
        if not rings:
            edge_class[edge.id] = "outer-boundary"
            continue
        if len(rings) == 1:
            # edge between a face in a ring and boundary (outer)
            edge_class[edge.id] = "ring-radial"
        else:
            if rings[0] == rings[1]:
                edge_class[edge.id] = "ring-tangential"
            else:
                edge_class[edge.id] = "ring-radial"

    return edge_class


def compute_edge_target_ratios(grid: PolyGrid, rings_map: dict) -> Dict[str, float]:
    """Compute per-edge target length ratios from ring length solver.

    Returns map edge_id -> ratio (relative units; will be scaled to absolute lengths later).
    """
    edge_ratios: Dict[str, list[float]] = {eid: [] for eid in grid.edges}

    for ring_idx, face_ids in rings_map.items():
        if ring_idx == 0:
            continue
        if ring_idx == 1:
            pattern = [1.0] * 6
        else:
            lengths_solution = solve_ring_hex_lengths(ring_idx, inner_edge_length=1.0)
            inner = lengths_solution["inner"]
            protrude = lengths_solution["protrude"]
            outer = lengths_solution["outer"]
            pattern = [inner, protrude, outer, outer, outer, protrude]

        for fid in face_ids:
            face = grid.faces.get(fid)
            if face is None or len(face.vertex_ids) != 6:
                continue
            verts = list(face.vertex_ids)
            # edges in order (v[i], v[i+1]) -> pattern[i]
            for i in range(6):
                a = verts[i]
                b = verts[(i + 1) % 6]
                key = tuple(sorted((a, b)))
                for edge in grid.edges.values():
                    if edge.vertex_ids == key and fid in edge.face_ids:
                        edge_ratios[edge.id].append(pattern[i])
                        break

    # average ratios for edges with multiple contributing faces
    averaged: Dict[str, float] = {}
    for eid, vals in edge_ratios.items():
        if not vals:
            continue
        averaged[eid] = sum(vals) / len(vals)

    return averaged


def compute_edge_target_ratios_by_class(grid: PolyGrid, rings_map: dict) -> Dict[str, float]:
    """Compute per-edge target ratios using ring adjacency (avoids face ordering dependence)."""
    face_to_ring: Dict[str, int] = {}
    for ring_idx, face_ids in rings_map.items():
        for fid in face_ids:
            face_to_ring[fid] = ring_idx

    ratios: Dict[str, float] = {}
    for edge in grid.edges.values():
        if not edge.face_ids:
            continue
        ring_indices = [face_to_ring.get(fid, -1) for fid in edge.face_ids]
        ring_indices = [r for r in ring_indices if r >= 0]
        if not ring_indices:
            continue
        ring_idx = max(ring_indices)
        if ring_idx <= 0:
            continue
        if ring_idx == 1:
            ratios[edge.id] = 1.0
            continue

        lengths_solution = solve_ring_hex_lengths(ring_idx, inner_edge_length=1.0)
        inner = lengths_solution["inner"]
        protrude = lengths_solution["protrude"]
        outer = lengths_solution["outer"]

        if any(grid.faces.get(fid) and grid.faces[fid].face_type == "pent" for fid in edge.face_ids):
            ratios[edge.id] = inner
            continue

        if len(ring_indices) == 1:
            ratios[edge.id] = outer
            continue

        if ring_indices[0] == ring_indices[1]:
            ratios[edge.id] = outer
        else:
            ratios[edge.id] = protrude

    return ratios


def compute_edge_target_lengths_by_ring(
    grid: PolyGrid,
    rings_map: dict,
    positions: Dict[str, Vertex],
) -> Dict[str, float]:
    """Compute per-edge absolute target lengths using ring specs and current inner-edge lengths."""
    face_to_ring: Dict[str, int] = {}
    for ring_idx, face_ids in rings_map.items():
        for fid in face_ids:
            face_to_ring[fid] = ring_idx

    targets: Dict[str, float] = {}
    for ring_idx, face_ids in rings_map.items():
        if ring_idx <= 0:
            continue
        inner_vertices = set(
            vid
            for fid in rings_map.get(ring_idx - 1, [])
            for vid in grid.faces[fid].vertex_ids
        )
        inner_edges = []
        for edge in grid.edges.values():
            if not any(fid in face_ids for fid in edge.face_ids):
                continue
            a, b = edge.vertex_ids
            if a in inner_vertices and b in inner_vertices:
                va = positions[a]
                vb = positions[b]
                if va.has_position() and vb.has_position():
                    inner_edges.append(math.hypot(vb.x - va.x, vb.y - va.y))
        inner_edge_len = sum(inner_edges) / len(inner_edges) if inner_edges else 1.0
        lengths_solution = solve_ring_hex_lengths(ring_idx, inner_edge_length=inner_edge_len)
        inner = lengths_solution["inner"]
        protrude = lengths_solution["protrude"]
        outer = lengths_solution["outer"]

        for edge in grid.edges.values():
            if not any(fid in face_ids for fid in edge.face_ids):
                continue
            a, b = edge.vertex_ids
            ring_indices = [face_to_ring.get(fid, -1) for fid in edge.face_ids]
            ring_indices = [r for r in ring_indices if r >= 0]
            if not ring_indices:
                continue
            if any(grid.faces.get(fid) and grid.faces[fid].face_type == "pent" for fid in edge.face_ids):
                targets[edge.id] = inner
                continue
            if a in inner_vertices and b in inner_vertices:
                targets[edge.id] = inner
            elif len(ring_indices) == 1:
                targets[edge.id] = outer
            elif ring_indices[0] == ring_indices[1]:
                targets[edge.id] = outer
            else:
                targets[edge.id] = protrude

    return targets

