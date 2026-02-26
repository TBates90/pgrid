"""Grid builders for hex (6-sided) and pentagon-centred (5-sided) polygrids."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .models import Edge, Face, Vertex
from .polygrid import PolyGrid
from .algorithms import build_face_adjacency, ring_faces


@dataclass(frozen=True)
class AxialCoord:
    q: int
    r: int


def hex_face_count(rings: int) -> int:
    if rings < 0:
        raise ValueError("rings must be >= 0")
    return 1 + 3 * rings * (rings + 1)


def _hex_area(rings: int) -> List[AxialCoord]:
    coords: List[AxialCoord] = []
    for q in range(-rings, rings + 1):
        for r in range(-rings, rings + 1):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) <= rings:
                coords.append(AxialCoord(q=q, r=r))
    return coords


def _axial_to_pixel(coord: AxialCoord, size: float) -> tuple[float, float]:
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


def build_pure_hex_grid(rings: int, size: float = 1.0) -> PolyGrid:
    """Build a pure hex grid in a hexagonal shape using axial coordinates."""
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

    return PolyGrid(
        vertex_map.values(),
        edge_map.values(),
        faces,
        metadata={"generator": "hex", "rings": rings, "sides": 6},
    )


def build_pentagon_centered_grid(
    rings: int,
    size: float = 1.0,
    embed: bool = True,
    embed_mode: str = "tutte+optimise",
    validate_topology: bool = False,
) -> PolyGrid:
    """Build a pentagon-centred Goldberg grid (5-sided shape)."""
    if rings < 0:
        raise ValueError("rings must be >= 0")

    from .goldberg_topology import build_goldberg_grid

    optimise = embed and (embed_mode in ("tutte+optimise",))
    grid = build_goldberg_grid(rings, size=size, optimise=optimise)

    if validate_topology:
        errors = validate_pentagon_topology(grid, rings=rings)
        if errors:
            raise RuntimeError("Pentagon grid topology errors: " + "; ".join(errors))

    return grid


def validate_pentagon_topology(grid: PolyGrid, rings: int) -> list[str]:
    """Validate that grid has correct pentagon-centred topology."""
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
