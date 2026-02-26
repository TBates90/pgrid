from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Iterable, List

from .models import Edge, Face

if TYPE_CHECKING:
    from .polygrid import PolyGrid


def build_face_adjacency(faces: Iterable[Face], edges: Iterable[Edge]) -> Dict[str, List[str]]:
    """Return face adjacency map based purely on shared edges."""
    edge_to_faces: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        for face_id in edge.face_ids:
            edge_to_faces[edge.id].append(face_id)

    neighbors: dict[str, set[str]] = {face.id: set() for face in faces}
    for face_ids in edge_to_faces.values():
        if len(face_ids) < 2:
            continue
        for i, face_id in enumerate(face_ids):
            for other_id in face_ids[i + 1 :]:
                neighbors[face_id].add(other_id)
                neighbors[other_id].add(face_id)

    return {face_id: sorted(list(neigh)) for face_id, neigh in neighbors.items()}


def get_face_adjacency(grid: "PolyGrid") -> Dict[str, List[str]]:
    """Return face adjacency for any PolyGrid, including globe grids.

    Strategy:
    1. If faces already carry non-empty ``neighbor_ids`` (e.g. globe grids
       whose adjacency comes from the models library), build the adjacency
       map directly from those — O(n).
    2. Otherwise fall back to :func:`build_face_adjacency` which infers
       adjacency from shared edges.

    This is the recommended way to obtain adjacency for algorithms that
    must work on **both** flat hex grids and globe grids.
    """
    # Check if faces already have neighbor_ids populated
    faces = list(grid.faces.values())
    has_neighbor_ids = any(f.neighbor_ids for f in faces)

    if has_neighbor_ids:
        return {
            f.id: sorted(f.neighbor_ids)
            for f in faces
        }

    return build_face_adjacency(faces, grid.edges.values())


def ring_faces(
    face_adjacency: Dict[str, List[str]],
    start_face_id: str,
    max_depth: int,
) -> Dict[int, List[str]]:
    """Return faces grouped by BFS ring distance from a start face."""
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")

    visited = {start_face_id}
    rings: Dict[int, List[str]] = {0: [start_face_id]}
    frontier = [start_face_id]

    for depth in range(1, max_depth + 1):
        next_frontier: List[str] = []
        for face_id in frontier:
            for neighbor in face_adjacency.get(face_id, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                next_frontier.append(neighbor)
        if not next_frontier:
            break
        rings[depth] = sorted(next_frontier)
        frontier = next_frontier

    return rings
