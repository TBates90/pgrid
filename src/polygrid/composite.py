from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from .models import Edge, Face, Vertex
from .polygrid import PolyGrid


@dataclass(frozen=True)
class CompositeGrid:
    """Composite container that keeps sub-grids and a merged view."""

    merged: PolyGrid
    components: Dict[str, PolyGrid]
    id_prefixes: Dict[str, str]


def join_grids(
    grids: Dict[str, PolyGrid],
    id_prefixes: Dict[str, str] | None = None,
) -> CompositeGrid:
    """Join grids at data layer by merging into a single PolyGrid.

    Uses prefixes to avoid id collisions. This does not stitch boundaries yet;
    it creates a merged topology view that algorithms can operate on.
    """
    prefixes = id_prefixes or {name: f"{name}_" for name in grids}

    merged_vertices: list[Vertex] = []
    merged_edges: list[Edge] = []
    merged_faces: list[Face] = []

    for name, grid in grids.items():
        prefix = prefixes[name]
        v_map = {vid: f"{prefix}{vid}" for vid in grid.vertices}
        e_map = {eid: f"{prefix}{eid}" for eid in grid.edges}
        f_map = {fid: f"{prefix}{fid}" for fid in grid.faces}

        for vertex in grid.vertices.values():
            merged_vertices.append(Vertex(v_map[vertex.id], vertex.x, vertex.y))

        for edge in grid.edges.values():
            merged_edges.append(
                Edge(
                    id=e_map[edge.id],
                    vertex_ids=(v_map[edge.vertex_ids[0]], v_map[edge.vertex_ids[1]]),
                    face_ids=tuple(f_map[fid] for fid in edge.face_ids),
                )
            )

        for face in grid.faces.values():
            merged_faces.append(
                Face(
                    id=f_map[face.id],
                    face_type=face.face_type,
                    vertex_ids=tuple(v_map[vid] for vid in face.vertex_ids),
                    edge_ids=tuple(e_map[eid] for eid in face.edge_ids),
                    neighbor_ids=tuple(f_map[nid] for nid in face.neighbor_ids),
                )
            )

    merged = PolyGrid(merged_vertices, merged_edges, merged_faces)
    return CompositeGrid(merged=merged, components=grids, id_prefixes=prefixes)


def split_composite(composite: CompositeGrid) -> Dict[str, PolyGrid]:
    """Return original component grids stored in the composite container."""
    return composite.components
