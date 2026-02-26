from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Dict, Iterable, List, Optional

from .algorithms import build_face_adjacency, get_face_adjacency
from .models import Edge, Face, MacroEdge, Vertex


class PolyGrid:
    """Topology-first container for vertices, edges, and faces.

    Optional *macro_edges* describe the sides of the grid's outer polygon
    (5 sides for a pentagon-centred grid, 6 for a hex grid).
    """

    VERSION = "1.0"

    def __init__(
        self,
        vertices: Iterable[Vertex],
        edges: Iterable[Edge],
        faces: Iterable[Face],
        metadata: Optional[dict] = None,
        macro_edges: Optional[List[MacroEdge]] = None,
    ) -> None:
        self.vertices: Dict[str, Vertex] = {v.id: v for v in vertices}
        self.edges: Dict[str, Edge] = {e.id: e for e in edges}
        self.faces: Dict[str, Face] = {f.id: f for f in faces}
        self.metadata = metadata or {}
        self.macro_edges: List[MacroEdge] = macro_edges or []

    # ── Boundary / macro-edge helpers ───────────────────────────────

    def boundary_edges(self) -> List[Edge]:
        """Return edges with fewer than 2 adjacent faces (boundary)."""
        return [e for e in self.edges.values() if len(e.face_ids) < 2]

    def boundary_vertex_cycle(self) -> List[str]:
        """Return an ordered cycle of boundary vertex ids."""
        from .geometry import boundary_vertex_cycle
        return boundary_vertex_cycle(self.edges.values())

    def compute_macro_edges(
        self,
        n_sides: int | None = None,
        corner_ids: List[str] | None = None,
    ) -> List[MacroEdge]:
        """Compute macro-edges from the boundary.

        *n_sides*: expected number of polygon sides (5 or 6).
        *corner_ids*: explicit corner vertex ids in boundary-cycle order.
            If not supplied, corners are detected as the *n_sides* vertices
            with the sharpest turning angle on the boundary.

        The result is also stored in ``self.macro_edges``.
        """
        cycle = self.boundary_vertex_cycle()
        if not cycle:
            return []

        if n_sides is None:
            n_sides = self.metadata.get("sides", 6)

        if corner_ids is None:
            corner_ids = _detect_corners(self.vertices, cycle, n_sides)

        # Build lookup: boundary edge by vertex-pair
        b_edges = self.boundary_edges()
        edge_lookup: Dict[tuple[str, str], Edge] = {}
        for e in b_edges:
            a, b = e.vertex_ids
            edge_lookup[(a, b)] = e
            edge_lookup[(b, a)] = e

        # Find starting index (first corner in cycle)
        corner_set = set(corner_ids)
        start_idx = None
        for i, vid in enumerate(cycle):
            if vid in corner_set:
                start_idx = i
                break
        if start_idx is None:
            return []

        # Rotate cycle so it starts at a corner
        cycle = cycle[start_idx:] + cycle[:start_idx]

        # Split cycle at corners
        macro_edges: List[MacroEdge] = []
        segment_start = 0
        corner_hits: List[int] = [0]
        for i in range(1, len(cycle)):
            if cycle[i] in corner_set:
                corner_hits.append(i)

        for seg_idx in range(len(corner_hits)):
            i_start = corner_hits[seg_idx]
            i_end = corner_hits[(seg_idx + 1) % len(corner_hits)]

            # Build vertex slice from i_start to i_end (inclusive)
            if i_end > i_start:
                seg_vids = cycle[i_start : i_end + 1]
            else:
                seg_vids = cycle[i_start:] + cycle[: i_end + 1]

            # Collect micro-edge ids along this segment
            seg_eids: List[str] = []
            for j in range(len(seg_vids) - 1):
                e = edge_lookup.get((seg_vids[j], seg_vids[j + 1]))
                if e:
                    seg_eids.append(e.id)

            macro_edges.append(MacroEdge(
                id=seg_idx,
                vertex_ids=tuple(seg_vids),
                edge_ids=tuple(seg_eids),
                corner_start=seg_vids[0],
                corner_end=seg_vids[-1],
            ))

        self.macro_edges = macro_edges
        return macro_edges

    # ── Existing methods ────────────────────────────────────────────

    def validate(self, strict: bool = False) -> list[str]:
        errors: list[str] = []

        for edge in self.edges.values():
            for vertex_id in edge.vertex_ids:
                if vertex_id not in self.vertices:
                    errors.append(f"Edge {edge.id} references missing vertex {vertex_id}")
            for face_id in edge.face_ids:
                if face_id not in self.faces:
                    errors.append(f"Edge {edge.id} references missing face {face_id}")

        for face in self.faces.values():
            for vertex_id in face.vertex_ids:
                if vertex_id not in self.vertices:
                    errors.append(f"Face {face.id} references missing vertex {vertex_id}")
            for edge_id in face.edge_ids:
                if edge_id not in self.edges:
                    errors.append(f"Face {face.id} references missing edge {edge_id}")
            if strict:
                errors.extend(face.validate_polygon())
                for edge_id in face.edge_ids:
                    edge = self.edges.get(edge_id)
                    if edge and face.id not in edge.face_ids:
                        errors.append(
                            f"Face {face.id} edge {edge_id} does not reference face in edge.face_ids"
                        )

        return errors

    def compute_face_neighbors(self) -> Dict[str, list[str]]:
        return build_face_adjacency(self.faces.values(), self.edges.values())

    def face_adjacency(self) -> Dict[str, list[str]]:
        """Return face adjacency — works for both flat and globe grids.

        Uses :func:`get_face_adjacency` which prefers ``neighbor_ids``
        when populated, falling back to shared-edge computation.
        """
        return get_face_adjacency(self)

    def with_neighbors(self) -> "PolyGrid":
        adjacency = self.compute_face_neighbors()
        faces = []
        for face in self.faces.values():
            neighbors = tuple(adjacency.get(face.id, []))
            faces.append(replace(face, neighbor_ids=neighbors))
        return PolyGrid(
            self.vertices.values(), self.edges.values(), faces,
            self.metadata, self.macro_edges,
        )

    def to_dict(self, include_neighbors: bool = True) -> dict:
        adjacency = self.compute_face_neighbors() if include_neighbors else {}
        faces_payload = []
        for face in sorted(self.faces.values(), key=lambda f: f.id):
            neighbors = (
                sorted(face.neighbor_ids)
                if face.neighbor_ids
                else adjacency.get(face.id, [])
            )
            face_data = {
                "id": face.id,
                "type": face.face_type,
                "vertices": list(face.vertex_ids),
                "edges": list(face.edge_ids),
                "neighbors": neighbors,
            }
            if face.metadata:
                face_data["metadata"] = face.metadata
            faces_payload.append(face_data)

        vertices_payload = []
        for vertex in sorted(self.vertices.values(), key=lambda v: v.id):
            payload = {"id": vertex.id}
            if vertex.has_position():
                pos = {"x": vertex.x, "y": vertex.y}
                if vertex.z is not None:
                    pos["z"] = vertex.z
                payload["position"] = pos
            vertices_payload.append(payload)

        edges_payload = []
        for edge in sorted(self.edges.values(), key=lambda e: e.id):
            edges_payload.append(
                {
                    "id": edge.id,
                    "vertices": list(edge.vertex_ids),
                    "faces": list(edge.face_ids),
                }
            )

        macro_edges_payload = []
        for me in self.macro_edges:
            macro_edges_payload.append(
                {
                    "id": me.id,
                    "vertices": list(me.vertex_ids),
                    "edges": list(me.edge_ids),
                    "corner_start": me.corner_start,
                    "corner_end": me.corner_end,
                }
            )

        data = {
            "version": self.VERSION,
            "metadata": self.metadata,
            "vertices": vertices_payload,
            "edges": edges_payload,
            "faces": faces_payload,
        }

        if include_neighbors:
            data["face_neighbors"] = adjacency

        if macro_edges_payload:
            data["macro_edges"] = macro_edges_payload

        return data

    @classmethod
    def from_dict(cls, payload: dict) -> "PolyGrid":
        vertices = [
            Vertex(
                id=vertex["id"],
                x=vertex.get("position", {}).get("x"),
                y=vertex.get("position", {}).get("y"),
                z=vertex.get("position", {}).get("z"),
            )
            for vertex in payload.get("vertices", [])
        ]
        edges = [
            Edge(
                id=edge["id"],
                vertex_ids=tuple(edge["vertices"]),
                face_ids=tuple(edge.get("faces", [])),
            )
            for edge in payload.get("edges", [])
        ]
        faces = [
            Face(
                id=face["id"],
                face_type=face.get("type", "other"),
                vertex_ids=tuple(face.get("vertices", [])),
                edge_ids=tuple(face.get("edges", [])),
                neighbor_ids=tuple(face.get("neighbors", [])),
                metadata=face.get("metadata", {}),
            )
            for face in payload.get("faces", [])
        ]
        macro_edges = [
            MacroEdge(
                id=me["id"],
                vertex_ids=tuple(me["vertices"]),
                edge_ids=tuple(me["edges"]),
                corner_start=me["corner_start"],
                corner_end=me["corner_end"],
            )
            for me in payload.get("macro_edges", [])
        ]

        return cls(
            vertices, edges, faces,
            payload.get("metadata", {}),
            macro_edges or None,
        )

    def to_json(self, include_neighbors: bool = True, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(include_neighbors=include_neighbors),
            indent=indent,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, json_data: str) -> "PolyGrid":
        return cls.from_dict(json.loads(json_data))


# ═══════════════════════════════════════════════════════════════════
# Corner detection helper
# ═══════════════════════════════════════════════════════════════════

def _detect_corners(
    vertices: Dict[str, Vertex],
    cycle: List[str],
    n_corners: int,
) -> List[str]:
    """Detect *n_corners* corner vertices from a boundary cycle.

    On a hex or pentagon polygrid the boundary zigzags (alternating left/
    right turns).  At the *n_corners* true corners, two consecutive turns
    have the **same sign** (both left or both right).

    The algorithm:
    1. Compute the signed turn direction at every boundary vertex.
    2. Find which sign of consecutive same-sign pair appears exactly
       *n_corners* times.
    3. Return the first vertex of each such pair, in boundary-cycle order.
    """
    n = len(cycle)
    if n < 3:
        return cycle[:n_corners]

    # Compute signed cross product (turn direction) at each vertex
    signs: List[int] = []
    for i in range(n):
        v_prev = vertices[cycle[(i - 1) % n]]
        v_curr = vertices[cycle[i]]
        v_next = vertices[cycle[(i + 1) % n]]
        if not (v_prev.has_position() and v_curr.has_position() and v_next.has_position()):
            signs.append(0)
            continue
        dx1 = v_curr.x - v_prev.x
        dy1 = v_curr.y - v_prev.y
        dx2 = v_next.x - v_curr.x
        dy2 = v_next.y - v_curr.y
        cross = dx1 * dy2 - dy1 * dx2
        signs.append(1 if cross > 0 else -1)

    # Find consecutive same-sign pairs; pick the sign that gives n_corners
    for target_sign in (1, -1):
        corners: List[str] = []
        for i in range(n):
            if signs[(i - 1) % n] == target_sign and signs[i] == target_sign:
                corners.append(cycle[(i - 1) % n])
        if len(corners) == n_corners:
            return corners

    # Fallback: pick the sharpest turning angles (unsigned)
    angles: List[tuple[float, str]] = []
    for i in range(n):
        v_prev = vertices[cycle[(i - 1) % n]]
        v_curr = vertices[cycle[i]]
        v_next = vertices[cycle[(i + 1) % n]]
        if not (v_prev.has_position() and v_curr.has_position() and v_next.has_position()):
            continue
        dx1 = v_curr.x - v_prev.x
        dy1 = v_curr.y - v_prev.y
        dx2 = v_next.x - v_curr.x
        dy2 = v_next.y - v_curr.y
        len1 = math.hypot(dx1, dy1)
        len2 = math.hypot(dx2, dy2)
        if len1 < 1e-12 or len2 < 1e-12:
            continue
        cos_a = (dx1 * dx2 + dy1 * dy2) / (len1 * len2)
        cos_a = max(-1.0, min(1.0, cos_a))
        angle = math.acos(cos_a)
        angles.append((angle, cycle[i]))
    angles.sort(key=lambda x: -x[0])
    chosen = {vid for _, vid in angles[:n_corners]}
    return [vid for vid in cycle if vid in chosen]
