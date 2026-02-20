from __future__ import annotations

import json
from dataclasses import replace
from typing import Dict, Iterable, Optional

from .algorithms import build_face_adjacency
from .models import Edge, Face, Vertex


class PolyGrid:
    """Topology-first container for vertices, edges, and faces."""

    VERSION = "1.0"

    def __init__(
        self,
        vertices: Iterable[Vertex],
        edges: Iterable[Edge],
        faces: Iterable[Face],
        metadata: Optional[dict] = None,
    ) -> None:
        self.vertices: Dict[str, Vertex] = {v.id: v for v in vertices}
        self.edges: Dict[str, Edge] = {e.id: e for e in edges}
        self.faces: Dict[str, Face] = {f.id: f for f in faces}
        self.metadata = metadata or {}

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

        return errors

    def compute_face_neighbors(self) -> Dict[str, list[str]]:
        return build_face_adjacency(self.faces.values(), self.edges.values())

    def with_neighbors(self) -> "PolyGrid":
        adjacency = self.compute_face_neighbors()
        faces = []
        for face in self.faces.values():
            neighbors = tuple(adjacency.get(face.id, []))
            faces.append(replace(face, neighbor_ids=neighbors))
        return PolyGrid(self.vertices.values(), self.edges.values(), faces, self.metadata)

    def to_dict(self, include_neighbors: bool = True) -> dict:
        adjacency = self.compute_face_neighbors() if include_neighbors else {}
        faces_payload = []
        for face in self.faces.values():
            neighbors = (
                list(face.neighbor_ids)
                if face.neighbor_ids
                else adjacency.get(face.id, [])
            )
            faces_payload.append(
                {
                    "id": face.id,
                    "type": face.face_type,
                    "vertices": list(face.vertex_ids),
                    "edges": list(face.edge_ids),
                    "neighbors": neighbors,
                }
            )

        vertices_payload = []
        for vertex in self.vertices.values():
            payload = {"id": vertex.id}
            if vertex.has_position():
                payload["position"] = {"x": vertex.x, "y": vertex.y}
            vertices_payload.append(payload)

        edges_payload = [
            {
                "id": edge.id,
                "vertices": list(edge.vertex_ids),
                "faces": list(edge.face_ids),
            }
            for edge in self.edges.values()
        ]

        data = {
            "version": self.VERSION,
            "metadata": self.metadata,
            "vertices": vertices_payload,
            "edges": edges_payload,
            "faces": faces_payload,
        }

        if include_neighbors:
            data["face_neighbors"] = adjacency

        return data

    @classmethod
    def from_dict(cls, payload: dict) -> "PolyGrid":
        vertices = [
            Vertex(
                id=vertex["id"],
                x=vertex.get("position", {}).get("x"),
                y=vertex.get("position", {}).get("y"),
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
            )
            for face in payload.get("faces", [])
        ]

        return cls(vertices, edges, faces, payload.get("metadata", {}))

    def to_json(self, include_neighbors: bool = True, indent: int = 2) -> str:
        return json.dumps(self.to_dict(include_neighbors=include_neighbors), indent=indent)

    @classmethod
    def from_json(cls, json_data: str) -> "PolyGrid":
        return cls.from_dict(json.loads(json_data))
