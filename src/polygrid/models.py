from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass(frozen=True)
class Vertex:
    id: str
    x: Optional[float] = None
    y: Optional[float] = None

    def has_position(self) -> bool:
        return self.x is not None and self.y is not None


@dataclass(frozen=True)
class Edge:
    id: str
    vertex_ids: tuple[str, str]
    face_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Face:
    id: str
    face_type: str
    vertex_ids: tuple[str, ...]
    edge_ids: tuple[str, ...] = field(default_factory=tuple)
    neighbor_ids: tuple[str, ...] = field(default_factory=tuple)

    def vertex_count(self) -> int:
        return len(self.vertex_ids)

    def validate_polygon(self) -> list[str]:
        errors: list[str] = []
        if len(set(self.vertex_ids)) != self.vertex_count():
            errors.append(f"Face {self.id} has repeated vertex ids")
        if self.edge_ids and len(self.edge_ids) != self.vertex_count():
            errors.append(
                f"Face {self.id} has {len(self.edge_ids)} edges but {self.vertex_count()} vertices"
            )
        if self.face_type == "pent" and self.vertex_count() != 5:
            errors.append(f"Face {self.id} is pent but has {self.vertex_count()} vertices")
        if self.face_type == "hex" and self.vertex_count() != 6:
            errors.append(f"Face {self.id} is hex but has {self.vertex_count()} vertices")
        return errors
