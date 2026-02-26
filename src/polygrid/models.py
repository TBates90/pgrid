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


@dataclass(frozen=True)
class MacroEdge:
    """One side of the polygrid's outer polygon.

    *id* is an integer 0 … N-1 (N = number of sides: 5 or 6).
    *vertex_ids* is an ordered tuple of boundary vertex ids from
    corner_start to corner_end inclusive.
    *edge_ids* is an ordered tuple of micro-edge ids along this side.
    *corner_start* / *corner_end* are the corner vertex ids shared
    with adjacent macro-edges.
    """

    id: int
    vertex_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    corner_start: str
    corner_end: str
