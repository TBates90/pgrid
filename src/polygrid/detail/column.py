"""3D column builders for planar polygrids.

This module keeps the geometry-only extrusion logic out of rendering
code so small 3D prototypes can be tested, serialised, and displayed
in notebooks before any heavier viewer integration is attempted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..building.builders import _hex_corners
from ..core.geometry import boundary_vertex_cycle, face_vertex_cycle
from ..core.models import Edge, Face, Vertex
from ..core.polygrid import PolyGrid


@dataclass(frozen=True)
class _FaceSpec:
    id: str
    face_type: str
    vertex_ids: tuple[str, ...]
    metadata: dict


def _build_polygrid_from_face_specs(
    vertices: Iterable[Vertex],
    faces: Iterable[_FaceSpec],
    *,
    metadata: dict | None = None,
) -> PolyGrid:
    vertex_list = list(vertices)
    face_specs = list(faces)
    edge_map: dict[tuple[str, str], Edge] = {}
    built_faces: list[Face] = []

    for spec in face_specs:
        edge_ids: list[str] = []
        count = len(spec.vertex_ids)
        for idx in range(count):
            a = spec.vertex_ids[idx]
            b = spec.vertex_ids[(idx + 1) % count]
            key = tuple(sorted((a, b)))
            edge = edge_map.get(key)
            if edge is None:
                edge = Edge(id=f"e{len(edge_map) + 1}", vertex_ids=key, face_ids=(spec.id,))
            else:
                edge = Edge(id=edge.id, vertex_ids=edge.vertex_ids, face_ids=edge.face_ids + (spec.id,))
            edge_map[key] = edge
            edge_ids.append(edge.id)

        built_faces.append(
            Face(
                id=spec.id,
                face_type=spec.face_type,
                vertex_ids=spec.vertex_ids,
                edge_ids=tuple(edge_ids),
                metadata=dict(spec.metadata),
            )
        )

    return PolyGrid(
        vertex_list,
        edge_map.values(),
        built_faces,
        metadata=metadata or {},
    )


def _require_positioned_2d(grid: PolyGrid) -> None:
    missing = [vertex.id for vertex in grid.vertices.values() if not vertex.has_position()]
    if missing:
        raise ValueError(
            "extrude_polygrid_column requires positioned 2D vertices; "
            f"missing positions for {len(missing)} vertices"
        )


def extrude_polygrid_column(
    grid: PolyGrid,
    height: float,
    *,
    include_base: bool = True,
    name: str = "polygrid-column",
) -> PolyGrid:
    """Extrude a planar polygrid into a shallow 3D column.

    The top surface preserves the source grid's face partitioning.
    Only the outer shell is extruded into side walls; internal edges do
    not produce vertical faces.
    """
    if height <= 0:
        raise ValueError("height must be > 0")

    _require_positioned_2d(grid)
    boundary = boundary_vertex_cycle(grid.edges.values())
    if len(boundary) < 3:
        raise ValueError("extrude_polygrid_column requires a simple outer boundary")

    vertices: list[Vertex] = []
    base_ids: dict[str, str] = {}
    top_ids: dict[str, str] = {}

    for vertex in grid.vertices.values():
        base_id = f"{vertex.id}_b"
        top_id = f"{vertex.id}_t"
        base_ids[vertex.id] = base_id
        top_ids[vertex.id] = top_id
        base_z = vertex.z if vertex.z is not None else 0.0
        vertices.append(Vertex(base_id, vertex.x, vertex.y, base_z))
        vertices.append(Vertex(top_id, vertex.x, vertex.y, base_z + height))

    face_specs: list[_FaceSpec] = []
    top_face_ids: list[str] = []
    side_face_ids: list[str] = []

    for face in grid.faces.values():
        ordered = tuple(top_ids[vid] for vid in face_vertex_cycle(face, grid.edges.values()))
        top_face_ids.append(face.id)
        metadata = dict(face.metadata)
        metadata.update({"surface_role": "top", "source_face_id": face.id})
        face_specs.append(_FaceSpec(face.id, face.face_type, ordered, metadata))

    for idx in range(len(boundary)):
        a = boundary[idx]
        b = boundary[(idx + 1) % len(boundary)]
        face_id = f"wall_{idx + 1}"
        side_face_ids.append(face_id)
        face_specs.append(
            _FaceSpec(
                face_id,
                "quad",
                (base_ids[a], base_ids[b], top_ids[b], top_ids[a]),
                {
                    "surface_role": "side",
                    "boundary_edge": (a, b),
                },
            )
        )

    base_face_id: str | None = None
    if include_base:
        base_face_id = "base"
        face_specs.append(
            _FaceSpec(
                base_face_id,
                f"{len(boundary)}-gon",
                tuple(base_ids[vid] for vid in reversed(boundary)),
                {"surface_role": "base"},
            )
        )

    column_metadata = dict(grid.metadata)
    column_metadata.update(
        {
            "generator": name,
            "source_generator": grid.metadata.get("generator"),
            "column_height": height,
            "source_face_count": len(grid.faces),
            "source_vertex_count": len(grid.vertices),
            "top_face_ids": top_face_ids,
            "side_face_ids": side_face_ids,
            "base_face_id": base_face_id,
        }
    )
    return _build_polygrid_from_face_specs(vertices, face_specs, metadata=column_metadata)


def build_hex_prism(
    *,
    radius: float = 1.0,
    height: float = 0.35,
    include_base: bool = True,
) -> PolyGrid:
    """Build a single shallow hex prism that reads like a tile."""
    if radius <= 0:
        raise ValueError("radius must be > 0")
    base_vertices = [
        Vertex(f"v{idx + 1}", x, y, 0.0)
        for idx, (x, y) in enumerate(_hex_corners((0.0, 0.0), radius))
    ]
    base_face = _FaceSpec(
        id="footprint",
        face_type="hex",
        vertex_ids=tuple(vertex.id for vertex in base_vertices),
        metadata={"surface_role": "footprint"},
    )
    footprint = _build_polygrid_from_face_specs(
        base_vertices,
        [base_face],
        metadata={"generator": "hex-footprint", "radius": radius, "sides": 6},
    )
    prism = extrude_polygrid_column(
        footprint,
        height,
        include_base=include_base,
        name="hex-prism",
    )
    prism.metadata["radius"] = radius
    prism.metadata["tile_like"] = True
    return prism