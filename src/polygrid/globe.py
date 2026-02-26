"""Globe grid builder — bridge between the ``models`` Goldberg polyhedron and PolyGrid.

Converts a :class:`~models.objects.goldberg.polyhedron.GoldbergPolyhedron`
into a :class:`PolyGrid` where every Goldberg tile becomes a face.
The resulting grid carries **both** 2-D and 3-D vertex coordinates:

* *x, y* — equirectangular projection ``(longitude_deg, latitude_deg)``
  for compatibility with existing 2-D noise / heightmap / terrain code.
* *z* — the third Cartesian coordinate of the tile vertex on the unit
  sphere, so that :func:`~heightmap.sample_noise_field_3d` can evaluate
  seamless 3-D noise without polar or seam artefacts.

Requires the ``models`` library (optional dependency).  Import this
module inside a ``try / except ImportError`` guard if you want the rest
of polygrid to remain usable without ``models`` installed.

Functions
---------
- :func:`build_globe_grid` — main entry point
- :class:`GlobeGrid` — thin wrapper with 3-D accessor helpers
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from .models import Edge, Face, Vertex
from .polygrid import PolyGrid

try:
    from models.objects.goldberg.polyhedron import GoldbergPolyhedron
    from models.objects.goldberg.tiles import GoldbergPolyhedronTile

    _HAS_MODELS = True
except ImportError:  # pragma: no cover
    _HAS_MODELS = False


# ═══════════════════════════════════════════════════════════════════
# Public helpers
# ═══════════════════════════════════════════════════════════════════

def _require_models() -> None:
    if not _HAS_MODELS:
        raise ImportError(
            "The 'models' library is required for globe grids.  "
            "Install it with: pip install models  "
            "or: pip install polygrid[globe]"
        )


def _face_id(tile: "GoldbergPolyhedronTile") -> str:
    """Stable face id from a Goldberg tile."""
    return f"t{tile.id}"


def _vertex_id(face_id: str, idx: int) -> str:
    """Vertex id for the *idx*-th vertex of a globe face."""
    return f"{face_id}_v{idx}"


def _edge_id(va: str, vb: str) -> str:
    """Canonical edge id from two vertex ids (sorted for uniqueness)."""
    a, b = sorted([va, vb])
    return f"e_{a}_{b}"


# ═══════════════════════════════════════════════════════════════════
# Builder
# ═══════════════════════════════════════════════════════════════════

def build_globe_grid(
    frequency: int,
    *,
    radius: float = 1.0,
) -> "GlobeGrid":
    """Build a :class:`GlobeGrid` from a Goldberg polyhedron of given *frequency*.

    Parameters
    ----------
    frequency : int
        Goldberg subdivision frequency (≥ 1).  Tile count = 10 × freq² + 2.
    radius : float
        Radius of the polyhedron.

    Returns
    -------
    GlobeGrid
        A PolyGrid-compatible globe with one face per Goldberg tile.
    """
    _require_models()

    poly = GoldbergPolyhedron.from_frequency(frequency, radius=radius)

    vertices: Dict[str, Vertex] = {}
    faces: List[Face] = []
    edge_set: Dict[str, Edge] = {}

    # Pre-build a mapping of tile id → face id for neighbour lookup
    tile_face_ids = {tile.id: _face_id(tile) for tile in poly.tiles}

    for tile in poly.tiles:
        fid = tile_face_ids[tile.id]
        face_type = "pent" if tile.kind == "pentagon" else "hex"

        # Create vertices for this tile's polygon
        vert_ids: List[str] = []
        for idx, vtx_3d in enumerate(tile.vertices):
            vid = _vertex_id(fid, idx)
            # 3D Cartesian coords as x, y, z on the vertex
            vertices[vid] = Vertex(
                id=vid,
                x=vtx_3d[0],
                y=vtx_3d[1],
                z=vtx_3d[2],
            )
            vert_ids.append(vid)

        # Build edges from consecutive vertex pairs around the polygon
        face_edge_ids: List[str] = []
        n_verts = len(vert_ids)
        for i in range(n_verts):
            va = vert_ids[i]
            vb = vert_ids[(i + 1) % n_verts]
            eid = _edge_id(va, vb)
            if eid not in edge_set:
                edge_set[eid] = Edge(id=eid, vertex_ids=(va, vb), face_ids=(fid,))
            else:
                # Edge shared with a previously-seen face — add this face
                existing = edge_set[eid]
                edge_set[eid] = Edge(
                    id=eid,
                    vertex_ids=existing.vertex_ids,
                    face_ids=existing.face_ids + (fid,),
                )
            face_edge_ids.append(eid)

        # Neighbour ids from the models library
        neighbor_ids = tuple(tile_face_ids[nid] for nid in tile.neighbor_ids)

        # Per-face metadata carrying 3D properties from the models tile
        meta = {
            "center_3d": tuple(tile.center),
            "normal_3d": tuple(tile.normal),
            "latitude_deg": tile.latitude_deg,
            "longitude_deg": tile.longitude_deg,
            "base_face_index": tile.base_face_index,
            "base_face_grid": tuple(tile.base_face_grid),
            "tile_id": tile.id,
        }

        faces.append(Face(
            id=fid,
            face_type=face_type,
            vertex_ids=tuple(vert_ids),
            edge_ids=tuple(face_edge_ids),
            neighbor_ids=neighbor_ids,
            metadata=meta,
        ))

    grid_metadata = {
        "generator": "globe",
        "frequency": frequency,
        "radius": radius,
        "tile_count": len(faces),
        "pentagon_count": sum(1 for f in faces if f.face_type == "pent"),
        "hexagon_count": sum(1 for f in faces if f.face_type == "hex"),
    }

    return GlobeGrid(
        vertices=vertices.values(),
        edges=edge_set.values(),
        faces=faces,
        metadata=grid_metadata,
        polyhedron=poly,
    )


# ═══════════════════════════════════════════════════════════════════
# GlobeGrid wrapper
# ═══════════════════════════════════════════════════════════════════

class GlobeGrid(PolyGrid):
    """A :class:`PolyGrid` whose faces are tiles on a Goldberg polyhedron.

    Extends PolyGrid with 3-D accessors for rendering and spherical
    noise evaluation.
    """

    def __init__(
        self,
        vertices,
        edges,
        faces,
        metadata=None,
        macro_edges=None,
        *,
        polyhedron: Optional["GoldbergPolyhedron"] = None,
    ) -> None:
        super().__init__(vertices, edges, faces, metadata, macro_edges)
        self._polyhedron = polyhedron

    # ── Properties ──────────────────────────────────────────────────

    @property
    def frequency(self) -> int:
        """Goldberg subdivision frequency."""
        return self.metadata.get("frequency", 0)

    @property
    def radius(self) -> float:
        """Polyhedron radius."""
        return self.metadata.get("radius", 1.0)

    @property
    def polyhedron(self) -> Optional["GoldbergPolyhedron"]:
        """The source :class:`GoldbergPolyhedron`, or *None* if deserialised."""
        return self._polyhedron

    # ── 3-D face accessors ──────────────────────────────────────────

    def tile_3d_center(self, face_id: str) -> Tuple[float, float, float] | None:
        """3-D Cartesian centre of the Goldberg tile."""
        face = self.faces.get(face_id)
        if face is None:
            return None
        return face.metadata.get("center_3d")

    def tile_normal(self, face_id: str) -> Tuple[float, float, float] | None:
        """Outward-pointing unit normal of the Goldberg tile."""
        face = self.faces.get(face_id)
        if face is None:
            return None
        return face.metadata.get("normal_3d")

    def tile_lat_lon(self, face_id: str) -> Tuple[float, float] | None:
        """``(latitude_deg, longitude_deg)`` of the tile centre."""
        face = self.faces.get(face_id)
        if face is None:
            return None
        lat = face.metadata.get("latitude_deg")
        lon = face.metadata.get("longitude_deg")
        if lat is None or lon is None:
            return None
        return (lat, lon)

    def tile_models_id(self, face_id: str) -> int | None:
        """Integer tile id in the models library (for cross-referencing)."""
        face = self.faces.get(face_id)
        if face is None:
            return None
        return face.metadata.get("tile_id")
