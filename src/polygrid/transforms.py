"""Topological transforms that operate on PolyGrids.

Each transform is a function ``PolyGrid → OverlayData`` (or similar)
that computes derived geometry without mutating the source grid.

The overlay model keeps the transform output separate from the grid
topology, so that transforms can be applied, visualised, and stripped
without altering the original mesh.

Architecture
------------
- A *transform* is a plain function: ``grid → overlay``.
- An :class:`Overlay` holds transform results (points, segments, regions)
  that can be drawn on top of a grid.
- :func:`apply_voronoi` is the first concrete transform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .geometry import face_center
from .models import Face, Vertex
from .polygrid import PolyGrid


# ═══════════════════════════════════════════════════════════════════
# Overlay data model
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OverlayPoint:
    """A labelled point in world space."""
    id: str
    x: float
    y: float
    label: str = ""
    source_face_id: str = ""


@dataclass
class OverlaySegment:
    """A line segment between two overlay points."""
    id: str
    start: Tuple[float, float]
    end: Tuple[float, float]
    source_edge_id: str = ""


@dataclass
class OverlayRegion:
    """A closed polygon (vertex positions, CCW)."""
    id: str
    points: List[Tuple[float, float]]
    source_vertex_id: str = ""


@dataclass
class Overlay:
    """Container for transform output that can be drawn on a grid.

    *kind* identifies the transform type (e.g. ``"voronoi"``).
    """
    kind: str
    points: List[OverlayPoint] = field(default_factory=list)
    segments: List[OverlaySegment] = field(default_factory=list)
    regions: List[OverlayRegion] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# Voronoi (dual) transform
# ═══════════════════════════════════════════════════════════════════

def apply_voronoi(grid: PolyGrid) -> Overlay:
    """Compute the Voronoi dual of a PolyGrid.

    For a polygonal mesh the Voronoi dual is straightforward:
    - **Sites** (overlay points) are the face centroids.
    - **Edges** (overlay segments) connect centroids of adjacent faces
      (i.e. faces that share an edge in the primal mesh).
    - **Regions** (overlay regions) are the dual cells around each
      primal vertex — formed by the ring of centroids of the faces
      incident on that vertex, in angular order.

    Boundary handling: edges with only one face get a segment from the
    face centroid to the midpoint of the boundary edge (clipped to the
    boundary rather than extending to infinity).
    """
    overlay = Overlay(kind="voronoi")

    # 1. Compute face centroids → overlay points
    centroids: Dict[str, Tuple[float, float]] = {}
    for face in grid.faces.values():
        c = face_center(grid.vertices, face)
        if c is None:
            continue
        centroids[face.id] = c
        overlay.points.append(OverlayPoint(
            id=f"vc_{face.id}",
            x=c[0], y=c[1],
            label=face.id,
            source_face_id=face.id,
        ))

    # 2. Dual edges — one per primal edge
    seg_idx = 0
    boundary_midpoints: Dict[str, Tuple[float, float]] = {}  # edge_id → midpoint
    for edge in grid.edges.values():
        fids = [fid for fid in edge.face_ids if fid in centroids]
        if len(fids) == 2:
            c0 = centroids[fids[0]]
            c1 = centroids[fids[1]]
            overlay.segments.append(OverlaySegment(
                id=f"vs_{seg_idx}",
                start=c0, end=c1,
                source_edge_id=edge.id,
            ))
            seg_idx += 1
        elif len(fids) == 1:
            # Boundary edge: segment from centroid to edge midpoint
            c0 = centroids[fids[0]]
            v0 = grid.vertices[edge.vertex_ids[0]]
            v1 = grid.vertices[edge.vertex_ids[1]]
            if v0.has_position() and v1.has_position():
                mid = ((v0.x + v1.x) / 2, (v0.y + v1.y) / 2)
                boundary_midpoints[edge.id] = mid
                overlay.segments.append(OverlaySegment(
                    id=f"vs_{seg_idx}",
                    start=c0, end=mid,
                    source_edge_id=edge.id,
                ))
                seg_idx += 1

    # 3. Dual regions — one per primal vertex
    #    Each region is the ring of centroids (and boundary midpoints)
    #    around that vertex, in angular order.
    vertex_to_faces: Dict[str, List[str]] = {}
    for face in grid.faces.values():
        for vid in face.vertex_ids:
            vertex_to_faces.setdefault(vid, []).append(face.id)

    vertex_to_boundary_edges: Dict[str, List[str]] = {}
    for edge in grid.edges.values():
        if len(edge.face_ids) < 2:
            for vid in edge.vertex_ids:
                vertex_to_boundary_edges.setdefault(vid, []).append(edge.id)

    for vid, adj_fids in vertex_to_faces.items():
        v = grid.vertices[vid]
        if not v.has_position():
            continue

        # Collect candidate points: face centroids + boundary midpoints
        ring_points: List[Tuple[float, float, str]] = []  # (x, y, source_id)
        for fid in adj_fids:
            if fid in centroids:
                cx, cy = centroids[fid]
                ring_points.append((cx, cy, fid))

        for eid in vertex_to_boundary_edges.get(vid, []):
            if eid in boundary_midpoints:
                mx, my = boundary_midpoints[eid]
                ring_points.append((mx, my, eid))

        if len(ring_points) < 2:
            continue

        # Sort by angle around the vertex
        ring_points.sort(key=lambda p: math.atan2(p[1] - v.y, p[0] - v.x))

        overlay.regions.append(OverlayRegion(
            id=f"vr_{vid}",
            points=[(p[0], p[1]) for p in ring_points],
            source_vertex_id=vid,
        ))

    overlay.metadata["n_sites"] = len(centroids)
    overlay.metadata["n_segments"] = len(overlay.segments)
    overlay.metadata["n_regions"] = len(overlay.regions)

    return overlay


# ═══════════════════════════════════════════════════════════════════
# Partition (coloured-section) transform
# ═══════════════════════════════════════════════════════════════════

def apply_partition(grid: PolyGrid, n_sections: int = 8) -> Overlay:
    """Partition faces of *grid* into *n_sections* coloured groups.

    The algorithm divides the angular space around the grid centroid
    into *n_sections* equal sectors.  Each face is assigned to the
    sector that contains its centroid.

    The resulting :class:`Overlay` has:
    - **regions**: one per face, coloured by section assignment.
      Each region stores the face polygon points.
    - **metadata**: includes ``section_assignments`` (face_id → section
      index) and ``n_sections``.

    This is designed to replace the Voronoi overlay with coloured
    polygons that show the different generated sections.
    """
    overlay = Overlay(kind="partition")

    # Compute face centroids
    centroids: Dict[str, Tuple[float, float]] = {}
    for face in grid.faces.values():
        c = face_center(grid.vertices, face)
        if c is not None:
            centroids[face.id] = c

    if not centroids:
        return overlay

    # Grid centroid (average of all vertex positions)
    xs = [v.x for v in grid.vertices.values() if v.has_position()]
    ys = [v.y for v in grid.vertices.values() if v.has_position()]
    gcx = sum(xs) / len(xs)
    gcy = sum(ys) / len(ys)

    # Assign each face to an angular sector
    sector_width = 2.0 * math.pi / n_sections
    assignments: Dict[str, int] = {}

    for fid, (cx, cy) in centroids.items():
        angle = math.atan2(cy - gcy, cx - gcx)   # [-π, π)
        # Shift to [0, 2π)
        if angle < 0:
            angle += 2.0 * math.pi
        sector = int(angle / sector_width) % n_sections
        assignments[fid] = sector

    # Build overlay regions — one per face, tagged with section index
    for face in grid.faces.values():
        if face.id not in assignments:
            continue
        pts: List[Tuple[float, float]] = []
        for vid in face.vertex_ids:
            v = grid.vertices[vid]
            if v.has_position():
                pts.append((v.x, v.y))
        if len(pts) < 3:
            continue
        region = OverlayRegion(
            id=f"part_{face.id}",
            points=pts,
            source_vertex_id=str(assignments[face.id]),  # section index
        )
        overlay.regions.append(region)

    overlay.metadata["n_sections"] = n_sections
    overlay.metadata["section_assignments"] = assignments

    return overlay
