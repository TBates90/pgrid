"""Boundary-aware detail terrain generation for Goldberg globe tiles.

This module generates intra-tile terrain that is **continuous** across
Goldberg tile boundaries.  Boundary faces of each detail grid are
interpolated toward their neighbours' elevations so that adjacent tiles
meet seamlessly.

Functions
---------
- :func:`compute_boundary_elevations` — per-tile edge-target elevations
- :func:`classify_detail_faces` — interior / boundary / corner classification
- :func:`generate_detail_terrain_bounded` — single tile, boundary-aware
- :func:`generate_all_detail_terrain` — batch, whole collection
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .algorithms import get_face_adjacency, ring_faces
from .detail_grid import detail_face_count
from .geometry import face_center, grid_center, boundary_vertex_cycle
from .heightmap import smooth_field
from .models import Edge
from .noise import fbm, domain_warp
from .polygrid import PolyGrid
from .tile_data import FieldDef, TileDataStore, TileSchema
from .tile_detail import TileDetailSpec, DetailGridCollection


# ═══════════════════════════════════════════════════════════════════
# 10B.1 — Compute boundary elevations
# ═══════════════════════════════════════════════════════════════════

def compute_boundary_elevations(
    globe_grid: PolyGrid,
    globe_store: TileDataStore,
    *,
    elevation_field: str = "elevation",
) -> Dict[str, Dict[str, float]]:
    """Compute target boundary elevations for each globe tile.

    For every globe tile, each neighbour contributes a boundary
    target — the average of the tile's own elevation and that
    neighbour's elevation.

    Parameters
    ----------
    globe_grid : PolyGrid
        The globe grid.
    globe_store : TileDataStore
        Globe-level tile data with an elevation field.
    elevation_field : str
        Name of the elevation field.

    Returns
    -------
    dict
        ``{face_id: {neighbor_id: boundary_elevation}}``
        For each face, each neighbour produces one boundary target.
    """
    adj = get_face_adjacency(globe_grid)
    result: Dict[str, Dict[str, float]] = {}

    for face_id in globe_grid.faces:
        own_elev = globe_store.get(face_id, elevation_field)
        neighbour_targets: Dict[str, float] = {}
        for nid in adj.get(face_id, []):
            n_elev = globe_store.get(nid, elevation_field)
            neighbour_targets[nid] = (own_elev + n_elev) / 2.0
        result[face_id] = neighbour_targets

    return result


# ═══════════════════════════════════════════════════════════════════
# 10B.2 — Classify detail faces
# ═══════════════════════════════════════════════════════════════════

def _boundary_face_ids(grid: PolyGrid) -> Set[str]:
    """Return the set of face ids that touch at least one boundary edge.

    A boundary edge is one with fewer than 2 face_ids (i.e. it's on
    the grid boundary with no neighbour on the other side).
    """
    boundary: Set[str] = set()
    for edge in grid.edges.values():
        if len(edge.face_ids) < 2:
            for fid in edge.face_ids:
                boundary.add(fid)
    return boundary


def classify_detail_faces(
    detail_grid: PolyGrid,
    boundary_depth: int = 1,
) -> Dict[str, str]:
    """Classify detail-grid faces as interior, boundary, or corner.

    Parameters
    ----------
    detail_grid : PolyGrid
        A detail grid (from :func:`build_detail_grid`).
    boundary_depth : int
        How many rings inward from the grid edge count as boundary.
        ``1`` = only the outermost ring of faces.

    Returns
    -------
    dict
        ``{face_id: "interior" | "boundary" | "corner"}``

    Corner faces are boundary faces that touch a vertex where ≥ 3
    boundary edges meet (i.e. the outer polygon's corners).
    """
    outermost = _boundary_face_ids(detail_grid)

    # Expand the boundary band inward using adjacency
    adj = get_face_adjacency(detail_grid)
    boundary_band: Set[str] = set(outermost)
    frontier = set(outermost)
    for _ in range(boundary_depth - 1):
        next_frontier: Set[str] = set()
        for fid in frontier:
            for nid in adj.get(fid, []):
                if nid not in boundary_band:
                    boundary_band.add(nid)
                    next_frontier.add(nid)
        frontier = next_frontier
        if not frontier:
            break

    # Identify corner faces — boundary faces sitting at the outer polygon's
    # corners.  These faces have more boundary vertices than ordinary edge
    # faces (≥ 4 of their vertices lie on the boundary).
    boundary_edges = [e for e in detail_grid.edges.values() if len(e.face_ids) < 2]
    boundary_verts: Set[str] = set()
    for edge in boundary_edges:
        boundary_verts.update(edge.vertex_ids)

    corner_faces: Set[str] = set()
    for fid in boundary_band:
        face = detail_grid.faces[fid]
        n_bv = sum(1 for vid in face.vertex_ids if vid in boundary_verts)
        # A hex face with ≥ 4 boundary vertices sits at a polygon corner;
        # a pent face with ≥ 4 similarly.  Edge faces have ≤ 3.
        if n_bv >= 4:
            corner_faces.add(fid)

    # Classify
    result: Dict[str, str] = {}
    for fid in detail_grid.faces:
        if fid in corner_faces:
            result[fid] = "corner"
        elif fid in boundary_band:
            result[fid] = "boundary"
        else:
            result[fid] = "interior"

    return result


# ═══════════════════════════════════════════════════════════════════
# 10B.3 — Generate boundary-aware detail terrain for one tile
# ═══════════════════════════════════════════════════════════════════

def _distance_to_boundary(
    grid: PolyGrid,
    face_id: str,
    boundary_faces: Set[str],
    adj: Dict[str, List[str]],
    max_depth: int,
) -> int:
    """BFS distance from *face_id* to the nearest boundary face."""
    if face_id in boundary_faces:
        return 0
    visited = {face_id}
    frontier = [face_id]
    for depth in range(1, max_depth + 1):
        next_f: List[str] = []
        for fid in frontier:
            for nid in adj.get(fid, []):
                if nid in visited:
                    continue
                if nid in boundary_faces:
                    return depth
                visited.add(nid)
                next_f.append(nid)
        frontier = next_f
        if not frontier:
            break
    return max_depth + 1


def generate_detail_terrain_bounded(
    detail_grid: PolyGrid,
    parent_elevation: float,
    neighbor_elevations: Dict[str, float],
    spec: TileDetailSpec,
    *,
    seed: int = 42,
) -> TileDataStore:
    """Generate boundary-aware terrain for a single detail grid.

    Interior faces get ``parent_elevation`` as base; boundary faces
    are interpolated toward the matching neighbour elevation so that
    adjacent tiles meet seamlessly.

    Parameters
    ----------
    detail_grid : PolyGrid
        Detail grid (from :func:`build_detail_grid`).
    parent_elevation : float
        Elevation of the parent globe tile.
    neighbor_elevations : dict
        ``{neighbor_face_id: boundary_target_elevation}``
        One entry per globe-level neighbour.  Boundary faces are
        blended toward the **mean** of these targets (since the
        detail grid does not carry per-edge direction information,
        we use the mean of all neighbours as the boundary target).
    spec : TileDetailSpec
        Noise and smoothing configuration.
    seed : int
        Base seed for noise.

    Returns
    -------
    TileDataStore
        Store with an ``"elevation"`` field for every sub-face.
    """
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=detail_grid, schema=schema)

    # Pre-compute classification and adjacency
    classification = classify_detail_faces(
        detail_grid, boundary_depth=max(1, spec.boundary_smoothing),
    )
    outermost = _boundary_face_ids(detail_grid)
    adj = get_face_adjacency(detail_grid)

    # Boundary target = mean of neighbour boundary elevations
    if neighbor_elevations:
        boundary_target = sum(neighbor_elevations.values()) / len(neighbor_elevations)
    else:
        boundary_target = parent_elevation

    # Determine maximum ring depth for distance calculation
    max_ring = max(1, spec.boundary_smoothing) + 1

    # Pre-compute grid center for distance calculations
    gc = grid_center(detail_grid.vertices)

    tile_seed = seed + spec.seed_offset + hash(
        detail_grid.metadata.get("parent_face_id", "")
    ) % 10000

    for fid in detail_grid.faces:
        face = detail_grid.faces[fid]
        c = face_center(detail_grid.vertices, face)
        if c is None:
            continue
        cx, cy = c

        # Compute noise contribution
        if spec.warp_strength > 0:
            noise_val = domain_warp(
                fbm, cx, cy,
                warp_strength=spec.warp_strength,
                warp_frequency=spec.noise_frequency * 0.5,
                warp_seed_x=tile_seed + 1000,
                warp_seed_y=tile_seed + 2000,
                octaves=spec.noise_octaves,
                frequency=spec.noise_frequency,
                seed=tile_seed,
            )
        else:
            noise_val = fbm(
                cx, cy,
                octaves=spec.noise_octaves,
                frequency=spec.noise_frequency,
                seed=tile_seed,
            )

        cls = classification[fid]
        if cls == "interior":
            base = parent_elevation
        else:
            # Boundary/corner: lerp toward boundary target based on
            # distance from interior.  Outermost ring gets full blend,
            # inner boundary rings get partial.
            dist = _distance_to_boundary(
                detail_grid, fid, outermost, adj, max_ring,
            )
            # dist=0 → outermost (full blend), dist=max_ring → interior-like
            t = 1.0 - min(dist / max(max_ring, 1), 1.0)
            base = parent_elevation * (1.0 - t) + boundary_target * t

        elevation = (
            base * spec.base_weight
            + noise_val * spec.amplitude * (1.0 - spec.base_weight)
        )
        store.set(fid, "elevation", elevation)

    # Smooth boundary band to reduce seam visibility
    if spec.boundary_smoothing > 0:
        boundary_fids = [
            fid for fid, cls in classification.items() if cls != "interior"
        ]
        if boundary_fids:
            smooth_field(
                detail_grid, store, "elevation",
                iterations=spec.boundary_smoothing,
                self_weight=0.5,
                face_ids=boundary_fids,
            )

    return store


# ═══════════════════════════════════════════════════════════════════
# 10B.4 — Batch: generate terrain for entire collection
# ═══════════════════════════════════════════════════════════════════

def generate_all_detail_terrain(
    collection: DetailGridCollection,
    globe_grid: PolyGrid,
    globe_store: TileDataStore,
    spec: Optional[TileDetailSpec] = None,
    *,
    seed: int = 42,
    elevation_field: str = "elevation",
) -> None:
    """Generate boundary-aware terrain for every tile in a collection.

    This is the preferred entry point — it computes boundary targets
    once and then populates every detail grid's store.

    Parameters
    ----------
    collection : DetailGridCollection
        The collection (must already contain grids).
    globe_grid : PolyGrid
        The globe grid.
    globe_store : TileDataStore
        Globe-level elevation data.
    spec : TileDetailSpec, optional
        Overrides the collection's spec if given.
    seed : int
        Base noise seed.
    elevation_field : str
        Name of the elevation field in *globe_store*.
    """
    if spec is None:
        spec = collection.spec

    boundary_elevs = compute_boundary_elevations(
        globe_grid, globe_store, elevation_field=elevation_field,
    )

    for face_id, detail_grid in collection.grids.items():
        parent_elev = globe_store.get(face_id, elevation_field)
        neighbor_elevs = boundary_elevs.get(face_id, {})

        store = generate_detail_terrain_bounded(
            detail_grid,
            parent_elev,
            neighbor_elevs,
            spec,
            seed=seed,
        )
        collection._stores[face_id] = store
