"""Boundary-aware detail terrain generation for Goldberg globe tiles.

This module generates intra-tile terrain that is **continuous** across
Goldberg tile boundaries.  Boundary faces of each detail grid are
interpolated toward their neighbours' elevations so that adjacent tiles
meet seamlessly.

The key to seamless boundaries is **per-edge directional blending**:
each boundary sub-face is assigned to the specific polygon side it faces,
and blended toward the elevation of the neighbour across that side rather
than toward a global mean of all neighbour elevations.

Functions
---------
- :func:`compute_boundary_elevations` — per-tile edge-target elevations
- :func:`classify_detail_faces` — interior / boundary / corner classification
- :func:`compute_neighbor_edge_mapping` — map each neighbour to a polygon edge
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
# 10B.1b — Neighbour-to-edge mapping for per-edge boundary blending
# ═══════════════════════════════════════════════════════════════════

def compute_neighbor_edge_mapping(
    globe_grid: PolyGrid,
    face_id: str,
) -> Dict[str, int]:
    """Map each neighbour of *face_id* to the polygon edge index it shares.

    Edge *k* of the polygon connects vertex *k* to vertex *(k+1) % N*.
    The mapping is computed by checking which two 3D vertices each
    neighbour shares with the face (by coordinate proximity).

    Parameters
    ----------
    globe_grid : PolyGrid
        The globe grid.
    face_id : str
        The face whose neighbours we map.

    Returns
    -------
    dict
        ``{neighbor_face_id: edge_index}``
    """
    face = globe_grid.faces[face_id]
    n = len(face.vertex_ids)

    # 3D positions of this face's vertices
    own_coords = []
    for vid in face.vertex_ids:
        v = globe_grid.vertices[vid]
        own_coords.append((v.x, v.y, v.z))

    result: Dict[str, int] = {}
    for nid in face.neighbor_ids:
        nface = globe_grid.faces[nid]
        # Find which of our vertices are shared (by 3D proximity)
        shared_indices: List[int] = []
        for i, (ox, oy, oz) in enumerate(own_coords):
            for nvid in nface.vertex_ids:
                nv = globe_grid.vertices[nvid]
                dx = ox - nv.x
                dy = oy - nv.y
                dz = oz - nv.z
                if dx * dx + dy * dy + dz * dz < 1e-10:
                    shared_indices.append(i)
                    break

        if len(shared_indices) == 2:
            a, b = sorted(shared_indices)
            # Edge k connects vertex k to vertex (k+1) % N
            if (b - a) == 1:
                edge_idx = a
            elif a == 0 and b == n - 1:
                # Wrapping edge: vertex (n-1) → vertex 0
                edge_idx = n - 1
            else:
                # Non-adjacent vertices — shouldn't happen for a proper polygon
                edge_idx = a
            result[nid] = edge_idx

    return result


def _compute_tutte_edge_angles(
    detail_grid: PolyGrid,
    n_sides: int,
    *,
    corner_vertex_ids: Optional[List[str]] = None,
) -> List[float]:
    """Compute the midpoint angle (radians) of each polygon edge in Tutte space.

    When *corner_vertex_ids* are supplied (from
    :func:`build_goldberg_grid` metadata) the corners are taken
    directly from those vertices, guaranteeing the same edge
    numbering as :meth:`PolyGrid.compute_macro_edges`.  Otherwise
    corners are detected by geometric clustering (legacy path).

    Parameters
    ----------
    detail_grid : PolyGrid
        A detail grid (hex or pent centred).
    n_sides : int
        Number of polygon sides (5 or 6).
    corner_vertex_ids : list of str, optional
        Ordered corner vertex ids.  When given, edge *k* connects
        ``corner_vertex_ids[k]`` to ``corner_vertex_ids[(k+1) % N]``,
        matching macro-edge numbering.

    Returns
    -------
    list of float
        ``edge_angles[k]`` is the angle (radians, atan2 convention)
        of the midpoint of edge *k* (vertex *k* → vertex *k+1*).
    """
    gc = grid_center(detail_grid.vertices)
    gcx, gcy = gc

    # ----- authoritative path: use known corner vertices -----
    if corner_vertex_ids is not None and len(corner_vertex_ids) == n_sides:
        # Compute macro edges so the numbering matches
        # ``PolyGrid.compute_macro_edges``.
        macro_edges = detail_grid.compute_macro_edges(
            n_sides, corner_ids=list(corner_vertex_ids),
        )
        corner_angles: List[float] = []
        for me in macro_edges:
            v = detail_grid.vertices[me.corner_start]
            corner_angles.append(math.atan2(v.y - gcy, v.x - gcx))

        edge_angles: List[float] = []
        for k in range(n_sides):
            a1 = corner_angles[k]
            a2 = corner_angles[(k + 1) % n_sides]
            # Shortest arc midpoint
            diff = a2 - a1
            if diff > math.pi:
                diff -= 2 * math.pi
            elif diff < -math.pi:
                diff += 2 * math.pi
            edge_angles.append(a1 + diff / 2.0)
        return edge_angles

    # ----- legacy fallback: detect corners by clustering -----
    # Identify boundary vertices
    boundary_verts: Set[str] = set()
    for edge in detail_grid.edges.values():
        if len(edge.face_ids) < 2:
            boundary_verts.update(edge.vertex_ids)

    # Find polygon corners: boundary vertices at maximum distance from centre.
    # Each corner of the N-gon may have 2 nearby vertices; we cluster them.
    vert_angles = []
    dists = {}
    for vid in boundary_verts:
        v = detail_grid.vertices[vid]
        dx, dy = v.x - gcx, v.y - gcy
        dists[vid] = math.hypot(dx, dy)
        vert_angles.append((vid, math.atan2(dy, dx), dists[vid]))

    max_d = max(d for _, _, d in vert_angles)
    # Keep only vertices within 2% of max distance (these are at the corners)
    corner_candidates = [
        (vid, ang) for vid, ang, d in vert_angles if d > max_d * 0.98
    ]
    corner_candidates.sort(key=lambda x: x[1])

    # Cluster into n_sides groups by angular proximity
    cluster_gap = math.radians(360.0 / n_sides * 0.4)  # ~40% of side width
    clusters: List[List[float]] = [[corner_candidates[0][1]]]
    for _, ang in corner_candidates[1:]:
        if ang - clusters[-1][-1] < cluster_gap:
            clusters[-1].append(ang)
        else:
            clusters.append([ang])

    # Handle wrap-around: merge first and last cluster if close
    if len(clusters) > n_sides:
        first_ang = clusters[0][0]
        last_ang = clusters[-1][-1]
        if (first_ang + 2 * math.pi) - last_ang < cluster_gap:
            clusters[-1].extend(clusters[0])
            clusters = clusters[1:]

    if len(clusters) != n_sides:
        # Fallback: evenly-spaced corners.  For hex: first corner at π,
        # for pent: first corner at π/2.
        base = math.pi if n_sides == 6 else math.pi / 2
        corner_angles = [base - k * 2 * math.pi / n_sides for k in range(n_sides)]
    else:
        # Average angle per cluster = polygon corner angle
        corner_angles = []
        for c in clusters:
            avg = sum(c) / len(c)
            corner_angles.append(avg)
        # Sort corners in descending angle (counter-clockwise in atan2)
        corner_angles.sort(reverse=True)

    # Edge k midpoint = average of corner k and corner (k+1) % N on the circle
    edge_angles = []
    for k in range(n_sides):
        a1 = corner_angles[k]
        a2 = corner_angles[(k + 1) % n_sides]
        # Handle wrap-around for the last edge
        if a2 > a1:
            a2 -= 2 * math.pi
        mid = (a1 + a2) / 2.0
        edge_angles.append(mid)

    return edge_angles


def _assign_boundary_faces_to_edges(
    detail_grid: PolyGrid,
    boundary_face_ids: Set[str],
    edge_angles: List[float],
    n_sides: int,
) -> Dict[str, int]:
    """Assign each boundary sub-face to the nearest polygon edge.

    Parameters
    ----------
    detail_grid : PolyGrid
        The detail grid.
    boundary_face_ids : set
        Face ids of all boundary sub-faces.
    edge_angles : list of float
        Midpoint angles of each polygon edge (from ``_compute_tutte_edge_angles``).
    n_sides : int
        Number of polygon sides.

    Returns
    -------
    dict
        ``{face_id: edge_index}``
    """
    gc = grid_center(detail_grid.vertices)
    gcx, gcy = gc

    result: Dict[str, int] = {}
    for fid in boundary_face_ids:
        face = detail_grid.faces[fid]
        c = face_center(detail_grid.vertices, face)
        if c is None:
            continue
        cx, cy = c
        face_angle = math.atan2(cy - gcy, cx - gcx)

        # Find the closest edge by angular distance
        best_edge = 0
        best_dist = float("inf")
        for k, ea in enumerate(edge_angles):
            d = abs(face_angle - ea)
            # Wrap to [-π, π]
            if d > math.pi:
                d = 2 * math.pi - d
            if d < best_dist:
                best_dist = d
                best_edge = k
        result[fid] = best_edge

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
    neighbor_edge_map: Optional[Dict[str, int]] = None,
) -> TileDataStore:
    """Generate boundary-aware terrain for a single detail grid.

    Interior faces get ``parent_elevation`` as base; boundary faces
    are interpolated toward the specific neighbour across the polygon
    edge they face, ensuring that adjacent tiles share identical
    boundary elevations.

    Parameters
    ----------
    detail_grid : PolyGrid
        Detail grid (from :func:`build_detail_grid`).
    parent_elevation : float
        Elevation of the parent globe tile.
    neighbor_elevations : dict
        ``{neighbor_face_id: boundary_target_elevation}``
        One entry per globe-level neighbour.
    spec : TileDetailSpec
        Noise and smoothing configuration.
    seed : int
        Base seed for noise.
    neighbor_edge_map : dict, optional
        ``{neighbor_face_id: edge_index}`` mapping each neighbour to the
        polygon edge it shares.  When provided, boundary sub-faces are
        blended toward the **specific** neighbour across the closest edge
        rather than toward the mean of all neighbours.  Computed by
        :func:`compute_neighbor_edge_mapping`.

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

    # --- Per-edge boundary targets ---
    # Build a lookup: edge_index → boundary target elevation.
    # Then assign each boundary/corner sub-face to an edge.
    n_sides = len(detail_grid.faces.get("f0", detail_grid.faces[next(iter(detail_grid.faces))]).vertex_ids)
    # Determine polygon side count from metadata or face type
    parent_ftype = detail_grid.metadata.get("parent_face_id", "")
    # The face type of the detail grid determines the polygon: hex=6, pent=5
    if any(f.face_type == "pent" for f in detail_grid.faces.values()):
        # Pentagon-centred grid: centre face is pent, rest are hex
        n_sides = 5
    else:
        n_sides = 6

    # Build per-edge target map: edge_index → elevation target
    edge_targets: Dict[int, float] = {}
    fallback_target = parent_elevation

    if neighbor_edge_map and neighbor_elevations:
        for nid, edge_idx in neighbor_edge_map.items():
            if nid in neighbor_elevations:
                edge_targets[edge_idx] = neighbor_elevations[nid]
        if edge_targets:
            fallback_target = sum(edge_targets.values()) / len(edge_targets)
    elif neighbor_elevations:
        # No edge mapping — fall back to uniform mean (legacy behaviour)
        fallback_target = sum(neighbor_elevations.values()) / len(neighbor_elevations)

    # Assign boundary sub-faces to polygon edges
    boundary_band = {
        fid for fid, cls in classification.items() if cls != "interior"
    }
    if edge_targets:
        corner_ids = detail_grid.metadata.get("corner_vertex_ids")
        edge_angles = _compute_tutte_edge_angles(
            detail_grid, n_sides, corner_vertex_ids=corner_ids,
        )
        face_edge_map = _assign_boundary_faces_to_edges(
            detail_grid, boundary_band, edge_angles, n_sides,
        )
    else:
        face_edge_map = {}

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
            noise_atten = 1.0
        else:
            # Boundary/corner: lerp toward the target elevation of the
            # specific polygon edge this sub-face belongs to.
            dist = _distance_to_boundary(
                detail_grid, fid, outermost, adj, max_ring,
            )
            # dist=0 → outermost (full blend), dist=max_ring → interior-like
            t = 1.0 - min(dist / max(max_ring, 1), 1.0)

            # Per-edge directional target
            edge_idx = face_edge_map.get(fid)
            if edge_idx is not None and edge_idx in edge_targets:
                target = edge_targets[edge_idx]
            else:
                target = fallback_target

            base = parent_elevation * (1.0 - t) + target * t

            # Attenuate noise at the boundary so that the outermost
            # sub-faces produce identical elevations on both sides of
            # a shared edge (where t → 1, noise → 0).
            noise_atten = 1.0 - t

        elevation = (
            base * spec.base_weight
            + noise_val * spec.amplitude * (1.0 - spec.base_weight) * noise_atten
        )
        store.set(fid, "elevation", elevation)

    # Smooth the inner boundary band to reduce noise discontinuities,
    # but EXCLUDE outermost faces — their elevations are pinned to the
    # shared boundary targets for cross-tile continuity.
    if spec.boundary_smoothing > 0:
        smooth_fids = [
            fid for fid, cls in classification.items()
            if cls != "interior" and fid not in outermost
        ]
        if smooth_fids:
            smooth_field(
                detail_grid, store, "elevation",
                iterations=spec.boundary_smoothing,
                self_weight=0.5,
                face_ids=smooth_fids,
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
    once, builds per-edge neighbour mappings, and then populates every
    detail grid's store with directional boundary blending.

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

    # Pre-compute neighbour→edge mappings for all faces.
    # ``compute_neighbor_edge_mapping`` returns PG-vertex-order edge
    # indices.  When the detail grid has ``corner_vertex_ids`` the
    # terrain generator uses macro-edge numbering (via
    # ``_compute_tutte_edge_angles`` with those corners), so we must
    # translate PG edge indices → macro-edge indices.
    from .tile_uv_align import compute_pg_to_macro_edge_map

    edge_mappings: Dict[str, Dict[str, int]] = {}
    for face_id in collection.grids:
        pg_map = compute_neighbor_edge_mapping(globe_grid, face_id)
        detail_grid = collection.grids[face_id]
        corner_ids = detail_grid.metadata.get("corner_vertex_ids")
        if corner_ids:
            # Ensure macro edges are computed so the conversion can read them
            n_sides = len(corner_ids)
            if not detail_grid.macro_edges:
                detail_grid.compute_macro_edges(n_sides, corner_ids=corner_ids)
            # Translate PG edge → macro edge so the indices match
            # ``_compute_tutte_edge_angles(corner_vertex_ids=...)``
            pg2macro = compute_pg_to_macro_edge_map(
                globe_grid, face_id, detail_grid,
            )
            edge_mappings[face_id] = {
                nid: pg2macro.get(pg_idx, pg_idx)
                for nid, pg_idx in pg_map.items()
            }
        else:
            edge_mappings[face_id] = pg_map

    for face_id, detail_grid in collection.grids.items():
        parent_elev = globe_store.get(face_id, elevation_field)
        neighbor_elevs = boundary_elevs.get(face_id, {})

        store = generate_detail_terrain_bounded(
            detail_grid,
            parent_elev,
            neighbor_elevs,
            spec,
            seed=seed,
            neighbor_edge_map=edge_mappings.get(face_id),
        )
        collection._stores[face_id] = store
