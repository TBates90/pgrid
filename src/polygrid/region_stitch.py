# TODO REMOVE — Not used by any live script. Phase 11C stitched detail grids.
"""Stitched sub-grid terrain — merge detail grids for adjacent tiles.

Phase 11C — the highest-quality terrain generation mode.  Adjacent
tiles' detail grids are merged into a single :class:`PolyGrid` so
that terrain algorithms (mountains, rivers, pipelines) operate on one
continuous surface with no tile-boundary discontinuities.

Workflow
--------
1. **Transform** each detail grid's local vertices to 3-D globe space
   (using the parent tile's tangent-plane transform, as in Phase 11A).
2. **Project** every 3-D point onto a shared tangent plane (gnomonic
   projection centred at the patch centroid) to get a flat 2-D grid
   that terrain algorithms can consume.
3. **Merge** coincident boundary vertices (within tolerance).
4. **Run** terrain algorithms on the combined flat grid.
5. **Split** results back into per-tile stores.

Public API
----------
- :func:`stitch_detail_grids` — merge detail grids for a group of tiles
- :func:`generate_terrain_on_stitched` — terrain gen on combined grid
- :func:`split_terrain_to_tiles` — distribute results back
- :func:`generate_stitched_patch_terrain` — end-to-end convenience
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .algorithms import get_face_adjacency
from .detail_terrain_3d import (
    Terrain3DSpec,
    _normalize,
    _tangent_basis,
)
from .geometry import face_center
from .heightmap import sample_noise_field, smooth_field
from .models import Edge, Face, Vertex
from .noise import fbm, ridged_noise
from .polygrid import PolyGrid
from .tile_data import FieldDef, TileDataStore, TileSchema
from .tile_detail import DetailGridCollection, TileDetailSpec


# ═══════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════

# Mapping from combined face_id → (original_tile_id, original_sub_face_id)
FaceMapping = Dict[str, Tuple[str, str]]


# ═══════════════════════════════════════════════════════════════════
# 11C.1 — stitch_detail_grids
# ═══════════════════════════════════════════════════════════════════

def stitch_detail_grids(
    collection: DetailGridCollection,
    globe_grid: PolyGrid,
    face_ids: List[str],
    *,
    merge_tolerance: float = 1e-4,
) -> Tuple[PolyGrid, FaceMapping]:
    """Merge detail grids for a group of adjacent globe tiles.

    Each tile's detail grid vertices are:
    1. Transformed to 3-D globe-surface positions.
    2. Projected onto a shared tangent plane (gnomonic projection at
       the group centroid).
    3. Merged where boundary vertices coincide (within *merge_tolerance*).

    Parameters
    ----------
    collection : DetailGridCollection
        Must contain grids for all *face_ids*.
    globe_grid : PolyGrid
        Globe grid with ``center_3d`` / ``normal_3d`` metadata.
    face_ids : list[str]
        Globe tile ids to stitch together.
    merge_tolerance : float
        Max 2-D distance for merging coincident vertices.

    Returns
    -------
    (PolyGrid, FaceMapping)
        The combined grid and a mapping
        ``{combined_face_id: (tile_id, sub_face_id)}``.
    """
    if not face_ids:
        raise ValueError("face_ids must be non-empty")

    # ── 1. Compute group centroid in 3-D ────────────────────────────
    centres_3d: Dict[str, Tuple[float, float, float]] = {}
    for fid in face_ids:
        gf = globe_grid.faces.get(fid)
        if gf is None:
            raise KeyError(f"Globe face '{fid}' not found")
        c3 = gf.metadata.get("center_3d")
        if c3 is None:
            raise ValueError(f"Globe face '{fid}' has no center_3d metadata")
        centres_3d[fid] = tuple(c3)

    # Mean of tile centres → centroid direction
    cx = sum(c[0] for c in centres_3d.values()) / len(face_ids)
    cy = sum(c[1] for c in centres_3d.values()) / len(face_ids)
    cz = sum(c[2] for c in centres_3d.values()) / len(face_ids)
    centroid_3d = _normalize((cx, cy, cz), 1.0)

    # Tangent frame at centroid
    tu, tv = _tangent_basis(centroid_3d)

    # ── 2. Transform each tile's detail-grid vertices to 3-D then 2-D ──
    # For each tile: local vertex → 3-D globe point → gnomonic 2-D
    all_vertices: Dict[str, Vertex] = {}  # combined_vid → Vertex
    all_edges: Dict[str, Edge] = {}
    all_faces: Dict[str, Face] = {}
    face_mapping: FaceMapping = {}

    # vertex 3-D cache per tile: {tile_id: {local_vid: (x3, y3, z3)}}
    vertex_3d_cache: Dict[str, Dict[str, Tuple[float, float, float]]] = {}

    for tile_id in face_ids:
        detail_grid = collection.grids[tile_id]
        globe_face = globe_grid.faces[tile_id]
        tile_centre = centres_3d[tile_id]
        normal_3d = globe_face.metadata.get("normal_3d")
        if normal_3d is None:
            raise ValueError(f"Globe face '{tile_id}' has no normal_3d")

        # Compute transform parameters for this tile (same as 11A)
        tile_tu, tile_tv = _tangent_basis(normal_3d)

        # Local extent for scaling
        xs = [v.x for v in detail_grid.vertices.values() if v.x is not None]
        ys = [v.y for v in detail_grid.vertices.values() if v.y is not None]
        local_extent = max(
            max(xs) - min(xs), max(ys) - min(ys), 1e-6,
        ) if xs and ys else 1e-6

        # Angular size of the globe tile
        tcx, tcy, tcz = tile_centre
        c_len = math.sqrt(tcx * tcx + tcy * tcy + tcz * tcz)
        globe_verts = [
            globe_grid.vertices[vid]
            for vid in globe_face.vertex_ids
            if vid in globe_grid.vertices
        ]
        angles = []
        for v in globe_verts:
            if v.x is None or v.y is None or v.z is None:
                continue
            v_len = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
            denom = c_len * v_len
            if denom < 1e-15:
                continue
            dot = (tcx * v.x + tcy * v.y + tcz * v.z) / denom
            dot = max(-1.0, min(1.0, dot))
            angles.append(math.acos(dot))
        tile_angular_radius = (sum(angles) / len(angles)) if angles else 0.15
        scale = tile_angular_radius / (local_extent / 2.0) if local_extent > 1e-6 else 1.0

        # Transform every vertex: local → 3-D → gnomonic 2-D
        v3d_map: Dict[str, Tuple[float, float, float]] = {}
        local_vid_to_combined: Dict[str, str] = {}

        for vid, vertex in detail_grid.vertices.items():
            if vertex.x is None or vertex.y is None:
                continue
            lx, ly = vertex.x, vertex.y

            # Local → 3-D (on sphere)
            dx = lx * scale
            dy = ly * scale
            px = tcx + dx * tile_tu[0] + dy * tile_tv[0]
            py = tcy + dx * tile_tu[1] + dy * tile_tv[1]
            pz = tcz + dx * tile_tu[2] + dy * tile_tv[2]
            p3d = _normalize((px, py, pz), 1.0)
            v3d_map[vid] = p3d

            # 3-D → gnomonic 2-D on the group tangent plane
            gx, gy = _gnomonic_project(p3d, centroid_3d, tu, tv)

            combined_vid = f"{tile_id}_{vid}"
            all_vertices[combined_vid] = Vertex(combined_vid, gx, gy)
            local_vid_to_combined[vid] = combined_vid

        vertex_3d_cache[tile_id] = v3d_map

        # Prefix edges
        for eid, edge in detail_grid.edges.items():
            a_local, b_local = edge.vertex_ids
            a_comb = local_vid_to_combined.get(a_local)
            b_comb = local_vid_to_combined.get(b_local)
            if a_comb is None or b_comb is None:
                continue
            combined_eid = f"{tile_id}_{eid}"
            combined_fids = tuple(f"{tile_id}_{fid}" for fid in edge.face_ids)
            all_edges[combined_eid] = Edge(combined_eid, (a_comb, b_comb), combined_fids)

        # Prefix faces
        for fid, face in detail_grid.faces.items():
            combined_fid = f"{tile_id}_{fid}"
            combined_vids = tuple(
                local_vid_to_combined[vid]
                for vid in face.vertex_ids
                if vid in local_vid_to_combined
            )
            combined_eids = tuple(f"{tile_id}_{eid}" for eid in face.edge_ids)
            combined_nids = tuple(f"{tile_id}_{nid}" for nid in face.neighbor_ids)
            all_faces[combined_fid] = Face(
                id=combined_fid,
                face_type=face.face_type,
                vertex_ids=combined_vids,
                edge_ids=combined_eids,
                neighbor_ids=combined_nids,
            )
            face_mapping[combined_fid] = (tile_id, fid)

    # ── 3. Merge coincident boundary vertices ───────────────────────
    # Boundary vertices of adjacent tiles may project to very close
    # positions.  Merge them.
    _merge_close_vertices(all_vertices, all_edges, all_faces, merge_tolerance)

    combined = PolyGrid(
        all_vertices.values(),
        all_edges.values(),
        all_faces.values(),
        metadata={
            "generator": "region_stitch",
            "source_tiles": list(face_ids),
            "tile_count": len(face_ids),
        },
    )

    return combined, face_mapping


# ═══════════════════════════════════════════════════════════════════
# 11C.2 — generate_terrain_on_stitched
# ═══════════════════════════════════════════════════════════════════

def generate_terrain_on_stitched(
    combined_grid: PolyGrid,
    face_mapping: FaceMapping,
    globe_grid: PolyGrid,
    globe_store: TileDataStore,
    spec: Optional[Terrain3DSpec] = None,
    *,
    elevation_field: str = "elevation",
) -> TileDataStore:
    """Generate terrain on a stitched combined grid.

    Uses 2-D noise sampling on the projected coordinates.  The
    combined grid's vertices are already in a shared gnomonic-projection
    space, so standard 2-D terrain algorithms produce continuous
    terrain across former tile boundaries.

    Parameters
    ----------
    combined_grid : PolyGrid
        Output of :func:`stitch_detail_grids`.
    face_mapping : FaceMapping
        ``{combined_face_id: (tile_id, sub_face_id)}``
    globe_grid : PolyGrid
    globe_store : TileDataStore
    spec : Terrain3DSpec, optional
    elevation_field : str

    Returns
    -------
    TileDataStore
        Store with ``"elevation"`` for every face in the combined grid.
    """
    if spec is None:
        spec = Terrain3DSpec()

    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=combined_grid, schema=schema)

    seed = spec.seed
    total_noise_weight = spec.fbm_weight + spec.ridge_weight
    if total_noise_weight < 1e-10:
        total_noise_weight = 1.0

    for cfid in combined_grid.faces:
        tile_id, _ = face_mapping[cfid]
        parent_elev = globe_store.get(tile_id, elevation_field)

        face = combined_grid.faces[cfid]
        c = face_center(combined_grid.vertices, face)
        if c is None:
            store.set(cfid, "elevation", parent_elev)
            continue
        x, y = c

        # Layer 1: rolling terrain via 2-D fbm
        fbm_val = 0.0
        if spec.fbm_weight > 0:
            fbm_val = fbm(
                x, y,
                octaves=spec.noise_octaves,
                frequency=spec.noise_frequency,
                seed=seed,
            )

        # Layer 2: ridges
        ridge_val = 0.0
        if spec.ridge_weight > 0:
            ridge_val = ridged_noise(
                x, y,
                octaves=spec.ridge_octaves,
                frequency=spec.ridge_frequency,
                seed=seed + 7919,
            )

        combined = (
            spec.fbm_weight * fbm_val
            + spec.ridge_weight * ridge_val
        ) / total_noise_weight

        elevation = (
            parent_elev * spec.base_weight
            + combined * spec.amplitude * (1.0 - spec.base_weight)
        )
        store.set(cfid, "elevation", elevation)

    # Smooth across the combined grid — this is the key advantage:
    # smoothing crosses former tile boundaries seamlessly
    if spec.boundary_smoothing > 0:
        smooth_field(
            combined_grid, store, "elevation",
            iterations=spec.boundary_smoothing,
            self_weight=0.6,
        )

    return store


# ═══════════════════════════════════════════════════════════════════
# 11C.3 — split_terrain_to_tiles
# ═══════════════════════════════════════════════════════════════════

def split_terrain_to_tiles(
    combined_store: TileDataStore,
    face_mapping: FaceMapping,
    collection: DetailGridCollection,
    *,
    elevation_field: str = "elevation",
) -> None:
    """Distribute combined-grid elevation back into per-tile stores.

    For each face in *face_mapping*, look up its combined-grid elevation
    and write it into the correct tile's :class:`TileDataStore`.

    Parameters
    ----------
    combined_store : TileDataStore
        Store from :func:`generate_terrain_on_stitched`.
    face_mapping : FaceMapping
        ``{combined_face_id: (tile_id, sub_face_id)}``
    collection : DetailGridCollection
        Tile stores will be created / updated in-place.
    elevation_field : str
        Name of the elevation field.
    """
    # Group by tile_id
    tile_elevations: Dict[str, Dict[str, float]] = {}
    for cfid, (tile_id, sub_fid) in face_mapping.items():
        elev = combined_store.get(cfid, elevation_field)
        tile_elevations.setdefault(tile_id, {})[sub_fid] = elev

    for tile_id, sub_elevs in tile_elevations.items():
        detail_grid = collection.grids[tile_id]

        # Create or update the tile store
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=detail_grid, schema=schema)
        for sub_fid, elev in sub_elevs.items():
            store.set(sub_fid, "elevation", elev)
        collection._stores[tile_id] = store


# ═══════════════════════════════════════════════════════════════════
# 11C.4 — generate_stitched_patch_terrain  (end-to-end)
# ═══════════════════════════════════════════════════════════════════

def generate_stitched_patch_terrain(
    collection: DetailGridCollection,
    globe_grid: PolyGrid,
    globe_store: TileDataStore,
    face_ids: List[str],
    spec: Optional[Terrain3DSpec] = None,
    *,
    merge_tolerance: float = 1e-4,
    elevation_field: str = "elevation",
) -> TileDataStore:
    """End-to-end stitched terrain for a group of tiles.

    1. Stitch detail grids into one combined :class:`PolyGrid`.
    2. Generate terrain on the combined grid.
    3. Split results back into per-tile stores in *collection*.

    Parameters
    ----------
    collection : DetailGridCollection
    globe_grid : PolyGrid
    globe_store : TileDataStore
    face_ids : list[str]
        Globe tile ids to process together.
    spec : Terrain3DSpec, optional
    merge_tolerance : float
    elevation_field : str

    Returns
    -------
    TileDataStore
        The combined-grid store (also written back to *collection*).
    """
    combined_grid, face_mapping = stitch_detail_grids(
        collection, globe_grid, face_ids,
        merge_tolerance=merge_tolerance,
    )

    combined_store = generate_terrain_on_stitched(
        combined_grid, face_mapping,
        globe_grid, globe_store,
        spec, elevation_field=elevation_field,
    )

    split_terrain_to_tiles(
        combined_store, face_mapping, collection,
        elevation_field=elevation_field,
    )

    return combined_store


# ═══════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════

def _gnomonic_project(
    point_3d: Tuple[float, float, float],
    centre_3d: Tuple[float, float, float],
    tangent_u: Tuple[float, float, float],
    tangent_v: Tuple[float, float, float],
) -> Tuple[float, float]:
    """Gnomonic (central) projection of a 3-D sphere point onto a tangent plane.

    The tangent plane is defined by *centre_3d* (contact point) and
    the orthonormal basis (*tangent_u*, *tangent_v*).  The projection
    maps any point on the same hemisphere to a 2-D coordinate.

    Parameters
    ----------
    point_3d : tuple
        ``(x, y, z)`` on the unit sphere.
    centre_3d : tuple
        Contact point of the tangent plane on the unit sphere.
    tangent_u, tangent_v : tuple
        Orthonormal basis vectors on the tangent plane.

    Returns
    -------
    (u, v) : tuple of float
        2-D coordinates on the tangent plane.
    """
    px, py, pz = point_3d
    cx, cy, cz = centre_3d

    # dot(point, centre) — how far from the tangent-plane normal
    d = px * cx + py * cy + pz * cz
    if d < 1e-12:
        # Point is on or behind the tangent plane — degenerate
        d = 1e-12

    # Gnomonic projection: scale = 1 / dot(point, centre)
    # Projected = (point / dot) projected onto tangent plane
    inv_d = 1.0 / d
    # Vector from centre to projected point on tangent plane
    qx = px * inv_d - cx
    qy = py * inv_d - cy
    qz = pz * inv_d - cz

    # Decompose into tangent basis
    u = qx * tangent_u[0] + qy * tangent_u[1] + qz * tangent_u[2]
    v = qx * tangent_v[0] + qy * tangent_v[1] + qz * tangent_v[2]

    return (u, v)


def _merge_close_vertices(
    vertices: Dict[str, Vertex],
    edges: Dict[str, Edge],
    faces: Dict[str, Face],
    tolerance: float,
) -> None:
    """Merge vertices that are closer than *tolerance* in 2-D.

    Modifies *vertices*, *edges*, and *faces* dicts **in-place**.
    Uses a simple spatial-hash bucket approach.

    This is O(n) expected with bucket size ~ tolerance.
    """
    if tolerance <= 0:
        return

    # Build spatial hash
    inv_tol = 1.0 / tolerance
    buckets: Dict[Tuple[int, int], List[str]] = {}
    for vid, v in vertices.items():
        if v.x is None or v.y is None:
            continue
        bx = int(math.floor(v.x * inv_tol))
        by = int(math.floor(v.y * inv_tol))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (bx + dx, by + dy)
                buckets.setdefault(key, []).append(vid)

    # Build merge map: vid → canonical_vid
    merge_map: Dict[str, str] = {}
    merged_set: Set[str] = set()

    vlist = list(vertices.keys())
    for vid in vlist:
        if vid in merged_set:
            continue
        v = vertices[vid]
        if v.x is None or v.y is None:
            continue
        bx = int(math.floor(v.x * inv_tol))
        by = int(math.floor(v.y * inv_tol))
        candidates = buckets.get((bx, by), [])
        for other_vid in candidates:
            if other_vid == vid or other_vid in merged_set:
                continue
            ov = vertices[other_vid]
            if ov.x is None or ov.y is None:
                continue
            dist = math.sqrt((v.x - ov.x) ** 2 + (v.y - ov.y) ** 2)
            if dist < tolerance:
                merge_map[other_vid] = vid
                merged_set.add(other_vid)

    if not merge_map:
        return

    def _canonical(vid: str) -> str:
        visited: Set[str] = set()
        while vid in merge_map:
            if vid in visited:
                break
            visited.add(vid)
            vid = merge_map[vid]
        return vid

    # Remove merged vertices
    for vid in merged_set:
        vertices.pop(vid, None)

    # Remap edges
    edge_by_pair: Dict[Tuple[str, str], Edge] = {}
    old_to_new_edge: Dict[str, str] = {}

    for eid, edge in list(edges.items()):
        a = _canonical(edge.vertex_ids[0])
        b = _canonical(edge.vertex_ids[1])
        if a == b:
            # Degenerate edge — remove
            edges.pop(eid, None)
            continue
        key = (min(a, b), max(a, b))
        fids = edge.face_ids
        if key in edge_by_pair:
            existing = edge_by_pair[key]
            # Merge face references
            all_fids = list(existing.face_ids) + list(fids)
            seen: Set[str] = set()
            unique: List[str] = []
            for f in all_fids:
                if f not in seen:
                    seen.add(f)
                    unique.append(f)
            edge_by_pair[key] = Edge(existing.id, existing.vertex_ids, tuple(unique))
            old_to_new_edge[eid] = existing.id
        else:
            new_edge = Edge(eid, key, fids)
            edge_by_pair[key] = new_edge
            old_to_new_edge[eid] = eid

    edges.clear()
    for e in edge_by_pair.values():
        edges[e.id] = e

    # Remap faces
    for fid in list(faces.keys()):
        face = faces[fid]
        new_vids = tuple(_canonical(vid) for vid in face.vertex_ids)
        new_eids = tuple(
            old_to_new_edge.get(eid, eid) for eid in face.edge_ids
            if old_to_new_edge.get(eid, eid) in edges
        )
        new_nids = face.neighbor_ids  # cross-tile neighbors not updated here
        faces[fid] = Face(
            id=fid,
            face_type=face.face_type,
            vertex_ids=new_vids,
            edge_ids=new_eids,
            neighbor_ids=new_nids,
            metadata=face.metadata,
        )
