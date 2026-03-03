"""Apron grid construction for seamless tile-boundary rendering.

Phase 18A — extends each tile's detail grid with the outermost
sub-faces from neighbouring tiles ("apron polygons").  When the
extended grid is rendered to a texture, the overlap zone ensures that
adjacent tiles agree on pixel content, eliminating visible seams.

Public API
----------
- :func:`classify_boundary_subfaces` — classify sub-faces as interior / boundary / edge_band
- :func:`compute_edge_subface_mapping` — map sub-faces along a shared globe edge
- :func:`build_apron_grid` — construct extended grid with neighbour edge sub-faces
- :func:`propagate_apron_terrain` — copy + smooth elevation into apron zone
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .algorithms import get_face_adjacency, ring_faces
from .geometry import face_center, grid_center, boundary_vertex_cycle
from .heightmap import smooth_field
from .models import Edge, Face, MacroEdge, Vertex
from .polygrid import PolyGrid
from .tile_data import FieldDef, TileDataStore, TileSchema
from .tile_detail import DetailGridCollection


# ═══════════════════════════════════════════════════════════════════
# 18A.1 — Classify boundary sub-faces
# ═══════════════════════════════════════════════════════════════════

def _boundary_face_ids(grid: PolyGrid) -> Set[str]:
    """Return sub-face ids that touch at least one boundary edge."""
    boundary: Set[str] = set()
    for edge in grid.edges.values():
        if len(edge.face_ids) < 2:
            for fid in edge.face_ids:
                boundary.add(fid)
    return boundary


def classify_boundary_subfaces(
    detail_grid: PolyGrid,
    *,
    edge_band_depth: int = 1,
) -> Dict[str, str]:
    """Classify every sub-face of a detail grid.

    Categories
    ----------
    ``"interior"``
        Not adjacent to any boundary edge and not in the edge band.
    ``"boundary"``
        Touches at least one boundary edge (the outermost ring).
    ``"edge_band"``
        Within *edge_band_depth* rings of the boundary but not itself
        on the outermost ring.  These are the transition zone.

    Parameters
    ----------
    detail_grid : PolyGrid
        A detail grid built by :func:`build_detail_grid`.
    edge_band_depth : int
        How many additional rings inward from the boundary count as
        ``edge_band``.  ``1`` means only the ring immediately inside
        the boundary ring.

    Returns
    -------
    dict
        ``{face_id: "interior" | "boundary" | "edge_band"}``
    """
    outermost = _boundary_face_ids(detail_grid)
    adj = get_face_adjacency(detail_grid)

    # Expand inward from boundary to build the edge_band
    edge_band: Set[str] = set()
    frontier = set(outermost)
    for _ in range(edge_band_depth):
        next_frontier: Set[str] = set()
        for fid in frontier:
            for nid in adj.get(fid, []):
                if nid not in outermost and nid not in edge_band:
                    edge_band.add(nid)
                    next_frontier.add(nid)
        frontier = next_frontier
        if not frontier:
            break

    result: Dict[str, str] = {}
    for fid in detail_grid.faces:
        if fid in outermost:
            result[fid] = "boundary"
        elif fid in edge_band:
            result[fid] = "edge_band"
        else:
            result[fid] = "interior"

    return result


def boundary_subface_ids(detail_grid: PolyGrid) -> Set[str]:
    """Convenience: return the set of boundary sub-face ids."""
    return _boundary_face_ids(detail_grid)


# ═══════════════════════════════════════════════════════════════════
# 18A.2 — Compute neighbour edge sub-face mapping
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EdgeSubfaceMapping:
    """Mapping of sub-faces along a shared macro-edge between two tiles.

    Attributes
    ----------
    face_id_a, face_id_b : str
        The two globe-level tile ids that share an edge.
    subfaces_a : list[str]
        Boundary sub-face ids from tile A that lie along the shared edge,
        ordered by their position along the macro-edge.
    subfaces_b : list[str]
        Corresponding boundary sub-face ids from tile B.
    """
    face_id_a: str
    face_id_b: str
    subfaces_a: List[str] = field(default_factory=list)
    subfaces_b: List[str] = field(default_factory=list)


def _macro_edge_for_neighbour(
    detail_grid: PolyGrid,
    neighbour_id: str,
    globe_grid: PolyGrid,
    face_id: str,
) -> Optional[int]:
    """Find which macro-edge index of *face_id*'s detail grid faces *neighbour_id*.

    Uses the globe grid's macro-edge information.  If macro-edges are not
    computed on the detail grid, returns ``None``.
    """
    # We need to figure out which macro-edge side corresponds to this neighbour.
    # This requires the globe face's edge list: find the globe edge shared
    # between face_id and neighbour_id.
    face_a = globe_grid.faces[face_id]
    face_b = globe_grid.faces[neighbour_id]
    shared_edge = None
    edges_a = set(face_a.edge_ids)
    edges_b = set(face_b.edge_ids)
    common = edges_a & edges_b
    if common:
        shared_edge = common.pop()

    if shared_edge is None:
        return None

    # If the globe face records which macro-edge side each edge belongs to,
    # we could use that.  Otherwise determine from the edge ordering.
    n_sides = len(face_a.vertex_ids)  # 5 or 6

    # The face's edge_ids are ordered around the polygon — edge i connects
    # vertex i to vertex (i+1)%n.  So the macro-edge index is the position
    # of the shared edge in face_a.edge_ids.
    for idx, eid in enumerate(face_a.edge_ids):
        if eid == shared_edge:
            return idx

    return None


def _boundary_faces_along_direction(
    detail_grid: PolyGrid,
    boundary_faces: Set[str],
    direction: Tuple[float, float],
) -> List[str]:
    """Sort boundary sub-faces by their projection onto *direction*.

    This gives an ordered list of boundary sub-faces along one side
    of the detail grid.
    """
    # Compute centroid of each boundary face
    face_projs: List[Tuple[float, str]] = []
    dx, dy = direction
    for fid in boundary_faces:
        face = detail_grid.faces[fid]
        c = face_center(detail_grid.vertices, face)
        if c is None:
            continue
        proj = c[0] * dx + c[1] * dy
        face_projs.append((proj, fid))

    face_projs.sort()
    return [fid for _, fid in face_projs]


def _side_direction(side_index: int, n_sides: int) -> Tuple[float, float]:
    """Unit direction vector along the *side_index*-th side of a regular polygon.

    Polygon is centred at origin, first vertex at top (90°).
    Side *i* runs from vertex *i* to vertex *(i+1) % n*.
    """
    angle_start = math.pi / 2 + 2 * math.pi * side_index / n_sides
    angle_end = math.pi / 2 + 2 * math.pi * ((side_index + 1) % n_sides) / n_sides
    dx = math.cos(angle_end) - math.cos(angle_start)
    dy = math.sin(angle_end) - math.sin(angle_start)
    length = math.hypot(dx, dy)
    if length < 1e-12:
        return (1.0, 0.0)
    return (dx / length, dy / length)


def _side_outward_normal(side_index: int, n_sides: int) -> Tuple[float, float]:
    """Outward-pointing normal of the *side_index*-th side."""
    dx, dy = _side_direction(side_index, n_sides)
    # Rotate 90° clockwise for outward normal (polygon wound CCW)
    return (dy, -dx)


def _boundary_faces_near_side(
    detail_grid: PolyGrid,
    boundary_faces: Set[str],
    side_index: int,
    n_sides: int,
    grid_center_xy: Tuple[float, float],
) -> List[str]:
    """Select boundary sub-faces closest to a specific polygon side.

    For a hex grid, each of the 6 sides has a subset of boundary faces.
    We pick faces whose centroid direction (from grid centre) is closest
    to the side's outward normal.
    """
    normal = _side_outward_normal(side_index, n_sides)
    gcx, gcy = grid_center_xy

    # For each boundary face, compute the dot product of its direction
    # from centre with the side's outward normal.
    scored: List[Tuple[float, str]] = []
    for fid in boundary_faces:
        face = detail_grid.faces[fid]
        c = face_center(detail_grid.vertices, face)
        if c is None:
            continue
        dx = c[0] - gcx
        dy = c[1] - gcy
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        # Dot with outward normal — faces on this side have highest dot
        dot = (dx / length) * normal[0] + (dy / length) * normal[1]
        scored.append((dot, fid))

    # Take faces with dot > threshold — for a regular polygon with n_sides,
    # the angular width per side is 2π/n, so the threshold is roughly
    # cos(π/n).  We use a slightly more permissive threshold.
    threshold = math.cos(math.pi / n_sides + 0.15)
    side_faces = {fid for dot, fid in scored if dot > threshold}

    # Sort along the side direction
    direction = _side_direction(side_index, n_sides)
    return _boundary_faces_along_direction(detail_grid, side_faces, direction)


def compute_edge_subface_mapping(
    globe_grid: PolyGrid,
    face_id_a: str,
    face_id_b: str,
    collection: DetailGridCollection,
) -> EdgeSubfaceMapping:
    """Map boundary sub-faces along the shared edge between two globe tiles.

    For each tile, selects boundary sub-faces that lie along the macro-edge
    shared with the other tile, sorted by position along that edge.

    Parameters
    ----------
    globe_grid : PolyGrid
        The globe grid (with face adjacency).
    face_id_a, face_id_b : str
        Two adjacent globe tile ids.
    collection : DetailGridCollection
        Contains detail grids for both tiles.

    Returns
    -------
    EdgeSubfaceMapping
        Sub-faces from A and B along their shared edge.
    """
    grid_a = collection.grids[face_id_a]
    grid_b = collection.grids[face_id_b]

    n_sides_a = len(globe_grid.faces[face_id_a].vertex_ids)
    n_sides_b = len(globe_grid.faces[face_id_b].vertex_ids)

    boundary_a = _boundary_face_ids(grid_a)
    boundary_b = _boundary_face_ids(grid_b)

    gc_a = grid_center(grid_a.vertices)
    gc_b = grid_center(grid_b.vertices)

    # Find which side of A faces B, and vice versa
    side_a = _macro_edge_for_neighbour(grid_a, face_id_b, globe_grid, face_id_a)
    side_b = _macro_edge_for_neighbour(grid_b, face_id_a, globe_grid, face_id_b)

    # If we can't determine the side from globe edges (e.g. missing edge_ids),
    # fall back to using the neighbour's direction.
    if side_a is None:
        # Find the globe-level direction from A to B
        face_a_globe = globe_grid.faces[face_id_a]
        face_b_globe = globe_grid.faces[face_id_b]
        # Use neighbour index ordering as a proxy for side
        neighbours_a = list(globe_grid.faces[face_id_a].neighbor_ids)
        if face_id_b in neighbours_a:
            side_a = neighbours_a.index(face_id_b) % n_sides_a
        else:
            side_a = 0

    if side_b is None:
        neighbours_b = list(globe_grid.faces[face_id_b].neighbor_ids)
        if face_id_a in neighbours_b:
            side_b = neighbours_b.index(face_id_a) % n_sides_b
        else:
            side_b = 0

    subfaces_a = _boundary_faces_near_side(
        grid_a, boundary_a, side_a, n_sides_a, gc_a,
    )
    subfaces_b = _boundary_faces_near_side(
        grid_b, boundary_b, side_b, n_sides_b, gc_b,
    )

    return EdgeSubfaceMapping(
        face_id_a=face_id_a,
        face_id_b=face_id_b,
        subfaces_a=subfaces_a,
        subfaces_b=subfaces_b,
    )


# ═══════════════════════════════════════════════════════════════════
# 18A.3 — Build apron grid
# ═══════════════════════════════════════════════════════════════════

def _transform_neighbour_vertices(
    grid_src: PolyGrid,
    face_ids: List[str],
    src_center: Tuple[float, float],
    dst_center: Tuple[float, float],
    src_side_index: int,
    dst_side_index: int,
    src_n_sides: int,
    dst_n_sides: int,
) -> Dict[str, Tuple[float, float]]:
    """Transform neighbour sub-face vertices into the target tile's local space.

    The transform:
    1. Translate so the source grid's centre is at origin.
    2. Rotate so the source's side direction aligns with the target's
       (opposite) side direction.
    3. Reflect across the shared edge (source faces outward, target inward).
    4. Scale to match target grid extent.
    5. Translate to the target's boundary zone.

    Returns vertex_id → (x, y) in the target's local coordinate system.
    """
    # Source side outward normal and direction
    src_dir = _side_direction(src_side_index, src_n_sides)
    src_normal = _side_outward_normal(src_side_index, src_n_sides)

    # Target side — the neighbour's sub-faces need to appear beyond the
    # target's boundary on the side facing the source
    dst_dir = _side_direction(dst_side_index, dst_n_sides)
    dst_normal = _side_outward_normal(dst_side_index, dst_n_sides)

    # Collect all vertex positions from the source sub-faces
    src_vids: Set[str] = set()
    for fid in face_ids:
        face = grid_src.faces[fid]
        src_vids.update(face.vertex_ids)

    # Compute the source boundary centroid (mean of boundary face centres)
    src_bnd_centers = []
    for fid in face_ids:
        face = grid_src.faces[fid]
        c = face_center(grid_src.vertices, face)
        if c:
            src_bnd_centers.append(c)

    if not src_bnd_centers:
        return {}

    src_bnd_cx = sum(c[0] for c in src_bnd_centers) / len(src_bnd_centers)
    src_bnd_cy = sum(c[1] for c in src_bnd_centers) / len(src_bnd_centers)

    # Compute target grid extent (for placing apron just outside boundary)
    dst_xs = [v.x for v in grid_src.vertices.values() if v.x is not None]
    dst_ys = [v.y for v in grid_src.vertices.values() if v.y is not None]
    if not dst_xs:
        return {}
    src_extent = max(max(dst_xs) - min(dst_xs), max(dst_ys) - min(dst_ys), 1e-6)

    # Simple approach: translate each source vertex so that the source
    # boundary centroid maps to a point just beyond the target boundary
    # on the appropriate side.
    #
    # Target boundary position = target_centre + outward_normal * extent/2
    # But we want the apron just beyond, so we use extent/2 + small offset.
    #
    # For simplicity and correctness, we mirror the vertices:
    # 1. Centre at source boundary centroid
    # 2. Reflect across the shared-edge line (flip the normal component)
    # 3. Translate to target boundary position

    # The shared edge "line" in source space passes through src_bnd_center
    # with direction src_dir.  Reflection flips the src_normal component.

    result: Dict[str, Tuple[float, float]] = {}
    for vid in src_vids:
        v = grid_src.vertices.get(vid)
        if v is None or v.x is None or v.y is None:
            continue

        # Relative to source boundary centroid
        rx = v.x - src_bnd_cx
        ry = v.y - src_bnd_cy

        # Decompose into tangent and normal components along source side
        tang_comp = rx * src_dir[0] + ry * src_dir[1]
        norm_comp = rx * src_normal[0] + ry * src_normal[1]

        # Reflect normal component (mirror across the edge)
        norm_comp = -norm_comp

        # Reconstruct in target coordinate space using target's side
        # The target side direction and normal define the placement
        # We place relative to the target boundary (centre + dst_normal * offset)
        offset = src_extent * 0.5  # half the grid extent
        target_anchor_x = dst_center[0] + dst_normal[0] * offset
        target_anchor_y = dst_center[1] + dst_normal[1] * offset

        # The tangent direction along the target side
        tx = target_anchor_x + tang_comp * dst_dir[0] + norm_comp * dst_normal[0]
        ty = target_anchor_y + tang_comp * dst_dir[1] + norm_comp * dst_normal[1]

        result[vid] = (tx, ty)

    return result


def build_apron_grid(
    globe_grid: PolyGrid,
    face_id: str,
    collection: DetailGridCollection,
) -> Tuple[PolyGrid, Dict[str, Tuple[str, str]]]:
    """Construct an extended PolyGrid with neighbour edge sub-faces.

    The resulting grid contains:

    - All of the tile's own sub-faces (unchanged).
    - The outermost ring of sub-faces from each neighbour, transformed
      into the tile's local coordinate space.

    Parameters
    ----------
    globe_grid : PolyGrid
        The globe grid with adjacency info.
    face_id : str
        The target tile whose grid we're extending.
    collection : DetailGridCollection
        Contains detail grids and stores for all tiles.

    Returns
    -------
    (PolyGrid, apron_mapping)
        The extended grid and a mapping
        ``{apron_face_id: (source_tile_id, source_sub_face_id)}``
        for the apron faces.  The tile's own faces keep their
        original ids.
    """
    detail_grid = collection.grids[face_id]
    globe_adj = get_face_adjacency(globe_grid)
    neighbours = globe_adj.get(face_id, [])

    n_sides = len(globe_grid.faces[face_id].vertex_ids)
    gc = grid_center(detail_grid.vertices)

    # Start with a copy of all own vertices, edges, faces
    all_vertices: Dict[str, Vertex] = dict(detail_grid.vertices)
    all_edges: Dict[str, Edge] = dict(detail_grid.edges)
    all_faces: Dict[str, Face] = dict(detail_grid.faces)
    apron_mapping: Dict[str, Tuple[str, str]] = {}
    apron_face_ids: Set[str] = set()

    for nid in neighbours:
        if nid not in collection.grids:
            continue

        ngrid = collection.grids[nid]
        n_sides_n = len(globe_grid.faces[nid].vertex_ids)

        # Find which boundary sub-faces of the neighbour to include
        mapping = compute_edge_subface_mapping(
            globe_grid, face_id, nid, collection,
        )
        # We want subfaces_b — the neighbour's boundary sub-faces
        apron_subfaces = mapping.subfaces_b

        if not apron_subfaces:
            continue

        # Determine side indices
        side_self = _macro_edge_for_neighbour(detail_grid, nid, globe_grid, face_id)
        side_nbr = _macro_edge_for_neighbour(ngrid, face_id, globe_grid, nid)

        if side_self is None:
            neighbours_list = list(globe_grid.faces[face_id].neighbor_ids)
            if nid in neighbours_list:
                side_self = neighbours_list.index(nid) % n_sides
            else:
                side_self = 0

        if side_nbr is None:
            neighbours_list_n = list(globe_grid.faces[nid].neighbor_ids)
            if face_id in neighbours_list_n:
                side_nbr = neighbours_list_n.index(face_id) % n_sides_n
            else:
                side_nbr = 0

        gc_n = grid_center(ngrid.vertices)

        # Transform neighbour vertices into our local space
        transformed = _transform_neighbour_vertices(
            ngrid, apron_subfaces,
            src_center=gc_n,
            dst_center=gc,
            src_side_index=side_nbr,
            dst_side_index=side_self,
            src_n_sides=n_sides_n,
            dst_n_sides=n_sides,
        )

        if not transformed:
            continue

        # Prefix neighbour sub-face ids to avoid collisions
        prefix = f"apron_{nid}_"
        vid_map: Dict[str, str] = {}  # old_vid → new_vid

        for old_vid, (px, py) in transformed.items():
            new_vid = f"{prefix}{old_vid}"
            all_vertices[new_vid] = Vertex(new_vid, px, py)
            vid_map[old_vid] = new_vid

        # Add edges for apron faces
        for fid in apron_subfaces:
            face = ngrid.faces[fid]
            new_fid = f"{prefix}{fid}"

            # Map vertex ids
            new_vids = []
            valid = True
            for vid in face.vertex_ids:
                if vid in vid_map:
                    new_vids.append(vid_map[vid])
                else:
                    valid = False
                    break
            if not valid:
                continue

            # Create edges for this face
            new_eids: List[str] = []
            for i in range(len(new_vids)):
                a = new_vids[i]
                b = new_vids[(i + 1) % len(new_vids)]
                eid = f"{prefix}e_{a}_{b}"
                if eid not in all_edges:
                    all_edges[eid] = Edge(eid, (a, b), (new_fid,))
                else:
                    # Update face_ids
                    existing = all_edges[eid]
                    all_edges[eid] = Edge(
                        eid, existing.vertex_ids,
                        existing.face_ids + (new_fid,),
                    )
                new_eids.append(eid)

            all_faces[new_fid] = Face(
                id=new_fid,
                face_type=face.face_type,
                vertex_ids=tuple(new_vids),
                edge_ids=tuple(new_eids),
                metadata={"apron_source_tile": nid, "apron_source_face": fid},
            )
            apron_mapping[new_fid] = (nid, fid)
            apron_face_ids.add(new_fid)

    extended = PolyGrid(
        all_vertices.values(),
        all_edges.values(),
        all_faces.values(),
        metadata={
            **detail_grid.metadata,
            "has_apron": True,
            "apron_face_count": len(apron_face_ids),
            "apron_source_tiles": sorted(
                {t for t, _ in apron_mapping.values()}
            ),
        },
    )

    return extended, apron_mapping


# ═══════════════════════════════════════════════════════════════════
# 18A.4 — Apron terrain propagation
# ═══════════════════════════════════════════════════════════════════

def propagate_apron_terrain(
    apron_grid: PolyGrid,
    apron_mapping: Dict[str, Tuple[str, str]],
    collection: DetailGridCollection,
    face_id: str,
    *,
    elevation_field: str = "elevation",
    smooth_iterations: int = 2,
    smooth_weight: float = 0.5,
) -> TileDataStore:
    """Build a TileDataStore for an apron grid with correct elevations.

    - Own sub-faces: copy elevation from the tile's existing store.
    - Apron sub-faces: copy elevation from the source neighbour's store.
    - Smooth the join zone to reduce discontinuities.

    Parameters
    ----------
    apron_grid : PolyGrid
        Extended grid from :func:`build_apron_grid`.
    apron_mapping : dict
        ``{apron_face_id: (source_tile_id, source_sub_face_id)}``
    collection : DetailGridCollection
        Contains stores for all tiles.
    face_id : str
        The target tile id.
    elevation_field : str
        Field name in stores.
    smooth_iterations : int
        Number of smoothing passes on the join zone.
    smooth_weight : float
        Self-weight for smoothing (0–1).

    Returns
    -------
    TileDataStore
        Store with elevation for every face in the apron grid.
    """
    schema = TileSchema([FieldDef(elevation_field, float, 0.0)])
    store = TileDataStore(grid=apron_grid, schema=schema)

    # Get own tile's store
    _, own_store = collection.get(face_id)

    # Populate own sub-faces
    for fid in apron_grid.faces:
        if fid in apron_mapping:
            continue  # apron face — handled below
        if own_store is not None:
            try:
                elev = own_store.get(fid, elevation_field)
                store.set(fid, elevation_field, elev)
            except (KeyError, ValueError):
                store.set(fid, elevation_field, 0.0)
        else:
            store.set(fid, elevation_field, 0.0)

    # Populate apron sub-faces from neighbour stores
    for apron_fid, (src_tile, src_face) in apron_mapping.items():
        try:
            _, src_store = collection.get(src_tile)
        except KeyError:
            store.set(apron_fid, elevation_field, 0.0)
            continue

        if src_store is not None:
            try:
                elev = src_store.get(src_face, elevation_field)
                store.set(apron_fid, elevation_field, elev)
            except (KeyError, ValueError):
                store.set(apron_fid, elevation_field, 0.0)
        else:
            store.set(apron_fid, elevation_field, 0.0)

    # Smooth the join zone on the *original* grid first (own boundary faces
    # only).  The apron faces are not topologically connected to the own
    # faces in the extended grid, so smoothing them together via adjacency
    # won't help.  Instead we smooth the own boundary faces so their
    # elevation gracefully transitions toward the apron values.
    own_grid = collection.grids[face_id]
    own_boundary_ids = _boundary_face_ids(own_grid)
    join_zone = [
        fid for fid in own_boundary_ids if fid in apron_grid.faces
    ]

    if join_zone and smooth_iterations > 0:
        # For smoothing, we use the original detail grid's adjacency
        # (which has correct neighbour relationships for own faces).
        # But we smooth using the apron store so elevations move toward
        # the neighbour-provided values.
        #
        # We wrap the smooth in a local adjacency context: for own
        # boundary faces, their "virtual neighbour" is the mean of the
        # nearby apron faces.  As a pragmatic approximation, we just
        # smooth the own boundary faces against each other — the apron
        # faces already carry the correct neighbour elevation.
        smooth_field(
            apron_grid, store, elevation_field,
            iterations=smooth_iterations,
            self_weight=smooth_weight,
            face_ids=join_zone,
        )

    return store


# ═══════════════════════════════════════════════════════════════════
# Convenience: build apron for all tiles
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ApronResult:
    """Result of building an apron grid for a single tile.

    Attributes
    ----------
    face_id : str
        Globe tile id.
    grid : PolyGrid
        Extended grid with apron sub-faces.
    store : TileDataStore
        Elevation store covering all sub-faces (own + apron).
    apron_mapping : dict
        ``{apron_face_id: (source_tile_id, source_sub_face_id)}``.
    own_face_count : int
        Number of the tile's own sub-faces (before apron).
    apron_face_count : int
        Number of apron sub-faces added.
    """
    face_id: str
    grid: PolyGrid
    store: TileDataStore
    apron_mapping: Dict[str, Tuple[str, str]]
    own_face_count: int
    apron_face_count: int


def build_all_apron_grids(
    globe_grid: PolyGrid,
    collection: DetailGridCollection,
    *,
    smooth_iterations: int = 2,
    smooth_weight: float = 0.5,
) -> Dict[str, ApronResult]:
    """Build apron grids for every tile in the collection.

    Parameters
    ----------
    globe_grid : PolyGrid
        Globe grid with adjacency.
    collection : DetailGridCollection
        Must have grids **and** stores populated (terrain generated).
    smooth_iterations : int
        Smoothing passes on join zones.
    smooth_weight : float
        Self-weight for smoothing.

    Returns
    -------
    dict
        ``{face_id: ApronResult}``
    """
    results: Dict[str, ApronResult] = {}

    for face_id in collection.face_ids:
        own_count = len(collection.grids[face_id].faces)

        apron_grid, apron_mapping = build_apron_grid(
            globe_grid, face_id, collection,
        )

        apron_store = propagate_apron_terrain(
            apron_grid, apron_mapping, collection, face_id,
            smooth_iterations=smooth_iterations,
            smooth_weight=smooth_weight,
        )

        results[face_id] = ApronResult(
            face_id=face_id,
            grid=apron_grid,
            store=apron_store,
            apron_mapping=apron_mapping,
            own_face_count=own_count,
            apron_face_count=len(apron_mapping),
        )

    return results
