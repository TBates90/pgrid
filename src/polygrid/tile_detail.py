"""Sub-tile detail grid infrastructure for Goldberg globe tiles.

Each Goldberg tile on a :class:`GlobeGrid` is expanded into a local
``PolyGrid`` — a hex grid for hexagonal tiles and a pentagon-centred
grid for pentagonal tiles.  The resulting collection of detail grids
carries per-face terrain data that can be rendered as textures and
UV-mapped onto the 3-D tile surfaces.

This module provides:

- :class:`TileDetailSpec` — configuration dataclass
- :func:`build_all_detail_grids` — batch detail grid construction
- :class:`DetailGridCollection` — container for all detail grids + stores
- :func:`find_polygon_corners` — locate polygon corners of a detail grid
- :func:`get_neighbour_border_faces` — neighbour boundary faces in local coords
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .detail_grid import build_detail_grid, detail_face_count
from .geometry import face_center, grid_center
from .heightmap import smooth_field
from .noise import fbm, domain_warp
from .polygrid import PolyGrid
from .tile_data import FieldDef, TileDataStore, TileSchema


# ═══════════════════════════════════════════════════════════════════
# 10A.1 — TileDetailSpec
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TileDetailSpec:
    """Configuration for sub-tile detail grid generation.

    Controls the resolution and noise parameters used when expanding
    each Goldberg tile into a local PolyGrid with terrain data.

    Parameters
    ----------
    detail_rings : int
        Ring count for sub-tile grids.  A hex grid with 4 rings has
        61 sub-faces; 6 rings → 127 sub-faces.
    noise_frequency : float
        Spatial frequency of intra-tile noise (higher = finer detail).
    noise_octaves : int
        Number of noise octaves.
    amplitude : float
        How much local noise varies from the parent elevation (0–1).
    base_weight : float
        Parent elevation dominance (0–1).  Higher values mean the
        detail grid follows the parent more closely.
    boundary_smoothing : int
        Smoothing passes applied to boundary-band faces to reduce
        seam visibility between adjacent tiles.
    warp_strength : float
        Domain-warp strength for organic-looking detail variation.
    seed_offset : int
        Added to the parent seed to produce per-tile variation while
        keeping the global seed deterministic.
    """

    detail_rings: int = 4
    noise_frequency: float = 6.0
    noise_octaves: int = 5
    amplitude: float = 0.12
    base_weight: float = 0.80
    boundary_smoothing: int = 2
    warp_strength: float = 0.15
    seed_offset: int = 0


# ═══════════════════════════════════════════════════════════════════
# 10A.2 — Build all detail grids
# ═══════════════════════════════════════════════════════════════════

def build_all_detail_grids(
    globe_grid: PolyGrid,
    spec: TileDetailSpec,
    *,
    size: float = 1.0,
) -> Dict[str, PolyGrid]:
    """Build a detail grid for every face in a globe grid.

    Parameters
    ----------
    globe_grid : PolyGrid
        A :class:`GlobeGrid` (or any PolyGrid whose faces have
        ``face_type`` of ``"pent"`` or ``"hex"``).
    spec : TileDetailSpec
        Detail grid configuration.
    size : float
        Cell size passed to the grid builders.

    Returns
    -------
    dict
        ``{face_id: PolyGrid}`` — one detail grid per globe face.
    """
    grids: Dict[str, PolyGrid] = {}
    for face_id in globe_grid.faces:
        grid = build_detail_grid(
            globe_grid, face_id,
            detail_rings=spec.detail_rings,
            size=size,
        )
        grids[face_id] = grid
    return grids


# ═══════════════════════════════════════════════════════════════════
# 10A.3 — DetailGridCollection
# ═══════════════════════════════════════════════════════════════════

class DetailGridCollection:
    """Container managing detail grids and their tile-data stores.

    Holds one ``PolyGrid`` and one ``TileDataStore`` per globe face,
    with convenience methods for batch terrain generation and queries.

    Parameters
    ----------
    globe_grid : PolyGrid
        The globe grid that owns the faces.
    spec : TileDetailSpec
        Configuration used for grid construction and terrain gen.
    grids : dict
        ``{face_id: PolyGrid}`` — pre-built detail grids.
    """

    def __init__(
        self,
        globe_grid: PolyGrid,
        spec: TileDetailSpec,
        grids: Dict[str, PolyGrid],
    ) -> None:
        self._globe_grid = globe_grid
        self._spec = spec
        self._grids = dict(grids)
        self._stores: Dict[str, TileDataStore] = {}

    # ── Factory ─────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        globe_grid: PolyGrid,
        spec: Optional[TileDetailSpec] = None,
        *,
        size: float = 1.0,
    ) -> "DetailGridCollection":
        """Build a :class:`DetailGridCollection` for every globe face.

        Parameters
        ----------
        globe_grid : PolyGrid
        spec : TileDetailSpec, optional
            Uses defaults if not given.
        size : float
            Cell size for detail grids.

        Returns
        -------
        DetailGridCollection
        """
        if spec is None:
            spec = TileDetailSpec()
        grids = build_all_detail_grids(globe_grid, spec, size=size)
        return cls(globe_grid, spec, grids)

    # ── Properties ──────────────────────────────────────────────────

    @property
    def globe_grid(self) -> PolyGrid:
        """The parent globe grid."""
        return self._globe_grid

    @property
    def spec(self) -> TileDetailSpec:
        """The detail spec used for construction."""
        return self._spec

    @property
    def grids(self) -> Dict[str, PolyGrid]:
        """``{face_id: PolyGrid}`` — all detail grids."""
        return dict(self._grids)

    @property
    def stores(self) -> Dict[str, TileDataStore]:
        """``{face_id: TileDataStore}`` — all tile data stores."""
        return dict(self._stores)

    # ── Accessors ───────────────────────────────────────────────────

    def get(self, face_id: str) -> Tuple[PolyGrid, Optional[TileDataStore]]:
        """Return ``(detail_grid, store)`` for a face.

        The store may be ``None`` if terrain has not been generated yet.

        Raises
        ------
        KeyError
            If *face_id* is not in the collection.
        """
        if face_id not in self._grids:
            raise KeyError(f"No detail grid for face '{face_id}'")
        return self._grids[face_id], self._stores.get(face_id)

    @property
    def face_ids(self) -> List[str]:
        """Sorted list of face ids in the collection."""
        return sorted(self._grids.keys())

    @property
    def total_face_count(self) -> int:
        """Sum of sub-face counts across all detail grids."""
        return sum(len(g.faces) for g in self._grids.values())

    def detail_face_count_for(self, face_id: str) -> int:
        """Number of sub-faces in the detail grid for *face_id*."""
        if face_id not in self._grids:
            raise KeyError(f"No detail grid for face '{face_id}'")
        return len(self._grids[face_id].faces)

    # ── Terrain generation ──────────────────────────────────────────

    def generate_all_terrain(
        self,
        globe_store: TileDataStore,
        *,
        seed: int = 42,
        elevation_field: str = "elevation",
    ) -> None:
        """Generate terrain for every detail grid in the collection.

        This is the basic (non-boundary-aware) version.  For boundary-
        continuous terrain, use the ``detail_terrain`` module's
        :func:`generate_all_detail_terrain` function instead.

        Each detail grid receives elevation from its parent tile plus
        high-frequency noise variation.

        Parameters
        ----------
        globe_store : TileDataStore
            Globe-level tile data with an elevation field.
        seed : int
            Base noise seed.
        elevation_field : str
            Name of the elevation field in *globe_store*.
        """
        spec = self._spec

        for face_id, detail_grid in self._grids.items():
            parent_elev = globe_store.get(face_id, elevation_field)
            tile_seed = seed + spec.seed_offset + hash(face_id) % 10000

            schema = TileSchema([FieldDef("elevation", float, 0.0)])
            store = TileDataStore(grid=detail_grid, schema=schema)

            for fid in detail_grid.faces:
                face = detail_grid.faces[fid]
                c = face_center(detail_grid.vertices, face)
                if c is None:
                    continue
                cx, cy = c

                # Layer domain-warped fbm for organic variation
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

                elevation = (
                    parent_elev * spec.base_weight
                    + noise_val * spec.amplitude * (1.0 - spec.base_weight)
                )
                store.set(fid, "elevation", elevation)

            # Smooth to soften cell-to-cell jumps
            if spec.boundary_smoothing > 0:
                smooth_field(
                    detail_grid, store, "elevation",
                    iterations=spec.boundary_smoothing,
                    self_weight=0.6,
                )

            self._stores[face_id] = store

    # ── Summary ─────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of the collection."""
        n_grids = len(self._grids)
        n_stores = len(self._stores)
        total = self.total_face_count
        n_pent = sum(
            1 for fid in self._grids
            if self._globe_grid.faces[fid].face_type == "pent"
        )
        n_hex = n_grids - n_pent

        pent_faces = detail_face_count("pent", self._spec.detail_rings)
        hex_faces = detail_face_count("hex", self._spec.detail_rings)

        lines = [
            f"DetailGridCollection: {n_grids} tiles, {total} total sub-faces",
            f"  Pentagon tiles: {n_pent} × {pent_faces} faces = {n_pent * pent_faces}",
            f"  Hexagon tiles:  {n_hex} × {hex_faces} faces = {n_hex * hex_faces}",
            f"  Detail rings:   {self._spec.detail_rings}",
            f"  Terrain stores: {n_stores} / {n_grids}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DetailGridCollection(tiles={len(self._grids)}, "
            f"sub_faces={self.total_face_count}, "
            f"rings={self._spec.detail_rings})"
        )


# ═══════════════════════════════════════════════════════════════════
# 10A.5 — Polygon corners and neighbour border transforms
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NeighbourBorderFace:
    """A single sub-face from a neighbour tile, transformed into the
    target tile's local coordinate space.

    Attributes
    ----------
    face_id : str
        Original face id within the neighbour's detail grid.
    neighbour_id : str
        Globe face id of the neighbour tile.
    vertices : tuple of (float, float)
        Polygon vertices in the **target** tile's coordinate space.
    elevation : float
        Elevation value (or 0.0 if unavailable).
    """

    face_id: str
    neighbour_id: str
    vertices: Tuple[Tuple[float, float], ...]
    elevation: float = 0.0


def find_polygon_corners(
    grid: PolyGrid,
    n_sides: int,
) -> List[Tuple[float, float]]:
    """Locate the polygon corners of a detail grid's boundary.

    The Tutte embedding (or hex-grid axial layout) places boundary
    vertices on a convex *n_sides*-gon.  This function clusters them
    and returns the average position of each corner, ordered by
    descending angle (counter-clockwise starting from the largest
    atan2 angle).

    Parameters
    ----------
    grid : PolyGrid
        A detail grid (hex- or pentagon-centred).
    n_sides : int
        Number of polygon sides (5 or 6).

    Returns
    -------
    list of (float, float)
        ``corners[k]`` is ``(x, y)`` of corner *k*.
        Edge *k* goes from ``corners[k]`` to ``corners[(k+1) % n_sides]``.
    """
    gcx, gcy = grid_center(grid.vertices)

    # Identify boundary vertices (those on edges with < 2 faces)
    boundary_vids: Set[str] = set()
    for edge in grid.edges.values():
        if len(edge.face_ids) < 2:
            boundary_vids.update(edge.vertex_ids)

    # Angle + distance from centre for each boundary vertex
    items: List[Tuple[str, float, float]] = []
    for vid in boundary_vids:
        v = grid.vertices[vid]
        dx, dy = v.x - gcx, v.y - gcy
        items.append((vid, math.atan2(dy, dx), math.hypot(dx, dy)))

    max_d = max(d for _, _, d in items)
    # Keep vertices within 2% of max distance (polygon corners)
    candidates = [(vid, ang) for vid, ang, d in items if d > max_d * 0.98]
    candidates.sort(key=lambda x: x[1])

    # Cluster by angular proximity
    cluster_gap = math.radians(360.0 / n_sides * 0.4)
    clusters: List[List[Tuple[str, float]]] = [[candidates[0]]]
    for vid, ang in candidates[1:]:
        if ang - clusters[-1][-1][1] < cluster_gap:
            clusters[-1].append((vid, ang))
        else:
            clusters.append([(vid, ang)])

    # Wrap-around: merge first and last if close
    if len(clusters) > n_sides:
        first_ang = clusters[0][0][1]
        last_ang = clusters[-1][-1][1]
        if (first_ang + 2 * math.pi) - last_ang < cluster_gap:
            clusters[-1].extend(clusters[0])
            clusters = clusters[1:]

    if len(clusters) != n_sides:
        # Fallback: regular polygon
        base = math.pi if n_sides == 6 else math.pi / 2
        r = max_d
        return [
            (gcx + r * math.cos(base - k * 2 * math.pi / n_sides),
             gcy + r * math.sin(base - k * 2 * math.pi / n_sides))
            for k in range(n_sides)
        ]

    # Average position per cluster
    corners: List[Tuple[float, float]] = []
    for cluster in clusters:
        xs = [grid.vertices[vid].x for vid, _ in cluster]
        ys = [grid.vertices[vid].y for vid, _ in cluster]
        corners.append((sum(xs) / len(xs), sum(ys) / len(ys)))

    # Sort by angle descending (same convention as edge indexing)
    corners.sort(
        key=lambda p: math.atan2(p[1] - gcy, p[0] - gcx),
        reverse=True,
    )
    return corners


def _compute_edge_transform(
    src_corners: List[Tuple[float, float]],
    src_edge_idx: int,
    dst_corners: List[Tuple[float, float]],
    dst_edge_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the similarity transform mapping source edge to destination edge.

    The two tiles share a globe edge.  On tile A (destination), edge
    ``dst_edge_idx`` goes from ``dst_corners[k]`` to
    ``dst_corners[(k+1) % N]``.  On tile B (source / neighbour), the
    *same* globe edge appears as ``src_corners[j]`` to
    ``src_corners[(j+1) % M]``, but in the **reverse** direction (the
    tiles face each other across the shared edge).

    This function returns ``(R, t)`` such that for any point ``p`` in
    the source tile's space, ``R @ p + t`` gives the position in the
    destination tile's space.

    Parameters
    ----------
    src_corners, dst_corners : list of (float, float)
        Polygon corners in source and destination tile spaces.
    src_edge_idx, dst_edge_idx : int
        Edge index in each tile's polygon.

    Returns
    -------
    R : np.ndarray, shape (2, 2)
        Rotation (+ optional uniform scale) matrix.
    t : np.ndarray, shape (2,)
        Translation vector.
    """
    n_src = len(src_corners)
    n_dst = len(dst_corners)

    # Source edge endpoints
    s0 = np.array(src_corners[src_edge_idx])
    s1 = np.array(src_corners[(src_edge_idx + 1) % n_src])

    # Destination edge endpoints — reversed because tiles face each other
    d0 = np.array(dst_corners[(dst_edge_idx + 1) % n_dst])
    d1 = np.array(dst_corners[dst_edge_idx])

    # Vectors along each edge
    sv = s1 - s0
    dv = d1 - d0

    s_len = np.linalg.norm(sv)
    d_len = np.linalg.norm(dv)

    if s_len < 1e-12 or d_len < 1e-12:
        return np.eye(2), np.zeros(2)

    scale = d_len / s_len

    # Rotation angle from source edge direction to destination edge direction
    s_angle = math.atan2(sv[1], sv[0])
    d_angle = math.atan2(dv[1], dv[0])
    theta = d_angle - s_angle

    cos_t = math.cos(theta) * scale
    sin_t = math.sin(theta) * scale
    R = np.array([[cos_t, -sin_t],
                   [sin_t, cos_t]])

    # Translation: s0 maps to d0
    t = d0 - R @ s0

    return R, t


def _boundary_face_ids_for_edge(
    grid: PolyGrid,
    edge_idx: int,
    n_sides: int,
    corners: List[Tuple[float, float]],
) -> Set[str]:
    """Return face ids of boundary faces closest to polygon edge ``edge_idx``.

    Uses the angle-based assignment: each boundary face's centroid
    angle (relative to grid centre) is compared to the edge midpoint
    angle.
    """
    gcx, gcy = grid_center(grid.vertices)

    # Boundary face ids
    boundary: Set[str] = set()
    for edge in grid.edges.values():
        if len(edge.face_ids) < 2:
            for fid in edge.face_ids:
                boundary.add(fid)

    # Edge midpoint angles
    edge_angles: List[float] = []
    for k in range(n_sides):
        c0 = corners[k]
        c1 = corners[(k + 1) % n_sides]
        mx = (c0[0] + c1[0]) / 2.0
        my = (c0[1] + c1[1]) / 2.0
        edge_angles.append(math.atan2(my - gcy, mx - gcx))

    result: Set[str] = set()
    for fid in boundary:
        face = grid.faces[fid]
        c = face_center(grid.vertices, face)
        if c is None:
            continue
        cx, cy = c
        face_angle = math.atan2(cy - gcy, cx - gcx)

        # Find closest edge
        best_edge = 0
        best_dist = float("inf")
        for k, ea in enumerate(edge_angles):
            d = abs(face_angle - ea)
            if d > math.pi:
                d = 2 * math.pi - d
            if d < best_dist:
                best_dist = d
                best_edge = k

        if best_edge == edge_idx:
            result.add(fid)

    return result


def get_neighbour_border_faces(
    coll: "DetailGridCollection",
    face_id: str,
    globe_grid: PolyGrid,
) -> List[NeighbourBorderFace]:
    """Return boundary faces from all neighbours, transformed into
    *face_id*'s local coordinate space.

    For each neighbour of *face_id* on the globe:

    1. Determine which polygon edge they share (via
       :func:`~polygrid.detail_terrain.compute_neighbor_edge_mapping`).
    2. Find the neighbour's boundary sub-faces along that shared edge.
    3. Apply a similarity transform to map those faces from the
       neighbour's Tutte space into the target tile's Tutte space.

    Parameters
    ----------
    coll : DetailGridCollection
        Must have terrain stores populated.
    face_id : str
        The target tile whose coordinate space the results are in.
    globe_grid : PolyGrid
        The globe grid (needed for adjacency / shared-vertex lookup).

    Returns
    -------
    list of NeighbourBorderFace
        One entry per boundary sub-face of each neighbour along the
        shared edge.  Vertices are in the target tile's local coords.
    """
    from .detail_terrain import compute_neighbor_edge_mapping

    target_grid, target_store = coll.get(face_id)
    target_face = globe_grid.faces[face_id]
    n_sides_target = len(target_face.vertex_ids)
    target_corners = find_polygon_corners(target_grid, n_sides_target)

    # Which edge of the target tile does each neighbour share?
    edge_mapping = compute_neighbor_edge_mapping(globe_grid, face_id)

    result: List[NeighbourBorderFace] = []

    for nid, target_edge_idx in edge_mapping.items():
        try:
            nbr_grid, nbr_store = coll.get(nid)
        except KeyError:
            continue

        nbr_face = globe_grid.faces[nid]
        n_sides_nbr = len(nbr_face.vertex_ids)
        nbr_corners = find_polygon_corners(nbr_grid, n_sides_nbr)

        # Which edge of the *neighbour* is shared with target?
        nbr_edge_mapping = compute_neighbor_edge_mapping(globe_grid, nid)
        nbr_edge_idx = nbr_edge_mapping.get(face_id)
        if nbr_edge_idx is None:
            continue

        # Compute transform: neighbour space → target space
        R, t = _compute_edge_transform(
            src_corners=nbr_corners,
            src_edge_idx=nbr_edge_idx,
            dst_corners=target_corners,
            dst_edge_idx=target_edge_idx,
        )

        # Get the neighbour's boundary faces on the shared edge
        edge_faces = _boundary_face_ids_for_edge(
            nbr_grid, nbr_edge_idx, n_sides_nbr, nbr_corners,
        )

        for fid in edge_faces:
            face = nbr_grid.faces[fid]
            # Transform vertices
            verts: List[Tuple[float, float]] = []
            for vid in face.vertex_ids:
                v = nbr_grid.vertices.get(vid)
                if v is None or not v.has_position():
                    break
                p = R @ np.array([v.x, v.y]) + t
                verts.append((float(p[0]), float(p[1])))
            else:
                if len(verts) >= 3:
                    elev = 0.0
                    if nbr_store is not None:
                        elev = nbr_store.get(fid, "elevation")
                    result.append(NeighbourBorderFace(
                        face_id=fid,
                        neighbour_id=nid,
                        vertices=tuple(verts),
                        elevation=elev,
                    ))

    return result


def get_neighbour_border_grid(
    coll: "DetailGridCollection",
    face_id: str,
    globe_grid: PolyGrid,
) -> Tuple[PolyGrid, TileDataStore]:
    """Build a :class:`PolyGrid` from the neighbour boundary faces,
    transformed into *face_id*'s local coordinate space.

    This is the grid-based counterpart to :func:`get_neighbour_border_faces`.
    It returns a proper ``PolyGrid`` (with vertices, edges, and faces)
    plus a ``TileDataStore`` carrying the elevation field, so the
    result can be rendered with the standard rendering pipeline.

    Parameters
    ----------
    coll : DetailGridCollection
        Must have terrain stores populated.
    face_id : str
        The target tile whose coordinate space the grid lives in.
    globe_grid : PolyGrid
        The globe grid (needed for adjacency / shared-vertex lookup).

    Returns
    -------
    (PolyGrid, TileDataStore)
        The neighbour border grid and its elevation store.
    """
    from .models import Vertex, Edge, Face

    border_faces = get_neighbour_border_faces(coll, face_id, globe_grid)

    # Build deduplicated vertices, edges, and faces for the new grid
    vertex_map: Dict[str, Vertex] = {}  # key → Vertex
    edge_map: Dict[Tuple[str, str], Edge] = {}  # sorted vertex pair → Edge
    faces_out: List[Face] = []

    def _vkey(x: float, y: float) -> str:
        """Snap-to-grid vertex key for deduplication."""
        return f"{x:.8f},{y:.8f}"

    def _get_or_create_vertex(x: float, y: float) -> str:
        key = _vkey(x, y)
        if key not in vertex_map:
            vid = f"nv{len(vertex_map)}"
            vertex_map[key] = Vertex(vid, x, y)
        return vertex_map[key].id

    vid_lookup: Dict[str, str] = {}  # maps vkey → vertex id (for edge building)

    for idx, bf in enumerate(border_faces):
        face_vid_list: List[str] = []
        for vx, vy in bf.vertices:
            key = _vkey(vx, vy)
            if key not in vertex_map:
                vid = f"nv{len(vertex_map)}"
                vertex_map[key] = Vertex(vid, vx, vy)
            face_vid_list.append(vertex_map[key].id)

        # Build edges for this face
        n = len(face_vid_list)
        edge_id_list: List[str] = []
        fid = f"nf{idx}"
        for i in range(n):
            a = face_vid_list[i]
            b = face_vid_list[(i + 1) % n]
            ekey = tuple(sorted((a, b)))
            if ekey not in edge_map:
                eid = f"ne{len(edge_map)}"
                edge_map[ekey] = Edge(eid, ekey, (fid,))
            else:
                old = edge_map[ekey]
                edge_map[ekey] = Edge(old.id, old.vertex_ids,
                                      old.face_ids + (fid,))
            edge_id_list.append(edge_map[ekey].id)

        faces_out.append(Face(
            id=fid,
            face_type="hex",  # neighbour border faces are always hex-shaped
            vertex_ids=tuple(face_vid_list),
            edge_ids=tuple(edge_id_list),
        ))

    grid = PolyGrid(
        vertex_map.values(),
        edge_map.values(),
        faces_out,
        metadata={"source": "neighbour_border", "target_face": face_id},
    )

    # Build store with elevations
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)
    for idx, bf in enumerate(border_faces):
        store.set(f"nf{idx}", "elevation", bf.elevation)

    return grid, store


# ═══════════════════════════════════════════════════════════════════
# Tile + full-neighbour stitching
# ═══════════════════════════════════════════════════════════════════

def build_tile_with_neighbours(
    coll: "DetailGridCollection",
    face_id: str,
    globe_grid: PolyGrid,
) -> "CompositeGrid":
    """Stitch a tile's detail grid with all its neighbours into one grid.

    Uses the existing :func:`_position_hex_for_stitch` +
    :func:`stitch_grids` infrastructure to:

    1. Position each neighbour flush against *face_id*'s shared macro-edge.
    2. Detect adjacent *outer* neighbour pairs and snap their shared
       boundary vertices to averaged positions (same technique as
       :func:`pent_hex_assembly`).
    3. Stitch **all** shared edges — center↔neighbour *and*
       neighbour↔neighbour — into a single merged :class:`CompositeGrid`.

    Parameters
    ----------
    coll : DetailGridCollection
        Must have detail grids built (terrain stores are optional).
    face_id : str
        Globe face whose tile is at the centre of the assembly.
    globe_grid : PolyGrid
        The globe-level grid (for adjacency / edge mapping).

    Returns
    -------
    CompositeGrid
        ``.merged`` is the unified PolyGrid.  Component prefixes let
        callers distinguish centre vs neighbour faces.
    """
    from .assembly import _position_hex_for_stitch
    from .composite import CompositeGrid, StitchSpec, stitch_grids
    from .detail_terrain import compute_neighbor_edge_mapping
    from .models import Vertex

    # ── Centre grid ────────────────────────────────────────────────
    n_sides = len(globe_grid.faces[face_id].vertex_ids)
    dg_center, _ = coll.get(face_id)
    dg_center.compute_macro_edges(n_sides=n_sides)

    neigh_map = compute_neighbor_edge_mapping(globe_grid, face_id)

    grids: Dict[str, PolyGrid] = {face_id: dg_center}
    stitches: List[StitchSpec] = []

    # ── Position each neighbour flush & add centre↔neighbour stitch ─
    for nid, center_edge_idx in neigh_map.items():
        n_sides_n = len(globe_grid.faces[nid].vertex_ids)
        dg_n, _ = coll.get(nid)
        dg_n.compute_macro_edges(n_sides=n_sides_n)

        neigh_edge_idx = compute_neighbor_edge_mapping(
            globe_grid, nid
        )[face_id]

        positioned = _position_hex_for_stitch(
            dg_center, center_edge_idx, dg_n, neigh_edge_idx,
        )
        positioned.compute_macro_edges(n_sides=n_sides_n)

        grids[nid] = positioned
        stitches.append(StitchSpec(
            grid_a=face_id,
            edge_a=center_edge_idx,
            grid_b=nid,
            edge_b=neigh_edge_idx,
            flip=True,
        ))

    # ── Discover neighbour↔neighbour adjacencies ──────────────────
    neighbours = list(neigh_map.keys())
    outer_pairs: List[Tuple[str, int, str, int]] = []  # (n1, e1, n2, e2)

    for i, n1 in enumerate(neighbours):
        n1_all_neighs = compute_neighbor_edge_mapping(globe_grid, n1)
        for n2 in neighbours[i + 1:]:
            if n2 in n1_all_neighs:
                e_on_n1 = n1_all_neighs[n2]
                e_on_n2 = compute_neighbor_edge_mapping(
                    globe_grid, n2
                )[n1]
                outer_pairs.append((n1, e_on_n1, n2, e_on_n2))

    # ── Snap outer boundary vertices to averaged positions ────────
    for n1, e1, n2, e2 in outer_pairs:
        g1 = grids[n1]
        g2 = grids[n2]

        me1 = next(m for m in g1.macro_edges if m.id == e1)
        me2 = next(m for m in g2.macro_edges if m.id == e2)

        vids_1 = list(me1.vertex_ids)
        vids_2 = list(me2.vertex_ids)[::-1]  # flip for natural alignment

        for va_id, vb_id in zip(vids_1, vids_2):
            va = g1.vertices[va_id]
            vb = g2.vertices[vb_id]
            if va.has_position() and vb.has_position():
                mx = (va.x + vb.x) / 2
                my = (va.y + vb.y) / 2
                g1.vertices[va_id] = Vertex(va_id, mx, my)
                g2.vertices[vb_id] = Vertex(vb_id, mx, my)

    # ── Add outer↔outer stitches ─────────────────────────────────
    for n1, e1, n2, e2 in outer_pairs:
        stitches.append(StitchSpec(
            grid_a=n1, edge_a=e1,
            grid_b=n2, edge_b=e2,
            flip=True,
        ))

    return stitch_grids(grids, stitches)
