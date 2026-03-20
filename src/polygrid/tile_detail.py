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
- :func:`build_tile_with_neighbours` — stitch a tile with its neighbours
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
# 10A.5 — Polygon corners
# ═══════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════
# Tile + full-neighbour stitching
# ═══════════════════════════════════════════════════════════════════


def _find_closest_macro_edge_pair(
    g1: PolyGrid,
    g2: PolyGrid,
    *,
    exclude_g1: Optional[int] = None,
    exclude_g2: Optional[int] = None,
) -> Tuple[int, int]:
    """Find the pair of macro edges (one from each grid) that overlap.

    After positioning, two adjacent neighbour grids share exactly one
    macro edge geometrically.  This function identifies that pair by
    comparing edge midpoints, excluding the stitch edges that connect
    each grid to the centre tile.

    Returns ``(edge_id_g1, edge_id_g2)``.
    """
    best_dist = float("inf")
    best_pair = (-1, -1)

    for me1 in g1.macro_edges:
        if me1.id == exclude_g1:
            continue
        v0 = g1.vertices[me1.vertex_ids[0]]
        v1 = g1.vertices[me1.vertex_ids[-1]]
        mid1 = np.array([(v0.x + v1.x) / 2, (v0.y + v1.y) / 2])

        for me2 in g2.macro_edges:
            if me2.id == exclude_g2:
                continue
            u0 = g2.vertices[me2.vertex_ids[0]]
            u1 = g2.vertices[me2.vertex_ids[-1]]
            mid2 = np.array([(u0.x + u1.x) / 2, (u0.y + u1.y) / 2])

            d = float(np.linalg.norm(mid1 - mid2))
            if d < best_dist:
                best_dist = d
                best_pair = (me1.id, me2.id)

    return best_pair


def _macro_edge_overlap_ok(
    g1: PolyGrid,
    e1: int,
    g2: PolyGrid,
    e2: int,
    *,
    tol_factor: float = 0.25,
) -> bool:
    """Check whether two macro edges overlap well enough to snap/stitch.

    Compares the maximum vertex-to-vertex distance between corresponding
    boundary vertices of the two macro edges against a tolerance derived
    from the edge length.  Returns ``True`` when the edges are close
    enough that averaging positions will not inject significant
    distortion.

    Parameters
    ----------
    g1, g2 : PolyGrid
        The two grids containing the macro edges.
    e1, e2 : int
        Macro-edge IDs to compare.
    tol_factor : float
        Maximum allowed vertex-to-vertex distance as a fraction of
        the macro-edge length.  Default 0.25 (25 %).
    """
    me1 = next(m for m in g1.macro_edges if m.id == e1)
    me2 = next(m for m in g2.macro_edges if m.id == e2)

    vids_1 = list(me1.vertex_ids)
    vids_2 = list(me2.vertex_ids)

    if len(vids_1) != len(vids_2):
        return False

    # Compute macro-edge length (from first to last vertex of edge 1)
    v_start = g1.vertices[vids_1[0]]
    v_end = g1.vertices[vids_1[-1]]
    edge_len = math.hypot(v_end.x - v_start.x, v_end.y - v_start.y)
    if edge_len < 1e-12:
        return False

    tol = edge_len * tol_factor

    # Determine alignment direction (same vs flipped)
    v1_start = g1.vertices[vids_1[0]]
    v2_start = g2.vertices[vids_2[0]]
    v2_end = g2.vertices[vids_2[-1]]
    d_same = math.hypot(v1_start.x - v2_start.x, v1_start.y - v2_start.y)
    d_flip = math.hypot(v1_start.x - v2_end.x, v1_start.y - v2_end.y)
    if d_flip < d_same:
        vids_2 = vids_2[::-1]

    # Check max vertex-to-vertex distance
    max_dist = 0.0
    for va_id, vb_id in zip(vids_1, vids_2):
        va = g1.vertices[va_id]
        vb = g2.vertices[vb_id]
        if va.has_position() and vb.has_position():
            d = math.hypot(va.x - vb.x, va.y - vb.y)
            if d > max_dist:
                max_dist = d

    return max_dist <= tol


def build_tile_with_neighbours(
    coll: "DetailGridCollection",
    face_id: str,
    globe_grid: PolyGrid,
) -> "CompositeGrid":
    """Stitch a tile's detail grid with all its neighbours into one grid.

    Uses the existing :func:`_position_hex_for_stitch` +
    :func:`stitch_grids` infrastructure to:

    1. Position each neighbour flush against *face_id*'s shared macro-edge.
    2. For **hex-centred** tiles, detect adjacent *outer* neighbour pairs
       and snap their shared boundary vertices to averaged positions
       (same technique as :func:`pent_hex_assembly`), then stitch those
       neighbour↔neighbour edges.
    3. For **pentagon-centred** tiles, skip neighbour↔neighbour closure.
       The 5-hex ring around a pentagon has an unavoidable angular
       deficit (~6° per corner) that makes planar closure impossible
       without injecting distortion.  Centre↔neighbour stitches are
       still created; corner gaps are handled by the downstream
       gap-fill infrastructure.
    4. Stitch all applicable shared edges into a single merged
       :class:`CompositeGrid`.

    An overlap quality check (:func:`_macro_edge_overlap_ok`) guards
    hex-centred composites against stitching edges that are too far
    apart to merge cleanly.

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
    from .tile_uv_align import compute_pg_to_macro_edge_map

    # ── Centre grid ────────────────────────────────────────────────
    n_sides = len(globe_grid.faces[face_id].vertex_ids)
    dg_center, _ = coll.get(face_id)
    dg_center.compute_macro_edges(
        n_sides=n_sides,
        corner_ids=dg_center.metadata.get("corner_vertex_ids"),
    )

    # PG vertex_ids edge index → detail-grid macro-edge index.
    # These differ because the Tutte boundary walk can have different
    # winding than the PolyGrid vertex_ids ordering.
    pg_to_macro_center = compute_pg_to_macro_edge_map(
        globe_grid, face_id, dg_center,
    )

    neigh_map = compute_neighbor_edge_mapping(globe_grid, face_id)

    grids: Dict[str, PolyGrid] = {face_id: dg_center}
    stitches: List[StitchSpec] = []

    # ── Position each neighbour flush & add centre↔neighbour stitch ─
    for nid, center_pg_edge in neigh_map.items():
        n_sides_n = len(globe_grid.faces[nid].vertex_ids)
        dg_n, _ = coll.get(nid)
        dg_n.compute_macro_edges(
            n_sides=n_sides_n,
            corner_ids=dg_n.metadata.get("corner_vertex_ids"),
        )

        neigh_pg_edge = compute_neighbor_edge_mapping(
            globe_grid, nid
        )[face_id]

        # Convert PG edge indices → macro-edge indices
        center_macro_edge = pg_to_macro_center[center_pg_edge]
        pg_to_macro_n = compute_pg_to_macro_edge_map(
            globe_grid, nid, dg_n,
        )
        neigh_macro_edge = pg_to_macro_n[neigh_pg_edge]

        positioned = _position_hex_for_stitch(
            dg_center, center_macro_edge, dg_n, neigh_macro_edge,
        )
        positioned.compute_macro_edges(
            n_sides=n_sides_n,
            corner_ids=positioned.metadata.get("corner_vertex_ids"),
        )

        grids[nid] = positioned
        stitches.append(StitchSpec(
            grid_a=face_id,
            edge_a=center_macro_edge,
            grid_b=nid,
            edge_b=neigh_macro_edge,
            flip=True,
        ))

    # ── Discover neighbour↔neighbour adjacencies ──────────────────
    # The PG→macro mapping was computed on *original* grids, but the
    # positioned grids may have been reflected/rotated, which changes
    # which geometric edge a given macro-edge ID refers to.  Instead
    # of trying to map through PG→macro indices, we detect shared
    # edges by geometric proximity on the already-positioned grids.
    #
    # IMPORTANT: For pentagon-centred composites (n_sides == 5) we
    # skip neighbour↔neighbour closure entirely.  A 5-hex ring around
    # a pentagon has an unavoidable ~6° angular deficit per corner
    # (120° hex corner vs 108° pent corner).  Forcing closure via
    # vertex averaging injects localised wedge/pinch distortion near
    # each pentagon corner.  Since the crop window
    # (compute_tile_view_limits) and gap-fill infrastructure
    # (fill_sentinel_pixels, _fill_warped_gaps) already handle any
    # resulting corner holes, skipping closure is both safe and
    # visually superior.
    is_pentagon_centre = n_sides == 5

    neighbours = list(neigh_map.keys())
    outer_pairs: List[Tuple[str, int, str, int]] = []  # (n1, e1, n2, e2)

    if not is_pentagon_centre:
        # Track which macro-edge on each neighbour was used for the
        # centre↔neighbour stitch so we can exclude it from candidates.
        stitch_edge_ids: Dict[str, int] = {}
        for s in stitches:
            stitch_edge_ids[s.grid_b] = s.edge_b

        for i, n1 in enumerate(neighbours):
            n1_all_neighs = compute_neighbor_edge_mapping(globe_grid, n1)
            for n2 in neighbours[i + 1:]:
                if n2 not in n1_all_neighs:
                    continue
                g1 = grids[n1]
                g2 = grids[n2]
                e1, e2 = _find_closest_macro_edge_pair(
                    g1, g2,
                    exclude_g1=stitch_edge_ids.get(n1),
                    exclude_g2=stitch_edge_ids.get(n2),
                )

                # ── Overlap quality check ─────────────────────────
                # Only snap/stitch if the candidate edges actually
                # overlap geometrically.  This guards against forcing
                # non-overlapping edges together (which causes
                # distortion), as can happen when the planar layout
                # leaves a gap that is too large to bridge cleanly.
                if _macro_edge_overlap_ok(g1, e1, g2, e2):
                    outer_pairs.append((n1, e1, n2, e2))

        # ── Snap outer boundary vertices to averaged positions ────
        for n1, e1, n2, e2 in outer_pairs:
            g1 = grids[n1]
            g2 = grids[n2]

            me1 = next(m for m in g1.macro_edges if m.id == e1)
            me2 = next(m for m in g2.macro_edges if m.id == e2)

            vids_1 = list(me1.vertex_ids)
            vids_2 = list(me2.vertex_ids)

            # Determine whether the two edges run in the same or
            # opposite direction by comparing the start vertex of
            # each edge.
            v1_start = g1.vertices[vids_1[0]]
            v2_start = g2.vertices[vids_2[0]]
            v2_end = g2.vertices[vids_2[-1]]
            d_same = math.hypot(v1_start.x - v2_start.x,
                                v1_start.y - v2_start.y)
            d_flip = math.hypot(v1_start.x - v2_end.x,
                                v1_start.y - v2_end.y)
            if d_flip < d_same:
                vids_2 = vids_2[::-1]

            for va_id, vb_id in zip(vids_1, vids_2):
                va = g1.vertices[va_id]
                vb = g2.vertices[vb_id]
                if va.has_position() and vb.has_position():
                    mx = (va.x + vb.x) / 2
                    my = (va.y + vb.y) / 2
                    g1.vertices[va_id] = Vertex(va_id, mx, my)
                    g2.vertices[vb_id] = Vertex(vb_id, mx, my)

        # ── Add outer↔outer stitches ─────────────────────────────
        for n1, e1, n2, e2 in outer_pairs:
            stitches.append(StitchSpec(
                grid_a=n1, edge_a=e1,
                grid_b=n2, edge_b=e2,
                flip=True,
            ))

    return stitch_grids(grids, stitches)
