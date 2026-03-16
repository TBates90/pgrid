"""Polygon-cut tile textures aligned to GoldbergTile UV space — Phase 21.

This module bridges **pre-rendered stitched polygrid images** and the
3D Goldberg globe.  It extracts the polygon boundary of each tile,
computes the piecewise-linear warp from polygrid 2D space into the
GoldbergTile's UV polygon space, and produces oriented tile images
ready for atlas packing.

Pipeline overview
-----------------
1. For each Goldberg tile, extract polygon corners from the detail
   grid's macro edges (Tutte embedding space).
2. Get the authoritative UV polygon from the ``models`` library's
   ``GoldbergTile.uv_vertices``.
3. Match grid corners → UV corners via angular alignment (handles
   both rotation and reflection between the two orderings).
4. Optionally equalise piecewise-warp sector ratios for irregular
   hex tiles adjacent to pentagons.
5. Compute a piecewise-linear (triangle-fan) warp that maps each
   pixel of the output slot back to the stitched source image.
6. Pack all warped tiles into a texture atlas with gutter padding.

Key functions
-------------
- :func:`get_macro_edge_corners`       — polygon corners from macro edges
- :func:`match_grid_corners_to_uv`     — angular alignment grid↔UV
- :func:`compute_polygon_corners_px`   — grid corners → pixel coords
- :func:`warp_tile_to_uv`             — piecewise-linear image warp
- :func:`build_polygon_cut_atlas`      — end-to-end atlas builder
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from .atlas_utils import fill_gutter, compute_atlas_layout
from .polygrid import PolyGrid
from .tile_detail import find_polygon_corners, DetailGridCollection

if TYPE_CHECKING:
    from PIL import Image


# ═══════════════════════════════════════════════════════════════════
# 21A — Polygon extraction helpers
# ═══════════════════════════════════════════════════════════════════

def compute_uv_to_polygrid_offset(
    globe_grid: PolyGrid,
    face_id: str,
) -> int:
    """Compute the rotational offset between GoldbergTile and PolyGrid vertex orderings.

    ``GoldbergTile.uv_vertices`` and ``PolyGrid.faces[fid].vertex_ids``
    use independently-constructed vertex orderings.  This function
    finds the integer *offset* such that::

        polygrid_vertex[k]  ==  goldberg_vertex[(k - offset) % N]

    Or equivalently::

        uv_corners_aligned[k] = uv_corners_raw[(k - offset) % N]

    where ``uv_corners_raw`` comes from ``get_tile_uv_vertices()``
    (GoldbergTile order) and ``uv_corners_aligned`` is in PolyGrid
    ``vertex_ids`` order.

    Returns
    -------
    int
        The rotation offset.  Apply as
        ``aligned = [raw[(i - offset) % n] for i in range(n)]``.
    """
    from .uv_texture import get_goldberg_tiles, _match_tile_to_face

    face = globe_grid.faces[face_id]
    n = len(face.vertex_ids)
    freq = globe_grid.metadata.get("frequency", 3)
    rad = globe_grid.metadata.get("radius", 1.0)
    tiles = get_goldberg_tiles(freq, rad)
    tile = _match_tile_to_face(tiles, face_id)

    # PolyGrid 3D coords in vertex_ids order
    pg_verts = []
    for vid in face.vertex_ids:
        v = globe_grid.vertices[vid]
        pg_verts.append(np.array([v.x, v.y, v.z]))

    # GoldbergTile 3D coords
    gt_verts = [np.array(v) for v in tile.vertices]

    # Match: GT[gi] ↔ PG[pi]
    # offset = (pi - gi) % n   (should be constant for all pairs)
    for gi, gv in enumerate(gt_verts):
        for pi, pv in enumerate(pg_verts):
            if np.linalg.norm(gv - pv) < 1e-4:
                return (pi - gi) % n

    # Fallback — shouldn't happen
    return 0


def align_uv_corners_to_polygrid(
    uv_corners: List[Tuple[float, float]],
    offset: int,
) -> List[Tuple[float, float]]:
    """Rotate *uv_corners* from GoldbergTile order into PolyGrid vertex_ids order.

    Parameters
    ----------
    uv_corners : list of (u, v)
        From ``get_tile_uv_vertices()`` (GoldbergTile order).
    offset : int
        From ``compute_uv_to_polygrid_offset()``.

    Returns
    -------
    list of (u, v)
        Reordered so that ``result[k]`` corresponds to ``vertex_ids[k]``.
    """
    n = len(uv_corners)
    return [uv_corners[(i - offset) % n] for i in range(n)]


def get_macro_edge_corners(
    grid: PolyGrid,
    n_sides: int,
) -> List[Tuple[float, float]]:
    """Return polygon corners ordered by macro-edge index.

    ``corners[k]`` is the start vertex of ``macro_edge[k]``, so edge *k*
    runs from ``corners[k]`` to ``corners[(k+1) % n_sides]``.  This
    ordering is consistent with :func:`compute_neighbor_edge_mapping`
    (which also uses macro-edge / ``vertex_ids`` numbering).

    Call ``grid.compute_macro_edges(n_sides=n_sides)`` before this.

    Parameters
    ----------
    grid : PolyGrid
        Detail grid with pre-computed macro edges.
    n_sides : int
        Number of polygon sides (5 or 6).

    Returns
    -------
    list of (float, float)
        ``corners[k]`` in macro-edge-id order.
    """
    corners: List[Tuple[float, float]] = []
    for k in range(n_sides):
        me = next(m for m in grid.macro_edges if m.id == k)
        v = grid.vertices[me.vertex_ids[0]]
        corners.append((v.x, v.y))
    return corners


def compute_pg_to_macro_edge_map(
    globe_grid: PolyGrid,
    face_id: str,
    detail_grid: PolyGrid,
) -> Dict[int, int]:
    """Map PolyGrid (vertex_ids) **edge** indices to macro-edge indices.

    ``compute_neighbor_edge_mapping`` returns edge indices in
    PolyGrid ``vertex_ids`` order — edge *k* connects ``vertex_ids[k]``
    to ``vertex_ids[(k+1) % n]``.  The detail-grid macro-edges are
    numbered by the Tutte boundary walk, which can have **opposite
    winding** for hexagonal tiles (CW macro vs CCW PG).

    This function first matches **corners** (macro corner → PG vertex)
    by angular proximity, then determines whether the cyclic ordering
    is preserved (rotation) or reversed (reflection).  For the
    reflected case, each macro edge connects two PG vertices in the
    *reverse* direction, so the edge mapping is shifted by one
    relative to the corner mapping.

    Parameters
    ----------
    globe_grid : PolyGrid
        Globe grid with 3D vertices.
    face_id : str
        Tile face id, e.g. ``"t0"``.
    detail_grid : PolyGrid
        Detail grid with pre-computed macro-edges.

    Returns
    -------
    dict
        ``{pg_edge_index: macro_edge_index}`` — for every polygon
        edge *k* (in ``vertex_ids`` numbering), the macro-edge id
        that spans the **same pair of globe vertices**.
    """
    from .uv_texture import compute_tile_basis

    face = globe_grid.faces[face_id]
    n = len(face.vertex_ids)

    # PG vertex angles on the tangent plane
    center_3d, _, tangent, bitangent = compute_tile_basis(globe_grid, face_id)
    pg_angles: List[float] = []
    for vid in face.vertex_ids:
        v = globe_grid.vertices[vid]
        d = np.array([v.x, v.y, v.z], dtype=np.float64) - center_3d
        pg_angles.append(math.atan2(float(np.dot(d, bitangent)),
                                    float(np.dot(d, tangent))))

    # Macro corner angles in Tutte 2D
    corners = get_macro_edge_corners(detail_grid, n)
    gc = np.array(corners, dtype=np.float64)
    gc_center = gc.mean(axis=0)
    macro_angles = [math.atan2(gc[k, 1] - gc_center[1],
                               gc[k, 0] - gc_center[0]) for k in range(n)]

    # Match each macro corner to closest PG vertex by angle
    def _adiff(a: float, b: float) -> float:
        d = abs(a - b) % (2 * math.pi)
        return min(d, 2 * math.pi - d)

    macro_corner_to_pg: Dict[int, int] = {}
    for mk in range(n):
        best_pk = min(range(n), key=lambda pk: _adiff(macro_angles[mk], pg_angles[pk]))
        macro_corner_to_pg[mk] = best_pk

    # Detect rotation vs reflection.
    # Rotation: macro_corner_to_pg[k] = (k + offset) % n  (constant offset)
    # Reflection: macro_corner_to_pg[k] = (R - k) % n     (constant sum)
    offsets = [(macro_corner_to_pg[k] - k) % n for k in range(n)]
    sums = [(macro_corner_to_pg[k] + k) % n for k in range(n)]
    is_reflected = len(set(sums)) == 1 and len(set(offsets)) > 1

    # Invert corner map: pg_vertex → macro_corner
    pg_to_macro_corner: Dict[int, int] = {
        pg: macro for macro, pg in macro_corner_to_pg.items()
    }

    if is_reflected:
        # Reflected winding: macro edge M goes from macro_corner M to
        # macro_corner (M+1)%n, which in PG terms is pg_vertex P to
        # pg_vertex P' where P' is the PG vertex at angle BEFORE P
        # (opposite direction).  PG edge k goes from pg_vertex k to
        # pg_vertex (k+1)%n.  The macro edge sharing those same two
        # globe vertices has its START corner at pg_vertex (k+1)%n.
        pg_edge_to_macro: Dict[int, int] = {}
        for k in range(n):
            pg_end = (k + 1) % n  # end vertex of PG edge k
            pg_edge_to_macro[k] = pg_to_macro_corner[pg_end]
    else:
        # Same winding: macro edge M starts at the same PG vertex as
        # PG edge (corner_to_pg[M]), so pg_edge → macro is just the
        # inverted corner map.
        pg_edge_to_macro = dict(pg_to_macro_corner)

    return pg_edge_to_macro


def match_grid_corners_to_uv(
    grid_corners: List[Tuple[float, float]],
    globe_grid: PolyGrid,
    face_id: str,
) -> List[Tuple[float, float]]:
    """Reorder *grid_corners* (macro-edge order) to match GoldbergTile UV order.

    The 3D renderer pairs ``GoldbergTile.vertices[k]`` with
    ``GoldbergTile.uv_vertices[k]`` — both in the generator's vertex
    ordering.  The atlas piecewise warp needs ``grid_corners[k]`` to
    pair with ``uv_corners[k]`` (also generator order).

    Macro-edge corners live in 2D Tutte space and GoldbergTile vertices
    live in 3D, but their cyclic angular order around the polygon
    centre is the same (up to a rotation **and possibly a reflection**
    due to the GoldbergPolyhedron layout pipeline flipping winding).

    This function matches by angular proximity in a reflection-aware
    way, producing a permutation that is guaranteed to be either a
    cyclic rotation or a cyclic reflection.

    Parameters
    ----------
    grid_corners : list of (x, y)
        From :func:`get_macro_edge_corners` — macro-edge order in
        Tutte 2D space.
    globe_grid : PolyGrid
        Globe grid with 3D vertices (from ``build_globe_grid``).
    face_id : str
        Tile face id, e.g. ``"t0"``.

    Returns
    -------
    list of (x, y)
        Grid corners reordered so that ``result[k]`` pairs with
        ``uv_corners[k]`` (GoldbergTile / generator order).
    """
    from .uv_texture import get_goldberg_tiles, _match_tile_to_face, compute_tile_basis

    n = len(grid_corners)
    freq = globe_grid.metadata.get("frequency", 3)
    rad = globe_grid.metadata.get("radius", 1.0)
    tiles = get_goldberg_tiles(freq, rad)
    tile = _match_tile_to_face(tiles, face_id)

    center_3d, _, tangent_3d, bitangent_3d = compute_tile_basis(globe_grid, face_id)

    # Angles of GoldbergTile vertices projected onto tangent plane
    gt_angles = np.empty(n, dtype=np.float64)
    for i, vtx in enumerate(tile.vertices):
        rel = np.array(vtx, dtype=np.float64) - center_3d
        u = float(np.dot(rel, tangent_3d))
        v = float(np.dot(rel, bitangent_3d))
        gt_angles[i] = math.atan2(v, u)

    # Angles of macro-edge corners in Tutte 2D space
    gc = np.array(grid_corners, dtype=np.float64)
    centroid = gc.mean(axis=0)
    macro_angles = np.arctan2(gc[:, 1] - centroid[1], gc[:, 0] - centroid[0])

    # Try both non-reflected and reflected orderings.
    # Non-reflected: macro_corner[k] → GT[(k + rot) % n]
    # Reflected:     macro_corner[k] → GT[(R - k) % n]  for some R
    # Pick the one with smallest total angular error.

    def _angular_diff(a: float, b: float) -> float:
        d = abs(a - b) % (2 * math.pi)
        return min(d, 2 * math.pi - d)

    # --- Non-reflected: find best rotation ---
    best_rot_err = float("inf")
    best_rot = 0
    for rot in range(n):
        err = sum(
            _angular_diff(macro_angles[k], gt_angles[(k + rot) % n])
            for k in range(n)
        )
        if err < best_rot_err:
            best_rot_err = err
            best_rot = rot

    # --- Reflected: find best reflection ---
    best_ref_err = float("inf")
    best_ref = 0
    for ref in range(n):
        err = sum(
            _angular_diff(macro_angles[k], gt_angles[(ref - k) % n])
            for k in range(n)
        )
        if err < best_ref_err:
            best_ref_err = err
            best_ref = ref

    if best_rot_err <= best_ref_err:
        # Pure rotation: result[gt_k] = grid_corners[(gt_k - best_rot) % n]
        # gt_k = (macro_k + best_rot) % n, so macro_k = (gt_k - best_rot) % n
        return [grid_corners[(k - best_rot) % n] for k in range(n)]
    else:
        # Reflection: macro_corner[k] → GT[(best_ref - k) % n]
        # For GT[gt_k], macro_k = (best_ref - gt_k) % n
        return [grid_corners[(best_ref - k) % n] for k in range(n)]


def compute_polygon_corners_px(
    corners_grid: List[Tuple[float, float]],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    img_w: int,
    img_h: int,
) -> List[Tuple[float, float]]:
    """Map polygon corners from grid coordinates to pixel coordinates.

    The rendering pipeline sets ``ax.set_xlim/ylim`` and produces an
    image of size ``(img_w, img_h)``.  This maps each grid-space
    corner to pixel (px_x, px_y) using that same linear transform.

    Parameters
    ----------
    corners_grid : list of (x, y)
        Polygon corners in grid (Tutte embedding) coordinates.
    xlim, ylim : (min, max)
        Axis limits used by the renderer.
    img_w, img_h : int
        Output image dimensions in pixels.

    Returns
    -------
    list of (px_x, px_y)
        Polygon corners in pixel coordinates (origin = top-left).
    """
    x_min, x_max = xlim
    y_min, y_max = ylim
    x_span = x_max - x_min
    y_span = y_max - y_min

    result = []
    for gx, gy in corners_grid:
        px_x = (gx - x_min) / x_span * img_w
        # Y axis is inverted (pixel y=0 is top, grid y_max is top)
        px_y = (1.0 - (gy - y_min) / y_span) * img_h
        result.append((px_x, px_y))
    return result


# ── Pentagon warp compensation ───────────────────────────────────────
# The models library assigns pentagons ~29 % more UV area per unit of
# 3D face area than hexagons (UV/3D density ratio ≈ 1.293).  In
# theory the grid corners should be expanded by sqrt(1.293) ≈ 1.137
# so the warp samples a wider region of the rendered image, shrinking
# the tile pattern to match hexagon density.
#
# In practice, the full theoretical correction (1.1373) over-zooms
# because the stitched image already includes apron data.  A small
# 2 % expansion is enough to avoid clipping the pentagon boundary
# without visibly shrinking the texture.  If pentagon tiles look
# noticeably different in density from their hex neighbours, increase
# this toward the theoretical value.
_PENTAGON_GRID_SCALE = 1.02


def _scale_corners_from_centroid(
    corners: List[Tuple[float, float]],
    scale: float,
) -> List[Tuple[float, float]]:
    """Scale *corners* outward from their centroid by *scale*."""
    cx = sum(x for x, _ in corners) / len(corners)
    cy = sum(y for _, y in corners) / len(corners)
    return [
        (cx + (x - cx) * scale, cy + (y - cy) * scale)
        for x, y in corners
    ]


def _smooth_pentagon_corners(
    grid_corners: List[Tuple[float, float]],
    detail_grid: "PolyGrid",
    n_sides: int,
) -> List[Tuple[float, float]]:
    """Shift pentagon grid corners inward to compensate zigzag bias.

    The Tutte embedding places pentagon corners at boundary vertices
    that protrude ~3.5 % beyond the smooth polygon edge (the zigzag
    is biased inward, so corners jut out).  This shifts each corner
    to the average of the midpoints of the two adjacent boundary
    segments::

        mid_before = (corner + prev_boundary_vertex) / 2
        mid_after  = (corner + next_boundary_vertex) / 2
        smooth     = (mid_before + mid_after) / 2

    The function operates on the **UV-matched** corner list, looking
    up the corresponding macro-edge boundary vertices by matching
    each corner position to the nearest macro-edge start vertex.

    Parameters
    ----------
    grid_corners : list of (x, y)
        Corners in UV-matched order (from ``match_grid_corners_to_uv``).
    detail_grid : PolyGrid
        Detail grid with pre-computed ``macro_edges``.
    n_sides : int
        Number of polygon sides (should be 5 for pentagons).

    Returns
    -------
    list of (x, y)
        Smoothed corner positions (same ordering as input).
    """
    if n_sides != 5 or not detail_grid.macro_edges:
        return grid_corners

    gc = np.array(grid_corners, dtype=np.float64)
    n = len(gc)

    # Build a map: corner position → macro_edge index (nearest).
    # grid_corners are in UV-matched order; macro_edges are in
    # boundary-walk order.  Match by proximity.
    me_starts = np.array([
        (detail_grid.vertices[me.vertex_ids[0]].x,
         detail_grid.vertices[me.vertex_ids[0]].y)
        for me in detail_grid.macro_edges
    ], dtype=np.float64)

    # For each UV-matched corner, find the closest macro-edge start
    gc_to_me = []
    for k in range(n):
        dists = np.linalg.norm(me_starts - gc[k], axis=1)
        gc_to_me.append(int(np.argmin(dists)))

    smoothed = np.empty_like(gc)
    for k in range(n):
        me_idx = gc_to_me[k]
        me = detail_grid.macro_edges[me_idx]
        # Previous macro edge's last-before-corner vertex
        me_prev = detail_grid.macro_edges[(me_idx - 1) % n_sides]
        prev_v = detail_grid.vertices[me_prev.vertex_ids[-2]]
        # Current macro edge's first-after-corner vertex
        next_v = detail_grid.vertices[me.vertex_ids[1]]

        corner = gc[k]
        mid_before = (corner + np.array([prev_v.x, prev_v.y])) / 2
        mid_after = (corner + np.array([next_v.x, next_v.y])) / 2
        smoothed[k] = (mid_before + mid_after) / 2

    return [(float(smoothed[i, 0]), float(smoothed[i, 1]))
            for i in range(n)]


def _rotate_corners(
    corners: List[Tuple[float, float]],
    angle_rad: float,
) -> List[Tuple[float, float]]:
    """Rotate *corners* around their centroid by *angle_rad*."""
    import math as _m
    cx = sum(x for x, _ in corners) / len(corners)
    cy = sum(y for _, y in corners) / len(corners)
    cos_a = _m.cos(angle_rad)
    sin_a = _m.sin(angle_rad)
    return [
        (cx + (x - cx) * cos_a - (y - cy) * sin_a,
         cy + (x - cx) * sin_a + (y - cy) * cos_a)
        for x, y in corners
    ]


def _compute_bulk_rotation(
    grid_corners: List[Tuple[float, float]],
    uv_corners: List[Tuple[float, float]],
    tile_size: int = 512,
    gutter: int = 4,
) -> float:
    """Compute the mean angular offset between paired grid and UV corners.

    Both corner lists must already be in the same pairing order (i.e.
    ``grid_corners[k]`` corresponds to ``uv_corners[k]``).

    The UV corners are converted to destination-pixel space (with the
    Y-flip) before computing angles, matching the convention used by
    the piecewise warp.

    Returns
    -------
    float
        Signed rotation in radians to apply to *grid_corners* (around
        their centroid) so that each grid corner's angle matches the
        corresponding UV corner's angle.
    """
    n = len(grid_corners)
    gc = np.array(grid_corners, dtype=np.float64)
    uv = np.array(uv_corners, dtype=np.float64)
    gc_c = gc.mean(axis=0)

    # Convert UV → destination pixel space (same as the warp pipeline)
    dst_px = np.empty_like(uv)
    for i in range(n):
        u, v = uv[i]
        dst_px[i, 0] = gutter + u * tile_size
        dst_px[i, 1] = gutter + (1.0 - v) * tile_size
    dst_px_c = dst_px.mean(axis=0)

    # Sort both by destination angle (same permutation for both)
    dst_angles = np.arctan2(
        dst_px[:, 1] - dst_px_c[1], dst_px[:, 0] - dst_px_c[0],
    )
    order = np.argsort(dst_angles)

    gc_sorted = gc[order]
    gc_sorted_angles = np.arctan2(
        gc_sorted[:, 1] - gc_c[1], gc_sorted[:, 0] - gc_c[0],
    )

    # Mean signed angular difference (grid − dst), wrapped to [−π, π]
    diffs = gc_sorted_angles - dst_angles[order]
    diffs = (diffs + math.pi) % (2 * math.pi) - math.pi
    return float(np.mean(diffs))


def _equalise_sector_ratios(
    grid_corners: List[Tuple[float, float]],
    uv_corners: List[Tuple[float, float]],
    tile_size: int = 512,
    gutter: int = 4,
) -> Tuple[List[Tuple[float, float]], Optional[np.ndarray]]:
    """Reshape grid corners so every piecewise-warp sector is conformal.

    The piecewise centroid-fan warp maps each triangle
    ``(src_centroid, src[i], src[i+1])`` → ``(dst_centroid, dst[i], dst[i+1])``.
    If the source and destination triangles are **similar** (same shape,
    different size), the per-sector affine is a pure rotation + uniform
    scale, giving zero anisotropic distortion.

    On a Goldberg polyhedron the hex faces adjacent to pentagons have
    irregular UV polygons — different corner radii and angular spans.
    The Tutte embedding produces a perfectly regular hexagon, so the
    sector triangles are *not* similar, causing up to ~23 % anisotropy.

    This function adjusts each grid corner's **angle** and **radius**
    from the centroid to match the destination polygon's geometry:

    - **Angles**: each grid angular span is set equal to the
      corresponding UV-pixel angular span.
    - **Radii**: each grid corner radius is proportional to the
      destination corner radius (preserving mean radius).

    Together these make every source triangle similar to its destination
    triangle, achieving anisotropy = 1.0 in every sector.

    Because moving corners shifts the polygon mean (which
    ``_compute_piecewise_warp_map`` uses as fan centre), the caller
    must pass the returned centroid as ``src_centroid_override``.

    For regular tiles (not adjacent to a pentagon) all UV corners are
    equidistant and equally spaced, so this is a no-op.

    Parameters
    ----------
    grid_corners : list of (x, y)
        Polygon corners in grid (Tutte) space, paired 1:1 with
        *uv_corners* by index.
    uv_corners : list of (u, v)
        UV polygon corners.
    tile_size : int
        Inner tile size in pixels (for computing dst_px sort order).
    gutter : int
        Gutter pixels (for computing dst_px sort order).

    Returns
    -------
    (corners, centroid)
        corners : list of (x, y) — adjusted grid corners.
        centroid : (2,) ndarray or None — the fixed centroid to pass
        as ``src_centroid_override`` to the warp.  ``None`` when no
        adjustment was needed.
    """
    n = len(grid_corners)
    if n != len(uv_corners) or n < 3:
        return grid_corners, None

    gc = np.array(grid_corners, dtype=np.float64)
    uv = np.array(uv_corners, dtype=np.float64)
    gc_c = gc.mean(axis=0)

    # ── Compute dst_px and sort order (same as _compute_piecewise_warp_map) ──
    dst_px = np.empty_like(uv)
    for i in range(n):
        u, v = uv[i]
        dst_px[i, 0] = gutter + u * tile_size
        dst_px[i, 1] = gutter + (1.0 - v) * tile_size
    dst_px_c = dst_px.mean(axis=0)

    dst_angles = np.arctan2(
        dst_px[:, 1] - dst_px_c[1],
        dst_px[:, 0] - dst_px_c[0],
    )
    order = np.argsort(dst_angles)

    dst_sorted = dst_px[order]
    gc_sorted = gc[order]

    # ── Destination corner radii ──
    dst_R = np.linalg.norm(dst_sorted - dst_px_c, axis=1)

    # ── Destination angular spans (sector angles at the centroid) ──
    dst_spans = np.empty(n, dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        d0 = dst_sorted[i] - dst_px_c
        d1 = dst_sorted[j] - dst_px_c
        # Signed angle from d0 to d1 (atan2 of cross, dot)
        dst_spans[i] = math.atan2(
            d0[0] * d1[1] - d0[1] * d1[0],
            d0[0] * d1[0] + d0[1] * d1[1],
        )

    # ── New grid corner radii: proportional to dst radii ──
    R_mean_src = np.mean(np.linalg.norm(gc - gc_c, axis=1))
    R_mean_dst = dst_R.mean()
    new_R = R_mean_src * (dst_R / R_mean_dst)

    # ── New grid corner angles: align to destination angles ──
    # Use the destination's first sorted angle as the start so that
    # the bulk rotation of the source polygon is corrected.  Without
    # this, the Tutte embedding's residual rotational offset carries
    # through to the warped tile, causing every texture to appear
    # rotated by a few degrees.
    dst_sorted_angles = np.arctan2(
        dst_sorted[:, 1] - dst_px_c[1],
        dst_sorted[:, 0] - dst_px_c[0],
    )
    start_angle = dst_sorted_angles[0]

    new_sorted = np.empty((n, 2), dtype=np.float64)
    angle = start_angle
    for i in range(n):
        new_sorted[i, 0] = gc_c[0] + new_R[i] * math.cos(angle)
        new_sorted[i, 1] = gc_c[1] + new_R[i] * math.sin(angle)
        angle += dst_spans[i]

    # ── Un-sort back to original index order ──
    result_arr = np.empty_like(gc)
    for sorted_idx in range(n):
        orig_idx = order[sorted_idx]
        result_arr[orig_idx] = new_sorted[sorted_idx]

    corners = [(float(result_arr[i, 0]), float(result_arr[i, 1]))
               for i in range(n)]
    return corners, gc_c


def compute_tile_view_limits(
    composite,
    face_id: str,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute the axis limits used by the stitched-tile renderer.

    Centre tile extent + 25 % padding, with aspect-ratio correction
    to make the view square.

    Returns
    -------
    (xlim, ylim) : ((x_min, x_max), (y_min, y_max))
    """
    mg = composite.merged
    center_prefix = composite.id_prefixes[face_id]

    center_xs, center_ys = [], []
    for fid, face in mg.faces.items():
        if not fid.startswith(center_prefix):
            continue
        for vid in face.vertex_ids:
            v = mg.vertices.get(vid)
            if v is not None and v.has_position():
                center_xs.append(v.x)
                center_ys.append(v.y)

    if not center_xs:
        return ((-1.0, 1.0), (-1.0, 1.0))

    cx_range = max(center_xs) - min(center_xs)
    cy_range = max(center_ys) - min(center_ys)
    half_span = max(cx_range, cy_range) * 0.5 * 1.25  # 25 % padding
    cx_mid = (min(center_xs) + max(center_xs)) * 0.5
    cy_mid = (min(center_ys) + max(center_ys)) * 0.5

    xlim = (cx_mid - half_span, cx_mid + half_span)
    ylim = (cy_mid - half_span, cy_mid + half_span)
    return xlim, ylim


# ═══════════════════════════════════════════════════════════════════
# 21A — Polygon masking
# ═══════════════════════════════════════════════════════════════════

def mask_to_polygon(
    img: "Image.Image",
    corners_px: List[Tuple[float, float]],
) -> "Image.Image":
    """Apply a polygon mask — pixels outside become transparent.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image (RGB or RGBA).
    corners_px : list of (px_x, px_y)
        Polygon corners in pixel coordinates.

    Returns
    -------
    PIL.Image.Image
        RGBA image with transparent pixels outside the polygon.
    """
    from PIL import Image, ImageDraw

    rgba = img.convert("RGBA")
    mask = Image.new("L", rgba.size, 0)
    draw = ImageDraw.Draw(mask)
    poly = [(int(round(x)), int(round(y))) for x, y in corners_px]
    draw.polygon(poly, fill=255)
    rgba.putalpha(mask)
    return rgba


def uv_polygon_px(
    uv_corners: List[Tuple[float, float]],
    tile_size: int,
    gutter: int = 0,
) -> List[Tuple[float, float]]:
    """Convert UV polygon corners to pixel coordinates in a warped slot.

    After the affine warp, the UV polygon should land at these pixel
    positions within the ``(tile_size + 2*gutter)``-sized slot image.

    Parameters
    ----------
    uv_corners : list of (u, v)
        UV polygon corners in [0, 1] normalised space.
    tile_size : int
    gutter : int

    Returns
    -------
    list of (px_x, px_y)
    """
    result = []
    for u, v in uv_corners:
        px_x = gutter + u * tile_size
        px_y = gutter + (1.0 - v) * tile_size
        result.append((px_x, px_y))
    return result


def mask_warped_to_uv_polygon(
    warped: "Image.Image",
    uv_corners: List[Tuple[float, float]],
    tile_size: int,
    gutter: int = 0,
    fill_colour: Tuple[int, int, int] = (0, 0, 0),
) -> "Image.Image":
    """Mask a warped slot image — pixels outside the UV polygon get filled.

    Parameters
    ----------
    warped : PIL.Image.Image
        Warped slot image (from :func:`warp_tile_to_uv`).
    uv_corners : list of (u, v)
        UV polygon corners in [0, 1] normalised space.
    tile_size : int
    gutter : int
    fill_colour : (R, G, B)
        Colour for pixels outside the polygon. Default black.

    Returns
    -------
    PIL.Image.Image
        RGB image with outside pixels filled.
    """
    from PIL import Image, ImageDraw

    corners_px = uv_polygon_px(uv_corners, tile_size, gutter)

    # Build polygon mask (white = inside)
    w, h = warped.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    poly = [(int(round(x)), int(round(y))) for x, y in corners_px]
    draw.polygon(poly, fill=255)

    # Composite: warped content inside polygon, fill_colour outside
    bg = Image.new("RGB", (w, h), fill_colour)
    rgb = warped.convert("RGB")
    bg.paste(rgb, mask=mask)
    return bg


def draw_debug_labels(
    img: "Image.Image",
    uv_corners: List[Tuple[float, float]],
    face_id: str,
    edge_neighbours: Dict[int, str],
    tile_size: int,
    gutter: int = 0,
) -> "Image.Image":
    """Draw tile-ID and per-edge labels on a warped slot image.

    Places the tile ID (e.g. ``t5``) at the polygon centroid and
    edge labels (e.g. ``e0→t12``) at the midpoint of each UV
    polygon edge, oriented along the edge.

    The edge numbering matches the globe grid's vertex ordering,
    so adjacent tiles sharing an edge should show the **same**
    edge index at their shared boundary.

    Parameters
    ----------
    img : PIL.Image.Image
        Warped slot image to annotate.
    uv_corners : list of (u, v)
        UV polygon corners in [0, 1] space **in original vertex order**
        (from :func:`get_tile_uv_vertices`).
    face_id : str
        Tile ID string (e.g. ``"t5"``).
    edge_neighbours : dict
        ``{edge_index: neighbour_face_id}`` from
        :func:`compute_neighbor_edge_mapping`.
    tile_size : int
    gutter : int

    Returns
    -------
    PIL.Image.Image
        Annotated copy of the image.
    """
    from PIL import ImageDraw, ImageFont

    out = img.copy()
    draw = ImageDraw.Draw(out)

    # Try to load a small monospace font; fall back to default
    font_size = max(14, tile_size // 10)
    font_small = max(11, tile_size // 14)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", font_size)
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", font_small)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
            font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_small)
        except (OSError, IOError):
            font = ImageFont.load_default()
            font_s = font

    # UV → pixel helper
    def _uv_to_px(u: float, v: float) -> Tuple[float, float]:
        return (gutter + u * tile_size,
                gutter + (1.0 - v) * tile_size)

    n = len(uv_corners)

    # ── Centre label ────────────────────────────────────────────
    cu = sum(c[0] for c in uv_corners) / n
    cv = sum(c[1] for c in uv_corners) / n
    cx, cy = _uv_to_px(cu, cv)
    label = face_id.upper()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # Dark background pill
    pad = 3
    draw.rounded_rectangle(
        [cx - tw / 2 - pad, cy - th / 2 - pad,
         cx + tw / 2 + pad, cy + th / 2 + pad],
        radius=4, fill=(0, 0, 0, 180),
    )
    draw.text((cx - tw / 2, cy - th / 2), label, fill="white", font=font)

    # ── Edge labels ─────────────────────────────────────────────
    from PIL import Image as _PILImage

    for k in range(n):
        j = (k + 1) % n
        u0, v0 = uv_corners[k]
        u1, v1 = uv_corners[j]

        # Edge midpoint in pixel space
        mx, my = _uv_to_px((u0 + u1) / 2, (v0 + v1) / 2)

        # Push label slightly inward toward centre
        dx, dy = cx - mx, cy - my
        dist = math.hypot(dx, dy) or 1.0
        inset = min(14, dist * 0.20)
        mx += dx / dist * inset
        my += dy / dist * inset

        # Label text
        nid = edge_neighbours.get(k, "?")
        label_e = f"e{k}\u2192{nid}"

        # Compute rotation angle from the edge direction (in pixel space).
        # Pixel-space: x right, y down — so negate y for standard atan2.
        px0x, px0y = _uv_to_px(u0, v0)
        px1x, px1y = _uv_to_px(u1, v1)
        edge_dx = px1x - px0x
        edge_dy = px1y - px0y
        angle_deg = -math.degrees(math.atan2(edge_dy, edge_dx))
        # Keep text roughly upright (never upside-down)
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180

        # Render the label into a small temp image, rotate, then paste
        bbox_e = draw.textbbox((0, 0), label_e, font=font_s)
        ew, eh = bbox_e[2] - bbox_e[0], bbox_e[3] - bbox_e[1]
        pad_t = 4
        tmp_w = ew + pad_t * 2
        tmp_h = eh + pad_t * 2
        tmp = _PILImage.new("RGBA", (tmp_w, tmp_h), (0, 0, 0, 0))
        tmp_draw = ImageDraw.Draw(tmp)
        # Dark pill background
        tmp_draw.rounded_rectangle(
            [pad_t - 2, pad_t - 2, pad_t + ew + 2, pad_t + eh + 2],
            radius=3, fill=(0, 0, 0, 180),
        )
        tmp_draw.text((pad_t, pad_t), label_e, fill="yellow", font=font_s)

        # Rotate around the centre of the temp image
        rotated = tmp.rotate(angle_deg, resample=_PILImage.BICUBIC, expand=True)

        # Paste centred on (mx, my)
        rw, rh = rotated.size
        paste_x = int(round(mx - rw / 2))
        paste_y = int(round(my - rh / 2))
        out.paste(rotated, (paste_x, paste_y), rotated)
        # Refresh draw handle after paste
        draw = ImageDraw.Draw(out)

    return out


# ═══════════════════════════════════════════════════════════════════
# 21B — Per-tile UV orientation alignment
# ═══════════════════════════════════════════════════════════════════

def _match_corners(
    src_corners: np.ndarray,
    dst_corners: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match source corners to destination corners by angular alignment.

    Both arrays have the same length N.  Returns reordered copies so
    that ``src_out[k]`` corresponds to ``dst_out[k]``.  Tries all N
    rotational offsets and picks the one with the smallest total
    angular error.

    Parameters
    ----------
    src_corners, dst_corners : (N, 2) arrays

    Returns
    -------
    (src_ordered, dst_ordered) : matched (N, 2) arrays
    """
    n = len(src_corners)
    src_c = src_corners.mean(axis=0)
    dst_c = dst_corners.mean(axis=0)

    src_angles = np.arctan2(
        src_corners[:, 1] - src_c[1],
        src_corners[:, 0] - src_c[0],
    )
    dst_angles = np.arctan2(
        dst_corners[:, 1] - dst_c[1],
        dst_corners[:, 0] - dst_c[0],
    )

    src_order = np.argsort(src_angles)
    dst_order = np.argsort(dst_angles)

    best_offset = 0
    best_score = float("inf")
    for offset in range(n):
        score = 0.0
        for k in range(n):
            dk = src_order[(k + offset) % n]
            uk = dst_order[k]
            diff = math.atan2(
                math.sin(src_angles[dk] - dst_angles[uk]),
                math.cos(src_angles[dk] - dst_angles[uk]),
            )
            score += diff * diff
        if score < best_score:
            best_score = score
            best_offset = offset

    src_matched = np.empty((n, 2), dtype=np.float64)
    dst_matched = np.empty((n, 2), dtype=np.float64)
    for k in range(n):
        dk = src_order[(k + best_offset) % n]
        uk = dst_order[k]
        src_matched[k] = src_corners[dk]
        dst_matched[k] = dst_corners[uk]

    return src_matched, dst_matched


def compute_grid_to_uv_affine(
    grid_corners: List[Tuple[float, float]],
    uv_corners: List[Tuple[float, float]],
) -> np.ndarray:
    """Compute the best-fit affine transform from grid corners to UV corners.

    Solves for the 2×3 affine matrix ``M`` such that for each
    matched corner pair ``(src, dst)``:

    .. math::

        \\begin{bmatrix} u \\\\ v \\end{bmatrix}
        = M \\begin{bmatrix} x \\\\ y \\\\ 1 \\end{bmatrix}

    The system is over-determined (N ≥ 3 points), solved via
    least-squares.

    Parameters
    ----------
    grid_corners : list of (x, y)
        Source polygon corners in grid (Tutte) coordinates.
    uv_corners : list of (u, v)
        Destination corners in UV [0,1] space.

    Returns
    -------
    np.ndarray, shape (2, 3)
        Affine matrix ``[[a, b, tx], [c, d, ty]]``.
    """
    src = np.array(grid_corners, dtype=np.float64)
    dst = np.array(uv_corners, dtype=np.float64)

    # Match corners by rotational alignment
    src_m, dst_m = _match_corners(src, dst)

    n = len(src_m)
    # Build system: for each point, [x, y, 1] @ [a, b, tx; c, d, ty]^T = [u, v]
    A = np.zeros((2 * n, 6), dtype=np.float64)
    b = np.zeros(2 * n, dtype=np.float64)

    for i in range(n):
        x, y = src_m[i]
        u, v = dst_m[i]
        A[2 * i, 0] = x
        A[2 * i, 1] = y
        A[2 * i, 2] = 1.0
        b[2 * i] = u
        A[2 * i + 1, 3] = x
        A[2 * i + 1, 4] = y
        A[2 * i + 1, 5] = 1.0
        b[2 * i + 1] = v

    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    M = result.reshape(2, 3)
    return M


def compute_grid_to_px_affine(
    grid_corners: List[Tuple[float, float]],
    uv_corners: List[Tuple[float, float]],
    tile_size: int,
    gutter: int = 0,
) -> np.ndarray:
    """Compute affine from grid coords to atlas-slot pixel coords.

    Maps grid corners → UV [0,1] → pixel [gutter, gutter+tile_size].

    Parameters
    ----------
    grid_corners : list of (x, y)
    uv_corners : list of (u, v) in [0, 1]
    tile_size : int
    gutter : int

    Returns
    -------
    np.ndarray, shape (2, 3)
        Affine matrix mapping grid (x, y) → pixel (px, py).
    """
    # UV corners → pixel corners
    px_corners = []
    for u, v in uv_corners:
        px_x = gutter + u * tile_size
        px_y = gutter + (1.0 - v) * tile_size  # V is flipped (v=0 → bottom → pixel y_max)
        px_corners.append((px_x, px_y))

    return compute_grid_to_uv_affine(grid_corners, px_corners)


# ═══════════════════════════════════════════════════════════════════
# 21B — Piecewise-linear warp (triangle-fan)
# ═══════════════════════════════════════════════════════════════════

def _build_sector_affines(
    src_corners: np.ndarray,
    src_centroid: np.ndarray,
    dst_corners: np.ndarray,
    dst_centroid: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Build per-sector 2×2 affine + translation for a triangle fan.

    Each sector is the triangle (centroid, corner[i], corner[i+1]).
    Returns a list of (A, t) where ``dst = A @ src + t``.
    """
    n = len(src_corners)
    sectors = []
    for i in range(n):
        j = (i + 1) % n
        S = np.column_stack([
            src_corners[i] - src_centroid,
            src_corners[j] - src_centroid,
        ])  # (2, 2)
        D = np.column_stack([
            dst_corners[i] - dst_centroid,
            dst_corners[j] - dst_centroid,
        ])  # (2, 2)
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        if abs(det) > 1e-20:
            S_inv = np.array([
                [S[1, 1], -S[0, 1]],
                [-S[1, 0], S[0, 0]],
            ], dtype=np.float64) / det
            A = D @ S_inv
        else:
            A = np.eye(2, dtype=np.float64)
        t = dst_centroid - A @ src_centroid
        sectors.append((A, t))
    return sectors


def _assign_sectors(
    points: np.ndarray,
    centroid: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    """Assign each point to a triangle-fan sector.

    Parameters
    ----------
    points : (M, 2) array
    centroid : (2,) array
    corners : (N, 2) array — polygon vertices in angular order

    Returns
    -------
    (M,) int array — sector index for each point
    """
    n = len(corners)
    corner_angles = np.arctan2(
        corners[:, 1] - centroid[1],
        corners[:, 0] - centroid[0],
    )
    point_angles = np.arctan2(
        points[:, 1] - centroid[1],
        points[:, 0] - centroid[0],
    )

    sectors = np.zeros(len(points), dtype=np.int32)
    for i in range(n):
        j = (i + 1) % n
        a0 = corner_angles[i]
        a1 = corner_angles[j]
        # Check if each point's angle is in the arc from a0 to a1 (CCW)
        a0n = a0 % (2.0 * math.pi)
        a1n = a1 % (2.0 * math.pi)
        pn = point_angles % (2.0 * math.pi)
        if a0n <= a1n:
            mask = (pn >= a0n) & (pn < a1n)
        else:
            # Arc wraps around 2π
            mask = (pn >= a0n) | (pn < a1n)
        sectors[mask] = i

    return sectors


def _compute_piecewise_warp_map(
    grid_corners: List[Tuple[float, float]],
    uv_corners: List[Tuple[float, float]],
    tile_size: int,
    gutter: int,
    img_w: int,
    img_h: int,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    output_size: int,
    src_centroid_override: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-pixel source-coordinate maps for a piecewise-linear warp.

    Uses a triangle-fan decomposition (centroid + polygon edges) to
    give **exact** corner alignment — matching the ``UVTransform``
    approach used by the working renderer.

    ``grid_corners[k]`` must correspond to ``uv_corners[k]``
    (same edge index).  Pass macro-edge corners (from
    :func:`get_macro_edge_corners`) for the grid side; these already
    share the ``vertex_ids`` numbering with
    :func:`get_tile_uv_vertices`.

    Parameters
    ----------
    src_centroid_override : (2,) array, optional
        If given, use this as the source centroid instead of
        ``mean(grid_corners)``.  This allows the caller to fix the
        centroid when grid corner angles have been adjusted, ensuring
        the triangle-fan geometry matches the intended sector
        decomposition.

    Returns
    -------
    (map_x, map_y) : (H, W) float arrays
        For each output pixel, the corresponding input-image pixel
        coordinate.  Suitable for ``scipy.ndimage.map_coordinates``
        or ``cv2.remap``.
    """
    src = np.array(grid_corners, dtype=np.float64)
    dst_uv = np.array(uv_corners, dtype=np.float64)

    # Corners are paired 1:1 by index (both use vertex_ids ordering).
    if src_centroid_override is not None:
        src_centroid = np.asarray(src_centroid_override, dtype=np.float64)
    else:
        src_centroid = src.mean(axis=0)
    dst_uv_centroid = dst_uv.mean(axis=0)

    # Destination: UV → slot pixel coordinates
    dst_px = np.empty_like(dst_uv)
    for i in range(len(dst_uv)):
        u, v = dst_uv[i]
        dst_px[i, 0] = gutter + u * tile_size
        dst_px[i, 1] = gutter + (1.0 - v) * tile_size
    dst_px_centroid = dst_px.mean(axis=0)

    # Sort both arrays by *destination-pixel* angle from their
    # centroid.  Using the destination angle ensures that sector
    # assignment (which operates in output/slot-pixel space) lines
    # up with the affine triangles.  Applying the SAME permutation
    # to both src and dst preserves the 1:1 pairing.
    dst_angles = np.arctan2(
        dst_px[:, 1] - dst_px_centroid[1],
        dst_px[:, 0] - dst_px_centroid[0],
    )
    order = np.argsort(dst_angles)
    src_ordered = src[order]
    dst_px_ordered = dst_px[order]

    src_centroid_ordered = src_centroid  # centroid is permutation-invariant

    # Build per-sector affines: slot_pixel → grid_space (inverse direction)
    # Forward: grid → slot_pixel
    fwd_sectors = _build_sector_affines(
        src_ordered, src_centroid_ordered, dst_px_ordered, dst_px_centroid,
    )
    # Inverse: slot_pixel → grid
    inv_sectors = _build_sector_affines(
        dst_px_ordered, dst_px_centroid, src_ordered, src_centroid_ordered,
    )

    # Build output pixel grid
    oy, ox = np.mgrid[0:output_size, 0:output_size]
    out_pts = np.stack([ox.ravel().astype(np.float64),
                        oy.ravel().astype(np.float64)], axis=1)  # (M, 2)

    # Assign each output pixel to a sector in dst_px space
    sector_ids = _assign_sectors(out_pts, dst_px_centroid, dst_px_ordered)

    # Map output pixels → grid space → input pixels
    x_min, x_max = xlim
    y_min, y_max = ylim
    x_span = x_max - x_min
    y_span = y_max - y_min

    map_x = np.full(len(out_pts), -1.0, dtype=np.float64)
    map_y = np.full(len(out_pts), -1.0, dtype=np.float64)

    for i in range(len(inv_sectors)):
        A_inv, t_inv = inv_sectors[i]
        mask = sector_ids == i
        if not mask.any():
            continue
        pts = out_pts[mask]
        # slot_pixel → grid
        grid_pts = (pts @ A_inv.T) + t_inv
        # grid → input pixel
        # px_x = (gx - x_min) / x_span * img_w
        # px_y = (1 - (gy - y_min) / y_span) * img_h
        src_px_x = (grid_pts[:, 0] - x_min) / x_span * img_w
        src_px_y = (1.0 - (grid_pts[:, 1] - y_min) / y_span) * img_h
        map_x[mask] = src_px_x
        map_y[mask] = src_px_y

    map_x = map_x.reshape(output_size, output_size)
    map_y = map_y.reshape(output_size, output_size)
    return map_x, map_y


def _dilate_cval_pixels(
    arr: np.ndarray,
    cval: int = 128,
    iterations: int = 4,
) -> np.ndarray:
    """Replace ``cval``-fill pixels with nearest valid neighbour colour.

    The piecewise warp leaves ``(cval, cval, cval)`` at bounding-box
    corners that fall outside the polygon sector decomposition.
    Bilinear/mipmap texture sampling can bleed into these pixels, so
    we dilate valid colours outward to eliminate them.

    Parameters
    ----------
    arr : (H, W, 3) uint8 array
        Warped image (modified in-place and returned).
    cval : int
        The constant fill value used by ``map_coordinates``.
    iterations : int
        Number of dilation passes.  Each pass expands the valid region
        by one pixel in all 8 directions.
    """
    from scipy.ndimage import maximum_filter, minimum_filter

    fill = np.array([cval, cval, cval], dtype=np.uint8)
    mask = np.all(arr == fill, axis=-1)
    if not mask.any():
        return arr

    for _ in range(iterations):
        if not mask.any():
            break
        # For each channel, dilate by taking the max of the 3×3 neighbourhood
        # but only write into masked (cval) pixels.
        for ch in range(3):
            dilated = maximum_filter(arr[:, :, ch], size=3)
            # maximum_filter pushes cval outward too; use minimum_filter
            # on the *non-masked* values to get nearest valid colour.
            # Simpler approach: overwrite masked pixels with the dilated
            # value, then re-check.
            arr[:, :, ch][mask] = dilated[mask]

        # Re-check which pixels are still exactly cval (may have been
        # overwritten with valid colours that happen to equal cval — rare)
        mask = np.all(arr == fill, axis=-1)

    return arr


def warp_tile_to_uv(
    img: "Image.Image",
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    affine_grid_to_slot: np.ndarray,
    output_size: int,
    *,
    grid_corners: Optional[List[Tuple[float, float]]] = None,
    uv_corners: Optional[List[Tuple[float, float]]] = None,
    tile_size: Optional[int] = None,
    gutter: int = 0,
    src_centroid_override: Optional[np.ndarray] = None,
) -> "Image.Image":
    """Warp a stitched tile image so its polygon maps to the UV layout.

    Uses a **piecewise-linear** (triangle-fan) warp when
    ``grid_corners`` and ``uv_corners`` are supplied, giving exact
    boundary alignment that matches the ``UVTransform`` approach.
    Falls back to a single-affine warp (``affine_grid_to_slot``) if
    the polygon data is not provided.

    Parameters
    ----------
    img : PIL.Image.Image
        The rendered stitched tile (any size).
    xlim, ylim : (min, max)
        Axis limits used when rendering the image.
    affine_grid_to_slot : (2, 3) array
        From :func:`compute_grid_to_px_affine`.  Used as fallback
        when ``grid_corners`` / ``uv_corners`` are not given.
    output_size : int
        Width and height of the output image (slot_size = tile_size + 2*gutter).
    grid_corners : list of (x, y), optional
        Polygon corners in grid (Tutte) space.
    uv_corners : list of (u, v), optional
        UV polygon corners in [0, 1].
    tile_size : int, optional
        Inner tile size (pixels).  Required when using piecewise warp.
    gutter : int
        Gutter pixels.
    src_centroid_override : (2,) array, optional
        If given, use this as the source (grid) centroid for the
        piecewise-affine warp.  Allows fixing the fan centre when
        grid corners have been adjusted for sector equalisation.

    Returns
    -------
    PIL.Image.Image
        Warped image of size ``(output_size, output_size)``.
    """
    from PIL import Image
    from scipy.ndimage import map_coordinates

    img_w, img_h = img.size

    if grid_corners is not None and uv_corners is not None and tile_size is not None:
        # ── Piecewise-linear warp (exact boundary alignment) ────
        map_x, map_y = _compute_piecewise_warp_map(
            grid_corners, uv_corners,
            tile_size=tile_size,
            gutter=gutter,
            img_w=img_w, img_h=img_h,
            xlim=xlim, ylim=ylim,
            output_size=output_size,
            src_centroid_override=src_centroid_override,
        )

        src_arr = np.array(img.convert("RGB"), dtype=np.float64)
        # map_coordinates expects (row, col) = (y, x)
        out_channels = []
        for ch in range(3):
            warped_ch = map_coordinates(
                src_arr[:, :, ch],
                [map_y, map_x],
                order=1,          # bilinear
                mode="constant",
                cval=128.0,
            )
            out_channels.append(warped_ch.astype(np.uint8))

        out_arr = np.stack(out_channels, axis=-1)

        # Dilate any remaining cval-fill pixels (bounding-box corners
        # outside the polygon) so bilinear/mipmap sampling never
        # encounters the grey fallback colour.
        out_arr = _dilate_cval_pixels(out_arr)

        return Image.fromarray(out_arr, "RGB")

    # ── Fallback: single-affine warp (legacy) ──────────────────
    x_min, x_max = xlim
    y_min, y_max = ylim
    x_span = x_max - x_min
    y_span = y_max - y_min

    P = np.array([
        [x_span / img_w, 0.0, x_min],
        [0.0, -y_span / img_h, y_min + y_span],
    ], dtype=np.float64)

    def _to_3x3(m23):
        m = np.eye(3, dtype=np.float64)
        m[:2, :] = m23
        return m

    P33 = _to_3x3(P)
    M33 = _to_3x3(affine_grid_to_slot)
    forward = M33 @ P33
    inv = np.linalg.inv(forward)

    coeffs = (
        inv[0, 0], inv[0, 1], inv[0, 2],
        inv[1, 0], inv[1, 1], inv[1, 2],
    )

    rgb = img.convert("RGB")
    warped = rgb.transform(
        (output_size, output_size),
        Image.AFFINE,
        coeffs,
        resample=Image.BICUBIC,
        fillcolor=(128, 128, 128),
    )
    return warped


def _fill_warped_gaps(img: "Image.Image", cval: int = 128) -> "Image.Image":
    """Fill any pixels equal to *cval* by copying the nearest valid pixel.

    The piecewise warp uses ``cval`` as the fill value for pixels that map
    outside the source image.  In some polygons (hexes) the bbox corners
    remain unfilled; this routine replaces those pixels with the nearest
    valid pixel colour so bilinear/mipmap sampling at tile edges doesn't
    pick up the fallback colour.
    """
    from PIL import Image
    import numpy as np
    try:
        from scipy.spatial import cKDTree
    except Exception:
        # If scipy not available, return the image unchanged
        return img

    arr = np.array(img.convert("RGB"))
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mask_valid = ~((r == cval) & (g == cval) & (b == cval))
    if mask_valid.all():
        return img

    valid_y, valid_x = np.nonzero(mask_valid)
    inv_y, inv_x = np.nonzero(~mask_valid)
    if len(valid_y) == 0:
        return img

    # Build KD-tree on valid pixel coordinates (use row, col order)
    valid_coords = np.column_stack((valid_y, valid_x))
    tree = cKDTree(valid_coords)

    inv_coords = np.column_stack((inv_y, inv_x))
    _, idxs = tree.query(inv_coords, k=1)

    filled = arr.copy()
    filled[inv_y, inv_x] = arr[valid_y[idxs], valid_x[idxs]]
    return Image.fromarray(filled, "RGB")


def fill_sentinel_pixels(
    img: "Image.Image",
    sentinel: Tuple[int, int, int] = (255, 0, 255),
) -> "Image.Image":
    """Replace sentinel-coloured pixels with the nearest valid pixel.

    Stitched tile images are rendered with a bright magenta background
    so that image corners not covered by polygon patches can be
    identified.  This function replaces those sentinel pixels with the
    nearest non-sentinel pixel, preventing per-tile background colour
    from bleeding into the atlas gutter and causing visible seams.

    Parameters
    ----------
    img : PIL.Image.Image
        Source image (RGB).
    sentinel : (R, G, B)
        The sentinel colour to replace.  Default is bright magenta
        ``(255, 0, 255)``.

    Returns
    -------
    PIL.Image.Image
        Image with sentinel pixels replaced.
    """
    from PIL import Image

    try:
        from scipy.spatial import cKDTree
    except Exception:
        return img

    arr = np.array(img.convert("RGB"))
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mask_sentinel = (
        (r == sentinel[0]) & (g == sentinel[1]) & (b == sentinel[2])
    )
    if not mask_sentinel.any():
        return img

    valid_y, valid_x = np.nonzero(~mask_sentinel)
    inv_y, inv_x = np.nonzero(mask_sentinel)
    if len(valid_y) == 0:
        return img

    valid_coords = np.column_stack((valid_y, valid_x))
    tree = cKDTree(valid_coords)

    inv_coords = np.column_stack((inv_y, inv_x))
    _, idxs = tree.query(inv_coords, k=1)

    filled = arr.copy()
    filled[inv_y, inv_x] = arr[valid_y[idxs], valid_x[idxs]]
    return Image.fromarray(filled, "RGB")


# ═══════════════════════════════════════════════════════════════════
# 21C — Atlas assembly from polygon-cut tiles
# ═══════════════════════════════════════════════════════════════════

def build_polygon_cut_atlas(
    tile_images: Dict[str, "Image.Image"],
    composites: Dict[str, object],
    detail_grids: Dict[str, PolyGrid],
    globe_grid: PolyGrid,
    face_ids: List[str],
    *,
    tile_size: int = 256,
    gutter: int = 4,
    mask_outside: bool = False,
    mask_colour: Tuple[int, int, int] = (0, 0, 0),
    debug_labels: bool = False,
    output_dir: Optional[Path] = None,
    pentagon_rotation_steps: int = 0,
) -> Tuple["Image.Image", Dict[str, Tuple[float, float, float, float]]]:
    """Build a texture atlas from stitched tile images, UV-aligned.

    For each tile:
    1. Finds the polygon corners in the original detail grid.
    2. Gets the GoldbergTile UV polygon from the models library.
    3. Computes the affine warp from grid space → atlas slot pixels.
    4. Warps the stitched image so the polygon lands in UV-correct
       orientation within the slot.

    The warped image fills the **full slot** (including gutter), with
    neighbour terrain from the stitched image naturally providing
    gutter content.

    Parameters
    ----------
    tile_images : dict
        ``{face_id: PIL.Image}`` — rendered stitched tile images.
    composites : dict
        ``{face_id: CompositeGrid}`` — stitched composites (for view limits).
    detail_grids : dict
        ``{face_id: PolyGrid}`` — original (un-stitched) detail grids.
    globe_grid : PolyGrid
        Globe grid with ``metadata["frequency"]``.
    face_ids : list of str
        Ordered list of face ids to pack.
    tile_size : int
    gutter : int
    mask_outside : bool
        If True, pixels outside the UV polygon are filled with
        ``mask_colour``.  Useful for debugging; off by default
        because the 3D renderer only samples inside the polygon.
    mask_colour : (R, G, B)
        Fill colour for outside-polygon pixels.  Default black.
    debug_labels : bool
        If True, draw tile ID and per-edge neighbour labels on each
        tile in the atlas.  Adjacent tiles sharing an edge should
        display the **same** edge index at the shared boundary.
    output_dir : Path, optional
        If given, saves individual masked tiles for debugging.
    pentagon_rotation_steps : int
        Extra rotation steps applied to pentagon tiles only.
        Positive = clockwise.  Use to correct any residual
        pentagon orientation mismatch (default 0).

    Returns
    -------
    (atlas, uv_layout)
        atlas : PIL.Image.Image
        uv_layout : ``{face_id: (u_min, v_min, u_max, v_max)}``
    """
    from PIL import Image
    from .uv_texture import get_tile_uv_vertices

    n = len(face_ids)
    columns, rows, atlas_w, atlas_h = compute_atlas_layout(
        n, tile_size, gutter,
    )
    slot_size = tile_size + 2 * gutter
    atlas = Image.new("RGB", (atlas_w, atlas_h), (128, 128, 128))

    uv_layout: Dict[str, Tuple[float, float, float, float]] = {}

    for idx, fid in enumerate(face_ids):
        if fid not in tile_images:
            continue

        col = idx % columns
        row = idx // columns
        slot_x = col * slot_size
        slot_y = row * slot_size

        tile_img = tile_images[fid]

        # Replace any sentinel-coloured background pixels (image
        # corners not covered by polygon patches) with nearest valid
        # pixel to prevent per-tile bg colour from causing seams.
        tile_img = fill_sentinel_pixels(tile_img)

        # Get polygon corners from macro edges.
        dg = detail_grids[fid]
        n_sides = len(globe_grid.faces[fid].vertex_ids)
        is_pentagon = n_sides == 5

        # Pentagon tiles store explicit corner_vertex_ids because
        # angle-based detection is unreliable in the Tutte embedding.
        corner_ids = dg.metadata.get("corner_vertex_ids") if is_pentagon else None
        dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)
        grid_corners_raw = get_macro_edge_corners(dg, n_sides)

        # Get UV polygon from GoldbergTile (raw = GoldbergTile order).
        uv_corners_raw = get_tile_uv_vertices(globe_grid, fid)

        # ── Match grid corners → UV corners ──────────────────────
        # Use angular matching (handles both rotation and reflection
        # between macro-edge and GoldbergTile orderings) then equalise
        # sector ratios to correct the Tutte embedding's residual
        # rotational offset and any irregular UV spacing.
        grid_corners_matched = match_grid_corners_to_uv(
            grid_corners_raw, globe_grid, fid,
        )
        uv_corners = uv_corners_raw

        grid_corners, src_centroid = _equalise_sector_ratios(
            grid_corners_matched, uv_corners,
            tile_size=tile_size, gutter=gutter,
        )
        if is_pentagon:
            grid_corners = _scale_corners_from_centroid(
                grid_corners, _PENTAGON_GRID_SCALE,
            )

        # Compute view limits (same as the renderer used)
        comp = composites[fid]
        xlim, ylim = compute_tile_view_limits(comp, fid)

        # Compute affine: grid coords → slot pixel coords (fallback)
        affine = compute_grid_to_px_affine(
            grid_corners, uv_corners,
            tile_size=tile_size,
            gutter=gutter,
        )

        # Warp the image (piecewise-linear for exact boundary alignment)
        warped = warp_tile_to_uv(
            tile_img, xlim, ylim, affine, slot_size,
            grid_corners=grid_corners,
            uv_corners=uv_corners,
            tile_size=tile_size,
            gutter=gutter,
            src_centroid_override=src_centroid,
        )

        # Fill any remaining fallback-colour pixels left by the
        # piecewise warp at bounding-box corners outside the polygon.
        warped = _fill_warped_gaps(warped, cval=128)

        # Mask outside the UV polygon
        if mask_outside:
            warped = mask_warped_to_uv_polygon(
                warped, uv_corners,
                tile_size=tile_size, gutter=gutter,
                fill_colour=mask_colour,
            )

        # Draw debug labels (tile ID + edge→neighbour)
        if debug_labels:
            from .detail_terrain import compute_neighbor_edge_mapping
            # compute_neighbor_edge_mapping returns {neighbour_id: edge_index}
            # in PolyGrid (vertex_ids) order.  We need edge indices in
            # GoldbergTile order (matching uv_corners) so the labels
            # appear at the correct UV edge.
            neigh_to_edge_pg = compute_neighbor_edge_mapping(globe_grid, fid)
            pg_gt_offset = compute_uv_to_polygrid_offset(globe_grid, fid)
            edge_to_neigh_gt = {}
            for nid, pg_eidx in neigh_to_edge_pg.items():
                gt_eidx = (pg_eidx - pg_gt_offset) % n_sides
                edge_to_neigh_gt[gt_eidx] = nid
            warped = draw_debug_labels(
                warped, uv_corners, fid, edge_to_neigh_gt,
                tile_size=tile_size, gutter=gutter,
            )

        # Paste into atlas
        atlas.paste(warped, (slot_x, slot_y))

        # Fill gutter from warped content (the neighbour terrain
        # should already be there from the stitched image, but
        # clamp edges as fallback for any boundary pixels)
        if gutter > 0:
            fill_gutter(atlas, slot_x, slot_y, tile_size, gutter)

        # Save debug tile if requested
        if output_dir is not None:
            warped.save(str(output_dir / f"{fid}_warped.png"))

        # UV coordinates (inner tile region in atlas)
        inner_x = slot_x + gutter
        inner_y = slot_y + gutter
        u_min = inner_x / atlas_w
        u_max = (inner_x + tile_size) / atlas_w
        v_min = 1.0 - (inner_y + tile_size) / atlas_h
        v_max = 1.0 - inner_y / atlas_h
        uv_layout[fid] = (u_min, v_min, u_max, v_max)

    return atlas, uv_layout
