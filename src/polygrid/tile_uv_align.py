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
   hex tiles adjacent to pentagons (``equalise_sectors``).
5. Compute a piecewise-linear (triangle-fan) warp that maps each
   pixel of the output slot back to the stitched source image.
6. Pack all warped tiles into a texture atlas with gutter padding.
7. Optionally blend boundary pixels along shared Goldberg edges so
   adjacent atlas tiles match exactly at the seam (``stitch_seams``).

Key functions
-------------
- :func:`get_macro_edge_corners`       — polygon corners from macro edges
- :func:`match_grid_corners_to_uv`     — angular alignment grid↔UV
- :func:`compute_polygon_corners_px`   — grid corners → pixel coords
- :func:`warp_tile_to_uv`             — piecewise-linear image warp
- :func:`_stitch_atlas_seams`          — seam enforcement post-pass
- :func:`build_polygon_cut_atlas`      — end-to-end atlas builder
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import numpy as np

from .atlas_utils import fill_gutter, compute_atlas_layout
from .polygrid import PolyGrid
from .tile_detail import find_polygon_corners, DetailGridCollection

if TYPE_CHECKING:
    from PIL import Image, ImageFont


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
    # The destination pixel space has Y flipped relative to grid space
    # (pixel y=0 is top, grid y increases upward).  Angles and angular
    # spans computed in dst_px space must be negated when placing
    # corners in grid space so the reconstructed polygon has the
    # correct (non-reflected) orientation.
    dst_sorted_angles = np.arctan2(
        dst_sorted[:, 1] - dst_px_c[1],
        dst_sorted[:, 0] - dst_px_c[0],
    )
    start_angle = -dst_sorted_angles[0]

    new_sorted = np.empty((n, 2), dtype=np.float64)
    angle = start_angle
    for i in range(n):
        new_sorted[i, 0] = gc_c[0] + new_R[i] * math.cos(angle)
        new_sorted[i, 1] = gc_c[1] + new_R[i] * math.sin(angle)
        angle -= dst_spans[i]

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


def _load_debug_font(size: int) -> "ImageFont.FreeTypeFont":
    """Load DejaVuSansMono-Bold at *size*, falling back gracefully."""
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _grid_to_slot_px(
    grid_corners: List[Tuple[float, float]],
    uv_corners: List[Tuple[float, float]],
    tile_size: int,
    gutter: int,
) -> "Callable[[float, float], Tuple[float, float]]":
    """Return a piecewise-affine mapper: grid (x, y) → slot pixel (px, py).

    Uses the same triangle-fan decomposition as the image warp so
    sub-face centroids land in the correct warped positions.
    """
    gc = np.array(grid_corners, dtype=np.float64)
    uv = np.array(uv_corners, dtype=np.float64)
    n = len(gc)
    src_c = gc.mean(axis=0)

    # UV → pixel
    dst_corners = np.empty_like(uv)
    for i in range(n):
        dst_corners[i, 0] = gutter + uv[i, 0] * tile_size
        dst_corners[i, 1] = gutter + (1.0 - uv[i, 1]) * tile_size
    dst_c = dst_corners.mean(axis=0)

    # Per-sector affines (grid → slot pixel)
    sectors = _build_sector_affines(gc, src_c, dst_corners, dst_c)

    # Corner angles for sector assignment
    corner_angles = np.arctan2(gc[:, 1] - src_c[1], gc[:, 0] - src_c[0])

    def _transform(gx: float, gy: float) -> Tuple[float, float]:
        pt = np.array([gx, gy])
        angle = math.atan2(gy - src_c[1], gx - src_c[0])
        # Find sector
        best_i = 0
        for i in range(n):
            j = (i + 1) % n
            a0 = corner_angles[i] % (2.0 * math.pi)
            a1 = corner_angles[j] % (2.0 * math.pi)
            pn = angle % (2.0 * math.pi)
            if a0 <= a1:
                if a0 <= pn < a1:
                    best_i = i
                    break
            else:
                if pn >= a0 or pn < a1:
                    best_i = i
                    break
        A, t = sectors[best_i]
        out = A @ pt + t
        return float(out[0]), float(out[1])

    return _transform


def draw_debug_labels(
    img: "Image.Image",
    uv_corners: List[Tuple[float, float]],
    face_id: str,
    edge_neighbours: Dict[int, str],
    tile_size: int,
    gutter: int = 0,
    *,
    detail_grid: Optional[PolyGrid] = None,
    grid_corners: Optional[List[Tuple[float, float]]] = None,
    face_type: Optional[str] = None,
    detail_rings: Optional[int] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> "Image.Image":
    """Draw comprehensive debug annotations on a warped slot image.

    Layers (drawn back-to-front):

    1. **Grid-line overlay** — sub-face outlines transformed to
       warped pixel space (semi-transparent white).
    2. **Tile ID watermark** — large semi-transparent text filling
       most of the polygon.  Pentagon tiles are prefixed with ``⬠``.
    3. **Sub-face labels** — local face ID (e.g. ``f3``) at each
       sub-face centroid, sized to fit the cell.  Drawn for
       ``detail_rings ≤ 4`` (at ``detail_rings = 5`` the 9 px font
       is too small to read).
    4. **Edge arrows** — directional arrows along each UV polygon
       edge showing winding order, with neighbour tile ID in black.
    5. **Corner vertex markers** — numbered green dots at each
       polygon corner, labels offset inward to avoid globe clipping.

    Parameters
    ----------
    img : PIL.Image.Image
        Warped slot image to annotate.
    uv_corners : list of (u, v)
        UV polygon corners in [0, 1] space.
    face_id : str
    edge_neighbours : dict
        ``{edge_index: neighbour_face_id}``.
    tile_size, gutter : int
    detail_grid : PolyGrid, optional
        The un-stitched detail grid for this tile.  Enables sub-face
        labels and grid-line overlay.
    grid_corners : list of (x, y), optional
        Polygon corners in grid (Tutte) space, matched to
        *uv_corners* ordering.  Required for sub-face labels.
    face_type : ``"pent"`` or ``"hex"``, optional
    detail_rings : int, optional
    xlim, ylim : (min, max), optional
        View limits — only needed for grid-line overlay (transform
        from grid-space to image-space).

    Returns
    -------
    PIL.Image.Image
        Annotated copy of the image (RGBA).
    """
    from PIL import Image as _PILImage, ImageDraw, ImageFont

    slot_size = tile_size + 2 * gutter
    out = img.convert("RGBA")

    # UV → pixel helper
    def _uv_to_px(u: float, v: float) -> Tuple[float, float]:
        return (gutter + u * tile_size,
                gutter + (1.0 - v) * tile_size)

    n = len(uv_corners)
    cu = sum(c[0] for c in uv_corners) / n
    cv = sum(c[1] for c in uv_corners) / n
    cx, cy = _uv_to_px(cu, cv)

    is_pentagon = (face_type == "pent") if face_type else (n == 5)
    n_subfaces = len(detail_grid.faces) if detail_grid else 0
    rings = detail_rings or (detail_grid.metadata.get("detail_rings") if detail_grid else None)
    can_draw_subfaces = (
        detail_grid is not None
        and grid_corners is not None
        and rings is not None
        and rings <= 4
    )

    # Build piecewise grid→pixel transform (if we have data)
    _g2px = None
    if detail_grid is not None and grid_corners is not None:
        _g2px = _grid_to_slot_px(
            grid_corners, uv_corners, tile_size, gutter,
        )

    # ── Layer 1: Grid-line overlay ──────────────────────────────
    if _g2px is not None and detail_grid is not None:
        overlay = _PILImage.new("RGBA", (slot_size, slot_size), (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)
        for face in detail_grid.faces.values():
            verts_px = []
            for vid in face.vertex_ids:
                v = detail_grid.vertices.get(vid)
                if v is None or not v.has_position():
                    break
                px, py = _g2px(v.x, v.y)
                verts_px.append((px, py))
            else:
                if len(verts_px) >= 3:
                    verts_px.append(verts_px[0])  # close polygon
                    ov_draw.line(verts_px, fill=(255, 255, 255, 90), width=1)
        out = _PILImage.alpha_composite(out, overlay)

    # ── Layer 2: Tile ID watermark ──────────────────────────────
    watermark = _PILImage.new("RGBA", (slot_size, slot_size), (0, 0, 0, 0))
    wm_draw = ImageDraw.Draw(watermark)
    wm_label = face_id.upper()
    wm_font_size = max(24, tile_size // 3)
    wm_font = _load_debug_font(wm_font_size)
    bbox = wm_draw.textbbox((0, 0), wm_label, font=wm_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    wm_draw.text(
        (cx - tw / 2, cy - th / 2), wm_label,
        fill=(255, 255, 255, 70), font=wm_font,
    )
    out = _PILImage.alpha_composite(out, watermark)

    # ── Layer 3: Sub-face labels ────────────────────────────────
    if can_draw_subfaces and _g2px is not None:
        from .geometry import face_center as _face_center

        # Estimate cell pixel size → font size
        cell_px = tile_size / math.sqrt(max(1, n_subfaces))
        sf_font_size = max(8, int(cell_px * 0.17))
        sf_font = _load_debug_font(sf_font_size)

        sf_layer = _PILImage.new("RGBA", (slot_size, slot_size), (0, 0, 0, 0))
        sf_draw = ImageDraw.Draw(sf_layer)

        for fid, face in detail_grid.faces.items():
            fc = _face_center(detail_grid.vertices, face)
            if fc is None:
                continue
            px, py = _g2px(fc[0], fc[1])
            # Short label — just the face id (e.g. "f3")
            lbl = fid
            bb = sf_draw.textbbox((0, 0), lbl, font=sf_font)
            lw, lh = bb[2] - bb[0], bb[3] - bb[1]
            tx = px - lw / 2
            ty = py - lh / 2
            # Outline for readability: dark shadow offset by 1px
            for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                sf_draw.text((tx + ox, ty + oy), lbl,
                             fill=(0, 0, 0, 200), font=sf_font)
            sf_draw.text((tx, ty), lbl, fill=(255, 255, 255, 220), font=sf_font)

        out = _PILImage.alpha_composite(out, sf_layer)

    # ── Layer 4: Edge arrows + neighbour labels ─────────────────
    edge_layer = _PILImage.new("RGBA", (slot_size, slot_size), (0, 0, 0, 0))
    edge_draw = ImageDraw.Draw(edge_layer)
    edge_font = _load_debug_font(max(11, tile_size // 18))

    for k in range(n):
        j = (k + 1) % n
        px0x, px0y = _uv_to_px(*uv_corners[k])
        px1x, px1y = _uv_to_px(*uv_corners[j])

        # Arrow line from 35% to 65% along the edge (kept inward
        # so it isn't clipped when rendered on the globe).
        ax0 = px0x + (px1x - px0x) * 0.35
        ay0 = px0y + (px1y - px0y) * 0.35
        ax1 = px0x + (px1x - px0x) * 0.65
        ay1 = px0y + (px1y - px0y) * 0.65
        # Push the arrow segment inward toward the polygon centre
        amx, amy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
        dx_a, dy_a = cx - amx, cy - amy
        da = math.hypot(dx_a, dy_a) or 1.0
        arrow_inset = min(14, da * 0.12)
        ax0 += dx_a / da * arrow_inset
        ay0 += dy_a / da * arrow_inset
        ax1 += dx_a / da * arrow_inset
        ay1 += dy_a / da * arrow_inset
        edge_draw.line([(ax0, ay0), (ax1, ay1)], fill=(0, 220, 220, 180), width=3)

        # Arrowhead at the 75% end
        edge_dx = px1x - px0x
        edge_dy = px1y - px0y
        elen = math.hypot(edge_dx, edge_dy) or 1.0
        ux, uy = edge_dx / elen, edge_dy / elen
        # Perpendicular
        px_perp, py_perp = -uy, ux
        head_len = min(8, elen * 0.08)
        head_w = head_len * 0.6
        tip_x, tip_y = ax1, ay1
        base_x = tip_x - ux * head_len
        base_y = tip_y - uy * head_len
        edge_draw.polygon([
            (tip_x, tip_y),
            (base_x + px_perp * head_w, base_y + py_perp * head_w),
            (base_x - px_perp * head_w, base_y - py_perp * head_w),
        ], fill=(0, 220, 220, 220))

        # Neighbour label near edge midpoint, pushed slightly inward
        mx = (px0x + px1x) / 2
        my = (px0y + px1y) / 2
        dx_in, dy_in = cx - mx, cy - my
        d_in = math.hypot(dx_in, dy_in) or 1.0
        inset = min(16, d_in * 0.18)
        mx += dx_in / d_in * inset
        my += dy_in / d_in * inset

        nid = edge_neighbours.get(k, "?")
        lbl_e = f"e{k}\u2192{nid}"

        # Rotation along edge
        angle_deg = -math.degrees(math.atan2(edge_dy, edge_dx))
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180

        bb_e = edge_draw.textbbox((0, 0), lbl_e, font=edge_font)
        ew, eh = bb_e[2] - bb_e[0], bb_e[3] - bb_e[1]
        pad_t = 4
        tmp = _PILImage.new("RGBA", (ew + pad_t * 2, eh + pad_t * 2), (0, 0, 0, 0))
        tmp_draw = ImageDraw.Draw(tmp)
        tmp_draw.text((pad_t, pad_t), lbl_e, fill=(0, 0, 0, 220), font=edge_font)
        rotated = tmp.rotate(angle_deg, resample=_PILImage.BICUBIC, expand=True)
        rw, rh = rotated.size
        paste_x = int(round(mx - rw / 2))
        paste_y = int(round(my - rh / 2))
        edge_layer.paste(rotated, (paste_x, paste_y), rotated)
        edge_draw = ImageDraw.Draw(edge_layer)

    out = _PILImage.alpha_composite(out, edge_layer)

    # ── Layer 5: Corner vertex markers ──────────────────────────
    corner_layer = _PILImage.new("RGBA", (slot_size, slot_size), (0, 0, 0, 0))
    c_draw = ImageDraw.Draw(corner_layer)
    c_font = _load_debug_font(max(9, tile_size // 22))
    dot_r = max(3, tile_size // 120)

    for k in range(n):
        vx, vy = _uv_to_px(*uv_corners[k])
        c_draw.ellipse(
            [vx - dot_r, vy - dot_r, vx + dot_r, vy + dot_r],
            fill=(50, 255, 50, 210),
        )
        vlbl = f"v{k}"
        # Offset label well inward so it isn't clipped on the globe
        dx_v, dy_v = cx - vx, cy - vy
        d_v = math.hypot(dx_v, dy_v) or 1.0
        off = max(dot_r + 10, d_v * 0.18)
        lx = vx + dx_v / d_v * off
        ly = vy + dy_v / d_v * off
        bb_v = c_draw.textbbox((0, 0), vlbl, font=c_font)
        vw, vh = bb_v[2] - bb_v[0], bb_v[3] - bb_v[1]
        # Outline
        for ox, oy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            c_draw.text((lx - vw / 2 + ox, ly - vh / 2 + oy), vlbl,
                        fill=(0, 0, 0, 200), font=c_font)
        c_draw.text((lx - vw / 2, ly - vh / 2), vlbl,
                    fill=(50, 255, 50, 240), font=c_font)

    out = _PILImage.alpha_composite(out, corner_layer)

    # Convert back to RGB for downstream pipeline
    return out.convert("RGB")


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
    src_grid = np.array(grid_corners, dtype=np.float64)
    dst_uv = np.array(uv_corners, dtype=np.float64)
    n = len(dst_uv)

    # ── Convert grid corners → source-image pixel coordinates ────
    # This keeps both sides of the affine in pixel space, eliminating
    # the mixed-coordinate-system bug where a Y-flip in the
    # grid→pixel conversion interacted badly with sector winding.
    x_min, x_max = xlim
    y_min, y_max = ylim
    x_span = x_max - x_min
    y_span = y_max - y_min

    src_px = np.empty_like(src_grid)
    for i in range(n):
        gx, gy = src_grid[i]
        src_px[i, 0] = (gx - x_min) / x_span * img_w
        src_px[i, 1] = (1.0 - (gy - y_min) / y_span) * img_h

    if src_centroid_override is not None:
        sc = np.asarray(src_centroid_override, dtype=np.float64)
        src_px_centroid = np.array([
            (sc[0] - x_min) / x_span * img_w,
            (1.0 - (sc[1] - y_min) / y_span) * img_h,
        ])
    else:
        src_px_centroid = src_px.mean(axis=0)

    # ── Convert UV corners → destination slot-pixel coordinates ──
    dst_px = np.empty_like(dst_uv)
    for i in range(n):
        u, v = dst_uv[i]
        dst_px[i, 0] = gutter + u * tile_size
        dst_px[i, 1] = gutter + (1.0 - v) * tile_size
    dst_px_centroid = dst_px.mean(axis=0)

    # Sort both arrays by *destination-pixel* angle from their
    # centroid.  Both arrays are now in pixel space (Y-down), so the
    # sort puts them in the same angular order with matching winding.
    dst_angles = np.arctan2(
        dst_px[:, 1] - dst_px_centroid[1],
        dst_px[:, 0] - dst_px_centroid[0],
    )
    order = np.argsort(dst_angles)
    src_px_ordered = src_px[order]
    dst_px_ordered = dst_px[order]

    # Build per-sector affines entirely in pixel space
    # Inverse: slot_pixel → source_pixel
    inv_sectors = _build_sector_affines(
        dst_px_ordered, dst_px_centroid, src_px_ordered, src_px_centroid,
    )

    # Build output pixel grid
    oy, ox = np.mgrid[0:output_size, 0:output_size]
    out_pts = np.stack([ox.ravel().astype(np.float64),
                        oy.ravel().astype(np.float64)], axis=1)  # (M, 2)

    # Assign each output pixel to a sector in dst_px space
    sector_ids = _assign_sectors(out_pts, dst_px_centroid, dst_px_ordered)

    # Map output pixels → source-image pixels directly
    map_x = np.full(len(out_pts), -1.0, dtype=np.float64)
    map_y = np.full(len(out_pts), -1.0, dtype=np.float64)

    for i in range(len(inv_sectors)):
        A_inv, t_inv = inv_sectors[i]
        mask = sector_ids == i
        if not mask.any():
            continue
        pts = out_pts[mask]
        mapped = (pts @ A_inv.T) + t_inv
        map_x[mask] = mapped[:, 0]
        map_y[mask] = mapped[:, 1]

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
    # The piecewise warp above is the production path.  If we reach
    # here, it means grid_corners / uv_corners were not provided.
    import warnings
    warnings.warn(
        "warp_tile_to_uv: falling back to single-affine warp — "
        "pass grid_corners and uv_corners for piecewise accuracy",
        DeprecationWarning,
        stacklevel=2,
    )
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
# 21B.1 — Atlas seam enforcement post-pass
# ═══════════════════════════════════════════════════════════════════

def _stitch_atlas_seams(
    atlas: "Image.Image",
    uv_layout: Dict[str, Tuple[float, float, float, float]],
    globe_grid: PolyGrid,
    face_ids: List[str],
    *,
    tile_size: int = 256,
    gutter: int = 4,
    stitch_width: int = 8,
) -> "Image.Image":
    """Cross-fade boundary pixels along shared edges in the packed atlas.

    For every pair of adjacent tiles that share a Goldberg edge, this
    function rasterises a band of pixels *stitch_width* deep on each
    side of that edge in **both** atlas slots and writes a weighted
    blend to both.  Pixels exactly on the edge receive a 50/50 mix;
    pixels further inside the tile fade linearly back to their
    original colour.  This **gradient cross-fade** hides structural
    misalignment of the sub-tile hex grid that simple averaging cannot
    fix — the two composites inevitably rasterise the boundary region
    from slightly different viewpoints, and averaging a narrow 2 px
    strip only conceals sub-pixel colour shifts, not the 2–3 px
    offset in grid line position.

    The default *stitch_width* of 8 is tuned for ``tile_size=512``
    with ``detail_rings=2`` (grid cells ≈ 100 px).  Increase for
    higher detail_rings; decrease if the blend visibly softens detail.

    Parameters
    ----------
    atlas : PIL.Image.Image
        The packed texture atlas — modified in-place and returned.
    uv_layout : dict
        ``{face_id: (u_min, v_min, u_max, v_max)}`` as returned by
        :func:`build_polygon_cut_atlas`.
    globe_grid : PolyGrid
    face_ids : list of str
    tile_size : int
    gutter : int
    stitch_width : int
        Half-width (in pixels) of the cross-fade band on each side
        of the shared edge.  Default 8.  Pixels at distance *d* from
        the edge receive blend weight ``d / stitch_width`` for the
        local tile and ``1 - d / stitch_width`` for the neighbour.

    Returns
    -------
    PIL.Image.Image
        The atlas with seams blended.
    """
    from PIL import Image as _PILImage

    try:
        from .uv_texture import _find_shared_edges, get_tile_uv_vertices
    except ImportError:
        return atlas

    shared_edges = _find_shared_edges(globe_grid, face_ids)
    if not shared_edges:
        return atlas

    atlas_w, atlas_h = atlas.size
    arr = np.array(atlas.convert("RGB"), dtype=np.float64)

    for fid_a, fid_b, shared_verts in shared_edges:
        if fid_a not in uv_layout or fid_b not in uv_layout:
            continue

        # UV corners for the shared edge (in tile-local [0,1] UV space)
        uv_verts_a = get_tile_uv_vertices(globe_grid, fid_a)
        uv_verts_b = get_tile_uv_vertices(globe_grid, fid_b)

        (ia0, ib0), (ia1, ib1) = shared_verts
        uv_a0 = np.array(uv_verts_a[ia0], dtype=np.float64)
        uv_a1 = np.array(uv_verts_a[ia1], dtype=np.float64)
        uv_b0 = np.array(uv_verts_b[ib0], dtype=np.float64)
        uv_b1 = np.array(uv_verts_b[ib1], dtype=np.float64)

        # Atlas slot origins (top-left pixel of each slot)
        u_min_a, v_min_a, u_max_a, v_max_a = uv_layout[fid_a]
        u_min_b, v_min_b, u_max_b, v_max_b = uv_layout[fid_b]

        # Slot pixel origin = top-left corner of inner tile region
        # (uv_layout u_min/v_max map to the inner tile, gutter sits outside)
        ox_a = round(u_min_a * atlas_w) - gutter
        oy_a = round((1.0 - v_max_a) * atlas_h) - gutter
        ox_b = round(u_min_b * atlas_w) - gutter
        oy_b = round((1.0 - v_max_b) * atlas_h) - gutter

        # Perpendicular direction (inward-pointing) for the blend band
        def _inward_perp(uv0, uv1, all_uv):
            edge = uv1 - uv0
            perp = np.array([-edge[1], edge[0]], dtype=np.float64)
            n = np.linalg.norm(perp)
            if n > 1e-12:
                perp /= n
            centroid = np.mean(all_uv, axis=0)
            mid = (uv0 + uv1) * 0.5
            if np.dot(perp, centroid - mid) < 0:
                perp = -perp
            return perp

        perp_a = _inward_perp(uv_a0, uv_a1,
                              np.array(uv_verts_a, dtype=np.float64))
        perp_b = _inward_perp(uv_b0, uv_b1,
                              np.array(uv_verts_b, dtype=np.float64))

        # Rasterise sample points along the shared edge
        n_along = max(tile_size * 4, 256)
        t_vals = np.linspace(0.0, 1.0, n_along)
        # Offsets from 0 (on the edge) to stitch_width (deepest inside)
        w_vals = np.arange(0, stitch_width + 1, dtype=np.float64)
        px_uv = 1.0 / tile_size  # one pixel in UV space

        # UV positions along the edge: (n_along, 2)
        base_a = (1.0 - t_vals[:, None]) * uv_a0 + t_vals[:, None] * uv_a1
        base_b = (1.0 - t_vals[:, None]) * uv_b0 + t_vals[:, None] * uv_b1

        # Cross-fade weights: at offset 0 → 0.5 local / 0.5 remote;
        # at offset stitch_width → 1.0 local / 0.0 remote.
        # alpha_local(d) = 0.5 + 0.5 * (d / stitch_width)
        alpha_local = 0.5 + 0.5 * w_vals / max(stitch_width, 1)

        for w_idx, w in enumerate(w_vals):
            # Offset *inward* into tile A (positive perp direction)
            off_a = w * px_uv * perp_a
            uv_a_pts = base_a + off_a[None, :]  # (n_along, 2)
            # Corresponding offset *inward* into tile B
            off_b = w * px_uv * perp_b
            uv_b_pts = base_b + off_b[None, :]

            ax = np.round(ox_a + gutter + uv_a_pts[:, 0] * tile_size).astype(np.int32)
            ay = np.round(oy_a + gutter + (1.0 - uv_a_pts[:, 1]) * tile_size).astype(np.int32)
            bx = np.round(ox_b + gutter + uv_b_pts[:, 0] * tile_size).astype(np.int32)
            by = np.round(oy_b + gutter + (1.0 - uv_b_pts[:, 1]) * tile_size).astype(np.int32)

            valid = (
                (ax >= 0) & (ax < atlas_w) & (ay >= 0) & (ay < atlas_h) &
                (bx >= 0) & (bx < atlas_w) & (by >= 0) & (by < atlas_h)
            )
            v_idx = np.where(valid)[0]
            if len(v_idx) == 0:
                continue

            ixa = ax[v_idx]
            iya = ay[v_idx]
            ixb = bx[v_idx]
            iyb = by[v_idx]

            ca = arr[iya, ixa]  # (N, 3)
            cb = arr[iyb, ixb]

            # For tile A at this offset: blend toward B's colour.
            # alpha_local[w_idx] is how much A keeps; rest comes from B.
            a_loc = alpha_local[w_idx]
            a_rem = 1.0 - a_loc
            arr[iya, ixa] = a_loc * ca + a_rem * cb

            # For tile B at this offset: blend toward A's colour
            # (symmetric — B keeps the same fraction of itself).
            arr[iyb, ixb] = a_loc * cb + a_rem * ca

    return _PILImage.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


# ═══════════════════════════════════════════════════════════════════
# 21B.2 — Corner-junction blending post-pass
# ═══════════════════════════════════════════════════════════════════

def _find_vertex_junctions(
    globe_grid: PolyGrid,
    face_ids: List[str],
) -> Dict[str, List[Tuple[str, int]]]:
    """Build a vertex → incident-tile map for globe polygon vertices.

    Returns ``{vertex_id: [(face_id, vertex_index), …]}`` where
    *vertex_index* is the position of the vertex within
    ``globe_grid.faces[face_id].vertex_ids``.

    Only vertices incident to **two or more** tiles that are in
    *face_ids* are returned (these are the junction points where
    texture mismatches can appear).
    """
    fid_set = set(face_ids)
    vertex_to_faces: Dict[str, List[Tuple[str, int]]] = {}

    for fid in face_ids:
        face = globe_grid.faces.get(fid)
        if face is None:
            continue
        for idx, vid in enumerate(face.vertex_ids):
            vertex_to_faces.setdefault(vid, []).append((fid, idx))

    # Keep only junctions (≥ 2 incident atlas tiles)
    return {
        vid: entries
        for vid, entries in vertex_to_faces.items()
        if len(entries) >= 2
    }


def _blend_corner_junctions(
    atlas: "Image.Image",
    uv_layout: Dict[str, Tuple[float, float, float, float]],
    globe_grid: PolyGrid,
    face_ids: List[str],
    *,
    tile_size: int = 256,
    gutter: int = 4,
    blend_radius: int = 2,
) -> "Image.Image":
    """Average atlas pixels at multi-tile vertex junctions.

    At every globe vertex shared by two or more tiles, each incident
    tile has a UV corner that maps to a small pixel neighbourhood in
    its atlas slot.  If the warped textures differ slightly at that
    corner (due to independent sampling), a tiny wedge artefact can
    appear on the rendered globe.

    This function samples a disc of *blend_radius* pixels around
    each junction in every incident tile's atlas slot, computes the
    mean colour, and writes it back to all of them — the atlas-space
    analogue of ``_stitch_atlas_seams`` but for point junctions
    rather than edge bands.

    Parameters
    ----------
    atlas : PIL.Image.Image
        Packed atlas — modified in-place and returned.
    uv_layout : dict
        ``{face_id: (u_min, v_min, u_max, v_max)}``.
    globe_grid : PolyGrid
    face_ids : list of str
    tile_size : int
    gutter : int
    blend_radius : int
        Radius in pixels of the averaging disc around each junction
        (default 2).  A small value is enough — the artefact is
        localised to 1–2 pixels.

    Returns
    -------
    PIL.Image.Image
        The atlas with corner junctions blended.
    """
    from PIL import Image as _PILImg

    try:
        from .uv_texture import get_tile_uv_vertices
    except ImportError:
        return atlas

    junctions = _find_vertex_junctions(globe_grid, face_ids)
    if not junctions:
        return atlas

    atlas_w, atlas_h = atlas.size
    arr = np.array(atlas.convert("RGB"), dtype=np.float64)

    # Pre-build a disc mask of offsets within *blend_radius* pixels.
    offsets = []
    for dy in range(-blend_radius, blend_radius + 1):
        for dx in range(-blend_radius, blend_radius + 1):
            if dx * dx + dy * dy <= blend_radius * blend_radius:
                offsets.append((dy, dx))
    offsets_arr = np.array(offsets, dtype=np.int32)  # (K, 2)  [dy, dx]

    for _vid, entries in junctions.items():
        # Collect atlas pixel centres for this junction across tiles.
        centres = []  # list of (atlas_y, atlas_x) per incident tile
        for fid, vtx_idx in entries:
            if fid not in uv_layout:
                continue
            uv_verts = get_tile_uv_vertices(globe_grid, fid)
            if vtx_idx >= len(uv_verts):
                continue
            u, v = uv_verts[vtx_idx]

            # Atlas slot pixel: same formula as _stitch_atlas_seams
            u_min, v_min, u_max, v_max = uv_layout[fid]
            ox = round(u_min * atlas_w) - gutter
            oy = round((1.0 - v_max) * atlas_h) - gutter
            px_x = int(round(ox + gutter + u * tile_size))
            px_y = int(round(oy + gutter + (1.0 - v) * tile_size))
            centres.append((px_y, px_x))

        if len(centres) < 2:
            continue

        # Gather pixel values from the disc around each centre,
        # compute the global mean, and write it back.
        all_vals = []
        pixel_locs = []  # (y, x) arrays per centre
        for cy, cx in centres:
            ys = offsets_arr[:, 0] + cy
            xs = offsets_arr[:, 1] + cx
            valid = (ys >= 0) & (ys < atlas_h) & (xs >= 0) & (xs < atlas_w)
            ys = ys[valid]
            xs = xs[valid]
            if len(ys) == 0:
                pixel_locs.append((np.array([], dtype=np.int32),
                                   np.array([], dtype=np.int32)))
                continue
            all_vals.append(arr[ys, xs])
            pixel_locs.append((ys, xs))

        if not all_vals:
            continue

        mean_colour = np.mean(np.concatenate(all_vals, axis=0), axis=0)

        for ys, xs in pixel_locs:
            if len(ys) > 0:
                arr[ys, xs] = mean_colour

    return _PILImg.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


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
    stitch_seams: bool = False,
    stitch_width: int = 8,
    equalise_sectors: bool = False,
    blend_corners: bool = False,
    blend_radius: int = 2,
) -> Tuple["Image.Image", Dict[str, Tuple[float, float, float, float]]]:
    """Build a texture atlas from stitched tile images, UV-aligned.

    For each tile:
    1. Finds the polygon corners in the original detail grid.
    2. Gets the GoldbergTile UV polygon from the models library.
    3. Computes the affine warp from grid space → atlas slot pixels.
    4. Warps the stitched image so the polygon lands in UV-correct
       orientation within the slot.
    5. (Optional) Blends boundary pixels along shared Goldberg edges
       so adjacent tiles match exactly at the seam.

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
    stitch_seams : bool
        If True, run a cross-fade post-pass along shared Goldberg
        edges so adjacent atlas tiles transition smoothly.
        Default False — disabled to avoid masking genuine alignment
        issues that should be fixed at the warp/composite level.
    stitch_width : int
        Half-width of the cross-fade band in pixels (default 8).
        Only used when ``stitch_seams=True``.
    equalise_sectors : bool
        If True, apply ``_equalise_sector_ratios`` to irregular hex
        tiles (those adjacent to pentagons) so the piecewise-warp
        sectors become conformal.  Default False (conservative).
    blend_corners : bool
        If True, run a post-pass that averages pixels at multi-tile
        vertex junctions.  Default False — disabled to avoid masking
        genuine alignment issues.
    blend_radius : int
        Radius in pixels of the averaging disc at each corner
        junction (default 2).  Only used when ``blend_corners=True``.

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
        # between macro-edge and GoldbergTile orderings).
        grid_corners_matched = match_grid_corners_to_uv(
            grid_corners_raw, globe_grid, fid,
        )
        uv_corners = uv_corners_raw

        # ── Corner adjustment strategy ───────────────────────────
        # Pentagon tiles: slight outward scale to avoid boundary
        # cropping in the warp.
        # Irregular hex tiles (adjacent to a pentagon): optionally
        # apply _equalise_sector_ratios so the piecewise-warp sectors
        # become conformal — the seam post-pass compensates for any
        # boundary shift this introduces.
        # Regular hex tiles: use matched corners directly.
        grid_corners = grid_corners_matched
        src_centroid = None
        if is_pentagon:
            grid_corners = _scale_corners_from_centroid(
                grid_corners, _PENTAGON_GRID_SCALE,
            )
        elif equalise_sectors:
            # Only worthwhile for irregular hexes (those adjacent to
            # a pentagon).  Regular hexes have uniform sector geometry
            # and _equalise_sector_ratios is a no-op for them.
            grid_corners, src_centroid = _equalise_sector_ratios(
                grid_corners_matched, uv_corners,
                tile_size=tile_size, gutter=gutter,
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

        # Draw debug labels (tile ID, sub-face IDs, edge arrows, etc.)
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
            face_type = globe_grid.faces[fid].face_type
            d_rings = dg.metadata.get("detail_rings")
            warped = draw_debug_labels(
                warped, uv_corners, fid, edge_to_neigh_gt,
                tile_size=tile_size, gutter=gutter,
                detail_grid=dg,
                grid_corners=grid_corners,
                face_type=face_type,
                detail_rings=d_rings,
                xlim=xlim,
                ylim=ylim,
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

    # ── Seam enforcement post-pass ───────────────────────────────
    if stitch_seams:
        atlas = _stitch_atlas_seams(
            atlas, uv_layout, globe_grid, face_ids,
            tile_size=tile_size, gutter=gutter,
            stitch_width=stitch_width,
        )

    # ── Corner-junction blending post-pass ───────────────────────
    if blend_corners:
        atlas = _blend_corner_junctions(
            atlas, uv_layout, globe_grid, face_ids,
            tile_size=tile_size, gutter=gutter,
            blend_radius=blend_radius,
        )

    return atlas, uv_layout
