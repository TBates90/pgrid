# NOTE: 2780-line monolith — consider splitting into tile_warp.py (piecewise warp),
# tile_atlas.py (atlas packing + gutter fill), tile_debug.py (debug labels/overlay).
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
7. Enforce orientation/corner correctness directly; no seam-blend
    post-pass compensation is used in the production path.

Key functions
-------------
- :func:`get_macro_edge_corners`       — polygon corners from macro edges
- :func:`match_grid_corners_to_uv`     — angular alignment grid↔UV
- :func:`compute_polygon_corners_px`   — grid corners → pixel coords
- :func:`warp_tile_to_uv`             — piecewise-linear image warp
- :func:`build_polygon_cut_atlas`      — end-to-end atlas builder
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .atlas_utils import fill_gutter, compute_atlas_layout
from ..core.polygrid import PolyGrid
from ..detail.tile_detail import find_polygon_corners, DetailGridCollection

if TYPE_CHECKING:
    from PIL import Image, ImageFont


LOGGER = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _count_rgb_fill_pixels(img: "Image.Image", *, fill: int = 128) -> int:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    return int(np.count_nonzero(np.all(arr == fill, axis=-1)))


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


def _is_rotation_map(index_map: List[int]) -> bool:
    n = len(index_map)
    if n < 3:
        return False
    return len({(index_map[k] - k) % n for k in range(n)}) == 1


def _is_reflection_map(index_map: List[int]) -> bool:
    n = len(index_map)
    if n < 3:
        return False
    return len({(index_map[k] + k) % n for k in range(n)}) == 1


def _select_corner_match_indices(
    macro_angles: np.ndarray,
    gt_angles: np.ndarray,
    *,
    allow_reflection: bool,
) -> Tuple[List[int], str, float, float, int, int]:
    """Select best corner-match permutation by angular error.

    Returns ``(indices, mode, rot_err, ref_err, best_rot, best_ref)``.
    ``mode`` is always ``"rotation"`` when *allow_reflection* is False.
    """
    n = len(macro_angles)

    def _angular_diff(a: float, b: float) -> float:
        d = abs(a - b) % (2 * math.pi)
        return min(d, 2 * math.pi - d)

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

    best_ref_err = float("inf")
    best_ref = 0
    if allow_reflection:
        for ref in range(n):
            err = sum(
                _angular_diff(macro_angles[k], gt_angles[(ref - k) % n])
                for k in range(n)
            )
            if err < best_ref_err:
                best_ref_err = err
                best_ref = ref

    if not allow_reflection or best_rot_err <= best_ref_err:
        indices = [(k - best_rot) % n for k in range(n)]
        mode = "rotation"
    else:
        indices = [(best_ref - k) % n for k in range(n)]
        mode = "reflection"

    return indices, mode, best_rot_err, best_ref_err, best_rot, best_ref


def _should_use_pent_reflection(
    best_rot_err: float,
    best_ref_err: float,
    *,
    min_relative_improvement: float = 0.15,
) -> bool:
    """Return True only when pent reflection is decisively better.

    This prevents near-tie reflection picks that can visibly mispair
    pent-hex seam corners in low-frequency layouts.
    """
    if not (math.isfinite(best_rot_err) and math.isfinite(best_ref_err)):
        return False
    if best_ref_err >= best_rot_err:
        return False
    denom = max(best_rot_err, 1e-12)
    improvement = (best_rot_err - best_ref_err) / denom
    return improvement >= float(min_relative_improvement)


def _signed_area(points: np.ndarray) -> float:
    """Return signed polygon area using the shoelace formula."""
    if points.ndim != 2 or points.shape[0] < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _winding_sign(points: np.ndarray, eps: float = 1e-12) -> int:
    """Return polygon winding sign: +1, -1, or 0 for degenerate."""
    area = _signed_area(points)
    if area > eps:
        return 1
    if area < -eps:
        return -1
    return 0


def _reverse_preserving_anchor(
    corners: np.ndarray,
    anchor: int = 0,
) -> np.ndarray:
    """Reverse cyclic winding while preserving one anchor corner index."""
    n = corners.shape[0]
    if n < 3:
        return corners
    out = np.empty_like(corners)
    out[anchor] = corners[anchor]
    write = (anchor + 1) % n
    read = (anchor - 1) % n
    for _ in range(n - 1):
        out[write] = corners[read]
        write = (write + 1) % n
        read = (read - 1) % n
    return out


def _normalize_ordered_pentagon_winding(
    src_px_ordered: np.ndarray,
    dst_px_ordered: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """Ensure ordered pentagon source corners share destination winding."""
    if src_px_ordered.shape[0] != 5 or dst_px_ordered.shape[0] != 5:
        return src_px_ordered, False

    src_sign = _winding_sign(src_px_ordered)
    dst_sign = _winding_sign(dst_px_ordered)
    if src_sign == 0 or dst_sign == 0 or src_sign == dst_sign:
        return src_px_ordered, False

    fixed = _reverse_preserving_anchor(src_px_ordered, anchor=0)
    if _winding_sign(fixed) == dst_sign:
        return fixed, True

    fallback = src_px_ordered[::-1].copy()
    if _winding_sign(fallback) == dst_sign:
        return fallback, True

    return src_px_ordered, False


def _normalize_pentagon_winding_for_warp(
    grid_corners: List[Tuple[float, float]],
    uv_corners: List[Tuple[float, float]],
    *,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    img_w: int,
    img_h: int,
    tile_size: int,
    gutter: int,
    face_id: Optional[str] = None,
) -> Tuple[List[Tuple[float, float]], bool]:
    """Normalize pentagon source winding to match destination UV winding."""
    if len(grid_corners) != 5 or len(uv_corners) != 5:
        return grid_corners, False

    x_min, x_max = xlim
    y_min, y_max = ylim
    x_span = x_max - x_min
    y_span = y_max - y_min
    if abs(x_span) < 1e-12 or abs(y_span) < 1e-12:
        return grid_corners, False

    src = np.asarray(grid_corners, dtype=np.float64)
    dst = np.asarray(uv_corners, dtype=np.float64)
    src_px = np.empty_like(src)
    dst_px = np.empty_like(dst)

    for i in range(5):
        gx, gy = src[i]
        src_px[i, 0] = (gx - x_min) / x_span * img_w
        src_px[i, 1] = (1.0 - (gy - y_min) / y_span) * img_h
        u, v = dst[i]
        dst_px[i, 0] = gutter + u * tile_size
        dst_px[i, 1] = gutter + (1.0 - v) * tile_size

    pre_src_sign = _winding_sign(src_px)
    pre_dst_sign = _winding_sign(dst_px)

    src_fixed, changed = _normalize_ordered_pentagon_winding(src_px, dst_px)
    if not changed:
        if _env_flag("PGRID_ORIENTATION_AUDIT") and face_id is not None:
            LOGGER.info(
                "orientation-audit face=%s stage=pent-winding changed=0 pre_src_sign=%d pre_dst_sign=%d post_src_sign=%d",
                face_id,
                pre_src_sign,
                pre_dst_sign,
                _winding_sign(src_fixed),
            )
        return grid_corners, False

    # Re-derive grid-space corners from the corrected source pixel corners.
    fixed = np.empty_like(src_fixed)
    fixed[:, 0] = x_min + (src_fixed[:, 0] / img_w) * x_span
    fixed[:, 1] = y_min + (1.0 - (src_fixed[:, 1] / img_h)) * y_span
    result = [(float(fixed[i, 0]), float(fixed[i, 1])) for i in range(5)]

    if _env_flag("PGRID_ORIENTATION_AUDIT") and face_id is not None:
        LOGGER.info(
            "orientation-audit face=%s stage=pent-winding changed=1 pre_src_sign=%d pre_dst_sign=%d post_src_sign=%d",
            face_id,
            pre_src_sign,
            pre_dst_sign,
            _winding_sign(src_fixed),
        )

    return result, True


def compute_pg_to_macro_corner_map(
    globe_grid: PolyGrid,
    face_id: str,
    detail_grid: PolyGrid,
) -> Dict[int, int]:
    """Map PolyGrid (vertex_ids) corner indices to macro-corner indices.

    ``compute_neighbor_edge_mapping`` returns edge indices in
    PolyGrid ``vertex_ids`` order — edge *k* connects ``vertex_ids[k]``
    to ``vertex_ids[(k+1) % n]``.  The detail-grid macro-edges are
    numbered by the Tutte boundary walk, which can have **opposite
    winding** for hexagonal tiles (CW macro vs CCW PG).

    This function matches **corners** (macro corner → PG vertex) by
    angular proximity, then inverts the map so callers can resolve a
    PolyGrid corner index directly to its corresponding macro corner.

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
        ``{pg_corner_index: macro_corner_index}`` — for every
        ``vertex_ids`` index in the PolyGrid face ordering.
    """
    from .uv_texture import compute_tile_basis

    face = globe_grid.faces[face_id]
    n = len(face.vertex_ids)
    if n < 3:
        raise ValueError(f"Face {face_id} has invalid side count: {n}")

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

    mapped_pg = [macro_corner_to_pg[k] for k in range(n)]
    if len(set(mapped_pg)) != n:
        raise ValueError(
            f"Corner mapping is non-bijective for face {face_id}: {mapped_pg}"
        )

    # Detect rotation vs reflection.
    # Rotation: macro_corner_to_pg[k] = (k + offset) % n  (constant offset)
    # Reflection: macro_corner_to_pg[k] = (R - k) % n     (constant sum)
    offsets = [(macro_corner_to_pg[k] - k) % n for k in range(n)]
    sums = [(macro_corner_to_pg[k] + k) % n for k in range(n)]
    is_reflected = len(set(sums)) == 1 and len(set(offsets)) > 1
    if not is_reflected and len(set(offsets)) != 1:
        raise ValueError(
            f"Corner mapping is neither pure rotation nor reflection for face {face_id}"
        )

    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "face %s corner-map orientation=%s n=%d mapped_pg=%s offsets=%s sums=%s",
            face_id,
            "reflection" if is_reflected else "rotation",
            n,
            mapped_pg,
            offsets,
            sums,
        )

    if _env_flag("PGRID_ORIENTATION_AUDIT"):
        LOGGER.info(
            "orientation-audit face=%s stage=corner-map orientation=%s n=%d mapped_pg=%s offsets=%s sums=%s",
            face_id,
            "reflection" if is_reflected else "rotation",
            n,
            mapped_pg,
            offsets,
            sums,
        )

    # Invert corner map: pg_vertex → macro_corner
    pg_to_macro_corner: Dict[int, int] = {
        pg: macro for macro, pg in macro_corner_to_pg.items()
    }

    return pg_to_macro_corner


def compute_pg_to_macro_edge_map(
    globe_grid: PolyGrid,
    face_id: str,
    detail_grid: PolyGrid,
) -> Dict[int, int]:
    """Map PolyGrid (vertex_ids) **edge** indices to macro-edge indices."""
    face = globe_grid.faces[face_id]
    n = len(face.vertex_ids)
    if n < 3:
        raise ValueError(f"Face {face_id} has invalid side count: {n}")

    pg_to_macro_corner = compute_pg_to_macro_corner_map(
        globe_grid,
        face_id,
        detail_grid,
    )

    macro_to_pg = {macro: pg for pg, macro in pg_to_macro_corner.items()}
    mapped_pg = [macro_to_pg[k] for k in range(n)]
    offsets = [(mapped_pg[k] - k) % n for k in range(n)]
    sums = [(mapped_pg[k] + k) % n for k in range(n)]
    is_reflected = len(set(sums)) == 1 and len(set(offsets)) > 1

    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "face %s edge-map orientation=%s n=%d mapped_pg=%s offsets=%s sums=%s",
            face_id,
            "reflection" if is_reflected else "rotation",
            n,
            mapped_pg,
            offsets,
            sums,
        )

    if _env_flag("PGRID_ORIENTATION_AUDIT"):
        LOGGER.info(
            "orientation-audit face=%s stage=edge-map orientation=%s n=%d mapped_pg=%s offsets=%s sums=%s",
            face_id,
            "reflection" if is_reflected else "rotation",
            n,
            mapped_pg,
            offsets,
            sums,
        )

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

    if set(pg_edge_to_macro.keys()) != set(range(n)):
        raise ValueError(f"Edge mapping keys invalid for face {face_id}")
    mapped_macro = [pg_edge_to_macro[k] for k in range(n)]
    if len(set(mapped_macro)) != n or any(m < 0 or m >= n for m in mapped_macro):
        raise ValueError(f"Edge mapping values invalid for face {face_id}: {mapped_macro}")

    return pg_edge_to_macro


def match_grid_corners_to_uv(
    grid_corners: List[Tuple[float, float]],
    globe_grid: PolyGrid,
    face_id: str,
    *,
    allow_reflection_override: Optional[bool] = None,
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
    if n < 3:
        raise ValueError("grid_corners must contain at least 3 corners")
    freq = globe_grid.metadata.get("frequency", 3)
    rad = globe_grid.metadata.get("radius", 1.0)
    tiles = get_goldberg_tiles(freq, rad)
    tile = _match_tile_to_face(tiles, face_id)
    if len(tile.vertices) != n:
        raise ValueError(
            f"Corner count mismatch for face {face_id}: grid={n}, tile={len(tile.vertices)}"
        )

    if not np.isfinite(np.asarray(grid_corners, dtype=np.float64)).all():
        raise ValueError("grid_corners contains non-finite values")

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

    # Pentagons use deterministic rotation-only matching.
    allow_reflection = (n != 5) if allow_reflection_override is None else bool(allow_reflection_override)
    indices, mode, best_rot_err, best_ref_err, best_rot, best_ref = _select_corner_match_indices(
        macro_angles,
        gt_angles,
        allow_reflection=allow_reflection,
    )

    if n == 5 and allow_reflection and mode == "reflection":
        if not _should_use_pent_reflection(best_rot_err, best_ref_err):
            indices = [(k - best_rot) % n for k in range(n)]
            mode = "rotation"

    if len(set(indices)) != n:
        raise ValueError(f"Corner reorder for face {face_id} is non-bijective: {indices}")
    if mode == "rotation" and not _is_rotation_map(indices):
        raise ValueError(f"Corner reorder for face {face_id} is not a pure rotation: {indices}")
    if mode == "reflection" and not _is_reflection_map(indices):
        raise ValueError(f"Corner reorder for face {face_id} is not a pure reflection: {indices}")
    if not (_is_rotation_map(indices) or _is_reflection_map(indices)):
        raise ValueError(f"Corner reorder for face {face_id} is invalid: {indices}")

    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "face %s corner-match mode=%s n=%d rot_err=%.6f ref_err=%.6f rot=%d ref=%d indices=%s",
            face_id,
            mode,
            n,
            best_rot_err,
            best_ref_err,
            best_rot,
            best_ref,
            indices,
        )

    if _env_flag("PGRID_ORIENTATION_AUDIT"):
        LOGGER.info(
            "orientation-audit face=%s stage=corner-match mode=%s allow_reflection=%d n=%d rot_err=%.6f ref_err=%.6f rot=%d ref=%d indices=%s",
            face_id,
            mode,
            int(allow_reflection),
            n,
            best_rot_err,
            best_ref_err,
            best_rot,
            best_ref,
            indices,
        )

    return [grid_corners[i] for i in indices]


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
# Minimum pentagon grid scale floor.
# Keep this mathematically neutral so low-frequency planets are not
# forced into an over-scaled pentagon warp.
_PENTAGON_GRID_SCALE_MIN = 1.0


def _avg_sector_scale(
    grid_corners: List[Tuple[float, float]],
    uv_corners: List[Tuple[float, float]],
    tile_size: int,
    gutter: int,
) -> float:
    """Average triangle-fan sector scale from *grid_corners* → atlas pixels.

    Each sector is the triangle (centroid, corner_i, corner_{i+1}).
    The scale is ``sqrt(det_dst / det_src)`` — i.e. the square-root of
    the area ratio — averaged over all sectors.  This gives a
    single representative "pixels per grid-unit" number that can be
    compared across tiles to detect density mismatch.
    """
    gc = np.asarray(grid_corners, dtype=float)
    uv = np.asarray(uv_corners, dtype=float)
    n = len(gc)

    gc_c = gc.mean(axis=0)

    # Map UV [0,1]² → atlas-slot pixel coordinates
    dst = np.empty_like(uv)
    for i in range(n):
        u, v = uv[i]
        dst[i, 0] = gutter + u * tile_size
        dst[i, 1] = gutter + (1.0 - v) * tile_size
    dst_c = dst.mean(axis=0)

    total = 0.0
    for i in range(n):
        j = (i + 1) % n
        S = np.column_stack([gc[i] - gc_c, gc[j] - gc_c])
        D = np.column_stack([dst[i] - dst_c, dst[j] - dst_c])
        det_s = abs(np.linalg.det(S))
        det_d = abs(np.linalg.det(D))
        if det_s > 1e-20:
            total += np.sqrt(det_d / det_s)
    return total / n


def _compute_pentagon_grid_scale(
    pent_grid_corners: List[Tuple[float, float]],
    pent_uv_corners: List[Tuple[float, float]],
    hex_neighbours_data: List[
        Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]
    ],
    tile_size: int,
    gutter: int,
) -> float:
    """Compute the dynamic grid-corner scale for a pentagon tile.

    The pentagon's UV polygon has more UV area per grid-unit area than
    its hex neighbours, so the warp maps it at a higher pixel density.
    This function computes the ratio
    ``pentagon_sector_scale / mean(hex_neighbour_sector_scales)``
    and returns it as the multiplicative scale to apply to the
    pentagon's grid corners (expanding them outward from the centroid
    so the warp brings the pixel density back in line with the hexes).

    Falls back to ``_PENTAGON_GRID_SCALE_MIN`` when there are no hex
    neighbours or when the computed ratio is below that floor.
    """
    pent_scale = _avg_sector_scale(
        pent_grid_corners, pent_uv_corners, tile_size, gutter,
    )

    if not hex_neighbours_data:
        return _PENTAGON_GRID_SCALE_MIN

    hex_scales = [
        _avg_sector_scale(gc, uc, tile_size, gutter)
        for gc, uc in hex_neighbours_data
    ]
    avg_hex = sum(hex_scales) / len(hex_scales)

    if avg_hex < 1e-12:
        return _PENTAGON_GRID_SCALE_MIN

    ratio = pent_scale / avg_hex
    return max(ratio, _PENTAGON_GRID_SCALE_MIN)


def _stabilize_pentagon_scale_for_frequency(scale: float, frequency: int) -> float:
    """Clamp pentagon expansion for low-frequency topology.

    At frequency 2, coarse geometry can over-expand pent edge sampling,
    which appears as midpoint bulge/clipping while corners still align.
    """
    s = max(float(scale), 1.0)
    if int(frequency) <= 2:
        return 1.0
    return s


def _pentagon_smoothing_alpha_for_frequency(frequency: int) -> float:
    """Return blend factor for pentagon corner smoothing.

    Lower-frequency grids can show midpoint seam bowing when full smoothing
    is applied. Use a partial blend there to keep corner alignment while
    reducing edge bulge.
    """
    # Keep freq<=2 seam anchors exact: any corner smoothing here shifts
    # pent boundary sampling away from neighboring hex boundaries.
    if int(frequency) <= 2:
        return 0.0
    return 1.0


def _blend_corner_sets(
    base_corners: List[Tuple[float, float]],
    adjusted_corners: List[Tuple[float, float]],
    alpha: float,
) -> List[Tuple[float, float]]:
    """Blend two ordered corner sets with weight ``alpha`` on adjusted."""
    if len(base_corners) != len(adjusted_corners):
        return adjusted_corners
    a = max(0.0, min(1.0, float(alpha)))
    out: List[Tuple[float, float]] = []
    for (bx, by), (ax, ay) in zip(base_corners, adjusted_corners):
        out.append((bx + (ax - bx) * a, by + (ay - by) * a))
    return out


def _is_hex_adjacent_to_pentagon(globe_grid: PolyGrid, face_id: str) -> bool:
    """Return True when a hex tile shares at least one edge with a pentagon."""
    face = globe_grid.faces.get(face_id)
    if face is None or face.face_type != "hex":
        return False
    for nid in face.neighbor_ids:
        nface = globe_grid.faces.get(nid)
        if nface is not None and nface.face_type == "pent":
            return True
    return False


def _scale_corners_from_centroid(
    corners: List[Tuple[float, float]],
    scale: float,
    center: Optional[Tuple[float, float]] = None,
) -> List[Tuple[float, float]]:
    """Scale *corners* outward from *center* by *scale*.

    If *center* is ``None`` the polygon's own centroid is used.
    """
    if center is not None:
        cx, cy = center
    else:
        cx = sum(x for x, _ in corners) / len(corners)
        cy = sum(y for _, y in corners) / len(corners)
    return [
        (cx + (x - cx) * scale, cy + (y - cy) * scale)
        for x, y in corners
    ]


def _shift_corners_to_edge_midpoints(
    corners: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Replace each corner with the midpoint of the edge it starts.

    For a pentagon with corners ``[V0, V1, V2, V3, V4]`` the result
    is ``[mid(V0,V1), mid(V1,V2), mid(V2,V3), mid(V3,V4), mid(V4,V0)]``.

    This rotates the polygon by half an interior angle (36° for a
    regular pentagon) anticlockwise and inscribes it within the
    original, reducing the circumradius by ``cos(π/n)``.
    """
    n = len(corners)
    return [
        (
            (corners[i][0] + corners[(i + 1) % n][0]) / 2.0,
            (corners[i][1] + corners[(i + 1) % n][1]) / 2.0,
        )
        for i in range(n)
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
    center: Optional[Tuple[float, float]] = None,
) -> List[Tuple[float, float]]:
    """Rotate *corners* around *center* by *angle_rad*.

    If *center* is ``None`` the polygon's own centroid is used.
    """
    import math as _m
    if center is not None:
        cx, cy = center
    else:
        cx = sum(x for x, _ in corners) / len(corners)
        cy = sum(y for _, y in corners) / len(corners)
    cos_a = _m.cos(angle_rad)
    sin_a = _m.sin(angle_rad)
    return [
        (cx + (x - cx) * cos_a - (y - cy) * sin_a,
         cy + (x - cx) * sin_a + (y - cy) * cos_a)
        for x, y in corners
    ]


def _apply_pentagon_uv_adjustments(
    uv_corners: List[Tuple[float, float]],
    *,
    tile_size: int,
    pent_uv_scale: float,
    pent_uv_rotation: float,
    pent_uv_x: float,
    pent_uv_y: float,
) -> List[Tuple[float, float]]:
    """Apply optional pentagon UV adjustments.

    Default settings are a strict no-op so pentagons use the
    authoritative models UV polygon without post-adjustment.
    """
    if (
        abs(float(pent_uv_rotation)) < 1e-12
        and abs(float(pent_uv_scale) - 1.0) < 1e-12
        and abs(float(pent_uv_x)) < 1e-12
        and abs(float(pent_uv_y)) < 1e-12
    ):
        return list(uv_corners)

    adjusted = list(uv_corners)
    uv_center = (0.5, 0.5)

    if abs(float(pent_uv_rotation)) >= 1e-12:
        adjusted = _rotate_corners(
            adjusted,
            math.radians(float(pent_uv_rotation)),
            center=uv_center,
        )

    # Keep optional transformed polygon centered in UV space.
    us = [u for u, _ in adjusted]
    vs = [v for _, v in adjusted]
    bbox_cx = (min(us) + max(us)) / 2
    bbox_cy = (min(vs) + max(vs)) / 2
    dx = 0.5 - bbox_cx
    dy = 0.5 - bbox_cy
    adjusted = [(u + dx, v + dy) for u, v in adjusted]

    if abs(float(pent_uv_scale) - 1.0) >= 1e-12:
        adjusted = _scale_corners_from_centroid(
            adjusted,
            float(pent_uv_scale),
            center=uv_center,
        )

    if abs(float(pent_uv_x)) >= 1e-12 or abs(float(pent_uv_y)) >= 1e-12:
        nudge_u = float(pent_uv_x) / tile_size
        nudge_v = float(pent_uv_y) / tile_size
        adjusted = [(u + nudge_u, v + nudge_v) for u, v in adjusted]

    return adjusted


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
    *,
    match_radii: bool = True,
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

    # ── New grid corner radii ──
    # ``match_radii=False`` keeps a uniform source radius, which helps
    # avoid pent-edge midpoint bulge while preserving angular alignment.
    R_mean_src = np.mean(np.linalg.norm(gc - gc_c, axis=1))
    if match_radii:
        R_mean_dst = dst_R.mean()
        new_R = R_mean_src * (dst_R / R_mean_dst)
    else:
        new_R = np.full(n, R_mean_src, dtype=np.float64)

    # ── New grid corner angles: align directly to destination angles ──
    # Destination pixel space is Y-down while grid space is Y-up, so
    # convert by negating the angles. Use unwrap to enforce a smooth,
    # monotonic traversal and avoid cumulative sector-step drift.
    dst_sorted_angles = np.arctan2(
        dst_sorted[:, 1] - dst_px_c[1],
        dst_sorted[:, 0] - dst_px_c[0],
    )
    target_angles = np.unwrap(-dst_sorted_angles)

    new_sorted = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        angle = float(target_angles[i])
        new_sorted[i, 0] = gc_c[0] + new_R[i] * math.cos(angle)
        new_sorted[i, 1] = gc_c[1] + new_R[i] * math.sin(angle)

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
    *,
    uniform_half_span: Optional[float] = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute the axis limits used by the stitched-tile renderer.

    Centre tile extent + 25 % padding, with aspect-ratio correction
    to make the view square.

    Parameters
    ----------
    composite : CompositeGrid
    face_id : str
    uniform_half_span : float, optional
        If given, **every** tile uses this half-span instead of
        computing its own from its vertex extent.  Pass the maximum
        half-span across all tiles (see
        :func:`compute_uniform_half_span`) to guarantee that every
        composite is rasterised at the **same** pixels-per-grid-unit.
        This eliminates the boundary artefact caused by adjacent tiles
        rendering shared hex cells at different effective resolutions.

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

    if uniform_half_span is not None:
        half_span = uniform_half_span

    cx_mid = (min(center_xs) + max(center_xs)) * 0.5
    cy_mid = (min(center_ys) + max(center_ys)) * 0.5

    xlim = (cx_mid - half_span, cx_mid + half_span)
    ylim = (cy_mid - half_span, cy_mid + half_span)
    return xlim, ylim


def compute_uniform_half_span(
    composites: Dict[str, Any],
    face_ids: List[str],
) -> float:
    """Return the maximum half-span across all tiles.

    Using this value as *uniform_half_span* in
    :func:`compute_tile_view_limits` ensures every composite is
    rasterised at the same pixels-per-grid-unit, eliminating the
    boundary artefact caused by scale mismatch between adjacent tiles.
    """
    max_span = 0.0
    for fid in face_ids:
        xlim, ylim = compute_tile_view_limits(composites[fid], fid)
        hs = (xlim[1] - xlim[0]) * 0.5
        if hs > max_span:
            max_span = hs
    return max_span


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
        from ..core.geometry import face_center as _face_center

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


def _augment_ordered_fan_with_edge_controls(
    src_px_ordered: np.ndarray,
    dst_px_ordered: np.ndarray,
    src_px_centroid: np.ndarray,
    dst_px_centroid: np.ndarray,
    edge_interior_pull: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Insert per-edge interior control points into ordered fan corners.

    Original corners remain exact anchors at even indices. Odd indices are
    edge-midpoint controls, with source controls optionally pulled inward
    toward the source centroid.
    """
    pull = max(0.0, min(0.95, float(edge_interior_pull)))
    if pull <= 1e-12:
        return src_px_ordered, dst_px_ordered

    n = int(src_px_ordered.shape[0])
    src_aug = np.empty((2 * n, 2), dtype=np.float64)
    dst_aug = np.empty((2 * n, 2), dtype=np.float64)

    for i in range(n):
        j = (i + 1) % n
        src_i = src_px_ordered[i]
        src_j = src_px_ordered[j]
        dst_i = dst_px_ordered[i]
        dst_j = dst_px_ordered[j]

        src_aug[2 * i] = src_i
        dst_aug[2 * i] = dst_i

        src_mid = 0.5 * (src_i + src_j)
        dst_mid = 0.5 * (dst_i + dst_j)

        src_aug[2 * i + 1] = src_mid + (src_px_centroid - src_mid) * pull
        dst_aug[2 * i + 1] = dst_mid

    return src_aug, dst_aug


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
    edge_interior_pull: float = 0.0,
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

    # Pentagon-only safety: if corner winding differs after ordering,
    # flip source winding while preserving an anchor corner.
    src_px_ordered, _ = _normalize_ordered_pentagon_winding(
        src_px_ordered,
        dst_px_ordered,
    )

    if n == 5:
        # Pentagon fail-fast checks: mirrored or degenerate sectors
        # are a strong signal of corner pairing regression.
        for i in range(n):
            j = (i + 1) % n
            sv0 = src_px_ordered[i] - src_px_centroid
            sv1 = src_px_ordered[j] - src_px_centroid
            dv0 = dst_px_ordered[i] - dst_px_centroid
            dv1 = dst_px_ordered[j] - dst_px_centroid
            sdet = float(sv0[0] * sv1[1] - sv0[1] * sv1[0])
            ddet = float(dv0[0] * dv1[1] - dv0[1] * dv1[0])
            if abs(sdet) < 1e-12 or abs(ddet) < 1e-12:
                raise ValueError(
                    f"Degenerate pentagon warp sector at index {i}: sdet={sdet:.3e} ddet={ddet:.3e}"
                )
            if math.copysign(1.0, sdet) != math.copysign(1.0, ddet):
                raise ValueError(
                    f"Inverted pentagon warp sector at index {i}: sdet={sdet:.3e} ddet={ddet:.3e}"
                )

    # Optional edge-interior controls for seam-focused refinement.
    src_px_fan, dst_px_fan = _augment_ordered_fan_with_edge_controls(
        src_px_ordered,
        dst_px_ordered,
        src_px_centroid,
        dst_px_centroid,
        edge_interior_pull=edge_interior_pull,
    )

    # Build per-sector affines entirely in pixel space
    # Inverse: slot_pixel → source_pixel
    inv_sectors = _build_sector_affines(
        dst_px_fan, dst_px_centroid, src_px_fan, src_px_centroid,
    )

    # Build output pixel grid
    oy, ox = np.mgrid[0:output_size, 0:output_size]
    out_pts = np.stack([ox.ravel().astype(np.float64),
                        oy.ravel().astype(np.float64)], axis=1)  # (M, 2)

    # Assign each output pixel to a sector in dst_px space
    sector_ids = _assign_sectors(out_pts, dst_px_centroid, dst_px_fan)

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


def _fill_invalid_sample_pixels(
    arr: np.ndarray,
    invalid_mask: np.ndarray,
) -> np.ndarray:
    """Fill invalidly sampled pixels using nearest valid RGB colour.

    ``map_coordinates(..., mode='constant')`` can blend constant-fill
    values into boundary samples, so invalid pixels are not always exactly
    equal to the fill colour. This helper uses a validity mask to replace
    all invalid samples with the nearest valid pixel colour.
    """
    from scipy.ndimage import distance_transform_edt

    if arr.ndim != 3 or arr.shape[2] != 3:
        return arr
    mask = np.asarray(invalid_mask, dtype=bool)
    if mask.shape != arr.shape[:2] or not mask.any():
        return arr

    valid_mask = ~mask
    if not valid_mask.any():
        return arr

    _, nearest = distance_transform_edt(mask, return_indices=True)
    out = arr.copy()
    out[mask] = arr[nearest[0][mask], nearest[1][mask]]
    return out


def warp_tile_to_uv(
    img: "Image.Image",
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    output_size: int,
    *,
    grid_corners: Optional[List[Tuple[float, float]]] = None,
    uv_corners: Optional[List[Tuple[float, float]]] = None,
    tile_size: Optional[int] = None,
    gutter: int = 0,
    src_centroid_override: Optional[np.ndarray] = None,
    sample_order: int = 1,
    dilate_cval: bool = True,
    edge_interior_pull: float = 0.0,
) -> "Image.Image":
    """Warp a stitched tile image so its polygon maps to the UV layout.

    Uses a **piecewise-linear** (triangle-fan) warp with required
    ``grid_corners`` and ``uv_corners`` inputs, giving exact boundary
    alignment that matches the ``UVTransform`` approach.

    Parameters
    ----------
    img : PIL.Image.Image
        The rendered stitched tile (any size).
    xlim, ylim : (min, max)
        Axis limits used when rendering the image.
    output_size : int
        Width and height of the output image (slot_size = tile_size + 2*gutter).
    grid_corners : list of (x, y), optional
        Polygon corners in grid (Tutte) space.
    uv_corners : list of (u, v), optional
        UV polygon corners in [0, 1].
    tile_size : int, optional
        Inner tile size (pixels). Required.
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
    if grid_corners is None or uv_corners is None or tile_size is None:
        raise ValueError(
            "warp_tile_to_uv requires grid_corners, uv_corners, and tile_size "
            "for mandatory piecewise mapping"
        )

    # ── Piecewise-linear warp (exact boundary alignment) ────
    map_x, map_y = _compute_piecewise_warp_map(
        grid_corners, uv_corners,
        tile_size=tile_size,
        gutter=gutter,
        img_w=img_w, img_h=img_h,
        xlim=xlim, ylim=ylim,
        output_size=output_size,
        src_centroid_override=src_centroid_override,
        edge_interior_pull=edge_interior_pull,
    )

    src_arr = np.array(img.convert("RGB"), dtype=np.float64)
    # map_coordinates expects (row, col) = (y, x)
    out_channels = []
    order = 0 if int(sample_order) <= 0 else 1
    for ch in range(3):
        warped_ch = map_coordinates(
            src_arr[:, :, ch],
            [map_y, map_x],
            order=order,
            mode="constant",
            cval=128.0,
        )
        out_channels.append(warped_ch.astype(np.uint8))

    # Treat only strictly out-of-bounds samples as invalid.
    # A softer validity threshold can over-classify near-edge pixels and
    # smear real detail along seams.
    invalid_mask = (
        (map_x < 0.0)
        | (map_x > float(img_w - 1))
        | (map_y < 0.0)
        | (map_y > float(img_h - 1))
    )

    out_arr = np.stack(out_channels, axis=-1)
    out_arr = _fill_invalid_sample_pixels(out_arr, invalid_mask)

    # Dilate any remaining cval-fill pixels (bounding-box corners
    # outside the polygon) so bilinear/mipmap sampling never
    # encounters the grey fallback colour.
    if dilate_cval:
        out_arr = _dilate_cval_pixels(out_arr)

    return Image.fromarray(out_arr, "RGB")


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
    pent_gutter: Optional[int] = None,
    hex_gutter: Optional[int] = None,
    mask_outside: bool = False,
    mask_colour: Tuple[int, int, int] = (0, 0, 0),
    debug_labels: bool = False,
    debug_uv_cut: bool = False,
    output_dir: Optional[Path] = None,
    pentagon_rotation_steps: int = 0,
    equalise_sectors: bool = False,
    warp_sample_order: int = 1,
    warp_dilate_cval: bool = True,
    pentagon_scale_override: Optional[float] = None,
    pent_uv_scale: float = 1.0,
    pent_uv_rotation: float = 0.0,
    pent_uv_x: float = 0.0,
    pent_uv_y: float = 0.0,
    pent_twist: float = 0.0,
    pent_edge_interior_pull: float = 0.0,
    hex_pent_edge_interior_pull: float = 0.0,
    pentagon_allow_reflection: bool = False,
    uniform_half_span: Optional[float] = None,
) -> Tuple["Image.Image", Dict[str, Tuple[float, float, float, float]]]:
    """Build a texture atlas from stitched tile images, UV-aligned.

    For each tile:
    1. Finds the polygon corners in the original detail grid.
    2. Gets the GoldbergTile UV polygon from the models library.
     3. Computes corner-matched piecewise warp maps from grid space
         into atlas slot pixels.
     4. Warps the stitched image so the polygon lands in UV-correct
       orientation within the slot.
     5. Enforces seam continuity through corner/order correctness and
         piecewise mapping only (no seam blend post-pass).

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
        Default gutter width used when ``pent_gutter`` or
        ``hex_gutter`` are not specified.
    pent_gutter : int or None
        Gutter pixels for pentagon tile slots.  Defaults to *gutter*.
    hex_gutter : int or None
        Gutter pixels for hexagon tile slots.  Defaults to *gutter*.
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
    equalise_sectors : bool
        If True, apply ``_equalise_sector_ratios`` to irregular hex
        tiles (those adjacent to pentagons) so the piecewise-warp
        sectors become conformal.  Default False (conservative).
    pentagon_scale_override : float or None
        When not None, replace the auto-computed pentagon grid-corner
        scale with this fixed value.  The scale controls how far the
        pentagon's source grid corners are expanded from their
        centroid before warping: values > 1.0 sample a wider region
        of the composite (coarser pixel density, larger terrain
        features); values < 1.0 tighten the crop (finer detail).
        Auto-computed values are typically ~1.08.  Pass ``None``
        (default) to use the automatic density-matching calculation.
    uniform_half_span : float or None
        When not None, every tile uses this half-span for its
        rasterisation view extent instead of computing a per-tile
        span from its vertex positions.  Pass the result of
        :func:`compute_uniform_half_span` to eliminate the boundary
        artefact caused by adjacent tiles rendering shared hex cells
        at different effective resolutions (up to ~11 % scale
        mismatch for freq-3 Goldberg grids).

    Returns
    -------
    (atlas, uv_layout)
        atlas : PIL.Image.Image
        uv_layout : ``{face_id: (u_min, v_min, u_max, v_max)}``
    """
    from PIL import Image
    from .uv_texture import get_tile_uv_vertices

    # ── Resolve per-type gutters ─────────────────────────────────
    _pent_g = pent_gutter if pent_gutter is not None else gutter
    _hex_g = hex_gutter if hex_gutter is not None else gutter
    max_gutter = max(_pent_g, _hex_g)

    n = len(face_ids)
    columns, rows, atlas_w, atlas_h = compute_atlas_layout(
        n, tile_size, max_gutter,
    )
    slot_size = tile_size + 2 * max_gutter
    atlas = Image.new("RGB", (atlas_w, atlas_h), (128, 128, 128))

    uv_layout: Dict[str, Tuple[float, float, float, float]] = {}
    pent_tiles_total = 0
    pent_winding_fixes = 0

    # ── Pre-pass: collect matched grid/UV corners for every tile ─
    # Pentagon scale depends on hex-neighbour geometry, so we gather
    # corners first and compute the dynamic scale before warping.
    _tile_corners: Dict[str, Tuple[
        List[Tuple[float, float]],   # grid_corners_matched
        List[Tuple[float, float]],   # uv_corners
        int,                         # n_sides
    ]] = {}

    for fid in face_ids:
        if fid not in tile_images:
            continue
        dg = detail_grids[fid]
        ns = len(globe_grid.faces[fid].vertex_ids)
        is_pent = ns == 5

        corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
        dg.compute_macro_edges(n_sides=ns, corner_ids=corner_ids)
        gc_raw = get_macro_edge_corners(dg, ns)
        uc_raw = get_tile_uv_vertices(globe_grid, fid)

        # Pentagon path: prefer topology-driven corner correspondence using
        # shared edge identity (PG edge -> macro edge) rather than pure angle
        # minimization. This is more stable for pent-hex seam endpoints.
        gc_matched: List[Tuple[float, float]]
        if is_pent:
            try:
                pg_to_macro_corner = compute_pg_to_macro_corner_map(globe_grid, fid, dg)
                pg_gt_offset = compute_uv_to_polygrid_offset(globe_grid, fid)
                gc_matched = [
                    gc_raw[pg_to_macro_corner[(k + pg_gt_offset) % ns]]
                    for k in range(ns)
                ]
            except Exception:
                gc_matched = match_grid_corners_to_uv(
                    gc_raw,
                    globe_grid,
                    fid,
                    allow_reflection_override=(True if pentagon_allow_reflection else None),
                )
        else:
            gc_matched = match_grid_corners_to_uv(
                gc_raw,
                globe_grid,
                fid,
                allow_reflection_override=None,
            )
        _tile_corners[fid] = (gc_matched, uc_raw, ns)

    # ── Compute dynamic pentagon scale factors ───────────────────
    _pent_scales: Dict[str, float] = {}
    pent_scale_values: List[float] = []
    freq_value = int(globe_grid.metadata.get("frequency", 0) or 0)
    for fid, (gc_m, uc, ns) in _tile_corners.items():
        if ns != 5:
            continue
        # Use the same pentagon source geometry basis as the warp path.
        dg = detail_grids[fid]
        smoothed_for_scale = _smooth_pentagon_corners(gc_m, dg, ns)
        smooth_alpha = _pentagon_smoothing_alpha_for_frequency(freq_value)
        gc_for_scale = _blend_corner_sets(gc_m, smoothed_for_scale, smooth_alpha)
        # Gather hex-neighbour corner data
        hex_data: List[
            Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]
        ] = []
        for nid in globe_grid.faces[fid].neighbor_ids:
            if nid in _tile_corners:
                ngc, nuc, nns = _tile_corners[nid]
                if nns == 6:
                    hex_data.append((ngc, nuc))
        _pent_scales[fid] = _compute_pentagon_grid_scale(
            gc_for_scale, uc, hex_data, tile_size, max_gutter,
        )
        _pent_scales[fid] = _stabilize_pentagon_scale_for_frequency(
            _pent_scales[fid],
            frequency=freq_value,
        )
        if pentagon_scale_override is not None:
            _pent_scales[fid] = pentagon_scale_override
        pent_scale_values.append(float(_pent_scales[fid]))

    if _env_flag("PGRID_ORIENTATION_AUDIT") and pent_scale_values:
        LOGGER.info(
            "orientation-audit stage=pent-scale-summary freq=%s pent_count=%d scale_min=%.6f scale_max=%.6f scale_mean=%.6f",
            str(globe_grid.metadata.get("frequency", "?")),
            len(pent_scale_values),
            min(pent_scale_values),
            max(pent_scale_values),
            sum(pent_scale_values) / len(pent_scale_values),
        )

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

        # Retrieve pre-computed corners.
        grid_corners_matched, uv_corners, n_sides = _tile_corners[fid]
        is_pentagon = n_sides == 5

        # Per-tile gutter
        g = _pent_g if is_pentagon else _hex_g

        # (macro edges already computed in pre-pass)
        dg = detail_grids[fid]

        # ── Corner adjustment strategy ───────────────────────────
        # Pentagon tiles: shift both grid and UV corners to their
        # edge midpoints (rotated ½-edge anticlockwise) so the warp
        # sectors align with the flat edges rather than the vertices.
        # This ensures the full apron content along each edge is
        # captured inside the warp polygon.  After the midpoint
        # shift, smooth → scale → equalise as before.
        #
        # Irregular hex tiles (adjacent to a pentagon): optionally
        # apply _equalise_sector_ratios so the piecewise-warp sectors
        # become conformal — the seam post-pass compensates for any
        # boundary shift this introduces.
        # Regular hex tiles: use matched corners directly.
        grid_corners = grid_corners_matched
        src_centroid = None
        if is_pentagon:
            uv_corners = _apply_pentagon_uv_adjustments(
                uv_corners,
                tile_size=tile_size,
                pent_uv_scale=pent_uv_scale,
                pent_uv_rotation=pent_uv_rotation,
                pent_uv_x=pent_uv_x,
                pent_uv_y=pent_uv_y,
            )

            # 1) Smooth corners to compensate Tutte zigzag bias
            smoothed_raw = _smooth_pentagon_corners(
                grid_corners_matched, dg, n_sides,
            )
            smooth_alpha = _pentagon_smoothing_alpha_for_frequency(freq_value)
            smoothed = _blend_corner_sets(
                grid_corners_matched,
                smoothed_raw,
                smooth_alpha,
            )
            # 2) Apply dynamic outward scale to match hex pixel density
            scaled = _scale_corners_from_centroid(
                smoothed, _pent_scales[fid],
            )
            # 3) Keep production pent path on smoothed+scaled source corners.
            #    Full sector equalisation can overfit irregular UV radii and
            #    visibly bend shared-edge progression near corners.
            grid_corners = scaled
            src_centroid = None
        elif equalise_sectors:
            # Only worthwhile for irregular hexes (those adjacent to
            # a pentagon).  Regular hexes have uniform sector geometry
            # and _equalise_sector_ratios is a no-op for them.
            grid_corners, src_centroid = _equalise_sector_ratios(
                grid_corners_matched, uv_corners,
                tile_size=tile_size, gutter=g,
            )

        # ── Pentagon twist: rotate grid corners (source sampling)
        #    clockwise by pent_twist degrees.  Rotating the source
        #    polygon clockwise makes the warped content appear to
        #    rotate clockwise in the output tile.
        if is_pentagon and pent_twist != 0.0:
            import math as _m
            # Negative angle = clockwise in standard coords
            grid_corners = _rotate_corners(
                grid_corners, -_m.radians(pent_twist),
            )

        # Compute view limits (same as the renderer used)
        comp = composites[fid]
        xlim, ylim = compute_tile_view_limits(
            comp, fid, uniform_half_span=uniform_half_span,
        )

        # Pentagon winding is normalized in ordered warp space inside
        # _compute_piecewise_warp_map. Applying an additional pre-warp
        # corner reorder here can disturb corner-to-corner pairing at
        # pent-hex seam endpoints.
        if is_pentagon:
            pent_tiles_total += 1

        # Warp the image (piecewise-linear for exact boundary alignment)
        tile_slot_size = tile_size + 2 * g
        edge_pull = 0.0
        if is_pentagon:
            edge_pull = pent_edge_interior_pull
        elif _is_hex_adjacent_to_pentagon(globe_grid, fid):
            edge_pull = hex_pent_edge_interior_pull

        warped = warp_tile_to_uv(
            tile_img, xlim, ylim, tile_slot_size,
            grid_corners=grid_corners,
            uv_corners=uv_corners,
            tile_size=tile_size,
            gutter=g,
            src_centroid_override=src_centroid,
            sample_order=warp_sample_order,
            dilate_cval=warp_dilate_cval,
            edge_interior_pull=edge_pull,
        )

        cval_after_warp = -1
        if _env_flag("PGRID_ATLAS_CVAL_AUDIT"):
            cval_after_warp = _count_rgb_fill_pixels(warped, fill=128)

        # Fill any remaining fallback-colour pixels left by the
        # piecewise warp at bounding-box corners outside the polygon.
        warped = _fill_warped_gaps(warped, cval=128)

        if _env_flag("PGRID_ATLAS_CVAL_AUDIT"):
            cval_after_gap_fill = _count_rgb_fill_pixels(warped, fill=128)
            LOGGER.info(
                "atlas-cval-audit face=%s n_sides=%d sample_order=%d dilate=%s cval_after_warp=%d cval_after_gap_fill=%d",
                fid,
                n_sides,
                int(warp_sample_order),
                bool(warp_dilate_cval),
                cval_after_warp,
                cval_after_gap_fill,
            )

        # Mask outside the UV polygon
        if mask_outside:
            warped = mask_warped_to_uv_polygon(
                warped, uv_corners,
                tile_size=tile_size, gutter=g,
                fill_colour=mask_colour,
            )

        # Draw debug labels (tile ID, sub-face IDs, edge arrows, etc.)
        if debug_labels:
            from ..detail.detail_terrain import compute_neighbor_edge_mapping
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
                tile_size=tile_size, gutter=g,
                detail_grid=dg,
                grid_corners=grid_corners,
                face_type=face_type,
                detail_rings=d_rings,
                xlim=xlim,
                ylim=ylim,
            )

        # ── Debug UV-cut overlay ─────────────────────────────────
        # Draw the UV inner rectangle (what the renderer samples) and
        # the UV polygon outline onto the warped image so we can
        # visually verify alignment.
        if debug_uv_cut:
            from PIL import ImageDraw
            dbg = warped.copy()
            draw = ImageDraw.Draw(dbg)
            tile_slot_size = tile_size + 2 * g

            # UV inner rectangle — the tile_size × tile_size region
            # that the renderer maps [0,1] UV into.  In the warped
            # image this sits at offset g from each edge.
            cut_x0, cut_y0 = g, g
            cut_x1, cut_y1 = g + tile_size, g + tile_size
            draw.rectangle(
                [cut_x0, cut_y0, cut_x1, cut_y1],
                outline=(255, 0, 0), width=2,
            )

            # UV polygon outline in warped-image coords
            px_corners = []
            for u, v in uv_corners:
                px_x = g + u * tile_size
                px_y = g + (1.0 - v) * tile_size
                px_corners.append((px_x, px_y))
            for i in range(len(px_corners)):
                j = (i + 1) % len(px_corners)
                draw.line(
                    [px_corners[i], px_corners[j]],
                    fill=(0, 255, 0), width=2,
                )

            # Centroid marker
            cx = sum(p[0] for p in px_corners) / len(px_corners)
            cy = sum(p[1] for p in px_corners) / len(px_corners)
            r = 4
            draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                         fill=(0, 255, 0))

            # Label
            label = f"{fid} {'pent' if is_pentagon else 'hex'} g={g}"
            draw.text((4, 4), label, fill=(255, 255, 255))

            if output_dir is not None:
                dbg.save(str(output_dir / f"{fid}_uvcut.png"))

        # Paste into atlas (centred within the max_gutter slot)
        pad = max_gutter - g
        atlas.paste(warped, (slot_x + pad, slot_y + pad))

        # Fill gutter from warped content — clamp the inner tile edge
        # pixels outward through the full max_gutter border so that
        # bilinear sampling at tile edges sees smooth colour.
        if max_gutter > 0:
            fill_gutter(atlas, slot_x, slot_y, tile_size, max_gutter)

        # Save debug tile if requested (per-tile size, not inflated)
        if output_dir is not None:
            warped.save(str(output_dir / f"{fid}_warped.png"))

        # UV coordinates (inner tile region in atlas)
        inner_x = slot_x + max_gutter
        inner_y = slot_y + max_gutter
        u_min = inner_x / atlas_w
        u_max = (inner_x + tile_size) / atlas_w
        v_min = 1.0 - (inner_y + tile_size) / atlas_h
        v_max = 1.0 - inner_y / atlas_h
        uv_layout[fid] = (u_min, v_min, u_max, v_max)

    if _env_flag("PGRID_ORIENTATION_AUDIT") and pent_tiles_total > 0:
        LOGGER.info(
            "orientation-audit stage=pent-summary pent_tiles=%d winding_fixes=%d",
            pent_tiles_total,
            pent_winding_fixes,
        )

    return atlas, uv_layout
