"""Full-slot tile texture rendering — Phase 16A.

Replaces the flat-colour background approach with per-pixel terrain
sampling so that *every* pixel in the tile slot has coherent terrain
colour, noise, and hillshade — even pixels outside the hex/pent
polygon footprint.

Strategy — Hybrid Polygon + IDW Fill
--------------------------------------
1. Rasterise sub-face polygons via PIL (same speed as ``detail_perf.py``).
2. Identify background pixels (those not covered by any polygon).
3. For background pixels, IDW-interpolate colour from the K nearest
   pre-computed face colours.  This is fast because face colours are
   already computed — no per-pixel noise calls.

The result: every pixel has coherent terrain colour with noise,
hillshade, and vegetation.  No flat fill.

Functions
---------
- :func:`render_detail_texture_fullslot` — full-slot PIL renderer
- :func:`build_face_lookup` — spatial index builder
- :func:`interpolate_at_pixel` — IDW interpolation helper
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detail_render import BiomeConfig, detail_elevation_to_colour, _detail_hillshade
from .geometry import face_center
from .polygrid import PolyGrid
from .tile_data import TileDataStore


# ═══════════════════════════════════════════════════════════════════
# 16A.1 — Spatial index for sub-face lookup
# ═══════════════════════════════════════════════════════════════════

def build_face_lookup(
    grid: PolyGrid,
    store: TileDataStore,
    elevation_field: str = "elevation",
) -> Tuple[
    np.ndarray,       # centroids (N, 2)
    np.ndarray,       # elevations (N,)
    List[str],         # face_ids (ordered)
]:
    """Build arrays of face centroids and elevations for spatial queries.

    Parameters
    ----------
    grid : PolyGrid
        The detail sub-grid.
    store : TileDataStore
        Must contain *elevation_field* for every face.
    elevation_field : str
        Field name in *store*.

    Returns
    -------
    centroids : np.ndarray, shape (N, 2)
        Face centroid positions.
    elevations : np.ndarray, shape (N,)
        Corresponding elevations.
    face_ids : list of str
        Face IDs in the same order.
    """
    centroids_list: List[Tuple[float, float]] = []
    elevs: List[float] = []
    fids: List[str] = []

    for fid, face in grid.faces.items():
        c = face_center(grid.vertices, face)
        if c is None:
            continue
        # Verify all vertices exist (face is renderable)
        ok = True
        for vid in face.vertex_ids:
            v = grid.vertices.get(vid)
            if v is None or not v.has_position():
                ok = False
                break
        if not ok:
            continue
        centroids_list.append(c)
        elevs.append(store.get(fid, elevation_field))
        fids.append(fid)

    if not centroids_list:
        return np.empty((0, 2)), np.empty(0), []

    return (
        np.array(centroids_list, dtype=np.float64),
        np.array(elevs, dtype=np.float64),
        fids,
    )


def interpolate_at_pixel(
    px: float,
    py: float,
    tree: "scipy.spatial.KDTree",
    centroids: np.ndarray,
    elevations: np.ndarray,
    hillshade_vals: np.ndarray,
    k: int = 4,
    eps: float = 1e-12,
) -> Tuple[float, float, float, float]:
    """IDW-interpolate elevation, hillshade, and noise coords at a pixel.

    Parameters
    ----------
    px, py : float
        Query point in grid coordinate space.
    tree : scipy.spatial.KDTree
    centroids, elevations, hillshade_vals : np.ndarray
    k : int
        Number of nearest neighbours.
    eps : float
        Distance floor to avoid division by zero.

    Returns
    -------
    (elevation, hillshade, noise_x, noise_y)
    """
    k_actual = min(k, len(centroids))
    if k_actual == 0:
        return (0.5, 0.5, px, py)

    dists, idxs = tree.query([px, py], k=k_actual)

    # tree.query returns scalar when k=1
    if k_actual == 1:
        dists = np.array([dists])
        idxs = np.array([idxs])

    # Check for exact hit (pixel on a centroid)
    if dists[0] < eps:
        idx = idxs[0]
        cx, cy = centroids[idx]
        return (elevations[idx], hillshade_vals[idx], cx, cy)

    # Inverse-distance weighting
    weights = 1.0 / np.maximum(dists, eps)
    w_sum = weights.sum()

    elev = float(np.dot(weights, elevations[idxs]) / w_sum)
    hs = float(np.dot(weights, hillshade_vals[idxs]) / w_sum)
    nx = float(np.dot(weights, centroids[idxs, 0]) / w_sum)
    ny = float(np.dot(weights, centroids[idxs, 1]) / w_sum)

    return (elev, hs, nx, ny)


# ═══════════════════════════════════════════════════════════════════
# 16A — Full-slot texture renderer (hybrid polygon + IDW fill)
# ═══════════════════════════════════════════════════════════════════

# Sentinel colour used to mark "background" pixels that were not
# covered by any sub-face polygon.  Chosen to be extremely unlikely
# in real terrain (hot magenta).
_BG_SENTINEL = (255, 0, 255)


def render_detail_texture_fullslot(
    detail_grid: PolyGrid,
    store: TileDataStore,
    output_path: Path | str,
    biome: Optional[BiomeConfig] = None,
    *,
    tile_size: int = 256,
    elevation_field: str = "elevation",
    noise_seed: int = 0,
    k_neighbours: int = 4,
    overscan: float = 0.15,
) -> Path:
    """Render a detail grid to a PNG, filling the entire tile slot.

    Uses a hybrid approach:

    1. **Polygon rasterisation** — sub-face polygons are drawn via
       PIL (same fast path as ``render_detail_texture_fast``).
    2. **IDW background fill** — any pixel that was not covered by
       a polygon gets IDW-interpolated colour from the K nearest
       pre-computed face colours.  Because face colours are already
       computed in step 1, no additional per-pixel noise calls are
       needed.

    This is ~2-3× the cost of the flat-fill renderer (vs. ~20× for
    the naïve per-pixel noise approach), while eliminating all flat-
    fill background pixels.

    Parameters
    ----------
    detail_grid : PolyGrid
    store : TileDataStore
    output_path : Path or str
    biome : BiomeConfig, optional
    tile_size : int
        Output image size in pixels (square).
    elevation_field : str
    noise_seed : int
    k_neighbours : int
        Number of nearest faces used for IDW interpolation of
        background pixels.  Default 4.
    overscan : float
        Fraction of extra padding beyond the grid bounding box.
        Larger values ensure corner pixels have good neighbour
        coverage.  Default 0.15 (15%).

    Returns
    -------
    Path
    """
    from PIL import Image, ImageDraw
    from scipy.spatial import KDTree

    if biome is None:
        biome = BiomeConfig()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Compute face colours and spatial data ───────────────────
    hs_dict = _detail_hillshade(
        detail_grid, store, elevation_field,
        azimuth=biome.azimuth, altitude=biome.altitude,
    )

    # Compute bounding box from vertex positions
    xs, ys = [], []
    for v in detail_grid.vertices.values():
        if v.has_position():
            xs.append(v.x)
            ys.append(v.y)

    if not xs:
        img = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
        img.save(str(output_path))
        return output_path

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add overscan padding
    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0
    span = max(x_range, y_range)
    pad = span * overscan
    x_min -= pad
    x_max += pad
    y_min -= pad
    y_max += pad

    x_range = x_max - x_min
    y_range = y_max - y_min

    scale = tile_size / max(x_range, y_range)
    ox = (tile_size - x_range * scale) / 2.0
    oy = (tile_size - y_range * scale) / 2.0

    def _to_pixel(vx: float, vy: float) -> Tuple[float, float]:
        px = (vx - x_min) * scale + ox
        py = tile_size - ((vy - y_min) * scale + oy)
        return (px, py)

    # ── Step 1: Polygon rasterisation ───────────────────────────
    # Compute per-face colours (same as detail_perf.render_detail_texture_fast)
    face_colours: Dict[str, Tuple[float, float, float]] = {}
    face_pixel_colours: Dict[str, Tuple[int, int, int]] = {}
    centroid_pixels: List[Tuple[float, float]] = []
    colour_array: List[Tuple[int, int, int]] = []
    face_id_order: List[str] = []

    for fid, face in detail_grid.faces.items():
        has_verts = True
        for vid in face.vertex_ids:
            v = detail_grid.vertices.get(vid)
            if v is None or not v.has_position():
                has_verts = False
                break
        if has_verts and len(face.vertex_ids) >= 3:
            elev = store.get(fid, elevation_field)
            c = face_center(detail_grid.vertices, face)
            cx, cy = c if c else (0.0, 0.0)
            r, g, b = detail_elevation_to_colour(
                elev, biome,
                hillshade_val=hs_dict.get(fid, 0.5),
                noise_x=cx, noise_y=cy,
                noise_seed=noise_seed,
            )
            face_colours[fid] = (r, g, b)
            pc = (
                max(0, min(255, int(r * 255))),
                max(0, min(255, int(g * 255))),
                max(0, min(255, int(b * 255))),
            )
            face_pixel_colours[fid] = pc

            # Record centroid in pixel space for the KDTree
            cpx, cpy = _to_pixel(cx, cy)
            centroid_pixels.append((cpx, cpy))
            colour_array.append(pc)
            face_id_order.append(fid)

    if not face_colours:
        img = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
        img.save(str(output_path))
        return output_path

    # Draw polygon fills on a sentinel background
    img = Image.new("RGB", (tile_size, tile_size), _BG_SENTINEL)
    draw = ImageDraw.Draw(img)

    for fid, face in detail_grid.faces.items():
        verts = []
        for vid in face.vertex_ids:
            v = detail_grid.vertices.get(vid)
            if v is None or not v.has_position():
                break
            verts.append(_to_pixel(v.x, v.y))
        else:
            if len(verts) >= 3 and fid in face_pixel_colours:
                colour = face_pixel_colours[fid]
                draw.polygon(verts, fill=colour, outline=colour)

    # ── Step 2: IDW fill for background pixels ──────────────────
    # Convert to numpy for fast pixel scanning
    pixels = np.array(img)  # (H, W, 3) uint8

    # Find background pixels (sentinel colour)
    bg_mask = (
        (pixels[:, :, 0] == _BG_SENTINEL[0]) &
        (pixels[:, :, 1] == _BG_SENTINEL[1]) &
        (pixels[:, :, 2] == _BG_SENTINEL[2])
    )  # (H, W) bool

    bg_count = bg_mask.sum()
    if bg_count > 0 and len(centroid_pixels) > 0:
        # Build KDTree of face centroid pixel positions
        centroid_arr = np.array(centroid_pixels, dtype=np.float64)  # (F, 2)
        colour_arr = np.array(colour_array, dtype=np.float64)      # (F, 3)
        tree = KDTree(centroid_arr)

        # Get coordinates of all background pixels
        bg_rows, bg_cols = np.where(bg_mask)  # each (M,)
        # KDTree query expects (x, y) = (col, row) to match centroid_pixels
        bg_points = np.column_stack([
            bg_cols.astype(np.float64),
            bg_rows.astype(np.float64),
        ])  # (M, 2)

        k_actual = min(k_neighbours, len(centroid_arr))
        dists, idxs = tree.query(bg_points, k=k_actual, workers=-1)

        if k_actual == 1:
            dists = dists[:, np.newaxis]
            idxs = idxs[:, np.newaxis]

        # IDW weights
        eps = 1e-12
        weights = 1.0 / np.maximum(dists, eps)           # (M, K)
        w_sum = weights.sum(axis=1, keepdims=True)        # (M, 1)
        norm_w = weights / w_sum                          # (M, K)

        # Gather neighbour colours and weighted-sum
        # colour_arr[idxs] → (M, K, 3)
        neighbour_colours = colour_arr[idxs]              # (M, K, 3)
        # Weighted colour: (M, K, 1) * (M, K, 3) → sum over K → (M, 3)
        blended = (norm_w[:, :, np.newaxis] * neighbour_colours).sum(axis=1)

        # Write back into the pixel array
        blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
        pixels[bg_rows, bg_cols] = blended_uint8

    img = Image.fromarray(pixels, "RGB")
    img.save(str(output_path))
    return output_path
