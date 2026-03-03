"""Full-slot tile texture rendering — Phase 16A + 16D hex softening.

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

Phase 16D — Hex Shape Softening
---------------------------------
Three techniques to break up the visible hexagonal sub-face grid:

- **16D.1 Sub-face edge dissolution** — small random jitter of polygon
  vertex positions (1–2 px) prevents perfectly straight sub-face edges.
- **16D.2 Pixel-level noise overlay** — high-frequency FBM noise adds
  micro-variation (±5 %) that masks uniform flat-colour sub-faces.
- **16D.3 Sub-face colour dithering** — each sub-face's colour is
  blended toward its neighbours' colours based on pixel distance from
  the centroid, softening hard colour boundaries.

Functions
---------
- :func:`render_detail_texture_fullslot` — full-slot PIL renderer
- :func:`build_face_lookup` — spatial index builder
- :func:`interpolate_at_pixel` — IDW interpolation helper
- :func:`jitter_polygon_vertices` — 16D.1 vertex jitter
- :func:`apply_noise_overlay` — 16D.2 pixel noise
- :func:`apply_colour_dithering` — 16D.3 neighbour-blended dithering
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
    vertex_jitter: float = 1.5,
    noise_overlay: bool = True,
    noise_frequency: float = 0.05,
    noise_amplitude: float = 0.05,
    colour_dither: bool = True,
    dither_radius: float = 6.0,
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

    Phase 16D hex-softening enhancements (all optional):

    3. **Vertex jitter** (16D.1) — sub-face polygon vertices are
       displaced by ±*vertex_jitter* pixels to dissolve straight
       edges.  Set ``vertex_jitter=0`` to disable.
    4. **Noise overlay** (16D.2) — a high-frequency FBM noise layer
       adds ±*noise_amplitude* micro-variation per pixel.
    5. **Colour dithering** (16D.3) — pixels near sub-face edges
       are blended toward neighbour colours, softening hard colour
       boundaries.

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
    vertex_jitter : float
        Maximum vertex displacement in pixels (16D.1).  Set 0 to
        disable.  Default 1.5.
    noise_overlay : bool
        Enable pixel-level noise overlay (16D.2).  Default True.
    noise_frequency : float
        Noise spatial frequency (16D.2).  Default 0.05.
    noise_amplitude : float
        Noise colour shift fraction (16D.2).  Default 0.05.
    colour_dither : bool
        Enable sub-face colour dithering (16D.3).  Default True.
    dither_radius : float
        Blend radius in pixels for dithering (16D.3).  Default 6.0.

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
                # 16D.1 — vertex jitter
                if vertex_jitter > 0:
                    verts = jitter_polygon_vertices(
                        verts,
                        max_jitter=vertex_jitter,
                        seed=noise_seed + hash(fid) % 100_000,
                    )
                colour = face_pixel_colours[fid]
                draw.polygon(verts, fill=colour, outline=colour)

    # Convert to numpy for further processing
    pixels = np.array(img)  # (H, W, 3) uint8

    # ── Step 1b: Colour dithering (16D.3) ───────────────────────
    # Applied before IDW fill so only polygon-covered pixels are
    # dithered — background pixels will be filled next.
    if colour_dither and len(centroid_pixels) >= 2:
        centroid_arr_d = np.array(centroid_pixels, dtype=np.float64)
        colour_arr_d = np.array(colour_array, dtype=np.float64)
        pixels = apply_colour_dithering(
            pixels, centroid_arr_d, colour_arr_d,
            blend_radius=dither_radius,
            k_neighbours=min(k_neighbours, len(centroid_arr_d)),
        )

    # ── Step 2: IDW fill for background pixels ──────────────────
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

    # ── Step 3: Pixel-level noise overlay (16D.2) ───────────────
    if noise_overlay:
        pixels = apply_noise_overlay(
            pixels,
            frequency=noise_frequency,
            amplitude=noise_amplitude,
            seed=noise_seed + 7777,
        )

    img = Image.fromarray(pixels, "RGB")
    img.save(str(output_path))
    return output_path


# ═══════════════════════════════════════════════════════════════════
# 16B — Soft tile-edge blending mask
# ═══════════════════════════════════════════════════════════════════

def _signed_distance_to_polygon(
    px: np.ndarray,
    py: np.ndarray,
    poly_x: np.ndarray,
    poly_y: np.ndarray,
) -> np.ndarray:
    """Signed distance from points to a convex polygon boundary.

    Positive = inside, negative = outside.

    Parameters
    ----------
    px, py : np.ndarray, shape (N,)
        Query point coordinates.
    poly_x, poly_y : np.ndarray, shape (V,)
        Polygon vertex coordinates (ordered, closed automatically).

    Returns
    -------
    np.ndarray, shape (N,)
        Signed distances (positive inside, negative outside).
    """
    n_verts = len(poly_x)
    n_pts = len(px)

    # Start with a large positive distance (inside)
    min_dist_sq = np.full(n_pts, np.inf, dtype=np.float64)

    # For each edge, compute distance to that segment
    for i in range(n_verts):
        j = (i + 1) % n_verts
        # Edge from (ax, ay) to (bx, by)
        ax, ay = poly_x[i], poly_y[i]
        bx, by = poly_x[j], poly_y[j]

        dx = bx - ax
        dy = by - ay
        edge_len_sq = dx * dx + dy * dy
        if edge_len_sq < 1e-20:
            continue

        # Project each point onto the edge
        t = ((px - ax) * dx + (py - ay) * dy) / edge_len_sq
        t = np.clip(t, 0.0, 1.0)

        # Closest point on edge
        cx = ax + t * dx
        cy = ay + t * dy

        dist_sq = (px - cx) ** 2 + (py - cy) ** 2
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)

    min_dist = np.sqrt(min_dist_sq)

    # Determine inside/outside using winding number (ray casting)
    inside = np.zeros(n_pts, dtype=bool)
    for i in range(n_verts):
        j = (i + 1) % n_verts
        yi, yj = poly_y[i], poly_y[j]
        xi, xj = poly_x[i], poly_x[j]

        # Points where the edge crosses the horizontal ray from (px, py)
        cond = ((yi <= py) & (yj > py)) | ((yj <= py) & (yi > py))
        if not np.any(cond):
            continue
        # x-coordinate of intersection
        slope = (xj - xi) / (yj - yi + 1e-30)
        x_intersect = xi + slope * (py - yi)
        inside[cond] ^= (px[cond] < x_intersect[cond])

    # Signed distance: positive inside, negative outside
    signed = np.where(inside, min_dist, -min_dist)
    return signed


def compute_tile_blend_mask(
    detail_grid: PolyGrid,
    tile_size: int = 256,
    fade_width: int = 16,
    *,
    overscan: float = 0.15,
) -> np.ndarray:
    """Compute a soft blending mask for a tile texture.

    The mask is 1.0 at the tile centre and fades to 0.0 at the tile
    edges over *fade_width* pixels, following the hex/pent polygon
    shape using signed distance from the convex hull boundary.

    Parameters
    ----------
    detail_grid : PolyGrid
        The sub-face grid whose vertex positions define the polygon.
    tile_size : int
        Image dimension.
    fade_width : int
        Width of the fade zone in pixels.  Pixels more than
        *fade_width* inside the polygon boundary are 1.0; pixels
        at the boundary are 0.5; pixels *fade_width* outside are 0.0.
    overscan : float
        Must match the overscan used in ``render_detail_texture_fullslot()``.

    Returns
    -------
    np.ndarray, shape (tile_size, tile_size), float32 in [0, 1]
    """
    from scipy.spatial import ConvexHull

    # Gather all vertex positions
    xs, ys = [], []
    for v in detail_grid.vertices.values():
        if v.has_position():
            xs.append(v.x)
            ys.append(v.y)

    if len(xs) < 3:
        return np.ones((tile_size, tile_size), dtype=np.float32)

    # Compute convex hull to get the outer polygon
    points = np.column_stack([xs, ys])
    hull = ConvexHull(points)
    hull_verts = points[hull.vertices]  # (V, 2) in grid coords

    # Grid → pixel coordinate transform (must match fullslot renderer)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
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

    # Convert hull vertices to pixel coordinates
    hull_px = (hull_verts[:, 0] - x_min) * scale + ox
    hull_py = tile_size - ((hull_verts[:, 1] - y_min) * scale + oy)

    # Build pixel coordinate grid
    col_coords = np.arange(tile_size, dtype=np.float64)
    row_coords = np.arange(tile_size, dtype=np.float64)
    px_grid, py_grid = np.meshgrid(col_coords, row_coords)

    # Compute signed distance for every pixel
    sd = _signed_distance_to_polygon(
        px_grid.ravel(), py_grid.ravel(),
        hull_px, hull_py,
    ).reshape(tile_size, tile_size)

    # Convert signed distance to blend weight:
    # sd >= fade_width → 1.0 (fully opaque, tile interior)
    # sd <= 0          → ramp from 0.5 down to 0.0 at -fade_width
    # 0 < sd < fade_width → ramp from 0.5 up to 1.0
    if fade_width <= 0:
        mask = np.where(sd >= 0, 1.0, 0.0).astype(np.float32)
    else:
        # Map signed distance to [0, 1]: -fade_width→0, +fade_width→1
        mask = np.clip((sd + fade_width) / (2.0 * fade_width), 0.0, 1.0)
        mask = mask.astype(np.float32)

    return mask


def apply_blend_mask_to_atlas(
    atlas_pixels: np.ndarray,
    masks: Dict[str, np.ndarray],
    face_ids: List[str],
    tile_size: int,
    gutter: int,
    columns: int,
) -> np.ndarray:
    """Apply per-tile blend masks to an assembled atlas.

    Multiplies each tile's slot by its mask, blending toward the
    gutter pixels (which come from edge clamping).

    Parameters
    ----------
    atlas_pixels : np.ndarray, shape (H, W, 3), uint8
        The assembled atlas image.
    masks : dict
        ``{face_id: mask_array}`` where each mask is
        ``(tile_size, tile_size)`` float32 in [0, 1].
    face_ids : list of str
        Ordered face IDs matching atlas layout.
    tile_size : int
    gutter : int
    columns : int

    Returns
    -------
    np.ndarray, shape (H, W, 3), uint8
        Atlas with blend masks applied.
    """
    result = atlas_pixels.copy()
    slot_size = tile_size + 2 * gutter

    for idx, fid in enumerate(face_ids):
        if fid not in masks:
            continue

        col = idx % columns
        row = idx // columns
        inner_x = col * slot_size + gutter
        inner_y = row * slot_size + gutter

        mask = masks[fid]  # (tile_size, tile_size) float32
        tile_region = result[inner_y:inner_y + tile_size,
                             inner_x:inner_x + tile_size].astype(np.float32)

        # The gutter already has edge-clamped pixels.  We blend
        # toward a neutral "background" colour that will be replaced
        # by the gutter.  For simplicity, we just darken/fade toward
        # the existing gutter values by applying the mask as a
        # brightness multiplier (mask=1 → full colour, mask=0 → dim).
        # A more sophisticated approach would composite with neighbour
        # tiles, but this already reduces boundary contrast significantly.
        blended = tile_region * mask[:, :, np.newaxis]
        result[inner_y:inner_y + tile_size,
               inner_x:inner_x + tile_size] = np.clip(
            blended, 0, 255,
        ).astype(np.uint8)

    return result


# ═══════════════════════════════════════════════════════════════════
# 16D — Hex Shape Softening
# ═══════════════════════════════════════════════════════════════════


# ── 16D.1 — Sub-face edge dissolution (vertex jitter) ──────────

def jitter_polygon_vertices(
    vertices: List[Tuple[float, float]],
    *,
    max_jitter: float = 1.5,
    seed: int = 0,
) -> List[Tuple[float, float]]:
    """Apply small deterministic jitter to pixel-space polygon vertices.

    The jitter is seeded so that the same vertex gets the same offset
    on every call, ensuring deterministic output.  The magnitude is
    small enough (1–2 px) that topology is not distorted but perfectly
    straight sub-face edges are broken up.

    Parameters
    ----------
    vertices : list of (float, float)
        Polygon vertices in pixel coordinates.
    max_jitter : float
        Maximum displacement in pixels (each axis).  Default 1.5 px.
    seed : int
        Noise seed — combined with vertex position hash for determinism.

    Returns
    -------
    list of (float, float)
        Jittered vertices.
    """
    rng = np.random.RandomState(seed)
    result: List[Tuple[float, float]] = []
    for vx, vy in vertices:
        # Hash the vertex position to get a per-vertex seed component
        h = hash((round(vx, 4), round(vy, 4)))
        rng_v = np.random.RandomState((seed + h) & 0x7FFFFFFF)
        dx = rng_v.uniform(-max_jitter, max_jitter)
        dy = rng_v.uniform(-max_jitter, max_jitter)
        result.append((vx + dx, vy + dy))
    return result


# ── 16D.2 — Pixel-level noise overlay ──────────────────────────

def apply_noise_overlay(
    pixels: np.ndarray,
    *,
    frequency: float = 0.05,
    amplitude: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Apply a high-frequency noise layer to an RGB pixel array.

    Adds micro-variation that breaks up the uniform colour within each
    sub-face and masks the regular grid pattern.

    The noise field is deterministic (same seed → same perturbation).

    Parameters
    ----------
    pixels : np.ndarray, shape (H, W, 3), uint8
        The tile image.
    frequency : float
        Spatial frequency of the noise in cycles per pixel.
        Default 0.05 (one cycle per 20 pixels).
    amplitude : float
        Maximum colour shift as a fraction of the base value.
        Default 0.05 (±5 %).
    seed : int
        Noise seed.

    Returns
    -------
    np.ndarray, shape (H, W, 3), uint8
        The modified image.
    """
    from .noise import fbm

    h, w = pixels.shape[:2]
    result = pixels.astype(np.float64)

    # Build a 2D noise field in one pass (vectorised via numpy broadcast).
    # We sample fbm at each pixel — but fbm is a pure-Python call,
    # so we vectorise the RNG part: pre-build the noise grid.
    noise_grid = np.empty((h, w), dtype=np.float64)
    for row in range(h):
        for col in range(w):
            n = fbm(
                col * frequency, row * frequency,
                octaves=3, lacunarity=2.0, persistence=0.5,
                frequency=1.0, seed=seed,
            )
            # n ∈ approx [-1, 1] — scale to [-amplitude, +amplitude]
            noise_grid[row, col] = n * amplitude

    # Apply as multiplicative brightness shift: pixel * (1 + noise)
    noise_3d = 1.0 + noise_grid[:, :, np.newaxis]  # (H, W, 1)
    result *= noise_3d
    return np.clip(result, 0, 255).astype(np.uint8)


# ── 16D.3 — Sub-face colour dithering ──────────────────────────

def apply_colour_dithering(
    pixels: np.ndarray,
    centroid_pixels: np.ndarray,
    colour_arr: np.ndarray,
    *,
    blend_radius: float = 6.0,
    k_neighbours: int = 4,
) -> np.ndarray:
    """Soften hard sub-face boundaries by blending toward neighbours.

    For each pixel, the existing polygon-rendered colour is blended
    with the IDW-weighted average of the K nearest sub-face colours.
    Pixels near their own centroid keep the face colour (blend_t ≈ 0);
    pixels near a sub-face edge get more blending (blend_t up to ~0.5).

    This is applied *before* the IDW background fill, so only polygon-
    covered pixels are affected.

    Parameters
    ----------
    pixels : np.ndarray, shape (H, W, 3), uint8
        Current tile pixels (polygon-rasterised, may contain sentinel).
    centroid_pixels : np.ndarray, shape (F, 2), float64
        Sub-face centroid positions in pixel coords.
    colour_arr : np.ndarray, shape (F, 3), float64
        Per-face colours (0–255 range).
    blend_radius : float
        Distance in pixels over which blending ramps up.  Pixels
        closer than ``blend_radius`` to their nearest centroid get
        linearly increasing blend; beyond ``blend_radius`` blend_t
        saturates at 0.5.
    k_neighbours : int
        Number of nearest centroids for the IDW colour estimate.

    Returns
    -------
    np.ndarray, shape (H, W, 3), uint8
    """
    from scipy.spatial import KDTree

    if len(centroid_pixels) < 2:
        return pixels

    h, w = pixels.shape[:2]
    tree = KDTree(centroid_pixels)

    # Only dither non-sentinel pixels
    is_sentinel = (
        (pixels[:, :, 0] == _BG_SENTINEL[0]) &
        (pixels[:, :, 1] == _BG_SENTINEL[1]) &
        (pixels[:, :, 2] == _BG_SENTINEL[2])
    )

    # Get all non-sentinel pixel coordinates
    rows, cols = np.where(~is_sentinel)
    if len(rows) == 0:
        return pixels

    pts = np.column_stack([cols.astype(np.float64), rows.astype(np.float64)])

    # Query K nearest centroids for every non-sentinel pixel
    k_actual = min(k_neighbours, len(centroid_pixels))
    dists, idxs = tree.query(pts, k=k_actual, workers=-1)

    if k_actual == 1:
        dists = dists[:, np.newaxis]
        idxs = idxs[:, np.newaxis]

    # IDW-weighted average colour from neighbours
    eps = 1e-12
    weights = 1.0 / np.maximum(dists, eps)       # (M, K)
    w_sum = weights.sum(axis=1, keepdims=True)    # (M, 1)
    norm_w = weights / w_sum                       # (M, K)
    neighbour_colours = colour_arr[idxs]           # (M, K, 3)
    idw_colour = (norm_w[:, :, np.newaxis] * neighbour_colours).sum(axis=1)  # (M, 3)

    # Blend factor based on distance to *nearest* centroid:
    # near centroid → blend_t ≈ 0 (keep face colour)
    # near edge → blend_t → 0.5 (half-blend with IDW average)
    nearest_dist = dists[:, 0]
    blend_t = np.clip(nearest_dist / max(blend_radius, eps), 0.0, 1.0) * 0.5

    # Blend original pixel colour with IDW colour
    result = pixels.copy()
    orig = pixels[rows, cols].astype(np.float64)  # (M, 3)
    blended = orig * (1.0 - blend_t[:, np.newaxis]) + idw_colour * blend_t[:, np.newaxis]
    result[rows, cols] = np.clip(blended, 0, 255).astype(np.uint8)

    return result
