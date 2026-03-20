"""Improved globe renderer — gap-free, subdivided, batched.

Phase 12 renderer that addresses three problems with the v1 renderer:

1. **Texture bleeding** — tile textures are flood-filled to remove
   black borders, so bilinear sampling never sees black pixels.

2. **Sphere subdivision** — each tile's triangle fan is subdivided
   and projected onto the sphere surface, producing smooth curvature
   instead of flat facets.

3. **Batched draw** — all tiles are merged into a single VBO with
   one draw call, eliminating per-tile overhead.

Public API
----------
- :func:`render_globe_v2` — launch an interactive pyglet 3D viewer
- :func:`flood_fill_tile_texture` — remove black borders from a tile PNG
- :func:`flood_fill_atlas` — apply flood-fill to a whole atlas
- :func:`classify_water_tiles` — identify water tiles by colour heuristic
- :func:`build_atmosphere_shell` — atmosphere sphere mesh (Phase 13G)
- :func:`build_background_quad` — fullscreen gradient background quad
- :func:`build_lod_batched_globe_mesh` — adaptive-LOD globe mesh (Phase 13F)
- :func:`select_lod_level` — choose subdivision level by screen fraction
- :func:`estimate_tile_screen_fraction` — angular-size LOD heuristic
- :func:`is_tile_backfacing` — frustum/backface culling test
- :func:`stitch_lod_boundary` — snap high-LOD boundary to low-LOD vertices
"""
from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from models.objects.goldberg import generate_goldberg_tiles
    _HAS_MODELS = True
except ImportError:
    _HAS_MODELS = False


def _compute_tile_uvs(
    tile_uv_vertices: List[Tuple[float, float]],
    atlas_slot: Tuple[float, float, float, float],
) -> List[Tuple[float, float]]:
    """Map a tile's normalised UV vertices into an atlas slot."""
    u_min, v_min, u_max, v_max = atlas_slot
    u_span, v_span = u_max - u_min, v_max - v_min
    return [
        (u_min + max(0.0, min(1.0, u)) * u_span,
         v_min + max(0.0, min(1.0, v)) * v_span)
        for u, v in tile_uv_vertices
    ]


# ═══════════════════════════════════════════════════════════════════
# 12A — Texture bleeding (flood-fill black borders)
# ═══════════════════════════════════════════════════════════════════

def flood_fill_tile_texture(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    iterations: int = 8,
) -> Path:
    """Remove black borders from a tile texture by flood-filling outward.

    Black pixels (R+G+B < threshold) adjacent to coloured pixels are
    replaced with the average of their coloured neighbours.  Repeated
    for *iterations* to fill the full border region.

    Parameters
    ----------
    image_path : Path
        Input tile texture PNG.
    output_path : Path, optional
        Output path.  Defaults to overwriting input.
    iterations : int
        Number of dilation passes.

    Returns
    -------
    Path
    """
    from PIL import Image

    image_path = Path(image_path)
    if output_path is None:
        output_path = image_path
    output_path = Path(output_path)

    img = Image.open(str(image_path)).convert("RGB")
    arr = np.array(img, dtype=np.float32)  # (H, W, 3)
    h, w = arr.shape[:2]

    # Mask: True where pixel has meaningful colour (not black)
    threshold = 10.0  # sum of RGB channels
    filled = arr[:, :, 0] + arr[:, :, 1] + arr[:, :, 2] > threshold

    for _ in range(iterations):
        # Find unfilled pixels that have at least one filled neighbour
        new_arr = arr.copy()
        new_filled = filled.copy()

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            # Shifted filled mask
            shifted = np.zeros_like(filled)
            shifted_col = np.zeros_like(arr)

            sy = slice(max(0, -dy), h + min(0, -dy))
            sx = slice(max(0, -dx), w + min(0, -dx))
            ty = slice(max(0, dy), h + min(0, dy))
            tx = slice(max(0, dx), w + min(0, dx))

            shifted[ty, tx] = filled[sy, sx]
            shifted_col[ty, tx] = arr[sy, sx]

            # Unfilled pixels that can be filled from this direction
            candidates = (~filled) & shifted
            if candidates.any():
                new_arr[candidates] += shifted_col[candidates]
                new_filled[candidates] = True

        # Average the accumulated colours
        # Count how many neighbours contributed
        count = np.zeros((h, w), dtype=np.float32)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            shifted = np.zeros_like(filled)
            sy = slice(max(0, -dy), h + min(0, -dy))
            sx = slice(max(0, -dx), w + min(0, -dx))
            ty = slice(max(0, dy), h + min(0, dy))
            tx = slice(max(0, dx), w + min(0, dx))
            shifted[ty, tx] = filled[sy, sx]
            count += (~filled) & shifted

        newly_filled = (~filled) & new_filled
        count_safe = np.maximum(count, 1.0)
        for c in range(3):
            # new_arr already has sum of neighbour colours for newly_filled
            # But we also added the original (which was ~0 for black pixels)
            # So subtract the original black value and divide by count
            new_arr[newly_filled, c] = (
                (new_arr[newly_filled, c] - arr[newly_filled, c])
                / count_safe[newly_filled]
            )

        arr = new_arr
        filled = new_filled

    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    out.save(str(output_path))
    return output_path


def flood_fill_atlas(
    atlas_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    iterations: int = 8,
) -> Path:
    """Apply flood-fill border removal to an entire atlas image.

    The atlas is a grid of tile textures.  Each tile may have black
    corners/borders.  This flood-fills every black pixel that borders
    a coloured pixel, extending tile colours to fill the full atlas.

    Parameters
    ----------
    atlas_path : Path
    output_path : Path, optional
    iterations : int

    Returns
    -------
    Path
    """
    return flood_fill_tile_texture(atlas_path, output_path, iterations=iterations)


# ═══════════════════════════════════════════════════════════════════
# 13C — UV polygon inset & clamping
# ═══════════════════════════════════════════════════════════════════

def compute_uv_polygon_inset(
    uv_polygon: List[Tuple[float, float]],
    inset_px: float = 1.5,
    texture_size: int = 256,
    atlas_size: int = 1,
) -> List[Tuple[float, float]]:
    """Shrink a UV polygon inward by *inset_px* texels.

    Each polygon edge is moved toward the centroid by an amount equal
    to *inset_px / atlas_size* in UV space.  The result is a slightly
    smaller polygon that stays safely inside the textured region.

    Parameters
    ----------
    uv_polygon : list of (u, v)
        The tile's polygon in atlas UV coordinates.
    inset_px : float
        Inset distance in *atlas* pixels.
    texture_size : int
        Individual tile texture size (for reference, not directly used).
    atlas_size : int
        Atlas width/height in pixels.  Used to convert pixel distance to
        UV distance.

    Returns
    -------
    list of (u, v)
        The inset polygon.
    """
    if atlas_size < 1:
        atlas_size = 1

    # Compute centroid
    n = len(uv_polygon)
    if n < 3:
        return list(uv_polygon)

    cu = sum(p[0] for p in uv_polygon) / n
    cv = sum(p[1] for p in uv_polygon) / n

    inset_uv = inset_px / atlas_size

    result = []
    for u, v in uv_polygon:
        du = u - cu
        dv = v - cv
        dist = math.sqrt(du * du + dv * dv)
        if dist < 1e-12:
            result.append((u, v))
            continue
        # Move toward centroid by inset_uv
        shrink = max(0.0, 1.0 - inset_uv / dist)
        result.append((cu + du * shrink, cv + dv * shrink))

    return result


def clamp_uv_to_polygon(
    u: float,
    v: float,
    polygon: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """Clamp a UV point to the nearest point inside *polygon*.

    If the point is inside the polygon it is returned unchanged.
    Otherwise it is projected onto the nearest polygon edge.

    Parameters
    ----------
    u, v : float
        The UV coordinate to clamp.
    polygon : list of (u, v)
        Convex polygon vertices in order.

    Returns
    -------
    (u, v) clamped
    """
    if _point_in_convex_polygon(u, v, polygon):
        return (u, v)
    return _nearest_point_on_polygon_edge(u, v, polygon)


def _point_in_convex_polygon(
    px: float, py: float,
    polygon: List[Tuple[float, float]],
) -> bool:
    """Test if (px, py) is inside a convex polygon (winding test)."""
    n = len(polygon)
    if n < 3:
        return False
    sign = None
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-7:
            continue  # treat as on-edge → skip
        s = cross > 0
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True


def _nearest_point_on_segment(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> Tuple[float, float]:
    """Project point (px,py) onto segment (a→b), clamped to [0,1]."""
    dx = bx - ax
    dy = by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-20:
        return (ax, ay)
    t = ((px - ax) * dx + (py - ay) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    return (ax + t * dx, ay + t * dy)


def _nearest_point_on_polygon_edge(
    px: float, py: float,
    polygon: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """Find the closest point on any polygon edge to (px, py)."""
    best_dist = float("inf")
    best_pt = (px, py)
    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        cx, cy = _nearest_point_on_segment(px, py, ax, ay, bx, by)
        d = (cx - px) ** 2 + (cy - py) ** 2
        if d < best_dist:
            best_dist = d
            best_pt = (cx, cy)
    return best_pt


# ═══════════════════════════════════════════════════════════════════
# 13D — Cross-tile colour harmonisation
# ═══════════════════════════════════════════════════════════════════

def blend_biome_configs(
    a: "BiomeConfig",
    b: "BiomeConfig",
    t: float,
) -> "BiomeConfig":
    """Linearly interpolate two :class:`BiomeConfig` instances.

    Parameters
    ----------
    a, b : BiomeConfig
        Source biome configs.
    t : float
        Blend weight (0.0 = purely *a*, 1.0 = purely *b*).

    Returns
    -------
    BiomeConfig
        A new config with every numeric field interpolated.
    """
    from .detail_render import BiomeConfig

    t = max(0.0, min(1.0, t))
    s = 1.0 - t

    return BiomeConfig(
        base_ramp=a.base_ramp,  # keep discrete field from a
        vegetation_density=s * a.vegetation_density + t * b.vegetation_density,
        rock_exposure=s * a.rock_exposure + t * b.rock_exposure,
        snow_line=s * a.snow_line + t * b.snow_line,
        water_level=s * a.water_level + t * b.water_level,
        moisture=s * a.moisture + t * b.moisture,
        hillshade_strength=s * a.hillshade_strength + t * b.hillshade_strength,
        azimuth=s * a.azimuth + t * b.azimuth,
        altitude=s * a.altitude + t * b.altitude,
    )


def compute_neighbour_average_colours(
    tile_colour_map: Dict[int, Tuple[float, float, float]],
    tiles: Any,
) -> Dict[int, Tuple[float, float, float]]:
    """Compute per-tile boundary colour as the average of neighbour colours.

    For each tile, the "edge colour" is the mean of all its neighbours'
    colours (from *tile_colour_map*).  Tiles without neighbours or
    without a colour entry keep their own colour.

    Parameters
    ----------
    tile_colour_map : dict
        ``{tile_index: (r, g, b)}``.
    tiles : sequence of GoldbergTile
        Tile objects with ``.index`` and ``.neighbor_indices``.

    Returns
    -------
    dict
        ``{tile_index: (r, g, b)}`` — the averaged neighbour colour per tile.
    """
    default_colour = (1.0, 1.0, 1.0)
    neighbour_avg: Dict[int, Tuple[float, float, float]] = {}

    for tile in tiles:
        nbs = getattr(tile, "neighbor_indices", ())
        if not nbs:
            neighbour_avg[tile.index] = tile_colour_map.get(
                tile.index, default_colour,
            )
            continue

        r_sum = g_sum = b_sum = 0.0
        count = 0
        for nb_idx in nbs:
            nb_col = tile_colour_map.get(nb_idx)
            if nb_col is not None:
                r_sum += nb_col[0]
                g_sum += nb_col[1]
                b_sum += nb_col[2]
                count += 1

        if count > 0:
            neighbour_avg[tile.index] = (
                r_sum / count, g_sum / count, b_sum / count,
            )
        else:
            neighbour_avg[tile.index] = tile_colour_map.get(
                tile.index, default_colour,
            )

    return neighbour_avg


def harmonise_tile_colours(
    tile_colour_map: Dict[int, Tuple[float, float, float]],
    tiles: Any,
    *,
    strength: float = 0.5,
) -> Dict[int, Tuple[float, float, float]]:
    """Blend each tile's colour toward the average of its neighbours.

    This produces a "smoothed" colour map where abrupt biome
    boundaries are softened.  The original map is not mutated.

    Parameters
    ----------
    tile_colour_map : dict
        ``{tile_index: (r, g, b)}``.
    tiles : sequence of GoldbergTile
        Tile objects with ``.index`` and ``.neighbor_indices``.
    strength : float
        How much to blend toward neighbours (0 = no change,
        1 = fully replace with neighbour average).

    Returns
    -------
    dict
        A new ``{tile_index: (r, g, b)}`` colour map.
    """
    strength = max(0.0, min(1.0, strength))
    if strength == 0.0:
        return dict(tile_colour_map)

    nb_avg = compute_neighbour_average_colours(tile_colour_map, tiles)
    result: Dict[int, Tuple[float, float, float]] = {}
    s = 1.0 - strength

    for tile in tiles:
        own = tile_colour_map.get(tile.index, (1.0, 1.0, 1.0))
        avg = nb_avg.get(tile.index, own)
        result[tile.index] = (
            s * own[0] + strength * avg[0],
            s * own[1] + strength * avg[1],
            s * own[2] + strength * avg[2],
        )

    return result


# ═══════════════════════════════════════════════════════════════════
# 13H — Water rendering
# ═══════════════════════════════════════════════════════════════════

DEFAULT_WATER_LEVEL = 0.12  # matches BiomeConfig.water_level


def classify_water_tiles(
    tile_colour_map: Dict[int, Tuple[float, float, float]],
    *,
    water_level: float = DEFAULT_WATER_LEVEL,
) -> Dict[int, bool]:
    """Classify tiles as water or land based on colour heuristic.

    A tile is considered water when its blue channel dominates red and
    green by at least *water_level* (i.e. ``blue - max(red, green) >=
    water_level``).  This mirrors the heuristic used in the PBR shader.

    Parameters
    ----------
    tile_colour_map : dict
        ``{tile_index: (r, g, b)}`` with values in ``[0, 1]``.
    water_level : float
        Minimum blue-dominance threshold.  Default ``0.12`` aligns
        with :data:`BiomeConfig.water_level`.

    Returns
    -------
    dict
        ``{tile_index: True}`` for water tiles, ``{tile_index: False}``
        for land tiles.
    """
    result: Dict[int, bool] = {}
    for idx, (r, g, b) in tile_colour_map.items():
        water_hint = b - max(r, g)
        result[idx] = water_hint >= water_level
    return result


def compute_water_depth(
    r: float, g: float, b: float,
    water_level: float = DEFAULT_WATER_LEVEL,
) -> float:
    """Compute a [0, 1] water depth proxy from an RGB tile colour.

    Returns 0.0 for land tiles or tiles at the water surface,
    and approaches 1.0 for the deepest ocean blue.

    Parameters
    ----------
    r, g, b : float
        Tile colour channels in ``[0, 1]``.
    water_level : float
        Blue-dominance threshold.

    Returns
    -------
    float
        Normalised depth in ``[0, 1]``.
    """
    water_hint = b - max(r, g)
    if water_hint < water_level:
        return 0.0
    # Normalise: water_hint in [water_level, 1.0] → [0, 1]
    return min(1.0, (water_hint - water_level) / max(1.0 - water_level, 0.01))


# ═══════════════════════════════════════════════════════════════════
# 13E — Normal-mapped lighting
# ═══════════════════════════════════════════════════════════════════

def encode_normal_to_rgb(
    nx: float, ny: float, nz: float,
) -> Tuple[int, int, int]:
    """Encode a unit normal vector as an RGB triplet.

    Maps each component from [-1, 1] → [0, 255] using the standard
    ``c * 0.5 + 0.5`` encoding.  A flat surface (0, 0, 1) becomes
    ``(128, 128, 255)`` — the classic "blue" of a flat normal map.
    """
    r = int(max(0, min(255, round((nx * 0.5 + 0.5) * 255))))
    g = int(max(0, min(255, round((ny * 0.5 + 0.5) * 255))))
    b = int(max(0, min(255, round((nz * 0.5 + 0.5) * 255))))
    return (r, g, b)


def decode_rgb_to_normal(
    r: int, g: int, b: int,
) -> Tuple[float, float, float]:
    """Decode an RGB-encoded normal back to a unit vector.

    Inverse of :func:`encode_normal_to_rgb`.
    """
    nx = (r / 255.0) * 2.0 - 1.0
    ny = (g / 255.0) * 2.0 - 1.0
    nz = (b / 255.0) * 2.0 - 1.0
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length > 1e-10:
        nx /= length
        ny /= length
        nz /= length
    return (nx, ny, nz)


def build_normal_map_atlas(
    normal_maps: Dict[str, Dict[str, Tuple[float, float, float]]],
    collection: Any,
    *,
    tile_size: int = 256,
    columns: int = 0,
    gutter: int = 4,
) -> Tuple[Any, Dict[str, Tuple[float, float, float, float]]]:
    """Build an atlas texture encoding per-sub-face normals as RGB.

    Each tile's sub-face normals (from :func:`compute_normal_map` or
    :func:`compute_all_normal_maps`) are rendered as an RGB image
    where ``(r, g, b) = (nx*0.5+0.5, ny*0.5+0.5, nz*0.5+0.5)``.

    The atlas layout mirrors :func:`build_detail_atlas`: each tile
    gets a ``(tile_size + 2*gutter)`` slot in a grid.

    Parameters
    ----------
    normal_maps : dict
        ``{face_id: {sub_face_id: (nx, ny, nz)}}``.
    collection : DetailGridCollection
        The detail grid collection (for grid geometry).
    tile_size : int
        Side length of each tile image.
    columns : int
        Number of atlas columns (0 = auto).
    gutter : int
        Padding pixels around each slot.

    Returns
    -------
    (atlas_image, uv_layout)
        *atlas_image* is a PIL Image (RGB).
        *uv_layout* maps ``face_id → (u_min, v_min, u_max, v_max)``.
    """
    from PIL import Image, ImageDraw

    face_ids = collection.face_ids
    n = len(face_ids)
    if n == 0:
        raise ValueError("No detail grids in the collection")

    if columns <= 0:
        columns = max(1, math.isqrt(n))
        if columns * columns < n:
            columns += 1
    rows = math.ceil(n / columns)

    slot_size = tile_size + 2 * gutter
    atlas_w = columns * slot_size
    atlas_h = rows * slot_size

    # Default flat normal (0, 0, 1) → blue
    flat_rgb = encode_normal_to_rgb(0.0, 0.0, 1.0)
    atlas = Image.new("RGB", (atlas_w, atlas_h), flat_rgb)

    uv_layout: Dict[str, Tuple[float, float, float, float]] = {}

    for idx, fid in enumerate(face_ids):
        col = idx % columns
        row = idx // columns
        slot_x = col * slot_size
        slot_y = row * slot_size

        # Render this tile's normal map
        tile_img = _render_normal_tile(
            collection, fid,
            normal_maps.get(fid, {}),
            tile_size=tile_size,
        )

        atlas.paste(tile_img, (slot_x + gutter, slot_y + gutter))

        # Fill gutter by clamping edge pixels outward
        if gutter > 0:
            _fill_normal_gutter(atlas, slot_x, slot_y, tile_size, gutter)

        # UV coordinates
        inner_x = slot_x + gutter
        inner_y = slot_y + gutter
        u_min = inner_x / atlas_w
        u_max = (inner_x + tile_size) / atlas_w
        v_min = 1.0 - (inner_y + tile_size) / atlas_h
        v_max = 1.0 - inner_y / atlas_h
        uv_layout[fid] = (u_min, v_min, u_max, v_max)

    return atlas, uv_layout


def _render_normal_tile(
    collection: Any,
    face_id: str,
    normals: Dict[str, Tuple[float, float, float]],
    *,
    tile_size: int = 256,
) -> Any:
    """Render a single tile's normal map as a PIL Image.

    Uses the same face-centroid → pixel mapping as the colour
    texture renderer.  Each sub-face is painted with its encoded
    normal colour.
    """
    from PIL import Image, ImageDraw
    from .geometry import face_center

    grid = collection.grids[face_id]
    flat_rgb = encode_normal_to_rgb(0.0, 0.0, 1.0)
    img = Image.new("RGB", (tile_size, tile_size), flat_rgb)
    draw = ImageDraw.Draw(img)

    # Compute bounding box of all face centres for UV → pixel mapping
    centres = {}
    for fid, face in grid.faces.items():
        c = face_center(grid.vertices, face)
        if c is not None:
            centres[fid] = c

    if not centres:
        return img

    xs = [c[0] for c in centres.values()]
    ys = [c[1] for c in centres.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = x_max - x_min if x_max > x_min else 1.0
    y_span = y_max - y_min if y_max > y_min else 1.0

    # Map each sub-face to a pixel and paint
    margin = 2  # pixel margin to avoid edge gaps
    usable = tile_size - 2 * margin

    for fid, face in grid.faces.items():
        c = centres.get(fid)
        if c is None:
            continue

        nx, ny, nz = normals.get(fid, (0.0, 0.0, 1.0))
        rgb = encode_normal_to_rgb(nx, ny, nz)

        # Map centroid to pixel
        px = margin + (c[0] - x_min) / x_span * usable
        py = margin + (c[1] - y_min) / y_span * usable

        # Draw a small filled circle at each sub-face centroid
        r = max(1, usable // (2 * max(1, int(math.sqrt(len(grid.faces))))))
        draw.ellipse(
            [px - r, py - r, px + r, py + r],
            fill=rgb,
        )

    return img


def _fill_normal_gutter(
    atlas: Any,
    slot_x: int, slot_y: int,
    tile_size: int, gutter: int,
) -> None:
    """Fill gutter pixels around a normal map slot (same logic as colour atlas)."""
    inner_x = slot_x + gutter
    inner_y = slot_y + gutter

    # Top gutter
    top_strip = atlas.crop((inner_x, inner_y, inner_x + tile_size, inner_y + 1))
    for g in range(gutter):
        atlas.paste(top_strip, (inner_x, slot_y + g))

    # Bottom gutter
    bot_y = inner_y + tile_size - 1
    bot_strip = atlas.crop((inner_x, bot_y, inner_x + tile_size, bot_y + 1))
    for g in range(gutter):
        atlas.paste(bot_strip, (inner_x, inner_y + tile_size + g))

    # Left gutter (full height)
    full_top = slot_y
    full_bot = slot_y + tile_size + 2 * gutter
    left_strip = atlas.crop((inner_x, full_top, inner_x + 1, full_bot))
    for g in range(gutter):
        atlas.paste(left_strip, (slot_x + g, full_top))

    # Right gutter (full height)
    right_x = inner_x + tile_size - 1
    right_strip = atlas.crop((right_x, full_top, right_x + 1, full_bot))
    for g in range(gutter):
        atlas.paste(right_strip, (inner_x + tile_size + g, full_top))


# ═══════════════════════════════════════════════════════════════════
# 12B — Sphere subdivision
# ═══════════════════════════════════════════════════════════════════

def _normalize_vec3(v: np.ndarray) -> np.ndarray:
    """Normalize a 3-vector to unit length."""
    length = np.linalg.norm(v)
    if length < 1e-12:
        return v
    return v / length


def _project_to_sphere(point: np.ndarray, radius: float) -> np.ndarray:
    """Project a point onto the sphere of given radius.

    The ``+ 0.0`` eliminates IEEE 754 negative-zero values so that
    shared boundary vertices from adjacent tiles produce bit-identical
    float32 positions.  Without this the GPU rasteriser can leave
    1-pixel gaps at tile seams.
    """
    return _normalize_vec3(point) * radius + 0.0


def subdivide_tile_mesh(
    center: Tuple[float, float, float],
    vertices: List[Tuple[float, float, float]],
    center_uv: Tuple[float, float],
    vertex_uvs: List[Tuple[float, float]],
    color: Tuple[float, float, float],
    radius: float = 1.0,
    subdivisions: int = 3,
    uv_clamp_polygon: Optional[List[Tuple[float, float]]] = None,
    edge_color: Optional[Tuple[float, float, float]] = None,
    tangent: Optional[Tuple[float, float, float]] = None,
    bitangent: Optional[Tuple[float, float, float]] = None,
    water_flag: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subdivide a tile's triangle fan and project onto sphere surface.

    Each triangle (center → v[i] → v[i+1]) is subdivided into a grid
    of ``subdivisions²`` smaller triangles using barycentric interpolation,
    then each vertex is projected onto the sphere.

    Parameters
    ----------
    center, vertices
        3D positions (center of tile and polygon boundary vertices).
    center_uv, vertex_uvs
        Corresponding UV coordinates.
    color
        RGB vertex colour at the tile centre.
    radius
        Sphere radius.
    subdivisions
        Number of subdivision levels per triangle edge.
    uv_clamp_polygon : list of (u, v), optional
        If provided, every generated UV is clamped to lie inside this
        convex polygon.  Typically the tile's atlas UV polygon inset
        by 1–2 texels (see :func:`compute_uv_polygon_inset`).
    edge_color : (r, g, b), optional
        If provided, vertex colours blend from *color* at the tile
        centre to *edge_color* at the tile boundary.  The blend
        weight is the barycentric distance from the centre vertex
        (``1 − b0``).  Used by Phase 13D colour harmonisation.
    tangent : (x, y, z), optional
        Tile tangent vector for normal mapping.  When provided
        (together with *bitangent*), the output vertex format
        expands from 8 to 14 floats (or 9 to 15 with water flag):
        pos(3) + col(3) + uv(2) + T(3) + B(3) [+ water(1)].
    bitangent : (x, y, z), optional
        Tile bitangent vector.  Must be provided with *tangent*.
    water_flag : float or None
        Per-tile water indicator.  ``None`` (default) means no water
        column is included.  When a float is given (``1.0`` for water,
        ``0.0`` for land), the vertex stride gains one extra float
        at the end: 9 (basic) or 15 (normal-mapped).

    Returns
    -------
    (vertex_data, index_data)
        vertex_data : np.ndarray — shape depends on options (see below)
        index_data : np.ndarray, shape (M, 3) — triangle indices

    The vertex stride depends on the combination of flags:

    ========== ======== ======
    TBN        water    stride
    ========== ======== ======
    False      None     8
    False      float    9
    True       None     14
    True       float    15
    ========== ======== ======
    """
    has_tbn = tangent is not None and bitangent is not None
    has_water = water_flag is not None
    base_stride = 14 if has_tbn else 8
    stride = base_stride + (1 if has_water else 0)
    center_pos = np.array(center, dtype=np.float64)
    n = len(vertices)
    all_verts = []  # list of (pos, uv, vert_color) tuples
    all_tris = []   # list of (i0, i1, i2) index tuples
    vert_map = {}   # (tri_idx, row, col) → global vertex index

    s = subdivisions

    # Pre-compute colour arrays for blending
    col_center = np.array(color, dtype=np.float64)
    col_edge = (
        np.array(edge_color, dtype=np.float64) if edge_color is not None
        else col_center
    )

    for tri_idx in range(n):
        i_next = (tri_idx + 1) % n

        p0 = center_pos
        p1 = np.array(vertices[tri_idx], dtype=np.float64)
        p2 = np.array(vertices[i_next], dtype=np.float64)

        uv0 = np.array(center_uv, dtype=np.float64)
        uv1 = np.array(vertex_uvs[tri_idx], dtype=np.float64)
        uv2 = np.array(vertex_uvs[i_next], dtype=np.float64)

        # Generate subdivided vertices using barycentric coords
        # For each row i (0..s) and col j (0..s-i):
        #   bary = ((s-i-j)/s, i/s, j/s) for (p0, p1, p2)
        for i in range(s + 1):
            for j in range(s - i + 1):
                k = s - i - j
                b0 = k / s
                b1 = i / s
                b2 = j / s

                pos = b0 * p0 + b1 * p1 + b2 * p2
                pos = _project_to_sphere(pos, radius)
                uv = b0 * uv0 + b1 * uv1 + b2 * uv2

                # Clamp interior UVs only.  Boundary vertices (b0 == 0)
                # lie on the tile polygon edge shared with adjacent
                # tiles.  Clamping them can push two fan sectors'
                # copies of the same spatial point to different
                # "nearest edge" solutions on the inset polygon,
                # and vertex dedup then picks one arbitrarily —
                # producing the small triangular wedge artefact at
                # tile vertices.  Skipping the clamp for boundary
                # points keeps their UVs coherent across sectors.
                if uv_clamp_polygon is not None and b0 > 0:
                    cu, cv = clamp_uv_to_polygon(
                        float(uv[0]), float(uv[1]), uv_clamp_polygon
                    )
                    uv = np.array([cu, cv], dtype=np.float64)

                # Per-vertex colour: blend from center to edge
                # b0=1 at center, b0=0 at boundary
                vert_color = b0 * col_center + (1.0 - b0) * col_edge

                key = (tri_idx, i, j)
                vert_map[key] = len(all_verts)
                all_verts.append((pos, uv, vert_color))

        # Generate triangles
        for i in range(s):
            for j in range(s - i):
                # Upper triangle
                v00 = vert_map[(tri_idx, i, j)]
                v10 = vert_map[(tri_idx, i + 1, j)]
                v01 = vert_map[(tri_idx, i, j + 1)]
                all_tris.append((v00, v10, v01))

                # Lower triangle (if it exists)
                if i + j + 1 < s:
                    v11 = vert_map[(tri_idx, i + 1, j + 1)]
                    all_tris.append((v10, v11, v01))

    # Deduplicate vertices by position AND UV.
    #
    # Earlier code used position-only keys, which was safe only when
    # UVs at shared positions were guaranteed identical.  UV clamping
    # (even with the boundary-skip guard above) can theoretically
    # produce different UVs at the same 3D position.  Including a
    # quantised UV in the key prevents merging vertices whose UVs
    # disagree, eliminating the "wedge triangle" artefact at tile
    # junctions.  The cost is a marginally larger vertex buffer at
    # polygon corners (negligible in practice).
    final_verts = []
    final_map = {}  # old_index → new_index
    pos_uv_hash = {}  # (rounded_pos, rounded_uv) → new_index

    for old_idx, (pos, uv, vc) in enumerate(all_verts):
        # Normalize near-zero components to +0.0 to avoid -0.0 vs +0.0
        # mismatches which can cause rasterisation edge ownership
        # differences on some GPUs.  Also clamp tiny noise to exact
        # zero before rounding.
        px0 = 0.0 if abs(pos[0]) < 1e-12 else pos[0]
        px1 = 0.0 if abs(pos[1]) < 1e-12 else pos[1]
        px2 = 0.0 if abs(pos[2]) < 1e-12 else pos[2]
        key = (
            round(px0, 7), round(px1, 7), round(px2, 7),
            round(float(uv[0]), 6), round(float(uv[1]), 6),
        )
        if key in pos_uv_hash:
            final_map[old_idx] = pos_uv_hash[key]
        else:
            new_idx = len(final_verts)
            pos_uv_hash[key] = new_idx
            final_map[old_idx] = new_idx
            final_verts.append((pos, uv, vc))

    # Build output arrays
    vertex_data = np.zeros((len(final_verts), stride), dtype=np.float32)
    for i, (pos, uv, vc) in enumerate(final_verts):
        vertex_data[i, 0:3] = pos
        vertex_data[i, 3:6] = vc
        vertex_data[i, 6:8] = uv
        if has_tbn:
            # Re-derive tangent/bitangent at this vertex to stay
            # orthogonal to the sphere normal at the projected position.
            sphere_n = _normalize_vec3(np.array(pos, dtype=np.float64))
            tile_t = np.array(tangent, dtype=np.float64)
            # Gram-Schmidt: make tangent perpendicular to sphere normal
            t_orth = tile_t - np.dot(tile_t, sphere_n) * sphere_n
            t_len = np.linalg.norm(t_orth)
            if t_len > 1e-10:
                t_orth /= t_len
            else:
                t_orth = tile_t  # degenerate — keep original
            # Bitangent = cross(N, T)
            b_orth = np.cross(sphere_n, t_orth)
            b_len = np.linalg.norm(b_orth)
            if b_len > 1e-10:
                b_orth /= b_len
            vertex_data[i, 8:11] = t_orth
            vertex_data[i, 11:14] = b_orth
        if has_water:
            vertex_data[i, base_stride] = water_flag

    index_data = np.array(
        [(final_map[a], final_map[b], final_map[c]) for a, b, c in all_tris],
        dtype=np.uint32,
    )

    return vertex_data, index_data


# ═══════════════════════════════════════════════════════════════════
# 12C — Batched globe mesh builder
# ═══════════════════════════════════════════════════════════════════

def build_batched_globe_mesh(
    frequency: int,
    uv_layout: Dict[str, Tuple[float, float, float, float]],
    tile_colour_map: Optional[Dict[int, Tuple[float, float, float]]] = None,
    *,
    radius: float = 1.0,
    subdivisions: int = 3,
    uv_inset_px: float = 0.0,
    atlas_size: Optional[int] = None,
    edge_blend: float = 0.0,
    normal_mapped: bool = False,
    water_tiles: Optional[Dict[int, bool]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a single merged mesh for the entire globe.

    Each tile is subdivided and sphere-projected, then all vertices
    and indices are concatenated into one VBO + IBO.

    Parameters
    ----------
    frequency : int
        Goldberg subdivision frequency.
    uv_layout : dict
        ``{face_id: (u_min, v_min, u_max, v_max)}`` from atlas.
    tile_colour_map : dict, optional
        ``{tile_index: (r, g, b)}``.  White if absent.
    radius : float
    subdivisions : int
        Per-triangle subdivision level.
    uv_inset_px : float
        How many atlas pixels to inset each tile's UV polygon so that no
        interpolated UV can sample outside the textured region.  Set to
        0 (default) to disable clamping.  Typical value: 1.0–2.0.
    atlas_size : int, optional
        Atlas width/height in pixels.  Required when *uv_inset_px* > 0.
    edge_blend : float
        Colour harmonisation strength (0–1).  When > 0, vertex colours
        at tile boundaries blend toward the average of the tile's
        neighbours, softening abrupt biome/colour transitions.  0 (default)
        disables blending.  Typical value: 0.3–0.6.
    normal_mapped : bool
        When True, include per-vertex tangent(3) + bitangent(3) in the
        vertex data, expanding the stride from 8 to 14 floats.  The
        tangent/bitangent are derived from each GoldbergTile's TBN
        frame and re-orthogonalised at each vertex against the sphere
        normal.  Required for normal-mapped PBR lighting (Phase 13E).
    water_tiles : dict, optional
        ``{tile_index: True/False}`` from :func:`classify_water_tiles`.
        When provided and any tile is flagged as water, the vertex
        stride gains one extra float (the water flag): 9 (basic),
        15 (normal-mapped).

    Returns
    -------
    (vertex_data, index_data)
        Concatenated arrays for a single draw call.
        Vertex stride is 8/9 (default) or 14/15 (when *normal_mapped=True*),
        depending on whether *water_tiles* contains any True values.
    """
    if not _HAS_MODELS:
        raise ImportError("models library required")

    if uv_inset_px > 0 and atlas_size is None:
        raise ValueError("atlas_size is required when uv_inset_px > 0")

    tiles = generate_goldberg_tiles(frequency=frequency, radius=radius)

    # Determine whether any tile is flagged as water
    has_any_water = water_tiles is not None and any(water_tiles.values())

    # Pre-compute neighbour-average edge colours for harmonisation
    edge_blend = max(0.0, min(1.0, edge_blend))
    nb_avg: Optional[Dict[int, Tuple[float, float, float]]] = None
    if edge_blend > 0.0 and tile_colour_map:
        nb_avg = compute_neighbour_average_colours(tile_colour_map, tiles)

    all_vertex_chunks = []
    all_index_chunks = []
    vertex_offset = 0

    for tile in tiles:
        fid = f"t{tile.index}"
        slot = uv_layout.get(fid)
        if slot is None:
            continue

        color = (1.0, 1.0, 1.0)
        if tile_colour_map:
            color = tile_colour_map.get(tile.index, color)

        # Compute edge colour for boundary harmonisation
        tile_edge_color: Optional[Tuple[float, float, float]] = None
        if nb_avg is not None and edge_blend > 0.0:
            avg = nb_avg.get(tile.index, color)
            # edge_color = lerp(own colour, neighbour avg, blend strength)
            s_own = 1.0 - edge_blend
            tile_edge_color = (
                s_own * color[0] + edge_blend * avg[0],
                s_own * color[1] + edge_blend * avg[1],
                s_own * color[2] + edge_blend * avg[2],
            )

        # Compute atlas-mapped UVs
        mapped_uvs = _compute_tile_uvs(list(tile.uv_vertices), slot)
        center_u = sum(uv[0] for uv in mapped_uvs) / len(mapped_uvs)
        center_v = sum(uv[1] for uv in mapped_uvs) / len(mapped_uvs)

        # Compute optional UV inset polygon for clamping
        clamp_poly: Optional[List[Tuple[float, float]]] = None
        if uv_inset_px > 0 and atlas_size is not None:
            clamp_poly = compute_uv_polygon_inset(
                mapped_uvs, inset_px=uv_inset_px, atlas_size=atlas_size,
            )

        # Per-tile water flag (only encode when at least one tile is water)
        wf: Optional[float] = None
        if has_any_water and water_tiles is not None:
            wf = 1.0 if water_tiles.get(tile.index, False) else 0.0

        vdata, idata = subdivide_tile_mesh(
            center=tuple(tile.center),
            vertices=[tuple(v) for v in tile.vertices],
            center_uv=(center_u, center_v),
            vertex_uvs=mapped_uvs,
            color=color,
            radius=radius,
            subdivisions=subdivisions,
            uv_clamp_polygon=clamp_poly,
            edge_color=tile_edge_color,
            tangent=tuple(tile.tangent) if normal_mapped else None,
            bitangent=tuple(tile.bitangent) if normal_mapped else None,
            water_flag=wf,
        )

        # Offset indices
        idata_offset = idata + vertex_offset
        vertex_offset += len(vdata)

        all_vertex_chunks.append(vdata)
        all_index_chunks.append(idata_offset)

    base_stride = 14 if normal_mapped else 8
    stride = base_stride + (1 if has_any_water else 0)
    if not all_vertex_chunks:
        return np.zeros((0, stride), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)

    vertex_data = np.concatenate(all_vertex_chunks, axis=0)
    index_data = np.concatenate(all_index_chunks, axis=0)

    return vertex_data, index_data


# ═══════════════════════════════════════════════════════════════════
# 13F — Adaptive mesh resolution (LOD)
# ═══════════════════════════════════════════════════════════════════

#: Standard LOD subdivision levels — coarsest to finest.
LOD_LEVELS: Tuple[int, ...] = (1, 2, 3, 5)

#: Default screen-space area thresholds (in normalised viewport fraction)
#: for switching between LOD levels.  ``LOD_THRESHOLDS[i]`` is the
#: *minimum* screen fraction at which ``LOD_LEVELS[i]`` is selected.
#: Tiles smaller than the lowest threshold get the coarsest LOD.
#: Must have ``len(LOD_THRESHOLDS) == len(LOD_LEVELS)``.
LOD_THRESHOLDS: Tuple[float, ...] = (0.0, 0.005, 0.02, 0.06)

#: Backface culling threshold — tiles whose ``dot(normal, view_dir)``
#: is below this value are culled.  Slightly negative to avoid popping
#: at the limb.
BACKFACE_THRESHOLD: float = -0.1


def select_lod_level(
    screen_fraction: float,
    *,
    lod_levels: Tuple[int, ...] = LOD_LEVELS,
    lod_thresholds: Tuple[float, ...] = LOD_THRESHOLDS,
) -> int:
    """Choose the subdivision level for a tile based on screen-space size.

    Parameters
    ----------
    screen_fraction : float
        Estimated fraction of the viewport occupied by the tile (0–1).
    lod_levels : tuple of int
        Available subdivision levels, coarsest first.
    lod_thresholds : tuple of float
        Minimum screen fraction for each LOD level.

    Returns
    -------
    int
        The subdivision level to use for this tile.
    """
    if len(lod_levels) != len(lod_thresholds):
        raise ValueError(
            f"lod_levels ({len(lod_levels)}) and lod_thresholds "
            f"({len(lod_thresholds)}) must have the same length"
        )
    selected = lod_levels[0]
    for level, threshold in zip(lod_levels, lod_thresholds):
        if screen_fraction >= threshold:
            selected = level
    return selected


def estimate_tile_screen_fraction(
    tile_center: Tuple[float, float, float],
    tile_edge_length: float,
    eye_position: Tuple[float, float, float],
    fov_y: float = math.radians(45),
) -> float:
    """Estimate what fraction of the viewport a tile covers.

    Uses the angular size of the tile as seen from the camera.

    Parameters
    ----------
    tile_center : (x, y, z)
        World-space position of the tile centre.
    tile_edge_length : float
        Average edge length of the tile (world units).
    eye_position : (x, y, z)
        Camera position in world space.
    fov_y : float
        Vertical field of view in radians.

    Returns
    -------
    float
        Estimated fraction of viewport height the tile subtends (0–1).
    """
    tc = np.array(tile_center, dtype=np.float64)
    ep = np.array(eye_position, dtype=np.float64)
    dist = np.linalg.norm(tc - ep)
    if dist < 1e-12:
        return 1.0  # camera inside tile — max LOD
    angular_size = 2.0 * math.atan2(tile_edge_length * 0.5, dist)
    fraction = angular_size / fov_y
    return min(1.0, max(0.0, fraction))


def is_tile_backfacing(
    tile_center: Tuple[float, float, float],
    tile_normal: Tuple[float, float, float],
    eye_position: Tuple[float, float, float],
    threshold: float = BACKFACE_THRESHOLD,
) -> bool:
    """Determine whether a tile faces away from the camera.

    Parameters
    ----------
    tile_center : (x, y, z)
    tile_normal : (x, y, z)
        Outward-pointing normal of the tile face.
    eye_position : (x, y, z)
    threshold : float
        Dot-product threshold.  Tiles with
        ``dot(normal, view_dir) < threshold`` are backfacing.
        A small negative value avoids popping at the limb.

    Returns
    -------
    bool
        True if the tile should be culled (not rendered).
    """
    tc = np.array(tile_center, dtype=np.float64)
    tn = np.array(tile_normal, dtype=np.float64)
    ep = np.array(eye_position, dtype=np.float64)
    view_dir = ep - tc
    length = np.linalg.norm(view_dir)
    if length < 1e-12:
        return False  # camera at tile — don't cull
    view_dir /= length
    n_len = np.linalg.norm(tn)
    if n_len < 1e-12:
        return False
    tn /= n_len
    return float(np.dot(tn, view_dir)) < threshold


def stitch_lod_boundary(
    high_verts: np.ndarray,
    low_verts: np.ndarray,
    shared_edge_start: Tuple[float, float, float],
    shared_edge_end: Tuple[float, float, float],
    tolerance: float = 1e-5,
) -> np.ndarray:
    """Snap higher-LOD boundary vertices to lower-LOD edge positions.

    When two adjacent tiles use different LOD levels, the higher-LOD
    tile has more vertices along the shared boundary than the lower-LOD
    tile.  This creates T-junctions that manifest as cracks.

    This function finds vertices in *high_verts* that lie on the edge
    from *shared_edge_start* to *shared_edge_end* (within *tolerance*)
    and snaps them to the nearest position that also exists in
    *low_verts* on that same edge.

    Parameters
    ----------
    high_verts : np.ndarray, shape (N, C)
        Vertex data from the higher-LOD tile.  Columns 0–2 are XYZ.
        **Modified in-place and returned.**
    low_verts : np.ndarray, shape (M, C)
        Vertex data from the lower-LOD tile.  Read-only.
    shared_edge_start, shared_edge_end : (x, y, z)
        Endpoints of the shared boundary edge.
    tolerance : float
        Maximum distance from the edge line for a vertex to be
        considered "on the boundary".

    Returns
    -------
    np.ndarray
        The *high_verts* array (same object, modified in-place).
    """
    p0 = np.array(shared_edge_start, dtype=np.float64)
    p1 = np.array(shared_edge_end, dtype=np.float64)
    edge_vec = p1 - p0
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-12:
        return high_verts

    edge_dir = edge_vec / edge_len

    # Find low-LOD vertices that lie on this edge
    low_on_edge = []
    for i in range(len(low_verts)):
        v = low_verts[i, :3].astype(np.float64)
        t = np.dot(v - p0, edge_dir)
        if t < -tolerance or t > edge_len + tolerance:
            continue
        projected = p0 + t * edge_dir
        dist = np.linalg.norm(v - projected)
        if dist < tolerance:
            low_on_edge.append((t, v))

    if not low_on_edge:
        return high_verts

    # Sort low-LOD edge vertices by parameter t
    low_on_edge.sort(key=lambda pair: pair[0])
    low_ts = np.array([pair[0] for pair in low_on_edge])
    low_positions = [pair[1] for pair in low_on_edge]

    # Snap high-LOD edge vertices to nearest low-LOD positions
    for i in range(len(high_verts)):
        v = high_verts[i, :3].astype(np.float64)
        t = np.dot(v - p0, edge_dir)
        if t < -tolerance or t > edge_len + tolerance:
            continue
        projected = p0 + t * edge_dir
        dist = np.linalg.norm(v - projected)
        if dist >= tolerance:
            continue
        # This vertex is on the shared edge — snap to nearest low-LOD pos
        idx = int(np.argmin(np.abs(low_ts - t)))
        high_verts[i, :3] = low_positions[idx].astype(np.float32)

    return high_verts


def build_lod_batched_globe_mesh(
    frequency: int,
    uv_layout: Dict[str, Tuple[float, float, float, float]],
    tile_colour_map: Optional[Dict[int, Tuple[float, float, float]]] = None,
    *,
    radius: float = 1.0,
    eye_position: Tuple[float, float, float] = (0.0, 0.0, 3.0),
    fov_y: float = math.radians(45),
    lod_levels: Tuple[int, ...] = LOD_LEVELS,
    lod_thresholds: Tuple[float, ...] = LOD_THRESHOLDS,
    backface_cull: bool = True,
    backface_threshold: float = BACKFACE_THRESHOLD,
    uv_inset_px: float = 0.0,
    atlas_size: Optional[int] = None,
    edge_blend: float = 0.0,
    normal_mapped: bool = False,
    water_tiles: Optional[Dict[int, bool]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """Build a batched globe mesh with per-tile adaptive LOD.

    Like :func:`build_batched_globe_mesh`, but each tile gets an
    individually-chosen subdivision level based on its screen-space
    size, and back-facing tiles are optionally culled.

    Parameters
    ----------
    frequency, uv_layout, tile_colour_map, radius
        Same as :func:`build_batched_globe_mesh`.
    eye_position : (x, y, z)
        Camera world-space position for LOD and culling decisions.
    fov_y : float
        Vertical field of view in radians.
    lod_levels : tuple of int
        Available subdivision levels (sorted coarsest → finest).
    lod_thresholds : tuple of float
        Screen fraction thresholds for each LOD level.
    backface_cull : bool
        Whether to skip back-facing tiles.
    backface_threshold : float
        Dot-product threshold for backface test.
    uv_inset_px, atlas_size, edge_blend, normal_mapped, water_tiles
        Same as :func:`build_batched_globe_mesh`.

    Returns
    -------
    (vertex_data, index_data, tile_lod_map)
        vertex_data, index_data — same format as build_batched_globe_mesh.
        tile_lod_map — ``{tile_index: subdivision_level}`` for each
        rendered tile (excludes culled tiles).
    """
    if not _HAS_MODELS:
        raise ImportError("models library required")

    if uv_inset_px > 0 and atlas_size is None:
        raise ValueError("atlas_size is required when uv_inset_px > 0")

    tiles = generate_goldberg_tiles(frequency=frequency, radius=radius)

    # Determine whether any tile is flagged as water
    has_any_water = water_tiles is not None and any(water_tiles.values())

    # Pre-compute neighbour-average edge colours for harmonisation
    edge_blend = max(0.0, min(1.0, edge_blend))
    nb_avg: Optional[Dict[int, Tuple[float, float, float]]] = None
    if edge_blend > 0.0 and tile_colour_map:
        nb_avg = compute_neighbour_average_colours(tile_colour_map, tiles)

    all_vertex_chunks: List[Tuple[int, np.ndarray]] = []  # (tile_idx, vdata)
    all_index_chunks: List[np.ndarray] = []
    tile_lod_map: Dict[int, int] = {}
    vertex_offset = 0

    for tile in tiles:
        # Backface culling
        if backface_cull and is_tile_backfacing(
            tuple(tile.center), tuple(tile.normal),
            eye_position, backface_threshold,
        ):
            continue

        # LOD selection
        screen_frac = estimate_tile_screen_fraction(
            tuple(tile.center), tile.edge_length,
            eye_position, fov_y,
        )
        subdiv = select_lod_level(
            screen_frac,
            lod_levels=lod_levels,
            lod_thresholds=lod_thresholds,
        )
        tile_lod_map[tile.index] = subdiv

        fid = f"t{tile.index}"
        slot = uv_layout.get(fid)
        if slot is None:
            continue

        color = (1.0, 1.0, 1.0)
        if tile_colour_map:
            color = tile_colour_map.get(tile.index, color)

        # Compute edge colour for boundary harmonisation
        tile_edge_color: Optional[Tuple[float, float, float]] = None
        if nb_avg is not None and edge_blend > 0.0:
            avg = nb_avg.get(tile.index, color)
            s_own = 1.0 - edge_blend
            tile_edge_color = (
                s_own * color[0] + edge_blend * avg[0],
                s_own * color[1] + edge_blend * avg[1],
                s_own * color[2] + edge_blend * avg[2],
            )

        # Compute atlas-mapped UVs
        mapped_uvs = _compute_tile_uvs(list(tile.uv_vertices), slot)
        center_u = sum(uv[0] for uv in mapped_uvs) / len(mapped_uvs)
        center_v = sum(uv[1] for uv in mapped_uvs) / len(mapped_uvs)

        # Compute optional UV inset polygon for clamping
        clamp_poly: Optional[List[Tuple[float, float]]] = None
        if uv_inset_px > 0 and atlas_size is not None:
            clamp_poly = compute_uv_polygon_inset(
                mapped_uvs, inset_px=uv_inset_px, atlas_size=atlas_size,
            )

        # Per-tile water flag
        wf: Optional[float] = None
        if has_any_water and water_tiles is not None:
            wf = 1.0 if water_tiles.get(tile.index, False) else 0.0

        vdata, idata = subdivide_tile_mesh(
            center=tuple(tile.center),
            vertices=[tuple(v) for v in tile.vertices],
            center_uv=(center_u, center_v),
            vertex_uvs=mapped_uvs,
            color=color,
            radius=radius,
            subdivisions=subdiv,
            uv_clamp_polygon=clamp_poly,
            edge_color=tile_edge_color,
            tangent=tuple(tile.tangent) if normal_mapped else None,
            bitangent=tuple(tile.bitangent) if normal_mapped else None,
            water_flag=wf,
        )

        # Offset indices
        idata_offset = idata + vertex_offset
        vertex_offset += len(vdata)

        all_vertex_chunks.append((tile.index, vdata))
        all_index_chunks.append(idata_offset)

    base_stride = 14 if normal_mapped else 8
    stride = base_stride + (1 if has_any_water else 0)
    if not all_vertex_chunks:
        return (
            np.zeros((0, stride), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint32),
            tile_lod_map,
        )

    vertex_data = np.concatenate([v for _, v in all_vertex_chunks], axis=0)
    index_data = np.concatenate(all_index_chunks, axis=0)

    return vertex_data, index_data, tile_lod_map


# ═══════════════════════════════════════════════════════════════════
# 13G — Atmosphere & post-processing
# ═══════════════════════════════════════════════════════════════════

ATMOSPHERE_SCALE = 1.025  # shell is 2.5% larger than the globe
ATMOSPHERE_COLOR = (0.4, 0.6, 1.0)  # pale blue haze
BLOOM_THRESHOLD = 0.8  # luminance cutoff for bloom extraction
BLOOM_INTENSITY = 0.3  # blend strength of bloom overlay

# Background gradient colours
BG_CENTER_COLOR = (0.02, 0.03, 0.08)  # dark blue at center
BG_EDGE_COLOR = (0.0, 0.0, 0.0)       # black at edges


def build_atmosphere_shell(
    radius: float = 1.0,
    *,
    scale: float = ATMOSPHERE_SCALE,
    lat_segments: int = 32,
    lon_segments: int = 64,
    color: Tuple[float, float, float] = ATMOSPHERE_COLOR,
    falloff: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a transparent atmosphere shell around the globe.

    Creates an icosphere-like UV sphere slightly larger than the globe.
    Each vertex stores a Fresnel-like alpha value: transparent when
    facing the camera (center of globe), opaque at the limb (edges).

    The alpha at each vertex is pre-computed as
    ``(1 - abs(dot(normal, view_up)))^falloff``, approximating
    limb brightening.  The actual view-dependent Fresnel is applied
    in the atmosphere fragment shader; the vertex alpha is a static
    hint for sorting and a reasonable fallback.

    Parameters
    ----------
    radius : float
        Globe radius.
    scale : float
        Multiplier for the atmosphere shell radius (default 1.025).
    lat_segments, lon_segments : int
        Tessellation resolution.
    color : (r, g, b)
        Base atmosphere tint.
    falloff : float
        Fresnel falloff exponent.  Higher = thinner haze.

    Returns
    -------
    (vertex_data, index_data)
        vertex_data : np.ndarray, shape (N, 7) — pos(3) + rgba(4)
        index_data : np.ndarray, shape (M, 3) — triangle indices (uint32)
    """
    atmo_r = radius * scale
    verts = []
    for lat in range(lat_segments + 1):
        theta = math.pi * lat / lat_segments  # 0 at north pole → π at south
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        for lon in range(lon_segments + 1):
            phi = 2.0 * math.pi * lon / lon_segments
            x = atmo_r * sin_t * math.cos(phi)
            y = atmo_r * cos_t
            z = atmo_r * sin_t * math.sin(phi)
            # Normal = normalised position
            nx, ny, nz = x / atmo_r, y / atmo_r, z / atmo_r
            # Fresnel hint: how much this vertex faces "outward"
            # Use |ny| as a proxy for facing the viewer (camera at +z)
            # The real shader will recompute per-fragment, but this
            # gives a reasonable per-vertex alpha.
            facing = abs(nz)  # 1 when facing camera, 0 at limb
            alpha = (1.0 - facing) ** falloff
            alpha = max(0.0, min(1.0, alpha))
            verts.append((x, y, z, color[0], color[1], color[2], alpha))

    vertex_data = np.array(verts, dtype=np.float32)

    # Build triangle indices
    tris = []
    for lat in range(lat_segments):
        for lon in range(lon_segments):
            a = lat * (lon_segments + 1) + lon
            b = a + lon_segments + 1
            c = a + 1
            d = b + 1
            tris.append((a, b, c))
            tris.append((c, b, d))

    index_data = np.array(tris, dtype=np.uint32)
    return vertex_data, index_data


def build_background_quad() -> np.ndarray:
    """Build a fullscreen quad for the radial gradient background.

    Returns a vertex array with 4 vertices in clip space (no MVP needed),
    each with position(2) + uv(2).  The fragment shader uses the UV
    distance from centre to compute a radial gradient.

    Returns
    -------
    np.ndarray, shape (4, 4) — (x, y, u, v) for each corner.
        Draw with ``GL_TRIANGLE_STRIP``.  No index buffer needed.
    """
    # Clip-space positions spanning the full viewport; UVs [0,1]²
    return np.array([
        [-1.0, -1.0, 0.0, 0.0],
        [ 1.0, -1.0, 1.0, 0.0],
        [-1.0,  1.0, 0.0, 1.0],
        [ 1.0,  1.0, 1.0, 1.0],
    ], dtype=np.float32)


def compute_bloom_threshold(
    r: float, g: float, b: float,
    threshold: float = BLOOM_THRESHOLD,
) -> float:
    """Compute the luminance-based bloom contribution for a pixel.

    Returns a value in ``[0, 1]`` indicating how much bloom glow
    the pixel should emit.  Used by the bloom extraction pass.

    Parameters
    ----------
    r, g, b : float
        Linear HDR colour channels.
    threshold : float
        Luminance cutoff below which no bloom is emitted.

    Returns
    -------
    float
        Bloom intensity factor.
    """
    # Perceptual luminance (Rec. 709)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if lum < threshold:
        return 0.0
    # Soft knee: ramp from 0 at threshold to 1
    return min(1.0, (lum - threshold) / max(1.0 - threshold, 0.001))


# ── Atmosphere shaders ──────────────────────────────────────────────

_ATMO_VERTEX_SHADER = """\
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color_alpha;

uniform mat4 u_mvp;
uniform mat4 u_model;

out vec3 v_world_pos;
out vec4 v_color;

void main() {
    vec4 world4 = u_model * vec4(position, 1.0);
    v_world_pos = world4.xyz;
    v_color = color_alpha;
    gl_Position = u_mvp * world4;
}
"""

_ATMO_FRAGMENT_SHADER = """\
#version 330 core
in vec3 v_world_pos;
in vec4 v_color;

uniform vec3 u_eye_pos;

// Atmosphere parameters
const float ATMO_FALLOFF = 3.0;
const float ATMO_DENSITY = 0.6;

out vec4 frag_color;

void main() {
    // View-dependent Fresnel for atmospheric scattering
    vec3 N = normalize(v_world_pos);
    vec3 V = normalize(u_eye_pos - v_world_pos);
    float NdotV = max(0.0, dot(N, V));

    // Limb brightening: strongest at grazing angles
    float fresnel = pow(1.0 - NdotV, ATMO_FALLOFF);
    float alpha = fresnel * ATMO_DENSITY;

    frag_color = vec4(v_color.rgb, alpha);
}
"""

# ── Background gradient shaders ─────────────────────────────────────

_BG_VERTEX_SHADER = """\
#version 330 core
layout(location = 0) in vec2 position;
layout(location = 1) in vec2 uv;

out vec2 v_uv;

void main() {
    v_uv = uv;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

_BG_FRAGMENT_SHADER = """\
#version 330 core
in vec2 v_uv;

uniform vec3 u_center_color;
uniform vec3 u_edge_color;

out vec4 frag_color;

void main() {
    // Radial distance from center (0 at center, 1 at corners)
    vec2 centered = v_uv * 2.0 - 1.0;
    float dist = length(centered);
    dist = smoothstep(0.0, 1.4, dist);  // 1.4 ≈ sqrt(2) for corners

    vec3 color = mix(u_center_color, u_edge_color, dist);
    frag_color = vec4(color, 1.0);
}
"""

# ── Bloom post-processing shaders ───────────────────────────────────

_BLOOM_EXTRACT_SHADER = """\
#version 330 core
in vec2 v_uv;

uniform sampler2D u_scene;
uniform float     u_threshold;

out vec4 frag_color;

void main() {
    vec3 color = texture(u_scene, v_uv).rgb;
    float lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float contribution = max(0.0, lum - u_threshold)
                       / max(1.0 - u_threshold, 0.001);
    contribution = min(1.0, contribution);
    frag_color = vec4(color * contribution, 1.0);
}
"""

_BLOOM_BLUR_SHADER = """\
#version 330 core
in vec2 v_uv;

uniform sampler2D u_source;
uniform vec2      u_direction;  // (1/w, 0) or (0, 1/h) for H/V pass

// 5-tap Gaussian weights
const float weights[5] = float[](0.227027, 0.1945946, 0.1216216,
                                   0.054054, 0.016216);

out vec4 frag_color;

void main() {
    vec3 result = texture(u_source, v_uv).rgb * weights[0];
    for (int i = 1; i < 5; i++) {
        vec2 offset = u_direction * float(i);
        result += texture(u_source, v_uv + offset).rgb * weights[i];
        result += texture(u_source, v_uv - offset).rgb * weights[i];
    }
    frag_color = vec4(result, 1.0);
}
"""

_BLOOM_COMPOSITE_SHADER = """\
#version 330 core
in vec2 v_uv;

uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float     u_bloom_intensity;

out vec4 frag_color;

void main() {
    vec3 scene = texture(u_scene, v_uv).rgb;
    vec3 bloom = texture(u_bloom, v_uv).rgb;
    vec3 color = scene + bloom * u_bloom_intensity;
    // Reinhard tone-map the combined result
    color = color / (color + vec3(1.0));
    frag_color = vec4(color, 1.0);
}
"""


def get_atmosphere_shader_sources() -> Tuple[str, str]:
    """Return the atmosphere vertex and fragment shader source strings.

    The atmosphere shell uses alpha blending with view-dependent
    Fresnel to simulate limb haze.

    Returns
    -------
    (vertex_source, fragment_source)
    """
    return _ATMO_VERTEX_SHADER, _ATMO_FRAGMENT_SHADER


def get_background_shader_sources() -> Tuple[str, str]:
    """Return the background gradient vertex and fragment shaders.

    The background quad renders a radial gradient from dark blue
    (center) to black (edges) to simulate deep space.

    Returns
    -------
    (vertex_source, fragment_source)
    """
    return _BG_VERTEX_SHADER, _BG_FRAGMENT_SHADER


def get_bloom_shader_sources() -> Tuple[str, str, str]:
    """Return the three bloom post-processing shader sources.

    Bloom is a three-pass effect:
    1. **Extract** — isolate bright pixels above the luminance threshold
    2. **Blur** — apply a separable Gaussian blur (two passes: H + V)
    3. **Composite** — add the blurred bloom back onto the scene

    All three shaders use the same fullscreen-quad vertex shader
    (see :func:`get_background_shader_sources` for the vertex stage).

    Returns
    -------
    (extract_source, blur_source, composite_source)
    """
    return _BLOOM_EXTRACT_SHADER, _BLOOM_BLUR_SHADER, _BLOOM_COMPOSITE_SHADER


# ═══════════════════════════════════════════════════════════════════
# 12D — Interactive OpenGL renderer (v2)
# ═══════════════════════════════════════════════════════════════════

# Legacy shaders (8-float vertex: pos + col + uv)
_V2_VERTEX_SHADER = """\
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 uv;

uniform mat4 u_mvp;
uniform mat4 u_model;

out vec3 v_color;
out vec3 v_normal;
out vec2 v_uv;

void main() {
    vec3 world_pos = (u_model * vec4(position, 1.0)).xyz;
    v_color  = color;
    v_normal = normalize(world_pos);   // sphere normal = normalised position
    v_uv     = uv;
    gl_Position = u_mvp * vec4(world_pos, 1.0);
}
"""

_V2_FRAGMENT_SHADER = """\
#version 330 core
in vec3 v_color;
in vec3 v_normal;
in vec2 v_uv;

uniform sampler2D u_atlas;
uniform int       u_use_texture;
uniform vec3      u_light_dir;

out vec4 frag_color;

void main() {
    vec3 base;
    if (u_use_texture == 1) {
        base = texture(u_atlas, v_uv).rgb;
    } else {
        base = v_color;
    }

    // Hemisphere lighting: key light + ambient
    vec3 n = normalize(v_normal);
    float ndotl = dot(n, u_light_dir);
    float light = clamp(ndotl * 0.6 + 0.4, 0.2, 1.0);

    frag_color = vec4(base * light, 1.0);
}
"""


# ───────────────────────────────────────────────────────────────────
# 13E+13H — PBR-lite shaders (15-float vertex: pos+col+uv+T+B+water)
# ───────────────────────────────────────────────────────────────────

_PBR_VERTEX_SHADER = """\
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec3 tangent;
layout(location = 4) in vec3 bitangent;
layout(location = 5) in float water_flag;

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat3 u_normal_matrix;

out vec3 v_world_pos;
out vec3 v_color;
out vec2 v_uv;
out mat3 v_tbn;
out float v_water;

void main() {
    vec4 world4 = u_model * vec4(position, 1.0);
    v_world_pos = world4.xyz;
    v_color = color;
    v_uv    = uv;
    v_water = water_flag;

    // Build TBN matrix in world space
    vec3 N = normalize(u_normal_matrix * normalize(position));
    vec3 T = normalize(u_normal_matrix * tangent);
    vec3 B = normalize(u_normal_matrix * bitangent);
    v_tbn = mat3(T, B, N);

    gl_Position = u_mvp * world4;
}
"""

_PBR_FRAGMENT_SHADER = """\
#version 330 core
in vec3 v_world_pos;
in vec3 v_color;
in vec2 v_uv;
in mat3 v_tbn;
in float v_water;

uniform sampler2D u_atlas;
uniform sampler2D u_normal_map;
uniform int       u_use_texture;
uniform int       u_use_normal_map;
uniform vec3      u_light_dir;      // warm key light direction (normalised)
uniform vec3      u_fill_dir;       // cool fill light direction
uniform vec3      u_eye_pos;        // camera position in world space
uniform float     u_time;           // elapsed seconds for wave animation

// Lighting colours
const vec3 KEY_COLOR  = vec3(1.00, 0.95, 0.85);   // warm sunlight
const vec3 FILL_COLOR = vec3(0.40, 0.50, 0.70);   // cool sky fill
const vec3 SKY_AMB    = vec3(0.20, 0.25, 0.35);   // sky hemisphere ambient
const vec3 GND_AMB    = vec3(0.10, 0.08, 0.06);   // ground hemisphere ambient

// Material
const float ROUGHNESS_TERRAIN = 0.85;
const float ROUGHNESS_WATER   = 0.15;
const float WATER_THRESHOLD   = 0.12;  // matches BiomeConfig.water_level
const float FRESNEL_POWER     = 4.0;
const float FRESNEL_STRENGTH  = 0.25;

// 13H — Water rendering constants
const vec3  WATER_SHALLOW     = vec3(0.15, 0.55, 0.60);   // turquoise
const vec3  WATER_DEEP        = vec3(0.04, 0.08, 0.22);   // dark navy
const float WAVE_SPEED         = 0.8;
const float WAVE_SCALE         = 12.0;
const float WAVE_AMPLITUDE     = 0.06;

// 17D — Enhanced water constants
const float WATER_F0           = 0.02;   // Schlick F0 for water (IOR ≈ 1.33)
const float WATER_TEXTURE_MIX  = 0.65;   // how much baked texture to keep
const float SUN_SPEC_POWER     = 256.0;  // tight sun hotspot on calm water
const float SUN_SPEC_STRENGTH  = 1.8;    // intensity of sun specular

// 13H — Coastline emphasis
const vec3  COAST_COLOR       = vec3(0.85, 0.92, 0.95);   // bright foam
const float COAST_WIDTH       = 0.012;                      // world-space width

out vec4 frag_color;

void main() {
    // ── Base colour ────────────────────────────────────────────────
    vec3 base;
    if (u_use_texture == 1) {
        base = texture(u_atlas, v_uv).rgb;
    } else {
        base = v_color;
    }

    // ── Surface normal ─────────────────────────────────────────────
    vec3 N;
    if (u_use_normal_map == 1) {
        // Sample normal map: decode from [0,1] → [-1,1]
        vec3 nm = texture(u_normal_map, v_uv).rgb * 2.0 - 1.0;
        N = normalize(v_tbn * nm);
    } else {
        N = normalize(v_tbn[2]);  // geometric normal (column 2 = N)
    }

    // ── View direction ─────────────────────────────────────────────
    vec3 V = normalize(u_eye_pos - v_world_pos);

    // ── Water detection ────────────────────────────────────────────
    // Use explicit per-vertex flag when available, fall back to
    // blue-channel heuristic for backward compatibility.
    float water_hint;
    if (v_water > 0.5) {
        water_hint = 1.0;
    } else {
        water_hint = max(0.0, base.b - max(base.r, base.g));
        water_hint = smoothstep(0.0, 0.3, water_hint);
    }

    // ── Roughness (water is shinier) ───────────────────────────────
    float roughness = mix(ROUGHNESS_TERRAIN, ROUGHNESS_WATER, water_hint);

    // ── 13H.2 / 17D: Texture-aware water shader ─────────────────
    if (water_hint > 0.5) {
        // 17D.1 — Preserve baked ocean texture, blend with procedural
        // The atlas now contains depth gradients, waves, coastal detail
        // from the ocean render pipeline (17A-17C).
        vec3 baked_ocean = base;  // keep the baked texture

        // Fallback procedural colour for untextured water tiles
        float depth = max(0.0, base.b - max(base.r, base.g));
        depth = smoothstep(WATER_THRESHOLD, 0.8, depth);
        vec3 procedural_ocean = mix(WATER_SHALLOW, WATER_DEEP, depth);

        // If using texture atlas, blend baked texture with procedural
        // enhancement; otherwise fall back to pure procedural (backward
        // compatible with untextured water tiles).
        if (u_use_texture == 1) {
            base = mix(procedural_ocean, baked_ocean, WATER_TEXTURE_MIX);
        } else {
            base = procedural_ocean;
        }

        // Animated wave normal perturbation
        vec3 world_n = normalize(v_world_pos);
        float wave_phase = u_time * WAVE_SPEED;
        float wx = sin(v_world_pos.x * WAVE_SCALE + wave_phase)
                 * cos(v_world_pos.z * WAVE_SCALE * 0.7 + wave_phase * 0.6);
        float wz = cos(v_world_pos.x * WAVE_SCALE * 0.8 - wave_phase * 0.5)
                 * sin(v_world_pos.z * WAVE_SCALE + wave_phase * 0.9);
        vec3 wave_offset = vec3(wx, 0.0, wz) * WAVE_AMPLITUDE;
        N = normalize(N + wave_offset);

        // 17D.2 — Fresnel-based reflection (water-specific IOR)
        // At glancing angles: highly reflective (sky-coloured)
        // At steep angles: transparent (shows baked texture)
        float NdotV_water = max(0.0, dot(N, V));
        float fresnel_water = WATER_F0 + (1.0 - WATER_F0)
                            * pow(1.0 - NdotV_water, 5.0);
        vec3 sky_reflection = SKY_AMB * 1.5;  // brighter sky for water
        base = mix(base, sky_reflection, fresnel_water * 0.6);
    }

    // ── 13H.3: Coastline emphasis ──────────────────────────────────
    // Detect water-land boundary using screen-space derivative of
    // the water flag.  Large gradient = coastline.
    float dw_dx = dFdx(v_water);
    float dw_dy = dFdy(v_water);
    float coast_factor = smoothstep(0.0, COAST_WIDTH,
                                     length(vec2(dw_dx, dw_dy)));

    // ── Key light (warm directional) ───────────────────────────────
    float NdotL_key = max(0.0, dot(N, u_light_dir));
    vec3 diffuse_key = NdotL_key * KEY_COLOR;

    // ── Fill light (cool, from opposite side) ──────────────────────
    float NdotL_fill = max(0.0, dot(N, u_fill_dir));
    vec3 diffuse_fill = NdotL_fill * FILL_COLOR * 0.35;

    // ── Blinn-Phong specular ───────────────────────────────────────
    vec3 H = normalize(u_light_dir + V);
    float NdotH = max(0.0, dot(N, H));
    float spec_power = 2.0 / max(0.001, roughness * roughness) - 2.0;
    float spec = pow(NdotH, spec_power);
    // Fresnel-Schlick at half angle
    float HdotV = max(0.0, dot(H, V));
    float F = 0.04 + 0.96 * pow(1.0 - HdotV, 5.0);
    vec3 specular = spec * F * KEY_COLOR * (1.0 - roughness);

    // ── 17D.3: Sun specular hotspot on water ───────────────────────
    // Tight, bright specular reflection only on water surfaces.
    // Position moves with globe rotation via u_light_dir.
    vec3 sun_specular = vec3(0.0);
    if (water_hint > 0.5) {
        float sun_spec = pow(NdotH, SUN_SPEC_POWER);
        float sun_F = WATER_F0 + (1.0 - WATER_F0) * pow(1.0 - HdotV, 5.0);
        sun_specular = sun_spec * sun_F * KEY_COLOR * SUN_SPEC_STRENGTH;
    }

    // ── Hemisphere ambient ─────────────────────────────────────────
    float up = N.y * 0.5 + 0.5;  // 0 = down, 1 = up
    vec3 ambient = mix(GND_AMB, SKY_AMB, up);

    // ── Fresnel rim ────────────────────────────────────────────────
    float NdotV = max(0.0, dot(N, V));
    float fresnel = FRESNEL_STRENGTH * pow(1.0 - NdotV, FRESNEL_POWER);
    vec3 rim = fresnel * SKY_AMB;

    // ── Combine ────────────────────────────────────────────────────
    vec3 color = base * (diffuse_key + diffuse_fill + ambient)
               + specular + sun_specular + rim;

    // Apply coastline highlight
    color = mix(color, COAST_COLOR, coast_factor * 0.6);

    // Tone-map (simple Reinhard) to avoid over-bright
    color = color / (color + vec3(1.0));

    frag_color = vec4(color, 1.0);
}
"""


def get_pbr_shader_sources() -> Tuple[str, str]:
    """Return the PBR vertex and fragment shader source strings.

    The PBR shaders support:
    - Normal-mapped lighting (Phase 13E)
    - Water rendering with animated waves, depth-based colour,
      coastline emphasis, and per-vertex water flag (Phase 13H)
    - Texture-aware ocean shader with baked texture preservation,
      Fresnel-based water reflection, and sun specular hotspot
      (Phase 17D)

    Useful for offline compilation checks and tests.

    Returns
    -------
    (vertex_source, fragment_source)
    """
    return _PBR_VERTEX_SHADER, _PBR_FRAGMENT_SHADER


def get_v2_shader_sources() -> Tuple[str, str]:
    """Return the legacy v2 vertex and fragment shader source strings.

    Returns
    -------
    (vertex_source, fragment_source)
    """
    return _V2_VERTEX_SHADER, _V2_FRAGMENT_SHADER


def render_globe_v2(
    payload: Dict[str, Any],
    atlas_path: Union[str, Path],
    uv_layout: Dict[str, Tuple[float, float, float, float]],
    *,
    radius: float = 1.0,
    subdivisions: int = 3,
    uv_inset_px: float = 1.5,
    width: int = 900,
    height: int = 700,
    title: str = "Polygrid Globe v2",
    flood_fill: bool = False,
    flood_fill_iterations: int = 8,
) -> None:
    """Launch an interactive pyglet window with the improved globe renderer.

    Improvements over v1:
    - Tile textures have terrain-coloured backgrounds (no black corners)
    - Atlas uses gutter pixels to prevent bilinear bleed at slot edges
    - Tiles are subdivided and sphere-projected for smooth curvature
    - All geometry in a single VBO + draw call

    Parameters
    ----------
    payload : dict
        Globe export payload.
    atlas_path : Path
        Detail texture atlas.
    uv_layout : dict
        ``{face_id: (u_min, v_min, u_max, v_max)}``.
    radius : float
    subdivisions : int
        Per-triangle subdivision level (3 = good quality, 5 = high).
    uv_inset_px : float
        Inset UV polygon by this many atlas pixels to prevent bilinear
        texture bleeding at tile edges.  Default 0.5.
    width, height : int
    title : str
    flood_fill : bool
        Whether to additionally flood-fill the atlas.  Usually not
        needed with the 13A background-colour fix.
    flood_fill_iterations : int
        Number of dilation passes for flood-fill.
    """
    if not _HAS_MODELS:
        raise ImportError("models library required for globe rendering")

    try:
        import pyglet
        from pyglet import gl
    except ImportError as exc:
        raise ImportError(
            "pyglet required for interactive rendering. "
            "Install with: pip install pyglet"
        ) from exc

    from PIL import Image
    import tempfile

    atlas_path = Path(atlas_path)

    # ── Flood-fill atlas to remove black borders ────────────────────
    if flood_fill:
        filled_path = atlas_path.parent / f"{atlas_path.stem}_filled{atlas_path.suffix}"
        flood_fill_atlas(atlas_path, filled_path, iterations=flood_fill_iterations)
        atlas_path = filled_path

    # ── Build tile colour map from payload ──────────────────────────
    tile_colour_map: Dict[int, Tuple[float, float, float]] = {}
    for tile in payload["tiles"]:
        idx = int(tile["id"][1:])
        tile_colour_map[idx] = tuple(tile["color"][:3])

    meta = payload["metadata"]
    frequency = meta["frequency"]

    # ── Build batched mesh ──────────────────────────────────────────
    # Read atlas dimensions for UV inset calculation
    atlas_img_check = Image.open(str(atlas_path))
    atlas_size = atlas_img_check.size[0]  # square atlas
    atlas_img_check.close()

    print(f"  Building subdivided globe mesh (subdivisions={subdivisions})...")
    vertex_data, index_data = build_batched_globe_mesh(
        frequency, uv_layout,
        tile_colour_map=tile_colour_map,
        radius=radius,
        subdivisions=subdivisions,
        uv_inset_px=uv_inset_px,
        atlas_size=atlas_size,
    )
    n_verts = len(vertex_data)
    n_tris = len(index_data)
    print(f"  → {n_verts:,} vertices, {n_tris:,} triangles")

    # ── Create window ───────────────────────────────────────────────
    config = pyglet.gl.Config(
        double_buffer=True, depth_size=24,
        major_version=3, minor_version=3,
        sample_buffers=1, samples=4,  # MSAA
    )
    try:
        window = pyglet.window.Window(
            width=width, height=height,
            caption=title, resizable=True, config=config,
        )
    except pyglet.window.NoSuchConfigException:
        # Fall back without MSAA
        config = pyglet.gl.Config(
            double_buffer=True, depth_size=24,
            major_version=3, minor_version=3,
        )
        window = pyglet.window.Window(
            width=width, height=height,
            caption=title, resizable=True, config=config,
        )

    # ── Compile shaders ─────────────────────────────────────────────
    def _compile_shader(source, shader_type):
        shader = gl.glCreateShader(shader_type)
        source_bytes = source.encode("utf-8")
        length = ctypes.c_int(len(source_bytes))
        src_buffer = ctypes.create_string_buffer(source_bytes)
        src_ptr = ctypes.cast(src_buffer, ctypes.POINTER(ctypes.c_char))
        src_array = (ctypes.POINTER(ctypes.c_char) * 1)(src_ptr)
        gl.glShaderSource(shader, 1, src_array, ctypes.byref(length))
        gl.glCompileShader(shader)
        status = gl.GLint()
        gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS, ctypes.byref(status))
        if not status.value:
            info_log = ctypes.create_string_buffer(1024)
            gl.glGetShaderInfoLog(shader, 1024, None, info_log)
            raise RuntimeError(f"Shader error: {info_log.value.decode()}")
        return shader

    vs = _compile_shader(_V2_VERTEX_SHADER, gl.GL_VERTEX_SHADER)
    fs = _compile_shader(_V2_FRAGMENT_SHADER, gl.GL_FRAGMENT_SHADER)
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vs)
    gl.glAttachShader(program, fs)
    gl.glLinkProgram(program)
    status = gl.GLint()
    gl.glGetProgramiv(program, gl.GL_LINK_STATUS, ctypes.byref(status))
    if not status.value:
        info_log = ctypes.create_string_buffer(1024)
        gl.glGetProgramInfoLog(program, 1024, None, info_log)
        raise RuntimeError(f"Link error: {info_log.value.decode()}")
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)

    mvp_loc = gl.glGetUniformLocation(program, b"u_mvp")
    model_loc = gl.glGetUniformLocation(program, b"u_model")
    atlas_loc = gl.glGetUniformLocation(program, b"u_atlas")
    use_tex_loc = gl.glGetUniformLocation(program, b"u_use_texture")
    light_loc = gl.glGetUniformLocation(program, b"u_light_dir")

    # ── Upload VBO + IBO ────────────────────────────────────────────
    vao = gl.GLuint()
    gl.glGenVertexArrays(1, ctypes.byref(vao))
    gl.glBindVertexArray(vao)

    vbo = gl.GLuint()
    gl.glGenBuffers(1, ctypes.byref(vbo))
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    vbo_data = vertex_data.astype(np.float32).tobytes()
    gl.glBufferData(gl.GL_ARRAY_BUFFER, len(vbo_data), vbo_data, gl.GL_STATIC_DRAW)

    stride = 8 * 4  # 8 floats × 4 bytes
    # position: location 0, 3 floats, offset 0
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
    # color: location 1, 3 floats, offset 12
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(12))
    # uv: location 2, 2 floats, offset 24
    gl.glEnableVertexAttribArray(2)
    gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(24))

    ibo = gl.GLuint()
    gl.glGenBuffers(1, ctypes.byref(ibo))
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ibo)
    ibo_data = index_data.astype(np.uint32).tobytes()
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, len(ibo_data), ibo_data, gl.GL_STATIC_DRAW)
    n_indices = index_data.size

    gl.glBindVertexArray(0)

    # ── Load atlas texture with mipmaps ─────────────────────────────
    atlas_img = Image.open(str(atlas_path)).convert("RGBA").transpose(
        Image.FLIP_TOP_BOTTOM,
    )
    tex_w, tex_h = atlas_img.size
    atlas_bytes = atlas_img.tobytes()

    tex_id = gl.GLuint()
    gl.glGenTextures(1, ctypes.byref(tex_id))
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, tex_w, tex_h, 0,
        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, atlas_bytes,
    )
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    # ── Camera state ────────────────────────────────────────────────
    yaw = [0.0]
    pitch = [0.0]
    zoom = [3.0]

    def _perspective(fovy, aspect, near, far):
        f = 1.0 / math.tan(fovy / 2.0)
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0],
        ], dtype=np.float32)

    def _look_at(eye, target):
        eye_v = np.array(eye, dtype=np.float32)
        target_v = np.array(target, dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        forward = target_v - eye_v
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        view = np.identity(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -eye_v @ view[:3, :3]
        return view

    def _rotation_y(angle):
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [ c, 0, s, 0],
            [ 0, 1, 0, 0],
            [-s, 0, c, 0],
            [ 0, 0, 0, 1],
        ], dtype=np.float32)

    def _rotation_x(angle):
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [1,  0,  0, 0],
            [0,  c, -s, 0],
            [0,  s,  c, 0],
            [0,  0,  0, 1],
        ], dtype=np.float32)

    # Normalised light direction
    light_dir = _normalize_vec3(np.array([0.3, 0.8, 0.5], dtype=np.float32))

    @window.event
    def on_draw():
        window.clear()
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.05, 0.05, 0.08, 1.0)
        gl.glClear(int(gl.GL_COLOR_BUFFER_BIT) | int(gl.GL_DEPTH_BUFFER_BIT))

        aspect = window.width / max(1, window.height)
        proj = _perspective(math.radians(45), aspect, 0.1, 100.0)
        eye = np.array([0.0, 0.0, zoom[0]], dtype=np.float32)
        view = _look_at(eye, (0.0, 0.0, 0.0))
        mvp = (proj @ view).astype(np.float32)
        model = (_rotation_y(yaw[0]) @ _rotation_x(pitch[0])).astype(np.float32)

        gl.glUseProgram(program)
        gl.glUniformMatrix4fv(
            mvp_loc, 1, gl.GL_TRUE,
            mvp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        gl.glUniformMatrix4fv(
            model_loc, 1, gl.GL_TRUE,
            model.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        gl.glUniform3f(light_loc, light_dir[0], light_dir[1], light_dir[2])

        # Bind atlas
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        gl.glUniform1i(atlas_loc, 0)
        gl.glUniform1i(use_tex_loc, 1)

        # Single draw call for entire globe
        gl.glBindVertexArray(vao)
        gl.glDrawElements(gl.GL_TRIANGLES, n_indices, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glUseProgram(0)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        yaw[0] += dx * 0.01
        pitch[0] += dy * 0.01
        pitch[0] = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, pitch[0]))

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        zoom[0] = max(1.5, min(10.0, zoom[0] - scroll_y * 0.3))

    pyglet.app.run()
