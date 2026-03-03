"""Apron-aware texture rendering — Phase 18B.

Renders tile textures using apron-extended grids so that the
gutter/bleed zone around each tile contains real terrain data
from neighbouring tiles.  This eliminates visible seams on the
3D globe.

Functions
---------
- :func:`render_detail_texture_apron` — render an apron grid to a PIL image
- :func:`build_apron_atlas` — full atlas pipeline using apron grids
- :func:`build_apron_feature_atlas` — atlas with biome overlays + apron gutters
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .apron_grid import (
    ApronResult,
    build_all_apron_grids,
)
from .detail_render import BiomeConfig, detail_elevation_to_colour, _detail_hillshade
from .geometry import face_center
from .polygrid import PolyGrid
from .tile_data import TileDataStore
from .tile_detail import DetailGridCollection


# Sentinel colour for background pixels (matches tile_texture.py).
_BG_SENTINEL = (255, 0, 255)


# ═══════════════════════════════════════════════════════════════════
# 18B.1 — Render apron texture
# ═══════════════════════════════════════════════════════════════════

def render_detail_texture_apron(
    apron_grid: PolyGrid,
    store: TileDataStore,
    biome: Optional[BiomeConfig] = None,
    *,
    tile_size: int = 256,
    elevation_field: str = "elevation",
    noise_seed: int = 0,
    overscan: float = 0.15,
    vertex_jitter: float = 1.5,
    noise_overlay: bool = True,
    noise_frequency: float = 0.05,
    noise_amplitude: float = 0.05,
    colour_dither: bool = True,
    dither_radius: float = 6.0,
    k_neighbours: int = 4,
) -> "Image.Image":
    """Render an apron-extended detail grid to a PIL Image.

    This is the apron-aware counterpart of
    :func:`tile_texture.render_detail_texture_fullslot`.  It renders
    **all** sub-faces in the apron grid (own + neighbour apron) so
    that the resulting image has correct terrain in the gutter zone
    surrounding the tile's polygon footprint.

    Parameters
    ----------
    apron_grid : PolyGrid
        Extended grid from :func:`build_apron_grid` (includes apron
        sub-faces from neighbours).
    store : TileDataStore
        Must have elevation for every face in *apron_grid*.
    biome : BiomeConfig, optional
        Colour ramp / hillshade config.
    tile_size : int
        Output image size (square).
    elevation_field : str
    noise_seed : int
    overscan : float
        Extra padding beyond the original tile's bounding box.
    vertex_jitter : float
        16D.1 edge dissolution amount in pixels.
    noise_overlay : bool
        16D.2 pixel noise layer.
    noise_frequency, noise_amplitude : float
        Parameters for the noise overlay.
    colour_dither : bool
        16D.3 sub-face colour dithering.
    dither_radius : float
    k_neighbours : int
        IDW neighbour count for background fill.

    Returns
    -------
    PIL.Image.Image
        RGB image of size (tile_size, tile_size).
    """
    from PIL import Image, ImageDraw
    from scipy.spatial import KDTree
    from .tile_texture import (
        jitter_polygon_vertices,
        apply_noise_overlay,
        apply_colour_dithering,
    )

    if biome is None:
        biome = BiomeConfig()

    # ── Hillshade for the full apron grid ───────────────────────
    hs_dict = _detail_hillshade(
        apron_grid, store, elevation_field,
        azimuth=biome.azimuth, altitude=biome.altitude,
    )

    # ── Bounding box from ALL vertices (own + apron) ────────────
    xs, ys = [], []
    for v in apron_grid.vertices.values():
        if v.has_position():
            xs.append(v.x)
            ys.append(v.y)

    if not xs:
        return Image.new("RGB", (tile_size, tile_size), (0, 0, 0))

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

    def _to_pixel(vx: float, vy: float) -> Tuple[float, float]:
        px = (vx - x_min) * scale + ox
        py = tile_size - ((vy - y_min) * scale + oy)
        return (px, py)

    # ── Step 1: Compute face colours + rasterise polygons ───────
    face_colours: Dict[str, Tuple[float, float, float]] = {}
    face_pixel_colours: Dict[str, Tuple[int, int, int]] = {}
    centroid_pixels: List[Tuple[float, float]] = []
    colour_array: List[Tuple[int, int, int]] = []
    face_id_order: List[str] = []

    for fid, face in apron_grid.faces.items():
        has_verts = True
        for vid in face.vertex_ids:
            v = apron_grid.vertices.get(vid)
            if v is None or not v.has_position():
                has_verts = False
                break
        if has_verts and len(face.vertex_ids) >= 3:
            elev = store.get(fid, elevation_field)
            c = face_center(apron_grid.vertices, face)
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

            cpx, cpy = _to_pixel(cx, cy)
            centroid_pixels.append((cpx, cpy))
            colour_array.append(pc)
            face_id_order.append(fid)

    if not face_colours:
        return Image.new("RGB", (tile_size, tile_size), (0, 0, 0))

    # Draw polygon fills on sentinel background
    img = Image.new("RGB", (tile_size, tile_size), _BG_SENTINEL)
    draw = ImageDraw.Draw(img)

    for fid, face in apron_grid.faces.items():
        verts = []
        for vid in face.vertex_ids:
            v = apron_grid.vertices.get(vid)
            if v is None or not v.has_position():
                break
            verts.append(_to_pixel(v.x, v.y))
        else:
            if len(verts) >= 3 and fid in face_pixel_colours:
                if vertex_jitter > 0:
                    verts = jitter_polygon_vertices(
                        verts,
                        max_jitter=vertex_jitter,
                        seed=noise_seed + hash(fid) % 100_000,
                    )
                colour = face_pixel_colours[fid]
                draw.polygon(verts, fill=colour, outline=colour)

    pixels = np.array(img)  # (H, W, 3) uint8

    # ── Step 1b: Colour dithering (16D.3) ───────────────────────
    if colour_dither and len(centroid_pixels) >= 2:
        centroid_arr_d = np.array(centroid_pixels, dtype=np.float64)
        colour_arr_d = np.array(colour_array, dtype=np.float64)
        pixels = apply_colour_dithering(
            pixels, centroid_arr_d, colour_arr_d,
            blend_radius=dither_radius,
            k_neighbours=min(k_neighbours, len(centroid_arr_d)),
        )

    # ── Step 2: IDW fill for remaining background pixels ────────
    bg_mask = (
        (pixels[:, :, 0] == _BG_SENTINEL[0]) &
        (pixels[:, :, 1] == _BG_SENTINEL[1]) &
        (pixels[:, :, 2] == _BG_SENTINEL[2])
    )
    bg_count = bg_mask.sum()
    if bg_count > 0 and len(centroid_pixels) > 0:
        centroid_arr = np.array(centroid_pixels, dtype=np.float64)
        colour_arr = np.array(colour_array, dtype=np.float64)
        tree = KDTree(centroid_arr)

        bg_rows, bg_cols = np.where(bg_mask)
        bg_points = np.column_stack([
            bg_cols.astype(np.float64),
            bg_rows.astype(np.float64),
        ])

        k_actual = min(k_neighbours, len(centroid_arr))
        dists, idxs = tree.query(bg_points, k=k_actual, workers=-1)

        if k_actual == 1:
            dists = dists[:, np.newaxis]
            idxs = idxs[:, np.newaxis]

        eps = 1e-12
        weights = 1.0 / np.maximum(dists, eps)
        w_sum = weights.sum(axis=1, keepdims=True)
        norm_w = weights / w_sum

        neighbour_colours = colour_arr[idxs]
        blended = (norm_w[:, :, np.newaxis] * neighbour_colours).sum(axis=1)

        blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
        pixels[bg_rows, bg_cols] = blended_uint8

    # ── Step 3: Noise overlay (16D.2) ───────────────────────────
    if noise_overlay:
        pixels = apply_noise_overlay(
            pixels,
            frequency=noise_frequency,
            amplitude=noise_amplitude,
            seed=noise_seed + 7777,
        )

    return Image.fromarray(pixels, "RGB")


# ═══════════════════════════════════════════════════════════════════
# 18B.4 — Atlas builder with apron-filled gutters
# ═══════════════════════════════════════════════════════════════════

def build_apron_atlas(
    collection: DetailGridCollection,
    globe_grid: PolyGrid,
    biome: Optional[BiomeConfig] = None,
    output_dir: Path | str = Path("exports/detail_tiles"),
    *,
    tile_size: int = 256,
    columns: int = 0,
    noise_seed: int = 0,
    gutter: int = 4,
    smooth_iterations: int = 2,
) -> Tuple[Path, Dict[str, Tuple[float, float, float, float]]]:
    """Build a texture atlas using apron-extended grids.

    Each tile is rendered with its apron sub-faces so the gutter zone
    contains real terrain data from neighbouring tiles.  This replaces
    the edge-clamping approach and eliminates visible seams.

    Parameters
    ----------
    collection : DetailGridCollection
        Must have stores populated (terrain generated).
    globe_grid : PolyGrid
        Globe grid with adjacency.
    biome : BiomeConfig, optional
    output_dir : Path or str
    tile_size : int
        Tile texture side length.
    columns : int
        Atlas columns (0 = auto).
    noise_seed : int
    gutter : int
        Gutter pixels around each slot.  These are now filled with
        actual terrain data from apron rendering.
    smooth_iterations : int
        Apron join-zone smoothing passes.

    Returns
    -------
    (atlas_path, uv_layout)
        ``uv_layout`` maps ``face_id → (u_min, v_min, u_max, v_max)``.
    """
    from PIL import Image

    if biome is None:
        biome = BiomeConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    face_ids = collection.face_ids
    n = len(face_ids)
    if n == 0:
        raise ValueError("No detail grids in the collection")

    # ── Build apron grids for all tiles ─────────────────────────
    apron_results = build_all_apron_grids(
        globe_grid, collection,
        smooth_iterations=smooth_iterations,
    )

    # ── Render each tile using its apron grid ───────────────────
    tile_images: Dict[str, Image.Image] = {}
    for fid in face_ids:
        ar = apron_results[fid]
        tile_img = render_detail_texture_apron(
            ar.grid, ar.store,
            biome,
            tile_size=tile_size,
            noise_seed=noise_seed + hash(fid) % 10000,
        )
        tile_images[fid] = tile_img

        # Also save individual tile
        tile_path = output_dir / f"tile_{fid}.png"
        tile_img.save(str(tile_path))

    # ── Assemble atlas ──────────────────────────────────────────
    if columns <= 0:
        columns = max(1, math.isqrt(n))
        if columns * columns < n:
            columns += 1
    rows = math.ceil(n / columns)

    slot_size = tile_size + 2 * gutter
    atlas_w = columns * slot_size
    atlas_h = rows * slot_size
    atlas = Image.new("RGB", (atlas_w, atlas_h), (128, 128, 128))

    uv_layout: Dict[str, Tuple[float, float, float, float]] = {}

    for idx, fid in enumerate(face_ids):
        col = idx % columns
        row = idx // columns
        slot_x = col * slot_size
        slot_y = row * slot_size

        tile_img = tile_images[fid].resize(
            (tile_size, tile_size), Image.LANCZOS,
        )

        # Paste tile into centre of slot
        atlas.paste(tile_img, (slot_x + gutter, slot_y + gutter))

        # Fill gutter from apron data — since the apron-rendered
        # image already has terrain in the border zone, we use the
        # edge pixels which now contain real neighbour terrain
        # rather than clamped tile-interior pixels.
        if gutter > 0:
            _fill_apron_gutter(atlas, slot_x, slot_y, tile_size, gutter)

        inner_x = slot_x + gutter
        inner_y = slot_y + gutter
        u_min = inner_x / atlas_w
        u_max = (inner_x + tile_size) / atlas_w
        v_min = 1.0 - (inner_y + tile_size) / atlas_h
        v_max = 1.0 - inner_y / atlas_h
        uv_layout[fid] = (u_min, v_min, u_max, v_max)

    atlas_path = output_dir / "detail_atlas.png"
    atlas.save(str(atlas_path))

    return atlas_path, uv_layout


def _fill_apron_gutter(
    atlas: "Image.Image",
    slot_x: int,
    slot_y: int,
    tile_size: int,
    gutter: int,
) -> None:
    """Fill gutter pixels from the apron-rendered tile's edge pixels.

    Unlike the original ``_fill_gutter()`` which clamps a single row
    outward (creating visible banding), this version extrapolates from
    the edge pixels that already contain real terrain from the apron
    rendering.  The visual result is the same approach — clamp edge
    pixels — but the *content* of those edge pixels is now correct
    apron terrain rather than tile-interior terrain.
    """
    inner_x = slot_x + gutter
    inner_y = slot_y + gutter

    # Top gutter — repeat top row (which now has apron terrain)
    top_strip = atlas.crop((inner_x, inner_y, inner_x + tile_size, inner_y + 1))
    for g in range(gutter):
        atlas.paste(top_strip, (inner_x, slot_y + g))

    # Bottom gutter
    bot_y = inner_y + tile_size - 1
    bot_strip = atlas.crop((inner_x, bot_y, inner_x + tile_size, bot_y + 1))
    for g in range(gutter):
        atlas.paste(bot_strip, (inner_x, inner_y + tile_size + g))

    # Left gutter (full height including top/bottom gutter)
    full_top = slot_y
    full_bot = slot_y + tile_size + 2 * gutter
    left_strip = atlas.crop((inner_x, full_top, inner_x + 1, full_bot))
    for g in range(gutter):
        atlas.paste(left_strip, (slot_x + g, full_top))

    # Right gutter
    right_x = inner_x + tile_size - 1
    right_strip = atlas.crop((right_x, full_top, right_x + 1, full_bot))
    for g in range(gutter):
        atlas.paste(right_strip, (inner_x + tile_size + g, full_top))


# ═══════════════════════════════════════════════════════════════════
# 18B.3 — Feature atlas with apron gutters
# ═══════════════════════════════════════════════════════════════════

def build_apron_feature_atlas(
    collection: DetailGridCollection,
    globe_grid: PolyGrid,
    *,
    biome_renderers: Optional[Dict[str, Any]] = None,
    density_map: Optional[Dict[str, float]] = None,
    biome_type_map: Optional[Dict[str, str]] = None,
    biome_config: Optional[BiomeConfig] = None,
    output_dir: Path | str = Path("exports/detail_tiles"),
    tile_size: int = 256,
    columns: int = 0,
    noise_seed: int = 0,
    gutter: int = 4,
    smooth_iterations: int = 2,
) -> Tuple[Path, Dict[str, Tuple[float, float, float, float]]]:
    """Build a texture atlas with biome features + apron gutters.

    Combines apron-aware ground rendering (18B.1) with biome feature
    overlays (forest, ocean) and fills gutters from apron data (18B.4).

    Parameters
    ----------
    collection : DetailGridCollection
        Must have stores populated.
    globe_grid : PolyGrid
        Globe grid with adjacency.
    biome_renderers : dict, optional
        ``{"forest": ForestRenderer(), "ocean": OceanRenderer(), ...}``.
    density_map : dict, optional
        ``{face_id: float}`` biome density per tile.
    biome_type_map : dict, optional
        ``{face_id: "forest"|"ocean"|...}`` routing to renderer.
    biome_config : BiomeConfig, optional
    output_dir : Path or str
    tile_size : int
    columns : int
    noise_seed : int
    gutter : int
    smooth_iterations : int

    Returns
    -------
    (atlas_path, uv_layout)
    """
    from PIL import Image
    from .geometry import face_center_3d

    if biome_config is None:
        biome_config = BiomeConfig()
    if density_map is None:
        density_map = {}
    if biome_renderers is None:
        biome_renderers = {}
    if biome_type_map is None:
        biome_type_map = {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    face_ids = collection.face_ids
    n = len(face_ids)
    if n == 0:
        raise ValueError("No detail grids in the collection")

    # Default renderer lookup
    default_renderer = (
        next(iter(biome_renderers.values())) if biome_renderers else None
    )

    def _get_renderer(fid: str):
        biome_key = biome_type_map.get(fid)
        if biome_key is not None and biome_key in biome_renderers:
            return biome_renderers[biome_key]
        return default_renderer

    # ── Build apron grids ───────────────────────────────────────
    apron_results = build_all_apron_grids(
        globe_grid, collection,
        smooth_iterations=smooth_iterations,
    )

    # ── Render ground + apply biome overlays ────────────────────
    tile_images: Dict[str, Image.Image] = {}

    for fid in face_ids:
        ar = apron_results[fid]

        # Ground texture from apron grid
        ground_img = render_detail_texture_apron(
            ar.grid, ar.store,
            biome_config,
            tile_size=tile_size,
            noise_seed=noise_seed + hash(fid) % 10000,
        )

        # Apply biome overlay
        tile_density = density_map.get(fid, 0.0)
        tile_renderer = _get_renderer(fid)

        if tile_density > 0.01 and tile_renderer is not None:
            center_3d = None
            face = globe_grid.faces.get(fid)
            if face is not None:
                center_3d = face_center_3d(globe_grid.vertices, face)

            # Set grid context for topology-aware renderers (18C)
            if hasattr(tile_renderer, "set_grid_context"):
                tile_renderer.set_grid_context(ar.grid, ar.store)

            ground_img = tile_renderer.render(
                ground_img,
                fid,
                tile_density,
                seed=noise_seed + hash(fid) % 100_000,
                globe_3d_center=center_3d,
            )

        tile_images[fid] = ground_img

        # Save individual tile
        tile_path = output_dir / f"tile_{fid}.png"
        ground_img.convert("RGB").save(str(tile_path))

    # ── Assemble atlas ──────────────────────────────────────────
    if columns <= 0:
        columns = max(1, math.isqrt(n))
        if columns * columns < n:
            columns += 1
    rows = math.ceil(n / columns)

    slot_size = tile_size + 2 * gutter
    atlas_w = columns * slot_size
    atlas_h = rows * slot_size
    atlas = Image.new("RGB", (atlas_w, atlas_h), (128, 128, 128))

    uv_layout: Dict[str, Tuple[float, float, float, float]] = {}

    for idx, fid in enumerate(face_ids):
        col = idx % columns
        row = idx // columns
        slot_x = col * slot_size
        slot_y = row * slot_size

        tile_img = tile_images[fid].convert("RGB").resize(
            (tile_size, tile_size), Image.LANCZOS,
        )

        atlas.paste(tile_img, (slot_x + gutter, slot_y + gutter))

        if gutter > 0:
            _fill_apron_gutter(atlas, slot_x, slot_y, tile_size, gutter)

        inner_x = slot_x + gutter
        inner_y = slot_y + gutter
        u_min = inner_x / atlas_w
        u_max = (inner_x + tile_size) / atlas_w
        v_min = 1.0 - (inner_y + tile_size) / atlas_h
        v_max = 1.0 - inner_y / atlas_h
        uv_layout[fid] = (u_min, v_min, u_max, v_max)

    atlas_path = output_dir / "detail_atlas.png"
    atlas.save(str(atlas_path))

    return atlas_path, uv_layout
