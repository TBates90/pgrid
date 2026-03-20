#!/usr/bin/env python3
"""Render per-tile polygrid PNGs for every Goldberg tile on a globe.

By default, produces polygon-cut UV-aligned tile textures with a packed
texture atlas ready for 3D globe rendering. All metadata needed by
``render_globe_from_tiles.py`` is exported alongside the tiles.

Usage
-----
::

    # Basic (frequency 3, detail_rings 4, polygon-cut atlas):
    python scripts/render_polygrids.py

    # Higher resolution:
    python scripts/render_polygrids.py -f 4 --detail-rings 6

    # Show grid edges overlaid on terrain:
    python scripts/render_polygrids.py --edges

    # Disable polygon-cut (plain stitched tiles):
    python scripts/render_polygrids.py --no-polygon-cut --stitched

    # Custom output directory:
    python scripts/render_polygrids.py -o exports/my_polygrids

    # Different terrain preset:
    python scripts/render_polygrids.py --preset alpine_peaks

Outputs one PNG per tile plus ``atlas.png``, ``uv_layout.json``,
``globe_payload.json``, and ``metadata.json`` to the output directory.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Sentinel background colour used by stitched-tile renderers.
# Any image pixels with this colour were NOT covered by polygon patches
# and must be replaced (nearest-neighbour fill) before atlas warping.
# Bright magenta is chosen because it never appears in terrain or
# colour-debug renders.
_SENTINEL_BG: tuple[float, float, float] = (1.0, 0.0, 1.0)
_SENTINEL_BG_RGB: tuple[int, int, int] = (255, 0, 255)


def _build_globe_and_terrain(frequency: int, preset: str, seed: int):
    """Build globe grid with mountain terrain."""
    from polygrid.globe import build_globe_grid
    from polygrid.mountains import MountainConfig, generate_mountains
    from polygrid.tile_data import FieldDef, TileDataStore, TileSchema

    print(f"Building globe (freq={frequency}, preset={preset}, seed={seed})...")
    grid = build_globe_grid(frequency)
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)

    presets = {
        "mountain_range": MountainConfig(
            seed=seed, ridge_frequency=2.0, ridge_octaves=4,
            peak_elevation=1.0, base_elevation=0.0,
        ),
        "alpine_peaks": MountainConfig(
            seed=seed, ridge_frequency=3.0, ridge_octaves=5,
            peak_elevation=1.0, base_elevation=0.1,
        ),
        "rolling_hills": MountainConfig(
            seed=seed, ridge_frequency=1.5, ridge_octaves=3,
            peak_elevation=0.5, base_elevation=0.2,
        ),
    }

    config = presets.get(preset, presets["mountain_range"])
    generate_mountains(grid, store, config)
    print(f"  → {len(grid.faces)} tiles")
    return grid, store


def _build_detail_grids(grid, store, detail_rings: int, seed: int):
    """Build detail grids and generate boundary-aware terrain."""
    from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
    from polygrid.detail_terrain import generate_all_detail_terrain

    spec = TileDetailSpec(detail_rings=detail_rings)

    t0 = time.perf_counter()
    print(f"Building detail grids (rings={detail_rings})...")
    coll = DetailGridCollection.build(grid, spec)
    print(f"  → {coll.total_face_count} sub-faces in "
          f"{time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    print("Generating detail terrain (boundary-aware)...")
    generate_all_detail_terrain(coll, grid, store, spec, seed=seed)
    print(f"  → done in {time.perf_counter() - t0:.2f}s")

    return coll


def _build_stitched_store(composite, coll, face_id, globe_grid):
    """Build a TileDataStore for a stitched CompositeGrid.

    Maps elevation data from each component's detail store into the
    merged grid using the composite's id-prefix mapping.
    """
    from polygrid.tile_data import FieldDef, TileDataStore, TileSchema

    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=composite.merged, schema=schema)

    for comp_name, prefix in composite.id_prefixes.items():
        _, comp_store = coll.get(comp_name)
        if comp_store is None:
            continue
        for fid in composite.components[comp_name].faces:
            prefixed_fid = f"{prefix}{fid}"
            if prefixed_fid in composite.merged.faces:
                store.set(prefixed_fid, "elevation",
                          comp_store.get(fid, "elevation"))

    return store


# ---------------------------------------------------------------------------
# Global hillshade pre-computation
# ---------------------------------------------------------------------------

def _compute_global_hillshade(coll, grid, face_ids, biome):
    """Pre-compute hillshade for every detail face across all tiles.

    For each globe tile, builds the composite (centre + neighbours),
    computes hillshade on the full composite grid, and records the
    values for the **centre tile's** faces only.  This means every
    face gets its hillshade computed with the maximum available
    neighbour context, eliminating boundary truncation artefacts.

    Returns a dict ``{prefixed_face_id: float}`` keyed by the
    merged-grid face IDs used by each composite.  Since the prefix
    changes per composite, the caller must look up values using the
    face IDs from the composite they are rendering.

    We return one dict per globe tile:
    ``{globe_face_id: {merged_face_id: hillshade_value}}``.
    The centre-tile faces have authoritative values; neighbour faces
    get their values from the composite where *they* are the centre.
    """
    from polygrid.detail_render import _detail_hillshade
    from polygrid.tile_detail import build_tile_with_neighbours

    print("Pre-computing global hillshade...")
    t0 = time.perf_counter()

    # Step 1: for each tile, compute hillshade on its composite and
    # store the ORIGINAL (un-prefixed) face_id → hillshade value.
    # This gives every detail face its hillshade from the composite
    # where it was the centre tile (best neighbour context).
    authoritative_hs: dict[str, float] = {}

    for fid in face_ids:
        composite = build_tile_with_neighbours(coll, fid, grid)
        stitched_store = _build_stitched_store(composite, coll, fid, grid)
        mg = composite.merged

        hs = _detail_hillshade(
            mg, stitched_store, "elevation",
            azimuth=biome.azimuth, altitude=biome.altitude,
        )

        # Keep only the centre tile's faces — these have full context.
        centre_prefix = composite.id_prefixes[fid]
        for merged_fid, val in hs.items():
            if merged_fid.startswith(centre_prefix):
                # Strip prefix to get original detail face id
                orig_fid = merged_fid[len(centre_prefix):]
                # Key by (globe_tile, detail_face) to be unique
                authoritative_hs[(fid, orig_fid)] = val

    elapsed = time.perf_counter() - t0
    print(f"  → global hillshade for {len(authoritative_hs)} faces "
          f"in {elapsed:.2f}s")
    return authoritative_hs


def _resolve_hillshade_for_composite(
    composite,
    face_id: str,
    global_hs: dict[tuple[str, str], float],
) -> dict[str, float]:
    """Build a hillshade dict for *composite*'s merged grid face IDs.

    For each face in the merged grid, look up its authoritative
    hillshade value from *global_hs* (keyed by globe tile + original
    detail face id).  Falls back to 0.5 for any face not found.
    """
    result: dict[str, float] = {}
    for comp_name, prefix in composite.id_prefixes.items():
        for orig_fid in composite.components[comp_name].faces:
            merged_fid = f"{prefix}{orig_fid}"
            # Look up the authoritative value (from the composite where
            # this globe tile was the centre)
            val = global_hs.get((comp_name, orig_fid), 0.5)
            result[merged_fid] = val
    return result


# ---------------------------------------------------------------------------
# Shared rendering helpers
# ---------------------------------------------------------------------------

from typing import Callable

# Colour callback type: (face_id, grid, face) -> (r, g, b) or None
ColourFunc = Callable[..., tuple[float, float, float] | None]


def _build_face_patches(grid, colour_fn: ColourFunc):
    """Iterate *grid* faces, build matplotlib patches + per-face colours.

    *colour_fn(face_id, grid, face)* returns an (r, g, b) tuple or
    ``None`` to skip a face.  Returns ``(patches, colours)`` lists.
    """
    from matplotlib.patches import Polygon as MplPolygon

    patches: list = []
    colours: list[tuple[float, float, float]] = []

    for fid, face in grid.faces.items():
        verts = []
        for vid in face.vertex_ids:
            v = grid.vertices.get(vid)
            if v is None or not v.has_position():
                break
            verts.append((v.x, v.y))
        else:
            if len(verts) < 3:
                continue
            colour = colour_fn(fid, grid, face)
            if colour is not None:
                patches.append(MplPolygon(verts, closed=True))
                colours.append(colour)

    return patches, colours


def _render_patches_to_png(
    patches,
    colours,
    output_path: Path,
    *,
    tile_size: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    edge_colour: str = "none",
    edge_lw: float = 0,
    bg_colour: tuple[float, float, float] = (0.12, 0.12, 0.12),
    extra_layers: list | None = None,
):
    """Render *patches* to a square PNG with consistent matplotlib setup.

    *extra_layers* is an optional list of ``PatchCollection`` objects to
    draw *before* the main patches (e.g. a neighbour-grid background).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection

    dpi = 100
    fig_size = tile_size / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for layer in (extra_layers or []):
        ax.add_collection(layer)

    pc = PatchCollection(
        patches, facecolors=colours,
        edgecolors=edge_colour, linewidths=edge_lw,
    )
    ax.add_collection(pc)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    fig.savefig(
        str(output_path), dpi=dpi, bbox_inches="tight", pad_inches=0,
        facecolor=bg_colour,
    )
    plt.close(fig)


def _render_analytical_to_png(
    grid,
    colour_fn,
    output_path: Path,
    *,
    tile_size: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    bg_colour: tuple[float, float, float] = (0.12, 0.12, 0.12),
):
    """Render face polygons to PNG via analytical (point-in-polygon) fill.

    Unlike :func:`_render_patches_to_png` this bypasses matplotlib
    rasterisation entirely.  Each pixel is assigned to exactly one face
    via vectorised ``MplPath.contains_points`` checks — no anti-aliasing,
    no sub-pixel jitter.  Two tiles sharing a face always produce the
    same RGB value for every pixel inside that face (given the same colour
    input), eliminating rasterisation seams.

    Parameters
    ----------
    grid : PolyGrid
        The merged grid whose faces will be rendered.
    colour_fn : callable
        ``colour_fn(fid, grid, face)`` → ``(r, g, b)`` floats in [0, 1],
        or ``None`` to skip a face.  Same signature as
        :func:`_build_face_patches`.
    output_path : Path
        Destination PNG path.
    tile_size : int
        Output image width/height in pixels.
    xlim, ylim : tuple[float, float]
        View limits in grid coordinates.
    bg_colour : tuple[float, float, float]
        Background colour for uncovered pixels (float [0, 1]).
    """
    import numpy as np
    from matplotlib.path import Path as MplPath
    from PIL import Image

    bg_rgb = np.array(
        [int(bg_colour[0] * 255), int(bg_colour[1] * 255),
         int(bg_colour[2] * 255)],
        dtype=np.uint8,
    )
    img = np.full((tile_size, tile_size, 3), bg_rgb, dtype=np.uint8)

    # Pre-compute pixel coordinate grids
    xs = np.linspace(xlim[0], xlim[1], tile_size)
    ys = np.linspace(ylim[1], ylim[0], tile_size)  # flip y (row 0 = top)
    px, py = np.meshgrid(xs, ys)

    for fid, face in grid.faces.items():
        colour = colour_fn(fid, grid, face)
        if colour is None:
            continue

        # Build vertex polygon
        verts = []
        for vid in face.vertex_ids:
            v = grid.vertices.get(vid)
            if v is None or not v.has_position():
                break
            verts.append((v.x, v.y))
        else:
            if len(verts) < 3:
                continue
        if len(verts) < 3:
            continue

        verts_closed = verts + [verts[0]]
        path = MplPath(verts_closed)

        # Bounding-box pre-filter for speed
        vx = [p[0] for p in verts]
        vy = [p[1] for p in verts]
        col_mask = (xs >= min(vx)) & (xs <= max(vx))
        row_mask = (ys >= min(vy)) & (ys <= max(vy))
        if not col_mask.any() or not row_mask.any():
            continue

        ci = np.where(col_mask)[0]
        ri = np.where(row_mask)[0]
        sub_px = px[np.ix_(ri, ci)].ravel()
        sub_py = py[np.ix_(ri, ci)].ravel()
        sub_pts = np.column_stack([sub_px, sub_py])
        mask = path.contains_points(sub_pts)

        rows_idx, cols_idx = np.meshgrid(ri, ci, indexing="ij")
        rows_flat = rows_idx.ravel()[mask]
        cols_flat = cols_idx.ravel()[mask]

        rgb = np.array(
            [int(colour[0] * 255), int(colour[1] * 255),
             int(colour[2] * 255)],
            dtype=np.uint8,
        )
        img[rows_flat, cols_flat] = rgb

    Image.fromarray(img).save(str(output_path))


def _avg_colour(
    colours: list[tuple[float, float, float]],
    fallback: tuple[float, float, float] = (0.15, 0.15, 0.15),
) -> tuple[float, float, float]:
    """Return the channel-wise mean of *colours*, or *fallback*."""
    if not colours:
        return fallback
    n = len(colours)
    return (
        sum(c[0] for c in colours) / n,
        sum(c[1] for c in colours) / n,
        sum(c[2] for c in colours) / n,
    )


def _grid_bbox(grid, pad: float = 1.15):
    """Return (xlim, ylim) covering all vertices in *grid*.

    *pad* is a multiplier applied around the midpoint (1.15 = 15% margin).
    """
    xs, ys = [], []
    for v in grid.vertices.values():
        if v.has_position():
            xs.append(v.x)
            ys.append(v.y)
    if not xs:
        return (0.0, 1.0), (0.0, 1.0)
    half = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.5 * pad
    mx = (min(xs) + max(xs)) * 0.5
    my = (min(ys) + max(ys)) * 0.5
    return (mx - half, mx + half), (my - half, my + half)


def _edge_style(show: bool, colour: str = "#00000040", lw: float = 0.5):
    """Return (edge_colour, edge_lw) for optional edge outlines."""
    return (colour, lw) if show else ("none", 0)


# ---------------------------------------------------------------------------
# Colour-debug helpers
# ---------------------------------------------------------------------------

def _compute_component_gradient_info(composite):
    """Pre-compute centroid + max-distance for each component.

    Returns ``(comp_centroids, comp_max_dist)`` dicts keyed by
    component name.
    """
    from polygrid.geometry import face_center as _face_center

    mg = composite.merged
    comp_centroids: dict[str, tuple[float, float]] = {}
    comp_max_dist: dict[str, float] = {}

    for name, prefix in composite.id_prefixes.items():
        xs, ys = [], []
        for fid, face in mg.faces.items():
            if not fid.startswith(prefix):
                continue
            c = _face_center(mg.vertices, face)
            if c is not None:
                xs.append(c[0])
                ys.append(c[1])
        if xs:
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            comp_centroids[name] = (cx, cy)
            max_d = max(
                ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                for x, y in zip(xs, ys)
            )
            comp_max_dist[name] = max_d if max_d > 1e-9 else 1.0
        else:
            comp_centroids[name] = (0.0, 0.0)
            comp_max_dist[name] = 1.0

    return comp_centroids, comp_max_dist


def _colour_debug_fn(
    composite, tile_hues, comp_centroids, comp_max_dist,
):
    """Return a colour callback for colour-debug rendering."""
    import colorsys
    from polygrid.geometry import face_center as _face_center

    comp_hues = {
        name: tile_hues.get(name, 0.0)
        for name in composite.id_prefixes
    }

    def _colour(fid, grid, face):
        comp_name = None
        for name, prefix in composite.id_prefixes.items():
            if fid.startswith(prefix):
                comp_name = name
                break
        if comp_name is None:
            return None
        hue = comp_hues[comp_name]
        cx, cy = comp_centroids[comp_name]
        c = _face_center(grid.vertices, face)
        fx, fy = c if c else (cx, cy)
        dist = ((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5
        t = min(dist / comp_max_dist[comp_name], 1.0)
        lightness = 0.72 - 0.20 * t
        saturation = 0.65 + 0.15 * t
        return colorsys.hls_to_rgb(hue, lightness, saturation)

    return _colour


def _colour_debug_single_fn(grid, hue: float):
    """Return a colour callback for a standalone colour-debug tile."""
    import colorsys
    from polygrid.geometry import face_center as _face_center

    xs, ys = [], []
    for fid, face in grid.faces.items():
        c = _face_center(grid.vertices, face)
        if c is not None:
            xs.append(c[0])
            ys.append(c[1])
    if not xs:
        return lambda *_: None
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    max_d = max(
        (((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 for x, y in zip(xs, ys)),
        default=1.0,
    )
    if max_d < 1e-9:
        max_d = 1.0

    def _colour(fid, grid, face):
        c = _face_center(grid.vertices, face)
        fx, fy = c if c else (cx, cy)
        t = min(((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5 / max_d, 1.0)
        lightness = 0.72 - 0.20 * t
        saturation = 0.65 + 0.15 * t
        return colorsys.hls_to_rgb(hue, lightness, saturation)

    return _colour


# ---------------------------------------------------------------------------
# Terrain colour helpers
# ---------------------------------------------------------------------------

def _terrain_colour_fn(grid, store, biome, noise_seed, hillshade=None):
    """Return a colour callback for terrain rendering.

    If *hillshade* is ``None``, a flat 0.5 value is used for every face
    (suitable for neighbour grids where hillshade is less important).
    """
    from polygrid.detail_render import detail_elevation_to_colour
    from polygrid.geometry import face_center as _face_center

    def _colour(fid, _grid, face):
        elev = store.get(fid, "elevation")
        c = _face_center(grid.vertices, face)
        cx, cy = c if c else (0.0, 0.0)
        hs_val = hillshade.get(fid, 0.5) if hillshade else 0.5
        return detail_elevation_to_colour(
            elev, biome,
            hillshade_val=hs_val,
            noise_x=cx, noise_y=cy,
            noise_seed=noise_seed,
        )

    return _colour


# ---------------------------------------------------------------------------
# Public rendering functions
# ---------------------------------------------------------------------------

def _render_stitched_tile(
    face_id: str,
    composite,
    stitched_store,
    output_path: Path,
    *,
    biome,
    tile_size: int,
    show_edges: bool,
    noise_seed: int,
    hillshade: dict[str, float] | None = None,
    renderer: str = "matplotlib",
):
    """Render a stitched tile+neighbours grid to PNG.

    The centre tile faces are rendered identically to neighbours —
    they form one seamless grid.  The view is cropped to the centre
    tile's extent (with a small margin to show surrounding context).

    Uses a sentinel background colour so that any uncovered pixels
    (image corners outside the polygon patch coverage) can be
    detected and filled by the atlas builder, preventing per-tile
    background colour from bleeding into the atlas gutter.

    Parameters
    ----------
    hillshade : dict, optional
        Pre-computed ``{merged_face_id: float}`` hillshade values.
        If provided, skips the per-composite hillshade computation
        (which suffers from boundary truncation).  Use
        :func:`_compute_global_hillshade` +
        :func:`_resolve_hillshade_for_composite` to build this.
    renderer : str, optional
        ``"matplotlib"`` (default) uses matplotlib PatchCollection
        rasterisation.  ``"analytical"`` bypasses matplotlib and fills
        each pixel via point-in-polygon tests — deterministic, no
        anti-aliasing, eliminates sub-pixel rasterisation seams.
    """
    from polygrid.detail_render import _detail_hillshade
    from polygrid.tile_uv_align import compute_tile_view_limits

    mg = composite.merged

    if hillshade is None:
        # Fallback: compute per-composite (has boundary truncation)
        hs = _detail_hillshade(
            mg, stitched_store, "elevation",
            azimuth=biome.azimuth, altitude=biome.altitude,
        )
    else:
        hs = hillshade

    colour_fn = _terrain_colour_fn(mg, stitched_store, biome, noise_seed, hs)
    xlim, ylim = compute_tile_view_limits(composite, face_id)

    if renderer == "analytical":
        _render_analytical_to_png(
            mg, colour_fn, output_path,
            tile_size=tile_size, xlim=xlim, ylim=ylim,
            bg_colour=_SENTINEL_BG,
        )
    else:
        patches, colours = _build_face_patches(mg, colour_fn)
        if not patches:
            return
        edge_col, edge_lw = _edge_style(show_edges)
        _render_patches_to_png(
            patches, colours, output_path,
            tile_size=tile_size, xlim=xlim, ylim=ylim,
            edge_colour=edge_col, edge_lw=edge_lw,
            bg_colour=_SENTINEL_BG,
        )


def _render_colour_debug_tile(
    face_id: str,
    composite,
    output_path: Path,
    *,
    tile_size: int,
    outline_tiles: bool,
    tile_hues: dict[str, float],
):
    """Render a stitched tile with each component in a distinct colour.

    Each component (centre tile + each neighbour) gets its own
    globe-wide unique hue from *tile_hues*.  No terrain data needed.
    """
    from polygrid.tile_uv_align import compute_tile_view_limits

    comp_centroids, comp_max_dist = _compute_component_gradient_info(
        composite,
    )
    colour_fn = _colour_debug_fn(
        composite, tile_hues, comp_centroids, comp_max_dist,
    )
    patches, colours = _build_face_patches(composite.merged, colour_fn)

    if not patches:
        return

    xlim, ylim = compute_tile_view_limits(composite, face_id)
    edge_col, edge_lw = _edge_style(outline_tiles, "#00000030", 0.4)

    _render_patches_to_png(
        patches, colours, output_path,
        tile_size=tile_size, xlim=xlim, ylim=ylim,
        edge_colour=edge_col, edge_lw=edge_lw,
        bg_colour=_SENTINEL_BG,
    )


def _render_colour_debug_single(
    face_id: str,
    detail_grid,
    output_path: Path,
    *,
    tile_size: int,
    outline_tiles: bool,
    hue: float,
):
    """Render a single detail grid (no stitching) in colour-debug style."""
    colour_fn = _colour_debug_single_fn(detail_grid, hue)
    patches, colours = _build_face_patches(detail_grid, colour_fn)

    if not patches:
        return

    xlim, ylim = _grid_bbox(detail_grid)
    edge_col, edge_lw = _edge_style(outline_tiles, "#00000030", 0.4)

    _render_patches_to_png(
        patches, colours, output_path,
        tile_size=tile_size, xlim=xlim, ylim=ylim,
        edge_colour=edge_col, edge_lw=edge_lw,
    )


def _render_tile(  # TODO REMOVE — Legacy single-tile renderer, only used by _main_simple/_main_neighbour_edges.
    face_id: str,
    detail_grid,
    detail_store,
    output_path: Path,
    *,
    biome,
    tile_size: int,
    show_edges: bool,
    noise_seed: int,
    neighbour_grid=None,
    neighbour_store=None,
):
    """Render a single tile polygrid to PNG.

    Parameters
    ----------
    neighbour_grid, neighbour_store : PolyGrid, TileDataStore, optional
        If provided, the neighbour border grid is rendered first
        (with edge outlines) as a background, then the tile grid is
        rendered on top.
    """
    from polygrid.detail_render import render_detail_texture_enhanced

    # Fast path: no extras — use existing renderer directly
    if not show_edges and neighbour_grid is None:
        render_detail_texture_enhanced(
            detail_grid, detail_store, output_path,
            biome=biome, tile_size=tile_size, noise_seed=noise_seed,
        )
        return

    from polygrid.detail_render import _detail_hillshade
    from matplotlib.collections import PatchCollection

    # ── Layer 1: neighbour border grid (background, with outlines) ──
    extra_layers = []
    if neighbour_grid is not None and neighbour_store is not None:
        nbr_colour_fn = _terrain_colour_fn(
            neighbour_grid, neighbour_store, biome, noise_seed,
        )
        nbr_patches, nbr_colours = _build_face_patches(
            neighbour_grid, nbr_colour_fn,
        )
        if nbr_patches:
            extra_layers.append(PatchCollection(
                nbr_patches, facecolors=nbr_colours,
                edgecolors="#00000080", linewidths=0.8,
            ))

    # ── Layer 2: tile grid (foreground) ──────────────────────────────
    hs = _detail_hillshade(
        detail_grid, detail_store, "elevation",
        azimuth=biome.azimuth, altitude=biome.altitude,
    )
    colour_fn = _terrain_colour_fn(
        detail_grid, detail_store, biome, noise_seed, hs,
    )
    patches, colours = _build_face_patches(detail_grid, colour_fn)

    if not patches:
        return

    edge_col, edge_lw = _edge_style(show_edges)

    # View limits
    if neighbour_grid is not None:
        all_x, all_y = [], []
        for g in (detail_grid, neighbour_grid):
            for v in g.vertices.values():
                if v.has_position():
                    all_x.append(v.x)
                    all_y.append(v.y)
        pad = 0.5
        xlim = (min(all_x) - pad, max(all_x) + pad)
        ylim = (min(all_y) - pad, max(all_y) + pad)
        bg = (0.15, 0.15, 0.15)
    else:
        xlim, ylim = _grid_bbox(detail_grid, pad=1.0)
        bg = _SENTINEL_BG

    _render_patches_to_png(
        patches, colours, output_path,
        tile_size=tile_size, xlim=xlim, ylim=ylim,
        edge_colour=edge_col, edge_lw=edge_lw,
        bg_colour=bg,
        extra_layers=extra_layers,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Render per-tile polygrid PNGs for a Goldberg globe.",
    )
    parser.add_argument(
        "-f", "--frequency", type=int, default=3,
        help="Goldberg polyhedron frequency (default: 3)",
    )
    parser.add_argument(
        "--detail-rings", type=int, default=4,
        help="Detail grid ring count (default: 4)",
    )
    parser.add_argument(
        "--preset", default="mountain_range",
        choices=["mountain_range", "alpine_peaks", "rolling_hills"],
        help="Terrain preset (default: mountain_range)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--tile-size", type=int, default=512,
        help="Output image size in pixels (default: 512)",
    )
    parser.add_argument(
        "--edges", action="store_true",
        help="Show grid edges overlaid on terrain colouring",
    )
    parser.add_argument(
        "--stitched", action="store_true",
        help="Render each tile stitched with all its neighbours "
             "as a single merged polygrid (no gaps)",
    )  # TODO REMOVE — Legacy flag, only triggers dead _main_stitched path.
    parser.add_argument(
        "--no-polygon-cut", action="store_true",
        help="Disable polygon-cut rendering. By default, polygon-cut "
             "UV-aligned tile textures are produced from stitched "
             "renders, generating an atlas ready for 3D globe mapping.",
    )
    parser.add_argument(
        "--debug-labels", action="store_true",
        help="Draw tile-ID and per-edge neighbour labels on each "
             "polygon-cut tile.",
    )
    parser.add_argument(
        "--polygon-mask", action="store_true",
        help="Apply black masking to pixels outside the UV polygon "
             "in polygon-cut tiles. Off by default; useful for "
             "debugging to visualise the polygon boundary.",
    )
    parser.add_argument(
        "--pent-rot", type=int, default=0, metavar="N",
        help="Extra rotation steps for pentagon tiles (positive = CW). "
             "Use to correct residual pentagon orientation mismatch.",
    )
    parser.add_argument(
        "--with-neighbour-edges", action="store_true",
        help="(Legacy) Show neighbour tile border faces around each tile",
    )  # TODO REMOVE — Legacy flag, only triggers dead _main_neighbour_edges path.
    parser.add_argument(
        "--colour-debug", action="store_true",
        help="Skip terrain generation and colour each polygrid tile "
             "with a unique hue. Centre tile faces are lightest, "
             "fading darker toward the edges. Useful for inspecting "
             "stitching topology without terrain noise.",
    )
    parser.add_argument(
        "--outline-tiles", action="store_true",
        help="Draw thin outlines on every child polygon (sub-face) "
             "within each polygrid tile.",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="Output directory (default: exports/polygrids/)",
    )
    parser.add_argument(
        "--renderer", default="analytical",
        choices=["matplotlib", "analytical"],
        help="Tile renderer backend (default: analytical). "
             "'analytical' uses point-in-polygon fill — deterministic, "
             "no anti-aliasing, eliminates sub-pixel rasterisation seams. "
             "'matplotlib' uses patch-based rasterisation with anti-aliasing.",
    )

    args = parser.parse_args()

    from polygrid.detail_render import BiomeConfig

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parent.parent / "exports" / "polygrids"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Colour-debug mode: skip terrain, colour tiles by identity ──
    if args.colour_debug:
        _main_colour_debug(args, output_dir)
        return

    # ── Terrain modes ──
    grid, store = _build_globe_and_terrain(
        args.frequency, args.preset, args.seed,
    )
    coll = _build_detail_grids(grid, store, args.detail_rings, args.seed)
    biome = BiomeConfig()
    face_ids = coll.face_ids

    # --outline-tiles is a synonym for --edges (both show sub-face outlines)
    show_edges = args.edges or args.outline_tiles

    # Derive effective polygon_cut flag (default on, unless --no-polygon-cut)
    polygon_cut = not args.no_polygon_cut

    if polygon_cut:
        _main_polygon_cut(args, output_dir, grid, store, coll, biome,
                          face_ids, show_edges)
    elif args.stitched:
        _main_stitched(args, output_dir, grid, coll, biome,
                       face_ids, show_edges)
    elif args.with_neighbour_edges:
        _main_neighbour_edges(args, output_dir, grid, coll, biome,
                              face_ids, show_edges)
    else:
        _main_simple(args, output_dir, coll, biome, face_ids, show_edges)


# ---------------------------------------------------------------------------
# Atlas + export helpers
# ---------------------------------------------------------------------------

def _build_atlas(
    tile_images, composites, detail_grids, grid, face_ids, args,
    output_dir,
):
    """Build polygon-cut atlas and write atlas.png + uv_layout.json.

    Returns ``(atlas, uv_layout)``.
    """
    import json
    from polygrid.tile_uv_align import build_polygon_cut_atlas

    print("Building polygon-cut atlas...")
    gutter = 4
    debug_dir = output_dir / "warped"
    debug_dir.mkdir(parents=True, exist_ok=True)

    atlas, uv_layout = build_polygon_cut_atlas(
        tile_images, composites, detail_grids, grid, face_ids,
        tile_size=args.tile_size,
        gutter=gutter,
        mask_outside=args.polygon_mask,
        debug_labels=args.debug_labels,
        output_dir=debug_dir,
        pentagon_rotation_steps=args.pent_rot,
    )

    atlas_path = output_dir / "atlas.png"
    atlas.save(str(atlas_path))
    print(f"  → Atlas: {atlas_path}")

    uv_path = output_dir / "uv_layout.json"
    uv_path.write_text(json.dumps(uv_layout, indent=2))
    print(f"  → UV layout: {uv_path}")

    return atlas, uv_layout


def _export_payload_and_metadata(
    grid, store, output_dir, *,
    frequency, seed, preset, detail_rings, tile_size,
    colour_overrides=None,
):
    """Export globe_payload.json and metadata.json.

    *colour_overrides* is an optional ``{tile_id: (r, g, b)}`` dict;
    when provided it replaces the terrain-derived tile colours in the
    payload (used by colour-debug mode).
    """
    import json
    from polygrid.globe_export import export_globe_payload

    payload = export_globe_payload(grid, store, ramp="satellite")

    if colour_overrides:
        for tile_entry in payload["tiles"]:
            c = colour_overrides.get(tile_entry["id"])
            if c is not None:
                tile_entry["color"] = [round(c[0], 4), round(c[1], 4),
                                       round(c[2], 4)]

    payload_path = output_dir / "globe_payload.json"
    payload_path.write_text(json.dumps(payload, indent=2))
    print(f"  → Payload: {payload_path}")

    metadata = {
        "frequency": frequency,
        "seed": seed,
        "preset": preset,
        "detail_rings": detail_rings,
        "tile_size": tile_size,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"  → Metadata: {metadata_path}")


# ---------------------------------------------------------------------------
# Mode-specific main branches
# ---------------------------------------------------------------------------

def _main_colour_debug(args, output_dir):
    """Colour-debug mode: skip terrain, colour tiles by identity."""
    import colorsys

    from PIL import Image

    from polygrid.globe import build_globe_grid
    from polygrid.tile_detail import (
        TileDetailSpec,
        DetailGridCollection,
        build_tile_with_neighbours,
    )
    from polygrid.tile_data import FieldDef, TileDataStore, TileSchema

    print(f"Building globe (freq={args.frequency})...")
    grid = build_globe_grid(args.frequency)
    spec = TileDetailSpec(detail_rings=args.detail_rings)

    t0 = time.perf_counter()
    print(f"Building detail grids (rings={args.detail_rings})...")
    coll = DetailGridCollection.build(grid, spec)
    print(f"  → {coll.total_face_count} sub-faces in "
          f"{time.perf_counter() - t0:.2f}s")

    face_ids = coll.face_ids

    # Pre-compute a unique hue for every globe tile (golden-ratio
    # spacing gives perceptually even colours across the whole globe).
    golden = 0.618033988749895
    tile_hues: dict[str, float] = {}
    for idx, fid in enumerate(face_ids):
        tile_hues[fid] = (0.08 + idx * golden) % 1.0

    # Phase 0: render standalone (un-stitched) polygrids
    singles_dir = output_dir / "singles"
    singles_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(face_ids)} standalone polygrids "
          f"to {singles_dir}/...")
    t0 = time.perf_counter()

    for i, fid in enumerate(face_ids):
        dg, _ = coll.get(fid)
        out_path = singles_dir / f"{fid}.png"
        _render_colour_debug_single(
            fid, dg, out_path,
            tile_size=args.tile_size,
            outline_tiles=args.outline_tiles,
            hue=tile_hues[fid],
        )
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  {i + 1}/{len(face_ids)}...")

    elapsed = time.perf_counter() - t0
    print(f"  → {len(face_ids)} singles in {elapsed:.2f}s")

    mode = "colour-debug" + (" + outlines" if args.outline_tiles else "")
    print(f"Rendering {len(face_ids)} tiles ({mode}) to {output_dir}/...")
    t0 = time.perf_counter()

    # Phase 1: render colour-debug tiles + collect data for atlas
    tile_images: dict[str, Image.Image] = {}
    composites: dict = {}
    detail_grids: dict = {}

    for i, fid in enumerate(face_ids):
        composite = build_tile_with_neighbours(coll, fid, grid)
        out_path = output_dir / f"{fid}.png"
        _render_colour_debug_tile(
            fid, composite, out_path,
            tile_size=args.tile_size,
            outline_tiles=args.outline_tiles,
            tile_hues=tile_hues,
        )
        tile_images[fid] = Image.open(str(out_path)).convert("RGB")
        composites[fid] = composite
        detail_grids[fid] = coll.get(fid)[0]

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  rendered {i + 1}/{len(face_ids)}...")

    elapsed = time.perf_counter() - t0
    print(f"  → {len(face_ids)} tiles rendered in {elapsed:.2f}s")

    # Phase 2: build polygon-cut atlas
    _build_atlas(
        tile_images, composites, detail_grids, grid, face_ids,
        args, output_dir,
    )

    # Phase 3: export payload + metadata
    # Build a dummy store (elevation=0) since there's no terrain.
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    dummy_store = TileDataStore(grid=grid, schema=schema)

    colour_overrides = {
        fid: colorsys.hls_to_rgb(tile_hues.get(fid, 0.0), 0.55, 0.70)
        for fid in face_ids
    }
    _export_payload_and_metadata(
        grid, dummy_store, output_dir,
        frequency=args.frequency, seed=0, preset="colour_debug",
        detail_rings=args.detail_rings, tile_size=args.tile_size,
        colour_overrides=colour_overrides,
    )

    print(f"Output: {output_dir}/")


def _main_polygon_cut(args, output_dir, grid, store, coll, biome,
                       face_ids, show_edges):
    """Default polygon-cut atlas pipeline."""
    from PIL import Image
    from polygrid.tile_detail import build_tile_with_neighbours

    mode = "polygon-cut" + (" + edges" if show_edges else "")
    print(f"Rendering {len(face_ids)} tiles ({mode}) to {output_dir}/...")
    t0 = time.perf_counter()

    # Phase 0: global hillshade pre-computation (eliminates boundary
    # truncation — each face gets hillshade from the composite where
    # it was the centre tile, with full neighbour context).
    global_hs = _compute_global_hillshade(coll, grid, face_ids, biome)

    # Phase 1: render stitched tiles + collect composites & detail grids
    tile_images = {}
    composites = {}
    detail_grids = {}
    for i, fid in enumerate(face_ids):
        composite = build_tile_with_neighbours(coll, fid, grid)
        stitched_store = _build_stitched_store(composite, coll, fid, grid)
        # Resolve pre-computed hillshade for this composite's merged grid
        hs = _resolve_hillshade_for_composite(composite, fid, global_hs)
        out_path = output_dir / f"{fid}.png"
        _render_stitched_tile(
            fid, composite, stitched_store, out_path,
            biome=biome,
            tile_size=args.tile_size,
            show_edges=show_edges,
            noise_seed=args.seed,
            hillshade=hs,
            renderer=args.renderer,
        )
        tile_images[fid] = Image.open(str(out_path)).convert("RGB")
        composites[fid] = composite
        detail_grids[fid] = coll.get(fid)[0]

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  rendered {i + 1}/{len(face_ids)}...")

    elapsed = time.perf_counter() - t0
    print(f"  → {len(face_ids)} tiles rendered in {elapsed:.2f}s")

    # Phase 2: build polygon-cut atlas
    _build_atlas(
        tile_images, composites, detail_grids, grid, face_ids,
        args, output_dir,
    )

    # Phase 3: export payload + metadata
    _export_payload_and_metadata(
        grid, store, output_dir,
        frequency=args.frequency, seed=args.seed, preset=args.preset,
        detail_rings=args.detail_rings, tile_size=args.tile_size,
    )

    print(f"Output: {output_dir}/")


def _main_stitched(args, output_dir, grid, coll, biome,  # TODO REMOVE — Legacy path, needs --no-polygon-cut --stitched.
                    face_ids, show_edges):
    """Stitched tile rendering (no atlas)."""
    from polygrid.tile_detail import build_tile_with_neighbours

    mode = "stitched" + (" + edges" if show_edges else "")
    print(f"Rendering {len(face_ids)} tiles ({mode}) to {output_dir}/...")
    t0 = time.perf_counter()

    # Global hillshade pre-computation
    global_hs = _compute_global_hillshade(coll, grid, face_ids, biome)

    for i, fid in enumerate(face_ids):
        composite = build_tile_with_neighbours(coll, fid, grid)
        stitched_store = _build_stitched_store(composite, coll, fid, grid)
        hs = _resolve_hillshade_for_composite(composite, fid, global_hs)
        out_path = output_dir / f"{fid}.png"
        _render_stitched_tile(
            fid, composite, stitched_store, out_path,
            biome=biome,
            tile_size=args.tile_size,
            show_edges=show_edges,
            noise_seed=args.seed,
            hillshade=hs,
            renderer=args.renderer,
        )
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  {i + 1}/{len(face_ids)}...")

    elapsed = time.perf_counter() - t0
    print(f"  → {len(face_ids)} tiles rendered in {elapsed:.2f}s")
    print(f"Output: {output_dir}/")


def _main_neighbour_edges(args, output_dir, grid, coll, biome,  # TODO REMOVE — Legacy path.
                           face_ids, show_edges):
    """Legacy neighbour-edge rendering."""
    from polygrid.tile_detail import get_neighbour_border_grid

    print(f"Rendering {len(face_ids)} tiles "
          f"(with neighbour edges) to {output_dir}/...")
    t0 = time.perf_counter()

    for i, fid in enumerate(face_ids):
        detail_grid, detail_store = coll.get(fid)
        if detail_store is None:
            print(f"  ⚠ {fid}: no terrain data, skipping")
            continue
        nbr_grid, nbr_store = get_neighbour_border_grid(coll, fid, grid)
        out_path = output_dir / f"{fid}.png"
        _render_tile(
            fid, detail_grid, detail_store, out_path,
            biome=biome,
            tile_size=args.tile_size,
            show_edges=show_edges,
            noise_seed=args.seed,
            neighbour_grid=nbr_grid,
            neighbour_store=nbr_store,
        )

    elapsed = time.perf_counter() - t0
    print(f"  → {len(face_ids)} tiles rendered in {elapsed:.2f}s")
    print(f"Output: {output_dir}/")


def _main_simple(args, output_dir, coll, biome, face_ids, show_edges):  # TODO REMOVE — Legacy path.
    """Simple single-tile rendering (no stitching, no atlas)."""
    print(f"Rendering {len(face_ids)} tile polygrids to {output_dir}/...")
    t0 = time.perf_counter()

    for i, fid in enumerate(face_ids):
        detail_grid, detail_store = coll.get(fid)
        if detail_store is None:
            print(f"  ⚠ {fid}: no terrain data, skipping")
            continue
        out_path = output_dir / f"{fid}.png"
        _render_tile(
            fid, detail_grid, detail_store, out_path,
            biome=biome,
            tile_size=args.tile_size,
            show_edges=show_edges,
            noise_seed=args.seed,
        )

    elapsed = time.perf_counter() - t0
    print(f"  → {len(face_ids)} tiles rendered in {elapsed:.2f}s")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
