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
):
    """Render a stitched tile+neighbours grid to PNG.

    The centre tile faces are rendered identically to neighbours —
    they form one seamless grid. The view is cropped to the centre
    tile's extent (with a small margin to show surrounding context).
    """
    from polygrid.detail_render import (
        BiomeConfig,
        detail_elevation_to_colour,
        _detail_hillshade,
    )
    from polygrid.geometry import face_center as _face_center

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    mg = composite.merged
    center_prefix = composite.id_prefixes[face_id]

    # Compute hillshade across the whole stitched grid
    hs = _detail_hillshade(
        mg, stitched_store, "elevation",
        azimuth=biome.azimuth, altitude=biome.altitude,
    )

    # Build patches + colours for every face
    patches = []
    colours = []
    for fid, face in mg.faces.items():
        verts = []
        for vid in face.vertex_ids:
            v = mg.vertices.get(vid)
            if v is None or not v.has_position():
                break
            verts.append((v.x, v.y))
        else:
            if len(verts) >= 3:
                elev = stitched_store.get(fid, "elevation")
                c = _face_center(mg.vertices, face)
                cx, cy = c if c else (0.0, 0.0)
                colour = detail_elevation_to_colour(
                    elev, biome,
                    hillshade_val=hs.get(fid, 0.5),
                    noise_x=cx, noise_y=cy,
                    noise_seed=noise_seed,
                )
                patches.append(MplPolygon(verts, closed=True))
                colours.append(colour)

    if not patches:
        return

    # Determine axis limits from the centre tile's extent
    center_xs, center_ys = [], []
    for fid, face in mg.faces.items():
        if not fid.startswith(center_prefix):
            continue
        for vid in face.vertex_ids:
            v = mg.vertices.get(vid)
            if v is not None and v.has_position():
                center_xs.append(v.x)
                center_ys.append(v.y)

    # Pad so the first ring of neighbour faces is fully visible.
    # Use equal x/y ranges so the output image is always square
    # (critical for polygon-cut UV alignment which assumes a
    # deterministic pixel ↔ grid mapping).
    cx_range = max(center_xs) - min(center_xs) if center_xs else 1
    cy_range = max(center_ys) - min(center_ys) if center_ys else 1
    half_span = max(cx_range, cy_range) * 0.5 * 1.25  # 25% padding
    cx_mid = (min(center_xs) + max(center_xs)) * 0.5
    cy_mid = (min(center_ys) + max(center_ys)) * 0.5
    xlim = (cx_mid - half_span, cx_mid + half_span)
    ylim = (cy_mid - half_span, cy_mid + half_span)

    # Render
    dpi = 100
    fig_size = tile_size / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    edge_col = "#00000040" if show_edges else "none"
    edge_lw = 0.5 if show_edges else 0
    pc = PatchCollection(
        patches, facecolors=colours,
        edgecolors=edge_col, linewidths=edge_lw,
    )
    ax.add_collection(pc)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Background: average colour of centre tile faces
    center_colours = [
        c for fid, c in zip(mg.faces.keys(), colours)
        if fid.startswith(center_prefix)
    ]
    if center_colours:
        avg_r = sum(c[0] for c in center_colours) / len(center_colours)
        avg_g = sum(c[1] for c in center_colours) / len(center_colours)
        avg_b = sum(c[2] for c in center_colours) / len(center_colours)
        bg_colour = (avg_r, avg_g, avg_b)
    else:
        bg_colour = (0.15, 0.15, 0.15)

    fig.savefig(
        str(output_path), dpi=dpi, bbox_inches="tight", pad_inches=0,
        facecolor=bg_colour,
    )
    plt.close(fig)


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
    globe-wide unique hue from *tile_hues*.  The centre component is
    rendered brightly; neighbour components use their own globe-tile
    hue but at reduced lightness so the centre stands out.

    No terrain data is needed.
    """
    import colorsys

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    from polygrid.geometry import face_center as _face_center

    mg = composite.merged
    center_prefix = composite.id_prefixes[face_id]

    # Each component uses the globe-wide hue assigned to that tile.
    comp_hues: dict[str, float] = {}
    for name in composite.id_prefixes:
        comp_hues[name] = tile_hues.get(name, 0.0)

    # Pre-compute each component's centroid + max face-distance for
    # the radial gradient.
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

    # Build patches + colours
    patches = []
    colours = []

    for fid, face in mg.faces.items():
        verts = []
        for vid in face.vertex_ids:
            v = mg.vertices.get(vid)
            if v is None or not v.has_position():
                break
            verts.append((v.x, v.y))
        else:
            if len(verts) < 3:
                continue

            # Identify which component this face belongs to.
            comp_name = None
            for name, prefix in composite.id_prefixes.items():
                if fid.startswith(prefix):
                    comp_name = name
                    break
            if comp_name is None:
                continue

            hue = comp_hues[comp_name]
            cx, cy = comp_centroids[comp_name]
            c = _face_center(mg.vertices, face)
            fx, fy = c if c else (cx, cy)

            # Normalised distance [0, 1] from component centroid
            dist = ((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5
            t = min(dist / comp_max_dist[comp_name], 1.0)

            # Same brightness for centre and neighbours so the
            # stitched result looks cohesive on the globe.
            lightness = 0.72 - 0.20 * t
            saturation = 0.65 + 0.15 * t

            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            patches.append(MplPolygon(verts, closed=True))
            colours.append((r, g, b))

    if not patches:
        return

    # Axis limits from centre tile
    center_xs, center_ys = [], []
    for fid, face in mg.faces.items():
        if not fid.startswith(center_prefix):
            continue
        for vid in face.vertex_ids:
            v = mg.vertices.get(vid)
            if v is not None and v.has_position():
                center_xs.append(v.x)
                center_ys.append(v.y)

    cx_range = max(center_xs) - min(center_xs) if center_xs else 1
    cy_range = max(center_ys) - min(center_ys) if center_ys else 1
    half_span = max(cx_range, cy_range) * 0.5 * 1.25
    cx_mid = (min(center_xs) + max(center_xs)) * 0.5
    cy_mid = (min(center_ys) + max(center_ys)) * 0.5
    xlim = (cx_mid - half_span, cx_mid + half_span)
    ylim = (cy_mid - half_span, cy_mid + half_span)

    # Render
    dpi = 100
    fig_size = tile_size / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    edge_col = "#00000030" if outline_tiles else "none"
    edge_lw = 0.4 if outline_tiles else 0
    pc = PatchCollection(
        patches, facecolors=colours,
        edgecolors=edge_col, linewidths=edge_lw,
    )
    ax.add_collection(pc)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    fig.savefig(
        str(output_path), dpi=dpi, bbox_inches="tight", pad_inches=0,
        facecolor=(0.12, 0.12, 0.12),
    )
    plt.close(fig)


def _render_colour_debug_single(
    face_id: str,
    detail_grid,
    output_path: Path,
    *,
    tile_size: int,
    outline_tiles: bool,
    hue: float,
):
    """Render a single detail grid (no stitching) in colour-debug style.

    All sub-faces use the tile's unique *hue* with a radial gradient
    from the grid centroid.  Produces a standalone PNG of just the
    one polygrid tile.
    """
    import colorsys

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    from polygrid.geometry import face_center as _face_center

    # Centroid + max distance for radial gradient
    xs, ys = [], []
    for fid, face in detail_grid.faces.items():
        c = _face_center(detail_grid.vertices, face)
        if c is not None:
            xs.append(c[0])
            ys.append(c[1])
    if not xs:
        return
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    max_d = max(
        (((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 for x, y in zip(xs, ys)),
        default=1.0,
    )
    if max_d < 1e-9:
        max_d = 1.0

    patches = []
    colours = []
    all_xs, all_ys = [], []

    for fid, face in detail_grid.faces.items():
        verts = []
        for vid in face.vertex_ids:
            v = detail_grid.vertices.get(vid)
            if v is None or not v.has_position():
                break
            verts.append((v.x, v.y))
            all_xs.append(v.x)
            all_ys.append(v.y)
        else:
            if len(verts) < 3:
                continue
            c = _face_center(detail_grid.vertices, face)
            fx, fy = c if c else (cx, cy)
            t = min(((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5 / max_d, 1.0)
            lightness = 0.72 - 0.20 * t
            saturation = 0.65 + 0.15 * t
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            patches.append(MplPolygon(verts, closed=True))
            colours.append((r, g, b))

    if not patches:
        return

    x_range = max(all_xs) - min(all_xs) if all_xs else 1
    y_range = max(all_ys) - min(all_ys) if all_ys else 1
    half_span = max(x_range, y_range) * 0.5 * 1.15
    x_mid = (min(all_xs) + max(all_xs)) * 0.5
    y_mid = (min(all_ys) + max(all_ys)) * 0.5

    dpi = 100
    fig_size = tile_size / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    edge_col = "#00000030" if outline_tiles else "none"
    edge_lw = 0.4 if outline_tiles else 0
    pc = PatchCollection(
        patches, facecolors=colours,
        edgecolors=edge_col, linewidths=edge_lw,
    )
    ax.add_collection(pc)
    ax.set_xlim(x_mid - half_span, x_mid + half_span)
    ax.set_ylim(y_mid - half_span, y_mid + half_span)

    fig.savefig(
        str(output_path), dpi=dpi, bbox_inches="tight", pad_inches=0,
        facecolor=(0.12, 0.12, 0.12),
    )
    plt.close(fig)


def _render_tile(
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

    # Manual rendering with matplotlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    from polygrid.detail_render import (
        detail_elevation_to_colour,
        _detail_hillshade,
    )
    from polygrid.geometry import face_center as _face_center

    dpi = 100
    fig_size = tile_size / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # ── Layer 1: neighbour border grid (background, with outlines) ──
    if neighbour_grid is not None and neighbour_store is not None:
        nbr_patches = []
        nbr_colours = []
        for fid, face in neighbour_grid.faces.items():
            verts = []
            for vid in face.vertex_ids:
                v = neighbour_grid.vertices.get(vid)
                if v is None or not v.has_position():
                    break
                verts.append((v.x, v.y))
            else:
                if len(verts) >= 3:
                    elev = neighbour_store.get(fid, "elevation")
                    c = _face_center(neighbour_grid.vertices, face)
                    cx, cy = c if c else (0.0, 0.0)
                    colour = detail_elevation_to_colour(
                        elev, biome,
                        hillshade_val=0.5,
                        noise_x=cx, noise_y=cy,
                        noise_seed=noise_seed,
                    )
                    nbr_patches.append(MplPolygon(verts, closed=True))
                    nbr_colours.append(colour)

        if nbr_patches:
            nbr_pc = PatchCollection(
                nbr_patches, facecolors=nbr_colours,
                edgecolors="#00000080", linewidths=0.8,
            )
            ax.add_collection(nbr_pc)

    # ── Layer 2: tile grid (foreground) ──────────────────────────────
    hs = _detail_hillshade(
        detail_grid, detail_store, "elevation",
        azimuth=biome.azimuth, altitude=biome.altitude,
    )

    tile_patches = []
    tile_colours = []
    for fid, face in detail_grid.faces.items():
        verts = []
        for vid in face.vertex_ids:
            v = detail_grid.vertices.get(vid)
            if v is None or not v.has_position():
                break
            verts.append((v.x, v.y))
        else:
            if len(verts) >= 3:
                elev = detail_store.get(fid, "elevation")
                c = _face_center(detail_grid.vertices, face)
                cx, cy = c if c else (0.0, 0.0)
                colour = detail_elevation_to_colour(
                    elev, biome,
                    hillshade_val=hs.get(fid, 0.5),
                    noise_x=cx, noise_y=cy,
                    noise_seed=noise_seed,
                )
                tile_patches.append(MplPolygon(verts, closed=True))
                tile_colours.append(colour)

    if not tile_patches:
        fig.savefig(str(output_path), dpi=dpi)
        plt.close(fig)
        return

    edge_col = "#00000040" if show_edges else "none"
    edge_lw = 0.5 if show_edges else 0
    tile_pc = PatchCollection(
        tile_patches, facecolors=tile_colours,
        edgecolors=edge_col, linewidths=edge_lw,
    )
    ax.add_collection(tile_pc)

    # Set explicit axis limits so neighbour faces are fully visible
    if neighbour_grid is not None:
        # Combine extents of tile + neighbour grids with padding
        all_x, all_y = [], []
        for v in detail_grid.vertices.values():
            if v.has_position():
                all_x.append(v.x)
                all_y.append(v.y)
        for v in neighbour_grid.vertices.values():
            if v.has_position():
                all_x.append(v.x)
                all_y.append(v.y)
        pad = 0.5
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    else:
        ax.autoscale_view()

    # Dark background when showing neighbours so they're visible;
    # otherwise use average tile colour for a seamless look
    if neighbour_grid is not None:
        bg_colour = (0.15, 0.15, 0.15)
    else:
        avg_r = sum(c[0] for c in tile_colours) / len(tile_colours)
        avg_g = sum(c[1] for c in tile_colours) / len(tile_colours)
        avg_b = sum(c[2] for c in tile_colours) / len(tile_colours)
        bg_colour = (avg_r, avg_g, avg_b)

    fig.savefig(
        str(output_path), dpi=dpi, bbox_inches="tight", pad_inches=0,
        facecolor=bg_colour,
    )
    plt.close(fig)


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
    )
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
    )
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

    args = parser.parse_args()

    from polygrid.detail_render import BiomeConfig

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parent.parent / "exports" / "polygrids"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Colour-debug mode: skip terrain, colour tiles by identity ──
    if args.colour_debug:
        import colorsys
        import json

        from PIL import Image

        from polygrid.globe import build_globe_grid
        from polygrid.tile_detail import (
            TileDetailSpec,
            DetailGridCollection,
            build_tile_with_neighbours,
        )
        from polygrid.tile_uv_align import build_polygon_cut_atlas
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
        print(f"Rendering {len(face_ids)} standalone polygrids to {singles_dir}/...")
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
        print("Building polygon-cut atlas (colour-debug)...")
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

        # Phase 3: export globe payload with colour-debug colours
        # Build a dummy TileDataStore (elevation=0) since there's no terrain.
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        dummy_store = TileDataStore(grid=grid, schema=schema)

        # Build the payload manually so we can inject the colour-debug
        # hues instead of terrain colours.
        from polygrid.globe_export import export_globe_payload

        payload = export_globe_payload(grid, dummy_store, ramp="satellite")

        # Override colours with the colour-debug hues (bright centre hue)
        for tile_entry in payload["tiles"]:
            tid = tile_entry["id"]
            hue = tile_hues.get(tid, 0.0)
            r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.70)
            tile_entry["color"] = [round(r, 4), round(g, 4), round(b, 4)]

        payload_path = output_dir / "globe_payload.json"
        payload_path.write_text(json.dumps(payload, indent=2))
        print(f"  → Payload: {payload_path}")

        metadata = {
            "frequency": args.frequency,
            "seed": 0,
            "preset": "colour_debug",
            "detail_rings": args.detail_rings,
            "tile_size": args.tile_size,
        }
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        print(f"  → Metadata: {metadata_path}")

        print(f"Output: {output_dir}/")
        return

    # Build globe + terrain
    grid, store = _build_globe_and_terrain(
        args.frequency, args.preset, args.seed,
    )

    # Build detail grids + terrain
    coll = _build_detail_grids(grid, store, args.detail_rings, args.seed)

    # Render each tile
    biome = BiomeConfig()
    face_ids = coll.face_ids

    # --outline-tiles is a synonym for --edges (both show sub-face outlines)
    show_edges = args.edges or args.outline_tiles

    # Derive effective polygon_cut flag (default on, unless --no-polygon-cut)
    polygon_cut = not args.no_polygon_cut

    if polygon_cut:
        from polygrid.tile_detail import build_tile_with_neighbours, find_polygon_corners
        from polygrid.tile_uv_align import (
            compute_tile_view_limits,
            compute_grid_to_px_affine,
            warp_tile_to_uv,
            build_polygon_cut_atlas,
            _fill_gutter,
        )
        from polygrid.uv_texture import get_tile_uv_vertices
        from PIL import Image

        mode = "polygon-cut" + (" + edges" if show_edges else "")
        print(f"Rendering {len(face_ids)} tiles ({mode}) to {output_dir}/...")
        t0 = time.perf_counter()

        # Phase 1: render stitched tiles + collect composites & detail grids
        tile_images = {}
        composites = {}
        detail_grids = {}
        for i, fid in enumerate(face_ids):
            composite = build_tile_with_neighbours(coll, fid, grid)
            stitched_store = _build_stitched_store(composite, coll, fid, grid)

            # Render to a temporary file
            out_path = output_dir / f"{fid}.png"
            _render_stitched_tile(
                fid, composite, stitched_store, out_path,
                biome=biome,
                tile_size=args.tile_size,
                show_edges=show_edges,
                noise_seed=args.seed + i,
            )

            tile_images[fid] = Image.open(str(out_path)).convert("RGB")
            composites[fid] = composite
            detail_grids[fid] = coll.get(fid)[0]

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  rendered {i + 1}/{len(face_ids)}...")

        # Phase 2: build polygon-cut atlas
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

        import json
        uv_path = output_dir / "uv_layout.json"
        uv_path.write_text(json.dumps(uv_layout, indent=2))
        print(f"  → UV layout: {uv_path}")

        # Phase 3: export globe payload + metadata for render_globe_from_tiles
        from polygrid.globe_export import export_globe_payload

        payload = export_globe_payload(grid, store, ramp="satellite")
        payload_path = output_dir / "globe_payload.json"
        payload_path.write_text(json.dumps(payload, indent=2))
        print(f"  → Payload: {payload_path}")

        metadata = {
            "frequency": args.frequency,
            "seed": args.seed,
            "preset": args.preset,
            "detail_rings": args.detail_rings,
            "tile_size": args.tile_size,
        }
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        print(f"  → Metadata: {metadata_path}")

    elif args.stitched:
        from polygrid.tile_detail import build_tile_with_neighbours

        mode = "stitched" + (" + edges" if show_edges else "")
        print(f"Rendering {len(face_ids)} tiles ({mode}) to {output_dir}/...")
        t0 = time.perf_counter()

        for i, fid in enumerate(face_ids):
            composite = build_tile_with_neighbours(coll, fid, grid)
            stitched_store = _build_stitched_store(
                composite, coll, fid, grid,
            )
            out_path = output_dir / f"{fid}.png"
            _render_stitched_tile(
                fid, composite, stitched_store, out_path,
                biome=biome,
                tile_size=args.tile_size,
                show_edges=show_edges,
                noise_seed=args.seed + i,
            )
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  {i + 1}/{len(face_ids)}...")

    elif args.with_neighbour_edges:
        from polygrid.tile_detail import get_neighbour_border_grid

        print(f"Rendering {len(face_ids)} tiles "
              f"(with neighbour edges) to {output_dir}/...")
        t0 = time.perf_counter()

        for i, fid in enumerate(face_ids):
            detail_grid, detail_store = coll.get(fid)
            if detail_store is None:
                print(f"  ⚠ {fid}: no terrain data, skipping")
                continue
            nbr_grid, nbr_store = get_neighbour_border_grid(
                coll, fid, grid,
            )
            out_path = output_dir / f"{fid}.png"
            _render_tile(
                fid, detail_grid, detail_store, out_path,
                biome=biome,
                tile_size=args.tile_size,
                show_edges=show_edges,
                noise_seed=args.seed + i,
                neighbour_grid=nbr_grid,
                neighbour_store=nbr_store,
            )

    else:
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
                noise_seed=args.seed + i,
            )

    elapsed = time.perf_counter() - t0
    print(f"  → {len(face_ids)} tiles rendered in {elapsed:.2f}s")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
