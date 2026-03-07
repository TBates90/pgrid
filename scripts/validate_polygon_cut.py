#!/usr/bin/env python3
"""Validate polygon-cut UV alignment for a single tile.

Renders a diagnostic image showing:
- The original stitched tile with polygon outline
- The warped tile with UV polygon outline
- Corner matching lines

Usage
-----
::

    python scripts/validate_polygon_cut.py -f 3 --detail-rings 3 --tile t5
    python scripts/validate_polygon_cut.py -f 3 --detail-rings 3 --tile t0 --tile-size 512

Saves diagnostic PNGs to ``exports/polygon_cut_validation/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Validate polygon-cut UV alignment for a tile.",
    )
    parser.add_argument(
        "-f", "--frequency", type=int, default=3,
    )
    parser.add_argument(
        "--detail-rings", type=int, default=3,
    )
    parser.add_argument(
        "--tile", type=str, default="t5",
        help="Face ID to validate (default: t5)",
    )
    parser.add_argument(
        "--preset", default="mountain_range",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--tile-size", type=int, default=256,
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=None,
    )
    args = parser.parse_args()

    from polygrid.globe import build_globe_grid
    from polygrid.mountains import MountainConfig, generate_mountains
    from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
    from polygrid.tile_detail import (
        TileDetailSpec, DetailGridCollection,
        build_tile_with_neighbours,
    )
    from polygrid.detail_terrain import generate_all_detail_terrain
    from polygrid.detail_render import BiomeConfig
    from polygrid.uv_texture import get_tile_uv_vertices
    from polygrid.tile_uv_align import (
        compute_tile_view_limits,
        compute_polygon_corners_px,
        compute_grid_to_px_affine,
        compute_grid_to_uv_affine,
        warp_tile_to_uv,
        get_macro_edge_corners,
    )
    from PIL import Image, ImageDraw

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parent.parent / "exports" / "polygon_cut_validation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fid = args.tile
    print(f"Validating polygon-cut alignment for {fid}...")

    # Build globe + terrain + detail grids
    print("Building globe...")
    grid = build_globe_grid(args.frequency)
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)
    config = MountainConfig(
        seed=args.seed, ridge_frequency=2.0, ridge_octaves=4,
        peak_elevation=1.0, base_elevation=0.0,
    )
    generate_mountains(grid, store, config)

    spec = TileDetailSpec(detail_rings=args.detail_rings)
    coll = DetailGridCollection.build(grid, spec)
    generate_all_detail_terrain(coll, grid, store, spec, seed=args.seed)

    if fid not in grid.faces:
        print(f"Error: {fid} not found (available: {list(grid.faces.keys())[:10]}...)")
        sys.exit(1)

    # Build stitched composite
    print("Building stitched tile...")
    composite = build_tile_with_neighbours(coll, fid, grid)

    # Build stitched store
    from polygrid.tile_data import FieldDef as FD, TileDataStore as TDS, TileSchema as TS
    s = TS([FD("elevation", float, 0.0)])
    stitched_store = TDS(grid=composite.merged, schema=s)
    for comp_name, prefix in composite.id_prefixes.items():
        _, comp_store = coll.get(comp_name)
        if comp_store is None:
            continue
        for face_id_inner in composite.components[comp_name].faces:
            pfid = f"{prefix}{face_id_inner}"
            if pfid in composite.merged.faces:
                stitched_store.set(pfid, "elevation",
                                   comp_store.get(face_id_inner, "elevation"))

    # Render stitched tile
    stitched_path = output_dir / f"{fid}_stitched.png"

    # Import the render function from render_polygrids
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from render_polygrids import _render_stitched_tile
    biome = BiomeConfig()

    _render_stitched_tile(
        fid, composite, stitched_store, stitched_path,
        biome=biome, tile_size=args.tile_size,
        show_edges=False, noise_seed=args.seed,
    )
    tile_img = Image.open(str(stitched_path)).convert("RGB")
    img_w, img_h = tile_img.size

    # Get polygon data
    dg, _ = coll.get(fid)
    n_sides = len(grid.faces[fid].vertex_ids)
    dg.compute_macro_edges(n_sides=n_sides)
    grid_corners_raw = get_macro_edge_corners(dg, n_sides)
    uv_corners = get_tile_uv_vertices(grid, fid)

    # Rotate grid corners from PolyGrid order into GoldbergTile order
    from polygrid.tile_uv_align import compute_uv_to_polygrid_offset
    offset = compute_uv_to_polygrid_offset(grid, fid)
    grid_corners = [grid_corners_raw[(k + offset) % n_sides] for k in range(n_sides)]

    xlim, ylim = compute_tile_view_limits(composite, fid)
    corners_px = compute_polygon_corners_px(
        grid_corners, xlim, ylim, img_w, img_h,
    )

    # ── Diagnostic 1: stitched tile with polygon outline ──
    diag1 = tile_img.copy()
    draw1 = ImageDraw.Draw(diag1)
    poly_pts = [(int(round(x)), int(round(y))) for x, y in corners_px]
    draw1.polygon(poly_pts, outline="red", width=2)
    for i, (px, py) in enumerate(poly_pts):
        draw1.text((px + 3, py - 12), str(i), fill="red")
    diag1.save(str(output_dir / f"{fid}_01_polygon_outline.png"))
    print(f"  → {fid}_01_polygon_outline.png")

    # ── Diagnostic 2: warped tile ──
    gutter = 4
    affine = compute_grid_to_px_affine(
        grid_corners, uv_corners,
        tile_size=args.tile_size, gutter=gutter,
    )
    slot_size = args.tile_size + 2 * gutter
    warped = warp_tile_to_uv(
        tile_img, xlim, ylim, affine, slot_size,
        grid_corners=grid_corners,
        uv_corners=uv_corners,
        tile_size=args.tile_size,
        gutter=gutter,
    )

    # Draw UV polygon on warped image
    diag2 = warped.copy()
    draw2 = ImageDraw.Draw(diag2)
    uv_px = []
    for u, v in uv_corners:
        px_x = gutter + u * args.tile_size
        px_y = gutter + (1.0 - v) * args.tile_size
        uv_px.append((int(round(px_x)), int(round(px_y))))
    draw2.polygon(uv_px, outline="cyan", width=2)
    for i, (px, py) in enumerate(uv_px):
        draw2.text((px + 3, py - 12), str(i), fill="cyan")
    diag2.save(str(output_dir / f"{fid}_02_warped.png"))
    print(f"  → {fid}_02_warped.png")

    # ── Diagnostic 3: affine accuracy report ──
    affine_uv = compute_grid_to_uv_affine(grid_corners, uv_corners)
    src = np.array(grid_corners)
    predicted = np.column_stack([src, np.ones(len(src))]) @ affine_uv.T

    print(f"\n  {fid} ({n_sides} sides):")
    print(f"  Image size: {img_w}×{img_h}")
    print(f"  xlim: {xlim}, ylim: {ylim}")
    print(f"  Grid corners: {[(round(x,2), round(y,2)) for x,y in grid_corners]}")
    print(f"  UV corners:   {[(round(u,3), round(v,3)) for u,v in uv_corners]}")
    print(f"  Pixel corners: {poly_pts}")
    print(f"\n  Corner mapping (grid → predicted UV → actual UV):")
    from scipy.spatial.distance import cdist
    dst = np.array(uv_corners)
    D = cdist(predicted, dst)
    for i in range(len(predicted)):
        best_j = D[i].argmin()
        err = D[i, best_j]
        print(f"    {i}: ({predicted[i][0]:.4f}, {predicted[i][1]:.4f}) "
              f"→ uv[{best_j}]=({dst[best_j][0]:.4f}, {dst[best_j][1]:.4f}) "
              f"err={err:.6f}")

    max_err = max(D[i, :].min() for i in range(len(predicted)))
    print(f"  Max corner error: {max_err:.6f}")

    print(f"\nDone. Diagnostics in {output_dir}/")


if __name__ == "__main__":
    main()
