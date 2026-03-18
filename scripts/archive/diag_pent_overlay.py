#!/usr/bin/env python3
"""Diagnostic: overlay polygon boundary on stitched & warped pentagon tiles.

Saves annotated images showing the polygon boundary, corner numbers, and
grid structure to help identify visual distortion.
"""
import sys, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from polygrid.globe import build_globe_grid
from polygrid.mountains import MountainConfig, generate_mountains
from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
from polygrid.tile_detail import (
    TileDetailSpec, DetailGridCollection, build_tile_with_neighbours,
)
from polygrid.detail_terrain import generate_all_detail_terrain
from polygrid.tile_uv_align import (
    compute_tile_view_limits,
    match_grid_corners_to_uv,
    get_macro_edge_corners,
    compute_polygon_corners_px,
)
from polygrid.uv_texture import get_tile_uv_vertices


def main():
    out_dir = ROOT / "exports" / "f3"

    grid = build_globe_grid(3)
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)
    generate_mountains(grid, store, MountainConfig(seed=42))

    spec = TileDetailSpec(detail_rings=4)
    coll = DetailGridCollection.build(grid, spec)
    generate_all_detail_terrain(coll, grid, store, seed=42)

    face_id = "t0"
    n_sides = len(grid.faces[face_id].vertex_ids)

    # Build composite & get corners
    composite = build_tile_with_neighbours(coll, face_id, grid)
    dg = coll.get(face_id)[0]
    corner_ids = dg.metadata.get("corner_vertex_ids")
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)
    grid_corners_raw = get_macro_edge_corners(dg, n_sides)
    uv_corners = get_tile_uv_vertices(grid, face_id)
    grid_corners = match_grid_corners_to_uv(grid_corners_raw, grid, face_id)
    xlim, ylim = compute_tile_view_limits(composite, face_id)

    # ── Annotate the stitched tile ────────────────────────────────
    stitched_path = out_dir / f"{face_id}.png"
    img = Image.open(str(stitched_path)).convert("RGB")
    img_w, img_h = img.size

    # Map grid corners to pixel space
    corners_px = compute_polygon_corners_px(grid_corners, xlim, ylim, img_w, img_h)

    draw = ImageDraw.Draw(img)
    # Draw polygon outline
    poly_pts = [(int(x), int(y)) for x, y in corners_px]
    for i in range(n_sides):
        j = (i + 1) % n_sides
        draw.line([poly_pts[i], poly_pts[j]], fill="red", width=2)

    # Label corners
    for i, (px, py) in enumerate(poly_pts):
        draw.ellipse([px-4, py-4, px+4, py+4], fill="yellow")
        draw.text((px+6, py-6), str(i), fill="white")

    annotated_path = out_dir / f"{face_id}_annotated.png"
    img.save(str(annotated_path))
    print(f"→ Saved: {annotated_path}")

    # ── Annotate the warped tile ──────────────────────────────────
    warped_path = out_dir / "warped" / f"{face_id}_warped.png"
    if warped_path.exists():
        wimg = Image.open(str(warped_path)).convert("RGB")
        wdraw = ImageDraw.Draw(wimg)
        tile_size = 512
        gutter = 4
        # UV corners → pixel space in warped image
        for i in range(n_sides):
            u0, v0 = uv_corners[i]
            u1, v1 = uv_corners[(i+1) % n_sides]
            x0 = gutter + u0 * tile_size
            y0 = gutter + (1.0 - v0) * tile_size
            x1 = gutter + u1 * tile_size
            y1 = gutter + (1.0 - v1) * tile_size
            wdraw.line([(int(x0), int(y0)), (int(x1), int(y1))],
                      fill="red", width=2)

        for i, (u, v) in enumerate(uv_corners):
            px = int(gutter + u * tile_size)
            py = int(gutter + (1.0 - v) * tile_size)
            wdraw.ellipse([px-4, py-4, px+4, py+4], fill="yellow")
            wdraw.text((px+6, py-6), str(i), fill="white")

        warped_ann_path = out_dir / f"{face_id}_warped_annotated.png"
        wimg.save(str(warped_ann_path))
        print(f"→ Saved: {warped_ann_path}")

    # ── Also do a hex tile for comparison ─────────────────────────
    hex_id = "t1"
    n_sides_h = len(grid.faces[hex_id].vertex_ids)
    composite_h = build_tile_with_neighbours(coll, hex_id, grid)
    dg_h = coll.get(hex_id)[0]
    corner_ids_h = dg_h.metadata.get("corner_vertex_ids")
    dg_h.compute_macro_edges(n_sides=n_sides_h, corner_ids=corner_ids_h)
    gc_raw_h = get_macro_edge_corners(dg_h, n_sides_h)
    gc_h = match_grid_corners_to_uv(gc_raw_h, grid, hex_id)
    xlim_h, ylim_h = compute_tile_view_limits(composite_h, hex_id)

    hex_path = out_dir / f"{hex_id}.png"
    if hex_path.exists():
        himg = Image.open(str(hex_path)).convert("RGB")
        hw, hh = himg.size
        cpx_h = compute_polygon_corners_px(gc_h, xlim_h, ylim_h, hw, hh)
        hdraw = ImageDraw.Draw(himg)
        poly_pts_h = [(int(x), int(y)) for x, y in cpx_h]
        for i in range(n_sides_h):
            j = (i + 1) % n_sides_h
            hdraw.line([poly_pts_h[i], poly_pts_h[j]], fill="red", width=2)
        for i, (px, py) in enumerate(poly_pts_h):
            hdraw.ellipse([px-4, py-4, px+4, py+4], fill="yellow")
            hdraw.text((px+6, py-6), str(i), fill="white")
        hann_path = out_dir / f"{hex_id}_annotated.png"
        himg.save(str(hann_path))
        print(f"→ Saved: {hann_path}")

    print("\nDone. Check annotated images to compare pentagon vs hex polygon coverage.")


if __name__ == "__main__":
    main()
