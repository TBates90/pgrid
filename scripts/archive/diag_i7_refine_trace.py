#!/usr/bin/env python3
"""Trace the refined warp to check if polyline refinement works."""
import sys
sys.path.insert(0, "src")

import numpy as np
from polygrid.globe import build_globe_grid
from polygrid.detail_grid import build_detail_grid
from polygrid.tile_uv_align import (
    get_macro_edge_corners, get_macro_edge_polylines,
    match_grid_corners_to_uv,
    _compute_piecewise_warp_map,
    _refine_with_polylines,
)
from polygrid.uv_texture import get_tile_uv_vertices

gg = build_globe_grid(3)

tile_size = 512
gutter = 4

for fid in ["t0", "t1"]:
    n_sides = len(gg.faces[fid].vertex_ids)
    dg = build_detail_grid(gg, fid, detail_rings=4)
    is_pent = n_sides == 5
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)

    corners_raw = get_macro_edge_corners(dg, n_sides)
    polylines_raw = get_macro_edge_polylines(dg, n_sides)
    uv_corners_raw = get_tile_uv_vertices(gg, fid)

    corners_matched, perm = match_grid_corners_to_uv(
        corners_raw, gg, fid, detail_grid=dg, return_permutation=True,
    )
    polylines_matched = [None] * n_sides
    for m in range(n_sides):
        k = perm[m]
        if k is not None:
            polylines_matched[m] = polylines_raw[k]

    uv_corners = uv_corners_raw
    grid_corners = corners_matched

    print(f"\n{'='*60}")
    print(f"{fid}: {n_sides} sides, {n_sides} matched polylines")
    print(f"  Grid corners: {len(grid_corners)}")
    print(f"  UV corners: {len(uv_corners)}")

    # Simulate what _compute_piecewise_warp_map does
    src_grid = np.array(grid_corners, dtype=np.float64)
    dst_uv = np.array(uv_corners, dtype=np.float64)
    n = len(dst_uv)

    # Fake image dims for conversion functions
    img_w, img_h = 800, 800
    xlim = (-0.3, 0.3)
    ylim = (-0.3, 0.3)
    x_min, x_max = xlim
    y_min, y_max = ylim
    x_span = x_max - x_min
    y_span = y_max - y_min

    def grid_to_px(gx, gy):
        return (
            (gx - x_min) / x_span * img_w,
            (1.0 - (gy - y_min) / y_span) * img_h,
        )

    def uv_to_slot(u, v):
        return (
            gutter + u * tile_size,
            gutter + (1.0 - v) * tile_size,
        )

    src_px = np.array([grid_to_px(gx, gy) for gx, gy in src_grid])
    dst_px = np.array([uv_to_slot(u, v) for u, v in dst_uv])
    src_px_centroid = src_px.mean(axis=0)
    dst_px_centroid = dst_px.mean(axis=0)

    print(f"\n  Without polylines:")
    print(f"    src_px corners: {n}")
    print(f"    dst_px corners: {n}")

    # Test with polylines
    src_r, dst_r = _refine_with_polylines(
        src_px, dst_px, src_px_centroid, dst_px_centroid,
        polylines_matched, grid_to_px, uv_corners, uv_to_slot,
    )
    print(f"\n  With polylines:")
    print(f"    refined src pts: {len(src_r)}")
    print(f"    refined dst pts: {len(dst_r)}")

    # Check that refined points include the original corners
    print(f"\n  Refined src points (first few):")
    for i in range(min(20, len(src_r))):
        print(f"    [{i}] ({src_r[i,0]:.2f}, {src_r[i,1]:.2f}) -> ({dst_r[i,0]:.2f}, {dst_r[i,1]:.2f})")

    # Check angular ordering
    angles = np.arctan2(
        dst_r[:, 1] - dst_px_centroid[1],
        dst_r[:, 0] - dst_px_centroid[0],
    )
    sorted_check = all(angles[i] <= angles[i+1] for i in range(len(angles)-1))
    print(f"\n  Angles monotonically increasing: {sorted_check}")
    if not sorted_check:
        # Find violations
        for i in range(len(angles)-1):
            if angles[i] > angles[i+1]:
                print(f"    VIOLATION at {i}: {angles[i]:.4f} > {angles[i+1]:.4f}")
                if i < 5 or i > len(angles) - 5:
                    continue
                # Only print first few violations
                break
