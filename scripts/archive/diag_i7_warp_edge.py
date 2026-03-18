#!/usr/bin/env python3
"""Check whether the warp for a specific bad edge (t0-t2, edge 2/5)
is correctly following the polyline."""
import sys
import numpy as np
sys.path.insert(0, "src")

from polygrid.globe import build_globe_grid
from polygrid.detail_grid import build_detail_grid
from polygrid.tile_uv_align import (
    get_macro_edge_corners, get_macro_edge_polylines,
    match_grid_corners_to_uv,
    _compute_piecewise_warp_map,
    compute_tile_view_limits,
)
from polygrid.tile_detail import build_tile_with_neighbours, DetailGridCollection, TileDetailSpec
from polygrid.uv_texture import get_tile_uv_vertices

gg = build_globe_grid(3)
coll = DetailGridCollection.build(gg, TileDetailSpec(detail_rings=4))

tile_size = 512
gutter = 4
output_size = tile_size + 2 * gutter

# Build composites and get the warp for both t0 and t2
for fid in ["t0", "t2"]:
    n_sides = len(gg.faces[fid].vertex_ids)
    dg, _ = coll.get(fid)
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

    comp = build_tile_with_neighbours(coll, fid, gg)
    xlim, ylim = compute_tile_view_limits(comp, fid)

    # Build warp maps
    map_x, map_y = _compute_piecewise_warp_map(
        grid_corners, uv_corners,
        tile_size=tile_size,
        gutter=gutter,
        img_w=800, img_h=800,  # arbitrary for testing
        xlim=xlim, ylim=ylim,
        output_size=output_size,
        grid_edge_polylines=polylines_matched,
    )

    print(f"\n{fid}:")
    print(f"  xlim={xlim}, ylim={ylim}")
    print(f"  grid_corners={[(f'{c[0]:.4f},{c[1]:.4f}') for c in grid_corners]}")
    print(f"  uv_corners={[(f'{c[0]:.4f},{c[1]:.4f}') for c in uv_corners]}")

    # Check what the warp does at the UV edge boundary
    # For t0, the shared edge with t2 is edge 2 (GT ordering)
    # For t2, the shared edge with t0 is edge 5
    if fid == "t0":
        edge_m = 2
    else:
        edge_m = 5

    u0, v0 = uv_corners[edge_m]
    u1, v1 = uv_corners[(edge_m + 1) % n_sides]

    print(f"  Edge {edge_m}: UV ({u0:.4f},{v0:.4f}) → ({u1:.4f},{v1:.4f})")

    # Sample the warp at several points along this UV edge
    for i in range(10):
        t = (i + 0.5) / 10
        u = u0 + t * (u1 - u0)
        v = v0 + t * (v1 - v0)
        # Convert to output pixel coords
        ox = gutter + u * tile_size
        oy = gutter + (1.0 - v) * tile_size
        ix = int(np.clip(ox, 0, output_size - 1))
        iy = int(np.clip(oy, 0, output_size - 1))
        src_x = map_x[iy, ix]
        src_y = map_y[iy, ix]
        # Convert back to grid coords
        x_min, x_max = xlim
        y_min, y_max = ylim
        x_span = x_max - x_min
        y_span = y_max - y_min
        gx = x_min + src_x / 800 * x_span
        gy = y_min + (1.0 - src_y / 800) * y_span
        print(f"    t={t:.2f}: UV({u:.4f},{v:.4f}) → px({ix},{iy}) → "
              f"src_px({src_x:.1f},{src_y:.1f}) → grid({gx:.4f},{gy:.4f})")

    # Also print the polyline for this edge
    polyline = polylines_matched[edge_m]
    if polyline:
        print(f"  Polyline for edge {edge_m}: {len(polyline)} pts")
        for k, (px, py) in enumerate(polyline):
            print(f"    [{k}] ({px:.4f}, {py:.4f})")
