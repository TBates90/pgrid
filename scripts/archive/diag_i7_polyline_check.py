#!/usr/bin/env python3
"""Check that macro-edge polylines are being extracted and refined."""
import sys
sys.path.insert(0, "src")

from polygrid.globe import build_globe_grid
from polygrid.detail_grid import build_detail_grid
from polygrid.tile_uv_align import (
    get_macro_edge_corners, get_macro_edge_polylines,
    match_grid_corners_to_uv,
)
from polygrid.uv_texture import get_tile_uv_vertices
import numpy as np

gg = build_globe_grid(3)

for fid in ["t0", "t1", "t3"]:
    n_sides = len(gg.faces[fid].vertex_ids)
    dg = build_detail_grid(gg, fid, detail_rings=4)
    is_pent = n_sides == 5
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)

    corners = get_macro_edge_corners(dg, n_sides)
    polylines = get_macro_edge_polylines(dg, n_sides)

    print(f"\n{fid} ({n_sides} sides):")
    for k in range(n_sides):
        pts = polylines[k]
        print(f"  edge {k}: {len(pts)} vertices, "
              f"start=({pts[0][0]:.4f},{pts[0][1]:.4f}) "
              f"end=({pts[-1][0]:.4f},{pts[-1][1]:.4f})  "
              f"corner=({corners[k][0]:.4f},{corners[k][1]:.4f})")

    # Check matching
    corners_matched, perm = match_grid_corners_to_uv(
        corners, gg, fid, detail_grid=dg, return_permutation=True,
    )
    print(f"  permutation: {perm}")
    print(f"  corners_matched: {[(f'{c[0]:.4f},{c[1]:.4f}') for c in corners_matched]}")

    # Check polyline reordering
    polylines_matched = [None] * n_sides
    for m in range(n_sides):
        k = perm[m]
        if k is not None:
            polylines_matched[m] = polylines[k]

    for m in range(n_sides):
        if polylines_matched[m] is not None:
            p = polylines_matched[m]
            print(f"  matched edge {m}: {len(p)} pts, "
                  f"start=({p[0][0]:.4f},{p[0][1]:.4f})")
