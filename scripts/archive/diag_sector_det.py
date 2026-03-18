#!/usr/bin/env python3
"""Diagnostic: check affine determinants in sector warp for pentagon vs hexagon.

Hypothesis: hexagon sector affines have det < 0 (reflection),
            pentagon sector affines have det > 0 (no reflection).
"""
import sys, numpy as np
sys.path.insert(0, "src")

from polygrid.globe import build_globe_grid
from polygrid.tile_detail import TileDetailSpec, DetailGridCollection, build_tile_with_neighbours
from polygrid.tile_uv_align import (
    match_grid_corners_to_uv, compute_tile_view_limits,
    get_macro_edge_corners, _build_sector_affines, _signed_area_2d,
)
from polygrid.uv_texture import get_tile_uv_vertices
from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
from polygrid.mountains import MountainConfig, generate_mountains

# Build the same grid that the render script uses
gg = build_globe_grid(3)
schema = TileSchema([FieldDef("elevation", float, 0.0)])
store = TileDataStore(grid=gg, schema=schema)
spec = TileDetailSpec(detail_rings=4)
coll = DetailGridCollection.build(gg, spec)

face_ids = list(gg.faces.keys())
poly = gg.polyhedron
tile_map = {f"t{t.id}": t for t in poly.tiles}
tile_size = 512
gutter = 4

# Check a few pentagons and a few hexagons
test_fids = ["t0", "t6", "t9", "t1", "t2", "t3"]
for fid in test_fids:
    face = gg.faces[fid]
    n_sides = len(face.vertex_ids)
    is_pent = n_sides == 5

    composite = build_tile_with_neighbours(coll, fid, gg)
    xlim, ylim = compute_tile_view_limits(composite, fid)

    dg = coll.get(fid)[0]
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)
    gc_raw = get_macro_edge_corners(dg, n_sides)
    uv_c = get_tile_uv_vertices(gg, fid)
    gc_matched = match_grid_corners_to_uv(gc_raw, gg, fid)

    x_min, x_max = xlim
    y_min, y_max = ylim
    x_span, y_span = x_max - x_min, y_max - y_min
    img_w = img_h = tile_size
    n = len(uv_c)

    src_grid = np.array(gc_matched, dtype=np.float64)
    dst_uv = np.array(uv_c, dtype=np.float64)

    src_px = np.empty_like(src_grid)
    for i in range(n):
        gx, gy = src_grid[i]
        src_px[i, 0] = (gx - x_min) / x_span * img_w
        src_px[i, 1] = (1.0 - (gy - y_min) / y_span) * img_h
    src_px_c = src_px.mean(axis=0)

    dst_px = np.empty_like(dst_uv)
    for i in range(n):
        u, v = dst_uv[i]
        dst_px[i, 0] = gutter + u * tile_size
        dst_px[i, 1] = gutter + (1.0 - v) * tile_size
    dst_px_c = dst_px.mean(axis=0)

    da = np.arctan2(dst_px[:, 1] - dst_px_c[1], dst_px[:, 0] - dst_px_c[0])
    order = np.argsort(da)
    src_o = src_px[order]
    dst_o = dst_px[order]

    # Build INVERSE sector affines: dst_px -> src_px
    inv_sec = _build_sector_affines(dst_o, dst_px_c, src_o, src_px_c)

    sa_src = _signed_area_2d(list(map(tuple, src_o)))
    sa_dst = _signed_area_2d(list(map(tuple, dst_o)))

    label = "PENT" if is_pent else "HEX"
    print(f"=== {fid} ({label}) ===")
    print(f"  src_px winding: {sa_src:+.1f} ({'CCW' if sa_src > 0 else 'CW'})")
    print(f"  dst_px winding: {sa_dst:+.1f} ({'CCW' if sa_dst > 0 else 'CW'})")

    dets = [np.linalg.det(A) for A, _ in inv_sec]
    signs = ["REFL" if d < 0 else "ROT" for d in dets]
    print(f"  sector dets: {', '.join(f'{d:+.4f}' for d in dets)}")
    print(f"  sector types: {', '.join(signs)}")
    print()
