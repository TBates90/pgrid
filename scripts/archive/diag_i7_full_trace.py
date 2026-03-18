#!/usr/bin/env python3
"""Trace the full pipeline for one specific edge pixel.

1. Take a point on the shared UV edge between t0 and t1
2. Map it to atlas pixel in tile t0's slot
3. Map it to atlas pixel in tile t1's slot  
4. Read the actual atlas RGB at both locations
5. Trace back through the warp to the source image pixel
"""
import sys
import json
import math
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, "src")

from polygrid.globe import build_globe_grid
from polygrid.uv_texture import get_tile_uv_vertices, get_goldberg_tiles, _match_tile_to_face
from polygrid.detail_terrain import compute_neighbor_edge_mapping

EXPORTS = Path("exports/f3")
atlas = Image.open(EXPORTS / "atlas.png").convert("RGB")
atlas_arr = np.array(atlas)
aw, ah = atlas.size

with open(EXPORTS / "uv_layout.json") as f:
    uv_layout = json.load(f)

gg = build_globe_grid(3)

# Find shared edge between t0 and t1
uv_t0 = get_tile_uv_vertices(gg, "t0")  # 5 corners for pentagon
uv_t1 = get_tile_uv_vertices(gg, "t1")  # 6 corners for hex

tiles_3d = get_goldberg_tiles(3, 1.0)

def _qv(v, prec=6):
    return tuple(round(c, prec) for c in v)

vert_to_tiles = {}
for t in tiles_3d:
    for vtx in t.vertices:
        vert_to_tiles.setdefault(_qv(vtx), []).append(t.index)

tile_t0 = _match_tile_to_face(tiles_3d, "t0")
tile_t1 = _match_tile_to_face(tiles_3d, "t1")

# Find shared 3D vertices
verts_t0 = [_qv(v) for v in tile_t0.vertices]
verts_t1 = [_qv(v) for v in tile_t1.vertices]

# Find the shared edge (2 shared vertices)
shared_verts = []
for i, v0 in enumerate(verts_t0):
    for j, v1 in enumerate(verts_t1):
        if v0 == v1:
            shared_verts.append((i, j))

print(f"Shared vertices between t0 and t1: {shared_verts}")
print(f"  t0 has {len(uv_t0)} UV corners, t1 has {len(uv_t1)} UV corners")

# The shared edge in UV space
for (i0, i1) in shared_verts:
    print(f"  t0 corner {i0}: UV=({uv_t0[i0][0]:.4f}, {uv_t0[i0][1]:.4f})")
    print(f"  t1 corner {i1}: UV=({uv_t1[i1][0]:.4f}, {uv_t1[i1][1]:.4f})")

# Take the midpoint of the shared UV edge in t0 and t1
if len(shared_verts) >= 2:
    (i0a, i1a), (i0b, i1b) = shared_verts[0], shared_verts[1]
    
    # t0's shared edge goes from corner i0a to i0b
    # t1's shared edge goes from corner i1a to i1b
    
    # Midpoint in t0's UV
    t0_mid_u = (uv_t0[i0a][0] + uv_t0[i0b][0]) / 2
    t0_mid_v = (uv_t0[i0a][1] + uv_t0[i0b][1]) / 2
    
    # Midpoint in t1's UV
    t1_mid_u = (uv_t1[i1a][0] + uv_t1[i1b][0]) / 2
    t1_mid_v = (uv_t1[i1a][1] + uv_t1[i1b][1]) / 2
    
    print(f"\nShared edge midpoint:")
    print(f"  in t0 UV: ({t0_mid_u:.4f}, {t0_mid_v:.4f})")
    print(f"  in t1 UV: ({t1_mid_u:.4f}, {t1_mid_v:.4f})")
    
    # Map to atlas pixels
    def uv_to_atlas_px(fid, u, v):
        slot = uv_layout[fid]
        su0, sv0, su1, sv1 = slot
        # u,v are in [0,1] tile-local UV space
        # Atlas inner region
        inner_x0 = su0 * aw
        inner_y0 = (1.0 - sv1) * ah
        inner_w = (su1 - su0) * aw
        inner_h = (sv1 - sv0) * ah
        # Map tile UV to atlas pixel
        ax = inner_x0 + u * inner_w
        ay = inner_y0 + (1.0 - v) * inner_h
        return ax, ay
    
    ax0, ay0 = uv_to_atlas_px("t0", t0_mid_u, t0_mid_v)
    ax1, ay1 = uv_to_atlas_px("t1", t1_mid_u, t1_mid_v)
    
    print(f"  Atlas pixel for t0: ({ax0:.1f}, {ay0:.1f})")
    print(f"  Atlas pixel for t1: ({ax1:.1f}, {ay1:.1f})")
    
    # Read atlas colours at these points
    def read_atlas(x, y):
        ix = int(np.clip(x, 0, aw-1))
        iy = int(np.clip(y, 0, ah-1))
        return atlas_arr[iy, ix]
    
    rgb0 = read_atlas(ax0, ay0)
    rgb1 = read_atlas(ax1, ay1)
    
    print(f"\n  Atlas RGB at t0's edge: {rgb0}")
    print(f"  Atlas RGB at t1's edge: {rgb1}")
    print(f"  Diff: {np.abs(rgb0.astype(int) - rgb1.astype(int))}")
    
    # Also check the warped tiles directly
    warped_t0 = Image.open(EXPORTS / "warped" / "t0_warped.png").convert("RGB")
    warped_t1 = Image.open(EXPORTS / "warped" / "t1_warped.png").convert("RGB")
    wt0_arr = np.array(warped_t0)
    wt1_arr = np.array(warped_t1)
    
    # The warped tile is slot_size x slot_size with gutter=4
    gutter = 4
    tile_size = 512
    slot_size = tile_size + 2 * gutter
    
    # UV to warped pixel
    def uv_to_warped_px(u, v):
        wx = gutter + u * tile_size
        wy = gutter + (1.0 - v) * tile_size
        return wx, wy
    
    wx0, wy0 = uv_to_warped_px(t0_mid_u, t0_mid_v)
    wx1, wy1 = uv_to_warped_px(t1_mid_u, t1_mid_v)
    
    print(f"\n  Warped pixel for t0: ({wx0:.1f}, {wy0:.1f})")
    print(f"  Warped pixel for t1: ({wx1:.1f}, {wy1:.1f})")
    
    wrgb0 = wt0_arr[int(wy0), int(wx0)]
    wrgb1 = wt1_arr[int(wy1), int(wx1)]
    
    print(f"  Warped RGB at t0's edge: {wrgb0}")
    print(f"  Warped RGB at t1's edge: {wrgb1}")
    print(f"  Diff: {np.abs(wrgb0.astype(int) - wrgb1.astype(int))}")
    
    # Also check the raw tile images
    raw_t0 = Image.open(EXPORTS / "t0.png").convert("RGB")
    raw_t1 = Image.open(EXPORTS / "t1.png").convert("RGB")
    raw_t0_arr = np.array(raw_t0)
    raw_t1_arr = np.array(raw_t1)
    
    print(f"\n  Raw t0 image size: {raw_t0.size}")
    print(f"  Raw t1 image size: {raw_t1.size}")
    
    # Sample a few points along the shared edge
    print("\n  Sampling along shared edge:")
    for t_param in np.linspace(0.1, 0.9, 5):
        u0 = uv_t0[i0a][0] + t_param * (uv_t0[i0b][0] - uv_t0[i0a][0])
        v0 = uv_t0[i0a][1] + t_param * (uv_t0[i0b][1] - uv_t0[i0a][1])
        u1 = uv_t1[i1a][0] + t_param * (uv_t1[i1b][0] - uv_t1[i1a][0])
        v1 = uv_t1[i1a][1] + t_param * (uv_t1[i1b][1] - uv_t1[i1a][1])
        
        wpx0, wpy0 = uv_to_warped_px(u0, v0)
        wpx1, wpy1 = uv_to_warped_px(u1, v1)
        
        c0 = wt0_arr[int(np.clip(wpy0, 0, slot_size-1)), int(np.clip(wpx0, 0, slot_size-1))]
        c1 = wt1_arr[int(np.clip(wpy1, 0, slot_size-1)), int(np.clip(wpx1, 0, slot_size-1))]
        diff = np.mean(np.abs(c0.astype(int) - c1.astype(int)))
        print(f"    t={t_param:.1f}: t0_rgb={c0}, t1_rgb={c1}, diff={diff:.1f}")
