#!/usr/bin/env python3
"""Pixel-level check: for t0/t1 shared edge, what colours appear at
the atlas boundary from each tile?"""
import sys, json, math
import numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, "src")

from polygrid.globe import build_globe_grid
from polygrid.uv_texture import get_tile_uv_vertices

EXPORTS = Path("exports/f3")
atlas = Image.open(EXPORTS / "atlas.png").convert("RGB")
arr = np.array(atlas)
aw, ah = atlas.size

with open(EXPORTS / "uv_layout.json") as f:
    uv_layout = json.load(f)

gg = build_globe_grid(3)


def get_edge_pixels(fid, edge_m, offset=0, n_samples=20):
    uv_corners = get_tile_uv_vertices(gg, fid)
    n = len(uv_corners)
    u0, v0 = uv_corners[edge_m]
    u1, v1 = uv_corners[(edge_m + 1) % n]
    
    slot = uv_layout[fid]
    su0, sv0, su1, sv1 = slot
    inner_x0 = su0 * aw
    inner_y0 = (1.0 - sv1) * ah
    inner_w = (su1 - su0) * aw
    inner_h = (sv1 - sv0) * ah

    def uv_to_atlas(u, v):
        return inner_x0 + u * inner_w, inner_y0 + (1.0 - v) * inner_h

    ax0, ay0 = uv_to_atlas(u0, v0)
    ax1, ay1 = uv_to_atlas(u1, v1)
    cu = sum(c[0] for c in uv_corners) / n
    cv = sum(c[1] for c in uv_corners) / n
    cax, cay = uv_to_atlas(cu, cv)
    emx, emy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    inx = cax - emx
    iny = cay - emy
    l = math.sqrt(inx**2 + iny**2)
    if l > 0: inx /= l; iny /= l

    pixels = []
    for i in range(n_samples):
        t = (i + 0.5) / n_samples
        ax = ax0 + t * (ax1 - ax0) + offset * inx
        ay = ay0 + t * (ay1 - ay0) + offset * iny
        ix = int(np.clip(ax, 0, aw - 1))
        iy = int(np.clip(ay, 0, ah - 1))
        pixels.append((t, arr[iy, ix].tolist(), ix, iy))
    return pixels


# Find the t0-t1 edge pair
from polygrid.uv_texture import get_goldberg_tiles, _match_tile_to_face
tiles_3d = get_goldberg_tiles(3, 1.0)

def _qv(v, prec=6):
    return tuple(round(c, prec) for c in v)

vert_to_tiles = {}
for t in tiles_3d:
    for vtx in t.vertices:
        vert_to_tiles.setdefault(_qv(vtx), []).append(t.index)

for fid in ["t0"]:
    tile = _match_tile_to_face(tiles_3d, fid)
    n = len(tile.vertices)
    for m in range(n):
        va = _qv(tile.vertices[m])
        vb = _qv(tile.vertices[(m + 1) % n])
        shared = (set(vert_to_tiles.get(va, [])) & set(vert_to_tiles.get(vb, []))) - {tile.index}
        if shared:
            neigh_idx = shared.pop()
            neigh_fid = f"t{neigh_idx}"
            # Find the matching edge on the neighbour
            neigh_tile = _match_tile_to_face(tiles_3d, neigh_fid)
            nn = len(neigh_tile.vertices)
            for nm in range(nn):
                nva = _qv(neigh_tile.vertices[nm])
                nvb = _qv(neigh_tile.vertices[(nm + 1) % nn])
                nshared = (set(vert_to_tiles.get(nva, [])) & set(vert_to_tiles.get(nvb, []))) - {neigh_tile.index}
                if nshared and f"t{nshared.pop()}" == fid:
                    # Found matching edge
                    pix_a = get_edge_pixels(fid, m, offset=0, n_samples=15)
                    pix_b = get_edge_pixels(neigh_fid, nm, offset=0, n_samples=15)
                    pix_b_rev = list(reversed(pix_b))
                    
                    print(f"\n{fid} edge {m} ↔ {neigh_fid} edge {nm}:")
                    print(f"  {'t':>4}  {'tile_a RGB':>12}  {'tile_b RGB':>12}  {'diff':>4}")
                    diffs = []
                    for (ta, ca, xa, ya), (tb, cb, xb, yb) in zip(pix_a, pix_b_rev):
                        d = sum(abs(a-b) for a,b in zip(ca, cb)) / 3
                        diffs.append(d)
                        print(f"  {ta:.3f}  {str(ca):>12}  {str(cb):>12}  {d:>4.0f}  "
                              f"({xa},{ya}) ({xb},{yb})")
                    print(f"  → mean diff: {np.mean(diffs):.1f}")
                    break
