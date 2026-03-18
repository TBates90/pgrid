#!/usr/bin/env python3
"""Baseline: what's the expected diff when sampling close to a UV edge?

Sample the SAME tile's UV edge at offset_inward=3 vs offset_inward=6
to establish baseline noise level.
"""
import sys
import json
import math
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, "src")

from polygrid.globe import build_globe_grid
from polygrid.uv_texture import get_tile_uv_vertices

EXPORTS = Path("exports/f3")

atlas = Image.open(EXPORTS / "atlas.png").convert("RGB")
atlas_arr = np.array(atlas)
aw, ah = atlas.size

with open(EXPORTS / "uv_layout.json") as f:
    uv_layout = json.load(f)

gg = build_globe_grid(3)


def sample_edge(fid, edge_m, offset, n_samples=50):
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
    if l > 0:
        inx /= l
        iny /= l
    
    pixels = []
    for i in range(n_samples):
        t = (i + 0.5) / n_samples
        ax = ax0 + t * (ax1 - ax0) + offset * inx
        ay = ay0 + t * (ay1 - ay0) + offset * iny
        ix = int(np.clip(ax, 0, aw - 1))
        iy = int(np.clip(ay, 0, ah - 1))
        pixels.append(atlas_arr[iy, ix])
    return np.array(pixels, dtype=np.float64)


# Same-tile comparison at different depths
print("Baseline: same tile, same edge, different inward offsets")
print("  This measures the natural gradient across the edge boundary.\n")

for fid in ["t0", "t1", "t3", "t6", "t10", "t20", "t50"]:
    n = len(gg.faces[fid].vertex_ids)
    diffs = []
    for m in range(n):
        p1 = sample_edge(fid, m, offset=1)
        p3 = sample_edge(fid, m, offset=3)
        p5 = sample_edge(fid, m, offset=5)
        p10 = sample_edge(fid, m, offset=10)
        d_1_3 = np.mean(np.abs(p1 - p3))
        d_3_5 = np.mean(np.abs(p3 - p5))
        d_1_5 = np.mean(np.abs(p1 - p5))
        d_1_10 = np.mean(np.abs(p1 - p10))
        diffs.append((m, d_1_3, d_3_5, d_1_5, d_1_10))
    
    kind = "PENT" if n == 5 else "HEX"
    print(f"  {fid} ({kind}):")
    for m, d13, d35, d15, d110 in diffs:
        print(f"    edge {m}: d(1,3)={d13:.1f}  d(3,5)={d35:.1f}  "
              f"d(1,5)={d15:.1f}  d(1,10)={d110:.1f}")

# Now check: sample at offset=0 (right at the edge) from both tiles
print("\n" + "=" * 70)
print("Cross-tile at offset=0 (right at UV boundary)")
print("=" * 70)

from polygrid.uv_texture import get_goldberg_tiles, _match_tile_to_face

tiles_3d = get_goldberg_tiles(3, 1.0)

def _qv(v, prec=6):
    return tuple(round(c, prec) for c in v)

vert_to_tiles = {}
for t in tiles_3d:
    for vtx in t.vertices:
        vert_to_tiles.setdefault(_qv(vtx), []).append(t.index)

def get_gt_neigh(fid):
    tile = _match_tile_to_face(tiles_3d, fid)
    n = len(tile.vertices)
    result = {}
    for m in range(n):
        va = _qv(tile.vertices[m])
        vb = _qv(tile.vertices[(m + 1) % n])
        shared = (set(vert_to_tiles.get(va, [])) & set(vert_to_tiles.get(vb, []))) - {tile.index}
        if shared:
            result[m] = f"t{shared.pop()}"
    return result


# Test at different offsets
for offset in [0, 1, 3, 5]:
    diffs_at_offset = []
    for fid_a in sorted(gg.faces.keys(), key=lambda x: int(x[1:])):
        gt_neigh = get_gt_neigh(fid_a)
        for m_a, fid_b in gt_neigh.items():
            if int(fid_a[1:]) >= int(fid_b[1:]):
                continue
            gt_neigh_b = get_gt_neigh(fid_b)
            m_b = None
            for m, nfid in gt_neigh_b.items():
                if nfid == fid_a:
                    m_b = m
                    break
            if m_b is None:
                continue
            
            pix_a = sample_edge(fid_a, m_a, offset=offset)
            pix_b = sample_edge(fid_b, m_b, offset=offset)
            diff_rev = np.mean(np.abs(pix_a - pix_b[::-1]))
            diffs_at_offset.append(diff_rev)
    
    arr = np.array(diffs_at_offset)
    print(f"  offset={offset:>2}: mean={np.mean(arr):.1f}  "
          f"median={np.median(arr):.1f}  max={np.max(arr):.1f}")
