#!/usr/bin/env python3
"""Check: for a single tile, does the warped atlas content along each UV edge
correspond to the composite content along the matching macro-edge?

This tests whether the warp correctly maps macro-edge k → UV edge k.
"""
import sys
import math
import json
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, "src")

from polygrid.globe import build_globe_grid
from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
from polygrid.detail_terrain import compute_neighbor_edge_mapping
from polygrid.tile_uv_align import (
    compute_pg_to_macro_edge_map, get_macro_edge_corners,
    match_grid_corners_to_uv, compute_tile_view_limits,
)
from polygrid.uv_texture import get_goldberg_tiles, _match_tile_to_face, get_tile_uv_vertices
from polygrid.tile_detail import build_tile_with_neighbours
from polygrid.composite import stitch_grids

EXPORTS = Path("exports/f3")

gg = build_globe_grid(3)
spec = TileDetailSpec(detail_rings=4)
coll = DetailGridCollection.build(gg, spec)
tiles_3d = get_goldberg_tiles(3, 1.0)

atlas = Image.open(EXPORTS / "atlas.png").convert("RGB")
atlas_arr = np.array(atlas)
aw, ah = atlas.size

with open(EXPORTS / "uv_layout.json") as f:
    uv_layout = json.load(f)

with open(EXPORTS / "metadata.json") as f:
    metadata = json.load(f)

tile_size = metadata.get("tile_size", 512)
gutter = metadata.get("gutter", 4)


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


def sample_atlas_uv_edge(fid, edge_m, n_samples=50, offset_inward=3):
    """Sample atlas pixels slightly inward from UV edge m."""
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
        ax = inner_x0 + u * inner_w
        ay = inner_y0 + (1.0 - v) * inner_h
        return ax, ay
    
    ax0, ay0 = uv_to_atlas(u0, v0)
    ax1, ay1 = uv_to_atlas(u1, v1)
    
    # Edge normal (pointing toward centroid)
    cu = sum(c[0] for c in uv_corners) / n
    cv = sum(c[1] for c in uv_corners) / n
    cax, cay = uv_to_atlas(cu, cv)
    
    emx = (ax0 + ax1) / 2
    emy = (ay0 + ay1) / 2
    inward_x = cax - emx
    inward_y = cay - emy
    inward_len = math.sqrt(inward_x**2 + inward_y**2)
    if inward_len > 0:
        inward_x /= inward_len
        inward_y /= inward_len
    
    pixels = []
    for i in range(n_samples):
        t = (i + 0.5) / n_samples
        ax = ax0 + t * (ax1 - ax0) + offset_inward * inward_x
        ay = ay0 + t * (ay1 - ay0) + offset_inward * inward_y
        ix = int(np.clip(ax, 0, aw - 1))
        iy = int(np.clip(ay, 0, ah - 1))
        pixels.append(atlas_arr[iy, ix])
    
    return np.array(pixels, dtype=np.float64)


# For a pair of adjacent tiles, compare their atlas edge pixel strips
print("=" * 70)
print("ATLAS UV EDGE COMPARISON (cross-tile)")
print("=" * 70)
print()
print("For each adjacent pair, sample ~3px inward from each tile's UV edge.")
print("The REV diff should be low if edges are correctly aligned.")
print()

all_diffs = []
for fid_a in sorted(gg.faces.keys(), key=lambda x: int(x[1:])):
    gt_neigh_a = get_gt_neigh(fid_a)
    n_a = len(gg.faces[fid_a].vertex_ids)
    
    for m_a, fid_b in gt_neigh_a.items():
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
        
        pix_a = sample_atlas_uv_edge(fid_a, m_a, n_samples=50, offset_inward=3)
        pix_b = sample_atlas_uv_edge(fid_b, m_b, n_samples=50, offset_inward=3)
        
        # B's edge is traversed in reverse relative to A
        diff_fwd = np.mean(np.abs(pix_a - pix_b))
        diff_rev = np.mean(np.abs(pix_a - pix_b[::-1]))
        
        n_b = len(gg.faces[fid_b].vertex_ids)
        kind = ("P" if n_a == 5 else "H") + "-" + ("P" if n_b == 5 else "H")
        all_diffs.append((fid_a, m_a, fid_b, m_b, diff_fwd, diff_rev, kind))

all_diffs.sort(key=lambda x: min(x[4], x[5]))

print(f"Total pairs: {len(all_diffs)}")
print()

# Show best 15
print("Best 15 (lowest diff):")
print(f"  {'A':>4} {'eA':>3} {'B':>4} {'eB':>3} {'fwd':>8} {'rev':>8} {'best':>6} {'kind':>5}")
for a, ma, b, mb, df, dr, k in all_diffs[:15]:
    best = "FWD" if df < dr else "REV"
    print(f"  {a:>4} {ma:>3} {b:>4} {mb:>3} {df:>8.1f} {dr:>8.1f} {best:>6} {k:>5}")

print()
print("Worst 15 (highest diff):")
for a, ma, b, mb, df, dr, k in all_diffs[-15:]:
    best = "FWD" if df < dr else "REV"
    print(f"  {a:>4} {ma:>3} {b:>4} {mb:>3} {df:>8.1f} {dr:>8.1f} {best:>6} {k:>5}")

# Stats
from collections import defaultdict
by_kind = defaultdict(list)
for _, _, _, _, df, dr, k in all_diffs:
    by_kind[k].append(min(df, dr))

print(f"\nBy pair type:")
for kind in sorted(by_kind):
    vals = by_kind[kind]
    print(f"  {kind}: mean={np.mean(vals):.1f} median={np.median(vals):.1f} "
          f"max={np.max(vals):.1f} n={len(vals)}")

# Direction stats
fwd_better = sum(1 for _, _, _, _, df, dr, _ in all_diffs if df < dr)
rev_better = len(all_diffs) - fwd_better
print(f"\nDirection: FWD better in {fwd_better}, REV better in {rev_better}")
