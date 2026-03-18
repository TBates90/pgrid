#!/usr/bin/env python3
"""Sample actual atlas pixels along shared UV edges to measure seam quality.

For each adjacent tile pair, sample pixels along both tiles' UV edges
(the shared boundary) and compute the mean absolute difference.
"""
import sys
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
    match_grid_corners_to_uv,
)
from polygrid.uv_texture import (
    get_goldberg_tiles, _match_tile_to_face, get_tile_uv_vertices,
)

EXPORTS = Path("exports/f3")
ATLAS_PATH = EXPORTS / "atlas.png"
UV_LAYOUT_PATH = EXPORTS / "uv_layout.json"

atlas = Image.open(ATLAS_PATH).convert("RGB")
atlas_arr = np.array(atlas)

with open(UV_LAYOUT_PATH) as f:
    uv_layout = json.load(f)

gg = build_globe_grid(3)
spec = TileDetailSpec(detail_rings=4)
coll = DetailGridCollection.build(gg, spec)
tiles_3d = get_goldberg_tiles(3, 1.0)


def _qv(v, prec=6):
    return tuple(round(c, prec) for c in v)


vert_to_tiles = {}
for t in tiles_3d:
    for vtx in t.vertices:
        vert_to_tiles.setdefault(_qv(vtx), []).append(t.index)


def get_atlas_slot(fid):
    """Get the atlas slot info for a tile."""
    slot = uv_layout[fid]
    # slot is [u0, v0, u1, v1] in normalised atlas coords
    aw, ah = atlas.size
    x0 = int(slot[0] * aw)
    y0 = int((1.0 - slot[3]) * ah)  # v is bottom-up, y is top-down
    x1 = int(slot[2] * aw)
    y1 = int((1.0 - slot[1]) * ah)
    return x0, y0, x1, y1


def sample_uv_edge(fid, edge_m, n_samples=50):
    """Sample pixels along UV edge m of tile fid in the atlas.
    
    UV edge m goes from uv_corners[m] to uv_corners[(m+1)%n].
    """
    uv_corners = get_tile_uv_vertices(gg, fid)
    n = len(uv_corners)
    u0, v0 = uv_corners[edge_m]
    u1, v1 = uv_corners[(edge_m + 1) % n]
    
    slot = uv_layout[fid]
    # slot: [u_min, v_min, u_max, v_max] in atlas normalised coords
    aw, ah = atlas.size
    
    # UV → atlas pixel coords
    # UV (0,0) is bottom-left of the tile slot; (1,1) is top-right
    # Atlas uses u_min..u_max horizontal, v_min..v_max vertical
    # Need to find the tile's pixel region in the atlas
    slot_u0, slot_v0, slot_u1, slot_v1 = slot
    slot_pw = (slot_u1 - slot_u0) * aw  # pixel width of slot
    slot_ph = (slot_v1 - slot_v0) * ah  # pixel height of slot
    
    pixels = []
    for i in range(n_samples):
        t = (i + 0.5) / n_samples
        u = u0 + t * (u1 - u0)
        v = v0 + t * (v1 - v0)
        
        # UV → atlas pixel
        px_x = slot_u0 * aw + u * slot_pw
        px_y = (1.0 - (slot_v0 + v * slot_ph / ah * 1)) # wrong
        
        # Actually let's be more careful.
        # The uv_layout gives normalised atlas coordinates.
        # UV corners are in [0,1]x[0,1] tile-local space.
        # In the atlas builder, the mapping is:
        #   atlas_x = slot_x + gutter + u * tile_size
        #   atlas_y = slot_y + gutter + (1 - v) * tile_size
        # where slot_x, slot_y are the top-left of the slot in atlas pixels.
        # The uv_layout stores:
        #   u_min = (slot_x + gutter) / atlas_w
        #   v_min = 1 - (slot_y + gutter + tile_size) / atlas_h
        #   u_max = (slot_x + gutter + tile_size) / atlas_w
        #   v_max = 1 - (slot_y + gutter) / atlas_h
        # So to convert UV to atlas pixels:
        #   atlas_x = u_min * aw + u * tile_size_px
        #   atlas_y = (1 - v_max) * ah + (1 - v) * tile_size_px
        
        tile_size_px_w = (slot_u1 - slot_u0) * aw
        tile_size_px_h = (slot_v1 - slot_v0) * ah
        
        ax = slot_u0 * aw + u * tile_size_px_w
        ay = (1.0 - slot_v1) * ah + (1.0 - v) * tile_size_px_h
        
        ix = int(np.clip(ax, 0, aw - 1))
        iy = int(np.clip(ay, 0, ah - 1))
        pixels.append(atlas_arr[iy, ix])
    
    return np.array(pixels, dtype=np.float64)


def get_gt_edge_neigh(fid):
    """GT edge → neighbour face_id."""
    tile = _match_tile_to_face(tiles_3d, fid)
    n = len(tile.vertices)
    result = {}
    for m in range(n):
        va = _qv(tile.vertices[m])
        vb = _qv(tile.vertices[(m + 1) % n])
        shared = (set(vert_to_tiles.get(va, []))
                  & set(vert_to_tiles.get(vb, []))
                  - {tile.index})
        if shared:
            result[m] = f"t{shared.pop()}"
    return result


print("=" * 70)
print("ATLAS PIXEL COMPARISON AT SHARED EDGES")
print("=" * 70)

# Sample edge pixels for all adjacent pairs
diffs = []
pair_data = []

for fid_a in sorted(gg.faces.keys(), key=lambda x: int(x[1:])):
    gt_neigh_a = get_gt_edge_neigh(fid_a)
    for m_a, fid_b in gt_neigh_a.items():
        if int(fid_a[1:]) >= int(fid_b[1:]):
            continue
        
        gt_neigh_b = get_gt_edge_neigh(fid_b)
        m_b = None
        for m, nfid in gt_neigh_b.items():
            if nfid == fid_a:
                m_b = m
                break
        if m_b is None:
            continue
        
        try:
            pix_a = sample_uv_edge(fid_a, m_a, n_samples=30)
            pix_b = sample_uv_edge(fid_b, m_b, n_samples=30)
            
            # Edge B should be traversed in reverse order relative to A
            pix_b_rev = pix_b[::-1]
            
            diff_fwd = np.mean(np.abs(pix_a - pix_b))
            diff_rev = np.mean(np.abs(pix_a - pix_b_rev))
            
            best_diff = min(diff_fwd, diff_rev)
            direction = "FWD" if diff_fwd <= diff_rev else "REV"
            
            n_a = len(gg.faces[fid_a].vertex_ids)
            n_b = len(gg.faces[fid_b].vertex_ids)
            kind_a = "P" if n_a == 5 else "H"
            kind_b = "P" if n_b == 5 else "H"
            pair_kind = f"{kind_a}-{kind_b}"
            
            pair_data.append((fid_a, m_a, fid_b, m_b, diff_fwd, diff_rev, 
                            direction, pair_kind))
            diffs.append(best_diff)
        except Exception as e:
            print(f"  Error: {fid_a} edge {m_a} ↔ {fid_b} edge {m_b}: {e}")

# Sort by diff (worst first)
pair_data.sort(key=lambda x: -min(x[4], x[5]))

print(f"\nTotal pairs: {len(pair_data)}")
print(f"\nWorst 20 edges:")
print(f"  {'A':>4} {'eA':>3} {'B':>4} {'eB':>3} {'diff_fwd':>9} {'diff_rev':>9} {'best':>5} {'kind':>5}")
for fid_a, m_a, fid_b, m_b, df, dr, d, kind in pair_data[:20]:
    best = min(df, dr)
    print(f"  {fid_a:>4} {m_a:>3} {fid_b:>4} {m_b:>3} {df:>9.1f} {dr:>9.1f} {d:>5} {kind:>5}")

print(f"\nBest 10 edges:")
for fid_a, m_a, fid_b, m_b, df, dr, d, kind in pair_data[-10:]:
    best = min(df, dr)
    print(f"  {fid_a:>4} {m_a:>3} {fid_b:>4} {m_b:>3} {df:>9.1f} {dr:>9.1f} {d:>5} {kind:>5}")

# Stats by pair kind
from collections import defaultdict
by_kind = defaultdict(list)
for _, _, _, _, df, dr, d, kind in pair_data:
    by_kind[kind].append(min(df, dr))

print(f"\nMean diff by pair type:")
for kind in sorted(by_kind):
    vals = by_kind[kind]
    print(f"  {kind}: mean={np.mean(vals):.1f} median={np.median(vals):.1f} "
          f"max={np.max(vals):.1f} n={len(vals)}")

overall = np.array(diffs)
print(f"\nOverall: mean={np.mean(overall):.1f} median={np.median(overall):.1f} "
      f"max={np.max(overall):.1f}")
