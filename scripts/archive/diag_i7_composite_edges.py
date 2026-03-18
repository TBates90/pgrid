#!/usr/bin/env python3
"""Check composite tile edge content matches BEFORE the warp.

For adjacent tiles A and B, the composite for A includes B's content
(via stitching) at the macro-edge facing B.  Similarly, B's composite
includes A's content at the macro-edge facing A.

If stitching is correct, the content along these edges should be
identical (it's the same terrain from the same underlying grid).

This checks the PRE-WARP composite, not the atlas.
"""
import sys
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
from polygrid.uv_texture import get_goldberg_tiles, _match_tile_to_face

EXPORTS = Path("exports/f3")

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


def sample_composite_macro_edge(fid, macro_edge_k, n_samples=40):
    """Sample pixels along macro-edge k in the composite image for fid.
    
    Macro-edge k goes from gc_raw[k] to gc_raw[(k+1)%n] in grid coords.
    Convert to pixel coords using the composite's view limits.
    """
    img = Image.open(EXPORTS / f"{fid}.png").convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    
    n_sides = len(gg.faces[fid].vertex_ids)
    is_pent = n_sides == 5
    
    dg, _ = coll.get(fid)
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)
    gc_raw = get_macro_edge_corners(dg, n_sides)
    
    # Get view limits
    from polygrid.composite import CompositeGrid
    # We need the composites... but we don't have them loaded.
    # Instead, compute view limits from the detail grid collection.
    # Actually, just use the image size and gc_raw bounds.
    # The renderer uses compute_tile_view_limits which needs the composite.
    # Let's approximate: the image covers the bounding box of gc_raw + margin.
    
    # Actually, let's just read the metadata or compute from the DG.
    gc_arr = np.array(gc_raw)
    # The renderer uses a specific xlim/ylim. Let's compute it the same way.
    # From tile_uv_align.compute_tile_view_limits:
    # It uses comp.merged extent. We don't have comp, but the composite PNG
    # was rendered with those limits.
    
    # Alternative: compute from the detail grid directly.
    xs = [dg.vertices[vid].x for vid in dg.vertices]
    ys = [dg.vertices[vid].y for vid in dg.vertices]
    margin = 0.05 * max(max(xs) - min(xs), max(ys) - min(ys))
    xlim = (min(xs) - margin, max(xs) + margin)
    ylim = (min(ys) - margin, max(ys) + margin)
    
    # Grid coord → pixel coord
    def grid_to_px(gx, gy):
        px_x = (gx - xlim[0]) / (xlim[1] - xlim[0]) * w
        px_y = (1.0 - (gy - ylim[0]) / (ylim[1] - ylim[0])) * h
        return px_x, px_y
    
    gx0, gy0 = gc_raw[macro_edge_k]
    gx1, gy1 = gc_raw[(macro_edge_k + 1) % n_sides]
    
    px0 = grid_to_px(gx0, gy0)
    px1 = grid_to_px(gx1, gy1)
    
    pixels = []
    for i in range(n_samples):
        t = (i + 0.5) / n_samples
        px = px0[0] + t * (px1[0] - px0[0])
        py = px0[1] + t * (px1[1] - px0[1])
        ix = int(np.clip(px, 0, w - 1))
        iy = int(np.clip(py, 0, h - 1))
        pixels.append(arr[iy, ix])
    
    return np.array(pixels, dtype=np.float64)


def get_macro_neigh(fid):
    """macro-edge k → neighbour face_id."""
    n_sides = len(gg.faces[fid].vertex_ids)
    is_pent = n_sides == 5
    dg, _ = coll.get(fid)
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)
    
    neigh_map = compute_neighbor_edge_mapping(gg, fid)
    pg_to_macro = compute_pg_to_macro_edge_map(gg, fid, dg)
    
    result = {}
    for nfid, pe in neigh_map.items():
        me = pg_to_macro.get(pe)
        if me is not None:
            result[me] = nfid
    return result


print("=" * 70)
print("COMPOSITE PRE-WARP EDGE PIXEL COMPARISON")
print("=" * 70)

# Check a selection of pairs
test_pairs = []
for fid_a in sorted(gg.faces.keys(), key=lambda x: int(x[1:])):
    macro_neigh_a = get_macro_neigh(fid_a)
    for me_a, fid_b in macro_neigh_a.items():
        if int(fid_a[1:]) >= int(fid_b[1:]):
            continue
        
        macro_neigh_b = get_macro_neigh(fid_b)
        me_b = None
        for me, nfid in macro_neigh_b.items():
            if nfid == fid_a:
                me_b = me
                break
        if me_b is None:
            continue
        
        test_pairs.append((fid_a, me_a, fid_b, me_b))

print(f"\nTotal pairs: {len(test_pairs)}")

diffs = []
for fid_a, me_a, fid_b, me_b in test_pairs:
    try:
        pix_a = sample_composite_macro_edge(fid_a, me_a)
        pix_b = sample_composite_macro_edge(fid_b, me_b)
        
        diff_fwd = np.mean(np.abs(pix_a - pix_b))
        diff_rev = np.mean(np.abs(pix_a - pix_b[::-1]))
        
        n_a = len(gg.faces[fid_a].vertex_ids)
        n_b = len(gg.faces[fid_b].vertex_ids)
        kind = ("P" if n_a == 5 else "H") + "-" + ("P" if n_b == 5 else "H")
        
        diffs.append((fid_a, me_a, fid_b, me_b, diff_fwd, diff_rev, kind))
    except Exception as e:
        print(f"  Error: {fid_a} macro {me_a} ↔ {fid_b} macro {me_b}: {e}")

diffs.sort(key=lambda x: -min(x[4], x[5]))

print(f"\nWorst 15 composite edges:")
print(f"  {'A':>4} {'mA':>3} {'B':>4} {'mB':>3} {'fwd':>8} {'rev':>8} {'kind':>5}")
for a, ma, b, mb, df, dr, k in diffs[:15]:
    print(f"  {a:>4} {ma:>3} {b:>4} {mb:>3} {df:>8.1f} {dr:>8.1f} {k:>5}")

print(f"\nBest 10 composite edges:")
for a, ma, b, mb, df, dr, k in diffs[-10:]:
    print(f"  {a:>4} {ma:>3} {b:>4} {mb:>3} {df:>8.1f} {dr:>8.1f} {k:>5}")

# Stats by kind
from collections import defaultdict
by_kind = defaultdict(list)
for _, _, _, _, df, dr, k in diffs:
    by_kind[k].append(min(df, dr))

print(f"\nMean composite-edge diff by pair type:")
for kind in sorted(by_kind):
    vals = by_kind[kind]
    print(f"  {kind}: mean={np.mean(vals):.1f} median={np.median(vals):.1f} "
          f"max={np.max(vals):.1f} n={len(vals)}")
