#!/usr/bin/env python3
"""Diagnostic: verify edge identity through the warp pipeline.

For each tile, check whether warp edge k (which maps gc_matched[k]→gc_matched[k+1] 
to uv[k]→uv[k+1]) corresponds to the SAME physical neighbour.

The key question: does the warp send the pixels near neighbour X
in the composite to the UV edge that also faces neighbour X?
"""
import sys, math
import numpy as np
from pathlib import Path

sys.path.insert(0, "src")

from polygrid.globe import build_globe_grid
from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
from polygrid.detail_terrain import compute_neighbor_edge_mapping
from polygrid.tile_uv_align import (
    compute_pg_to_macro_edge_map, get_macro_edge_corners,
    match_grid_corners_to_uv,
)
from polygrid.uv_texture import (
    get_tile_uv_vertices, get_goldberg_tiles, _match_tile_to_face,
    compute_tile_basis,
)

gg = build_globe_grid(3)
spec = TileDetailSpec(detail_rings=4)
coll = DetailGridCollection.build(gg, spec)

# Get GoldbergTile neighbour info for UV edges
tiles_3d = get_goldberg_tiles(3, 1.0)

print("=" * 70)
print("EDGE IDENTITY: macro-edge neighbour vs UV-edge neighbour")
print("=" * 70)
print()
print("For each tile, warp edge k maps composite pixels near macro-edge")
print("gc_raw[raw_k]→gc_raw[raw_k+1] to atlas UV edge uv[k]→uv[k+1].")
print("macro-edge raw_k faces some neighbour (from polygrid adjacency).")
print("UV edge k faces some neighbour (from GoldbergTile 3D adjacency).")
print("These MUST be the same neighbour for the globe seams to match.")
print()

for fid in ["t0", "t6", "t9", "t1", "t3"]:
    n_sides = len(gg.faces[fid].vertex_ids)
    is_pent = n_sides == 5
    kind = "PENT" if is_pent else "HEX"
    
    # Macro-edge neighbours
    dg, _ = coll.get(fid)
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)
    gc_raw = get_macro_edge_corners(dg, n_sides)
    gc_matched = match_grid_corners_to_uv(gc_raw, gg, fid, detail_grid=dg)
    
    neigh_map = compute_neighbor_edge_mapping(gg, fid)
    pg_to_macro = compute_pg_to_macro_edge_map(gg, fid, dg)
    macro_to_neigh = {}
    for nid, cpg in neigh_map.items():
        macro_to_neigh[pg_to_macro[cpg]] = nid
    
    # UV-edge neighbours from GoldbergTile
    tile_3d = _match_tile_to_face(tiles_3d, fid)
    # GoldbergTile.neighbor_ids[k] is the neighbour across edge k→k+1
    # (or the neighbour sharing vertices k and k+1)
    uv_edge_neighs = tile_3d.neighbor_indices  # list of int
    
    # Find which gc_raw index each gc_matched[k] came from
    gc_raw_arr = np.array(gc_raw)
    gc_m_arr = np.array(gc_matched)
    
    raw_idx = []
    for k in range(n_sides):
        dists = np.linalg.norm(gc_raw_arr - gc_m_arr[k], axis=1)
        raw_idx.append(int(np.argmin(dists)))
    
    print(f"\n{fid} ({kind}):")
    print(f"  {'warp_k':>7} {'raw_edge':>10} {'macro_neigh':>12} "
          f"{'uv_neigh':>10} {'ok':>5}")
    
    all_ok = True
    for k in range(n_sides):
        k1 = (k + 1) % n_sides
        rk = raw_idx[k]
        rk1 = raw_idx[k1]
        
        # What macro-edge does gc_raw[rk]→gc_raw[rk1] correspond to?
        if rk1 == (rk + 1) % n_sides:
            macro_ei = rk
        elif rk == (rk1 + 1) % n_sides:
            macro_ei = rk1  # reversed
        else:
            macro_ei = -1
        
        macro_n = macro_to_neigh.get(macro_ei, "?")
        uv_n_idx = uv_edge_neighs[k] if k < len(uv_edge_neighs) else -1
        uv_n = f"t{uv_n_idx}" if uv_n_idx >= 0 else "?"
        ok = macro_n == uv_n
        all_ok = all_ok and ok
        
        print(f"  edge {k}→{k1:>2}   raw {rk}→{rk1}      {macro_n:>8}"
              f"    {uv_n:>8}  {'  ✓' if ok else '  ✗ WRONG!'}")
    
    if all_ok:
        print(f"  → ALL EDGES CORRECT ✓")
    else:
        print(f"  → *** EDGE MISMATCH DETECTED ***")

print()
print("=" * 70)
print()

# Also show what neighbor_ids look like for the GoldbergTile
print("GoldbergTile neighbor_ids for reference:")
for fid in ["t0", "t1"]:
    tile_3d = _match_tile_to_face(tiles_3d, fid)
    print(f"  {fid}: neighbor_indices={tile_3d.neighbor_indices}")

print()
print("Macro-edge neighbours for reference:")
for fid in ["t0", "t1"]:
    n_sides = len(gg.faces[fid].vertex_ids)
    is_pent = n_sides == 5
    dg, _ = coll.get(fid)
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)
    neigh_map = compute_neighbor_edge_mapping(gg, fid)
    pg_to_macro = compute_pg_to_macro_edge_map(gg, fid, dg)
    neighs_by_edge = []
    for ei in range(n_sides):
        for nid, cpg in neigh_map.items():
            if pg_to_macro[cpg] == ei:
                neighs_by_edge.append(nid)
                break
    print(f"  {fid}: {neighs_by_edge}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
