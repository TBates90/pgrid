#!/usr/bin/env python3
"""Direct test of match_grid_corners_to_uv neighbour identity.

For each tile, match grid corners to UV, then check:
  gc_matched edge k → which macro-edge? → which neighbour?
  uv edge k → which GoldbergTile neighbour?
  Do they match?
"""
import sys, math
import numpy as np

sys.path.insert(0, "src")

from polygrid.globe import build_globe_grid
from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
from polygrid.detail_terrain import compute_neighbor_edge_mapping
from polygrid.tile_uv_align import (
    compute_pg_to_macro_edge_map, get_macro_edge_corners,
    match_grid_corners_to_uv,
)
from polygrid.uv_texture import get_goldberg_tiles, _match_tile_to_face

gg = build_globe_grid(3)
spec = TileDetailSpec(detail_rings=4)
coll = DetailGridCollection.build(gg, spec)
tiles_3d = get_goldberg_tiles(3, 1.0)


def _qv(v, prec=6):
    return tuple(round(c, prec) for c in v)


# Build GT edge → neighbour lookup
vert_to_tiles = {}
for t in tiles_3d:
    for vtx in t.vertices:
        vert_to_tiles.setdefault(_qv(vtx), []).append(t.index)


total_ok = 0
total_bad = 0

for fid in sorted(gg.faces.keys(), key=lambda x: int(x[1:])):
    n_sides = len(gg.faces[fid].vertex_ids)
    is_pent = n_sides == 5
    
    dg, _ = coll.get(fid)
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)
    gc_raw = get_macro_edge_corners(dg, n_sides)
    gc_matched = match_grid_corners_to_uv(gc_raw, gg, fid, detail_grid=dg)
    
    tile = _match_tile_to_face(tiles_3d, fid)
    
    # gc_matched[m] should pair with uv_corners[m].
    # gc_matched[m] is some grid_corners[k] — the start of macro-edge k.
    # So "matched edge m" goes from gc_matched[m] to gc_matched[(m+1)%n],
    # which is macro-edge k (going from grid_corners[k] to grid_corners[(k+1)%n]).
    # We need to find k from gc_matched[m].
    
    gc_raw_arr = np.array(gc_raw)
    gc_m_arr = np.array(gc_matched)
    
    # Find which raw index each matched corner came from
    raw_for_m = {}
    for m in range(n_sides):
        dists = np.linalg.norm(gc_raw_arr - gc_m_arr[m], axis=1)
        raw_for_m[m] = int(np.argmin(dists))
    
    # Neighbour map
    neigh_map = compute_neighbor_edge_mapping(gg, fid)
    pg_to_macro = compute_pg_to_macro_edge_map(gg, fid, dg)
    macro_to_neigh = {}
    for nid, cpg in neigh_map.items():
        macro_to_neigh[pg_to_macro[cpg]] = nid
    
    # GT edge → neighbour
    gt_edge_neigh = {}
    for m in range(n_sides):
        va = _qv(tile.vertices[m])
        vb = _qv(tile.vertices[(m + 1) % n_sides])
        shared = (set(vert_to_tiles.get(va, [])) & set(vert_to_tiles.get(vb, []))) - {tile.index}
        if shared:
            gt_edge_neigh[m] = f"t{shared.pop()}"
    
    # Check each matched edge
    tile_ok = True
    for m in range(n_sides):
        # Matched edge m: gc_matched[m] = gc_raw[raw_for_m[m]]
        # This is the start corner of macro-edge raw_for_m[m]
        macro_k = raw_for_m[m]
        macro_neigh = macro_to_neigh.get(macro_k, "?")
        uv_neigh = gt_edge_neigh.get(m, "?")
        
        if macro_neigh != uv_neigh:
            tile_ok = False
    
    if tile_ok:
        total_ok += 1
    else:
        total_bad += 1
        if total_bad <= 10:  # Show first 10 failures
            kind = "PENT" if is_pent else "HEX"
            print(f"\n{fid} ({kind}): FAILED")
            for m in range(n_sides):
                macro_k = raw_for_m[m]
                macro_neigh = macro_to_neigh.get(macro_k, "?")
                uv_neigh = gt_edge_neigh.get(m, "?")
                ok = macro_neigh == uv_neigh
                print(f"  edge {m}: raw={macro_k} macro_neigh={macro_neigh} "
                      f"uv_neigh={uv_neigh} {'✓' if ok else '✗'}")

print(f"\n\nSummary: {total_ok} tiles correct, {total_bad} tiles with mismatches")
print(f"Total: {total_ok + total_bad}")
