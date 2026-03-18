#!/usr/bin/env python3
"""Diagnostic: check if edge mismatches are consistent between neighbours.

Theory: hex-hex seams look correct because BOTH hex tiles have the 
same rotation error (content shifted by same amount), so their shared
edge still shows matching content. But pent-hex pairs have DIFFERENT
rotation errors → mismatch at the shared edge.
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


def get_edge_mapping(fid):
    """Return dict: {macro_edge_neigh -> uv_edge_neigh} for each warp edge."""
    n_sides = len(gg.faces[fid].vertex_ids)
    is_pent = n_sides == 5
    dg, _ = coll.get(fid)
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)
    gc_raw = get_macro_edge_corners(dg, n_sides)
    gc_matched = match_grid_corners_to_uv(gc_raw, gg, fid)
    
    neigh_map = compute_neighbor_edge_mapping(gg, fid)
    pg_to_macro = compute_pg_to_macro_edge_map(gg, fid, dg)
    macro_to_neigh = {}
    for nid, cpg in neigh_map.items():
        macro_to_neigh[pg_to_macro[cpg]] = nid
    
    tile_3d = _match_tile_to_face(tiles_3d, fid)
    uv_edge_neighs = tile_3d.neighbor_indices
    
    gc_raw_arr = np.array(gc_raw)
    gc_m_arr = np.array(gc_matched)
    raw_idx = []
    for k in range(n_sides):
        dists = np.linalg.norm(gc_raw_arr - gc_m_arr[k], axis=1)
        raw_idx.append(int(np.argmin(dists)))
    
    # For each warp edge k, find which macro-edge it comes from
    # and which UV-edge it goes to
    result = {}  # {warp_k: (macro_neigh, uv_neigh)}
    for k in range(n_sides):
        k1 = (k + 1) % n_sides
        rk = raw_idx[k]
        rk1 = raw_idx[k1]
        if rk1 == (rk + 1) % n_sides:
            macro_ei = rk
        elif rk == (rk1 + 1) % n_sides:
            macro_ei = rk1
        else:
            macro_ei = -1
        macro_n = macro_to_neigh.get(macro_ei, "?")
        uv_n = f"t{uv_edge_neighs[k]}"
        result[k] = (macro_n, uv_n)
    
    return result


print("=" * 70)
print("SHARED EDGE CONSISTENCY CHECK")
print("=" * 70)
print()
print("For each shared edge between tiles A and B:")
print("  A's warp sends content-for-X to UV-edge-facing-Y")
print("  B's warp sends content-for-P to UV-edge-facing-Q")
print("  If A faces B at UV edge, then B faces A at its UV edge.")
print("  The seam is OK if both send the SAME actual content to their shared edge.")
print()

# Check all edges
all_mappings = {}
for fid in gg.faces:
    all_mappings[fid] = get_edge_mapping(fid)

# For each pair of adjacent tiles, check consistency
pent_hex_ok = 0
pent_hex_bad = 0
hex_hex_ok = 0
hex_hex_bad = 0

checked = set()
for fid in gg.faces:
    n_a = len(gg.faces[fid].vertex_ids)
    mapping_a = all_mappings[fid]
    
    for wk_a, (macro_n_a, uv_n_a) in mapping_a.items():
        nid = uv_n_a  # The tile that A's UV edge faces
        if nid not in all_mappings:
            continue
        pair = tuple(sorted([fid, nid]))
        if pair in checked:
            continue
        checked.add(pair)
        
        n_b = len(gg.faces[nid].vertex_ids)
        mapping_b = all_mappings[nid]
        
        # Find B's warp edge that faces A
        wk_b = None
        for wb, (mn_b, un_b) in mapping_b.items():
            if un_b == fid:
                wk_b = wb
                break
        
        if wk_b is None:
            continue
        
        macro_n_b, uv_n_b = mapping_b[wk_b]
        
        # A's UV edge wk_a faces nid. A sends content from macro_n_a to this edge.
        # B's UV edge wk_b faces fid. B sends content from macro_n_b to this edge.
        # For seam to work: 
        #   A must send content-about-nid to its edge (macro_n_a == nid)
        #   B must send content-about-fid to its edge (macro_n_b == fid)
        a_correct = macro_n_a == nid
        b_correct = macro_n_b == fid
        
        is_pent_hex = (n_a == 5) != (n_b == 5)
        
        if a_correct and b_correct:
            if is_pent_hex:
                pent_hex_ok += 1
            else:
                hex_hex_ok += 1
        else:
            if is_pent_hex:
                pent_hex_bad += 1
            else:
                hex_hex_bad += 1
            
            # Check: do A and B at least send content from the SAME actual 
            # neighbour pair? (i.e., same rotation offset?)
            # A sends content from macro_n_a; B sends content from macro_n_b
            # If macro_n_a == macro_n_b's expected AND vice versa, then
            # the content at the seam would still match
            
            # Actually, the real question is: at the shared UV edge,
            # does A's pixel content (from macro_n_a's apron) match
            # B's pixel content (from macro_n_b's centre)?
            # This depends on whether macro_n_a == fid for B, etc.
            
            if not is_pent_hex or (pent_hex_bad <= 5):
                print(f"  {fid}({'P' if n_a==5 else 'H'})↔{nid}({'P' if n_b==5 else 'H'}): "
                      f"A sends {macro_n_a} (want {nid}), "
                      f"B sends {macro_n_b} (want {fid}) "
                      f"{'PH' if is_pent_hex else 'HH'}")

print()
print(f"Pent-hex edges: {pent_hex_ok} correct, {pent_hex_bad} wrong")
print(f"Hex-hex edges:  {hex_hex_ok} correct, {hex_hex_bad} wrong")
print()

# Summary: how many tiles have CORRECT edge mapping (all edges)?
correct_tiles = 0
wrong_tiles = 0
for fid, mapping in all_mappings.items():
    all_ok = all(mn == un for mn, un in mapping.values())
    if all_ok:
        correct_tiles += 1
    else:
        wrong_tiles += 1

print(f"Tiles with ALL edges correct: {correct_tiles}")
print(f"Tiles with some edges wrong:  {wrong_tiles}")

# Which tiles are correct?
print("\nCorrect tiles:")
for fid, mapping in all_mappings.items():
    all_ok = all(mn == un for mn, un in mapping.values())
    if all_ok:
        n_sides = len(gg.faces[fid].vertex_ids)
        print(f"  {fid} ({'PENT' if n_sides == 5 else 'HEX'})")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
