#!/usr/bin/env python3
"""Diagnostic: verify the warp maps pentagon edges to correct UV edges.

The stitch is correct (composites match at edges). The question is:
does the piecewise warp map the pentagon's edge-3 (towards t1) in the
composite image to the correct UV edge in the atlas?

We check whether gc_matched[k] corresponds to the SAME physical edge
as uv_corners[k] — i.e., do they share the same 3D vertex?
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
    match_grid_corners_to_uv, _signed_area_2d,
)
from polygrid.uv_texture import get_tile_uv_vertices, get_goldberg_tiles, _match_tile_to_face, compute_tile_basis

gg = build_globe_grid(3)
spec = TileDetailSpec(detail_rings=4)
coll = DetailGridCollection.build(gg, spec)

print("=" * 70)
print("EDGE IDENTITY THROUGH THE WARP PIPELINE")
print("=" * 70)
print()
print("For each tile, we check: does warp edge k (gc_matched[k]→gc_matched[k+1])")
print("map to UV edge k (uv[k]→uv[k+1]) and is that the same 3D edge?")
print("If gc_matched and uv_corners are correctly paired, then:")
print("  grid_corner gc_matched[k] ←→ uv_corners[k] ←→ GoldbergTile.vertex[k]")
print()

for fid in ["t0", "t1"]:
    n_sides = len(gg.faces[fid].vertex_ids)
    is_pent = n_sides == 5
    kind = "PENT" if is_pent else "HEX"
    dg, _ = coll.get(fid)
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)
    gc_raw = get_macro_edge_corners(dg, n_sides)
    gc_matched = match_grid_corners_to_uv(gc_raw, gg, fid)
    uv = get_tile_uv_vertices(gg, fid)

    # Get GoldbergTile vertices and their neighbour identities
    tiles = get_goldberg_tiles(3, 1.0)
    tile = _match_tile_to_face(tiles, fid)
    
    # Get 3D vertex IDs from the globe face
    face = gg.faces[fid]
    vid3d = face.vertex_ids  # 3D vertex IDs in generator order
    
    # Get neighbour edge mapping (which neighbour is on which macro-edge)
    neigh_map = compute_neighbor_edge_mapping(gg, fid)
    pg_to_macro = compute_pg_to_macro_edge_map(gg, fid, dg)
    
    # Build: macro_edge_idx → neighbour_id
    macro_to_neigh = {}
    for nid, cpg in neigh_map.items():
        mei = pg_to_macro[cpg]
        macro_to_neigh[mei] = nid
    
    print(f"\n{'='*60}")
    print(f"{fid} ({kind})")
    print(f"{'='*60}")
    print(f"  Globe face vertex_ids (3D): {vid3d}")
    print()
    
    # For each edge, check which neighbour it faces
    # gc_raw edges: edge k goes from gc_raw[k] to gc_raw[(k+1)%n]
    # This is macro_edge k, facing macro_to_neigh[k]
    print(f"  gc_raw edges (macro-edge order):")
    for k in range(n_sides):
        nid = macro_to_neigh.get(k, "?")
        print(f"    edge {k}: gc_raw[{k}]→gc_raw[{(k+1)%n_sides}]"
              f"  →  neigh={nid}")
    
    # gc_matched is reordered to match GoldbergTile vertex order
    # So gc_matched[k] maps to uv[k] which maps to GoldbergTile vertex k
    # Edge k in matched order: gc_matched[k]→gc_matched[(k+1)%n]
    # This corresponds to UV edge uv[k]→uv[(k+1)%n]
    # Which 3D edge is that? vertex_ids[k]→vertex_ids[(k+1)%n]
    
    # Find which gc_raw index each gc_matched[k] came from
    gc_raw_arr = np.array(gc_raw)
    gc_m_arr = np.array(gc_matched)
    
    raw_idx_for_matched = []
    for k in range(n_sides):
        dists = np.linalg.norm(gc_raw_arr - gc_m_arr[k], axis=1)
        raw_k = int(np.argmin(dists))
        raw_idx_for_matched.append(raw_k)
    
    print(f"\n  gc_matched[k] ← gc_raw[?]:")
    for k in range(n_sides):
        rk = raw_idx_for_matched[k]
        print(f"    gc_matched[{k}] ← gc_raw[{rk}]")
    
    print(f"\n  Edge mapping through warp:")
    print(f"  {'warp_edge':>10} {'gc_raw_edge':>12} {'macro_neigh':>12} "
          f"{'uv_edge':>10} {'3D_verts':>12} {'3D_neigh':>12} {'MATCH':>7}")
    
    for k in range(n_sides):
        k1 = (k + 1) % n_sides
        # Warp edge k: gc_matched[k] → gc_matched[k+1]
        raw_a = raw_idx_for_matched[k]
        raw_b = raw_idx_for_matched[k1]
        
        # What macro-edge does this correspond to?
        # gc_raw edge from raw_a to raw_b — but which macro-edge is that?
        # Macro-edge i goes from gc_raw[i] to gc_raw[i+1]
        # So macro-edge = raw_a if raw_b == (raw_a+1)%n
        if raw_b == (raw_a + 1) % n_sides:
            macro_ei = raw_a
            direction = "FWD"
        elif raw_a == (raw_b + 1) % n_sides:
            macro_ei = raw_b
            direction = "REV"
        else:
            macro_ei = -1
            direction = "???"
        
        macro_neigh = macro_to_neigh.get(macro_ei, "?") if macro_ei >= 0 else "?"
        
        # UV edge k: vid3d[k] → vid3d[k+1]
        # What 3D neighbour shares this edge?
        v3a = vid3d[k]
        v3b = vid3d[k1]
        uv_neigh = "?"
        for nid in face.neighbor_ids:
            nface = gg.faces[nid]
            if v3a in nface.vertex_ids and v3b in nface.vertex_ids:
                uv_neigh = nid
                break
        
        match = "✓" if macro_neigh == uv_neigh else "✗ MISMATCH"
        
        print(f"  {k}→{k1:>2} (warp) "
              f"  {raw_a}→{raw_b} ({direction})"
              f"    {macro_neigh:>8}"
              f"   {k}→{k1:>2} (UV)"
              f"  {v3a}→{v3b}"
              f"  {uv_neigh:>8}"
              f"  {match}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
