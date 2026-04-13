#!/usr/bin/env python3
"""Trace the full vertex chain at shared pent-hex corners.

For each shared corner between a pentagon and a hexagon, verify that:
- gc_matched[k] for pent corresponds to the SAME 3D vertex as
  gc_matched[j] for the hex neighbor
- The warp maps these to the correct UV positions
"""
import sys, math
from pathlib import Path

venv_path = Path(__file__).resolve().parent.parent / ".venv" / "lib"
for p in sorted(venv_path.glob("python3.*")):
    sp = str(p / "site-packages")
    if sp not in sys.path:
        sys.path.insert(0, sp)
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
from polygrid.globe.globe import build_globe_grid
from polygrid.rendering.uv_texture import (
    get_tile_uv_vertices, get_goldberg_tiles, compute_tile_basis,
)
from polygrid.rendering.tile_uv_align import (
    get_macro_edge_corners,
    match_grid_corners_to_uv,
    compute_pg_to_macro_corner_map,
    compute_gt_to_pg_corner_map,
)
from polygrid.detail.detail_grid import build_detail_grid

FREQ = 2
DETAIL_RINGS = 4

globe = build_globe_grid(FREQ)
tiles_gb = get_goldberg_tiles(FREQ, 1.0)
tile_by_fid = {f"t{t.index}": t for t in tiles_gb}


def get_gc_matched(fid):
    """Get gc_matched corners for a tile."""
    ns = len(globe.faces[fid].vertex_ids)
    is_pent = ns == 5
    dg = build_detail_grid(globe, fid, detail_rings=DETAIL_RINGS)
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=ns, corner_ids=corner_ids)
    gc_raw = get_macro_edge_corners(dg, ns)
    uc_raw = get_tile_uv_vertices(globe, fid)
    
    if is_pent:
        gt_to_pg = compute_gt_to_pg_corner_map(globe, fid)
        pg_to_macro = compute_pg_to_macro_corner_map(globe, fid, dg)
        gc_matched = [
            gc_raw[pg_to_macro[gt_to_pg[k]]]
            for k in range(ns)
        ]
    else:
        gc_matched = match_grid_corners_to_uv(gc_raw, globe, fid)
    
    return gc_matched, uc_raw, ns


# Focus on pentagon t0 and its neighbors
pent_fid = "t0"
pent_tile = tile_by_fid[pent_fid]
pent_gc, pent_uc, pent_ns = get_gc_matched(pent_fid)

print(f"Pentagon {pent_fid}: {pent_ns} sides")
print(f"  GT vertices (3D): {[tuple(round(c,4) for c in v) for v in pent_tile.vertices]}")
print(f"  UV corners:       {pent_uc}")
print(f"  GC matched corners: {pent_gc}")
print()

# For each pentagon edge, find the hex neighbor and check shared vertex alignment
for gt_edge_k in range(pent_ns):
    # Pentagon edge k connects GT vertex k and (k+1)%n
    v_k = np.array(pent_tile.vertices[gt_edge_k], dtype=np.float64)
    v_k1 = np.array(pent_tile.vertices[(gt_edge_k + 1) % pent_ns], dtype=np.float64)
    
    # Find neighbor that shares this edge
    for other_tile in tiles_gb:
        if other_tile.index == pent_tile.index:
            continue
        other_fid = f"t{other_tile.index}"
        other_ns = len(other_tile.vertices)
        
        # Check if both vertices are shared
        matches_k = []
        matches_k1 = []
        for oi, ov in enumerate(other_tile.vertices):
            if np.linalg.norm(np.array(ov) - v_k) < 1e-6:
                matches_k.append(oi)
            if np.linalg.norm(np.array(ov) - v_k1) < 1e-6:
                matches_k1.append(oi)
        
        if matches_k and matches_k1:
            other_gc, other_uc, _ = get_gc_matched(other_fid)
            
            pent_vert_k_idx = gt_edge_k
            pent_vert_k1_idx = (gt_edge_k + 1) % pent_ns
            other_vert_k_idx = matches_k[0]
            other_vert_k1_idx = matches_k1[0]
            
            print(f"Edge {gt_edge_k}: {pent_fid}[{pent_vert_k_idx}→{pent_vert_k1_idx}] <-> {other_fid}[{other_vert_k_idx}→{other_vert_k1_idx}]")
            print(f"  Pent GC[{pent_vert_k_idx}]  = ({pent_gc[pent_vert_k_idx][0]:.6f}, {pent_gc[pent_vert_k_idx][1]:.6f})")
            print(f"  Pent GC[{pent_vert_k1_idx}] = ({pent_gc[pent_vert_k1_idx][0]:.6f}, {pent_gc[pent_vert_k1_idx][1]:.6f})")
            print(f"  Hex  GC[{other_vert_k_idx}]  = ({other_gc[other_vert_k_idx][0]:.6f}, {other_gc[other_vert_k_idx][1]:.6f})")
            print(f"  Hex  GC[{other_vert_k1_idx}] = ({other_gc[other_vert_k1_idx][0]:.6f}, {other_gc[other_vert_k1_idx][1]:.6f})")
            print(f"  Pent UV[{pent_vert_k_idx}]  = ({pent_uc[pent_vert_k_idx][0]:.6f}, {pent_uc[pent_vert_k_idx][1]:.6f})")
            print(f"  Pent UV[{pent_vert_k1_idx}] = ({pent_uc[pent_vert_k1_idx][0]:.6f}, {pent_uc[pent_vert_k1_idx][1]:.6f})")
            print(f"  Hex  UV[{other_vert_k_idx}]  = ({other_uc[other_vert_k_idx][0]:.6f}, {other_uc[other_vert_k_idx][1]:.6f})")
            print(f"  Hex  UV[{other_vert_k1_idx}] = ({other_uc[other_vert_k1_idx][0]:.6f}, {other_uc[other_vert_k1_idx][1]:.6f})")
            
            # Check: does interpolating along the pent edge (in GC space) produce
            # the same 3D vertex as interpolating along the hex edge?
            # More importantly: does the pent GC corner at the shared vertex
            # correspond to the hex GC corner at the same shared vertex?
            print(f"  3D vertex match: pent[{pent_vert_k_idx}] ≈ hex[{other_vert_k_idx}] = {np.linalg.norm(v_k - np.array(other_tile.vertices[other_vert_k_idx])):.8f}")
            print()
            break
