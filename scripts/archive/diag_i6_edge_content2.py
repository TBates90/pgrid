#!/usr/bin/env python3
"""Diagnostic: compare pre-warp composite tile content at shared edges.

Uses the already-rendered composite tile PNGs (t0.png, t1.png etc.)
to check whether the pixels near shared macro-edges match between
adjacent tiles.

This isolates the question: is the STITCH correct (before any warp)?
"""
import json, sys, math
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, "src")

from polygrid.globe import build_globe_grid
from polygrid.tile_detail import TileDetailSpec, DetailGridCollection, build_tile_with_neighbours
from polygrid.detail_terrain import compute_neighbor_edge_mapping
from polygrid.tile_uv_align import (
    compute_pg_to_macro_edge_map, get_macro_edge_corners,
    match_grid_corners_to_uv, compute_tile_view_limits,
    _signed_area_2d,
)
from polygrid.uv_texture import get_tile_uv_vertices
from polygrid.assembly import _signed_area_macro

EXPORT_DIR = Path("exports/f3")

gg = build_globe_grid(3)
spec = TileDetailSpec(detail_rings=4)
coll = DetailGridCollection.build(gg, spec)

pentagons = sorted(fid for fid in gg.faces if len(gg.faces[fid].vertex_ids) == 5)

def get_tile_info(fid):
    """Get corners, view limits, and image for a tile."""
    n_sides = len(gg.faces[fid].vertex_ids)
    is_pent = n_sides == 5
    dg, _ = coll.get(fid)
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)
    gc = get_macro_edge_corners(dg, n_sides)
    comp = build_tile_with_neighbours(coll, fid, gg)
    xlim, ylim = compute_tile_view_limits(comp, fid)
    img = np.array(Image.open(EXPORT_DIR / f"{fid}.png").convert("RGB"))
    return dg, gc, xlim, ylim, img, n_sides


def grid_to_pixel(gx, gy, xlim, ylim, w, h):
    x_min, x_max = xlim
    y_min, y_max = ylim
    px = (gx - x_min) / (x_max - x_min) * w
    py = (1.0 - (gy - y_min) / (y_max - y_min)) * h
    return int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))


def sample_edge_in_composite(img_arr, gc, edge_start, edge_end, xlim, ylim, n=50, inset=0.08):
    """Sample pixels along a macro-edge, inset towards centroid."""
    h, w = img_arr.shape[:2]
    centroid = np.mean(gc, axis=0)
    a = np.array(gc[edge_start])
    b = np.array(gc[edge_end])
    a_in = a + inset * (centroid - a)
    b_in = b + inset * (centroid - b)
    pixels = []
    for i in range(n):
        t = (i + 0.5) / n
        pt = a_in * (1 - t) + b_in * t
        px, py = grid_to_pixel(pt[0], pt[1], xlim, ylim, w, h)
        pixels.append(img_arr[py, px].astype(float))
    return np.array(pixels)


# ═══════════════════════════════════════════════════════════════════
# 1. Pentagon composite edge vs hex composite edge
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. COMPOSITE EDGE COMPARISON (pre-warp)")
print("=" * 70)
print("Sampling pixels along each shared edge in BOTH composites.")
print("If the stitch is correct, pentagon's apron region near the edge")
print("should show the same content as hex's centre region near the edge.")
print()

for pid in pentagons[:3]:  # First 3 pentagons
    dg_p, gc_p, xlim_p, ylim_p, img_p, ns_p = get_tile_info(pid)
    neigh_map = compute_neighbor_edge_mapping(gg, pid)
    pg_to_macro = compute_pg_to_macro_edge_map(gg, pid, dg_p)

    print(f"\n{pid} (PENT):")
    for nid, cpg in neigh_map.items():
        mei_p = pg_to_macro[cpg]
        ei_s = mei_p
        ei_e = (mei_p + 1) % ns_p

        dg_n, gc_n, xlim_n, ylim_n, img_n, ns_n = get_tile_info(nid)
        npg = compute_neighbor_edge_mapping(gg, nid)[pid]
        pg_to_macro_n = compute_pg_to_macro_edge_map(gg, nid, dg_n)
        mei_n = pg_to_macro_n[npg]
        ni_s = mei_n
        ni_e = (mei_n + 1) % ns_n

        # Sample from pentagon side (near its edge towards nid)
        pix_p = sample_edge_in_composite(img_p, gc_p, ei_s, ei_e, xlim_p, ylim_p)
        
        # Sample from hex side (near its edge towards pid)
        # Try both directions
        pix_n_fwd = sample_edge_in_composite(img_n, gc_n, ni_s, ni_e, xlim_n, ylim_n)
        pix_n_rev = sample_edge_in_composite(img_n, gc_n, ni_e, ni_s, xlim_n, ylim_n)

        diff_fwd = np.mean(np.abs(pix_p - pix_n_fwd))
        diff_rev = np.mean(np.abs(pix_p - pix_n_rev))
        best = min(diff_fwd, diff_rev)
        result = "FWD" if diff_fwd < diff_rev else "REV"
        
        print(f"  → {nid} edge[{ei_s}→{ei_e}]↔[{ni_s}→{ni_e}]: "
              f"fwd={diff_fwd:.1f} rev={diff_rev:.1f} best={best:.1f} ({result})")

# ═══════════════════════════════════════════════════════════════════
# 2. Hex composite edge vs hex composite edge (control)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. HEX-HEX COMPOSITE EDGE COMPARISON (control)")
print("=" * 70)

for hid in ["t1", "t3", "t7"]:
    dg_h, gc_h, xlim_h, ylim_h, img_h, ns_h = get_tile_info(hid)
    neigh_map_h = compute_neighbor_edge_mapping(gg, hid)
    pg_to_macro_h = compute_pg_to_macro_edge_map(gg, hid, dg_h)

    print(f"\n{hid} (HEX):")
    for nid_h in list(gg.faces[hid].neighbor_ids):
        if len(gg.faces[nid_h].vertex_ids) == 5:
            continue  # skip pent neighbours
        cpg_h = neigh_map_h[nid_h]
        mei_h = pg_to_macro_h[cpg_h]
        ei_s = mei_h
        ei_e = (mei_h + 1) % ns_h

        dg_n, gc_n, xlim_n, ylim_n, img_n, ns_n = get_tile_info(nid_h)
        npg_h = compute_neighbor_edge_mapping(gg, nid_h)[hid]
        pg_to_macro_n = compute_pg_to_macro_edge_map(gg, nid_h, dg_n)
        mei_n = pg_to_macro_n[npg_h]
        ni_s = mei_n
        ni_e = (mei_n + 1) % ns_n

        pix_h = sample_edge_in_composite(img_h, gc_h, ei_s, ei_e, xlim_h, ylim_h)
        pix_n_fwd = sample_edge_in_composite(img_n, gc_n, ni_s, ni_e, xlim_n, ylim_n)
        pix_n_rev = sample_edge_in_composite(img_n, gc_n, ni_e, ni_s, xlim_n, ylim_n)

        diff_fwd = np.mean(np.abs(pix_h - pix_n_fwd))
        diff_rev = np.mean(np.abs(pix_h - pix_n_rev))
        best = min(diff_fwd, diff_rev)
        result = "FWD" if diff_fwd < diff_rev else "REV"
        
        print(f"  → {nid_h} edge[{ei_s}→{ei_e}]↔[{ni_s}→{ni_e}]: "
              f"fwd={diff_fwd:.1f} rev={diff_rev:.1f} best={best:.1f} ({result})")

# ═══════════════════════════════════════════════════════════════════
# 3. Macro-edge direction consistency check
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. MACRO-EDGE DIRECTION CHECK")
print("=" * 70)
print("For each pent-hex shared edge, check vertex positions at shared")
print("macro-edge endpoints in BOTH composites' coordinate systems.")
print()

for pid in ["t0"]:
    dg_p, gc_p, xlim_p, ylim_p, img_p, ns_p = get_tile_info(pid)
    neigh_map = compute_neighbor_edge_mapping(gg, pid)
    pg_to_macro = compute_pg_to_macro_edge_map(gg, pid, dg_p)
    
    print(f"{pid} (PENT) gc_raw: {gc_p}")
    print(f"  SA = {_signed_area_2d(gc_p):+.4f}")
    
    for nid, cpg in neigh_map.items():
        mei_p = pg_to_macro[cpg]
        
        dg_n, gc_n, xlim_n, ylim_n, img_n, ns_n = get_tile_info(nid)
        npg = compute_neighbor_edge_mapping(gg, nid)[pid]
        pg_to_macro_n = compute_pg_to_macro_edge_map(gg, nid, dg_n)
        mei_n = pg_to_macro_n[npg]

        # Get actual vertex positions on the macro-edge
        me_p = next(m for m in dg_p.macro_edges if m.id == mei_p)
        me_n = next(m for m in dg_n.macro_edges if m.id == mei_n)
        
        # Endpoint vertices in pentagon's grid
        p_v0 = dg_p.vertices[me_p.vertex_ids[0]]
        p_v1 = dg_p.vertices[me_p.vertex_ids[-1]]
        
        # Now position the neighbour (as done during stitch)
        from polygrid.assembly import _position_hex_for_stitch
        positioned = _position_hex_for_stitch(dg_p, mei_p, dg_n, mei_n)
        me_pos = next(m for m in positioned.macro_edges if m.id == mei_n)
        n_v0 = positioned.vertices[me_pos.vertex_ids[0]]
        n_v1 = positioned.vertices[me_pos.vertex_ids[-1]]
        
        # Check alignment
        d00 = math.hypot(p_v0.x - n_v0.x, p_v0.y - n_v0.y)
        d01 = math.hypot(p_v0.x - n_v1.x, p_v0.y - n_v1.y)
        d10 = math.hypot(p_v1.x - n_v0.x, p_v1.y - n_v0.y)
        d11 = math.hypot(p_v1.x - n_v1.x, p_v1.y - n_v1.y)
        
        fwd = d00 < 0.001 and d11 < 0.001
        rev = d01 < 0.001 and d10 < 0.001
        
        print(f"\n  → {nid}: pent_edge={mei_p} hex_edge={mei_n}")
        print(f"    pent vertices: ({p_v0.x:.4f},{p_v0.y:.4f}) → ({p_v1.x:.4f},{p_v1.y:.4f})")
        print(f"    hex  vertices: ({n_v0.x:.4f},{n_v0.y:.4f}) → ({n_v1.x:.4f},{n_v1.y:.4f})")
        print(f"    d00={d00:.6f} d01={d01:.6f} d10={d10:.6f} d11={d11:.6f}")
        print(f"    Alignment: {'FWD' if fwd else 'REV' if rev else 'MISALIGNED!'}")

        # Also check: in the COMPOSITE, what do pixels look like at
        # the actual shared vertex positions?
        # Pentagon side
        px0_p, py0_p = grid_to_pixel(p_v0.x, p_v0.y, xlim_p, ylim_p, img_p.shape[1], img_p.shape[0])
        px1_p, py1_p = grid_to_pixel(p_v1.x, p_v1.y, xlim_p, ylim_p, img_p.shape[1], img_p.shape[0])
        # Hex side (uses hex's own coord space)
        # Need to sample from hex's OWN composite at its macro-edge
        me_n_orig = next(m for m in dg_n.macro_edges if m.id == mei_n)
        h_v0 = dg_n.vertices[me_n_orig.vertex_ids[0]]
        h_v1 = dg_n.vertices[me_n_orig.vertex_ids[-1]]
        px0_n, py0_n = grid_to_pixel(h_v0.x, h_v0.y, xlim_n, ylim_n, img_n.shape[1], img_n.shape[0])
        px1_n, py1_n = grid_to_pixel(h_v1.x, h_v1.y, xlim_n, ylim_n, img_n.shape[1], img_n.shape[0])
        
        print(f"    Pent composite at edge start: pixel ({px0_p},{py0_p}) = {img_p[py0_p, px0_p]}")
        print(f"    Pent composite at edge end:   pixel ({px1_p},{py1_p}) = {img_p[py1_p, px1_p]}")
        print(f"    Hex composite at edge start:  pixel ({px0_n},{py0_n}) = {img_n[py0_n, px0_n]}")
        print(f"    Hex composite at edge end:    pixel ({px1_n},{py1_n}) = {img_n[py1_n, px1_n]}")


# ═══════════════════════════════════════════════════════════════════
# 4. Corner pairing angle analysis
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. CORNER PAIRING: angles in GoldbergTile vs grid corners")
print("=" * 70)

from polygrid.uv_texture import get_goldberg_tiles, _match_tile_to_face, compute_tile_basis

for fid in ["t0", "t1"]:
    n_sides = len(gg.faces[fid].vertex_ids)
    dg, _ = coll.get(fid)
    is_pent = n_sides == 5
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)
    gc = get_macro_edge_corners(dg, n_sides)
    gc_m = match_grid_corners_to_uv(gc, gg, fid)
    
    tiles = get_goldberg_tiles(3, 1.0)
    tile = _match_tile_to_face(tiles, fid)
    center_3d, _, tangent, bitangent = compute_tile_basis(gg, fid)
    
    print(f"\n{fid} ({'PENT' if is_pent else 'HEX'}):")
    print(f"  GoldbergTile angles (3D projected):")
    for i, vtx in enumerate(tile.vertices):
        rel = np.array(vtx) - center_3d
        u = float(np.dot(rel, tangent))
        v = float(np.dot(rel, bitangent))
        angle = math.degrees(math.atan2(v, u))
        print(f"    [{i}] 3D=({vtx[0]:.4f},{vtx[1]:.4f},{vtx[2]:.4f}) "
              f"proj=({u:.4f},{v:.4f}) angle={angle:.1f}°")
    
    gc_arr = np.array(gc)
    centroid = gc_arr.mean(axis=0)
    print(f"  Grid corners raw (centroid=({centroid[0]:.4f},{centroid[1]:.4f})):")
    for i, c in enumerate(gc):
        angle = math.degrees(math.atan2(c[1] - centroid[1], c[0] - centroid[0]))
        print(f"    [{i}] ({c[0]:+.4f},{c[1]:+.4f}) angle={angle:.1f}°")
    
    gc_m_arr = np.array(gc_m)
    centroid_m = gc_m_arr.mean(axis=0)
    print(f"  Grid corners matched (centroid=({centroid_m[0]:.4f},{centroid_m[1]:.4f})):")
    for i, c in enumerate(gc_m):
        angle = math.degrees(math.atan2(c[1] - centroid_m[1], c[0] - centroid_m[0]))
        print(f"    [{i}] ({c[0]:+.4f},{c[1]:+.4f}) angle={angle:.1f}°")
    
    uv = get_tile_uv_vertices(gg, fid)
    uv_arr = np.array(uv)
    uv_centroid = uv_arr.mean(axis=0)
    print(f"  UV corners (centroid=({uv_centroid[0]:.4f},{uv_centroid[1]:.4f})):")
    for i, c in enumerate(uv):
        angle = math.degrees(math.atan2(c[1] - uv_centroid[1], c[0] - uv_centroid[0]))
        print(f"    [{i}] ({c[0]:.4f},{c[1]:.4f}) angle={angle:.1f}°")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
