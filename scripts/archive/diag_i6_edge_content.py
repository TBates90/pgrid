#!/usr/bin/env python3
"""Diagnostic: compare composite tile content at shared edges.

For a pentagon–hex pair sharing an edge, sample pixels from both
composite images (pre-warp) along their respective macro-edges.
If the composites are correct, these pixels should match.

Also visualises what `match_grid_corners_to_uv` does to the corner
ordering — does it produce correct pairing for the warp?
"""
import json, sys, math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, "src")

from polygrid.globe import build_globe_grid
from polygrid.tile_detail import TileDetailSpec, DetailGridCollection, build_tile_with_neighbours  # noqa
from polygrid.detail_terrain import compute_neighbor_edge_mapping
from polygrid.tile_uv_align import (
    compute_pg_to_macro_edge_map, get_macro_edge_corners,
    match_grid_corners_to_uv, compute_tile_view_limits,
    _signed_area_2d, compute_polygon_corners_px,
)
from polygrid.uv_texture import get_tile_uv_vertices
from polygrid.assembly import _signed_area_macro

EXPORT_DIR = Path("exports/f3")

gg = build_globe_grid(3)
spec = TileDetailSpec(detail_rings=4)
coll = DetailGridCollection.build(gg, spec)

# ═══════════════════════════════════════════════════════════════════
# 1. Composite edge analysis — sample pixels along shared macro-edges
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. COMPOSITE EDGE ANALYSIS (pre-warp)")
print("=" * 70)

def get_composite_and_corners(fid):
    """Build composite and get macro-edge corners + view limits."""
    n_sides = len(gg.faces[fid].vertex_ids)
    is_pent = n_sides == 5
    dg, _ = coll.get(fid)
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)
    gc_raw = get_macro_edge_corners(dg, n_sides)

    comp = build_tile_with_neighbours(coll, fid, gg)
    img = comp.to_image(512, show_grid=False)
    xlim, ylim = compute_tile_view_limits(comp, fid)
    return img, gc_raw, xlim, ylim, dg

def grid_to_pixel(gx, gy, xlim, ylim, img_w, img_h):
    """Convert grid coordinates to image pixel coordinates."""
    x_min, x_max = xlim
    y_min, y_max = ylim
    px = (gx - x_min) / (x_max - x_min) * img_w
    py = (1.0 - (gy - y_min) / (y_max - y_min)) * img_h
    return int(np.clip(px, 0, img_w - 1)), int(np.clip(py, 0, img_h - 1))


def sample_composite_edge(img_arr, gc, edge_start, edge_end, xlim, ylim, n=50, inset=0.05):
    """Sample pixel colors along a macro-edge in the composite."""
    h, w = img_arr.shape[:2]
    gc_arr = np.array(gc)
    centroid = gc_arr.mean(axis=0)
    a = np.array(gc[edge_start])
    b = np.array(gc[edge_end])
    # Inset slightly towards centre to avoid boundary artefacts
    a_in = a + inset * (centroid - a)
    b_in = b + inset * (centroid - b)
    pixels = []
    for i in range(n):
        t = (i + 0.5) / n
        pt = a_in * (1 - t) + b_in * t
        px, py = grid_to_pixel(pt[0], pt[1], xlim, ylim, w, h)
        pixels.append(img_arr[py, px].astype(float))
    return np.array(pixels)


# Find shared edge between t0 and each neighbour
pid = "t0"
face = gg.faces[pid]
neigh_map = compute_neighbor_edge_mapping(gg, pid)

img_p, gc_p, xlim_p, ylim_p, dg_p = get_composite_and_corners(pid)
img_p_arr = np.array(img_p.convert("RGB"))

# For finding which macro-edge corresponds to which neighbour
pg_to_macro_p = compute_pg_to_macro_edge_map(gg, pid, dg_p)

print(f"\n{pid} (PENT) corners: {len(gc_p)}, image: {img_p_arr.shape}")
print(f"Grid corners raw:")
for i, c in enumerate(gc_p):
    print(f"  [{i}] ({c[0]:+.4f}, {c[1]:+.4f})")

for nid, cpg in neigh_map.items():
    macro_edge_idx = pg_to_macro_p[cpg]
    n_sides_p = len(gg.faces[pid].vertex_ids)

    # Get the macro-edge vertex indices for the pentagon
    me_p = next(m for m in dg_p.macro_edges if m.id == macro_edge_idx)
    # Edge goes from gc_p[macro_edge_idx] to gc_p[(macro_edge_idx+1) % n]
    ei_start = macro_edge_idx
    ei_end = (macro_edge_idx + 1) % n_sides_p

    # Get the neighbour's composite
    img_n, gc_n, xlim_n, ylim_n, dg_n = get_composite_and_corners(nid)
    img_n_arr = np.array(img_n.convert("RGB"))
    n_sides_n = len(gg.faces[nid].vertex_ids)
    npg = compute_neighbor_edge_mapping(gg, nid)[pid]
    pg_to_macro_n = compute_pg_to_macro_edge_map(gg, nid, dg_n)
    macro_edge_n = pg_to_macro_n[npg]
    ni_start = macro_edge_n
    ni_end = (macro_edge_n + 1) % n_sides_n

    # Sample from pentagon composite along this edge
    pix_p = sample_composite_edge(img_p_arr, gc_p, ei_start, ei_end, xlim_p, ylim_p)
    # Sample from hex composite along corresponding edge (same direction in grid space)
    pix_n_fwd = sample_composite_edge(img_n_arr, gc_n, ni_start, ni_end, xlim_n, ylim_n)
    pix_n_rev = sample_composite_edge(img_n_arr, gc_n, ni_end, ni_start, xlim_n, ylim_n)

    diff_fwd = np.mean(np.abs(pix_p - pix_n_fwd))
    diff_rev = np.mean(np.abs(pix_p - pix_n_rev))

    result = "MATCH" if diff_fwd < diff_rev else "REVERSED"
    print(f"\n  Edge {pid}[{ei_start}→{ei_end}] ↔ {nid}[{ni_start}→{ni_end}]")
    print(f"    FWD diff: {diff_fwd:.1f}  REV diff: {diff_rev:.1f}  → {result}")
    print(f"    Sample colors (inset=0.05):")
    for si in [0, 12, 24, 37, 49]:
        pp = pix_p[si].astype(int)
        pf = pix_n_fwd[si].astype(int)
        pr = pix_n_rev[si].astype(int)
        print(f"      [{si:2d}] pent=[{pp[0]:3d},{pp[1]:3d},{pp[2]:3d}] "
              f"hex_fwd=[{pf[0]:3d},{pf[1]:3d},{pf[2]:3d}] "
              f"hex_rev=[{pr[0]:3d},{pr[1]:3d},{pr[2]:3d}]")


# ═══════════════════════════════════════════════════════════════════
# 2. Compare hex-hex composite edge (control)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. HEX-HEX COMPOSITE EDGE ANALYSIS (control)")
print("=" * 70)

hid = "t1"
face_h = gg.faces[hid]
neigh_map_h = compute_neighbor_edge_mapping(gg, hid)
img_h, gc_h, xlim_h, ylim_h, dg_h = get_composite_and_corners(hid)
img_h_arr = np.array(img_h.convert("RGB"))
pg_to_macro_h = compute_pg_to_macro_edge_map(gg, hid, dg_h)
n_sides_h = 6

for nid_h in list(face_h.neighbor_ids)[:3]:
    if len(gg.faces[nid_h].vertex_ids) == 5:
        continue  # skip pentagon neighbour
    cpg_h = neigh_map_h[nid_h]
    mei_h = pg_to_macro_h[cpg_h]
    ei_s = mei_h
    ei_e = (mei_h + 1) % n_sides_h

    img_n2, gc_n2, xlim_n2, ylim_n2, dg_n2 = get_composite_and_corners(nid_h)
    img_n2_arr = np.array(img_n2.convert("RGB"))
    n_sides_n2 = len(gg.faces[nid_h].vertex_ids)
    npg_h = compute_neighbor_edge_mapping(gg, nid_h)[hid]
    pg_to_macro_n2 = compute_pg_to_macro_edge_map(gg, nid_h, dg_n2)
    mei_n2 = pg_to_macro_n2[npg_h]
    ni_s = mei_n2
    ni_e = (mei_n2 + 1) % n_sides_n2

    pix_c = sample_composite_edge(img_h_arr, gc_h, ei_s, ei_e, xlim_h, ylim_h)
    pix_n_fwd = sample_composite_edge(img_n2_arr, gc_n2, ni_s, ni_e, xlim_n2, ylim_n2)
    pix_n_rev = sample_composite_edge(img_n2_arr, gc_n2, ni_e, ni_s, xlim_n2, ylim_n2)

    diff_fwd = np.mean(np.abs(pix_c - pix_n_fwd))
    diff_rev = np.mean(np.abs(pix_c - pix_n_rev))
    result = "MATCH" if diff_fwd < diff_rev else "REVERSED"
    print(f"\n  Edge {hid}[{ei_s}→{ei_e}] ↔ {nid_h}[{ni_s}→{ni_e}]")
    print(f"    FWD diff: {diff_fwd:.1f}  REV diff: {diff_rev:.1f}  → {result}")
    for si in [0, 12, 24, 37, 49]:
        pp = pix_c[si].astype(int)
        pf = pix_n_fwd[si].astype(int)
        pr = pix_n_rev[si].astype(int)
        print(f"      [{si:2d}] hex=[{pp[0]:3d},{pp[1]:3d},{pp[2]:3d}] "
              f"neigh_fwd=[{pf[0]:3d},{pf[1]:3d},{pf[2]:3d}] "
              f"neigh_rev=[{pr[0]:3d},{pr[1]:3d},{pr[2]:3d}]")


# ═══════════════════════════════════════════════════════════════════
# 3. Warp mapping verification: check whether the warp maps
#    pentagon edge pixels to the correct atlas location
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. WARP MAPPING VERIFICATION")
print("=" * 70)

# For pentagon t0, get both raw and matched corners
dg_t0, _ = coll.get("t0")
dg_t0.compute_macro_edges(n_sides=5, corner_ids=dg_t0.metadata.get("corner_vertex_ids"))
gc_raw_t0 = get_macro_edge_corners(dg_t0, 5)
gc_matched_t0 = match_grid_corners_to_uv(gc_raw_t0, gg, "t0")
uv_t0 = get_tile_uv_vertices(gg, "t0")

# Check that the matched corners map correctly
print("\nPentagon t0 corner pairing:")
print(f"  gc_raw SA = {_signed_area_2d(gc_raw_t0):+.4f}")
print(f"  gc_matched SA = {_signed_area_2d(gc_matched_t0):+.4f}")
print(f"  uv_corners SA = {_signed_area_2d(list(uv_t0)):+.4f}")
print()

# Which macro-edge index corresponds to the t0→t1 edge?
neigh_map_t0 = compute_neighbor_edge_mapping(gg, "t0")
pg_to_macro_t0 = compute_pg_to_macro_edge_map(gg, "t0", dg_t0)

for nid, cpg in neigh_map_t0.items():
    mei = pg_to_macro_t0[cpg]
    ei_s = mei
    ei_e = (mei + 1) % 5
    gc_s = gc_raw_t0[ei_s]
    gc_e = gc_raw_t0[ei_e]
    gcm_s = gc_matched_t0[ei_s]
    gcm_e = gc_matched_t0[ei_e]
    uv_s = uv_t0[ei_s]
    uv_e = uv_t0[ei_e]
    print(f"  Edge {ei_s}→{ei_e} ({pid}→{nid}):")
    print(f"    gc_raw:    ({gc_s[0]:+.4f},{gc_s[1]:+.4f}) → ({gc_e[0]:+.4f},{gc_e[1]:+.4f})")
    print(f"    gc_match:  ({gcm_s[0]:+.4f},{gcm_s[1]:+.4f}) → ({gcm_e[0]:+.4f},{gcm_e[1]:+.4f})")
    print(f"    uv:        ({uv_s[0]:.4f},{uv_s[1]:.4f}) → ({uv_e[0]:.4f},{uv_e[1]:.4f})")

# Same for hex t1
dg_t1, _ = coll.get("t1")
dg_t1.compute_macro_edges(n_sides=6)
gc_raw_t1 = get_macro_edge_corners(dg_t1, 6)
gc_matched_t1 = match_grid_corners_to_uv(gc_raw_t1, gg, "t1")
uv_t1 = get_tile_uv_vertices(gg, "t1")

print(f"\nHexagon t1 corner pairing:")
print(f"  gc_raw SA = {_signed_area_2d(gc_raw_t1):+.4f}")
print(f"  gc_matched SA = {_signed_area_2d(gc_matched_t1):+.4f}")
print(f"  uv_corners SA = {_signed_area_2d(list(uv_t1)):+.4f}")
print()

neigh_map_t1 = compute_neighbor_edge_mapping(gg, "t1")
pg_to_macro_t1 = compute_pg_to_macro_edge_map(gg, "t1", dg_t1)

for nid, cpg in neigh_map_t1.items():
    mei = pg_to_macro_t1[cpg]
    ei_s = mei
    ei_e = (mei + 1) % 6
    gc_s = gc_raw_t1[ei_s]
    gc_e = gc_raw_t1[ei_e]
    gcm_s = gc_matched_t1[ei_s]
    gcm_e = gc_matched_t1[ei_e]
    uv_s = uv_t1[ei_s]
    uv_e = uv_t1[ei_e]
    print(f"  Edge {ei_s}→{ei_e} ({hid}→{nid}):")
    print(f"    gc_raw:    ({gc_s[0]:+.4f},{gc_s[1]:+.4f}) → ({gc_e[0]:+.4f},{gc_e[1]:+.4f})")
    print(f"    gc_match:  ({gcm_s[0]:+.4f},{gcm_s[1]:+.4f}) → ({gcm_e[0]:+.4f},{gcm_e[1]:+.4f})")
    print(f"    uv:        ({uv_s[0]:.4f},{uv_s[1]:.4f}) → ({uv_e[0]:.4f},{uv_e[1]:.4f})")


# ═══════════════════════════════════════════════════════════════════
# 4. Verify the stitch itself: do apron cells in the pentagon
#    composite show the same content as the corresponding centre cells
#    in the hexagon composite?
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. STITCH CONTENT VERIFICATION")
print("=" * 70)

# Build the pentagon composite
comp_t0 = build_tile_with_neighbours(coll, "t0", gg)
# Build the hex composite
comp_t1 = build_tile_with_neighbours(coll, "t1", gg)

# Check: pentagon composite should contain apron cells from t1
# Look at cell values near the shared edge in both composites
print("\nt0 composite cells: total =", len(comp_t0.cells))
print("t1 composite cells: total =", len(comp_t1.cells))

# Find cells near the t0-t1 shared edge in the pentagon composite
# that originated from the t1 grid (apron cells)
dg_t0_center, _ = coll.get("t0")
dg_t0_center.compute_macro_edges(n_sides=5, corner_ids=dg_t0_center.metadata.get("corner_vertex_ids"))
me_idx = pg_to_macro_t0[neigh_map_t0["t1"]]
me = next(m for m in dg_t0_center.macro_edges if m.id == me_idx)

# Get edge vertex positions
ev0 = dg_t0_center.vertices[me.vertex_ids[0]]
ev1 = dg_t0_center.vertices[me.vertex_ids[-1]]
print(f"\nt0 macro edge {me_idx} towards t1:")
print(f"  from ({ev0.x:.4f}, {ev0.y:.4f}) to ({ev1.x:.4f}, {ev1.y:.4f})")

# Check what values the composite assigns near this edge
# Sample the composite image near the edge midpoint
mid_x = (ev0.x + ev1.x) / 2
mid_y = (ev0.y + ev1.y) / 2

# Get edge normal (pointing outward from centre)
edge_dx = ev1.x - ev0.x
edge_dy = ev1.y - ev0.y
# Normal pointing outward (away from centre)
gc_centroid = np.mean(gc_raw_t0, axis=0)
outward_x = -(edge_dy)
outward_y = edge_dx
# Make sure it points away from centre
if outward_x * (mid_x - gc_centroid[0]) + outward_y * (mid_y - gc_centroid[1]) < 0:
    outward_x, outward_y = -outward_x, -outward_y
norm = math.hypot(outward_x, outward_y)
outward_x /= norm
outward_y /= norm

img_t0 = comp_t0.to_image(512, show_grid=False)
img_t0_arr = np.array(img_t0.convert("RGB"))

print(f"\nSampling along edge normal at midpoint ({mid_x:.4f}, {mid_y:.4f}):")
print(f"  Outward direction: ({outward_x:.4f}, {outward_y:.4f})")
for dist in [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]:
    sx = mid_x + dist * outward_x
    sy = mid_y + dist * outward_y
    px, py = grid_to_pixel(sx, sy, xlim_p, ylim_p, img_t0_arr.shape[1], img_t0_arr.shape[0])
    col = img_t0_arr[py, px]
    side = "inside" if dist < 0 else ("edge" if dist == 0 else "apron")
    print(f"  d={dist:+.3f} ({side:6s}): pixel ({px:3d},{py:3d}) = [{col[0]:3d},{col[1]:3d},{col[2]:3d}]")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
