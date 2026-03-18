#!/usr/bin/env python3
"""Comprehensive Issue 6 diagnostic.

Systematically examines:
1. The stitched composite for pentagon tiles — vertex alignment
2. The atlas edge sampling — forward vs reversed pixel match
3. The full pipeline trace from Tutte embedding through to atlas

Run AFTER generating tiles:
    .venv/bin/python scripts/render_polygrids.py -f 3 --detail-rings 4 \
        -o exports/f3 --renderer analytical --tile-size 512
"""
import json, sys, math
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, "src")

from polygrid.globe import build_globe_grid
from polygrid.tile_detail import (
    TileDetailSpec, DetailGridCollection, build_tile_with_neighbours,
)
from polygrid.detail_terrain import compute_neighbor_edge_mapping
from polygrid.tile_uv_align import (
    compute_pg_to_macro_edge_map, get_macro_edge_corners,
    match_grid_corners_to_uv, compute_tile_view_limits,
    _signed_area_2d,
)
from polygrid.uv_texture import get_tile_uv_vertices
from polygrid.assembly import _position_hex_for_stitch, _signed_area_macro

EXPORT_DIR = Path("exports/f3")

# ══════════════════════════════════════════════════════════════════
# 1. SETUP
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("ISSUE 6 COMPREHENSIVE DIAGNOSTIC")
print("=" * 70)

gg = build_globe_grid(3)
spec = TileDetailSpec(detail_rings=4)
coll = DetailGridCollection.build(gg, spec)

atlas = np.array(Image.open(EXPORT_DIR / "atlas.png").convert("RGB"))
with open(EXPORT_DIR / "uv_layout.json") as f:
    uv_layout = json.load(f)
atlas_h, atlas_w = atlas.shape[:2]

pentagons = sorted(fid for fid in gg.faces if len(gg.faces[fid].vertex_ids) == 5)
hexagons = sorted(fid for fid in gg.faces if len(gg.faces[fid].vertex_ids) == 6)

print(f"Pentagons: {len(pentagons)}, Hexagons: {len(hexagons)}")
print(f"Atlas: {atlas_w}x{atlas_h}")
print()

# ══════════════════════════════════════════════════════════════════
# 2. STITCH WINDING ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("2. STITCH WINDING ANALYSIS")
print("=" * 70)

for fid in ["t0", "t1"]:
    n_sides = len(gg.faces[fid].vertex_ids)
    kind = "PENT" if n_sides == 5 else "HEX"
    dg_center, _ = coll.get(fid)
    dg_center.compute_macro_edges(
        n_sides=n_sides,
        corner_ids=dg_center.metadata.get("corner_vertex_ids"),
    )
    sa_center = _signed_area_macro(dg_center)
    print(f"\n{fid} ({kind}): SA={sa_center:+.4f} ({'CCW' if sa_center > 0 else 'CW'})")

    neigh_map = compute_neighbor_edge_mapping(gg, fid)
    pg_to_macro = compute_pg_to_macro_edge_map(gg, fid, dg_center)

    for nid, cpg in neigh_map.items():
        n_sides_n = len(gg.faces[nid].vertex_ids)
        nkind = "pent" if n_sides_n == 5 else "hex"
        dg_n, _ = coll.get(nid)
        dg_n.compute_macro_edges(
            n_sides=n_sides_n,
            corner_ids=dg_n.metadata.get("corner_vertex_ids"),
        )
        sa_n = _signed_area_macro(dg_n)
        same_winding = (sa_center > 0) == (sa_n > 0)

        npg = compute_neighbor_edge_mapping(gg, nid)[fid]
        cm = pg_to_macro[cpg]
        pm = compute_pg_to_macro_edge_map(gg, nid, dg_n)[npg]

        positioned = _position_hex_for_stitch(dg_center, cm, dg_n, pm)
        sa_pos = _signed_area_macro(positioned)
        reflected = (sa_n > 0) != (sa_pos > 0)

        # Check edge alignment
        tme = next(m for m in dg_center.macro_edges if m.id == cm)
        sme = next(m for m in positioned.macro_edges if m.id == pm)
        t0v = dg_center.vertices[tme.vertex_ids[0]]
        t1v = dg_center.vertices[tme.vertex_ids[-1]]
        s0v = positioned.vertices[sme.vertex_ids[0]]
        s1v = positioned.vertices[sme.vertex_ids[-1]]

        d_fwd_start = math.hypot(t0v.x - s0v.x, t0v.y - s0v.y)
        d_fwd_end = math.hypot(t1v.x - s1v.x, t1v.y - s1v.y)
        d_rev_start = math.hypot(t0v.x - s1v.x, t0v.y - s1v.y)
        d_rev_end = math.hypot(t1v.x - s0v.x, t1v.y - s0v.y)

        fwd_ok = d_fwd_start < 0.001 and d_fwd_end < 0.001
        rev_ok = d_rev_start < 0.001 and d_rev_end < 0.001
        align = "FWD" if fwd_ok else ("REV" if rev_ok else "BAD")

        print(f"  {nid}({nkind}): same_w={same_winding} reflected={reflected} "
              f"align={align} (fwd={d_fwd_start:.4f},{d_fwd_end:.4f} "
              f"rev={d_rev_start:.4f},{d_rev_end:.4f})")


# ══════════════════════════════════════════════════════════════════
# 3. COMPOSITE VISUAL CHECK — does each stitched tile look correct?
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("3. COMPOSITE TILE VISUAL STATS")
print("=" * 70)

for fid in ["t0", "t6", "t1", "t3"]:
    img_path = EXPORT_DIR / f"{fid}.png"
    if not img_path.exists():
        print(f"  {fid}: MISSING")
        continue
    img = np.array(Image.open(img_path).convert("RGB"))
    n_sides = len(gg.faces[fid].vertex_ids)
    kind = "PENT" if n_sides == 5 else "HEX"
    uniq = len(np.unique(img.reshape(-1, 3), axis=0))
    magenta = np.all(img == [255, 0, 255], axis=-1).sum()
    print(f"  {fid} ({kind}): {img.shape[1]}x{img.shape[0]} "
          f"unique_colors={uniq} magenta_pixels={magenta}")


# ══════════════════════════════════════════════════════════════════
# 4. ATLAS EDGE SAMPLING — compare pixel strips along shared edges
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("4. ATLAS EDGE SAMPLING")
print("=" * 70)


def uv_to_atlas_px(fid, u, v):
    """Map normalised UV (in [0,1]) to atlas pixel coords."""
    box = uv_layout[fid]
    u0, v0, u1, v1 = box
    ax = (u0 + u * (u1 - u0)) * atlas_w
    ay = (1.0 - (v0 + v * (v1 - v0))) * atlas_h
    return int(np.clip(ax, 0, atlas_w - 1)), int(np.clip(ay, 0, atlas_h - 1))


def sample_edge_pixels(fid, vi_start, vi_end, n_samples=50, inset=0.05):
    """Sample pixels along an edge, slightly inset from the boundary."""
    verts = get_tile_uv_vertices(gg, fid)
    a = np.array(verts[vi_start])
    b = np.array(verts[vi_end])
    center = np.mean(verts, axis=0)
    a_in = a + inset * (center - a)
    b_in = b + inset * (center - b)
    pixels = []
    for i in range(n_samples):
        t = (i + 0.5) / n_samples
        pt = a_in * (1 - t) + b_in * t
        px, py = uv_to_atlas_px(fid, pt[0], pt[1])
        pixels.append(atlas[py, px].astype(float))
    return np.array(pixels)


# Build 3D vertex table for finding shared edges
vert3d = {}
for fid, face in gg.faces.items():
    coords = []
    for vid in face.vertex_ids:
        v = gg.vertices[vid]
        coords.append(np.array([v.x, v.y, v.z]))
    vert3d[fid] = coords


def find_shared_edge(fid_a, fid_b, tol=1e-4):
    """Find shared edge vertex indices between two faces."""
    va = vert3d[fid_a]
    vb = vert3d[fid_b]
    na, nb = len(va), len(vb)
    matches = []
    for i in range(na):
        for j in range(nb):
            if np.linalg.norm(va[i] - vb[j]) < tol:
                matches.append((i, j))
    if len(matches) < 2:
        return None
    (ai, aj), (bi, bj) = matches[0], matches[1]
    if (bi - ai) % na != 1 and (ai - bi) % na != 1:
        return None
    if (bi - ai) % na != 1:
        ai, bi = bi, ai
        aj, bj = bj, aj
    return ai, bi, aj, bj


# Test all pentagon edges
print("\nPentagon-hex edge sampling (forward vs reversed):")
print(f"{'pent':>5} {'edge':>5} {'hex':>5} {'fwd_diff':>10} {'rev_diff':>10} {'result':>10} {'margin':>8}")
print("-" * 60)

pent_match, pent_rev, pent_total = 0, 0, 0
for pid in pentagons:
    face = gg.faces[pid]
    for nid in face.neighbor_ids:
        result = find_shared_edge(pid, nid)
        if result is None:
            continue
        ai, bi, aj, bj = result
        pix_a = sample_edge_pixels(pid, ai, bi)
        pix_b = sample_edge_pixels(nid, bj, aj)  # reversed on neighbour

        diff_fwd = np.mean(np.abs(pix_a - pix_b))
        diff_rev = np.mean(np.abs(pix_a - pix_b[::-1]))
        margin = diff_rev - diff_fwd
        status = "MATCH" if diff_fwd < diff_rev else "REVERSED"

        pent_total += 1
        if diff_fwd < diff_rev:
            pent_match += 1
        else:
            pent_rev += 1

        print(f"{pid:>5} [{ai}-{bi}] {nid:>5} {diff_fwd:>10.2f} {diff_rev:>10.2f} "
              f"{status:>10} {margin:>+8.2f}")

print(f"\nPentagon edges: {pent_match}/{pent_total} MATCH, {pent_rev} REVERSED")

# Compare with hex-hex edges
print("\nHex-hex edge sampling (sample):")
hh_match, hh_rev, hh_total = 0, 0, 0
for hid in hexagons[:8]:
    face = gg.faces[hid]
    for nid in face.neighbor_ids:
        if len(gg.faces[nid].vertex_ids) == 5:
            continue  # skip pent neighbours, already tested above
        result = find_shared_edge(hid, nid)
        if result is None:
            continue
        ai, bi, aj, bj = result
        pix_a = sample_edge_pixels(hid, ai, bi)
        pix_b = sample_edge_pixels(nid, bj, aj)

        diff_fwd = np.mean(np.abs(pix_a - pix_b))
        diff_rev = np.mean(np.abs(pix_a - pix_b[::-1]))
        status = "MATCH" if diff_fwd < diff_rev else "REVERSED"
        hh_total += 1
        if diff_fwd < diff_rev:
            hh_match += 1
        else:
            hh_rev += 1

print(f"Hex-hex edges: {hh_match}/{hh_total} MATCH, {hh_rev} REVERSED")

# ══════════════════════════════════════════════════════════════════
# 5. WARP CORNER PAIRING ANALYSIS
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("5. WARP CORNER PAIRING — grid_corners vs uv_corners")
print("=" * 70)

for fid in ["t0", "t1"]:
    n_sides = len(gg.faces[fid].vertex_ids)
    kind = "PENT" if n_sides == 5 else "HEX"
    dg, _ = coll.get(fid)
    is_pent = n_sides == 5
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)

    gc_raw = get_macro_edge_corners(dg, n_sides)
    uv_c = get_tile_uv_vertices(gg, fid)
    gc_matched = match_grid_corners_to_uv(gc_raw, gg, fid)

    sa_gc_raw = _signed_area_2d(gc_raw)
    sa_gc_matched = _signed_area_2d(gc_matched)
    sa_uv = _signed_area_2d(list(uv_c))

    print(f"\n{fid} ({kind}):")
    print(f"  gc_raw winding:     {sa_gc_raw:+.4f} ({'CCW' if sa_gc_raw > 0 else 'CW'})")
    print(f"  gc_matched winding: {sa_gc_matched:+.4f} ({'CCW' if sa_gc_matched > 0 else 'CW'})")
    print(f"  uv_corners winding: {sa_uv:+.4f} ({'CCW' if sa_uv > 0 else 'CW'})")

    # Show the actual corner positions
    for i in range(n_sides):
        gcr = gc_raw[i]
        gcm = gc_matched[i]
        uvc = uv_c[i]
        print(f"  corner[{i}]: gc_raw=({gcr[0]:+.3f},{gcr[1]:+.3f})  "
              f"gc_matched=({gcm[0]:+.3f},{gcm[1]:+.3f})  "
              f"uv=({uvc[0]:.3f},{uvc[1]:.3f})")


# ══════════════════════════════════════════════════════════════════
# 6. ACTUAL PIXEL COMPARISON AT SPECIFIC EDGE LOCATIONS
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("6. PIXEL DEEP DIVE — specific edge locations")
print("=" * 70)

# Pick pentagon t0 and its first neighbour
pid = "t0"
face = gg.faces[pid]
nid = face.neighbor_ids[0]
result = find_shared_edge(pid, nid)
if result:
    ai, bi, aj, bj = result
    print(f"\nShared edge: {pid}[{ai}→{bi}] ↔ {nid}[{bj}→{aj}]")

    pent_verts = get_tile_uv_vertices(gg, pid)
    hex_verts = get_tile_uv_vertices(gg, nid)

    # Sample at 5 evenly spaced points along the edge
    for t_idx in range(5):
        t = (t_idx + 0.5) / 5
        inset = 0.05

        # Pentagon side
        a_p = np.array(pent_verts[ai])
        b_p = np.array(pent_verts[bi])
        center_p = np.mean(pent_verts, axis=0)
        pt_p = (a_p * (1 - t) + b_p * t) + inset * (center_p - (a_p * (1 - t) + b_p * t))
        px_p, py_p = uv_to_atlas_px(pid, pt_p[0], pt_p[1])
        col_p = atlas[py_p, px_p]

        # Hex side (reversed direction on the shared edge)
        a_h = np.array(hex_verts[bj])
        b_h = np.array(hex_verts[aj])
        center_h = np.mean(hex_verts, axis=0)
        pt_h = (a_h * (1 - t) + b_h * t) + inset * (center_h - (a_h * (1 - t) + b_h * t))
        px_h, py_h = uv_to_atlas_px(nid, pt_h[0], pt_h[1])
        col_h = atlas[py_h, px_h]

        diff = np.abs(col_p.astype(float) - col_h.astype(float)).mean()
        print(f"  t={t:.1f}: pent={col_p} hex={col_h} diff={diff:.1f}")


# ══════════════════════════════════════════════════════════════════
# 7. ASSEMBLY CODE PATH — trace what _position_hex_for_stitch does
# ══════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("7. ASSEMBLY PATH TRACE")
print("=" * 70)

# Check whether the original code (before our changes) would have
# reflected the grid, and what the current code does
for fid in ["t0", "t1"]:
    n_sides = len(gg.faces[fid].vertex_ids)
    kind = "PENT" if n_sides == 5 else "HEX"
    dg_center, _ = coll.get(fid)
    dg_center.compute_macro_edges(
        n_sides=n_sides,
        corner_ids=dg_center.metadata.get("corner_vertex_ids"),
    )
    sa_center = _signed_area_macro(dg_center)

    neigh_map = compute_neighbor_edge_mapping(gg, fid)
    pg_to_macro = compute_pg_to_macro_edge_map(gg, fid, dg_center)

    print(f"\n{fid} ({kind}, SA={sa_center:+.4f}):")

    for nid, cpg in list(neigh_map.items())[:2]:  # first 2 neighbours
        n_sides_n = len(gg.faces[nid].vertex_ids)
        nkind = "pent" if n_sides_n == 5 else "hex"
        dg_n, _ = coll.get(nid)
        dg_n.compute_macro_edges(
            n_sides=n_sides_n,
            corner_ids=dg_n.metadata.get("corner_vertex_ids"),
        )
        sa_n = _signed_area_macro(dg_n)
        same_winding = (sa_center > 0) == (sa_n > 0)

        npg = compute_neighbor_edge_mapping(gg, nid)[fid]
        cm = pg_to_macro[cpg]
        pm = compute_pg_to_macro_edge_map(gg, nid, dg_n)[npg]

        positioned = _position_hex_for_stitch(dg_center, cm, dg_n, pm)
        sa_pos = _signed_area_macro(positioned)
        winding_changed = (sa_n > 0) != (sa_pos > 0)

        print(f"  → {nid}({nkind}): sa_orig={sa_n:+.4f} sa_pos={sa_pos:+.4f} "
              f"same_w={same_winding} winding_changed={winding_changed} "
              f"flip_in_stitch={same_winding}")


print()
print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
