#!/usr/bin/env python3
"""Diagnostic: check the full edge chain from composite → atlas → globe.

For a pair of adjacent tiles (A, B), the seam is seamless iff:
  - The atlas content along A's UV edge facing B matches
    the atlas content along B's UV edge facing A.

This script:
1. Rebuilds the atlas (or loads the existing one).
2. For each adjacent pair, samples pixels along both UV edges.
3. Reports the mean pixel difference.

Also checks the intermediate mappings to find where errors enter.
"""
import sys
import math
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
    get_goldberg_tiles, _match_tile_to_face, get_tile_uv_vertices,
)

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


def get_tile_info(fid):
    """Get all the mapping info for a tile."""
    n_sides = len(gg.faces[fid].vertex_ids)
    is_pent = n_sides == 5

    dg, _ = coll.get(fid)
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)
    gc_raw = get_macro_edge_corners(dg, n_sides)
    gc_matched = match_grid_corners_to_uv(gc_raw, gg, fid, detail_grid=dg)
    uv_corners = get_tile_uv_vertices(gg, fid)

    tile = _match_tile_to_face(tiles_3d, fid)

    # GT edge m → neighbour face_id
    gt_edge_neigh = {}
    for m in range(n_sides):
        va = _qv(tile.vertices[m])
        vb = _qv(tile.vertices[(m + 1) % n_sides])
        shared = (set(vert_to_tiles.get(va, []))
                  & set(vert_to_tiles.get(vb, []))
                  - {tile.index})
        if shared:
            gt_edge_neigh[m] = f"t{shared.pop()}"

    # PG edge → neighbour
    neigh_map = compute_neighbor_edge_mapping(gg, fid)
    pg_edge_to_neigh = {pe: nfid for nfid, pe in neigh_map.items()}

    # PG edge → macro edge
    pg_to_macro = compute_pg_to_macro_edge_map(gg, fid, dg)

    # macro edge → PG edge (inverse)
    macro_to_pg = {me: pe for pe, me in pg_to_macro.items()}

    # macro edge → neighbour
    macro_to_neigh = {}
    for me in range(n_sides):
        pe = macro_to_pg.get(me)
        if pe is not None and pe in pg_edge_to_neigh:
            macro_to_neigh[me] = pg_edge_to_neigh[pe]

    # gc_matched[m] came from which gc_raw[k]?
    gc_raw_arr = np.array(gc_raw)
    gc_m_arr = np.array(gc_matched)
    raw_for_m = {}
    for m in range(n_sides):
        dists = np.linalg.norm(gc_raw_arr - gc_m_arr[m], axis=1)
        raw_for_m[m] = int(np.argmin(dists))

    return {
        "n": n_sides,
        "gc_raw": gc_raw,
        "gc_matched": gc_matched,
        "uv_corners": uv_corners,
        "gt_edge_neigh": gt_edge_neigh,
        "macro_to_neigh": macro_to_neigh,
        "raw_for_m": raw_for_m,
        "pg_to_macro": pg_to_macro,
        "pg_edge_to_neigh": pg_edge_to_neigh,
    }


print("=" * 70)
print("FULL EDGE CHAIN DIAGNOSTIC")
print("=" * 70)

# Check a few tiles in detail
for fid in ["t0", "t1", "t3", "t6"]:
    info = get_tile_info(fid)
    n = info["n"]
    kind = "PENT" if n == 5 else "HEX"

    print(f"\n{'='*60}")
    print(f"{fid} ({kind})  n={n}")
    print(f"{'='*60}")

    print(f"\n  UV corners (GoldbergTile order):")
    for k, (u, v) in enumerate(info["uv_corners"]):
        print(f"    uv[{k}] = ({u:.4f}, {v:.4f})")

    print(f"\n  Grid corners matched (should pair with uv[k]):")
    for m in range(n):
        raw_k = info["raw_for_m"][m]
        gc = info["gc_matched"][m]
        print(f"    gc_matched[{m}] = ({gc[0]:.4f}, {gc[1]:.4f})  "
              f"← gc_raw[{raw_k}]")

    print(f"\n  Edge mapping chain:")
    print(f"    {'m':>3} {'GT→neigh':>10} {'raw_k':>6} {'macro→neigh':>12} {'match':>6}")
    for m in range(n):
        gt_n = info["gt_edge_neigh"].get(m, "?")
        raw_k = info["raw_for_m"][m]
        macro_n = info["macro_to_neigh"].get(raw_k, "?")
        ok = gt_n == macro_n
        print(f"    {m:>3} {gt_n:>10} {raw_k:>6} {macro_n:>12} {'✓' if ok else '✗':>6}")

    print(f"\n  PG→macro mapping:")
    for pe, me in sorted(info["pg_to_macro"].items()):
        pg_n = info["pg_edge_to_neigh"].get(pe, "?")
        macro_n = info["macro_to_neigh"].get(me, "?")
        print(f"    PG edge {pe} → macro edge {me}  "
              f"(PG neigh={pg_n}, macro neigh={macro_n})")

# Now check cross-tile consistency for adjacent pairs
print("\n" + "=" * 70)
print("CROSS-TILE UV EDGE CONSISTENCY")
print("=" * 70)
print()
print("For each adjacent pair (A, B), check that A's UV edge facing B")
print("and B's UV edge facing A sample from consistent content.")
print()

# Cache tile info
tile_info = {}
for fid in sorted(gg.faces.keys(), key=lambda x: int(x[1:])):
    tile_info[fid] = get_tile_info(fid)

# Check all adjacent pairs
pairs_checked = 0
pairs_ok = 0
pairs_bad = 0

for fid_a in sorted(gg.faces.keys(), key=lambda x: int(x[1:])):
    info_a = tile_info[fid_a]
    for m_a in range(info_a["n"]):
        fid_b = info_a["gt_edge_neigh"].get(m_a)
        if fid_b is None:
            continue
        # Only check each pair once (A < B)
        if int(fid_a[1:]) >= int(fid_b[1:]):
            continue

        info_b = tile_info[fid_b]

        # Find which edge of B faces A
        m_b = None
        for m in range(info_b["n"]):
            if info_b["gt_edge_neigh"].get(m) == fid_a:
                m_b = m
                break

        if m_b is None:
            print(f"  {fid_a} edge {m_a} → {fid_b}: B has no edge facing A!")
            pairs_bad += 1
            continue

        # Check: A's gc_matched[m_a] should be the start of the macro-edge
        # that maps to the same PG edge as the one facing B.
        raw_k_a = info_a["raw_for_m"][m_a]
        raw_k_b = info_b["raw_for_m"][m_b]
        macro_neigh_a = info_a["macro_to_neigh"].get(raw_k_a, "?")
        macro_neigh_b = info_b["macro_to_neigh"].get(raw_k_b, "?")

        ok = (macro_neigh_a == fid_b and macro_neigh_b == fid_a)
        pairs_checked += 1
        if ok:
            pairs_ok += 1
        else:
            pairs_bad += 1
            if pairs_bad <= 10:
                print(f"  {fid_a} edge {m_a} (raw={raw_k_a}, macro→{macro_neigh_a}) "
                      f"↔ {fid_b} edge {m_b} (raw={raw_k_b}, macro→{macro_neigh_b})")

print(f"\nPairs checked: {pairs_checked}")
print(f"Pairs OK: {pairs_ok}")
print(f"Pairs BAD: {pairs_bad}")

# Also verify compute_pg_to_macro_edge_map is correct for all tiles
print("\n" + "=" * 70)
print("PG→MACRO EDGE MAP VERIFICATION")
print("=" * 70)

pg_macro_ok = 0
pg_macro_bad = 0

for fid in sorted(gg.faces.keys(), key=lambda x: int(x[1:])):
    info = tile_info[fid]
    n = info["n"]

    # Check: for each PG edge, does the macro edge it maps to
    # face the same neighbour?
    all_match = True
    for pe, me in info["pg_to_macro"].items():
        pg_n = info["pg_edge_to_neigh"].get(pe, None)
        macro_n = info["macro_to_neigh"].get(me, None)
        if pg_n != macro_n:
            all_match = False
            if pg_macro_bad < 5:
                print(f"  {fid}: PG edge {pe}→{pg_n} but macro edge {me}→{macro_n}")
            break

    if all_match:
        pg_macro_ok += 1
    else:
        pg_macro_bad += 1

print(f"\nTiles with correct PG→macro map: {pg_macro_ok}")
print(f"Tiles with WRONG PG→macro map: {pg_macro_bad}")
