#!/usr/bin/env python3
"""Verify whether pentagon and hex gc_matched have consistent winding."""
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
from polygrid.rendering.uv_texture import get_tile_uv_vertices, get_goldberg_tiles
from polygrid.rendering.tile_uv_align import (
    get_macro_edge_corners,
    match_grid_corners_to_uv,
    compute_pg_to_macro_corner_map,
    compute_gt_to_pg_corner_map,
)
from polygrid.detail.detail_grid import build_detail_grid

FREQ = 2
DETAIL_RINGS = 4


def winding_sign(pts):
    """Shoelace winding sign. >0 = CCW in standard axes, <0 = CW."""
    n = len(pts)
    s = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        s += (x1 - x0) * (y1 + y0)
    return -1 if s > 0 else (1 if s < 0 else 0)


globe = build_globe_grid(FREQ)
tiles_gb = get_goldberg_tiles(FREQ, 1.0)

print("Face  | Sides | Mode       | gc_matched winding | uv_corners winding | Same?")
print("------|-------|------------|--------------------|--------------------|------")

for fid in sorted(globe.faces.keys(), key=lambda f: int(f[1:])):
    ns = len(globe.faces[fid].vertex_ids)
    is_pent = ns == 5
    dg = build_detail_grid(globe, fid, detail_rings=DETAIL_RINGS)
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=ns, corner_ids=corner_ids)
    gc_raw = get_macro_edge_corners(dg, ns)
    uc_raw = get_tile_uv_vertices(globe, fid)

    if is_pent:
        try:
            gt_to_pg = compute_gt_to_pg_corner_map(globe, fid)
            pg_to_macro = compute_pg_to_macro_corner_map(globe, fid, dg)
            gc_matched = [
                gc_raw[pg_to_macro[gt_to_pg[k]]]
                for k in range(ns)
            ]
            mode = "topology"
        except Exception as e:
            gc_matched = match_grid_corners_to_uv(gc_raw, globe, fid, allow_reflection_override=True)
            mode = "angle-ref"
    else:
        gc_matched = match_grid_corners_to_uv(gc_raw, globe, fid, allow_reflection_override=None)
        mode = "angle-ref"

    gc_sign = winding_sign(gc_matched)
    uv_sign = winding_sign(uc_raw)
    same = "YES" if gc_sign == uv_sign else "NO"
    
    label = "pent" if is_pent else "hex"
    print(f"{fid:5s} | {ns}     | {mode:10s} | {gc_sign:+d} ({['CW','zero','CCW'][gc_sign+1]:>4s})        | {uv_sign:+d} ({['CW','zero','CCW'][uv_sign+1]:>4s})        | {same}")
