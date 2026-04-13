#!/usr/bin/env python3
"""Compare topology-driven vs angle-based corner matching for pentagons.

If these disagree, the atlas warp is using a different corner pairing
than what's expected, which could cause the visual discontinuity.
"""
import sys
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
from polygrid.rendering.uv_texture import get_tile_uv_vertices
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

print("Comparing topology-driven vs angle-based corner matching for all 12 pentagons")
print("=" * 80)
any_mismatch = False

for fid in sorted(globe.faces.keys(), key=lambda f: int(f[1:])):
    ns = len(globe.faces[fid].vertex_ids)
    if ns != 5:
        continue
    
    dg = build_detail_grid(globe, fid, detail_rings=DETAIL_RINGS)
    corner_ids = dg.metadata.get("corner_vertex_ids")
    dg.compute_macro_edges(n_sides=ns, corner_ids=corner_ids)
    gc_raw = get_macro_edge_corners(dg, ns)
    
    # Method A: Topology-driven (what build_polygon_cut_atlas actually uses)
    try:
        gt_to_pg = compute_gt_to_pg_corner_map(globe, fid)
        pg_to_macro = compute_pg_to_macro_corner_map(globe, fid, dg)
        gc_topo = [gc_raw[pg_to_macro[gt_to_pg[k]]] for k in range(ns)]
        topo_ok = True
    except Exception as e:
        gc_topo = None
        topo_ok = False
        print(f"  {fid}: topology FAILED: {e}")
    
    # Method B: Angle-based (what the test uses)
    gc_angle = match_grid_corners_to_uv(gc_raw, globe, fid, allow_reflection_override=None)
    
    # Method C: Angle-based with reflection allowed (fallback for pentagon)
    gc_angle_ref = match_grid_corners_to_uv(gc_raw, globe, fid, allow_reflection_override=True)
    
    if topo_ok:
        # Compare: are the corner orderings the same?
        raw_to_idx = {id(c): i for i, c in enumerate(gc_raw)}
        
        topo_indices = []
        angle_indices = []
        angle_ref_indices = []
        
        for k in range(ns):
            # Find which gc_raw index each method selected for position k
            for ri, rc in enumerate(gc_raw):
                if rc[0] == gc_topo[k][0] and rc[1] == gc_topo[k][1]:
                    topo_indices.append(ri)
                    break
            for ri, rc in enumerate(gc_raw):
                if rc[0] == gc_angle[k][0] and rc[1] == gc_angle[k][1]:
                    angle_indices.append(ri)
                    break
            for ri, rc in enumerate(gc_raw):
                if rc[0] == gc_angle_ref[k][0] and rc[1] == gc_angle_ref[k][1]:
                    angle_ref_indices.append(ri)
                    break
        
        match_topo_angle = topo_indices == angle_indices
        match_topo_angleref = topo_indices == angle_ref_indices
        
        status = "MATCH" if match_topo_angle else "MISMATCH"
        if not match_topo_angle:
            any_mismatch = True
        
        print(f"{fid}: {status}")
        print(f"  topology:         {topo_indices}")
        print(f"  angle(no-ref):    {angle_indices}  {'✓' if match_topo_angle else '✗'}")
        print(f"  angle(allow-ref): {angle_ref_indices}  {'✓' if match_topo_angleref else '✗'}")
        
        if not match_topo_angle:
            # Show which corners differ
            for k in range(ns):
                if topo_indices[k] != angle_indices[k]:
                    print(f"    Position {k}: topo→raw[{topo_indices[k]}] vs angle→raw[{angle_indices[k]}]")

print()
if any_mismatch:
    print("⚠️  MISMATCHES FOUND! Topology and angle-based paths disagree.")
    print("   This means the atlas warp uses different corner pairings than expected.")
else:
    print("✓ All 12 pentagons: topology and angle-based paths AGREE.")
