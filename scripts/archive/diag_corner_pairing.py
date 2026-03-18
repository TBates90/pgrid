#!/usr/bin/env python3
"""Diagnostic: verify corner pairing chain for pentagon and hex tiles.

For each tile, prints:
  1. PG vertex_ids order (3D coords)
  2. GT vertex order (3D coords)
  3. gt_offset from compute_uv_to_polygrid_offset
  4. Macro-edge corner order (2D Tutte coords) + which PG vertex_id each corner matches
  5. UV corners (GT order) from get_tile_uv_vertices
  6. The final paired corners after offset rotation

This reveals whether macro-edge corners are in PG vertex_ids order or not.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from polygrid.goldberg_topology import build_goldberg_grid
from polygrid.detail_grid import build_detail_grid
from polygrid.tile_uv_align import (
    compute_uv_to_polygrid_offset,
    get_macro_edge_corners,
    match_grid_corners_to_uv,
)
from polygrid.uv_texture import get_tile_uv_vertices, get_goldberg_tiles, _match_tile_to_face, compute_tile_basis
from polygrid.detail_terrain import compute_neighbor_edge_mapping


def main():
    freq = 3
    from polygrid.globe import build_globe_grid
    globe = build_globe_grid(freq)
    tiles_gt = get_goldberg_tiles(freq, 1.0)

    # Pick a couple of pentagons and hexagons
    pent_ids = [fid for fid, f in globe.faces.items() if len(f.vertex_ids) == 5]
    hex_ids = [fid for fid, f in globe.faces.items() if len(f.vertex_ids) == 6][:5]

    # Quick summary check for all tiles
    print("=== SUMMARY: match quality across all tiles ===")
    all_ok = True
    for fid in list(globe.faces.keys()):
        face = globe.faces[fid]
        n = len(face.vertex_ids)
        tile_gt = _match_tile_to_face(tiles_gt, fid)
        
        center_3d, _, tangent_3d, bitangent_3d = compute_tile_basis(globe, fid)
        gt_angles = []
        for vtx in tile_gt.vertices:
            rel = np.array(vtx, dtype=np.float64) - center_3d
            u = float(np.dot(rel, tangent_3d))
            v = float(np.dot(rel, bitangent_3d))
            gt_angles.append(np.arctan2(v, u))
        
        dg = build_detail_grid(globe, fid, detail_rings=4)
        corner_ids = dg.metadata.get("corner_vertex_ids")
        dg.compute_macro_edges(n_sides=n, corner_ids=corner_ids)
        grid_corners = get_macro_edge_corners(dg, n)
        
        matched = match_grid_corners_to_uv(grid_corners, globe, fid)
        
        mc = np.array(matched)
        mc_centroid = mc.mean(axis=0)
        mc_angles = np.arctan2(mc[:, 1] - mc_centroid[1], mc[:, 0] - mc_centroid[0])
        
        max_diff = 0
        for k in range(n):
            diff = abs(mc_angles[k] - gt_angles[k])
            if diff > np.pi:
                diff = 2*np.pi - diff
            max_diff = max(max_diff, diff)
        
        status = "OK" if np.degrees(max_diff) < 15 else "BAD"
        if status == "BAD":
            all_ok = False
        kind = "pent" if n == 5 else "hex"
        print(f"  {fid} ({kind}): max_angle_diff={np.degrees(max_diff):.1f}° [{status}]")
    
    print(f"\n{'ALL OK' if all_ok else 'SOME TILES HAVE BAD MATCHING'}")
    if not all_ok:
        return

    # Detailed output for a few selected tiles
    for fid in pent_ids[:1] + hex_ids[:1]:
        face = globe.faces[fid]
        n = len(face.vertex_ids)
        print(f"\n{'='*70}")
        print(f"Tile {fid}  (n_sides={n})")
        print(f"{'='*70}")

        # --- PG vertex_ids and 3D coords ---
        print(f"\nPG vertex_ids order:")
        pg_3d = []
        for i, vid in enumerate(face.vertex_ids):
            v = globe.vertices[vid]
            c = np.array([v.x, v.y, v.z])
            pg_3d.append(c)
            print(f"  PG[{i}] vid={vid}  ({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f})")

        # --- GT vertices ---
        tile_gt = _match_tile_to_face(tiles_gt, fid)
        print(f"\nGT vertex order:")
        gt_3d = [np.array(v) for v in tile_gt.vertices]
        for i, c in enumerate(gt_3d):
            print(f"  GT[{i}]  ({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f})")

        # --- GT offset ---
        offset = compute_uv_to_polygrid_offset(globe, fid)
        print(f"\ngt_offset = {offset}")
        print(f"  Meaning: PG[k] == GT[(k - offset) % {n}]")
        # Verify
        for pi in range(n):
            gi = (pi - offset) % n
            dist = np.linalg.norm(pg_3d[pi] - gt_3d[gi])
            print(f"  PG[{pi}] ↔ GT[{gi}]  dist={dist:.8f}")

        # --- Build detail grid and compute macro edges ---
        dg = build_detail_grid(globe, fid, detail_rings=4)
        corner_ids = dg.metadata.get("corner_vertex_ids")
        dg.compute_macro_edges(n_sides=n, corner_ids=corner_ids)
        grid_corners = get_macro_edge_corners(dg, n)

        print(f"\nMacro-edge corners (Tutte 2D, metadata corners):")
        for k in range(n):
            me = next(m for m in dg.macro_edges if m.id == k)
            print(f"  macro_edge[{k}] corner_start={me.corner_start}  "
                  f"2D=({grid_corners[k][0]:.6f}, {grid_corners[k][1]:.6f})")

        # --- Which 3D vertex does each macro corner map to? ---
        # The detail grid corners are 2D Tutte vertices. We need
        # to figure out which PG (and thus GT) vertex each maps to.
        #
        # compute_neighbor_edge_mapping uses face.vertex_ids[k] to
        # face.vertex_ids[(k+1)%n] as "edge k".  If macro_edge[k] in
        # the detail grid corresponds to PG edge k, then the pairing
        # is straightforward.  But we need to verify this.
        #
        # Strategy: use compute_neighbor_edge_mapping to find which
        # neighbour is on each PG edge, then check which macro_edge
        # face each detail-grid macro edge maps to.
        
        # Instead, let's use the 2D angles of the macro corners from
        # the grid center, and compare with the projected 3D angles
        # of PG/GT vertices in the tangent plane.
        center_3d, normal_3d, tangent_3d, bitangent_3d = compute_tile_basis(globe, fid)
        
        print(f"\nMacro-corner to GT vertex matching via angle:")
        # Project GT vertices onto tangent plane
        gt_angles = []
        for i, gv in enumerate(gt_3d):
            rel = gv - center_3d
            u = float(np.dot(rel, tangent_3d))
            v = float(np.dot(rel, bitangent_3d))
            a = np.arctan2(v, u)
            gt_angles.append(a)
            # print(f"  GT[{i}] angle={np.degrees(a):.1f}°")
        
        # Compute angles of macro corners from detail grid center
        gc = np.array([sum(c[0] for c in grid_corners)/n,
                       sum(c[1] for c in grid_corners)/n])
        macro_angles = []
        for k in range(n):
            dx = grid_corners[k][0] - gc[0]
            dy = grid_corners[k][1] - gc[1]
            a = np.arctan2(dy, dx)
            macro_angles.append(a)
        
        # Match each macro corner to its nearest GT vertex by angle
        # The detail grid is a Tutte embedding so its angles should
        # preserve the cyclic order of the 3D polygon vertices.
        gt_used = set()
        macro_to_gt = {}
        for k in range(n):
            best_gi = -1
            best_diff = 999
            for gi in range(n):
                if gi in gt_used:
                    continue
                diff = abs(macro_angles[k] - gt_angles[gi])
                if diff > np.pi:
                    diff = 2*np.pi - diff
                if diff < best_diff:
                    best_diff = diff
                    best_gi = gi
            macro_to_gt[k] = best_gi
            gt_used.add(best_gi)
            print(f"  macro_corner[{k}] angle={np.degrees(macro_angles[k]):.1f}° "
                  f"→ GT[{best_gi}] angle={np.degrees(gt_angles[best_gi]):.1f}° "
                  f"(diff={np.degrees(best_diff):.1f}°)")
        
        # Check if macro_to_gt is a pure rotation
        offsets = [(macro_to_gt[k] - k) % n for k in range(n)]
        if len(set(offsets)) == 1:
            print(f"  → Pure rotation: macro_to_gt offset = {offsets[0]}")
        else:
            print(f"  → NOT a pure rotation! offsets = {offsets}")
            # Check if it's a reflection
            reflected = [(macro_to_gt[k] + k) % n for k in range(n)]
            if len(set(reflected)) == 1:
                print(f"  → Reflection + rotation: sum = {reflected[0]}")
            else:
                print(f"  → Complex mapping")

        # --- Neighbour edge mapping (PG order) ---
        neigh_map = compute_neighbor_edge_mapping(globe, fid)
        print(f"\nNeighbour edge mapping (PG vertex_ids order):")
        for nid, eidx in sorted(neigh_map.items(), key=lambda x: x[1]):
            print(f"  edge {eidx}: {fid} ↔ {nid}")

        # --- UV corners ---
        uv_corners = get_tile_uv_vertices(globe, fid)
        print(f"\nUV corners (GT order):")
        for i, (u, v) in enumerate(uv_corners):
            print(f"  UV[{i}]  ({u:.6f}, {v:.6f})")

        # --- Test the new match_grid_corners_to_uv function ---
        matched_corners = match_grid_corners_to_uv(grid_corners, globe, fid)
        
        print(f"\nNew match_grid_corners_to_uv result:")
        for k in range(n):
            print(f"  k={k}: matched_corner = {matched_corners[k]} "
                  f"↔ uv_corner = {uv_corners[k]}")
        
        # Verify: angles of matched corners should follow GT angles
        mc = np.array(matched_corners)
        mc_centroid = mc.mean(axis=0)
        mc_angles = np.arctan2(mc[:, 1] - mc_centroid[1], mc[:, 0] - mc_centroid[0])
        print(f"\n  Angular check:")
        for k in range(n):
            diff = abs(mc_angles[k] - gt_angles[k])
            if diff > np.pi:
                diff = 2*np.pi - diff
            print(f"  matched[{k}] angle={np.degrees(mc_angles[k]):.1f}° "
                  f"vs GT[{k}] angle={np.degrees(gt_angles[k]):.1f}° "
                  f"(diff={np.degrees(diff):.1f}°)")


if __name__ == "__main__":
    main()
