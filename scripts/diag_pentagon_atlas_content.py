#!/usr/bin/env python3
"""Diagnose pentagon atlas content orientation.

Build a placeholder atlas at f=2 d=4, extract the pentagon tiles,
and verify that:
1. Edge labels (which neighbor is on which edge) match what the 3D mesh expects
2. The winding conventions are consistent
3. The detail grid cell-to-UV assignment is correct

Prints a per-pentagon diagnostic and generates debug images.
"""
import sys
import math
import json
import os
from pathlib import Path

# Activate venv
venv_path = Path(__file__).resolve().parent.parent / ".venv" / "lib"
for p in sorted(venv_path.glob("python3.*")):
    sp = str(p / "site-packages")
    if sp not in sys.path:
        sys.path.insert(0, sp)

src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np

# ─── Build globe grid and tiles ─────────────────────────────────
from polygrid.globe.globe import build_globe_grid
from polygrid.rendering.uv_texture import get_goldberg_tiles, get_tile_uv_vertices, compute_tile_basis
from polygrid.rendering.globe_renderer_v2 import build_batched_globe_mesh

FREQ = 2

globe_grid = build_globe_grid(FREQ)
face_ids = sorted(globe_grid.faces.keys(), key=lambda f: int(f[1:]))

tiles = get_goldberg_tiles(FREQ, 1.0)
tile_by_fid = {f"t{t.index}": t for t in tiles}


def get_tile_neighbors(gg, fid):
    """Return ordered neighbor face IDs from globe grid."""
    return list(gg.faces[fid].neighbor_ids)


def angular_order_3d(tile):
    """Return CCW angular ordering of tile vertices in tangent plane."""
    c = np.array(tile.center, dtype=float)
    t = np.array(tile.tangent, dtype=float)
    b = np.array(tile.bitangent, dtype=float)
    angles = []
    for i, v in enumerate(tile.vertices):
        rel = np.array(v, dtype=float) - c
        u = float(np.dot(rel, t))
        vv = float(np.dot(rel, b))
        angles.append((i, math.atan2(vv, u)))
    return angles


print("=" * 70)
print("PENTAGON ATLAS CONTENT DIAGNOSTIC — f=2")
print("=" * 70)

# ─── For each pentagon tile, compare 3D edge structure with atlas expectations ─
pent_fids = [fid for fid in face_ids if len(globe_grid.faces[fid].vertex_ids) == 5]

for fid in pent_fids:
    tile = tile_by_fid[fid]
    face = globe_grid.faces[fid]
    n_sides = len(face.vertex_ids)
    neighbors = get_tile_neighbors(globe_grid, fid)

    print(f"\n{'─' * 70}")
    print(f"Tile: {fid} (pentagon, {n_sides} sides)")
    print(f"Globe grid neighbors (PG vertex_ids order): {neighbors}")

    # GoldbergTile vertex angles in tangent plane
    gt_angles = angular_order_3d(tile)
    print(f"GoldbergTile vertex angles (rad):")
    for i, a in gt_angles:
        print(f"  v{i}: {a:.4f} rad ({math.degrees(a):.2f}°)")

    # UV vertices
    uv_verts = get_tile_uv_vertices(globe_grid, fid)
    print(f"UV vertices (from get_tile_uv_vertices):")
    for i, (u, v) in enumerate(uv_verts):
        print(f"  uv{i}: ({u:.6f}, {v:.6f})")

    # GoldbergTile has .neighbor_ids — check if it exists
    if hasattr(tile, 'neighbor_ids') and tile.neighbor_ids:
        print(f"GoldbergTile neighbor_ids: {tile.neighbor_ids}")

    # ── Compute GT edge -> neighbor mapping via 3D vertex sharing ──
    # GT edge k connects tile.vertices[k] and tile.vertices[(k+1)%n]
    # The shared edge with a neighbor is identified by shared 3D vertices
    print(f"\n  GT edge -> neighbor mapping (from 3D vertex sharing):")

    # Build vertex -> tile index lookup
    vertex_sharing = {}  # (rounded vertex) -> set of tile indices
    for other_tile in tiles:
        for vi, v in enumerate(other_tile.vertices):
            key = tuple(round(c, 6) for c in v)
            if key not in vertex_sharing:
                vertex_sharing[key] = set()
            vertex_sharing[key].add(other_tile.index)

    for edge_k in range(n_sides):
        v0 = tile.vertices[edge_k]
        v1 = tile.vertices[(edge_k + 1) % n_sides]
        k0 = tuple(round(c, 6) for c in v0)
        k1 = tuple(round(c, 6) for c in v1)
        shared_tiles = vertex_sharing.get(k0, set()) & vertex_sharing.get(k1, set())
        shared_tiles.discard(tile.index)
        shared_neighbor = shared_tiles.pop() if shared_tiles else None
        neigh_fid = f"t{shared_neighbor}" if shared_neighbor is not None else "???"
        print(f"    GT edge {edge_k} (v{edge_k}-v{(edge_k + 1) % n_sides}): shared with {neigh_fid}")

    # ── Now check the PG (PolyGrid) edge -> neighbor mapping ──
    from polygrid.detail.detail_terrain import compute_neighbor_edge_mapping
    pg_neigh_edge = compute_neighbor_edge_mapping(globe_grid, fid)
    print(f"\n  PG edge -> neighbor mapping (compute_neighbor_edge_mapping):")
    for nid, eidx in sorted(pg_neigh_edge.items(), key=lambda x: int(x[1])):
        print(f"    PG edge {eidx}: neighbor {nid}")

    # ── Compute the PG->GT offset ──
    from polygrid.rendering.tile_uv_align import compute_uv_to_polygrid_offset
    try:
        pg_gt_offset = compute_uv_to_polygrid_offset(globe_grid, fid)
        print(f"\n  PG->GT offset (compute_uv_to_polygrid_offset): {pg_gt_offset}")
    except Exception as e:
        pg_gt_offset = None
        print(f"\n  PG->GT offset: FAILED ({e})")

    # ── Compute the full corner mapping chain ──
    from polygrid.rendering.tile_uv_align import compute_gt_to_pg_corner_map, compute_pg_to_macro_corner_map, get_macro_edge_corners
    from polygrid.detail.tile_detail import DetailGridCollection, TileDetailSpec

    spec = TileDetailSpec(detail_rings=4)
    coll = DetailGridCollection.build(globe_grid, spec)
    dg = coll.get(fid)[0]

    corner_ids = dg.metadata.get("corner_vertex_ids")
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)
    gc_raw = get_macro_edge_corners(dg, n_sides)

    print(f"\n  Macro-edge corners (gc_raw) — Tutte 2D positions:")
    for i, (x, y) in enumerate(gc_raw):
        print(f"    macro[{i}]: ({x:.6f}, {y:.6f})")

    try:
        gt_to_pg = compute_gt_to_pg_corner_map(globe_grid, fid)
        pg_to_macro = compute_pg_to_macro_corner_map(globe_grid, fid, dg)
        gc_matched = [gc_raw[pg_to_macro[gt_to_pg[k]]] for k in range(n_sides)]

        print(f"\n  Corner mapping chain:")
        for k in range(n_sides):
            pg_idx = gt_to_pg[k]
            macro_idx = pg_to_macro[pg_idx]
            print(f"    GT[{k}] -> PG[{pg_idx}] -> macro[{macro_idx}] -> gc=({gc_matched[k][0]:.6f}, {gc_matched[k][1]:.6f})")

        # Verify pairing with UV corners by checking angular consistency
        center_3d, _, tan_3d, bitan_3d = compute_tile_basis(globe_grid, fid)
        gc_centroid = np.mean(gc_matched, axis=0)
        print(f"\n  Angular verification (GT angle vs GC angle):")
        for k in range(n_sides):
            vtx = np.array(tile.vertices[k], dtype=float)
            rel = vtx - center_3d
            gt_angle = math.atan2(float(np.dot(rel, bitan_3d)), float(np.dot(rel, tan_3d)))

            gc = np.array(gc_matched[k], dtype=float)
            rel_gc = gc - gc_centroid
            gc_angle = math.atan2(rel_gc[1], rel_gc[0])

            diff = math.degrees(gt_angle - gc_angle)
            # Normalize to [-180, 180]
            while diff > 180: diff -= 360
            while diff < -180: diff += 360
            print(f"    k={k}: GT={math.degrees(gt_angle):.2f}° GC={math.degrees(gc_angle):.2f}° diff={diff:.2f}°")

    except Exception as e:
        print(f"\n  Corner mapping FAILED: {e}")
        import traceback
        traceback.print_exc()

    # ── Check winding consistency ──
    print(f"\n  Winding analysis:")
    # UV winding
    uv_arr = np.array(uv_verts, dtype=float)
    uv_cross_sum = 0
    for i in range(n_sides):
        j = (i + 1) % n_sides
        uv_cross_sum += uv_arr[i, 0] * uv_arr[j, 1] - uv_arr[j, 0] * uv_arr[i, 1]
    uv_winding = "CCW" if uv_cross_sum > 0 else "CW"
    print(f"    UV polygon winding: {uv_winding} (cross sum = {uv_cross_sum:.6f})")

    # GC_matched winding (Tutte space)
    if 'gc_matched' in dir():
        gc_arr = np.array(gc_matched, dtype=float)
        gc_cross_sum = 0
        for i in range(n_sides):
            j = (i + 1) % n_sides
            gc_cross_sum += gc_arr[i, 0] * gc_arr[j, 1] - gc_arr[j, 0] * gc_arr[i, 1]
        gc_winding = "CCW" if gc_cross_sum > 0 else "CW"
        print(f"    GC matched winding: {gc_winding} (cross sum = {gc_cross_sum:.6f})")

    # 3D vertex winding projected on tangent plane
    gt_proj = []
    for v in tile.vertices:
        rel = np.array(v, dtype=float) - center_3d
        gt_proj.append([float(np.dot(rel, tan_3d)), float(np.dot(rel, bitan_3d))])
    gt_proj = np.array(gt_proj)
    gt_cross_sum = 0
    for i in range(n_sides):
        j = (i + 1) % n_sides
        gt_cross_sum += gt_proj[i, 0] * gt_proj[j, 1] - gt_proj[j, 0] * gt_proj[i, 1]
    gt_winding_3d = "CCW" if gt_cross_sum > 0 else "CW"
    print(f"    3D projected winding: {gt_winding_3d} (cross sum = {gt_cross_sum:.6f})")

    print()


# ─── Now build the actual placeholder atlas and check ──────────
print("\n" + "=" * 70)
print("PLACEHOLDER ATLAS BUILD TEST")
print("=" * 70)

from polygrid.placeholder_atlas import PlaceholderAtlasArtifact

artifact = PlaceholderAtlasArtifact.build(frequency=FREQ, detail_rings=4, tile_size=128, gutter=4)

print(f"Atlas shape: {artifact.atlas_image.size}")
print(f"UV layout keys: {sorted(artifact.uv_layout.keys(), key=lambda f: int(f[1:]))}")
print(f"Vertex data shape: {artifact.vertex_data.shape}")
print(f"Index data shape: {artifact.index_data.shape}")

# Check if atlas was built successfully with valid UV layout
for fid in pent_fids:
    slot = artifact.uv_layout.get(fid)
    if slot is None:
        print(f"  WARNING: {fid} not in UV layout!")
    else:
        u_min, v_min, u_max, v_max = slot
        print(f"  {fid} UV slot: u=[{u_min:.4f}, {u_max:.4f}] v=[{v_min:.4f}, {v_max:.4f}]")

# ─── Build the 3D mesh from this atlas and compare ──────────────
print("\n" + "=" * 70)
print("3D MESH UV vs ATLAS UV COMPARISON")
print("=" * 70)

vdata, idata = build_batched_globe_mesh(
    frequency=FREQ,
    uv_layout=artifact.uv_layout,
    subdivisions=3,
)

print(f"Mesh: {len(vdata)} vertices, {len(idata)} triangles")

# For each pentagon, extract the mesh vertices that belong to it
# and check their UV coordinates against the atlas
for fid in pent_fids:
    tile = tile_by_fid[fid]
    n = len(tile.vertices)
    slot = artifact.uv_layout[fid]
    u_min, v_min, u_max, v_max = slot

    # Find mesh vertices near this tile's center
    tile_center = np.array(tile.center, dtype=float)
    all_pos = vdata[:, :3]
    dists = np.linalg.norm(all_pos - tile_center, axis=1)

    # Get the n closest vertices (these are the corner vertices)
    # Actually, the mesh is subdivided, so there are many vertices.
    # Let's just look at the UVs that are within this slot
    mesh_uvs = vdata[:, 6:8]  # columns 6,7 are u,v (after pos_3 + color_3)
    in_slot = (
        (mesh_uvs[:, 0] >= u_min - 0.001) &
        (mesh_uvs[:, 0] <= u_max + 0.001) &
        (mesh_uvs[:, 1] >= v_min - 0.001) &
        (mesh_uvs[:, 1] <= v_max + 0.001)
    )
    slot_mask = in_slot
    n_verts_in_slot = slot_mask.sum()

    # Compute corner UVs as mesh expects them
    from polygrid.rendering.globe_renderer_v2 import _compute_tile_uvs
    expected_corner_uvs = _compute_tile_uvs(list(tile.uv_vertices), slot)

    print(f"\n  {fid}: {n_verts_in_slot} vertices in atlas slot")
    print(f"    Expected corner UVs (from mesh builder):")
    for i, (eu, ev) in enumerate(expected_corner_uvs):
        print(f"      corner {i}: ({eu:.6f}, {ev:.6f})")

    # Check if these corner UVs appear in the mesh
    for i, (eu, ev) in enumerate(expected_corner_uvs):
        uv_dists = np.sqrt((mesh_uvs[:, 0] - eu) ** 2 + (mesh_uvs[:, 1] - ev) ** 2)
        closest_idx = np.argmin(uv_dists)
        closest_dist = uv_dists[closest_idx]
        closest_pos = all_pos[closest_idx]
        expected_pos = np.array(tile.vertices[i], dtype=float)
        pos_error = np.linalg.norm(closest_pos / np.linalg.norm(closest_pos) - expected_pos / np.linalg.norm(expected_pos))
        print(f"      corner {i}: UV dist={closest_dist:.6f}, 3D pos error={pos_error:.6f}")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
