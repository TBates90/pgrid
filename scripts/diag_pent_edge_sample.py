#!/usr/bin/env python3
"""Sample cell indices along shared pentagon-hex edges in the placeholder atlas.

For each pentagon, walks along each GT edge in the atlas UV space and
collects the cell index keys. Then does the same for the NEIGHBOR's
corresponding edge. If the pentagon orientation is correct, the boundary
cells should show consistent content (same physical region on the globe).

Also writes edge-comparison images for visual inspection.
"""
import sys
import math
import os
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
from PIL import Image, ImageDraw

FREQ = 2
DETAIL_RINGS = 4
TILE_SIZE = 256
GUTTER = 4
OUT_DIR = Path("/tmp/pent_edge_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Build artifact ──────────────────────────────────────────────
from polygrid.placeholder_atlas import _build_artifact
from polygrid.globe.globe import build_globe_grid
from polygrid.rendering.uv_texture import get_tile_uv_vertices, get_goldberg_tiles
from polygrid.rendering.atlas_utils import compute_atlas_layout

artifact = _build_artifact(FREQ, DETAIL_RINGS, TILE_SIZE, GUTTER)
globe_grid = build_globe_grid(FREQ)
tiles = get_goldberg_tiles(FREQ, 1.0)
tile_by_fid = {f"t{t.index}": t for t in tiles}
face_ids = sorted(globe_grid.faces.keys(), key=lambda f: int(f[1:]))

atlas_h, atlas_w = artifact.tile_index_map.shape
idx_map = artifact.tile_index_map

print(f"Atlas size: {atlas_w}x{atlas_h}")
print(f"Index keys: {len(artifact.index_keys)}")


def uv_to_atlas_px(u, v, slot):
    """Convert tile-local UV (0-1) to atlas pixel coords."""
    u_min, v_min, u_max, v_max = slot
    ax = (u_min + u * (u_max - u_min)) * atlas_w
    ay = (1.0 - (v_min + v * (v_max - v_min))) * atlas_h
    return ax, ay


def sample_edge_indices(fid, edge_k, n_samples=50):
    """Sample cell indices along GT edge k of tile fid in atlas space."""
    slot = artifact.uv_layout[fid]
    uv_verts = get_tile_uv_vertices(globe_grid, fid)
    n_sides = len(uv_verts)

    v0 = uv_verts[edge_k]
    v1 = uv_verts[(edge_k + 1) % n_sides]

    indices = []
    for t in np.linspace(0.1, 0.9, n_samples):
        u = v0[0] + t * (v1[0] - v0[0])
        v = v0[1] + t * (v1[1] - v0[1])
        ax, ay = uv_to_atlas_px(u, v, slot)
        ix = int(np.clip(ax, 0, atlas_w - 1))
        iy = int(np.clip(ay, 0, atlas_h - 1))
        cell_idx = int(idx_map[iy, ix])
        key = artifact.index_keys[cell_idx] if cell_idx < len(artifact.index_keys) else "BG"
        indices.append((t, key, ix, iy))
    return indices


def find_shared_gt_edge(fid_a, fid_b):
    """Find which GT edge of fid_a borders fid_b."""
    tile_a = tile_by_fid[fid_a]
    tile_b = tile_by_fid[fid_b]
    n_a = len(tile_a.vertices)

    # Build a vertex lookup for tile_b
    b_verts = set()
    for v in tile_b.vertices:
        b_verts.add(tuple(round(c, 6) for c in v))

    for eidx in range(n_a):
        va = tuple(round(c, 6) for c in tile_a.vertices[eidx])
        vb = tuple(round(c, 6) for c in tile_a.vertices[(eidx + 1) % n_a])
        if va in b_verts and vb in b_verts:
            return eidx
    return None


# ── For each pentagon, check shared edges ───────────────────────
pent_fids = [fid for fid in face_ids if len(globe_grid.faces[fid].vertex_ids) == 5]

# We'll focus on t0 for detailed analysis
for fid in ["t0"]:
    tile = tile_by_fid[fid]
    n_sides = len(tile.vertices)

    print(f"\n{'='*70}")
    print(f"Pentagon: {fid}")
    print(f"{'='*70}")

    for gt_edge in range(n_sides):
        # Find neighbor for this GT edge
        va = tile.vertices[gt_edge]
        vb = tile.vertices[(gt_edge + 1) % n_sides]
        va_key = tuple(round(c, 6) for c in va)
        vb_key = tuple(round(c, 6) for c in vb)

        # Find which neighbor shares this edge
        neigh_fid = None
        for other_tile in tiles:
            if other_tile.index == tile.index:
                continue
            ov = set(tuple(round(c, 6) for c in v) for v in other_tile.vertices)
            if va_key in ov and vb_key in ov:
                neigh_fid = f"t{other_tile.index}"
                break

        if neigh_fid is None:
            print(f"\n  GT edge {gt_edge}: NO NEIGHBOR FOUND")
            continue

        # Find which GT edge of the neighbor corresponds
        neigh_gt_edge = find_shared_gt_edge(neigh_fid, fid)
        if neigh_gt_edge is None:
            print(f"\n  GT edge {gt_edge}: shared with {neigh_fid} but can't find reverse edge")
            continue

        print(f"\n  GT edge {gt_edge} -> neighbor {neigh_fid} (neighbor GT edge {neigh_gt_edge})")

        # Sample cell indices along the pentagon's edge
        pent_samples = sample_edge_indices(fid, gt_edge)
        # Sample along neighbor's edge (reversed direction for alignment)
        neigh_samples = sample_edge_indices(neigh_fid, neigh_gt_edge)

        # The shared edge should show the same macro-tile boundary cells
        print(f"    Pentagon {fid} edge {gt_edge} cells (first/mid/last):")
        for i in [0, len(pent_samples)//2, -1]:
            t, key, ix, iy = pent_samples[i]
            # Parse the key to get the source tile
            parts = key.split(":")
            src_tile = parts[0] if len(parts) > 1 else "?"
            cell_id = parts[1] if len(parts) > 1 else key
            print(f"      t={t:.2f}: src={src_tile} cell={cell_id} (atlas px={ix},{iy})")

        print(f"    Neighbor {neigh_fid} edge {neigh_gt_edge} cells (first/mid/last):")
        for i in [0, len(neigh_samples)//2, -1]:
            t, key, ix, iy = neigh_samples[i]
            parts = key.split(":")
            src_tile = parts[0] if len(parts) > 1 else "?"
            cell_id = parts[1] if len(parts) > 1 else key
            print(f"      t={t:.2f}: src={src_tile} cell={cell_id} (atlas px={ix},{iy})")

        # Check: cells along pentagon edge should be from fid's grid (interior)
        # AND/OR from the neighbor (gutter). Similarly for neighbor's edge.
        # Check for cross-contamination: pentagon edge should NOT show
        # cells from a WRONG neighbor.
        pent_sources = set()
        for _, key, _, _ in pent_samples:
            src = key.split(":")[0] if ":" in key else key
            pent_sources.add(src)

        neigh_sources = set()
        for _, key, _, _ in neigh_samples:
            src = key.split(":")[0] if ":" in key else key
            neigh_sources.add(src)

        print(f"    Pentagon edge sources: {pent_sources}")
        print(f"    Neighbor edge sources: {neigh_sources}")

        # Validate: pentagon edge should show cells from fid or neigh_fid
        unexpected_pent = pent_sources - {fid, neigh_fid, "BG"}
        unexpected_neigh = neigh_sources - {fid, neigh_fid, "BG"}

        if unexpected_pent:
            print(f"    *** WRONG SOURCE IN PENTAGON EDGE: {unexpected_pent} ***")
            print(f"    Expected only {fid} or {neigh_fid}")
        else:
            print(f"    Pentagon edge: OK (only {fid}/{neigh_fid} cells)")

        if unexpected_neigh:
            print(f"    *** WRONG SOURCE IN NEIGHBOR EDGE: {unexpected_neigh} ***")
            print(f"    Expected only {fid} or {neigh_fid}")
        else:
            print(f"    Neighbor edge: OK (only {fid}/{neigh_fid} cells)")


# ── Also: render a comparison strip for the shared edge ─────────
# For t0 and each neighbor, extract a strip of pixels around the shared
# edge from both tiles' atlas slots and show them side-by-side
print("\n\nGenerating edge comparison strips...")

for fid in ["t0"]:
    tile = tile_by_fid[fid]
    n_sides = len(tile.vertices)

    for gt_edge in range(n_sides):
        va = tile.vertices[gt_edge]
        vb = tile.vertices[(gt_edge + 1) % n_sides]
        va_key = tuple(round(c, 6) for c in va)
        vb_key = tuple(round(c, 6) for c in vb)

        neigh_fid = None
        for other_tile in tiles:
            if other_tile.index == tile.index:
                continue
            ov = set(tuple(round(c, 6) for c in v) for v in other_tile.vertices)
            if va_key in ov and vb_key in ov:
                neigh_fid = f"t{other_tile.index}"
                break

        if neigh_fid is None:
            continue

        # Sample a strip of pixels perpendicular to the edge
        slot_p = artifact.uv_layout[fid]
        slot_n = artifact.uv_layout[neigh_fid]

        uv_p = get_tile_uv_vertices(globe_grid, fid)
        uv_n = get_tile_uv_vertices(globe_grid, neigh_fid)

        # Edge endpoints in atlas pixel space (for the pentagon)
        u0, v0 = uv_p[gt_edge]
        u1, v1 = uv_p[(gt_edge + 1) % n_sides]

        STRIP_W = 200
        STRIP_H = 20
        strip = np.zeros((STRIP_H * 2, STRIP_W, 3), dtype=np.uint8)

        for sx in range(STRIP_W):
            t = sx / (STRIP_W - 1)
            eu = u0 + t * (u1 - u0)
            ev = v0 + t * (v1 - v0)

            for sy in range(STRIP_H):
                offset = (sy - STRIP_H // 2) / TILE_SIZE
                # Sample pentagon atlas at edge + perpendicular offset
                ax, ay = uv_to_atlas_px(eu, ev + offset * 0.5, slot_p)
                ix = int(np.clip(ax, 0, atlas_w - 1))
                iy = int(np.clip(ay, 0, atlas_h - 1))
                cidx = int(idx_map[iy, ix])
                r = (cidx * 73) % 256
                g = (cidx * 151) % 256
                b = (cidx * 199) % 256
                strip[sy, sx] = [r, g, b]

        # Also sample the neighbor's edge
        neigh_gt_edge = find_shared_gt_edge(neigh_fid, fid)
        if neigh_gt_edge is not None:
            n_s = len(uv_n)
            nu0, nv0 = uv_n[neigh_gt_edge]
            nu1, nv1 = uv_n[(neigh_gt_edge + 1) % n_s]

            for sx in range(STRIP_W):
                # Reversed direction to align with pentagon edge
                t = 1.0 - sx / (STRIP_W - 1)
                eu = nu0 + t * (nu1 - nu0)
                ev = nv0 + t * (nv1 - nv0)

                for sy in range(STRIP_H):
                    offset = (sy - STRIP_H // 2) / TILE_SIZE
                    ax, ay = uv_to_atlas_px(eu, ev + offset * 0.5, slot_n)
                    ix = int(np.clip(ax, 0, atlas_w - 1))
                    iy = int(np.clip(ay, 0, atlas_h - 1))
                    cidx = int(idx_map[iy, ix])
                    r = (cidx * 73) % 256
                    g = (cidx * 151) % 256
                    b = (cidx * 199) % 256
                    strip[STRIP_H + sy, sx] = [r, g, b]

        Image.fromarray(strip, "RGB").save(OUT_DIR / f"{fid}_e{gt_edge}_{neigh_fid}.png")
        print(f"  Saved {fid}_e{gt_edge}_{neigh_fid}.png")

print("\nDone!")
