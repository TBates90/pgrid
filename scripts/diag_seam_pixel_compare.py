#!/usr/bin/env python3
"""Generate a direct pixel comparison at pentagon-hex boundaries.

For each pent-hex shared edge, extract a narrow strip of pixels from
both tiles' atlas slots and check:
1. Do pixel colors match across the shared boundary?
2. Is the cell grid pattern continuous or broken?

Generates comparison images for visual inspection.
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
OUT_DIR = Path("/tmp/pent_seam_compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Build a recolored atlas with high noise for visibility ──────
from polygrid.placeholder_atlas import _build_artifact, recolor_atlas
from polygrid.integration import PlaceholderAtlasSpec
from polygrid.globe.globe import build_globe_grid
from polygrid.rendering.uv_texture import get_tile_uv_vertices, get_goldberg_tiles

artifact = _build_artifact(FREQ, DETAIL_RINGS, TILE_SIZE, GUTTER)
globe_grid = build_globe_grid(FREQ)
tiles = get_goldberg_tiles(FREQ, 1.0)
tile_by_fid = {f"t{t.index}": t for t in tiles}

# Recolor with high noise so cells are clearly distinguishable
spec = PlaceholderAtlasSpec(
    frequency=FREQ,
    detail_rings=DETAIL_RINGS,
    tile_size=TILE_SIZE,
    gutter=GUTTER,
    base_color=(0.4, 0.6, 0.8),
    noise_amount=0.15,
    seed=42,
)
atlas_png = recolor_atlas(artifact, spec)
atlas_img = Image.open(__import__("io").BytesIO(atlas_png)).convert("RGB")
atlas_arr = np.array(atlas_img, dtype=np.uint8)
atlas_h, atlas_w = atlas_arr.shape[:2]

atlas_img.save(OUT_DIR / "atlas_recolored.png")
print(f"Atlas: {atlas_w}x{atlas_h}")


def uv_to_atlas_px(u, v, slot):
    """Convert tile-local UV to atlas pixel coords."""
    u_min, v_min, u_max, v_max = slot
    # Map local UV [0,1] to atlas UV, then to pixel
    atlas_u = u_min + u * (u_max - u_min)
    atlas_v = v_min + v * (v_max - v_min)
    ax = atlas_u * atlas_w
    ay = (1.0 - atlas_v) * atlas_h
    return ax, ay


def find_shared_gt_edge(fid_a, fid_b):
    """Find which GT edge of fid_a borders fid_b, and which edge of fid_b borders fid_a."""
    tile_a = tile_by_fid[fid_a]
    tile_b = tile_by_fid[fid_b]
    n_a = len(tile_a.vertices)
    n_b = len(tile_b.vertices)

    b_verts = {}
    for vi, v in enumerate(tile_b.vertices):
        b_verts[tuple(round(c, 6) for c in v)] = vi

    for ea in range(n_a):
        va0 = tuple(round(c, 6) for c in tile_a.vertices[ea])
        va1 = tuple(round(c, 6) for c in tile_a.vertices[(ea + 1) % n_a])
        if va0 in b_verts and va1 in b_verts:
            # Find the corresponding edge in tile_b
            bi0 = b_verts[va0]
            bi1 = b_verts[va1]
            for eb in range(n_b):
                if (eb == bi0 and (eb + 1) % n_b == bi1):
                    return ea, eb, False  # same direction
                if (eb == bi1 and (eb + 1) % n_b == bi0):
                    return ea, eb, True   # reversed direction
            # Edge exists but couldn't match direction
            return ea, None, None
    return None, None, None


def sample_strip_along_edge(fid, edge_k, perpendicular_offset_uv, n_samples=100):
    """Sample a strip of atlas pixels along a UV edge with perpendicular offset.
    
    perpendicular_offset_uv: positive = inward (toward center), negative = outward
    Returns list of RGB values.
    """
    slot = artifact.uv_layout[fid]
    uv_verts = get_tile_uv_vertices(globe_grid, fid)
    n = len(uv_verts)
    
    u0, v0 = uv_verts[edge_k]
    u1, v1 = uv_verts[(edge_k + 1) % n]
    
    # Edge direction and perpendicular
    eu = u1 - u0
    ev = v1 - v0
    edge_len = math.sqrt(eu * eu + ev * ev)
    if edge_len < 1e-10:
        return []
    
    # Perpendicular (inward = toward polygon center)
    cx = sum(u for u, _ in uv_verts) / n
    cy = sum(v for _, v in uv_verts) / n
    # Perpendicular direction
    perp_u = -ev / edge_len
    perp_v = eu / edge_len
    # Make sure perpendicular points inward
    mid_u = (u0 + u1) / 2
    mid_v = (v0 + v1) / 2
    to_center_u = cx - mid_u
    to_center_v = cy - mid_v
    if perp_u * to_center_u + perp_v * to_center_v < 0:
        perp_u, perp_v = -perp_u, -perp_v
    
    pixels = []
    for i in range(n_samples):
        t = (i + 0.5) / n_samples
        u = u0 + t * eu + perpendicular_offset_uv * perp_u
        v = v0 + t * ev + perpendicular_offset_uv * perp_v
        ax, ay = uv_to_atlas_px(u, v, slot)
        ix = int(np.clip(ax, 0, atlas_w - 1))
        iy = int(np.clip(ay, 0, atlas_h - 1))
        pixels.append(tuple(atlas_arr[iy, ix]))
    
    return pixels


# ── Compare pent-hex edges ──────────────────────────────────────
pent_fids = [fid for fid in sorted(globe_grid.faces.keys(), key=lambda f: int(f[1:]))
             if len(globe_grid.faces[fid].vertex_ids) == 5]

STRIP_H = 30   # pixels of strip height
STRIP_W = 200  # pixels of strip width
OFFSET_UV = 0.03  # UV offset from edge (inward)

comparison_strips = []

for pent_fid in pent_fids[:3]:  # Just check first 3 pentagons for speed
    tile = tile_by_fid[pent_fid]
    n_sides = len(tile.vertices)

    for gt_edge in range(n_sides):
        # Find neighbor
        neigh_fid = None
        for other in tiles:
            if other.index == tile.index:
                continue
            pent_ea, neigh_eb, reversed_dir = find_shared_gt_edge(pent_fid, f"t{other.index}")
            if pent_ea == gt_edge:
                neigh_fid = f"t{other.index}"
                break
        
        if neigh_fid is None:
            continue
        
        _, neigh_eb, reversed_dir = find_shared_gt_edge(pent_fid, neigh_fid)
        if neigh_eb is None:
            continue
        
        # Sample strips just inside each tile's boundary
        pent_strip = sample_strip_along_edge(pent_fid, gt_edge, OFFSET_UV, STRIP_W)
        
        # For the neighbor, sample in the direction that aligns with pentagon edge
        if reversed_dir:
            neigh_strip = sample_strip_along_edge(neigh_fid, neigh_eb, OFFSET_UV, STRIP_W)
            neigh_strip = list(reversed(neigh_strip))
        else:
            neigh_strip = sample_strip_along_edge(neigh_fid, neigh_eb, OFFSET_UV, STRIP_W)
        
        if not pent_strip or not neigh_strip:
            continue
        
        # Create comparison image: pentagon strip on top, neighbor strip below
        strip_img = Image.new("RGB", (STRIP_W, 4 + STRIP_H * 2), (0, 0, 0))
        draw = ImageDraw.Draw(strip_img)
        
        # Draw pentagon strip
        for x, rgb in enumerate(pent_strip):
            for y in range(STRIP_H):
                strip_img.putpixel((x, y), rgb)
        
        # Separator line
        for x in range(STRIP_W):
            strip_img.putpixel((x, STRIP_H), (255, 255, 0))
            strip_img.putpixel((x, STRIP_H + 1), (255, 255, 0))
            strip_img.putpixel((x, STRIP_H + 2), (255, 255, 0))
            strip_img.putpixel((x, STRIP_H + 3), (255, 255, 0))
        
        # Draw neighbor strip
        for x, rgb in enumerate(neigh_strip):
            for y in range(STRIP_H):
                strip_img.putpixel((x, STRIP_H + 4 + y), rgb)
        
        fname = f"{pent_fid}_e{gt_edge}_{neigh_fid}_e{neigh_eb}.png"
        strip_img.save(OUT_DIR / fname)
        
        # Compute color difference metric
        diffs = []
        for p, n in zip(pent_strip, neigh_strip):
            d = sum(abs(a - b) for a, b in zip(p, n))
            diffs.append(d)
        avg_diff = sum(diffs) / len(diffs) if diffs else 0
        max_diff = max(diffs) if diffs else 0
        
        comparison_strips.append((pent_fid, gt_edge, neigh_fid, neigh_eb, avg_diff, max_diff))
        print(f"  {pent_fid} e{gt_edge} <-> {neigh_fid} e{neigh_eb}: avg_diff={avg_diff:.1f} max_diff={max_diff}")


# ── Also check a hex-hex reference edge for comparison ──────────
print("\nReference hex-hex edges:")
hex_fids = [fid for fid in sorted(globe_grid.faces.keys(), key=lambda f: int(f[1:]))
            if len(globe_grid.faces[fid].vertex_ids) == 6]

for hex_fid in hex_fids[:2]:
    htile = tile_by_fid[hex_fid]
    for gt_edge in range(6):
        for other in tiles:
            if other.index == htile.index:
                continue
            if len(other.vertices) == 6:
                ea, eb, rev = find_shared_gt_edge(hex_fid, f"t{other.index}")
                if ea == gt_edge and eb is not None:
                    h_strip = sample_strip_along_edge(hex_fid, gt_edge, OFFSET_UV, STRIP_W)
                    n_strip = sample_strip_along_edge(f"t{other.index}", eb, OFFSET_UV, STRIP_W)
                    if rev:
                        n_strip = list(reversed(n_strip))
                    if h_strip and n_strip:
                        diffs = [sum(abs(a - b) for a, b in zip(p, n)) for p, n in zip(h_strip, n_strip)]
                        avg_d = sum(diffs) / len(diffs)
                        max_d = max(diffs)
                        print(f"  {hex_fid} e{gt_edge} <-> t{other.index} e{eb}: avg_diff={avg_d:.1f} max_diff={max_d}")
                        
                        strip_img = Image.new("RGB", (STRIP_W, 4 + STRIP_H * 2), (0, 0, 0))
                        for x, rgb in enumerate(h_strip):
                            for y in range(STRIP_H):
                                strip_img.putpixel((x, y), rgb)
                        for x in range(STRIP_W):
                            for dy in range(4):
                                strip_img.putpixel((x, STRIP_H + dy), (255, 255, 0))
                        for x, rgb in enumerate(n_strip):
                            for y in range(STRIP_H):
                                strip_img.putpixel((x, STRIP_H + 4 + y), rgb)
                        strip_img.save(OUT_DIR / f"hexref_{hex_fid}_e{gt_edge}_t{other.index}_e{eb}.png")
                    break
        break  # Just first edge per hex

print(f"\nDone! Check {OUT_DIR}")
