#!/usr/bin/env python3
"""Visual diagnostic: extract and compare atlas edge strips for a tile pair.

Save side-by-side comparison images of the pixel strips along shared
UV edges.  This makes it easy to see if the content matches or is
flipped/shifted.
"""
import sys
import json
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "src")

from polygrid.globe import build_globe_grid
from polygrid.uv_texture import get_goldberg_tiles, _match_tile_to_face, get_tile_uv_vertices

EXPORTS = Path("exports/f3")
OUT = Path("exports/f3_diag")
OUT.mkdir(exist_ok=True)

atlas = Image.open(EXPORTS / "atlas.png").convert("RGB")
atlas_arr = np.array(atlas)
aw, ah = atlas.size

with open(EXPORTS / "uv_layout.json") as f:
    uv_layout = json.load(f)

with open(EXPORTS / "metadata.json") as f:
    metadata = json.load(f)

tile_size = metadata.get("tile_size", 512)
gutter = metadata.get("gutter", 4)
slot_size = tile_size + 2 * gutter

gg = build_globe_grid(3)
tiles_3d = get_goldberg_tiles(3, 1.0)


def _qv(v, prec=6):
    return tuple(round(c, prec) for c in v)

vert_to_tiles = {}
for t in tiles_3d:
    for vtx in t.vertices:
        vert_to_tiles.setdefault(_qv(vtx), []).append(t.index)


def get_gt_edge_neigh(fid):
    tile = _match_tile_to_face(tiles_3d, fid)
    n = len(tile.vertices)
    result = {}
    for m in range(n):
        va = _qv(tile.vertices[m])
        vb = _qv(tile.vertices[(m + 1) % n])
        shared = (set(vert_to_tiles.get(va, []))
                  & set(vert_to_tiles.get(vb, []))
                  - {tile.index})
        if shared:
            result[m] = f"t{shared.pop()}"
    return result


def extract_edge_strip(fid, edge_m, width=10, n_samples=200):
    """Extract a strip of pixels along UV edge m of tile fid.
    
    Returns a (n_samples, width, 3) array and the edge endpoints in atlas pixels.
    """
    uv_corners = get_tile_uv_vertices(gg, fid)
    n = len(uv_corners)
    u0, v0 = uv_corners[edge_m]
    u1, v1 = uv_corners[(edge_m + 1) % n]
    
    slot = uv_layout[fid]
    su0, sv0, su1, sv1 = slot
    
    # UV → atlas pixel
    # The atlas builder does:
    #   atlas_x = slot_x + gutter + u * tile_size
    #   atlas_y = slot_y + gutter + (1 - v) * tile_size
    # uv_layout stores normalised coords of the inner tile region:
    #   u_min = (slot_x + gutter) / atlas_w
    #   ...
    # So inner tile pixel region is:
    inner_x0 = su0 * aw
    inner_y0 = (1.0 - sv1) * ah  # top of inner region
    inner_w = (su1 - su0) * aw
    inner_h = (sv1 - sv0) * ah
    
    def uv_to_atlas(u, v):
        ax = inner_x0 + u * inner_w
        ay = inner_y0 + (1.0 - v) * inner_h
        return ax, ay
    
    # Edge direction
    ax0, ay0 = uv_to_atlas(u0, v0)
    ax1, ay1 = uv_to_atlas(u1, v1)
    
    # Normal to edge (pointing inward)
    dx = ax1 - ax0
    dy = ay1 - ay0
    edge_len = math.sqrt(dx*dx + dy*dy)
    nx = -dy / edge_len  # perpendicular
    ny = dx / edge_len
    
    strip = np.zeros((n_samples, width, 3), dtype=np.uint8)
    for i in range(n_samples):
        t = (i + 0.5) / n_samples
        cx = ax0 + t * dx
        cy = ay0 + t * dy
        for j in range(width):
            # j=0 is at the edge, j increases inward
            offset = j - width // 2
            px = cx + offset * nx
            py = cy + offset * ny
            ix = int(np.clip(px, 0, aw - 1))
            iy = int(np.clip(py, 0, ah - 1))
            strip[i, j] = atlas_arr[iy, ix]
    
    return strip, (ax0, ay0, ax1, ay1)


# Check specific pairs
test_fids = ["t0", "t1", "t3", "t6"]
for fid_a in test_fids:
    gt_neigh = get_gt_edge_neigh(fid_a)
    for m_a, fid_b in gt_neigh.items():
        # Find B's edge facing A
        gt_neigh_b = get_gt_edge_neigh(fid_b)
        m_b = None
        for m, nfid in gt_neigh_b.items():
            if nfid == fid_a:
                m_b = m
                break
        if m_b is None:
            continue
        
        strip_a, endpts_a = extract_edge_strip(fid_a, m_a, width=20)
        strip_b, endpts_b = extract_edge_strip(fid_b, m_b, width=20)
        
        # B's edge should be traversed in reverse direction
        strip_b_rev = strip_b[::-1]
        
        # Compute diffs
        diff_fwd = np.mean(np.abs(strip_a.astype(float) - strip_b.astype(float)))
        diff_rev = np.mean(np.abs(strip_a.astype(float) - strip_b_rev.astype(float)))
        
        # Create comparison image
        h, w = strip_a.shape[:2]
        gap = 4
        comp = Image.new("RGB", (w * 3 + gap * 2, h), (128, 128, 128))
        comp.paste(Image.fromarray(strip_a), (0, 0))
        comp.paste(Image.fromarray(strip_b_rev), (w + gap, 0))
        
        # Diff image (amplified)
        diff_arr = np.abs(strip_a.astype(float) - strip_b_rev.astype(float))
        diff_img = np.clip(diff_arr * 4, 0, 255).astype(np.uint8)  # 4x amplified
        comp.paste(Image.fromarray(diff_img), (2 * (w + gap), 0))
        
        # Scale up for visibility
        scale = 3
        comp = comp.resize((comp.width * scale, comp.height * scale), Image.NEAREST)
        
        n_a = len(gg.faces[fid_a].vertex_ids)
        n_b = len(gg.faces[fid_b].vertex_ids)
        kind_a = "P" if n_a == 5 else "H"
        kind_b = "P" if n_b == 5 else "H"
        
        fname = f"edge_{fid_a}_e{m_a}_{fid_b}_e{m_b}_{kind_a}{kind_b}.png"
        comp.save(str(OUT / fname))
        
        print(f"{fid_a}(e{m_a}) ↔ {fid_b}(e{m_b}) [{kind_a}-{kind_b}]: "
              f"fwd={diff_fwd:.1f} rev={diff_rev:.1f}  "
              f"→ saved {fname}")

print(f"\nImages saved to {OUT}/")
