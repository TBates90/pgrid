#!/usr/bin/env python3
"""Trace the new refined warp: for points along a shared UV edge,
what grid-space point does each tile's warp map to, and do they
hit the same sub-face?
"""
import sys
import json
import math
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, "src")

from polygrid.globe import build_globe_grid
from polygrid.uv_texture import get_tile_uv_vertices, get_goldberg_tiles, _match_tile_to_face

EXPORTS = Path("exports/f3")
atlas = Image.open(EXPORTS / "atlas.png").convert("RGB")
atlas_arr = np.array(atlas)
aw, ah = atlas.size

with open(EXPORTS / "uv_layout.json") as f:
    uv_layout = json.load(f)

gg = build_globe_grid(3)
tiles_3d = get_goldberg_tiles(3, 1.0)


def _qv(v, prec=6):
    return tuple(round(c, prec) for c in v)

vert_to_tiles = {}
for t in tiles_3d:
    for vtx in t.vertices:
        vert_to_tiles.setdefault(_qv(vtx), []).append(t.index)


def get_gt_neigh(fid):
    tile = _match_tile_to_face(tiles_3d, fid)
    n = len(tile.vertices)
    result = {}
    for m in range(n):
        va = _qv(tile.vertices[m])
        vb = _qv(tile.vertices[(m + 1) % n])
        shared = (set(vert_to_tiles.get(va, [])) & set(vert_to_tiles.get(vb, []))) - {tile.index}
        if shared:
            result[m] = f"t{shared.pop()}"
    return result


def sample_edge_strip(fid, edge_m, offset, n_samples=50):
    """Sample a strip parallel to UV edge m, offset pixels inward."""
    uv_corners = get_tile_uv_vertices(gg, fid)
    n = len(uv_corners)
    u0, v0 = uv_corners[edge_m]
    u1, v1 = uv_corners[(edge_m + 1) % n]

    slot = uv_layout[fid]
    su0, sv0, su1, sv1 = slot
    inner_x0 = su0 * aw
    inner_y0 = (1.0 - sv1) * ah
    inner_w = (su1 - su0) * aw
    inner_h = (sv1 - sv0) * ah

    def uv_to_atlas(u, v):
        return inner_x0 + u * inner_w, inner_y0 + (1.0 - v) * inner_h

    ax0, ay0 = uv_to_atlas(u0, v0)
    ax1, ay1 = uv_to_atlas(u1, v1)
    cu = sum(c[0] for c in uv_corners) / n
    cv = sum(c[1] for c in uv_corners) / n
    cax, cay = uv_to_atlas(cu, cv)
    emx, emy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    inx = cax - emx
    iny = cay - emy
    l = math.sqrt(inx**2 + iny**2)
    if l > 0:
        inx /= l
        iny /= l

    pixels = []
    for i in range(n_samples):
        t = (i + 0.5) / n_samples
        ax = ax0 + t * (ax1 - ax0) + offset * inx
        ay = ay0 + t * (ay1 - ay0) + offset * iny
        ix = int(np.clip(ax, 0, aw - 1))
        iy = int(np.clip(ay, 0, ah - 1))
        pixels.append(atlas_arr[iy, ix])
    return np.array(pixels, dtype=np.float64)


# Count how many edge pairs have perfect/near-perfect matches
print("Cross-tile diff analysis at different offsets:")
print("=" * 70)

all_pairs = []
for fid_a in sorted(gg.faces.keys(), key=lambda x: int(x[1:])):
    gt_neigh = get_gt_neigh(fid_a)
    for m_a, fid_b in gt_neigh.items():
        if int(fid_a[1:]) >= int(fid_b[1:]):
            continue
        gt_neigh_b = get_gt_neigh(fid_b)
        m_b = None
        for m, nfid in gt_neigh_b.items():
            if nfid == fid_a:
                m_b = m
                break
        if m_b is None:
            continue
        all_pairs.append((fid_a, m_a, fid_b, m_b))

for offset in [0, 1, 2]:
    diffs = []
    for fid_a, m_a, fid_b, m_b in all_pairs:
        pix_a = sample_edge_strip(fid_a, m_a, offset=offset)
        pix_b = sample_edge_strip(fid_b, m_b, offset=offset)
        # Reverse b because it's sampled from the opposite direction
        diff = np.mean(np.abs(pix_a - pix_b[::-1]))
        diffs.append(diff)

    arr = np.array(diffs)
    perfect = np.sum(arr < 1.0)
    good = np.sum(arr < 5.0)
    ok = np.sum(arr < 10.0)
    print(f"  offset={offset}: mean={arr.mean():.1f}  median={np.median(arr):.1f}  "
          f"max={arr.max():.1f}  <1:{perfect}/{len(arr)}  <5:{good}/{len(arr)}  <10:{ok}/{len(arr)}")

# Break down by pair type
print("\nPer-type breakdown (offset=0):")
categories = {"H-H": [], "H-P": [], "P-H": [], "P-P": []}
for fid_a, m_a, fid_b, m_b in all_pairs:
    pix_a = sample_edge_strip(fid_a, m_a, offset=0)
    pix_b = sample_edge_strip(fid_b, m_b, offset=0)
    diff = np.mean(np.abs(pix_a - pix_b[::-1]))

    na = len(gg.faces[fid_a].vertex_ids)
    nb = len(gg.faces[fid_b].vertex_ids)
    ta = "P" if na == 5 else "H"
    tb = "P" if nb == 5 else "H"
    key = f"{ta}-{tb}"
    categories[key].append(diff)

for key, vals in sorted(categories.items()):
    if vals:
        arr = np.array(vals)
        print(f"  {key}: n={len(vals)}  mean={arr.mean():.1f}  "
              f"median={np.median(arr):.1f}  max={arr.max():.1f}")

# Show worst 10 pairs
print("\nWorst 10 pairs (offset=0):")
pair_diffs = []
for fid_a, m_a, fid_b, m_b in all_pairs:
    pix_a = sample_edge_strip(fid_a, m_a, offset=0)
    pix_b = sample_edge_strip(fid_b, m_b, offset=0)
    diff = np.mean(np.abs(pix_a - pix_b[::-1]))
    pair_diffs.append((diff, fid_a, m_a, fid_b, m_b))

pair_diffs.sort(reverse=True)
for diff, fid_a, m_a, fid_b, m_b in pair_diffs[:10]:
    na = len(gg.faces[fid_a].vertex_ids)
    nb = len(gg.faces[fid_b].vertex_ids)
    ta = "P" if na == 5 else "H"
    tb = "P" if nb == 5 else "H"
    print(f"  {fid_a} e{m_a} ({ta}) – {fid_b} e{m_b} ({tb}): diff={diff:.1f}")
