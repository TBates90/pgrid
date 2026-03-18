#!/usr/bin/env python3
"""Definitive edge-match test for pentagon tiles.

For each pentagon edge, find the neighbouring face that shares that edge
(via 3-D vertex matching), sample pixels along both sides in the atlas,
and compare forward vs reversed orientation.
"""
import json, sys, numpy as np
from PIL import Image

sys.path.insert(0, "src")
from polygrid.globe import build_globe_grid
from polygrid.uv_texture import get_tile_uv_vertices

# ── Build globe & load atlas ───────────────────────────────────────
gg = build_globe_grid(3)
atlas = np.array(Image.open("exports/f3/atlas.png").convert("RGB"))
with open("exports/f3/uv_layout.json") as f:
    uv_layout = json.load(f)

atlas_h, atlas_w = atlas.shape[:2]

# ── Build 3-D vertex table (face_id → list of (x,y,z)) ───────────
vert3d = {}
for fid, face in gg.faces.items():
    coords = []
    for vid in face.vertex_ids:
        v = gg.vertices[vid]
        coords.append(np.array([v.x, v.y, v.z]))
    vert3d[fid] = coords


def find_shared_edge(fid_a, fid_b, tol=1e-4):
    """Return (ai, bi, aj, bj) vertex indices of the shared edge.

    Edge goes from vertex ai→bi in face A and from vertex aj→bj in face B,
    where A[ai]≈B[aj] and A[bi]≈B[bj] (same 3-D point).
    Returns None if no shared edge found.
    """
    va = vert3d[fid_a]
    vb = vert3d[fid_b]
    na = len(va)
    nb = len(vb)
    matches = []
    for i in range(na):
        for j in range(nb):
            if np.linalg.norm(va[i] - vb[j]) < tol:
                matches.append((i, j))
    if len(matches) < 2:
        return None
    # We expect exactly 2 shared vertices
    (ai, aj), (bi, bj) = matches[0], matches[1]
    # Ensure ai→bi is an actual edge of face A (consecutive)
    if (bi - ai) % na != 1 and (ai - bi) % na != 1:
        return None
    # Make sure ai→bi follows the face winding (ai, ai+1)
    if (bi - ai) % na != 1:
        ai, bi = bi, ai
        aj, bj = bj, aj
    return ai, bi, aj, bj


def uv_to_atlas_px(fid, u, v):
    box = uv_layout[fid]
    u0, v0, u1, v1 = box
    ax = (u0 + u * (u1 - u0)) * atlas_w
    ay = (1.0 - (v0 + v * (v1 - v0))) * atlas_h
    return int(np.clip(ax, 0, atlas_w - 1)), int(np.clip(ay, 0, atlas_h - 1))


def sample_edge(fid, vi_start, vi_end, n_samples=100, inset=0.1):
    """Sample pixels along the edge from vertex vi_start to vi_end."""
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


# ── Test all pentagon edges ───────────────────────────────────────
pentagons = sorted(fid for fid, face in gg.faces.items()
                   if len(face.vertex_ids) == 5)
hexagons = sorted(fid for fid, face in gg.faces.items()
                  if len(face.vertex_ids) == 6)

match_p, rev_p, total_p = 0, 0, 0

print("=== Pentagon edges ===")
for pid in pentagons:
    face = gg.faces[pid]
    n = len(face.vertex_ids)
    nids = face.neighbor_ids

    for ni, nid in enumerate(nids):
        result = find_shared_edge(pid, nid)
        if result is None:
            continue
        ai, bi, aj, bj = result

        # Sample pentagon edge ai→bi
        pix_a = sample_edge(pid, ai, bi)

        # Sample neighbor edge: shared vertices are A[ai]≈B[aj], A[bi]≈B[bj]
        # On the globe, the shared edge runs in opposite directions on the two faces.
        # So the matching direction on B is bj→aj (reversed).
        pix_b = sample_edge(nid, bj, aj)

        diff = np.mean(np.abs(pix_a - pix_b))
        diff_rev = np.mean(np.abs(pix_a - pix_b[::-1]))

        status = "MATCH" if diff < diff_rev else "REVERSED"
        total_p += 1
        if diff < diff_rev:
            match_p += 1
        else:
            rev_p += 1

        margin = abs(diff - diff_rev)
        flag = " ← WEAK" if margin < 2.0 else ""
        print(f"  {pid}[{ai}→{bi}] vs {nid}[{bj}→{aj}]: "
              f"fwd={diff:.1f} rev={diff_rev:.1f} → {status}{flag}")

print(f"\nPentagon edges: {match_p}/{total_p} MATCH, {rev_p} REVERSED")

# ── Sanity check: a few hexagon edges ────────────────────────────
print("\n=== Hex-hex edges (sample) ===")
match_h, rev_h, total_h = 0, 0, 0
for hid in hexagons[:5]:
    face = gg.faces[hid]
    nids = face.neighbor_ids
    for nid in nids[:2]:  # just first 2 neighbors
        result = find_shared_edge(hid, nid)
        if result is None:
            continue
        ai, bi, aj, bj = result
        pix_a = sample_edge(hid, ai, bi)
        pix_b = sample_edge(nid, bj, aj)
        diff = np.mean(np.abs(pix_a - pix_b))
        diff_rev = np.mean(np.abs(pix_a - pix_b[::-1]))
        status = "MATCH" if diff < diff_rev else "REVERSED"
        total_h += 1
        if diff < diff_rev:
            match_h += 1
        else:
            rev_h += 1
        print(f"  {hid}[{ai}→{bi}] vs {nid}[{bj}→{aj}]: "
              f"fwd={diff:.1f} rev={diff_rev:.1f} → {status}")

print(f"\nHex edges (sample): {match_h}/{total_h} MATCH, {rev_h} REVERSED")
