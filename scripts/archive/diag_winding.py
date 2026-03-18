#!/usr/bin/env python3
"""Diagnostic: check winding of tile.vertices (3D) and tile.uv_vertices (2D)
for all Goldberg tiles, comparing pentagons vs hexagons.

Determines:
  - 3D winding: sign of dot(normal, cross(v1-v0, v2-v0))
    positive => CCW from outside, negative => CW from outside
  - 2D UV winding: signed area of uv_vertices polygon
    positive => CCW, negative => CW  (standard math convention)
"""

import sys
sys.path.insert(0, "src")

import numpy as np
from models.objects.goldberg.generator import generate_goldberg_tiles


def signed_area_2d(pts):
    """Shoelace signed area. Positive => CCW (math convention)."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        area += (x0 * y1 - x1 * y0)
    return area / 2.0


def winding_3d(vertices, center):
    """Check if polygon vertices are CCW when viewed from outside (along -normal).
    Returns signed value: positive => CCW from outside."""
    c = np.array(center, dtype=float)
    normal_out = c / np.linalg.norm(c)  # radial outward

    v0 = np.array(vertices[0], dtype=float)
    v1 = np.array(vertices[1], dtype=float)
    v2 = np.array(vertices[2], dtype=float)

    cross = np.cross(v1 - v0, v2 - v0)
    return float(np.dot(cross, normal_out))


tiles = generate_goldberg_tiles(frequency=3, radius=1.0)

print(f"{'tile':>5} {'kind':>8} {'#v':>3} {'3D_wind':>10} {'3D_dir':>8} {'uv_area':>10} {'uv_dir':>8} {'MATCH':>6}")
print("-" * 72)

pent_3d = []
pent_uv = []
hex_3d = []
hex_uv = []

for tile in tiles:
    w3d = winding_3d(tile.vertices, tile.center)
    sa = signed_area_2d(tile.uv_vertices)
    dir_3d = "CCW" if w3d > 0 else "CW"
    dir_uv = "CCW" if sa > 0 else "CW"
    match = "YES" if (w3d > 0) == (sa > 0) else "NO"

    if tile.kind == "pentagon":
        pent_3d.append(w3d)
        pent_uv.append(sa)
    else:
        hex_3d.append(w3d)
        hex_uv.append(sa)

    if tile.kind == "pentagon" or tile.index < 5:
        print(f"t{tile.index:>4} {tile.kind:>8} {len(tile.vertices):>3} {w3d:>10.6f} {dir_3d:>8} {sa:>10.6f} {dir_uv:>8} {match:>6}")

# Summary
print("\n=== SUMMARY ===")
print(f"Pentagons (n={len(pent_3d)}):")
print(f"  3D winding: all {'CCW' if all(w > 0 for w in pent_3d) else 'MIXED/CW'}")
print(f"  UV winding: all {'CCW' if all(s > 0 for s in pent_uv) else ('CW' if all(s < 0 for s in pent_uv) else 'MIXED')}")
print(f"  3D range: [{min(pent_3d):.6f}, {max(pent_3d):.6f}]")
print(f"  UV range: [{min(pent_uv):.6f}, {max(pent_uv):.6f}]")

print(f"Hexagons (n={len(hex_3d)}):")
print(f"  3D winding: all {'CCW' if all(w > 0 for w in hex_3d) else 'MIXED/CW'}")
print(f"  UV winding: all {'CCW' if all(s > 0 for s in hex_uv) else ('CW' if all(s < 0 for s in hex_uv) else 'MIXED')}")
print(f"  3D range: [{min(hex_3d):.6f}, {max(hex_3d):.6f}]")
print(f"  UV range: [{min(hex_uv):.6f}, {max(hex_uv):.6f}]")

# Check if 3D and UV winding match for all tiles
all_match = all(
    (winding_3d(t.vertices, t.center) > 0) == (signed_area_2d(t.uv_vertices) > 0)
    for t in tiles
)
print(f"\n3D↔UV winding consistent for ALL tiles: {all_match}")

# Now check what the mesh builder triangle fan produces
# The fan is: center → v[i] → v[i+1]
# For OpenGL, front face is CCW when viewed from outside
# Check the first triangle of each tile type
print("\n=== TRIANGLE FAN WINDING (first triangle of each tile) ===")
for tile in tiles:
    if tile.index > 0 and tile.kind == "hexagon":
        continue  # just show first hex
    c = np.array(tile.center, dtype=float)
    v0 = np.array(tile.vertices[0], dtype=float)
    v1 = np.array(tile.vertices[1], dtype=float)

    normal_out = c / np.linalg.norm(c)

    # Triangle: center, v0, v1
    cross = np.cross(v0 - c, v1 - c)
    dot = float(np.dot(cross, normal_out))
    dir_fan = "CCW" if dot > 0 else "CW"

    # UV triangle: center_uv, uv[0], uv[1]
    uv_center = (
        sum(u for u, v in tile.uv_vertices) / len(tile.uv_vertices),
        sum(v for u, v in tile.uv_vertices) / len(tile.uv_vertices),
    )
    ux0, uy0 = tile.uv_vertices[0]
    ux1, uy1 = tile.uv_vertices[1]
    cx, cy = uv_center
    uv_cross = (ux0 - cx) * (uy1 - cy) - (ux1 - cx) * (uy0 - cy)
    dir_uv_fan = "CCW" if uv_cross > 0 else "CW"
    fan_match = "YES" if (dot > 0) == (uv_cross > 0) else "NO"

    print(f"  t{tile.index} ({tile.kind}): 3D fan={dir_fan} ({dot:.6f})  UV fan={dir_uv_fan} ({uv_cross:.6f})  match={fan_match}")
