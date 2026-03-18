#!/usr/bin/env python3
"""Check: how much does the macro-edge deviate from the straight line
between its endpoints (the grid corners)?

The piecewise warp treats the edge between two UV corners as a straight
line, mapping to a straight line between the corresponding grid corners.
But the actual macro-edge is a polyline. If it deviates significantly,
the warp will sample from the wrong side of the boundary.
"""
import sys
import math
import numpy as np
sys.path.insert(0, "src")

from polygrid.globe import build_globe_grid
from polygrid.tile_detail import (
    DetailGridCollection,
    TileDetailSpec,
)
from polygrid.detail_terrain import generate_all_detail_terrain
from polygrid.mountains import MountainConfig, generate_mountains
from polygrid.tile_data import FieldDef, TileDataStore, TileSchema

grid = build_globe_grid(3)
schema = TileSchema([FieldDef("elevation", float, 0.0)])
store = TileDataStore(grid=grid, schema=schema)
config = MountainConfig(seed=42, ridge_frequency=2.0, ridge_octaves=4,
                        peak_elevation=1.0, base_elevation=0.0)
generate_mountains(grid, store, config)
coll = DetailGridCollection.build(grid, TileDetailSpec(detail_rings=4))

# For each tile, check macro-edge straightness
all_deviations = []
all_relative_deviations = []

for fid in sorted(coll.face_ids, key=lambda x: int(x[1:])):
    n_sides = len(grid.faces[fid].vertex_ids)
    dg, _ = coll.get(fid)
    is_pent = n_sides == 5
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)
    
    for me in dg.macro_edges:
        vids = list(me.vertex_ids)
        pts = [(dg.vertices[vid].x, dg.vertices[vid].y) for vid in vids]
        
        # Straight line from first to last
        ax, ay = pts[0]
        bx, by = pts[-1]
        edge_len = math.hypot(bx - ax, by - ay)
        
        if edge_len < 1e-10:
            continue
        
        # Direction unit vector
        dx, dy = (bx - ax) / edge_len, (by - ay) / edge_len
        # Normal
        nx, ny = -dy, dx
        
        # Compute perpendicular distance of each intermediate vertex
        max_dev = 0
        for px, py in pts[1:-1]:
            # Distance from line
            dev = abs((px - ax) * nx + (py - ay) * ny)
            max_dev = max(max_dev, dev)
        
        all_deviations.append(max_dev)
        all_relative_deviations.append(max_dev / edge_len if edge_len > 0 else 0)

devs = np.array(all_deviations)
rel_devs = np.array(all_relative_deviations)

print(f"Macro-edge straightness analysis ({len(devs)} edges total):")
print(f"\n  Absolute max deviation from straight line:")
print(f"    mean={devs.mean():.6f}")
print(f"    max={devs.max():.6f}")
print(f"    median={np.median(devs):.6f}")
print(f"\n  Relative deviation (max_dev / edge_length):")
print(f"    mean={rel_devs.mean():.6f}")
print(f"    max={rel_devs.max():.6f}")
print(f"    median={np.median(rel_devs):.6f}")
print(f"    % with relative dev > 1%: {np.sum(rel_devs > 0.01) / len(rel_devs) * 100:.1f}%")
print(f"    % with relative dev > 5%: {np.sum(rel_devs > 0.05) / len(rel_devs) * 100:.1f}%")

# What fraction of edge_length does the pixel spacing correspond to?
# For a tile_size=512 image with typical xlim span ~0.53
typical_span = 0.53
pixel_spacing = typical_span / 512
cell_size_approx = typical_span / 8  # roughly 8 cells across for rings=4
print(f"\n  For reference:")
print(f"    Typical pixel spacing: {pixel_spacing:.6f}")
print(f"    Typical cell size: {cell_size_approx:.4f}")
print(f"    Max deviation / cell size: {devs.max() / cell_size_approx:.2f}")
