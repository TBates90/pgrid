#!/usr/bin/env python3
"""Check: does the same sub-face get different centroids in different composites?

Pick an adjacent pair (t0, t1). Look at faces along their shared macro-edge.
In composite_t0, t1's faces near the edge are positioned relative to t0.
In composite_t1, those same faces are in the centre (original position).
Do the centroids differ?
"""
import sys
import math
sys.path.insert(0, "src")

from polygrid.globe import build_globe_grid
from polygrid.tile_detail import (
    build_tile_with_neighbours,
    DetailGridCollection,
    TileDetailSpec,
)
from polygrid.detail_terrain import generate_all_detail_terrain
from polygrid.mountains import MountainConfig, generate_mountains
from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
from polygrid.geometry import face_center

# Build globe + detail grids (same as render_polygrids)
grid = build_globe_grid(3)
schema = TileSchema([FieldDef("elevation", float, 0.0)])
store = TileDataStore(grid=grid, schema=schema)
config = MountainConfig(seed=42, ridge_frequency=2.0, ridge_octaves=4,
                        peak_elevation=1.0, base_elevation=0.0)
generate_mountains(grid, store, config)
spec = TileDetailSpec(detail_rings=4)
coll = DetailGridCollection.build(grid, spec)
generate_all_detail_terrain(coll, grid, store, spec, seed=42)

face_ids = coll.face_ids

# Build composites for t0 and t1
comp_t0 = build_tile_with_neighbours(coll, "t0", grid)
comp_t1 = build_tile_with_neighbours(coll, "t1", grid)

# t1 is a neighbour of t0; find t1's faces in both composites
# In comp_t0.merged, t1's faces are prefixed "t1_"
# In comp_t1.merged, t1's faces are the centre: also prefixed "t1_"

prefix_t1_in_t0 = comp_t0.id_prefixes.get("t1", "t1_")
prefix_t1_in_t1 = comp_t1.id_prefixes.get("t1", "t1_")

print(f"Prefix for t1 in comp_t0: '{prefix_t1_in_t0}'")
print(f"Prefix for t1 in comp_t1: '{prefix_t1_in_t1}'")

# Find faces in t1 that appear in both composites
mg_t0 = comp_t0.merged
mg_t1 = comp_t1.merged

t1_faces_in_t0 = {fid for fid in mg_t0.faces if fid.startswith(prefix_t1_in_t0)}
t1_faces_in_t1 = {fid for fid in mg_t1.faces if fid.startswith(prefix_t1_in_t1)}

# Map to unprefixed IDs
def unprefix(fid, prefix):
    return fid[len(prefix):]

unprefixed_in_t0 = {unprefix(fid, prefix_t1_in_t0) for fid in t1_faces_in_t0}
unprefixed_in_t1 = {unprefix(fid, prefix_t1_in_t1) for fid in t1_faces_in_t1}

common = sorted(unprefixed_in_t0 & unprefixed_in_t1)
print(f"\nFaces of t1 in comp_t0: {len(t1_faces_in_t0)}")
print(f"Faces of t1 in comp_t1: {len(t1_faces_in_t1)}")
print(f"Common (same sub-face in both): {len(common)}")

# Compare centroids
diffs = []
for ufid in common[:20]:
    fid_in_t0 = prefix_t1_in_t0 + ufid
    fid_in_t1 = prefix_t1_in_t1 + ufid
    
    c0 = face_center(mg_t0.vertices, mg_t0.faces[fid_in_t0])
    c1 = face_center(mg_t1.vertices, mg_t1.faces[fid_in_t1])
    
    if c0 and c1:
        dx = c0[0] - c1[0]
        dy = c0[1] - c1[1]
        dist = math.sqrt(dx*dx + dy*dy)
        diffs.append(dist)
        if dist > 0.001:
            print(f"  {ufid}: comp_t0=({c0[0]:.4f},{c0[1]:.4f}) "
                  f"comp_t1=({c1[0]:.4f},{c1[1]:.4f}) dist={dist:.4f}")

if diffs:
    import numpy as np
    arr = np.array(diffs)
    print(f"\nCentroid distance stats for {len(diffs)} common faces:")
    print(f"  mean={arr.mean():.6f}  max={arr.max():.6f}  min={arr.min():.6f}")
    zero_count = sum(1 for d in diffs if d < 1e-8)
    print(f"  exact matches (dist<1e-8): {zero_count}/{len(diffs)}")
