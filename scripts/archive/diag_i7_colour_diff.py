#!/usr/bin/env python3
"""Check: how much do adjacent sub-face colours differ along a shared macro-edge?

If two sub-faces adjacent along the macro-edge have very different colours,
then bilinear interpolation at the cell boundary will produce different
results depending on where the pixel grid falls — causing the seam.
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
from polygrid.geometry import face_center as _face_center
from polygrid.detail_render import detail_elevation_to_colour, BiomeConfig
from polygrid.detail_render import _detail_hillshade
import numpy as np

grid = build_globe_grid(3)
schema = TileSchema([FieldDef("elevation", float, 0.0)])
store = TileDataStore(grid=grid, schema=schema)
config = MountainConfig(seed=42, ridge_frequency=2.0, ridge_octaves=4,
                        peak_elevation=1.0, base_elevation=0.0)
generate_mountains(grid, store, config)
coll = DetailGridCollection.build(grid, TileDetailSpec(detail_rings=4))
generate_all_detail_terrain(coll, grid, store, TileDetailSpec(detail_rings=4), seed=42)

biome = BiomeConfig()

# For t1, compute colours of all sub-faces
dg1, ds1 = coll.get("t1")
hs1 = _detail_hillshade(dg1, ds1, "elevation",
                         azimuth=biome.azimuth, altitude=biome.altitude)

colours = {}
for fid, face in dg1.faces.items():
    elev = ds1.get(fid, "elevation")
    c = _face_center(dg1.vertices, face)
    cx, cy = c if c else (0.0, 0.0)
    hs_val = hs1.get(fid, 0.5)
    rgb = detail_elevation_to_colour(
        elev, biome,
        hillshade_val=hs_val,
        noise_x=cx, noise_y=cy,
        noise_seed=42,
    )
    colours[fid] = np.array(rgb)

# Get adjacency
from polygrid.polygrid import get_face_adjacency
adj = get_face_adjacency(dg1)

# Compute colour differences between adjacent faces
diffs = []
for fid, neighs in adj.items():
    if fid not in colours:
        continue
    for nid in neighs:
        if nid not in colours:
            continue
        diff = np.abs(colours[fid] - colours[nid])
        diffs.append(np.mean(diff))

diffs = np.array(diffs)
print(f"Adjacent sub-face colour diffs (in [0,1] space) for t1:")
print(f"  mean={diffs.mean():.4f}  median={np.median(diffs):.4f}")
print(f"  max={diffs.max():.4f}  min={diffs.min():.4f}")
print(f"  As 0-255: mean={diffs.mean()*255:.1f}  max={diffs.max()*255:.1f}")
print(f"  Count: {len(diffs)} adjacencies")

# How many have diff > some threshold?
for thresh in [0.01, 0.02, 0.05, 0.1]:
    cnt = np.sum(diffs > thresh)
    print(f"  Pairs with diff > {thresh}: {cnt} ({cnt/len(diffs)*100:.1f}%)")
