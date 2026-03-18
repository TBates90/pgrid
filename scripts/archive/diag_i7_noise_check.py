#!/usr/bin/env python3
"""Quick check: are noise_x/noise_y being stored and different from centroid?"""
import sys
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

# Inline _build_stitched_store from render_polygrids
def _build_stitched_store(composite, coll, face_id, globe_grid):
    schema = TileSchema([
        FieldDef("elevation", float, 0.0),
        FieldDef("noise_x", float, 0.0),
        FieldDef("noise_y", float, 0.0),
    ])
    store = TileDataStore(grid=composite.merged, schema=schema)
    for comp_name, prefix in composite.id_prefixes.items():
        dg_orig, comp_store = coll.get(comp_name)
        if comp_store is None:
            continue
        for fid in composite.components[comp_name].faces:
            prefixed_fid = f"{prefix}{fid}"
            if prefixed_fid in composite.merged.faces:
                store.set(prefixed_fid, "elevation",
                          comp_store.get(fid, "elevation"))
                if fid in dg_orig.faces:
                    c = _face_center(dg_orig.vertices, dg_orig.faces[fid])
                    if c:
                        store.set(prefixed_fid, "noise_x", c[0])
                        store.set(prefixed_fid, "noise_y", c[1])
    return store

grid = build_globe_grid(3)
schema = TileSchema([FieldDef("elevation", float, 0.0)])
store = TileDataStore(grid=grid, schema=schema)
config = MountainConfig(seed=42, ridge_frequency=2.0, ridge_octaves=4,
                        peak_elevation=1.0, base_elevation=0.0)
generate_mountains(grid, store, config)
coll = DetailGridCollection.build(grid, TileDetailSpec(detail_rings=4))
generate_all_detail_terrain(coll, grid, store, TileDetailSpec(detail_rings=4), seed=42)

# Compare noise coords in comp_t0 vs comp_t1 for t1's faces
comp_t0 = build_tile_with_neighbours(coll, "t0", grid)
comp_t1 = build_tile_with_neighbours(coll, "t1", grid)

st_t0 = _build_stitched_store(comp_t0, coll, "t0", grid)
st_t1 = _build_stitched_store(comp_t1, coll, "t1", grid)

prefix_t1_in_t0 = comp_t0.id_prefixes["t1"]
prefix_t1_in_t1 = comp_t1.id_prefixes["t1"]

mg_t0 = comp_t0.merged
mg_t1 = comp_t1.merged

# Pick a few t1 faces common to both composites
common_fids = []
for fid in comp_t0.components["t1"].faces:
    pfid_0 = f"{prefix_t1_in_t0}{fid}"
    pfid_1 = f"{prefix_t1_in_t1}{fid}"
    if pfid_0 in mg_t0.faces and pfid_1 in mg_t1.faces:
        common_fids.append(fid)

print(f"Common t1 faces: {len(common_fids)}")

for fid in sorted(common_fids)[:10]:
    pfid_0 = f"{prefix_t1_in_t0}{fid}"
    pfid_1 = f"{prefix_t1_in_t1}{fid}"
    
    nx_0 = st_t0.get(pfid_0, "noise_x")
    ny_0 = st_t0.get(pfid_0, "noise_y")
    nx_1 = st_t1.get(pfid_1, "noise_x")
    ny_1 = st_t1.get(pfid_1, "noise_y")
    
    # Also get merged centroid for comparison
    c0 = _face_center(mg_t0.vertices, mg_t0.faces[pfid_0])
    c1 = _face_center(mg_t1.vertices, mg_t1.faces[pfid_1])
    
    print(f"  {fid}:")
    print(f"    noise_x: t0={nx_0:.4f}  t1={nx_1:.4f}  same={abs(nx_0-nx_1)<1e-6}")
    print(f"    noise_y: t0={ny_0:.4f}  t1={ny_1:.4f}  same={abs(ny_0-ny_1)<1e-6}")
    print(f"    merged_cx: t0={c0[0]:.4f}  t1={c1[0]:.4f}")
    print(f"    merged_cy: t0={c0[1]:.4f}  t1={c1[1]:.4f}")
