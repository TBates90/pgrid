#!/usr/bin/env python3
"""For the same 3D edge point, which sub-face does each composite
put it in, and do those sub-faces have the same colour?
"""
import sys
import math
import numpy as np
from matplotlib.path import Path as MplPath
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
from polygrid.tile_uv_align import compute_tile_view_limits

grid = build_globe_grid(3)
schema = TileSchema([FieldDef("elevation", float, 0.0)])
store = TileDataStore(grid=grid, schema=schema)
config = MountainConfig(seed=42, ridge_frequency=2.0, ridge_octaves=4,
                        peak_elevation=1.0, base_elevation=0.0)
generate_mountains(grid, store, config)
coll = DetailGridCollection.build(grid, TileDetailSpec(detail_rings=4))
generate_all_detail_terrain(coll, grid, store, TileDetailSpec(detail_rings=4), seed=42)

biome = BiomeConfig()

comp_t0 = build_tile_with_neighbours(coll, "t0", grid)
comp_t1 = build_tile_with_neighbours(coll, "t1", grid)

# Test points from the warp trace (grid coords in each composite)
test_points = [
    (0.1, (0.1356, -0.1524), (-0.1174, -0.1913)),
    (0.3, (0.1519, -0.1037), (-0.1436, -0.1519)),
    (0.5, (0.1682, -0.0541), (-0.1703, -0.1134)),
    (0.7, (0.1837, -0.0053), (-0.1964, -0.0740)),
    (0.9, (0.2000,  0.0434), (-0.2218, -0.0350)),
]

# Build stitched stores with noise coords
def _build_stitched_store(composite, coll):
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

# Build hillshade (authoritative)
from polygrid.detail_render import _detail_hillshade

def compute_authoritative_hs(coll, grid, fids, biome):
    result = {}
    for fid in fids:
        comp = build_tile_with_neighbours(coll, fid, grid)
        st = _build_stitched_store(comp, coll)
        hs = _detail_hillshade(comp.merged, st, "elevation",
                               azimuth=biome.azimuth, altitude=biome.altitude)
        prefix = comp.id_prefixes[fid]
        for mfid, val in hs.items():
            if mfid.startswith(prefix):
                orig = mfid[len(prefix):]
                result[(fid, orig)] = val
    return result

print("Computing authoritative hillshade...")
auth_hs = compute_authoritative_hs(coll, grid, coll.face_ids, biome)

# Resolve for each composite
def resolve_hs(composite, auth_hs):
    result = {}
    for comp_name, prefix in composite.id_prefixes.items():
        for orig_fid in composite.components[comp_name].faces:
            mfid = f"{prefix}{orig_fid}"
            result[mfid] = auth_hs.get((comp_name, orig_fid), 0.5)
    return result

hs_t0 = resolve_hs(comp_t0, auth_hs)
hs_t1 = resolve_hs(comp_t1, auth_hs)

st_t0 = _build_stitched_store(comp_t0, coll)
st_t1 = _build_stitched_store(comp_t1, coll)

mg_t0 = comp_t0.merged
mg_t1 = comp_t1.merged


def find_face_at(merged_grid, x, y):
    """Find which face contains point (x, y)."""
    for fid, face in merged_grid.faces.items():
        verts = []
        for vid in face.vertex_ids:
            v = merged_grid.vertices.get(vid)
            if v and v.has_position():
                verts.append((v.x, v.y))
        if len(verts) < 3:
            continue
        path = MplPath(verts + [verts[0]])
        if path.contains_point((x, y)):
            return fid
    return None


def compute_colour(fid, merged_grid, store, hs_dict, biome, noise_seed=42):
    """Compute the colour for a face using the same pipeline as rendering."""
    face = merged_grid.faces[fid]
    elev = store.get(fid, "elevation")
    try:
        cx = store.get(fid, "noise_x")
        cy = store.get(fid, "noise_y")
    except:
        c = _face_center(merged_grid.vertices, face)
        cx, cy = c if c else (0, 0)
    hs_val = hs_dict.get(fid, 0.5)
    return detail_elevation_to_colour(
        elev, biome, hillshade_val=hs_val,
        noise_x=cx, noise_y=cy, noise_seed=noise_seed,
    )


print("\nTracing each test point:")
for t_param, (gx0, gy0), (gx1, gy1) in test_points:
    fid_in_t0 = find_face_at(mg_t0, gx0, gy0)
    fid_in_t1 = find_face_at(mg_t1, gx1, gy1)
    
    if fid_in_t0 and fid_in_t1:
        # Get component and original face ID
        for cname, prefix in comp_t0.id_prefixes.items():
            if fid_in_t0.startswith(prefix):
                orig_fid_0 = fid_in_t0[len(prefix):]
                comp_name_0 = cname
                break
        for cname, prefix in comp_t1.id_prefixes.items():
            if fid_in_t1.startswith(prefix):
                orig_fid_1 = fid_in_t1[len(prefix):]
                comp_name_1 = cname
                break
        
        col0 = compute_colour(fid_in_t0, mg_t0, st_t0, hs_t0, biome)
        col1 = compute_colour(fid_in_t1, mg_t1, st_t1, hs_t1, biome)
        
        elev0 = st_t0.get(fid_in_t0, "elevation")
        elev1 = st_t1.get(fid_in_t1, "elevation")
        hs0 = hs_t0.get(fid_in_t0, 0.5)
        hs1 = hs_t1.get(fid_in_t1, 0.5)
        nx0 = st_t0.get(fid_in_t0, "noise_x")
        ny0 = st_t0.get(fid_in_t0, "noise_y")
        nx1 = st_t1.get(fid_in_t1, "noise_x")
        ny1 = st_t1.get(fid_in_t1, "noise_y")
        
        rgb0 = tuple(int(c*255) for c in col0)
        rgb1 = tuple(int(c*255) for c in col1)
        diff = sum(abs(a-b) for a,b in zip(rgb0, rgb1)) / 3
        
        same_face = (comp_name_0 == comp_name_1 and orig_fid_0 == orig_fid_1)
        
        print(f"  t={t_param:.1f}:")
        print(f"    comp_t0: {fid_in_t0} ({comp_name_0}:{orig_fid_0}) "
              f"elev={elev0:.4f} hs={hs0:.4f} noise=({nx0:.4f},{ny0:.4f})")
        print(f"    comp_t1: {fid_in_t1} ({comp_name_1}:{orig_fid_1}) "
              f"elev={elev1:.4f} hs={hs1:.4f} noise=({nx1:.4f},{ny1:.4f})")
        print(f"    Same underlying face: {same_face}")
        print(f"    Colour t0: {rgb0}")
        print(f"    Colour t1: {rgb1}")
        print(f"    Diff: {diff:.1f}")
    else:
        print(f"  t={t_param:.1f}: face not found (t0={fid_in_t0}, t1={fid_in_t1})")
