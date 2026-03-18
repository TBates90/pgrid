#!/usr/bin/env python3
"""Trace: for a specific point on a shared macro-edge, what colour
does each composite produce?

Pick the midpoint of the t0-t1 shared macro-edge in UV space,
trace back to grid space in each composite, determine which sub-face
that point falls in, and compare colours.
"""
import sys
import math
import numpy as np
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

# Build both composites
comp_t0 = build_tile_with_neighbours(coll, "t0", grid)
comp_t1 = build_tile_with_neighbours(coll, "t1", grid)

# Render parameters
tile_size = 512

# Get view limits
xlim_t0, ylim_t0 = compute_tile_view_limits(comp_t0, "t0")
xlim_t1, ylim_t1 = compute_tile_view_limits(comp_t1, "t1")

print(f"t0 view: xlim={xlim_t0}, ylim={ylim_t0}")
print(f"t1 view: xlim={xlim_t1}, ylim={ylim_t1}")

# Check what linspace pixel centres look like
xs_t0 = np.linspace(xlim_t0[0], xlim_t0[1], tile_size)
ys_t0 = np.linspace(ylim_t0[1], ylim_t0[0], tile_size)
xs_t1 = np.linspace(xlim_t1[0], xlim_t1[1], tile_size)
ys_t1 = np.linspace(ylim_t1[1], ylim_t1[0], tile_size)

print(f"\nt0 pixel spacing: dx={xs_t0[1]-xs_t0[0]:.6f}, dy={ys_t0[1]-ys_t0[0]:.6f}")
print(f"t1 pixel spacing: dx={xs_t1[1]-xs_t1[0]:.6f}, dy={ys_t1[1]-ys_t1[0]:.6f}")

# Find the shared macro-edge between t0 and t1 in the composites
# Look at sub-faces near the boundary in each composite
mg_t0 = comp_t0.merged
mg_t1 = comp_t1.merged

# Sample points along the shared boundary
# First, find the macro-edge boundary between t0 and t1 in comp_t0
# The macro-edge of t0 (detail grid) that faces t1

from polygrid.detail_terrain import compute_neighbor_edge_mapping
from polygrid.tile_uv_align import compute_pg_to_macro_edge_map

dg_t0, _ = coll.get("t0")
n_sides_t0 = len(grid.faces["t0"].vertex_ids)
dg_t0.compute_macro_edges(
    n_sides=n_sides_t0,
    corner_ids=dg_t0.metadata.get("corner_vertex_ids"),
)
neigh_map_t0 = compute_neighbor_edge_mapping(grid, "t0")
pg_to_macro_t0 = compute_pg_to_macro_edge_map(grid, "t0", dg_t0)

t0_pg_edge_to_t1 = neigh_map_t0.get("t1")
if t0_pg_edge_to_t1 is not None:
    t0_macro_edge_to_t1 = pg_to_macro_t0[t0_pg_edge_to_t1]
    me = next(m for m in dg_t0.macro_edges if m.id == t0_macro_edge_to_t1)
    vids = list(me.vertex_ids)
    print(f"\nt0 macro-edge {t0_macro_edge_to_t1} faces t1")
    print(f"  {len(vids)} vertices on this edge")
    
    # Get edge vertex positions
    edge_pts = []
    for vid in vids:
        v = dg_t0.vertices[vid]
        edge_pts.append((v.x, v.y))
    
    ep = np.array(edge_pts)
    print(f"  x range: [{ep[:,0].min():.4f}, {ep[:,0].max():.4f}]")
    print(f"  y range: [{ep[:,1].min():.4f}, {ep[:,1].max():.4f}]")
    
    # Now sample a point near the midpoint of this edge
    mid_idx = len(vids) // 2
    mid_v = dg_t0.vertices[vids[mid_idx]]
    test_x, test_y = mid_v.x, mid_v.y
    print(f"\n  Test point (t0 grid space): ({test_x:.4f}, {test_y:.4f})")
    
    # Convert to pixel in t0 image
    px_x = (test_x - xlim_t0[0]) / (xlim_t0[1] - xlim_t0[0]) * tile_size
    px_y = (1.0 - (test_y - ylim_t0[0]) / (ylim_t0[1] - ylim_t0[0])) * tile_size
    print(f"  → t0 image pixel: ({px_x:.1f}, {px_y:.1f})")
    
    # Find which face in mg_t0 contains this point
    from matplotlib.path import Path as MplPath
    prefix_t0 = comp_t0.id_prefixes["t0"]
    
    found_face = None
    for fid, face in mg_t0.faces.items():
        verts = []
        for vid2 in face.vertex_ids:
            v = mg_t0.vertices.get(vid2)
            if v and v.has_position():
                verts.append((v.x, v.y))
        if len(verts) < 3:
            continue
        path = MplPath(verts + [verts[0]])
        if path.contains_point((test_x, test_y)):
            found_face = fid
            break
    
    print(f"  Face at test point in comp_t0: {found_face}")
    
    if found_face:
        # Now find the same face in comp_t1
        # Extract component and original face id
        for comp_name, prefix in comp_t0.id_prefixes.items():
            if found_face.startswith(prefix):
                orig_fid = found_face[len(prefix):]
                print(f"  Component: {comp_name}, original face: {orig_fid}")
                
                # Look it up in comp_t1
                prefix_in_t1 = comp_t1.id_prefixes.get(comp_name)
                if prefix_in_t1:
                    merged_fid_in_t1 = f"{prefix_in_t1}{orig_fid}"
                    if merged_fid_in_t1 in mg_t1.faces:
                        face_in_t1 = mg_t1.faces[merged_fid_in_t1]
                        c_t1 = _face_center(mg_t1.vertices, face_in_t1)
                        print(f"  Found in comp_t1: {merged_fid_in_t1}")
                        if c_t1:
                            print(f"  Centroid in comp_t1: ({c_t1[0]:.4f}, {c_t1[1]:.4f})")
                    else:
                        print(f"  NOT found in comp_t1!")
                break
