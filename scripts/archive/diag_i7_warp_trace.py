#!/usr/bin/env python3
"""Trace: for points along shared edge, what source-image pixel do the
warp maps point to, and what colour is there?

This reveals whether the warp is pointing to the wrong location or
whether the source images have genuinely different content.
"""
import sys
import json
import math
import numpy as np
from pathlib import Path
from PIL import Image

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
from polygrid.uv_texture import get_tile_uv_vertices
from polygrid.tile_uv_align import (
    compute_tile_view_limits,
    get_macro_edge_corners,
    match_grid_corners_to_uv,
    _compute_piecewise_warp_map,
    compute_pg_to_macro_edge_map,
)
from polygrid.detail_terrain import compute_neighbor_edge_mapping

EXPORTS = Path("exports/f3")

grid = build_globe_grid(3)
schema = TileSchema([FieldDef("elevation", float, 0.0)])
store = TileDataStore(grid=grid, schema=schema)
config = MountainConfig(seed=42, ridge_frequency=2.0, ridge_octaves=4,
                        peak_elevation=1.0, base_elevation=0.0)
generate_mountains(grid, store, config)
coll = DetailGridCollection.build(grid, TileDetailSpec(detail_rings=4))
generate_all_detail_terrain(coll, grid, store, TileDetailSpec(detail_rings=4), seed=42)

tile_size = 512
gutter = 4
slot_size = tile_size + 2 * gutter

uv_t0 = get_tile_uv_vertices(grid, "t0")
uv_t1 = get_tile_uv_vertices(grid, "t1")

# Build composites
comp_t0 = build_tile_with_neighbours(coll, "t0", grid)
comp_t1 = build_tile_with_neighbours(coll, "t1", grid)

xlim_t0, ylim_t0 = compute_tile_view_limits(comp_t0, "t0")
xlim_t1, ylim_t1 = compute_tile_view_limits(comp_t1, "t1")

# Get grid corners for t0 and t1
for fid, comp, xlim, ylim, uv_corners in [
    ("t0", comp_t0, xlim_t0, ylim_t0, uv_t0),
    ("t1", comp_t1, xlim_t1, ylim_t1, uv_t1),
]:
    n_sides = len(grid.faces[fid].vertex_ids)
    dg, _ = coll.get(fid)
    is_pent = n_sides == 5
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)
    grid_corners = get_macro_edge_corners(dg, n_sides)
    matched = match_grid_corners_to_uv(grid_corners, grid, fid, detail_grid=dg)
    
    print(f"\n{fid} ({n_sides} sides):")
    print(f"  Grid corners (matched): {[(f'{x:.4f}',f'{y:.4f}') for x,y in matched]}")
    print(f"  UV corners: {[(f'{u:.4f}',f'{v:.4f}') for u,v in uv_corners]}")
    print(f"  xlim={xlim}, ylim={ylim}")

# Now compute warp maps for both tiles
for fid, comp, xlim, ylim, uv_corners in [
    ("t0", comp_t0, xlim_t0, ylim_t0, uv_t0),
    ("t1", comp_t1, xlim_t1, ylim_t1, uv_t1),
]:
    n_sides = len(grid.faces[fid].vertex_ids)
    dg, _ = coll.get(fid)
    is_pent = n_sides == 5
    corner_ids = dg.metadata.get("corner_vertex_ids") if is_pent else None
    dg.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)
    grid_corners = get_macro_edge_corners(dg, n_sides)
    matched = match_grid_corners_to_uv(grid_corners, grid, fid, detail_grid=dg)
    
    raw_img = Image.open(EXPORTS / f"{fid}.png").convert("RGB")
    img_w, img_h = raw_img.size
    
    map_x, map_y = _compute_piecewise_warp_map(
        matched, uv_corners,
        tile_size=tile_size,
        gutter=gutter,
        img_w=img_w, img_h=img_h,
        xlim=xlim, ylim=ylim,
        output_size=slot_size,
    )
    
    print(f"\n{fid} warp map range:")
    print(f"  map_x: [{map_x.min():.1f}, {map_x.max():.1f}]")
    print(f"  map_y: [{map_y.min():.1f}, {map_y.max():.1f}]")
    
    # For the shared edge, sample the warp map
    # Shared edge: t0 corners 1→2, t1 corners 5→4
    if fid == "t0":
        edge_uv = [(uv_t0[1], uv_t0[2])]
    else:
        edge_uv = [(uv_t1[5], uv_t1[4])]
    
    for (uv_start, uv_end) in edge_uv:
        print(f"  Edge UV: ({uv_start[0]:.4f},{uv_start[1]:.4f}) → ({uv_end[0]:.4f},{uv_end[1]:.4f})")
        
        for t_param in np.linspace(0.1, 0.9, 5):
            u = uv_start[0] + t_param * (uv_end[0] - uv_start[0])
            v = uv_start[1] + t_param * (uv_end[1] - uv_start[1])
            
            # UV → warped pixel
            wpx = gutter + u * tile_size
            wpy = gutter + (1.0 - v) * tile_size
            
            ix = int(np.clip(wpx, 0, slot_size-1))
            iy = int(np.clip(wpy, 0, slot_size-1))
            
            # Warp map tells us where in source image
            src_x = map_x[iy, ix]
            src_y = map_y[iy, ix]
            
            # Read source image at that location
            raw_arr = np.array(raw_img)
            six = int(np.clip(src_x, 0, img_w-1))
            siy = int(np.clip(src_y, 0, img_h-1))
            src_rgb = raw_arr[siy, six]
            
            # Also convert src pixel back to grid coords
            gx = xlim[0] + src_x / img_w * (xlim[1] - xlim[0])
            gy = ylim[0] + (1.0 - src_y / img_h) * (ylim[1] - ylim[0])
            
            print(f"    t={t_param:.1f}: UV=({u:.3f},{v:.3f}) → warp({ix},{iy}) "
                  f"→ src({src_x:.1f},{src_y:.1f}) grid=({gx:.4f},{gy:.4f}) "
                  f"rgb={src_rgb}")
