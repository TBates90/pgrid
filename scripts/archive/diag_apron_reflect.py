#!/usr/bin/env python3
"""Diagnostic: check which neighbour grids get REFLECTED when stitched
around pentagon vs hexagon centre tiles.

The key question: does `_position_hex_for_stitch` reflect neighbour
grids differently depending on whether the centre tile is a pentagon
or hexagon?  A reflection in the Tutte 2D space flips the internal
cell content — and if the warp pipeline doesn't account for this,
the texture content at those apron cells will appear mirrored.
"""
import sys, math
sys.path.insert(0, "src")

import numpy as np
from polygrid.globe import build_globe_grid
from polygrid.tile_detail import (
    TileDetailSpec, DetailGridCollection,
    build_tile_with_neighbours,
)
from polygrid.detail_terrain import compute_neighbor_edge_mapping
from polygrid.tile_uv_align import compute_pg_to_macro_edge_map
from polygrid.assembly import (
    _position_hex_for_stitch,
    _macro_edge_outward_normal,
    _macro_edge_midpoint,
)

grid = build_globe_grid(3)
spec = TileDetailSpec(detail_rings=4)
coll = DetailGridCollection.build(grid, spec)

# Check a pentagon centre (t0) and a hexagon centre (t1)
test_faces = ["t0", "t1"]

for fid in test_faces:
    n_sides = len(grid.faces[fid].vertex_ids)
    kind = "PENTAGON" if n_sides == 5 else "HEXAGON"
    print(f"\n{'='*60}")
    print(f"Centre tile: {fid} ({kind}, {n_sides} sides)")
    print(f"{'='*60}")

    dg_center, _ = coll.get(fid)
    dg_center.compute_macro_edges(
        n_sides=n_sides,
        corner_ids=dg_center.metadata.get("corner_vertex_ids"),
    )

    pg_to_macro_center = compute_pg_to_macro_edge_map(grid, fid, dg_center)
    neigh_map = compute_neighbor_edge_mapping(grid, fid)

    for nid, center_pg_edge in neigh_map.items():
        n_sides_n = len(grid.faces[nid].vertex_ids)
        nkind = "pent" if n_sides_n == 5 else "hex"
        dg_n, _ = coll.get(nid)
        dg_n.compute_macro_edges(
            n_sides=n_sides_n,
            corner_ids=dg_n.metadata.get("corner_vertex_ids"),
        )

        neigh_pg_edge = compute_neighbor_edge_mapping(grid, nid)[fid]
        center_macro_edge = pg_to_macro_center[center_pg_edge]
        pg_to_macro_n = compute_pg_to_macro_edge_map(grid, nid, dg_n)
        neigh_macro_edge = pg_to_macro_n[neigh_pg_edge]

        # Replicate the logic from _position_hex_for_stitch to check
        # whether reflection is triggered
        positioned = _position_hex_for_stitch(
            dg_center, center_macro_edge, dg_n, neigh_macro_edge,
        )

        # Check: was the grid reflected?
        # We can detect this by checking if the positioned grid's
        # winding changed relative to the original.
        # Use signed area of the polygon formed by macro-edge corners.
        def signed_area(grid_obj):
            corners = []
            for me in grid_obj.macro_edges:
                v = grid_obj.vertices[me.vertex_ids[0]]
                corners.append((v.x, v.y))
            s = 0.0
            n = len(corners)
            for i in range(n):
                j = (i + 1) % n
                s += corners[i][0] * corners[j][1] - corners[j][0] * corners[i][1]
            return s / 2.0

        sa_orig = signed_area(dg_n)
        sa_pos = signed_area(positioned)
        reflected = (sa_orig > 0) != (sa_pos > 0)

        # Also check: what's the dot product that triggers reflection?
        # Redo the check from _position_hex_for_stitch
        normal = _macro_edge_outward_normal(dg_center, center_macro_edge)
        emx, emy = _macro_edge_midpoint(dg_center, center_macro_edge)

        # Get centroid of the positioned grid BEFORE reflection
        # (we need to re-do positioning without reflection to check)
        # Instead, just report whether reflection happened
        print(f"  Neighbour {nid:>3} ({nkind}): "
              f"orig_SA={sa_orig:+.4f} pos_SA={sa_pos:+.4f} "
              f"reflected={'YES' if reflected else 'no'}")
