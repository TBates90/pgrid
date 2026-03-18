#!/usr/bin/env python3
"""Diagnostic: visualise pentagon t0 stitching geometry.

Checks whether neighbours are positioned correctly around the center pentagon.
Outputs a scatter plot of vertex positions and macro-edge corners, and prints
the PG→macro edge map and corner data.
"""
import sys, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from polygrid.goldberg_topology import build_goldberg_grid
from polygrid.globe import build_globe_grid
from polygrid.detail_grid import build_detail_grid
from polygrid.detail_terrain import (
    compute_neighbor_edge_mapping,
    generate_all_detail_terrain,
)
from polygrid.tile_detail import (
    build_tile_with_neighbours,
    TileDetailSpec,
    DetailGridCollection,
)
from polygrid.tile_uv_align import compute_pg_to_macro_edge_map

FREQ = 3


def main():
    # Build globe
    grid, store = build_globe_grid(FREQ), None
    from polygrid.mountains import MountainConfig, generate_mountains
    from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)
    generate_mountains(grid, store, MountainConfig(seed=42))
    face_id = "t0"

    n_sides = len(grid.faces[face_id].vertex_ids)
    print(f"Face {face_id}: {n_sides} sides (pentagon)")
    print(f"  vertex_ids = {grid.faces[face_id].vertex_ids}")

    # Build detail grids for t0 and its neighbours
    neigh_map = compute_neighbor_edge_mapping(grid, face_id)
    print(f"\nNeighbour edge mapping (PG edge indices):")
    for nid, eidx in neigh_map.items():
        print(f"  {face_id} edge {eidx} ↔ {nid}")

    all_faces = [face_id] + list(neigh_map.keys())

    # Build detail grids using the proper pipeline
    spec = TileDetailSpec(detail_rings=4)
    coll = DetailGridCollection.build(grid, spec)
    generate_all_detail_terrain(coll, grid, store, seed=42)

    # Now look at the center detail grid's macro-edge mapping
    dg_center = coll.grids[face_id]
    corner_ids = dg_center.metadata.get("corner_vertex_ids")
    print(f"\nCenter detail grid metadata:")
    print(f"  corner_vertex_ids = {corner_ids}")
    dg_center.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)

    print(f"\nMacro edges (center):")
    for me in dg_center.macro_edges:
        v0 = dg_center.vertices[me.vertex_ids[0]]
        vn = dg_center.vertices[me.vertex_ids[-1]]
        print(f"  edge {me.id}: {len(me.vertex_ids)} verts, "
              f"({v0.x:.3f},{v0.y:.3f}) → ({vn.x:.3f},{vn.y:.3f})")

    # PG→macro mapping
    pg_to_macro = compute_pg_to_macro_edge_map(grid, face_id, dg_center)
    print(f"\nPG→macro edge map: {pg_to_macro}")

    # Now build composite (the real stitching)
    composite = build_tile_with_neighbours(coll, face_id, grid)
    mg = composite.merged

    # Print stats about composite
    n_verts = len(mg.vertices)
    n_faces = len(mg.faces)
    print(f"\nComposite: {n_verts} vertices, {n_faces} faces")

    # Get all vertex positions for scatter plot
    xs = [v.x for v in mg.vertices.values() if v.has_position()]
    ys = [v.y for v in mg.vertices.values() if v.has_position()]
    print(f"  x range: [{min(xs):.3f}, {max(xs):.3f}]")
    print(f"  y range: [{min(ys):.3f}, {max(ys):.3f}]")

    # Get the center tile vertices
    center_prefix = face_id + "_"  # component prefix
    # Actually, let's look at the components
    print(f"\nComponents: {list(composite.components.keys())}")

    # Check each neighbour's position relative to center
    # For each macro edge, check that the neighbour centroid is on the outside
    for nid, pg_eidx in neigh_map.items():
        # Get neighbour vertices from the composite
        n_prefix = nid + "_"
        n_verts = [(v.x, v.y) for vid, v in mg.vertices.items()
                   if isinstance(vid, str) and vid.startswith(n_prefix)
                   and v.has_position()]
        if n_verts:
            ncx = sum(x for x,y in n_verts) / len(n_verts)
            ncy = sum(y for x,y in n_verts) / len(n_verts)
            print(f"  Neighbour {nid} (PG edge {pg_eidx}): "
                  f"centroid ({ncx:.3f}, {ncy:.3f}), "
                  f"{len(n_verts)} verts")
        else:
            print(f"  Neighbour {nid}: NO positioned vertices!")

    # Check center's centroid
    c_verts = [(v.x, v.y) for vid, v in mg.vertices.items()
               if isinstance(vid, str) and vid.startswith(face_id + "_")
               and v.has_position()]
    if c_verts:
        ccx = sum(x for x,y in c_verts) / len(c_verts)
        ccy = sum(y for x,y in c_verts) / len(c_verts)
        print(f"  Center {face_id}: centroid ({ccx:.3f}, {ccy:.3f}), "
              f"{len(c_verts)} verts")

    # Save a visual diagnostic
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plot all vertices
    ax.scatter(xs, ys, s=1, c="gray", alpha=0.3)

    # Plot macro edges of center
    colors = ["red", "blue", "green", "orange", "purple"]
    for me in dg_center.macro_edges:
        exs = [dg_center.vertices[vid].x for vid in me.vertex_ids]
        eys = [dg_center.vertices[vid].y for vid in me.vertex_ids]
        c = colors[me.id % len(colors)]
        ax.plot(exs, eys, c=c, linewidth=2, label=f"edge {me.id}")
        # Label the edge
        mx = sum(exs) / len(exs)
        my = sum(eys) / len(eys)
        ax.text(mx, my, str(me.id), fontsize=12, fontweight="bold", color=c)

    # Plot neighbour centroids
    for nid, pg_eidx in neigh_map.items():
        n_prefix = nid + "_"
        n_verts_list = [(v.x, v.y) for vid, v in mg.vertices.items()
                        if isinstance(vid, str) and vid.startswith(n_prefix)
                        and v.has_position()]
        if n_verts_list:
            ncx = sum(x for x,y in n_verts_list) / len(n_verts_list)
            ncy = sum(y for x,y in n_verts_list) / len(n_verts_list)
            ax.plot(ncx, ncy, 'x', markersize=10, markeredgewidth=2)
            ax.annotate(f"{nid}\n(PG e{pg_eidx}→mac e{pg_to_macro.get(pg_eidx,'?')})",
                       (ncx, ncy), fontsize=8)

    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title(f"Pentagon {face_id} stitching diagnostic")

    out = ROOT / "exports" / "f3" / "pent_stitch_diag.png"
    fig.savefig(str(out), dpi=100)
    print(f"\n→ Saved: {out}")


if __name__ == "__main__":
    main()
