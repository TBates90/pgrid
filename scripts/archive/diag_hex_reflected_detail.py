#!/usr/bin/env python3
"""Show which specific neighbours of hex centres get reflected."""
import sys
sys.path.insert(0, "src")

from polygrid.globe import build_globe_grid
from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
from polygrid.detail_terrain import compute_neighbor_edge_mapping
from polygrid.tile_uv_align import compute_pg_to_macro_edge_map
from polygrid.assembly import _position_hex_for_stitch

grid = build_globe_grid(3)
spec = TileDetailSpec(detail_rings=4)
coll = DetailGridCollection.build(grid, spec)

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

hex_ids = [fid for fid in grid.faces if len(grid.faces[fid].vertex_ids) == 6]

for fid in hex_ids[:10]:
    n_sides = 6
    dg_center, _ = coll.get(fid)
    dg_center.compute_macro_edges(
        n_sides=n_sides,
        corner_ids=dg_center.metadata.get("corner_vertex_ids"),
    )
    pg_to_macro = compute_pg_to_macro_edge_map(grid, fid, dg_center)
    neigh_map = compute_neighbor_edge_mapping(grid, fid)

    reflected_neighs = []
    for nid, cpg in neigh_map.items():
        n_sides_n = len(grid.faces[nid].vertex_ids)
        dg_n, _ = coll.get(nid)
        dg_n.compute_macro_edges(
            n_sides=n_sides_n,
            corner_ids=dg_n.metadata.get("corner_vertex_ids"),
        )
        npg = compute_neighbor_edge_mapping(grid, nid)[fid]
        cm = pg_to_macro[cpg]
        pm = compute_pg_to_macro_edge_map(grid, nid, dg_n)[npg]

        positioned = _position_hex_for_stitch(dg_center, cm, dg_n, pm)
        sa_orig = signed_area(dg_n)
        sa_pos = signed_area(positioned)
        reflected = (sa_orig > 0) != (sa_pos > 0)
        nkind = "PENT" if n_sides_n == 5 else "hex"
        if reflected:
            reflected_neighs.append(f"{nid}({nkind})")

    if reflected_neighs:
        print(f"Hex centre {fid}: reflected neighbours = {', '.join(reflected_neighs)}")
