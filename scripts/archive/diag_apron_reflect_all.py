#!/usr/bin/env python3
"""Diagnostic: check reflection pattern for ALL pentagon centres.
Confirms whether every pentagon's hex neighbours always get reflected."""
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

pent_ids = [fid for fid in grid.faces if len(grid.faces[fid].vertex_ids) == 5]
hex_ids = [fid for fid in grid.faces if len(grid.faces[fid].vertex_ids) == 6]

# Check all pentagons
pent_reflected_count = 0
pent_total = 0

for fid in pent_ids:
    n_sides = 5
    dg_center, _ = coll.get(fid)
    dg_center.compute_macro_edges(
        n_sides=n_sides,
        corner_ids=dg_center.metadata.get("corner_vertex_ids"),
    )
    pg_to_macro = compute_pg_to_macro_edge_map(grid, fid, dg_center)
    neigh_map = compute_neighbor_edge_mapping(grid, fid)

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

        pent_total += 1
        if reflected:
            pent_reflected_count += 1

# Check a sample of hexagons
hex_reflected_count = 0
hex_total = 0

for fid in hex_ids[:10]:  # sample 10 hexagons
    n_sides = 6
    dg_center, _ = coll.get(fid)
    dg_center.compute_macro_edges(
        n_sides=n_sides,
        corner_ids=dg_center.metadata.get("corner_vertex_ids"),
    )
    pg_to_macro = compute_pg_to_macro_edge_map(grid, fid, dg_center)
    neigh_map = compute_neighbor_edge_mapping(grid, fid)

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

        nkind = "pent" if n_sides_n == 5 else "hex"
        hex_total += 1
        if reflected:
            hex_reflected_count += 1

print("=== PENTAGON CENTRES ===")
print(f"  Neighbours reflected: {pent_reflected_count}/{pent_total} "
      f"({100*pent_reflected_count/pent_total:.0f}%)")

print("=== HEXAGON CENTRES (sample of 10) ===")
print(f"  Neighbours reflected: {hex_reflected_count}/{hex_total} "
      f"({100*hex_reflected_count/hex_total:.0f}%)")
print()
print("If pentagon centres have 100% reflection and hex centres have ~0% (or")
print("only pent neighbours), the apron stitching orientation is the root cause.")
