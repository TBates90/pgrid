#!/usr/bin/env python3
"""Test: does skipping edge reversal when windings differ eliminate the
need for reflection (and thus preserve internal cell orientation)?"""
import sys, math
sys.path.insert(0, "src")

import numpy as np
from polygrid.globe import build_globe_grid
from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
from polygrid.detail_terrain import compute_neighbor_edge_mapping
from polygrid.tile_uv_align import compute_pg_to_macro_edge_map
from polygrid.assembly import (
    _macro_edge_outward_normal, _macro_edge_midpoint,
    scale_grid, rotate_grid, translate_grid,
    _reflect_across_edge,
)
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


def position_winding_aware(target_grid, target_edge, source_grid, source_edge):
    """Like _position_hex_for_stitch but conditionally reverses the
    source edge based on winding agreement."""
    target_sa = signed_area(target_grid)
    source_sa = signed_area(source_grid)
    same_winding = (target_sa > 0) == (source_sa > 0)

    tme = next(m for m in target_grid.macro_edges if m.id == target_edge)
    t0 = target_grid.vertices[tme.vertex_ids[0]]
    t1 = target_grid.vertices[tme.vertex_ids[-1]]
    t_len = math.hypot(t1.x - t0.x, t1.y - t0.y)
    t_angle = math.atan2(t1.y - t0.y, t1.x - t0.x)

    sme = next(m for m in source_grid.macro_edges if m.id == source_edge)

    if same_winding:
        # Same winding: reverse the source edge (original behaviour)
        s0 = source_grid.vertices[sme.vertex_ids[-1]]
        s1 = source_grid.vertices[sme.vertex_ids[0]]
    else:
        # Different winding: DON'T reverse — edges already run same direction
        s0 = source_grid.vertices[sme.vertex_ids[0]]
        s1 = source_grid.vertices[sme.vertex_ids[-1]]

    s_len = math.hypot(s1.x - s0.x, s1.y - s0.y)

    scale = t_len / s_len if s_len > 1e-12 else 1.0
    scx, scy = (s0.x + s1.x) / 2, (s0.y + s1.y) / 2
    g = scale_grid(source_grid, scale, scx, scy)

    sme2 = next(m for m in g.macro_edges if m.id == source_edge)
    if same_winding:
        s0b = g.vertices[sme2.vertex_ids[-1]]
        s1b = g.vertices[sme2.vertex_ids[0]]
    else:
        s0b = g.vertices[sme2.vertex_ids[0]]
        s1b = g.vertices[sme2.vertex_ids[-1]]
    s_angle2 = math.atan2(s1b.y - s0b.y, s1b.x - s0b.x)

    rotation = t_angle - s_angle2
    rcx, rcy = (s0b.x + s1b.x) / 2, (s0b.y + s1b.y) / 2
    g = rotate_grid(g, rotation, rcx, rcy)

    sme3 = next(m for m in g.macro_edges if m.id == source_edge)
    if same_winding:
        s0c = g.vertices[sme3.vertex_ids[-1]]
    else:
        s0c = g.vertices[sme3.vertex_ids[0]]
    dx = t0.x - s0c.x
    dy = t0.y - s0c.y
    g = translate_grid(g, dx, dy)

    # Check: does the body end up on the outside?
    normal = _macro_edge_outward_normal(target_grid, target_edge)
    emx, emy = _macro_edge_midpoint(target_grid, target_edge)

    src_xs = [v.x for v in g.vertices.values() if v.has_position()]
    src_ys = [v.y for v in g.vertices.values() if v.has_position()]
    src_cx = sum(src_xs) / len(src_xs)
    src_cy = sum(src_ys) / len(src_ys)

    dot = (src_cx - emx) * normal[0] + (src_cy - emy) * normal[1]
    needs_reflect = dot < 0

    return g, same_winding, needs_reflect


# Test pentagon centre t0
fid = "t0"
n_sides = 5
dg_center, _ = coll.get(fid)
dg_center.compute_macro_edges(n_sides=n_sides, corner_ids=dg_center.metadata.get("corner_vertex_ids"))
pg_to_macro = compute_pg_to_macro_edge_map(grid, fid, dg_center)
neigh_map = compute_neighbor_edge_mapping(grid, fid)

print("Pentagon t0 — winding-aware positioning:")
for nid, cpg in neigh_map.items():
    n_sides_n = len(grid.faces[nid].vertex_ids)
    dg_n, _ = coll.get(nid)
    dg_n.compute_macro_edges(n_sides=n_sides_n, corner_ids=dg_n.metadata.get("corner_vertex_ids"))
    npg = compute_neighbor_edge_mapping(grid, nid)[fid]
    cm = pg_to_macro[cpg]
    pm = compute_pg_to_macro_edge_map(grid, nid, dg_n)[npg]

    g, same_w, needs_ref = position_winding_aware(dg_center, cm, dg_n, pm)
    sa_orig = signed_area(dg_n)
    sa_new = signed_area(g)
    reflected = (sa_orig > 0) != (sa_new > 0)

    print(f"  {nid}: same_winding={same_w}, needs_reflect_after={needs_ref}, "
          f"winding_changed={reflected}")

# Test hexagon centre t1
fid = "t1"
n_sides = 6
dg_center, _ = coll.get(fid)
dg_center.compute_macro_edges(n_sides=n_sides, corner_ids=dg_center.metadata.get("corner_vertex_ids"))
pg_to_macro = compute_pg_to_macro_edge_map(grid, fid, dg_center)
neigh_map = compute_neighbor_edge_mapping(grid, fid)

print("\nHexagon t1 — winding-aware positioning:")
for nid, cpg in neigh_map.items():
    n_sides_n = len(grid.faces[nid].vertex_ids)
    dg_n, _ = coll.get(nid)
    dg_n.compute_macro_edges(n_sides=n_sides_n, corner_ids=dg_n.metadata.get("corner_vertex_ids"))
    npg = compute_neighbor_edge_mapping(grid, nid)[fid]
    cm = pg_to_macro[cpg]
    pm = compute_pg_to_macro_edge_map(grid, nid, dg_n)[npg]

    g, same_w, needs_ref = position_winding_aware(dg_center, cm, dg_n, pm)
    sa_orig = signed_area(dg_n)
    sa_new = signed_area(g)
    reflected = (sa_orig > 0) != (sa_new > 0)
    nkind = "PENT" if n_sides_n == 5 else "hex"

    print(f"  {nid}({nkind}): same_winding={same_w}, needs_reflect_after={needs_ref}, "
          f"winding_changed={reflected}")
