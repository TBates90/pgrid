import math

from polygrid.builders import build_pentagon_centered_grid, _build_fixed_positions
from polygrid.algorithms import build_face_adjacency, ring_faces


def test_pentagon_centered_grid_center_face():
    grid = build_pentagon_centered_grid(1, embed=False)
    pent_faces = [face for face in grid.faces.values() if face.face_type == "pent"]
    assert len(pent_faces) == 1
    assert len(pent_faces[0].vertex_ids) == 5
    assert all(len(face.vertex_ids) in (5, 6) for face in grid.faces.values())


def test_angle_first_layout_positions_pentagon():
    grid = build_pentagon_centered_grid(1, embed=True, embed_mode="angle")
    pent_face = next(face for face in grid.faces.values() if face.face_type == "pent")
    for vid in pent_face.vertex_ids:
        v = grid.vertices[vid]
        assert v.x is not None and v.y is not None


import pytest


@pytest.mark.xfail(reason="Ring constraint snapping is being replaced by angle-first solver.")
def test_ring1_constraints_equalized():
    grid = build_pentagon_centered_grid(1, embed=True)
    pent_face = next(face for face in grid.faces.values() if face.face_type == "pent")
    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings = ring_faces(adjacency, pent_face.id, max_depth=3)
    ring1_face_ids = rings.get(1, [])
    ring1_vertices = set()
    for fid in ring1_face_ids:
        ring1_vertices.update(grid.faces[fid].vertex_ids)
    inner_vertices = set(pent_face.vertex_ids)

    protruding_map = {}
    for edge in grid.edges.values():
        a, b = edge.vertex_ids
        if (a in ring1_vertices and b in inner_vertices) or (b in ring1_vertices and a in inner_vertices):
            inner = a if a in inner_vertices else b
            outer = b if inner == a else a
            protruding_map.setdefault(outer, []).append(inner)

    fixed_positions = _build_fixed_positions(grid)
    protruding_lengths = []
    for outer, inners in protruding_map.items():
        if len(inners) != 1:
            continue
        if outer in fixed_positions:
            continue
        inner = inners[0]
        va = grid.vertices[inner]
        vb = grid.vertices[outer]
        protruding_lengths.append(math.hypot(vb.x - va.x, vb.y - va.y))

    assert protruding_lengths
    assert max(protruding_lengths) - min(protruding_lengths) < 1e-3

    center_x = sum(v.x for v in grid.vertices.values()) / len(grid.vertices)
    center_y = sum(v.y for v in grid.vertices.values()) / len(grid.vertices)
    inner_neighbor_counts = {vid: 0 for vid in ring1_vertices}
    for edge in grid.edges.values():
        a, b = edge.vertex_ids
        if a in ring1_vertices and b in inner_vertices:
            inner_neighbor_counts[a] += 1
        elif b in ring1_vertices and a in inner_vertices:
            inner_neighbor_counts[b] += 1
    pointy_lengths = []
    for fid in ring1_face_ids:
        face = grid.faces[fid]
        verts = _ordered_face_vertices(grid, face)
        if not verts:
            continue
        candidates = [
            vid
            for vid in verts
            if vid in ring1_vertices and inner_neighbor_counts.get(vid, 0) == 0
        ]
        if not candidates:
            continue
        pointy = max(
            candidates,
            key=lambda vid: math.hypot(grid.vertices[vid].x - center_x, grid.vertices[vid].y - center_y),
        )
        idx = verts.index(pointy)
        prev_vid = verts[(idx - 1) % len(verts)]
        next_vid = verts[(idx + 1) % len(verts)]
        pv = grid.vertices[pointy]
        prev_v = grid.vertices[prev_vid]
        next_v = grid.vertices[next_vid]
        pointy_lengths.append(math.hypot(prev_v.x - pv.x, prev_v.y - pv.y))
        pointy_lengths.append(math.hypot(next_v.x - pv.x, next_v.y - pv.y))

    assert pointy_lengths
    assert max(pointy_lengths) - min(pointy_lengths) < 1e-2


def _ordered_face_vertices(grid, face):
    verts = list(face.vertex_ids)
    if not verts:
        return verts
    xs = [grid.vertices[vid].x for vid in verts]
    ys = [grid.vertices[vid].y for vid in verts]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    return sorted(verts, key=lambda vid: math.atan2(grid.vertices[vid].y - cy, grid.vertices[vid].x - cx))
