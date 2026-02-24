from __future__ import annotations

import math
from typing import Dict, List

from ..models import Vertex
from ..polygrid import PolyGrid
from ..algorithms import build_face_adjacency, ring_faces
from ..angle_solver import ring_angle_spec, solve_ring_hex_lengths
from ..builders import (
    _find_pentagon_face,
    _face_vertex_cycle,
    _ordered_face_vertices,
    _collect_face_vertices,
    _order_ring_faces,
    _rotate_vertices,
    _rotate_list,
    _find_edge_start_index,
    _hex_points_from_edge,
    _hex_points_from_edge_candidates,
    _ordered_boundary_edges,
    _optimize_ring1_positions,
)


def apply_angle_first_layout(grid: PolyGrid, rings: int, size: float) -> Dict[str, Vertex]:
    """Experimental angle-first layout. Kept for diagnostics/debugging."""
    pent_face = _find_pentagon_face(grid)
    if pent_face is None:
        return grid.vertices

    pent_vertices = list(pent_face.vertex_ids)
    if len(pent_vertices) != 5:
        return grid.vertices

    side_length = size
    radius = side_length / (2 * math.sin(math.pi / 5))
    center = (0.0, 0.0)
    pent_points = [
        (
            radius * math.cos(2 * math.pi * i / 5),
            radius * math.sin(2 * math.pi * i / 5),
        )
        for i in range(5)
    ]

    ordered_pent = _face_vertex_cycle(pent_face, grid.edges.values())
    if len(ordered_pent) != 5:
        ordered_pent = _ordered_face_vertices(grid.vertices, pent_face)
    if len(ordered_pent) != 5:
        ordered_pent = pent_vertices

    current: Dict[str, Vertex] = {vid: Vertex(vid, v.x, v.y) for vid, v in grid.vertices.items()}
    for vid, (x, y) in zip(ordered_pent, pent_points):
        current[vid] = Vertex(vid, x, y)

    adjacency = build_face_adjacency(grid.faces.values(), grid.edges.values())
    rings_map = ring_faces(adjacency, pent_face.id, max_depth=rings)

    prev_outer = side_length
    for ring_idx in range(1, rings + 1):
        ring_face_ids = rings_map.get(ring_idx, [])
        if not ring_face_ids:
            continue
        ring_faces_list = [grid.faces[fid] for fid in ring_face_ids if fid in grid.faces]

        region_faces = []
        for depth in range(0, ring_idx):
            region_faces.extend(rings_map.get(depth, []))
        inner_vertices = set(_collect_face_vertices(grid, region_faces))
        region_set = set(region_faces)
        ring_set = set(ring_face_ids)

        spec = ring_angle_spec(ring_idx)
        angles = [
            spec.inner_angle_deg,
            spec.outer_angle_deg,
            spec.outer_angle_deg,
            spec.outer_angle_deg,
            spec.outer_angle_deg,
            spec.inner_angle_deg,
        ]

        if ring_idx == 1:
            edge_lookup = {frozenset(edge.vertex_ids): edge for edge in grid.edges.values()}
            pent_index = {vid: idx for idx, vid in enumerate(ordered_pent)}
            ordered_faces = _order_ring_faces(current, ring_faces_list, inner_vertices, center)
            ring1_targets: Dict[str, List[float]] = {}
            ring1_counts: Dict[str, int] = {}
            for face in ordered_faces:
                ordered = _face_vertex_cycle(face, grid.edges.values())
                if len(ordered) != 6:
                    continue

                # identify the pent edge for this face to compute inner edge length
                pent_edge = None
                for edge_id in face.edge_ids:
                    edge = grid.edges.get(edge_id)
                    if edge is None:
                        continue
                    a, b = edge.vertex_ids
                    if a in inner_vertices and b in inner_vertices:
                        pent_edge = edge
                        break
                if pent_edge is None:
                    continue
                v0_id, v1_id = pent_edge.vertex_ids
                if v0_id in pent_index and v1_id in pent_index:
                    if (pent_index[v1_id] - pent_index[v0_id]) % len(ordered_pent) != 1:
                        v0_id, v1_id = v1_id, v0_id
                v0_inner = current[v0_id]
                v1_inner = current[v1_id]
                if v0_inner.x is None or v1_inner.x is None:
                    continue
                inner_edge_len = math.hypot(v1_inner.x - v0_inner.x, v1_inner.y - v0_inner.y)
                # build edge-length sequence based on topology instead of fixed order
                edge_types = []
                for i in range(len(ordered)):
                    a = ordered[i]
                    b = ordered[(i + 1) % len(ordered)]
                    edge = edge_lookup.get(frozenset((a, b)))
                    if edge is None:
                        edge_types.append("outer")
                        continue
                    if a in inner_vertices and b in inner_vertices:
                        edge_types.append("inner")
                    elif any(fid in ring_set and fid != face.id for fid in edge.face_ids):
                        edge_types.append("ring1")
                    else:
                        edge_types.append("outer")

                angles_seq = [
                    spec.inner_angle_deg if vid in inner_vertices else spec.outer_angle_deg
                    for vid in ordered
                ]

                inner_idx = next((i for i, t in enumerate(edge_types) if t == "inner"), None)
                if inner_idx is None:
                    continue
                ordered = _rotate_vertices(ordered, inner_idx)
                edge_types_rot = _rotate_list(edge_types, inner_idx)
                angles_vertex = _rotate_list(angles_seq, inner_idx)

                if not (ordered[0] == v0_id and ordered[1] == v1_id):
                    reversed_order = list(reversed(ordered))
                    rev_start = _find_edge_start_index(reversed_order, v0_id, v1_id)
                    if rev_start is not None:
                        ordered = _rotate_vertices(reversed_order, rev_start)
                        angles_vertex = _rotate_list(list(reversed(angles_vertex)), rev_start)

                # Recompute edge types based on the finalized ordering.
                edge_types_rot = []
                for i in range(len(ordered)):
                    a = ordered[i]
                    b = ordered[(i + 1) % len(ordered)]
                    edge = edge_lookup.get(frozenset((a, b)))
                    if edge is None:
                        edge_types_rot.append("outer")
                    elif a in inner_vertices and b in inner_vertices:
                        edge_types_rot.append("inner")
                    elif any(fid in ring_set and fid != face.id for fid in edge.face_ids):
                        edge_types_rot.append("ring1")
                    else:
                        edge_types_rot.append("outer")

                lengths = solve_ring_hex_lengths(ring_idx, inner_edge_length=inner_edge_len)

                lengths_by_type = {
                    "inner": lengths["inner"],
                    "ring1": lengths["protrude"],
                    "outer": lengths["outer"],
                }
                lengths_rot = [lengths_by_type[t] for t in edge_types_rot]
                angles_turn = _rotate_list(angles_vertex, 1)

                v0 = current[ordered[0]]
                v1 = current[ordered[1]]
                if v0.x is None or v1.x is None:
                    continue

                preferred_points = _hex_points_from_edge(
                    (v0.x, v0.y),
                    (v1.x, v1.y),
                    lengths_rot,
                    angles_turn,
                    center,
                )
                cw_points, ccw_points = _hex_points_from_edge_candidates(
                    (v0.x, v0.y),
                    (v1.x, v1.y),
                    lengths_rot,
                    angles_turn,
                )

                candidates = [
                    (ordered, cw_points),
                    (ordered, ccw_points),
                ]

                best_order = ordered
                best_points = preferred_points
                best_error = float("inf")
                for cand_order, cand_points in candidates:
                    error = 0.0
                    count = 0
                    for vid, (px, py) in zip(cand_order, cand_points):
                        v = current[vid]
                        if not v.has_position():
                            continue
                        error += (v.x - px) ** 2 + (v.y - py) ** 2
                        count += 1
                    if count == 0:
                        error = 0.0
                    if error < best_error - 1e-12:
                        best_error = error
                        best_order = cand_order
                        best_points = cand_points
                    elif abs(error - best_error) <= 1e-12:
                        if cand_points == preferred_points:
                            best_order = cand_order
                            best_points = cand_points

                for vid, (px, py) in zip(best_order, best_points):
                    if vid in inner_vertices:
                        continue
                    if vid not in ring1_targets:
                        ring1_targets[vid] = [0.0, 0.0]
                        ring1_counts[vid] = 0
                    ring1_targets[vid][0] += px
                    ring1_targets[vid][1] += py
                    ring1_counts[vid] += 1
            for vid, (sx, sy) in ring1_targets.items():
                count = ring1_counts[vid]
                if count <= 0:
                    continue
                current[vid] = Vertex(vid, sx / count, sy / count)
            # refine ring-1 vertex positions to satisfy shared edge lengths
            pent_lengths: list[float] = []
            for idx, vid in enumerate(ordered_pent):
                nxt = ordered_pent[(idx + 1) % len(ordered_pent)]
                v0 = current[vid]
                v1 = current[nxt]
                if v0.has_position() and v1.has_position():
                    pent_lengths.append(math.hypot(v1.x - v0.x, v1.y - v0.y))
            inner_edge_len = sum(pent_lengths) / len(pent_lengths) if pent_lengths else side_length
            current = _optimize_ring1_positions(
                grid,
                current,
                ring_faces_list,
                inner_vertices,
                inner_edge_len,
            )
            continue

        boundary_edges = _ordered_boundary_edges(grid, region_set, center)

        outer_lengths: list[float] = []
        for edge in boundary_edges:
            v0_id, v1_id = edge.vertex_ids
            if v0_id not in inner_vertices or v1_id not in inner_vertices:
                continue
            face_id = next((fid for fid in edge.face_ids if fid in ring_set), None)
            if face_id is None:
                continue
            face = grid.faces.get(face_id)
            if face is None:
                continue
            v0_inner = current[v0_id]
            v1_inner = current[v1_id]
            if v0_inner.x is None or v1_inner.x is None:
                continue
            inner_edge_len = math.hypot(v1_inner.x - v0_inner.x, v1_inner.y - v0_inner.y)
            lengths = solve_ring_hex_lengths(ring_idx, inner_edge_length=inner_edge_len)
            edge_lengths = [
                lengths["inner"],
                lengths["protrude"],
                lengths["outer"],
                lengths["outer"],
                lengths["outer"],
                lengths["protrude"],
            ]
            ordered = _face_vertex_cycle(face, grid.edges.values())
            if len(ordered) != 6:
                continue
            start_idx = _find_edge_start_index(ordered, v0_id, v1_id)
            if start_idx is None:
                continue
            ordered = _rotate_vertices(ordered, start_idx)
            lengths_rot = _rotate_list(edge_lengths, start_idx)
            angles_vertex = [
                spec.inner_angle_deg if vid in inner_vertices else spec.outer_angle_deg
                for vid in ordered
            ]
            angles_turn = _rotate_list(angles_vertex, 1)

            v0 = current[ordered[0]]
            v1 = current[ordered[1]]
            if v0.x is None or v1.x is None:
                continue
            hex_points = _hex_points_from_edge(
                (v0.x, v0.y),
                (v1.x, v1.y),
                lengths_rot,
                angles_turn,
                center,
            )

            for idx, vid in enumerate(ordered):
                if vid in inner_vertices:
                    continue
                px, py = hex_points[idx]
                if vid not in current:
                    continue
                current[vid] = Vertex(vid, px, py)

        prev_outer *= 1.1

    return current
