from __future__ import annotations

import math
from typing import Dict

from .models import Vertex
from .polygrid import PolyGrid


def apply_embedding(
    grid: PolyGrid,
    rings: int,
    size: float,
    embed_mode: str,
) -> PolyGrid:
    """Apply embedding strategy to a grid.

    Supported embed_mode values: "tutte", "tutte+optimise", "angle".
    """
    if embed_mode == "angle":
        from .experiments.angle_first import apply_angle_first_layout

        positioned = apply_angle_first_layout(grid, rings, size)
        return PolyGrid(positioned.values(), grid.edges.values(), grid.faces.values())

    from .embedding import tutte_embedding
    from .algorithms import build_face_adjacency, ring_faces
    from .angle_solver import ring_angle_spec
    from .builders import (
        _build_fixed_positions,
        _face_vertex_cycle,
        _find_pentagon_face,
        _interior_angle,
        _laplacian_relax,
        _ordered_face_vertices,
        _ring1_pent_angle_snap,
        _ring1_symmetry_relax,
        _ring1_symmetry_snap,
        _ring_pointy_edge_snap,
        _ring_protruding_edge_snap,
        _ring_constraints_snap,
        compute_edge_target_ratios,
        _global_optimize_positions,
    )

    fixed_positions = _build_fixed_positions(grid)
    if not fixed_positions:
        return grid

    embedded = tutte_embedding(grid.vertices, grid.edges.values(), fixed_positions)
    if embed_mode == "tutte":
        return PolyGrid(embedded.values(), grid.edges.values(), grid.faces.values())

    if rings <= 1:
        updated = PolyGrid(embedded.values(), grid.edges.values(), grid.faces.values())
        if rings == 0:
            return updated
        adjacency = build_face_adjacency(updated.faces.values(), updated.edges.values())
        pent_face = _find_pentagon_face(updated)
        if pent_face is None:
            return updated
        rings_map = ring_faces(adjacency, pent_face.id, max_depth=rings)
        inner_vertices = set(pent_face.vertex_ids)
        center = (
            sum(updated.vertices[vid].x for vid in pent_face.vertex_ids) / len(pent_face.vertex_ids),
            sum(updated.vertices[vid].y for vid in pent_face.vertex_ids) / len(pent_face.vertex_ids),
        )

        # Identify pointy vertices (outermost per ring-1 face).
        pointy_vertices: set[str] = set()
        for fid in rings_map.get(1, []):
            face = updated.faces.get(fid)
            if face is None:
                continue
            ordered = _face_vertex_cycle(face, updated.edges.values())
            if len(ordered) != len(face.vertex_ids):
                ordered = _ordered_face_vertices(updated.vertices, face)
            candidates = [vid for vid in ordered if vid not in inner_vertices]
            if not candidates:
                continue
            pointy_vertices.add(
                max(
                    candidates,
                    key=lambda vid: math.hypot(
                        updated.vertices[vid].x - center[0],
                        updated.vertices[vid].y - center[1],
                    ),
                )
            )

        def apply_pointy_scale(
            factor: float,
            base_vertices: Dict[str, Vertex],
        ) -> Dict[str, Vertex]:
            scaled = {vid: Vertex(vid, v.x, v.y) for vid, v in base_vertices.items()}
            for vid in pointy_vertices:
                v = scaled.get(vid)
                if v is None:
                    continue
                dx = v.x - center[0]
                dy = v.y - center[1]
                scaled[vid] = Vertex(vid, center[0] + dx * factor, center[1] + dy * factor)
            return scaled

        ring_vertices = set(
            vid for face in rings_map.get(1, []) for vid in updated.faces[face].vertex_ids
        )
        outer_ring_vertices = ring_vertices - inner_vertices
        from .builders import _fivefold_symmetry_snap

        def pointy_angle_mean(vertices: Dict[str, Vertex]) -> float:
            snapped_vertices = _fivefold_symmetry_snap(updated, vertices, outer_ring_vertices)
            angles: list[float] = []
            for fid in rings_map.get(1, []):
                face = updated.faces.get(fid)
                if face is None:
                    continue
                ordered = _face_vertex_cycle(face, updated.edges.values())
                if len(ordered) != len(face.vertex_ids):
                    ordered = _ordered_face_vertices(snapped_vertices, face)
                candidates = [vid for vid in ordered if vid not in inner_vertices]
                if not candidates:
                    continue
                pointy = max(
                    candidates,
                    key=lambda vid: math.hypot(
                        snapped_vertices[vid].x - center[0],
                        snapped_vertices[vid].y - center[1],
                    ),
                )
                idx = ordered.index(pointy)
                prev_vid = ordered[(idx - 1) % len(ordered)]
                next_vid = ordered[(idx + 1) % len(ordered)]
                angles.append(
                    math.degrees(
                        _interior_angle(snapped_vertices, prev_vid, pointy, next_vid)
                    )
                )
            return sum(angles) / len(angles) if angles else 0.0

        best_factor = 1.0
        target_pointy_angle = ring_angle_spec(1).outer_angle_deg
        if pointy_vertices:
            best_diff = float("inf")
            for factor in (0.90 + 0.005 * idx for idx in range(31)):
                scaled = apply_pointy_scale(factor, updated.vertices)
                mean_angle = pointy_angle_mean(scaled)
                diff = abs(mean_angle - target_pointy_angle)
                if diff < best_diff:
                    best_diff = diff
                    best_factor = factor

        scaled_vertices = apply_pointy_scale(best_factor, updated.vertices)

        snapped = _fivefold_symmetry_snap(updated, scaled_vertices, outer_ring_vertices)
        return PolyGrid(snapped.values(), updated.edges.values(), updated.faces.values())

    relaxed = _laplacian_relax(
        embedded,
        grid.edges.values(),
        fixed_positions,
        iterations=40,
        alpha=0.25,
    )
    ring1_relaxed = _ring1_symmetry_relax(
        grid,
        relaxed,
        fixed_positions,
        iterations=120,
        strength=0.25,
    )
    snapped = _ring1_symmetry_snap(grid, ring1_relaxed, fixed_positions)
    constrained = _ring_constraints_snap(
        grid,
        snapped,
        fixed_positions,
        max_ring=len(grid.faces),
        iterations=5,
    )
    protruded = _ring_protruding_edge_snap(
        grid,
        constrained,
        fixed_positions,
        max_ring=len(grid.faces),
    )
    angled = _ring1_pent_angle_snap(
        grid,
        protruded,
        fixed_positions,
        target_angle_deg=126.0,
        strength=1.0,
    )
    pointy_final = _ring_pointy_edge_snap(
        grid,
        angled,
        fixed_positions,
        max_ring=len(grid.faces),
    )
    updated = PolyGrid(pointy_final.values(), grid.edges.values(), grid.faces.values())

    adjacency = build_face_adjacency(updated.faces.values(), updated.edges.values())
    pent_face = _find_pentagon_face(updated)
    if pent_face is None:
        return updated

    rings_map = ring_faces(adjacency, pent_face.id, max_depth=rings)
    edge_targets = compute_edge_target_ratios(updated, rings_map)
    optimized = _global_optimize_positions(
        updated,
        updated.vertices,
        fixed_positions,
        edge_targets,
        rings_map,
    )
    return PolyGrid(optimized.values(), updated.edges.values(), updated.faces.values())
