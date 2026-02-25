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
    if embed_mode in {"tutte", "tutte+optimise"} and all(
        v.has_position() for v in grid.vertices.values()
    ):
        if rings <= 1:
            return grid
    if embed_mode == "angle":
        from .experiments.angle_first import apply_angle_first_layout

        positioned = apply_angle_first_layout(grid, rings, size)
        return PolyGrid(positioned.values(), grid.edges.values(), grid.faces.values())

    from .embedding import tutte_embedding
    from .algorithms import build_face_adjacency, ring_faces
    from .angle_solver import ring_angle_spec, solve_ring_hex_lengths
    from .builders import (
        _apply_outer_ring_geometry,
        _build_fixed_positions,
        compute_edge_target_lengths_by_ring,
        compute_edge_target_ratios_by_class,
    )
    from .embedding_utils import (
        _fivefold_symmetry_snap,
        _laplacian_relax,
        _ring1_pent_angle_snap,
        _ring1_symmetry_relax,
        _ring1_symmetry_snap,
        _ring_constraints_snap,
        _ring_pointy_edge_snap,
        _ring_protruding_edge_snap,
    )
    from .geometry import _face_vertex_cycle, _interior_angle, _ordered_face_vertices
    from .grid_utils import _collect_face_vertices, _find_pentagon_face
    from .optimisation import (
        _global_optimize_positions,
        optimise_positions_to_edge_targets,
        optimize_outer_rings,
    )

    pre_positioned = all(v.has_position() for v in grid.vertices.values())
    if not pre_positioned and rings > 1:
        positioned = _apply_outer_ring_geometry(grid, rings)
        if all(v.has_position() for v in positioned.values()):
            grid = PolyGrid(positioned.values(), grid.edges.values(), grid.faces.values())
            pre_positioned = True

    if rings > 1 and all(v.has_position() for v in grid.vertices.values()):
        fixed_positions = _build_fixed_positions(grid)
    else:
        fixed_positions = _build_fixed_positions(grid)
    if not fixed_positions:
        return grid

    if pre_positioned:
        embedded = grid.vertices
    else:
        embedded = tutte_embedding(grid.vertices, grid.edges.values(), fixed_positions)
    if embed_mode == "tutte":
        return PolyGrid(embedded.values(), grid.edges.values(), grid.faces.values())

    if rings <= 1:
        return PolyGrid(embedded.values(), grid.edges.values(), grid.faces.values())

    if pre_positioned:
        updated = PolyGrid(embedded.values(), grid.edges.values(), grid.faces.values())
    else:
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
    edge_targets = compute_edge_target_lengths_by_ring(
        updated,
        rings_map,
        updated.vertices,
    )
    movable = len(updated.vertices) - len(fixed_positions)
    if pre_positioned:
        tightened = optimize_outer_rings(
            updated,
            updated.vertices,
            edge_targets,
            rings_map,
            fixed_positions,
            start_ring=2,
            iterations=60,
        )
        return PolyGrid(tightened.values(), updated.edges.values(), updated.faces.values())
    if movable > 120 or rings > 2:
        relaxed_positions = optimise_positions_to_edge_targets(
            updated,
            updated.vertices,
            compute_edge_target_ratios_by_class(updated, rings_map),
            fixed_positions,
            iterations=40,
        )
        tightened = optimize_outer_rings(
            updated,
            relaxed_positions,
            edge_targets,
            rings_map,
            fixed_positions,
            start_ring=2,
            iterations=60,
        )
        return PolyGrid(tightened.values(), updated.edges.values(), updated.faces.values())

    optimized = _global_optimize_positions(
        updated,
        updated.vertices,
        fixed_positions,
        compute_edge_target_ratios_by_class(updated, rings_map),
        rings_map,
        iterations=40,
    )
    tightened = optimize_outer_rings(
        updated,
        optimized,
        edge_targets,
        rings_map,
        fixed_positions,
        start_ring=2,
        iterations=60,
    )
    return PolyGrid(tightened.values(), updated.edges.values(), updated.faces.values())
