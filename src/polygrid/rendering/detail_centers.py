"""Detail cell 3D centre computation for Goldberg tiles.

For each globe tile, the detail grid contains ~60 sub-cells (rings=4).
This module computes the 3D sphere position of every detail-cell centre
so that the game engine can resolve a world-space hit point to a
sub-tile cell by nearest-centre lookup.

The pipeline per tile:
  1. Build the detail grid (same grid used for texture rendering).
  2. Get the tile's tangent-plane basis (``compute_tile_basis``).
  3. Build the piecewise-linear warp from detail-2D → tile UV space
     (``compute_detail_to_uv_transform``).
  4. For every detail cell, compute its 2D centroid, apply the warp
     to get normalised UV coordinates, invert the UV normalisation to
     recover raw tangent-plane coordinates, then reconstruct a 3D point
     and normalise it onto the unit sphere.

Output format written to ``detail_cells.json``:

.. code-block:: json

    {
      "metadata": {"frequency": 3, "detail_rings": 4},
            "tiles": {
                "t0": [
                    {
                        "id": "f0",
                        "center_3d": [x, y, z],
                        "canonical_center_3d": [x, y, z],
                        "ring_index": 0,
                        "position_in_ring": 0
                    },
                    ...
                ],
        "t1": [...]
      }
    }
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from ..core.geometry import face_center, ordered_face_vertices
from ..core.polygrid import PolyGrid
from .uv_texture import (
    compute_tile_basis,
    compute_tile_uv_bounds,
    compute_detail_to_uv_transform,
)
from .detail_topology import build_detail_cell_addresses, build_detail_ring_positions
from .detail_cell_contract import normalize_detail_cells_tiles
from ..detail.detail_grid import build_detail_grid

LOGGER = logging.getLogger(__name__)


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _invert_uv_normalization(
    u_norm: float,
    v_norm: float,
    uv_bounds: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """Invert ``project_and_normalize`` to recover raw tangent-plane coords.

    ``project_and_normalize`` applies uniform scaling (uses the larger span
    for both axes) with centering offsets.  Inverting:

        u_raw = (u_norm - offset_u) * span + u_min
        v_raw = (v_norm - offset_v) * span + v_min
    """
    u_min, v_min, u_max, v_max = uv_bounds
    u_span = max(u_max - u_min, 1e-12)
    v_span = max(v_max - v_min, 1e-12)
    span = max(u_span, v_span)
    offset_u = (span - u_span) / (2.0 * span)
    offset_v = (span - v_span) / (2.0 * span)
    u_raw = (u_norm - offset_u) * span + u_min
    v_raw = (v_norm - offset_v) * span + v_min
    return u_raw, v_raw


def _sphere_point_from_detail_xy(
    xy: tuple[float, float],
    *,
    center: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
    uv_bounds: Tuple[float, float, float, float],
    xform,
) -> np.ndarray:
    if xform is not None:
        try:
            u_norm, v_norm = xform.apply(xy[0], xy[1])
        except Exception:
            u_norm, v_norm = xy[0], xy[1]
    else:
        u_norm, v_norm = xy[0], xy[1]
    u_raw, v_raw = _invert_uv_normalization(u_norm, v_norm, uv_bounds)
    pt3d = center + u_raw * tangent + v_raw * bitangent
    return _normalize_vec(pt3d)


def compute_detail_cell_centers_3d(
    globe_grid: PolyGrid,
    globe_face_id: str,
    detail_rings: int = 4,
) -> List[Dict[str, Any]]:
    """Compute 3D sphere positions for every detail cell within a globe tile.

    Parameters
    ----------
    globe_grid:
        The ``GlobeGrid`` (or compatible ``PolyGrid`` with ``frequency``
        and ``radius`` metadata).
    globe_face_id:
        Face ID of the globe tile, e.g. ``"t0"``, ``"t42"``.
    detail_rings:
        Ring count used when building the detail grid.

    Returns
    -------
    list of ``{"id": str, "center_3d": [x, y, z]}`` sorted by face ID.
    Each ``center_3d`` is a unit-sphere point.
    """
    try:
        center, normal, tangent, bitangent = compute_tile_basis(globe_grid, globe_face_id)
    except Exception:
        LOGGER.warning(
            "compute_tile_basis failed for face %s — skipping", globe_face_id
        )
        return []

    try:
        grid = build_detail_grid(globe_grid, globe_face_id, detail_rings=detail_rings)
    except Exception:
        LOGGER.warning(
            "build_detail_grid failed for face %s — skipping", globe_face_id
        )
        return []

    uv_bounds = compute_tile_uv_bounds(globe_grid, globe_face_id, center, tangent, bitangent)

    try:
        xform = compute_detail_to_uv_transform(
            globe_grid, globe_face_id, grid, center, tangent, bitangent, uv_bounds
        )
    except Exception:
        LOGGER.warning(
            "compute_detail_to_uv_transform failed for face %s — "
            "falling back to raw tangent-plane projection",
            globe_face_id,
        )
        xform = None

    ring_positions = build_detail_ring_positions(grid, max_depth=max(0, int(detail_rings)))
    addresses = build_detail_cell_addresses(grid.faces.keys(), ring_positions)
    results: List[Dict[str, Any]] = []

    for face_id_local, face in sorted(grid.faces.items()):
        xy = face_center(grid.vertices, face)
        if xy is None:
            continue

        sphere_pt = _sphere_point_from_detail_xy(
            xy,
            center=center,
            tangent=tangent,
            bitangent=bitangent,
            uv_bounds=uv_bounds,
            xform=xform,
        )

        ordered_ids = ordered_face_vertices(grid.vertices, face)
        vertices_3d: list[list[float]] = []
        for vertex_id in ordered_ids:
            vertex = grid.vertices.get(vertex_id)
            if vertex is None or not vertex.has_position():
                continue
            sphere_vertex = _sphere_point_from_detail_xy(
                (float(vertex.x), float(vertex.y)),
                center=center,
                tangent=tangent,
                bitangent=bitangent,
                uv_bounds=uv_bounds,
                xform=xform,
            )
            vertices_3d.append([round(float(value), 9) for value in sphere_vertex])

        # Canonical center is derived from spherical vertex centroid. This is
        # deterministic for a given grid topology + ring count and provides a
        # stable anchor for downstream marker placement.
        canonical_center = np.array(sphere_pt, dtype=np.float64)
        if vertices_3d:
            verts_np = np.array(vertices_3d, dtype=np.float64)
            if verts_np.size:
                centroid = np.mean(verts_np, axis=0)
                canonical_center = _normalize_vec(centroid)

        address = addresses.get(str(face_id_local))
        ring_index = int(address.ring_index) if address is not None else -1
        position_in_ring = int(address.position_in_ring) if address is not None else -1
        detail_index = int(address.detail_index) if address is not None else -1

        results.append(
            {
                "id": face_id_local,
            "detail_index": detail_index,
                "center_3d": [round(float(v), 9) for v in sphere_pt],
                "canonical_center_3d": [round(float(v), 9) for v in canonical_center],
                "vertices_3d": vertices_3d,
                "sides": int(len(vertices_3d) or face.vertex_count()),
                "ring_index": int(ring_index),
                "position_in_ring": int(position_in_ring),
            }
        )

    return results


def compute_all_detail_centers(
    globe_grid: PolyGrid,
    detail_rings: int = 4,
) -> Dict[str, List[Dict[str, Any]]]:
    """Compute detail-cell 3D centres for every globe tile.

    Parameters
    ----------
    globe_grid:
        The complete globe grid.
    detail_rings:
        Ring count for the sub-tile hex/pent grids.

    Returns
    -------
    ``{face_id: [{id, center_3d}, ...]}`` for every face in *globe_grid*.
    """
    result: Dict[str, List[Dict[str, Any]]] = {}
    face_ids = sorted(globe_grid.faces.keys())
    for i, face_id in enumerate(face_ids):
        if i % 10 == 0:
            LOGGER.info(
                "Computing detail centres: %d/%d tiles", i, len(face_ids)
            )
        result[face_id] = compute_detail_cell_centers_3d(
            globe_grid, face_id, detail_rings=detail_rings
        )
    return result


def build_slug_keyed_detail_centers(
    globe_grid: PolyGrid,
    detail_rings: int = 4,
) -> Dict[str, List[Dict[str, Any]]]:
    """Like ``compute_all_detail_centers`` but keys by Goldberg tile slug.

    Returns ``{tile_slug: [{id, center_3d}, ...]}`` where *tile_slug* matches
    the canonical IDs used by playground's picking system (e.g.
    ``"3:f4:3-0-0"``).  Falls back to raw face IDs if the grid does not
    expose a slug lookup.
    """
    raw = compute_all_detail_centers(globe_grid, detail_rings=detail_rings)

    # Build slug lookup if available (GlobeGrid exposes build_slug_lookup).
    slug_lookup: Dict[str, str] = {}
    try:
        slug_lookup = globe_grid.build_slug_lookup()  # type: ignore[attr-defined]
    except AttributeError:
        LOGGER.debug("Globe grid has no build_slug_lookup — using raw face IDs")

    result: Dict[str, List[Dict[str, Any]]] = {}
    for face_id, cells in raw.items():
        key = slug_lookup.get(face_id, face_id)
        result[key] = cells
    return normalize_detail_cells_tiles(result)
