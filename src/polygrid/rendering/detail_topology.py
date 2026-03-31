"""Topology helpers for detail-cell export metadata.

This module centralizes deterministic ring/address indexing for detail grids so
export and runtime consumers can share one canonical ordering contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from ..core.algorithms import get_face_adjacency, ring_faces
from ..core.geometry import face_center, grid_center
from ..core.polygrid import PolyGrid


@dataclass(frozen=True)
class DetailCellAddress:
    """Canonical address metadata for one detail cell."""

    id: str
    detail_index: int
    ring_index: int
    position_in_ring: int


def build_detail_ring_positions(
    grid: PolyGrid,
    *,
    max_depth: int,
) -> dict[str, tuple[int, int]]:
    """Return ``{face_id: (ring_index, position_in_ring)}`` for a detail grid.

    The center face is chosen as the face whose 2D centroid is nearest to the
    grid centroid. Faces in each ring are ordered by polar angle around the
    center face centroid for deterministic clockwise indexing.
    """

    grid_c = grid_center(grid.vertices)
    center_face_id: str | None = None
    center_face_xy: tuple[float, float] | None = None
    best_dist = float("inf")

    for face_id, face in grid.faces.items():
        xy = face_center(grid.vertices, face)
        if xy is None:
            continue
        dist = float((xy[0] - grid_c[0]) ** 2 + (xy[1] - grid_c[1]) ** 2)
        if dist < best_dist:
            best_dist = dist
            center_face_id = str(face_id)
            center_face_xy = (float(xy[0]), float(xy[1]))

    if center_face_id is None or center_face_xy is None:
        return {}

    adjacency = get_face_adjacency(grid)
    rings = ring_faces(adjacency, center_face_id, max_depth=max_depth)
    mapping: dict[str, tuple[int, int]] = {}

    for ring_index, face_ids in sorted(rings.items()):
        if ring_index == 0:
            mapping[center_face_id] = (0, 0)
            continue
        sortable: list[tuple[float, str]] = []
        for face_id in face_ids:
            face = grid.faces.get(face_id)
            if face is None:
                continue
            xy = face_center(grid.vertices, face)
            if xy is None:
                continue
            dx = float(xy[0] - center_face_xy[0])
            dy = float(xy[1] - center_face_xy[1])
            angle = float(math.atan2(dy, dx))
            sortable.append((angle, str(face_id)))
        sortable.sort(key=lambda item: item[0])
        for position_in_ring, (_, face_id) in enumerate(sortable):
            mapping[face_id] = (int(ring_index), int(position_in_ring))

    return mapping


def build_detail_cell_addresses(
    face_ids: Iterable[str],
    ring_positions: dict[str, tuple[int, int]],
) -> dict[str, DetailCellAddress]:
    """Build deterministic addresses for detail-cell IDs.

    The canonical ordering is by:
    1) ``ring_index`` ascending
    2) ``position_in_ring`` ascending
    3) numeric suffix in local ID (e.g. ``f12``)
    4) raw local ID string

    ``detail_index`` is 1-based to reserve 0 as a sentinel value for packed
    selection IDs in downstream renderers.
    """

    local_ids = [str(face_id) for face_id in face_ids]

    def _numeric_suffix(token: str) -> int:
        token_l = token.lower()
        if token_l.startswith("f") and token_l[1:].isdigit():
            return int(token_l[1:])
        return 1_000_000

    def _sort_key(token: str) -> tuple[int, int, int, str]:
        ring_index, position = ring_positions.get(token, (-1, -1))
        ring_sort = ring_index if ring_index >= 0 else 1_000_000
        pos_sort = position if position >= 0 else 1_000_000
        return (ring_sort, pos_sort, _numeric_suffix(token), token)

    ordered = sorted(local_ids, key=_sort_key)
    result: dict[str, DetailCellAddress] = {}
    for idx, token in enumerate(ordered, start=1):
        ring_index, position = ring_positions.get(token, (-1, -1))
        result[token] = DetailCellAddress(
            id=token,
            detail_index=int(idx),
            ring_index=int(ring_index),
            position_in_ring=int(position),
        )
    return result
