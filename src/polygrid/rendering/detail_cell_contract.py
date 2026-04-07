"""Detail-cell payload contract normalization for export pipelines.

This module canonicalizes detail-cell tile maps produced by pgrid so runtime
consumers receive stable IDs, finite unit-sphere vectors, and contiguous
detail indices.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class DetailCellNormalizationReport:
    """Summary of producer-side detail-cell contract normalization."""

    tiles_seen: int
    tiles_emitted: int
    tiles_dropped: int
    invalid_tile_entries: int
    cells_seen: int
    cells_emitted: int
    cells_dropped: int
    repaired_index_tiles: int
    repaired_index_cells: int

    def to_dict(self) -> dict[str, int]:
        return {
            "tiles_seen": int(self.tiles_seen),
            "tiles_emitted": int(self.tiles_emitted),
            "tiles_dropped": int(self.tiles_dropped),
            "invalid_tile_entries": int(self.invalid_tile_entries),
            "cells_seen": int(self.cells_seen),
            "cells_emitted": int(self.cells_emitted),
            "cells_dropped": int(self.cells_dropped),
            "repaired_index_tiles": int(self.repaired_index_tiles),
            "repaired_index_cells": int(self.repaired_index_cells),
        }


def normalize_detail_cells_tiles(
    detail_cells: Mapping[str, Any] | None,
    *,
    strict: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Return normalized detail-cell tiles keyed by canonical tile slug.

    Accepted input shape: ``{tile_slug: [cell, ...]}``.
    Invalid tiles/cells are dropped.
    """

    normalized_tiles, _report = normalize_detail_cells_tiles_with_report(
        detail_cells,
        strict=strict,
    )
    return normalized_tiles


def normalize_detail_cells_tiles_with_report(
    detail_cells: Mapping[str, Any] | None,
    *,
    strict: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], DetailCellNormalizationReport]:
    """Normalize detail-cell tiles and return a repair/drop summary report."""

    normalized_tiles: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(detail_cells, Mapping):
        return normalized_tiles, DetailCellNormalizationReport(
            tiles_seen=0,
            tiles_emitted=0,
            tiles_dropped=0,
            invalid_tile_entries=0,
            cells_seen=0,
            cells_emitted=0,
            cells_dropped=0,
            repaired_index_tiles=0,
            repaired_index_cells=0,
        )

    tiles_seen = 0
    invalid_tile_entries = 0
    cells_seen = 0
    cells_dropped = 0
    repaired_index_tiles = 0
    repaired_index_cells = 0

    for tile_id, raw_cells in detail_cells.items():
        tiles_seen += 1
        tile_token = _normalize_token(tile_id)
        if tile_token is None:
            continue
        if not isinstance(raw_cells, Sequence) or isinstance(raw_cells, (str, bytes, bytearray)):
            invalid_tile_entries += 1
            continue

        normalized_cells: list[dict[str, Any]] = []
        for cell in raw_cells:
            cells_seen += 1
            if not isinstance(cell, Mapping):
                cells_dropped += 1
                continue
            normalized = _normalize_cell(cell)
            if normalized is not None:
                normalized_cells.append(normalized)
            else:
                cells_dropped += 1

        if not normalized_cells:
            continue

        repaired_for_tile = _ensure_contiguous_detail_indices(normalized_cells)
        if repaired_for_tile > 0:
            repaired_index_tiles += 1
            repaired_index_cells += repaired_for_tile
        normalized_tiles[tile_token] = normalized_cells

    tiles_emitted = len(normalized_tiles)
    cells_emitted = sum(len(cells) for cells in normalized_tiles.values())
    report = DetailCellNormalizationReport(
        tiles_seen=tiles_seen,
        tiles_emitted=tiles_emitted,
        tiles_dropped=max(0, tiles_seen - tiles_emitted),
        invalid_tile_entries=invalid_tile_entries,
        cells_seen=cells_seen,
        cells_emitted=cells_emitted,
        cells_dropped=cells_dropped,
        repaired_index_tiles=repaired_index_tiles,
        repaired_index_cells=repaired_index_cells,
    )
    if strict and _report_has_adjustments(report):
        raise ValueError(f"detail_cells payload violates strict contract: {report.to_dict()}")
    return normalized_tiles, report


def _report_has_adjustments(report: DetailCellNormalizationReport) -> bool:
    return any(
        (
            report.tiles_dropped,
            report.invalid_tile_entries,
            report.cells_dropped,
            report.repaired_index_tiles,
            report.repaired_index_cells,
        )
    )


def _normalize_cell(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    local_id = _normalize_token(raw.get("id"), lowercase=False)
    if local_id is None:
        return None

    canonical_center = _normalize_vec3(raw.get("canonical_center_3d"))
    center = _normalize_vec3(raw.get("center_3d"))
    if canonical_center is None and center is None:
        return None
    if canonical_center is None:
        canonical_center = center
    if center is None:
        center = canonical_center

    vertices = _normalize_vertices(raw.get("vertices_3d"))
    sides = _normalize_sides(raw.get("sides"), fallback=len(vertices) if vertices else None)

    cell: dict[str, Any] = {
        "id": local_id,
        "center_3d": list(center),
        "canonical_center_3d": list(canonical_center),
        "sides": int(sides),
    }

    detail_index = _normalize_positive_int(raw.get("detail_index"))
    if detail_index is not None:
        cell["detail_index"] = int(detail_index)

    ring_index = _normalize_int(raw.get("ring_index"))
    if ring_index is not None:
        cell["ring_index"] = int(ring_index)

    position_in_ring = _normalize_int(raw.get("position_in_ring"))
    if position_in_ring is not None:
        cell["position_in_ring"] = int(position_in_ring)

    if vertices:
        cell["vertices_3d"] = [list(vertex) for vertex in vertices]

    return cell


def _ensure_contiguous_detail_indices(cells: list[dict[str, Any]]) -> int:
    assigned: set[int] = set()
    valid = True

    for cell in cells:
        raw_index = cell.get("detail_index")
        if not isinstance(raw_index, int) or raw_index <= 0 or raw_index in assigned:
            valid = False
            break
        assigned.add(raw_index)

    if valid and assigned == set(range(1, len(cells) + 1)):
        return 0

    changed = 0
    for idx, cell in enumerate(cells, start=1):
        if cell.get("detail_index") != idx:
            changed += 1
        cell["detail_index"] = idx
    return changed


def _normalize_vertices(value: Any) -> list[tuple[float, float, float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    vertices: list[tuple[float, float, float]] = []
    for item in value:
        normalized = _normalize_vec3(item)
        if normalized is not None:
            vertices.append(normalized)
    if len(vertices) < 3:
        return []
    return vertices


def _normalize_vec3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) < 3:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
        z = float(value[2])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    mag = math.sqrt(x * x + y * y + z * z)
    if mag <= 1e-9:
        return None
    return (x / mag, y / mag, z / mag)


def _normalize_sides(value: Any, *, fallback: int | None) -> int:
    parsed = _normalize_int(value)
    if parsed is not None and parsed >= 3:
        return int(parsed)
    if fallback is not None and int(fallback) >= 3:
        return int(fallback)
    return 6


def _normalize_positive_int(value: Any) -> int | None:
    parsed = _normalize_int(value)
    if parsed is None or parsed <= 0:
        return None
    return int(parsed)


def _normalize_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_token(value: Any, *, lowercase: bool = True) -> str | None:
    token = str(value).strip() if value not in (None, "") else ""
    if not token:
        return None
    return token.lower() if lowercase else token
