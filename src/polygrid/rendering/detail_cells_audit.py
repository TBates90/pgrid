"""Audit helpers for ``detail_cells.json`` canonical metadata coverage.

These checks are intended for migration tracking: they report where runtime
decoders would still need legacy local-ID fallback instead of canonical
``detail_index`` lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TileAuditResult:
    """Per-tile canonical metadata audit summary."""

    tile_id: str
    total_cells: int
    missing_detail_index: int
    invalid_detail_index: int
    non_contiguous_detail_index: bool
    missing_ring_index: int
    missing_position_in_ring: int

    @property
    def needs_fallback(self) -> bool:
        """Return True when canonical index decode is not fully safe."""

        return (
            self.missing_detail_index > 0
            or self.invalid_detail_index > 0
            or self.non_contiguous_detail_index
        )


def _parse_detail_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _audit_tile(tile_id: str, cells: Any) -> TileAuditResult:
    if not isinstance(cells, list):
        return TileAuditResult(
            tile_id=str(tile_id),
            total_cells=0,
            missing_detail_index=0,
            invalid_detail_index=0,
            non_contiguous_detail_index=False,
            missing_ring_index=0,
            missing_position_in_ring=0,
        )

    total = len(cells)
    missing_detail_index = 0
    invalid_detail_index = 0
    missing_ring_index = 0
    missing_position = 0
    valid_indices: list[int] = []

    for cell in cells:
        if not isinstance(cell, dict):
            invalid_detail_index += 1
            missing_ring_index += 1
            missing_position += 1
            continue

        if "detail_index" not in cell:
            missing_detail_index += 1
        else:
            parsed = _parse_detail_index(cell.get("detail_index"))
            if parsed is None or parsed < 1:
                invalid_detail_index += 1
            else:
                valid_indices.append(parsed)

        if "ring_index" not in cell:
            missing_ring_index += 1
        if "position_in_ring" not in cell:
            missing_position += 1

    non_contiguous = False
    if total > 0 and len(valid_indices) == total:
        expected = list(range(1, total + 1))
        non_contiguous = sorted(valid_indices) != expected

    return TileAuditResult(
        tile_id=str(tile_id),
        total_cells=int(total),
        missing_detail_index=int(missing_detail_index),
        invalid_detail_index=int(invalid_detail_index),
        non_contiguous_detail_index=bool(non_contiguous),
        missing_ring_index=int(missing_ring_index),
        missing_position_in_ring=int(missing_position),
    )


def audit_detail_cells_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return canonical-metadata coverage stats for one payload."""

    tiles = payload.get("tiles", {})
    if not isinstance(tiles, dict):
        tiles = {}

    tile_results = [_audit_tile(str(tile_id), cells) for tile_id, cells in tiles.items()]

    total_tiles = len(tile_results)
    total_cells = sum(item.total_cells for item in tile_results)
    fallback_tiles = sum(1 for item in tile_results if item.needs_fallback)
    fallback_cells = sum(
        item.missing_detail_index + item.invalid_detail_index
        for item in tile_results
    )
    ring_missing_cells = sum(item.missing_ring_index for item in tile_results)
    position_missing_cells = sum(item.missing_position_in_ring for item in tile_results)
    non_contiguous_tiles = sum(1 for item in tile_results if item.non_contiguous_detail_index)

    return {
        "tiles": total_tiles,
        "cells": total_cells,
        "fallback_tiles": fallback_tiles,
        "fallback_cells": fallback_cells,
        "non_contiguous_tiles": non_contiguous_tiles,
        "missing_ring_index_cells": ring_missing_cells,
        "missing_position_in_ring_cells": position_missing_cells,
        "canonical_index_coverage": (
            1.0 if total_cells == 0 else max(0.0, float(total_cells - fallback_cells) / float(total_cells))
        ),
        "tile_results": [
            {
                "tile_id": item.tile_id,
                "total_cells": item.total_cells,
                "missing_detail_index": item.missing_detail_index,
                "invalid_detail_index": item.invalid_detail_index,
                "non_contiguous_detail_index": item.non_contiguous_detail_index,
                "missing_ring_index": item.missing_ring_index,
                "missing_position_in_ring": item.missing_position_in_ring,
                "needs_fallback": item.needs_fallback,
            }
            for item in tile_results
        ],
    }
