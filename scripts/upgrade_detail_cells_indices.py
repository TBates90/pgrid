#!/usr/bin/env python3
"""Backfill canonical ``detail_index`` fields in legacy detail_cells exports.

This script updates existing ``detail_cells.json`` files in-place by assigning
deterministic 1-based ``detail_index`` values per tile when missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _discover_targets(raw_dirs: list[str]) -> list[Path]:
    targets: list[Path] = []
    for raw in raw_dirs:
        path = Path(raw)
        if not path.exists():
            print(f"WARNING: {path} does not exist - skipping")
            continue
        if (path / "detail_cells.json").exists():
            targets.append(path)
            continue
        children = sorted(
            child
            for child in path.iterdir()
            if child.is_dir() and (child / "detail_cells.json").exists()
        )
        if children:
            targets.extend(children)
        else:
            print(
                f"WARNING: {path} has no detail_cells.json and no children with it - skipping"
            )
    return targets


def _numeric_suffix(token: str) -> int:
    token_l = token.lower()
    if token_l.startswith("f") and token_l[1:].isdigit():
        return int(token_l[1:])
    return 1_000_000


def _upgrade_export(export_dir: Path) -> tuple[int, int]:
    path = export_dir / "detail_cells.json"
    payload = json.loads(path.read_text())
    tiles = payload.get("tiles", {})
    if not isinstance(tiles, dict):
        return 0, 0

    tiles_changed = 0
    cells_changed = 0

    for tile_id, cells in tiles.items():
        if not isinstance(cells, list):
            continue

        # Keep stable deterministic order: ring/position if present, then ID suffix.
        def _sort_key(cell: object) -> tuple[int, int, int, str]:
            if not isinstance(cell, dict):
                return (1_000_000, 1_000_000, 1_000_000, "")
            ring = cell.get("ring_index")
            position = cell.get("position_in_ring")
            ring_i = int(ring) if isinstance(ring, int) and ring >= 0 else 1_000_000
            pos_i = int(position) if isinstance(position, int) and position >= 0 else 1_000_000
            local_id = str(cell.get("id", ""))
            return (ring_i, pos_i, _numeric_suffix(local_id), local_id)

        ordered = sorted(cells, key=_sort_key)
        tile_changed = False
        for idx, cell in enumerate(ordered, start=1):
            if not isinstance(cell, dict):
                continue
            current = cell.get("detail_index")
            if isinstance(current, int) and current == idx:
                continue
            cell["detail_index"] = idx
            tile_changed = True
            cells_changed += 1

        if tile_changed:
            tiles_changed += 1

    if tiles_changed > 0:
        path.write_text(json.dumps(payload, indent=2))

    return tiles_changed, cells_changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill detail_index for legacy detail_cells.json exports."
    )
    parser.add_argument(
        "dirs",
        nargs="+",
        metavar="DIR",
        help=(
            "Export directory (or parent directory containing multiple exports). "
            "Children with detail_cells.json are auto-discovered."
        ),
    )
    args = parser.parse_args()

    targets = _discover_targets(args.dirs)
    if not targets:
        print("No valid export directories found.")
        sys.exit(1)

    changed_exports = 0
    changed_tiles = 0
    changed_cells = 0

    for target in targets:
        try:
            tile_count, cell_count = _upgrade_export(target)
        except Exception as exc:
            print(f"ERROR {target.name}: {exc}")
            continue
        if tile_count > 0:
            changed_exports += 1
            changed_tiles += tile_count
            changed_cells += cell_count
        print(
            f"{target.name}: tiles_changed={tile_count} cells_changed={cell_count}"
        )

    print(
        "\n"
        f"Done: exports_changed={changed_exports}/{len(targets)} "
        f"tiles_changed={changed_tiles} cells_changed={changed_cells}"
    )


if __name__ == "__main__":
    main()
