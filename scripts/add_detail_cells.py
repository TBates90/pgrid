#!/usr/bin/env python3
"""Add (or regenerate) ``detail_cells.json`` for an existing pgrid export.

Reads ``metadata.json`` from the export directory to determine frequency
and detail_rings, rebuilds the globe topological grid (no terrain, no
textures — fast), computes sub-tile detail cell 3-D centres keyed by
Goldberg tile slug, and writes ``detail_cells.json`` alongside the
existing export files.

Usage
-----
::

    # Regenerate for a single export directory:
    python scripts/add_detail_cells.py exports/f3-d4

    # Batch-regenerate every export sub-directory:
    python scripts/add_detail_cells.py exports/

"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _rebuild_globe_grid(frequency: int):
    """Rebuild the globe topology grid (no terrain needed for cell centres)."""
    from polygrid.globe import build_globe_grid

    return build_globe_grid(frequency)


def _write_detail_cells(export_dir: Path) -> bool:
    """Compute and write detail_cells.json for one export directory.

    Returns True on success, False on failure.
    """
    metadata_path = export_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"  SKIP {export_dir}: metadata.json not found")
        return False

    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"  SKIP {export_dir}: metadata.json parse error — {exc}")
        return False

    frequency = int(metadata.get("frequency", 3))
    detail_rings = int(metadata.get("detail_rings", 4))

    print(f"Processing {export_dir.name}: freq={frequency}, rings={detail_rings}")

    try:
        grid = _rebuild_globe_grid(frequency)
    except Exception as exc:
        print(f"  ERROR building globe grid: {exc}")
        return False

    try:
        from polygrid.rendering.detail_centers import build_slug_keyed_detail_centers

        t0 = time.perf_counter()
        all_centres = build_slug_keyed_detail_centers(grid, detail_rings=detail_rings)
        elapsed = time.perf_counter() - t0
    except Exception as exc:
        print(f"  ERROR computing detail centres: {exc}")
        return False

    detail_cells = {
        "metadata": {"frequency": frequency, "detail_rings": detail_rings},
        "tiles": all_centres,
    }

    output_path = export_dir / "detail_cells.json"
    output_path.write_text(json.dumps(detail_cells, indent=2))

    sample_count = len(next(iter(all_centres.values()), []))
    print(
        f"  → {output_path} "
        f"({len(all_centres)} tiles × ~{sample_count} cells, {elapsed:.2f}s)"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate detail_cells.json for existing pgrid exports."
    )
    parser.add_argument(
        "dirs",
        nargs="+",
        metavar="DIR",
        help=(
            "Export directory (or parent directory containing multiple exports). "
            "If a directory contains subdirectories with metadata.json files, "
            "each subdirectory is processed."
        ),
    )
    args = parser.parse_args()

    targets: list[Path] = []
    for raw in args.dirs:
        p = Path(raw)
        if not p.exists():
            print(f"WARNING: {p} does not exist — skipping")
            continue
        # If it has metadata.json directly, treat it as an export dir
        if (p / "metadata.json").exists():
            targets.append(p)
        else:
            # Treat as a parent directory; collect immediate children with metadata.json
            children = sorted(c for c in p.iterdir() if c.is_dir() and (c / "metadata.json").exists())
            if children:
                targets.extend(children)
            else:
                print(f"WARNING: {p} has no metadata.json and no children with it — skipping")

    if not targets:
        print("No valid export directories found.")
        sys.exit(1)

    success = 0
    for target in targets:
        if _write_detail_cells(target):
            success += 1

    print(f"\nDone: {success}/{len(targets)} exports updated.")


if __name__ == "__main__":
    main()
