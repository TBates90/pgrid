#!/usr/bin/env python3
"""Demo: build a pent+hex assembly and partition it into terrain regions.

Produces a 4-panel PNG:
  1. Exploded components with edge IDs and stitch arrows
  2. Stitched composite
  3. Stitched with colour-coded regions
  4. Unstitched components with regions overlaid

Usage:
    python scripts/demo_regions.py                        # default output
    python scripts/demo_regions.py --rings 3 --regions 6  # customise
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polygrid import (
    pent_hex_assembly,
    partition_voronoi,
    regions_to_overlay,
    validate_region_map,
    render_assembly_panels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Terrain region partitioning demo")
    parser.add_argument("--rings", type=int, default=2, help="Grid rings (default: 2)")
    parser.add_argument("--regions", type=int, default=5, help="Number of regions (default: 5)")
    parser.add_argument("--out", type=str, default="exports/regions.png", help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI (default: 150)")
    args = parser.parse_args()

    print(f"Building pent+hex assembly with {args.rings} rings …")
    plan = pent_hex_assembly(rings=args.rings)
    composite = plan.build()
    grid = composite.merged
    print(f"  {len(grid.faces)} faces, {len(grid.vertices)} vertices")

    # Pick well-spaced seed faces
    face_ids = sorted(grid.faces.keys())
    n = min(args.regions, len(face_ids))
    step = len(face_ids) // n
    seeds = [face_ids[i * step] for i in range(n)]

    region_names = [f"region_{i}" for i in range(n)]

    print(f"Partitioning into {n} Voronoi regions …")
    rm = partition_voronoi(grid, seeds, names=region_names)

    result = validate_region_map(rm)
    if not result.ok:
        print("Validation errors:")
        for e in result.errors:
            print(f"  • {e}")
        raise SystemExit(1)

    for r in rm.regions:
        print(f"  {r.name}: {r.size} faces")

    overlay = regions_to_overlay(rm, grid)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Rendering to {out} …")
    render_assembly_panels(plan, str(out), overlay=overlay, dpi=args.dpi)
    print("Done ✓")


if __name__ == "__main__":
    main()
