#!/usr/bin/env python3
"""Generate the 4-panel assembly demo image.

Usage:
    python scripts/demo_assembly.py [--rings N] [--out PATH] [--sections N]

Produces:
    Panel 1: Exploded components with macro-edge IDs + stitch arrows
    Panel 2: Stitched composite
    Panel 3: Stitched + partition overlay (coloured sections)
    Panel 4: Unstitched with partition preserved
"""

from __future__ import annotations

import argparse
from pathlib import Path

from polygrid.assembly import pent_hex_assembly
from polygrid.transforms import apply_voronoi, apply_partition
from polygrid.visualize import render_assembly_panels, render_single_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Assembly visualisation demo")
    parser.add_argument("--rings", type=int, default=3, help="Rings per component grid")
    parser.add_argument("--out", default="exports/assembly_demo.png", help="Output path for 4-panel image")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--sections", type=int, default=8, help="Number of partition sections")
    args = parser.parse_args()

    print(f"Building pent+hex assembly with {args.rings} rings …")
    plan = pent_hex_assembly(rings=args.rings)

    print(f"Components: {list(plan.components.keys())}")
    for name, grid in plan.components.items():
        print(f"  {name}: {len(grid.faces)} faces, {len(grid.macro_edges)} macro-edges")

    print("Stitching …")
    composite = plan.build()
    print(f"Merged grid: {len(composite.merged.faces)} faces, "
          f"{len(composite.merged.vertices)} vertices, "
          f"{len(composite.merged.edges)} edges")

    print(f"Computing partition overlay ({args.sections} sections) …")
    overlay = apply_partition(composite.merged, n_sections=args.sections)
    print(f"Partition: {overlay.metadata['n_sections']} sections, "
          f"{len(overlay.regions)} face regions")

    print(f"Rendering 4-panel image to {args.out} …")
    render_assembly_panels(plan, args.out, overlay=overlay, dpi=args.dpi,
                           figsize=(28, 7))

    # Also render individual panels at higher quality
    out_dir = Path(args.out).parent
    render_single_panel(composite.merged, out_dir / "stitched.png",
                        title="Stitched composite")
    render_single_panel(composite.merged, out_dir / "stitched_partition.png",
                        overlay=overlay, title=f"Stitched + partition ({args.sections} sections)")

    # Also render Voronoi for reference
    voronoi = apply_voronoi(composite.merged)
    render_single_panel(composite.merged, out_dir / "stitched_voronoi.png",
                        overlay=voronoi, title="Stitched + Voronoi")

    print("Done ✓")


if __name__ == "__main__":
    main()
