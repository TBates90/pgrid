#!/usr/bin/env python3
"""Demo: full terrain pipeline — mountains + rivers on a pent-hex assembly.

Usage
-----
    python scripts/demo_terrain.py --rings 3 --preset mountain_range --out exports/terrain.png
    python scripts/demo_terrain.py --rings 3 --preset alpine_peaks --seed 7 --out exports/terrain2.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polygrid import pent_hex_assembly
from polygrid.mountains import (
    ALPINE_PEAKS,
    MESA_PLATEAU,
    MOUNTAIN_RANGE,
    ROLLING_HILLS,
    MountainConfig,
    generate_mountains,
)
from polygrid.rivers import (
    RiverConfig,
    carve_river_valleys,
    generate_rivers,
    river_to_overlay,
)
from polygrid.terrain_render import (
    elevation_to_overlay,
    hillshade,
    render_terrain,
)
from polygrid.tile_data import FieldDef, TileSchema, TileDataStore

PRESETS = {
    "mountain_range": MOUNTAIN_RANGE,
    "alpine_peaks": ALPINE_PEAKS,
    "rolling_hills": ROLLING_HILLS,
    "mesa_plateau": MESA_PLATEAU,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Full terrain demo: mountains + rivers")
    parser.add_argument("--rings", type=int, default=3, help="Assembly ring count")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="mountain_range")
    parser.add_argument("--out", default="exports/terrain.png")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--ramp", default="satellite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-acc", type=int, default=4, help="River min flow accumulation")
    parser.add_argument("--carve", type=float, default=0.06, help="River carve depth")
    args = parser.parse_args()

    # ── 1. Build grid ───────────────────────────────────────────────
    print(f"Building pent-hex assembly (rings={args.rings})…")
    plan = pent_hex_assembly(rings=args.rings)
    composite = plan.build()
    grid = composite.merged

    # ── 2. Generate mountains ───────────────────────────────────────
    print(f"Generating mountains (preset={args.preset})…")
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid, schema=schema)
    store.initialise_all()

    config = replace(PRESETS[args.preset], seed=args.seed)
    generate_mountains(grid, store, config)

    # ── 3. Generate rivers ──────────────────────────────────────────
    print("Generating rivers…")
    river_cfg = RiverConfig(
        min_accumulation=args.min_acc,
        min_length=3,
        carve_depth=args.carve,
        seed=args.seed,
    )
    network = generate_rivers(grid, store, river_cfg)
    print(f"  → {len(network)} river segments, {len(network.all_river_face_ids())} river faces")

    # ── 4. Carve river valleys ──────────────────────────────────────
    if len(network) > 0:
        print("Carving river valleys…")
        carve_river_valleys(grid, store, network, carve_depth=args.carve)

    # ── 5. Render ───────────────────────────────────────────────────
    print(f"Rendering terrain…")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # We render terrain (elevation + hillshade), then overlay rivers on top
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    shade = hillshade(grid, store)
    elev_overlay = elevation_to_overlay(grid, store, ramp=args.ramp, shade=shade)
    riv_overlay = river_to_overlay(grid, network) if len(network) > 0 else None

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw elevation
    for region in elev_overlay.regions:
        if len(region.points) < 3:
            continue
        fid = region.source_vertex_id
        color = elev_overlay.metadata.get(f"color_{fid}", (0.5, 0.5, 0.5))
        poly = Polygon(region.points, closed=True,
                        facecolor=color, edgecolor=color,
                        linewidth=0.3, zorder=1)
        ax.add_patch(poly)

    # Draw rivers on top
    if riv_overlay:
        for region in riv_overlay.regions:
            if len(region.points) < 3:
                continue
            fid = region.source_vertex_id
            color = riv_overlay.metadata.get(f"color_{fid}", (0.2, 0.4, 0.8))
            poly = Polygon(region.points, closed=True,
                            facecolor=color, edgecolor=color,
                            linewidth=0.3, alpha=0.85, zorder=2)
            ax.add_patch(poly)

    ax.autoscale_view()
    ax.set_title(
        f"Terrain — {args.preset} + rivers (rings={args.rings})",
        fontsize=14,
    )
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"Done ✓  →  {args.out}")


if __name__ == "__main__":
    main()
