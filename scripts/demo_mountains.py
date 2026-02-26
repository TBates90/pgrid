#!/usr/bin/env python3
"""Demo: generate mountain terrain on a pent-hex assembly and render it.

Usage
-----
    python scripts/demo_mountains.py --rings 3 --preset mountain_range --out exports/mountains.png
    python scripts/demo_mountains.py --rings 2 --preset alpine_peaks --out exports/alpine.png

Available presets: mountain_range, alpine_peaks, rolling_hills, mesa_plateau
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the path when run as a script
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
from polygrid.terrain_render import render_terrain
from polygrid.tile_data import FieldDef, TileSchema, TileDataStore

PRESETS = {
    "mountain_range": MOUNTAIN_RANGE,
    "alpine_peaks": ALPINE_PEAKS,
    "rolling_hills": ROLLING_HILLS,
    "mesa_plateau": MESA_PLATEAU,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mountain terrain demo")
    parser.add_argument("--rings", type=int, default=3, help="Assembly ring count")
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="mountain_range",
        help="Mountain preset",
    )
    parser.add_argument("--out", default="exports/mountains.png", help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI")
    parser.add_argument("--ramp", default="satellite", help="Colour ramp: terrain, greyscale, satellite")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Building pent-hex assembly (rings={args.rings})…")
    plan = pent_hex_assembly(rings=args.rings)
    composite = plan.build()
    grid = composite.merged

    print(f"Generating mountains (preset={args.preset}, seed={args.seed})…")
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid, schema=schema)
    store.initialise_all()

    config = PRESETS[args.preset]
    # Override seed if specified
    from dataclasses import replace
    config = replace(config, seed=args.seed)

    generate_mountains(grid, store, config)

    print(f"Rendering terrain (ramp={args.ramp})…")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    render_terrain(
        grid,
        store,
        args.out,
        ramp=args.ramp,
        dpi=args.dpi,
        title=f"Mountains — {args.preset} (rings={args.rings})",
    )

    print(f"Done ✓  →  {args.out}")


if __name__ == "__main__":
    main()
