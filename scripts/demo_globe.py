#!/usr/bin/env python3
"""Demo: generate terrain on a Goldberg polyhedron and render it.

Usage:
    python scripts/demo_globe.py [--frequency N] [--preset PRESET] [--out DIR]

Presets: mountain_range, alpine_peaks, rolling_hills, mesa_plateau
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is on the path when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polygrid.globe import build_globe_grid
from polygrid.mountains import (
    MountainConfig,
    generate_mountains,
    MOUNTAIN_RANGE,
    ALPINE_PEAKS,
    ROLLING_HILLS,
    MESA_PLATEAU,
)
from polygrid.tile_data import FieldDef, TileSchema, TileDataStore
from polygrid.globe_render import (
    render_globe_flat,
    render_globe_3d,
    globe_to_tile_colours,
)

PRESETS = {
    "mountain_range": MOUNTAIN_RANGE,
    "alpine_peaks": ALPINE_PEAKS,
    "rolling_hills": ROLLING_HILLS,
    "mesa_plateau": MESA_PLATEAU,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Globe terrain demo")
    parser.add_argument("--frequency", type=int, default=3, help="Goldberg frequency (default: 3)")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="mountain_range")
    parser.add_argument("--out", type=str, default="exports", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    freq = args.frequency
    print(f"Building globe grid (frequency={freq})...")
    grid = build_globe_grid(freq)
    print(f"  → {len(grid.faces)} tiles ({grid.metadata['pentagon_count']} pent, {grid.metadata['hexagon_count']} hex)")

    # Set up tile data
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)

    # Get mountain config from preset, override seed
    config = PRESETS[args.preset]
    from dataclasses import replace
    config = replace(config, seed=args.seed)

    print(f"Generating terrain (preset={args.preset}, seed={args.seed})...")
    generate_mountains(grid, store, config)

    # Stats
    elevations = [store.get(fid, "elevation") for fid in grid.faces]
    print(f"  → elevation range: [{min(elevations):.3f}, {max(elevations):.3f}]")

    # Render flat map
    flat_path = out_dir / f"globe_f{freq}_flat.png"
    print(f"Rendering flat projection → {flat_path}")
    render_globe_flat(grid, store, flat_path)

    # Render 3D
    d3_path = out_dir / f"globe_f{freq}_3d.png"
    print(f"Rendering 3D polyhedron → {d3_path}")
    render_globe_3d(grid, store, d3_path)

    # Export colour JSON
    colours_path = out_dir / f"globe_f{freq}_colours.json"
    print(f"Exporting tile colours → {colours_path}")
    tile_colours = globe_to_tile_colours(grid, store)
    colours_path.write_text(json.dumps(tile_colours, indent=2))

    print("Done!")


if __name__ == "__main__":
    main()
