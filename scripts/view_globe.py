#!/usr/bin/env python3
"""Interactive terrain globe viewer — loads a globe export JSON and
renders it as a terrain-coloured Goldberg polyhedron.

Usage
-----
::

    # Generate a globe export first:
    python scripts/demo_globe_3d.py --preset mountain_range --out exports/

    # Then view it interactively:
    python scripts/view_globe.py exports/globe_f3_mountain_range_colours.json

    # Or generate + view in one go:
    python scripts/view_globe.py --frequency 4 --preset alpine_peaks

Requires ``pyglet`` and an OpenGL 3.3+ capable display.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the package is importable from the scripts/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _generate_payload(frequency: int, preset: str, seed: int):
    """Generate a globe export payload inline."""
    from polygrid.globe import build_globe_grid
    from polygrid.mountains import MountainConfig, generate_mountains
    from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
    from polygrid.globe_export import export_globe_payload

    grid = build_globe_grid(frequency)
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)

    presets = {
        "mountain_range": MountainConfig(
            seed=seed, ridge_frequency=2.0, ridge_octaves=4,
            peak_elevation=1.0, base_elevation=0.0,
        ),
        "alpine_peaks": MountainConfig(
            seed=seed, ridge_frequency=3.0, ridge_octaves=5,
            peak_elevation=1.0, base_elevation=0.1,
        ),
        "rolling_hills": MountainConfig(
            seed=seed, ridge_frequency=1.5, ridge_octaves=3,
            peak_elevation=0.5, base_elevation=0.2,
        ),
        "mesa_plateau": MountainConfig(
            seed=seed, ridge_frequency=1.0, ridge_octaves=2,
            peak_elevation=0.7, base_elevation=0.3,
            terrace_steps=5,
        ),
    }

    config = presets.get(preset, presets["mountain_range"])
    generate_mountains(grid, store, config)
    return export_globe_payload(grid, store, ramp="satellite")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive terrain globe viewer",
    )
    parser.add_argument(
        "json_file", nargs="?", type=str,
        help="Path to a globe export JSON file",
    )
    parser.add_argument(
        "--frequency", "-f", type=int, default=3,
        help="Goldberg frequency for inline generation (default: 3)",
    )
    parser.add_argument(
        "--preset", "-p", type=str, default="mountain_range",
        choices=["mountain_range", "alpine_peaks", "rolling_hills", "mesa_plateau"],
        help="Terrain preset for inline generation",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42,
        help="Noise seed",
    )
    parser.add_argument(
        "--width", type=int, default=1024,
        help="Window width (default: 1024)",
    )
    parser.add_argument(
        "--height", type=int, default=768,
        help="Window height (default: 768)",
    )
    parser.add_argument(
        "--textured", action="store_true",
        help="Enable detail texture atlas rendering (Phase 10E)",
    )
    parser.add_argument(
        "--detail-rings", type=int, default=4,
        help="Sub-tile detail ring count (default: 4, used with --textured)",
    )
    args = parser.parse_args()

    if args.json_file:
        path = Path(args.json_file)
        if not path.exists():
            print(f"Error: {path} not found", file=sys.stderr)
            sys.exit(1)
        print(f"Loading globe export from {path}...")
        payload = json.loads(path.read_text())
    else:
        print(f"Generating globe (freq={args.frequency}, preset={args.preset}, seed={args.seed})...")
        payload = _generate_payload(args.frequency, args.preset, args.seed)

    freq = payload["metadata"]["frequency"]
    tile_count = payload["metadata"]["tile_count"]
    print(f"Globe: frequency={freq}, tiles={tile_count}")

    if args.textured:
        _launch_textured(payload, args)
    else:
        _launch_flat(payload, args)


def _launch_flat(payload, args):
    """Launch the flat-colour vertex renderer (original mode)."""
    from polygrid.globe_renderer import render_terrain_globe_opengl

    freq = payload["metadata"]["frequency"]
    tile_count = payload["metadata"]["tile_count"]
    render_terrain_globe_opengl(
        payload,
        width=args.width,
        height=args.height,
        title=f"Polygrid Globe — freq={freq}, {tile_count} tiles",
    )


def _launch_textured(payload, args):
    """Generate detail grids, render atlas, launch textured renderer."""
    from polygrid.globe import build_globe_grid
    from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
    from polygrid.detail_terrain import generate_all_detail_terrain
    from polygrid.detail_render import BiomeConfig
    from polygrid.texture_pipeline import build_detail_atlas
    from polygrid.globe_renderer import render_textured_globe_opengl
    from polygrid.tile_data import FieldDef, TileDataStore, TileSchema

    freq = payload["metadata"]["frequency"]
    tile_count = payload["metadata"]["tile_count"]
    seed = args.seed

    # Rebuild the globe grid and terrain store from the payload
    print(f"Building detail grids (rings={args.detail_rings})...")
    grid = build_globe_grid(freq)
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)

    # Populate store from the payload tile data
    for tile_info in payload["tiles"]:
        fid = tile_info["id"]
        if fid in grid.faces:
            elev = tile_info.get("elevation", 0.5)
            store.set(fid, "elevation", elev)

    spec = TileDetailSpec(detail_rings=args.detail_rings)
    coll = DetailGridCollection.build(grid, spec)

    print("Generating detail terrain...")
    generate_all_detail_terrain(coll, grid, store, spec, seed=seed)

    print("Rendering detail atlas...")
    biome = BiomeConfig()
    output_dir = Path("exports/detail_atlas")
    atlas_path, uv_layout = build_detail_atlas(
        coll, biome, output_dir,
        tile_size=256,
        noise_seed=seed,
    )
    print(f"Atlas saved to {atlas_path} ({len(uv_layout)} tiles)")

    render_textured_globe_opengl(
        payload,
        atlas_path,
        uv_layout,
        width=args.width,
        height=args.height,
        title=f"Polygrid Textured Globe — freq={freq}, {tile_count} tiles",
    )


if __name__ == "__main__":
    main()
