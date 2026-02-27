#!/usr/bin/env python3
"""End-to-end demo: globe → mountains → detail grids → textures → atlas.

Usage
-----
::

    # Basic (frequency 3, detail_rings 4):
    python scripts/demo_detail_globe.py

    # Higher resolution:
    python scripts/demo_detail_globe.py -f 4 --detail-rings 4 --preset alpine_peaks

    # Interactive 3D (requires pyglet + OpenGL 3.3+):
    python scripts/demo_detail_globe.py -f 3 --view

    # Fast renderer (PIL, no matplotlib):
    python scripts/demo_detail_globe.py -f 3 --fast

    # Compare flat vs textured at multiple detail levels:
    python scripts/demo_detail_globe.py --compare

Outputs are written to ``exports/detail_demo/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _build_globe_terrain(frequency: int, preset: str, seed: int):
    """Build a globe grid with mountain terrain and return (grid, store, payload)."""
    from polygrid.globe import build_globe_grid
    from polygrid.mountains import MountainConfig, generate_mountains
    from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
    from polygrid.globe_export import export_globe_payload

    print(f"Building globe (freq={frequency}, preset={preset}, seed={seed})...")
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
    payload = export_globe_payload(grid, store, ramp="satellite")
    print(f"  → {len(grid.faces)} tiles generated")
    return grid, store, payload


def _generate_detail_pipeline(
    grid, store, payload, *, detail_rings: int, seed: int,
    output_dir: Path, fast: bool, tile_size: int,
):
    """Run the full detail pipeline: grids → terrain → atlas."""
    from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
    from polygrid.detail_render import BiomeConfig

    spec = TileDetailSpec(detail_rings=detail_rings)

    # Build detail grids
    t0 = time.perf_counter()
    print(f"Building detail grids (rings={detail_rings})...")
    coll = DetailGridCollection.build(grid, spec)
    print(f"  → {coll.total_face_count} sub-faces in {time.perf_counter() - t0:.2f}s")

    # Generate terrain (parallel)
    t0 = time.perf_counter()
    print("Generating detail terrain (parallel)...")
    from polygrid.detail_perf import generate_all_detail_terrain_parallel
    generate_all_detail_terrain_parallel(
        coll, grid, store, spec, seed=seed,
    )
    print(f"  → done in {time.perf_counter() - t0:.2f}s")

    # Render atlas
    t0 = time.perf_counter()
    biome = BiomeConfig()
    if fast:
        print("Rendering detail atlas (fast PIL)...")
        from polygrid.detail_perf import build_detail_atlas_fast
        atlas_path, uv_layout = build_detail_atlas_fast(
            coll, biome, output_dir / "tiles",
            tile_size=tile_size, noise_seed=seed,
        )
    else:
        print("Rendering detail atlas (matplotlib)...")
        from polygrid.texture_pipeline import build_detail_atlas
        atlas_path, uv_layout = build_detail_atlas(
            coll, biome, output_dir / "tiles",
            tile_size=tile_size, noise_seed=seed,
        )
    print(f"  → atlas saved to {atlas_path} in {time.perf_counter() - t0:.2f}s")

    return atlas_path, uv_layout


def _render_comparison_panel(
    grid, store, payload, *, output_dir: Path, seed: int, fast: bool,
):
    """Render a 2×2 comparison: flat colour, rings=2, rings=4, rings=6."""
    from PIL import Image

    levels = [
        ("flat", 0),
        ("rings=2", 2),
        ("rings=4", 4),
    ]

    panel_size = 512
    images = []

    for label, rings in levels:
        if rings == 0:
            # Flat colour — render a simple globe export overview
            from polygrid.terrain_render import render_terrain
            flat_path = output_dir / "flat_overview.png"
            render_terrain(grid, store, flat_path, ramp="satellite", tile_size=panel_size)
            images.append((label, Image.open(flat_path)))
        else:
            atlas_path, _ = _generate_detail_pipeline(
                grid, store, payload,
                detail_rings=rings, seed=seed,
                output_dir=output_dir / f"detail_r{rings}",
                fast=fast, tile_size=128,
            )
            images.append((label, Image.open(atlas_path)))

    # Assemble 1×3 panel
    cols = len(images)
    panel = Image.new("RGB", (panel_size * cols, panel_size), (0, 0, 0))
    for i, (label, img) in enumerate(images):
        resized = img.resize((panel_size, panel_size), Image.LANCZOS)
        panel.paste(resized, (i * panel_size, 0))

    comparison_path = output_dir / "detail_comparison.png"
    panel.save(str(comparison_path))
    print(f"Comparison panel saved to {comparison_path}")
    return comparison_path


def main():
    parser = argparse.ArgumentParser(
        description="Detail globe demo — end-to-end pipeline",
    )
    parser.add_argument(
        "-f", "--frequency", type=int, default=3,
        help="Goldberg frequency (default: 3)",
    )
    parser.add_argument(
        "--detail-rings", type=int, default=4,
        help="Sub-tile detail ring count (default: 4)",
    )
    parser.add_argument(
        "-p", "--preset", type=str, default="mountain_range",
        choices=["mountain_range", "alpine_peaks", "rolling_hills", "mesa_plateau"],
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--tile-size", type=int, default=256,
        help="Individual tile texture size in pixels (default: 256)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="exports/detail_demo",
        help="Output directory",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use fast PIL renderer instead of matplotlib",
    )
    parser.add_argument(
        "--view", action="store_true",
        help="Launch interactive 3D viewer after generation",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Render side-by-side comparison at multiple detail levels",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()

    grid, store, payload = _build_globe_terrain(
        args.frequency, args.preset, args.seed,
    )

    # Save payload
    payload_path = output_dir / "globe_payload.json"
    payload_path.write_text(json.dumps(payload, indent=2))
    print(f"Payload saved to {payload_path}")

    if args.compare:
        _render_comparison_panel(
            grid, store, payload,
            output_dir=output_dir, seed=args.seed, fast=args.fast,
        )
    else:
        atlas_path, uv_layout = _generate_detail_pipeline(
            grid, store, payload,
            detail_rings=args.detail_rings, seed=args.seed,
            output_dir=output_dir, fast=args.fast,
            tile_size=args.tile_size,
        )

        if args.view:
            print("Launching interactive 3D viewer...")
            from polygrid.globe_renderer import render_textured_globe_opengl
            render_textured_globe_opengl(
                payload, atlas_path, uv_layout,
                title=f"Detail Globe — freq={args.frequency}, "
                      f"rings={args.detail_rings}",
            )

    elapsed = time.perf_counter() - t_total
    print(f"\nTotal time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
