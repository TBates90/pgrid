#!/usr/bin/env python3
"""End-to-end demo: globe with forest biome features.

Usage
-----
::

    # Basic forest world (freq 3, detail_rings 4):
    python scripts/demo_forest_globe.py

    # Higher resolution:
    python scripts/demo_forest_globe.py -f 4 --detail-rings 4

    # Different terrain preset:
    python scripts/demo_forest_globe.py --terrain earthlike

    # Deep forest (100% forest, no ocean):
    python scripts/demo_forest_globe.py --terrain deep_forest

    # Different forest style:
    python scripts/demo_forest_globe.py --forest tropical

    # Interactive 3D viewer:
    python scripts/demo_forest_globe.py -f 3 --view

Outputs are written to ``exports/forest_demo/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _build_globe_with_terrain(frequency: int, terrain_preset: str, seed: int):
    """Build a globe grid with terrain patches and detail terrain."""
    from polygrid.globe import build_globe_grid
    from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
    from polygrid.terrain_patches import (
        TERRAIN_PRESETS,
        generate_terrain_patches,
    )
    from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
    from polygrid.detail_terrain import generate_all_detail_terrain

    print(f"Building globe (freq={frequency}, terrain={terrain_preset}, "
          f"seed={seed})...")
    grid = build_globe_grid(frequency)
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)

    # Assign terrain patches
    dist = TERRAIN_PRESETS.get(terrain_preset)
    if dist is None:
        avail = ", ".join(sorted(TERRAIN_PRESETS.keys()))
        raise ValueError(
            f"Unknown terrain preset '{terrain_preset}'. "
            f"Available: {avail}"
        )

    patches = generate_terrain_patches(grid, distribution=dist, seed=seed)
    print(f"  → {len(grid.faces)} tiles, {len(patches)} terrain patches")

    return grid, store, patches


def _build_feature_atlas(
    grid, store, patches, *,
    detail_rings: int,
    forest_preset: str,
    seed: int,
    tile_size: int,
    output_dir: Path,
):
    """Build detail grids + feature atlas with forest overlays."""
    from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
    from polygrid.detail_terrain import generate_all_detail_terrain
    from polygrid.biome_pipeline import (
        ForestRenderer,
        build_feature_atlas,
        identify_forest_tiles,
    )
    from polygrid.biome_continuity import build_biome_density_map
    from polygrid.biome_render import FOREST_PRESETS
    import random

    spec = TileDetailSpec(detail_rings=detail_rings)

    # Build detail grids
    t0 = time.perf_counter()
    print(f"Building detail grids (rings={detail_rings})...")
    coll = DetailGridCollection.build(grid, spec)
    print(f"  → {coll.total_face_count} sub-faces in "
          f"{time.perf_counter() - t0:.2f}s")

    # Generate detail terrain
    t0 = time.perf_counter()
    print("Generating detail terrain...")

    # Seed elevations from terrain type
    rng = random.Random(seed)
    for fid in grid.faces:
        store.set(fid, "elevation", rng.uniform(0.1, 0.9))

    generate_all_detail_terrain(coll, grid, store, spec, seed=seed)
    print(f"  → done in {time.perf_counter() - t0:.2f}s")

    # Identify forest tiles
    forest_faces = identify_forest_tiles(patches)
    print(f"  → {len(forest_faces)} forest tiles out of {len(grid.faces)}")

    # Build globe-wide density map with transitions
    t0 = time.perf_counter()
    face_ids = list(grid.faces.keys())
    density_map = build_biome_density_map(
        grid, face_ids,
        biome_faces=forest_faces,
        seed=seed + 1000,
    )
    nonzero = sum(1 for d in density_map.values() if d > 0.01)
    print(f"  → density map: {nonzero} tiles with features "
          f"(computed in {time.perf_counter() - t0:.2f}s)")

    # Choose forest renderer
    forest_config = FOREST_PRESETS.get(forest_preset)
    if forest_config is None:
        avail = ", ".join(sorted(FOREST_PRESETS.keys()))
        raise ValueError(
            f"Unknown forest preset '{forest_preset}'. Available: {avail}"
        )
    renderer = ForestRenderer(config=forest_config)

    # Build feature atlas
    t0 = time.perf_counter()
    print(f"Building feature atlas (tile_size={tile_size}, "
          f"forest={forest_preset})...")
    atlas_path, uv_layout = build_feature_atlas(
        coll, globe_grid=grid,
        biome_renderers={"forest": renderer},
        density_map=density_map,
        output_dir=output_dir / "tiles",
        tile_size=tile_size,
        noise_seed=seed,
    )
    print(f"  → atlas saved to {atlas_path} in "
          f"{time.perf_counter() - t0:.2f}s")

    return atlas_path, uv_layout


def main():
    parser = argparse.ArgumentParser(
        description="Forest globe demo — biome feature rendering",
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
        "--terrain", type=str, default="forest_world",
        help="Terrain preset (default: forest_world)",
    )
    parser.add_argument(
        "--forest", type=str, default="temperate",
        choices=["temperate", "tropical", "boreal", "sparse_woodland"],
        help="Forest style preset (default: temperate)",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--tile-size", type=int, default=256,
        help="Individual tile texture size in pixels (default: 256)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="exports/forest_demo",
        help="Output directory",
    )
    parser.add_argument(
        "--view", action="store_true",
        help="Launch interactive 3D viewer after generation",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()

    grid, store, patches = _build_globe_with_terrain(
        args.frequency, args.terrain, args.seed,
    )

    atlas_path, uv_layout = _build_feature_atlas(
        grid, store, patches,
        detail_rings=args.detail_rings,
        forest_preset=args.forest,
        seed=args.seed,
        tile_size=args.tile_size,
        output_dir=output_dir,
    )

    # Save UV layout
    uv_path = output_dir / "uv_layout.json"
    uv_path.write_text(json.dumps(uv_layout, indent=2))
    print(f"UV layout saved to {uv_path}")

    if args.view:
        print("Launching interactive 3D viewer...")
        from polygrid.globe_export import export_globe_payload
        payload = export_globe_payload(grid, store, ramp="satellite")
        from polygrid.globe_renderer import render_textured_globe_opengl
        render_textured_globe_opengl(
            payload, atlas_path, uv_layout,
            title=f"Forest Globe — freq={args.frequency}, "
                  f"rings={args.detail_rings}, {args.forest}",
        )

    elapsed = time.perf_counter() - t_total
    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Atlas: {atlas_path}")


if __name__ == "__main__":
    main()
