#!/usr/bin/env python3
"""End-to-end demo: globe with ocean biome features.

Usage
-----
::

    # Ocean world (80% ocean, scattered islands):
    python scripts/demo_ocean_globe.py

    # Archipelago with ocean features:
    python scripts/demo_ocean_globe.py --terrain archipelago

    # Earth-like with forest AND ocean features:
    python scripts/demo_ocean_globe.py --terrain earthlike --features

    # Higher resolution:
    python scripts/demo_ocean_globe.py -f 4 --detail-rings 4

    # Interactive 3D viewer:
    python scripts/demo_ocean_globe.py -f 3 --view

    # With soft-blend pipeline (Phase 16):
    python scripts/demo_ocean_globe.py --soft-blend

Outputs are written to ``exports/ocean_demo/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _build_globe_with_terrain(frequency: int, terrain_preset: str, seed: int):
    """Build a globe grid with terrain patches."""
    from polygrid.globe import build_globe_grid
    from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
    from polygrid.terrain_patches import (
        TERRAIN_PRESETS,
        generate_terrain_patches,
    )

    print(f"Building globe (freq={frequency}, terrain={terrain_preset}, "
          f"seed={seed})...")
    grid = build_globe_grid(frequency)
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)

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


def _build_ocean_atlas(
    grid, store, patches, *,
    detail_rings: int,
    ocean_preset: str,
    forest_preset: str,
    features: bool,
    seed: int,
    tile_size: int,
    output_dir: Path,
    soft_blend: bool = False,
):
    """Build detail grids + feature atlas with ocean (and optionally forest) overlays."""
    from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
    from polygrid.detail_terrain import generate_all_detail_terrain
    from polygrid.biome_pipeline import (
        ForestRenderer,
        OceanRenderer,
        build_feature_atlas,
        identify_forest_tiles,
    )
    from polygrid.biome_continuity import build_biome_density_map
    from polygrid.ocean_render import (
        OCEAN_PRESETS,
        identify_ocean_tiles,
        compute_ocean_depth_map,
    )
    import random

    spec = TileDetailSpec(detail_rings=detail_rings)

    # Build detail grids
    t0 = time.perf_counter()
    print(f"Building detail grids (rings={detail_rings})...")
    coll = DetailGridCollection.build(grid, spec)
    print(f"  → {coll.total_face_count} sub-faces in "
          f"{time.perf_counter() - t0:.2f}s")

    # Seed elevations from terrain type
    t0 = time.perf_counter()
    print("Generating detail terrain...")
    rng = random.Random(seed)
    for fid in grid.faces:
        store.set(fid, "elevation", rng.uniform(0.1, 0.9))
    generate_all_detail_terrain(coll, grid, store, spec, seed=seed)
    print(f"  → done in {time.perf_counter() - t0:.2f}s")

    # ── Identify ocean and forest tiles ─────────────────────────
    ocean_faces = identify_ocean_tiles(patches)
    print(f"  → {len(ocean_faces)} ocean tiles out of {len(grid.faces)}")

    forest_faces = identify_forest_tiles(patches) if features else set()
    if features:
        print(f"  → {len(forest_faces)} forest tiles")

    # ── Ocean depth map ─────────────────────────────────────────
    t0 = time.perf_counter()
    ocean_depth_map = compute_ocean_depth_map(grid, store, ocean_faces)
    print(f"  → ocean depth map computed in {time.perf_counter() - t0:.2f}s")
    if ocean_depth_map:
        depths = list(ocean_depth_map.values())
        print(f"    depth range: [{min(depths):.3f}, {max(depths):.3f}]")

    # ── Density maps ────────────────────────────────────────────
    face_ids = list(grid.faces.keys())
    density_map = {}
    biome_type_map = {}

    # Ocean density
    t0 = time.perf_counter()
    ocean_density = build_biome_density_map(
        grid, face_ids,
        biome_faces=ocean_faces,
        seed=seed + 2000,
    )
    for fid, d in ocean_density.items():
        if d > 0.01:
            density_map[fid] = d
            biome_type_map[fid] = "ocean"

    ocean_nonzero = sum(1 for d in ocean_density.values() if d > 0.01)
    print(f"  → ocean density: {ocean_nonzero} tiles with features")

    # Forest density (if --features)
    if features and forest_faces:
        forest_density = build_biome_density_map(
            grid, face_ids,
            biome_faces=forest_faces,
            seed=seed + 1000,
        )
        for fid, d in forest_density.items():
            if d > 0.01 and fid not in ocean_faces:
                density_map[fid] = d
                biome_type_map[fid] = "forest"

        forest_nonzero = sum(
            1 for fid, d in forest_density.items()
            if d > 0.01 and fid not in ocean_faces
        )
        print(f"  → forest density: {forest_nonzero} land tiles with features")

    print(f"  → total: {len(density_map)} tiles with biome features "
          f"(computed in {time.perf_counter() - t0:.2f}s)")

    # ── Configure renderers ─────────────────────────────────────
    ocean_config = OCEAN_PRESETS.get(ocean_preset)
    if ocean_config is None:
        avail = ", ".join(sorted(OCEAN_PRESETS.keys()))
        raise ValueError(
            f"Unknown ocean preset '{ocean_preset}'. Available: {avail}"
        )

    renderers = {
        "ocean": OceanRenderer(
            config=ocean_config,
            ocean_depth_map=ocean_depth_map,
            ocean_faces=ocean_faces,
            globe_grid=grid,
        ),
    }

    if features:
        from polygrid.biome_render import FOREST_PRESETS
        forest_config = FOREST_PRESETS.get(forest_preset)
        if forest_config is None:
            avail = ", ".join(sorted(FOREST_PRESETS.keys()))
            raise ValueError(
                f"Unknown forest preset '{forest_preset}'. Available: {avail}"
            )
        renderers["forest"] = ForestRenderer(
            config=forest_config, fullslot=soft_blend,
        )

    # ── Build atlas ─────────────────────────────────────────────
    t0 = time.perf_counter()
    blend_label = " + soft-blend" if soft_blend else ""
    biome_label = "ocean" + (" + forest" if features else "")
    print(f"Building feature atlas (tile_size={tile_size}, "
          f"{biome_label}{blend_label})...")

    atlas_path, uv_layout = build_feature_atlas(
        coll, globe_grid=grid,
        biome_renderers=renderers,
        density_map=density_map,
        biome_type_map=biome_type_map,
        output_dir=output_dir / "tiles",
        tile_size=tile_size,
        noise_seed=seed,
        soft_blend=soft_blend,
    )
    print(f"  → atlas saved to {atlas_path} in "
          f"{time.perf_counter() - t0:.2f}s")

    return atlas_path, uv_layout


def main():
    parser = argparse.ArgumentParser(
        description="Ocean globe demo — ocean biome feature rendering",
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
        "--terrain", type=str, default="ocean_world",
        help="Terrain preset (default: ocean_world)",
    )
    parser.add_argument(
        "--ocean", type=str, default="temperate",
        choices=["tropical", "temperate", "arctic", "deep"],
        help="Ocean style preset (default: temperate)",
    )
    parser.add_argument(
        "--forest", type=str, default="temperate",
        choices=["temperate", "tropical", "boreal", "sparse_woodland"],
        help="Forest style for --features mode (default: temperate)",
    )
    parser.add_argument(
        "--features", action="store_true", default=False,
        help="Enable forest features on land tiles (combined biome demo)",
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--tile-size", type=int, default=256,
        help="Individual tile texture size in pixels (default: 256)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="exports/ocean_demo",
        help="Output directory",
    )
    parser.add_argument(
        "--view", action="store_true",
        help="Launch interactive 3D viewer after generation",
    )
    parser.add_argument(
        "--soft-blend", action="store_true", default=False,
        help="Enable Phase 16 soft tile-edge blending",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()

    grid, store, patches = _build_globe_with_terrain(
        args.frequency, args.terrain, args.seed,
    )

    atlas_path, uv_layout = _build_ocean_atlas(
        grid, store, patches,
        detail_rings=args.detail_rings,
        ocean_preset=args.ocean,
        forest_preset=args.forest,
        features=args.features,
        seed=args.seed,
        tile_size=args.tile_size,
        output_dir=output_dir,
        soft_blend=args.soft_blend,
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
            title=f"Ocean Globe — freq={args.frequency}, "
                  f"rings={args.detail_rings}, {args.ocean}",
        )

    elapsed = time.perf_counter() - t_total
    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Atlas: {atlas_path}")


if __name__ == "__main__":
    main()
