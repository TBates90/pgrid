#!/usr/bin/env python3
"""Demo: Phase 19 coastline transitions — natural biome boundaries.

Usage
-----
::

    # Default (freq=3, earthlike, coastline transitions):
    python scripts/demo_coastline.py

    # Interactive 3D viewer:
    python scripts/demo_coastline.py --view

    # Compare: with vs without coastlines:
    python scripts/demo_coastline.py --compare

    # Higher resolution:
    python scripts/demo_coastline.py -f 4 --detail-rings 4 --tile-size 256

    # Specific coastline preset:
    python scripts/demo_coastline.py --coastline gentle
    python scripts/demo_coastline.py --coastline rugged
    python scripts/demo_coastline.py --coastline archipelago

Outputs to ``exports/phase19_demo/``.

Pipeline stages:
  1. Globe → mountain terrain → detail grids
  2. Terrain patches → biome identification (forest + ocean)
  3. Density maps with neighbour transitions
  4. Coastline mask generation for transition tiles (Phase 19A)
  5. Dual-biome rendering with noise-warped coastlines (Phase 19B)
  6. Atlas assembly with apron gutters
  7. Flat + 3D comparison renders
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════
# Globe + terrain
# ═══════════════════════════════════════════════════════════════════

def _build_globe(frequency: int, seed: int):
    """Build globe with mountain terrain."""
    from polygrid.globe import build_globe_grid
    from polygrid.mountains import MountainConfig, generate_mountains
    from polygrid.tile_data import FieldDef, TileDataStore, TileSchema

    print(f"Building globe (freq={frequency}, seed={seed})...")
    grid = build_globe_grid(frequency)
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)

    config = MountainConfig(
        seed=seed, ridge_frequency=2.0, ridge_octaves=4,
        peak_elevation=1.0, base_elevation=0.0,
    )
    generate_mountains(grid, store, config)
    print(f"  → {len(grid.faces)} tiles")
    return grid, store


def _build_detail(grid, store, detail_rings, seed):
    """Build detail grid collection + terrain."""
    from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
    from polygrid.detail_terrain import generate_all_detail_terrain

    print(f"Building detail grids (rings={detail_rings})...")
    spec = TileDetailSpec(detail_rings=detail_rings)
    coll = DetailGridCollection.build(grid, spec)
    generate_all_detail_terrain(coll, grid, store, spec, seed=seed)
    print(f"  → {coll.total_face_count} sub-faces across {len(coll.face_ids)} tiles")
    return coll


# ═══════════════════════════════════════════════════════════════════
# Biome setup
# ═══════════════════════════════════════════════════════════════════

def _setup_biomes(grid, store, seed):
    """Identify biomes and build density/renderer maps."""
    from polygrid.biome_pipeline import ForestRenderer, OceanRenderer, identify_forest_tiles
    from polygrid.biome_render import FOREST_PRESETS
    from polygrid.ocean_render import OCEAN_PRESETS, identify_ocean_tiles, compute_ocean_depth_map
    from polygrid.biome_continuity import build_biome_density_map
    from polygrid.terrain_patches import TERRAIN_PRESETS, generate_terrain_patches

    print("Setting up biomes...")
    dist = TERRAIN_PRESETS.get("earthlike")
    patches = generate_terrain_patches(grid, distribution=dist, seed=seed)

    ocean_faces = identify_ocean_tiles(patches)
    forest_faces = identify_forest_tiles(patches)
    ocean_depth_map = compute_ocean_depth_map(grid, store, ocean_faces)

    face_ids = list(grid.faces.keys())
    density_map = {}
    biome_type_map = {}

    ocean_density = build_biome_density_map(
        grid, face_ids, biome_faces=ocean_faces, seed=seed + 2000,
    )
    for fid, d in ocean_density.items():
        if d > 0.01:
            density_map[fid] = d
            biome_type_map[fid] = "ocean"

    forest_density = build_biome_density_map(
        grid, face_ids, biome_faces=forest_faces, seed=seed + 1000,
    )
    for fid, d in forest_density.items():
        if d > 0.01 and fid not in ocean_faces:
            density_map[fid] = d
            biome_type_map[fid] = "forest"

    n_ocean = sum(1 for v in biome_type_map.values() if v == "ocean")
    n_forest = sum(1 for v in biome_type_map.values() if v == "forest")
    n_bare = len(face_ids) - n_ocean - n_forest
    print(f"  → {n_ocean} ocean, {n_forest} forest, {n_bare} bare terrain")

    renderers = {
        "ocean": OceanRenderer(
            config=OCEAN_PRESETS["temperate"],
            ocean_depth_map=ocean_depth_map,
            ocean_faces=ocean_faces,
            globe_grid=grid,
        ),
        "forest": ForestRenderer(config=FOREST_PRESETS["temperate"]),
    }

    return density_map, biome_type_map, renderers


# ═══════════════════════════════════════════════════════════════════
# Atlas building
# ═══════════════════════════════════════════════════════════════════

def _build_coastline_atlas(
    coll, grid, density_map, biome_type_map, renderers,
    output_dir, tile_size, seed, coastline_preset,
):
    """Build atlas with coastline transitions."""
    from polygrid.apron_texture import build_apron_feature_atlas
    from polygrid.coastline import COASTLINE_PRESETS

    coast_cfg = COASTLINE_PRESETS.get(coastline_preset)
    if coast_cfg is None:
        print(f"  Unknown preset '{coastline_preset}', using default")
        coast_cfg = COASTLINE_PRESETS["default"]

    print(f"Building atlas with coastline transitions (preset={coastline_preset})...")
    t0 = time.perf_counter()

    atlas_path, uv_layout = build_apron_feature_atlas(
        coll, grid,
        biome_renderers=renderers,
        density_map=density_map,
        biome_type_map=biome_type_map,
        output_dir=output_dir / "coastline_tiles",
        tile_size=tile_size,
        noise_seed=seed,
        coastline_config=coast_cfg,
        enable_coastlines=True,
    )

    elapsed = time.perf_counter() - t0
    print(f"  → {atlas_path} in {elapsed:.2f}s")
    return atlas_path, uv_layout, elapsed


def _build_no_coastline_atlas(
    coll, grid, density_map, biome_type_map, renderers,
    output_dir, tile_size, seed,
):
    """Build atlas WITHOUT coastline transitions (for comparison)."""
    from polygrid.apron_texture import build_apron_feature_atlas

    print("Building atlas WITHOUT coastline transitions (comparison)...")
    t0 = time.perf_counter()

    atlas_path, uv_layout = build_apron_feature_atlas(
        coll, grid,
        biome_renderers=renderers,
        density_map=density_map,
        biome_type_map=biome_type_map,
        output_dir=output_dir / "no_coastline_tiles",
        tile_size=tile_size,
        noise_seed=seed,
        enable_coastlines=False,
    )

    elapsed = time.perf_counter() - t0
    print(f"  → {atlas_path} in {elapsed:.2f}s")
    return atlas_path, uv_layout, elapsed


# ═══════════════════════════════════════════════════════════════════
# Flat rendering
# ═══════════════════════════════════════════════════════════════════

def _render_flat(grid, atlas_path, uv_layout, output_path):
    """Save a copy of the atlas as the flat render output."""
    from PIL import Image

    atlas_img = Image.open(str(atlas_path)).convert("RGB")
    atlas_img.save(str(output_path))
    print(f"  Flat render (atlas): {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Comparison panel
# ═══════════════════════════════════════════════════════════════════

def _build_comparison(
    coastline_path, no_coastline_path, output_dir,
):
    """Build a 2-panel comparison image."""
    from PIL import Image, ImageDraw

    panel_w = 512
    imgs = [
        ("Without Coastlines", no_coastline_path),
        ("With Coastlines (Phase 19)", coastline_path),
    ]

    n_panels = len(imgs)
    total_w = panel_w * n_panels
    panel_h = panel_w + 40
    comp = Image.new("RGB", (total_w, panel_h), (30, 30, 30))

    draw = ImageDraw.Draw(comp)
    for i, (label, atlas_path) in enumerate(imgs):
        atlas = Image.open(str(atlas_path)).resize(
            (panel_w, panel_w), Image.LANCZOS,
        )
        x_off = i * panel_w
        comp.paste(atlas, (x_off, 0))
        draw.text((x_off + 10, panel_w + 10), label, fill=(255, 255, 255))

    comp_path = output_dir / "comparison.png"
    comp.save(str(comp_path))
    print(f"  Comparison: {comp_path}")
    return comp_path


# ═══════════════════════════════════════════════════════════════════
# Coastline statistics
# ═══════════════════════════════════════════════════════════════════

def _print_coastline_stats(grid, biome_type_map, seed, tile_size):
    """Print statistics about coastline transitions."""
    from polygrid.algorithms import get_face_adjacency
    from polygrid.coastline import (
        classify_all_tiles,
        build_coastline_mask,
        CoastlineConfig,
    )

    adjacency = get_face_adjacency(grid)
    contexts = classify_all_tiles(biome_type_map, adjacency)

    n_interior = sum(1 for c in contexts.values() if c.is_interior)
    n_edge = sum(1 for c in contexts.values() if c.is_edge)

    print(f"\n  Coastline Statistics:")
    print(f"    Interior tiles: {n_interior}")
    print(f"    Edge tiles (transition): {n_edge}")
    print(f"    Transition ratio: {n_edge / max(1, len(contexts)):.1%}")

    # Sample a few transition masks
    cfg = CoastlineConfig()
    transition_fracs = []
    for fid, ctx in contexts.items():
        if ctx.is_edge and len(transition_fracs) < 5:
            cm = build_coastline_mask(
                fid, ctx, grid,
                tile_size=tile_size,
                config=cfg, seed=seed,
            )
            transition_fracs.append(cm.transition_fraction)

    if transition_fracs:
        import statistics
        print(f"    Avg transition zone: {statistics.mean(transition_fracs):.1%} of tile area")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 19 — Coastline transition demo",
    )
    parser.add_argument("-f", "--frequency", type=int, default=3)
    parser.add_argument("--detail-rings", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coastline", type=str, default="default",
                        help="Coastline preset: default, gentle, rugged, archipelago")
    parser.add_argument("--compare", action="store_true",
                        help="Build both with and without coastlines for comparison")
    parser.add_argument("--view", action="store_true",
                        help="Open interactive 3D viewer")
    args = parser.parse_args()

    output_dir = Path("exports/phase19_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build globe + terrain
    grid, store = _build_globe(args.frequency, args.seed)
    coll = _build_detail(grid, store, args.detail_rings, args.seed)

    # 2. Setup biomes
    density_map, biome_type_map, renderers = _setup_biomes(grid, store, args.seed)

    # 3. Print coastline stats
    _print_coastline_stats(grid, biome_type_map, args.seed, args.tile_size)

    # 4. Build atlas with coastlines
    atlas_path, uv_layout, t_coast = _build_coastline_atlas(
        coll, grid, density_map, biome_type_map, renderers,
        output_dir, args.tile_size, args.seed, args.coastline,
    )

    # 5. Optional comparison
    if args.compare:
        atlas_no_coast, uv_no_coast, t_no_coast = _build_no_coastline_atlas(
            coll, grid, density_map, biome_type_map, renderers,
            output_dir, args.tile_size, args.seed,
        )
        _build_comparison(atlas_path, atlas_no_coast, output_dir)
        print(f"\n  Timing: coastlines={t_coast:.2f}s, no-coastlines={t_no_coast:.2f}s")

    # 6. Render flat view
    try:
        flat_path = output_dir / "coastline_flat.png"
        _render_flat(grid, atlas_path, uv_layout, flat_path)
    except Exception as e:
        print(f"  Flat render skipped: {e}")

    # 7. Interactive viewer
    if args.view:
        try:
            from polygrid.globe_export import export_globe_payload
            from polygrid.globe_renderer_v2 import render_globe_v2

            payload = export_globe_payload(grid, store, ramp="satellite")
            print("\nLaunching 3D viewer...")
            render_globe_v2(
                payload, atlas_path, uv_layout,
                title=f"Phase 19 Coastlines — freq={args.frequency}",
                subdivisions=3,
            )
        except ImportError as e:
            print(f"  3D viewer not available: {e}")

    print(f"\nDone! Output in {output_dir}/")


if __name__ == "__main__":
    main()
