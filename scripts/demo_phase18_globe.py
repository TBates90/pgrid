#!/usr/bin/env python3
# TODO REMOVE — Demo for dead Phase 18 modules (apron/texture_export/visual_cohesion).
"""Demo: Phase 18 visual cohesion — apron rendering + topology biomes + export.

Usage
-----
::

    # Default (freq=3, earthlike, all Phase 18 features):
    python scripts/demo_phase18_globe.py

    # Interactive 3D viewer:
    python scripts/demo_phase18_globe.py --view

    # With material export (KTX2 + glTF):
    python scripts/demo_phase18_globe.py --export

    # Performance benchmark (apron vs baseline):
    python scripts/demo_phase18_globe.py --bench

    # Higher resolution:
    python scripts/demo_phase18_globe.py -f 4 --detail-rings 5 --tile-size 256

    # Skip biome features (terrain-only):
    python scripts/demo_phase18_globe.py --no-features

Outputs to ``exports/phase18_demo/``.

Pipeline stages (Phase 18):
  1. Globe → mountain terrain → detail grids
  2. Apron grid construction (18A) — extended grids with neighbour sub-faces
  3. Apron-aware texture rendering (18B) — seamless gutter fill
  4. Topology-aware biome features (18C) — forest + ocean follow sub-face structure
  5. Texture export (18D) — PoT atlas, mipmaps, KTX2, ORM, glTF
  6. Seam measurement + visual comparison (18E)
"""

from __future__ import annotations

import argparse
import json
import statistics
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
    return coll, spec


# ═══════════════════════════════════════════════════════════════════
# Baseline atlas (no apron — Phase 16 pipeline)
# ═══════════════════════════════════════════════════════════════════

def _build_baseline_atlas(coll, output_dir, tile_size, seed):
    """Build atlas using the old (non-apron) pipeline for comparison."""
    from polygrid.detail_render import BiomeConfig
    from polygrid.texture_pipeline import build_detail_atlas

    print("Building baseline atlas (no apron)...")
    t0 = time.perf_counter()
    atlas_path, uv_layout = build_detail_atlas(
        coll, BiomeConfig(), output_dir / "baseline_tiles",
        tile_size=tile_size, noise_seed=seed,
    )
    elapsed = time.perf_counter() - t0
    print(f"  → {atlas_path} in {elapsed:.2f}s")
    return atlas_path, uv_layout, elapsed


# ═══════════════════════════════════════════════════════════════════
# Apron atlas (Phase 18A+B) — ground only
# ═══════════════════════════════════════════════════════════════════

def _build_apron_atlas(coll, grid, output_dir, tile_size, seed):
    """Build atlas using apron grids for seamless gutters."""
    from polygrid.apron_texture import build_apron_atlas

    print("Building apron atlas (Phase 18A+B)...")
    t0 = time.perf_counter()
    atlas_path, uv_layout = build_apron_atlas(
        coll, grid,
        output_dir=output_dir / "apron_tiles",
        tile_size=tile_size, noise_seed=seed,
    )
    elapsed = time.perf_counter() - t0
    print(f"  → {atlas_path} in {elapsed:.2f}s")
    return atlas_path, uv_layout, elapsed


# ═══════════════════════════════════════════════════════════════════
# Apron + biome features (Phase 18A+B+C)
# ═══════════════════════════════════════════════════════════════════

def _build_feature_atlas(coll, grid, store, output_dir, tile_size, seed):
    """Build apron atlas with topology-aware biome features."""
    from polygrid.apron_texture import build_apron_feature_atlas
    from polygrid.biome_pipeline import ForestRenderer, OceanRenderer, identify_forest_tiles
    from polygrid.biome_render import FOREST_PRESETS
    from polygrid.ocean_render import OCEAN_PRESETS, identify_ocean_tiles, compute_ocean_depth_map
    from polygrid.biome_continuity import build_biome_density_map
    from polygrid.terrain_patches import TERRAIN_PRESETS, generate_terrain_patches

    print("Building feature atlas (Phase 18C — topology biomes)...")
    t0 = time.perf_counter()

    # Terrain patches for biome identification
    dist = TERRAIN_PRESETS.get("earthlike")
    patches = generate_terrain_patches(grid, distribution=dist, seed=seed)

    ocean_faces = identify_ocean_tiles(patches)
    forest_faces = identify_forest_tiles(patches)
    ocean_depth_map = compute_ocean_depth_map(grid, store, ocean_faces)

    # Build density maps
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
    print(f"  → {n_ocean} ocean tiles, {n_forest} forest tiles")

    renderers = {
        "ocean": OceanRenderer(
            config=OCEAN_PRESETS["temperate"],
            ocean_depth_map=ocean_depth_map,
            ocean_faces=ocean_faces,
            globe_grid=grid,
        ),
        "forest": ForestRenderer(config=FOREST_PRESETS["temperate"]),
    }

    atlas_path, uv_layout = build_apron_feature_atlas(
        coll, grid,
        biome_renderers=renderers,
        density_map=density_map,
        biome_type_map=biome_type_map,
        output_dir=output_dir / "feature_tiles",
        tile_size=tile_size,
        noise_seed=seed,
    )

    elapsed = time.perf_counter() - t0
    print(f"  → {atlas_path} in {elapsed:.2f}s")
    return atlas_path, uv_layout, biome_type_map, elapsed


# ═══════════════════════════════════════════════════════════════════
# Seam measurement
# ═══════════════════════════════════════════════════════════════════

def _measure_seams(atlas_path, uv_layout, label):
    """Measure and print seam visibility for an atlas."""
    from PIL import Image
    from polygrid.visual_cohesion import measure_seam_visibility

    img = Image.open(str(atlas_path)).convert("RGB")
    seam = measure_seam_visibility(img, uv_layout, n_samples=40)

    print(f"\n  [{label}] Seam Visibility:")
    print(f"    Boundary variance:  {seam['boundary_variance']:.1f}")
    print(f"    Interior variance:  {seam['interior_variance']:.1f}")
    print(f"    Ratio (boundary/interior): {seam['ratio']:.2f}")
    if seam["ratio"] < 2.0:
        print(f"    ✅ Within target (< 2.0×)")
    else:
        print(f"    ⚠  Above target (> 2.0×)")

    return seam


# ═══════════════════════════════════════════════════════════════════
# Export pipeline (Phase 18D)
# ═══════════════════════════════════════════════════════════════════

def _run_export(
    coll, atlas_path, uv_layout, biome_type_map,
    output_dir, tile_size, frequency,
):
    """Run KTX2 + ORM + glTF export."""
    from PIL import Image
    from polygrid.texture_export import (
        resize_atlas_pot,
        generate_atlas_mipmaps,
        export_atlas_ktx2,
        validate_ktx2_header,
        build_orm_atlas,
        export_globe_gltf,
    )

    export_dir = output_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning texture export pipeline (Phase 18D)...")
    t0 = time.perf_counter()

    # PoT atlas
    atlas_img = Image.open(str(atlas_path)).convert("RGB")
    pot_atlas = resize_atlas_pot(atlas_img)
    pot_path = export_dir / "atlas_pot.png"
    pot_atlas.save(str(pot_path))
    print(f"  PoT atlas: {pot_atlas.size[0]}×{pot_atlas.size[1]}")

    # Mipmaps
    mip_paths = generate_atlas_mipmaps(pot_path, output_dir=export_dir / "mipmaps")
    print(f"  Mipmaps: {len(mip_paths)} levels")

    # KTX2
    ktx_path = export_dir / "atlas.ktx2"
    export_atlas_ktx2(pot_path, ktx_path, include_mipmaps=True)
    valid = validate_ktx2_header(ktx_path)
    size_kb = ktx_path.stat().st_size / 1024
    print(f"  KTX2: {ktx_path.name} ({size_kb:.0f} KB, valid={valid})")

    # ORM
    orm_img, orm_uv = build_orm_atlas(
        coll, biome_type_map=biome_type_map,
        tile_size=tile_size, gutter=4,
    )
    orm_path = export_dir / "orm.png"
    orm_img.save(str(orm_path))
    print(f"  ORM atlas: {orm_img.size[0]}×{orm_img.size[1]}")

    # glTF
    try:
        gltf_path = export_dir / "globe.gltf"
        export_globe_gltf(
            frequency=frequency,
            uv_layout=uv_layout,
            albedo_path=pot_path,
            orm_path=orm_path,
            output_path=gltf_path,
            embed_textures=True,
        )
        gltf_kb = gltf_path.stat().st_size / 1024
        print(f"  glTF: {gltf_path.name} ({gltf_kb:.0f} KB)")
    except ImportError:
        print("  glTF: skipped (models library not available)")

    elapsed = time.perf_counter() - t0
    print(f"  Export completed in {elapsed:.2f}s")
    return elapsed


# ═══════════════════════════════════════════════════════════════════
# Comparison panel
# ═══════════════════════════════════════════════════════════════════

def _build_comparison(
    baseline_path, apron_path, feature_path, output_dir,
):
    """Build a 3-panel comparison image."""
    from PIL import Image, ImageDraw

    panel_w = 512
    has_features = feature_path is not None

    imgs = [
        ("Baseline (Phase 16)", baseline_path),
        ("Apron (Phase 18B)", apron_path),
    ]
    if has_features:
        imgs.append(("Apron + Biomes (Phase 18C)", feature_path))

    n_panels = len(imgs)
    total_w = panel_w * n_panels
    panel_h = panel_w + 40
    comp = Image.new("RGB", (total_w, panel_h), (30, 30, 30))

    for i, (label, path) in enumerate(imgs):
        try:
            img = Image.open(str(path)).resize((panel_w, panel_w), Image.LANCZOS)
            comp.paste(img, (i * panel_w, 40))
        except Exception:
            pass
        try:
            draw = ImageDraw.Draw(comp)
            draw.text((i * panel_w + 10, 8), label, fill=(255, 255, 255))
        except Exception:
            pass

    comp_path = output_dir / "phase18_comparison.png"
    comp.save(str(comp_path))
    print(f"\n  Comparison panel → {comp_path}")
    return comp_path


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 18 visual cohesion demo — apron + topology + export",
    )
    parser.add_argument("-f", "--frequency", type=int, default=3)
    parser.add_argument("--detail-rings", type=int, default=4)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("-o", "--output", type=str, default="exports/phase18_demo")
    parser.add_argument("--view", action="store_true",
                        help="Launch interactive 3D viewer")
    parser.add_argument("--export", action="store_true",
                        help="Run KTX2 + glTF export pipeline")
    parser.add_argument("--bench", action="store_true",
                        help="Run performance benchmark")
    parser.add_argument("--no-features", action="store_true",
                        help="Skip biome features (terrain-only)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    # ── Build globe + detail grids ──────────────────────────────
    grid, store = _build_globe(args.frequency, args.seed)
    coll, spec = _build_detail(grid, store, args.detail_rings, args.seed)

    # ── Baseline atlas (no apron) ───────────────────────────────
    print(f"\n{'═' * 60}")
    print("  Baseline — Phase 16 pipeline (no apron)")
    print(f"{'═' * 60}")
    baseline_path, baseline_uv, t_baseline = _build_baseline_atlas(
        coll, output_dir, args.tile_size, args.seed,
    )
    seam_baseline = _measure_seams(baseline_path, baseline_uv, "Baseline")

    # ── Apron atlas (Phase 18A+B) ───────────────────────────────
    print(f"\n{'═' * 60}")
    print("  Phase 18A+B — Apron atlas (ground only)")
    print(f"{'═' * 60}")
    apron_path, apron_uv, t_apron = _build_apron_atlas(
        coll, grid, output_dir, args.tile_size, args.seed,
    )
    seam_apron = _measure_seams(apron_path, apron_uv, "Apron")

    # ── Apron + biome features (Phase 18C) ──────────────────────
    feature_path = None
    biome_type_map = {}
    feature_uv = apron_uv

    if not args.no_features:
        print(f"\n{'═' * 60}")
        print("  Phase 18C — Apron + topology biomes")
        print(f"{'═' * 60}")
        feature_path, feature_uv, biome_type_map, t_feature = _build_feature_atlas(
            coll, grid, store, output_dir, args.tile_size, args.seed,
        )
        seam_feature = _measure_seams(feature_path, feature_uv, "Apron+Biomes")

    # ── Comparison panel ────────────────────────────────────────
    try:
        _build_comparison(baseline_path, apron_path, feature_path, output_dir)
    except ImportError:
        print("  (Pillow ImageDraw not available — skipping comparison)")

    # ── Export (Phase 18D) ──────────────────────────────────────
    if args.export:
        best_atlas = feature_path if feature_path else apron_path
        best_uv = feature_uv
        _run_export(
            coll, best_atlas, best_uv, biome_type_map,
            output_dir, args.tile_size, args.frequency,
        )

    # ── Summary ─────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_total
    print(f"\n{'═' * 60}")
    print(f"  PHASE 18 SUMMARY")
    print(f"{'═' * 60}")
    print(f"  Globe: freq={args.frequency}, {len(grid.faces)} tiles, "
          f"rings={args.detail_rings}, tile_size={args.tile_size}")
    print(f"  Sub-faces: {coll.total_face_count}")
    print()
    print(f"  Baseline atlas:   {t_baseline:.2f}s  "
          f"(seam ratio: {seam_baseline['ratio']:.2f})")
    print(f"  Apron atlas:      {t_apron:.2f}s  "
          f"(seam ratio: {seam_apron['ratio']:.2f})")
    if not args.no_features and feature_path:
        print(f"  Feature atlas:    {t_feature:.2f}s  "
              f"(seam ratio: {seam_feature['ratio']:.2f})")

    speedup = t_apron / t_baseline if t_baseline > 0 else float("inf")
    print(f"\n  Apron overhead: {speedup:.2f}× baseline")
    if speedup < 2.0:
        print(f"  ✅ Within 2× performance budget")
    else:
        print(f"  ⚠  Exceeds 2× budget — consider caching apron grids")

    print(f"\n  Total time: {total_elapsed:.2f}s")
    print(f"  Output: {output_dir.resolve()}")
    print(f"{'═' * 60}")

    # ── Optional: 3D viewer ─────────────────────────────────────
    if args.view:
        best_atlas = feature_path if feature_path else apron_path
        best_uv = feature_uv
        try:
            from polygrid.globe_export import export_globe_payload
            payload = export_globe_payload(grid, store, ramp="satellite")
            from polygrid.globe_renderer_v2 import render_globe_v2
            print("\nLaunching 3D viewer...")
            render_globe_v2(
                payload, best_atlas, best_uv,
                title=f"Phase 18 Globe — freq={args.frequency}, "
                      f"rings={args.detail_rings}",
                subdivisions=3,
            )
        except ImportError as e:
            print(f"  Cannot launch viewer: {e}")

    # ── Optional: benchmark ─────────────────────────────────────
    if args.bench:
        from polygrid.visual_cohesion import benchmark_apron_pipeline

        print(f"\n{'═' * 60}")
        print("  BENCHMARK — apron vs baseline")
        print(f"{'═' * 60}")
        bench = benchmark_apron_pipeline(
            frequency=args.frequency,
            detail_rings=args.detail_rings,
            tile_size=args.tile_size,
            seed=args.seed,
            n_runs=3,
        )
        print(f"  Baseline: {bench['baseline_mean']:.3f}s")
        print(f"  Apron:    {bench['apron_mean']:.3f}s")
        print(f"  Ratio:    {bench['ratio']:.2f}×")
        if bench["within_budget"]:
            print(f"  ✅ Within 2× performance budget")
        else:
            print(f"  ⚠  Exceeds 2× budget")


if __name__ == "__main__":
    main()
