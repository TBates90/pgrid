#!/usr/bin/env python3
# TODO REMOVE — Demo for dead Phase 11 modules.
"""Demo: Phase 11 full cohesive terrain pipeline vs Phase 10 per-tile.

Usage
-----
::

    # Default comparison (freq=3, detail_rings=4, earthlike preset):
    python scripts/demo_cohesive_globe.py

    # Choose a terrain preset:
    python scripts/demo_cohesive_globe.py --preset mountainous

    # Higher resolution:
    python scripts/demo_cohesive_globe.py -f 3 --detail-rings 5

    # Custom noise parameters:
    python scripts/demo_cohesive_globe.py --freq-3d 5.0 --ridge-weight 0.6

    # Interactive 3D viewer (if pyglet/OpenGL available):
    python scripts/demo_cohesive_globe.py --view

    # Performance benchmark:
    python scripts/demo_cohesive_globe.py --bench

Outputs side-by-side comparison to ``exports/cohesive_demo/``.

Pipeline stages (Phase 11):
  1. Globe → global mountain terrain
  2. Terrain patches (11B) — region-based terrain distribution
  3. 3D-coherent noise (11A) applied per patch
  4. Region stitching (11C) — cross-tile terrain continuity
  5. Enhanced mountains + rivers + erosion (11D) on stitched grids
  6. Automatic biome assignment (11E)
  7. Seamless texture rendering (11E) with normal maps
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════
# Globe construction
# ═══════════════════════════════════════════════════════════════════

def _build_globe_terrain(frequency: int, seed: int):
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


# ═══════════════════════════════════════════════════════════════════
# Phase 10 — per-tile terrain
# ═══════════════════════════════════════════════════════════════════

def _generate_phase10_terrain(grid, store, *, detail_rings, seed):
    """Generate terrain using Phase 10 (per-tile) approach."""
    from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
    from polygrid.detail_terrain import generate_all_detail_terrain

    spec = TileDetailSpec(detail_rings=detail_rings)
    coll = DetailGridCollection.build(grid, spec)
    generate_all_detail_terrain(coll, grid, store, spec, seed=seed)
    return coll, spec


# ═══════════════════════════════════════════════════════════════════
# Phase 11 — full cohesive pipeline
# ═══════════════════════════════════════════════════════════════════

def _generate_phase11_terrain(grid, store, *, detail_rings, seed,
                               noise_frequency, ridge_frequency,
                               fbm_weight, ridge_weight, base_weight,
                               amplitude, preset_name):
    """Generate terrain using the full Phase 11 pipeline.

    Steps:
    1. Build detail grid collection
    2. Generate 3D-coherent base terrain (11A)
    3. Generate terrain patches (11B)
    4. Stitch and apply cross-tile terrain for each patch (11C)
    5. Run enhanced mountains + rivers + erosion (11D)
    """
    from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
    from polygrid.detail_terrain_3d import Terrain3DSpec, generate_all_detail_terrain_3d
    from polygrid.terrain_patches import (
        TERRAIN_PRESETS, generate_terrain_patches, apply_terrain_patches,
    )
    from polygrid.region_stitch import generate_stitched_patch_terrain
    from polygrid.globe_terrain import (
        MountainConfig3D, GLOBE_MOUNTAIN_RANGE, GLOBE_CONTINENTAL_DIVIDE,
        generate_mountains_3d, generate_rivers_on_stitched,
        ErosionConfig, erode_terrain,
    )
    from polygrid.rivers import RiverConfig
    from polygrid.algorithms import get_face_adjacency

    spec_grid = TileDetailSpec(detail_rings=detail_rings)
    coll = DetailGridCollection.build(grid, spec_grid)

    # ── Stage 1: 3D-coherent base terrain (11A) ────────────────────
    print("  Stage 1: 3D-coherent base terrain...")
    spec_3d = Terrain3DSpec(
        noise_frequency=noise_frequency,
        ridge_frequency=ridge_frequency,
        fbm_weight=fbm_weight,
        ridge_weight=ridge_weight,
        base_weight=base_weight,
        amplitude=amplitude,
        seed=seed,
    )
    generate_all_detail_terrain_3d(coll, grid, store, spec_3d)

    # ── Stage 2: terrain patches (11B) ─────────────────────────────
    print("  Stage 2: Terrain patches...")
    preset = TERRAIN_PRESETS.get(preset_name, TERRAIN_PRESETS["earthlike"])
    patches = generate_terrain_patches(grid, distribution=preset, seed=seed)
    apply_terrain_patches(coll, grid, store, patches, seed=seed)
    print(f"    → {len(patches)} patches")
    for p in patches[:6]:
        print(f"      {p.name}: {p.terrain_type} ({len(p.face_ids)} tiles)")

    # ── Stage 3: stitched cross-tile terrain (11C) ─────────────────
    print("  Stage 3: Stitched cross-tile terrain...")
    patched_count = 0
    for patch in patches:
        if len(patch.face_ids) < 2:
            continue
        try:
            generate_stitched_patch_terrain(
                coll, grid, store, patch.face_ids,
                spec=patch.to_terrain_3d_spec(),
            )
            patched_count += 1
        except Exception as e:
            print(f"    ⚠ Patch '{patch.name}' stitch failed: {e}")
    print(f"    → {patched_count} patches stitched")

    # ── Stage 4: enhanced mountains + rivers + erosion (11D) ───────
    print("  Stage 4: Mountains, rivers, erosion...")
    mountain_config = GLOBE_MOUNTAIN_RANGE
    tiles_with_mountains = 0
    for face_id in coll.face_ids:
        s = coll._stores.get(face_id)
        if s is None:
            continue
        g = coll.grids[face_id]
        generate_mountains_3d(g, s, mountain_config)
        tiles_with_mountains += 1
    print(f"    → mountains applied to {tiles_with_mountains} tiles")

    # Rivers on a stitched group
    adj = get_face_adjacency(grid)
    start = list(grid.faces.keys())[5]
    river_group = [start]
    for _ in range(15):
        for fid in list(river_group):
            for nid in adj.get(fid, []):
                if nid not in river_group:
                    river_group.append(nid)
                if len(river_group) >= 12:
                    break
            if len(river_group) >= 12:
                break

    from polygrid.region_stitch import stitch_detail_grids, generate_terrain_on_stitched, split_terrain_to_tiles
    combined, mapping = stitch_detail_grids(coll, grid, river_group)
    combined_store = generate_terrain_on_stitched(
        combined, mapping, grid, store, spec_3d,
    )

    river_config = RiverConfig(min_accumulation=3, seed=seed)
    rivers = generate_rivers_on_stitched(combined, combined_store, river_config)
    split_terrain_to_tiles(combined_store, mapping, coll)
    print(f"    → {len(rivers.segments)} river segments across {len(river_group)} tiles")

    # Erosion
    erosion_config = ErosionConfig(iterations=3, erosion_rate=0.02, deposition_rate=0.01)
    for face_id in coll.face_ids:
        s = coll._stores.get(face_id)
        if s is None:
            continue
        g = coll.grids[face_id]
        erode_terrain(g, s, erosion_config)
    print(f"    → erosion applied ({erosion_config.iterations} iterations)")

    return coll, spec_grid


# ═══════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════

def _render_atlas(coll, biome, output_dir, *, tile_size, seed, fast):
    """Render detail atlas for a collection."""
    if fast:
        from polygrid.detail_perf import build_detail_atlas_fast
        return build_detail_atlas_fast(
            coll, biome, output_dir / "tiles",
            tile_size=tile_size, noise_seed=seed,
        )
    else:
        from polygrid.texture_pipeline import build_detail_atlas
        return build_detail_atlas(
            coll, biome, output_dir / "tiles",
            tile_size=tile_size, noise_seed=seed,
        )


def _render_with_biomes(coll, output_dir, *, tile_size, seed):
    """Render using Phase 11E auto-biome assignment."""
    from polygrid.render_enhanced import (
        assign_all_biomes, render_seamless_texture,
    )

    biomes = assign_all_biomes(coll)

    # Summarise biome distribution
    from collections import Counter
    biome_names = Counter()
    from polygrid.render_enhanced import BIOME_PRESETS
    preset_lookup = {id(v): k for k, v in BIOME_PRESETS.items()}
    for b in biomes.values():
        name = preset_lookup.get(id(b), "custom")
        biome_names[name] += 1
    print(f"    Biome distribution: {dict(biome_names)}")

    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for face_id in sorted(coll.face_ids)[:40]:  # cap at 40 for speed
        render_seamless_texture(
            coll, face_id,
            tiles_dir / f"{face_id}.png",
            biome=biomes.get(face_id),
            tile_size=tile_size,
            noise_seed=seed,
        )
        rendered += 1

    print(f"    → {rendered} biome-assigned tiles rendered")
    return tiles_dir


def _render_normal_map_sample(coll, output_dir):
    """Compute and save a normal-map sample for a few tiles."""
    from polygrid.render_enhanced import compute_all_normal_maps

    normals = compute_all_normal_maps(coll, scale=2.0)

    # Stats
    total_faces = sum(len(nm) for nm in normals.values())
    tilt_values = []
    for nm in normals.values():
        for nx, ny, nz in nm.values():
            tilt_values.append(math.sqrt(nx * nx + ny * ny))

    mean_tilt = statistics.mean(tilt_values) if tilt_values else 0
    max_tilt = max(tilt_values) if tilt_values else 0
    print(f"    Normal maps: {len(normals)} tiles, {total_faces} faces")
    print(f"    Mean tilt: {mean_tilt:.4f}, max tilt: {max_tilt:.4f}")

    # Save sample as JSON
    sample_path = output_dir / "normal_map_sample.json"
    sample = {}
    for fid in list(normals.keys())[:3]:
        sample[fid] = {
            sfid: list(n) for sfid, n in list(normals[fid].items())[:10]
        }
    with open(sample_path, "w") as f:
        json.dump(sample, f, indent=2)
    print(f"    → sample saved to {sample_path}")


# ═══════════════════════════════════════════════════════════════════
# Boundary statistics
# ═══════════════════════════════════════════════════════════════════

def _compute_boundary_stats(coll, globe, *, label):
    """Compute and print cross-tile boundary elevation statistics."""
    from polygrid.detail_terrain_3d import precompute_3d_positions
    from polygrid.detail_terrain import _boundary_face_ids

    # Intra-tile variance
    intra_vars = []
    for fid in list(coll.face_ids)[:20]:
        store = coll._stores.get(fid)
        if store is None:
            continue
        grid = coll.grids[fid]
        elevs = [store.get(sf, "elevation") for sf in grid.faces]
        if len(elevs) >= 2:
            intra_vars.append(statistics.variance(elevs))

    # Cross-tile boundary diffs
    cross_diffs = []
    for face_a in list(coll.face_ids)[:20]:
        store_a = coll._stores.get(face_a)
        if store_a is None:
            continue
        pos_a = precompute_3d_positions(globe, face_a, coll.grids[face_a])
        for face_b in globe.faces[face_a].neighbor_ids:
            store_b = coll._stores.get(face_b)
            if store_b is None:
                continue
            pos_b = precompute_3d_positions(globe, face_b, coll.grids[face_b])
            bnd_b = _boundary_face_ids(coll.grids[face_b])
            for fa_id in _boundary_face_ids(coll.grids[face_a]):
                if fa_id not in pos_a:
                    continue
                pa = pos_a[fa_id]
                best_d, best_e = float("inf"), None
                for fb_id in bnd_b:
                    if fb_id not in pos_b:
                        continue
                    d = math.sqrt(sum((a - b)**2 for a, b in zip(pa, pos_b[fb_id])))
                    if d < best_d:
                        best_d = d
                        best_e = store_b.get(fb_id, "elevation")
                if best_d < 0.15 and best_e is not None:
                    ea = store_a.get(fa_id, "elevation")
                    cross_diffs.append(abs(ea - best_e))

    mean_intra = statistics.mean(intra_vars) if intra_vars else 0
    mean_cross = statistics.mean(cross_diffs) if cross_diffs else 0
    max_cross = max(cross_diffs) if cross_diffs else 0

    print(f"\n  [{label}] Boundary Statistics:")
    print(f"    Mean intra-tile variance:     {mean_intra:.6f}")
    print(f"    Mean cross-tile elev diff:    {mean_cross:.6f}")
    print(f"    Max cross-tile elev diff:     {max_cross:.6f}")
    print(f"    Boundary pairs checked:       {len(cross_diffs)}")
    return mean_intra, mean_cross


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 11 cohesive terrain demo — full pipeline comparison",
    )
    parser.add_argument("-f", "--frequency", type=int, default=3)
    parser.add_argument("--detail-rings", type=int, default=4)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("-o", "--output", type=str, default="exports/cohesive_demo")
    parser.add_argument("--fast", action="store_true",
                        help="Use fast PIL renderer for Phase 10")
    parser.add_argument("--view", action="store_true",
                        help="Launch interactive 3D viewer")
    parser.add_argument("--bench", action="store_true",
                        help="Run performance benchmark")
    # Terrain preset
    parser.add_argument("--preset", type=str, default="earthlike",
                        choices=["earthlike", "mountainous", "archipelago", "pangaea"],
                        help="Terrain distribution preset (Phase 11B)")
    # 3D noise params
    parser.add_argument("--freq-3d", type=float, default=4.0,
                        help="3D noise frequency")
    parser.add_argument("--ridge-freq", type=float, default=3.0,
                        help="Ridge noise frequency")
    parser.add_argument("--fbm-weight", type=float, default=0.6)
    parser.add_argument("--ridge-weight", type=float, default=0.4)
    parser.add_argument("--base-weight", type=float, default=0.70)
    parser.add_argument("--amplitude", type=float, default=0.15)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    from polygrid.detail_render import BiomeConfig
    biome = BiomeConfig()

    grid, store = _build_globe_terrain(args.frequency, args.seed)

    # ── Phase 10: per-tile terrain ──────────────────────────────────
    print("\n" + "═" * 60)
    print("  Phase 10 — Per-tile noise terrain")
    print("═" * 60)
    t0 = time.perf_counter()
    coll_10, spec_10 = _generate_phase10_terrain(
        grid, store, detail_rings=args.detail_rings, seed=args.seed,
    )
    t10_terrain = time.perf_counter() - t0
    print(f"  Terrain generated in {t10_terrain:.2f}s")

    t0 = time.perf_counter()
    atlas_10, uv_10 = _render_atlas(
        coll_10, biome, output_dir / "phase10",
        tile_size=args.tile_size, seed=args.seed, fast=args.fast,
    )
    t10_render = time.perf_counter() - t0
    print(f"  Atlas rendered in {t10_render:.2f}s → {atlas_10}")

    intra_10, cross_10 = _compute_boundary_stats(coll_10, grid, label="Phase 10")

    # ── Phase 11: full cohesive pipeline ────────────────────────────
    print("\n" + "═" * 60)
    print(f"  Phase 11 — Full cohesive pipeline (preset: {args.preset})")
    print("═" * 60)
    t0 = time.perf_counter()
    coll_11, spec_11 = _generate_phase11_terrain(
        grid, store,
        detail_rings=args.detail_rings, seed=args.seed,
        noise_frequency=args.freq_3d,
        ridge_frequency=args.ridge_freq,
        fbm_weight=args.fbm_weight,
        ridge_weight=args.ridge_weight,
        base_weight=args.base_weight,
        amplitude=args.amplitude,
        preset_name=args.preset,
    )
    t11_terrain = time.perf_counter() - t0
    print(f"  Full pipeline completed in {t11_terrain:.2f}s")

    # ── Phase 11E: biome assignment + rendering ─────────────────────
    print("\n  Rendering with auto-biome assignment (11E)...")
    t0 = time.perf_counter()
    biome_dir = _render_with_biomes(
        coll_11, output_dir / "phase11",
        tile_size=args.tile_size, seed=args.seed,
    )
    t11_render = time.perf_counter() - t0
    print(f"  Biome rendering completed in {t11_render:.2f}s")

    # Also render a standard atlas for comparison
    t0 = time.perf_counter()
    atlas_11, uv_11 = _render_atlas(
        coll_11, biome, output_dir / "phase11_standard",
        tile_size=args.tile_size, seed=args.seed, fast=args.fast,
    )
    print(f"  Standard atlas rendered in {time.perf_counter() - t0:.2f}s → {atlas_11}")

    # ── Normal maps (11E) ───────────────────────────────────────────
    print("\n  Computing normal maps (11E)...")
    _render_normal_map_sample(coll_11, output_dir / "phase11")

    intra_11, cross_11 = _compute_boundary_stats(coll_11, grid, label="Phase 11")

    # ── Comparison panel ────────────────────────────────────────────
    try:
        from PIL import Image

        panel_w = 512
        img_10 = Image.open(atlas_10).resize((panel_w, panel_w), Image.LANCZOS)
        img_11 = Image.open(atlas_11).resize((panel_w, panel_w), Image.LANCZOS)

        panel = Image.new("RGB", (panel_w * 2, panel_w + 40), (30, 30, 30))
        panel.paste(img_10, (0, 40))
        panel.paste(img_11, (panel_w, 40))

        # Add labels
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(panel)
            draw.text((10, 5), "Phase 10 (per-tile)", fill=(255, 255, 255))
            draw.text((panel_w + 10, 5), "Phase 11 (cohesive)", fill=(255, 255, 255))
        except ImportError:
            pass

        comp_path = output_dir / "comparison.png"
        panel.save(str(comp_path))
        print(f"\n  Comparison panel → {comp_path}")
    except ImportError:
        print("\n  (Pillow not available — skipping comparison panel)")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  SUMMARY")
    print(f"{'═' * 60}")
    print(f"  Globe: freq={args.frequency}, {len(grid.faces)} tiles, "
          f"detail_rings={args.detail_rings}")
    print(f"  Preset: {args.preset}")
    print()
    print(f"  Phase 10:  terrain {t10_terrain:.2f}s  |  render {t10_render:.2f}s")
    print(f"  Phase 11:  terrain {t11_terrain:.2f}s  |  render {t11_render:.2f}s")
    if cross_10 > 0 and cross_11 > 0:
        improvement = cross_10 / cross_11
        print(f"\n  Cross-tile continuity improvement: {improvement:.1f}x")
    print(f"\n  Output: {output_dir.resolve()}")
    print(f"{'═' * 60}")

    # ── Optional: 3D viewer ─────────────────────────────────────────
    if args.view:
        try:
            from polygrid.globe_export import export_globe_payload
            payload = export_globe_payload(grid, store, ramp="satellite")
            from polygrid.globe_renderer_v2 import render_globe_v2
            print("\nLaunching 3D viewer (v2 — subdivided, batched, flood-filled)...")
            render_globe_v2(
                payload, atlas_11, uv_11,
                title=f"Cohesive Globe v2 — freq={args.frequency}, "
                      f"rings={args.detail_rings}, preset={args.preset}",
                subdivisions=3,
            )
        except ImportError as e:
            print(f"  Cannot launch viewer: {e}")

    # ── Optional: benchmark ─────────────────────────────────────────
    if args.bench:
        print("\n" + "═" * 60)
        print("  BENCHMARK — 3 runs each")
        print("═" * 60)
        for label, gen_fn in [
            ("Phase 10", lambda: _generate_phase10_terrain(
                grid, store, detail_rings=args.detail_rings, seed=args.seed)),
            ("Phase 11", lambda: _generate_phase11_terrain(
                grid, store,
                detail_rings=args.detail_rings, seed=args.seed,
                noise_frequency=args.freq_3d,
                ridge_frequency=args.ridge_freq,
                fbm_weight=args.fbm_weight,
                ridge_weight=args.ridge_weight,
                base_weight=args.base_weight,
                amplitude=args.amplitude,
                preset_name=args.preset,
            )),
        ]:
            times = []
            for i in range(3):
                t0 = time.perf_counter()
                gen_fn()
                times.append(time.perf_counter() - t0)
            mean_t = statistics.mean(times)
            std_t = statistics.stdev(times) if len(times) > 1 else 0
            print(f"  {label}: {mean_t:.2f}s ± {std_t:.2f}s")


if __name__ == "__main__":
    main()
