# Code Cleanup Review

### Cleanup Review Prompt

Review the pgrid codebase. Tag code using the following tags:

 # TODO REFACTOR
 # TODO REMOVE
 # TODO REVIEW
 # TODO OPTIMISE

Review and update the CLEANUP_REVIEW.md document.

### Latest Most Accurate Runs
```
python scripts/render_polygrids.py --frequency 2 --detail-rings 2 --pent-gutter 4 --hex-gutter 4 --pent-uv-scale 0.95 --pent-twist -1 --no-neighbours

python scripts/render_polygrids.py --frequency 2 --detail-rings 4 --pent-gutter 4 --hex-gutter 4 --pent-uv-scale 0.975 --pent-twist -7.55 --no-neighbours
```

**Note:** `--gutter 4` and `--no-neighbours` are now the default behaviour.
Equivalent shorter commands:
```
python scripts/render_polygrids.py -f 2 --detail-rings 2 --pent-uv-scale 0.95 --pent-twist -1
python scripts/render_polygrids.py -f 2 --detail-rings 4 --pent-uv-scale 0.975 --pent-twist -7.55
```
Use `--neighbours` to re-enable neighbour↔neighbour closure.

---

## Key Imports

| Module | Used by |
|--------|---------|
| `polygrid.globe` | render_polygrids, render_globe_from_tiles |
| `polygrid.mountains` | render_polygrids, render_globe_from_tiles |
| `polygrid.tile_data` | render_polygrids, render_globe_from_tiles |
| `polygrid.tile_detail` | render_polygrids |
| `polygrid.detail_terrain` | render_polygrids |
| `polygrid.detail_render` | render_polygrids |
| `polygrid.geometry` | render_polygrids |
| `polygrid.tile_uv_align` | render_polygrids |
| `polygrid.atlas_utils` | render_globe_from_tiles, tile_uv_align |
| `polygrid.globe_export` | render_polygrids, render_globe_from_tiles |
| `polygrid.globe_renderer_v2` | render_polygrids (viewer), render_globe_from_tiles (viewer) |
| `polygrid.uv_texture` | tile_uv_align, detail_grid, tests |

| Module | Required by |
|--------|-------------|
| `polygrid.models` | core data types — used everywhere |
| `polygrid.polygrid` | core container — used everywhere |
| `polygrid.algorithms` | geometry, tile_data, detail_render, detail_terrain, heightmap |
| `polygrid.noise` | detail_render, detail_terrain, tile_detail, mountains |
| `polygrid.heightmap` | mountains, detail_terrain, tile_detail |
| `polygrid.detail_grid` | tile_detail, detail_terrain |
| `polygrid.builders` | detail_grid (build_pure_hex_grid, build_pentagon_centered_grid) |
| `polygrid.goldberg_topology` | detail_grid, __init__ |
| `polygrid.composite` | tile_detail, __init__ |
| `polygrid.assembly` | __init__, visualize (dead) |
| `polygrid.regions` | __init__, tests, terrain_patches (dead) |
| `polygrid.transforms` | __init__, regions, visualize (dead), rivers (dead), terrain_render (dead) |

---

## � TODO REMOVE — Dead modules (tagged, safe to delete)

| Module | Tag | Reason |
|--------|-----|--------|
| `cli.py` | ✅ **DELETED** | CLI wrapper around dead `io.py`; not used by any live script |
| `io.py` | ✅ **DELETED** | Old JSON I/O for standalone polygrids; only used by dead `cli.py` |
| `visualize.py` | ✅ **DELETED** | Old Phase 1–4 matplotlib visualisation; only used by dead `cli.py` |
| `pipeline.py` | ✅ **DELETED** | Old Phase 7F composable pipeline framework; no live callers |
| `rivers.py` | ✅ **DELETED** | Phase 7E river generation; only used by dead `pipeline.py` and `globe_terrain.py` |
| `terrain_render.py` | ✅ **DELETED** | Phase 7D overlay-based terrain rendering; no live callers |
| `terrain_patches.py` | ✅ **DELETED** | Phase 11B terrain distribution patches; no live callers |
| `region_stitch.py` | ✅ **DELETED** | Phase 11C stitched detail grids; only used by dead `globe_terrain.py` |
| `globe_terrain.py` | ✅ **DELETED** | Phase 11D globe-scale terrain; no live callers |
| `detail_terrain_3d.py` | ✅ **DELETED** | Phase 11A 3D-coherent terrain gen; no live callers |
| `render_enhanced.py` | ✅ **DELETED** | Phase 11E biome assignment / normal maps; only used by dead `texture_export.py` |
| `detail_perf.py` | ✅ **DELETED** | Phase 10F parallel/cached detail gen; no live callers |
| `diagnostics.py` | ✅ **DELETED** | Quality-gate diagnostics for standalone grids; no live callers |
| `biome_render.py` | ✅ **DELETED** | Phase 14B forest rendering; only used by dead biome modules |
| `biome_scatter.py` | ✅ **DELETED** | Phase 14A biome feature scattering; no live callers |
| `biome_continuity.py` | ✅ **DELETED** | Phase 14C cross-tile biome continuity; only used by dead `visual_cohesion.py` |
| `biome_pipeline.py` | ✅ **DELETED** | Phase 14D biome atlas pipeline; no live callers |
| `biome_topology.py` | ✅ **DELETED** | Phase 18C topology-aware biome rendering; no live callers |
| `tile_texture.py` | ✅ **DELETED** | Phase 16A full-slot tile texture rendering; no live callers |
| `texture_pipeline.py` | ✅ **DELETED** | Replaced by render_polygrids.py atlas builder; no live callers |
| `texture_export.py` | ✅ **DELETED** | Phase 18D KTX2/glTF texture export; no live callers |
| `apron_grid.py` | ✅ **DELETED** | Phase 18A apron grid construction; no live callers |
| `apron_texture.py` | ✅ **DELETED** | Phase 18B apron-aware texture rendering; no live callers |
| `ocean_render.py` | ✅ **DELETED** | Phase 17A ocean biome rendering; no live callers |
| `coastline.py` | ✅ **DELETED** | Phase 19 coastline transitions; no live callers |
| `globe_mesh.py` | ✅ **DELETED** | Old mesh bridge for models library renderer; no live callers |
| `visual_cohesion.py` | ✅ **DELETED** | Phase 18E visual cohesion validation; no live callers |

**27 dead modules deleted.**

---

## ✅ REFACTOR — Resolved

| Location | Status | Resolution |
|----------|--------|------------|
| `render_polygrids.py :: _load_env` | ✅ **DONE** | Extracted to `scripts/_script_utils.py`; both scripts now import from shared module |
| `render_globe_from_tiles.py :: _load_env` | ✅ **DONE** | Same — uses shared `_script_utils.load_env` |
| `render_polygrids.py :: presets dict` | ✅ **DONE** | Replaced with canonical `MOUNTAIN_RANGE`/`ALPINE_PEAKS`/`ROLLING_HILLS` from `polygrid.mountains` + `dataclasses.replace(preset, seed=seed)` |
| `render_globe_from_tiles.py :: presets dict` | ✅ **DONE** | Same canonical preset replacement |
| `render_polygrids.py :: _colour_debug_fn / _colour_debug_single_fn` | ✅ **DONE** | Extracted shared `_radial_debug_colour()` helper |
| `globe_renderer_v2.py` | ⏳ Deferred | 2500-line monolith — noted for future split (mesh, shaders, viewer) |
| `tile_uv_align.py` | ⏳ Deferred | 2780-line monolith — noted for future split (warp, atlas, debug) |
| `uv_texture.py :: _normalize_vec` | ✅ **RESOLVED** | Trivial 3-line helper; not worth shared-import coupling. Tag removed. |
| `uv_texture.py :: _find_polygon_corners` | ✅ **RESOLVED** | Private to `uv_texture.py` monolith; consolidation deferred to monolith split. Tag removed. |
| `regions.py` | ✅ REVIEWED | `Region` type used by `mountains.py`. Re-exported for library consumers. Kept. |
| `transforms.py` | ✅ REVIEWED | `Overlay`, `OverlayRegion` used by `regions.py`. Re-exported for library consumers. Kept. |
| `__init__.py` | ✅ REVIEWED | Pruned in prior cleanup. Dead module re-exports already removed. |

---

## ✅ REMOVE — Dead code in live files (resolved)

| Location | Status | Resolution |
|----------|--------|------------|
| `render_polygrids.py :: _SENTINEL_BG_RGB` | ✅ **DELETED** | Unused constant removed |
| `render_polygrids.py :: _avg_colour` | ✅ **DELETED** | Uncalled function removed |

---

## ✅ OPTIMISE — Performance (resolved)

| Location | Status | Resolution |
|----------|--------|------------|
| `render_polygrids.py :: _compute_global_hillshade` | ✅ **DONE** | `_main_polygon_cut` now builds composites first, passes them to `_compute_global_hillshade` — eliminates redundant `build_tile_with_neighbours` calls |
| `render_polygrids.py :: _render_analytical_to_png` | ⏳ Deferred | Per-face `contains_points` loop is O(faces × pixels); noted for future vectorisation with rasterio or GPU |

---

## ✅ REVIEW — Resolved

| Location | Status | Resolution |
|----------|--------|------------|
| `render_polygrids.py :: _render_patches_to_png` | ✅ **RESOLVED** | Kept — still used for `--renderer=matplotlib` path. Tag converted to NOTE comment. |
| `detail_render.py :: render_detail_texture_enhanced` | ✅ **RESOLVED** | Kept — public API, used by tests, re-exported in `__init__.py`. Tag removed. |

---

## 🟡 TODO REVIEW — Tests

| Test file | Status |
|-----------|--------|
| `test_atlas_seams.py` | ✅ Live — tests `_stitch_atlas_seams` in `tile_uv_align.py` |
| `test_core_topology.py` | ✅ Live — tests core `PolyGrid`, `builders`, `algorithms` |
| `test_corner_blend.py` | ✅ Live — tests `_blend_corner_junctions` in `tile_uv_align.py` |
| `test_detail_render.py` | ✅ Live — tests `BiomeConfig`, `detail_elevation_to_colour`, `render_detail_texture_enhanced` |
| `test_detail_terrain.py` | ✅ Live — tests boundary-aware terrain generation |
| `test_globe.py` | ✅ Live — tests `build_globe_grid`, `export_globe_payload`, `partition_voronoi` |
| `test_globe_renderer_v2.py` | ✅ Live — tests 3D renderer mesh building and shader setup |
| `test_grid_deformation.py` | ✅ Live — tests UV shape-matched grid deformation |
| `test_heightmap.py` | ✅ Live — tests noise field sampling and smoothing |
| `test_mountains.py` | ✅ Live — tests `generate_mountains` and preset configs |
| `test_noise.py` | ✅ Live — tests `fbm`, `ridged_noise`, `domain_warp` etc. |
| `test_tile_data.py` | ✅ Live — tests `TileDataStore`, `TileSchema`, serialisation |
| `test_tile_detail.py` | ✅ Live — tests `DetailGridCollection`, `build_tile_with_neighbours` |
| `test_uv_texture.py` | ✅ Live — tests `UVTransform`, `_find_polygon_corners`, UV projection |

All 14 test files exercise live modules. No test-only dead code found.

---

## 🟡 TODO REVIEW — Docs

| File | Status |
|------|--------|
| `docs/ARCHITECTURE.md` | References some removed modules (`render.py`) in layer diagram; layer table still accurate. Low-priority update. |
| `docs/MODULE_REFERENCE.md` | Line counts and test counts stale; module list includes removed files. Needs rewrite when cleanup stabilises. |
| `docs/JSON_CONTRACT.md` | Describes live payload format, still accurate. |
| `docs/RENDERING_ISSUES.md` | Documents sentinel bg, seam issues — still accurate for current pipeline. |
| `docs/TILE_TEXTURE_MAPPING.md` | Documents UV warp pipeline — still accurate. |
| `docs/CLEANUP_REVIEW.md` | ✅ This file — kept up to date. |
| `test_results.txt` | ✅ Up to date (605 tests / 14 files). |
| `TESTING.md` | ✅ Up to date. |
| `README.md` | ✅ Up to date. |

---

## Defaults Changed

| Flag | Old default | New default | Reason |
|------|-------------|-------------|--------|
| `--gutter` | `tile_size // 32` (16 at 512) | `4` | Matches proven rendering runs |
| `--pent-gutter` | Falls back to `--gutter` | Falls back to `--gutter` (4) | Same |
| `--hex-gutter` | Falls back to `--gutter` | Falls back to `--gutter` (4) | Same |
| `--no-neighbours` | `False` (off) | `True` (on) | Default behaviour now skips neighbour closure; use `--neighbours` to re-enable |

