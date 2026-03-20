# Code Cleanup Review

**Date:** 2026-03-20  
**Status:** ✅ All files tagged — ready for deletion pass.  
**Scope:** Everything not needed by the three live scripts:
- `scripts/render_polygrids.py`
- `scripts/render_globe_from_tiles.py`
- `scripts/debug_pipeline.py`

---

## 1. Live Dependency Tree

### Direct imports from the 3 live scripts:

| Module | Used by |
|--------|---------|
| `polygrid.globe` | render_polygrids, render_globe_from_tiles, debug_pipeline |
| `polygrid.mountains` | render_polygrids, render_globe_from_tiles |
| `polygrid.tile_data` | render_polygrids, render_globe_from_tiles |
| `polygrid.tile_detail` | render_polygrids, debug_pipeline |
| `polygrid.detail_terrain` | render_polygrids |
| `polygrid.detail_render` | render_polygrids, debug_pipeline |
| `polygrid.geometry` | render_polygrids, debug_pipeline |
| `polygrid.tile_uv_align` | render_polygrids, debug_pipeline |
| `polygrid.atlas_utils` | render_globe_from_tiles |
| `polygrid.globe_export` | render_polygrids, render_globe_from_tiles |
| `polygrid.globe_renderer` | render_globe_from_tiles (legacy v1, deprecated) |
| `polygrid.globe_renderer_v2` | render_globe_from_tiles |
| `polygrid.uv_texture` | debug_pipeline |

### Transitive dependencies (internal imports of the above):

| Module | Required by |
|--------|-------------|
| `polygrid.models` | core data types — used everywhere |
| `polygrid.polygrid` | core container — used everywhere |
| `polygrid.algorithms` | geometry, tile_data, detail_render, detail_terrain, heightmap |
| `polygrid.noise` | detail_render, detail_terrain, tile_detail, mountains |
| `polygrid.heightmap` | mountains, detail_terrain, tile_detail |
| `polygrid.detail_grid` | tile_detail, detail_terrain |
| `polygrid.globe_render` | globe_export (globe_to_colour_map) |
| `polygrid.regions` | mountains (Region type) |
| `polygrid.transforms` | regions (Overlay types) |

### Complete LIVE module set (24 modules):

```
models, polygrid, algorithms, geometry, noise, heightmap, 
detail_grid, tile_data, tile_detail, detail_terrain, detail_render,
globe, globe_export, globe_render, globe_renderer, globe_renderer_v2,
mountains, tile_uv_align, atlas_utils, uv_texture,
regions, transforms,
__init__
```

---

## 2. Scripts — Tagging

### ✅ KEEP (live)
- `scripts/render_polygrids.py`
- `scripts/render_globe_from_tiles.py`
- `scripts/debug_pipeline.py`

### 🔴 TODO REMOVE — Not used by live pipeline
- `scripts/demo_assembly.py` — early Phase 1–4 demo, uses assembly/visualize
- `scripts/demo_terrain.py` — Phase 7 demo, uses terrain_render/rivers
- `scripts/demo_mountains.py` — Phase 7 demo, standalone mountain vis
- `scripts/demo_regions.py` — Phase 5–6 demo, uses regions/visualize
- `scripts/demo_globe.py` — early globe vis using globe_render (flat map)
- `scripts/demo_globe_3d.py` — matplotlib 3D globe, uses globe_mesh (not v2)
- `scripts/demo_detail_globe.py` — Phase 10 demo, uses texture_pipeline/detail_perf
- `scripts/demo_phase18_globe.py` — Phase 18 demo, uses old apron/biome pipeline
- `scripts/demo_forest_globe.py` — Phase 14 demo, uses biome_pipeline
- `scripts/demo_ocean_globe.py` — Phase 17A demo, uses ocean_render pipeline
- `scripts/demo_coastline.py` — Phase 19 demo, uses coastline/apron pipeline
- `scripts/demo_cohesive_globe.py` — Phase 11C/D demo, uses region_stitch/globe_terrain
- `scripts/debug_globe_labels.py` — debug script for UV labels
- `scripts/debug_polygrids.py` — older debug script, replaced by debug_pipeline.py
- `scripts/validate_polygon_cut.py` — one-off validation script
- `scripts/view_globe_v3.py` — standalone viewer, uses render_enhanced/texture_pipeline
- `scripts/globe_generation.sh` — shell wrapper
- `scripts/run_fast.sh` — shell wrapper
- `scripts/archive/` — **entire directory** (36 diagnostic scripts from old iterations)

### 🟡 TODO REVIEW
- `scripts/run_tests.py` — useful test runner utility, but consider if pytest alone is sufficient

---

## 3. Source Modules — Tagging

### ✅ KEEP (in live dependency tree)
| Module | Why |
|--------|-----|
| `models.py` | Core data types |
| `polygrid.py` | Core container |
| `algorithms.py` | Face adjacency, ring_faces |
| `geometry.py` | face_center etc. |
| `noise.py` | fbm, domain_warp |
| `heightmap.py` | smooth_field |
| `detail_grid.py` | build_detail_grid |
| `tile_data.py` | TileDataStore |
| `tile_detail.py` | DetailGridCollection |
| `detail_terrain.py` | generate_all_detail_terrain |
| `detail_render.py` | BiomeConfig, hillshade, colour mapping |
| `globe.py` | build_globe_grid |
| `globe_export.py` | export_globe_payload |
| `globe_render.py` | globe_to_colour_map (used by globe_export) |
| `globe_renderer.py` | v1 legacy renderer (still reachable via --legacy-renderer) |
| `globe_renderer_v2.py` | Primary 3D renderer |
| `mountains.py` | MountainConfig, generate_mountains |
| `tile_uv_align.py` | Polygon-cut atlas, warp, corner matching |
| `atlas_utils.py` | fill_gutter, compute_atlas_layout |
| `uv_texture.py` | get_tile_uv_vertices, UV transforms |
| `regions.py` | Region type (used by mountains) |
| `transforms.py` | Overlay types (used by regions) |
| `__init__.py` | Package re-exports |

### 🟡 TODO REFACTOR — In live tree but has issues
| Module | Issue |
|--------|-------|
| `globe_renderer.py` | Deprecated v1 renderer. Only reachable via `--legacy-renderer` flag. Consider removing from `render_globe_from_tiles.py` entirely and keeping only v2. |
| `globe_render.py` | Only `globe_to_colour_map` is used (by `globe_export.py`). The rest (`render_globe_flat`, `render_globe_3d`, `globe_to_tile_colours`) is unused by live scripts. Consider inlining `globe_to_colour_map` into `globe_export.py`. |
| `regions.py` | Only `Region` type is used (by `mountains.py`). The full partitioning API (`partition_voronoi`, `partition_noise`, etc.) is not used by any live script. |
| `transforms.py` | Only `Overlay` and `OverlayRegion` types are used (by `regions.py`). The functions `apply_voronoi`, `apply_partition` are unused by live scripts. |
| `__init__.py` | Re-exports ~200 symbols from 30+ modules. Most are from dead modules. Needs major pruning once dead modules are removed. |

### 🔴 TODO REMOVE — Not in live dependency tree
| Module | What it was |
|--------|-------------|
| `assembly.py` | Phase 1–4 pent+hex assembly. Not used by globe pipeline. |
| `builders.py` | build_pure_hex_grid, build_pentagon_centered_grid — used only by assembly.py |
| `composite.py` | CompositeGrid/StitchSpec — the old stitching model, used by assembly/visualize |
| `visualize.py` | matplotlib panel rendering (render_exploded, render_stitched etc.) |
| `io.py` | load_json/save_json — not used by any live script |
| `cli.py` | CLI wrapper around io.py |
| `goldberg_topology.py` | Stand-alone goldberg builder — globe.py has its own via models library |
| `diagnostics.py` | Ring quality diagnostics — not used by live pipeline |
| `pipeline.py` | TerrainPipeline/MountainStep/RiverStep — not used by live scripts |
| `rivers.py` | River generation — not used by any live script |
| `terrain_render.py` | Old terrain overlay renderer (elevation_to_overlay etc.) |
| `detail_perf.py` | Parallel terrain gen / fast renderer / disk cache — not used by live scripts |
| `texture_pipeline.py` | Old texture atlas builder — replaced by render_polygrids.py |
| `render_enhanced.py` | Phase 11E biomes/normals — not used by live scripts |
| `detail_terrain_3d.py` | Phase 11A 3D terrain — not used by live scripts |
| `terrain_patches.py` | Phase 11B terrain patches — not used by live scripts |
| `region_stitch.py` | Phase 11C stitched detail grids — not used by live scripts |
| `globe_terrain.py` | Phase 11D globe terrain (mountains_3d, erosion) — not used by live scripts |
| `globe_mesh.py` | Bridge to models library meshes — not used by live scripts |
| `apron_grid.py` | Phase 18A apron grids — not used by live scripts |
| `apron_texture.py` | Phase 18B apron texture rendering — not used by live scripts |
| `biome_scatter.py` | Phase 14A feature scattering — not used by live scripts |
| `biome_render.py` | Phase 14B forest rendering — not used by live scripts |
| `biome_pipeline.py` | Phase 14D biome atlas pipeline — not used by live scripts |
| `biome_continuity.py` | Phase 14C cross-tile biome continuity — not used by live scripts |
| `biome_topology.py` | Phase 18C topology-aware biomes — not used by live scripts |
| `ocean_render.py` | Phase 17A ocean rendering — not used by live scripts |
| `coastline.py` | Phase 19 coastline transitions — not used by live scripts |
| `tile_texture.py` | Phase 16A full-slot tile texture — not used by live scripts |
| `texture_export.py` | Phase 18D KTX2/glTF export — not used by live scripts |
| `visual_cohesion.py` | Phase 18E visual validation — not used by live scripts |

---

## 4. Tests — Tagging

### ✅ KEEP — Tests for live modules
| Test file | Tests module(s) |
|-----------|-----------------|
| `test_core_topology.py` | models, polygrid, algorithms |
| `test_goldberg.py` | goldberg_topology (but see note below) |
| `test_tile_data.py` | tile_data |
| `test_noise.py` | noise |
| `test_heightmap.py` | heightmap |
| `test_mountains.py` | mountains |
| `test_globe.py` | globe, globe integration |
| `test_tile_detail.py` | tile_detail |
| `test_detail_terrain.py` | detail_terrain |
| `test_detail_render.py` | detail_render |
| `test_uv_texture.py` | uv_texture |
| `test_globe_renderer_v2.py` | globe_renderer_v2 |
| `test_grid_deformation.py` | detail_grid, tile_uv_align |
| `test_atlas_seams.py` | tile_uv_align |
| `test_corner_blend.py` | tile_uv_align, globe_renderer_v2 |
| `conftest.py` | shared fixtures |

### 🟡 TODO REVIEW — Tests for live modules but may overlap / be unnecessary
| Test file | Issue |
|-----------|-------|
| `test_goldberg.py` | Tests standalone `goldberg_topology.py` which may be removed. BUT also tests core goldberg embedding maths that `globe.py` depends on indirectly. Check which tests exercise code actually used by `globe.py`. |
| `test_globe.py` | Has pre-existing failures (3). Some tests exercise `pipeline.py`, `rivers.py` which are dead code. Audit individual test methods. |
| `test_globe_renderer_v2.py` | Has 1 pre-existing failure. Some tests reference `globe_renderer.py` (v1) and `texture_pipeline.py`. Audit individual test methods. |

### 🔴 TODO REMOVE — Tests for dead modules only
| Test file | Tests module(s) |
|-----------|-----------------|
| `test_assembly.py` | assembly, composite, builders |
| `test_stitching.py` | composite (stitch_grids etc.) |
| `test_macro_edges.py` | macro edges on assembly grids |
| `test_pentagon_centered.py` | builders (build_pentagon_centered_grid) |
| `test_transforms.py` | transforms (apply_voronoi, apply_partition) |
| `test_diagnostics.py` | diagnostics |
| `test_visualize.py` | visualize |
| `test_regions.py` | regions (partitioning functions) |
| `test_rivers.py` | rivers |
| `test_pipeline.py` | pipeline (TerrainPipeline) |
| `test_terrain_render.py` | terrain_render |
| `test_determinism.py` | builders (build_pentagon_centered_grid) |
| `test_detail_perf.py` | detail_perf |
| `test_detail_terrain_3d.py` | detail_terrain_3d |
| `test_terrain_patches.py` | terrain_patches |
| `test_globe_terrain.py` | globe_terrain |
| `test_region_stitch.py` | region_stitch |
| `test_render_enhanced.py` | render_enhanced |
| `test_texture_pipeline.py` | texture_pipeline |
| `test_phase13_rendering.py` | texture_pipeline, detail_perf |
| `test_apron_grid.py` | apron_grid |
| `test_apron_texture.py` | apron_texture |
| `test_biome_scatter.py` | biome_scatter |
| `test_biome_render.py` | biome_render |
| `test_biome_pipeline.py` | biome_pipeline |
| `test_biome_continuity.py` | biome_continuity |
| `test_biome_topology.py` | biome_topology |
| `test_ocean_render.py` | ocean_render |
| `test_coastline.py` | coastline |
| `test_tile_texture.py` | tile_texture |
| `test_texture_export.py` | texture_export |
| `test_visual_cohesion.py` | visual_cohesion |

---

## 5. Other Files — Tagging

### 🔴 TODO REMOVE
- `notebooks/analytical_fill.ipynb` — one-off analysis
- `notebooks/rendering_diagnostics.ipynb` — one-off analysis
- `schemas/globe.schema.json` — JSON schema for old globe format
- `schemas/polygrid.schema.json` — JSON schema for old polygrid format
- `examples/minimal_grid.json` — example for io.py (which is dead)
- `docs/POLYGRID_DISTORTION.md` — about old distortion issues
- `docs/POLYGRID_DISTORTION_FOLLOWUP.md` — follow-up on above
- `docs/RENDERING_ISSUES.md` — old rendering issue tracking
- `docs/TILE_TEXTURE_MAPPING.md` — about old tile_texture.py approach
- `exports/f3/` — old export artifacts

### 🟡 TODO REVIEW
- `docs/ARCHITECTURE.md` — likely needs update to reflect stripped-down codebase
- `docs/MODULE_REFERENCE.md` — will need heavy rewrite
- `docs/JSON_CONTRACT.md` — check if it describes the live payload format
- `docs/CLEANUP.md` — existing cleanup notes, merge with this review
- `test_results.md` — stale test timing data
- `TESTING.md` — testing guide, will need update
- `README.md` — will need update after cleanup

---

## 6. Potential Logic Issues (TODO REVIEW)

### 6.1 `globe_export.py` → `globe_render.py` dependency
`export_globe_payload` calls `globe_to_colour_map` from `globe_render.py`. This pulls in the entire `globe_render` module (with its matplotlib flat/3D renderers) just for a colour-ramp function. The colour-ramp logic should be inlined or extracted to a small utility module.

### 6.2 `mountains.py` → `regions.py` dependency
`mountains.py` imports `Region` from `regions.py` which in turn imports `transforms.py`. This creates a dependency chain `mountains → regions → transforms → geometry`. The live scripts only use `MountainConfig` and `generate_mountains`. Check if `Region` is actually used inside `generate_mountains` or if it's only used in the `generate_mountains_regional` variant.

### 6.3 `globe_renderer.py` v1 deprecation
`render_globe_from_tiles.py` still supports `--legacy-renderer` which imports `globe_renderer.py`. This module depends on the `models` library's old mesh API. Since it's already deprecated with a warning, consider removing the flag entirely.

### 6.4 `render_polygrids.py` has dead code paths
- `_main_neighbour_edges()` uses `get_neighbour_border_grid` — this is the old Phase 10 neighbour approach, replaced by stitched composites. The `--with-neighbour-edges` flag is labeled "Legacy" in the argparse help.
- `_main_simple()` — simple un-stitched rendering, may still be useful but is not the default path.
- `_render_tile()` — only called by `_main_simple` and `_main_neighbour_edges`, not by the default polygon-cut path.

### 6.5 `__init__.py` catches ImportError silently
The `try/except ImportError: pass` blocks in `__init__.py` mean missing dependencies produce zero feedback. After cleanup, these should be more explicit about what's optional.

### 6.6 `goldberg_topology.py` vs `globe.py`
`goldberg_topology.py` provides standalone goldberg grid building. `globe.py` builds on the `models` library's goldberg generator. It's unclear if `globe.py` uses anything from `goldberg_topology.py` or if they're fully independent implementations. If independent, `goldberg_topology.py` is dead code. If `globe.py` delegates to it, it's live.

---

## 7. Summary Statistics

| Category | Keep | Remove | Refactor/Review |
|----------|------|--------|-----------------|
| **Scripts** | 3 | 18+ archive | 1 |
| **Source modules** | 22 | 30 | 5 |
| **Test files** | 16 | 31 | 3 |
| **Other files** | — | 10 | 7 |

**Estimated removal: ~55% of the codebase.**

---

## 8. Recommended Execution Order

1. **Tag all files** with `# TODO REMOVE` / `# TODO REFACTOR` / `# TODO REVIEW` comments
2. **Delete dead scripts** (`scripts/demo_*`, `scripts/debug_polygrids.py`, `scripts/archive/`, etc.)
3. **Delete dead source modules** (30 modules listed above)
4. **Delete dead test files** (31 test files listed above)
5. **Clean up `__init__.py`** — remove all imports of dead modules
6. **Refactor `globe_render.py`** — inline `globe_to_colour_map` into `globe_export.py`
7. **Audit `mountains.py`** → `regions.py` dependency
8. **Remove `--legacy-renderer`** from `render_globe_from_tiles.py`
9. **Clean `render_polygrids.py`** — remove dead code paths if desired
10. **Update documentation** — README, ARCHITECTURE, MODULE_REFERENCE
11. **Run full test suite** to verify nothing breaks
