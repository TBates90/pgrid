# Code Cleanup Review

### Cleanup Review Prompt

Review the pgrid codebase. Tag code using the following tags:

 # TODO REFACTOR
 # TODO REMOVE
 # TODO REVIEW
 # TODO OPTIMISE

Review and update the CLEANUP_REVIEW.md document.

### Key Imports

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
| `polygrid.globe_renderer_v2` | render_globe_from_tiles |
| `polygrid.uv_texture` | debug_pipeline |

| Module | Required by |
|--------|-------------|
| `polygrid.models` | core data types — used everywhere |
| `polygrid.polygrid` | core container — used everywhere |
| `polygrid.algorithms` | geometry, tile_data, detail_render, detail_terrain, heightmap |
| `polygrid.noise` | detail_render, detail_terrain, tile_detail, mountains |
| `polygrid.heightmap` | mountains, detail_terrain, tile_detail |
| `polygrid.detail_grid` | tile_detail, detail_terrain |
| `polygrid.regions` | terrain_patches, pipeline (RegionMap + partitioning) |
| `polygrid.transforms` | regions, terrain_render, rivers, visualize (Overlay types) |

---

### 🟡 TODO REFACTOR — In live tree but has issues
| Module | Issue |
|--------|-------|
| `regions.py` | ✅ REVIEWED — `Region` type used by `mountains.py` (via `models.py`). `RegionMap`, `partition_flood_fill`, `partition_noise` used by `terrain_patches.py`. `partition_voronoi` used in `test_globe.py`. The remaining functions (`partition_angular`, `assign_field`, `assign_biome`, `regions_to_overlay`, `validate_region_map`) are part of the public API for library consumers. All kept. |
| `transforms.py` | ✅ REVIEWED — `Overlay`, `OverlayRegion` used by `regions.py`, `visualize.py`, `terrain_render.py`, `rivers.py`. `OverlayPoint`, `OverlaySegment` used by `visualize.py`. `apply_voronoi` used by `cli.py`. `apply_partition` removed from `__init__.py` re-exports (zero callers); kept in module for library consumers who import directly. |
| `__init__.py` | ✅ DONE — Pruned from ~200 to ~95 re-exported symbols. Removed bulk re-exports for `globe_renderer_v2`, `uv_texture`, `tile_uv_align`, `atlas_utils`. Added `ImportWarning` for missing `globe`/`models` optional dependency (was silent `pass`). Specialised subsystems now imported directly from their modules. |

### 🟡 TODO REVIEW — Tests for live modules but may overlap / be unnecessary
| Test file | Issue |
|-----------|-------|
| `test_goldberg.py` | ✅ REVIEWED — File does not exist (removed in prior cleanup). `goldberg_topology.py` is exercised indirectly via `build_pentagon_centered_grid` in `test_mountains.py` and detail-grid tests. Standalone `goldberg_face_count`, `goldberg_embed_tutte`, and `goldberg_optimise` could benefit from targeted tests but are not blocking. |
| `test_determinism.py` | ✅ REVIEWED — File does not exist (removed in prior cleanup). |

### 🟡 TODO REVIEW - docs
- `docs/ARCHITECTURE.md` — references some removed modules (`render.py`) in layer diagram; layer table still accurate. Low-priority update.
- `docs/MODULE_REFERENCE.md` — line counts and test counts stale; module list includes removed files. Needs rewrite when cleanup stabilises.
- `docs/JSON_CONTRACT.md` — describes live payload format, still accurate.
- `docs/CLEANUP.md` — ✅ kept up to date with all cleanup actions.
- `test_results.txt` — ✅ UPDATED (was 1,478 tests/47 files, now 605 tests/14 files).
- `TESTING.md` — ✅ UPDATED (removed references to deleted `run_fast.sh` and `run_tests.py` scripts, updated test counts and instructions).
- `README.md` — ✅ UPDATED (removed stale `TASKLIST.md` reference, updated test counts).

---

