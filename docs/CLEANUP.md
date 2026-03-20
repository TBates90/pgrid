# Cleanup & Refactoring Tracker

Identified legacy code, dead modules, and refactoring opportunities.
Items are grouped by priority and effort.

---

## 1. Dead Modules — Safe to Delete

### 1.1 `src/polygrid/render.py` (20 lines)
- **Status**: Deprecated shim — emits `DeprecationWarning` and re-exports
  `render_png` from `visualize.py`.
- **Evidence**: No imports from any source file, script, or test.  Not in
  `__init__.py` imports.  Only exists as a compatibility bridge.
- **Action**: Delete the file and remove from `__all__` if listed.
- **Risk**: None — nothing imports it.

---

## 2. Diagnostic Scripts — Archive or Delete

The `scripts/` directory contains **32 one-off diagnostic scripts**
(~5,700 lines total) created during issue investigation.  These are
not part of the production pipeline and are not referenced by tests
or other scripts.

### 2.1 `diag_i6_*` (6 scripts, ~1,225 lines)
Issue 6 diagnostics (edge mismatch / pentagon warp identity).
The underlying issues have been resolved.

### 2.2 `diag_i7_*` (17 scripts, ~2,520 lines)
Issue 7 diagnostics (seam checking, pixel tracing, warp tracing).
The underlying issues have been resolved.

### 2.3 Miscellaneous `diag_*` (9 scripts, ~965 lines)
- `diag_apron_reflect.py`, `diag_apron_reflect_all.py` — apron reflection debugging
- `diag_corner_pairing.py` — corner matching investigation
- `diag_edge_match.py` — edge matching investigation
- `diag_hex_reflected_detail.py` — hex reflection check
- `diag_pent_overlay.py`, `diag_pent_stitch.py` — pentagon overlay/stitch debugging
- `diag_sector_det.py` — sector determinant check
- `diag_winding.py`, `diag_winding_fix_test.py` — winding investigation

### Action
Move all `diag_*` scripts into `scripts/archive/` (or delete outright).
The two *retained* diagnostic tools (`debug_pipeline.py`,
`validate_polygon_cut.py`) are production-quality and should stay.

---

## 3. Legacy Script Consolidation

### 3.1 `scripts/view_globe.py` → superseded by `view_globe_v3.py`  ✅ DONE
- Deleted.  `view_globe_v3.py` already provides `--no-bloom`,
  `--no-atmosphere`, `--no-water` for a simple view.

### 3.2 `scripts/demo.py` (25 lines)
- Trivial script that loads `minimal_grid.json` and prints vertex/edge/face
  counts.  Superseded by the CLI (`polygrid info`) and richer demo scripts.
- **Action**: Delete.

### 3.3 `scripts/render_validation.py` (27 lines)
- Renders two static PNGs for manual inspection.  Superseded by the test
  suite and `debug_pipeline.py`.
- **Action**: Delete.

### 3.4 `scripts/angle_diagnostics.py` (65 lines)
- One-off angle analysis of pentagon grids using the deprecated `"angle"`
  embed mode.
- **Action**: Delete (the embed mode it tests is no longer the default).

---

## 4. Renderer Duplication

### 4.1 `globe_renderer.py` (v1) removal  ✅ DONE
- v1 renderer (`globe_renderer.py`) deleted.
- `--legacy-renderer` and `--v2` flags removed from `render_globe_from_tiles.py`.
- `globe_renderer_v2.py` is now the only renderer, used by default.
- v1 tests removed from `test_globe.py` (`TestGlobeRenderer`) and
  `test_globe_renderer_v2.py` (`TestTexturedRenderer`).
- `__init__.py` no longer re-exports v1 symbols.

### 4.2 `globe_render.py` inlined into `globe_export.py`  ✅ DONE
- `globe_to_colour_map` and its colour-ramp helpers (`_lerp_colour`,
  `_ramp_satellite`, `_ramp_topo`, `_RAMPS`) inlined into `globe_export.py`.
- `globe_render.py` deleted (removed unused `render_globe_flat`,
  `render_globe_3d`, `globe_to_tile_colours`).
- Tests for `globe_to_colour_map` moved to import from `globe_export`;
  tests for dead functions removed.
- `__init__.py` re-exports `globe_to_colour_map` from `globe_export`.

### 4.2 Legacy shaders in `globe_renderer_v2.py`
- `_V2_VERTEX_SHADER` / `_V2_FRAGMENT_SHADER` (lines 1852–1895) are the
  old 8-float vertex shaders, kept for `get_v2_shader_sources()`.
- `get_v2_shader_sources()` is exported in `__init__.py` / `__all__` but
  has **zero callers** outside the module itself (used only by the internal
  `_build_v2_shader_program`).
- **Action**: Remove the public export of `get_v2_shader_sources` from
  `__init__.py` and `__all__`.  Keep the shaders as internal helpers.

---

## 5. Legacy Fallback Code in Production Modules

### 5.1 `detail_terrain.py` — legacy corner detection (~60 lines)
- `_compute_tutte_edge_angles()` has a "legacy fallback" path
  (lines 209–260) that detects polygon corners by geometric clustering
  when `corner_vertex_ids` is not supplied.
- The Goldberg pipeline *always* sets `corner_vertex_ids` in grid
  metadata via `goldberg_topology.py`, so the fallback is never
  triggered in production.
- **Action**: Add a deprecation warning to the fallback path now.
  Remove the fallback in a future release once confirmed unused by any
  downstream consumer.

### 5.2 `detail_terrain.py` — legacy uniform-mean elevation fallback
- Line 524: `"No edge mapping — fall back to uniform mean (legacy behaviour)"`.
- Same situation: the edge mapping is always available in the Goldberg
  pipeline.
- **Action**: Add a deprecation warning.  Remove later.

### 5.3 `tile_uv_align.py` — legacy single-affine warp fallback
- Line 1499: `"Fallback: single-affine warp (legacy)"`.
- The piecewise warp is the production path.  The single-affine fallback
  exists as a safety net.
- **Action**: Keep for now (it's a genuine safety net), but add a
  warning log when it triggers so we know if it's ever hit.

---

## 6. `__init__.py` Hygiene

### 6.1 Module is 858 lines
- The flat re-export style works but is fragile — every new public symbol
  must be added in two places (import + `__all__`).
- **Action (future)**: Consider switching to explicit sub-package imports
  (e.g. `from polygrid.terrain import MountainConfig`) and a generated
  `__all__`.  Not urgent.

### 6.2 Missing `__all__` entries
- `validate_globe_payload` is imported (line 161) but not in `__all__`.
- **Action**: Add it, or remove the import if it's only used internally.

---

## 7. Test Housekeeping

### 7.1 `test_determinism.py` (13 lines)
- Extremely thin — likely a placeholder.  Check if it provides value
  beyond what other tests cover.
- **Action**: Review; expand or merge into another test file.

### 7.2 `test_globe_renderer_v2.py` (3,091 lines)
- The largest test file by far.  May contain redundant or
  over-specified tests from incremental development.
- **Action (future)**: Audit for redundancy; consolidate where
  possible.

---

## 8. Export Artefacts

### 8.1 `exports/` (40 subdirectories, ~697 MB)
- Already `.gitignore`d, so not a repo-size problem.
- Many directories are one-off debug outputs (`colour_debug_fix`,
  `colour_debug_NO_ROTATION`, `f3_seam_test`, `pent_debug`, etc.).
- **Action**: Prune locally.  Consider adding a `make clean-exports`
  target or a note in README about which directories are canonical
  outputs vs debug artefacts.

---

## Execution Plan

| Phase | Items | Effort | Impact | Status |
|-------|-------|--------|--------|--------|
| **A — Quick wins** | 1.1, 3.2, 3.3, 3.4, 4.2, 6.2 | ~30 min | Remove dead code, fix `__all__` | ✅ Done |
| **B — Diag archive** | 2.1, 2.2, 2.3 | ~15 min | Remove ~5,700 lines of stale scripts | ✅ Done |
| **C — Script consolidation** | 3.1 | ~15 min | Delete v1 viewer | ✅ Done |
| **D — Legacy fallbacks** | 5.1, 5.2, 5.3 | ~30 min | Add deprecation warnings | ✅ Done |
| **E — Renderer migration** | 4.1 | ~1 hr | Migrate script callers to v2, deprecate v1 module | ✅ Done |
| **F — Structural** | 6.1, 7.1, 7.2 | ~4+ hrs | `__init__.py` refactor, test audit | Remaining |
