# PolyGrid Task List

Active and upcoming phases.

---

## Goals

1. **`models` and `pgrid` as proper installable Python packages** — clean `pyproject.toml`, correct `src/` layout metadata, editable installs that work from any consuming project.
2. **`pgrid` integrated into `playground`** — pgrid generates terrain data and texture atlases consumed by the playground world-builder; pgrid's globe renderer code is retired in favour of playground's existing OpenGL pipeline.
3. **Legacy cleanup in `playground`** — remove dead approaches (hexgrid dependency, graph-seeded tile geometry, legacy UV-sphere remnants) and replace with the clean pgrid + models pipeline.
4. **Clear separation of concerns across the three repos** — `models` owns Goldberg polyhedron geometry, `pgrid` owns 2D topology/terrain/texture generation, `playground` owns the application UI, OpenGL rendering, persistence, and world-building workflows.
5. **Tile debug panel in `playground`** — selecting a tile in the app surfaces rich debug info: pgrid face data, biome, elevation, region, neighbours, atlas UV coords, and models-level geometry.
6. **Existing playground planet features preserved** — tile selection, region painting, planet glow/sizing, biome textures, orbit cameras, and all current UX remain functional throughout the migration.

---

## Legend

- ✅ Done
- 🔲 To do
- 🔶 In progress / partially done

---

## Ongoing — Code Quality & Refactoring

- [ ] **CI pipeline** — set up GitHub Actions for `pytest` + linting on push across all three repos
- [ ] **Design patterns** — Strategy / Pipeline / Observer / Repository as terrain algorithms grow

---

## Phase 30 — Package Hygiene & Installability

Make `models` and `pgrid` proper, independently installable Python packages so `playground` (and any future consumer) can depend on them via `pip install -e`.

### 30A — `models` package cleanup

- [x] **30A.1** — Review `models/pyproject.toml`: confirm `[tool.setuptools.packages.find]` covers the `models/` package tree correctly (currently `where = ["."]`, `include = ["models*"]` — looks correct). ✅
- [x] **30A.2** — Add a `models/models/py.typed` marker for downstream type-checkers. ✅
- [x] **30A.3** — Ensure `models` exports a clean public API from `__init__.py` — audit what `playground` and `pgrid` actually import and make those the documented surface. ✅ (audit complete; API doc created)
- [x] **30A.4** — Pin minimum dependency versions in `pyproject.toml` (`numpy>=1.26`, `PyOpenGL>=3.1`) and verify editable install from a clean venv: `pip install -e ../models`. ✅ (already pinned; install verified)
- [x] **30A.5** — Add a brief `docs/API.md` listing public modules/classes consumed by `pgrid` and `playground`. ✅

### 30B — `pgrid` package cleanup

- [x] **30B.1** — Fix `pyproject.toml`: add `[tool.setuptools.packages.find]` with `where = ["src"]`, `include = ["polygrid*"]` so the `src/` layout is discovered correctly. ✅
- [x] **30B.2** — Rename the PyPI package from `polygrid` to `pgrid` (or keep `polygrid` as the import name with `pgrid` as the distribution name) — decide and document the convention. Currently `import polygrid` works but the repo is called `pgrid`. ✅ **Decision:** keep `polygrid` as both dist + import name; repo dir stays `pgrid`.
- [x] **30B.3** — Move rendering-only dependencies (`matplotlib`, `Pillow`, `pyglet`) into optional extras; keep the core package zero-dependency for library consumers. ✅ (`render`, `terrain`, `demo` extras)
- [x] **30B.4** — Add `[project.optional-dependencies] terrain = ["opensimplex>=0.4", "numpy>=1.26", "scipy>=1.12", "Pillow>=10"]` for the full terrain+texture pipeline. ✅
- [x] **30B.5** — Declare explicit dependency on `models` in the `globe` extra: `globe = ["models>=0.1.1"]`. ✅ (already present)
- [x] **30B.6** — Verify editable install from a clean venv: `pip install -e "../pgrid[globe,terrain,render]"`. ✅
- [x] **30B.7** — Add `src/polygrid/py.typed` marker. ✅

### 30C — `playground` dependency wiring

- [x] **30C.1** — Replace `requirements.txt` entries: swap `-e ../hexgrid` for `-e ../pgrid[globe,terrain]` and keep `-e ../models`. ✅
- [ ] **30C.2** — Remove the `hexgrid` dependency entirely (adapter, texture baker, and any other references). pgrid replaces this.
- [x] **30C.3** — Verify `playground` can `import polygrid` and `import models` after the editable installs. ✅
- [x] **30C.4** — Update `.env` / `.vscode/settings.json` if `PYTHONPATH` hacks existed for the old layout. ✅ (none found — editable installs are sufficient)

---

## Phase 31 — Separation of Concerns Audit

Review the boundaries between the three repos and relocate code that is in the wrong place.

### 31A — `models` boundary

- [ ] **31A.1** — Audit `models/models/rendering/opengl.py` — if it contains GL helpers only used by `playground`, move them to `playground/opengl/`.
- [ ] **31A.2** — Audit `models/models/objects/render_helpers.py` — relocate playground-specific rendering utilities.
- [ ] **31A.3** — Confirm `models` has **no** dependency on `pgrid` or `playground`. It should be a leaf package.
- [ ] **31A.4** — Document the `models` public contract: Goldberg geometry (tiles, polyhedron, slug generation, mesh building, layout helpers) and core primitives (mesh, transform, colour). Nothing else.

### 31B — `pgrid` boundary

- [ ] **31B.1** — Audit pgrid's `globe_renderer.py` and `globe_renderer_v2.py` — these are standalone OpenGL renderers that duplicate what `playground` already does better. Mark them as **standalone demo scripts only** (not part of the library API); move to `scripts/` or gate behind an optional `[demo]` extra.
- [ ] **31B.2** — Audit pgrid's `globe_mesh.py` — if it builds `ShapeMesh` objects for the 3D renderer, decide whether playground should consume this or keep its own `PlanetTileController` mesh pipeline.
- [ ] **31B.3** — Confirm pgrid's library surface is: core topology, grid building, terrain generation, detail grids, texture atlas building, UV alignment, and data export (`globe_export`, `tile_data`). Rendering modules are optional extras, not core API.
- [ ] **31B.4** — Ensure pgrid scripts (`render_polygrids.py`, `render_globe_from_tiles.py`) remain functional as standalone demo/diagnostic tools but are not imported by `playground`.
- [ ] **31B.5** — Review pgrid's `globe_export.py` output format — this is the main data contract between pgrid and playground. Document the JSON schema (`globe_payload.json`, `metadata.json`, `uv_layout.json`).

### 31C — `playground` boundary

- [ ] **31C.1** — Confirm playground never imports pgrid rendering modules (only data/topology/terrain/texture layers).
- [ ] **31C.2** — Confirm playground never imports `models.rendering` (only `models.core` and `models.objects`).
- [ ] **31C.3** — Document which pgrid modules playground will depend on: `polygrid.globe`, `polygrid.tile_data`, `polygrid.globe_export`, `polygrid.tile_detail`, `polygrid.detail_terrain`, `polygrid.detail_render`, `polygrid.tile_uv_align`, `polygrid.uv_texture`, `polygrid.mountains`, `polygrid.noise`.
- [ ] **31C.4** — Document which `models` modules playground depends on: `models.objects.goldberg`, `models.core.mesh`, `models.core.geometry`.

---

## Phase 32 — Legacy Cleanup in Playground

Prune dead code and outdated approaches from playground so the pgrid integration has a clean foundation.

### 32A — Remove hexgrid dependency

- [ ] **32A.1** — Delete `controllers/hexgrid_adapter.py`.
- [ ] **32A.2** — Delete `controllers/hexgrid_texture_baker.py`.
- [ ] **32A.3** — Delete `controllers/hexgrid_spec_sequence.py`.
- [ ] **32A.4** — Delete `controllers/hexgrid_region_stitching.py`.
- [ ] **32A.5** — Delete `controllers/planet_hexgrid_service.py`.
- [ ] **32A.6** — Delete `controllers/region_hexgrid_service.py`.
- [ ] **32A.7** — Remove `-e ../hexgrid` from `requirements.txt`.
- [ ] **32A.8** — Grep the entire playground tree for remaining `hexgrid` references and remove/replace them. Update any tests that exercised hexgrid paths.
- [ ] **32A.9** — Remove the hexgrid UV rotation slider and related code from `DebugInfoPanel`.

### 32B — Remove graph-seeded tile geometry path

- [ ] **32B.1** — Audit `controllers/graph_controller.py` — identify which graph-tile lookups are still needed (region painting, selections) vs. which are legacy (mesh topology).
- [ ] **32B.2** — Audit `controllers/tile_geometry_provider.py` — determine if `TileGeometrySnapshot` is still needed or if pgrid + models can serve the same data.
- [ ] **32B.3** — Audit `persistence/graph/runtime.py` — identify graph-tile seeding code that can be removed once pgrid provides tile topology.
- [ ] **32B.4** — Remove the graph-tile seeding CLI (`persistence.graph.seed --layout`) if no longer needed.
- [ ] **32B.5** — Replace `TileGeometryProvider` with a thin wrapper over `TileTopologyService` + pgrid data where possible.

### 32C — Remove legacy renderer remnants

- [ ] **32C.1** — Delete `opengl/rendering/planet_surface_controller.py` (already a stub raising `ImportError`).
- [ ] **32C.2** — Delete `opengl/rendering/icosahedron/controller.py` (already a stub raising `ImportError`).
- [ ] **32C.3** — Check if `opengl/rendering/icosahedron/geometry.py` is still referenced — if not, delete the entire `icosahedron/` directory.
- [ ] **32C.4** — Audit `opengl/rendering/planet_renderer.py` for dead code paths that reference the old UV-sphere or atlas approaches.

### 32D — General pruning

- [ ] **32D.1** — Delete `biome_debug_main.py` and `tmp_debug_astral.py` from the playground root (temp debug scripts).
- [ ] **32D.2** — Audit `scripts/` for stale one-off scripts.
- [ ] **32D.3** — Clean up `__pycache__` references in `.gitignore`; ensure no compiled files are tracked.
- [ ] **32D.4** — Audit `Orientation and Refactoring Task List.md`, `refactor_ico_controller.md`, `planet_rendering_plan.md`, `llm_plan.md`, `travel_notes.txt`, `world_buiilding_ux.md` — archive or delete any that are fully superseded by this tasklist and the design docs.

---

## Phase 33 — pgrid Integration: Terrain Data Pipeline

Wire pgrid's terrain generation into playground's planet pipeline so terrain data (elevation, biome seeds, detail grids) flows from pgrid into the existing renderer.

### 33A — Planet terrain generation service

- [ ] **33A.1** — Create `controllers/planet_terrain_service.py` in playground. This service wraps pgrid's terrain pipeline:
  - Accepts a layout slug (e.g. `gb3`) + seed + preset.
  - Calls `polygrid.globe.build_globe_grid(frequency)` to get a `PolyGrid` globe.
  - Calls `polygrid.mountains.generate_mountains()` for elevation data.
  - Calls `polygrid.tile_detail.DetailGridCollection.build()` for sub-tile detail grids.
  - Calls `polygrid.detail_terrain.generate_all_detail_terrain()` for boundary-aware detail.
  - Returns a structured payload (globe grid, tile data store, detail grid collection).
- [ ] **33A.2** — Map playground's layout slugs (`gb3`–`gb7`) to pgrid frequencies (`3`–`7`). Create a shared lookup in the service.
- [ ] **33A.3** — Extend the `Planet` model to store terrain generation parameters: `terrain_seed`, `terrain_preset`, `terrain_detail_rings`. Default to `seed=42`, `preset=mountain_range`, `detail_rings=4`.
- [ ] **33A.4** — Wire the terrain service into `PlanetController`: when a planet is created or its terrain params change, regenerate terrain data. Cache the result per `(layout, seed, preset, detail_rings)` tuple.
- [ ] **33A.5** — Emit `EventTopic.PLANET_TERRAIN_GENERATED` when terrain data is ready, carrying the pgrid data payload so downstream consumers (texture builder, debug panel) can react.

### 33B — Tile data bridge

- [ ] **33B.1** — Create `controllers/pgrid_tile_bridge.py` that translates between pgrid's `TileDataStore` face IDs (`t0`, `t1`, …) and playground's Goldberg tile slugs (`freq:f<face>:i-j-k`). The models library's `GoldbergPolyhedron` provides the mapping.
- [ ] **33B.2** — Extend `PlanetTileSnapshot` to include pgrid-sourced per-tile fields: `elevation`, `terrain_preset`, `detail_ring_count`.
- [ ] **33B.3** — Enrich the `PlanetSessionState` / `PlanetSnapshot` with pgrid terrain metadata so the debug panel and renderers have access.
- [ ] **33B.4** — Update `TileTopologyService` to optionally source neighbour data from the pgrid globe grid (which has the same topology as the models polyhedron, but with additional terrain metadata).

---

## Phase 34 — pgrid Integration: Texture Atlas Pipeline

Use pgrid's polygon-cut texture atlas as the source for planet tile textures in playground's OpenGL renderer.

### 34A — Atlas generation integration

- [ ] **34A.1** — Create `controllers/planet_atlas_service.py` in playground. This service:
  - Takes the terrain data from Phase 33 + biome assignments from region painting.
  - Calls pgrid's stitched tile renderer + `build_polygon_cut_atlas()` to produce `atlas.png` + `uv_layout.json`.
  - Stores the atlas under `artifacts/worlds/<world>/planets/<planet>/atlas/`.
  - Emits `EventTopic.PLANET_ATLAS_GENERATED`.
- [ ] **34A.2** — Define the atlas cache contract: when to regenerate (terrain params changed, biome assignments changed, layout changed) vs. reuse cached atlas.
- [ ] **34A.3** — Wire atlas generation into the planet creation / terrain update flow so an initial atlas is produced on planet creation.
- [ ] **34A.4** — Support incremental atlas updates: when a single tile's biome changes (region painting), re-render only that tile + its neighbours and patch the atlas in-place rather than regenerating the entire thing.

### 34B — Renderer integration

- [ ] **34B.1** — Update `PlanetTileController` to accept a pgrid-generated atlas texture + UV layout instead of (or alongside) per-tile biome textures.
- [ ] **34B.2** — Add an atlas-sampling path to the planet tile shader: when an atlas is bound, sample from the atlas using the `uv_layout.json` coordinates for the current tile instead of the per-tile biome texture.
- [ ] **34B.3** — Keep the existing per-tile biome texture path as a fallback for planets without pgrid terrain (backwards compatibility).
- [ ] **34B.4** — Wire `PLANET_ATLAS_GENERATED` events into `PlanetRenderer.update_planets()` so the renderer binds the new atlas texture when it arrives.
- [ ] **34B.5** — Ensure atlas UVs align correctly with the models library's tile mesh UVs. Both pgrid and models derive UVs from `GoldbergTile.uv_vertices`, so they should match — but verify with a visual test.

### 34C — Biome-aware terrain rendering

- [ ] **34C.1** — Extend pgrid's `BiomeConfig` / `detail_render` to accept playground's biome palette definitions (from `BiomeTileRegistry` / `BiomeSurfaceProfile`).
- [ ] **34C.2** — Create an adapter that converts playground's `Biome` model + palette into pgrid's `BiomeConfig` so atlas textures use the same colours as the existing per-tile renderer.
- [ ] **34C.3** — Support per-tile biome overrides in the atlas: when region painting assigns a biome to a tile, the atlas re-renders that tile with the assigned biome's palette.
- [ ] **34C.4** — Ensure coastline / ecotone effects at biome boundaries are preserved (pgrid's stitched rendering handles boundary continuity; verify this still works with mixed biomes).

---

## Phase 35 — Preserve & Migrate Existing Planet Features

Ensure all current playground planet functionality remains working throughout the migration.

### 35A — Tile selection

- [ ] **35A.1** — Audit `opengl/selection/picking.py` + `PlanetRenderer` tile picking. Confirm the ray-sphere → tile-slug resolution still works with pgrid-generated data.
- [ ] **35A.2** — Ensure `TileTopologyService.tile_slug_for_direction()` continues to work (it uses models' `GoldbergPolyhedron` directly — no pgrid dependency).
- [ ] **35A.3** — Verify tile highlight rendering (`PlanetTileController._highlight_tiles`) works identically whether textures come from the atlas or per-tile biome textures.

### 35B — Region painting

- [ ] **35B.1** — Verify `RegionPaintingController` brush operations still work. The brush uses tile slugs and `TileTopologyService` for neighbour expansion — no pgrid dependency in the hot path.
- [ ] **35B.2** — When a region paint stroke completes (`REGION_TILES_CHANGED`), trigger an incremental atlas update (Phase 34A.4) so the painted tiles reflect the new biome immediately.
- [ ] **35B.3** — Ensure `REGION_SUBMIT` → `persist_active_region()` flow still works and triggers a full atlas regeneration if needed.
- [ ] **35B.4** — Verify region tile highlights (union of all painted regions) still render correctly on top of the atlas-textured planet.

### 35C — Planet visuals (glow, sizing, orbit)

- [ ] **35C.1** — Confirm `Planet.size` slider still controls the planet's rendered radius. This is purely a `PlanetRenderer` concern and should be unaffected by pgrid integration.
- [ ] **35C.2** — Confirm `Planet.color` primary tint still applies as a global colour modulation. May need to blend with atlas colours.
- [ ] **35C.3** — Confirm `Planet.glow` / atmosphere rendering is unaffected.
- [ ] **35C.4** — Confirm planet orbit distance, layout selection (`gb3`–`gb7`), and system-level positioning are unaffected.

### 35D — Planet layout switching

- [ ] **35D.1** — When a planet's layout slug changes (e.g. `gb3` → `gb5`), regenerate terrain data + atlas via the terrain service (Phase 33) and rebind the renderer.
- [ ] **35D.2** — Verify that switching layouts doesn't break tile selections, region painting state, or biome assignments (these are slug-based and will need remapping).
- [ ] **35D.3** — Consider whether layout changes should preserve or reset terrain / region state (user-facing design decision).

---

## Phase 36 — Tile Debug Panel

Enrich the existing `DebugInfoPanel` with per-tile information when a tile is selected in the viewport.

### 36A — Debug data collection

- [ ] **36A.1** — When a tile is picked in the viewport, emit `EventTopic.TILE_SELECTED` carrying the tile slug.
- [ ] **36A.2** — Create `controllers/tile_debug_service.py` that assembles a `TileDebugPayload` from multiple sources:
  - **models**: tile slug, polygon index, vertex count, face normal, tangent, neighbour slugs, 3D center position.
  - **pgrid**: pgrid face ID (`t0`, `t1`, …), elevation, terrain preset contribution, detail grid face count, atlas UV rect.
  - **playground**: assigned biome slug, region ID, hexgrid metadata (if any), render texture handle, highlight state.
- [ ] **36A.3** — Cache the debug payload per tile so repeated clicks don't recompute.

### 36B — Debug panel UI

- [ ] **36B.1** — Extend `DebugInfoPanel` with a collapsible "Selected Tile" section that appears when a tile is selected.
- [ ] **36B.2** — Display **identity**: tile slug, pgrid face ID, polygon index.
- [ ] **36B.3** — Display **geometry**: 3D center, face normal, vertex count, neighbour count + slugs.
- [ ] **36B.4** — Display **terrain**: elevation value, terrain preset, detail ring count, detail sub-face count.
- [ ] **36B.5** — Display **biome/region**: assigned biome slug, region ID, region name.
- [ ] **36B.6** — Display **atlas**: UV rect (u_min, v_min, u_max, v_max), atlas slot index.
- [ ] **36B.7** — Display **render**: texture source (atlas vs. per-tile), highlight state, LOD level.
- [ ] **36B.8** — Clear the tile section when the selection is deselected or when switching away from planet/region mode.

### 36C — Wiring

- [ ] **36C.1** — Wire `TILE_SELECTED` → `TileDebugService` → `DebugInfoPanel` refresh.
- [ ] **36C.2** — Ensure the debug panel updates are throttled (no per-frame updates; only on selection change).
- [ ] **36C.3** — Gate the tile debug section behind `DEBUG_MODE=1` so production builds don't pay the cost.

---

## Phase 37 — Refactoring & Polish

Final cleanup pass across all three repos after integration is complete.

### 37A — Design doc updates

- [ ] **37A.1** — Update `playground/design.md` to reflect the pgrid integration: new services, data flow, removed legacy modules.
- [ ] **37A.2** — Update `playground/opengl/design.md` to describe the atlas-based texture pipeline.
- [ ] **37A.3** — Update `pgrid/docs/ARCHITECTURE.md` to clarify pgrid's role as a library consumed by playground (not a standalone renderer).
- [ ] **37A.4** — Update `models/README.md` to note its role as the shared geometry primitive layer.
- [ ] **37A.5** — Update `playground/AGENTS.md` with the new phase summary.

### 37B — Test coverage

- [ ] **37B.1** — Add integration tests in playground that exercise the full pipeline: create planet → generate terrain → build atlas → render tiles → pick tile → verify debug payload.
- [ ] **37B.2** — Add unit tests for `PlanetTerrainService`, `PlanetAtlasService`, `PgridTileBridge`, `TileDebugService`.
- [ ] **37B.3** — Ensure existing playground tests pass with the hexgrid dependency removed.
- [ ] **37B.4** — Ensure pgrid tests pass when installed as a proper package (not just `sys.path` hacks).
- [ ] **37B.5** — Ensure models tests pass in isolation.

### 37C — Final code cleanup

- [ ] **37C.1** — Remove any remaining `sys.path.insert` hacks in pgrid scripts (they should use the installed package).
- [ ] **37C.2** — Run a linter pass across all three repos (ruff or flake8) and fix obvious issues.
- [ ] **37C.3** — Verify all three repos have clean `pyproject.toml` with consistent metadata (authors, license, Python version).
- [ ] **37C.4** — Final review of import graphs: confirm no circular dependencies between the three packages.
- [ ] **37C.5** — Archive or delete obsolete planning docs from playground root (see 32D.4).

---

## Phase Summary

| Phase | Scope | Description |
|-------|-------|-------------|
| **30** | All repos | Package hygiene — make `models` and `pgrid` proper installable packages, wire `playground` dependencies |
| **31** | All repos | SoC audit — review boundaries, relocate misplaced code, document contracts |
| **32** | Playground | Legacy cleanup — remove hexgrid, dead graph-tile code, legacy renderer stubs, stale docs |
| **33** | Playground + pgrid | Terrain data pipeline — wire pgrid terrain generation into playground's planet flow |
| **34** | Playground + pgrid | Texture atlas pipeline — use pgrid's polygon-cut atlas in playground's OpenGL renderer |
| **35** | Playground | Feature preservation — verify tile selection, region painting, glow, sizing all still work |
| **36** | Playground | Tile debug panel — rich per-tile debug info from models + pgrid + playground |
| **37** | All repos | Polish — design docs, tests, linting, final cleanup |
