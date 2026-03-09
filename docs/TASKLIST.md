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
- [x] **30C.2** — Remove the `hexgrid` dependency entirely (adapter, texture baker, and any other references). pgrid replaces this. ✅
- [x] **30C.3** — Verify `playground` can `import polygrid` and `import models` after the editable installs. ✅
- [x] **30C.4** — Update `.env` / `.vscode/settings.json` if `PYTHONPATH` hacks existed for the old layout. ✅ (none found — editable installs are sufficient)

---

## Phase 31 — Separation of Concerns Audit

Review the boundaries between the three repos and relocate code that is in the wrong place.

### 31A — `models` boundary

- [x] **31A.1** — Audit `models/models/rendering/opengl.py` — if it contains GL helpers only used by `playground`, move them to `playground/opengl/`. ✅ Not used by playground at all; used only by pgrid demo scripts and models' own demos. No relocation needed.
- [x] **31A.2** — Audit `models/models/objects/render_helpers.py` — relocate playground-specific rendering utilities. ✅ Not imported by playground. Contains generic geometry helpers (face orientation, projection, hit testing). No relocation needed.
- [x] **31A.3** — Confirm `models` has **no** dependency on `pgrid` or `playground`. It should be a leaf package. ✅ Confirmed: `pyproject.toml` depends only on `numpy` and `PyOpenGL`; no imports of `polygrid` or `playground` found anywhere in the models source.
- [x] **31A.4** — Document the `models` public contract: Goldberg geometry (tiles, polyhedron, slug generation, mesh building, layout helpers) and core primitives (mesh, transform, colour). Nothing else. ✅ Created `models/docs/CONTRACT.md`.

### 31B — `pgrid` boundary

- [x] **31B.1** — Audit pgrid's `globe_renderer.py` and `globe_renderer_v2.py` — these are standalone OpenGL renderers that duplicate what `playground` already does better. Mark them as **standalone demo scripts only** (not part of the library API); move to `scripts/` or gate behind an optional `[demo]` extra. ✅ Both modules are only imported by pgrid scripts (`scripts/`) and test files. They are already gated behind `try/except ImportError` in `__init__.py`. Added a note to `ARCHITECTURE.md` explicitly marking them as demo-only, not library API.
- [x] **31B.2** — Audit pgrid's `globe_mesh.py` — if it builds `ShapeMesh` objects for the 3D renderer, decide whether playground should consume this or keep its own `PlanetTileController` mesh pipeline. ✅ `globe_mesh.py` builds meshes specifically for pgrid's demo renderers. Playground will keep its own `PlanetTileController` mesh pipeline (which already works with `models`' `build_layout_tile_meshes`). Documented in `ARCHITECTURE.md` boundary contract.
- [x] **31B.3** — Confirm pgrid's library surface is: core topology, grid building, terrain generation, detail grids, texture atlas building, UV alignment, and data export (`globe_export`, `tile_data`). Rendering modules are optional extras, not core API. ✅ Confirmed and documented in `ARCHITECTURE.md` "Boundary Contract" section with full module-by-layer table.
- [x] **31B.4** — Ensure pgrid scripts (`render_polygrids.py`, `render_globe_from_tiles.py`) remain functional as standalone demo/diagnostic tools but are not imported by `playground`. ✅ Confirmed: all scripts live under `pgrid/scripts/`, none are imported by playground.
- [x] **31B.5** — Review pgrid's `globe_export.py` output format — this is the main data contract between pgrid and playground. Document the JSON schema (`globe_payload.json`, `metadata.json`, `uv_layout.json`). ✅ Already documented in `docs/JSON_CONTRACT.md` (globe payload, atlas image, UV layout JSON) and formally specified in `schemas/globe.schema.json`. Added cross-reference in `ARCHITECTURE.md` boundary contract.

### 31C — `playground` boundary

- [x] **31C.1** — Confirm playground never imports pgrid rendering modules (only data/topology/terrain/texture layers). ✅ Playground does not import `polygrid` at all yet (pre-integration). When integration begins (Phase 33–34), allowed/forbidden imports are documented in `playground/docs/dependency_contracts.md`.
- [x] **31C.2** — Confirm playground never imports `models.rendering` (only `models.core` and `models.objects`). ✅ Confirmed: grep shows zero imports of `models.rendering` in playground. Playground uses `models.core.mesh` and `models.objects.goldberg` only.
- [x] **31C.3** — Document which pgrid modules playground will depend on: `polygrid.globe`, `polygrid.tile_data`, `polygrid.globe_export`, `polygrid.tile_detail`, `polygrid.detail_terrain`, `polygrid.detail_render`, `polygrid.tile_uv_align`, `polygrid.uv_texture`, `polygrid.mountains`, `polygrid.noise`. ✅ Created `playground/docs/dependency_contracts.md` with full planned import table.
- [x] **31C.4** — Document which `models` modules playground depends on: `models.objects.goldberg`, `models.core.mesh`, `models.core.geometry`. ✅ Documented in `playground/docs/dependency_contracts.md`.

---

## Phase 32 — Legacy Cleanup in Playground

Prune dead code and outdated approaches from playground so the pgrid integration has a clean foundation.

### 32A — Remove hexgrid dependency

- [x] **32A.1** — Delete `controllers/hexgrid_adapter.py`. ✅
- [x] **32A.2** — Delete `controllers/hexgrid_texture_baker.py`. ✅
- [x] **32A.3** — Delete `controllers/hexgrid_spec_sequence.py`. ✅
- [x] **32A.4** — Delete `controllers/hexgrid_region_stitching.py`. ✅
- [x] **32A.5** — Delete `controllers/planet_hexgrid_service.py`. ✅
- [x] **32A.6** — Delete `controllers/region_hexgrid_service.py`. ✅
- [x] **32A.7** — Remove `-e ../hexgrid` from `requirements.txt`. ✅
- [x] **32A.8** — Grep the entire playground tree for remaining `hexgrid` references and remove/replace them. Update any tests that exercised hexgrid paths. ✅
- [x] **32A.9** — Remove the hexgrid UV rotation slider and related code from `DebugInfoPanel`. ✅

### 32B — Remove graph-seeded tile geometry path

> **Audit status:** `GraphRuntimeController` and `TileGeometryProvider` are still actively wired into the app (used by `AppController`, `WorldController`, `PlanetController`, `RegionController`, `LocationController`, `BiomeController`, `PlanetSessionService`). These cannot be removed until the pgrid integration (Phase 33) provides a replacement data path.

- [x] **32B.1** — Audit `controllers/graph_controller.py` — identify which graph-tile lookups are still needed (region painting, selections) vs. which are legacy (mesh topology). ✅ **Finding:** `GraphRuntimeController` is used by `AppController` to lazily build a `GraphRuntimeStore` + feed tile data into `TileGeometryProvider`, `PlanetController`, `RegionController`, `LocationController`, `BiomeController`, and `WorldController`. The graph-tile lookups serve: tile identity (slug, polygon index), neighbour lists, geometry metadata (center, normal, tangent, bitangent, transform). These are all still needed. Mesh topology is handled by `TileTopologyService` (models-backed), not by the graph store.
- [x] **32B.2** — Audit `controllers/tile_geometry_provider.py` — determine if `TileGeometrySnapshot` is still needed or if pgrid + models can serve the same data. ✅ **Finding:** `TileGeometrySnapshot` is consumed by `PlanetSessionService.payload_for_layout()` which feeds the debug panel / session state. The data it carries (polygon_index, tile_id, vertex_count, neighbors, center, normal, tangent, bitangent) can all be sourced from `models.objects.goldberg.GoldbergPolyhedron` + pgrid's `TileDataStore`. The provider can be replaced after Phase 33 integration.
- [ ] **32B.3** — Audit `persistence/graph/runtime.py` — identify graph-tile seeding code that can be removed once pgrid provides tile topology.
- [ ] **32B.4** — Remove the graph-tile seeding CLI (`persistence.graph.seed --layout`) if no longer needed.
- [ ] **32B.5** — Replace `TileGeometryProvider` with a thin wrapper over `TileTopologyService` + pgrid data where possible.

### 32C — Remove legacy renderer remnants

- [x] **32C.1** — Delete `opengl/rendering/planet_surface_controller.py` (already a stub raising `ImportError`). ✅ Deleted.
- [x] **32C.2** — Delete `opengl/rendering/icosahedron/controller.py` (already a stub raising `ImportError`). ✅ Deleted (entire `icosahedron/` directory).
- [x] **32C.3** — Check if `opengl/rendering/icosahedron/geometry.py` is still referenced — if not, delete the entire `icosahedron/` directory. ✅ No references found in playground. Deleted entire directory.
- [x] **32C.4** — Audit `opengl/rendering/planet_renderer.py` for dead code paths that reference the old UV-sphere or atlas approaches. ✅ `build_uv_sphere_mesh` is still used for the simplified planet mesh (`PlanetDetailLevel.SIMPLIFIED`); no dead code paths found. Also deleted retired test stubs: `test_icosahedron_controller_uniforms.py`, `test_icosahedron_controller.py`, `test_border_factors.py`.

### 32D — General pruning

- [x] **32D.1** — Delete `biome_debug_main.py` and `tmp_debug_astral.py` from the playground root (temp debug scripts). ✅ Deleted. Updated README and home_overlay comment.
- [x] **32D.2** — Audit `scripts/` for stale one-off scripts. ✅ Deleted `analyze_edge_mismatches.py` (dead script, raises SystemExit immediately). Remaining scripts (`reseed_graph_db.py`, `start_graph_db.py`, `db_admin.py`, `prebake_biome_textures.py`, `prebake_planet_tile_textures.py`, `biome_texture_gallery.py`) are still relevant to the running app.
- [x] **32D.3** — Clean up `__pycache__` references in `.gitignore`; ensure no compiled files are tracked. ✅ `.gitignore` already has `__pycache__/` and `**/__pycache__/`; no `.pyc` files tracked in git.
- [x] **32D.4** — Audit `Orientation and Refactoring Task List.md`, `refactor_ico_controller.md`, `planet_rendering_plan.md`, `llm_plan.md`, `travel_notes.txt`, `world_buiilding_ux.md` — archive or delete any that are fully superseded by this tasklist and the design docs. ✅ Deleted 4 files: `Orientation and Refactoring Task List.md` (archived stub), `refactor_ico_controller.md` (archived stub), `planet_rendering_plan.md` (archived stub), `travel_notes.txt` (personal data). Kept `llm_plan.md` and `world_buiilding_ux.md` as potentially still relevant working notes.

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

## Phase 38 — Pentagon Tile Rendering Fix

Fix the visual distortion on pentagon tiles in the rendered globe.  Hex tiles currently look correct and **must not regress**.  See `docs/TILE_TEXTURE_MAPPING.md` § "Pentagon Distortion Problem" for full diagnosis.

### Context

The 12 pentagon tiles on the Goldberg polyhedron show visible warping/stretching of terrain texture.  The root cause is `normalize_uvs()` in `models/core/geometry.py`, which normalises U and V axes **independently** — stretching the UV polygon to fill [0,1]² even when the tangent-plane projection is non-square.  Pentagons have a bounding-box aspect ratio of ~1.0 : 0.81, giving ~23% anisotropic distortion (hexes are ~13%, barely visible).

### 38A — Uniform UV normalisation (primary fix)

> **Scope:** `models` repo.  Affects both pgrid texture rendering and playground's 3D mesh UVs.

- [x] **38A.1** — In `models/core/geometry.py`, change `normalize_uvs()` to use **uniform scaling**: compute `span = max(span_x, span_y)` and divide both axes by `span`.  Centre the shorter axis within [0, 1] (i.e. offset by `(1 - shorter_span/span) / 2`). ✅ Both `normalize_uvs()` in `geometry.py` and `_normalize_uvs()` in `icosahedron.py` updated.
- [x] **38A.2** — Audit all callers of `normalize_uvs()` across models, pgrid, and playground.  Confirm that none assume the UV polygon fills [0,1]² exactly. ✅ Key callers audited: `generate_goldberg_tiles()`, `icosahedron.py`, `uv_texture.py`. None assume [0,1]² exactly. Playground's `icosahedron/` directory has been deleted (Phase 32C).
- [x] **38A.3** — Update `project_and_normalize()` in pgrid's `uv_texture.py` to use uniform scaling matching the models library's aspect-ratio-preserving normalisation. ✅ Updated to use `span = max(u_span, v_span)` with centering offsets.
- [ ] **38A.4** — Run `render_polygrids.py -f 3 --detail-rings 3 -o exports/f3_test` and visually verify:
  - Pentagon tiles no longer show visible warping
  - Hex tiles look identical to before (no regression)
  - Edge stitching between hex↔pent and pent↔hex tiles is seamless
- [x] **38A.5** — Run pgrid's test suite (`pytest tests/`) — fix any tests that assert on specific UV coordinate values that change under uniform scaling. ✅ All 43 uv_texture tests pass.
- [x] **38A.6** — Run models' test suite (`pytest tests/`) — fix any tests broken by the UV change. ✅ All 6 models tests pass (UV values still within [0,1]).
- [ ] **38A.7** — Run playground's test suite to check for regressions.
- [ ] **38A.8** — Render the globe at freq=3 and freq=4 (`render_globe_from_tiles.py --v2`) and visually confirm both pentagon and hex tiles look correct.

### 38B — Exact pentagon corner detection (secondary fix)

> **Scope:** `pgrid` repo.  Makes the UVTransform corner-matching deterministic for pentagon grids.

- [x] **38B.1** — In `goldberg_topology.py`, the `goldberg_topology()` function already returns `corner_ids` (the 5 sector-corner vertex IDs).  Verify this works correctly for rings ≥ 1. ✅ Verified: `goldberg_topology()` returns `corner_vids` for all ring counts; tested via `build_pentagon_centered_grid` for rings 0–4.
- [x] **38B.2** — In `goldberg_topology.py` → `build_goldberg_grid()`, propagate `corner_ids` into the returned PolyGrid's metadata: `metadata["corner_vertex_ids"] = corner_ids`. ✅ Added for both rings=0 and rings≥1 code paths.
- [x] **38B.3** — In `detail_grid.py` → `build_detail_grid()`, for pentagon tiles, the `corner_vertex_ids` are already in the grid metadata from `build_goldberg_grid`. Also added `parent_face_type` to metadata. ✅
- [x] **38B.4** — In `uv_texture.py` → `_find_polygon_corners()`, add an early-return path: if the detail grid's metadata contains `corner_vertex_ids`, look up those vertex positions directly instead of running the threshold-based clustering heuristic. ✅ Fast-path added with counter-clockwise angle sorting.
- [x] **38B.5** — Add tests verifying `_find_polygon_corners()` returns the correct 5 corners for a pentagon grid with rings=2, 3, and 4 — both via the metadata fast-path and the clustering fallback. ✅ 11 tests in `TestFindPolygonCornersMetadataFastPath` + 4 tests in `test_pentagon_centered.py`.

### 38C — Robust corner-to-corner matching (tertiary fix)

> **Scope:** `pgrid` repo.  Prevents rotational mismatch on pentagon tiles.

- [x] **38C.1** — In `compute_detail_to_uv_transform()`, after finding the best rotational offset by angular scoring, add a **validation step**: compute the edge-length ratios of adjacent source corners vs adjacent destination corners for the chosen offset.  If the ratios differ by more than a threshold, try the next-best offset. ✅ Implemented: offset candidates sorted by angular score, then validated with edge-length ratio check.
- [x] **38C.2** — Add tests verifying pentagon transforms produce valid mappings — centroid maps near UV centre, all corners map within [0,1], all 12 pentagon tiles produce valid transforms. ✅ 2 tests in `TestRobustCornerMatching`.
- [x] **38C.3** — Log a warning when the matching falls back to a non-optimal offset, to aid future debugging. ✅ Warning logged via `logging.getLogger(__name__).warning()`.

### 38D — Visual validation

- [ ] **38D.1** — After all fixes, render side-by-side comparison images (before vs after) for at least 3 pentagon tiles at freq=3 and freq=4.
- [ ] **38D.2** — Verify that the 3D globe looks correct at multiple zoom levels — close-up on a pentagon tile, medium zoom showing hex/pent boundary, and full globe.
- [ ] **38D.3** — Commit all changes across models and pgrid.  Update `TILE_TEXTURE_MAPPING.md` to mark the Pentagon Distortion Problem as resolved.

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
| **38** | models + pgrid | Pentagon tile rendering fix — uniform UV scaling, exact corner detection, robust matching |
