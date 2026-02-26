# PolyGrid Task List

Comprehensive task list for evolving PolyGrid from a topology toolkit into a procedural terrain generation system for a Goldberg polyhedron game.

---

## Legend

- ✅ Done
- 🔲 To do
- 🔶 In progress / partially done

---

## Phase 1 — Core Topology ✅

- [x] `Vertex`, `Edge`, `Face` frozen dataclasses
- [x] `PolyGrid` container with validation
- [x] JSON serialisation round-trip
- [x] Face adjacency algorithm (`build_face_adjacency`)
- [x] BFS ring detection (`ring_faces`)
- [x] Boundary walking and macro-edge detection
- [x] CLI: `validate`, `render`, `build-hex`, `build-pent`

## Phase 2 — Goldberg Topology ✅

- [x] Combinatorial Goldberg topology via cone-triangulation + dualisation
- [x] Tutte embedding (Laplacian solve with pentagonal boundary)
- [x] Least-squares optimisation (edge length, angle, area-inversion)
- [x] CCW winding fix
- [x] Structural invariant tests (face counts, vertex degrees, boundary counts, corners)
- [x] Pentagon-centred grid builder (`build_pentagon_centered_grid`)
- [x] Quality diagnostics (per-ring edge lengths, angles, quality gates)

## Phase 3 — Stitching & Assembly ✅

- [x] `CompositeGrid` with vertex merging and edge dedup
- [x] `StitchSpec` for declaring which macro-edges join
- [x] `AssemblyPlan` with `.build()` and `.exploded(gap)`
- [x] `pent_hex_assembly(rings)` — 1 pent + 5 hex recipe
- [x] Hex positioning: scale, rotate, translate, reflect to outside
- [x] Hex-hex boundary snapping (vertex averaging)
- [x] 10 stitches: 5 pent↔hex + 5 hex↔hex
- [x] Macro-edge compatibility checks in tests

## Phase 4 — Transforms & Visualisation ✅

- [x] Overlay data model (`Overlay`, `OverlayPoint`, `OverlaySegment`, `OverlayRegion`)
- [x] Voronoi dual transform (`apply_voronoi`)
- [x] Partition transform (`apply_partition`)
- [x] Multi-panel visualisation (`render_assembly_panels`, etc.)
- [x] Partition colouring with per-section colours
- [x] Unstitched overlay view with per-component translation

---

## Phase 5 — Tile Data Layer ✅

The foundation for terrain generation: a way to attach and query per-face data.

- [x] **`TileData` model** — container mapping face ids to key-value data, validated against a `TileSchema`. Separate from `PolyGrid` (SoC). Wraps `Dict[str, Dict[str, Any]]`.
- [x] **Schema / typed fields** — `TileSchema` declares field names, types (`int`, `float`, `str`, `bool`), and optional defaults. `FieldDef` dataclass per field. Validates on every write.
- [x] **`TileDataStore`** — binds a `TileData` to a `PolyGrid`. Provides `get`, `set`, `bulk_set`, `initialise_all`, lazy-built adjacency cache.
- [x] **JSON serialisation** — `TileData` round-trips to JSON (schema + tiles dict). File I/O via `save_tile_data` / `load_tile_data`. Handles JSON int→float coercion.
- [x] **Neighbour-aware queries** — `get_neighbors_data(face_id, key)` returns `[(neighbor_id, value)]` for adjacent faces.
- [x] **Ring-based queries** — `get_ring_data(face_id, radius, key)` returns `{ring: [(face_id, value)]}` via BFS.
- [x] **Bulk operations** — `apply_to_all(key, fn)`, `apply_to_ring(center, radius, key, fn)`, `apply_to_faces(face_ids, key, fn)`.
- [x] **Tests** — 52 tests: CRUD, schema validation, serialisation round-trip, neighbour queries, ring queries, bulk ops, file I/O, pent-grid integration.

## Phase 6 — Terrain Partitioning 🔲

Splitting the grid into named regions (continents, oceans, biome zones) that algorithms operate on.

- [ ] **Region model** — a named collection of face ids with metadata (e.g. `Region(name="continent_1", face_ids=[...], biome="temperate")`).
- [ ] **Region assignment algorithms:**
  - [ ] *Angular sectors* (already have `apply_partition` — adapt to produce `Region` objects)
  - [ ] *Flood-fill from seeds* — pick N seed faces, flood-fill outward, each seed claims territory
  - [ ] *Voronoi-based partitioning* — use face centroids as sites, assign each face to the nearest seed
  - [ ] *Noise-based boundaries* — perturb region boundaries with Perlin/simplex noise for organic shapes
- [ ] **Constraints** — minimum region size, maximum region count, adjacency requirements (e.g. every continent must touch ocean).
- [ ] **Region ↔ TileData integration** — `assign_biome(region, biome_type)` sets tile data for all faces in a region.
- [ ] **Visualise regions** — colour-coded overlay showing region boundaries and biome types.
- [ ] **Tests** — all faces assigned, no gaps, constraint satisfaction.

## Phase 7 — Terrain Generation Algorithms 🔲

Per-biome algorithms that populate tile data with elevation, features, etc.

- [ ] **Algorithm registry / pipeline** — a way to declare and chain algorithms. Each algorithm is a function `(grid, tile_data, region) → tile_data`. Run in sequence.
- [ ] **Elevation generation:**
  - [ ] *Base heightmap* — Perlin/simplex noise scaled per biome (mountains high, plains flat)
  - [ ] *Ridge lines* — for mountain ranges, use ridged-multifractal noise or graph-based ridge paths
  - [ ] *Coastal falloff* — elevation drops to sea level near ocean regions
  - [ ] *Smoothing pass* — average with neighbours to remove harsh transitions
- [ ] **Moisture / temperature:**
  - [ ] *Latitude-based temperature* (for globe: distance from poles)
  - [ ] *Elevation-based temperature* (higher = colder)
  - [ ] *Moisture from oceans* — BFS spread from ocean tiles, decreasing with distance
  - [ ] *Rain shadow* — reduce moisture on the lee side of mountains
- [ ] **Biome classification:**
  - [ ] *Whittaker diagram* — classify biome from (temperature, moisture) pair
  - [ ] Types: ocean, lake, beach, grassland, forest, desert, tundra, mountain, snow, swamp
- [ ] **Feature placement:**
  - [ ] *Rivers* — follow elevation gradient from high to low, merge at confluences
  - [ ] *Lakes* — fill depressions where water can't drain
  - [ ] *Roads / paths* — shortest path between settlements, weighted by terrain difficulty
  - [ ] *Settlements* — place at favourable locations (river junctions, coastlines, flat areas)
- [ ] **Tests** — elevation within bounds, biome coverage, river connectivity, no orphan features.

## Phase 8 — Globe-Scale Topology 🔲

Represent the whole Goldberg polyhedron as a `PolyGrid` for macro-scale algorithms.

- [ ] **Globe grid builder** — build a `PolyGrid` where each face represents one face of the Goldberg polyhedron (12 pentagons + N hexagons). Topology only, no 2D positions needed.
- [ ] **Face-type tagging** — each globe face knows whether it's a pentagon or hex face, and its frequency/subdivision level.
- [ ] **Globe ↔ detail grid mapping** — each globe face maps to a detail `PolyGrid` (built by `build_pentagon_centered_grid` or `build_pure_hex_grid`). Stitch specs describe how detail grids join at globe-face boundaries.
- [ ] **Macro algorithms on globe:**
  - [ ] *Continent placement* — partition globe faces into continents/oceans
  - [ ] *Climate zones* — latitude bands, ocean currents
  - [ ] *Tectonic plates* — for mountain range generation at plate boundaries
- [ ] **Drill-down** — run per-face detail algorithms using the globe partition as input (e.g. "this globe face is desert → run desert terrain gen on its detail grid").
- [ ] **Tests** — globe face count matches Goldberg formula, adjacency is correct, all faces mapped to detail grids.

## Phase 9 — Export & 3D Integration 🔲

Prepare per-tile data and textures for the 3D Goldberg renderer.

- [ ] **Per-tile texture export** — render each face's detail grid to a PNG/texture atlas. One image per Goldberg face, with consistent UV mapping.
- [ ] **Tile data export** — JSON export of all tile data (biome, elevation, features) per Goldberg face, keyed by face id within the detail grid.
- [ ] **Globe metadata export** — JSON describing the globe topology, face assignments, and paths to per-face texture/data files.
- [ ] **3D package integration** — define the interface between PolyGrid exports and the 3D renderer. Likely: the 3D package reads globe metadata JSON, loads per-face textures, maps them onto the polyhedron.
- [ ] **Coordinate mapping** — 2D detail grid positions → 3D polyhedron face UV coordinates. Needed so textures wrap correctly onto curved faces.
- [ ] **Tests** — export files exist, JSON is valid, textures have correct dimensions.

---

## Ongoing — Code Quality & Refactoring

- [x] **Ensure gitignore covers relevant files** — added `exports/`, `.venv/`, `.coverage`, `htmlcov/` to `.gitignore`; removed tracked `__pycache__/` and `exports/` from git.
- [x] **Review `render.py` vs `visualize.py`** — merged `render_png` (with pent-axes support) into `visualize.py`. `render.py` is now a deprecation shim re-exporting from `visualize`. All CLI and script imports updated.
- [x] **Clean up `__init__.py` exports** — organised imports by architectural layer (Core / Building / Transforms / Rendering / Diagnostics) with section comments. `__all__` grouped the same way.
- [x] **Type hints** — added full type annotations to all untyped private helpers in `diagnostics.py` and `cli.py`. Removed bare `dict`/`list` gaps.
- [x] **Docstrings** — filled gaps in `diagnostics.py` (`min_face_signed_area`, `has_edge_crossings`, `summarize_ring_stats`, `ring_quality_gates`, `diagnostics_report`, helpers) and `geometry.py` (`interior_angle`, `face_signed_area`, `collect_face_vertices`, `boundary_vertex_cycle`).
- [x] **Remove legacy aliases from `geometry.py`** — removed the `_xxx = xxx` alias block at the bottom (unused).
- [x] **Move `notes.md`** — `src/polygrid/notes.md` is an early planning doc. Move to `docs/` or remove if superseded.
- [x] **Remove `experiments/`** — `src/polygrid/experiments/` contains ad-hoc experimental code. Clean up or move to `scripts/`.
- [ ] **CI pipeline** — set up GitHub Actions for `pytest` + linting on push.
- [ ] **Design patterns** — as terrain algorithms grow, consider:
  - *Strategy pattern* for swappable terrain generators per biome
  - *Pipeline/chain pattern* for algorithm sequencing
  - *Observer pattern* if tile data changes need to trigger recalculation
  - *Repository pattern* for tile data persistence
- [ ] **Performance** — profile for large ring counts (rings ≥ 5). The optimiser and stitching are the bottlenecks. Consider caching, lazy evaluation.

---

## Dependency Roadmap

| Phase | New dependencies |
|-------|-----------------|
| 5 (Tile Data) | None |
| 6 (Partitioning) | `noise` or `opensimplex` (for noise-based boundaries) |
| 7 (Terrain Gen) | `noise` / `opensimplex`, possibly `Pillow` for texture generation |
| 8 (Globe) | None (topology only) |
| 9 (Export) | `Pillow` for PNG texture output |
