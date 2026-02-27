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

## Phase 6 — Terrain Partitioning ✅

Splitting the grid into named regions (continents, oceans, biome zones) that algorithms operate on.

- [x] **Region model** — a named collection of face ids with metadata (e.g. `Region(name="continent_1", face_ids=[...], biome="temperate")`). `RegionMap` container with validation, adjacency queries, and face↔region lookups.
- [x] **Region assignment algorithms:**
  - [x] *Angular sectors* — `partition_angular(grid, n_sections)` divides faces into equal angular wedges
  - [x] *Flood-fill from seeds* — `partition_flood_fill(grid, seeds)` competitive BFS expansion with random tie-breaking
  - [x] *Voronoi-based partitioning* — `partition_voronoi(grid, seeds)` assigns each face to nearest seed by centroid distance
  - [x] *Noise-based boundaries* — `partition_noise(grid, seeds)` Voronoi with distance perturbation (uses opensimplex if available, deterministic fallback otherwise)
- [x] **Constraints** — `validate_region_map()` checks: full coverage, no gaps/overlaps, min region size, max region count, required adjacency between regions.
- [x] **Region ↔ TileData integration** — `assign_field(region, store, key, value)` and `assign_biome(region, store, biome_type)` bulk-set tile data for all faces in a region.
- [x] **Visualise regions** — `regions_to_overlay(region_map, grid)` converts to an `Overlay` compatible with existing partition rendering.
- [x] **Tests** — 76 tests: Region/RegionMap model, validation (coverage, overlaps, extras, constraints), all 4 algorithms (full coverage, no gaps, determinism, edge cases), TileData integration, overlay conversion, region adjacency, cross-algorithm parametrized tests on hex and pent grids.

## Phase 7 — Terrain Generation Algorithms ✅

Build reusable, composable primitives first, then assemble them into higher-level terrain features. Initial focus: **mountains** and **rivers** — the two features that most define realistic landscape character (see satellite-view reference image). Rivers are distinct from region partitioning because they flow *through* existing biomes rather than defining region boundaries.

### 7A — Noise Primitives (`noise.py`) ✅

A library of reusable noise functions that all terrain algorithms draw from. Each function operates on `(x, y)` and returns a float, making them easy to compose, layer, and test in isolation.

- [x] **7A.1 — `fbm` (Fractal Brownian Motion)** — standard multi-octave noise. Params: `octaves`, `lacunarity`, `persistence`, `frequency`, `seed`. Uses `opensimplex` when available, deterministic fallback hash-noise otherwise. Returns values in `[−1, 1]`.
- [x] **7A.2 — `ridged_noise`** — `abs(fbm)` inverted so ridges form at zero-crossings. Produces sharp mountain-ridge-like features. Same params as `fbm` plus `ridge_offset`.
- [x] **7A.3 — `domain_warp`** — feed warped coordinates into any noise fn: `f(x + fbm₁(x,y), y + fbm₂(x,y))`. Creates organic, swirly distortions. Params: `warp_strength`, `warp_frequency`.
- [x] **7A.4 — `gradient_mask`** — radial or directional linear falloff. Used to fade elevation toward edges, coasts, etc. Params: `center`, `radius`, `falloff_curve` (linear / smooth / exponential).
- [x] **7A.5 — `terrace`** — remap a continuous value into stepped plateaus: `floor(v * n_steps) / n_steps` with optional smooth blending. Gives mesa / plateau shapes.
- [x] **7A.6 — `normalize` / `remap`** — utility to rescale any `[a, b]` range into `[c, d]`. Used everywhere.
- [x] **7A.7 — Tests** — 40 tests: output range assertions (`fbm` ∈ `[−1,1]`), determinism (same seed → same output), composability (`ridged_noise(domain_warp(x,y))` doesn't crash).

> **Design note:** every function is a plain `(x, y, **config) → float`. No grid dependency. This keeps them testable and reusable outside PolyGrid.

### 7B — Grid-Noise Bridge (`heightmap.py`) ✅

Connect the noise primitives to the grid/tile-data world. This is the thin adapter layer between `noise.py` (pure math) and `TileDataStore` (grid-aware data).

- [x] **7B.1 — `sample_noise_field`** — given a `PolyGrid` + a noise function + config, evaluate the noise at every face centroid and write results into a TileData field. Signature: `sample_noise_field(grid, store, field_name, noise_fn, **config)`.
- [x] **7B.2 — `sample_noise_field_region`** — same as above but restricted to faces in a `Region`. Faces outside the region are untouched.
- [x] **7B.3 — `smooth_field`** — neighbour-averaging pass: for each face, new value = weighted average of self + neighbours. Params: `iterations`, `self_weight`. Operates on an existing TileData field in-place.
- [x] **7B.4 — `blend_fields`** — combine two TileData fields into a third using a blend function: `out = fn(a, b)`. Useful for e.g. `elevation = base_noise * ridge_mask`.
- [x] **7B.5 — `clamp_field` / `normalize_field`** — clamp or normalize a TileData field across all faces to `[min, max]`.
- [x] **7B.6 — Tests** — 19 tests: every face gets a value, values within expected range, smoothing reduces variance, region-restricted sampling doesn't touch other faces.

> **Design note:** `sample_noise_field` is the core pattern — it means *any* noise primitive can be applied to *any* grid by just passing a different `noise_fn`. No special-case code per noise type.

### 7C — Mountain Generation (`mountains.py`) ✅

Assemble the noise primitives + grid bridge into a high-level mountain-terrain generator. The goal is satellite-realistic mountain ranges with ridges, peaks, foothills, and valleys.

- [x] **7C.1 — `MountainConfig` dataclass** — all tuneable parameters in one place:
  - `peak_elevation` (float, default 1.0) — max elevation at peaks
  - `base_elevation` (float, default 0.1) — elevation of surrounding lowlands
  - `ridge_octaves` (int, default 6) — detail level of ridge noise
  - `ridge_lacunarity` (float, default 2.2) — frequency scaling between octaves
  - `ridge_persistence` (float, default 0.5) — amplitude scaling between octaves
  - `ridge_frequency` (float, default 1.5) — base spatial frequency
  - `warp_strength` (float, default 0.3) — domain warp for organic shapes
  - `foothill_blend` (float, default 0.4) — how far foothills extend from ridges
  - `terrace_steps` (int, default 0) — 0 = smooth, >0 = mesa/plateau terracing
  - `seed` (int, default 42)
- [x] **7C.2 — `generate_mountains`** — the main entry point. Orchestrates noise primitives:
  1. Generate a ridged-noise heightmap (7A.2) → sharp peaks and ridge lines
  2. Apply domain warp (7A.3) → organic, non-geometric ridge shapes
  3. Blend with fbm (7A.1) at lower amplitude → foothills and micro-variation
  4. Optionally apply terrace (7A.5) → stepped plateaus
  5. Apply gradient mask (7A.4) if the mountain region has boundaries → elevation fades to `base_elevation` at region edges
  6. Normalize to `[base_elevation, peak_elevation]`
  7. Smooth (7B.3) → soften any harsh cell-to-cell jumps
  8. Write to TileData field `"elevation"`
  - Signature: `generate_mountains(grid, store, config, *, region=None)`
  - If `region` is given, only those faces are affected (via `sample_noise_field_region`)
- [x] **7C.3 — Preset configs** — named presets for common mountain types:
  - `MOUNTAIN_RANGE` — long ridged ranges (high warp, many octaves)
  - `ALPINE_PEAKS` — isolated sharp peaks (high frequency, low warp)
  - `ROLLING_HILLS` — gentle, low-amplitude terrain (few octaves, high persistence)
  - `MESA_PLATEAU` — flat-topped with steep edges (terrace_steps=4, low octaves)
- [x] **7C.4 — Tests** — 16 tests:
  - Elevation field exists for all target faces after generation
  - Values within `[base_elevation, peak_elevation]`
  - Peak faces (top 10%) have elevation > 0.7 × peak_elevation
  - Different configs produce measurably different height distributions (mean, std)
  - Region-restricted generation doesn't modify faces outside the region
  - Determinism: same config + seed → identical output

### 7D — Elevation-Aware Rendering (`terrain_render.py`) ✅

Visualise elevation data with colour ramps and shading so we can see the mountains. Builds on the existing `visualize.py` overlay system.

- [x] **7D.1 — `elevation_to_overlay`** — convert a TileData `"elevation"` field into an `Overlay` where each face's colour comes from a configurable colour ramp. Per-face fill colour, not region-based.
  - Colour ramps: `"terrain"` (blue→green→brown→white), `"greyscale"`, `"satellite"` (ocean-blue → lowland-green → highland-brown → rocky-grey → snow-white, matching the reference image palette).
  - Accepts `vmin`/`vmax` to control range mapping.
- [x] **7D.2 — `hillshade`** — compute per-face hillshade from elevation differences with neighbours. Simulates directional lighting (sun azimuth + altitude). Returns a `[0, 1]` brightness value per face that multiplies the base colour → gives 3D depth illusion like the satellite image.
- [x] **7D.3 — `render_terrain`** — convenience function: builds overlay from elevation + hillshade, renders via existing `render_stitched_with_overlay` / `render_assembly_panels`. Single call to go from TileData → PNG.
- [x] **7D.4 — Demo script `scripts/demo_mountains.py`** — CLI script: `python scripts/demo_mountains.py --rings 3 --preset alpine_peaks --out exports/mountains.png`. Builds grid, generates mountains, renders terrain.
- [x] **7D.5 — Tests** — 24 tests:
  - Overlay has correct number of regions (= number of faces)
  - Colour values are valid RGB tuples
  - Hillshade values in `[0, 1]`
  - PNG file is produced without error

### 7E — River Generation (`rivers.py`) ✅

Rivers are fundamentally different from regions — they are **linear features that flow through existing terrain**, following the elevation gradient downhill. They modify the existing elevation (carving valleys) and add a new `"river"` tile-data field rather than changing region boundaries.

#### 7E.1 — River Primitives ✅

Low-level building blocks for river pathfinding.

- [x] **7E.1.1 — `steepest_descent_path`** — from a starting face, greedily follow the neighbour with the lowest elevation until reaching a local minimum or grid boundary. Returns an ordered list of face ids (the river path). Handles plateaus by BFS to find the nearest lower face.
- [x] **7E.1.2 — `find_drainage_basins`** — for every face, determine which local minimum it drains to (following steepest descent). Returns `{face_id: basin_id}`. This tells us the "watershed" structure of the terrain.
- [x] **7E.1.3 — `fill_depressions`** — raise elevation of local minima that aren't at the grid boundary (endorheic basins) so that water can flow outward. This is a standard hydrological "pit filling" step. Modifies elevation in-place.
- [x] **7E.1.4 — `flow_accumulation`** — for each face, count how many upstream faces drain through it (BFS/DFS from all faces following descent). Faces with high accumulation are where rivers form. Returns `{face_id: int}`.
- [x] **7E.1.5 — Tests** — steepest descent always descends, every face in one basin, no interior minima after fill, accumulation ≥ 1 everywhere.

#### 7E.2 — River Network Construction ✅

Build the actual river network from the hydrological primitives.

- [x] **7E.2.1 — `RiverSegment` dataclass** — ordered list of face ids forming one river stretch, plus metadata: `width` (from flow accumulation), `name`, `order` (Strahler or Shreve stream order).
- [x] **7E.2.2 — `RiverNetwork` dataclass** — collection of `RiverSegment`s with convenience queries: `segments_through(face_id)`, `main_stem()` (longest/highest-order river), `tributaries_of(segment)`.
- [x] **7E.2.3 — `generate_rivers`** — the main entry point:
  1. Fill depressions (7E.1.3) to ensure continuous drainage
  2. Compute flow accumulation (7E.1.4)
  3. Threshold: faces with accumulation above `min_accumulation` are river faces
  4. Trace river paths by following steepest descent from each river-head (local accumulation peak)
  5. Merge paths that converge → confluences
  6. Assign Strahler stream order → higher order = wider river
  7. Return a `RiverNetwork`
  - Params: `RiverConfig` with `min_accumulation`, `min_length`, `carve_depth`, `seed`
- [x] **7E.2.4 — Tests** — rivers always flow downhill, min_length respected, stream ordering correct.

#### 7E.3 — River ↔ Terrain Integration ✅

Rivers modify the existing terrain — they aren't a separate biome partition.

- [x] **7E.3.1 — `carve_river_valleys`** — for each river face, lower its elevation by `carve_depth × width_factor`. Wider rivers carve deeper. Also lower immediate neighbours slightly (valley walls). This modifies the `"elevation"` field in TileData.
- [x] **7E.3.2 — `assign_river_data`** — set a `"river"` (bool) field and `"river_width"` (float) field in TileData for all river faces. Downstream consumers (renderers, biome classifiers) can read these.
- [x] **7E.3.3 — `river_to_overlay`** — convert a `RiverNetwork` into an `Overlay`. River faces become `OverlayRegion`s coloured by width/order (thin tributaries = light blue, main stems = dark blue). Distinct from the elevation overlay — meant to be rendered on top of it.
- [x] **7E.3.4 — Tests** — 25 tests total: river carving, data assignment, overlay colours, valley carving isolation.

#### 7E.4 — Combined Mountain + River Demo ✅

- [x] **7E.4.1 — Demo script `scripts/demo_terrain.py`** — end-to-end terrain pipeline:
  1. Build pent_hex_assembly
  2. Generate mountains (7C.2) with a chosen preset
  3. Generate rivers (7E.2.3) flowing through the mountain terrain
  4. Carve river valleys (7E.3.1)
  5. Render: elevation overlay + river overlay on top → satellite-style PNG
  - CLI: `python scripts/demo_terrain.py --rings 3 --preset mountain_range --out exports/terrain.png`
- [x] **7E.4.2 — Tests:**
  - Full pipeline runs without error on rings=2 and rings=3
  - Output PNG exists and has reasonable file size
  - Rivers originate in high-elevation areas and terminate at low elevation

### 7F — Pipeline & Composition Framework (`pipeline.py`) ✅

A lightweight framework for chaining terrain generation steps. Not needed for the mountain/river work above (those are standalone functions), but provides structure as we add more terrain types.

- [x] **7F.1 — `TerrainStep` protocol** — any callable with signature `(grid, store, region_map, config) → None`. Mutates `store` in place. Has a `.name` attribute for logging.
- [x] **7F.2 — `TerrainPipeline`** — ordered list of `TerrainStep`s. `.run(grid, store, region_map)` executes each step in sequence. Supports `before` / `after` hooks for logging/validation.
- [x] **7F.3 — Built-in steps** — wrap `generate_mountains` and `generate_rivers` as `TerrainStep`s so they can be composed in a pipeline. Also `CustomStep` for inline lambdas.
- [x] **7F.4 — Tests** — 20 tests:
  - Pipeline runs steps in declared order
  - A step can read data written by a previous step
  - Empty pipeline is a no-op

### Summary — Phase 7 Implementation Order

| Step | Module | Depends on | Delivers |
|------|--------|-----------|----------|
| 7A | `noise.py` | nothing (pure math) | Reusable noise primitives |
| 7B | `heightmap.py` | 7A + tile_data | Grid↔noise bridge |
| 7C | `mountains.py` | 7A + 7B | Mountain terrain generation |
| 7D | `terrain_render.py` | 7C + visualize | Elevation rendering + hillshade |
| 7E.1 | `rivers.py` (primitives) | 7B (elevation data) | Hydrological building blocks |
| 7E.2 | `rivers.py` (network) | 7E.1 | River network construction |
| 7E.3 | `rivers.py` (integration) | 7E.2 + 7C | River↔terrain integration |
| 7E.4 | `demo_terrain.py` | 7C + 7E + 7D | End-to-end satellite-style demo |
| 7F | `pipeline.py` | 7C + 7E | Composable pipeline framework |

## Phase 8 — Globe-Scale Topology & Models Integration ✅

Bridge the `polygrid` terrain system with the `models` library's Goldberg polyhedron generator. The `models` library already produces a complete 3D Goldberg polyhedron (`GoldbergPolyhedron`) with per-tile 3D vertices, normals, adjacency, and transforms. This phase creates a `GlobeGrid` — a specialised PolyGrid whose faces are the tiles of a Goldberg polyhedron — so that all existing terrain algorithms (noise, mountains, rivers, regions, tile data) work on the globe without modification.

### Key design decisions

- **Coordinate system:** Globe tiles live in 3D. For noise/terrain sampling, we use spherical coordinates `(lat, lon)` projected to 2D via equirectangular projection `(lon, lat)` for the face centroids. This avoids seam artefacts because our noise primitives already handle wrapping when the domain spans a continuous surface.
- **PolyGrid compatibility:** `GlobeGrid` extends `PolyGrid` (or wraps it) so that `TileDataStore`, `RegionMap`, `sample_noise_field`, `generate_mountains`, `generate_rivers`, and all pipeline steps work without modification.
- **models as a dependency:** `polygrid` takes an *optional* dependency on `models`. The `globe.py` module imports from `models.objects.goldberg` and is gated behind an import check. All other polygrid modules remain independent.
- **3D metadata preserved:** Each globe face stores the original 3D center, normal, vertices, and transform from the models tile in `Face` metadata or a side-channel dict, so rendering can project back to 3D.

### 8A — Refactoring & Integration Prep ✅

Small changes to both libraries that make integration cleaner.

- [x] **8A.1 — Add `models` as optional dependency to `polygrid`** — add `models` to `pyproject.toml` under `[project.optional-dependencies]` as `globe = ["models>=0.1.1"]`. Guard all globe-related imports with `try/except ImportError`.
- [x] **8A.2 — Extend `Vertex` model for optional `z` coordinate** — currently `Vertex` has `x: Optional[float], y: Optional[float]`. Add `z: Optional[float] = None` so globe grids can store 3D positions. Ensure all existing code handles `z=None` gracefully (backward compatible — default is `None`).
- [x] **8A.3 — Extend `Face` metadata for 3D properties** — add an optional `metadata: dict` field to `Face` (or use the existing `PolyGrid.metadata`). Globe faces will store `{"normal_3d": (x,y,z), "center_3d": (x,y,z), "transform": [...]}` so downstream code can access 3D properties.
- [x] **8A.4 — Noise 3D support** — add `fbm_3d(x, y, z, ...)` and `ridged_noise_3d(x, y, z, ...)` variants to `noise.py` that use 3D noise (opensimplex `noise3` or a 3D hash fallback). This is critical for seamless globe terrain — 2D noise on `(lat, lon)` has a polar singularity and an equatorial seam.
- [x] **8A.5 — Heightmap 3D bridge** — add `sample_noise_field_3d(grid, store, field_name, noise_fn_3d)` to `heightmap.py` that samples at `(x, y, z)` face centroids instead of `(x, y)`. Works on any PolyGrid with 3D vertex positions.
- [x] **8A.6 — Tests:**
  - `Vertex(id, x, y, z=1.0)` round-trips through JSON
  - Existing code still works with `z=None`
  - `fbm_3d` returns values in `[-1, 1]`, is deterministic
  - `sample_noise_field_3d` writes to all faces

### 8B — Globe Grid Builder (`globe.py`) ✅

The core module that converts a `models.GoldbergPolyhedron` into a `PolyGrid`.

- [x] **8B.1 — `build_globe_grid(frequency, radius=1.0)` → PolyGrid** — the main entry point.
  1. Call `GoldbergPolyhedron.from_frequency(frequency, radius=radius)`
  2. For each tile: create a `Vertex` at the tile's 3D center `(x, y, z)`, and also project to 2D using `(longitude_deg, latitude_deg)` as `(x, y)` for noise compatibility.
  3. For each tile: create a `Face` with `face_type="pent"` or `"hex"`, `vertex_ids` from the tile's 3D vertices (each as a `Vertex` with 3D coords), and `neighbor_ids` from the tile's adjacency.
  4. Build edges from shared vertices between adjacent tiles.
  5. Store 3D metadata (center, normal, transform, base_face_index, base_face_grid) in `PolyGrid.metadata` or per-face dict.
  6. Return a `PolyGrid` with `metadata={"generator": "globe", "frequency": frequency, "radius": radius}`.
- [x] **8B.2 — `GlobeGrid` wrapper class** — thin subclass or wrapper around `PolyGrid` that adds:
  - `.frequency` property
  - `.radius` property
  - `.tile_3d_center(face_id)` → `(x, y, z)` — 3D center of the original Goldberg tile
  - `.tile_normal(face_id)` → `(x, y, z)` — outward normal
  - `.tile_transform(face_id)` → `4x4 matrix` — model matrix from models library
  - `.tile_lat_lon(face_id)` → `(lat_deg, lon_deg)` — spherical coords
  - `.polyhedron` — cached reference to the source `GoldbergPolyhedron`
- [x] **8B.3 — Face ID scheme** — stable face IDs that match the models tile slug format: `"globe_f{base_face_index}_{i}-{j}-{k}"` or simply `"t{tile_id}"`. Must be deterministic across runs.
- [x] **8B.4 — Adjacency validation** — ensure that `build_face_adjacency(globe_grid)` matches the models library's `tile.neighbor_ids` exactly. This validates the topology bridge.
- [x] **8B.5 — Tests:**
  - `build_globe_grid(3)` produces 92 faces (12 pent + 80 hex)
  - `build_globe_grid(4)` produces 162 faces
  - All faces have correct vertex counts (5 for pent, 6 for hex)
  - Adjacency matches: pentagons have 5 neighbors, hexagons have 6
  - Face IDs are deterministic (same frequency → same IDs)
  - `GlobeGrid` 3D accessors return correct values
  - Grid validates clean (`grid.validate()` returns no errors)
  - `TileDataStore` can be created and populated on a globe grid
  - Round-trip: `build_globe_grid` → `to_dict` → `from_dict` preserves all data

### 8C — Globe Terrain Generation ✅

Apply existing terrain algorithms to the globe grid. Because `GlobeGrid` is a `PolyGrid`, all existing algorithms *should* work — this sub-phase validates and adapts where needed.

- [x] **8C.1 — Globe noise sampling** — verify that `sample_noise_field` works on `GlobeGrid` (it reads face centroids via `face_center()`). If `face_center` uses 2D `(x, y)` (lat/lon projection), noise works but may have polar distortion. For high-quality results, add a `sample_noise_field_globe` that uses the 3D noise bridge from 8A.5.
- [x] **8C.2 — Globe mountains** — `generate_mountains(globe_grid, store, config)` should "just work" since it calls `sample_noise_field` internally. Verify and adapt if needed. Test that elevation data covers all 92 (freq=3) faces.
- [x] **8C.3 — Globe rivers** — `generate_rivers(globe_grid, store, config)` works via universal `get_face_adjacency()` helper. Rivers on a sphere naturally flow from high to low elevation. Required adding `get_face_adjacency(grid)` to `algorithms.py` that prefers `face.neighbor_ids` (globe grids) with fallback to shared-edge computation (flat grids).
- [x] **8C.4 — Globe regions** — `partition_noise(globe_grid, seeds)` and `partition_voronoi` should work for continent/ocean placement. Verify that seed selection and distance computation work on the lat/lon projected positions.
- [x] **8C.5 — Globe pipeline** — `TerrainPipeline` with `MountainStep` + `RiverStep` runs on `GlobeGrid`.
- [x] **8C.6 — Tests:**
  - Mountains on globe: all faces have elevation, range is correct
  - Rivers on globe: segments flow downhill
  - Regions on globe: full coverage, no gaps
  - Pipeline runs end-to-end on globe grid

### 8D — Globe Rendering ✅

Render the globe with terrain data visible — both as a 2D map projection and as 3D per-tile colours for the models renderer.

- [x] **8D.1 — `globe_to_colour_map(globe_grid, store, ramp="satellite")` → `{tile_id: (r, g, b)}`** — map each tile's elevation/biome to an RGB colour using the existing terrain_render colour ramps. Returns a dict keyed by face ID (or tile ID).
- [x] **8D.2 — `render_globe_flat(globe_grid, store, out_path, ramp="satellite")`** — render a 2D equirectangular projection of the globe grid. Each face plotted at its `(lon, lat)` position, coloured by terrain. Produces a flat "world map" PNG.
- [x] **8D.3 — `globe_to_tile_colours(globe_grid, store, ramp)` → JSON** — export per-tile colours as a JSON payload compatible with the models renderer: `{"tile_id": {"color": [r, g, b], "elevation": float, ...}}`. This is the hand-off format between polygrid and models.
- [x] **8D.4 — Demo script `scripts/demo_globe.py`** — CLI script:
  - `python scripts/demo_globe.py --frequency 3 --preset mountain_range --out exports/globe_flat.png`
  - Builds globe grid → generates mountains → renders flat projection
  - Also exports `exports/globe_colours.json` for 3D rendering
- [x] **8D.5 — Tests:**
  - Colour map has entry for every face
  - All colours are valid RGB tuples `(0-1, 0-1, 0-1)`
  - Flat render produces a PNG
  - JSON export is valid and has correct tile count

### 8E — 3D Goldberg Rendering with Terrain ✅

Use the models library's rendering system to visualise terrain on the actual 3D Goldberg polyhedron. This is the "money shot" — a spinning globe with terrain colours.

- [x] **8E.1 — `models` colour injection** — `globe_mesh.py` bridge module: `terrain_colors_for_layout`, `terrain_colors_from_tile_colours`, `build_terrain_layout_mesh`, `build_terrain_face_meshes`, `build_terrain_tile_meshes`, `build_terrain_edge_mesh`. Converts polygrid colour maps into models `Color` sequences and builds terrain-coloured `ShapeMesh` objects.
- [x] **8E.2 — `render_globe_3d(frequency, store, out_path, ramp="satellite")`** — render the Goldberg polyhedron as a static 3D image (matplotlib 3D scatter/polycollection, or off-screen models render if GL context available). Falls back to a wireframe-plus-colour view if no GL.
- [x] **8E.3 — Matplotlib 3D fallback** — for CI/headless environments, render the globe as a matplotlib `Poly3DCollection` plot. Each Goldberg tile is a 3D polygon with its terrain colour. Camera at an isometric-ish angle. Produces a PNG.
- [x] **8E.4 — Demo script `scripts/demo_globe_3d.py`** — CLI with presets: mountain_range, alpine_peaks, rolling_hills, mesa_plateau, regions, rivers. Outputs 3D render, flat render, mesh metadata JSON, and tile colours JSON.
- [x] **8E.5 — Tests:**
  - 3D render produces a PNG file
  - All tiles are coloured (no missing faces)
  - Wireframe mode works without GL
  - Globe mesh bridge: colour mapping, layout mesh, face meshes, tile meshes, edge mesh

### Summary — Phase 8 Implementation Order

| Step | Module | Depends on | Delivers |
|------|--------|-----------|----------|
| 8A | refactoring (Vertex z, noise 3D, heightmap 3D) | Phase 7 | Integration-ready primitives |
| 8B | `globe.py` | 8A + models library | `build_globe_grid`, `GlobeGrid` |
| 8C | globe terrain validation | 8B + Phase 7 modules | Mountains/rivers/regions on globe |
| 8D | globe rendering | 8C + terrain_render | 2D flat map + colour export |
| 8E | 3D rendering | 8D + models rendering | 3D Goldberg with terrain |

### Design notes — Topology contracts per frequency

| Frequency | Tiles | Pentagons | Hexagons | Formula |
|-----------|-------|-----------|----------|---------|
| 1 | 12 | 12 | 0 | Dodecahedron |
| 2 | 42 | 12 | 30 | 10×2²+2 |
| 3 | 92 | 12 | 80 | 10×3²+2 |
| 4 | 162 | 12 | 150 | 10×4²+2 |
| 5 | 252 | 12 | 240 | 10×5²+2 |

Every pentagon always has 5 neighbours. Every hexagon always has 6. This is the fundamental Goldberg invariant that our `GlobeGrid` must preserve.

## Phase 9 — Export & 3D Integration ✅

Prepare per-tile data and textures for the 3D Goldberg renderer. Phase 8 gets terrain on a globe grid and renders it in 2D and via matplotlib 3D. Phase 9 takes it further — producing exports that plug into the `models` library's OpenGL renderer and potentially other 3D engines.

### 9A — Per-Tile Data Export ✅

- [x] **9A.1 — `export_globe_payload(globe_grid, store, ramp)` → dict** — produce a single JSON-serialisable dict with:
  - `globe.metadata`: frequency, radius, tile count, generator info
  - `globe.tiles[]`: for each tile — id, face_type, center_3d, normal_3d, lat, lon, elevation, biome/region, colour RGB, vertex positions (3D)
  - `globe.adjacency`: edge list for the full graph
- [x] **9A.2 — `export_globe_json(globe_grid, store, path, ramp)`** — write the payload to a JSON file. Validate against a schema.
- [x] **9A.3 — Globe JSON schema** — add `schemas/globe.schema.json` defining the export format. Reference it from `JSON_CONTRACT.md`.
- [x] **9A.4 — Tests:**
  - Exported JSON validates against schema
  - Tile count matches expected for frequency
  - All tiles have colour, elevation, and 3D coords

### 9B — Multi-Resolution Detail Grids ✅

For higher fidelity: each Goldberg tile can expand into a local detail grid (pentagon-centered or hex grid built by the existing builders), giving sub-tile terrain detail.

- [x] **9B.1 — `build_detail_grid(globe_grid, face_id, detail_rings)` → PolyGrid** — for a given globe face, build a detail grid (pent-centered or hex) at the given ring count. Anchor it to the globe face's 2D projection.
- [x] **9B.2 — Detail ↔ globe mapping** — maintain a mapping between detail-grid faces and their parent globe tile. Store as `detail_grid.metadata["parent_face_id"]`.
- [x] **9B.3 — Detail terrain gen** — run terrain generation on the detail grid, seeded/constrained by the parent globe tile's elevation and biome.
- [x] **9B.4 — Per-tile texture export** — render each detail grid to a small PNG texture. UVs mapped so the texture wraps onto the Goldberg tile's surface in 3D.
- [x] **9B.5 — Texture atlas** — combine per-tile PNGs into a single atlas image for efficient GPU rendering.
- [x] **9B.6 — Tests:**
  - Detail grid has expected face count for given ring count
  - Texture files are created and have correct dimensions
  - Atlas has correct layout

### 9C — Models Library Renderer Integration ✅

Feed per-tile colours and textures into the `models` library's rendering pipeline.

- [x] **9C.1 — Colour mesh builder** — `build_coloured_globe_mesh(frequency, tile_colours)` and `build_coloured_globe_mesh_from_export(payload)` in `globe_renderer.py`. Also `build_terrain_layout_mesh` in `globe_mesh.py` (from Phase 8E).
- [x] **9C.2 — `render_terrain_globe_opengl(payload)`** — full OpenGL render of the Goldberg polyhedron with terrain colours. Uses `SimpleMeshRenderer` from models + pyglet window with mouse rotation/zoom. `prepare_terrain_scene(payload)` for CPU-side mesh prep.
- [x] **9C.3 — Interactive demo** — `scripts/view_globe.py` loads a globe export JSON (or generates inline) and renders the terrain-coloured polyhedron with rotation/zoom.
- [ ] **9C.4 — Textured mesh builder** (stretch goal) — produce UV-mapped meshes that reference the per-tile textures from 9B, for sub-tile detail rendering.
- [x] **9C.5 — Tests:**
  - Coloured mesh has correct vertex count
  - Mesh vertex/index counts match reference terrain_layout_mesh
  - Scene preparation with/without edges
  - Edge mesh builder

## Phase 10 — Sub-Tile Detail Rendering �

**Goal:** Replace the current flat-colour-per-tile rendering with high-resolution per-tile terrain using full polygrids inside each Goldberg tile (hex grids for hexagons, pentagon-centred grids for pentagons). The globe-scale terrain from Phase 8 provides the macro heightfield; Phase 10 adds intra-tile detail that brings us toward satellite-imagery realism (see reference image: organic ridges, river valleys, vegetation gradients, coastal water).

### Design overview

The approach has three layers:

1. **Globe layer** (existing) — a `GlobeGrid` with one face per Goldberg tile. `generate_mountains()` / `generate_rivers()` produce a coarse elevation field (92 tiles at freq=3, 252 at freq=5). This layer controls the large-scale terrain character: which tiles are ocean, lowland, highland, mountain, etc.

2. **Detail layer** (new) — each Goldberg tile is expanded into a local `PolyGrid` (hex grid for hex tiles, pentagon-centred grid for pent tiles) with many sub-faces. The parent tile's elevation, biome, and neighbours' elevations seed and constrain the detail terrain so that sub-tile elevation transitions are smooth across Goldberg tile boundaries.

3. **Render layer** (new) — each detail grid is rendered to a small PNG texture using elevation-to-colour mapping, hillshade lighting, and river overlays. The textures are UV-mapped onto the 3D Goldberg tile surfaces in the OpenGL viewer, replacing the current flat vertex colours.

Key constraint: **boundary continuity**. Adjacent Goldberg tiles share an edge. The detail grids inside those tiles must produce compatible elevation along their shared boundary — no visible seam where two textures meet. This is achieved by sharing elevation values at boundary faces and interpolating between parent tile elevations.

### 10A — Detail Grid Infrastructure (`tile_detail.py`) ✅

Refactor and extend `detail_grid.py` into a production-ready system for building and managing detail grids across all globe tiles.

- [x] **10A.1 — `TileDetailSpec` dataclass** — configuration for detail grid generation:
  - `detail_rings` (int, default 4) — ring count for sub-tile grids. Controls resolution: a hex grid with 4 rings has 61 sub-faces per tile; with 6 rings it has 127.
  - `noise_frequency` (float, default 6.0) — spatial frequency of intra-tile noise
  - `noise_octaves` (int, default 5) — detail noise octaves
  - `amplitude` (float, default 0.12) — how much local noise varies from the parent elevation
  - `base_weight` (float, default 0.8) — parent elevation dominance (0–1)
  - `boundary_smoothing` (int, default 2) — smoothing passes at tile boundaries
  - `seed_offset` (int, default 0) — added to parent seed for per-tile variation

- [x] **10A.2 — `build_all_detail_grids(globe_grid, spec)` → `Dict[str, PolyGrid]`** — build a detail grid for every globe tile in one call. Returns `{face_id: detail_grid}`. Each detail grid's metadata stores `parent_face_id`, `parent_face_type`, `parent_elevation`.

- [x] **10A.3 — `DetailGridCollection` container** — manages the full set of detail grids:
  - `.grids` — dict of `{face_id: PolyGrid}`
  - `.stores` — dict of `{face_id: TileDataStore}`
  - `.get(face_id)` → `(PolyGrid, TileDataStore)`
  - `.total_face_count` — sum of all sub-faces across all tiles
  - `.generate_all_terrain(globe_grid, globe_store, spec)` — populate elevation for every detail grid
  - `.summary()` → human-readable stats string

- [x] **10A.4 — Tests:**
  - `build_all_detail_grids` produces one grid per globe tile
  - Hex tiles get hex grids, pent tiles get pent grids
  - Correct sub-face counts for the given ring count
  - `DetailGridCollection` stores and retrieves grids correctly
  - Total face count matches sum of `detail_face_count` per tile type

### 10B — Boundary-Aware Detail Terrain (`detail_terrain.py`) ✅

The critical piece: generate intra-tile terrain that is continuous across Goldberg tile boundaries. Without this, each tile's texture would have hard seam lines where it meets its neighbours.

- [x] **10B.1 — `compute_boundary_elevations(globe_grid, globe_store)` → `Dict[str, Dict[int, float]]`** — for each globe tile, compute the elevation that each of its edges should have at the boundary. This is the average of the parent tile's elevation and each neighbour's elevation along that shared edge. Returns `{face_id: {edge_index: boundary_elevation}}`.

- [x] **10B.2 — `classify_detail_faces(detail_grid, boundary_depth)` → `Dict[str, str]`** — for each face in a detail grid, classify it as `"interior"`, `"boundary"`, or `"corner"`. Boundary faces are those within `boundary_depth` rings of the grid edge. Corner faces are at vertices shared by 3+ Goldberg tiles.

- [x] **10B.3 — `generate_detail_terrain_bounded(detail_grid, parent_elevation, neighbor_elevations, spec)` → `TileDataStore`** — enhanced version of `generate_detail_terrain` that:
  1. Assigns parent elevation as the base for interior faces
  2. Interpolates toward neighbour elevations for boundary faces (lerp based on distance from edge)
  3. Adds high-frequency noise on top (fbm, domain-warped for organic shapes)
  4. Smooths the boundary band to eliminate discontinuities
  5. Applies the parent tile's terrain character (ridge direction, biome type) to influence the local noise parameters

- [x] **10B.4 — `generate_all_detail_terrain(collection, globe_grid, globe_store, spec)`** — populate terrain for every detail grid in a `DetailGridCollection`, using boundary-aware generation. This is the main entry point for terrain.

- [x] **10B.5 — Tests:**
  - Boundary faces have elevations between parent and neighbour elevations
  - Interior faces cluster around parent elevation
  - Adjacent tile boundary faces have similar elevations (seam test: max difference < threshold)
  - Determinism: same inputs → identical outputs
  - No NaN or infinite values

### 10C — Enhanced Colour Ramps & Biome Rendering (`detail_render.py`) ✅

The target image shows rich, multi-tonal terrain: not just elevation-banded colour, but vegetation gradients, exposed rock, water. This step enhances the colour system for satellite realism.

- [x] **10C.1 — `BiomeConfig` dataclass** — per-biome rendering parameters:
  - `base_ramp` — colour ramp name (satellite, terrain, etc.)
  - `vegetation_density` (float, 0–1) — how much green appears at low/mid elevations
  - `rock_exposure` (float, 0–1) — how much bare rock shows at high elevations
  - `snow_line` (float) — elevation above which snow appears
  - `water_level` (float) — elevation below which water appears
  - `moisture` (float, 0–1) — affects green vs brown balance

- [x] **10C.2 — `_RAMP_DETAIL_SATELLITE` colour ramp** — a richer satellite ramp with more control points, designed for sub-tile resolution:
  - Deep water → shallow water → sandy coast → lowland green → lush vegetation → dry grass → highland brown → exposed rock → grey scree → snow line → snow
  - At least 12 control points for smooth gradients

- [x] **10C.3 — `detail_elevation_to_colour(elevation, biome_config, *, hillshade, moisture_noise)` → `(r, g, b)`** — per-face colour function that combines:
  - Base elevation colour from ramp
  - Hillshade darkening (light direction, slope from neighbours)
  - Vegetation noise overlay (patchy green at mid-elevations)
  - Rock/scree noise at high elevations
  - Snow line with fractal edge (not a hard cutoff)

- [x] **10C.4 — `render_detail_texture_enhanced(detail_grid, store, output_path, biome_config)` → Path** — render a detail grid texture using the enhanced colour system. Produces a higher-quality PNG than the existing `render_detail_texture`.

- [x] **10C.5 — Tests:**
  - Colour output is valid RGB for all elevation values
  - Hillshade darkens south-facing slopes
  - Water colour appears below water_level
  - Snow appears above snow_line
  - Vegetation noise varies between faces (not uniform)

### 10D — Texture Atlas & UV Mapping (`texture_pipeline.py`) ✅

Build the full texture pipeline: render all detail grids, assemble into a texture atlas, and map UVs.

- [x] **10D.1 — `build_detail_atlas(collection, biome_config, output_dir, *, tile_size)` → `(Path, Dict)`** — render every detail grid to a texture, then assemble into an atlas. Returns `(atlas_path, uv_layout)` where `uv_layout` maps `face_id → (u_min, v_min, u_max, v_max)` in atlas UV space.

- [x] **10D.2 — `compute_tile_uvs(tile, atlas_layout)` → `List[Tuple[float, float]]`** — for a `GoldbergTile`, compute per-vertex UV coordinates that map into the correct atlas slot. Uses the tile's `uv_vertices` (already provided by the models library as normalised 2D projections of the tile's 3D vertices onto its tangent plane).

- [x] **10D.3 — `build_textured_tile_mesh(tile, atlas_layout)` → `ShapeMesh`** — build a per-tile mesh with position(3) + color(3) + uv(2) where the UV coordinates point into the atlas. Colour is set to white `(1, 1, 1)` so the texture provides all colour information.

- [x] **10D.4 — `build_textured_globe_meshes(frequency, atlas_layout, *, radius)` → `List[ShapeMesh]`** — build textured meshes for all tiles. Each mesh's UVs are mapped to its slot in the atlas.

- [x] **10D.5 — Tests:**
  - Atlas image exists and has correct dimensions
  - UV layout covers all tiles
  - UVs are within [0, 1] range
  - Per-tile mesh has correct vertex count and stride

### 10E — Textured OpenGL Renderer (`globe_renderer.py` extension) ✅

Extend the existing OpenGL renderer to support texture-mapped tiles instead of (or in addition to) flat vertex colours.

- [x] **10E.1 — Textured shader pair** — new vertex/fragment shaders that:
  - Vertex shader: pass through UV coordinates, compute world position and normal
  - Fragment shader: sample the atlas texture at the interpolated UV, apply directional lighting, output final colour
  - Fallback: if no texture, use vertex colour (backward compatible with current renderer)

- [x] **10E.2 — `render_textured_globe_opengl(payload, atlas_path, uv_layout, ...)`** — new entry point:
  1. Load the atlas image as an OpenGL texture
  2. Build per-tile textured meshes with atlas UVs
  3. Upload to `SimpleMeshRenderer`
  4. Render with the textured shader, binding the atlas texture
  5. Same mouse rotation/zoom interaction as current viewer

- [x] **10E.3 — `view_globe.py` updates** — add `--detail-rings N` and `--textured` flags:
  - `--textured`: generate detail grids, render textures, build atlas, launch textured renderer
  - `--detail-rings 4`: control sub-tile resolution (default: 4)
  - Without `--textured`: same flat-colour rendering as before (backward compatible)

- [ ] **10E.4 — Tests:**
  - Textured shader compiles and links (unit test with mock context)
  - Textured mesh has correct stride (position + color + uv = 32 bytes)
  - Atlas texture loading produces a valid texture ID
  - Renderer falls back to vertex colour when no atlas provided

### 10F — Performance & Scale (`detail_perf.py`) ✅

At frequency=5 (252 tiles) with detail_rings=4 (61 sub-faces per hex tile), we have ~15,000 sub-faces total. At detail_rings=6 we hit ~32,000. This step ensures the pipeline scales.

- [x] **10F.1 — Parallel detail grid generation** — use `concurrent.futures.ProcessPoolExecutor` to generate detail terrains in parallel (each tile is independent except for boundary values, which are precomputed).

- [x] **10F.2 — Texture rendering optimisation** — batch matplotlib renders or switch to direct PIL/numpy rendering for detail textures (matplotlib is slow for hundreds of small renders).

- [x] **10F.3 — Atlas packing optimisation** — pack textures tightly (hexagons and pentagons have different aspect ratios). Consider hex-shaped texture regions rather than square tiles.

- [x] **10F.4 — Caching** — cache generated detail grids and textures to disk. Only regenerate when the parent terrain or spec changes. Keyed by `(face_id, spec_hash, parent_elevation)`.

- [x] **10F.5 — Benchmarks and profiling** — measure time for: detail grid construction, terrain generation, texture rendering, atlas assembly, OpenGL upload. Target: < 5 seconds for freq=3 with detail_rings=4; < 30 seconds for freq=5 with detail_rings=6.

- [x] **10F.6 — Tests:**
  - Parallel generation produces identical results to serial
  - Cached results match fresh generation
  - Pipeline completes within timeout for freq=3, detail_rings=4

### 10G — Demo & Integration ✅

End-to-end demos showing the full pipeline.

- [x] **10G.1 — `scripts/demo_detail_globe.py`** — CLI script:
  - `python scripts/demo_detail_globe.py --frequency 4 --detail-rings 4 --preset mountain_range`
  - Generates globe → mountains → detail grids → textures → atlas → 3D render
  - Outputs: `exports/detail_atlas.png`, `exports/detail_globe.png`, individual tile textures in `exports/detail_tiles/`

- [x] **10G.2 — Side-by-side comparison** — render the same globe at multiple detail levels (flat colour, detail_rings=2, detail_rings=4, detail_rings=6) in a 2×2 panel for visual comparison.

- [x] **10G.3 — Higher frequency demo** — `--frequency 5 --detail-rings 4` producing a globe with 252 tiles × 61 sub-faces = ~15,000 visible terrain cells — enough to start seeing realistic terrain patterns.

- [ ] **10G.4 — Documentation** — update `README.md` and `JSON_CONTRACT.md` with the detail grid system, atlas format, and viewer usage.

### Summary — Phase 10 Implementation Order

| Step | Module | Depends on | Delivers | Status |
|------|--------|-----------|----------|--------|
| 10A | `tile_detail.py` | detail_grid.py, globe.py | Detail grid infrastructure | ✅ |
| 10B | `detail_terrain.py` | 10A + heightmap + noise | Boundary-aware sub-tile terrain | ✅ |
| 10C | `detail_render.py` | 10B + terrain_render | Enhanced colour & texture rendering | ✅ |
| 10D | `texture_pipeline.py` | 10C + atlas | Full texture pipeline with UV mapping | ✅ |
| 10E | `globe_renderer.py` ext | 10D + OpenGL | Textured 3D rendering | ✅ |
| 10F | `detail_perf.py` | 10A–10E | Performance: parallelism, caching | ✅ |
| 10G | demos + docs | 10A–10F | End-to-end demos and documentation | ✅ |

### Design notes — Resolution scaling

| Frequency | Tiles | detail_rings=4 | detail_rings=6 | detail_rings=8 |
|-----------|-------|-----------------|-----------------|-----------------|
| 3 | 92 | ~5,500 sub-faces | ~11,500 | ~20,000 |
| 4 | 162 | ~9,800 | ~20,400 | ~35,500 |
| 5 | 252 | ~15,300 | ~31,800 | ~55,300 |
| 8 | 642 | ~39,000 | ~81,000 | ~141,000 |

The target image's level of detail corresponds roughly to freq=5 with detail_rings=6–8. Starting at freq=3, detail_rings=4 gives a good first visual while keeping iteration fast (< 5 seconds).

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
| 8 (Globe) | `models>=0.1.1` (optional, under `globe` extra) |
| 9 (Export) | `Pillow` for PNG texture output |
