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
- [x] **9C.4 — Textured mesh builder** (stretch goal) — superseded by Phase 12–13's `globe_renderer_v2.py` which provides batched, subdivided, PBR-lit textured mesh rendering with atlas UV mapping.
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

- [x] **10E.4 — Tests:** Covered in `test_globe.py`: `test_textured_vertex_shader_is_string`, `test_textured_fragment_shader_is_string`, `test_textured_vertex_shader_has_uv_passthrough`, `test_textured_fragment_shader_fallback`, `test_textured_shaders_version_330`, `test_textured_mesh_stride`, `test_textured_mesh_vertex_count`, `test_textured_mesh_has_uv_attribute`, `test_textured_mesh_uvs_within_atlas_slot`.

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

- [x] **10G.4 — Documentation** — updated `README.md` (full Phase 10–13 coverage, package layout, CLI, usage examples), `JSON_CONTRACT.md` (added atlas/UV layout and vertex format sections), `ARCHITECTURE.md` (updated layer diagram, added terrain/globe/detail/rendering layers, Phase 12–13 shader architecture), `MODULE_REFERENCE.md` (all 40 source modules with line counts and dependencies).

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

## Phase 11 — Cohesive Globe Terrain �

**Goal:** Replace the current per-tile-in-isolation terrain generation with a system that produces terrain features that span across Goldberg tile boundaries — mountain ranges that flow across multiple tiles, river systems that cross tile edges, and vegetation gradients that are globally coherent. The current Phase 10 system generates each tile's sub-faces independently with per-tile noise seeds, which produces a "patchwork quilt" artefact: every tile has its own terrain character with visible seam boundaries even after boundary smoothing.

### Problem analysis

The root cause is architectural. Currently:

1. **Noise is local.** Each detail grid is sampled with a per-tile seed (`tile_seed = seed + hash(face_id)`). The noise field has no spatial relationship between adjacent tiles — tile A's interior noise and tile B's interior noise are unrelated functions.

2. **Boundary smoothing is a patch, not a solution.** `generate_detail_terrain_bounded` averages parent/neighbour elevations at the boundary band and smooths. This prevents hard seam *lines*, but it can't make a ridge that crosses from tile A into tile B because the interior noise on each side was generated independently.

3. **Globe-level terrain is too coarse.** `generate_mountains` writes one elevation value per Goldberg tile (92 values at freq=3). The detail layer adds high-frequency variation on top, but the large-scale shape (ridges, valleys) is locked to the coarse grid.

### Approach: 3D-Coherent Noise + Region Patches

The key insight: **sample a single, globe-wide noise field** at every sub-face position, using the sub-face's real 3D coordinates (derived from its parent tile's transform). This makes the noise field continuous across the entire sphere regardless of tile boundaries.

There are three layers to this approach:

**Layer 1 — Globe-coherent noise via 3D coordinates ("easy win")**
Instead of per-tile noise with local (x, y) coordinates and per-tile seeds, transform each sub-face centroid into globe 3D space and sample a single `fbm_3d(x, y, z)` field. The existing `noise.py` already has `fbm_3d` and `ridged_noise_3d`. The existing `heightmap.py` already has `sample_noise_field_3d`. This alone eliminates the patchwork problem because the noise is spatially continuous on the sphere.

**Layer 2 — Region patches for feature-scale terrain ("medium complexity")**
Group adjacent globe tiles into *terrain patches* (using the existing region/partition system on the globe grid). Within each patch, apply a coherent terrain recipe — e.g. a mountain range that spans 5–8 tiles, or an ocean basin that spans 10 tiles. Each patch uses the same noise parameters so features flow naturally across tile boundaries within the patch.

**Layer 3 — Full-resolution algorithms on stitched sub-grids ("stretch")**
For the highest quality, stitch together the detail grids of adjacent tiles into a combined PolyGrid (using the existing `CompositeGrid` / `stitch_grids` machinery), run terrain algorithms (mountains, rivers) on the combined grid, then split the results back to per-tile stores for texture rendering. This is the most complex but gives the most realistic results — rivers that flow across tiles, ridge lines that span a region.

### 11A — 3D-Coherent Noise Sampling (`detail_terrain_3d.py`) ✅

The critical first step: make sub-face noise globally continuous by using 3D sphere coordinates instead of local 2D grid coordinates.

- [x] **11A.1 — `compute_subface_3d_position(globe_grid, face_id, detail_grid, sub_face_id)` → `(x, y, z)`** — given a sub-face in a detail grid, compute its approximate 3D position on the globe sphere. Uses the parent tile's `center_3d`, `normal_3d`, and transform to project the sub-face's local 2D position onto the sphere surface (tangent-plane projection + normalisation to sphere radius).

- [x] **11A.2 — `precompute_3d_positions(globe_grid, face_id, detail_grid)` → `Dict[str, Tuple[float, float, float]]`** — batch version: compute 3D positions for all sub-faces in a detail grid. Cache-friendly.

- [x] **11A.3 — `generate_detail_terrain_3d(detail_grid, positions_3d, spec, *, seed)` → `TileDataStore`** — replacement for `generate_detail_terrain_bounded` that:
  1. Samples `fbm_3d(x, y, z)` at each sub-face's 3D position (globally coherent — same seed for all tiles)
  2. Layers `ridged_noise_3d` for mountain ridges that span tile boundaries
  3. Blends with parent elevation for macro-scale shape
  4. No per-tile seed — the spatial position IS the seed
  - Params: `Terrain3DSpec` with `ridge_weight`, `fbm_weight`, `base_weight`

- [x] **11A.4 — `generate_all_detail_terrain_3d(collection, globe_grid, globe_store, spec)` → None** — batch entry point: precompute all 3D positions, then generate terrain for every tile using the global noise field. Drop-in replacement for `generate_all_detail_terrain`.

- [x] **11A.5 — Tests:** 29 tests in `tests/test_detail_terrain_3d.py`
  - Adjacent tiles' boundary sub-faces have similar elevation (tighter threshold than current)
  - Same sub-face position produces same elevation regardless of which tile "owns" it
  - No patchwork artefact: elevation variance across tile boundaries ≈ elevation variance within tiles
  - Determinism: same inputs → identical output
  - Performance: not significantly slower than current approach

### 11B — Terrain Patches via Globe Regions (`terrain_patches.py`) ✅

Group globe tiles into terrain patches (continents, mountain ranges, ocean basins) and apply patch-wide terrain recipes. This gives the globe-level terrain more structure than plain noise.

- [x] **11B.1 — `TerrainPatch` dataclass** — a named group of globe face IDs with a terrain recipe:
  - `name` (str) — e.g. `"mountain_range_1"`, `"ocean_basin_1"`
  - `face_ids` (List[str]) — globe tile IDs in this patch
  - `terrain_type` (str) — `"mountain"`, `"ocean"`, `"plains"`, `"hills"`, `"desert"`
  - `params` (dict) — noise parameters specific to this patch (frequency, octaves, ridge weight, etc.)
  - `elevation_range` (Tuple[float, float]) — target min/max elevation for this patch

- [x] **11B.2 — `generate_terrain_patches(globe_grid, *, n_patches, seed)` → `List[TerrainPatch]`** — auto-generate terrain patches:
  1. Use `partition_noise(globe_grid, seeds)` to create organic regions
  2. Assign terrain types based on region size and position (large = ocean/plains, small + clustered = mountains)
  3. Assign noise parameters per terrain type (mountains get high ridge weight + frequency, oceans get low amplitude)
  4. Return patch list

- [x] **11B.3 — `apply_terrain_patches(collection, globe_grid, globe_store, patches, spec)` → None** — for each patch:
  1. Collect all sub-faces from all tiles in the patch
  2. Precompute their 3D positions
  3. Sample terrain using patch-specific noise parameters + the global 3D noise field
  4. Apply patch-specific elevation range normalisation
  5. Smooth boundaries between patches (cross-patch boundary faces get blended)

- [x] **11B.4 — Preset terrain distributions:**
  - `"earthlike"` — ~30% ocean, ~20% plains, ~15% hills, ~25% forest/vegetation, ~10% mountains
  - `"mountainous"` — ~60% mountains/hills, ~20% highland plains, ~20% valleys
  - `"archipelago"` — ~70% ocean, ~30% scattered island clusters
  - `"pangaea"` — one large continent, rest ocean

- [x] **11B.5 — Tests:**
  - Patches cover all globe tiles (no gaps)
  - Mountain patches produce higher elevation than ocean patches
  - Cross-patch boundaries are smooth (no hard terrain-type transitions)
  - Different presets produce measurably different terrain distributions

### 11C — Stitched Sub-Grid Terrain (`region_stitch.py`) ✅

For the highest quality: stitch together detail grids of adjacent tiles into a combined PolyGrid, run existing terrain algorithms on the combined grid, then split results back.

- [x] **11C.1 — `stitch_detail_grids(collection, face_ids)` → `(PolyGrid, FaceMapping)`** — merge the detail grids for a set of adjacent globe tiles into a single PolyGrid:
  1. Transform each detail grid's vertices from local space to 3-D globe positions (tangent-plane transform)
  2. Project to shared 2-D via gnomonic projection at group centroid
  3. Merge coincident boundary vertices (within tolerance)
  4. Return combined grid + mapping `{combined_face_id: (original_tile_id, original_sub_face_id)}`

- [x] **11C.2 — `generate_terrain_on_stitched(combined_grid, face_mapping, globe_grid, globe_store, spec)` → `TileDataStore`** — run terrain generation on the stitched grid:
  1. 2-D noise sampling (fbm + ridged) on gnomonic-projected coordinates
  2. Smoothing crosses former tile boundaries seamlessly
  3. The combined grid can be large (a patch of 8 tiles × 61 sub-faces = ~500 faces) — still manageable

- [x] **11C.3 — `split_terrain_to_tiles(combined_store, mapping, collection)` → None** — distribute the combined grid's elevation data back into per-tile stores using the face-id mapping. Each tile's store gets exactly the sub-faces that belong to it.

- [x] **11C.4 — `generate_stitched_patch_terrain(collection, globe_grid, globe_store, face_ids, spec)` → `TileDataStore`** — end-to-end for one terrain patch:
  1. Stitch detail grids for the patch's tiles
  2. Generate terrain on combined grid
  3. Split results back to tile stores

- [x] **11C.5 — Tests (19 passing):**
  - Combined grid has correct face count matching source tiles
  - Face mapping is complete and bidirectional
  - Gnomonic coordinates are valid and reasonable
  - Split-back values exactly match combined grid
  - Deterministic: same inputs → identical output
  - Cross-tile elevation is continuous
  - End-to-end convenience function works

### 11D — Enhanced Mountain & River Generation (`globe_terrain.py`) ✅

Tune the existing terrain algorithms for globe-scale realism.

- [x] **11D.1 — `MountainConfig3D` presets** — globe-optimised mountain presets:
  - `GLOBE_MOUNTAIN_RANGE` — long ridges spanning tiles (freq=6.0)
  - `GLOBE_VOLCANIC_CHAIN` — isolated peaks along curved paths (freq=12.0)
  - `GLOBE_CONTINENTAL_DIVIDE` — single dominant ridge (freq=3.0)

- [x] **11D.2 — Globe-scale river generation** — `generate_rivers_on_stitched()` runs on combined grids so rivers flow across tile boundaries. Includes depression filling, flow accumulation, Strahler ordering, and valley carving.

- [x] **11D.3 — Erosion simulation** — `erode_terrain()` with `ErosionConfig`:
  1. Drop virtual "water particles" at random high-elevation sub-faces
  2. Flow downhill (steepest descent), depositing sediment and eroding
  3. Creates realistic valley shapes, alluvial fans, and drainage patterns
  4. Operates on stitched sub-grids for cross-tile continuity

- [x] **11D.4 — Tests (17 passing):**
  - Mountain ridges span multiple tiles
  - Rivers cross tile boundaries (verified in 6/6 tiles)
  - Erosion reduces peak elevation
  - Deterministic: same inputs → same output
  - Flat terrain: no erosion, no rivers

### 11E — Rendering Enhancements ✅

Update the rendering pipeline for the improved terrain.

- [x] **11E.1 — Seamless texture rendering** — `render_seamless_texture()` renders tiles with auto biome selection and fallback for missing stores. Delegates to `render_detail_texture_enhanced()`.

- [x] **11E.2 — Elevation-dependent biome assignment** — `assign_biome()` auto-classifies tiles: ocean (mean<0.15), snow (>0.70), mountain (>0.55), desert (flat+low), vegetation (default). Five presets: `OCEAN_BIOME`, `VEGETATION_BIOME`, `MOUNTAIN_BIOME`, `DESERT_BIOME`, `SNOW_BIOME`.

- [x] **11E.3 — Normal-map generation** — `compute_normal_map()` derives per-face unit normals from neighbour elevation gradients with configurable vertical exaggeration. `compute_all_normal_maps()` for batch processing.

- [x] **11E.4 — Tests:** 32 tests in `tests/test_render_enhanced.py` — all passing.
  - Biome presets valid and complete
  - Elevation-threshold biome assignment (ocean, snow, mountain, desert, vegetation)
  - Boundary conditions and empty-grid fallback
  - Batch assignment across full collection
  - Flat-terrain normals point up; gradient normals tilt; scale increases tilt
  - Unit-length validation across all normals
  - Deterministic assignments and normals
  - Texture file output, biome override, no-store fallback, seed variation

### 11F — Demo & Comparison ✅

- [x] **11F.1 — `scripts/demo_cohesive_globe.py`** — updated end-to-end demo running the full Phase 11 pipeline:
  - `python scripts/demo_cohesive_globe.py -f 3 --detail-rings 4 --preset earthlike`
  - Pipeline: globe → terrain patches (11B) → 3D-coherent noise (11A) → stitched region terrain (11C) → mountains + rivers + erosion (11D) → auto biome assignment + seamless textures + normal maps (11E) → comparison panel
  - Supports `--preset` (earthlike/mountainous/archipelago/pangaea), `--bench` for performance, `--view` for 3D

- [x] **11F.2 — Before/after gallery** — comparison panel exported as `comparison.png` with Phase 10 vs Phase 11 side-by-side. Biome-assigned tiles rendered separately. Normal-map sample saved as JSON.

- [x] **11F.3 — Performance comparison** — `--bench` flag runs 3× timed iterations of Phase 10 and Phase 11 pipelines with mean ± stdev. Inline timing for each pipeline stage.

### Summary — Phase 11 Implementation Order

| Step | Module | Depends on | Delivers | Complexity |
|------|--------|-----------|----------|------------|
| 11A | `detail_terrain_3d.py` | noise.py (3D), globe.py | Globally-coherent noise field | Low-Medium |
| 11B | `terrain_patches.py` | 11A + regions.py | Structured terrain distribution | Medium |
| 11C | `region_stitch.py` | 11B + composite.py | Stitched sub-grid terrain gen | High |
| 11D | mountains/rivers extensions | 11A–C | Globe-scale terrain features | Medium |
| 11E | rendering enhancements | 11A–D + detail_render | Seamless textures, biomes, normals | Medium |
| 11F | demos + docs | 11A–E | End-to-end demos, comparison | Low |

### Design notes — Why not merge ALL sub-faces into one giant PolyGrid?

It's tempting to merge all 92×61 = ~5,600 sub-faces (at freq=3, detail_rings=4) into a single PolyGrid and run terrain algorithms once. This would give perfect continuity but has serious drawbacks:

1. **Stitching complexity.** The existing `stitch_grids` works on macro-edges between grids with compatible boundary vertex counts. Detail grids from adjacent Goldberg tiles don't share a macro-edge structure — they're different grid topologies (some are pent-centred, some are hex) with different boundary shapes. Stitching 92 arbitrary-shaped grids together is a substantial engineering effort.

2. **Scale.** At freq=5 with detail_rings=6 we'd have ~32,000 sub-faces in one grid. Algorithms like `generate_rivers` (which runs BFS/DFS on every face) would become slow. The current per-tile approach parallelises naturally.

3. **Memory.** A single combined TileDataStore with 32k faces and multiple fields becomes a memory concern.

The **region-patch approach (11B–C)** is the sweet spot: stitch together groups of 5–15 adjacent tiles at a time, run terrain on each group, then split back. This keeps each combined grid at ~300–900 faces (manageable), enables cross-tile features, and parallelises at the patch level.

---

## Phase 12 — Rendering Quality ✅

**Goal:** Fix the three root causes of "gappy" 3D globe rendering:
(1) black texture borders bleeding, (2) flat faceted tiles, (3) per-tile draw call overhead.

### 12A — Texture Flood Fill ✅

- [x] **12A.1 — `flood_fill_tile_texture()`** — iterative dilation: black pixels adjacent to coloured pixels are replaced with the average of their coloured neighbours. Repeated for *N* iterations to fill the full border region around each tile's textured polygon.
- [x] **12A.2 — `flood_fill_atlas()`** — convenience wrapper that applies the same flood-fill to an entire atlas image.
- [x] **12A.3 — Tests** — 7 tests: reduces black pixels, preserves coloured centre, overwrites in place, edge pixels get colour, atlas alias, zero iterations, all-coloured.

### 12B — Sphere Subdivision ✅

- [x] **12B.1 — `subdivide_tile_mesh()`** — each tile's triangle fan (center → v[i] → v[i+1]) is subdivided into a grid of `subdivisions²` smaller triangles using barycentric interpolation, then each vertex is projected onto the sphere surface (`normalize(pos) * radius`). This eliminates flat faceting.
- [x] **12B.2 — Vertex deduplication** — shared boundary vertices between adjacent sub-triangles within a tile are merged by position (rounded to 7 decimal places).
- [x] **12B.3 — Tests** — 14 tests: correct shapes, tri count at s=1/2/3, hex and pent, vertices on sphere, custom radius, UV range, colour propagation, no degenerate triangles, index bounds, deduplication.

### 12C — Batched Globe Mesh ✅

- [x] **12C.1 — `build_batched_globe_mesh()`** — iterates all Goldberg tiles, calls `subdivide_tile_mesh()` for each, concatenates all vertex and index arrays with offset indexing into a single VBO + IBO.
- [x] **12C.2 — Tests** — 9 tests: output shapes, nonzero, all on sphere, valid indices, more tris with higher subdiv, colour map, empty layout, f=3 tile count (540 tris at s=1), custom radius.

### 12D — Interactive Renderer v2 ✅

- [x] **12D.1 — `render_globe_v2()`** — pyglet 3D viewer using:
  - Flood-filled atlas (black border removal)
  - Subdivided + sphere-projected mesh (smooth curvature)
  - Single VBO + IBO + `glDrawElements` draw call (92 tiles → 1 call)
  - MSAA (4× multisampling, graceful fallback)
  - Mipmap-filtered atlas texture
  - Hemisphere lighting (directional + ambient)
  - Mouse drag rotation, scroll zoom
- [x] **12D.2 — Demo integration** — `demo_cohesive_globe.py --view` now launches v2 renderer.
- [x] **12D.3 — GLSL shaders** — v2 vertex/fragment shaders with per-vertex position/colour/UV, model/MVP uniforms, atlas sampling, hemisphere lighting.

### Summary — Phase 12

| Fix | Module | Problem solved | Impact |
|-----|--------|---------------|--------|
| 12A | `globe_renderer_v2.py` | Black texture borders → visible seams | Eliminates black gaps |
| 12B | `globe_renderer_v2.py` | Flat triangle fans → faceted look | Smooth sphere curvature |
| 12C | `globe_renderer_v2.py` | 92 draw calls → overhead | Single draw call |
| 12D | `globe_renderer_v2.py` | Combined viewer | Production-quality viewer |

**Tests:** 33 total (7 flood-fill + 14 subdivision + 3 helpers + 9 batched mesh)

### 12 — Retrospective

Phase 12 addressed sphere subdivision (smooth curvature) and batched draw calls (performance), but the **black seam problem remains visually dominant**. Root cause analysis:

- Each tile's detail texture is a hex/pent polygon rendered on a black background. The polygon occupies ~56% of the square tile slot; the remaining ~44% is black.
- The GoldbergTile UV vertices span [0,1]² but form a hex/pent inscribed in that square. Triangle fan subdivision maps UV coordinates into the atlas, and any UV sample that falls outside the polygon outline samples black.
- The flood-fill dilation (12A) only extends a few pixels from the polygon border — too little to cover the large black corners.
- Result: every tile boundary shows a thick black seam where adjacent tiles' UV triangles overlap the black corner regions.

The fundamental fix requires **ensuring no UV sample ever sees a black pixel**, which means either (a) filling the *entire* tile slot with colour, (b) clamping/remapping UVs to stay inside the polygon, or (c) rendering the texture differently so there are no black corners at all.

---

## Phase 13 — Cohesive Globe Rendering 🔶

**Goal:** Achieve a production-quality globe where tile boundaries are invisible, the surface is smooth and continuous, lighting is realistic, and rendering is efficient. This phase replaces the broken flood-fill approach with a comprehensive solution.

### Problem analysis (from screenshot review)

1. **Black seams (dominant)** — 44% of each atlas tile slot is black. UV interpolation across tile triangle fans samples these black regions at every tile boundary. Flood-fill only covers a thin border ring.
2. **Colour discontinuity** — adjacent tiles have abrupt colour/biome transitions because each tile's texture is rendered independently with per-tile noise seeds and biome assignment.
3. **Texture aliasing** — 256×256 tile textures with 61 sub-faces each are coarse. When sampled at oblique angles (near globe edges), aliasing makes seams more visible.
4. **Lighting flatness** — current hemisphere lighting (`dot(N,L)*0.6+0.4`) is simplistic. No ambient occlusion, no normal mapping, no specular.

### Architecture: Three-layer solution

```
Layer 1: Texture Pipeline    — eliminate black pixels entirely
Layer 2: Mesh & UV Pipeline  — seamless UV mapping across tile boundaries
Layer 3: Shader & Lighting   — realistic PBR-lite rendering
```

---

### 13A — Full-coverage tile textures ✅

**Problem:** `render_detail_texture_enhanced()` renders sub-face polygons on a black `facecolor="black"` background. The hex/pent polygon doesn't fill the square tile slot.

**Fix:** Render tile textures so the *entire* square image is filled with terrain colour.

- [x] **13A.1 — Background colour fill** — In `detail_render.py`, change `facecolor="black"` to the tile's average terrain colour. This ensures corners outside the polygon are terrain-coloured instead of black.
- [x] **13A.2 — PIL renderer fix** — Same fix in `detail_perf.py`: replace `Image.new("RGB", ..., (0,0,0))` with the tile's average colour.
- [x] **13A.3 — Polygon edge extension** — Extend the outermost sub-face polygons by 2-3 pixels beyond the tile boundary so they overlap slightly into the border region, ensuring full coverage.
- [x] **13A.4 — Aggressive flood-fill** — Replace the fixed-iteration flood-fill with a full diffusion fill: repeat until *all* black pixels below a threshold are filled. Use a proper distance-weighted propagation (nearest-coloured-pixel sampling) instead of iterative averaging.
- [x] **13A.5 — Tests** — full-coverage assertions: after rendering, no pixel in the tile slot should be below a minimum brightness threshold. Verify corner pixels are terrain-coloured.

### 13B — Atlas gutter system ✅

**Problem:** Adjacent tile slots in the atlas share pixel boundaries. Bilinear/trilinear sampling bleeds pixels across slot boundaries, creating seams even when tiles are individually correct.

**Fix:** Add a gutter (padding) around each atlas slot filled with neighbouring tile colours.

- [x] **13B.1 — Gutter parameter** — Add `gutter_px: int = 4` parameter to `build_detail_atlas()` and `build_detail_atlas_fast()`. Each tile slot expands by `gutter_px` on all sides in the atlas image.
- [x] **13B.2 — Gutter fill strategy** — Fill gutter pixels by mirroring/clamping the tile edge pixels. For a 4px gutter, the 4 outermost pixel columns/rows of each tile are duplicated into the gutter.
- [x] **13B.3 — UV layout adjustment** — Update `compute_tile_uvs()` to account for the gutter: atlas UVs map to the inner (non-gutter) region. UV coordinates must be inset by `gutter_px / atlas_size`.
- [x] **13B.4 — Tests** — verify gutter pixels match tile edge pixels; verify UVs after inset still map correctly.

### 13C — UV inset clamping ✅

**Problem:** The GoldbergTile UV vertices (e.g. pentagon: `(0.191,0) .. (1.0, 0.618) .. (0.0, 0.618)`) define a polygon inscribed in [0,1]². The triangle fan from center to these vertices generates UV coordinates that stay inside the polygon. But barycentric subdivision (12B) can generate UV coordinates that interpolate towards the *center UV* (average of edge UVs), which is correct — the problem is that the *triangle edges* between adjacent tiles' triangle fans cover the *gap* between the two tile polygons, and that gap maps to the black border region.

**Fix:** Clamp all UV coordinates to stay strictly within the polygon interior, with a configurable inset.

- [x] **13C.1 — UV polygon inset** — `compute_uv_polygon_inset()` shrinks a tile's UV polygon toward its centroid by a configurable number of atlas pixels. Helper functions: `clamp_uv_to_polygon()`, `_point_in_convex_polygon()` (winding test), `_nearest_point_on_segment()`, `_nearest_point_on_polygon_edge()`.
- [x] **13C.2 — Vertex-level UV clamping** — `subdivide_tile_mesh()` accepts optional `uv_clamp_polygon` parameter; after barycentric UV interpolation each UV is clamped. `build_batched_globe_mesh()` accepts `uv_inset_px` + `atlas_size` and computes the inset polygon per tile automatically.
- [x] **13C.3 — Tests** — 25 new tests: point-in-polygon containment, nearest-point-on-segment, polygon-edge projection, inset shrinkage/centroid/scaling, clamp interior-unchanged + exterior-clamped, subdivision with/without clamping (topology, positions, backward compat), batched mesh with inset (output shapes, topology, positions, validation, zero-inset identity). 914 total tests passing.

### 13D — Cross-tile colour harmonisation ✅

**Problem:** Adjacent tiles can have dramatically different biomes/colours. The ocean-to-grassland transition in the screenshot is a single-pixel step.

**Fix:** Smooth colour transitions at tile boundaries.

- [x] **13D.1 — Boundary colour matching** — `compute_neighbour_average_colours()` computes the mean colour of each tile's neighbours from the adjacency graph (`GoldbergTile.neighbor_indices`). `harmonise_tile_colours()` blends each tile toward its neighbour average by a configurable `strength` (0–1), producing a smoothed colour map.
- [x] **13D.2 — Biome transition blending** — `blend_biome_configs(a, b, t)` linearly interpolates every numeric field of two `BiomeConfig` instances (vegetation_density, rock_exposure, snow_line, water_level, moisture, hillshade_strength, azimuth, altitude). Weight is clamped to [0,1].
- [x] **13D.3 — Per-vertex edge colour gradient** — `subdivide_tile_mesh()` accepts optional `edge_color` parameter; vertex colours blend from `color` at tile centre (b0=1) to `edge_color` at tile boundary (b0=0) using the barycentric centre weight. `build_batched_globe_mesh()` accepts `edge_blend` float (0–1); when >0 it computes per-tile edge colours from `compute_neighbour_average_colours()` and passes them through automatically.
- [x] **13D.4 — Tests** — 26 new tests: BiomeConfig blending (zero/half/full/clamp/base_ramp), neighbour average colours (single/pair/triangle/missing), harmonise colours (zero/full/half strength, immutability, distance reduction), edge colour in subdivision (uniform without, centre keeps colour, boundary blends, gradient variation, same-as-centre uniform, topology preserved, positions/UVs preserved), batched mesh edge blend (zero identity, output valid, topology unchanged, colours changed, no-colour-map no-effect). 940 total tests passing.

### 13E — Normal-mapped lighting ✅

**Problem:** Current lighting is flat (just `dot(N,L)` with the sphere normal). The globe looks plastic. Terrain detail (hills, valleys) is invisible in the lighting.

**Fix:** Use the Phase 11E normal maps in the shader.

- [x] **13E.1 — Normal map atlas** — `build_normal_map_atlas()` builds a second atlas with per-sub-face normals encoded as RGB using `encode_normal_to_rgb()` / `decode_rgb_to_normal()`. Each tile slot rendered via `_render_normal_tile()` with gutter fill via `_fill_normal_gutter()`. Returns `(PIL.Image, uv_layout)`.
- [x] **13E.2 — Tangent-space normals** — `subdivide_tile_mesh()` accepts optional `tangent` and `bitangent` params. When provided, vertex format expands from 8 to 14 floats: pos(3)+col(3)+uv(2)+T(3)+B(3). Tangent/bitangent are Gram-Schmidt re-orthogonalised against the sphere normal at each vertex. `build_batched_globe_mesh()` gains `normal_mapped=True` flag that passes each GoldbergTile's `.tangent` / `.bitangent` through.
- [x] **13E.3 — PBR-lite shader** — New `_PBR_VERTEX_SHADER` and `_PBR_FRAGMENT_SHADER` (GLSL 330 core) with:
  - Diffuse: warm key light (`KEY_COLOR`) + cool fill light (`FILL_COLOR`) from separate directions
  - Specular: Blinn-Phong with roughness auto-derived from water heuristic (blue channel)
  - Ambient: hemisphere ambient (sky colour above, ground colour below)
  - Fresnel: Schlick rim lighting at glancing angles
  - Normal mapping: samples `u_normal_map` atlas, transforms via TBN matrix
  - Tone mapping: Reinhard to prevent over-bright
  - Legacy v2 shaders preserved as fallback. `get_pbr_shader_sources()` and `get_v2_shader_sources()` convenience accessors.
- [x] **13E.4 — Tests** — 34 new tests across 5 classes: encode/decode round-trip (7 tests), subdivide with tangent (6 tests), batched mesh normal_mapped (6 tests), normal map atlas (5 tests), PBR shader source validation (10 tests). 974 total tests passing.

### 13F — Adaptive mesh resolution ✅

**Problem:** Uniform subdivision (s=3 → 4,860 tris) is wasteful. Tiles near the camera need more detail; tiles on the far side of the globe need almost none.

**Fix:** View-dependent level of detail.

- [x] **13F.1 — Multi-resolution mesh** — `select_lod_level()` picks from `LOD_LEVELS = (1, 2, 3, 5)` based on `estimate_tile_screen_fraction()` (angular size vs FOV). `LOD_THRESHOLDS` defines screen-fraction breakpoints per level. `build_lod_batched_globe_mesh()` produces per-tile adaptive subdivision with a `tile_lod_map` return.
- [x] **13F.2 — Stitching at LOD boundaries** — `stitch_lod_boundary()` finds vertices on a shared edge and snaps higher-LOD boundary vertices to the nearest lower-LOD position, preventing T-junction cracks. Operates in-place on the vertex array, preserving colour/UV columns.
- [x] **13F.3 — GPU frustum culling** — `is_tile_backfacing()` uses `dot(tile_normal, view_dir) < BACKFACE_THRESHOLD` (default −0.1, slightly negative to avoid limb popping). `build_lod_batched_globe_mesh()` skips back-facing tiles entirely, halving triangle count.
- [x] **13F.4 — Tests** — 43 new tests across 5 classes: `TestSelectLodLevel` (10), `TestEstimateTileScreenFraction` (7), `TestIsTileBackfacing` (7), `TestStitchLodBoundary` (6), `TestBuildLodBatchedGlobeMesh` (7 + fixture), `TestLodConstants` (6). 1101 total tests, 0 failures.

### 13G — Atmosphere & post-processing ✅

**Problem:** The globe floats in black space with no atmospheric context.

**Fix:** Add atmospheric scattering and post-processing effects.

- [x] **13G.1 — Atmosphere shell** — `build_atmosphere_shell()` builds a UV sphere at `radius * ATMOSPHERE_SCALE` (1.025×) with per-vertex RGBA (7-float stride). Fresnel-based alpha: transparent at front, opaque at limb via `falloff` exponent. `_ATMO_VERTEX_SHADER` / `_ATMO_FRAGMENT_SHADER` use view-dependent `pow(1-dot(N,V), ATMO_FALLOFF)` for edge glow.
- [x] **13G.2 — Glow/bloom** — 3-pass bloom pipeline: `_BLOOM_EXTRACT_SHADER` (Rec.709 luminance threshold), `_BLOOM_BLUR_SHADER` (separable 5-tap Gaussian, `u_direction` for H/V pass), `_BLOOM_COMPOSITE_SHADER` (additive blend + Reinhard tone mapping). `compute_bloom_threshold()` gives CPU-side luminance check. Constants: `BLOOM_THRESHOLD=0.8`, `BLOOM_INTENSITY=0.3`.
- [x] **13G.3 — Background gradient** — `build_background_quad()` returns clip-space fullscreen quad (4×4 float32). `_BG_VERTEX_SHADER` / `_BG_FRAGMENT_SHADER` compute radial gradient with `smoothstep` from `BG_CENTER_COLOR` (dark blue) to `BG_EDGE_COLOR` (black).
- [x] **13G.4 — Tests** — 41 new tests across 7 classes: `TestBuildAtmosphereShell` (10), `TestBuildBackgroundQuad` (4), `TestComputeBloomThreshold` (8), `TestAtmosphereShaderSources` (6), `TestBackgroundShaderSources` (3), `TestBloomShaderSources` (6), `TestAtmosphereConstants` (4). 1058 total tests, 0 failures.

### 13H — Water rendering ✅

**Problem:** Ocean tiles are flat blue with no visual distinction from land.

**Fix:** Differentiated water rendering.

- [x] **13H.1 — Water detection** — `classify_water_tiles()` identifies tiles by blue-channel dominance vs `water_level` threshold (default 0.12, matching `BiomeConfig`). `compute_water_depth()` returns normalised [0,1] depth proxy. Per-vertex `water_flag` float added to vertex stride (Optional[float]: None=no column, 0.0=land, 1.0=water). Stride: 8→9 (basic) or 14→15 (normal-mapped).
- [x] **13H.2 — Water shader** — PBR fragment shader upgraded:
  - Water uses low roughness (0.15) for shiny reflective surface
  - Depth-based colour: shallow turquoise → deep navy via `WATER_SHALLOW`/`WATER_DEEP` constants
  - Animated wave normal perturbation via `u_time` uniform (sin/cos offset, `WAVE_SPEED`/`WAVE_SCALE`/`WAVE_AMPLITUDE`)
  - Per-vertex water flag (`v_water`) falls back to blue-channel heuristic for backward compat
- [x] **13H.3 — Coastline emphasis** — `dFdx`/`dFdy` screen-space derivatives of `v_water` detect water-land boundary; `coast_factor` blends `COAST_COLOR` (bright foam) at transitions.
- [x] **13H.4 — Tests** — 43 new tests across 5 classes: `TestClassifyWaterTiles` (10), `TestComputeWaterDepth` (5), `TestSubdivideWithWaterFlag` (9), `TestBatchedMeshWithWater` (9), `TestPBRShaderWaterFeatures` (10). 1017 total tests, 0 failures.

---

### Summary — Phase 13 Implementation Priority

| Step | Focus | Impact | Effort | Priority |
|------|-------|--------|--------|----------|
| **13A** | Full-coverage textures | **Critical** — eliminates 90% of visible seams | Medium | ✅ Done |
| **13B** | Atlas gutters | **High** — prevents sampling across tile boundaries | Low | ✅ Done |
| **13C** | UV inset clamping | **High** — backstop for any remaining UV bleed | Low | ✅ Done |
| **13D** | Colour harmonisation | **Medium** — softens biome transitions | Medium | ✅ Done |
| **13E** | Normal-mapped lighting | **Medium** — adds terrain depth & realism | Medium | ✅ Done |
| **13F** | Adaptive LOD | **Low** — performance (visible only at high freq) | High | ✅ Done |
| **13G** | Atmosphere | **Low** — aesthetic polish | Low | ✅ Done |
| **13H** | Water rendering | **Low** — aesthetic polish | Medium | ✅ Done |

**Recommended implementation order:** 13A → 13B → 13C → 13D → 13E → 13H → 13G → 13F

**13A + 13B together should eliminate the black seams entirely.** 13C is a safety net. 13D makes the globe look cohesive rather than tiled. 13E adds physical realism. 13F-H are polish.

### Design notes — Why the flood-fill approach (12A) failed

The flood-fill dilation with 8 iterations only extends the coloured region by ~8 pixels in each direction. For a 256×256 tile slot where the hex polygon has corners 50-80 pixels from the slot edges, this covers less than 20% of the black region. Increasing iterations to 50+ would work but is slow and produces smeared colours.

The proper fix (13A) changes the renderer itself: by setting `facecolor` to the tile's average colour instead of black, the *entire* image becomes terrain-coloured. The hex polygon then adds detail *on top* of this base colour. No flood-fill needed. Combined with atlas gutters (13B), bilinear sampling across slot boundaries sees terrain colour from both sides.

### Design notes — Shared vertices between adjacent Goldberg tiles

Adjacent GoldbergTile instances share exactly 2 vertices (verified: inter-vertex distance < 1e-6). This means the *3D mesh* is watertight — there are no geometric gaps. The seams are purely a *texture sampling* problem. The 3D positions are correct; only the UV mapping needs to be fixed.

---

## Phase 14 — Biome Feature Rendering ✅

**Goal:** Replace the current flat elevation-ramp colouring with **recognisable visual features** rendered per-biome onto tile textures. Starting with **forests**: from above, a forest should look like a dense canopy of overlapping tree crowns with shadows, gaps, undergrowth, colour variation, and natural edges — not just green polygons. The system must be cross-tile continuous (a forest that spans multiple Goldberg tiles should look like one forest, not a patchwork) and extensible to other biomes (desert dunes, grassland, tundra, wetlands, etc.).

### Problem analysis

The current rendering pipeline colours each sub-face polygon with a single RGB value derived from `detail_elevation_to_colour()` — a colour ramp lookup plus noise-based vegetation/rock/snow overlays. The result is abstract satellite-style terrain: smooth colour gradients that suggest vegetation or rock but contain no recognisable features. Every biome looks like a colour wash.

Real satellite imagery of forests shows:
- **Tree canopy crowns** — irregular circles/blobs of varying sizes (3-15m diameter, ~1-6 px at typical tile resolution), packed tightly with small dark gaps between them
- **Shadow** — each canopy casts a short shadow opposite the sun direction, creating depth
- **Colour variation** — species mix, seasonal state, health; multiple greens, yellow-greens, dark patches
- **Undergrowth/ground** — darker browns/greens visible in canopy gaps
- **Density gradients** — dense core, thinning edges at biome boundaries, clearings
- **Edge structure** — forests don't end in straight lines; they have fractal, irregular borders

### Architecture

The approach has four layers:

**Layer 1 — Feature placement** (`biome_scatter.py`)
Scatter feature instances (tree positions, sizes, colours) across tile textures using Poisson disk sampling for natural spacing. Positions are derived from 3D globe coordinates so placement is continuous across tile boundaries. Density is modulated by a noise field so forest edges are organic.

**Layer 2 — Feature rendering** (`biome_render.py`)
Stamp visual elements onto PIL images — canopy circles with soft edges, cast shadows, undergrowth fill. Each biome type provides a renderer that knows how to draw its features. Operates at the pixel level on the existing 256×256 tile textures.

**Layer 3 — Biome definitions** (`biome_features.py`)
Dataclass-based feature configuration per biome type. `ForestFeatureConfig` defines canopy radius range, colour palette, density, shadow params, etc. Extensible: `DesertFeatureConfig`, `GrasslandFeatureConfig`, etc. in future phases.

**Layer 4 — Pipeline integration** (`biome_pipeline.py`)
Hooks into the existing `build_detail_atlas()` / `render_detail_texture_fast()` pipeline. For tiles assigned a feature-rich biome, the feature renderer replaces (or overlays on top of) the flat colour ramp. Cross-tile density is computed once from the globe-wide 3D noise field and passed to each tile's renderer.

### Key design decisions

- **Pixel-level rendering, not per-sub-face.** The current system colours each sub-face polygon (61 faces per hex tile → ~4 px average at 256×256). Feature rendering operates at the pixel level on the PIL image, giving much higher visual resolution. Individual tree canopies are 2-8 px in diameter.

- **3D-coherent placement for cross-tile continuity.** Tree positions are generated by sampling a global Poisson disk field (seeded by 3D sphere coordinates, not per-tile local coords). A tree near a tile boundary appears consistently in both tiles' textures. This eliminates tile-edge discontinuities.

- **Density-from-noise for organic boundaries.** A globe-wide density field (using `fbm_3d()`) controls how densely trees are packed. Where density falls below a threshold, the forest thins out naturally. This is how forest edges and clearings form — not by hard biome-boundary lines.

- **Compositing model.** Features are drawn on top of the existing elevation-based ground colour. The ground texture (current system) provides the "floor" visible through canopy gaps; the feature renderer adds the canopy layer. This means we keep all existing terrain colouring and add features as an overlay.

---

### 14A — Feature Scattering (`biome_scatter.py`) ✅

A general-purpose system for placing feature instances (trees, rocks, bushes, etc.) across a tile's texture area with natural spacing and cross-tile coherence.

- [x] **14A.1 — `PoissonDiskSampler` class** — 2D Poisson disk sampling on a bounded rectangle. Given a minimum distance `r`, produces a set of points where no two points are closer than `r`. Uses Bridson's algorithm (O(n)) with configurable `k` candidate attempts per active point (default 30).
  - Input: `(width, height, min_distance, seed)`
  - Output: `List[Tuple[float, float]]` — point positions in pixel coordinates
  - Supports variable density via a density function `density_fn(x, y) → float` that scales the minimum distance spatially: `local_r = r / max(density, 0.01)`. Dense regions pack tighter; sparse regions spread out.

- [x] **14A.2 — `scatter_features_on_tile(tile_3d_center, tile_transform, tile_size, config)` → `List[FeatureInstance]`** — place feature instances for one tile:
  1. Map pixel coordinates to 3D globe positions via the tile's tangent-plane transform
  2. Evaluate the global density field at each candidate position (`fbm_3d()`)
  3. Run Poisson disk sampling with spatially-varying density
  4. For each accepted point, generate feature attributes: size (from distribution), colour (from palette + noise), rotation (random), species (weighted random from palette)
  5. Return a list of `FeatureInstance` dataclasses

- [x] **14A.3 — `FeatureInstance` dataclass** — a single placed feature:
  - `position` — `(px_x, px_y)` in tile-local pixel coordinates
  - `radius` — canopy radius in pixels
  - `color` — `(r, g, b)` canopy top colour
  - `shadow_color` — `(r, g, b)` for drop shadow
  - `species_id` — int, indexes into a colour/shape palette
  - `depth` — drawing order (back-to-front for proper overlap)

- [x] **14A.4 — `compute_density_field(globe_grid, face_ids, *, seed)` → `Dict[str, float]`** — for each globe tile, compute a density value (0–1) from the global 3D noise field. Tiles in the forest's core have density ~0.8–1.0; tiles at the edge thin down to 0.1–0.3; non-forest tiles get 0.0. Uses `fbm_3d` at the tile's 3D centre with configurable frequency/octaves.

- [x] **14A.5 — Cross-tile boundary overlap** — for tiles at forest edges, scatter features in a margin zone that extends slightly beyond the tile boundary (using the neighbour tiles' texture space). This ensures canopies that straddle a tile boundary render correctly in both tiles. The overflow region is clipped to the tile's texture bounds, producing a half-canopy at the edge that matches the other half in the neighbour's texture.

- [x] **14A.6 — Tests:**
  - Poisson disk: no two points closer than `min_distance`
  - Poisson disk: covers the area adequately (point count within expected bounds)
  - Variable density: dense regions have more points than sparse regions
  - Feature scatter: all instances within tile bounds (plus margin)
  - Cross-tile: features near boundary appear in both adjacent tiles' scatter lists
  - Density field: values in [0, 1], higher at forest centres, zero outside forests
  - Determinism: same inputs → identical scatter output

### 14B — Forest Feature Rendering (`biome_render.py`) ✅

Pixel-level rendering of forest features onto PIL tile textures. Produces the actual "satellite forest" look.

- [x] **14B.1 — `render_canopy(image, instance, config)` → None** — draw a single tree canopy onto a PIL image (in-place):
  1. **Canopy circle** — filled ellipse (slightly irregular via noise-perturbed radius) in the tree's top-colour. Soft edges via alpha blending with a radial falloff.
  2. **Internal texture** — subtle noise pattern within the canopy to break up uniformity (leaf clumps, small gaps). 2-3 shades of green per species.
  3. **Highlight** — small bright spot offset toward the sun direction (specular from waxy leaves).
  4. **Shadow** — a darker, slightly offset ellipse drawn *before* the canopy (painter's algorithm). Offset direction matches sun azimuth.

- [x] **14B.2 — `render_forest_tile(ground_image, instances, config)` → `PIL.Image`** — render all tree features for one tile:
  1. Sort instances back-to-front (by latitude / y-position for parallax-correct overlap)
  2. Draw undergrowth fill in canopy-gap regions (dark green-brown, dappled noise)
  3. Draw shadows for all instances
  4. Draw canopies for all instances
  5. Return the composited image

- [x] **14B.3 — `render_undergrowth(image, density, config)` → None** — fill ground areas between canopies with undergrowth texture:
  - Dark green/brown noise field at high frequency
  - Density-modulated: dense forest has minimal visible ground; sparse areas show more ground
  - Small shrub/bush dots scattered in gaps (much smaller radius than tree canopies)

- [x] **14B.4 — `ForestFeatureConfig` dataclass** — all tuneable parameters for forest rendering:
  - `canopy_radius_range` — `(min_px, max_px)` e.g. `(3, 8)` at 256px tile size
  - `canopy_colors` — list of `(r, g, b)` base colours for different tree species
  - `color_noise_amplitude` — how much each tree's colour varies from its species base
  - `density_scale` — global density multiplier (0–1)
  - `shadow_offset` — `(dx, dy)` in pixels, derived from sun direction
  - `shadow_opacity` — 0–1
  - `highlight_strength` — 0–1
  - `undergrowth_color` — `(r, g, b)` base colour for ground between trees
  - `edge_thinning` — how aggressively density drops at forest edges (0–1)
  - Preset configs: `TEMPERATE_FOREST`, `TROPICAL_FOREST`, `BOREAL_FOREST`, `SPARSE_WOODLAND`

- [x] **14B.5 — Tests:**
  - Canopy modifies pixels within the expected radius
  - Shadow is offset in the correct direction
  - Forest tile has significantly different pixel statistics than flat-colour tile
  - Dense config produces more green pixels than sparse config
  - Undergrowth is visible in gaps between canopies
  - Species colour variation: not all trees are the same shade
  - Empty instance list → returns ground image unchanged

### 14C — Cross-Tile Feature Continuity (`biome_continuity.py`) ✅

Ensure that biome features flow seamlessly across Goldberg tile boundaries. A forest that spans 5 tiles should look like one continuous forest canopy, not 5 separate patches.

- [x] **14C.1 — `build_biome_density_map(globe_grid, store, *, biome_type, seed)` → `Dict[str, float]`** — globe-wide density map for a biome type. For "forest": high density (0.7–1.0) at tiles classified as forest by terrain patches; moderate (0.3–0.6) at adjacent transition tiles; zero elsewhere. Computed from 3D noise field modulated by the patch assignment.

- [x] **14C.2 — `get_tile_margin_features(tile_id, scatter, neighbour_scatters)` → `List[FeatureInstance]`** — for each neighbouring tile, collect feature instances that fall within the margin zone (within `max_canopy_radius` of the shared boundary). These are features that partially overlap into the current tile and need to be drawn (clipped) for seamless edges.

- [x] **14C.3 — `compute_biome_transition_mask(tile_id, density_map, neighbours)` → `np.ndarray`** — a 2D float mask (tile_size × tile_size) that represents the biome transition zone. 1.0 = full forest, 0.0 = no forest, gradient at edges. Used to control both feature density and opacity of the forest overlay. Derived from interpolating the tile's and neighbours' density values.

- [x] **14C.4 — `stitch_feature_textures(tile_a, tile_b, shared_edge)` → None** — post-process: compare the pixel strips along the shared boundary of two adjacent tiles. If there's a visible discontinuity (colour difference exceeds threshold), blend the boundary strips using a narrow feather gradient (2-4 pixels). This is a safety net — primary continuity comes from shared scatter positions (14A.5).

- [x] **14C.5 — Tests:**
  - Density map covers all tiles, values in [0, 1]
  - Forest-interior tiles have density > 0.5
  - Non-forest tiles have density ≈ 0.0
  - Margin features: at least some features collected from neighbours
  - Transition mask gradient: no hard edges (max pixel-to-pixel change < threshold)
  - Boundary stitch: max colour difference across boundary < threshold

### 14D — Biome Feature Pipeline Integration (`biome_pipeline.py`) ✅

Wire the feature rendering system into the existing texture atlas pipeline so that `build_detail_atlas()` automatically uses feature rendering for applicable biomes.

- [x] **14D.1 — `BiomeRenderer` protocol** — interface that all biome feature renderers implement:
  ```python
  class BiomeRenderer(Protocol):
      def render(self, ground_image: Image, tile_id: str,
                 density: float, config: Any) -> Image: ...
  ```
  The atlas builder calls `renderer.render(ground_img, ...)` for each tile. Returns the composited image with features on top of the ground texture.

- [x] **14D.2 — `ForestRenderer` implementation** — implements `BiomeRenderer`:
  1. Compute scatter for this tile (using cached density map + 3D positions)
  2. Collect margin features from neighbours
  3. Render undergrowth
  4. Render all canopies (own + margin)
  5. Apply transition mask at edges
  6. Return composited image

- [x] **14D.3 — `build_feature_atlas(collection, globe_grid, store, biome_renderers, *, tile_size)` → `(Path, Dict)`** — extended atlas builder:
  1. Build globe-wide density maps for each biome type
  2. For each tile, determine which biome renderer(s) apply
  3. Render ground texture (existing `render_detail_texture_fast()`)
  4. Apply biome renderer overlay
  5. Assemble into atlas with gutters

- [x] **14D.4 — Demo script `scripts/demo_forest_globe.py`** — end-to-end:
  ```
  python scripts/demo_forest_globe.py -f 3 --detail-rings 4 --view
  ```
  Generates a globe where all (or most) tiles are forested, demonstrating seamless cross-tile forest canopy. Options for density, forest type presets (temperate, tropical, boreal).

- [x] **14D.5 — Tests:**
  - ForestRenderer produces different output than ground-only rendering
  - Feature atlas has same dimensions/UV layout as standard atlas
  - Dense forest tiles have lower average brightness (shadows)
  - Renderer protocol is satisfied by ForestRenderer
  - Demo script runs without error
  - Atlas with features validates same as standard atlas (gutter, UV, dimensions)

### 14E — Forest Globe Demo & Tuning ✅

Full integration demo: a globe entirely (or predominantly) covered in forest, viewed in the Phase 13 v3 viewer with PBR lighting.

- [x] **14E.1 — "All forest" terrain preset** — a `TERRAIN_PRESETS["forest_world"]` distribution that assigns forest to 80-90% of land tiles (oceans remain). Also `"deep_forest"` (100% forest, no ocean).

- [x] **14E.2 — Viewer integration** — update `view_globe_v3.py` to use `build_feature_atlas()` when biome features are enabled. Add `--features` / `--no-features` flag. Default: features enabled.

- [x] **14E.3 — Visual tuning** — iterate on:
  - Canopy size/density at various zoom levels
  - Colour palette for convincing forest-from-above look
  - Shadow length/opacity vs sun direction
  - Undergrowth visibility in gaps
  - Edge thinning at biome boundaries
  - Normal map interaction with canopy bumps

- [x] **14E.4 — Performance** — forest rendering adds a pixel-level pass per tile. Profile and ensure it stays under 2× the current atlas build time. Poisson disk sampling should be < 10ms per tile at 256×256.

- [x] **14E.5 — Tests:**
  - Forest-world globe renders without error
  - Viewer launches with feature atlas
  - All-forest globe: every land tile has forest features (non-uniform pixel colours)
  - Performance: atlas build time < 2× baseline for freq=3, detail_rings=4

### Summary — Phase 14 Implementation Order

| Step | Module | Depends on | Delivers | Complexity |
|------|--------|-----------|----------|------------|
| 14A | `biome_scatter.py` | noise.py (3D), globe.py | Poisson disk scattering, feature placement | Medium |
| 14B | `biome_render.py` | 14A + PIL | Forest canopy/shadow/undergrowth rendering | Medium |
| 14C | `biome_continuity.py` | 14A + 14B + globe topology | Cross-tile continuity, density maps, edge blending | High |
| 14D | `biome_pipeline.py` | 14A–C + texture_pipeline | Atlas integration, BiomeRenderer protocol | Medium |
| 14E | demo + tuning | 14A–D | Full forest globe demo, visual tuning | Low |

### Design notes — Why Poisson disk sampling?

Random uniform scattering produces clusters and gaps that look unnatural. Grid-based placement looks artificial (visible rows). Poisson disk sampling guarantees a minimum distance between points while maximising density — exactly how real tree canopies pack in a forest: each tree needs a minimum crown spacing, but they fill available space as densely as possible. Bridson's algorithm runs in O(n) time, making it practical at 256×256 resolution (~1000-3000 trees per tile).

### Design notes — Why pixel-level rendering instead of per-sub-face?

At detail_rings=4, each hex tile has 61 sub-faces, giving each sub-face ~4px diameter at 256×256 tile resolution. A tree canopy is 5-15px. Colouring entire sub-faces as "tree" vs "gap" would produce a blocky Minecraft-like appearance. Pixel-level PIL rendering gives ~65,000 addressable pixels per tile — enough for smooth canopy circles, soft shadows, and gradient edges.

### Design notes — Extensibility to other biomes

The `BiomeRenderer` protocol is deliberately generic. Phase 14 implements `ForestRenderer`. Future phases can add:
- **DesertRenderer** — sand dunes (sinusoidal ridges), rock outcrops, sparse scrub
- **GrasslandRenderer** — wind-aligned grass streaks, wildflower dots, cattle tracks
- **TundraRenderer** — lichen patches, polygonal frost cracks, sparse stunted trees
- **WetlandRenderer** — standing water pools, reed clusters, mud flats
- **UrbanRenderer** — building footprints, road grids, park patches

Each renderer implements the same `BiomeRenderer.render()` interface and slots into the existing atlas pipeline without architectural changes.

### Design notes — Interaction with existing PBR / normal map pipeline

Forest canopy features add visual detail at the texture level. The existing normal map pipeline (13E) computes per-sub-face normals from elevation gradients. For forest tiles, the normal map should be enhanced to include canopy-bump normals — each tree canopy creates a slight convex bump in the normal map, giving PBR lighting something to work with (specular highlights on canopy tops, shadow in gaps). This is a stretch goal for 14E.3.

---

## Phase 15 — Test Infrastructure Overhaul ✅

**Goal:** Restructure the test suite for faster execution, clearer progress output, removal of duplicate tests, and a single-command runner that groups tests by phase/speed and gives real-time terminal feedback.

### Results

| Metric | Before | After |
|--------|--------|-------|
| Total tests | 1,101 | 1,004 |
| Test files | 36 | 31 |
| `test_globe.py` | 3,025 lines / 217 tests / 82.7s | 1,367 lines / 110 tests / 17.8s |
| Full suite | ~5 min (subprocess) | ~2m 6s (single process) |
| Fast tier | N/A | 342 tests in ~12s |
| Progress visibility | Wall of dots | Grouped output with per-phase timing |

### 15A — Remove Duplicate Tests from `test_globe.py` ✅

Removed 7 duplicate test classes (~97 tests, ~1,658 lines) from `test_globe.py`:

- `TestDetailGrid` (16 tests) → covered by `test_tile_detail.py` + `test_detail_terrain.py` + `test_texture_pipeline.py`
- `TestTileDetail` (24 tests) → covered by `test_tile_detail.py`
- `TestDetailTerrain` (19 tests) → covered by `test_detail_terrain.py`
- `TestDetailRender` (12 tests) → covered by `test_detail_render.py`
- `TestTexturePipeline` (11 tests) → covered by `test_texture_pipeline.py`
- `TestTexturedRenderer` (10 tests) → moved to `test_globe_renderer_v2.py`
- `TestDetailPerf` (15 tests) → covered by `test_detail_perf.py`

Kept: 17 unique classes (110 tests) including `TestVertexZ`, `TestBuildGlobeGrid`, `TestGlobeExport`, `TestGlobeRenderer`, `TestDetailIntegration`.

### 15B — Pytest Markers & Groups ✅

Registered markers in `pyproject.toml`:
- `fast` — tests that run in < 3s per file (Phases 1–4, 5–7)
- `medium` — tests that run in 3–30s per file (Goldberg, detail, rendering)
- `slow` — tests that run in > 30s per file (globe integration)
- `needs_models` — requires the models library

Auto-applied via `pytest_collection_modifyitems` in `conftest.py`.

Usage:
```bash
pytest -m fast            # 342 tests in ~12s
pytest -m medium          # 552 tests
pytest -m slow            # 110 tests
pytest -m "not slow"      # 894 tests (skip expensive globe build)
```

### 15C — Test Runner Script (`scripts/run_tests.py`) ✅

Developer-facing script with grouped progress, timing, and colour output:
```bash
python scripts/run_tests.py              # full suite
python scripts/run_tests.py --fast       # fast-tier only
python scripts/run_tests.py --phase 13   # only Phase 13 tests
python scripts/run_tests.py --summary    # summary table only
```

### 15D — Consolidate Single-Test Files ✅

Merged 6 single-test files into `tests/test_core_topology.py`:
- `test_adjacency.py` → `TestFaceAdjacency`
- `test_build_hex.py` → `TestBuildHex`
- `test_composite.py` → `TestComposite`
- `test_hex_shape.py` → `TestHexShape`
- `test_rings.py` → `TestRingFaces`
- `test_serialization.py` → `TestSerialization`

Updated `conftest.py` tier map and `run_tests.py` group list.

### 15E — Documentation ✅

Testing guide (see below).

#### Running Tests

```bash
# Full suite (1,004 tests)
pytest tests/

# Speed tiers
pytest -m fast            # 342 tests, ~12s — no globe/collection build
pytest -m medium          # 552 tests — Goldberg, detail, rendering
pytest -m "not slow"      # 894 tests — skip expensive globe integration

# Grouped runner with progress output
python scripts/run_tests.py              # all groups
python scripts/run_tests.py --fast       # fast tier only (~35s)
python scripts/run_tests.py --phase 13   # single phase
python scripts/run_tests.py --summary    # summary table only

# Single file
pytest tests/test_globe.py -v
```

#### Test File Structure (31 files)

| Group | Files | Tests | Typical Time |
|-------|-------|-------|-------------|
| Phase 1–4 Core | `test_core_topology`, `test_stitching`, `test_assembly`, `test_macro_edges`, `test_pentagon_centered`, `test_transforms`, `test_diagnostics`, `test_visualize` | 68 | ~21s |
| Phase 2 Goldberg | `test_goldberg` | 79 | ~18s |
| Phase 5–7 Terrain | `test_tile_data`, `test_regions`, `test_noise`, `test_heightmap`, `test_mountains`, `test_rivers`, `test_pipeline`, `test_terrain_render`, `test_determinism` | 274 | ~14s |
| Phase 8–9 Globe | `test_globe` | 110 | ~18s |
| Phase 10 Detail | `test_tile_detail`, `test_detail_render`, `test_detail_perf` | 60 | ~46s |
| Phase 11 Cohesive | `test_detail_terrain`, `test_detail_terrain_3d`, `test_terrain_patches`, `test_globe_terrain`, `test_region_stitch`, `test_render_enhanced`, `test_texture_pipeline` | 153 | ~125s |
| Phase 12–13 Rendering | `test_globe_renderer_v2`, `test_phase13_rendering` | 260 | ~25s |

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
- [x] **Test performance** — added `lru_cache`-based caching in `tests/conftest.py` for `build_globe_grid()` and monkeypatched `DetailGridCollection.build()` to cache the expensive grids dict. Each test gets a fresh mutable wrapper with empty `_stores`. Full suite (974 tests) runs in ~5 minutes instead of 50+.
- [ ] **Design patterns** — as terrain algorithms grow, consider:
  - *Strategy pattern* for swappable terrain generators per biome
  - *Pipeline/chain pattern* for algorithm sequencing
  - *Observer pattern* if tile data changes need to trigger recalculation
  - *Repository pattern* for tile data persistence
- [x] **Performance** — profile for large ring counts (rings ≥ 5). The optimiser and stitching are the bottlenecks. Test-time caching (see above) mitigates the optimiser cost. Consider lazy evaluation for production paths.

---

## Dependency Roadmap

| Phase | New dependencies |
|-------|-----------------|
| 5 (Tile Data) | None |
| 6 (Partitioning) | `noise` or `opensimplex` (for noise-based boundaries) |
| 7 (Terrain Gen) | `noise` / `opensimplex`, possibly `Pillow` for texture generation |
| 8 (Globe) | `models>=0.1.1` (optional, under `globe` extra) |
| 9 (Export) | `Pillow` for PNG texture output |
| 10 (Sub-Tile Detail) | None (uses Phase 8/9 infra) |
| 11 (Cohesive Terrain) | None (uses Phase 10 + existing noise/composite infra) |
| 12 (Rendering Quality) | None (uses Phase 10/11 infra + pyglet) |
| 13 (Cohesive Rendering) | None (extends Phase 12 renderer + texture pipeline) |
