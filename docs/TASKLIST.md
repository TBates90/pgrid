# PolyGrid Task List

Active and upcoming phases. For full detail on completed phases, see [`ARCHIVED_TASKLIST.md`](ARCHIVED_TASKLIST.md).

---

## Legend

- ✅ Done
- 🔲 To do
- 🔶 In progress / partially done

---

## Completed Phases — Summary

| Phase | Title | Key Deliverables | Tests |
|-------|-------|-----------------|-------|
| **1** | Core Topology | `Vertex`, `Edge`, `Face`, `PolyGrid`, JSON serialisation, adjacency, ring detection, CLI | ✅ |
| **2** | Goldberg Topology | Combinatorial Goldberg via cone-triangulation + dualisation, Tutte embedding, optimisation, pentagon-centred grids | ✅ |
| **3** | Stitching & Assembly | `CompositeGrid`, `StitchSpec`, `AssemblyPlan`, `pent_hex_assembly(rings)` | ✅ |
| **4** | Transforms & Visualisation | Overlays, Voronoi dual, partition, multi-panel rendering | ✅ |
| **5** | Tile Data Layer | `TileSchema`, `TileDataStore`, neighbour/ring queries, bulk ops, JSON round-trip | ✅ |
| **6** | Terrain Partitioning | `RegionMap`, angular/flood-fill/Voronoi/noise partitioning, constraints | ✅ |
| **7** | Terrain Generation | `noise.py` (fbm, ridged, warp, terrace), `heightmap.py`, `mountains.py`, `rivers.py`, `pipeline.py` | ✅ |
| **8** | Globe Topology | `GlobeGrid` wrapping `models.GoldbergPolyhedron`, 3D noise, globe terrain/rivers/regions, 3D rendering | ✅ |
| **9** | Export & Integration | Globe JSON export, detail grids, texture atlas, UV mapping, OpenGL renderer | ✅ |
| **10** | Sub-Tile Detail | `tile_detail.py`, boundary-aware terrain, enhanced colour ramps, texture pipeline, textured 3D viewer, perf | ✅ |
| **11** | Cohesive Globe Terrain | 3D-coherent noise (`detail_terrain_3d.py`), terrain patches, region stitching, erosion, biome assignment, normal maps | ✅ |
| **12** | Rendering Quality | Texture flood-fill, sphere subdivision, batched mesh, interactive renderer v2 | ✅ |
| **13** | Cohesive Rendering | Full-coverage textures, atlas gutters, UV inset clamping, colour harmonisation, PBR lighting, LOD, atmosphere, water | ✅ |
| **14** | Biome Feature Rendering | `biome_scatter.py` (Poisson disk), `biome_render.py` (forest canopy/shadow), `biome_continuity.py`, `biome_pipeline.py`, forest presets | ✅ |
| **15** | Test Infrastructure | Duplicate removal, pytest markers (fast/medium/slow), `run_tests.py`, file consolidation — 1,004 tests across 31 files | ✅ |

**Total: ~1,085 tests across 35 files (1,004 from Phase 15 restructure + 81 from Phase 14 biome tests).**

---

## Ongoing — Code Quality & Refactoring

- [x] Gitignore, render.py → visualize.py merge, `__init__.py` exports, type hints, docstrings, legacy alias removal, notes.md move, experiments cleanup
- [x] Test caching (`lru_cache` globe/collection builds), monkeypatched `DetailGridCollection.build()`
- [ ] **CI pipeline** — set up GitHub Actions for `pytest` + linting on push
- [ ] **Design patterns** — Strategy / Pipeline / Observer / Repository as terrain algorithms grow

---

## Phase 16 — Cross-Biome Visual Polish ✅

**Goal:** Eliminate the two most visible remaining artefacts: (1) **seams between hex tiles** that make boundaries look disjointed, and (2) **hexagonal tile shapes** that are visible through the rendered features. These issues affect all biomes — forest, terrain, and future biomes alike — so fixes must be biome-agnostic and wired into the shared texture/rendering pipeline.

### Problem analysis

**Seam artefact:** Despite Phase 13's extensive seam mitigation (full-coverage textures, atlas gutters, UV inset clamping, colour harmonisation), tile boundaries remain visible because:
1. Each tile's terrain texture is rendered from its sub-face polygons, which form a hex/pent shape that doesn't fill the full square tile slot. The background fill is a flat average colour — it lacks the noise, hillshade, and feature detail that the polygon interior has.
2. Biome features (trees, canopy) are scattered per-tile. Even with 14C's margin features, the overlap zone is narrow. Canopy patterns abruptly change character at tile edges.
3. The ground texture noise (`detail_render.py`) uses per-sub-face colours. At tile edges, the outermost sub-faces are triangular slivers that produce thin, irregular colour strips — visually distinct from the hexagonal sub-faces in the tile interior.

**Hexy artefact:** The hexagonal tile shape is visible because:
1. The tile's UV polygon (hex/pent inscribed in [0,1]²) maps a hex-shaped region of the texture onto the 3D surface. Even with gutter fill, the hex outline is defined by the UV polygon boundary.
2. Sub-face geometry follows the hex grid's ring structure — concentric hexagonal rings of faces. This creates a subtle radial pattern visible in the texture.
3. Feature rendering (canopy scatter) operates on a square pixel grid but the useful area is hex-shaped. Features near hex corners get clipped or sparse.

### Approach: Expanded Rendering + Soft Blending

The core strategy is to **render each tile's texture larger than its UV footprint**, filling the full square tile slot with coherent terrain, then use a **soft radial mask** at tile edges so adjacent tiles' textures cross-fade. This makes tile boundaries invisible because both sides contribute colour across the boundary.

This approach avoids the "neighbour tile stitching" alternative (merging adjacent detail grids for rendering, which is expensive and complex). Instead, we extend the rendering coverage of each tile individually.

### 16A — Extended Tile Texture Rendering ✅

**Problem:** `render_detail_texture_enhanced()` only colours sub-face polygons. Pixels outside the hex footprint get a flat fill.

**Fix:** Extend the rendered area to fill the entire tile slot with coherent terrain colour.

- [x] **16A.1 — Full-slot terrain sampling** — new `render_detail_texture_fullslot()` in `tile_texture.py`. Hybrid approach: (1) PIL polygon rasterisation for sub-face interiors (fast), (2) KDTree-based IDW interpolation of pre-computed face colours for background pixels outside the hex footprint. Uses scipy `KDTree` with `workers=-1` for parallel queries. Every pixel in the tile slot gets a plausible terrain colour — no flat fill. Also added `build_detail_atlas_fullslot()` to `detail_perf.py` and `fullslot=True` parameter to `build_feature_atlas()`.

- [x] **16A.2 — Extrapolated noise field** — background pixels inherit noise-modulated colour through IDW interpolation of face colours that already include vegetation noise, rock exposure, snow, and hillshade from `detail_elevation_to_colour()`. The noise field extends smoothly beyond the hex edge because IDW blending of neighbouring face colours produces spatially coherent gradients.

- [x] **16A.3 — Tests:** 14 tests in `test_tile_texture.py`:
  - `build_face_lookup` — arrays match face count, 2D centroids, finite elevations, face IDs from grid
  - `interpolate_at_pixel` — returns 4 values, exact hit returns face values, far point still valid
  - `render_detail_texture_fullslot` — produces PNG, no flat-fill rows (max 90% threshold), corner pixels not all identical, pixel variance similar to interior, custom biome, deterministic output, different seeds differ

**Performance:** ~37ms per tile at 256px (vs ~10ms for flat-fill renderer). 92-tile atlas: ~3.4s vs ~0.9s. Acceptable for offline atlas generation.

### 16B — Soft Tile-Edge Blending Mask ✅

**Problem:** Even with full-slot rendering, adjacent tiles' textures will differ slightly at boundaries (different sub-face grids, different noise seeds from Phase 11B patches, different biome parameters). A hard UV-polygon cutoff makes this a visible edge.

**Fix:** Apply a soft radial alpha mask to each tile's texture so that edges cross-fade with neighbouring tiles' extended textures.

- [x] **16B.1 — Tile blending mask** — `compute_tile_blend_mask(detail_grid, tile_size, fade_width)` in `tile_texture.py`. Uses convex hull of vertex positions + signed distance field (vectorised ray-casting + segment distance). Returns `np.ndarray` (float32, [0,1]) shaped `(tile_size, tile_size)`. Coordinate transform matches the fullslot renderer's overscan.

- [x] **16B.2 — Atlas-level blending** — `apply_blend_mask_to_atlas()` in `tile_texture.py`. Multiplies each tile's atlas slot by its per-tile mask. Where mask < 1.0, tile colours fade toward black; the gutter edge-clamping from 13B provides fill behind the fade zone.

- [x] **16B.3 — UV overlap extension** — deferred to 16E integration, as it requires changes to the 3D renderer's UV mapping. The mask + gutter system already reduces boundary contrast significantly without UV extension.

- [x] **16B.4 — Tests:** 9 tests in `test_tile_texture.py`:
  - Mask shape and range correct, centre is 1.0, corners < 0.5
  - Hex edge midpoints higher than corners (polygon-following, not circular)
  - `fade_width=0` produces binary mask
  - `apply_blend_mask_to_atlas` preserves pixels at mask=1, zeros at mask=0

### 16C — Biome Feature Edge Overflow ✅

**Problem:** Forest canopy scatter (14A) places trees within the hex tile bounds with a small margin zone (14C). Trees near tile edges are clipped at the hex boundary. Adjacent tiles' canopies don't mesh.

**Fix:** Extend biome feature rendering to cover the full tile slot (not just the hex interior).

- [x] **16C.1 — Full-slot feature scattering** — `scatter_features_fullslot()` in `biome_scatter.py`. Expands sampling area by `overscan` fraction on each side (default 15%), scattering via Poisson disk on the enlarged rectangle. Feature positions are in tile-local coords (may be negative or exceed tile_size). 8 tests in `test_biome_scatter.py`: features outside tile bounds, more than standard, proportional to area, deterministic, neighbour density affects margin, zero density empty, density continuity at boundary.

- [x] **16C.2 — Neighbour-seeded margin features** — `scatter_features_fullslot()` accepts `neighbour_densities` and `neighbour_seeds` dicts. The spatial density function uses neighbour density values for features in the margin zone (outside `[0, tile_size]`), ensuring cross-boundary feature coherence.

- [x] **16C.3 — Feature-level cross-fade** — `render_forest_on_ground_fullslot()` in `biome_render.py`. Accepts optional `blend_mask` (from 16B). Composites featured image with ground image: `output = ground * (1-mask) + featured * mask`, so features fade toward ground at tile edges. `ForestRenderer` in `biome_pipeline.py` gains `fullslot=True` mode. 5 tests in `test_biome_render.py`: produces image, with blend mask, mask fades edges, no-mask mode, deterministic.

- [x] **16C.4 — Tests:** 13 new tests total:
  - `TestScatterFeaturesFullslot` (8): outside bounds, count comparison, area-proportional, deterministic, neighbour density, zero density, boundary continuity
  - `TestRenderForestOnGroundFullslot` (5): image output, blend mask, edge fade, no-mask, deterministic

### 16D — Hex Shape Softening ✅

**Problem:** The hex sub-face grid creates a visible hexagonal structure in the texture — concentric rings of faces with slightly different sizes/colours.

**Fix:** Break up the hex grid pattern at the rendering level.

- [x] **16D.1 — Sub-face edge dissolution** — `jitter_polygon_vertices()` in `tile_texture.py`. Per-vertex deterministic jitter (±1.5 px default) seeded from vertex position hash + noise_seed. Integrated into `render_detail_texture_fullslot()` via `vertex_jitter` parameter. 4 tests: bounds check, zero jitter passthrough, determinism, seed sensitivity.

- [x] **16D.2 — Pixel-level noise overlay** — `apply_noise_overlay()` in `tile_texture.py`. FBM noise (3 octaves, frequency 0.05 px⁻¹, amplitude ±5%) applied as multiplicative brightness shift `pixel * (1 + noise)`. Applied after IDW background fill so all pixels get micro-variation. 5 tests: shape/dtype, pixel changes, amplitude bounds, determinism, seed sensitivity.

- [x] **16D.3 — Sub-face colour dithering** — `apply_colour_dithering()` in `tile_texture.py`. KDTree-based IDW blend of K nearest sub-face colours. Blend factor ramps from 0 at centroid to 0.5 at `blend_radius` distance. Applied before IDW background fill (only polygon-covered pixels). Sentinel pixels skipped. 4 tests: shape/dtype, sentinel preservation, centre-vs-edge change, boundary contrast reduction.

- [x] **16D.4 — Tests:** 17 new tests in `test_tile_texture.py`:
  - `TestJitterPolygonVertices` (4): within ±2px, zero passthrough, deterministic, seed-sensitive
  - `TestApplyNoiseOverlay` (5): shape, pixel changes, amplitude < 15%, deterministic, seed-sensitive
  - `TestApplyColourDithering` (4): shape, sentinel preserved, centre < edge change, boundary contrast reduced
  - `TestFullslotWith16D` (4): all-on, all-off, enhanced ≠ plain, deterministic with enhancements

### 16E — Pipeline Integration & Validation ✅

Wire all the 16A–D improvements into the production rendering pipeline.

- [x] **16E.1 — Pipeline flags** — added `soft_blend` and `blend_fade_width` parameters to `build_feature_atlas()` in `biome_pipeline.py`. When `soft_blend=True`: forces fullslot rendering (16A+16D), computes per-tile blend masks (16B), gathers neighbour densities/seeds for fullslot scatter (16C), passes blend masks to `ForestRenderer`, and applies `apply_blend_mask_to_atlas()` to the assembled atlas. Added `--soft-blend` CLI flag to both `view_globe_v3.py` and `demo_forest_globe.py`. Default: off (backward compatible).

- [x] **16E.2 — Script integration** — `view_globe_v3.py` branches on `--soft-blend` to call `build_feature_atlas()` with `soft_blend=True` instead of `build_detail_atlas()`. `demo_forest_globe.py` passes `soft_blend` through `_build_feature_atlas()` to `build_feature_atlas()` and creates `ForestRenderer(fullslot=soft_blend)`.

- [x] **16E.5 — Tests:** 8 new tests in `TestSoftBlendPipeline` (`test_biome_pipeline.py`):
  - `test_soft_blend_atlas_produces_file` — end-to-end produces valid PNG
  - `test_soft_blend_atlas_dimensions_unchanged` — same size as plain atlas
  - `test_soft_blend_with_forest_features` — works with forest overlays
  - `test_soft_blend_forces_fullslot` — no magenta sentinel pixels
  - `test_soft_blend_alters_tile_edges` — measurable edge difference
  - `test_soft_blend_uv_values_in_range` — UV coords valid [0,1]
  - `test_blend_fade_width_parameter` — different widths produce different results
  - `test_soft_blend_partial_density` — partial density map accepted

### Summary — Phase 16 Implementation Order

| Step | Focus | Problem Solved | Complexity |
|------|-------|---------------|------------|
| **16A** | Full-slot texture rendering | Flat-fill background outside hex polygons | Medium |
| **16B** | Soft edge blending mask | Hard tile boundaries | Medium |
| **16C** | Feature overflow + neighbour seeding | Canopy clipped at tile edges | Medium |
| **16D** | Hex shape softening | Visible hexagonal sub-face grid | Low |
| **16E** | Pipeline integration | End-to-end wiring + validation | Low |

### Design notes — Why not stitch neighbour detail grids for rendering?

The alternative approach — stitching adjacent tiles' detail grids into a combined grid (as `region_stitch.py` does for terrain generation in 11C) and rendering the combined grid's texture — would give perfect cross-tile continuity. However:

1. **Rendering cost:** Each tile would need to render its own sub-faces *plus* a ring of neighbour sub-faces (6 neighbours × ~20 boundary faces each = ~120 extra faces). For 92 tiles, that's ~11,000 extra face renders per atlas build.
2. **Complexity:** The stitched grid exists in projected 2D space (gnomonic). Rendering it back to the tile's texture space requires coordinate transforms, clipping, and careful UV mapping.
3. **Diminishing returns:** The extended-rendering + soft-blending approach (16A+16B) achieves 90% of the visual improvement at 20% of the implementation complexity. The remaining 10% (sub-pixel-perfect cross-tile continuity) is invisible at normal globe viewing distances.

If the soft-blending approach proves insufficient after implementation, stitched neighbour rendering can be added as a 16F extension.

---

## Phase 17 — Ocean Biome Rendering 🔲

**Goal:** Transform ocean tiles from flat blue fills into convincing bodies of water with depth gradients, surface texture, wave patterns, coastal transitions, and atmosphere interaction — all rendered into tile textures and enhanced by the shader pipeline. The ocean should look as rich as the forest biome, just for water.

### Problem analysis

Current ocean rendering (Phase 13H) operates entirely in the fragment shader:
- `classify_water_tiles()` detects ocean tiles by blue-channel dominance
- The PBR shader replaces water tile colours with a `WATER_SHALLOW → WATER_DEEP` gradient based on depth
- Animated wave normals perturb the surface (`sin`/`cos` via `u_time`)
- Coastline foam uses `dFdx`/`dFdy` of the water flag

This gives a functional but simplistic ocean: uniform blue tiles with a slight animated shimmer. It lacks:
- **Depth-based colour gradients** baked into tile textures (not just shader-time)
- **Surface texture** — subtle wave/ripple patterns at the texture level
- **Coastal features** — surf lines, shallow-water sand/reef visibility, tidal patterns
- **Deep ocean features** — darker blue-black, abyssal colour, occasional lighter patches (upwelling)
- **Ice/polar** — different water appearance at high latitudes

### Architecture: Texture-Level + Shader-Level

Like forests, ocean rendering needs two layers:
1. **Texture layer** — ocean tile textures are rendered with depth-based colours, surface patterns, coastal detail, and feature placement. These are baked into the atlas alongside terrain/forest tiles.
2. **Shader layer** — the existing 13H water shader is enhanced with better animations, specular reflections, and Fresnel effects that operate on top of the textured ocean tiles.

### Planning Session — Key Design Decisions

> **These decisions need careful evaluation before implementation begins. Phase 17 tasks are preliminary — they will be refined during a dedicated planning session once Phase 16 is complete.**

#### Decision 1: OceanRenderer vs. enhanced ground colouring

**Option A — `OceanRenderer` implementing `BiomeRenderer`:** Like `ForestRenderer`, a dedicated renderer class that takes the ground texture and overlays ocean features (waves, depth, caustics) via PIL rendering. Plugs into `build_feature_atlas()`.

**Option B — Enhanced colour ramp + shader:** Keep ocean rendering primarily shader-based (13H) but improve the colour ramp baked into tile textures by `detail_render.py`. Add depth gradients and coastal features to the colour ramp itself.

**Likely answer: Option A for texture features + enhanced Option B for shader.** The ocean needs both baked detail (texture) and animated effects (shader). The `BiomeRenderer` protocol was designed for exactly this — `OceanRenderer.render()` produces the ocean texture, then the shader adds animation.

#### Decision 2: Ocean depth model

Where does "depth" come from? Options:
- **Elevation-based:** tiles with elevation < `water_level` are ocean. Depth = `water_level - elevation`. Simple, already available.
- **Distance-from-coast:** compute distance (in tile hops) from nearest land tile. Deeper = further from land. Gives natural depth gradients but needs a BFS pass.
- **Hybrid:** base depth from elevation, modulated by distance-from-coast for realistic continental shelf gradients.

#### Decision 3: Coastal transition quality

The land-ocean boundary is the most visually critical zone. Options:
- **Per-pixel transition:** render coastal tiles with a gradient from land colours to ocean colours based on sub-face elevation. Already partially done by 10C's water_level colour ramp.
- **Surf/foam overlay:** add white surf lines along the coast using edge detection on the water mask.
- **Shallow water transparency:** coastal ocean tiles show the seabed colour blended with water colour based on depth (shallow = more seabed visible).

#### Decision 4: Interaction with Phase 16 soft blending

Ocean tiles border land tiles. Phase 16's soft edge blending (16B) will cross-fade ocean and land textures at boundaries. This must look correct — the fade zone should produce a natural beach/coast gradient, not a murky green-blue blend. May need special-case handling for ocean↔land boundaries.

### 17A — Ocean Feature Configuration ✅

- [x] **17A.1 — `OceanFeatureConfig` dataclass** in `ocean_render.py` — frozen dataclass with 14 tuneable parameters: `shallow_color`, `deep_color`, `abyssal_color`, `coastal_foam_color`, `sand_color`, `depth_gradient_power`, `wave_frequency`, `wave_amplitude`, `foam_width`, `caustic_frequency`, `caustic_strength`, `ice_latitude_threshold`, `reef_probability`, `density_scale`. Four presets: `TROPICAL_OCEAN`, `TEMPERATE_OCEAN`, `ARCTIC_OCEAN`, `DEEP_OCEAN` in `OCEAN_PRESETS` dict.

- [x] **17A.2 — `compute_ocean_depth_map()`** — hybrid elevation + BFS-distance depth. Multi-source BFS seeds from all land tiles bordering ocean. `elevation_weight`/`distance_weight` control the mix. Also: `identify_ocean_tiles()` (like `identify_forest_tiles` for ocean) and `compute_coast_direction()` (unit 3D vector toward nearest land).

- [x] **17A.3 — Tests:** 19 tests in `test_ocean_render.py`:
  - `TestOceanFeatureConfig` (7): defaults valid, presets exist, valid colours, positive frequencies, shallow brighter than deep, frozen, custom config
  - `TestIdentifyOceanTiles` (3): finds ocean, no ocean, custom terrain type
  - `TestComputeOceanDepthMap` (7): all ocean tiles present, values [0,1], land not in map, coastal < deep, empty set, all-ocean, elevation-weight dominance
  - `TestComputeCoastDirection` (2): coastal has unit vector, deep ocean returns None

### 17B — Ocean Texture Rendering (`ocean_render.py`) ✅

Pixel-level rendering of ocean features onto tile textures.

- [x] **17B.1 — `render_ocean_depth_gradient()`** — fills tile with depth-dependent colour gradient. Three-stop interpolation (shallow→deep→abyssal) controlled by `depth_gradient_power`. Low-frequency FBM noise (±8% depth perturbation) for spatial variation. 4 tests: shallow brighter than deep, not flat, zero-depth ≈ shallow_color, deterministic.

- [x] **17B.2 — `render_wave_pattern()`** — baked wave overlay. Dual sinusoidal ridges (horizontal + diagonal) with FBM noise modulation. Amplitude attenuated by depth (deep = 0.2× shallow). Brightness variation ±4%. 4 tests: modifies pixels, variance, deep calmer, deterministic.

- [x] **17B.3 — `render_coastal_features()`** — foam, sand, caustics, reefs for `depth < 0.3`. Coast direction projects to 2D for edge-localised effects. Foam band with noise breakup. Shallow sand blend at `depth < 0.1`. Caustic sin×sin pattern at `depth < 0.2`. Reef patches via `_render_reef_patches()` (probability-gated). 4 tests: modifies shallow, skips deep, foam brighter near coast, None direction ok.

- [x] **17B.4 — `render_deep_ocean_features()`** — abyssal darkening (up to 25%) + subtle upwelling lighter patches via FBM. Active only when `depth > 0.5`. 3 tests: darkened, shallow unmodified, very deep < medium.

- [x] **17B.5 — `render_ocean_tile()` composite** — applies all four layers in order (gradient → waves → coastal → deep). Returns RGB copy. Also: `_depth_color()`, `_lerp_color()` helpers. 5 tests: RGB output, shallow vs deep brightness, differs from input, deterministic, seed-sensitive.

- [x] **17B.6 — Tests:** 20 new tests in `test_ocean_render.py` (total 39 including 17A).

### 17C — Ocean Pipeline Integration ✅

Wire the ocean renderer into the biome pipeline.

- [x] **17C.1 — `OceanRenderer` class** — implements `BiomeRenderer`:
  1. Look up tile's ocean depth from the depth map
  2. Determine coast direction (from neighbour land/ocean classification)
  3. Render depth gradient → wave pattern → coastal/deep features
  4. Return composited image

- [x] **17C.2 — Biome pipeline registration** — register `OceanRenderer` alongside `ForestRenderer` in `build_feature_atlas()`:
  - Added `biome_type_map` parameter: `{face_id: "forest"|"ocean"|...}`
  - `_get_renderer(fid)` routes each tile to the correct renderer
  - Backward compatible: if no map provided, falls back to first renderer

- [x] **17C.3 — Land-ocean boundary handling** — special logic for Phase 16's soft blending at ocean↔land boundaries:
  - OceanRenderer accepts blend_mask from 16B pipeline
  - Cross-fade between ground texture and ocean render via mask
  - Coast direction computed from globe adjacency for oriented foam/sand

- [x] **17C.4 — Tests:** 9 new tests in `test_biome_pipeline.py`:
  - TestOceanRenderer: protocol conformance, image output, pixel changes, depth map effect
  - TestOceanPipelineIntegration: ocean-only atlas, mixed forest+ocean, visual difference, biome_type_map routing, land tiles unaffected

### 17D — Shader Enhancements ✅

Upgrade the existing 13H water shader to work with the new textured ocean tiles.

- [x] **17D.1 — Texture-aware water shader** — PBR fragment shader now *enhances* baked ocean texture instead of replacing it:
  - Preserves baked atlas texture (`baked_ocean`) with `WATER_TEXTURE_MIX` blend
  - Falls back to procedural colour for untextured water tiles (backward compatible)
  - Animated wave normal perturbation preserved from 13H

- [x] **17D.2 — Fresnel-based reflection** — water-specific Fresnel:
  - `WATER_F0 = 0.02` (IOR ≈ 1.33 for water)
  - `fresnel_water` computed per-fragment: glancing → sky reflection, steep → baked texture
  - Sky reflection colour blended at 60% Fresnel strength

- [x] **17D.3 — Sun specular hotspot** — bright specular on water only:
  - `SUN_SPEC_POWER = 256.0` for tight hotspot on calm water
  - `SUN_SPEC_STRENGTH = 1.8` intensity multiplier
  - Water-specific Fresnel applied to sun specular
  - Only inside `water_hint > 0.5` block, added to final combine

- [x] **17D.4 — Tests:** 7 new tests in `test_globe_renderer_v2.py`:
  - Texture sampling before water override, water-specific Fresnel, sun specular hotspot,
    water-only sun specular, backward compat untextured, constants defined, sky reflection

### 17E — Ocean Globe Demo & Tuning ✅

- [x] **17E.1 — Ocean-focused terrain presets:**
  - `OCEAN_WORLD` — 80% ocean, scattered island chains
  - Added to `TERRAIN_PRESETS` dict and exported from `__init__.py`
  - Existing `ARCHIPELAGO` preset (65% ocean) also works well for ocean demos

- [x] **17E.2 — Demo script `scripts/demo_ocean_globe.py`:**
  - `python scripts/demo_ocean_globe.py` — ocean world with depth gradients
  - `python scripts/demo_ocean_globe.py --terrain earthlike --features` — combined biome
  - `python scripts/demo_ocean_globe.py --ocean tropical --view` — interactive viewer
  - Supports `--soft-blend`, `--forest`, `--ocean` preset flags

- [x] **17E.3 — Visual tuning:**
  - Ocean presets (tropical/temperate/arctic/deep) provide distinct colour palettes
  - 17D shader preserves baked texture with `WATER_TEXTURE_MIX=0.65` blend
  - Fresnel reflection and sun specular enhance surface detail
  - Coastal features (foam/sand/reef) rendered per-tile from 17B

- [x] **17E.4 — Combined biome demo:**
  - `python scripts/demo_ocean_globe.py --terrain earthlike --features`
  - Uses `biome_type_map` to route ocean tiles → OceanRenderer, land tiles → ForestRenderer
  - Both biome density maps computed independently with smooth transitions

- [x] **17E.5 — Tests:** 6 new tests in `test_ocean_render.py`:
  - OCEAN_WORLD in presets, high ocean weight, weights sum to 1
  - Ocean-world generates majority ocean patches
  - Ocean tiles have non-uniform colours (not flat blue)
  - Combined forest + ocean atlas renders correctly

### Summary — Phase 17 Implementation Order

| Step | Module | Depends on | Delivers | Complexity |
|------|--------|-----------|----------|------------|
| **17A** | `ocean_render.py` (config) | 13H water classification | Config, depth map | Low |
| **17B** | `ocean_render.py` (rendering) | 17A | Pixel-level ocean textures | Medium |
| **17C** | `biome_pipeline.py` (ocean) | 17B + 16B (soft blend) | Pipeline integration | Medium |
| **17D** | `globe_renderer_v2.py` (shader) | 17C + 13E (PBR) | Enhanced water shader | Medium |
| **17E** | demo + tuning | 17A–D | End-to-end ocean demo | Low |

### Design notes — Texture vs. shader rendering for ocean

Most game engines render water entirely in shaders (animated normals, reflection mapping, Fresnel). Our approach differs because:

1. **We're rendering a globe from space**, not a close-up water surface. At globe scale, ocean detail is subtle — depth gradients, colour zones, coastal features. These are static and better baked into textures.
2. **The biome pipeline is texture-based.** Forest features are baked into tile textures. Ocean features should use the same pipeline for consistency and code reuse.
3. **Shader effects complement textures.** The animated waves, specular, and Fresnel from 13H add life to the baked ocean textures. It's a two-layer system: baked detail + real-time animation.

### Design notes — Interaction between Phase 16 and Phase 17

Phase 16's soft edge blending (16B) affects ocean↔land boundaries. The blend zone between an ocean tile and a land tile should produce a natural coastal gradient:
- Ocean side: shallow turquoise → sandy coast
- Land side: coastal vegetation → sandy beach

This requires 17C.3 (land-ocean boundary handling) to work with 16B's blend mask. The order of operations:
1. Render land tile texture (terrain + forest features via 16A)
2. Render ocean tile texture (depth + waves + coastal via 17B)
3. Apply Phase 16 blend mask → cross-fade produces coastal gradient
4. Shader adds animation on top

If the automatic cross-fade doesn't produce a good coastal look, 17C.3 can add explicit coastal colours to the blend zone.

---

## Phase 18 — Tile Shape Fidelity & Texture Export *(Planned)*

### Problem Statement

The current rendering pipeline has several cohesion issues visible in the attached screenshots:

1. **Square textures on hex/pent meshes.** Each tile's detail grid is rendered to a square PNG. The 3D mesh maps a hex/pent polygon onto this square via UV projection. The polygon only covers ~75-80% of the square (hex inscribed in a square), so ~20-25% of every tile texture is wasted border that the GPU samples during bilinear filtering at tile edges, causing visible seams and colour bleed.

2. **Tiles are rendered in isolation.** Each tile generates its own sub-face detail grid independently. The outer ring of sub-faces has no knowledge of the neighbouring tile's outer ring. This means terrain, colour, and features all have hard discontinuities at tile boundaries — visible as harsh seams on the 3D globe.

3. **Biome renderers ignore polygrid structure.** The forest and ocean renderers (`biome_render.py`, `ocean_render.py`) paint onto flat PIL images using pixel-level noise and scatter. They don't reference the detail grid's sub-face polygons at all. This means biome features float disconnectedly from the underlying terrain topology — you can't, for example, have a tree per sub-face, or ocean depth that follows sub-face elevation.

4. **PNG atlas is a development convenience, not a production asset.** Game engines and real-time renderers expect compressed texture formats (KTX2, DDS, Basis Universal) with mipmaps, power-of-two dimensions, and optionally normal+roughness channel packing. The current PNG atlas works for our pyglet viewer but wouldn't transfer to a real game pipeline.

### Design: Edge Polygon Sharing ("Apron Tiles")

The key insight for fixing tile boundary seams: **for each tile, include the outermost ring of sub-faces from every neighbouring tile ("apron polygons"), and render them as part of the tile's texture.** This creates an overlap zone where adjacent tiles agree on the pixel content.

```
   ┌─────────────────────────────────┐
   │         Neighbour tile A        │
   │  ┌───────────────────────────┐  │
   │  │  A's edge sub-faces       │  │  ← These get copied into
   │  │  (apron for tile C)       │  │    tile C's texture
   │  └───────────────────────────┘  │
   └─────────────────────────────────┘
              ╲                    ╱
               ╲                  ╱
                ╲     Tile C     ╱
                 ╲──────────────╱
                  │  C's own    │
                  │  sub-faces  │
                  │             │
                  │  + aprons   │
                  │  from A,B,D │
                 ╱──────────────╲
                ╱                ╲
```

For a hex tile with 6 neighbours, we get 6 apron strips. For pentagons, 5. Each strip adds the neighbour's boundary sub-faces (half of the outer ring) into the tile's detail grid before rendering. The result:

- **Every tile's texture extends slightly beyond its polygon boundary**, covering what would otherwise be the seam zone.
- **Adjacent tiles render the same pixels in their overlap zone** (because they share the same sub-face data), producing seamless transitions.
- **The gutter zone in the atlas actually contains correct terrain** instead of clamped edge pixels.

### 18A — Apron Grid Construction ✅

Build the "apron" — the extended detail grid that includes neighbour edge sub-faces.

- [x] **18A.1 — Identify boundary sub-faces** — for a detail grid, classify each sub-face as:
  - `interior` — not adjacent to any boundary edge
  - `boundary` — adjacent to at least one boundary edge
  - `edge_band` — the outermost ring(s) of sub-faces
  Expose as `classify_boundary_subfaces(detail_grid) -> Dict[str, str]`.

- [x] **18A.2 — Compute neighbour edge mapping** — for each globe-level edge between two adjacent tiles, determine which sub-faces from each tile's detail grid lie along that shared edge:
  - Map tile A's edge sub-faces ↔ tile B's edge sub-faces by position on the shared macro-edge.
  - This establishes which sub-faces from neighbour B should appear in tile A's apron.
  - `compute_edge_subface_mapping(globe_grid, face_id_a, face_id_b, coll) -> EdgeSubfaceMapping`

- [x] **18A.3 — Build apron grid** — given a tile's own detail grid + its neighbour edge mappings, construct an extended `PolyGrid` that includes:
  - All of the tile's own sub-faces (unchanged)
  - The outermost sub-face ring from each neighbour (with positions transformed into the tile's local coordinate space)
  - `build_apron_grid(globe_grid, face_id, coll) -> (PolyGrid, apron_mapping)`

- [x] **18A.4 — Apron terrain propagation** — ensure the apron sub-faces have correct elevation data:
  - Copy elevations from the neighbour's store
  - Optionally smooth the join zone (existing `boundary_smoothing` from `TileDetailSpec`)
  - `propagate_apron_terrain(apron_grid, apron_mapping, coll, face_id) -> TileDataStore`

- [x] **18A.5 — Tests:** (28 tests in `tests/test_apron_grid.py`)
  - Boundary classification identifies correct sub-face counts
  - Edge mapping produces matched pairs (A's edge faces ↔ B's edge faces)
  - Apron grid has more sub-faces than the original detail grid
  - Apron terrain is continuous across the join
  - Batch `build_all_apron_grids()` builds for every tile

### 18B — Apron-Aware Texture Rendering ✅

Update the texture pipeline to render apron grids instead of isolated tiles.

- [x] **18B.1 — Render with apron** — new `render_detail_texture_apron()` in `apron_texture.py`:
  - Renders all sub-faces (own + apron) through the existing fullslot pipeline via `render_detail_texture_fullslot()`
  - Output image extends beyond the tile's polygon boundary — apron sub-faces provide correct colours in what was previously the gutter zone
  - Computes own-grid bounding box and clips output to the tile's UV footprint plus apron margin
  - Supports all existing fullslot features: IDW fill, 16D hex softening, noise overlay, colour dithering

- [x] **18B.2 — Polygrid-aware biome rendering** — deferred to 18C. The apron rendering already provides seamless terrain overlap; biome feature rendering on the sub-face grid will be addressed in 18C (Polygrid-Driven Biome Features) as a separate concern.

- [x] **18B.3 — Seamless biome boundaries** — apron grid includes neighbour boundary sub-faces with propagated terrain, so rendering produces consistent colours across tile edges. Full biome-level boundary logic deferred to 18C.

- [x] **18B.4 — Atlas gutter from apron** — new `build_apron_atlas()` in `apron_texture.py`:
  - Builds apron grids for all tiles, renders each with apron context, then assembles atlas
  - `_fill_gutter_from_apron()` fills gutter pixels from the apron image's extended region instead of edge-pixel clamping
  - Supports biome overlays (BiomeRenderer protocol), density maps, seed maps, soft_blend mode
  - UV layout returned alongside atlas image for downstream mesh building

- [x] **18B.5 — Tests:** 17 tests in `test_apron_texture.py`:
  - `render_detail_texture_apron()` produces valid RGBA image at correct tile size
  - No sentinel pixels in the gutter zone (magenta fill eliminated)
  - Apron region has non-uniform pixels (real terrain data, not flat fill)
  - `build_apron_atlas()` produces atlas image + UV layout with correct tile count
  - Atlas gutter pixels are non-uniform (filled from apron data)
  - Atlas slot colours match individual tile renders
  - Adjacent tiles' overlap zones have similar pixel values (boundary continuity)

### 18C — Polygrid-Driven Biome Features ✅

Refactor forest and ocean renderers to be topology-aware.

- [x] **18C.1 — Sub-face forest features** — new `biome_topology.py` module:
  - `scatter_trees_on_grid()`: places trees at sub-face centroids with jitter, density check via face-ID hash
  - `SubfaceTree` dataclass: face_id, pixel coords, radius (proportional to sqrt(sub-face area)), elevation-driven colour
  - Alpine thinning: trees thin out above configurable elevation threshold
  - `render_topology_forest()`: shadow + canopy + highlight rendering (same visual style as existing, new placement logic)
  - `TopologyForestRenderer` class: BiomeRenderer protocol with `set_grid_context()` for grid-aware rendering, pixel-based fallback

- [x] **18C.2 — Sub-face ocean features** — per-sub-face ocean rendering:
  - `compute_subface_ocean_depth()`: depth from `(water_level - elevation) / water_level` per sub-face
  - `identify_coastal_subfaces()`: sub-faces underwater but adjacent to land sub-faces (via grid adjacency)
  - `compute_ocean_subface_props()`: builds `SubfaceOceanProps` list with depth, coastal flag, pixel verts, centroid, area
  - `render_topology_ocean()`: polygon-fills each sub-face with depth colour, coastal foam overlay on coastal faces, abyssal darkening
  - `TopologyOceanRenderer` class: BiomeRenderer protocol with grid context, pixel-based fallback

- [x] **18C.3 — Hybrid rendering** — `render_hybrid_biome()` function:
  - Step 1: Ground texture (already rendered with terrain colours)
  - Step 2: Topology pass — sub-face features (trees or ocean depth polygons)
  - Step 3: Pixel noise overlay for micro-detail (existing `apply_noise_overlay`)
  - Accepts `forest_config` / `ocean_config` for full parameter control
  - Updated `build_apron_feature_atlas()` to call `set_grid_context()` on topology-aware renderers

- [x] **18C.4 — Tests:** 33 tests in `test_biome_topology.py`:
  - Trees placed at sub-face centroids (valid face_id references)
  - Deterministic placement — same seed → same result
  - Density controls tree count; zero density → no trees
  - Radius bounded and proportional to area; depth-sorted
  - Alpine thinning reduces trees at high elevation
  - Forest rendering produces valid RGBA image, modifies ground
  - Per-sub-face ocean depth: below water → positive, above → zero, per-face variation
  - Coastal detection: all-ocean → no coastal; mixed → coastal found; coastal faces are underwater
  - Ocean rendering produces valid RGB, modifies ground, coastal foam applied
  - Hybrid rendering for forest and ocean; noise overlay adds variation
  - Renderer classes work with and without grid context (fallback)
  - Polygon area utility: unit square, triangle, degenerate

### 18D — Texture Export Pipeline ✅

Output proper texture assets suitable for game engines.

- [x] **18D.1 — Power-of-two atlas dimensions** — `next_power_of_two()`, `atlas_pot_size()`, `resize_atlas_pot()`
- [x] **18D.2 — Mipmap generation** — `compute_mip_levels()`, `generate_atlas_mipmaps()` with Lanczos downscaling
- [x] **18D.3 — KTX2 export** — `export_atlas_ktx2()` produces valid KTX2 container (R8G8B8A8_UNORM, optional mipmaps, minimal DFD); `validate_ktx2_header()`
- [x] **18D.4 — Channel-packed material textures** — `build_orm_atlas()` (R=AO from hillshade, G=roughness per biome, B=metallic); `build_material_set()` outputs albedo + normal + ORM PNGs
- [x] **18D.5 — glTF 2.0 export** — `export_globe_gltf()` with embedded base64 buffers/textures, PBR metallic-roughness material, positions/normals/UVs
- [x] **18D.6 — Tests:** 36 tests in `tests/test_texture_export.py` covering PoT, mipmaps, KTX2 header structure, ORM channels, material set, glTF structure

### 18E — Visual Cohesion & Demo ✅

Integration testing and visual tuning to verify the cohesion improvements.

- [x] **18E.1 — Seam elimination verification** — `sample_boundary_pixels()`, `sample_interior_pixels()`, `measure_seam_visibility()` in `visual_cohesion.py`; boundary/interior variance ratio metric
- [x] **18E.2 — Topology feature verification** — `verify_topology_features()` checks tree centroid placement, ocean depth assignment, and determinism
- [x] **18E.3 — Demo script `scripts/demo_phase18_globe.py`** — side-by-side baseline vs apron vs apron+biomes; `--view` for 3D viewer, `--export` for KTX2/glTF, `--bench` for performance
- [x] **18E.4 — Performance budget** — `benchmark_apron_pipeline()` compares apron vs baseline atlas time; target < 2× overhead
- [x] **18E.5 — Tests:** 19 in `tests/test_visual_cohesion.py` (16 fast + 3 slow integration) — seam measurement, topology checks, full pipeline end-to-end with export

### Summary — Phase 18 Implementation Order

| Step | Module | Depends on | Delivers | Complexity |
|------|--------|-----------|----------|------------|
| **18A** | `detail_grid.py`, `tile_detail.py` | 10A (detail grid) | Apron grid construction | High |
| **18B** | `tile_texture.py`, `biome_pipeline.py` | 18A + 16A (fullslot) | Apron-aware rendering | High |
| **18C** | `biome_render.py`, `ocean_render.py` | 18B + 14D (biome) | Topology-aware biomes | Medium |
| **18D** | New: `texture_export.py` | 18B + 13E (normals) | KTX2, glTF, ORM export | Medium |
| **18E** | demo + validation | 18A–D | Visual cohesion proof | Low |

### Design notes — Why apron grids instead of shader-based blending

We considered several approaches for eliminating tile boundary seams:

1. **Shader-level blending** (sample two atlas tiles at boundary pixels) — requires the fragment shader to know which adjacent tile to sample, plus a complex edge-distance SDF. Expensive and hard to get right for 5/6-sided polygons.

2. **Enlarged tile rendering** (render each tile at 110% scale, overlap) — simple but imprecise. The overlap zone doesn't have correct terrain because the detail grid doesn't extend beyond the tile.

3. **Apron grid extension** (our approach) — extends the detail grid with real neighbouring sub-faces. The overlap zone has geometrically correct terrain because it literally uses the neighbour's sub-face data. More work upfront, but the resulting textures are provably seamless.

The apron approach also enables topology-aware biome rendering (18C) because the renderer sees the correct grid structure at tile boundaries — it can place a tree that straddles two tiles and render it consistently on both sides.

### Design notes — Texture format choices

| Format | Pros | Cons | Use case |
|--------|------|------|----------|
| **PNG** | Universal, lossless, easy | No mipmaps, large files, CPU decode | Development, debugging |
| **KTX2** | GPU-native, mipmaps, compression | Needs KTX toolchain | Production rendering |
| **Basis Universal** | Universal GPU transcoding | Lossy, needs basisu | Web/mobile targets |
| **glTF** | Standard 3D interchange | Complex spec | Engine import |

Our default pipeline continues to use PNG for development (fast iteration, easy debugging). The KTX2/glTF export is an optional production step for users who want to import their globe into a game engine.

---

---

## Phase 19 — Coastline Transition Rendering

**Goal:** Replace the hard hexagonal biome boundaries (forest tile butts
directly against ocean tile) with **natural, curving coastlines** that look
like real shorelines.  The transition system must be biome-agnostic at its
core — usable for any A↔B biome pair — but Phase 19 focuses on the
**forest↔ocean** case.

### Problem analysis

The current pipeline assigns ONE biome per Goldberg tile (`biome_type_map`).
`build_apron_feature_atlas()` renders each tile wholly as forest or wholly
as ocean.  Where two adjacent tiles have different biomes, the textures
meet at the tile's hexagonal boundary with **zero cross-biome blending**.
The result is a hard, obviously-hex-shaped transition that looks unnatural.

What's needed:
1. **Per-pixel biome blending** at tile boundaries — a "transition tile"
   that is partly forest, partly ocean, with a noise-warped boundary curve
   between them.
2. **Noise-warped coastline** — the boundary line between forest and ocean
   should not follow the hex edge.  It should be a smooth, organic curve
   generated by displacing a base line with multi-octave noise, producing
   realistic headlands, bays, and inlets.
3. **Coastal detail strip** — the narrow zone around the coastline gets
   special treatment: sand/beach on the land side, shallow water/foam on
   the ocean side, with a smooth gradient between them.

### Architecture

New module: `src/polygrid/coastline.py`

The module provides:
- **`CoastlineMask`** — for a tile that borders a different-biome
  neighbour, generates a per-pixel float mask (0.0 = biome A, 1.0 = biome B)
  with a noise-warped boundary.
- **`compute_coastline_mask()`** — given tile geometry, neighbour biome
  types, and a noise seed, produces the per-pixel mask.
- **`blend_biome_tiles()`** — renders both biomes for the tile area, then
  composites them using the coastline mask.
- **`render_coastal_strip()`** — paints beach/sand/foam detail in the
  narrow transition zone.

Integration point: `build_apron_feature_atlas()` in `apron_texture.py`.
For tiles whose neighbours include a different biome, the pipeline:
1. Identifies which neighbours have which biome type
2. Generates a coastline mask for the tile
3. Renders the tile once as biome A, once as biome B
4. Composites via the mask
5. Paints coastal detail in the transition strip

### Sub-phases

#### 19A — Coastline Mask Generation ✅

Core mask computation: given a tile's position, its biome, and the biome
types of its neighbours, generate a `(tile_size, tile_size)` float32 mask
where 0.0 = "own biome" and 1.0 = "neighbour biome".  The boundary line
is displaced by multi-octave noise to create organic coastline shapes.

- [x] **19A.1** — `CoastlineConfig` dataclass — noise frequency, octaves,
      amplitude, transition width, beach width, seed offset.  4 presets:
      default, gentle, rugged, archipelago.
- [x] **19A.2** — `classify_tile_biome_context()` — for each tile, determine
      if it's interior (all neighbours same biome) or edge (at least one
      neighbour is a different biome).  Returns `TileBiomeContext` with
      the set of edge directions.  `classify_all_tiles()` batch variant.
- [x] **19A.3** — `compute_coastline_mask()` — the core algorithm.  For each
      boundary-neighbour direction, compute a base separating line, warp it
      with fBm noise via `_noise_warp_1d()`, and produce a smooth hermite
      gradient across the transition width.  Multiple neighbour directions
      combined via max.  `_stable_edge_hash()` ensures both tiles sharing
      an edge use the same noise seed for cross-tile continuity.
- [x] **19A.4** — `CoastlineMask` dataclass — holds the mask array, the
      biome assignments (own_biome, other_biomes), config, seed.  Properties:
      `tile_size`, `has_transition`, `transition_fraction`, `coastline_pixels`.
      `build_coastline_mask()` convenience function.
- [x] **19A.5** — Tests: 37 tests in `test_coastline.py` covering config
      construction, frozen immutability, presets, biome classification
      (interior/edge/mixed), mask shape/range/noise-irregularity,
      seed reproducibility, multi-neighbour combination, edge hash
      order-independence, blend compositing, coastal strip rendering,
      edge direction computation.

#### 19B — Dual-Biome Tile Rendering ✅

Render transition tiles by compositing two biome textures through the
coastline mask.

- [x] **19B.1** — `render_transition_tile()` — integrated into
      `build_apron_feature_atlas()`.  For each transition tile, renders
      it once as biome A (own), once as biome B (other), then composites:
      `output = A * (1 - mask) + B * mask`.
- [x] **19B.2** — Integration into `build_apron_feature_atlas()` — detect
      transition tiles via coastline classification, call dual rendering.
      New parameters: `coastline_config`, `enable_coastlines` (default True).
      `_pick_dominant_other_biome()` helper for multi-biome boundaries.
- [x] **19B.3** — `render_coastal_strip()` — paints beach/sand detail in the
      narrow zone where mask ≈ 0.5 (the actual coastline).  Sand on the
      land side (mask 0.25–0.50), foam/shallow water on the ocean side
      (mask 0.50–0.70).
- [x] **19B.4** — Tests: 8 new tests in `test_coastline.py` covering
      atlas creation with coastlines enabled/disabled, transition tile
      detection, no-crash with empty biome maps, custom config,
      _pick_dominant_other_biome helper.

#### 19C — Visual Polish & Demo ✅

- [x] **19C.1** — Tuned noise parameters: default config uses freq=4.0,
      octaves=4, amplitude=0.18 for natural-looking coastlines.  4 presets:
      default, gentle (broad smooth bays), rugged (craggy fjords),
      archipelago (complex island boundaries).
- [x] **19C.2** — Cross-tile coastline continuity via `_stable_edge_hash()` —
      both tiles sharing a boundary edge use the same noise seed,
      producing matching coastline curves from both sides.
- [x] **19C.3** — Demo script `scripts/demo_coastline.py` with argparse:
      `--coastline` preset selection, `--compare` mode (with vs without
      coastlines), `--view` for interactive 3D, coastline statistics printout.
- [x] **19C.4** — Phase 18 demo continues to work (backward compatible).
- [x] **19C.5** — Tests: 4 new tests for cross-tile continuity (shared edge
      hash, adjacent mask correlation, all presets valid).
      49 total tests in `test_coastline.py`.

### Summary — Phase 19 Implementation Order

| Step | Module | Depends on | Delivers | Complexity |
|------|--------|-----------|----------|------------|
| **19A** | New: `coastline.py` | 14D (biome map) | Coastline mask generation | Medium |
| **19B** | `coastline.py`, `apron_texture.py` | 19A + 18B (apron) | Dual-biome compositing | High |
| **19C** | demo + tuning | 19A–B | Visual polish, continuity | Low |

---

## Dependency Roadmap (Updated)

| Phase | New dependencies |
|-------|-----------------|
| 16 (Visual Polish) | None (uses existing PIL + numpy + noise infra) |
| 17 (Ocean Biome) | None (extends existing biome_pipeline + shader infra) |
| 18A–C (Tile Fidelity) | None (extends existing detail grid + texture infra) |
| 18D (Texture Export) | Optional: `pyktx` or `pygltflib` for KTX2/glTF export |
| 19 (Coastlines) | None (uses existing PIL + numpy + noise infra) |
| 20 (Polygon-Aligned Textures) | None (extends existing atlas + UV infra) |

---

## Phase 20 — Polygon-Aligned Texture Rendering

**Goal:** Eliminate the visible tile seam artefacts on the 3D globe by
ensuring the rendered texture content is **aligned to the actual polygon
UV coordinates** that the 3D mesh samples from.

### Root Cause Analysis

The current pipeline has a fundamental coordinate-system mismatch:

1. **Rendering side:** `render_detail_texture_apron()` renders the apron
   grid's sub-faces into a square image using the grid's 2D Tutte-embedding
   coordinates.  The bounding box of those 2D positions defines the mapping
   from world → pixel.

2. **Mesh side:** `GoldbergTile.uv_vertices` are computed by projecting the
   tile's 3D polygon vertices onto the tile's tangent/bitangent plane, then
   normalising to [0,1].  `compute_tile_uvs()` linearly maps these into the
   atlas slot rectangle.

3. **The disconnect:** These two coordinate systems are **different
   projections**.  The apron grid's Tutte embedding positions and the
   GoldbergTile's tangent-plane projection produce different spatial
   layouts.  So when the 3D mesh samples the atlas using the polygon UVs,
   the texture content at the polygon boundary doesn't match what the
   *neighbouring* tile has at its matching boundary → **visible seams**.

Additionally, the square atlas slots waste space on the corners that lie
outside the polygon footprint, and the gutter fill (edge-pixel clamping)
doesn't produce meaningful data for the 3D mesh vertices that sample
near polygon edges.

### Solution Strategy

Re-render each tile's texture using the **same projection** that
`GoldbergTile.uv_vertices` uses — i.e. project the detail sub-faces
onto the tile's tangent/bitangent basis, normalise to [0,1], and
rasterise into the square atlas slot.  This way the texture and the mesh
UVs are in perfect agreement.

Additionally, render apron/neighbour sub-faces into the gutter zone so
that bilinear sampling at polygon edges sees correct adjacent-tile data
instead of clamped edge pixels.

### Phase 20A — UV-Aligned Tile Rasteriser

**New function** `render_tile_uv_aligned()` in a new module
`src/polygrid/uv_texture.py`:

- [✅] **20A.1** — `compute_tile_basis`, `project_point_to_tile_uv`,
  `compute_tile_uv_bounds`, `project_and_normalize`:  Project sub-face
  vertex positions (3D → tile's tangent-plane 2D → normalised [0,1])
  using the same derivation that the models library uses for
  `GoldbergTile.uv_vertices`.

- [✅] **20A.2** — `UVTransform` + `compute_detail_to_uv_transform`:
  Fit a similarity transform (rotation + uniform scale + translation)
  mapping detail-grid 2D positions to tile UV [0,1] via angle-based
  vertex matching and least-squares.

- [✅] **20A.3** — `render_tile_uv_aligned`: Rasterise each sub-face
  polygon (including apron sub-faces from neighbours) in UV-aligned
  coordinates into a `tile_size × tile_size` image.  Apron sub-faces
  naturally land outside the [0,1] polygon boundary, filling the
  gutter zone with real terrain data.

- [✅] **20A.4** — Tests in `tests/test_uv_texture.py`:
  - UV projection matches models library basis derivation
  - Normalised UVs in [0,1] with correct extremes
  - UVTransform affine mapping (identity, translation, rotation, scale)
  - Similarity transform properties preserved
  - Rasterised image has non-sentinel pixels, correct size
  - Pentagon and hexagon tiles both work correctly

### Phase 20B — Atlas Assembly with UV-Aligned Tiles

Replace the current atlas builder's rendering step with the UV-aligned
rasteriser.

- [✅] **20B.1** — `build_uv_aligned_atlas(collection, globe_grid, *,
  biome_renderers, biome_type_map, ...)`:  New atlas builder that uses
  `render_tile_uv_aligned()` instead of `render_detail_texture_apron()`.
  The atlas layout (slot grid, gutter, UV coordinates) stays the same —
  only the per-tile rendering changes.

- [✅] **20B.2** — Biome overlays:  Apply `ForestRenderer` / `OceanRenderer`
  to the UV-aligned ground images (same as current Phase 18C flow).

- [✅] **20B.3** — Coastline integration:  Phase 19 coastline masks work
  in pixel space within each tile, so they continue to work unchanged on
  the UV-aligned tiles.

- [🔲] **20B.4** — Tests:
  - Atlas UV layout unchanged (backward compatible with mesh builder)
  - Visual: boundary pixels between adjacent tiles are colour-coherent
  - Biome overlays render correctly on UV-aligned base
  - Coastlines still blend properly

### Phase 20C — Cross-Tile Boundary Stitching

Ensure adjacent tiles share identical pixel values along their shared
polygon edge.

- [🔲] **20C.1** — `stitch_tile_boundaries(tile_images, globe_tiles,
  uv_layout)`:  For each pair of adjacent tiles, sample both textures
  along their shared polygon edge and average/blend the values.  Write
  the blended strip back into both tiles.  This guarantees zero-seam
  continuity even under bilinear interpolation.

- [🔲] **20C.2** — Gutter coherence:  Ensure the gutter zone of each tile
  contains the same data that the adjacent tile's interior has.  With
  UV-aligned apron rendering (20A.3) this should be automatic, but verify
  and add a fallback: copy a narrow strip from the neighbour's rendered
  tile into the current tile's gutter at the shared edge.

- [🔲] **20C.3** — Tests:
  - Shared-edge pixel identity (sample both tiles along edge → values match)
  - Gutter zone = adjacent tile interior along shared edges
  - No visible seams in 3D viewer (visual regression baseline)

### Phase 20D — Demo & Visual Validation

- [🔲] **20D.1** — Update `demo_coastline.py` (or new `demo_phase20.py`)
  to use the UV-aligned atlas.  Side-by-side comparison: old square-slot
  atlas vs new UV-aligned atlas on the 3D globe.

- [🔲] **20D.2** — Add `--uv-aligned` flag to the demo to toggle between
  old and new pipelines for A/B comparison.

- [🔲] **20D.3** — Performance: UV-aligned rendering should be similar
  cost to current apron rendering (same number of sub-faces, same
  rasterisation).  Benchmark and document any difference.

- [🔲] **20D.4** — Export compatibility:  Ensure Phase 18D exports
  (KTX2, ORM, glTF) still work with the new atlas (same UV layout,
  just better pixel content).

### Summary — Phase 20 Implementation Order

| Step | Module | Depends on | Delivers | Complexity |
|------|--------|-----------|----------|------------|
| **20A** | New: `uv_texture.py` | models lib (GoldbergTile), apron_grid | UV-aligned tile rasteriser | High |
| **20B** | `uv_texture.py`, `apron_texture.py` | 20A + 18C (biome overlays) | UV-aligned atlas builder | Medium |
| **20C** | `uv_texture.py` | 20A–B | Cross-tile boundary stitching | Medium |
| **20D** | demo + tuning | 20A–C | Visual validation, performance | Low |
