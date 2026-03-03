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

## Phase 16 — Cross-Biome Visual Polish 🔲

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

### 16B — Soft Tile-Edge Blending Mask 🔲

**Problem:** Even with full-slot rendering, adjacent tiles' textures will differ slightly at boundaries (different sub-face grids, different noise seeds from Phase 11B patches, different biome parameters). A hard UV-polygon cutoff makes this a visible edge.

**Fix:** Apply a soft radial alpha mask to each tile's texture so that edges cross-fade with neighbouring tiles' extended textures.

- [ ] **16B.1 — Tile blending mask** — `compute_tile_blend_mask(tile_shape, tile_size, fade_width)` → `np.ndarray` (float32, [0,1]). A 2D mask that is 1.0 at the tile centre and fades to 0.0 at the tile edges over `fade_width` pixels. The mask follows the hex/pent shape (not a simple circle), using signed distance from the polygon boundary.

- [ ] **16B.2 — Atlas-level blending** — after rendering all tile textures, apply the blend mask: each tile's atlas slot is multiplied by its mask. Adjacent tiles' gutter overlap (from 13B) now contributes colour where the mask < 1.0. The atlas becomes a seamless image rather than a grid of isolated tiles.

- [ ] **16B.3 — UV overlap extension** — extend each tile's UV polygon slightly beyond its strict boundary (by `fade_width` pixels in atlas space). This means each triangle fan samples a slightly larger region of the atlas, overlapping with its neighbour's fade zone. Combined with the alpha mask, this produces cross-fade blending at every tile boundary.

- [ ] **16B.4 — Tests:**
  - Blend mask is 1.0 at tile centre
  - Blend mask is 0.0 at tile corners
  - Mask follows hex/pent shape (verified at midpoints of hex edges vs corners)
  - Applied mask reduces max colour difference across adjacent tile boundaries
  - Atlas with blending has lower boundary contrast than without

### 16C — Biome Feature Edge Overflow 🔲

**Problem:** Forest canopy scatter (14A) places trees within the hex tile bounds with a small margin zone (14C). Trees near tile edges are clipped at the hex boundary. Adjacent tiles' canopies don't mesh.

**Fix:** Extend biome feature rendering to cover the full tile slot (not just the hex interior).

- [ ] **16C.1 — Full-slot feature scattering** — modify `scatter_features_on_tile()` to scatter features across the full square tile area (0,0 → tile_size, tile_size), not just the hex interior. Features outside the hex footprint will be visible through the fade zone (16B). Poisson disk sampling already works on rectangles — just expand the bounds.

- [ ] **16C.2 — Neighbour-seeded margin features** — for tile edges, read the neighbours' density and seed values (from the globe-wide density map) and use them to scatter features in the margin zone with the *neighbour's* parameters. This ensures that trees flowing across a tile boundary have consistent species, size, and spacing from the neighbour's perspective.

- [ ] **16C.3 — Feature-level cross-fade** — features near the tile edge are rendered with reduced opacity (following the blend mask from 16B). Features from the neighbour (rendered into the neighbour's extended tile area) provide the matching opacity. The sum is a seamless canopy.

- [ ] **16C.4 — Tests:**
  - Features are scattered across the full square, not just the hex interior
  - Margin features use neighbour density values
  - Feature count in the extended area is proportional to the expanded bounds
  - No visible gap in feature density at hex boundary (statistical test)

### 16D — Hex Shape Softening 🔲

**Problem:** The hex sub-face grid creates a visible hexagonal structure in the texture — concentric rings of faces with slightly different sizes/colours.

**Fix:** Break up the hex grid pattern at the rendering level.

- [ ] **16D.1 — Sub-face edge dissolution** — when rendering sub-face polygons, add a small random jitter (1-2px) to sub-face vertex positions. This prevents the perfectly straight edges between sub-faces from creating visible grid lines. The jitter is noise-seeded (deterministic) and much smaller than a sub-face, so it doesn't distort topology.

- [ ] **16D.2 — Pixel-level noise overlay** — after rendering sub-face polygons, apply a high-frequency pixel-level noise layer (Perlin/simplex at frequency ~0.05 px⁻¹) that adds micro-variation. This breaks up the uniform colour within each sub-face and masks the regular grid pattern. Amplitude: ±5% of the base colour.

- [ ] **16D.3 — Sub-face colour dithering** — instead of a single colour per sub-face, vary the colour across the sub-face area using interpolation from the sub-face's neighbours' colours. This softens the hard colour boundaries between adjacent sub-faces. Implemented in the PIL renderer (not matplotlib).

- [ ] **16D.4 — Tests:**
  - Jittered vertex positions are within ±2px of originals
  - Noise overlay changes pixel values by < 15% of base
  - Dithered sub-faces have lower boundary contrast than non-dithered
  - Output image is deterministic (same seed → same pixels)

### 16E — Pipeline Integration & Validation 🔲

Wire all the 16A–D improvements into the production rendering pipeline.

- [ ] **16E.1 — Pipeline flags** — add `--soft-blend` / `--no-soft-blend` flags to `view_globe_v3.py` and `build_feature_atlas()`. Default: enabled. Controls whether 16A–D enhancements are active.

- [ ] **16E.2 — Performance budget** — the extended rendering (16A) adds ~44% more pixel work per tile (filling corners). Soft blending (16B) adds a mask multiplication pass. Feature overflow (16C) adds ~20% more scatter points. Total target: < 1.5× current atlas build time.

- [ ] **16E.3 — Visual regression test** — render the same globe (freq=3, rings=4, temperate forest) with and without Phase 16 enhancements. Compute boundary contrast metrics (mean absolute colour difference along tile edges). Phase 16 version must have < 50% of the non-enhanced boundary contrast.

- [ ] **16E.4 — Demo update** — update `demo_forest_globe.py` to use Phase 16 pipeline by default. Add `--compare` flag that renders side-by-side panels showing the improvement.

- [ ] **16E.5 — Tests:**
  - Pipeline runs end-to-end with all enhancements enabled
  - Atlas dimensions and UV layout unchanged (backward compatible)
  - Performance within budget
  - Boundary contrast metric meets threshold

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

### 17A — Ocean Feature Configuration 🔲

- [ ] **17A.1 — `OceanFeatureConfig` dataclass** — tuneable parameters:
  - `shallow_color` — `(r, g, b)` for shallow coastal water (turquoise)
  - `deep_color` — `(r, g, b)` for deep ocean (dark navy)
  - `abyssal_color` — `(r, g, b)` for very deep ocean (near-black blue)
  - `coastal_foam_color` — `(r, g, b)` for surf/foam (white/cream)
  - `depth_gradient_power` — controls shallow→deep transition curve
  - `wave_frequency` — spatial frequency of baked wave texture
  - `wave_amplitude` — visual strength of wave pattern in texture
  - `foam_width` — width of coastal foam band in pixels
  - `caustic_frequency` — underwater caustic pattern frequency
  - `caustic_strength` — visual intensity of caustic patterns
  - `ice_latitude_threshold` — latitude above which ocean becomes icy
  - `reef_probability` — chance of shallow-water reef colour patches
  - Presets: `TROPICAL_OCEAN`, `TEMPERATE_OCEAN`, `ARCTIC_OCEAN`, `DEEP_OCEAN`

- [ ] **17A.2 — `compute_ocean_depth_map(globe_grid, globe_store)` → `Dict[str, float]`** — globe-wide ocean depth map:
  1. Identify ocean tiles (elevation < water_level)
  2. Compute BFS distance from nearest land tile
  3. Combine: `depth = (water_level - elevation) * distance_factor`
  4. Normalise to [0, 1] (0 = coastline, 1 = deepest ocean)

- [ ] **17A.3 — Tests:**
  - Config presets have valid colour tuples
  - Depth map: coastal tiles have depth ≈ 0, deep tiles ≈ 1
  - All ocean tiles present in depth map
  - Land tiles not in depth map (or have depth 0)

### 17B — Ocean Texture Rendering (`ocean_render.py`) 🔲

Pixel-level rendering of ocean features onto tile textures.

- [ ] **17B.1 — `render_ocean_depth_gradient(image, depth, config)` → None** — fill the tile with a depth-dependent colour gradient. Uses cubic interpolation between shallow/deep/abyssal colours based on the tile's depth value. Not a flat fill — adds subtle spatial variation using low-frequency noise.

- [ ] **17B.2 — `render_wave_pattern(image, depth, config)` → None** — overlay a subtle wave texture:
  - Low-frequency sinusoidal ridges modulated by noise for natural wave patterns
  - Wave amplitude decreases with depth (deep ocean = calmer surface)
  - Orientation varies by latitude (trade winds, westerlies)
  - Rendered as subtle brightness variation (±3-5% of base colour)

- [ ] **17B.3 — `render_coastal_features(image, depth, coast_direction, config)` → None** — for shallow ocean tiles near coast:
  - Foam/surf line: white band along the coast-facing edge
  - Shallow-water sand visibility: warm beige-green blend where depth < 0.1
  - Reef patches: irregular darker/lighter spots in shallow water (noise-based)
  - Caustic ripple pattern: bright undulating network (underwater light refraction)

- [ ] **17B.4 — `render_deep_ocean_features(image, depth, config)` → None** — for deep ocean tiles:
  - Abyssal darkness: colour tends toward near-black blue
  - Occasional lighter patches: upwelling/current indicators (very subtle noise)
  - Minimal wave texture (deep ocean surface is smoother)

- [ ] **17B.5 — Tests:**
  - Depth gradient: shallow tile has higher RGB mean than deep tile
  - Wave pattern: pixel variance > 0 (not flat)
  - Coastal features: foam pixels near edge are brighter than interior
  - Deep ocean: average brightness lower than shallow ocean
  - Empty/zero-depth tile: returns plausible shallow ocean

### 17C — Ocean Pipeline Integration 🔲

Wire the ocean renderer into the biome pipeline.

- [ ] **17C.1 — `OceanRenderer` class** — implements `BiomeRenderer`:
  1. Look up tile's ocean depth from the depth map
  2. Determine coast direction (from neighbour land/ocean classification)
  3. Render depth gradient → wave pattern → coastal/deep features
  4. Return composited image

- [ ] **17C.2 — Biome pipeline registration** — register `OceanRenderer` alongside `ForestRenderer` in `build_feature_atlas()`:
  ```python
  biome_renderers = {
      "forest": ForestRenderer(config),
      "ocean": OceanRenderer(ocean_config),
  }
  ```

- [ ] **17C.3 — Land-ocean boundary handling** — special logic for Phase 16's soft blending at ocean↔land boundaries:
  - Coastal land tiles: blend toward sandy/beach colours at their ocean-facing edges
  - Coastal ocean tiles: blend toward shallow turquoise at their land-facing edges
  - The blend zone should produce a natural beach gradient

- [ ] **17C.4 — Tests:**
  - OceanRenderer satisfies BiomeRenderer protocol
  - Feature atlas with both forest and ocean tiles renders correctly
  - Land tiles unaffected by ocean renderer
  - Coastal transition: boundary pixels between land and ocean tiles have intermediate colours

### 17D — Shader Enhancements 🔲

Upgrade the existing 13H water shader to work with the new textured ocean tiles.

- [ ] **17D.1 — Texture-aware water shader** — the PBR fragment shader currently replaces water tile colours entirely. Update it to *enhance* the baked ocean texture instead:
  - Sample the atlas texture (which now has depth gradients, waves, etc.)
  - Add animated wave normal perturbation on top (existing 13H)
  - Add specular reflection enhancement (water is more reflective than land)
  - Preserve the baked depth gradient and coastal features

- [ ] **17D.2 — Fresnel-based reflection** — enhance water Fresnel:
  - At glancing angles (near globe edge): water is highly reflective (sky-coloured)
  - At steep angles (looking straight down): water is transparent (shows baked texture)
  - Uses existing Schlick approximation but with water-specific IOR

- [ ] **17D.3 — Sun specular hotspot** — bright specular reflection of the sun on the water surface:
  - Position moves with globe rotation
  - Size controlled by roughness (calm water = tight hotspot, rough = broad)
  - Only on water tiles (use water_flag)

- [ ] **17D.4 — Tests:**
  - Shader source contains texture sampling before water colour override
  - Fresnel keywords present in shader
  - Specular hotspot code present
  - Backward compatibility: shader still works with untextured water tiles

### 17E — Ocean Globe Demo & Tuning 🔲

- [ ] **17E.1 — Ocean-focused terrain presets:**
  - `OCEAN_WORLD` — 80% ocean, scattered island chains (ideal for ocean demo)
  - `ARCHIPELAGO_OCEAN` — existing archipelago preset with enhanced ocean features
  - Update existing presets to enable ocean rendering by default

- [ ] **17E.2 — Demo script `scripts/demo_ocean_globe.py`:**
  - `python scripts/demo_ocean_globe.py -f 3 --detail-rings 4 --preset ocean_world --view`
  - Shows ocean depth gradients, coastal features, wave patterns, sun reflections

- [ ] **17E.3 — Visual tuning** — iterate on:
  - Depth gradient colour stops for convincing ocean-from-space look
  - Wave pattern scale relative to tile size
  - Coastal foam width and brightness
  - Deep vs shallow ocean contrast
  - Interaction with atmosphere shader (ocean should be slightly hazier near horizon)

- [ ] **17E.4 — Combined biome demo** — globe with forests on land AND oceans with depth:
  - `python scripts/demo_ocean_globe.py -f 3 --detail-rings 4 --preset earthlike --features --view`
  - The "money shot": a full planet with rich forests, varied terrain, and detailed oceans

- [ ] **17E.5 — Tests:**
  - Ocean-world globe renders without error
  - All ocean tiles have non-uniform pixel colours (not flat blue)
  - Combined forest + ocean globe renders correctly
  - Performance: atlas build with ocean + forest < 3× baseline

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

## Dependency Roadmap (Updated)

| Phase | New dependencies |
|-------|-----------------|
| 16 (Visual Polish) | None (uses existing PIL + numpy + noise infra) |
| 17 (Ocean Biome) | None (extends existing biome_pipeline + shader infra) |
