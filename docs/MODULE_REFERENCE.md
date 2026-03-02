# PolyGrid Module Reference

Quick reference for every module in `src/polygrid/`, what it does, and what it depends on.

**40 source files · 17,224 lines · 1,101 tests**

---

## Core Layer (zero optional dependencies)

### `models.py` (71 lines)
Frozen dataclasses: `Vertex`, `Edge`, `Face`, `MacroEdge`. Pure value objects with no logic beyond basic validation.

### `polygrid.py` (379 lines)
`PolyGrid` — the central container. Holds vertex/edge/face dicts, macro-edges. Provides validation, serialisation, boundary detection, adjacency computation.

**Depends on:** `models`, `algorithms`, `geometry`

### `algorithms.py` (83 lines)
Pure graph algorithms: `build_face_adjacency`, `ring_faces` (BFS ring grouping).

**Depends on:** `models`

### `geometry.py` (200 lines)
Geometry helpers: vertex ordering, interior angles, signed areas, edge lengths, centroids, boundary walking.

**Depends on:** `models`

### `io.py` (19 lines)
`load_json`, `save_json` — thin wrappers around `PolyGrid` serialisation.

**Depends on:** `polygrid`

---

## Building Layer

### `builders.py` (168 lines)
Grid constructors: `build_pure_hex_grid`, `build_pentagon_centered_grid`, `validate_pentagon_topology`.

**Depends on:** `models`, `polygrid`, `algorithms`, `goldberg_topology`

### `goldberg_topology.py` (617 lines)
Goldberg polyhedron face topology: triangulation, dualisation, Tutte embedding, least-squares optimisation, winding fix.

**Depends on:** `models`, `polygrid`; optionally `numpy`, `scipy`

### `composite.py` (233 lines)
Multi-grid stitching: `StitchSpec`, `stitch_grids` → `CompositeGrid`, `join_grids`, `split_composite`.

**Depends on:** `models`, `polygrid`

### `assembly.py` (391 lines)
High-level assembly recipes: `AssemblyPlan`, `pent_hex_assembly`, rigid transforms, hex positioning, hex-hex boundary snapping.

**Depends on:** `builders`, `composite`, `models`, `polygrid`

---

## Transform Layer

### `transforms.py` (258 lines)
Overlay data model (`Overlay`, `OverlayPoint`, `OverlaySegment`, `OverlayRegion`) + transforms: `apply_voronoi`, `apply_partition`.

**Depends on:** `geometry`, `models`, `polygrid`

### `regions.py` (730 lines)
Terrain partitioning: `Region`, `RegionMap`, `RegionValidation`. Four algorithms: `partition_angular`, `partition_flood_fill`, `partition_voronoi`, `partition_noise`. TileData integration: `assign_field`, `assign_biome`. Visualisation: `regions_to_overlay`.

**Depends on:** `algorithms`, `geometry`, `models`, `polygrid`, `transforms`

### `region_stitch.py` (615 lines)
Cross-region boundary stitching: ensures terrain continuity at region boundaries with smooth transitions.

**Depends on:** `regions`, `tile_data`

---

## Tile Data Layer

### `tile_data.py` (490 lines)
Per-face key-value storage: `FieldDef`, `TileSchema`, `TileData`, `TileDataStore`. Schema-validated writes, neighbour/ring queries, bulk operations, JSON serialisation.

**Depends on:** `algorithms`, `polygrid`

---

## Terrain Generation Layer

### `noise.py` (475 lines)
Noise primitives: simplex noise with octave layering (`octave_noise`), domain warping (`warped_noise`), ridged noise, billowed noise. Used by heightmap and mountain generators.

**Depends on:** optionally `opensimplex`; deterministic fallback

### `heightmap.py` (267 lines)
Grid-noise bridge: `apply_heightmap` maps noise to per-face elevations. Supports erosion simulation and normalisation.

**Depends on:** `noise`, `tile_data`

### `mountains.py` (346 lines)
Mountain range generation: ridgeline-based peaks with erosion, slope-dependent colouring, configurable profiles.

**Depends on:** `noise`, `heightmap`, `tile_data`

### `rivers.py` (657 lines)
River network generation: downhill flow, watershed detection, confluence points, river width based on accumulated flow.

**Depends on:** `algorithms`, `tile_data`, `heightmap`

### `terrain_render.py` (388 lines)
Elevation-to-colour mapping: `BiomeConfig` palettes, elevation-aware colouring, biome blending. Used for satellite-style textures.

**Depends on:** `tile_data`

### `terrain_patches.py` (620 lines)
Terrain patch stitching: ensures smooth terrain transitions across tile boundaries.

**Depends on:** `tile_data`, `terrain_render`

### `pipeline.py` (314 lines)
Pipeline/composition framework: chain terrain generation passes with `Pipeline` and `PipelineStep`.

**Depends on:** `tile_data`, `noise`, `heightmap`

### `render_enhanced.py` (346 lines)
Enhanced rendering utilities: additional render modes and colour processing.

**Depends on:** `terrain_render`

---

## Globe Layer

### `globe.py` (254 lines)
Goldberg polyhedron globe builder: `build_globe_grid` creates a globe-scale `PolyGrid` with `TileDataStore` from the `models` library's `generate_goldberg_tiles`.

**Depends on:** `polygrid`, `tile_data`, `models` library

### `globe_terrain.py` (552 lines)
Globe-scale terrain generation: applies noise, mountains, biome colouring at the global scale. Supports presets (`alpine_peaks`, `mountain_range`, etc.).

**Depends on:** `globe`, `noise`, `heightmap`, `mountains`, `terrain_render`

### `globe_export.py` (241 lines)
Globe JSON export: `export_globe_json`, `validate_globe_payload`. Produces per-tile payloads with 3D positions, normals, colours, adjacency.

**Depends on:** `globe`, `tile_data`

### `globe_mesh.py` (243 lines)
3D mesh bridge: `build_coloured_globe_mesh`, `build_terrain_layout_mesh`, `prepare_terrain_scene`. Converts terrain colours to `models` library's `ShapeMesh`.

**Depends on:** `globe`, `models` library

### `globe_render.py` (265 lines)
Render helpers for globe output.

**Depends on:** `globe_mesh`, `models` library

---

## Sub-Tile Detail Layer

### `detail_grid.py` (322 lines)
Detail grid builder: `build_detail_grid`, `generate_detail_terrain`, `render_detail_texture`, `build_texture_atlas`.

**Depends on:** `polygrid`, `tile_data`

### `tile_detail.py` (335 lines)
Sub-tile infrastructure: `TileDetailSpec`, `build_all_detail_grids`, `DetailGridCollection`.

**Depends on:** `detail_grid`, `globe`

### `detail_terrain.py` (373 lines)
Boundary-aware detail terrain: `compute_boundary_elevations`, `classify_detail_faces`, `generate_detail_terrain_bounded`.

**Depends on:** `tile_detail`, `heightmap`, `noise`

### `detail_terrain_3d.py` (511 lines)
3D terrain detail with elevation displacement for realistic terrain rendering.

**Depends on:** `detail_terrain`, `heightmap`

### `detail_render.py` (437 lines)
Satellite-style detail textures: biome-aware colour palettes, matplotlib and PIL fast-path rendering.

**Depends on:** `detail_terrain`, `terrain_render`

### `detail_perf.py` (570 lines)
Performance: parallel generation (`ProcessPoolExecutor`), PIL fast-path rendering, caching.

**Depends on:** `detail_render`, `detail_terrain`

### `texture_pipeline.py` (336 lines)
Texture atlas + UV mapping: `build_atlas`, `compute_tile_uvs`, `build_uv_layout`. Packs tile textures into a single GPU-ready atlas PNG.

**Depends on:** `detail_render`

---

## GPU Rendering Layer

### `globe_renderer.py` (851 lines)
OpenGL renderer v1: flat-colour and textured modes using `models` library's `SimpleMeshRenderer`. Includes `render_textured_globe_opengl`, `render_terrain_globe_opengl`.

**Depends on:** `globe_mesh`, `models` library, `pyglet`

### `globe_renderer_v2.py` (2,403 lines)
Phase 12–13 renderer — the main rendering engine:

| Section | Functions |
|---------|-----------|
| **12A Flood-fill** | `flood_fill_tile_texture`, `flood_fill_atlas` |
| **13C UV clamping** | `compute_uv_polygon_inset`, `clamp_uv_to_polygon` |
| **13D Colour harmonisation** | `blend_biome_configs`, `compute_neighbour_average_colours`, `harmonise_tile_colours` |
| **13H Water** | `classify_water_tiles`, `compute_water_depth` |
| **13E Normal maps** | `encode_normal_to_rgb`, `decode_rgb_to_normal`, `build_normal_map_atlas` |
| **12B Subdivision** | `subdivide_tile_mesh` |
| **12C Batched mesh** | `build_batched_globe_mesh` |
| **13F Adaptive LOD** | `select_lod_level`, `estimate_tile_screen_fraction`, `is_tile_backfacing`, `stitch_lod_boundary`, `build_lod_batched_globe_mesh` |
| **13G Atmosphere** | `build_atmosphere_shell`, `build_background_quad`, `compute_bloom_threshold` |
| **Shaders** | PBR, atmosphere, background, bloom (vertex + fragment) |
| **Viewer** | `render_globe_v2` (interactive pyglet window) |

**Depends on:** `texture_pipeline`, `models` library, `pyglet`, `numpy`, `Pillow`

---

## 2D Visualisation Layer

### `visualize.py` (714 lines)
Multi-panel composite visualisation: `render_png`, `render_single_panel`, `render_exploded`, `render_stitched`, `render_assembly_panels`. Partition colouring.

**Depends on:** `assembly`, `composite`, `models`, `polygrid`, `transforms`; requires `matplotlib`

### `render.py` (20 lines)
Deprecated shim — re-exports `render_png` from `visualize`.

---

## Entry Points

### `cli.py` (249 lines)
CLI: `validate`, `render`, `build-hex`, `build-pent`, `build-pent-all`, `assembly`.

### `diagnostics.py` (311 lines)
Per-ring quality diagnostics: edge lengths, interior angles, area checks, quality gates.

---

## Stats

| Metric | Count |
|--------|-------|
| Source files | 40 |
| Source lines | ~17,200 |
| Test files | 36 |
| Tests | 1,101 |
