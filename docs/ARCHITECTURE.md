# PolyGrid Architecture

## Project Vision

PolyGrid is a **topology-first toolkit** for building, composing, and running algorithms on polygon grids — primarily **hex grids** and **pentagon-centred Goldberg grids**. The end goal is **procedural terrain generation for a game** where each tile sits on a Goldberg polyhedron (a sphere tiled by hexagons and 12 pentagons).

The workflow is:

1. **Build** individual polygon grids (hex, pentagon-centred Goldberg faces).
2. **Compose** them into assemblies (e.g. one pentagon + five hex grids stitched together).
3. **Run algorithms** on the composed grid — partitioning into terrain zones, assigning biome types, generating heightmaps, placing features.
4. **Store per-tile data** (biome, elevation, moisture, etc.) against individual faces.
5. **Generate detail** — expand each Goldberg tile into a local sub-grid with boundary-continuous terrain.
6. **Render textures** — satellite-style detail textures packed into a texture atlas.
7. **GPU render** — PBR-lit, normal-mapped 3D globe with water, atmosphere, bloom, and adaptive LOD.

A separate `models` package provides the Goldberg polyhedron geometry primitives (tile positions, normals, tangents, adjacency). PolyGrid handles the **topology, terrain, textures, and rendering** side.

---

## Separation of Concerns

The project is organised around strict layering:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           scripts / CLI                                │
├─────────────────────────────────────────────────────────────────────────┤
│  globe_renderer_v2.py │ visualize.py │ render.py                     │  GPU + 2D rendering
├─────────────────────────────────────────────────────────────────────────┤
│  texture_pipeline.py │ detail_render.py │ detail_perf.py              │  Texture pipeline
├─────────────────────────────────────────────────────────────────────────┤
│  detail_grid.py │ tile_detail.py │ detail_terrain.py                  │  Sub-tile detail
│                                  │ detail_terrain_3d.py               │
├─────────────────────────────────────────────────────────────────────────┤
│  globe.py │ globe_terrain.py │ globe_export.py │ globe_mesh.py       │  Globe scale
├─────────────────────────────────────────────────────────────────────────┤
│  noise.py │ heightmap.py │ mountains.py │ rivers.py │ pipeline.py    │  Terrain generation
│  terrain_render.py │ terrain_patches.py │ render_enhanced.py         │
├─────────────────────────────────────────────────────────────────────────┤
│  transforms.py │ regions.py │ region_stitch.py                       │  Algorithms
├─────────────────────────────────────────────────────────────────────────┤
│  tile_data.py                                                         │  Per-tile data
├─────────────────────────────────────────────────────────────────────────┤
│  assembly.py │ composite.py │ builders.py │ goldberg_topology.py     │  Composition
├─────────────────────────────────────────────────────────────────────────┤
│  polygrid.py │ models.py │ algorithms.py │ geometry.py │ io.py       │  Core topology
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer rules

| Layer | Knows about | Does NOT know about |
|-------|------------|---------------------|
| **Core** (`models`, `polygrid`, `algorithms`, `geometry`, `io`) | Vertices, edges, faces, adjacency | Rendering, assembly, transforms, tile data |
| **Building** (`builders`, `goldberg_topology`, `composite`, `assembly`) | Core layer | Rendering, game-specific data |
| **Tile Data** (`tile_data`) | Core layer | Rendering, transforms, regions |
| **Transforms** (`transforms`, `regions`, `region_stitch`) | Core layer, tile data | Rendering |
| **Terrain** (`noise`, `heightmap`, `mountains`, `rivers`, `pipeline`) | Core, tile data, transforms | Rendering, globe topology |
| **Globe** (`globe`, `globe_terrain`, `globe_export`, `globe_mesh`) | Core, tile data, terrain, `models` lib | Rendering |
| **Detail** (`detail_grid`, `tile_detail`, `detail_terrain`, `detail_render`, `detail_perf`) | Core, terrain, globe | GPU rendering |
| **Texture** (`texture_pipeline`) | Detail, globe | GPU rendering |
| **Rendering** (`globe_renderer_v2`, `visualize`) | Everything above | Nothing below it depends on rendering |
| **Scripts / CLI** | Everything | — |

**Key principle:** the core topology layer has **zero rendering dependencies**. Matplotlib and pyglet are optional installs. All algorithm work operates on the abstract `PolyGrid` graph.

---

## Core Data Model

### `PolyGrid`

The central container. Holds dictionaries of vertices, edges, and faces, plus optional macro-edges (the sides of the grid's outer polygon). Supports:

- **Validation** — referential integrity checks between faces ↔ edges ↔ vertices.
- **Serialisation** — lossless JSON round-trip via `to_dict()` / `from_dict()`.
- **Boundary detection** — identifies boundary edges, walks the boundary cycle, detects corners, computes macro-edges.
- **Adjacency** — `compute_face_neighbors()` builds a face adjacency map from shared edges.

### Frozen dataclasses

`Vertex`, `Edge`, `Face`, `MacroEdge` are all `@dataclass(frozen=True)` — immutable value objects. This makes them safe to share, hash, and reason about.

### Topology-first

Vertices have **optional** positions (`x`, `y` may be `None`). All topology operations (adjacency, ring BFS, stitching) work on the graph structure alone. Positions are only needed for embedding, rendering, and geometry-based algorithms.

---

## Grid Types

### Pure hex grid (`build_pure_hex_grid`)

A hexagonal-shaped grid of hexagons, built from axial coordinates. Parameter: `rings` (0 = single hex, 1 = 7 hexes, etc.). Face count: `1 + 3·R·(R+1)`. Has 6 macro-edges.

### Pentagon-centred Goldberg grid (`build_pentagon_centered_grid`)

One central pentagon surrounded by rings of hexagons, forming a pentagonal shape. Built via:

1. **Triangulation** — construct a triangular grid on a cone with 5-fold symmetry.
2. **Dualisation** — each triangle becomes a dual vertex; each interior triangulation vertex becomes a dual face.
3. **Tutte embedding** — pin boundary to a regular pentagon, solve Laplacian for interior positions.
4. **Optimisation** — `scipy.optimize.least_squares` minimising edge-length variance, angle deviation, and area-inversion penalties.
5. **Winding fix** — ensure all faces have CCW vertex ordering.

Has 5 macro-edges. Face count: `1 + 5·R·(R+1)/2`.

---

## Composition

### `CompositeGrid` & `StitchSpec`

Multiple `PolyGrid`s are composed via stitching — merging boundary vertices along matching macro-edges. The process:

1. **Prefix** all ids to avoid collisions.
2. **Merge** boundary vertex pairs into canonical vertices.
3. **Deduplicate** edges that now reference the same vertex pair.
4. **Rebuild** faces with remapped ids.

### `AssemblyPlan` & `pent_hex_assembly`

An `AssemblyPlan` is a named collection of `PolyGrid` components + stitch specs. The current recipe `pent_hex_assembly(rings)` builds 1 pentagon + 5 hex grids stitched into a unified mesh.

---

## Terrain Generation (Phases 6–7)

### Partitioning

The `regions` module splits grids into named regions using four algorithms:

| Algorithm | Strategy | Use case |
|-----------|----------|----------|
| `partition_angular` | Equal angular sectors | Quick geometric split |
| `partition_flood_fill` | Competitive BFS from seeds | Organic, topology-aware shapes |
| `partition_voronoi` | Nearest seed by centroid distance | Clean, regular boundaries |
| `partition_noise` | Voronoi + distance perturbation | Organic, irregular boundaries |

### Noise & heightmaps

`noise.py` provides simplex noise with octave layering and domain warping. `heightmap.py` bridges noise to grids, computing per-face elevations. `mountains.py` adds ridgeline-based mountain ranges with erosion simulation.

### Rivers

`rivers.py` generates river networks following downhill flow, with watershed detection and confluence points.

### Pipeline

`pipeline.py` provides a composition framework for chaining terrain generation passes.

---

## Globe-Scale Topology (Phase 8)

`globe.py` builds a Goldberg polyhedron globe using the `models` library's `generate_goldberg_tiles()`. Each tile gets per-face terrain via `globe_terrain.py` (which applies noise, mountains, and biome colouring at the global scale). `globe_export.py` serialises the globe as JSON with per-tile colour, elevation, and adjacency.

---

## Sub-Tile Detail (Phase 10)

Each Goldberg tile is expanded into a local hex sub-grid:

1. **`tile_detail.py`** — `TileDetailSpec` + `DetailGridCollection` manage per-tile detail grids.
2. **`detail_terrain.py`** — boundary-aware terrain generation that ensures continuity between adjacent tiles.
3. **`detail_render.py`** — satellite-style texture rendering with biome-aware colour palettes.
4. **`texture_pipeline.py`** — packs individual tile textures into a GPU-ready texture atlas with UV layout.
5. **`detail_perf.py`** — parallel generation, PIL fast-path rendering, caching.

---

## GPU Rendering (Phases 12–13)

> **Note:** `globe_renderer_v2.py` is a **standalone demo/diagnostic
> renderer** — it is NOT part of the library API.
> `playground` has its own full OpenGL pipeline and does not import this
> module.  It exists so pgrid can be developed and tested independently
> with a 3D viewer.  It is gated behind the `[demo]` / `[globe]` extras
> and is only imported by scripts in `scripts/` and by its own test files.

`globe_renderer_v2.py` is the main renderer (~2,300 lines), implementing:

### Phase 12 — Core rendering quality
- **12A Flood-fill** — `flood_fill_tile_texture()` removes black borders.
- **12B Subdivision** — `subdivide_tile_mesh()` subdivides triangle fans and projects onto sphere.
- **12C Batched mesh** — `build_batched_globe_mesh()` merges all tiles into one VBO.

### Phase 13 — Cohesive rendering
- **13A Full-coverage textures** — terrain-coloured backgrounds eliminate seams.
- **13B Atlas gutters** — padding pixels prevent bilinear bleed.
- **13C UV clamping** — `compute_uv_polygon_inset()` + `clamp_uv_to_polygon()` prevent UV overshoot.
- **13D Colour harmonisation** — `harmonise_tile_colours()` blends boundary vertex colours toward neighbour averages.
- **13E Normal-mapped PBR** — tangent-space normal maps, `encode_normal_to_rgb()` / `decode_rgb_to_normal()`, PBR fragment shader with Fresnel specular, roughness, and Reinhard tone mapping.
- **13F Adaptive LOD** — `select_lod_level()` picks from `LOD_LEVELS = (1, 2, 3, 5)` per tile. `estimate_tile_screen_fraction()` for view-dependent LOD. `is_tile_backfacing()` for culling. `stitch_lod_boundary()` for crack prevention. `build_lod_batched_globe_mesh()` for the adaptive pipeline.
- **13G Atmosphere** — `build_atmosphere_shell()` (Fresnel limb haze), `build_background_quad()` (radial gradient), 3-pass bloom (extract → Gaussian blur → composite with Reinhard).
- **13H Water** — `classify_water_tiles()`, `compute_water_depth()`, per-vertex `water_flag`. PBR shader with depth-based colour, animated waves (`u_time`), coastline foam via `dFdx`/`dFdy`.

### Shader architecture

| Shader set | Purpose |
|-----------|---------|
| `_V2_VERTEX/FRAGMENT_SHADER` | Legacy v2 (basic lighting) |
| `_PBR_VERTEX/FRAGMENT_SHADER` | Full PBR with normal maps, water, specular |
| `_ATMO_VERTEX/FRAGMENT_SHADER` | Atmosphere shell with Fresnel alpha |
| `_BG_VERTEX/FRAGMENT_SHADER` | Background radial gradient |
| `_BLOOM_EXTRACT/BLUR/COMPOSITE_SHADER` | 3-pass bloom post-processing |

---

## Per-Tile Data

`tile_data.py` provides per-face key-value storage:

| Class | Responsibility |
|-------|---------------|
| `FieldDef` | Typed field definition (name, dtype, default) |
| `TileSchema` | Field set declaration, validates on write |
| `TileData` | Raw data container, JSON-serialisable |
| `TileDataStore` | Binds data to grid; neighbour/ring queries, bulk ops |

---

## Testing

1,101 tests across 36 test files, covering:

- Core topology: model validation, serialisation, adjacency
- Goldberg grids: face counts, vertex degrees, embedding quality
- Composition: stitching, assembly, boundary alignment
- Terrain: noise, heightmaps, mountains, rivers, pipeline
- Globe: globe builder, terrain generation, export, rendering
- Detail: sub-tile grids, boundary continuity, texture atlas
- Renderer v2: subdivision, batching, UV clamping, colour harmonisation, normal maps, water, atmosphere, bloom, LOD
- Partitioning: all 4 algorithms, validation, constraints

Tests run in ~3 min with caching via `conftest.py` (globe grid + detail grid collection cached with `lru_cache`).

---

## Boundary Contract

### What pgrid owns (library API)

| Layer | Modules | Description |
|-------|---------|-------------|
| **Core topology** | `polygrid`, `models`, `algorithms`, `geometry`, `io` | PolyGrid container, vertices/edges/faces, adjacency, serialisation |
| **Building** | `builders`, `goldberg_topology`, `composite`, `assembly` | Grid constructors, Goldberg embedding, stitching |
| **Tile data** | `tile_data` | Per-tile key-value storage (`TileDataStore`) |
| **Transforms** | `transforms`, `regions`, `region_stitch` | Overlays, partitioning, region stitching |
| **Terrain** | `noise`, `heightmap`, `mountains`, `rivers`, `pipeline`, `terrain_patches` | Procedural terrain generation |
| **Globe** | `globe`, `globe_terrain`, `globe_export` | Globe-scale topology, terrain, JSON export |
| **Detail** | `detail_grid`, `tile_detail`, `detail_terrain`, `detail_terrain_3d`, `detail_render`, `detail_perf` | Sub-tile detail grids and terrain |
| **Texture** | `tile_texture`, `uv_texture`, `tile_uv_align`, `texture_export`, `apron_grid`, `apron_texture` | Texture atlas building, UV alignment, polygon-cut, export |
| **Biome** | `biome_scatter`, `biome_render`, `biome_pipeline`, `biome_continuity`, `biome_topology`, `coastline`, `ocean_render`, `render_enhanced` | Biome-aware terrain rendering |
| **Visual QA** | `visual_cohesion`, `diagnostics` | Seam checks, quality gates |

### What pgrid does NOT own

| Concern | Owner |
|---------|-------|
| Goldberg polyhedron geometry (tiles, normals, tangents, mesh) | `models` |
| Application UI, OpenGL pipeline, persistence | `playground` |
| Region painting, world-building UX | `playground` |
| Interactive 3D globe viewer (production) | `playground` |

### Standalone demo/diagnostic modules (NOT library API)

These modules are for pgrid's own development and testing. They are **not**
imported by `playground`.

| Module | Purpose |
|--------|---------|
| `globe_renderer_v2.py` | Globe renderer: subdivision, batching, PBR, water, atmosphere, LOD, bloom |
| `globe_mesh.py` | Builds `ShapeMesh` objects for the 3D renderer |
| `texture_pipeline.py` | Builds textured globe meshes for the 3D renderers |
| `visualize.py` | 2D matplotlib rendering |

These are gated behind optional extras (`[demo]`, `[globe]`, `[render]`)
and are exposed in `__init__.py` via `try/except ImportError` blocks.

### Data contracts with playground

The primary integration point between pgrid and playground is the **globe
export payload**.  See `schemas/globe.schema.json` and `docs/JSON_CONTRACT.md`
for the full specification.

| Artifact | Format | Producer | Consumer |
|----------|--------|----------|----------|
| Globe payload | JSON (`globe_export.py`) | pgrid | playground |
| Texture atlas | PNG (`tile_uv_align.py`, `detail_perf.py`) | pgrid | playground |
| UV layout | JSON `{ tile_id: [u_min, v_min, u_max, v_max] }` | pgrid | playground |
| Normal map atlas | PNG (`render_enhanced.py`) | pgrid | playground |

### Playground's expected pgrid imports

When integration is complete, playground will import from these pgrid modules:

- `polygrid.globe` — `build_globe_grid`, `GlobeGrid`
- `polygrid.tile_data` — `TileDataStore`, `TileSchema`, `FieldDef`
- `polygrid.globe_export` — `export_globe_payload`, `validate_globe_payload`
- `polygrid.tile_detail` — `DetailGridCollection`, `TileDetailSpec`
- `polygrid.detail_terrain` — `generate_all_detail_terrain`
- `polygrid.detail_render` — `BiomeConfig`, `render_detail_texture_enhanced`
- `polygrid.tile_uv_align` — `build_polygon_cut_atlas`
- `polygrid.uv_texture` — `compute_tile_uv_bounds`, `UVTransform`
- `polygrid.mountains` — `generate_mountains`, `MountainConfig`
- `polygrid.noise` — `fbm`, `ridged_noise`
- `polygrid.terrain_patches` — `TERRAIN_PRESETS`, `generate_patched_terrain`
- `polygrid.coastline` — `CoastlineConfig`, `classify_all_tiles`

Playground must **never** import pgrid rendering modules (`globe_renderer_v2`,
`globe_mesh`, `visualize`).

---

## Dependencies

| Dependency | Required for | Install group |
|-----------|-------------|--------------|
| (none) | Core topology | default |
| pytest | Testing | `dev` |
| matplotlib | 2D rendering | `render` |
| numpy, scipy | Embedding & optimisation | `embed` |
| opensimplex | Noise-based terrain & boundaries | `noise` |
| Pillow | Texture rendering | `render` |
| pyglet | Interactive 3D viewer | `globe` |
| models | Goldberg polyhedron geometry | `globe` |
