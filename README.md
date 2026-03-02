# PolyGrid

A **topology-first polygon grid toolkit** for building, composing, and running algorithms on hex and pentagon-centred grids. Designed for procedural terrain generation on Goldberg polyhedra — from abstract topology all the way to GPU-rendered 3D globes with PBR lighting, water, atmosphere, and adaptive LOD.

## What it does

- **Builds** pure hex grids and pentagon-centred Goldberg grids with correct topology and optimised 2D embeddings.
- **Composes** grids into assemblies — stitching multiple grids along shared macro-edges into a single unified mesh.
- **Generates terrain** — noise-based heightmaps, mountain ranges, rivers, biome-aware colouring, boundary-continuous sub-tile detail grids.
- **Renders globes** — GPU-accelerated Goldberg polyhedron rendering with texture atlases, PBR lighting, normal maps, water effects, atmosphere, bloom, and view-dependent adaptive LOD.
- **Runs algorithms** on grids — Voronoi duals, angular partitioning, flood-fill, noise-perturbed boundaries.
- **Visualises** grids and overlays with multi-panel composite renders (exploded, stitched, overlay views).

## Project goal

Procedural terrain generation for a game. Each tile sits on a **Goldberg polyhedron** (a sphere tiled by hexagons and 12 pentagons). PolyGrid handles the 2D topology, algorithms, terrain generation, texture pipeline, and 3D rendering. A separate `models` package provides the Goldberg polyhedron geometry primitives.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design and [`docs/TASKLIST.md`](docs/TASKLIST.md) for the roadmap.

## Quick start

```bash
# Install (editable, with all optional deps)
pip install -e ".[dev,render,embed]"

# Build a pentagon-centred grid
polygrid build-pent --rings 3 --out pent.json --render-out pent.png

# Build a hex grid
polygrid build-hex --rings 3 --out hex.json --render-out hex.png

# Build a pent+hex assembly with visualisation
polygrid assembly --rings 3 --out exports/assembly.png

# Run tests
pytest
```

### Terrain partitioning

Partition an assembly into named regions and render the stitched +
unstitched views with colour-coded regions:

```bash
# 5 Voronoi regions on a rings-2 assembly (default)
python scripts/demo_regions.py

# Customise rings, region count, and output path
python scripts/demo_regions.py --rings 3 --regions 6 --out exports/regions.png
```

Other partitioning algorithms are available — `partition_angular`,
`partition_flood_fill`, and `partition_noise` (organic boundaries).
See [`docs/MODULE_REFERENCE.md`](docs/MODULE_REFERENCE.md) for details.

### Globe terrain with sub-tile detail (Phase 10)

Generate a Goldberg polyhedron globe with high-resolution detail
textures:

```bash
# End-to-end: globe → mountains → detail grids → texture atlas
python scripts/demo_detail_globe.py -f 3 --detail-rings 4 --preset mountain_range

# Fast renderer (PIL, ~5× faster than matplotlib):
python scripts/demo_detail_globe.py -f 3 --detail-rings 4 --fast

# Interactive 3D viewer with detail textures (requires pyglet):
python scripts/demo_detail_globe.py -f 3 --detail-rings 4 --view

# Side-by-side comparison at multiple detail levels:
python scripts/demo_detail_globe.py --compare

# Flat-colour 3D viewer (original mode):
python scripts/view_globe.py -f 3 -p alpine_peaks

# Textured 3D viewer:
python scripts/view_globe.py -f 3 --textured --detail-rings 4
```

The detail pipeline expands each Goldberg tile into a local sub-grid
(61 sub-faces at `detail_rings=4`), generates boundary-continuous
terrain, renders satellite-style textures, and packs them into a
texture atlas for GPU rendering.

### Cohesive 3D globe (Phases 11–13)

The full rendering pipeline produces a seamless, PBR-lit globe:

```bash
# Cohesive globe: terrain → atlas → PBR-lit 3D viewer
python scripts/demo_cohesive_globe.py --view

# 3D terrain with elevation displacement
python scripts/demo_globe_3d.py -f 3 --detail-rings 4 --preset mountain_range
```

The Phase 12–13 rendering pipeline includes:
- **Flood-fill textures** (12A) — terrain-coloured backgrounds eliminate black seams
- **Sphere subdivision** (12B) — smooth curvature instead of flat facets
- **Batched draw** (12C) — single VBO + draw call for entire globe
- **Atlas gutters** (13B) — prevent bilinear bleed at tile boundaries
- **UV inset clamping** (13C) — safety net for UV edge cases
- **Colour harmonisation** (13D) — smooth biome transitions between tiles
- **Normal-mapped PBR lighting** (13E) — tangent-space normal maps with Fresnel specular and Reinhard tone mapping
- **Water rendering** (13H) — depth-based colour (shallow turquoise → deep navy), animated waves, coastline foam via screen-space derivatives
- **Atmosphere** (13G) — Fresnel limb haze shell, radial background gradient, 3-pass bloom post-processing
- **Adaptive LOD** (13F) — per-tile subdivision level based on screen-space size (`LOD_LEVELS = (1, 2, 3, 5)`), backface culling, LOD boundary stitching

## Package layout

```
src/polygrid/
    # ── Core topology ──────────────────────────────────────────
    models.py              # Vertex, Edge, Face, MacroEdge (frozen dataclasses)
    polygrid.py            # PolyGrid container, validation, serialisation
    algorithms.py          # Face adjacency, BFS ring detection
    geometry.py            # Geometry helpers (centroids, angles, areas, boundary)
    io.py                  # JSON load/save

    # ── Building & composition ─────────────────────────────────
    builders.py            # Grid constructors (hex, pentagon-centred)
    goldberg_topology.py   # Goldberg topology, Tutte embedding, optimisation
    composite.py           # Multi-grid stitching (vertex merge, edge dedup)
    assembly.py            # Assembly recipes (pent+hex), grid transforms

    # ── Transforms & partitioning ──────────────────────────────
    transforms.py          # Voronoi dual, partition, overlay model
    regions.py             # Terrain partitioning (Region, RegionMap, algorithms)
    region_stitch.py       # Cross-region boundary stitching
    tile_data.py           # Per-face data layer (schema, store, queries)

    # ── Terrain generation ─────────────────────────────────────
    noise.py               # Noise primitives (simplex, domain warp, octaves)
    heightmap.py           # Grid-noise bridge (per-face elevation)
    mountains.py           # Mountain range generation (ridgeline, erosion)
    rivers.py              # River network generation (flow, watershed)
    terrain_render.py      # Elevation → colour (biome-aware palette)
    terrain_patches.py     # Terrain patch stitching
    pipeline.py            # Pipeline/composition framework

    # ── Globe-scale topology & terrain ─────────────────────────
    globe.py               # Goldberg polyhedron globe builder
    globe_terrain.py       # Globe-scale terrain generation
    globe_export.py        # Globe JSON export with terrain colours
    globe_mesh.py          # 3D mesh bridge (terrain colours → ShapeMesh)

    # ── Sub-tile detail ────────────────────────────────────────
    detail_grid.py         # Detail grid builder
    tile_detail.py         # Sub-tile detail grid infrastructure
    detail_terrain.py      # Boundary-aware detail terrain gen
    detail_terrain_3d.py   # 3D terrain detail with elevation
    detail_render.py       # Satellite-style detail textures
    detail_perf.py         # Parallel gen, fast render, caching
    texture_pipeline.py    # Texture atlas + UV mapping

    # ── GPU rendering ──────────────────────────────────────────
    globe_renderer.py      # OpenGL renderer v1 (flat + textured modes)
    globe_renderer_v2.py   # Phase 12-13 renderer (PBR, LOD, water, atmosphere)
    globe_render.py        # Render helpers
    render_enhanced.py     # Enhanced rendering utilities

    # ── 2D visualisation ───────────────────────────────────────
    visualize.py           # Multi-panel composite visualisation
    render.py              # Deprecated shim (use visualize)
    diagnostics.py         # Per-ring quality diagnostics
    cli.py                 # Command-line interface

docs/
    ARCHITECTURE.md        # Design, layers, separation of concerns
    MODULE_REFERENCE.md    # Per-module reference
    TASKLIST.md            # Comprehensive roadmap
    JSON_CONTRACT.md       # JSON serialisation format

tests/                     # 1,101 tests across 36 test files
scripts/                   # Demo and diagnostic scripts
```

## Architecture

Strict layering with clear separation of concerns:

| Layer | Modules | Responsibility |
|-------|---------|---------------|
| **Core** | `models`, `polygrid`, `algorithms`, `geometry`, `io` | Topology, graph operations, serialisation |
| **Building** | `builders`, `goldberg_topology`, `composite`, `assembly` | Grid construction and composition |
| **Tile Data** | `tile_data` | Per-face key-value storage, schema validation, queries |
| **Transforms** | `transforms`, `regions`, `region_stitch` | Overlays, partitioning, region management |
| **Terrain** | `noise`, `heightmap`, `mountains`, `rivers`, `pipeline`, `terrain_render`, `terrain_patches` | Procedural terrain generation |
| **Globe** | `globe`, `globe_terrain`, `globe_export`, `globe_mesh` | Globe-scale topology & terrain |
| **Detail** | `detail_grid`, `tile_detail`, `detail_terrain`, `detail_terrain_3d`, `detail_render`, `detail_perf` | Sub-tile detail grids & textures |
| **Texture** | `texture_pipeline` | Texture atlas, UV mapping |
| **Rendering** | `globe_renderer`, `globe_renderer_v2`, `globe_render`, `render_enhanced`, `visualize` | GPU rendering, PBR, LOD |

The core layer has **zero rendering dependencies**. All algorithm work operates on abstract graph topology.

## Key concepts

- **`PolyGrid`** — the central container. Vertices, edges, faces, macro-edges. Immutable value-object primitives.
- **`MacroEdge`** — one side of the grid's outer polygon. Used for stitching.
- **`CompositeGrid`** — multiple grids merged by stitching along macro-edges.
- **`AssemblyPlan`** — a recipe: named components + stitch specs. Builds into a `CompositeGrid`.
- **`TileDataStore`** — per-face data (elevation, biome, etc.) bound to a grid, with neighbour/ring queries.
- **`RegionMap`** — partitions a grid into named `Region`s (e.g. continents, oceans) with validation and adjacency queries.
- **`Overlay`** — algorithm output (points, segments, regions) drawn on top of a grid.
- **`BiomeConfig`** — biome palette definition for terrain colouring (water level, colour ramp, etc.).
- **Globe renderer v2** — batched mesh with subdivision, PBR shaders, normal maps, water, atmosphere, bloom, adaptive LOD.

## Dependencies

| Package | Required for | Install group |
|---------|-------------|--------------|
| (none) | Core topology | default |
| pytest | Tests | `dev` |
| matplotlib | 2D rendering | `render` |
| numpy, scipy | Embedding & optimisation | `embed` |
| opensimplex | Noise-based terrain & boundaries | `noise` |
| Pillow | Texture rendering | `render` |
| pyglet | Interactive 3D viewer | `globe` |
| models | Goldberg polyhedron geometry | `globe` |

## Tests

```bash
pytest                     # 1,101 tests, ~3 min
pytest -v                  # verbose output
pytest -k gold             # run only Goldberg tests
pytest tests/test_globe_renderer_v2.py  # run renderer tests only
```

## CLI commands

| Command | Description |
|---------|------------|
| `polygrid validate --in grid.json` | Validate a grid JSON file |
| `polygrid render --in grid.json --out grid.png` | Render grid to PNG |
| `polygrid build-hex --rings N --out hex.json` | Build a pure hex grid |
| `polygrid build-pent --rings N --out pent.json` | Build a pentagon-centred grid |
| `polygrid build-pent-all --max-rings N --dir exports` | Batch-build pent grids |
| `polygrid assembly --rings N --out assembly.png` | Build pent+hex assembly |
