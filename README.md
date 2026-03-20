# PolyGrid

A **topology-first polygon grid toolkit** for building, composing, and running algorithms on hex and pentagon-centred grids. Designed for procedural terrain generation on Goldberg polyhedra — from abstract topology all the way to GPU-rendered 3D globes with PBR lighting, water, atmosphere, and adaptive LOD.

## What it does

- **Builds** pure hex grids and pentagon-centred Goldberg grids with correct topology and optimised 2D embeddings.
- **Composes** grids into assemblies — stitching multiple grids along shared macro-edges into a single unified mesh.
- **Generates terrain** — noise-based heightmaps, mountain ranges, rivers, biome-aware colouring, boundary-continuous sub-tile detail grids.
- **Renders globes** — GPU-accelerated Goldberg polyhedron rendering with polygon-cut texture atlases, PBR lighting, normal maps, water effects, atmosphere, bloom, and view-dependent adaptive LOD.
- **Runs algorithms** on grids — Voronoi duals, angular partitioning, flood-fill, noise-perturbed boundaries.
- **Visualises** grids and overlays with multi-panel composite renders (exploded, stitched, overlay views).

## Project goal

Procedural terrain generation for a game. Each tile sits on a **Goldberg polyhedron** (a sphere tiled by hexagons and 12 pentagons). PolyGrid handles the 2D topology, algorithms, terrain generation, texture pipeline, and 3D rendering. A separate `models` package provides the Goldberg polyhedron geometry primitives.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design and [`docs/TASKLIST.md`](docs/TASKLIST.md) for the roadmap.

## Quick start

```bash
# Install (editable, with all optional deps)
pip install -e ".[dev,render,embed]"

# Run tests
pytest
```

## Generating globe textures

The texture pipeline uses **polygon-cut** rendering: each Goldberg tile
is rendered as a stitched 2D polygrid (with its neighbours for seamless
boundaries), then warped into the tile's UV polygon space and packed
into a texture atlas. This atlas is consumed directly by the 3D globe
renderer.

### Step 1 — Generate tile textures

```bash
python scripts/render_polygrids.py -f 3 --seed 42 -o exports/f3
```
    
This produces an output directory containing:
- `atlas.png` — the packed texture atlas
- `uv_layout.json` — per-tile UV coordinates in the atlas
- `globe_payload.json` — globe geometry for the 3D viewer
- `metadata.json` — frequency, seed, preset (consumed by Step 2)
- `t0.png`, `t1.png`, … — individual stitched tile renders
- `warped/` — individual UV-warped tiles (for inspection)

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-f`, `--frequency` | `3` | Goldberg polyhedron frequency (number of tiles scales as `10f² + 2`) |
| `--detail-rings` | `4` | Sub-tile detail grid ring count (higher = finer terrain) |
| `--preset` | `mountain_range` | Terrain preset: `mountain_range`, `alpine_peaks`, `rolling_hills` |
| `--seed` | `42` | Random seed for reproducible terrain |
| `--tile-size` | `512` | Tile image size in pixels |
| `-o`, `--output-dir` | `exports/polygrids/` | Output directory |
| `--renderer` | `analytical` | Tile renderer backend: `analytical` (deterministic point-in-polygon fill, no anti-aliasing) or `matplotlib` (patch rasterisation with anti-aliasing) |
| `--no-polygon-cut` | off | Disable polygon-cut atlas (plain tile PNGs only) |

#### Debug flags

| Flag | Description |
|------|-------------|
| `--debug-labels` | Draw tile-ID and per-edge neighbour labels on each tile |
| `--polygon-mask` | Black-fill pixels outside the UV polygon (visualise polygon boundary) |
| `--edges` | Show grid edges overlaid on terrain colouring |
| `--outline-tiles` | Draw thin outlines on every sub-face within each polygrid (synonym for `--edges`) |
| `--colour-debug` | Skip terrain generation; colour each polygrid tile with a unique hue and a centre→edge gradient. Useful for inspecting stitching topology without terrain noise. No `--seed` needed. |

Example with debug overlays:

```bash
python scripts/render_polygrids.py --debug-labels --polygon-mask \
    -f 3 --seed 142 --tile-size 256 -o exports/debug_f3
```

Example colour-debug (no terrain, just topology):

```bash
python scripts/render_polygrids.py --colour-debug --outline-tiles \
    -f 3 --detail-rings 4 -o exports/colour_debug
```

### Debug pipeline visualiser

For diagnosing rendering issues, the debug pipeline script traces the
full rendering pipeline step-by-step and produces annotated diagnostic
images at each stage:

```bash
python scripts/debug_pipeline.py -f 3 --detail-rings 4

# Debug specific tiles only:
python scripts/debug_pipeline.py --tiles t0 t5 t11
```

This produces images for each stage:

| Stage | What it shows |
|-------|---------------|
| **1. Globe topology** | All faces on the polyhedron with IDs, pentagon markers flameshot gui|
| **2. Detail grid** | Tutte embedding with boundary, macro-edge segments, detected corners with angles |
| **3. Stitched composite** | Centre tile + neighbour aprons with the view-limits box overlaid |
| **4. Corner matching** | Grid corners (macro-edge order) ↔ UV corners showing angular alignment |
| **5. Sector equalisation** | Before/after grid corner adjustment with per-sector angle annotations |
| **6. Warp sectors** | Triangle-fan decomposition in source + dest space with anisotropy values |
| **7. Warped tile** | Final warped slot image with UV polygon boundary + gutter overlay |

Output goes to `exports/debug_pipeline/` by default, with a subfolder
per tile.

### Step 2 — View the 3D globe

```bash
python scripts/render_globe_from_tiles.py exports/f3 --v2
```

```bash
python scripts/render_globe_from_tiles.py exports/colour_debug --v2
```

This loads the pre-generated atlas and metadata from the export
directory and launches an interactive 3D viewer with PBR lighting,
water, atmosphere, and bloom. No need to re-specify frequency or seed —
they are read automatically from `metadata.json`.

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--v2` | off | Use the v2 renderer (recommended — better sphere projection) |
| `--subdivisions` | `3` | Triangle subdivision level (higher = smoother sphere) |
| `--width` / `--height` | `900` / `700` | Window dimensions |
| `--no-view` | off | Build atlas/payload only, don't launch the viewer |
| `--no-polygon-cut` | off | Pack a fresh atlas from raw tile PNGs instead of using the pre-built one |
| `-f`, `--frequency` | from metadata | Override the Goldberg frequency |
| `--seed` | from metadata | Override the random seed |
| `--preset` | from metadata | Override the terrain preset |
| `--payload` | — | Path to an existing `globe_payload.json` (skip globe generation) |

### Topology-only quick start

For working with grids outside of the globe pipeline:

```bash
# Build a pentagon-centred grid
polygrid build-pent --rings 3 --out pent.json --render-out pent.png

# Build a hex grid
polygrid build-hex --rings 3 --out hex.json --render-out hex.png

# Build a pent+hex assembly with visualisation
polygrid assembly --rings 3 --out exports/assembly.png
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
    atlas_utils.py         # Shared atlas helpers (fill_gutter, layout calc)
    texture_pipeline.py    # Texture atlas + UV mapping
    uv_texture.py          # GoldbergTile UV extraction
    tile_uv_align.py       # Polygon-cut warp, atlas builder

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
| **Texture** | `texture_pipeline`, `uv_texture`, `tile_uv_align` | Polygon-cut warp, texture atlas, UV mapping |
| **Rendering** | `globe_renderer`, `globe_renderer_v2`, `globe_render`, `render_enhanced`, `visualize` | GPU rendering, PBR, LOD |

The core layer has **zero rendering dependencies**. All algorithm work operates on abstract graph topology.

## Key concepts

- **Goldberg polyhedron** — a sphere tiled by hexagons and exactly 12 pentagons. Created from an icosahedron by subdivision at a given *frequency*. Total tile count = `10f² + 2`.
- **Polygrid** — a 2D detail grid for a single Goldberg tile. **Pentgrids** (5-sided) are used for pentagon tiles and **hexgrids** (6-sided) for hexagon tiles. Each consists of a central hexagon or pentagon surrounded by rings of hexagons, as close to regular as possible.
- **Stitching / Aprons** — polygrids can be stitched together. This is primarily used to add *apron* visual data around each polygrid so that when it is rendered onto a Goldberg tile, the jagged hex outline of the polygrid connects seamlessly to adjacent tiles — like puzzle pieces slotting together.
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
