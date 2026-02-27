# PolyGrid

A **topology-first polygon grid toolkit** for building, composing, and running algorithms on hex and pentagon-centred grids. Designed for procedural terrain generation on Goldberg polyhedra.

## What it does

- **Builds** pure hex grids and pentagon-centred Goldberg grids with correct topology and optimised 2D embeddings.
- **Composes** grids into assemblies — stitching multiple grids along shared macro-edges into a single unified mesh.
- **Runs algorithms** on grids — Voronoi duals, angular partitioning, with a transform/overlay architecture that keeps algorithm output separate from grid topology.
- **Visualises** grids and overlays with multi-panel composite renders (exploded, stitched, overlay views).

## Project goal

Procedural terrain generation for a game. Each tile sits on a **Goldberg polyhedron** (a sphere tiled by hexagons and 12 pentagons). PolyGrid handles the 2D topology, algorithms, and data layer. A separate package handles 3D rendering.

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

## Package layout

```
src/polygrid/
    models.py              # Vertex, Edge, Face, MacroEdge (frozen dataclasses)
    polygrid.py            # PolyGrid container, validation, serialisation
    algorithms.py          # Face adjacency, BFS ring detection
    geometry.py            # Geometry helpers (centroids, angles, areas, boundary)
    io.py                  # JSON load/save
    builders.py            # Grid constructors (hex, pentagon-centred)
    goldberg_topology.py   # Goldberg topology, Tutte embedding, optimisation
    composite.py           # Multi-grid stitching (vertex merge, edge dedup)
    assembly.py            # Assembly recipes (pent+hex), grid transforms
    transforms.py          # Voronoi dual, partition, overlay model
    regions.py             # Terrain partitioning (Region, RegionMap, algorithms)
    tile_data.py           # Per-face data layer (schema, store, queries)
    globe.py               # Goldberg polyhedron globe builder
    globe_mesh.py          # 3D mesh bridge (terrain colours → ShapeMesh)
    globe_export.py        # Globe JSON export with terrain colours
    globe_renderer.py      # OpenGL renderer (flat + textured modes)
    tile_detail.py         # Sub-tile detail grid infrastructure (Phase 10A)
    detail_terrain.py      # Boundary-aware detail terrain gen (Phase 10B)
    detail_render.py       # Satellite-style detail textures (Phase 10C)
    texture_pipeline.py    # Texture atlas + UV mapping (Phase 10D)
    detail_perf.py         # Parallel gen, fast render, caching (Phase 10F)
    visualize.py           # Multi-panel composite visualisation
    render.py              # Deprecated shim (use visualize)
    diagnostics.py         # Per-ring quality diagnostics
    cli.py                 # Command-line interface

docs/
    ARCHITECTURE.md        # Design, layers, separation of concerns
    MODULE_REFERENCE.md    # Per-module reference
    TASKLIST.md            # Comprehensive roadmap
    JSON_CONTRACT.md       # JSON serialisation format

tests/                     # 630+ tests
scripts/                   # Demo and diagnostic scripts
```

## Architecture

Strict layering with clear separation of concerns:

| Layer | Modules | Responsibility |
|-------|---------|---------------|
| **Core** | `models`, `polygrid`, `algorithms`, `geometry`, `io` | Topology, graph operations, serialisation |
| **Building** | `builders`, `goldberg_topology`, `composite`, `assembly` | Grid construction and composition |
| **Tile Data** | `tile_data` | Per-face key-value storage, schema validation, queries |
| **Transforms** | `transforms`, `regions` | Overlays, partitioning algorithms, region management |
| **Rendering** | `visualize` | Matplotlib visualisation (optional dep) |

The core layer has **zero rendering dependencies**. All algorithm work operates on abstract graph topology.

## Key concepts

- **`PolyGrid`** — the central container. Vertices, edges, faces, macro-edges. Immutable value-object primitives.
- **`MacroEdge`** — one side of the grid's outer polygon. Used for stitching.
- **`CompositeGrid`** — multiple grids merged by stitching along macro-edges.
- **`AssemblyPlan`** — a recipe: named components + stitch specs. Builds into a `CompositeGrid`.
- **`TileDataStore`** — per-face data (elevation, biome, etc.) bound to a grid, with neighbour/ring queries.
- **`RegionMap`** — partitions a grid into named `Region`s (e.g. continents, oceans) with validation and adjacency queries.
- **`Overlay`** — algorithm output (points, segments, regions) drawn on top of a grid.

## Dependencies

| Package | Required for | Install group |
|---------|-------------|--------------|
| (none) | Core topology | default |
| pytest | Tests | `dev` |
| matplotlib | Rendering | `render` |
| numpy, scipy | Embedding & optimisation | `embed` |

## Tests

```bash
pytest          # 630+ tests, ~3 min
pytest -v       # verbose output
pytest -k gold  # run only Goldberg tests
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
