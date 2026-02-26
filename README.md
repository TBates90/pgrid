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
    visualize.py           # Multi-panel composite visualisation
    render.py              # Simple single-grid PNG renderer
    diagnostics.py         # Per-ring quality diagnostics
    cli.py                 # Command-line interface

docs/
    ARCHITECTURE.md        # Design, layers, separation of concerns
    MODULE_REFERENCE.md    # Per-module reference
    TASKLIST.md            # Comprehensive roadmap
    JSON_CONTRACT.md       # JSON serialisation format

tests/                     # 149 tests across 16 files
scripts/                   # Demo and diagnostic scripts
```

## Architecture

Strict layering with clear separation of concerns:

| Layer | Modules | Responsibility |
|-------|---------|---------------|
| **Core** | `models`, `polygrid`, `algorithms`, `geometry`, `io` | Topology, graph operations, serialisation |
| **Building** | `builders`, `goldberg_topology`, `composite`, `assembly` | Grid construction and composition |
| **Transforms** | `transforms` | Algorithms that produce overlay data |
| **Rendering** | `visualize`, `render` | Matplotlib visualisation (optional dep) |

The core layer has **zero rendering dependencies**. All algorithm work operates on abstract graph topology.

## Key concepts

- **`PolyGrid`** — the central container. Vertices, edges, faces, macro-edges. Immutable value-object primitives.
- **`MacroEdge`** — one side of the grid's outer polygon. Used for stitching.
- **`CompositeGrid`** — multiple grids merged by stitching along macro-edges.
- **`AssemblyPlan`** — a recipe: named components + stitch specs. Builds into a `CompositeGrid`.
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
pytest          # 149 tests, ~22s
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
