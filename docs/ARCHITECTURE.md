# PolyGrid Architecture

## Project Vision

PolyGrid is a **topology-first toolkit** for building, composing, and running algorithms on polygon grids — primarily **hex grids** and **pentagon-centred Goldberg grids**. The end goal is **procedural terrain generation for a game** where each tile sits on a Goldberg polyhedron (a sphere tiled by hexagons and 12 pentagons).

The workflow is:

1. **Build** individual polygon grids (hex, pentagon-centred Goldberg faces).
2. **Compose** them into assemblies (e.g. one pentagon + five hex grids stitched together).
3. **Run algorithms** on the composed grid — partitioning into terrain zones, assigning biome types, generating heightmaps, placing features.
4. **Store per-tile data** (biome, elevation, moisture, etc.) against individual faces.
5. **Export** per-tile textures / PNGs for a 3D renderer that maps them onto a Goldberg polyhedron.

A separate package handles the 3D Goldberg polyhedron mechanics and rendering. PolyGrid focuses purely on the **2D topology, data, and algorithms** side.

---

## Separation of Concerns

The project is organised around a strict layering:

```
┌─────────────────────────────────────────────────────┐
│                   scripts / CLI                      │  Entry points
├─────────────────────────────────────────────────────┤
│                    visualize.py                      │  Rendering (matplotlib)
│                     render.py                        │
├─────────────────────────────────────────────────────┤
│                   transforms.py                      │  Algorithms (Voronoi,
│                   (future: terrain.py, biomes.py)    │  partition, terrain gen)
├─────────────────────────────────────────────────────┤
│    assembly.py  │  composite.py  │  builders.py      │  Composition & building
│                 │                │  goldberg_topology │
├─────────────────────────────────────────────────────┤
│   polygrid.py  │  models.py  │  algorithms.py       │  Core topology
│   geometry.py  │             │  io.py               │
└─────────────────────────────────────────────────────┘
```

### Layer rules

| Layer | Knows about | Does NOT know about |
|-------|------------|---------------------|
| **Core** (`models`, `polygrid`, `algorithms`, `geometry`, `io`) | Vertices, edges, faces, adjacency | Rendering, assembly, transforms |
| **Building** (`builders`, `goldberg_topology`, `composite`, `assembly`) | Core layer | Rendering, game-specific data |
| **Transforms** (`transforms`, future terrain/biome modules) | Core layer, optionally building layer | Rendering |
| **Rendering** (`visualize`, `render`) | Everything above | Nothing below it depends on rendering |
| **Scripts / CLI** | Everything | — |

**Key principle:** the core topology layer has **zero rendering dependencies**. Matplotlib is an optional install. All algorithm work (adjacency, ring detection, partitioning, future terrain generation) operates on the abstract `PolyGrid` graph, not on pixel positions.

---

## Core Data Model

### `PolyGrid`

The central container. Holds dictionaries of vertices, edges, and faces, plus optional macro-edges (the sides of the grid's outer polygon). Supports:

- **Validation** — referential integrity checks between faces ↔ edges ↔ vertices.
- **Serialisation** — lossless JSON round-trip via `to_dict()` / `from_dict()`.
- **Boundary detection** — identifies boundary edges, walks the boundary cycle, detects corners, computes macro-edges.
- **Adjacency** — `compute_face_neighbors()` builds a face adjacency map from shared edges.

### Frozen dataclasses

`Vertex`, `Edge`, `Face`, `MacroEdge` are all `@dataclass(frozen=True)` — immutable value objects. This makes them safe to share, hash, and reason about. When you need a modified copy, use `dataclasses.replace()`.

### Topology-first

Vertices have **optional** positions (`x`, `y` may be `None`). All topology operations (adjacency, ring BFS, stitching) work on the graph structure alone. Positions are only needed for embedding, rendering, and geometry-based algorithms.

---

## Grid Types

### Pure hex grid (`build_pure_hex_grid`)

A hexagonal-shaped grid of hexagons, built from axial coordinates. Parameter: `rings` (0 = single hex, 1 = 7 hexes, etc.). Face count: `1 + 3·R·(R+1)`. Has 6 macro-edges.

### Pentagon-centred Goldberg grid (`build_pentagon_centered_grid`)

One central pentagon surrounded by rings of hexagons, forming a pentagonal shape. Built via:

1. **Triangulation** — construct a triangular grid on a cone with 5-fold symmetry.
2. **Dualisation** — each triangle becomes a dual vertex; each interior triangulation vertex becomes a dual face (apex → pentagon, rest → hexagons).
3. **Tutte embedding** — pin boundary to a regular pentagon, solve Laplacian for interior positions.
4. **Optimisation** — `scipy.optimize.least_squares` minimising edge-length variance, angle deviation, and area-inversion penalties.
5. **Winding fix** — ensure all faces have CCW vertex ordering.

Has 5 macro-edges. Face count: `1 + 5·R·(R+1)/2`.

---

## Composition

### `CompositeGrid` & `StitchSpec`

Multiple `PolyGrid`s are composed via stitching — merging boundary vertices along matching macro-edges. The process:

1. **Prefix** all ids to avoid collisions.
2. **Merge** boundary vertex pairs (from `StitchSpec`s) into canonical vertices.
3. **Deduplicate** edges that now reference the same vertex pair.
4. **Rebuild** faces with remapped ids.

Result: a single unified `PolyGrid` inside a `CompositeGrid` wrapper that also tracks the original components.

### `AssemblyPlan` & `pent_hex_assembly`

An `AssemblyPlan` is a named collection of `PolyGrid` components + stitch specs. The current recipe `pent_hex_assembly(rings)` builds:

- 1 pentagon-centred grid ("pent")
- 5 hex grids ("hex0" … "hex4"), each positioned flush against a pent macro-edge
- 5 pent↔hex stitches (pent edge i ↔ hex{i} edge 3)
- 5 hex↔hex stitches (hex{i} edge 2 ↔ hex{(i+1)%5} edge 4)

Hex grids are reflected to sit on the **outside** of the pentagon, then hex-hex boundary vertices are **snapped** to averaged positions to close the small angular gap (12° total at each pent corner, ~6° per flanking edge).

---

## Transforms & Overlays

Transforms are functions `PolyGrid → Overlay`. An `Overlay` holds derived geometry (points, segments, regions) that can be drawn on top of a grid without mutating it.

### Current transforms

- **`apply_voronoi`** — Voronoi dual: face centroids as sites, dual edges between adjacent centroids, dual cells around each vertex.
- **`apply_partition`** — Angular partitioning: divides faces into N sectors around the grid centroid, each sector gets a colour index.

### Future transforms (terrain generation)

The transform pattern extends naturally to terrain algorithms — each terrain pass produces an overlay or attaches data to faces. See the tasklist for planned algorithms.

---

## Rendering

Two rendering modules, both requiring `matplotlib` (optional dependency):

- **`render.py`** — simple single-grid PNG output (legacy).
- **`visualize.py`** — multi-panel composite visualisation: exploded views, stitched views, overlay rendering, partition colouring.

Rendering is strictly a **leaf** dependency — nothing in the core or algorithm layers imports from rendering modules.

---

## Per-Tile Data

The **tile data layer** (`tile_data.py`) provides per-face key-value storage, kept strictly separate from grid topology.

### Components

| Class | Responsibility |
|-------|---------------|
| `FieldDef` | Defines a single field: name, type (`int`/`float`/`str`/`bool`), optional default |
| `TileSchema` | Declares the set of fields; validates values on write |
| `TileData` | Raw data container — `{face_id: {key: value}}` with schema enforcement |
| `TileDataStore` | Binds `TileData` to a `PolyGrid`; adds neighbour/ring queries and bulk operations |

### Design principles
- Data lives **alongside** the PolyGrid, not inside it — SoC between topology and game data.
- Data is keyed by face id.
- Schema validates every write — catches type errors early.
- JSON-serialisable (separate from grid JSON; face ids are the join key).
- Neighbour-aware: `get_neighbors_data()`, `get_ring_data()` use the grid's adjacency graph.
- Bulk operations: `apply_to_all()`, `apply_to_ring()`, `apply_to_faces()` for algorithm passes.
- Adjacency is lazily built and cached.

### Intended usage
```python
schema = TileSchema([
    FieldDef("elevation", float, 0.0),
    FieldDef("biome", str, "none"),
])
store = TileDataStore(grid, schema=schema)
store.initialise_all()                        # fill with defaults
store.set("f1", "elevation", 42.0)            # set one face
store.bulk_set(face_ids, "biome", "mountain") # set many
store.apply_to_all("elevation", lambda fid, v: v + noise(fid))
neighbors = store.get_neighbors_data("f1", "elevation")
```

---

## Goldberg Polyhedron Integration (Planned)

The full Goldberg polyhedron has 12 pentagonal faces and a configurable number of hexagonal faces. Each face of the polyhedron is itself a `PolyGrid` (pentagon-centred for pent faces, hex for hex faces).

PolyGrid can represent the **whole-globe topology** as well — just the 12+N faces with their adjacency, no positions needed. This allows running algorithms (continent placement, ocean currents, climate) at the macro scale, then drilling into per-face detail grids.

---

## Testing

201 tests across 17 test files, covering:

- Core model validation and serialisation
- Goldberg topology invariants (face counts, vertex degrees, boundary counts, corner detection)
- Goldberg embedding quality (no crossings, positive areas)
- Macro-edge detection and serialisation
- Stitching correctness (vertex merging, edge dedup)
- Assembly (component count, stitch count, boundary alignment)
- Transforms (Voronoi properties, partition coverage)
- Visualisation (PNG output, overlay rendering)

All tests run in ~22s. No external service dependencies.

---

## Dependencies

| Dependency | Required for | Install group |
|-----------|-------------|--------------|
| (none) | Core topology | default |
| pytest | Testing | `dev` |
| matplotlib | Rendering | `render` |
| numpy, scipy | Embedding & optimisation | `embed` |
