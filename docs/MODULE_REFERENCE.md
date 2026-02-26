# PolyGrid Module Reference

Quick reference for every module in `src/polygrid/`, what it does, and what it depends on.

---

## Core Layer (zero optional dependencies)

### `models.py` (66 lines)
Frozen dataclasses: `Vertex`, `Edge`, `Face`, `MacroEdge`. Pure value objects with no logic beyond basic validation (`Face.validate_polygon()`).

### `polygrid.py` (366 lines)
`PolyGrid` — the central container. Holds vertex/edge/face dicts, macro-edges. Provides:
- `validate(strict)` — referential integrity checks
- `boundary_edges()`, `boundary_vertex_cycle()`, `compute_macro_edges(n_sides)`
- `compute_face_neighbors()`, `with_neighbors()`
- `to_dict()` / `from_dict()` / `to_json()` / `from_json()` — serialisation
- `_detect_corners()` — private helper for corner detection via boundary turn analysis

**Depends on:** `models`, `algorithms`, `geometry`

### `algorithms.py` (54 lines)
Pure graph algorithms:
- `build_face_adjacency(faces, edges)` → face adjacency map
- `ring_faces(adjacency, start, max_depth)` → BFS ring grouping

**Depends on:** `models`

### `geometry.py` (189 lines)
Geometry helpers (all pure functions):
- `ordered_face_vertices`, `face_vertex_cycle` — vertex ordering
- `interior_angle`, `face_signed_area`, `edge_length` — measurements
- `face_center`, `grid_center` — centroids
- `find_pentagon_face`, `collect_face_vertices` — queries
- `boundary_vertex_cycle` — boundary walking

**Depends on:** `models`

### `io.py` (19 lines)
`load_json(path)`, `save_json(grid, path)` — thin wrappers around `PolyGrid.from_dict` / `to_dict`.

**Depends on:** `polygrid`

---

## Building Layer

### `builders.py` (168 lines)
Grid constructors:
- `build_pure_hex_grid(rings, size)` — hex grid from axial coordinates
- `build_pentagon_centered_grid(rings, size, embed, embed_mode)` — delegates to `goldberg_topology`
- `validate_pentagon_topology(grid, rings)` — structural checks
- `hex_face_count(rings)` — formula: `1 + 3R(R+1)`

**Depends on:** `models`, `polygrid`, `algorithms`, `goldberg_topology`

### `goldberg_topology.py` (618 lines)
The largest module. Implements Goldberg polyhedron face topology:
- `goldberg_topology(rings)` — pure combinatorial graph (no positions)
- `goldberg_embed_tutte(...)` — Tutte embedding with pentagonal boundary
- `goldberg_optimise(...)` — least-squares edge/angle/area optimisation
- `build_goldberg_grid(rings)` — end-to-end builder
- `fix_face_winding(...)` — ensure CCW ordering
- `goldberg_face_count(rings)` — formula: `1 + 5R(R+1)/2`

**Depends on:** `models`, `polygrid`; **optionally** `numpy`, `scipy`

### `composite.py` (234 lines)
Multi-grid stitching:
- `StitchSpec` — which edges to join
- `stitch_grids(grids, stitches)` → `CompositeGrid`
- `join_grids(grids)` — merge without stitching
- `split_composite(composite)` — recover original components

**Depends on:** `models`, `polygrid`

### `assembly.py` (391 lines)
High-level assembly recipes:
- `AssemblyPlan` — named components + stitch specs, with `.build()` and `.exploded(gap)`
- `pent_hex_assembly(rings)` — 1 pent + 5 hex recipe
- `translate_grid`, `rotate_grid`, `scale_grid` — rigid transforms
- `_position_hex_for_stitch` — align source edge to target edge + reflect to outside
- `_snap_hex_hex_boundaries` — average vertex positions on hex-hex boundaries

**Depends on:** `builders`, `composite`, `models`, `polygrid`

---

## Transform Layer

### `transforms.py` (259 lines)
Overlay data model + transform functions:
- `Overlay`, `OverlayPoint`, `OverlaySegment`, `OverlayRegion` — output containers
- `apply_voronoi(grid)` — Voronoi dual (centroids, dual edges, dual cells)
- `apply_partition(grid, n_sections)` — angular sector partitioning

**Depends on:** `geometry`, `models`, `polygrid`

---

## Tile Data Layer

### `tile_data.py`
Per-face key-value storage for terrain generation data:
- `FieldDef` — typed field definition (name, dtype, optional default)
- `TileSchema` — declares which fields exist and their types; validates on write
- `TileData` — raw data container (`{face_id: {key: value}}`); JSON-serialisable
- `TileDataStore` — binds `TileData` to a `PolyGrid`; adds neighbour/ring queries and bulk operations
- `save_tile_data` / `load_tile_data` — file I/O helpers

**Depends on:** `algorithms`, `polygrid`

---

## Rendering Layer (requires matplotlib)

### `render.py` *(deprecated shim)*
Re-exports `render_png` from `visualize.py` with a `DeprecationWarning`. Kept for backwards compatibility — all new code should import from `visualize`.

### `visualize.py`
All rendering lives here:
- `render_png` — single-grid PNG with optional pentagon symmetry axes
- `render_single_panel` — single grid + optional overlay to PNG
- `render_exploded`, `render_stitched` — component/merged views
- `render_stitched_with_overlay`, `render_unstitched_with_overlay` — overlay views
- `render_assembly_panels` — 4-panel PNG output
- Partition colouring with 16-colour palette

**Depends on:** `assembly`, `composite`, `models`, `polygrid`, `transforms`; **requires** `matplotlib`

---

## Entry Points

### `cli.py` (246 lines)
Command-line interface: `validate`, `render`, `build-hex`, `build-pent`, `build-pent-all`, `assembly`.

### `diagnostics.py` (277 lines)
Per-ring quality diagnostics: edge lengths, interior angles, area checks, quality gates. Used by CLI and tests.

### `scripts/` (directory)
Ad-hoc scripts: `demo.py`, `demo_assembly.py`, `angle_diagnostics.py`, `render_validation.py`.

---

## Stats

| Metric | Count |
|--------|-------|
| Source files | 15 |
| Source lines | ~3,600 |
| Test files | 16 |
| Test lines | ~925 |
| Tests | 149 |
