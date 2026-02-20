# PolyGrid (Python)

A topology-first grid data model designed to support pentagon‑centered grids, pure hex grids, and composite grids. This scaffold focuses on **clean data contracts**, **JSON import/export**, and **topology-based algorithms** so future polygon types and join/unjoin workflows remain straightforward.

## What’s included
- Lightweight `PolyGrid` data model (faces, edges, vertices)
- JSON import/export with versioned contract
- Topology-based algorithms (face adjacency)
- Composite grid join/unjoin (data-layer merge with id prefixes)
- Minimal CLI for validation and round-trip checks
- Tests for serialization and adjacency

## Quick start
1. Create or load a JSON grid file.
2. Validate and round-trip it using the CLI.

## JSON contract overview
See `schemas/polygrid.schema.json` for the formal contract. At minimum:
- **vertices**: list of vertex objects (`id`, optional `position`)
- **edges**: list of edge objects (`id`, `vertices`, optional `faces`)
- **faces**: list of face objects (`id`, `type`, `vertices`, optional `edges`, optional `neighbors`)
- **face_neighbors**: optional adjacency map (face id → list of adjacent face ids)

## CLI usage
- Validate a grid and re-save it:
  - `polygrid validate --in grid.json --out roundtrip.json`
- Render a grid to PNG (requires matplotlib):
  - `polygrid render --in grid.json --out grid.png`
- Build a pure hex grid (topology + positions):
  - `polygrid build-hex --rings 2 --out hex.json`
  - `polygrid build-hex --rings 2 --out hex.json --render-out hex.png`
- Build a pentagon-centered grid (triangulation + dual + Tutte embed):
  - `polygrid build-pent --rings 2 --out pent.json`
  - `polygrid build-pent --rings 2 --out pent.json --render-out pent.png`
  - `polygrid build-pent --rings 2 --out pent.json --embed tutte`
  - `polygrid build-pent --rings 2 --out pent.json --embed none`

## Project layout
```
src/polygrid/
  models.py       # core dataclasses
  polygrid.py     # PolyGrid container + validation
  algorithms.py   # topology-based algorithms
  io.py           # JSON load/save helpers
  cli.py          # minimal CLI
schemas/
  polygrid.schema.json
examples/
  minimal_grid.json
```

## Next steps
- Implement pentagon-centered and pure hex grid generators
- Add embedder strategies for geometry relaxation
- Expand algorithms library (ring extraction, region selection, shortest path)
