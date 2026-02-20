# PolyGrid JSON Contract

This document describes the lightweight JSON contract used to export/import `PolyGrid` objects.

## Goals
- Minimal but complete topology representation (faces, edges, vertices).
- Optional geometry (vertex positions).
- Explicit neighbor relations for algorithms that require adjacency.
- Stable, versioned format for long-term compatibility.

## Required fields
- `version`: string
- `vertices`: list of `{ id, position? }`
- `edges`: list of `{ id, vertices, faces? }`
- `faces`: list of `{ id, type, vertices, edges?, neighbors? }`

## Optional fields
- `metadata`: arbitrary object
- `face_neighbors`: map of face id → list of adjacent face ids

## Notes
- `type` is typically `pent` or `hex`, but other values are allowed for transitional data.
- `neighbors` may be omitted; it will be recomputed from edges when loading.
