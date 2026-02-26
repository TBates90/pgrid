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

---

## Globe Export Format

The globe export format (`schemas/globe.schema.json`) provides a comprehensive
per-tile payload for consumption by 3D renderers and downstream tools.

### Top-level keys

- `metadata`: frequency, radius, tile count, colour ramp info
- `tiles[]`: per-tile data (see below)
- `adjacency`: edge-list `[[face_id_a, face_id_b], ...]`

### Per-tile fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Polygrid face ID (`"t0"`, `"t1"`, ...) |
| `face_type` | string | `"pent"` or `"hex"` |
| `models_tile_id` | int | Models library tile index |
| `vertices_3d` | `[[x,y,z],...]` | 3D vertex positions |
| `center_3d` | `[x,y,z]` | 3D tile centre |
| `normal_3d` | `[x,y,z]` | Outward unit normal |
| `latitude_deg` | float | Latitude (degrees) |
| `longitude_deg` | float | Longitude (degrees) |
| `elevation` | float | Terrain elevation |
| `color` | `[r,g,b]` | RGB colour (0-1 range) |
| `neighbor_ids` | `[str,...]` | Adjacent tile IDs |

### Usage

```python
from polygrid.globe_export import export_globe_json, validate_globe_payload

export_globe_json(globe_grid, store, "exports/globe.json")

# Validate
import json
payload = json.loads(Path("exports/globe.json").read_text())
errors = validate_globe_payload(payload)
```
