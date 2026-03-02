# PolyGrid JSON Contract

This document describes the JSON formats used across PolyGrid for serialisation and data exchange.

---

## 1. PolyGrid JSON Format

The lightweight contract for exporting/importing `PolyGrid` objects.

### Goals
- Minimal but complete topology representation (faces, edges, vertices).
- Optional geometry (vertex positions).
- Explicit neighbor relations for algorithms that require adjacency.
- Stable, versioned format for long-term compatibility.

### Required fields
- `version`: string
- `vertices`: list of `{ id, position? }`
- `edges`: list of `{ id, vertices, faces? }`
- `faces`: list of `{ id, type, vertices, edges?, neighbors? }`

### Optional fields
- `metadata`: arbitrary object
- `face_neighbors`: map of face id → list of adjacent face ids

### Notes
- `type` is typically `pent` or `hex`, but other values are allowed for transitional data.
- `neighbors` may be omitted; it will be recomputed from edges when loading.

---

## 2. Globe Export Format

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

---

## 3. Texture Atlas & UV Layout

The texture pipeline (Phase 10D) generates a texture atlas and UV layout
for mapping sub-tile detail textures onto 3D Goldberg tiles.

### Atlas image

A single PNG image arranged as a grid of tile slots:

- Each slot is `tile_size × tile_size` pixels (default 256×256)
- Atlas gutters (Phase 13B): each slot is padded by `gutter` pixels on all sides, filled with the tile's edge colour to prevent bilinear bleed
- Slot arrangement: row-major, `ceil(sqrt(n_tiles))` columns

### UV layout JSON

```json
{
  "t0": [u_min, v_min, u_max, v_max],
  "t1": [u_min, v_min, u_max, v_max],
  ...
}
```

Each entry maps a tile ID to its normalised UV rectangle within the atlas (0–1 range). The renderer uses these to compute per-vertex UV coordinates.

### Normal map atlas

Phase 13E adds a separate normal map atlas with the same layout as the colour atlas. Normal vectors are encoded as RGB: `(nx*0.5+0.5, ny*0.5+0.5, nz*0.5+0.5)`.

---

## 4. Vertex Formats (GPU)

The Phase 12-13 renderer uses several vertex formats depending on enabled features:

| Mode | Stride | Layout |
|------|--------|--------|
| Basic | 8 floats | `pos(3) + col(3) + uv(2)` |
| Basic + water | 9 floats | `pos(3) + col(3) + uv(2) + water(1)` |
| Normal-mapped | 14 floats | `pos(3) + col(3) + uv(2) + T(3) + B(3)` |
| Normal-mapped + water | 15 floats | `pos(3) + col(3) + uv(2) + T(3) + B(3) + water(1)` |
| Atmosphere shell | 7 floats | `pos(3) + rgba(4)` |
| Background quad | 4 floats | `clip_xy(2) + uv(2)` |

Where:
- `T(3)` = tangent vector, `B(3)` = bitangent vector (for TBN normal mapping)
- `water(1)` = per-vertex water flag (1.0 = water, 0.0 = land)
