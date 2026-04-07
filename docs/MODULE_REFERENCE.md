# PolyGrid Module Reference

This document is a concise index of module entry points in `src/polygrid/`.

Scope:
- Module lookup, package ownership, import targets.
- Flat-path compatibility shims still present for transition.

Out of scope:
- System design and layer rationale: see [ARCHITECTURE.md](ARCHITECTURE.md).
- Rendering algorithm details: see [RENDERING_PIPELINE.md](RENDERING_PIPELINE.md) and [TILE_TEXTURE_MAPPING.md](TILE_TEXTURE_MAPPING.md).
- JSON schemas/contracts: see [JSON_CONTRACT.md](JSON_CONTRACT.md).

---

## Canonical Package Entry Points

Use sub-package imports for new code.

| Package | Purpose | Common modules |
|---------|---------|----------------|
| `polygrid.core` | Core graph/topology model | `models`, `polygrid`, `algorithms`, `geometry` |
| `polygrid.building` | Grid builders and composition | `builders`, `goldberg_topology`, `composite`, `assembly` |
| `polygrid.data` | Data binding and transforms | `tile_data`, `transforms` |
| `polygrid.terrain` | Terrain classification and generation | `classification`, `features`, `noise`, `heightmap`, `mountains`, `regions`, `moisture`, `temperature` |
| `polygrid.globe` | Globe generation and export | `globe`, `globe_export` |
| `polygrid.detail` | Per-tile detail topology, terrain, rendering | `detail_grid`, `tile_detail`, `detail_terrain`, `detail_render`, `column` |
| `polygrid.rendering` | Atlas/UV mapping and rendering helpers | `atlas_utils`, `uv_texture`, `tile_uv_align`, `globe_renderer_v2`, `detail_centers`, `detail_cell_contract`, `seam_strips`, `detail_topology` |
| `polygrid.debug` | Internal diagnostics and experiments | `nebula_core` |

---

## Rendering-Focused Symbols

These are the primary symbols used by the polygon-cut mapping pipeline.

| Module | Key symbols |
|--------|-------------|
| `polygrid.rendering.tile_uv_align` | `build_polygon_cut_atlas`, `_compute_pentagon_grid_scale`, `_smooth_pentagon_corners`, `_equalise_sector_ratios` |
| `polygrid.rendering.uv_texture` | `UVTransform`, `compute_detail_to_uv_transform`, `compute_tile_uv_bounds` |
| `polygrid.rendering.atlas_utils` | `fill_gutter`, atlas-slot helpers |
| `polygrid.detail.tile_detail` | `build_all_detail_grids`, `build_tile_with_neighbours`, `find_polygon_corners` |

---

## Top-Level Compatibility Shims

The package still exposes legacy flat paths such as `polygrid.tile_uv_align` and `polygrid.uv_texture`.

- Keep using shims only for backward compatibility.
- Prefer canonical imports from sub-packages in all new code and docs.
- When updating older scripts, migrate flat imports to sub-package imports first.

---

## Quick Import Map

```python
from polygrid.globe.globe import build_globe_grid
from polygrid.data.tile_data import TileDataStore
from polygrid.detail.tile_detail import build_all_detail_grids
from polygrid.rendering.tile_uv_align import build_polygon_cut_atlas
from polygrid.rendering.uv_texture import UVTransform
```

---

## Related Docs

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [RENDERING_PIPELINE.md](RENDERING_PIPELINE.md)
- [TILE_TEXTURE_MAPPING.md](TILE_TEXTURE_MAPPING.md)
- [JSON_CONTRACT.md](JSON_CONTRACT.md)
