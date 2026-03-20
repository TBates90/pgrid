# Code Cleanup Review

### Cleanup Review Prompt

Review the pgrid codebase. Tag code using the following tags:

 # TODO REFACTOR
 # TODO REMOVE
 # TODO REVIEW
 # TODO OPTIMISE

Review and update the CLEANUP_REVIEW.md document.

### Key Imports

| Module | Used by |
|--------|---------|
| `polygrid.globe` | render_polygrids, render_globe_from_tiles, debug_pipeline |
| `polygrid.mountains` | render_polygrids, render_globe_from_tiles |
| `polygrid.tile_data` | render_polygrids, render_globe_from_tiles |
| `polygrid.tile_detail` | render_polygrids, debug_pipeline |
| `polygrid.detail_terrain` | render_polygrids |
| `polygrid.detail_render` | render_polygrids, debug_pipeline |
| `polygrid.geometry` | render_polygrids, debug_pipeline |
| `polygrid.tile_uv_align` | render_polygrids, debug_pipeline |
| `polygrid.atlas_utils` | render_globe_from_tiles |
| `polygrid.globe_export` | render_polygrids, render_globe_from_tiles |
| `polygrid.globe_renderer_v2` | render_globe_from_tiles |
| `polygrid.uv_texture` | debug_pipeline |

| Module | Required by |
|--------|-------------|
| `polygrid.models` | core data types — used everywhere |
| `polygrid.polygrid` | core container — used everywhere |
| `polygrid.algorithms` | geometry, tile_data, detail_render, detail_terrain, heightmap |
| `polygrid.noise` | detail_render, detail_terrain, tile_detail, mountains |
| `polygrid.heightmap` | mountains, detail_terrain, tile_detail |
| `polygrid.detail_grid` | tile_detail, detail_terrain |
| `polygrid.regions` | mountains (Region type) |
| `polygrid.transforms` | regions (Overlay types) |

---

### 🟡 TODO REFACTOR — In live tree but has issues
| Module | Issue |
|--------|-------|
| `regions.py` | Only `Region` type is used (by `mountains.py`). The full partitioning API (`partition_voronoi`, `partition_noise`, etc.) is not used by any live script. |
| `transforms.py` | Only `Overlay` and `OverlayRegion` types are used (by `regions.py`). The functions `apply_voronoi`, `apply_partition` are unused by live scripts. |
| `__init__.py` | Re-exports ~200 symbols from 30+ modules. Most are from dead modules. Needs major pruning once dead modules are removed. |

### 🟡 TODO REVIEW — Tests for live modules but may overlap / be unnecessary
| Test file | Issue |
|-----------|-------|
| `test_goldberg.py` | Tests standalone `goldberg_topology.py` which may be removed. BUT also tests core goldberg embedding maths that `globe.py` depends on indirectly. Check which tests exercise code actually used by `globe.py`. |

### 🟡 TODO REVIEW - docs
- `docs/ARCHITECTURE.md` — likely needs update to reflect stripped-down codebase
- `docs/MODULE_REFERENCE.md` — will need heavy rewrite
- `docs/JSON_CONTRACT.md` — check if it describes the live payload format
- `docs/CLEANUP.md` — existing cleanup notes, merge with this review
- `test_results.md` — stale test timing data
- `TESTING.md` — testing guide, will need update
- `README.md` — will need update after cleanup

---

## 6. Potential Logic Issues (TODO REVIEW)

### 6.1 `mountains.py` → `regions.py` dependency
`mountains.py` imports `Region` from `regions.py` which in turn imports `transforms.py`. This creates a dependency chain `mountains → regions → transforms → geometry`. The live scripts only use `MountainConfig` and `generate_mountains`. Check if `Region` is actually used inside `generate_mountains` or if it's only used in the `generate_mountains_regional` variant.

### 6.2 `render_polygrids.py` has dead code paths
- `_main_neighbour_edges()` uses `get_neighbour_border_grid` — this is the old Phase 10 neighbour approach, replaced by stitched composites. The `--with-neighbour-edges` flag is labeled "Legacy" in the argparse help.
- `_main_simple()` — simple un-stitched rendering, may still be useful but is not the default path.
- `_render_tile()` — only called by `_main_simple` and `_main_neighbour_edges`, not by the default polygon-cut path.

### 6.3 `__init__.py` catches ImportError silently
The `try/except ImportError: pass` blocks in `__init__.py` mean missing dependencies produce zero feedback. After cleanup, these should be more explicit about what's optional.

### 6.4 `goldberg_topology.py` vs `globe.py`
`goldberg_topology.py` provides standalone goldberg grid building. `globe.py` builds on the `models` library's goldberg generator. It's unclear if `globe.py` uses anything from `goldberg_topology.py` or if they're fully independent implementations. If independent, `goldberg_topology.py` is dead code. If `globe.py` delegates to it, it's live.
