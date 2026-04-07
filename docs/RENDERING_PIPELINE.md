# Rendering Pipeline

This document is the operational overview of the pgrid rendering flow.

Scope:
- End-to-end pipeline from tile generation to atlas-driven globe view.
- Runtime/debug commands and expected outputs.

Related docs:
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [TILE_TEXTURE_MAPPING.md](TILE_TEXTURE_MAPPING.md)
- [JSON_CONTRACT.md](JSON_CONTRACT.md)

Out of scope:
- Deep UV warp math and pentagon-specific compensation internals.
- See [TILE_TEXTURE_MAPPING.md](TILE_TEXTURE_MAPPING.md) for low-level algorithm details.

---

## Pipeline Stages

1. Build Goldberg topology and terrain inputs.
1. Generate per-tile detail grids with neighbour context.
1. Render stitched tile images.
1. Warp tile content into UV polygon-aligned atlas slots.
1. Export atlas, UV layout, payload, and metadata.
1. Load exports in globe viewer.

Core implementation anchors:
- `polygrid.detail.tile_detail.build_tile_with_neighbours`
- `polygrid.rendering.tile_uv_align.build_polygon_cut_atlas`
- `polygrid.rendering.uv_texture.UVTransform`
- `polygrid.rendering.globe_renderer_v2.render_globe_v2`

---

## Generate Exports

```bash
python scripts/render_polygrids.py -f 3 --seed 42 -o exports/f3
```

Expected artifacts:
- `atlas.png`
- `uv_layout.json`
- `globe_payload.json`
- `metadata.json`
- per-tile outputs (`t0.png`, `t1.png`, ...)
- `warped/` inspection outputs

Common options:

| Flag | Default | Description |
|------|---------|-------------|
| `-f`, `--frequency` | `3` | Goldberg frequency; tile count scales as `10f^2 + 2` |
| `--detail-rings` | `4` | Detail-grid ring count |
| `--preset` | `mountain_range` | Terrain preset |
| `--seed` | `42` | Deterministic terrain seed |
| `--tile-size` | `512` | Per-tile atlas slot size |
| `--renderer` | `analytical` | `analytical` or `matplotlib` |
| `--no-polygon-cut` | off | Disable polygon-cut atlas path |

---

## Debugging Modes

Useful flags in `render_polygrids.py`:

| Flag | Use |
|------|-----|
| `--debug-labels` | Draw tile IDs and neighbour-edge labels |
| `--polygon-mask` | Visualize polygon cutoff |
| `--edges` | Overlay detail-grid edges |
| `--outline-tiles` | Alias for edge outlines |
| `--colour-debug` | Topology-only coloration (no terrain) |

Debug pipeline trace:

```bash
python scripts/debug_pipeline.py -f 3 --detail-rings 4
python scripts/debug_pipeline.py --tiles t0 t5 t11
```

The debug export includes staged images for topology, corner matching,
sector equalization, and final warp output.

---

## View Globe

```bash
python scripts/render_globe_from_tiles.py exports/f3
```

The viewer loads `metadata.json`, `atlas.png`, `uv_layout.json`, and payload
from the export directory.

Common viewer options:

| Flag | Default | Description |
|------|---------|-------------|
| `--subdivisions` | `3` | Sphere smoothness |
| `--width` / `--height` | `900` / `700` | Window size |
| `--no-view` | off | Export only; skip interactive viewer |
| `--payload` | unset | Use existing payload JSON |

---

## Ownership Notes

- This doc is the operational guide.
- [TILE_TEXTURE_MAPPING.md](TILE_TEXTURE_MAPPING.md) is the deep technical source of truth for UV alignment details.
- [JSON_CONTRACT.md](JSON_CONTRACT.md) defines export formats consumed by downstream systems.
