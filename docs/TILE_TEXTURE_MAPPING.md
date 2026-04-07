# Tile Texture Mapping (Deep Technical)

This document is the deep technical reference for polygon-cut tile mapping.

Scope:
- Coordinate systems used by the UV pipeline.
- Implemented warp/atlas behavior.
- Pentagon-specific compensation and edge-case handling.

Related docs:
- [RENDERING_PIPELINE.md](RENDERING_PIPELINE.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [JSON_CONTRACT.md](JSON_CONTRACT.md)

Out of scope:
- Day-to-day pipeline commands and user workflow.
- See [RENDERING_PIPELINE.md](RENDERING_PIPELINE.md) for the operational guide.

---

## Current Implementation

The production path is based on per-tile corner alignment and piecewise UV warping.

Primary implementation anchors:
- `polygrid.rendering.tile_uv_align.build_polygon_cut_atlas`
- `polygrid.rendering.uv_texture.UVTransform`
- `polygrid.rendering.uv_texture.compute_detail_to_uv_transform`
- `polygrid.detail.tile_detail.find_polygon_corners`

Key behavior:
1. Detail grids are generated with neighbour context.
1. Tile boundary corners are detected in detail-grid space.
1. Corners are matched to tile UV polygon corners with rotation/reflection handling.
1. A piecewise transform maps source detail coordinates into UV-aligned atlas space.
1. Gutter is filled to stabilize bilinear sampling near polygon boundaries.

This is implemented behavior, not a proposal.

---

## Coordinate Systems

### 1. Detail-grid 2D space

The source terrain and topology are rendered in 2D detail-grid coordinates (Tutte/derived embedding).

### 2. Tile-local UV polygon space

Each Goldberg tile has UV polygon vertices generated from local tangent-plane projection.

### 3. Atlas pixel space

The UV polygon-aligned content is packed into atlas slots with gutter support.

---

## Warp Strategy

`UVTransform` applies piecewise mapping from detail-grid coordinates to UV-aligned coordinates. This avoids relying on a single global affine transform for all tiles and handles per-tile shape/orientation differences.

Corner-matching behavior includes:
- rotational offset search
- winding compatibility checks
- reflection-safe alignment when orientation differs

The result is robust tile-by-tile orientation alignment with shared-edge continuity.

---

## Pentagon Handling

Pentagons are explicitly compensated in the current pipeline.

Implemented compensation hooks include:
- `_compute_pentagon_grid_scale`
- `_smooth_pentagon_corners`
- `_equalise_sector_ratios`

These routines address pentagon-specific geometry drift and sector imbalance that can appear when mapping from detail-grid boundaries to UV polygons.

Operationally this means:
- pentagon corner placement is stabilized before final warp
- local sector ratios are normalized to reduce visible anisotropy artifacts
- edge continuity is preserved while reducing interior shape distortion

---

## Seam Continuity Model

Continuity is achieved by geometry-aware alignment, not a post-render seam-blend pass.

Requirements for continuity:
1. Neighbor-aware detail generation for source content.
1. Consistent corner ordering and orientation matching.
1. UV-aligned warping into atlas slots.
1. Atlas gutter fill to prevent edge sampling bleed.

---

## Orientation and Diagnostics

Orientation behavior varies by tile basis, so per-tile diagnostics matter.

Useful diagnostics:
- `scripts/debug_pipeline.py` stage images for corner matching and sector equalization
- per-tile warped outputs under export debug directories

If orientation issues appear, compare source-corner order against destination UV-corner order first, then inspect pentagon compensation stages.

---

## Cross-Repo Note

Some UV shape behavior is influenced by geometry primitives in the external `models` repository. When documenting or fixing shape normalization behavior that originates there, link the relevant GitHub issue directly in this document.

Tracking policy for unresolved items:
- Use explicit issue links (for example: `github.com/<org>/<repo>/issues/<id>`).
- Do not reference local task files that are not present in this repository.

---

## Related Docs

- [RENDERING_PIPELINE.md](RENDERING_PIPELINE.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [JSON_CONTRACT.md](JSON_CONTRACT.md)
