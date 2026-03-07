# Tile Texture Mapping — Polygon-Cut Textures for the Goldberg Globe

## Overview

This document outlines the approach for generating **polygon-shaped tile
textures** from the stitched polygrid images and mapping them correctly
onto the 3D Goldberg globe.  The goal is to produce textures where the
content visible through the hex/pent UV polygon on the 3D mesh is
**exactly** the terrain data from the stitched polygrid — no wasted
square corners, no orientation mismatches, and seamless continuity
between adjacent tiles.

---

## Current State

### What we have

1. **Stitched polygrid PNGs** — `render_polygrids.py --stitched` produces
   one PNG per globe tile.  Each image contains the tile's detail grid
   **plus all its neighbours** merged into a single PolyGrid.  The view
   is cropped to the centre tile's extent with ~25 % padding, so
   neighbour terrain is visible around the edges.

2. **Globe 3D viewer** — `render_globe_from_tiles.py` packs tile PNGs into
   an atlas with gutter + UV layout and launches an interactive 3D globe.

3. **UV transform infrastructure** — `UVTransform` (Phase 20A) maps
   detail-grid 2D positions to tile UV coordinates via a piecewise-linear
   polygon warp.  This handles the per-tile rotation/scale differences
   between the polygrid's Tutte embedding and the GoldbergTile's
   tangent-plane projection.

4. **Atlas pipeline** — `texture_pipeline.py` packs tiles into a square
   grid with gutter padding, computes `{face_id: (u_min, v_min, u_max, v_max)}`
   UV layout.

### The problem

The current stitched PNGs are **square images** rendered with matplotlib.
When packed into the atlas, the 3D mesh's UV polygon samples a
hex/pent-shaped region from the square tile slot.  This works, but:

- **Orientation mismatch** — the stitched PNG's orientation comes from the
  detail grid's 2D Tutte embedding.  The 3D mesh's UV polygon orientation
  comes from the GoldbergTile's tangent/bitangent projection.  These are
  **different coordinate systems** that don't align automatically.  Each
  tile on the globe has a unique tangent basis, so the rotation between
  polygrid space and UV space varies per tile.

- **Wasted atlas space** — the square tile slot wastes the four corner
  regions that fall outside the UV polygon.  For hexes, ~17 % of pixels
  are outside the polygon.

- **Pentagon wonkiness** — the pentagon detail grid's Tutte embedding
  produces a slightly irregular boundary.  The 3D pentagon UV polygon
  is also irregular (non-uniform `normalize_uvs` scaling).  Matching
  the two requires the per-tile UVTransform warp, not a simple rotation.

---

## Coordinate Systems

Understanding the three coordinate systems is essential:

### 1. Polygrid 2D space (Tutte embedding)

- **Hex grids:** built from axial coordinates.  Individual cells are
  **pointy-topped** (vertex at top, flat edges left/right).  The overall
  grid boundary is a **flat-topped hexagon** (flat edges at top/bottom,
  vertices at left/right).  Corners at angles 0°, ±60°, ±120°, 180°
  from centre.

- **Pentagon grids:** built via cone triangulation + Tutte embedding +
  optimisation.  The boundary is a roughly regular pentagon with one
  vertex near the top (~95°).  Cells near the centre are pentagons;
  outer cells are hexagons.

### 2. GoldbergTile UV space (tangent-plane projection)

Each tile on the 3D globe has a local tangent/bitangent basis derived
from the first polygon edge.  The 3D polygon vertices are projected
onto this basis, then `normalize_uvs()` maps to [0, 1] × [0, 1] by
independently scaling X and Y.

- **Hex UV polygons:** roughly flat-topped, but the independent X/Y
  scaling makes them slightly irregular.  The angular spacing between
  vertices varies from ~48° to ~72° (vs the ideal 60°).  **Every tile
  has a different UV polygon shape** because each tile's tangent basis
  is different.

- **Pentagon UV polygons:** bottom edge flat (vertices at y ≈ 0), top
  vertex at y ≈ 1.  Similar independent X/Y scaling irregularity.

### 3. Atlas pixel space

Tiles are packed into a square grid of slots.  Each slot is
`tile_size + 2 * gutter` pixels.  UV coordinates map into the inner
`tile_size × tile_size` region.  The gutter is filled by clamping edge
pixels outward to prevent bilinear sampling artefacts.

---

## Proposed Approach

### Step 1: Polygon-mask the stitched PNG

Instead of outputting a square PNG, clip the rendered image to the
**polygon shape** (hex or pent) of the centre tile.  Pixels outside the
polygon become transparent (alpha = 0).

**Implementation:** After rendering the stitched grid, compute the
centre tile's polygon boundary in pixel coordinates (from the
`find_polygon_corners()` result, mapped through the matplotlib axis
transform).  Apply a polygon mask using PIL's `ImageDraw.polygon()`.

This gives us a hex/pent-shaped RGBA image with transparent corners.

### Step 2: Rotate to match the UV polygon

The stitched PNG's polygon is oriented according to the Tutte embedding.
The atlas slot expects content oriented to match the GoldbergTile's UV
polygon.  We need to rotate (and possibly scale) the polygon-masked
image so that its corners align with the UV polygon's corners.

**Per-tile approach:** For each tile:

1. Compute the polygrid boundary corners (from `find_polygon_corners()`).
2. Compute the GoldbergTile UV polygon corners (from `tile.uv_vertices`).
3. Fit a similarity transform (rotation + uniform scale) mapping the
   polygrid corners to the UV corners.
4. Apply that transform to the masked PNG (affine warp via PIL or scipy).

The existing `UVTransform` class already solves this matching problem
with a piecewise-linear warp.  For the image-space approach, a simpler
global similarity transform (best-fit rotation + scale) should suffice
because we're mapping polygon → polygon, not individual sub-faces.

**However:** because `normalize_uvs()` independently scales X and Y,
the UV polygon is slightly stretched.  A pure similarity transform
(uniform scale) won't be a perfect fit.  Options:

- **Affine transform** (non-uniform scale + rotation + translation):
  handles the stretch, but distorts the terrain slightly.
- **Piecewise warp** (UVTransform approach): zero distortion at corners,
  smooth interpolation inside.  More complex but already implemented.
- **Accept small error**: use similarity transform and let the slight
  stretch (~5-10%) blend away visually.

**Recommendation:** Start with a global affine transform (6 parameters:
2×2 matrix + translation).  If visible distortion is a problem, fall
back to the piecewise UVTransform warp.

### Step 3: Fill the atlas slot with the oriented polygon

Paste the rotated polygon-masked image into the atlas slot.  The
transparent corners land in the gutter/background region.  Fill the
gutter by clamping edge pixels outward (existing `_fill_gutter()`
approach), but operating on the polygon edge rather than the square
edge.

**Better gutter fill:** Because the stitched PNG includes neighbour
terrain beyond the centre tile's polygon, we can render the **full
stitched image** (not just the polygon-clipped version) into the atlas
slot.  The UV polygon on the 3D mesh will only sample the hex/pent
interior, and the surrounding neighbour terrain acts as natural gutter
content.  This gives perfect bilinear sampling at polygon edges —
adjacent terrain data instead of clamped edge pixels.

### Step 4: Pentagon handling

Pentagons need special treatment:

- **Boundary detection:** `find_polygon_corners(grid, 5)` identifies the
  5 corners.  The Tutte embedding places them at roughly regular angles,
  but with some irregularity from the optimisation step.

- **Orientation matching:** The pentagon UV polygon has its bottom edge
  flat (two vertices at y ≈ 0).  The Tutte embedding has one vertex at
  the top (~95° from centre).  The affine transform must rotate the
  grid to match the UV orientation — this is the same corner-matching
  procedure as for hexes, just with 5 corners instead of 6.

- **Grid regularity:** The pentagon grid's cells are less regular than
  hex cells (mixed pentagon + hex faces, varying sizes).  This is
  inherent to the Goldberg construction and doesn't affect the
  texture mapping — it only matters for the visual appearance of the
  sub-face grid pattern.

---

## Should Individual Cells Be Flat-Topped?

The user raised the question of making child polygons (hex cells within
the hex-shaped grid) **flat-topped** instead of the current
**pointy-topped** orientation.

### Current state

- Individual cells: **pointy-topped** (vertex at top, flat edges
  left/right).  This comes from `_hex_corners()` using
  `angle = 60° * i - 30°` — the first corner is at -30° (bottom-right).

- Grid boundary: **flat-topped** (flat edges at top/bottom).  This
  emerges naturally from the axial coordinate layout with pointy-topped
  cells.

### Impact of switching to flat-topped cells

If we rotated each cell by 30° to make them flat-topped:

- **Grid boundary becomes pointy-topped** — the outer shape would have
  vertices at the top and bottom, with flat edges at the sides.  This
  would **no longer match** the GoldbergTile's UV polygon, which is
  flat-topped.  We'd need an extra 30° rotation in the UV alignment
  transform to compensate.

- **Pentagon grid interaction** — the pentagon grid is built via Goldberg
  topology (triangulation + dualisation + Tutte embedding), not axial
  hex construction.  The cell orientations in the pentagon grid are
  determined by the topology, not by a simple angular parameter.
  Changing the hex grid's cell orientation wouldn't automatically change
  the pentagon grid's cells.

- **Stitching compatibility** — the existing stitching infrastructure
  (`_position_hex_for_stitch`, `stitch_grids`) matches macro-edges by
  their vertex positions.  Cell orientation doesn't affect stitching.

- **Visual impact** — flat-topped cells produce a grid pattern where
  cells align in horizontal rows (like a brick wall).  Pointy-topped
  cells produce offset columns.  The visual difference is subtle and
  mostly aesthetic.

### Recommendation

**Keep pointy-topped cells.**  The grid boundary orientation (flat-topped)
already matches the UV polygon orientation.  Switching to flat-topped
cells would misalign the boundary and require compensating transforms.
The cell orientation is invisible once terrain colouring and hillshade
are applied — it only matters when viewing the grid with edge outlines.

If flat-topped cells are desired for aesthetic reasons (e.g., for a
visible grid overlay), it should be an **optional parameter** on the
grid builder, not a global change.

---

## Orientation Robustness

The per-tile UV polygon orientation varies across the globe because each
tile's tangent basis depends on its 3D position and the direction of its
first polygon edge (`derive_basis_from_vertices`).  The pipeline must
handle this correctly for all 92 tiles (freq=3) or 642 tiles (freq=5).

### Robustness checklist

1. **Corner ordering** — `find_polygon_corners()` returns corners sorted
   by descending angle (CCW from largest atan2).  The UV polygon vertices
   from `GoldbergTile.uv_vertices` are ordered by the 3D polygon's winding.
   The `UVTransform` constructor matches them by angular offset, trying all
   N rotational alignments to find the best fit.  ✅ Already handles
   arbitrary rotations.

2. **Pentagon vs hex detection** — the grid builder tags faces as `"pent"`
   or `"hex"`.  `build_detail_grid()` dispatches to the correct builder.
   The number of macro-edges (5 vs 6) follows automatically.  ✅

3. **Reflection** — `_position_hex_for_stitch()` includes a reflection
   check (dot product with outward normal) to ensure the source grid is
   on the correct side.  The stitched result has consistent winding.  ✅

4. **Irregular UV polygons** — `normalize_uvs()` independently scales X
   and Y, producing slightly non-regular hexagons.  The angular spacing
   varies 48°–72°.  A simple rotation won't match these — need at least
   an affine transform.  🔶 Needs implementation.

5. **Edge alignment continuity** — adjacent tiles on the globe share a
   polygon edge.  In the stitched PNG, the centre tile and its neighbour
   share terrain along that edge.  In the atlas, the UV polygons of
   adjacent tiles must sample matching content along the shared edge.
   The UVTransform + gutter fill ensures this, but it requires that both
   tiles' textures have been generated from the same stitched terrain
   data.  ✅ Guaranteed by `build_tile_with_neighbours()`.

---

## Pipeline Summary

```
┌─────────────────────────────────────────────────────┐
│  build_globe_grid(freq) + terrain + detail grids    │
│  build_tile_with_neighbours() for each tile         │
└────────────────┬────────────────────────────────────┘
                 │
    ┌────────────▼────────────────┐
    │  Render stitched polygrid   │
    │  (matplotlib, full extent)  │
    └────────────┬────────────────┘
                 │
    ┌────────────▼────────────────────────────────────┐
    │  For each tile:                                  │
    │  1. Find polygrid boundary corners               │
    │  2. Find GoldbergTile UV polygon corners         │
    │  3. Fit affine/UVTransform mapping               │
    │  4. Warp image to align polygon → UV polygon     │
    │  5. Optionally mask to polygon shape (or keep    │
    │     full stitched extent as natural gutter)       │
    └────────────┬────────────────────────────────────┘
                 │
    ┌────────────▼────────────────┐
    │  Pack into atlas with       │
    │  gutter + UV layout         │
    └────────────┬────────────────┘
                 │
    ┌────────────▼────────────────┐
    │  3D globe renderer          │
    │  (render_globe_v2)          │
    └─────────────────────────────┘
```

---

## Open Questions

1. **Render resolution** — the stitched PNGs are currently 256×256.  For
   polygon masking + rotation, we may want to render at 2× and downsample
   to avoid aliasing at the polygon edges.

2. **Gutter strategy** — should we (a) polygon-mask and then fill gutter
   from edge pixels, or (b) keep the full stitched image in the slot and
   let the neighbour terrain act as natural gutter?  Option (b) is simpler
   and gives better bilinear sampling.

3. **Performance** — fitting an affine/UVTransform per tile adds overhead.
   For 92 tiles at freq=3 this is negligible.  For freq=5 (642 tiles) it
   may be worth caching the transforms.

4. **Pentagon validation** — the 12 pentagon tiles need visual validation
   that the orientation mapping works correctly.  Their UV polygons are
   more irregular than hexes.

5. **Interaction with Phase 16/20** — Phases 16 (soft blending) and 20
   (UV-aligned rendering) have partial overlap with this approach.  The
   polygon-cut stitched texture supersedes both: Phase 16's soft blending
   is unnecessary when the stitched PNG already contains neighbour terrain,
   and Phase 20's UV-aligned rasteriser can be replaced by the image-space
   affine warp.
