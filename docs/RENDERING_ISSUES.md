# Globe Rendering Issues — Investigation & Task Tracker

Tracks diagnosed issues in the polygon-cut atlas → 3D globe rendering
pipeline.  Each issue has a **diagnosis**, **investigation notes**,
**tried/rejected approaches**, and **actionable tasks**.

Use `scripts/debug_pipeline.py` to produce per-stage diagnostic images
for any tile (see `README.md § Debug Pipeline Visualiser`).

---

## Table of Contents

1. [Issue 1 — Pentagon corner-to-UV rotation mismatch (stage 4)](#issue-1--pentagon-corner-to-uv-rotation-mismatch-stage-4)
2. [Issue 2 — Pentagon zigzag bias causes uneven UV cut (stage 7)](#issue-2--pentagon-zigzag-bias-causes-uneven-uv-cut-stage-7)
3. [Issue 3 — Hex UV polygon irregularity (stage 7)](#issue-3--hex-uv-polygon-area--shape-looks-wrong-stage-7)
4. [Issue 4 — Code quality / duplication in render_polygrids.py](#issue-4--code-quality--duplication-in-render_polygridspy)
5. [Issue 5 — Tile-to-tile brightness seams on the globe](#issue-5--tile-to-tile-brightness-seams-on-the-globe)
6. [Completed work](#completed-work)

---

## Issue 1 — Pentagon corner-to-UV rotation mismatch (stage 4)

**Status:** ✅ Resolved (confirmed working)

### Symptom

Initial visual inspection of the stage 4 debug image suggested grid
corners C0–C4 were being paired with the wrong UV corners, producing a
~144° rotational offset in the warped tile texture.

### Investigation

`match_grid_corners_to_uv()` in `tile_uv_align.py` matches grid corners
(macro-edge order in Tutte 2D space) to GoldbergTile UV corners
(generator vertex order) by:

1. Computing angles of both sets around their centroids.
2. Trying all N rotations and N reflections.
3. Picking the rotation/reflection with the smallest total angular error.

**Verified numerically** (March 2026): for both t0 (pentagon) and t5
(hexagon), the angular errors between matched pairs are 2–5° — well
within tolerance.  The matching is correct.

The stage 4 debug image was initially misread because panel 1 (raw grid
corners) and panel 2 (UV corners) use different index-start conventions:
C0 starts at the top of the Tutte embedding (~90°), while UV0 starts at
the bottom-left (~−123°).  Panel 3 shows the actual matched pairs after
reordering, which align correctly.

### Conclusion

No code change needed.  The matching algorithm is working correctly.
Phase 38C tasks in TASKLIST.md (robust corner matching improvements)
are already implemented and passing tests.

---

## Issue 2 — Pentagon zigzag bias causes uneven UV cut (stage 7)

**Status:** 🔶 Diagnosed — needs design decision before implementation

### Symptom

In stage 7 (warped tile) for pentagon t0, the UV polygon boundary cuts
unevenly into the hex content:

- **Near corners** (e.g. UV3): the UV polygon boundary is close to the
  inner hex faces — very little apron space.
- **Near edge midpoints**: the hex content sits comfortably inside the
  UV polygon — generous apron space.

This creates an asymmetric appearance: some sides of the tile have
visible neighbour colour bleeding close to the polygon edge, while
others have a wider buffer of the tile's own colour.

### Root cause — zigzag asymmetry

The pentagon detail grid (Goldberg topology + Tutte embedding) has a
hex-cell boundary that zigzags around the ideal smooth polygon edge.
Critically, the zigzag is **biased inward**:

```
Pentagon ME0 — signed distance from corner-to-corner line:
  v108 (C0):  +0.000  (corner, ON the line)
  v109:       −0.352  (inward)
  v110:       −0.063  (slightly inward)
  v111:       −0.391  (inward)
  v112:       −0.077  (slightly inward)
  v113:       −0.353  (inward)
  v114:       +0.012  (≈ zero)
  v115:       −0.212  (inward)
  v116:       +0.200  (outward — first real outward!)
  v117 (C1):  +0.000  (corner, ON the line)
```

Almost all intermediate vertices are **inward** of the corner-to-corner
line.  The corners jut outward beyond the ideal smooth edge by ~3.2% of
the centroid-to-corner distance.

In contrast, the **hex grid** zigzag is **symmetric** around the
corner-to-corner line:

```
Hex ME0 — signed distance from corner-to-corner line:
  v22 (C0):  +0.000
  v21:       +0.444  (outward)
  v20:       −0.111  (inward)
  v37:       +0.333  (outward)
  v36:       −0.222  (inward)
  v54:       +0.222  (outward)
  v53:       −0.333  (inward)
  v73:       +0.111  (outward)
  v72:       −0.444  (inward)
  v94 (C1):  +0.000
```

Hex corners sit exactly **on** the ideal smooth edge.  Pentagon corners
protrude beyond it.

### Why this matters

The UV polygon corners are derived from globe face vertices (the same
3D points as the grid corner vertices).  The warp maps:

    grid_corner → UV_corner

For hex tiles, the corner IS on the smooth edge line, so the warp
produces an even cut.

For pentagon tiles, the corner protrudes outward.  The warp must
"compress" content near corners and "expand" content near edge midpoints,
producing the visible asymmetry.

### Quantified impact

| Metric | Pentagon | Hex |
|--------|----------|-----|
| Corner offset from smooth edge | 3.2% of radius | 0% |
| Smooth-to-corner distance | 0.144 grid units | 0.0 grid units |
| At 512px tile size | ~15–20px displacement | 0px |

### Proposed fix — smoothed grid corners for pentagons

Shift each pentagon grid corner inward to the "smooth" position —
the average of the midpoints of the two edges adjacent to the corner:

```python
mid_before = (corner + prev_boundary_vertex) / 2
mid_after  = (corner + next_boundary_vertex) / 2
smooth_corner = (mid_before + mid_after) / 2
```

This would produce an even cut along each edge, matching the hex
behaviour.

### Trade-off analysis

**Pros:**
- Even apron space around all edges — matches hex tile behaviour.
- The warp's anisotropy decreases near corners.

**Cons:**
- `smooth_corner ≠ globe_vertex`.  The grid corner identity
  (`grid_corner → globe_vertex → UV_corner`) is broken by ~3.2%.
- At 3D globe seams (where two tiles share a vertex), there could be a
  small offset (~15–20px at 512px tile size) producing visible
  triangular gaps or overlaps at the 5 pentagon corners.
- Only affects 12 pentagon tiles; hex tiles are unaffected.

**Alternative — shrink UV polygon instead:**
Scale the pentagon UV polygon inward by the same 3.2%, keeping the
grid-to-UV mapping at the corner peaks.  But this changes the 3D mesh
UV coordinates, requiring `models` repo changes and potentially causing
visible gaps between pentagon and hex tiles on the globe.

### Decision needed

Whether the seam artefact from the smoothed-corner approach is worse
than the current uneven cut.  Needs visual testing.

### Tasks

- [ ] **I2.1** — Implement smoothed pentagon corners as an optional mode
  in `get_macro_edge_corners()` (or a new wrapper function).
- [ ] **I2.2** — Test visually: render t0 with smoothed corners,
  compare stage 7 output to current.
- [ ] **I2.3** — Test seam quality: render full globe at f=3 with
  smoothed pentagon corners, inspect seams at pentagon vertices (where
  5 tiles meet) for gaps/overlaps.
- [ ] **I2.4** — If seam quality is acceptable, make smoothed corners
  the default for pentagons; otherwise, document the trade-off and
  choose the lesser visual artefact.

---

## Issue 3 — Hex UV polygon area / shape looks wrong (stage 7)

**Status:** � Diagnosed — inherent geometry; possible mitigation

### Symptom

In stage 7 for hex tile t5, the UV polygon (red outline) appears
irregular — some edges are visibly shorter than others, and the hex
content doesn't fill the UV polygon evenly.

### Root cause — Goldberg polyhedron geometry

Investigation (March 2026) confirmed the irregularity is **not**
introduced by the UV projection.  It's in the **3D geometry itself**.

On a Goldberg polyhedron (freq=3), hex faces fall into two categories:

| Category | Count | Edge ratio | Dist ratio | Example tiles |
|----------|-------|------------|------------|---------------|
| Regular | ~20% (16 tiles) | 1.000 | 1.000 | t3, t11, t17, t23 … |
| Irregular | ~80% (64 tiles) | **1.274** | 1.087 | t5, t10, t1, t2 … |

The regular hexes are "equatorial" tiles equidistant from all pentagons.
The irregular hexes are adjacent to pentagons and have edges that vary
by **27.4%** in 3D length.  The UV projection preserves this ratio
exactly (3D edge ratio = UV edge ratio = 1.274).

The `normalize_uvs()` uniform scaling fix (Phase 38A) is correctly
applied — both `models/core/geometry.py` and `pgrid/uv_texture.py`
use `max(span_x, span_y)`.  Pentagon UV polygons are perfectly regular
(edge_ratio=1.000 for all 12 tiles, confirmed).

### Impact on rendering

The detail hex grid is built from `build_pure_hex_grid()`, which
produces a **perfectly regular** hexagonal grid (edge_ratio=1.000).
The piecewise warp maps this regular grid onto the irregular UV polygon,
introducing up to 27% local stretch in the texture content.

This is most visible in colour-debug mode where the hex sub-face
outlines reveal the distortion.  With actual terrain textures, the
stretch is less obvious but still present.

### Possible mitigations

1. **Build irregular detail grids** — construct the hex detail grid with
   edge lengths proportional to the 3D face's actual edge lengths, so
   the warp is closer to identity.  This would require a new hex grid
   builder that accepts per-edge scale factors.

2. **Accept the distortion** — the 27% stretch is comparable to typical
   map projection distortion (Mercator is far worse).  With terrain
   textures, hillshade, and at globe-view zoom levels, it may be
   imperceptible.

3. **Scale detail grid to match** — after building the regular hex grid,
   apply an affine pre-warp to approximate the 3D face shape before the
   piecewise warp.  This reduces the work the piecewise warp has to do.

### Decision

Low priority.  The distortion is inherent to the geometry and consistent
between the texture and the 3D mesh.  Focus on Issues 2 and 4 first;
revisit if the distortion is visible with real terrain textures.

### Investigation tasks (for reference)

- [x] **I3.1** — Run debug pipeline on hex tiles t5, t10, t20.
  ✅ Confirmed irregularity visible in stage 7.
- [x] **I3.2** — Check `normalize_uvs()` uniform scaling.
  ✅ Correctly applied in both models and pgrid.
- [x] **I3.3** — Verify hex corner detection.
  ✅ Corners detected correctly; irregularity is in the 3D geometry.
- [x] **I3.4** — Quantify hex UV polygon irregularity.
  ✅ 80% of hex tiles have 27.4% edge ratio; 20% are perfectly regular.
- [x] **I3.5** — Trace irregularity to source.
  ✅ Confirmed: 3D edge ratio = UV edge ratio = 1.274.  Source is
  `models` library's Goldberg polyhedron geometry, not UV normalisation.

---

## Issue 4 — Code quality / duplication in render_polygrids.py

**Status:** ✅ Resolved (March 2026)

### Description

`render_polygrids.py` contained ~400 lines of near-duplicate matplotlib
patch-rendering code across four functions:

- `_render_stitched_tile()`
- `_render_colour_debug_tile()`
- `_render_colour_debug_single()`
- `_render_tile()`

Each function built face patches, colours, and figure setup independently
with minor variations (colour source, outline style, view limits).

### Impact

- Maintenance burden: a rendering fix had to be applied in 4 places.
- Divergent behaviour: the four functions had drifted slightly
  (e.g. different padding calculations, inconsistent edge styles).

### Resolution

**Refactoring pass 2** (March 2026) extracted shared helpers:

| Helper | Purpose |
|--------|---------|
| `_build_face_patches(grid, colour_fn)` | Single face-iteration loop with colour callback |
| `_render_patches_to_png(...)` | Shared matplotlib figure setup, PatchCollection, save/close |
| `_avg_colour(colours)` | Channel-wise mean (was inlined 3×) |
| `_grid_bbox(grid, pad)` | Vertex bounding box (was inlined 2×) |
| `_edge_style(show, colour, lw)` | Edge visibility toggle (was inlined 4×) |
| `_terrain_colour_fn(...)` | Terrain+hillshade colour callback |
| `_colour_debug_fn(...)` | Stitched colour-debug colour callback |
| `_colour_debug_single_fn(...)` | Single-tile colour-debug colour callback |
| `_compute_component_gradient_info(...)` | Component centroid/max-dist for radial gradient |
| `_build_atlas(...)` | Atlas building + file export (was duplicated in 2 branches) |
| `_export_payload_and_metadata(...)` | Payload/metadata JSON export (was duplicated in 2 branches) |

The four render functions are now thin wrappers (5–20 lines each) that
compose a colour callback + call the shared patch builder + renderer.

The `main()` function was split into mode-specific branches
(`_main_colour_debug`, `_main_polygon_cut`, `_main_stitched`,
`_main_neighbour_edges`, `_main_simple`) with shared atlas/export
helpers, eliminating the 400-line monolith.

### Verification

All four rendering modes smoke-tested successfully:
- `--colour-debug --outline-tiles` — 92 tiles + atlas ✅
- Default polygon-cut — 92 tiles + atlas ✅
- `--no-polygon-cut --stitched` — 92 tiles ✅
- `--no-polygon-cut --edges` — 92 tiles ✅

### Tasks

- [x] **I4.1** — Extract shared `_build_face_patches()` helper.
- [x] **I4.2** — Parameterise via colour callbacks for each variant.
- [x] **I4.3** — Rewrite the four `_render_*` functions as thin wrappers.
- [x] **I4.4** — Verify visual output across all modes.

---

## Issue 5 — Tile-to-tile brightness seams on the globe

**Status:** � Two main artefact sources fixed (noise_seed + hillshade); residual is genuine terrain variation

### Symptom

On the rendered 3D globe (both terrain and colour-debug modes), each
tile is visible as a distinct patch with slightly different overall
brightness from its neighbours.  The seam lines follow the Goldberg
polyhedron face boundaries exactly.  The effect is clearly visible in
the terrain mode globe screenshot — adjacent tiles have noticeably
different lightness, giving the globe a "soccer ball" patchwork look.

### Investigation summary (March 2026)

#### Hypothesis 1: Per-tile colour normalisation — RULED OUT

Initially suspected that each stitched tile applied independent colour
normalisation (e.g. stretching elevation → colour separately per tile).
Investigation of `detail_render.py` confirmed the colour pipeline is
**fully deterministic**: given the same elevation, biome, hillshade,
noise coordinates, and seed, it always produces the same RGB.  There
is no per-tile normalisation, histogram stretching, or local contrast
adjustment.

#### Hypothesis 2: Background colour bleeding — RULED OUT

Suspected that the matplotlib figure background colour (which was
computed as `_avg_colour(center_colours)` — the average colour of the
centre tile's faces) differed per tile, and that this background leaked
into the atlas through image corners.

**Fix attempted:** Changed all renderers to use a bright magenta
sentinel `(255, 0, 255)` as background, then added
`fill_sentinel_pixels()` to replace any sentinel pixels with the
nearest valid pixel before atlas warping.

**Result:** After re-rendering, **zero sentinel pixels** were found in
any tile image.  The stitched composite's polygon patches completely
cover the entire matplotlib view area — there is no exposed background.
The zero-variance corner patches initially observed were actual terrain
faces (from the flat apron of the neighbour grid) that happen to share
the same colour, not background fill.

The sentinel infrastructure remains in the code as defensive code but
has no practical effect.

#### Hypothesis 3: Rasterisation differences — PARTIALLY CONFIRMED

The same face appears in multiple overlapping stitched renders (once as
the "centre tile" render and again as a "neighbour" in an adjacent
tile's render).  Each render has different `xlim`/`ylim` view windows,
so the same face polygon lands at different sub-pixel positions in each
512×512 PNG.

When the piecewise warp samples these images using bilinear interpolation
(`scipy.ndimage.map_coordinates`), the sub-pixel positioning differences
cause slightly different RGB values to be produced for the shared edge
region.  Adjacent tiles in the atlas therefore have subtly different
pixel colours along their boundary.

**Key evidence:**
- All stitched tile images are exactly 512×512 pixels (consistent)
- Colour pipeline is fully deterministic (same inputs → same RGB)
- No background pixels exist (patches cover everything)
- The seam brightness difference is small (~1-5 RGB values) but
  visually obvious because the human eye is very sensitive to
  edges/discontinuities

#### Hypothesis 4: Hillshade boundary truncation — CONFIRMED & FIXED ✅

`_detail_hillshade()` computes per-face hillshade from the elevation
gradient between adjacent faces using `get_face_adjacency(grid)`.  In
the stitched pipeline, each tile is rendered as a composite: the centre
tile + its immediate neighbours.  The merged grid's face adjacency is
**local to the composite** — faces at the outer boundary of the
composite have fewer neighbours than they would in the full globe.

When face F appears in two different composites (A's and B's), it gets
**different adjacency neighbourhoods** and therefore **different
hillshade values**.  Only the innermost ring of centre-tile faces (fully
surrounded by composite faces) is consistent.

**Quantified impact (verified in notebook, July 2025):**

```
Hillshade inconsistency across ALL boundary face pairs:
  Total boundary face comparisons: 9,834
  Max hillshade diff:    0.9914
  Mean hillshade diff:   0.1078
  Median hillshade diff: 0.0230
  95th pctile:           0.4523
  99th pctile:           0.6886
  Faces with diff > 0.001: 7,699 / 9,834 (78.3%)
  
  Estimated RGB impact:
    Mean:    3.5 RGB
    99th:   22.4 RGB
    Max:    32.2 RGB  ← the DOMINANT seam contributor
```

The hillshade contribution is **systematic** — it biases ALL boundary
faces consistently in one direction within a tile, creating a coherent
per-tile brightness shift rather than random noise.

**Fix:** `_compute_global_hillshade()` in `render_polygrids.py`
pre-computes hillshade for every detail face from the composite where
that face's tile is the centre (maximum neighbour context).  The
`_render_stitched_tile()` function accepts this pre-computed dict and
looks up values instead of computing per-composite hillshade.  This
eliminates the entire distribution above.

#### Root cause summary

Three confirmed contributors (ranked by impact):

| Contributor | Mean RGB | Worst RGB | Status |
|---|---|---|---|
| **Hillshade boundary truncation** | ~3.5 | ~32.2 | ✅ Fixed |
| **noise_seed per-tile bias** | ~1.5 | ~26.3 | ✅ Fixed |
| **Rasterisation sub-pixel** | ~0.1 | ~1 | ✅ Measured (negligible) |
| **Genuine terrain variation** | ~12.5 std | n/a | Not a bug |

### Fix approaches (ordered by effort/impact)

1. ✅ **Fix noise_seed** — Changed `noise_seed = seed + i` → `seed` in
   all 4 render paths.  Noise coordinates already vary per face via
   `noise_x, noise_y`.  Eliminates up to ~26.3 RGB worst-case.

2. ✅ **Fix hillshade boundary truncation** — Added
   `_compute_global_hillshade()` which pre-computes hillshade for every
   detail face from its centre composite (full neighbour context), then
   passes the values into `_render_stitched_tile()`.  Eliminates up to
   ~32.2 RGB worst-case.

3. ❌ **Shader-level `edge_blend`** — INVESTIGATED, NOT APPLICABLE.
   The `edge_blend` parameter in `build_batched_globe_mesh` blends
   per-tile **vertex colours** toward neighbour averages.  However,
   the PBR fragment shader uses `base = texture(u_atlas, v_uv).rgb`
   when texturing is enabled and **never references `v_color`**.
   So `edge_blend` has zero effect on texture-based seams.  A true
   texture-level seam fix would require a shader that detects tile
   boundary proximity and samples/blends across tile boundaries in
   the atlas — a significantly more complex approach.

4. ⬜ **Cross-fade gutter** — At each tile's gutter, blend in content
   from the neighbouring tile's render.  The current gutter just clamps
   edge pixels; replacing it with actual neighbour content would smooth
   the transition.

5. ✅ **Analytical fill renderer** — Renders stitched terrain colours
   via point-in-polygon fill, bypassing matplotlib rasterisation.
   Implemented as `_render_analytical_to_png()` in `render_polygrids.py`.
   **Now the default renderer** (`--renderer analytical`).  Prototyped
   and measured in `notebooks/analytical_fill.ipynb`.

   **Results:** Vectorised fill runs in 0.07s per 512×512 tile (7× faster
   than matplotlib).  100% pixel coverage, deterministic (no anti-aliasing).
   Full atlas comparison (92 tiles, 270 pairs): median seam diff 14.1 vs
   matplotlib's 14.0 — **negligible difference** because tile-mean seams
   are dominated by genuine terrain variation (~12.5 RGB std), not
   rasterisation artefacts.  The sub-pixel rasterisation contribution
   is ~0.1 RGB median per pair.

### Quantified severity

```
BEFORE fixes — Adjacent tile mean brightness differences (freq=3, seed=42):
  Pairs analysed: 270
  Median max-channel diff:  13.5 RGB
  Mean max-channel diff:    13.8 RGB
  Maximum:                  37.6 RGB  (t13-t32, hex-hex)
  Minimum:                   0.8 RGB  (t85-t86)
  
  Pentagon-adjacent edges:  60, median = 13.4
  Hex-only edges:          210, median = 13.6
  → Seams are NOT concentrated at pentagons — uniform across globe.

AFTER fixes (noise_seed + global hillshade):
  Median max-channel diff:  13.2 RGB
  Maximum:                  37.2 RGB
  
  The aggregate improvement is modest because genuine terrain variation
  (~12.5 RGB std within tiles) dominates the tile-mean metric.  The
  per-face artefact elimination is significant:
    noise_seed:  up to 26.3 RGB worst-case eliminated
    hillshade:   up to 32.2 RGB worst-case eliminated

ANALYTICAL fill renderer (--renderer analytical):
  Median max-channel diff:  14.1 RGB
  Maximum:                  40.6 RGB
  
  Essentially identical to matplotlib (median Δ = -0.1, within noise).
  Confirms rasterisation sub-pixel contribution is negligible at the
  tile-mean level.  Analytical fill is 7× faster (0.07s vs 0.5s per
  tile) but does not improve seam metrics.

Within-tile RGB std:  median = 12.5 (terrain has significant variation)
```

### Key insight — terrain variation vs artefact

The visible "soccer ball" pattern was a combination of:
1. **Hillshade boundary truncation** (~3.5 mean, ~32.2 worst) —
   the DOMINANT contributor, now eliminated ✅
2. **noise_seed per-tile bias** (~1.5 mean, ~26.3 worst) —
   eliminated ✅
3. **Genuine terrain variation** (~12.5 RGB std) — different tiles have
   different mean elevations, which is correct terrain rendering
4. **Rasterisation sub-pixel differences** (~0.1 RGB median) —
   measured via analytical fill A/B comparison, negligible impact ✅

Fixes 1 + 2 together eliminate the two identified artefact sources
(up to ~32 RGB hillshade + ~26 RGB noise worst-case).  The post-fix
atlas brightness analysis shows median 13.2 / max 37.2 tile-pair diffs,
which is dominated by genuine terrain variation (~12.5 RGB std within
tiles).  Fix 5 (analytical fill) confirmed that rasterisation sub-pixel
contributes only ~0.1 RGB to tile-mean seams — within measurement noise.
Remaining visible seams after these fixes are genuine terrain variation.

### Tasks

- [x] **I5.1** — Quantify per-tile mean brightness differences
  (270 pairs, median 13.5, max 37.6 RGB)
- [x] **I5.2** — Check noise_seed consistency
  (contributes ~1.5 mean, ~26.3 worst)
- [x] **I5.3** — Check hillshade boundary truncation
  (confirmed: max 0.99 hillshade diff, ~32.2 RGB worst-case)
- [x] **I5.4** — Fix noise_seed: use `noise_seed = seed` for all tiles
  (applied: eliminates up to 26.3 RGB worst-case, 13.4 at 99th pctile)
- [x] **I5.5** — Fix hillshade: `_compute_global_hillshade()` added to
  `render_polygrids.py`.  Pre-computes authoritative hillshade for each
  face from the composite where it was the centre tile.  Eliminates up
  to 32.2 RGB worst-case, 22.4 at 99th pctile.
- [x] **I5.6** — Re-render and re-measure after both fixes.
  Post-fix atlas: median 13.2 (was 13.5), max 37.2 (was 37.6).
  Small aggregate improvement because genuine terrain variation (~12.5 std)
  dominates the tile-mean metric.
- [x] **I5.7** — Investigate shader `edge_blend` parameter.
  FINDING: `edge_blend` blends per-tile vertex colours only; the PBR
  fragment shader ignores `v_color` when texturing is enabled
  (`base = texture(u_atlas, v_uv).rgb`).  No effect on texture seams.
  A texture-level fix would require tile-boundary-aware shader blending
  or cross-fade gutters — out of scope for now.

---

## Completed work

### Refactoring — March 2026

| Change | Files | Notes |
|--------|-------|-------|
| Consolidated `_fill_gutter()` | `atlas_utils.py` (new), `tile_uv_align.py`, `texture_pipeline.py`, `detail_perf.py`, `biome_pipeline.py`, `render_globe_from_tiles.py` | Was duplicated in 4 files |
| Consolidated `compute_atlas_layout()` | `atlas_utils.py`, `texture_pipeline.py`, `render_globe_from_tiles.py` | Was duplicated in 4+ inline implementations |
| Consolidated view-limits calculation | `render_polygrids.py` | Replaced ~18 lines × 2 with `compute_tile_view_limits()` calls |
| Removed dead code | `tile_uv_align.py` | `_reorder_grid_corners_to_uv()` was unused no-op |
| Created debug pipeline | `scripts/debug_pipeline.py` (new) | 7-stage diagnostic tool, 33+ images |
| Added boundary vertex IDs to stage 2 | `scripts/debug_pipeline.py` | Colour-coded per macro edge, nudged outward |
| Extracted shared render helpers | `render_polygrids.py` | `_build_face_patches()`, `_render_patches_to_png()`, colour callbacks; 4 render functions → thin wrappers |
| Extracted atlas/export helpers | `render_polygrids.py` | `_build_atlas()`, `_export_payload_and_metadata()`; `main()` split into 5 mode-specific branches |
| Documentation updates | `README.md`, `TILE_TEXTURE_MAPPING.md`, `RENDERING_ISSUES.md` | Debug pipeline docs, key concepts, issue tracker |
| Sentinel background infrastructure | `render_polygrids.py`, `tile_uv_align.py` | `_SENTINEL_BG` constant, `fill_sentinel_pixels()` — defensive code, no practical effect since patches cover entire view |
| Atlas gap filling | `tile_uv_align.py` | `_fill_warped_gaps()` replaces cval=128 pixels; `_dilate_cval_pixels()` for piecewise warp corners |
| Negative-zero normalisation | `globe_renderer_v2.py` | `_project_to_sphere()` + vertex dedup key normalisation eliminates -0.0 vs +0.0 mismatch |
| UV inset parameter | `globe_renderer_v2.py` | `uv_inset_px` parameter wired through to `render_globe_v2()` |

### Phase 38 — Pentagon distortion fixes (prior work)

| Task | Status | Notes |
|------|--------|-------|
| 38A — Uniform UV normalisation | ✅ Done | `normalize_uvs()` in models + `uv_texture.py` use `max(span)` |
| 38B — Exact pentagon corner detection | ✅ Done | `corner_vertex_ids` metadata + fast-path in `_find_polygon_corners()` |
| 38C — Robust corner matching | ✅ Done | Edge-length ratio validation in `compute_detail_to_uv_transform()` |
| 38D — Visual validation | ✅ Done | Quantitative analysis confirms pipeline is healthy |

---

## Priority & Sequence

```
1. [Issue 3] Diagnose hex UV polygon issue    ✅ Diagnosed — inherent geometry
2. [Issue 4] Refactor render_polygrids.py     ✅ Done — shared helpers + thin wrappers
3. [Issue 5] Tile brightness seams            ✅ Resolved — noise_seed, hillshade, analytical fill
4. [Issue 2] Pentagon zigzag smoothing         ← blocked on pentagon overrides being disabled
5. [Phase 38D] Full visual validation          ✅ Done — see results below
```

### Current state (March 2026)

**What works:**
- Hex tiles render correctly (orientation, colour, warp all good)
- **Pentagon tiles warp correctly** — 100% atlas coverage, 0.0 RGB
  inner↔gutter discontinuity across all 12 pentagon tiles
- Atlas has no grey (128,128,128) pixels inside tile regions
- Tile-to-tile brightness seams resolved (noise_seed fix + global
  hillshade precomputation + analytical fill option)
- Negative-zero vertex normalisation prevents GPU edge gaps
- UV inset (1.5px) prevents bilinear texture bleeding
- `fill_sentinel_pixels()` and `_fill_warped_gaps()` handle edge cases
- Colour pipeline is fully deterministic (no per-tile normalisation)
- `render_polygrids.py` cleanly refactored with shared helpers
- `debug_pipeline.py` produces 7-stage per-tile diagnostic images

**Remaining visual issues (cosmetic, not blocking):**
- Pentagon tiles show slightly higher edge discontinuity than hex tiles
  (mean 6.5 vs 4.6 RGB edge_diff) — inherent to 5-sided warping geometry
- Pentagon zigzag bias (Issue 2) still present — pentagon overrides
  (smoothing, scaling) remain disabled in `build_polygon_cut_atlas`
- Pentagon quad-range (brightness imbalance across quadrants) slightly
  higher than hex (11.5 vs 10.2) — again inherent to 5→4 sector mapping

**Dead/unused code to clean up:**
- `match_grid_corners_to_uv()` — only used by debug/diag scripts, not
  by the atlas pipeline (which uses `compute_uv_to_polygrid_offset`)
- `_equalise_sector_ratios()` — only used by `debug_pipeline.py`
- `_smooth_pentagon_corners()` — defined but never called (commented out)
- `_scale_corners_from_centroid()` — only used by `debug_pipeline.py`
- `_rotate_corners()` — defined but never called anywhere
- `_PENTAGON_GRID_SCALE` — only used by `debug_pipeline.py`
- Numerous old `diag_*.py` scripts in `scripts/` that may be stale

> **Note:** Issue 1 was initially thought to be a rotation mismatch
> but investigation confirmed the matching is correct.  Issue 3
> turned out to be inherent Goldberg polyhedron geometry (27% edge
> variation in hex faces), not a UV projection bug.  Both remain
> documented here for reference.

### Phase 38D — Visual validation results (March 2026)

Colour-debug globe rendered at freq=3, detail_rings=4, tile_size=512.
Actual pentagon tile IDs: t0, t6, t9, t14, t20, t26, t35, t41, t47,
t53, t59, t61 (NOT sequential t0–t11 as initially assumed).

| Metric | Pentagon (n=12) | Hex (n=80) | Notes |
|--------|----------------|------------|-------|
| Stitched coverage | 100.0% | 100.0% | All 262,144 px valid |
| Warped coverage | 100.0% | 100.0% | All 270,400 px valid (520×520 slot) |
| Centre coverage | 100.0% | 100.0% | No holes in inner 50% |
| Atlas inner↔gutter disc. | 0.0 RGB | 0.0 RGB | Perfect gutter fill |
| Edge discontinuity | 6.5 RGB | 4.6 RGB | Moderate, inherent to geometry |
| Quad-range | 11.5 | 10.2 | Brightness balance across quadrants |
| Black pixels | 0 | 0 | No sentinel/gap pixels anywhere |

**Conclusion:** The polygon-cut atlas pipeline produces clean, gap-free
tiles for both pentagons and hexagons.  Pentagon tiles are slightly more
uneven than hex tiles due to the 5→rectangular sector mapping, but the
differences are within acceptable bounds.  No blocking issues found.

---

## How to reproduce

```bash
# Generate colour-debug tiles
.venv/bin/python scripts/render_polygrids.py \
    --colour-debug --outline-tiles -f 3 --detail-rings 4 \
    -o exports/colour_debug

# Generate debug pipeline images for specific tiles
.venv/bin/python scripts/debug_pipeline.py --tiles t0 t5

# Render full globe from colour-debug tiles
.venv/bin/python scripts/render_globe_from_tiles.py exports/colour_debug --v2
```

Output directories:
- `exports/colour_debug/` — per-tile colour-debug PNGs + atlas
- `exports/debug_pipeline/` — per-stage diagnostic images
