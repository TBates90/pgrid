Diagnosis and Remediation Plan for Pentagon–Hex Polygrid Distortion in pgrid
Observed artefact in your render

In the screenshot (freq
    No localised wedge/pinch artifacts at the pentagon corners when rings=1 (the extreme case), and
    No regression in hex↔hex continuity (your current "works perfectly" case).

---

Implementation Status

Fixes 1 and 2 from the plan above have been implemented:

1. **Pentagon-centred composites skip neighbour↔neighbour closure**
   (`tile_detail.py: build_tile_with_neighbours`).  When `n_sides == 5`,
   the entire outer-pair discovery → snap → stitch block is skipped.
   Only centre↔neighbour stitches are created.  Corner gaps are handled
   by the existing gap-fill infrastructure (`fill_sentinel_pixels`,
   `_fill_warped_gaps`, `_dilate_cval_pixels`).

2. **Overlap quality guard for hex composites**
   (`tile_detail.py: _macro_edge_overlap_ok`).  Before snapping any
   outer neighbour pair, the maximum vertex-to-vertex distance is
   checked against 25 % of the macro-edge length.  Pairs that exceed
   the tolerance are silently skipped, preventing forced closure of
   edges that don't geometrically align.

Remaining items (not yet implemented, per the staged plan):

- **Seam-enforcement post-pass in atlas space** (plan step 3) — ✅ DONE.
  `_stitch_atlas_seams()` in `tile_uv_align.py` averages boundary pixels
  along shared Goldberg edges after all tiles are warped into the atlas.
  Integrated into `build_polygon_cut_atlas()` via `stitch_seams=True`
  (default) and `stitch_width=2`.  Mirrors the vectorised approach from
  `_stitch_tile_edges()` in `uv_texture.py` but operates in atlas pixel
  space.
- **Irregular hex detail grids** (plan step 4 / structural mitigation) —
  NOT IMPLEMENTED.  This is the "most work, best geometry" path per the
  plan — building tile-adaptive detail grids whose boundary macro-edges
  match per-edge length ratios from the GoldbergTile UV polygon.  The
  practical mitigations (steps 1–3 + 5) should be validated visually
  first.  If residual stretch in pent-adjacent hexes is still
  unacceptable with real textures, this becomes the next lever.  The
  plan recommends implementing behind a feature flag, starting only
  with irregular hexes and leaving regular hexes on the fast path.
- **Re-evaluate `_equalise_sector_ratios`** (plan step 5) — ✅ DONE.
  Re-enabled as an opt-in `equalise_sectors=False` parameter on
  `build_polygon_cut_atlas`.  When `True`, irregular hex tiles get
  `_equalise_sector_ratios` applied (making each piecewise-warp sector
  conformal); the seam post-pass compensates for the boundary shift.
  Pentagon tiles continue using `_scale_corners_from_centroid`.
  Regular hex tiles are unaffected (the function is a no-op for them).
detail-rings=1), the region around the central pentagon tile shows localised “pinching/shearing” where the pentagon-centred polygrid meets the surrounding hex-centred polygrids. The artefact is not a uniform stretch across an entire tile; it appears concentrated near the pentagon’s corners and along the boundaries where two neighbour-hex regions are expected to meet cleanly.

That pattern is strongly suggestive of a layout/stitching issue in the “centre tile + neighbours” composite construction for pentagon-centred tiles (where there are only 5 neighbours), rather than a generic hex↔hex edge continuity issue (where the neighbourhood can close in the plane cleanly).
Current pipeline and likely insertion point for the distortion

The default atlas path used by the repo’s “polygon-cut” pipeline is:

    Build detail grids per globe face (pent tiles get a pentagon-centred Goldberg grid; hex tiles get a pure regular hex grid).
    For each tile, build a stitched composite consisting of the centre tile + all neighbouring tiles, and then render that composite to a square PNG. This is done by build_tile_with_neighbours() and _render_stitched_tile() in scripts/render_polygrids.py.
    Warp the rendered stitched PNG into the tile’s UV polygon and pack into an atlas via build_polygon_cut_atlas() in tile_uv_align.py.

The repo already includes two targeted diagnostic tools that are ideal for isolating whether the artefact is created during neighbour stitching (composite build) or during UV warp:

    scripts/debug_pipeline.py (multi-stage “where did it go wrong?” breakdown, including “Stitched composite” and “Warped tile” stages).
    scripts/validate_polygon_cut.py (single-tile, stitched vs warped, with outlines and corner mapping).

The fastest way to prove the insertion point is: compare stage 3 vs stage 7 for a pent tile. If stage 3 (“stitched composite”) already shows the wedge distortion near pent corners, the root cause is the neighbour-neighbour closure logic in build_tile_with_neighbours. If stage 3 is clean and the distortion appears only after stage 7, then the UV polygon warp / polygon irregularity path is the primary issue.
Primary cause found in code: forced neighbour–neighbour closure around a pentagon-centred composite

The key behaviour difference between hex-centred and pent-centred composites is captured directly in build_tile_with_neighbours():

    It positions each neighbour flush against the centre tile (centre↔neighbour seams).
    It then discovers neighbour↔neighbour adjacencies among the positioned neighbours and tries to make the outer ring close by:
        picking a “shared” pair of neighbour macro-edges purely by minimum midpoint distance (_find_closest_macro_edge_pair),
        snapping vertex pairs by averaging positions (forcing edges to coincide),
        and finally stitching those neighbour↔neighbour edges into the composite.

This logic is explicitly written with an assumption that is valid for a hex-centred neighbourhood but fragile for a pent-centred one:

    “After positioning, two adjacent neighbour grids share exactly one macro edge geometrically.”

That assumption is not generally true around a pentagon-centred tile in a flat (Euclidean) layout, because the neighbourhood has unavoidable “curvature deficit” relative to a hex tiling.

The repo has a very direct acknowledgement of the geometric incompatibility in the pent+5-hex planar assembly recipe (assembly.pent_hex_assembly). It states that adjacent hex grids around a pentagon corner naturally leave a gap because a hex corner angle (120°) does not match a pent corner angle (108°), and it then “snaps” boundary vertices to force stitches anyway—accepting “minimal local distortion” in the outer boundary cells as the trade.

With detail_rings=1, almost everything is “outer boundary” (there’s only one ring of hexes around the centre cell), so that “minimal boundary distortion” becomes visually prominent and can manifest exactly as the kind of localised wedge/pinch you see near the pentagon corners.

In short: the code currently tries to make the 5-neighbour ring close cleanly in the plane; it cannot do so without distorting something, and the distortion is being injected by the neighbour↔neighbour snapping/stitching pass.
Secondary contributor: genuine irregularity of hex tiles adjacent to pentagons at freq=3

Even if you completely neutralise the neighbour↔neighbour closure artefacts, there is a second effect that can still read as “distortion” at pent↔hex neighbourhoods on a Goldberg globe:

    For freq=3, most hex tiles adjacent to pentagons have unequal edge lengths (the repo’s own investigation quantifies this as ~80% “irregular” hexes with edge-length ratio ~1.274). That irregularity is present in the underlying 3D Goldberg geometry and is preserved by UV projection.
    Meanwhile, the hex-centred detail grid builder (build_pure_hex_grid) generates a perfectly regular hex grid in 2D.
    Any mapping that places a regular internal pattern onto an irregular polygon will introduce some degree of local stretch/compression. The repo documents this as an inherent property of the geometry, and explicitly lists “build irregular detail grids” as a potential mitigation if the distortion becomes visually objectionable.

This secondary effect is usually subtle with terrain/noise textures, but it becomes obvious with low-frequency, cell-structured debug renders and especially at very low ring counts (like rings=1), where each cell is large and deformation is easier to perceive.
Comprehensive plan to resolve the pent↔hex distortion

This plan is staged so you can stop once the visual target is met.
Isolate whether the distortion is introduced by stitching or by UV warp

Run the debug pipeline on the central pent tile and a representative adjacent hex tile with your exact parameters (freq=3, rings=1). The script is designed to reveal “stage of first failure”.

    If stage 3 “Stitched composite” already shows the wedge distortion near pent corners, focus on build_tile_with_neighbours neighbour↔neighbour closure.
    If stage 3 looks geometrically sensible but stage 7 “Warped tile” introduces the visible deformation, focus on UV-warp handling and (optionally) the sector equalisation / irregular-hex mitigation path.

This step prevents “fixing the wrong subsystem”.
Fix the neighbour–neighbour closure logic for pentagon-centred composites

The safest resolution is to stop forcing neighbour↔neighbour closure when it is not geometrically valid, while preserving the thing you actually need for seamless tiles: centre↔neighbour correctness along each shared edge.

A robust approach is to add an explicit overlap quality check before snapping/stitching any outer neighbour pair:

    After _find_closest_macro_edge_pair() returns (e1, e2), compute an “edge overlap error” using the macro-edge boundary vertices before you average them.
    If the maximum (or mean) vertex-to-vertex distance exceeds a tolerance, treat that pair as “non-overlapping” and do not snap/stitch it.

This directly addresses the implicit assumption (“share exactly one macro edge geometrically”) that fails in pentagon neighbourhoods.

A simpler (and often sufficient) variant is:

    If the centre tile is a pentagon (n_sides == 5), skip the entire neighbour↔neighbour closure section (outer_pairs discovery + snapping + outer stitches). Keep only centre↔neighbour stitches.

Why this is usually safe in your pipeline:

    The crop window used for rendering stitched tiles (compute_tile_view_limits) is computed from centre tile faces only (prefixed by the centre tile ID), with padding. So you aren’t relying on neighbour↔neighbour manifold closure for the crop itself.
    Pixels outside the centre polygon are not directly sampled by the 3D mesh; they mainly act as “natural gutter” content. Any corner holes created by skipping outer closure can be handled by the existing sentinel + gap-fill infrastructure (fill_sentinel_pixels, _fill_warped_gaps, _dilate_cval_pixels).

Expected impact:

    The distortion caused by forcibly collapsing the pentagon’s “missing neighbour wedge” gets removed (or heavily reduced), because you stop injecting artificial deformation into the neighbour tiles near the pentagon corners.

Add a seam-enforcement step in atlas space for pent↔hex edges

Even after fixing the planar closure artefact, you can still get residual discontinuities at pent↔hex edges due to warp sampling differences and the fact that each tile is independently resampled into UV space. A high-leverage mitigation is to enforce consistency in atlas pixel space along shared edges:

    The uv_texture.py pipeline already includes an explicit “stitch edges” concept (averaging/propagating edge pixels across shared boundaries after UV-aligned rendering).
    Port that idea to the polygon-cut atlas builder (build_polygon_cut_atlas) as an optional post-pass:
        iterate all adjacency edges in the globe grid,
        for each shared edge, compute the corresponding pixel polyline along both tiles’ UV polygon edges in their atlas slots,
        and blend/average pixels (optionally confined to a narrow band) so both sides match.

This gives you a backstop: even if the tile-generation process produces slightly different pixels at the boundary, the atlas becomes self-consistent.

This also “unblocks” some more aggressive corner/sector adjustments (next step), because seam risk becomes manageable with deterministic post-stitching.
Decide how far you want to go on the “irregular hex” distortion

If (after the neighbour↔neighbour closure fix) what remains is primarily a smooth stretch inside hex tiles adjacent to pentagons, that is consistent with the repo’s documented “inherent Goldberg geometry” effect at freq=3.

You then have three paths:

    Accept it (often fine for natural terrain textures, less fine for visible grid overlays). The repo explicitly frames this as “low priority unless visible with real textures”.
    Reduce its visibility (practical): increase detail_rings above 1 so individual cells are smaller, making any deformation less noticeable at globe-view zoom levels. This doesn’t remove distortion but can make it visually negligible in the intended viewpoint. The detail grid scale is controlled by TileDetailSpec(detail_rings=...) and is already wired through the render scripts.
    Mitigate structurally (most work, best geometry): implement tile-adaptive hex detail grids for irregular hex faces by reshaping the detail grid boundary (and optionally interior) to match per-edge length ratios derived from the tile’s UV polygon / 3D geometry. The repo calls this out as a key possible mitigation: “Build irregular detail grids … accepts per-edge scale factors.”

If you choose the structural route, I’d implement it behind a feature flag and start only with the “irregular hex” class (the ones adjacent to pentagons), leaving perfectly-regular hex tiles on the current fast path.
Re-evaluate sector equalisation with seam protection

tile_uv_align.py contains _equalise_sector_ratios() specifically to reduce anisotropy when mapping regular grid corners to irregular UV corner configurations, but it is currently intentionally disabled in build_polygon_cut_atlas because (as the comment explains) moving corners causes the warp to sample “the wrong side” of edges and creates visible seams.

Once you have either:

    the neighbour↔neighbour closure artefact removed (so the stitched source image is “cleaner” near pent corners), and/or
    a post-atlas seam enforcement pass (previous step),

you can run controlled experiments re-enabling _equalise_sector_ratios for the subset of tiles where it matters (typically irregular hexes adjacent to pentagons) and see whether the perceived distortion improves enough to justify any remaining complexity. The debug pipeline already visualises the equalisation and sector anisotropy, so it’s well-supported for iteration.
Validation and success criteria

Use the tooling already in the repo to confirm “fixed in the right place” and guard against regressions:

    scripts/debug_pipeline.py:
        Compare stage 3 (“Stitched composite”) before/after the build_tile_with_neighbours change on a pent tile, using rings=1; the pent-corner wedge distortion should no longer appear in the composite.
        Confirm stage 7 (“Warped tile”) does not introduce new corner holes beyond what _fill_warped_gaps / dilation already manages.

    scripts/validate_polygon_cut.py on:
        the pent tile (t0 in most freq=3 layouts),
        and at least one adjacent irregular hex tile, to ensure the polygon outline and UV outline still align and corner mapping hasn’t regressed.

    Full-globe render in colour-debug mode (scripts/render_polygrids.py --colour-debug --outline-tiles -f 3 --detail-rings 1) to visually confirm that the pent neighbourhood no longer shows the concentrated wedge artefact while hex↔hex remains unchanged.

A practical “definition of done” for your specific complaint:

    No localised wedge/pinch artifacts at the pentagon corners when rings=1 (the extreme case), and
    No regression in hex↔hex continuity (your current “works perfectly” case).
