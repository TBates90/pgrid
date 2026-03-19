══════════════════════════════════════════════════════════
 PolyGrid Test Suite — 1,459 tests across 46 files
══════════════════════════════════════════════════════════

  Phase 1-4 — Core Topology & Transforms
    ✅ test_core_topology.py .................    6 tests    0.7s
    ✅ test_stitching.py .....................    5 tests    1.4s
    ✅ test_assembly.py ......................   16 tests    2.5s
    ✅ test_macro_edges.py ...................   10 tests    1.8s
    ✅ test_pentagon_centered.py .............    9 tests    1.6s
    ✅ test_transforms.py ....................   15 tests    1.5s
    ✅ test_diagnostics.py ...................    6 tests    1.2s
    ✅ test_visualize.py .....................    4 tests    4.2s
                                           ────────────────────────
                                             71 tests    14.9s

  Phase 2 — Goldberg Topology
    ✅ test_goldberg.py ......................   79 tests    8.8s
                                           ────────────────────────
                                             79 tests     8.8s

  Phase 5-7 — Tile Data & Terrain
    ✅ test_tile_data.py .....................   52 tests    1.5s
    ✅ test_regions.py .......................   76 tests    1.3s
    ✅ test_noise.py .........................   40 tests   23.4s
    ✅ test_heightmap.py .....................   19 tests    0.8s
    ✅ test_mountains.py .....................   16 tests    2.2s
    ✅ test_rivers.py ........................   25 tests    2.0s
    ✅ test_pipeline.py ......................   20 tests    1.6s
    ✅ test_terrain_render.py ................   24 tests    1.8s
    ✅ test_determinism.py ...................    2 tests    1.2s
                                           ────────────────────────
                                            274 tests    35.8s

  Phase 8-9 — Globe & Export
    ❌ test_globe.py .........................  110 tests   25.2s  (1 failed)
                                           ────────────────────────
                                            110 tests    25.2s

  Phase 10 — Sub-Tile Detail
    ✅ test_tile_detail.py ...................   24 tests   41.8s
    ✅ test_detail_render.py .................   26 tests   17.8s
    ❌ test_detail_perf.py ...................   15 tests   59.4s  (2 failed)
                                           ────────────────────────
                                             65 tests  1m 59.1s

  Phase 11 — Cohesive Terrain
    ✅ test_detail_terrain.py ................   15 tests   35.8s
    ✅ test_detail_terrain_3d.py .............   29 tests   45.1s
    ✅ test_terrain_patches.py ...............   22 tests  1m 15.6s
    ❌ test_globe_terrain.py .................   17 tests   10.3s  (2 failed)
    ✅ test_region_stitch.py .................   19 tests    7.5s
    ✅ test_render_enhanced.py ...............   32 tests   57.6s
    ✅ test_texture_pipeline.py ..............    9 tests   34.4s
                                           ────────────────────────
                                            143 tests  4m 26.2s

  Phase 12-13 — Rendering & PBR
    ❌ test_globe_renderer_v2.py .............  262 tests   11.5s  (1 failed)
    ✅ test_phase13_rendering.py .............   15 tests   39.3s
                                           ────────────────────────
                                            277 tests    50.8s

  Phase 14 — Biome Features
    ✅ test_biome_scatter.py .................   37 tests  1m 17.0s
    ✅ test_biome_render.py ..................   23 tests  1m 21.4s
    ✅ test_biome_pipeline.py ................   29 tests  12m 06.3s
    ✅ test_biome_continuity.py ..............   21 tests    0.8s
                                           ────────────────────────
                                            110 tests  14m 45.6s

  Ungrouped
    ✅ test_apron_grid.py ....................   28 tests   16.0s
    ✅ test_apron_texture.py .................   17 tests  7m 04.5s
    ❌ test_atlas_seams.py ...................    7 tests    0.8s  (2 failed)
    ✅ test_biome_topology.py ................   33 tests  1m 21.1s
    ✅ test_coastline.py .....................   49 tests  8m 01.1s
    ❌ test_corner_blend.py ..................   13 tests    0.8s  (1 failed)
    ✅ test_ocean_render.py ..................   44 tests  2m 06.1s
    ✅ test_texture_export.py ................   36 tests    1.0s
    ✅ test_tile_texture.py ..................   40 tests  2m 11.3s
    ✅ test_uv_texture.py ....................   43 tests   38.9s
    ✅ test_visual_cohesion.py ...............   19 tests  17m 21.6s
                                           ────────────────────────
                                            329 tests  39m 03.4s

══════════════════════════════════════════════════════════
 SUMMARY
══════════════════════════════════════════════════════════
  Phase 1-4 — Core Topology & Trans...   71 tests    14.9s  ✅
  Phase 2 — Goldberg Topology            79 tests     8.8s  ✅
  Phase 5-7 — Tile Data & Terrain       274 tests    35.8s  ✅
  Phase 8-9 — Globe & Export            110 tests    25.2s  ❌
  Phase 10 — Sub-Tile Detail             65 tests  1m 59.1s  ❌
  Phase 11 — Cohesive Terrain           143 tests  4m 26.2s  ❌
  Phase 12-13 — Rendering & PBR         277 tests    50.8s  ❌
  Phase 14 — Biome Features             110 tests  14m 45.6s  ✅
  Ungrouped                             329 tests  39m 03.4s  ❌
──────────────────────────────────────────────────────────
  TOTAL                                1458 tests  62m 29.6s  9 FAILED