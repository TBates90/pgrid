# pgrid ↔ playground Integration Plan

> **Single source of truth** for integrating pgrid's procedural terrain
> pipeline into playground's world-building and planet rendering stack.
>
> Related docs:
> - pgrid: [`BIOME_BIBLE.md`](./BIOME_BIBLE.md), [`JSON_CONTRACT.md`](./JSON_CONTRACT.md), [`MODULE_REFERENCE.md`](./MODULE_REFERENCE.md), [`ARCHITECTURE.md`](./ARCHITECTURE.md)
> - playground: [`design.md`](../../playground/design.md), [`planet_rendering.md`](../../playground/docs/planet_rendering.md), [`hexgrid_biome_plan.md`](../../playground/docs/hexgrid_biome_plan.md)

---

## Table of Contents

1. [Current State Audit](#1-current-state-audit)
2. [Integration Architecture](#2-integration-architecture)
3. [Data Model Gap Analysis](#3-data-model-gap-analysis)
4. [Tile ID Bridging](#4-tile-id-bridging)
5. [API Boundary Contract](#5-api-boundary-contract)
6. [Phased Roadmap](#6-phased-roadmap)
7. [Open Questions](#7-open-questions)

---

## 1. Current State Audit

### pgrid — what exists

| Capability | Module(s) | Status |
|-----------|-----------|--------|
| Goldberg globe construction | `globe/globe.py` | ✅ |
| Elevation generation (noise, mountains, smoothing) | `terrain/noise.py`, `heightmap.py`, `mountains.py` | ✅ |
| Region partitioning (angular, flood-fill, voronoi, noise) | `terrain/regions.py` | ✅ |
| Per-tile key-value storage with schema validation | `data/tile_data.py` | ✅ |
| Sub-tile detail grids + boundary blending | `detail/detail_terrain.py`, `tile_detail.py` | ✅ |
| Satellite-style rendering with BiomeConfig palettes | `detail/detail_render.py`, `terrain/terrain_render.py` | ✅ |
| Texture atlas + UV mapping | `detail/texture_pipeline.py` | ✅ |
| Globe JSON export (geometry, elevation, colour, adjacency) | `globe/globe_export.py` | ✅ |
| River generation (flow accumulation, watershed) | `terrain/rivers.py` | ✅ |
| GPU renderer with water, normals, LOD, atmosphere | `globe_renderer_v2.py` | ✅ |
| Temperature / moisture fields | `terrain/temperature.py`, `terrain/moisture.py` | ✅ |
| Terrain classification rules (priority table) | `terrain/classification.py` | ✅ |
| Feature detection (coast, lake, forest) | `terrain/features.py` | ✅ |
| Tile ID bridging (face_id ↔ tile_slug) | `globe/globe.py` | ✅ |
| Well-known biome fields in export schema | `globe/globe_export.py` | ✅ |
| Public API — `generate_planet()` orchestrator | `integration.py` | ✅ |
| Any awareness of playground's world model | — | ❌ (by design) |

### playground — what exists

| Capability | Module(s) | Status |
|-----------|-----------|--------|
| World → System → Planet → Region hierarchy | `app/world_models/` | ✅ |
| Biome catalog (11 classifications, colours, patterns) | `app/world_models/biome.py` | ✅ |
| Region painting (brush, tile assignment) | `controllers/region_paint_controller.py` | ✅ |
| Per-tile Goldberg rendering (PlanetTileController) | `opengl/rendering/` | ✅ |
| Biome texture generation + GPU cache | `BiomeTileRenderer`, `BiomeTextureCache` | ✅ |
| Tile topology service (slug ↔ index, neighbours) | `controllers/tile_topology_service.py` | ✅ |
| Hexgrid sub-tile grids (Phases 0–4 of hexgrid plan) | `opengl/hextiles/` | ✅ |
| Planet-level biome attributes (water_abundance, roughness, etc.) | `app/world_models/planet.py` | ✅ |
| Region modifiers (elevation, moisture, feature weights) | `app/world_models/region.py` | ✅ |
| Star heat for temperature derivation | `app/world_models/system.py` | ✅ |
| Any call into pgrid's generation pipeline | — | ❌ (Phase 5) |

### Shared dependency: `models` library

Both repos depend on the `models` library for Goldberg polyhedron geometry. This is the existing common ground:

- `GoldbergPolyhedron`, `GoldbergTile`, `GoldbergIndex`
- `generate_goldberg_tiles()` → tile ID, vertices, normals, tangent basis
- `tile_slug()` → canonical `freq:f<face>:i-j-k` identifiers
- `ShapeMesh` → GPU-ready vertex/index buffers

---

## 2. Integration Architecture

### High-level data flow

```
playground (authoring)             pgrid (generation)               playground (rendering)
─────────────────────              ──────────────────               ───────────────────────
User sets planet/region    ──►     Receives planet params    ──►   Receives enriched
attributes in the UI               + region modifiers               tile payload
                                        │
                                   Runs generation pipeline:
                                   elevation → temperature →
                                   moisture → terrain → features
                                        │
                                   Returns per-tile data
                                   (terrain, features, colours,
                                    elevation, temperature, moisture)
```

### Dependency direction

```
playground ──depends-on──► pgrid  (generation API)
playground ──depends-on──► models (geometry)
pgrid      ──depends-on──► models (geometry)
pgrid      ──does-NOT-depend-on──► playground
```

pgrid is a **pure library** — it never imports playground code. playground calls pgrid's public API, passing planet/region parameters and receiving enriched tile data.

### Integration surface

The integration happens through a single, well-defined contract:

1. **Input:** playground packages planet + region parameters into a request dict.
2. **Processing:** pgrid runs its generation pipeline on the Goldberg globe.
3. **Output:** pgrid returns an enriched payload (per-tile terrain, features, colours, etc.).
4. **Consumption:** playground maps the payload onto its rendering pipeline.

---

## 3. Data Model Gap Analysis

### 3.1 Planet model gaps

playground's `Planet` model (`app/world_models/planet.py`) needs new fields to feed pgrid's pipeline.

| Field | Type | Default | Purpose | pgrid consumer |
|-------|------|---------|---------|----------------|
| `water_abundance` | `float` | 0.5 | Global water level threshold + moisture base | `BiomeConfig.water_level`, terrain classification |
| `roughness` | `float` | 0.5 | Elevation noise amplitude scaling | Noise amplitude in heightmap generation |
| `temperature` | `float` | 0.5 | Base planetary temperature (0=frozen, 1=scorching) | Terrain classification, feature placement |
| `minerals` | `list[str]` | `[]` | Mineral palette (`iron`, `copper`, `crystal`, `volcanic`) | BiomeConfig colour tints |

> `temperature` can be directly set or derived from `star_heat × star_distance`. Derivation is lower-priority (see Biome Bible decision #2).

### 3.2 Region model gaps

playground's `Region` model (`app/world_models/region.py`) needs modifier fields.

| Field | Type | Default | Purpose | pgrid consumer |
|-------|------|---------|---------|----------------|
| `elevation_modifier` | `float` | 1.0 | Multiplicative scalar for elevation | `heightmap` pass |
| `humidity_modifier` | `float` | 1.0 | Multiplicative scalar for moisture | `moisture` pass |
| `feature_weights` | `dict[str, float]` | `{}` | Per-feature density biases, e.g. `{"forests": 1.5}` | Feature generation passes |

### 3.3 System/Star model gaps

playground's `System` model has visual star fields (size, glow, core_color, etc.) but no biome-relevant properties.

| Field | Type | Default | Purpose | pgrid consumer |
|-------|------|---------|---------|----------------|
| `star_heat` | `float` | 1.0 | Luminosity / energy output | Temperature derivation (lower priority) |

> `star_distance` already has a proxy — `Planet.orbit_distance`. We can derive a relative distance from that slider. No new field needed on System.

### 3.4 Biome classification alignment

playground has 11 `BiomeClassificationStyle` slugs. pgrid's Biome Bible defines 8 terrain types + 5 feature types. The mapping isn't 1:1 because playground's biomes blend terrain + features.

| playground slug | pgrid terrain | pgrid feature | Notes |
|----------------|---------------|---------------|-------|
| `ocean` | Ocean | — | Direct match |
| `coast` | any land | Coast | Coast is a feature in pgrid, a biome in playground |
| `tundra` | Tundra | — | Direct match |
| `mountain` | Mountains / Snow | — | Snow is a sub-type of mountain area |
| `desert` | Desert | — | Direct match |
| `wetlands` | Wetland | — | Direct match |
| `grassland` | Plains | — | Plains with low-to-moderate moisture |
| `savanna` | Plains | — | Plains with low moisture + high temperature |
| `temperate_forest` | Plains / Hills | Forest | Terrain + forest feature combo |
| `tropical_rainforest` | Plains / Hills | Forest | Terrain + forest feature + high temp/moisture |
| `wasteland` | — | — | No pgrid equivalent yet (backlogged) |

**Bridging strategy:** pgrid returns `terrain` + `features` per tile. A mapping function (owned by the integration layer) converts `(terrain, features, temperature, moisture)` → playground `BiomeClassificationStyle` slug. This runs on the playground side so pgrid stays agnostic to playground's biome catalog.

### 3.5 Globe export payload gaps

pgrid's current `globe_payload.json` exports per tile:
- `id`, `face_type`, `models_tile_id`
- `vertices_3d`, `center_3d`, `normal_3d`
- `latitude_deg`, `longitude_deg`
- `elevation`, `color`
- `neighbor_ids`

**Needs to also export:**
- `temperature` (float)
- `moisture` (float)
- `terrain` (string — terrain classification)
- `features` (list of strings)
- `region_id` (string — which region this tile belongs to)

The schema's `additionalProperties: true` on tiles makes this backward-compatible.

---

## 4. Tile ID Bridging

This is the most critical integration seam.

### The problem

| Repo | ID format | Example | Source |
|------|-----------|---------|--------|
| pgrid | `t<index>` (face index) | `t0`, `t5`, `t41` | `PolyGrid` face numbering |
| playground / models | `freq:f<face>:i-j-k` (Goldberg slug) | `2:f0:0-0-0` | `GoldbergPolyhedron.tile_slug()` |

The same physical tile has different IDs in each system. They are deterministic and 1:1 — every pgrid face maps to exactly one models tile — but the mapping must be explicit.

### The bridge

pgrid already exports `models_tile_id` (integer) per tile. The models library's `GoldbergPolyhedron` can resolve integer ID → slug. The bridge is:

```
pgrid face_id (t0) ──► models_tile_id (0) ──► GoldbergPolyhedron.tile_slug(0) ──► "2:f0:0-0-0"
```

**Implementation:** The integration layer (or pgrid's export) should include the canonical Goldberg slug in the payload so playground never needs to reverse-engineer the mapping.

**Action:** Add `tile_slug` field to the globe export payload.

---

## 5. API Boundary Contract

### pgrid public API for integration

pgrid should expose a clean, high-level function that playground calls. All the internal pipeline complexity is hidden.

```python
# pgrid/integration.py (proposed)

@dataclass
class PlanetParams:
    """Parameters playground sends to pgrid."""
    layout: str                    # "gb3", "gb4", etc.
    seed: int                      # Deterministic generation seed
    water_abundance: float         # 0.0–1.0
    roughness: float               # 0.0–1.0
    temperature: float             # 0.0–1.0
    minerals: list[str]            # e.g. ["iron", "crystal"]
    regions: list[RegionParams]    # One per painted region

@dataclass
class RegionParams:
    """Per-region modifier set."""
    region_id: str                 # playground's region UUID
    tile_slugs: list[str]          # Goldberg slugs assigned to this region
    elevation_modifier: float      # Multiplicative, default 1.0
    humidity_modifier: float       # Multiplicative, default 1.0
    feature_weights: dict[str, float]  # e.g. {"forests": 1.5}
    biome_hint: str | None         # Optional playground classification slug

@dataclass
class TileResult:
    """Per-tile output from pgrid's pipeline."""
    tile_slug: str                 # Canonical Goldberg slug
    face_id: str                   # pgrid face ID (t0, t1, ...)
    face_type: str                 # "pent" or "hex"
    elevation: float
    temperature: float
    moisture: float
    terrain: str                   # pgrid terrain classification
    features: list[str]            # e.g. ["coast", "forest"]
    color: tuple[float, float, float]  # RGB [0,1]
    region_id: str | None          # Which region this tile belongs to

@dataclass
class GenerationResult:
    """Full result from a generation run."""
    tiles: list[TileResult]
    metadata: dict                 # Seed, frequency, stats, etc.

def generate_planet(params: PlanetParams) -> GenerationResult:
    """
    Run the full terrain generation pipeline for a planet.

    This is the single entry point playground calls.
    """
    ...
```

### playground consumption API

playground's integration adapter converts the result into its rendering pipeline:

```python
# playground integration adapter (proposed sketch)

def apply_generation_result(planet: Planet, result: GenerationResult) -> None:
    """Map pgrid output onto playground's planet tile data."""
    for tile in result.tiles:
        slug = tile.tile_slug
        biome_slug = terrain_to_biome_classification(
            terrain=tile.terrain,
            features=tile.features,
            temperature=tile.temperature,
            moisture=tile.moisture,
        )
        # Update planet tile snapshot, biome texture cache, etc.
        ...

def terrain_to_biome_classification(
    terrain: str,
    features: list[str],
    temperature: float,
    moisture: float,
) -> str:
    """Convert pgrid terrain + features → playground BiomeClassificationStyle slug."""
    ...
```

---

## 6. Phased Roadmap

### Phase 0 — Foundation (both repos) ✅

> **Goal:** Fill data model gaps and establish the tile ID bridge. No generation logic yet.
>
> **Completed.** All tasks done; all tests pass (90 pgrid, 33 playground).

| # | Task | Repo | Effort | Status |
|---|------|------|--------|--------|
| 0.1 | Add `water_abundance`, `roughness`, `temperature`, `minerals` fields to `Planet` model + UI sliders | playground | S | ✅ |
| 0.2 | Add `elevation_modifier`, `humidity_modifier`, `feature_weights` fields to `Region` model + UI controls | playground | S | ✅ |
| 0.3 | Add `star_heat` field to `System` model | playground | XS | ✅ |
| 0.4 | Add `tile_slug` to globe export payload (resolve via `models_tile_id` → `GoldbergPolyhedron`) | pgrid | XS | ✅ |
| 0.5 | Add `temperature`, `moisture`, `terrain`, `features`, `region_id` to globe export schema + payload | pgrid | S | ✅ |
| 0.6 | Write tile-ID bridging utility: `face_id ↔ tile_slug` lookup table per layout | pgrid | XS | ✅ |

<details>
<summary>Phase 0 implementation notes</summary>

- **0.1:** Planet fields use `int` sliders (0–100) with `_FIELD_BOUNDS` validation, matching existing planet model conventions. `minerals` is a `tuple[str, ...]` — no UI control yet (future custom widget).
- **0.2:** Region fields use `int` sliders (25–200) for modifiers. `feature_weights` is a `dict[str, float]` — no UI control yet (future custom widget). Added `REGION_FIELD_CHANGED` event to `event_names.py`.
- **0.3:** `star_heat` is an `int` slider (0–100) in a new "SystemTerrain" UI section.
- **0.4 + 0.6:** `GlobeGrid.tile_slug(face_id)`, `build_slug_lookup()`, and `build_reverse_slug_lookup()` added to `globe.py`. Export payload includes `tile_slug` per tile.
- **0.5:** Well-known biome fields auto-extracted from `TileDataStore` when present, defaulting to `null`. `features` stored as comma-separated string in store, split to list in export.
- **Pre-existing test failure:** `tests/test_planet_session_service.py::test_service_builds_planet_snapshots_payloads` fails due to `layout="custom"` being normalized to `"gb3"` — not caused by our changes.
</details>

### Phase 1 — Temperature & Moisture Generation (pgrid) ✅

> **Goal:** pgrid generates temperature and moisture fields per tile, stored in `TileDataStore`.
>
> **Completed.** Both generators implemented and tested (26 new tests, 642 total pass).

| # | Task | Repo | Effort | Status |
|---|------|------|--------|--------|
| 1.1 | Implement temperature field generation: planet baseline + latitude gradient + elevation lapse rate | pgrid | M | ✅ |
| 1.2 | Implement moisture field generation: water_abundance × region modifier, distance-to-ocean, elevation factor | pgrid | M | ✅ |
| 1.3 | Add `temperature` and `moisture` to `TileSchema` defaults | pgrid | XS | ✅ (consumers add to their schema) |
| 1.4 | Unit tests for temperature/moisture with known inputs | pgrid | S | ✅ |

<details>
<summary>Phase 1 implementation notes</summary>

- **Temperature** (`terrain/temperature.py`):
  - `compute_temperature(latitude_deg, elevation, planet_temperature)` — pure function, single tile.
  - `generate_temperature_field(grid, store, planet_temperature)` — grid-wide, reads `latitude_deg` from face metadata and `elevation` from store.
  - Formula: `temp = planet_temp × (1 − lat_weight) + cos(lat) × lat_weight − lapse_rate × elevation`, clamped [0, 1].
  - Default constants: `LATITUDE_WEIGHT = 0.5`, `LAPSE_RATE = 0.6`.
  - Ocean tiles use elevation = 0.0 for lapse-rate (surface temperature, not seabed).
- **Moisture** (`terrain/moisture.py`):
  - `compute_ocean_distance(grid, store)` — BFS from ocean tiles, returns `{face_id: hop_count}`.
  - `generate_moisture_field(grid, store, water_abundance)` — grid-wide.
  - Formula: `moisture = base × (1 − ocean_wt) + proximity × ocean_wt − elev_penalty × elevation`, clamped [0, 1].
  - Ocean tiles get moisture = 1.0 unconditionally.
  - Supports `region_humidity` dict + `region_field` for per-region scaling.
  - Default constants: `OCEAN_PROXIMITY_WEIGHT = 0.35`, `ELEVATION_PENALTY = 0.25`, `MAX_OCEAN_DISTANCE = 20`.
- **Schema:** Temperature/moisture fields are added by consumers to their `TileSchema` (not hard-coded). This follows the existing pattern where `elevation` is also added per-use.
- Both modules are re-exported from `polygrid.terrain` and top-level `polygrid`.
</details>

### Phase 2 — Terrain Classification (pgrid) ✅

> **Goal:** pgrid classifies each tile into a terrain type using the priority table from the Biome Bible.
>
> **Completed.** Classification module implemented with all 8 terrain types, 27 new tests (669 total pass).

| # | Task | Repo | Effort | Status |
|---|------|------|--------|--------|
| 2.1 | Implement terrain classification function (priority-ordered rules: Ocean → Snow → Tundra → Mountains → Desert → Wetland → Hills → Plains) | pgrid | M | ✅ |
| 2.2 | Store `terrain` field in `TileDataStore` | pgrid | XS | ✅ |
| 2.3 | Unit tests with edge-case tiles (all terrain types, boundary conditions) | pgrid | S | ✅ |

<details>
<summary>Phase 2 implementation notes</summary>

- **Module:** `terrain/classification.py` with `classify_tile()` (pure function) and `generate_terrain_field()` (grid-wide).
- **Priority order:** Ocean → Snow → Tundra → Mountains → Desert → Wetland → Hills → Plains — first match wins.
- **Thresholds** are module-level constants (e.g. `MOUNTAIN_THRESHOLD = 0.65`, `SNOW_ELEVATION = 0.75`) for easy tuning.
- All 8 terrain types are reachable and tested with explicit edge cases plus a full-pipeline integration test.
- String constants exported: `OCEAN`, `SNOW`, `TUNDRA`, `MOUNTAINS`, `DESERT`, `WETLAND`, `HILLS`, `PLAINS`, `TERRAIN_TYPES`.
</details>

### Phase 3 — Feature Generation (pgrid) ✅

> **Goal:** pgrid detects and tags features (coast, lakes, forests) post-terrain.
>
> **Completed.** Feature module with coast, lake, and forest detection; 21 new tests (690 total pass).

| # | Task | Repo | Effort | Status |
|---|------|------|--------|--------|
| 3.1 | Implement coast detection (adjacency scan for land tiles neighbouring ocean) | pgrid | S | ✅ |
| 3.2 | Implement lake detection (depression fill / basin detection) | pgrid | M | ✅ |
| 3.3 | Implement forest placement (noise + threshold, masked by terrain/moisture/temperature) | pgrid | M | ✅ |
| 3.4 | Wire `rivers.py` into the feature pipeline (already exists, needs integration) | pgrid | S | ⏳ (rivers.py does not exist yet — deferred) |
| 3.5 | Store `features` list in `TileDataStore` (may need new list dtype or comma-separated string) | pgrid | S | ✅ (comma-separated string) |
| 3.6 | Apply `feature_weights` from region params to scale densities | pgrid | S | ✅ (`forest_weight` param) |
| 3.7 | Unit tests for each feature type | pgrid | M | ✅ |

<details>
<summary>Phase 3 implementation notes</summary>

- **Module:** `terrain/features.py` with per-feature functions and a `generate_features()` orchestrator.
- **Features stored** as comma-separated strings in TileDataStore (`"coast,forest"`). Helper functions `get_features()` and `add_feature()` manage the parsing.
- **Coast:** Adjacency scan — any land tile with ≥1 ocean neighbour gets `"coast"`.
- **Lake:** Simplified depression-fill — find local elevation minima, flood-fill up to threshold, discard clusters < 3 tiles.
- **Forest:** Hash-based noise scored by `forest_weight × (0.5 + moisture × 0.5)`, thresholded at 0.45. Only on Plains/Hills with moisture ≥ 0.4 and temperature 0.2–0.8.
- **Rivers:** Deferred — `rivers.py` doesn't exist yet. The feature pipeline has a clean extension point for it.
- **`_hash_noise_simple`:** Cheap deterministic hash for forest density; avoids the heavier `fbm_3d` for this use case.
</details>

### Phase 4 — Public API + Pipeline Orchestration (pgrid) ✅

> **Goal:** Expose the clean `generate_planet()` entry point.
>
> **Completed.** Integration module with full pipeline orchestrator, 41 new tests (731 total pass).

| # | Task | Repo | Effort | Status |
|---|------|------|--------|--------|
| 4.1 | Create `pgrid/integration.py` (or `pgrid/api.py`) with `PlanetParams`, `RegionParams`, `GenerationResult` dataclasses | pgrid | S | ✅ |
| 4.2 | Implement `generate_planet()` — orchestrate: build globe → assign regions → run pipeline passes → collect results | pgrid | M | ✅ |
| 4.3 | Accept `tile_slugs` in region params and resolve to pgrid face IDs internally | pgrid | S | ✅ |
| 4.4 | Integration tests: round-trip with known params → validate output fields | pgrid | M | ✅ |
| 4.5 | Updated globe export that includes all new fields | pgrid | S | ✅ (already done in Phase 0.5) |

<details>
<summary>Phase 4 implementation notes</summary>

- **Module:** `integration.py` at the top of the `polygrid` package (not inside `terrain/`), re-exported from `polygrid.__init__`.
- **Dataclasses:** `PlanetParams`, `RegionParams`, `TileResult`, `GenerationResult` — all frozen, with sensible defaults.
- **`parse_layout(layout)`:** Extracts Goldberg frequency from `"gb3"` / `"GB5"` / `"3"` strings. Validates ≥ 1.
- **Pipeline stages (12 steps):**
  1. Parse layout → build `GlobeGrid`
  2. Resolve regions via `build_reverse_slug_lookup()`
  3. Generate elevation (3-D FBM + ridged noise blended by `roughness`)
  4. Apply per-region elevation modifiers
  5. Normalise elevation to [0, 1]
  6. Compute water level from `water_abundance` (linear: `wl = wa × 0.6`)
  7. Generate temperature field
  8. Generate moisture field (with per-region humidity modifiers)
  9. Classify terrain types
  10. Detect features (coast, lakes, forests)
  11. Compute colours via `_ramp_satellite`
  12. Collect `TileResult` objects with all fields
- **Elevation noise:** `fbm_3d` (smooth) blended with `ridged_noise_3d` (mountains) using `roughness` as mix factor. Octaves scale with roughness (4–8). Uses 3-D noise on unit sphere — no seam/polar artefacts.
- **Determinism:** Same `PlanetParams` always produces identical output (tested).
- **Region modifiers:** `elevation_modifier` scales raw elevation pre-normalisation; `humidity_modifier` feeds into moisture generation; `feature_weights.forests` controls forest density.
- **Tests:** 41 tests covering layout parsing, water level, schema, round-trip pipeline, field ranges, determinism, region assignment, modifiers, coast logic, and top-level imports.
</details>

### Phase 5 — Playground Integration Adapter (playground) ✅

> **Goal:** playground calls pgrid and maps results to its rendering stack.
>
> **Completed.** Integration adapter, biome mapper, UI action, renderer support, and 52 new tests (570 total pass in playground).

| # | Task | Repo | Effort | Status |
|---|------|------|--------|--------|
| 5.1 | Add pgrid as a dependency (pip install or path-based) | playground | XS | ✅ |
| 5.2 | Implement `terrain_to_biome_classification()` mapper (see §3.4 table) | playground | S | ✅ |
| 5.3 | Implement integration adapter: collect planet/region params → call `generate_planet()` → map results to tile snapshot | playground | M | ✅ |
| 5.4 | Wire "Generate Terrain" button/action in planet mode UI | playground | S | ✅ |
| 5.5 | Update `PlanetTileController` / `BiomeTextureCache` to consume pgrid-derived biome assignments | playground | M | ✅ |
| 5.6 | Refresh planet rendering after generation completes | playground | S | ✅ |
| 5.7 | Integration tests with mock pgrid results | playground | M | ✅ |

<details>
<summary>Phase 5 implementation notes</summary>

- **5.1:** `pip install -e ../pgrid` — editable install, verified importable.
- **5.2:** `controllers/terrain_generation/biome_mapper.py` — priority-based rules: coast feature override → lake → wetlands → terrain base lookup → climate refinement (forest + temperature → tropical/temperate, hot + dry → savanna). All 11 playground slugs reachable.
- **5.3:** `controllers/terrain_generation/adapter.py`:
  - `_slider_to_float()` — int 0–100 → 0.0–1.0.
  - `_modifier_to_float()` — int 25–200 → multiplier (100 = 1.0).
  - `build_planet_params()` — collects planet/region model fields, derives deterministic seed from planet ID hash.
  - `apply_generation_result()` — maps `GenerationResult` tiles → `tile_biome_map` + `tile_metadata` dicts.
  - `run_terrain_generation()` — full orchestration: build params → generate → apply → return result dict.
  - Lazy pgrid import with `_ensure_pgrid()` guard and clear error message.
- **5.4:** Added `GENERATE_TERRAIN = "ui.planet.generate_terrain"` to `UIAction` enum. Added accent "Generate Terrain" button (globe icon) to planet mode bottom actions. Registered handler in `app_controller.py`.
- **5.4 handler:** `_handle_generate_terrain_action()` runs generation in background thread via `ThreadPoolExecutor`, shows progress/result banners, persists results to planet metadata (`generated_tile_biome_map`, `generated_tile_metadata`, `terrain_generation`), saves tile snapshot artifact, marks world dirty, refreshes planet session, and invalidates viewport caches.
- **5.5:** Extended `planet_renderer._tile_biomes_from_metadata()` to read `generated_tile_biome_map` from metadata as base layer (region paint overrides on top). Extended `_tile_metadata_from_regions()` to seed from `generated_tile_metadata` similarly.
- **5.6:** Handler calls `space_viewport.invalidate_planet_payload_cache()` + `viewport_bridge.refresh_planet_visibility()` + `planet_session.refresh(force=True)` to trigger full re-render.
- **5.7:** 52 tests in `tests/test_terrain_generation.py`:
  - 15 biome mapper tests (feature overrides, terrain base, climate refinement)
  - 11 slider/modifier conversion tests
  - 6 `build_planet_params` tests (basic planet, seed derivation, regions, minerals)
  - 4 `apply_generation_result` tests (mapping, metadata fields, empty)
  - 7 `run_terrain_generation` integration tests (end-to-end with real pgrid, determinism, water_abundance effect)
  - All 52 pass. 518 existing playground tests unchanged (11 pre-existing failures unrelated to our work).

**Files created:**
- `controllers/terrain_generation/__init__.py`
- `controllers/terrain_generation/biome_mapper.py`
- `controllers/terrain_generation/adapter.py`
- `tests/test_terrain_generation.py`

**Files modified:**
- `app/events/event_names.py` — `GENERATE_TERRAIN` action
- `app/modes/bottom_actions.py` — generate terrain button spec
- `app/modes/planet.py` — added button to planet mode
- `controllers/app_controller.py` — action registration + handler
- `opengl/rendering/planet_renderer.py` — generated biome map / metadata support
</details>

### Phase 6 — Detail Terrain (future)

> **Goal:** Sub-tile detail grids driven by pgrid's detail pipeline.

| # | Task | Repo | Effort |
|---|------|------|--------|
| 6.1 | Expose detail grid generation in pgrid's public API (per-tile detail terrain + atlas) | pgrid | L |
| 6.2 | Feed pgrid detail textures into playground's hexgrid texture pipeline | playground | L |
| 6.3 | Boundary blending between adjacent tiles at detail level | both | L |

### Dependency graph

```
Phase 0 (foundation)
  │
  ├──► Phase 1 (temperature + moisture)
  │       │
  │       └──► Phase 2 (terrain classification)
  │               │
  │               └──► Phase 3 (features)
  │                       │
  │                       └──► Phase 4 (public API)
  │                               │
  │                               └──► Phase 5 (playground adapter)
  │                                       │
  │                                       └──► Phase 6 (detail terrain)
  │
  └──► Phase 5.1–5.2 can start in parallel (adapter scaffolding + mapper)
```

---

## 7. Open Questions

| # | Question | Impact | Notes |
|---|----------|--------|-------|
| 1 | **Should pgrid be installed as a pip package or path-referenced?** | Phase 5.1 | ✅ Resolved: `pip install -e ../pgrid` editable install for co-development. |
| 2 | **Where does the integration adapter live?** | Phase 5 | ✅ Resolved: `controllers/terrain_generation/` in playground (option a). |
| 3 | **Async generation?** | Phase 5 | ✅ Resolved: Background `ThreadPoolExecutor` thread with UI dispatch back to main thread. Status banners show progress. |
| 4 | **Incremental regeneration?** | Phase 4–5 | Full regeneration for now. Optimise later. |
| 5 | **Region-less tiles?** | Phase 4 | ✅ Resolved: Tiles not assigned to any region use planet defaults (modifier = 1.0, no biome hint). |
| 6 | **Wasteland biome mapping?** | Phase 5.2 | No pgrid equivalent yet. Not reachable from terrain generation — manual override only. |
| 7 | **Detail terrain ownership?** | Phase 6 | pgrid generates data, playground renders it. Deferred. |
