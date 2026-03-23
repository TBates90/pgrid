# Biome Bible

---

## Resolved Decisions

> Decisions made from the initial design review — kept here as a reference.

| # | Topic | Decision |
|---|-------|----------|
| 1 | **Star/Planet hierarchy ownership** | Lives in `playground` (World → AstralBody → System → Planet → Region). pgrid receives planet/region-level parameters as inputs — it does not model the star/system hierarchy itself. Integration planning is a separate task (see backlog). |
| 2 | **Temperature model** | Physics-lite: inverse-square luminosity → equilibrium temperature → latitude bands → elevation lapse rate. Lower priority — implement after core terrain pipeline. |
| 3 | **Minerals** | Start simple: a small palette enum (`iron`, `copper`, `crystal`, `volcanic`) mapped to colour tints in `BiomeConfig`. Expand later if gameplay needs dictate. |
| 4 | **Region modifiers** | Multiplicative scalars as default. Expect some trial-and-error tuning; the API should make it easy to swap to additive if needed. |
| 5 | **Temperature modifier on regions** | Dropped for now. Temperature is derived from latitude + elevation + planet heat. May revisit for volcanic hotspots later. |
| 6 | **Feature weights** | Weight map per feature type, e.g. `{"forests": 1.5, "lakes": 0.5}`. Maps naturally to per-feature UI sliders in `playground`. |
| 7 | **Wetness vs Humidity** | Consolidated into a single **`moisture`** field per tile. Represents how wet the tile is overall — derived from planet `water_abundance` × region `humidity_modifier`, distance-to-ocean, and elevation. Drives lake/river frequency and marsh/wetland classification. No separate humidity vs wetness split needed. |
| 8 | **Wetland variants** | Ship with **Marsh** (renamed to **Wetland** in the terrain table to match playground's existing `BiomeClassificationStyle`). Future variants (swamp, bog, mangrove) noted in backlog. |
| 9 | **Cities** | Backlogged. pgrid reserves a `features` list field on tiles; `playground` owns placement. |
| 10 | **Snow / Tundra** | First-class terrain types: **Tundra** (frozen ground, low temp) and **Snow** (high elevation + low temp mountain caps). |
| 11 | **Volcanic terrain** | Backlogged. Potential future work with tectonic plate simulation. |

---

## Overview

This document defines the rules and data that drive **procedural biome, terrain, and feature generation** across the polygrid system. It covers:

- The **tiered attribute hierarchy** (Star → Planet → Region → Tile) and which values are user-set vs procedurally generated.
- **Terrain classification** rules — how tile attributes map to terrain types.
- **Feature generation** rules — how secondary features (rivers, forests, coasts, etc.) are placed after terrain is assigned.
- **Procedural generation pipeline** — the ordered sequence of generation passes.

### Design Principles

1. **Parent cascades down** — each tier inherits and is constrained by its parent's attributes.
2. **Procedural only at tile level** — Star, Planet, and Region attributes are user-set or randomised. Only TileBiome attributes are fully procedural.
3. **Elevation is king** — elevation is generated first; most other attributes (temperature, humidity, terrain class) can be derived from elevation + parent context.
4. **Separation of concerns** — biome _data_ lives in `TileData` / `TileDataStore`. Biome _rendering_ lives in `BiomeConfig` / `detail_render`. This document describes the data rules; rendering is documented in `TILE_TEXTURE_MAPPING.md`.

---

## Biome Attribute Hierarchy

The system uses a **four-tier hierarchy**. Each tier narrows the parameter space for the tier below it.

```
Star
 └─ Planet
     └─ Region (multiple per planet)
         └─ Tile (many per region — Goldberg face or detail sub-face)
```

### Tier 1 — StarBiome Attributes

Top-level properties of the star. These are **user-set or randomised** and influence planetary conditions.

| Attribute      | Type    | Range / Notes |
|----------------|---------|---------------|
| `star_size`    | `float` | Relative scale (e.g. 0.5 = red dwarf, 1.0 = solar, 2.0 = giant). Affects luminosity. |
| `star_heat`    | `float` | Luminosity / energy output (0.0–2.0). Combined with distance to derive planetary temperature. In reality this correlates with size and spectral class, but we treat it as an independent tuneable for creative freedom. |

### Tier 2 — PlanetBiome Attributes

Properties of a single planet. **User-set or randomised**, optionally influenced by star attributes.

| Attribute          | Type    | Range / Notes |
|--------------------|---------|---------------|
| `water_abundance`  | `float` | 0.0 (bone dry) – 1.0 (ocean world). Controls the global water level threshold and moisture availability. Maps to `BiomeConfig.water_level` for rendering. |
| `roughness`        | `float` | 0.0 (billiard ball) – 1.0 (extreme relief). Scales the amplitude of elevation noise — higher roughness = deeper oceans, taller mountains, more dramatic terrain. |
| `star_distance`    | `float` | Relative orbital distance (0.5 = scorching, 1.0 = habitable zone, 2.0 = frozen). Combined with `star_heat` to derive a base planetary temperature. |
| `temperature`      | `float` | 0.0 (frozen) – 1.0 (scorching). _Derived_ from `star_heat` and `star_distance`, or directly set by the user. Acts as the baseline for latitude-adjusted temperature bands. |
| `minerals`         | `list[str]` | Mineral groups present on the planet. Initial palette: `iron`, `copper`, `crystal`, `volcanic`. Mapped to colour tints in `BiomeConfig` for rendering. May expand to gameplay stats later. |

### Tier 3 — RegionBiome Attributes

Regions partition the globe (via `partition_noise`, `partition_voronoi`, etc. from `terrain/regions.py`). Each region has **modifier attributes** that shift the planet baseline. These are **user-set or randomised**, constrained by parent values.

| Attribute              | Type    | Range / Notes |
|------------------------|---------|---------------|
| `elevation_modifier`   | `float` | Multiplicative scalar (0.5 = flattened lowland, 1.0 = no change, 1.5 = exaggerated highlands). Applied to the base elevation noise for faces within this region. |
| `humidity_modifier`    | `float` | Multiplicative scalar for moisture availability. 0.5 = rain shadow / arid zone, 1.5 = lush jungle belt. Scales the planet's `water_abundance` for this region. |
| `feature_weights`      | `dict[str, float]` | Per-feature density biases, e.g. `{"forests": 1.5, "lakes": 0.5}`. Default 1.0 = no bias. Maps to per-feature UI sliders in `playground`. |

### Tier 4 — TileBiome Attributes (Procedurally Generated)

These are the per-face values stored in `TileDataStore` (see `data/tile_data.py`). They are **computed procedurally** from parent attributes + noise.

| Attribute     | Type        | Range / Notes | Source |
|---------------|-------------|---------------|--------|
| `elevation`   | `float`     | 0.0–1.0 normalised | Noise + mountains + smoothing (existing pipeline) |
| `temperature` | `float`     | 0.0 (frozen) – 1.0 (scorching) | Derived from latitude + planet temperature + elevation lapse rate |
| `moisture`    | `float`     | 0.0 (arid) – 1.0 (saturated) | Derived from planet `water_abundance` × region `humidity_modifier`, distance-to-ocean, elevation. Drives lake/river frequency and wetland classification. |
| `terrain`     | `str`       | Enum (see Terrain table below) | Classified from elevation + temperature + moisture |
| `features`    | `list[str]` | Feature tags (e.g. `["forest", "river"]`) | Assigned by feature generators post-terrain |

> **Mapping to existing code:** `TileSchema` already supports `float`, `str`, `int`, `bool` fields. The `elevation` field exists today. The additional fields (`temperature`, `moisture`, `terrain`, `features`) will need to be added to the schema. `features` will likely need a new `list` dtype or be stored as a comma-separated string.

> **Mapping to playground:** The existing `Biome` model in playground (`app/world_models/biome.py`) already has `elevation` and `moisture` sliders, plus a `classification` field with styles that closely align with the terrain types here (temperate_forest, tropical_rainforest, savanna, desert, tundra, wetlands, ocean, coast, grassland, mountain, wasteland). The terrain classification rules below should produce results compatible with playground's `BiomeClassificationStyle` slugs.

---

## Terrain Classification

Terrain is the **primary biome label** assigned to each tile based on its attributes. A tile has exactly one terrain type.

### Terrain Table

| Terrain       | Elevation        | Temperature    | Moisture       | Description |
|---------------|------------------|----------------|----------------|-------------|
| **Ocean**     | < `water_level`  | any            | any            | Below the global water line. Depth = `water_level − elevation`. |
| **Snow**      | > 0.75           | < 0.2          | any            | Permanent snow and ice caps on high mountain peaks. |
| **Tundra**    | any land         | < 0.15         | any            | Frozen ground. High latitude or high elevation + low temperature. Permafrost, sparse lichen. |
| **Mountains** | > 0.65           | any            | any            | High elevation, steep. Ridge lines from `generate_mountains()`. |
| **Desert**    | > `water_level`  | > 0.6          | < 0.2          | Hot and dry. Typically low-to-mid elevation. Sand dunes, cracked earth. |
| **Wetland**   | `water_level` – `water_level + 0.05` | > 0.3 | > 0.6 | Low-lying saturated land. Flat, near water level, warm and humid. |
| **Hills**     | 0.40 – 0.65     | any            | any            | Moderate elevation with rolling relief. Transitional between plains and mountains. |
| **Plains**    | 0.15 – 0.40     | 0.2 – 0.8     | 0.2 – 0.7     | Flat or gently rolling lowlands. The "default" land terrain. |

> **Classification priority** (highest → lowest): Ocean → Snow → Tundra → Mountains → Desert → Wetland → Hills → Plains. A tile that qualifies for multiple types takes the highest-priority match.

> **Alignment with playground:** These terrain types correspond to playground's `BiomeClassificationStyle` slugs: `ocean`, `tundra`, `mountain`, `desert`, `wetlands`, `grassland` (Plains), `coast` (feature), `savanna` (dry Plains variant), `temperate_forest` / `tropical_rainforest` (Plains/Hills + Forest feature). The mapping isn't 1:1 because playground biomes blend terrain + features — that bridging logic will be part of the integration work.

### Per-Terrain Detail

#### Ocean

- **Trigger:** `elevation < water_level` (planet-level attribute).
- **Sub-categories** (for rendering, not separate terrain types):
  - Deep ocean: `elevation < water_level × 0.3`
  - Mid ocean: `water_level × 0.3` – `water_level × 0.7`
  - Shallow water: `water_level × 0.7` – `water_level`
- **Rendering:** Blue colour ramp keyed on depth. Animated waves in the PBR shader. See `classify_water_tiles()` and `compute_water_depth()` in `globe_renderer_v2.py`.
- **Adjacency rule:** Ocean tiles should form contiguous bodies. Isolated single-tile ocean pockets should be smoothed away or promoted to lakes.

#### Mountains

- **Trigger:** `elevation > 0.65` (after noise + mountain generation).
- **Generation:** Produced by `generate_mountains()` with `MountainConfig` presets (`MOUNTAIN_RANGE`, `ALPINE_PEAKS`, `MESA_PLATEAU`).
- **Sub-variation by config:**
  - Sharp ridges → `ALPINE_PEAKS` (high frequency, low warp)
  - Broad ranges → `MOUNTAIN_RANGE` (high warp, foothill blend)
  - Stepped plateaus → `MESA_PLATEAU` (terrace steps > 0)
- **Temperature interaction:** High mountains have lower temperature (elevation lapse rate: ~−0.006 per unit elevation). Snow terrain takes over above `snow_line` when temperature < 0.2.
- **Adjacency rule:** Mountains should generally have Hills as a transitional buffer to Plains. Abrupt mountain-to-plains transitions look unnatural.

#### Snow

- **Trigger:** `elevation > 0.75` AND `temperature < 0.2`.
- **Character:** Permanent snow and ice caps on the highest peaks. A sub-zone of Mountains that gets its own terrain classification.
- **Rendering:** Bright white/blue palette. Maps to the top end of the existing `_RAMP_DETAIL_SATELLITE` colour ramp (snow field → bright snow above 0.85).
- **Adjacency rule:** Snow borders Mountains below and Tundra at lower elevations with matching low temperature.

#### Tundra

- **Trigger:** Temperature < 0.15, any land elevation.
- **Character:** Frozen ground, permafrost. Sparse lichen and moss. Found at high latitudes (polar regions) or at high elevation in cold zones.
- **Rendering:** Pale blue-grey palette. Sparse vegetation dots.
- **Adjacency rule:** Tundra borders Snow (upslope/colder) and Plains (equatorward/warmer). Should form latitude bands on the globe.

#### Hills

- **Trigger:** `elevation` in 0.40–0.65 range.
- **Generation:** Produced naturally by `ROLLING_HILLS` config or as the foothill fringe of mountain regions (`foothill_blend` in `MountainConfig`).
- **Character:** Gently undulating. Lower noise frequency than mountains.
- **Adjacency rule:** Hills typically border mountains (upslope) and plains (downslope). Can also surround isolated volcanic features.

#### Plains

- **Trigger:** Moderate elevation (0.15–0.40), moderate temperature, no extreme moisture.
- **Character:** The default land terrain — flat to gently rolling. Visual character shifts with moisture: dry plains → savanna/grassland; moist plains → lush meadow.
- **Rendering:** Vegetation density in `BiomeConfig` drives green vs brown balance.

#### Desert

- **Trigger:** Temperature > 0.6 AND moisture < 0.2 AND above water level.
- **Character:** Hot and dry. Can occur at any elevation but most common at low-to-mid.
- **Sub-types (rendering variation, not separate terrains):**
  - Sandy desert (low elevation, high temperature)
  - Rocky desert (mid elevation)
  - Salt flat (very low moisture, near former water)
- **Adjacency rule:** Deserts should transition through dry plains/savanna, not directly border lush jungle or wetland.

#### Wetland

- **Trigger:** Elevation just above water level (within ~0.05) AND high moisture (> 0.6) AND temperature > 0.3.
- **Character:** Waterlogged lowland. Transition zone between land and water.
- **Adjacency rule:** Wetlands should border ocean/lakes on one side and plains on the other. They act as a coastal or lakeside buffer.
- **Future variants** (backlogged):
  - Swamp — forested wetland (wetland + forest feature)
  - Bog — acidic peat, colder wetlands
  - Mangrove — coastal tropical wetland

---

## Features

Features are **secondary tags** applied to tiles _after_ terrain classification. A tile can have zero or more features. Features are generated by dedicated procedural passes that run sequentially after the terrain pass.

| Feature      | Applies To           | Generation Method | Description |
|--------------|----------------------|-------------------|-------------|
| **Coast**    | Land tiles adjacent to Ocean | Adjacency scan | Marks the land-sea boundary. Drives coastline foam rendering and coastal erosion visuals. |
| **Lake**     | Interior low-elevation clusters | Depression fill / basin detection | Inland water bodies. Tiles below a local threshold that aren't connected to the ocean. |
| **River**    | Land tiles, downhill paths | Flow accumulation / watershed | Linear water features following steepest-descent paths from high to low elevation. Confluence where tributaries meet. |
| **Forest**   | Plains, Hills (moisture > 0.4, temperature 0.2–0.8) | Noise-based density + clustering | Tree-covered areas. Density varies with moisture and temperature. Tropical (high temp + high moisture) vs temperate vs boreal (low temp). |
| **City**     | Plains, Hills (low gradient) | _Backlogged — playground-layer concern_ | Urban areas. pgrid reserves the `features` list field; `playground` owns placement logic. |

### Feature Generation Order

Features have dependencies — some can only be placed once others exist:

```
1. Coast       (requires: terrain classification with ocean identified)
2. Lakes       (requires: elevation field, terrain classification)
3. Rivers      (requires: elevation field, lake positions)
4. Forests     (requires: moisture, temperature, terrain — avoids ocean/desert/mountain/tundra)
5. Cities      (requires: all of the above — placed last, backlogged)
```

### Feature Rules

#### Coast
- **Detection:** Any land tile (terrain ≠ Ocean) with at least one Ocean neighbour.
- **Effect:** Adds `"coast"` to the tile's feature list. Rendering adds foam/sand fringe.
- **Width:** Could optionally extend 1–2 rings inland for a broader coastal zone.

#### Lakes
- **Detection:** Find local elevation minima on land that aren't connected to ocean. Fill depressions up to a spillover threshold.
- **Representation:** Tiles within a lake basin get terrain overridden to Ocean (or a dedicated "Lake" terrain) and `"lake"` added to features.
- **Size constraint:** Minimum 3 tiles, maximum ~5% of a region's land area.
- **Adjacency:** Lakes should be bordered by Coast-tagged tiles.

#### Rivers
- **Detection:** Flow accumulation — for each land tile, trace the steepest-descent neighbour. Tiles where accumulated flow exceeds a threshold are marked as river.
- **Representation:** `"river"` added to the tile's feature list. River tiles may have a `flow_direction` metadata entry.
- **Rules:**
  - Rivers flow downhill (strictly decreasing elevation along the path).
  - Rivers terminate at ocean, lake, or map edge.
  - Tributaries merge at confluence points — they don't cross.
  - River density scales with planet `water_abundance` and region `humidity_modifier` × `feature_weights["rivers"]`.

#### Forests
- **Detection:** Noise-based density field, thresholded and masked to suitable terrain.
- **Placement rules:**
  - Only on Plains or Hills terrain.
  - Moisture > 0.4 required.
  - Temperature 0.2–0.8 (no frozen tundra forests, no scorching desert forests).
  - Density varies: tropical forest (temp > 0.6, moisture > 0.7) = dense; boreal (temp 0.2–0.35) = sparse.
  - Density scaled by region `feature_weights["forests"]` (default 1.0).
- **Clustering:** Forests should form contiguous patches, not random salt-and-pepper noise. Use a smoothed noise field + threshold.

---

## Procedural Generation Pipeline

This section defines the **ordered sequence** of generation passes when building a globe.

### Globe-Level Pipeline (Goldberg Polyhedron Tiles)

When building the world initially, we operate on the **Goldberg polyhedron face level** (one value per hex/pent tile). This is the simpler, coarser pass.

```
Pass 1 — Elevation
  ├─ Sample 3D noise at tile centroids (fbm_3d / ridged_noise_3d)
  ├─ Apply mountain generation (generate_mountains per region)
  ├─ Apply region elevation_modifier (multiplicative)
  ├─ Smooth field (neighbour averaging, 2–3 iterations)
  ├─ Normalize to [0, 1]
  └─ Scale by planet roughness

Pass 2 — Temperature
  ├─ Base from planet temperature (derived from star_heat × star_distance)
  ├─ Latitude gradient (poles cold, equator hot)
  └─ Elevation lapse rate (higher = colder, ~−0.006 per unit elevation)

Pass 3 — Moisture
  ├─ Base from planet water_abundance
  ├─ Apply region humidity_modifier (multiplicative)
  ├─ Distance-to-ocean attenuation (further inland = drier)
  └─ Elevation factor (mountains create rain shadow)

Pass 4 — Terrain Classification
  └─ Apply terrain table rules (priority-ordered)

Pass 5 — Feature Generation
  ├─ Coast detection (adjacency scan)
  ├─ Lake detection (depression fill)
  ├─ River generation (flow accumulation)
  └─ Forest placement (noise + threshold, scaled by feature_weights)
```

> **Anti-hexagonal measures:** At the globe level, using 3D noise (`sample_noise_field_3d`) sampled at true Cartesian centroids avoids the hexagonal-grid artefacts that 2D noise on projected coordinates would produce. The existing `smooth_field` passes further soften cell boundaries.

### Region-Level Pipeline (Detail Sub-Grid)

When creating a region's detail view, we work at the **sub-tile polygrid level** (the fine-grained grid inside each Goldberg tile, from `detail/`).

```
Pass 1 — Detail Elevation
  ├─ Inherit parent tile elevation as baseline
  ├─ Add high-frequency noise for local variation
  ├─ Boundary blending (detail_terrain.py ensures continuity)
  └─ Smooth

Pass 2 — Detail Temperature / Moisture
  └─ Interpolate from parent tile values (no additional noise needed at this scale)

Pass 3 — Detail Terrain Classification
  └─ Same rules as globe level, applied to sub-faces

Pass 4 — Detail Features
  └─ Refine parent tile features at higher resolution
      (e.g. river course refined to specific sub-faces)
```

---

## Mapping to Existing Code

### pgrid

| Concept | Current Module | Status |
|---------|---------------|--------|
| Elevation generation | `terrain/noise.py`, `terrain/heightmap.py`, `terrain/mountains.py` | ✅ Implemented |
| Region partitioning | `terrain/regions.py` | ✅ Implemented |
| Biome assignment | `terrain/regions.py` → `assign_biome()` | ✅ Basic (string label only) |
| Per-tile data storage | `data/tile_data.py` | ✅ Implemented |
| Detail terrain | `detail/detail_terrain.py` | ✅ Implemented |
| Detail rendering | `detail/detail_render.py` → `BiomeConfig` | ✅ Implemented |
| Temperature field | — | ❌ Not yet implemented |
| Moisture field | — | ❌ Not yet implemented |
| Terrain classification rules | — | ❌ Not yet implemented (terrain is currently just a string label) |
| Feature generation (coast, lakes, rivers, forests) | — | ❌ Not yet implemented |
| Mineral colour tints | — | ❌ Not yet implemented |

### playground

| Concept | Current Module | Status |
|---------|---------------|--------|
| World → System → Planet → Region hierarchy | `app/world_models/` | ✅ Implemented |
| Biome catalog (classification, colours, patterns) | `app/world_models/biome.py` | ✅ Implemented |
| Region painting + tile assignment | `controllers/region_paint_controller.py` | ✅ Implemented |
| Planet rendering (per-tile Goldberg) | `opengl/rendering/`, `PlanetTileController` | ✅ Implemented |
| Biome texture generation | `BiomeTileRenderer`, `BiomeTextureCache` | ✅ Implemented |
| Star/System attributes (heat, distance) | `app/world_models/system.py` | ⚠️ Partial (size/glow/colour, no heat/distance for biome purposes) |
| Planet biome attributes (water_abundance, roughness) | `app/world_models/planet.py` | ❌ Not yet implemented |
| Region modifiers (elevation, moisture, feature weights) | `app/world_models/region.py` | ❌ Not yet implemented |
| pgrid ↔ playground integration (procedural terrain pipeline) | — | ❌ Not yet implemented |

---

## Backlog

Items deferred from this design for future work:

| Item | Notes |
|------|-------|
| **Volcanic terrain** | Potential tectonic plate simulation. Terrain type influenced by minerals + temperature. |
| **City placement** | `playground`-owned. pgrid reserves `features` field; placement logic TBD. |
| **Wetland variants** | Swamp (forested), bog (acidic/cold), mangrove (coastal tropical). |
| **Physics-lite temperature** | Inverse-square luminosity → equilibrium temp. Lower priority. |
| **Mineral gameplay stats** | Expand beyond colour tints if gameplay needs dictate. |
| **pgrid ↔ playground integration** | Major task: design the API boundary, refactor both repos, create integration pipeline. **See [`INTEGRATION_PLAN.md`](./INTEGRATION_PLAN.md)** for the full phased roadmap. |
