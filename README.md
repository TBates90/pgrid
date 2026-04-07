# PolyGrid

PolyGrid is a topology-first toolkit for polygon-grid generation, terrain synthesis, atlas production, and Goldberg-globe rendering.

## Quick Start

```bash
# Install pgrid with development extras
pip install -e ".[dev,render,embed]"

# Required for globe-related tests and scripts
pip install -e ../models

# Fast iteration
pytest -m fast

# Full suite
pytest
```

For full testing commands and tier guidance, see [TESTING.md](TESTING.md).

## Globe Workflow

Step 1: build detail textures, atlas, UV layout, and payload.

```bash
python scripts/render_polygrids.py -f 3 --seed 42 -o exports/f3
```

Step 2: load export into the globe viewer.

```bash
python scripts/render_globe_from_tiles.py exports/f3
```

For debugging stages and deep rendering details, see [docs/RENDERING_PIPELINE.md](docs/RENDERING_PIPELINE.md) and [docs/TILE_TEXTURE_MAPPING.md](docs/TILE_TEXTURE_MAPPING.md).

## Topology CLI

```bash
polygrid build-pent --rings 3 --out pent.json --render-out pent.png
polygrid build-hex --rings 3 --out hex.json --render-out hex.png
polygrid assembly --rings 3 --out exports/assembly.png
```

## Documentation Map

| Doc | Scope |
|-----|-------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Canonical system architecture, package boundaries, phase-level design |
| [docs/RENDERING_PIPELINE.md](docs/RENDERING_PIPELINE.md) | High-level rendering pipeline and operational workflow |
| [docs/TILE_TEXTURE_MAPPING.md](docs/TILE_TEXTURE_MAPPING.md) | Deep technical reference for polygon-cut UV mapping and tile alignment |
| [docs/JSON_CONTRACT.md](docs/JSON_CONTRACT.md) | Export and interchange contracts (payload, UV layout, detail cells, seam manifests) |
| [docs/BIOME_BIBLE.md](docs/BIOME_BIBLE.md) | Terrain and biome classification rules |
| [docs/MODULE_REFERENCE.md](docs/MODULE_REFERENCE.md) | Concise module index and import-map reference |
| [TESTING.md](TESTING.md) | Test tiers, prerequisites, and run commands |

## Topic Ownership

- Architecture and boundaries: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Rendering workflow: [docs/RENDERING_PIPELINE.md](docs/RENDERING_PIPELINE.md)
- UV/polygon-cut implementation details: [docs/TILE_TEXTURE_MAPPING.md](docs/TILE_TEXTURE_MAPPING.md)
- Data contracts and schemas: [docs/JSON_CONTRACT.md](docs/JSON_CONTRACT.md)
- Module lookup: [docs/MODULE_REFERENCE.md](docs/MODULE_REFERENCE.md)
