# Testing quickguide

Scope:
- Test prerequisites, tiers, and run commands.

This repository has a test suite of **605 tests across 14 files**. Use these quick commands to avoid long runs while iterating.

Related docs:
- [README.md](README.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

Prerequisites
- Install the sibling `models` package in editable mode (required for globe tests):

```bash
# from the pgrid repo root, with the pgrid venv active
pip install -e ../models
```

Run only fast tests (recommended during development)

```bash
pytest -m fast
```

Run the full test-suite:

```bash
pytest
```

Run only the medium+fast but not slow tests:

```bash
pytest -m "not slow"
```

How tests are grouped
- **Fast**: quick unit tests — core topology, tile data, noise, heightmap, mountains, UV, atlas, corner blend
- **Medium**: moderately expensive — detail grids, detail render, detail terrain, grid deformation, globe renderer v2
- **Slow**: long-running — globe tests

Tiers are assigned automatically by `tests/conftest.py` based on filename.
If a particular test file is taking too long for your iteration flow, run it
explicitly or adjust its tier in `_FILE_TIER` in `tests/conftest.py`.
