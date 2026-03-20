import sys
from pathlib import Path
from functools import lru_cache

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ═══════════════════════════════════════════════════════════════════
# Auto-apply speed-tier markers based on test-file grouping
# ═══════════════════════════════════════════════════════════════════
#   pytest -m fast        → only fast tests  (~3s total)
#   pytest -m "not slow"  → skip the expensive globe tests
#   pytest -m medium      → mid-tier only

_FILE_TIER: dict[str, str] = {
    # Core — fast
    "test_core_topology.py": "fast",
    "test_tile_data.py": "fast",
    "test_noise.py": "fast",
    "test_heightmap.py": "fast",
    "test_mountains.py": "fast",
    # Globe — slow
    "test_globe.py": "slow",
    # Detail grids — medium
    "test_tile_detail.py": "medium",
    "test_detail_render.py": "medium",
    "test_detail_terrain.py": "medium",
    "test_grid_deformation.py": "medium",
    # Globe renderer — medium
    "test_globe_renderer_v2.py": "medium",
    # UV / atlas — fast
    "test_uv_texture.py": "fast",
    "test_atlas_seams.py": "fast",
    "test_corner_blend.py": "fast",
}


def pytest_collection_modifyitems(config, items):
    """Automatically tag every test item with fast / medium / slow."""
    for item in items:
        fname = Path(item.fspath).name
        tier = _FILE_TIER.get(fname, "fast")  # default unlisted files to fast
        item.add_marker(getattr(pytest.mark, tier))

# ═══════════════════════════════════════════════════════════════════
# Cached globe builders — shared across all test files
# ═══════════════════════════════════════════════════════════════════
# Building a Goldberg globe is expensive (~15-20s).  Tests that only
# *read* the globe/collection/store can share a single cached instance
# instead of rebuilding it from scratch every test method.
#
# Usage in test files:
#   from conftest import cached_build_globe, cached_build_globe_and_collection
#   globe = cached_build_globe(3)
#   globe, coll, store = cached_build_globe_and_collection(3, 3)

@lru_cache(maxsize=8)
def cached_build_globe(frequency: int = 3, radius: float = 1.0):
    """Build and cache a globe grid.  Safe for read-only tests."""
    try:
        from polygrid.globe import build_globe_grid
        return build_globe_grid(frequency, radius=radius)
    except ImportError:
        return None


# ═══════════════════════════════════════════════════════════════════
# Cached DetailGridCollection.build  (monkeypatch)
# ═══════════════════════════════════════════════════════════════════
# ``DetailGridCollection.build`` calls ``build_all_detail_grids``
# which runs an optimizer for each of 92 tiles — very expensive.
# We cache the resulting *grids dict* keyed on (globe_id, detail_rings, size)
# and return a fresh ``DetailGridCollection`` wrapper each time so that
# each test gets its own mutable ``_stores`` dict.

_collection_grids_cache: dict = {}

def _install_collection_cache():
    """Monkey-patch ``DetailGridCollection.build`` to cache grids."""
    try:
        from polygrid.tile_detail import DetailGridCollection, TileDetailSpec
    except ImportError:
        return  # polygrid not importable yet — skip

    _original_build = DetailGridCollection.build.__func__  # unwrap classmethod

    @classmethod
    def _cached_build(cls, globe_grid, spec=None, *, size=1.0):
        if spec is None:
            spec = TileDetailSpec()
        key = (id(globe_grid), spec.detail_rings, size)
        grids = _collection_grids_cache.get(key)
        if grids is None:
            # First time — do the real (expensive) build
            coll = _original_build(cls, globe_grid, spec, size=size)
            grids = coll._grids
            _collection_grids_cache[key] = grids
        # Return a fresh instance with shared grids, empty _stores
        return cls(globe_grid, spec, grids)

    DetailGridCollection.build = _cached_build

_install_collection_cache()


@lru_cache(maxsize=8)
def _cached_collection_internals(frequency: int = 3, detail_rings: int = 3):
    """Cache the *expensive* parts: globe, grids dict, spec, globe_store.

    Returns ``(globe, grids_dict, spec, globe_store)`` or ``None``.
    These are never mutated by tests so it is safe to cache them.
    """
    try:
        from polygrid.globe import build_globe_grid  # noqa: F401
    except ImportError:
        return None

    from polygrid import (
        DetailGridCollection,
        TileDetailSpec,
        TileDataStore,
        TileSchema,
        FieldDef,
    )
    from polygrid.heightmap import sample_noise_field
    from polygrid.noise import fbm

    globe = cached_build_globe(frequency)
    if globe is None:
        return None

    spec = TileDetailSpec(detail_rings=detail_rings)
    collection = DetailGridCollection.build(globe, spec)

    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    globe_store = TileDataStore(grid=globe, schema=schema)
    sample_noise_field(
        globe, globe_store, "elevation",
        lambda x, y: fbm(x, y, frequency=2.0, seed=42),
    )

    # Return the raw internals so we can build fresh collections
    return globe, collection._grids, spec, globe_store


def cached_build_globe_and_collection(frequency: int = 3, detail_rings: int = 3):
    """Build a globe + DetailGridCollection + globe store.

    The expensive grid-build is cached, but each call returns a
    **fresh** ``DetailGridCollection`` with an empty ``_stores``
    dict so that tests can mutate it safely without polluting the
    cache.

    Returns ``(globe, collection, globe_store)`` or ``None`` if
    the models library is not installed.
    """
    result = _cached_collection_internals(frequency, detail_rings)
    if result is None:
        return None

    globe, grids_dict, spec, globe_store = result

    from polygrid import DetailGridCollection
    # Build a fresh collection that shares the (immutable) grids
    # but has its own empty _stores dict.
    collection = DetailGridCollection(globe, spec, grids_dict)

    return globe, collection, globe_store
