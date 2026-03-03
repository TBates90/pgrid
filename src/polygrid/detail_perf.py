"""Performance utilities for the detail grid pipeline.

Provides parallel generation, fast texture rendering, caching, and
benchmarking for sub-tile detail grids.

Functions
---------
- :func:`generate_all_detail_terrain_parallel` — threaded terrain gen
- :func:`render_detail_texture_fast` — PIL/numpy renderer (no matplotlib)
- :func:`build_detail_atlas_fast` — atlas with fast renderer
- :class:`DetailCache` — disk cache keyed by spec + elevation hash
- :func:`benchmark_pipeline` — end-to-end timing measurements
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .detail_render import BiomeConfig, detail_elevation_to_colour, _detail_hillshade
from .detail_terrain import (
    compute_boundary_elevations,
    generate_detail_terrain_bounded,
)
from .geometry import face_center
from .polygrid import PolyGrid
from .tile_data import FieldDef, TileDataStore, TileSchema
from .tile_detail import DetailGridCollection, TileDetailSpec


# ═══════════════════════════════════════════════════════════════════
# 10F.1 — Parallel detail terrain generation
# ═══════════════════════════════════════════════════════════════════

def generate_all_detail_terrain_parallel(
    collection: DetailGridCollection,
    globe_grid: PolyGrid,
    globe_store: TileDataStore,
    spec: Optional[TileDetailSpec] = None,
    *,
    seed: int = 42,
    elevation_field: str = "elevation",
    max_workers: Optional[int] = None,
) -> None:
    """Generate boundary-aware terrain for every tile using threads.

    Functionally identical to
    :func:`~polygrid.detail_terrain.generate_all_detail_terrain` but
    distributes per-tile work across a thread pool.

    Parameters
    ----------
    collection : DetailGridCollection
        Must contain pre-built grids.
    globe_grid : PolyGrid
    globe_store : TileDataStore
    spec : TileDetailSpec, optional
    seed : int
    elevation_field : str
    max_workers : int, optional
        Thread pool size (default: ``min(8, tile_count)``).
    """
    if spec is None:
        spec = collection.spec

    boundary_elevs = compute_boundary_elevations(
        globe_grid, globe_store, elevation_field=elevation_field,
    )

    grids = collection.grids
    face_ids = sorted(grids.keys())

    if max_workers is None:
        max_workers = min(8, len(face_ids))

    def _generate_one(face_id: str) -> Tuple[str, TileDataStore]:
        detail_grid = grids[face_id]
        parent_elev = globe_store.get(face_id, elevation_field)
        neighbor_elevs = boundary_elevs.get(face_id, {})
        store = generate_detail_terrain_bounded(
            detail_grid, parent_elev, neighbor_elevs, spec, seed=seed,
        )
        return face_id, store

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_one, fid): fid for fid in face_ids
        }
        for future in as_completed(futures):
            face_id, store = future.result()
            collection._stores[face_id] = store


# ═══════════════════════════════════════════════════════════════════
# 10F.2 — Fast PIL/numpy texture renderer (no matplotlib)
# ═══════════════════════════════════════════════════════════════════

def render_detail_texture_fast(
    detail_grid: PolyGrid,
    store: TileDataStore,
    output_path: Path | str,
    biome: Optional[BiomeConfig] = None,
    *,
    tile_size: int = 256,
    elevation_field: str = "elevation",
    noise_seed: int = 0,
) -> Path:
    """Render a detail grid to a PNG using PIL rasterisation.

    This is a faster alternative to
    :func:`~polygrid.detail_render.render_detail_texture_enhanced`
    that avoids matplotlib entirely.  Each sub-face polygon is
    rasterised using PIL's ``ImageDraw.polygon``.

    Parameters
    ----------
    detail_grid : PolyGrid
    store : TileDataStore
    output_path : Path or str
    biome : BiomeConfig, optional
    tile_size : int
        Output square image size in pixels.
    elevation_field : str
    noise_seed : int

    Returns
    -------
    Path
    """
    from PIL import Image, ImageDraw

    if biome is None:
        biome = BiomeConfig()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute bounding box of the grid
    xs, ys = [], []
    for v in detail_grid.vertices.values():
        if v.has_position():
            xs.append(v.x)
            ys.append(v.y)

    if not xs:
        # Empty grid
        img = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
        img.save(str(output_path))
        return output_path

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Add small padding
    pad = max(x_max - x_min, y_max - y_min) * 0.02
    x_min -= pad
    x_max += pad
    y_min -= pad
    y_max += pad

    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0

    # Scale to tile_size, keeping aspect ratio
    scale = tile_size / max(x_range, y_range)
    ox = (tile_size - x_range * scale) / 2.0
    oy = (tile_size - y_range * scale) / 2.0

    def _to_pixel(vx: float, vy: float) -> Tuple[float, float]:
        px = (vx - x_min) * scale + ox
        # Flip Y: image y increases downward
        py = tile_size - ((vy - y_min) * scale + oy)
        return (px, py)

    # Compute hillshade
    hs = _detail_hillshade(
        detail_grid, store, elevation_field,
        azimuth=biome.azimuth, altitude=biome.altitude,
    )

    # First pass: compute all face colours to get the average for
    # the background.  This ensures pixels outside the polygon are
    # terrain-coloured instead of black, eliminating seams in the
    # 3D atlas texture.
    face_colours = {}
    for fid, face in detail_grid.faces.items():
        has_verts = True
        for vid in face.vertex_ids:
            v = detail_grid.vertices.get(vid)
            if v is None or not v.has_position():
                has_verts = False
                break
        if has_verts and len(face.vertex_ids) >= 3:
            elev = store.get(fid, elevation_field)
            c = face_center(detail_grid.vertices, face)
            cx, cy = c if c else (0.0, 0.0)
            r, g, b = detail_elevation_to_colour(
                elev, biome,
                hillshade_val=hs.get(fid, 0.5),
                noise_x=cx, noise_y=cy,
                noise_seed=noise_seed,
            )
            face_colours[fid] = (r, g, b)

    if face_colours:
        avg_r = sum(c[0] for c in face_colours.values()) / len(face_colours)
        avg_g = sum(c[1] for c in face_colours.values()) / len(face_colours)
        avg_b = sum(c[2] for c in face_colours.values()) / len(face_colours)
        bg = (
            max(0, min(255, int(avg_r * 255))),
            max(0, min(255, int(avg_g * 255))),
            max(0, min(255, int(avg_b * 255))),
        )
    else:
        bg = (0, 0, 0)

    # Rasterise
    img = Image.new("RGB", (tile_size, tile_size), bg)
    draw = ImageDraw.Draw(img)

    for fid, face in detail_grid.faces.items():
        verts = []
        for vid in face.vertex_ids:
            v = detail_grid.vertices.get(vid)
            if v is None or not v.has_position():
                break
            verts.append(_to_pixel(v.x, v.y))
        else:
            if len(verts) >= 3 and fid in face_colours:
                r, g, b = face_colours[fid]
                colour = (
                    max(0, min(255, int(r * 255))),
                    max(0, min(255, int(g * 255))),
                    max(0, min(255, int(b * 255))),
                )
                draw.polygon(verts, fill=colour, outline=colour)

    img.save(str(output_path))
    return output_path


# ═══════════════════════════════════════════════════════════════════
# 10F.2b — Fast atlas builder (uses PIL renderer)
# ═══════════════════════════════════════════════════════════════════

def build_detail_atlas_fast(
    collection: DetailGridCollection,
    biome: Optional[BiomeConfig] = None,
    output_dir: Path | str = Path("exports/detail_tiles"),
    *,
    tile_size: int = 256,
    columns: int = 0,
    noise_seed: int = 0,
    gutter: int = 4,
) -> Tuple[Path, Dict[str, Tuple[float, float, float, float]]]:
    """Build a detail atlas using the fast PIL renderer.

    Same interface as :func:`~polygrid.texture_pipeline.build_detail_atlas`
    but uses :func:`render_detail_texture_fast` instead of the
    matplotlib-based renderer.

    Returns
    -------
    tuple
        ``(atlas_path, uv_layout)``
    """
    from PIL import Image
    from .texture_pipeline import _fill_gutter

    if biome is None:
        biome = BiomeConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    face_ids = collection.face_ids
    n = len(face_ids)
    if n == 0:
        raise ValueError("No detail grids in the collection")

    # Render individual tiles
    tile_paths: Dict[str, Path] = {}
    for fid in face_ids:
        grid, store = collection.get(fid)
        if store is None:
            raise ValueError(
                f"No terrain store for face '{fid}' — "
                "call generate_all_detail_terrain first"
            )
        path = output_dir / f"tile_{fid}.png"
        render_detail_texture_fast(
            grid, store, path, biome,
            tile_size=tile_size,
            noise_seed=noise_seed + hash(fid) % 10000,
        )
        tile_paths[fid] = path

    # Compute atlas layout — each slot is tile_size + 2*gutter
    if columns <= 0:
        columns = max(1, math.isqrt(n))
        if columns * columns < n:
            columns += 1
    rows = math.ceil(n / columns)

    slot_size = tile_size + 2 * gutter
    atlas_w = columns * slot_size
    atlas_h = rows * slot_size
    atlas = Image.new("RGB", (atlas_w, atlas_h), (128, 128, 128))

    uv_layout: Dict[str, Tuple[float, float, float, float]] = {}

    for idx, fid in enumerate(face_ids):
        col = idx % columns
        row = idx // columns
        slot_x = col * slot_size
        slot_y = row * slot_size

        tile_img = Image.open(tile_paths[fid]).convert("RGB")
        tile_img = tile_img.resize((tile_size, tile_size), Image.LANCZOS)
        atlas.paste(tile_img, (slot_x + gutter, slot_y + gutter))

        if gutter > 0:
            _fill_gutter(atlas, slot_x, slot_y, tile_size, gutter)

        inner_x = slot_x + gutter
        inner_y = slot_y + gutter
        u_min = inner_x / atlas_w
        u_max = (inner_x + tile_size) / atlas_w
        v_min = 1.0 - (inner_y + tile_size) / atlas_h
        v_max = 1.0 - inner_y / atlas_h
        uv_layout[fid] = (u_min, v_min, u_max, v_max)

    atlas_path = output_dir / "detail_atlas.png"
    atlas.save(str(atlas_path))

    return atlas_path, uv_layout


# ═══════════════════════════════════════════════════════════════════
# 16A — Full-slot atlas builder (uses tile_texture renderer)
# ═══════════════════════════════════════════════════════════════════

def build_detail_atlas_fullslot(
    collection: DetailGridCollection,
    biome: Optional[BiomeConfig] = None,
    output_dir: Path | str = Path("exports/detail_tiles"),
    *,
    tile_size: int = 256,
    columns: int = 0,
    noise_seed: int = 0,
    gutter: int = 4,
    k_neighbours: int = 4,
    overscan: float = 0.15,
) -> Tuple[Path, Dict[str, Tuple[float, float, float, float]]]:
    """Build a detail atlas using the full-slot renderer (Phase 16A).

    Same interface as :func:`build_detail_atlas_fast` but uses
    :func:`~polygrid.tile_texture.render_detail_texture_fullslot`
    which fills every pixel with coherent terrain colour —
    no flat-fill background.

    Returns
    -------
    tuple
        ``(atlas_path, uv_layout)``
    """
    from PIL import Image
    from .texture_pipeline import _fill_gutter
    from .tile_texture import render_detail_texture_fullslot

    if biome is None:
        biome = BiomeConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    face_ids = collection.face_ids
    n = len(face_ids)
    if n == 0:
        raise ValueError("No detail grids in the collection")

    # Render individual tiles with full-slot renderer
    tile_paths: Dict[str, Path] = {}
    for fid in face_ids:
        grid, store = collection.get(fid)
        if store is None:
            raise ValueError(
                f"No terrain store for face '{fid}' — "
                "call generate_all_detail_terrain first"
            )
        path = output_dir / f"tile_{fid}.png"
        render_detail_texture_fullslot(
            grid, store, path, biome,
            tile_size=tile_size,
            noise_seed=noise_seed + hash(fid) % 10000,
            k_neighbours=k_neighbours,
            overscan=overscan,
        )
        tile_paths[fid] = path

    # Compute atlas layout
    if columns <= 0:
        columns = max(1, math.isqrt(n))
        if columns * columns < n:
            columns += 1
    rows = math.ceil(n / columns)

    slot_size = tile_size + 2 * gutter
    atlas_w = columns * slot_size
    atlas_h = rows * slot_size
    atlas = Image.new("RGB", (atlas_w, atlas_h), (128, 128, 128))

    uv_layout: Dict[str, Tuple[float, float, float, float]] = {}

    for idx, fid in enumerate(face_ids):
        col = idx % columns
        row = idx // columns
        slot_x = col * slot_size
        slot_y = row * slot_size

        tile_img = Image.open(tile_paths[fid]).convert("RGB")
        tile_img = tile_img.resize((tile_size, tile_size), Image.LANCZOS)
        atlas.paste(tile_img, (slot_x + gutter, slot_y + gutter))

        if gutter > 0:
            _fill_gutter(atlas, slot_x, slot_y, tile_size, gutter)

        inner_x = slot_x + gutter
        inner_y = slot_y + gutter
        u_min = inner_x / atlas_w
        u_max = (inner_x + tile_size) / atlas_w
        v_min = 1.0 - (inner_y + tile_size) / atlas_h
        v_max = 1.0 - inner_y / atlas_h
        uv_layout[fid] = (u_min, v_min, u_max, v_max)

    atlas_path = output_dir / "detail_atlas.png"
    atlas.save(str(atlas_path))

    return atlas_path, uv_layout


# ═══════════════════════════════════════════════════════════════════
# 10F.4 — Detail grid / texture caching
# ═══════════════════════════════════════════════════════════════════

class DetailCache:
    """Disk-backed cache for generated detail terrain stores.

    Keys are derived from ``(face_id, spec, parent_elevation)`` so
    that regeneration only happens when inputs change.

    Parameters
    ----------
    cache_dir : Path or str
        Directory where cache files are stored.
    """

    def __init__(self, cache_dir: Path | str = Path("exports/.detail_cache")):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_key(
        face_id: str,
        spec: TileDetailSpec,
        parent_elevation: float,
        seed: int,
    ) -> str:
        """Produce a hash key from the cache inputs."""
        data = {
            "face_id": face_id,
            "spec": asdict(spec),
            "parent_elevation": round(parent_elevation, 8),
            "seed": seed,
        }
        raw = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def _path_for(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def has(
        self,
        face_id: str,
        spec: TileDetailSpec,
        parent_elevation: float,
        seed: int,
    ) -> bool:
        """Check if a cached store exists for these inputs."""
        key = self._make_key(face_id, spec, parent_elevation, seed)
        return self._path_for(key).exists()

    def get(
        self,
        face_id: str,
        spec: TileDetailSpec,
        parent_elevation: float,
        seed: int,
        detail_grid: PolyGrid,
    ) -> Optional[TileDataStore]:
        """Load a cached store, or return ``None`` if miss."""
        key = self._make_key(face_id, spec, parent_elevation, seed)
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            schema = TileSchema([FieldDef("elevation", float, 0.0)])
            store = TileDataStore(grid=detail_grid, schema=schema)
            for fid, elev in data["elevations"].items():
                store.set(fid, "elevation", elev)
            return store
        except (json.JSONDecodeError, KeyError):
            return None

    def put(
        self,
        face_id: str,
        spec: TileDetailSpec,
        parent_elevation: float,
        seed: int,
        store: TileDataStore,
        detail_grid: PolyGrid,
    ) -> None:
        """Write a store to the cache."""
        key = self._make_key(face_id, spec, parent_elevation, seed)
        elevations = {}
        for fid in detail_grid.faces:
            elevations[fid] = store.get(fid, "elevation")
        data = {"elevations": elevations}
        self._path_for(key).write_text(json.dumps(data))

    def clear(self) -> int:
        """Remove all cached files.  Returns the number removed."""
        count = 0
        for p in self._dir.glob("*.json"):
            p.unlink()
            count += 1
        return count

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(list(self._dir.glob("*.json")))


def generate_all_detail_terrain_cached(
    collection: DetailGridCollection,
    globe_grid: PolyGrid,
    globe_store: TileDataStore,
    spec: Optional[TileDetailSpec] = None,
    *,
    seed: int = 42,
    elevation_field: str = "elevation",
    cache: Optional[DetailCache] = None,
) -> int:
    """Generate terrain with cache lookup, returning cache-hit count.

    Falls back to serial generation for cache misses.

    Parameters
    ----------
    Returns
    -------
    int
        Number of cache hits.
    """
    if spec is None:
        spec = collection.spec
    if cache is None:
        cache = DetailCache()

    boundary_elevs = compute_boundary_elevations(
        globe_grid, globe_store, elevation_field=elevation_field,
    )

    hits = 0
    for face_id, detail_grid in collection.grids.items():
        parent_elev = globe_store.get(face_id, elevation_field)
        neighbor_elevs = boundary_elevs.get(face_id, {})

        cached = cache.get(face_id, spec, parent_elev, seed, detail_grid)
        if cached is not None:
            collection._stores[face_id] = cached
            hits += 1
        else:
            store = generate_detail_terrain_bounded(
                detail_grid, parent_elev, neighbor_elevs, spec, seed=seed,
            )
            cache.put(face_id, spec, parent_elev, seed, store, detail_grid)
            collection._stores[face_id] = store

    return hits


# ═══════════════════════════════════════════════════════════════════
# 10F.5 — Benchmark utilities
# ═══════════════════════════════════════════════════════════════════

def benchmark_pipeline(
    frequency: int = 3,
    detail_rings: int = 4,
    seed: int = 42,
    *,
    use_parallel: bool = True,
    use_fast_render: bool = True,
    tile_size: int = 128,
) -> Dict[str, float]:
    """Run the full pipeline and return timing measurements.

    Returns a dict with keys:
    - ``"grid_build"`` — time to build globe + detail grids
    - ``"terrain_gen"`` — time for terrain generation
    - ``"texture_render"`` — time for tile texture rendering
    - ``"atlas_assembly"`` — time for atlas packing
    - ``"total"`` — wall-clock total

    All values in seconds.
    """
    import tempfile
    from .globe import build_globe_grid
    from .mountains import MountainConfig, generate_mountains

    timings: Dict[str, float] = {}
    t_total = time.perf_counter()

    # 1. Build grids
    t0 = time.perf_counter()
    grid = build_globe_grid(frequency)
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)
    config = MountainConfig(seed=seed)
    generate_mountains(grid, store, config)
    spec = TileDetailSpec(detail_rings=detail_rings)
    coll = DetailGridCollection.build(grid, spec)
    timings["grid_build"] = time.perf_counter() - t0

    # 2. Terrain generation
    t0 = time.perf_counter()
    if use_parallel:
        generate_all_detail_terrain_parallel(
            coll, grid, store, spec, seed=seed,
        )
    else:
        from .detail_terrain import generate_all_detail_terrain
        generate_all_detail_terrain(coll, grid, store, spec, seed=seed)
    timings["terrain_gen"] = time.perf_counter() - t0

    # 3. Texture rendering + atlas
    biome = BiomeConfig()
    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time.perf_counter()
        if use_fast_render:
            atlas_path, uv_layout = build_detail_atlas_fast(
                coll, biome, tmpdir, tile_size=tile_size, noise_seed=seed,
            )
        else:
            from .texture_pipeline import build_detail_atlas
            atlas_path, uv_layout = build_detail_atlas(
                coll, biome, tmpdir, tile_size=tile_size, noise_seed=seed,
            )
        timings["texture_render"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - t_total
    timings["tile_count"] = len(coll.grids)
    timings["sub_face_count"] = coll.total_face_count
    return timings
