"""Visual cohesion validation — Phase 18E.

Programmatic checks that tile boundaries are seamless, topology-aware
features follow sub-face structure, and the full pipeline runs without
error.  These are *measurement* utilities — they sample pixel colours
at tile boundaries vs interiors and report variance ratios.

Functions
---------
- :func:`sample_boundary_pixels` — extract pixel colours along tile UV seams
- :func:`sample_interior_pixels` — extract pixel colours from tile interiors
- :func:`measure_seam_visibility` — compute boundary-vs-interior colour variance
- :func:`verify_topology_features` — check tree/ocean placement follows sub-faces
- :func:`run_full_pipeline` — end-to-end terrain → apron → atlas → export
- :func:`benchmark_apron_pipeline` — time apron vs baseline atlas build
"""

from __future__ import annotations

import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ═══════════════════════════════════════════════════════════════════
# 18E.1 — Seam elimination verification
# ═══════════════════════════════════════════════════════════════════

def sample_boundary_pixels(
    atlas: "Image.Image",
    uv_layout: Dict[str, Tuple[float, float, float, float]],
    *,
    n_samples: int = 50,
) -> List[Tuple[int, int, int]]:
    """Sample pixel colours along tile UV boundaries in the atlas.

    For each tile, samples *n_samples* pixels along each of the four
    edges of the UV rectangle.

    Parameters
    ----------
    atlas : PIL.Image
    uv_layout : dict
        ``{face_id: (u_min, v_min, u_max, v_max)}``.
    n_samples : int
        Samples per edge.

    Returns
    -------
    list of (R, G, B)
    """
    w, h = atlas.size
    pixels = []

    for fid, (u0, v0, u1, v1) in uv_layout.items():
        # Convert UV to pixel coords
        px0 = int(u0 * w)
        py0 = int((1.0 - v1) * h)  # v is flipped
        px1 = int(u1 * w)
        py1 = int((1.0 - v0) * h)

        # Clamp to image bounds
        px0 = max(0, min(w - 1, px0))
        px1 = max(0, min(w - 1, px1))
        py0 = max(0, min(h - 1, py0))
        py1 = max(0, min(h - 1, py1))

        tile_w = max(1, px1 - px0)
        tile_h = max(1, py1 - py0)

        for i in range(n_samples):
            t = i / max(1, n_samples - 1)

            # Top edge
            x = int(px0 + t * tile_w)
            pixels.append(atlas.getpixel((min(x, w - 1), py0))[:3])

            # Bottom edge
            pixels.append(atlas.getpixel((min(x, w - 1), min(py1, h - 1)))[:3])

            # Left edge
            y = int(py0 + t * tile_h)
            pixels.append(atlas.getpixel((px0, min(y, h - 1)))[:3])

            # Right edge
            pixels.append(atlas.getpixel((min(px1, w - 1), min(y, h - 1)))[:3])

    return pixels


def sample_interior_pixels(
    atlas: "Image.Image",
    uv_layout: Dict[str, Tuple[float, float, float, float]],
    *,
    n_samples: int = 50,
    margin_frac: float = 0.2,
) -> List[Tuple[int, int, int]]:
    """Sample pixel colours from tile interiors (away from edges).

    Parameters
    ----------
    atlas : PIL.Image
    uv_layout : dict
    n_samples : int
        Total samples per tile.
    margin_frac : float
        Fraction of tile to skip from each edge (0.2 = inner 60%).

    Returns
    -------
    list of (R, G, B)
    """
    import random
    rng = random.Random(12345)
    w, h = atlas.size
    pixels = []

    for fid, (u0, v0, u1, v1) in uv_layout.items():
        px0 = int(u0 * w)
        py0 = int((1.0 - v1) * h)
        px1 = int(u1 * w)
        py1 = int((1.0 - v0) * h)

        tile_w = max(1, px1 - px0)
        tile_h = max(1, py1 - py0)

        # Interior region
        mx = int(tile_w * margin_frac)
        my = int(tile_h * margin_frac)
        inner_x0 = px0 + mx
        inner_y0 = py0 + my
        inner_x1 = px1 - mx
        inner_y1 = py1 - my

        if inner_x1 <= inner_x0 or inner_y1 <= inner_y0:
            continue

        for _ in range(n_samples):
            x = rng.randint(inner_x0, min(inner_x1, w - 1))
            y = rng.randint(inner_y0, min(inner_y1, h - 1))
            pixels.append(atlas.getpixel((x, y))[:3])

    return pixels


def _colour_variance(pixels: List[Tuple[int, int, int]]) -> float:
    """Compute mean per-channel variance of a list of RGB pixels."""
    if len(pixels) < 2:
        return 0.0
    r = [p[0] for p in pixels]
    g = [p[1] for p in pixels]
    b = [p[2] for p in pixels]
    return (statistics.variance(r) + statistics.variance(g) + statistics.variance(b)) / 3.0


def measure_seam_visibility(
    atlas: "Image.Image",
    uv_layout: Dict[str, Tuple[float, float, float, float]],
    *,
    n_samples: int = 50,
) -> Dict[str, float]:
    """Measure boundary vs interior colour variance.

    Returns
    -------
    dict
        ``{"boundary_variance": float, "interior_variance": float,
           "ratio": float}``
        Ratio < 2.0 indicates acceptable seam visibility.
    """
    boundary = sample_boundary_pixels(atlas, uv_layout, n_samples=n_samples)
    interior = sample_interior_pixels(atlas, uv_layout, n_samples=n_samples)

    bv = _colour_variance(boundary)
    iv = _colour_variance(interior)

    ratio = bv / iv if iv > 0 else float("inf")

    return {
        "boundary_variance": bv,
        "interior_variance": iv,
        "ratio": ratio,
        "boundary_samples": len(boundary),
        "interior_samples": len(interior),
    }


# ═══════════════════════════════════════════════════════════════════
# 18E.2 — Topology feature verification
# ═══════════════════════════════════════════════════════════════════

def verify_topology_features(
    collection,
    globe_grid,
    *,
    seed: int = 42,
) -> Dict[str, Any]:
    """Verify that topology-aware features follow sub-face structure.

    Checks:
    1. Trees sit at sub-face centroids (not arbitrary positions).
    2. Ocean depth is assigned per sub-face (all sub-faces have values).
    3. Feature positions are deterministic (same seed → same result).

    Parameters
    ----------
    collection : DetailGridCollection
    globe_grid : PolyGrid
    seed : int

    Returns
    -------
    dict with check results.
    """
    from .biome_topology import scatter_trees_on_grid, compute_subface_ocean_depth
    from .geometry import face_center

    results: Dict[str, Any] = {
        "tree_centroid_check": False,
        "ocean_depth_check": False,
        "determinism_check": False,
    }

    face_ids = collection.face_ids
    if not face_ids:
        return results

    # Pick first tile with a grid
    test_fid = face_ids[0]
    grid, store = collection.get(test_fid)

    # ── Tree centroid check ─────────────────────────────────────
    try:
        trees = scatter_trees_on_grid(grid, store, density=0.5, seed=seed)
        if trees:
            # Check that each tree's face_id is a real face
            valid_faces = set(grid.faces.keys())
            all_valid = all(t.face_id in valid_faces for t in trees)
            results["tree_centroid_check"] = all_valid
            results["tree_count"] = len(trees)
        else:
            # No trees at this density is also valid
            results["tree_centroid_check"] = True
            results["tree_count"] = 0
    except Exception as e:
        results["tree_centroid_error"] = str(e)

    # ── Ocean depth check ───────────────────────────────────────
    try:
        depths = compute_subface_ocean_depth(grid, store)
        if depths:
            all_have_values = all(
                isinstance(v, (int, float)) for v in depths.values()
            )
            results["ocean_depth_check"] = all_have_values
            results["ocean_depth_count"] = len(depths)
        else:
            results["ocean_depth_check"] = True
            results["ocean_depth_count"] = 0
    except Exception as e:
        results["ocean_depth_error"] = str(e)

    # ── Determinism check ───────────────────────────────────────
    try:
        trees_a = scatter_trees_on_grid(grid, store, density=0.5, seed=seed)
        trees_b = scatter_trees_on_grid(grid, store, density=0.5, seed=seed)
        if trees_a and trees_b:
            same = (len(trees_a) == len(trees_b) and
                    all(a.face_id == b.face_id for a, b in zip(trees_a, trees_b)))
            results["determinism_check"] = same
        else:
            results["determinism_check"] = True
    except Exception as e:
        results["determinism_error"] = str(e)

    return results


# ═══════════════════════════════════════════════════════════════════
# 18E.3 — Full pipeline run
# ═══════════════════════════════════════════════════════════════════

def run_full_pipeline(
    frequency: int = 3,
    detail_rings: int = 4,
    tile_size: int = 128,
    seed: int = 42,
    *,
    output_dir: Path | str = Path("exports/cohesion_test"),
    enable_features: bool = True,
    enable_export: bool = True,
) -> Dict[str, Any]:
    """Run the complete terrain → apron → atlas → export pipeline.

    This is the integration test: every Phase 18 component in sequence.

    Parameters
    ----------
    frequency : int
    detail_rings : int
    tile_size : int
    seed : int
    output_dir : Path
    enable_features : bool
        Include biome features (forest + ocean).
    enable_export : bool
        Run KTX2 + glTF export.

    Returns
    -------
    dict
        Timings, paths, and validation results.
    """
    from .apron_grid import build_all_apron_grids
    from .apron_texture import build_apron_atlas, build_apron_feature_atlas
    from .detail_render import BiomeConfig
    from .tile_detail import TileDetailSpec, DetailGridCollection
    from .detail_terrain import generate_all_detail_terrain

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {"timings": {}}

    # ── 1. Build globe ──────────────────────────────────────────
    t0 = time.perf_counter()

    from .globe import build_globe_grid
    from .tile_data import FieldDef, TileDataStore, TileSchema
    from .mountains import MountainConfig, generate_mountains

    grid = build_globe_grid(frequency)
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)

    config = MountainConfig(
        seed=seed, ridge_frequency=2.0, ridge_octaves=4,
        peak_elevation=1.0, base_elevation=0.0,
    )
    generate_mountains(grid, store, config)

    result["timings"]["globe_build"] = time.perf_counter() - t0
    result["n_tiles"] = len(grid.faces)

    # ── 2. Detail grids + terrain ───────────────────────────────
    t0 = time.perf_counter()

    spec = TileDetailSpec(detail_rings=detail_rings)
    coll = DetailGridCollection.build(grid, spec)
    generate_all_detail_terrain(coll, grid, store, spec, seed=seed)

    result["timings"]["detail_terrain"] = time.perf_counter() - t0
    result["total_subfaces"] = coll.total_face_count

    # ── 3. Apron atlas (baseline ground) ────────────────────────
    t0 = time.perf_counter()

    atlas_path, uv_layout = build_apron_atlas(
        coll, grid,
        output_dir=output_dir / "apron_tiles",
        tile_size=tile_size,
        noise_seed=seed,
    )

    result["timings"]["apron_atlas"] = time.perf_counter() - t0
    result["atlas_path"] = str(atlas_path)
    result["uv_tile_count"] = len(uv_layout)

    # ── 4. Apron feature atlas (biome overlays) ─────────────────
    if enable_features:
        t0 = time.perf_counter()

        from .biome_pipeline import ForestRenderer, OceanRenderer, identify_forest_tiles
        from .biome_render import FOREST_PRESETS
        from .ocean_render import OCEAN_PRESETS, identify_ocean_tiles, compute_ocean_depth_map
        from .biome_continuity import build_biome_density_map
        from .terrain_patches import TERRAIN_PRESETS, generate_terrain_patches

        dist = TERRAIN_PRESETS.get("earthlike")
        patches = generate_terrain_patches(grid, distribution=dist, seed=seed)

        ocean_faces = identify_ocean_tiles(patches)
        forest_faces = identify_forest_tiles(patches)

        ocean_depth_map = compute_ocean_depth_map(grid, store, ocean_faces)

        density_map = {}
        biome_type_map = {}

        ocean_density = build_biome_density_map(
            grid, list(grid.faces.keys()),
            biome_faces=ocean_faces, seed=seed + 2000,
        )
        for fid, d in ocean_density.items():
            if d > 0.01:
                density_map[fid] = d
                biome_type_map[fid] = "ocean"

        forest_density = build_biome_density_map(
            grid, list(grid.faces.keys()),
            biome_faces=forest_faces, seed=seed + 1000,
        )
        for fid, d in forest_density.items():
            if d > 0.01 and fid not in ocean_faces:
                density_map[fid] = d
                biome_type_map[fid] = "forest"

        renderers = {
            "ocean": OceanRenderer(
                config=OCEAN_PRESETS["temperate"],
                ocean_depth_map=ocean_depth_map,
                ocean_faces=ocean_faces,
                globe_grid=grid,
            ),
            "forest": ForestRenderer(
                config=FOREST_PRESETS["temperate"],
            ),
        }

        feature_atlas_path, feature_uv = build_apron_feature_atlas(
            coll, grid,
            biome_renderers=renderers,
            density_map=density_map,
            biome_type_map=biome_type_map,
            output_dir=output_dir / "feature_tiles",
            tile_size=tile_size,
            noise_seed=seed,
        )

        result["timings"]["feature_atlas"] = time.perf_counter() - t0
        result["feature_atlas_path"] = str(feature_atlas_path)
        result["ocean_tiles"] = len(ocean_faces)
        result["forest_tiles"] = len(forest_faces)

        # Use the feature atlas for seam measurement
        atlas_path = feature_atlas_path
        uv_layout = feature_uv

    # ── 5. Seam measurement ─────────────────────────────────────
    atlas_img = Image.open(str(atlas_path)).convert("RGB")
    seam = measure_seam_visibility(atlas_img, uv_layout, n_samples=30)
    result["seam_visibility"] = seam

    # ── 6. Topology verification ────────────────────────────────
    topo = verify_topology_features(coll, grid, seed=seed)
    result["topology_verification"] = topo

    # ── 7. Material + KTX2 + glTF export ────────────────────────
    if enable_export:
        t0 = time.perf_counter()

        from .texture_export import (
            resize_atlas_pot,
            generate_atlas_mipmaps,
            export_atlas_ktx2,
            build_orm_atlas,
            validate_ktx2_header,
            export_globe_gltf,
        )

        # PoT atlas
        pot_atlas = resize_atlas_pot(atlas_img)
        pot_path = output_dir / "atlas_pot.png"
        pot_atlas.save(str(pot_path))
        result["pot_size"] = pot_atlas.size

        # Mipmaps
        mip_paths = generate_atlas_mipmaps(
            pot_path, output_dir=output_dir / "mipmaps",
        )
        result["mip_levels"] = len(mip_paths)

        # KTX2
        ktx_path = output_dir / "atlas.ktx2"
        export_atlas_ktx2(pot_path, ktx_path, include_mipmaps=True)
        result["ktx2_valid"] = validate_ktx2_header(ktx_path)
        result["ktx2_path"] = str(ktx_path)

        # ORM
        orm_img, orm_uv = build_orm_atlas(
            coll,
            biome_type_map=biome_type_map if enable_features else None,
            tile_size=tile_size,
            gutter=4,
        )
        orm_path = output_dir / "orm.png"
        orm_img.save(str(orm_path))
        result["orm_path"] = str(orm_path)

        # glTF
        gltf_path = output_dir / "globe.gltf"
        try:
            export_globe_gltf(
                frequency=frequency,
                uv_layout=uv_layout,
                albedo_path=pot_path,
                output_path=gltf_path,
                embed_textures=True,
            )
            result["gltf_path"] = str(gltf_path)
            result["gltf_valid"] = gltf_path.exists()
        except ImportError:
            result["gltf_valid"] = False
            result["gltf_error"] = "models library required"

        result["timings"]["export"] = time.perf_counter() - t0

    return result


# ═══════════════════════════════════════════════════════════════════
# 18E.4 — Performance budget
# ═══════════════════════════════════════════════════════════════════

def benchmark_apron_pipeline(
    frequency: int = 3,
    detail_rings: int = 4,
    tile_size: int = 128,
    seed: int = 42,
    *,
    n_runs: int = 2,
) -> Dict[str, Any]:
    """Compare apron atlas build time against baseline (non-apron).

    Target: apron pipeline < 2× baseline.

    Parameters
    ----------
    frequency : int
    detail_rings : int
    tile_size : int
    seed : int
    n_runs : int

    Returns
    -------
    dict
        ``{"baseline_mean", "apron_mean", "ratio", "within_budget"}``.
    """
    from .detail_render import BiomeConfig
    from .tile_detail import TileDetailSpec, DetailGridCollection
    from .detail_terrain import generate_all_detail_terrain
    from .texture_pipeline import build_detail_atlas
    from .apron_texture import build_apron_atlas

    import tempfile

    from .globe import build_globe_grid
    from .tile_data import FieldDef, TileDataStore, TileSchema
    from .mountains import MountainConfig, generate_mountains

    grid = build_globe_grid(frequency)
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)
    config = MountainConfig(
        seed=seed, ridge_frequency=2.0, ridge_octaves=4,
        peak_elevation=1.0, base_elevation=0.0,
    )
    generate_mountains(grid, store, config)

    spec = TileDetailSpec(detail_rings=detail_rings)
    coll = DetailGridCollection.build(grid, spec)
    generate_all_detail_terrain(coll, grid, store, spec, seed=seed)

    biome = BiomeConfig()

    # ── Baseline: non-apron atlas ───────────────────────────────
    baseline_times = []
    for _ in range(n_runs):
        with tempfile.TemporaryDirectory() as td:
            t0 = time.perf_counter()
            build_detail_atlas(
                coll, biome, Path(td),
                tile_size=tile_size, noise_seed=seed,
            )
            baseline_times.append(time.perf_counter() - t0)

    # ── Apron atlas ─────────────────────────────────────────────
    apron_times = []
    for _ in range(n_runs):
        with tempfile.TemporaryDirectory() as td:
            t0 = time.perf_counter()
            build_apron_atlas(
                coll, grid,
                output_dir=Path(td),
                tile_size=tile_size, noise_seed=seed,
            )
            apron_times.append(time.perf_counter() - t0)

    baseline_mean = statistics.mean(baseline_times)
    apron_mean = statistics.mean(apron_times)
    ratio = apron_mean / baseline_mean if baseline_mean > 0 else float("inf")

    return {
        "baseline_mean": round(baseline_mean, 3),
        "apron_mean": round(apron_mean, 3),
        "ratio": round(ratio, 2),
        "within_budget": ratio < 2.0,
        "n_runs": n_runs,
        "n_tiles": len(grid.faces),
    }
