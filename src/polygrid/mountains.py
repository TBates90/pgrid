"""Mountain terrain generation — Phase 7C.

Assembles noise primitives (:mod:`noise`) and grid-noise bridge
(:mod:`heightmap`) into a high-level mountain generator that writes
elevation data into a :class:`~tile_data.TileDataStore`.

Usage
-----
>>> from polygrid.mountains import generate_mountains, MOUNTAIN_RANGE
>>> generate_mountains(grid, store, MOUNTAIN_RANGE)

To restrict mountains to a specific region::

    generate_mountains(grid, store, config, region=my_region)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .geometry import face_center, grid_center
from .heightmap import (
    clamp_field,
    normalize_field,
    sample_noise_field,
    sample_noise_field_region,
    smooth_field,
)
from .noise import (
    domain_warp,
    fbm,
    gradient_mask,
    ridged_noise,
    normalize,
    terrace as terrace_fn,
)
from .polygrid import PolyGrid
from .regions import Region
from .tile_data import TileDataStore


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════


@dataclass
class MountainConfig:
    """All tuneable parameters for mountain generation.

    Attributes
    ----------
    peak_elevation : float
        Maximum elevation at the highest peaks.
    base_elevation : float
        Elevation of surrounding lowlands / base terrain.
    ridge_octaves : int
        Number of noise octaves for the ridge signal.
    ridge_lacunarity : float
        Frequency multiplier between octaves.
    ridge_persistence : float
        Amplitude multiplier between octaves.
    ridge_frequency : float
        Base spatial frequency for ridges.
    warp_strength : float
        Domain-warp amplitude (organic shape distortion).
    warp_frequency : float
        Spatial frequency of the warp noise.
    foothill_blend : float
        Weight of fbm noise blended in for foothills (0 = no foothills,
        1 = heavy foothill variation).
    foothill_octaves : int
        Octaves for the foothill fbm signal.
    terrace_steps : int
        0 = smooth terrain, >0 = stepped mesa/plateau terracing.
    terrace_smoothing : float
        Blend between terraced and smooth (0 = pure terrace, 1 = smooth).
    edge_falloff : bool
        If *True* and a ``region`` is supplied, apply a radial gradient
        mask so elevation fades to *base_elevation* near region edges.
    smooth_iterations : int
        Number of neighbour-averaging passes after generation.
    smooth_self_weight : float
        Self-weight in the smoothing kernel.
    seed : int
        Master random seed.
    """

    peak_elevation: float = 1.0
    base_elevation: float = 0.1
    ridge_octaves: int = 6
    ridge_lacunarity: float = 2.2
    ridge_persistence: float = 0.5
    ridge_frequency: float = 1.5
    warp_strength: float = 0.3
    warp_frequency: float = 1.0
    foothill_blend: float = 0.4
    foothill_octaves: int = 4
    terrace_steps: int = 0
    terrace_smoothing: float = 0.0
    edge_falloff: bool = True
    smooth_iterations: int = 1
    smooth_self_weight: float = 0.6
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════
# Preset configs
# ═══════════════════════════════════════════════════════════════════

MOUNTAIN_RANGE = MountainConfig(
    peak_elevation=1.0,
    base_elevation=0.05,
    ridge_octaves=6,
    ridge_lacunarity=2.2,
    ridge_persistence=0.5,
    ridge_frequency=1.5,
    warp_strength=0.4,
    warp_frequency=0.8,
    foothill_blend=0.35,
    foothill_octaves=4,
    terrace_steps=0,
    smooth_iterations=2,
    seed=42,
)

ALPINE_PEAKS = MountainConfig(
    peak_elevation=1.0,
    base_elevation=0.05,
    ridge_octaves=7,
    ridge_lacunarity=2.5,
    ridge_persistence=0.45,
    ridge_frequency=2.5,
    warp_strength=0.15,
    warp_frequency=1.2,
    foothill_blend=0.2,
    foothill_octaves=3,
    terrace_steps=0,
    smooth_iterations=1,
    seed=42,
)

ROLLING_HILLS = MountainConfig(
    peak_elevation=0.4,
    base_elevation=0.05,
    ridge_octaves=3,
    ridge_lacunarity=2.0,
    ridge_persistence=0.6,
    ridge_frequency=1.0,
    warp_strength=0.2,
    warp_frequency=0.6,
    foothill_blend=0.7,
    foothill_octaves=4,
    terrace_steps=0,
    smooth_iterations=3,
    smooth_self_weight=0.4,
    seed=42,
)

MESA_PLATEAU = MountainConfig(
    peak_elevation=0.8,
    base_elevation=0.1,
    ridge_octaves=3,
    ridge_lacunarity=2.0,
    ridge_persistence=0.5,
    ridge_frequency=1.2,
    warp_strength=0.2,
    warp_frequency=0.8,
    foothill_blend=0.3,
    foothill_octaves=2,
    terrace_steps=4,
    terrace_smoothing=0.15,
    smooth_iterations=1,
    seed=42,
)


# ═══════════════════════════════════════════════════════════════════
# Generator
# ═══════════════════════════════════════════════════════════════════

def generate_mountains(
    grid: PolyGrid,
    store: TileDataStore,
    config: MountainConfig,
    *,
    region: Optional[Region] = None,
) -> None:
    """Generate mountain terrain and write elevation into *store*.

    Orchestrates noise primitives in this order:

    1. Ridged noise → sharp peaks and ridge lines
    2. Domain warp → organic, non-geometric ridge shapes
    3. Blend with fbm → foothills and micro-variation
    4. Optional terrace → stepped plateaus
    5. Optional edge falloff → fade to base_elevation at region edges
    6. Normalize to ``[base_elevation, peak_elevation]``
    7. Smooth → soften harsh cell-to-cell jumps

    Parameters
    ----------
    grid : PolyGrid
    store : TileDataStore
        Must have an ``"elevation"`` field (type ``float``) in its schema.
    config : MountainConfig
    region : Region, optional
        If given, only the faces in this region are affected.
    """
    face_ids = list(region.face_ids) if region else list(grid.faces.keys())
    cfg = config

    # ── 1-2. Ridged noise with domain warp ──────────────────────────
    def _ridged_warped(x: float, y: float) -> float:
        if cfg.warp_strength > 0:
            return domain_warp(
                ridged_noise,
                x,
                y,
                warp_strength=cfg.warp_strength,
                warp_frequency=cfg.warp_frequency,
                warp_seed_x=cfg.seed + 100,
                warp_seed_y=cfg.seed + 200,
                octaves=cfg.ridge_octaves,
                lacunarity=cfg.ridge_lacunarity,
                persistence=cfg.ridge_persistence,
                frequency=cfg.ridge_frequency,
                seed=cfg.seed,
            )
        return ridged_noise(
            x,
            y,
            octaves=cfg.ridge_octaves,
            lacunarity=cfg.ridge_lacunarity,
            persistence=cfg.ridge_persistence,
            frequency=cfg.ridge_frequency,
            seed=cfg.seed,
        )

    sample_noise_field(grid, store, "elevation", _ridged_warped, face_ids=face_ids)

    # ── 3. Blend with fbm for foothills ─────────────────────────────
    if cfg.foothill_blend > 0:
        # We need a temporary field for the foothill noise.
        # Check if "temp_foothill" exists in schema; if not, use a
        # manual approach to avoid schema dependency.
        _blend_foothills(grid, store, cfg, face_ids)

    # ── 4. Terrace ──────────────────────────────────────────────────
    if cfg.terrace_steps > 0:
        for fid in face_ids:
            v = store.get(fid, "elevation")
            store.set(
                fid,
                "elevation",
                terrace_fn(v, steps=cfg.terrace_steps, smoothing=cfg.terrace_smoothing),
            )

    # ── 5. Edge falloff ─────────────────────────────────────────────
    if cfg.edge_falloff and region is not None:
        _apply_edge_falloff(grid, store, face_ids)

    # ── 6. Normalize ────────────────────────────────────────────────
    normalize_field(store, "elevation", lo=cfg.base_elevation, hi=cfg.peak_elevation, face_ids=face_ids)

    # ── 7. Smooth ───────────────────────────────────────────────────
    if cfg.smooth_iterations > 0:
        smooth_field(
            grid,
            store,
            "elevation",
            iterations=cfg.smooth_iterations,
            self_weight=cfg.smooth_self_weight,
            face_ids=face_ids,
        )


# ═══════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════

def _blend_foothills(
    grid: PolyGrid,
    store: TileDataStore,
    cfg: MountainConfig,
    face_ids: list,
) -> None:
    """Blend fbm foothill noise into the existing elevation field."""
    def _fbm_fn(x: float, y: float) -> float:
        return fbm(
            x,
            y,
            octaves=cfg.foothill_octaves,
            frequency=cfg.ridge_frequency * 0.7,
            seed=cfg.seed + 50,
        )

    # Compute foothill noise and blend in-place
    for fid in face_ids:
        face = grid.faces.get(fid)
        if face is None:
            continue
        c = face_center(grid.vertices, face)
        if c is None:
            continue
        cx, cy = c
        ridge_val = store.get(fid, "elevation")
        foothill_val = _fbm_fn(cx, cy)
        # Normalize fbm from [-1,1] to [0,1]
        foothill_01 = normalize(foothill_val, src_min=-1.0, src_max=1.0, dst_min=0.0, dst_max=1.0)
        # Blend: elevation = (1-blend)*ridge + blend*foothill
        blended = (1.0 - cfg.foothill_blend) * ridge_val + cfg.foothill_blend * foothill_01
        store.set(fid, "elevation", blended)


def _apply_edge_falloff(
    grid: PolyGrid,
    store: TileDataStore,
    face_ids: list,
) -> None:
    """Apply radial gradient falloff from region centroid to region edges."""
    # Compute centroid and max radius of the region
    positions = []
    fid_pos = {}
    for fid in face_ids:
        face = grid.faces.get(fid)
        if face is None:
            continue
        c = face_center(grid.vertices, face)
        if c is not None:
            positions.append(c)
            fid_pos[fid] = c

    if not positions:
        return

    cx = sum(p[0] for p in positions) / len(positions)
    cy = sum(p[1] for p in positions) / len(positions)
    import math
    max_r = max(math.hypot(p[0] - cx, p[1] - cy) for p in positions) or 1.0

    for fid, (fx, fy) in fid_pos.items():
        mask = gradient_mask(fx, fy, center=(cx, cy), radius=max_r * 1.1, falloff="smooth")
        v = store.get(fid, "elevation")
        store.set(fid, "elevation", v * mask)
