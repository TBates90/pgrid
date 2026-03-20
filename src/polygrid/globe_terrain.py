# TODO REMOVE — Not used by any live script. Phase 11D globe-scale terrain (mountains_3d, erosion).
"""Globe-scale terrain enhancements — Phase 11D.

Provides globe-optimised mountain presets, cross-tile river generation,
and lightweight hydraulic erosion for stitched detail grids.

Public API
----------
- :class:`MountainConfig3D` — 3-D noise mountain configuration
- Presets: ``GLOBE_MOUNTAIN_RANGE``, ``GLOBE_VOLCANIC_CHAIN``,
  ``GLOBE_CONTINENTAL_DIVIDE``
- :func:`generate_mountains_3d` — mountains on a stitched grid
  using 3-D coherent noise
- :func:`generate_rivers_on_stitched` — rivers that cross tile boundaries
- :func:`erode_terrain` — lightweight hydraulic erosion
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .algorithms import get_face_adjacency
from .geometry import face_center
from .heightmap import normalize_field, smooth_field
from .noise import fbm, ridged_noise
from .polygrid import PolyGrid
from .region_stitch import FaceMapping
from .rivers import (
    RiverConfig,
    RiverNetwork,
    RiverSegment,
    fill_depressions,
    flow_accumulation,
)
from .tile_data import TileDataStore


# ═══════════════════════════════════════════════════════════════════
# 11D.1 — MountainConfig3D & presets
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MountainConfig3D:
    """Globe-optimised mountain generation configuration.

    Works with 2-D projected coordinates from stitched grids
    (see :mod:`region_stitch`).

    Parameters
    ----------
    peak_elevation : float
        Maximum peak elevation (target, before blending with parent).
    base_elevation : float
        Minimum base elevation.
    ridge_frequency : float
        Spatial frequency for ridged noise.  Lower values → longer
        ridges spanning more tiles.
    ridge_octaves : int
        Octave count for ridged noise.
    ridge_lacunarity : float
        Frequency multiplier between octaves.
    ridge_persistence : float
        Amplitude multiplier between octaves.
    fbm_frequency : float
        Spatial frequency for the fbm layer (foothills).
    fbm_octaves : int
        Octaves for fbm.
    fbm_blend : float
        Weight of fbm blended into the ridge signal (0–1).
    smooth_iterations : int
        Smoothing passes after generation.
    seed : int
        Master seed.
    """

    peak_elevation: float = 1.0
    base_elevation: float = 0.05
    ridge_frequency: float = 8.0
    ridge_octaves: int = 6
    ridge_lacunarity: float = 2.2
    ridge_persistence: float = 0.5
    fbm_frequency: float = 6.0
    fbm_octaves: int = 4
    fbm_blend: float = 0.35
    smooth_iterations: int = 2
    seed: int = 42


# ── Presets ─────────────────────────────────────────────────────────

GLOBE_MOUNTAIN_RANGE = MountainConfig3D(
    peak_elevation=1.0,
    base_elevation=0.05,
    ridge_frequency=6.0,     # low freq → long ridges spanning tiles
    ridge_octaves=6,
    ridge_lacunarity=2.2,
    ridge_persistence=0.50,
    fbm_frequency=4.0,
    fbm_octaves=4,
    fbm_blend=0.35,
    smooth_iterations=2,
    seed=42,
)

GLOBE_VOLCANIC_CHAIN = MountainConfig3D(
    peak_elevation=1.0,
    base_elevation=0.10,
    ridge_frequency=12.0,    # high freq → isolated peaks
    ridge_octaves=4,
    ridge_lacunarity=2.5,
    ridge_persistence=0.40,
    fbm_frequency=8.0,
    fbm_octaves=3,
    fbm_blend=0.20,
    smooth_iterations=1,
    seed=42,
)

GLOBE_CONTINENTAL_DIVIDE = MountainConfig3D(
    peak_elevation=1.0,
    base_elevation=0.02,
    ridge_frequency=3.0,     # very low freq → single dominant ridge
    ridge_octaves=7,
    ridge_lacunarity=2.0,
    ridge_persistence=0.55,
    fbm_frequency=2.5,
    fbm_octaves=5,
    fbm_blend=0.45,
    smooth_iterations=3,
    seed=42,
)

MOUNTAIN_3D_PRESETS = {
    "mountain_range": GLOBE_MOUNTAIN_RANGE,
    "volcanic_chain": GLOBE_VOLCANIC_CHAIN,
    "continental_divide": GLOBE_CONTINENTAL_DIVIDE,
}


# ═══════════════════════════════════════════════════════════════════
# generate_mountains_3d
# ═══════════════════════════════════════════════════════════════════

def generate_mountains_3d(
    grid: PolyGrid,
    store: TileDataStore,
    config: MountainConfig3D,
    *,
    face_ids: Optional[List[str]] = None,
) -> None:
    """Generate mountain terrain on a (stitched) grid.

    Uses ridged noise + fbm blend in the grid's 2-D coordinate space,
    which — for stitched grids — is the gnomonic projection of the
    globe surface.  This means ridges cross tile boundaries naturally.

    Parameters
    ----------
    grid : PolyGrid
        Typically a stitched grid from :func:`~region_stitch.stitch_detail_grids`.
    store : TileDataStore
        Must have an ``"elevation"`` field.
    config : MountainConfig3D
    face_ids : list, optional
        Subset of faces to affect.  If ``None``, all faces.
    """
    targets = face_ids or list(grid.faces.keys())
    cfg = config

    # 1. Ridged noise
    for fid in targets:
        face = grid.faces.get(fid)
        if face is None:
            continue
        c = face_center(grid.vertices, face)
        if c is None:
            continue
        cx, cy = c

        ridge_val = ridged_noise(
            cx, cy,
            octaves=cfg.ridge_octaves,
            lacunarity=cfg.ridge_lacunarity,
            persistence=cfg.ridge_persistence,
            frequency=cfg.ridge_frequency,
            seed=cfg.seed,
        )
        store.set(fid, "elevation", ridge_val)

    # 2. Blend with fbm foothills
    if cfg.fbm_blend > 0:
        for fid in targets:
            face = grid.faces.get(fid)
            if face is None:
                continue
            c = face_center(grid.vertices, face)
            if c is None:
                continue
            cx, cy = c

            fbm_val = fbm(
                cx, cy,
                octaves=cfg.fbm_octaves,
                frequency=cfg.fbm_frequency,
                seed=cfg.seed + 50,
            )
            # Normalise fbm from [-1,1] to [0,1]
            fbm_01 = (fbm_val + 1.0) * 0.5

            ridge_val = store.get(fid, "elevation")
            blended = (1.0 - cfg.fbm_blend) * ridge_val + cfg.fbm_blend * fbm_01
            store.set(fid, "elevation", blended)

    # 3. Normalize
    normalize_field(
        store, "elevation",
        lo=cfg.base_elevation, hi=cfg.peak_elevation,
        face_ids=targets,
    )

    # 4. Smooth
    if cfg.smooth_iterations > 0:
        smooth_field(
            grid, store, "elevation",
            iterations=cfg.smooth_iterations,
            self_weight=0.6,
            face_ids=targets,
        )


# ═══════════════════════════════════════════════════════════════════
# 11D.2 — Globe-scale river generation on stitched grids
# ═══════════════════════════════════════════════════════════════════

def generate_rivers_on_stitched(
    combined_grid: PolyGrid,
    combined_store: TileDataStore,
    config: Optional[RiverConfig] = None,
    *,
    field_name: str = "elevation",
) -> RiverNetwork:
    """Generate rivers on a stitched combined grid.

    Because the combined grid merges sub-faces from multiple tiles into
    one contiguous :class:`PolyGrid`, rivers naturally flow across
    former tile boundaries.

    Parameters
    ----------
    combined_grid : PolyGrid
        Stitched grid from :func:`~region_stitch.stitch_detail_grids`.
    combined_store : TileDataStore
        Must have an elevation field.
    config : RiverConfig, optional
    field_name : str

    Returns
    -------
    RiverNetwork
    """
    if config is None:
        config = RiverConfig()

    adj = get_face_adjacency(combined_grid)

    # 1. Fill depressions
    fill_depressions(combined_grid, combined_store, adj, field_name)

    # 2. Flow accumulation
    acc = flow_accumulation(adj, combined_store, field_name)

    # 3. River faces
    river_faces: Set[str] = {
        fid for fid, a in acc.items() if a >= config.min_accumulation
    }
    if not river_faces:
        return RiverNetwork()

    # 4. Find river heads
    heads: List[Tuple[int, str]] = []
    for fid in river_faces:
        upstream_rivers = [
            nid for nid in adj.get(fid, [])
            if nid in river_faces
            and combined_store.get(nid, field_name) > combined_store.get(fid, field_name)
        ]
        if not upstream_rivers:
            heads.append((acc[fid], fid))

    if not heads:
        # Fallback: highest-elevation river faces
        heads = sorted(
            [(acc[fid], fid) for fid in river_faces],
            key=lambda x: x[0], reverse=True,
        )[:max(1, len(river_faces) // 10)]

    # 5. Trace paths from heads downhill
    segments: List[RiverSegment] = []
    used: Set[str] = set()

    for _, head in sorted(heads, key=lambda x: x[0], reverse=True):
        if head in used:
            continue
        path = _trace_river_downhill(adj, combined_store, head, river_faces, used, field_name)
        if len(path) >= config.min_length:
            seg = RiverSegment(
                name=f"river_{len(segments)}",
                face_ids=path,
                order=1,
                width=max(1.0, math.log2(acc.get(path[-1], 1) + 1)),
            )
            segments.append(seg)
            used.update(path)

    # 6. Assign Strahler order
    _assign_strahler_orders(segments, adj)

    network = RiverNetwork(segments=segments)

    # 7. Carve valleys
    if config.carve_depth > 0:
        _carve_river_valleys(
            combined_grid, combined_store, network, adj,
            config.carve_depth, config.valley_width, field_name,
        )

    return network


# ═══════════════════════════════════════════════════════════════════
# 11D.3 — Lightweight hydraulic erosion
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ErosionConfig:
    """Configuration for lightweight hydraulic erosion.

    Parameters
    ----------
    iterations : int
        Number of water droplets to simulate.
    erosion_rate : float
        How much material each droplet removes per step (0–1).
    deposition_rate : float
        How much sediment is deposited per step when the droplet
        slows down (0–1).
    max_steps : int
        Max steps per droplet before it evaporates.
    seed : int
        Random seed for droplet start positions.
    """

    iterations: int = 500
    erosion_rate: float = 0.03
    deposition_rate: float = 0.02
    max_steps: int = 50
    seed: int = 42


def erode_terrain(
    grid: PolyGrid,
    store: TileDataStore,
    config: Optional[ErosionConfig] = None,
    *,
    field_name: str = "elevation",
) -> Dict[str, float]:
    """Lightweight hydraulic erosion on a grid.

    Drops virtual water particles at random high-elevation faces.
    Each particle flows downhill (steepest descent), eroding the
    surface and depositing sediment when it slows.  Produces
    realistic valley shapes and drainage patterns.

    Works on any :class:`PolyGrid` but is most effective on stitched
    combined grids where water can flow across tile boundaries.

    Parameters
    ----------
    grid : PolyGrid
    store : TileDataStore
        Elevation field is modified **in-place**.
    config : ErosionConfig, optional
    field_name : str

    Returns
    -------
    dict
        ``{face_id: cumulative_erosion}`` — total material removed
        at each face (useful for diagnostics).
    """
    if config is None:
        config = ErosionConfig()

    adj = get_face_adjacency(grid)
    rng = random.Random(config.seed)

    # Collect faces sorted by elevation (descending) for weighted start
    face_ids = list(grid.faces.keys())
    elevations = {fid: store.get(fid, field_name) for fid in face_ids}

    # Weight start positions toward higher elevations
    sorted_faces = sorted(face_ids, key=lambda f: elevations[f], reverse=True)
    top_fraction = max(1, len(sorted_faces) // 4)
    high_faces = sorted_faces[:top_fraction]

    cumulative_erosion: Dict[str, float] = {fid: 0.0 for fid in face_ids}

    for _ in range(config.iterations):
        # Start at a random high-elevation face
        current = rng.choice(high_faces)
        sediment = 0.0

        for _ in range(config.max_steps):
            cur_elev = store.get(current, field_name)

            # Find steepest downhill neighbour
            best_nid: Optional[str] = None
            best_drop = 0.0

            for nid in adj.get(current, []):
                ne = store.get(nid, field_name)
                drop = cur_elev - ne
                if drop > best_drop:
                    best_drop = drop
                    best_nid = nid

            if best_nid is None:
                # At a local minimum — deposit all remaining sediment
                new_elev = cur_elev + sediment * config.deposition_rate
                store.set(current, field_name, new_elev)
                break

            # Erode current face
            erosion = min(config.erosion_rate * best_drop, cur_elev * 0.1)
            store.set(current, field_name, cur_elev - erosion)
            cumulative_erosion[current] += erosion
            sediment += erosion

            # Deposit some sediment if flow is slowing
            if best_drop < sediment * 0.5:
                deposit = min(sediment * config.deposition_rate, sediment)
                ne = store.get(best_nid, field_name)
                store.set(best_nid, field_name, ne + deposit)
                sediment -= deposit

            current = best_nid

    return cumulative_erosion


# ═══════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════

def _trace_river_downhill(
    adj: Dict[str, List[str]],
    store: TileDataStore,
    start: str,
    river_faces: Set[str],
    used: Set[str],
    field_name: str,
) -> List[str]:
    """Trace a river path downhill from *start*."""
    path = [start]
    visited: Set[str] = {start}

    for _ in range(len(adj)):
        current = path[-1]
        cur_elev = store.get(current, field_name)

        best_nid = None
        best_elev = cur_elev

        for nid in adj.get(current, []):
            if nid in visited:
                continue
            ne = store.get(nid, field_name)
            if ne < best_elev:
                best_elev = ne
                best_nid = nid

        if best_nid is None:
            break

        visited.add(best_nid)
        path.append(best_nid)

        # Stop at grid boundary or if we leave river faces
        if best_nid not in river_faces and best_nid not in adj:
            break

    return path


def _assign_strahler_orders(
    segments: List[RiverSegment],
    adj: Dict[str, List[str]],
) -> None:
    """Assign Strahler stream orders based on confluence structure."""
    if not segments:
        return

    # Build mouth → segment lookup
    mouth_map: Dict[str, RiverSegment] = {}
    for seg in segments:
        if seg.face_ids:
            mouth_map[seg.face_ids[-1]] = seg

    # Find confluences: segment mouths that are in another segment
    for seg in segments:
        mouth = seg.face_ids[-1] if seg.face_ids else None
        if mouth is None:
            continue
        # Check if this mouth feeds into another segment
        for other in segments:
            if other is seg:
                continue
            if mouth in other.face_ids:
                # seg flows into other — other should have higher order
                other.order = max(other.order, seg.order + 1)


def _carve_river_valleys(
    grid: PolyGrid,
    store: TileDataStore,
    network: RiverNetwork,
    adj: Dict[str, List[str]],
    carve_depth: float,
    valley_width: float,
    field_name: str,
) -> None:
    """Lower elevation along rivers and their valley banks."""
    river_fids = network.all_river_face_ids()

    # Carve river channels
    for fid in river_fids:
        elev = store.get(fid, field_name)
        store.set(fid, field_name, max(0.0, elev - carve_depth))

    # Widen valleys: lower neighbours of river faces
    if valley_width > 0:
        for fid in river_fids:
            for nid in adj.get(fid, []):
                if nid not in river_fids:
                    elev = store.get(nid, field_name)
                    store.set(
                        nid, field_name,
                        max(0.0, elev - carve_depth * valley_width),
                    )
