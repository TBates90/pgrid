"""Terrain patches — structured terrain distribution across the globe.

Phase 11B — groups adjacent Goldberg tiles into *terrain patches*
(continents, mountain ranges, ocean basins, plains) and assigns each
patch a terrain recipe.  When combined with the 3-D coherent noise
from Phase 11A, this produces globe-level terrain that has large-scale
structure (continents vs oceans, mountain belts, plains) rather than
uniform noise everywhere.

Key idea
--------
1.  Partition the globe into N organic-shaped regions using
    :func:`~regions.partition_noise` (noise-perturbed Voronoi).
2.  Classify each region by terrain type based on position, size,
    and a seeded random assignment drawn from a *terrain distribution*
    preset (e.g. "earthlike" with ~30 % ocean, ~20 % plains, etc.).
3.  Each terrain type carries its own noise parameters — mountains get
    high ridge weight and frequency; oceans get low amplitude; plains
    get smooth, low-frequency noise.
4.  :func:`apply_terrain_patches` generates 3-D coherent terrain for
    every tile, using the patch-specific parameters.  Cross-patch
    boundaries are smoothed so terrain types blend organically.

Functions
---------
- :class:`TerrainPatch` — single patch definition
- :class:`TerrainDistribution` — preset terrain distributions
- :func:`generate_terrain_patches` — auto-generate patches
- :func:`apply_terrain_patches` — generate terrain using patches
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .algorithms import get_face_adjacency
from .detail_terrain_3d import (
    Terrain3DSpec,
    generate_detail_terrain_3d,
    precompute_3d_positions,
)
from .geometry import face_center
from .heightmap import smooth_field
from .polygrid import PolyGrid
from .regions import RegionMap, partition_flood_fill, partition_noise
from .tile_data import FieldDef, TileDataStore, TileSchema
from .tile_detail import DetailGridCollection


# ═══════════════════════════════════════════════════════════════════
# 11B.1 — TerrainPatch dataclass
# ═══════════════════════════════════════════════════════════════════

# Terrain type → default noise recipe
_TERRAIN_TYPE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "mountain": {
        "noise_frequency": 8.0,
        "ridge_frequency": 5.0,
        "fbm_weight": 0.3,
        "ridge_weight": 0.7,
        "base_weight": 0.35,
        "amplitude": 0.35,
        "elevation_range": (0.55, 1.0),
    },
    "hills": {
        "noise_frequency": 6.0,
        "ridge_frequency": 4.0,
        "fbm_weight": 0.6,
        "ridge_weight": 0.4,
        "base_weight": 0.45,
        "amplitude": 0.25,
        "elevation_range": (0.35, 0.65),
    },
    "plains": {
        "noise_frequency": 3.0,
        "ridge_frequency": 2.0,
        "fbm_weight": 0.9,
        "ridge_weight": 0.1,
        "base_weight": 0.60,
        "amplitude": 0.10,
        "elevation_range": (0.25, 0.45),
    },
    "ocean": {
        "noise_frequency": 3.0,
        "ridge_frequency": 2.0,
        "fbm_weight": 0.85,
        "ridge_weight": 0.15,
        "base_weight": 0.65,
        "amplitude": 0.08,
        "elevation_range": (0.0, 0.25),
    },
    "desert": {
        "noise_frequency": 4.0,
        "ridge_frequency": 3.0,
        "fbm_weight": 0.7,
        "ridge_weight": 0.3,
        "base_weight": 0.50,
        "amplitude": 0.15,
        "elevation_range": (0.20, 0.40),
    },
    "forest": {
        "noise_frequency": 5.0,
        "ridge_frequency": 3.5,
        "fbm_weight": 0.7,
        "ridge_weight": 0.3,
        "base_weight": 0.50,
        "amplitude": 0.18,
        "elevation_range": (0.30, 0.55),
    },
}

# All known terrain types
TERRAIN_TYPES = list(_TERRAIN_TYPE_DEFAULTS.keys())


@dataclass
class TerrainPatch:
    """A named group of globe tile IDs sharing a terrain recipe.

    Parameters
    ----------
    name : str
        Patch identifier (e.g. ``"mountain_range_1"``).
    face_ids : list of str
        Globe-level face IDs in this patch.
    terrain_type : str
        One of: ``"mountain"``, ``"hills"``, ``"plains"``, ``"ocean"``,
        ``"desert"``, ``"forest"``.
    params : dict
        Noise parameters for this patch.  Missing keys are filled
        from :data:`_TERRAIN_TYPE_DEFAULTS`.
    elevation_range : tuple
        ``(min_elevation, max_elevation)`` target range.
    """

    name: str
    face_ids: List[str]
    terrain_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    elevation_range: Tuple[float, float] = (0.0, 1.0)

    def to_terrain_3d_spec(self, *, seed: int = 42) -> Terrain3DSpec:
        """Build a :class:`Terrain3DSpec` from this patch's parameters.

        Merges the patch's explicit ``params`` with the terrain-type
        defaults, then constructs a ``Terrain3DSpec``.
        """
        defaults = _TERRAIN_TYPE_DEFAULTS.get(self.terrain_type, {})
        merged = {**defaults, **self.params}
        # Remove elevation_range — it's not a Terrain3DSpec field
        merged.pop("elevation_range", None)
        return Terrain3DSpec(
            noise_frequency=merged.get("noise_frequency", 6.0),
            noise_octaves=merged.get("noise_octaves", 5),
            ridge_frequency=merged.get("ridge_frequency", 4.0),
            ridge_octaves=merged.get("ridge_octaves", 5),
            fbm_weight=merged.get("fbm_weight", 0.6),
            ridge_weight=merged.get("ridge_weight", 0.4),
            base_weight=merged.get("base_weight", 0.55),
            amplitude=merged.get("amplitude", 0.25),
            boundary_smoothing=0,  # we do cross-patch smoothing separately
            seed=seed,
        )

    @property
    def size(self) -> int:
        """Number of globe tiles in this patch."""
        return len(self.face_ids)

    def __repr__(self) -> str:
        return (
            f"TerrainPatch(name={self.name!r}, type={self.terrain_type!r}, "
            f"tiles={self.size}, elev={self.elevation_range})"
        )


# ═══════════════════════════════════════════════════════════════════
# 11B.4 — Preset terrain distributions
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TerrainDistribution:
    """A named distribution of terrain types across the globe.

    The ``weights`` dict maps terrain type → relative probability.
    The probabilities are normalised internally.

    Parameters
    ----------
    name : str
        Preset name.
    weights : dict
        ``{terrain_type: relative_weight}``.
    """

    name: str
    weights: Dict[str, float]

    @property
    def normalised_weights(self) -> Dict[str, float]:
        """Weights normalised to sum to 1.0."""
        total = sum(self.weights.values())
        if total < 1e-10:
            n = len(self.weights)
            return {k: 1.0 / n for k in self.weights}
        return {k: v / total for k, v in self.weights.items()}


# Built-in presets
EARTHLIKE = TerrainDistribution(
    name="earthlike",
    weights={
        "ocean": 0.30,
        "plains": 0.20,
        "forest": 0.15,
        "hills": 0.15,
        "desert": 0.10,
        "mountain": 0.10,
    },
)

MOUNTAINOUS = TerrainDistribution(
    name="mountainous",
    weights={
        "mountain": 0.35,
        "hills": 0.25,
        "plains": 0.20,
        "forest": 0.10,
        "desert": 0.05,
        "ocean": 0.05,
    },
)

ARCHIPELAGO = TerrainDistribution(
    name="archipelago",
    weights={
        "ocean": 0.65,
        "hills": 0.12,
        "plains": 0.08,
        "forest": 0.08,
        "mountain": 0.05,
        "desert": 0.02,
    },
)

PANGAEA = TerrainDistribution(
    name="pangaea",
    weights={
        "plains": 0.25,
        "forest": 0.20,
        "hills": 0.15,
        "mountain": 0.10,
        "desert": 0.10,
        "ocean": 0.20,
    },
)

FOREST_WORLD = TerrainDistribution(
    name="forest_world",
    weights={
        "forest": 0.80,
        "hills": 0.05,
        "plains": 0.05,
        "ocean": 0.10,
    },
)

DEEP_FOREST = TerrainDistribution(
    name="deep_forest",
    weights={
        "forest": 1.0,
    },
)

OCEAN_WORLD = TerrainDistribution(
    name="ocean_world",
    weights={
        "ocean": 0.80,
        "hills": 0.06,
        "plains": 0.05,
        "forest": 0.04,
        "mountain": 0.03,
        "desert": 0.02,
    },
)

TERRAIN_PRESETS: Dict[str, TerrainDistribution] = {
    "earthlike": EARTHLIKE,
    "mountainous": MOUNTAINOUS,
    "archipelago": ARCHIPELAGO,
    "pangaea": PANGAEA,
    "forest_world": FOREST_WORLD,
    "deep_forest": DEEP_FOREST,
    "ocean_world": OCEAN_WORLD,
}


# ═══════════════════════════════════════════════════════════════════
# 11B.2 — Generate terrain patches
# ═══════════════════════════════════════════════════════════════════

def generate_terrain_patches(
    globe_grid: PolyGrid,
    *,
    n_patches: int = 8,
    distribution: Optional[TerrainDistribution] = None,
    seed: int = 42,
) -> List[TerrainPatch]:
    """Auto-generate terrain patches for a globe grid.

    1.  Select *n_patches* seed faces spread across the globe.
    2.  Partition the globe using noise-perturbed Voronoi (organic shapes).
    3.  Assign terrain types drawn from *distribution* (weighted random).

    Parameters
    ----------
    globe_grid : PolyGrid
        The globe grid.
    n_patches : int
        Number of terrain patches (regions).  Good defaults:
        6–8 for freq=3, 10–15 for freq=4+.
    distribution : TerrainDistribution, optional
        Terrain type distribution.  Defaults to :data:`EARTHLIKE`.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of TerrainPatch
    """
    if distribution is None:
        distribution = EARTHLIKE

    rng = random.Random(seed)
    face_ids = sorted(globe_grid.faces.keys())
    n_patches = min(n_patches, len(face_ids))

    # Select well-spread seed faces using a greedy farthest-point approach
    # based on 3D tile centres (if available) or random otherwise.
    seed_faces = _select_spread_seeds(globe_grid, n_patches, rng)

    # Partition the globe into organic regions
    region_map = partition_noise(
        globe_grid,
        seed_faces,
        noise_scale=2.0,
        noise_weight=0.35,
        seed=seed,
    )

    # Assign terrain types from the distribution
    nw = distribution.normalised_weights
    terrain_types = list(nw.keys())
    terrain_weights = [nw[t] for t in terrain_types]

    patches: List[TerrainPatch] = []
    for region in region_map.regions:
        # Weighted random terrain type
        tt = rng.choices(terrain_types, weights=terrain_weights, k=1)[0]

        # Get the default elevation range for this terrain type
        defaults = _TERRAIN_TYPE_DEFAULTS.get(tt, {})
        elev_range = defaults.get("elevation_range", (0.0, 1.0))

        patch = TerrainPatch(
            name=region.name,
            face_ids=sorted(region.face_ids),
            terrain_type=tt,
            elevation_range=elev_range,
        )
        patches.append(patch)

    return patches


def _select_spread_seeds(
    globe_grid: PolyGrid,
    n: int,
    rng: random.Random,
) -> List[str]:
    """Select *n* seed faces that are well-spread across the globe.

    Uses a greedy farthest-point strategy: pick a random first seed,
    then iteratively add the face that maximises its minimum distance
    to all existing seeds.

    Falls back to random selection if face centres aren't available.
    """
    face_ids = sorted(globe_grid.faces.keys())
    if n >= len(face_ids):
        return face_ids

    # Try to use 3D centres for distance calculation
    centres: Dict[str, Tuple[float, float, float]] = {}
    for fid in face_ids:
        face = globe_grid.faces[fid]
        c3d = face.metadata.get("center_3d")
        if c3d is not None:
            centres[fid] = c3d

    if len(centres) < len(face_ids):
        # Fallback: random selection
        return rng.sample(face_ids, n)

    # Greedy farthest-point
    seeds: List[str] = [rng.choice(face_ids)]
    remaining = set(face_ids) - {seeds[0]}

    for _ in range(n - 1):
        if not remaining:
            break
        # Find the face in `remaining` with max min-distance to seeds
        best_fid = None
        best_dist = -1.0
        for fid in remaining:
            c = centres[fid]
            min_d = min(
                math.sqrt(sum((a - b) ** 2 for a, b in zip(c, centres[s])))
                for s in seeds
            )
            if min_d > best_dist:
                best_dist = min_d
                best_fid = fid
        if best_fid is not None:
            seeds.append(best_fid)
            remaining.discard(best_fid)

    return seeds


# ═══════════════════════════════════════════════════════════════════
# 11B.3 — Apply terrain patches to a detail collection
# ═══════════════════════════════════════════════════════════════════

def apply_terrain_patches(
    collection: DetailGridCollection,
    globe_grid: PolyGrid,
    globe_store: TileDataStore,
    patches: List[TerrainPatch],
    *,
    seed: int = 42,
    elevation_field: str = "elevation",
    radius: float = 1.0,
    cross_patch_smoothing: int = 3,
) -> None:
    """Generate terrain for every tile using patch-specific noise parameters.

    For each patch:
    1.  Precompute 3-D positions for all sub-faces in the patch's tiles.
    2.  Generate terrain using the patch's :class:`Terrain3DSpec`.
    3.  Rescale elevations to the patch's ``elevation_range``.

    After all patches are generated, boundary sub-faces between
    different patches are smoothed to eliminate hard terrain-type
    transitions.

    Parameters
    ----------
    collection : DetailGridCollection
        Must already contain grids.
    globe_grid : PolyGrid
        The globe grid.
    globe_store : TileDataStore
        Globe-level elevation data.
    patches : list of TerrainPatch
        Terrain patches (must cover all globe tiles).
    seed : int
        Global noise seed.
    elevation_field : str
        Elevation field name in *globe_store*.
    radius : float
        Sphere radius.
    cross_patch_smoothing : int
        Smoothing iterations on faces near patch boundaries.
    """
    # Build face→patch lookup
    face_to_patch: Dict[str, TerrainPatch] = {}
    for patch in patches:
        for fid in patch.face_ids:
            face_to_patch[fid] = patch

    # Generate terrain per patch
    for patch in patches:
        spec_3d = patch.to_terrain_3d_spec(seed=seed)

        for face_id in patch.face_ids:
            if face_id not in collection.grids:
                continue

            detail_grid = collection.grids[face_id]
            positions = precompute_3d_positions(
                globe_grid, face_id, detail_grid, radius=radius,
            )
            parent_elev = globe_store.get(face_id, elevation_field)

            store = generate_detail_terrain_3d(
                detail_grid, positions, parent_elev, spec_3d,
            )
            collection._stores[face_id] = store

        # Rescale elevations to the patch's target range
        _rescale_patch_elevations(collection, patch)

    # Smooth cross-patch boundaries
    if cross_patch_smoothing > 0:
        _smooth_cross_patch_boundaries(
            collection, globe_grid, patches, face_to_patch,
            iterations=cross_patch_smoothing,
        )


def _rescale_patch_elevations(
    collection: DetailGridCollection,
    patch: TerrainPatch,
) -> None:
    """Rescale a patch's elevations to its target elevation range.

    Performs a min-max rescale across ALL sub-faces in the patch
    so that the minimum elevation maps to ``elevation_range[0]``
    and the maximum to ``elevation_range[1]``.
    """
    lo, hi = patch.elevation_range
    if abs(hi - lo) < 1e-10:
        return

    # Collect all elevations in the patch
    all_elevs: List[Tuple[str, str, float]] = []  # (face_id, sub_face_id, elev)
    for face_id in patch.face_ids:
        store = collection._stores.get(face_id)
        if store is None:
            continue
        grid = collection.grids[face_id]
        for sf_id in grid.faces:
            elev = store.get(sf_id, "elevation")
            all_elevs.append((face_id, sf_id, elev))

    if not all_elevs:
        return

    vals = [e for _, _, e in all_elevs]
    vmin, vmax = min(vals), max(vals)
    span = vmax - vmin
    if span < 1e-15:
        # All elevations identical — set to midpoint of target range
        mid = (lo + hi) / 2.0
        for face_id, sf_id, _ in all_elevs:
            collection._stores[face_id].set(sf_id, "elevation", mid)
        return

    for face_id, sf_id, elev in all_elevs:
        t = (elev - vmin) / span
        collection._stores[face_id].set(sf_id, "elevation", lo + t * (hi - lo))


def _smooth_cross_patch_boundaries(
    collection: DetailGridCollection,
    globe_grid: PolyGrid,
    patches: List[TerrainPatch],
    face_to_patch: Dict[str, TerrainPatch],
    *,
    iterations: int = 3,
) -> None:
    """Smooth sub-face elevations at the boundaries between different patches.

    Identifies globe tiles that border a different terrain patch, then
    smooths ALL sub-faces in those tiles to soften the terrain-type
    transition.
    """
    adj = get_face_adjacency(globe_grid)

    # Find boundary tiles: tiles whose neighbours belong to a different patch
    boundary_tiles: Set[str] = set()
    for patch in patches:
        for fid in patch.face_ids:
            for nid in adj.get(fid, []):
                neighbor_patch = face_to_patch.get(nid)
                if neighbor_patch is not None and neighbor_patch.name != patch.name:
                    boundary_tiles.add(fid)
                    boundary_tiles.add(nid)

    # For each boundary tile, smooth all its sub-faces
    for face_id in boundary_tiles:
        store = collection._stores.get(face_id)
        if store is None:
            continue
        detail_grid = collection.grids.get(face_id)
        if detail_grid is None:
            continue
        smooth_field(
            detail_grid, store, "elevation",
            iterations=iterations,
            self_weight=0.6,
        )


# ═══════════════════════════════════════════════════════════════════
# Convenience: single-call pipeline
# ═══════════════════════════════════════════════════════════════════

def generate_patched_terrain(
    collection: DetailGridCollection,
    globe_grid: PolyGrid,
    globe_store: TileDataStore,
    *,
    n_patches: int = 8,
    distribution: Optional[TerrainDistribution] = None,
    seed: int = 42,
    elevation_field: str = "elevation",
    radius: float = 1.0,
    cross_patch_smoothing: int = 3,
) -> List[TerrainPatch]:
    """One-call pipeline: generate patches then apply terrain.

    Combines :func:`generate_terrain_patches` and
    :func:`apply_terrain_patches` into a single convenience function.

    Parameters
    ----------
    collection : DetailGridCollection
    globe_grid : PolyGrid
    globe_store : TileDataStore
    n_patches : int
        Number of terrain patches.
    distribution : TerrainDistribution, optional
        Defaults to :data:`EARTHLIKE`.
    seed : int
    elevation_field : str
    radius : float
    cross_patch_smoothing : int

    Returns
    -------
    list of TerrainPatch
        The generated patches (for inspection / debugging).
    """
    patches = generate_terrain_patches(
        globe_grid,
        n_patches=n_patches,
        distribution=distribution,
        seed=seed,
    )
    apply_terrain_patches(
        collection, globe_grid, globe_store, patches,
        seed=seed,
        elevation_field=elevation_field,
        radius=radius,
        cross_patch_smoothing=cross_patch_smoothing,
    )
    return patches
