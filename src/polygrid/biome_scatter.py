# TODO REMOVE — Not used by any live script. Phase 14A biome feature scattering.
"""Biome feature scattering — Poisson disk placement for natural features.

Places feature instances (trees, rocks, bushes, etc.) across tile
textures using Poisson disk sampling for natural spacing.  Positions
are derived from 3-D globe coordinates so placement is continuous
across tile boundaries.

Functions
---------
- :func:`poisson_disk_sample` — 2-D Poisson disk sampling
- :func:`scatter_features_on_tile` — place features for one tile
- :func:`compute_density_field` — globe-wide density map per biome
- :func:`collect_margin_features` — cross-tile boundary overlap

Classes
-------
- :class:`FeatureInstance` — a single placed feature (position, size, colour)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from .noise import fbm_3d


# ═══════════════════════════════════════════════════════════════════
# 14A.3 — FeatureInstance dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FeatureInstance:
    """A single placed feature (e.g. a tree) in tile-local pixel space.

    Attributes
    ----------
    px : float
        X position in tile-local pixel coordinates.
    py : float
        Y position in tile-local pixel coordinates.
    radius : float
        Feature radius in pixels (e.g. canopy radius).
    color : tuple of int
        ``(r, g, b)`` primary colour.
    shadow_color : tuple of int
        ``(r, g, b)`` drop-shadow colour.
    species_id : int
        Index into a species palette (for shape/colour variation).
    depth : float
        Drawing order — higher values drawn later (on top).
    """

    px: float
    py: float
    radius: float
    color: Tuple[int, int, int] = (40, 120, 35)
    shadow_color: Tuple[int, int, int] = (15, 35, 10)
    species_id: int = 0
    depth: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# 14A.1 — Poisson Disk Sampling (Bridson's algorithm)
# ═══════════════════════════════════════════════════════════════════

def poisson_disk_sample(
    width: float,
    height: float,
    min_distance: float,
    *,
    seed: int = 42,
    k: int = 30,
    density_fn: Optional[Callable[[float, float], float]] = None,
) -> List[Tuple[float, float]]:
    """Generate 2-D Poisson disk points in a ``width × height`` rectangle.

    Uses Bridson's fast O(n) algorithm.  Supports spatially varying
    density via *density_fn*.

    Parameters
    ----------
    width, height : float
        Bounding rectangle.
    min_distance : float
        Global minimum distance between any two accepted points.
    seed : int
        Random seed for reproducibility.
    k : int
        Candidate attempts per active point (higher = denser fill).
    density_fn : callable, optional
        ``(x, y) → float`` in ``[0, 1]``.  The local minimum distance
        is ``min_distance / max(density, 0.01)`` — dense regions pack
        tighter.  If *None*, uniform density is used.

    Returns
    -------
    list of (float, float)
        Accepted point positions.
    """
    rng = random.Random(seed)

    # Grid cell size so each cell has at most one sample
    cell_size = min_distance / math.sqrt(2)
    cols = max(1, int(math.ceil(width / cell_size)))
    rows = max(1, int(math.ceil(height / cell_size)))

    # -1 means empty
    grid: List[int] = [-1] * (cols * rows)
    points: List[Tuple[float, float]] = []
    active: List[int] = []

    def _grid_idx(px: float, py: float) -> int:
        c = int(px / cell_size)
        r = int(py / cell_size)
        c = min(c, cols - 1)
        r = min(r, rows - 1)
        return r * cols + c

    def _local_r(px: float, py: float) -> float:
        if density_fn is None:
            return min_distance
        d = max(density_fn(px, py), 0.01)
        return min_distance / d

    def _in_bounds(px: float, py: float) -> bool:
        return 0.0 <= px < width and 0.0 <= py < height

    def _too_close(px: float, py: float, local_r: float) -> bool:
        c0 = int(px / cell_size)
        r0 = int(py / cell_size)
        search = max(2, int(math.ceil(local_r / cell_size)))
        for dr in range(-search, search + 1):
            for dc in range(-search, search + 1):
                r2 = r0 + dr
                c2 = c0 + dc
                if 0 <= r2 < rows and 0 <= c2 < cols:
                    idx = r2 * cols + c2
                    pi = grid[idx]
                    if pi != -1:
                        ox, oy = points[pi]
                        dist_sq = (px - ox) ** 2 + (py - oy) ** 2
                        # Use the stricter of the two local distances
                        other_r = _local_r(ox, oy)
                        effective_r = min(local_r, other_r)
                        if dist_sq < effective_r * effective_r:
                            return True
        return False

    # Seed point
    x0 = rng.uniform(0, width)
    y0 = rng.uniform(0, height)
    points.append((x0, y0))
    grid[_grid_idx(x0, y0)] = 0
    active.append(0)

    while active:
        ai = rng.randint(0, len(active) - 1)
        pi = active[ai]
        px, py = points[pi]
        lr = _local_r(px, py)

        found = False
        for _ in range(k):
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(lr, 2 * lr)
            nx = px + dist * math.cos(angle)
            ny = py + dist * math.sin(angle)

            if not _in_bounds(nx, ny):
                continue

            nlr = _local_r(nx, ny)
            if _too_close(nx, ny, nlr):
                continue

            # Accept point
            new_idx = len(points)
            points.append((nx, ny))
            gi = _grid_idx(nx, ny)
            grid[gi] = new_idx
            active.append(new_idx)
            found = True
            break

        if not found:
            active.pop(ai)

    return points


# ═══════════════════════════════════════════════════════════════════
# 14A.4 — Globe-wide density field
# ═══════════════════════════════════════════════════════════════════

def compute_density_field(
    globe_grid,
    face_ids: Sequence[str],
    *,
    biome_faces: Optional[set] = None,
    seed: int = 42,
    noise_frequency: float = 3.0,
    noise_octaves: int = 4,
    base_density: float = 0.85,
    edge_falloff: float = 0.35,
) -> Dict[str, float]:
    """Compute a density value (0–1) for each globe tile.

    Tiles in ``biome_faces`` get a high density modulated by 3-D noise.
    Tiles not in ``biome_faces`` get 0.0.

    Parameters
    ----------
    globe_grid : PolyGrid (GlobeGrid)
        The globe topology — must have 3-D vertex positions.
    face_ids : sequence of str
        All face IDs to evaluate.
    biome_faces : set of str, optional
        Face IDs that belong to the target biome.  If *None*, all
        *face_ids* are treated as biome members.
    seed : int
        Noise seed.
    noise_frequency, noise_octaves : float, int
        fbm_3d parameters for spatial variation.
    base_density : float
        Density value at biome interior (before noise modulation).
    edge_falloff : float
        How much density drops at biome edges (0–1).

    Returns
    -------
    dict
        ``{face_id: density}`` with values in ``[0.0, 1.0]``.
    """
    from .geometry import face_center_3d

    if biome_faces is None:
        biome_faces = set(face_ids)

    density: Dict[str, float] = {}

    for fid in face_ids:
        if fid not in biome_faces:
            density[fid] = 0.0
            continue

        face = globe_grid.faces.get(fid)
        if face is None:
            density[fid] = 0.0
            continue

        c3d = face_center_3d(globe_grid.vertices, face)
        if c3d is None:
            density[fid] = base_density
        else:
            x, y, z = c3d
            noise_val = fbm_3d(
                x, y, z,
                frequency=noise_frequency,
                octaves=noise_octaves,
                seed=seed,
            )
            # Map noise from [-1,1] to [0,1], then modulate base density
            noise_01 = (noise_val + 1.0) * 0.5
            d = base_density * (1.0 - edge_falloff + edge_falloff * noise_01)
            density[fid] = max(0.0, min(1.0, d))

    return density


# ═══════════════════════════════════════════════════════════════════
# 14A.2 — Scatter features on a tile
# ═══════════════════════════════════════════════════════════════════

def scatter_features_on_tile(
    tile_density: float,
    tile_size: int = 256,
    *,
    min_radius: float = 2.0,
    max_radius: float = 7.0,
    min_distance: float = 4.0,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    shadow_color: Tuple[int, int, int] = (15, 35, 10),
    color_noise: float = 15.0,
    seed: int = 42,
    globe_3d_center: Optional[Tuple[float, float, float]] = None,
) -> List[FeatureInstance]:
    """Place feature instances for one tile using Poisson disk sampling.

    Parameters
    ----------
    tile_density : float
        Overall density for this tile (0–1).  0 = no features,
        1 = maximum packing.
    tile_size : int
        Tile image size in pixels (square).
    min_radius, max_radius : float
        Canopy radius range in pixels.
    min_distance : float
        Minimum pixel distance between feature centres.
    colors : list of (r, g, b), optional
        Species palette.  Defaults to temperate forest greens.
    shadow_color : (r, g, b)
        Drop-shadow colour.
    color_noise : float
        Max per-channel random offset to each tree's colour.
    seed : int
        Random seed for deterministic output.
    globe_3d_center : (x, y, z), optional
        3-D globe position of this tile's centre.  Used to seed
        sub-tile noise for cross-tile coherence.

    Returns
    -------
    list of FeatureInstance
    """
    if tile_density <= 0.01:
        return []

    if colors is None:
        colors = [
            (34, 120, 30),   # dark green
            (50, 135, 40),   # mid green
            (65, 145, 35),   # yellow-green
            (28, 100, 28),   # deep forest green
            (45, 128, 50),   # bright green
        ]

    rng = random.Random(seed)

    # Spatial density function — use globe 3D position to create
    # sub-tile variation (denser in some patches, gaps elsewhere)
    if globe_3d_center is not None:
        gx, gy, gz = globe_3d_center

        def density_fn(px: float, py: float) -> float:
            # Map pixel to fractional offset
            fx = px / tile_size - 0.5
            fy = py / tile_size - 0.5
            # Perturb the 3D position slightly for sub-tile noise
            local_noise = fbm_3d(
                gx + fx * 0.3, gy + fy * 0.3, gz,
                frequency=8.0, octaves=3, seed=seed + 100,
            )
            # noise in [-1,1] → density modulation
            mod = 0.7 + 0.3 * (local_noise + 1.0) * 0.5
            return tile_density * mod
    else:
        def density_fn(px: float, py: float) -> float:
            return tile_density

    # Effective min_distance scales inversely with density
    effective_min_d = min_distance / max(tile_density, 0.1)
    # Clamp to sensible range
    effective_min_d = max(min_distance * 0.5, min(effective_min_d, tile_size * 0.5))

    points = poisson_disk_sample(
        float(tile_size),
        float(tile_size),
        effective_min_d,
        seed=seed,
        density_fn=density_fn,
    )

    instances: List[FeatureInstance] = []
    for i, (px, py) in enumerate(points):
        species = rng.randint(0, len(colors) - 1)
        base_r, base_g, base_b = colors[species]

        # Per-tree colour variation
        dr = rng.randint(-int(color_noise), int(color_noise))
        dg = rng.randint(-int(color_noise), int(color_noise))
        db = rng.randint(-int(color_noise), int(color_noise))
        tree_color = (
            max(0, min(255, base_r + dr)),
            max(0, min(255, base_g + dg)),
            max(0, min(255, base_b + db)),
        )

        radius = rng.uniform(min_radius, max_radius)

        instances.append(FeatureInstance(
            px=px,
            py=py,
            radius=radius,
            color=tree_color,
            shadow_color=shadow_color,
            species_id=species,
            depth=py,  # back-to-front: higher py = further south = drawn later
        ))

    # Sort by depth for painter's algorithm
    instances.sort(key=lambda f: f.depth)
    return instances


# ═══════════════════════════════════════════════════════════════════
# 14A.5 — Cross-tile margin features
# ═══════════════════════════════════════════════════════════════════

def collect_margin_features(
    own_instances: List[FeatureInstance],
    tile_size: int = 256,
    margin: float = 8.0,
) -> Tuple[List[FeatureInstance], List[FeatureInstance]]:
    """Split features into interior and margin.

    Features within *margin* pixels of any tile edge are "margin"
    features — they partially overlap into a neighbour tile and
    should be shared.

    Parameters
    ----------
    own_instances : list of FeatureInstance
        All features placed on this tile.
    tile_size : int
        Tile image size.
    margin : float
        Pixel margin from edge (typically ``max_canopy_radius``).

    Returns
    -------
    (interior, margin_list) : tuple of lists
        *interior* — features fully inside the tile.
        *margin_list* — features near an edge.
    """
    interior: List[FeatureInstance] = []
    margin_list: List[FeatureInstance] = []

    for inst in own_instances:
        near_edge = (
            inst.px - inst.radius < margin
            or inst.px + inst.radius > tile_size - margin
            or inst.py - inst.radius < margin
            or inst.py + inst.radius > tile_size - margin
        )
        if near_edge:
            margin_list.append(inst)
        else:
            interior.append(inst)

    return interior, margin_list


# ═══════════════════════════════════════════════════════════════════
# 16C.1 — Full-slot feature scattering
# ═══════════════════════════════════════════════════════════════════

def scatter_features_fullslot(
    tile_density: float,
    tile_size: int = 256,
    *,
    overscan: float = 0.15,
    min_radius: float = 2.0,
    max_radius: float = 7.0,
    min_distance: float = 4.0,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    shadow_color: Tuple[int, int, int] = (15, 35, 10),
    color_noise: float = 15.0,
    seed: int = 42,
    globe_3d_center: Optional[Tuple[float, float, float]] = None,
    neighbour_densities: Optional[Dict[str, float]] = None,
    neighbour_seeds: Optional[Dict[str, int]] = None,
) -> List[FeatureInstance]:
    """Place features across the full square tile slot (not just hex interior).

    Extends :func:`scatter_features_on_tile` by:

    1. Expanding the sampling area by *overscan* on each side so
       that features populate the corners/edges of the square tile.
    2. Optionally incorporating neighbour density/seed values in the
       margin zone (16C.2) so that features near tile boundaries are
       seeded from the *neighbour's* parameters — producing coherent
       cross-boundary canopy.

    Positions are mapped back to tile-local coordinates after
    scattering in the expanded area.  Features whose centre falls
    outside ``[0, tile_size]`` are included (they'll be partially
    visible through the fade zone from 16B).

    Parameters
    ----------
    tile_density : float
        Density for the tile interior (0–1).
    tile_size : int
        Tile image size in pixels (square).
    overscan : float
        Extra padding fraction.  The scatter area is
        ``tile_size * (1 + 2 * overscan)`` on each axis, centred
        on the tile.
    min_radius, max_radius : float
        Canopy radius range in pixels.
    min_distance : float
        Minimum pixel distance between feature centres.
    colors : list of (r, g, b), optional
    shadow_color : (r, g, b)
    color_noise : float
    seed : int
    globe_3d_center : (x, y, z), optional
    neighbour_densities : dict, optional
        ``{direction: float}`` where direction is one of
        ``"left", "right", "top", "bottom"``.  Density values used
        for margin features in the corresponding overflow zone.
        Missing directions use *tile_density*.
    neighbour_seeds : dict, optional
        ``{direction: int}`` seeds for margin-zone features.  When
        a margin feature falls in a neighbour's overflow zone, it is
        seeded from the neighbour's seed for cross-tile coherence.
        Missing directions use *seed*.

    Returns
    -------
    list of FeatureInstance
        Features covering the full slot area.  Positions may be
        negative or exceed *tile_size* (in the overscan margin).
    """
    if tile_density <= 0.01:
        return []

    if colors is None:
        colors = [
            (34, 120, 30),
            (50, 135, 40),
            (65, 145, 35),
            (28, 100, 28),
            (45, 128, 50),
        ]

    if neighbour_densities is None:
        neighbour_densities = {}
    if neighbour_seeds is None:
        neighbour_seeds = {}

    # Expanded sampling area
    margin_px = tile_size * overscan
    sample_w = tile_size + 2 * margin_px
    sample_h = tile_size + 2 * margin_px

    rng = random.Random(seed)

    # Spatial density function — varies across the expanded area
    def density_fn(px: float, py: float) -> float:
        # Translate from sample coords to tile coords
        tx = px - margin_px
        ty = py - margin_px

        # Determine which zone this point is in
        in_left = tx < 0
        in_right = tx > tile_size
        in_top = ty < 0
        in_bottom = ty > tile_size

        # Use neighbour density for margin zones
        if in_left:
            d = neighbour_densities.get("left", tile_density)
        elif in_right:
            d = neighbour_densities.get("right", tile_density)
        elif in_top:
            d = neighbour_densities.get("top", tile_density)
        elif in_bottom:
            d = neighbour_densities.get("bottom", tile_density)
        else:
            d = tile_density

        # Modulate with globe 3D noise for spatial variation
        if globe_3d_center is not None:
            gx, gy, gz = globe_3d_center
            fx = tx / tile_size - 0.5
            fy = ty / tile_size - 0.5
            local_noise = fbm_3d(
                gx + fx * 0.3, gy + fy * 0.3, gz,
                frequency=8.0, octaves=3, seed=seed + 100,
            )
            mod = 0.7 + 0.3 * (local_noise + 1.0) * 0.5
            d = d * mod

        return max(0.0, d)

    # Effective min_distance scales inversely with density
    effective_min_d = min_distance / max(tile_density, 0.1)
    effective_min_d = max(min_distance * 0.5, min(effective_min_d, tile_size * 0.5))

    points = poisson_disk_sample(
        sample_w,
        sample_h,
        effective_min_d,
        seed=seed,
        density_fn=density_fn,
    )

    instances: List[FeatureInstance] = []
    for i, (px, py) in enumerate(points):
        # Translate to tile-local coordinates
        tx = px - margin_px
        ty = py - margin_px

        species = rng.randint(0, len(colors) - 1)
        base_r, base_g, base_b = colors[species]

        dr = rng.randint(-int(color_noise), int(color_noise))
        dg = rng.randint(-int(color_noise), int(color_noise))
        db = rng.randint(-int(color_noise), int(color_noise))
        tree_color = (
            max(0, min(255, base_r + dr)),
            max(0, min(255, base_g + dg)),
            max(0, min(255, base_b + db)),
        )

        radius = rng.uniform(min_radius, max_radius)

        instances.append(FeatureInstance(
            px=tx,
            py=ty,
            radius=radius,
            color=tree_color,
            shadow_color=shadow_color,
            species_id=species,
            depth=ty,
        ))

    instances.sort(key=lambda f: f.depth)
    return instances
