"""Ocean biome rendering — depth gradients, wave textures, coastal features.

Phase 17 renders ocean tiles with rich depth-based colour gradients,
surface wave patterns, coastal detail (foam, shallow sand, reefs),
and deep-ocean variation — all baked into the tile texture atlas.

Modules
-------
- **17A** — ``OceanFeatureConfig`` (presets) + ``compute_ocean_depth_map()``
- **17B** — Pixel-level ocean rendering functions
- **17C** — ``OceanRenderer`` (``BiomeRenderer`` implementation)

Functions
---------
- :func:`compute_ocean_depth_map` — BFS-based depth map for ocean tiles
- :func:`render_ocean_depth_gradient` — depth-based colour fill
- :func:`render_wave_pattern` — baked wave texture overlay
- :func:`render_coastal_features` — foam, shallow sand, reef patches
- :func:`render_deep_ocean_features` — abyssal darkness, upwelling

Classes
-------
- :class:`OceanFeatureConfig` — tuneable ocean rendering parameters
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# ═══════════════════════════════════════════════════════════════════
# 17A.1 — OceanFeatureConfig
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class OceanFeatureConfig:
    """Tuneable parameters for ocean tile rendering.

    Controls the visual appearance of ocean tiles including depth
    colour gradients, wave patterns, coastal detail, and deep-ocean
    effects.

    Parameters
    ----------
    shallow_color : tuple of int
        RGB colour for shallow coastal water (0–255).
    deep_color : tuple of int
        RGB colour for mid-depth ocean (0–255).
    abyssal_color : tuple of int
        RGB colour for very deep ocean (0–255).
    coastal_foam_color : tuple of int
        RGB colour for surf/foam along coastlines (0–255).
    sand_color : tuple of int
        RGB colour for shallow-water sand/seabed visibility (0–255).
    depth_gradient_power : float
        Controls the shallow→deep transition curve.
        Higher values concentrate shallow colour near the coast.
    wave_frequency : float
        Spatial frequency of baked wave texture (cycles per tile).
    wave_amplitude : float
        Visual strength of wave pattern (fraction of base colour, 0–1).
    foam_width : float
        Width of coastal foam band as fraction of tile size (0–1).
    caustic_frequency : float
        Spatial frequency of underwater caustic ripple pattern.
    caustic_strength : float
        Visual intensity of caustic patterns (0–1).
    ice_latitude_threshold : float
        Latitude (0–1, 0=equator, 1=pole) above which ocean
        becomes icy / lighter.
    reef_probability : float
        Chance [0,1] of shallow-water reef colour patches per tile.
    density_scale : float
        Multiplied with the biome density from the density map.
        Allows per-preset strength adjustment.
    """

    shallow_color: Tuple[int, int, int] = (64, 164, 192)
    deep_color: Tuple[int, int, int] = (16, 48, 112)
    abyssal_color: Tuple[int, int, int] = (6, 16, 42)
    coastal_foam_color: Tuple[int, int, int] = (220, 230, 235)
    sand_color: Tuple[int, int, int] = (180, 170, 130)
    depth_gradient_power: float = 1.8
    wave_frequency: float = 6.0
    wave_amplitude: float = 0.04
    foam_width: float = 0.08
    caustic_frequency: float = 12.0
    caustic_strength: float = 0.06
    ice_latitude_threshold: float = 0.85
    reef_probability: float = 0.3
    density_scale: float = 1.0


# ── Presets ─────────────────────────────────────────────────────

TROPICAL_OCEAN = OceanFeatureConfig(
    shallow_color=(72, 200, 210),
    deep_color=(12, 52, 128),
    abyssal_color=(4, 14, 48),
    coastal_foam_color=(235, 240, 245),
    sand_color=(200, 190, 150),
    depth_gradient_power=2.2,
    wave_frequency=5.0,
    wave_amplitude=0.03,
    foam_width=0.10,
    caustic_frequency=14.0,
    caustic_strength=0.08,
    reef_probability=0.5,
)

TEMPERATE_OCEAN = OceanFeatureConfig(
    shallow_color=(56, 140, 168),
    deep_color=(20, 52, 100),
    abyssal_color=(8, 18, 44),
    coastal_foam_color=(210, 220, 225),
    sand_color=(165, 158, 125),
    depth_gradient_power=1.6,
    wave_frequency=7.0,
    wave_amplitude=0.05,
    foam_width=0.07,
    caustic_frequency=10.0,
    caustic_strength=0.05,
    reef_probability=0.15,
)

ARCTIC_OCEAN = OceanFeatureConfig(
    shallow_color=(100, 150, 170),
    deep_color=(30, 60, 90),
    abyssal_color=(12, 24, 50),
    coastal_foam_color=(230, 235, 240),
    sand_color=(150, 148, 135),
    depth_gradient_power=1.2,
    wave_frequency=4.0,
    wave_amplitude=0.02,
    foam_width=0.05,
    caustic_frequency=6.0,
    caustic_strength=0.02,
    ice_latitude_threshold=0.60,
    reef_probability=0.0,
)

DEEP_OCEAN = OceanFeatureConfig(
    shallow_color=(40, 120, 160),
    deep_color=(10, 36, 90),
    abyssal_color=(4, 10, 32),
    coastal_foam_color=(200, 210, 215),
    sand_color=(140, 135, 110),
    depth_gradient_power=2.5,
    wave_frequency=3.0,
    wave_amplitude=0.02,
    foam_width=0.04,
    caustic_frequency=8.0,
    caustic_strength=0.03,
    reef_probability=0.05,
)

OCEAN_PRESETS: Dict[str, OceanFeatureConfig] = {
    "tropical": TROPICAL_OCEAN,
    "temperate": TEMPERATE_OCEAN,
    "arctic": ARCTIC_OCEAN,
    "deep": DEEP_OCEAN,
}


# ═══════════════════════════════════════════════════════════════════
# 17A.2 — Ocean depth map
# ═══════════════════════════════════════════════════════════════════

def identify_ocean_tiles(
    patches: Sequence,
    *,
    terrain_type: str = "ocean",
) -> Set[str]:
    """Return the set of face IDs classified as ocean.

    Works identically to ``identify_forest_tiles()`` but defaults to
    the ``"ocean"`` terrain type.

    Parameters
    ----------
    patches : sequence of TerrainPatch
    terrain_type : str

    Returns
    -------
    set of str
    """
    face_ids: Set[str] = set()
    for patch in patches:
        if patch.terrain_type == terrain_type:
            face_ids.update(patch.face_ids)
    return face_ids


def compute_ocean_depth_map(
    globe_grid,
    globe_store,
    ocean_faces: Set[str],
    *,
    water_level: float = 0.12,
    max_bfs_depth: int = 50,
    elevation_weight: float = 0.5,
    distance_weight: float = 0.5,
) -> Dict[str, float]:
    """Compute a normalised [0, 1] depth value for every ocean tile.

    Depth is a hybrid of:
    - **Elevation-based depth:** ``(water_level - elevation) / water_level``
    - **Distance from coast:** BFS hop count from the nearest land tile,
      normalised by the maximum distance found.

    The two are combined as::

        depth = elevation_weight * elev_depth + distance_weight * dist_depth

    Parameters
    ----------
    globe_grid : PolyGrid / GlobeGrid
        The globe grid (for face adjacency).
    globe_store : TileDataStore
        Must contain ``"elevation"`` field.
    ocean_faces : set of str
        Face IDs classified as ocean.
    water_level : float
        Elevation threshold — tiles below this are ocean.
    max_bfs_depth : int
        Maximum BFS distance to search (caps computation).
    elevation_weight, distance_weight : float
        Relative weights for the two depth components.
        Need not sum to 1 — they are normalised internally.

    Returns
    -------
    dict
        ``{face_id: depth}`` for every face in *ocean_faces*.
        Values are in ``[0, 1]``: 0 = coastline, 1 = deepest ocean.
    """
    from .algorithms import get_face_adjacency

    adjacency = get_face_adjacency(globe_grid)
    land_faces = set(globe_grid.faces.keys()) - ocean_faces

    # ── BFS distance from coast ─────────────────────────────────
    # Seed the BFS from all land tiles (distance 0 for their
    # ocean neighbours = distance 1)
    bfs_dist: Dict[str, int] = {}
    queue: deque[Tuple[str, int]] = deque()

    # Initialise: land tiles bordering ocean are the BFS seeds
    for land_fid in land_faces:
        for nbr in adjacency.get(land_fid, []):
            if nbr in ocean_faces and nbr not in bfs_dist:
                bfs_dist[nbr] = 1
                queue.append((nbr, 1))

    # Also handle ocean tiles that might not border land (isolated deep ocean)
    # — they'll get distance = max_bfs_depth

    while queue:
        fid, dist = queue.popleft()
        if dist >= max_bfs_depth:
            continue
        for nbr in adjacency.get(fid, []):
            if nbr in ocean_faces and nbr not in bfs_dist:
                bfs_dist[nbr] = dist + 1
                queue.append((nbr, dist + 1))

    # Assign max distance to unreached ocean tiles
    max_dist_found = max(bfs_dist.values()) if bfs_dist else 1
    for fid in ocean_faces:
        if fid not in bfs_dist:
            bfs_dist[fid] = max_dist_found

    # ── Elevation-based depth ───────────────────────────────────
    w_total = elevation_weight + distance_weight
    if w_total <= 0:
        w_total = 1.0
    w_elev = elevation_weight / w_total
    w_dist = distance_weight / w_total

    # Normalise BFS distances to [0, 1]
    max_dist = max(bfs_dist.values()) if bfs_dist else 1
    if max_dist == 0:
        max_dist = 1

    depth_map: Dict[str, float] = {}
    for fid in ocean_faces:
        # Elevation component
        elev = globe_store.get(fid, "elevation")
        if elev is None:
            elev = 0.0
        elev_depth = max(0.0, min(1.0,
            (water_level - elev) / max(water_level, 0.001)
        ))

        # Distance component
        dist_depth = bfs_dist.get(fid, max_dist) / max_dist

        depth = w_elev * elev_depth + w_dist * dist_depth
        depth_map[fid] = max(0.0, min(1.0, depth))

    return depth_map


def compute_coast_direction(
    globe_grid,
    face_id: str,
    ocean_faces: Set[str],
) -> Optional[Tuple[float, float, float]]:
    """Compute the direction from an ocean tile toward the nearest coast.

    Returns a normalised 3D vector pointing from the ocean tile's
    centre toward the average position of its land-bordering neighbours.
    Returns ``None`` if the tile has no land neighbours.

    Parameters
    ----------
    globe_grid : PolyGrid / GlobeGrid
    face_id : str
    ocean_faces : set of str

    Returns
    -------
    tuple or None
        ``(dx, dy, dz)`` normalised direction, or *None*.
    """
    from .algorithms import get_face_adjacency
    from .geometry import face_center_3d

    adjacency = get_face_adjacency(globe_grid)
    neighbours = adjacency.get(face_id, [])
    land_neighbours = [n for n in neighbours if n not in ocean_faces]

    if not land_neighbours:
        return None

    # This tile's centre
    face = globe_grid.faces.get(face_id)
    if face is None:
        return None
    cx, cy, cz = face_center_3d(globe_grid.vertices, face)

    # Average land-neighbour centre
    lx, ly, lz = 0.0, 0.0, 0.0
    count = 0
    for nid in land_neighbours:
        nface = globe_grid.faces.get(nid)
        if nface is not None:
            nx, ny, nz = face_center_3d(globe_grid.vertices, nface)
            lx += nx
            ly += ny
            lz += nz
            count += 1

    if count == 0:
        return None

    lx /= count
    ly /= count
    lz /= count

    # Direction: land centre - ocean centre
    dx = lx - cx
    dy = ly - cy
    dz = lz - cz
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-12:
        return None
    return (dx / length, dy / length, dz / length)


# ═══════════════════════════════════════════════════════════════════
# 17B — Ocean Texture Rendering
# ═══════════════════════════════════════════════════════════════════

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def _lerp_color(
    c1: Tuple[int, int, int],
    c2: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    """Linear interpolation between two RGB colours."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _depth_color(
    depth: float,
    config: OceanFeatureConfig,
) -> Tuple[int, int, int]:
    """Map a [0, 1] depth to an RGB colour using the config's three stops.

    - depth 0.0 → shallow_color
    - depth 0.5 → deep_color
    - depth 1.0 → abyssal_color

    The ``depth_gradient_power`` bends the curve.
    """
    d = max(0.0, min(1.0, depth)) ** config.depth_gradient_power
    if d < 0.5:
        t = d * 2.0  # 0→1 within shallow→deep
        return _lerp_color(config.shallow_color, config.deep_color, t)
    else:
        t = (d - 0.5) * 2.0  # 0→1 within deep→abyssal
        return _lerp_color(config.deep_color, config.abyssal_color, t)


# ── 17B.1 — Depth gradient fill ────────────────────────────────

def render_ocean_depth_gradient(
    image: "Image.Image",
    depth: float,
    config: Optional[OceanFeatureConfig] = None,
    *,
    seed: int = 0,
) -> None:
    """Fill a tile image with a depth-dependent colour gradient.

    Uses cubic interpolation between shallow/deep/abyssal colours
    based on the tile's normalised depth value, with low-frequency
    noise for subtle spatial variation.

    Parameters
    ----------
    image : PIL.Image (RGB)
        Modified **in place**.
    depth : float
        Normalised ocean depth in ``[0, 1]``.
    config : OceanFeatureConfig, optional
    seed : int
        Noise seed for spatial variation.
    """
    from .noise import fbm

    if config is None:
        config = TEMPERATE_OCEAN

    w, h = image.size
    pixels = image.load()
    base_color = _depth_color(depth, config)

    for y in range(h):
        for x in range(w):
            # Low-frequency noise → local depth perturbation
            nx = x / max(w, 1)
            ny = y / max(h, 1)
            noise_val = fbm(
                nx * 3.0, ny * 3.0,
                octaves=2, frequency=1.0, seed=seed,
            )
            # Perturb depth ±0.08
            local_depth = max(0.0, min(1.0, depth + noise_val * 0.08))
            c = _depth_color(local_depth, config)
            pixels[x, y] = c


# ── 17B.2 — Wave pattern overlay ───────────────────────────────

def render_wave_pattern(
    image: "Image.Image",
    depth: float,
    config: Optional[OceanFeatureConfig] = None,
    *,
    seed: int = 0,
) -> None:
    """Overlay a baked wave texture on an ocean tile.

    Produces low-frequency sinusoidal ridges modulated by noise,
    rendered as subtle brightness variation.  Wave amplitude decreases
    with depth (deep ocean is calmer).

    Parameters
    ----------
    image : PIL.Image (RGB)
        Modified **in place**.
    depth : float
        Normalised depth [0, 1].
    config : OceanFeatureConfig, optional
    seed : int
    """
    from .noise import fbm

    if config is None:
        config = TEMPERATE_OCEAN

    w, h = image.size
    pixels = image.load()
    freq = config.wave_frequency
    amp = config.wave_amplitude

    # Amplitude decreases with depth — deep ocean calmer
    depth_attenuation = max(0.2, 1.0 - depth * 0.7)
    effective_amp = amp * depth_attenuation

    # Direction perturbation from noise
    for y in range(h):
        for x in range(w):
            nx = x / max(w, 1)
            ny = y / max(h, 1)

            # Primary wave (roughly horizontal)
            wave1 = math.sin(ny * freq * 2.0 * math.pi + nx * 1.5)
            # Secondary wave (diagonal)
            wave2 = math.sin((nx + ny) * freq * 1.2 * math.pi)
            # Noise modulation breaks up regularity
            noise_mod = fbm(
                nx * 8.0, ny * 8.0,
                octaves=2, frequency=1.0, seed=seed + 7777,
            )
            combined = (wave1 * 0.6 + wave2 * 0.3 + noise_mod * 0.1)
            brightness = 1.0 + combined * effective_amp

            r, g, b = pixels[x, y]
            pixels[x, y] = (
                max(0, min(255, int(r * brightness))),
                max(0, min(255, int(g * brightness))),
                max(0, min(255, int(b * brightness))),
            )


# ── 17B.3 — Coastal features ───────────────────────────────────

def render_coastal_features(
    image: "Image.Image",
    depth: float,
    coast_direction: Optional[Tuple[float, float, float]],
    config: Optional[OceanFeatureConfig] = None,
    *,
    seed: int = 0,
) -> None:
    """Render coastal detail for shallow ocean tiles.

    Draws foam/surf bands along the coast-facing edge, shallow-water
    sand visibility, and optional reef patches with caustic ripples.

    Only active when ``depth < 0.3`` (near coast). For deeper tiles
    this is a no-op.

    Parameters
    ----------
    image : PIL.Image (RGB)
        Modified **in place**.
    depth : float
    coast_direction : (dx, dy, dz) or None
        Direction toward coast from :func:`compute_coast_direction`.
        If None, a default direction is used.
    config : OceanFeatureConfig, optional
    seed : int
    """
    from .noise import fbm
    import random

    if config is None:
        config = TEMPERATE_OCEAN

    if depth > 0.3:
        return  # too deep for coastal features

    w, h = image.size
    pixels = image.load()
    rng = random.Random(seed)

    # Use coast direction to determine which edge of the tile faces land.
    # Project to 2D: simplify to (dx, dy) in [−1, 1] representing
    # which tile edge the coast is on.
    if coast_direction is not None:
        cdx, cdy = coast_direction[0], coast_direction[1]
        cd_len = math.sqrt(cdx * cdx + cdy * cdy)
        if cd_len > 1e-6:
            cdx /= cd_len
            cdy /= cd_len
        else:
            cdx, cdy = 0.0, -1.0
    else:
        cdx, cdy = 0.0, -1.0  # default: coast is "north"

    foam_width_px = max(1, int(config.foam_width * w))

    for y in range(h):
        for x in range(w):
            # Normalised position [−0.5, 0.5] from tile centre
            nx = (x / max(w, 1)) - 0.5
            ny = (y / max(h, 1)) - 0.5

            # Distance along coast direction (how close to coast edge)
            coast_dist = nx * cdx + ny * cdy  # positive = toward coast
            # Remap: 0.5 = at tile edge toward coast
            coast_proximity = (coast_dist + 0.5)  # 0=away, 1=toward coast

            r, g, b = pixels[x, y]

            # ── Foam/surf line along coast edge ─────────────────
            foam_zone = max(0.0, coast_proximity - (1.0 - config.foam_width))
            foam_zone = min(1.0, foam_zone / max(config.foam_width, 0.001))
            # Noise breaks up the foam line
            foam_noise = fbm(
                x / max(w, 1) * 20.0, y / max(h, 1) * 20.0,
                octaves=2, frequency=1.0, seed=seed + 3333,
            )
            foam_zone *= max(0.0, 0.5 + foam_noise * 0.6)
            # Foam fades with depth
            foam_strength = foam_zone * max(0.0, 1.0 - depth * 4.0)

            if foam_strength > 0.01:
                fc = config.coastal_foam_color
                r = int(r + (fc[0] - r) * foam_strength)
                g = int(g + (fc[1] - g) * foam_strength)
                b = int(b + (fc[2] - b) * foam_strength)

            # ── Shallow-water sand visibility ───────────────────
            if depth < 0.1:
                sand_t = (1.0 - depth / 0.1) * coast_proximity * 0.3
                sc = config.sand_color
                r = int(r + (sc[0] - r) * sand_t)
                g = int(g + (sc[1] - g) * sand_t)
                b = int(b + (sc[2] - b) * sand_t)

            # ── Caustic ripple pattern ──────────────────────────
            if depth < 0.2 and config.caustic_strength > 0:
                cx_freq = config.caustic_frequency
                caustic = math.sin(
                    x / max(w, 1) * cx_freq * math.pi
                ) * math.sin(
                    y / max(h, 1) * cx_freq * 1.3 * math.pi
                )
                caustic_noise = fbm(
                    x / max(w, 1) * 15.0, y / max(h, 1) * 15.0,
                    octaves=2, frequency=1.0, seed=seed + 5555,
                )
                caustic_val = (caustic * 0.7 + caustic_noise * 0.3)
                caustic_bright = 1.0 + caustic_val * config.caustic_strength * (1.0 - depth * 5.0)
                r = max(0, min(255, int(r * caustic_bright)))
                g = max(0, min(255, int(g * caustic_bright)))
                b = max(0, min(255, int(b * caustic_bright)))

            pixels[x, y] = (max(0, min(255, r)),
                            max(0, min(255, g)),
                            max(0, min(255, b)))

    # ── Reef patches (probability-based) ────────────────────────
    if depth < 0.15 and rng.random() < config.reef_probability:
        _render_reef_patches(image, depth, config, seed=seed)


def _render_reef_patches(
    image: "Image.Image",
    depth: float,
    config: OceanFeatureConfig,
    *,
    seed: int = 0,
) -> None:
    """Draw 1-3 irregular reef colour patches in shallow water."""
    from .noise import fbm
    import random

    w, h = image.size
    pixels = image.load()
    rng = random.Random(seed + 9999)

    n_reefs = rng.randint(1, 3)
    for _ in range(n_reefs):
        cx = rng.uniform(0.2, 0.8) * w
        cy = rng.uniform(0.2, 0.8) * h
        radius = rng.uniform(0.05, 0.12) * min(w, h)

        # Reef colour: darker greenish variant of shallow
        reef_color = (
            max(0, config.shallow_color[0] - 20),
            min(255, config.shallow_color[1] + 15),
            max(0, config.shallow_color[2] - 30),
        )

        for y in range(max(0, int(cy - radius)), min(h, int(cy + radius) + 1)):
            for x in range(max(0, int(cx - radius)), min(w, int(cx + radius) + 1)):
                dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist > radius:
                    continue
                # Soft edge
                t = 1.0 - (dist / radius)
                noise_val = fbm(
                    x / max(w, 1) * 25.0, y / max(h, 1) * 25.0,
                    octaves=2, frequency=1.0, seed=seed + 8888,
                )
                t *= max(0.0, 0.5 + noise_val * 0.5)
                t *= 0.35  # subtle

                r, g, b = pixels[x, y]
                r = int(r + (reef_color[0] - r) * t)
                g = int(g + (reef_color[1] - g) * t)
                b = int(b + (reef_color[2] - b) * t)
                pixels[x, y] = (max(0, min(255, r)),
                                max(0, min(255, g)),
                                max(0, min(255, b)))


# ── 17B.4 — Deep ocean features ────────────────────────────────

def render_deep_ocean_features(
    image: "Image.Image",
    depth: float,
    config: Optional[OceanFeatureConfig] = None,
    *,
    seed: int = 0,
) -> None:
    """Add deep-ocean detail: abyssal darkness + subtle upwelling.

    Only active when ``depth > 0.5``.  For shallower tiles this is
    a no-op.

    Parameters
    ----------
    image : PIL.Image (RGB)
        Modified **in place**.
    depth : float
    config : OceanFeatureConfig, optional
    seed : int
    """
    from .noise import fbm

    if config is None:
        config = TEMPERATE_OCEAN

    if depth <= 0.5:
        return  # not deep enough

    w, h = image.size
    pixels = image.load()

    # How "deep" within the deep range (0.5→1.0 mapped to 0→1)
    deep_factor = (depth - 0.5) * 2.0

    for y in range(h):
        for x in range(w):
            nx = x / max(w, 1)
            ny = y / max(h, 1)

            r, g, b = pixels[x, y]

            # ── Abyssal darkening ───────────────────────────────
            darken = 1.0 - deep_factor * 0.25  # up to 25% darker
            r = int(r * darken)
            g = int(g * darken)
            b = int(b * darken)

            # ── Subtle upwelling lighter patches ────────────────
            upwelling = fbm(
                nx * 4.0, ny * 4.0,
                octaves=2, frequency=1.0, seed=seed + 6666,
            )
            # Only lighten, never darken further
            if upwelling > 0.3:
                brighten = (upwelling - 0.3) * 0.08 * deep_factor
                r = min(255, int(r * (1.0 + brighten)))
                g = min(255, int(g * (1.0 + brighten)))
                b = min(255, int(b * (1.0 + brighten)))

            pixels[x, y] = (max(0, r), max(0, g), max(0, b))


# ═══════════════════════════════════════════════════════════════════
# 17B.5 — Composite ocean rendering
# ═══════════════════════════════════════════════════════════════════

def render_ocean_tile(
    ground_image: "Image.Image",
    depth: float,
    *,
    config: Optional[OceanFeatureConfig] = None,
    coast_direction: Optional[Tuple[float, float, float]] = None,
    seed: int = 0,
) -> "Image.Image":
    """Render a complete ocean tile texture.

    Applies all ocean rendering layers in order:
    1. Depth gradient fill
    2. Wave pattern overlay
    3. Coastal features (if shallow)
    4. Deep ocean features (if deep)

    Parameters
    ----------
    ground_image : PIL.Image
        The base ground texture (will be replaced with ocean colours).
    depth : float
        Normalised depth [0, 1].
    config : OceanFeatureConfig, optional
    coast_direction : tuple or None
    seed : int

    Returns
    -------
    PIL.Image (RGB)
    """
    if config is None:
        config = TEMPERATE_OCEAN

    # Work on a copy
    img = ground_image.copy().convert("RGB")

    # 1. Base depth gradient
    render_ocean_depth_gradient(img, depth, config, seed=seed)

    # 2. Wave pattern
    render_wave_pattern(img, depth, config, seed=seed)

    # 3. Coastal features (only for shallow tiles)
    if depth < 0.3:
        render_coastal_features(img, depth, coast_direction, config, seed=seed)

    # 4. Deep ocean features (only for deep tiles)
    if depth > 0.5:
        render_deep_ocean_features(img, depth, config, seed=seed)

    return img
