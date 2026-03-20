# TODO REMOVE — Not used by any live script. Phase 19 coastline transitions.
"""Coastline transition rendering — natural biome boundaries.

Phase 19 replaces hard hexagonal biome boundaries with **noise-warped
coastlines** that look like real shorelines.  The transition system is
biome-agnostic at its core but Phase 19 focuses on forest↔ocean.

Architecture
------------
For each tile that borders a different biome, we generate a per-pixel
**coastline mask** — a ``(tile_size, tile_size)`` float32 array where:

- 0.0 = fully "own" biome (the tile's assigned biome)
- 1.0 = fully "other" biome (the neighbouring tile's biome)

The boundary between 0 and 1 follows a **noise-warped curve** that
produces organic headlands, bays, and inlets instead of a straight line
along the hex edge.

Integration: ``build_apron_feature_atlas()`` detects transition tiles,
renders both biomes, and composites through the mask.

Functions
---------
- :func:`classify_tile_biome_context` — interior vs edge vs transition
- :func:`compute_coastline_mask` — per-pixel mask with noise-warped boundary
- :func:`compute_edge_base_line` — base separating line for one neighbour
- :func:`blend_biome_images` — composite two biome images via mask
- :func:`render_coastal_strip` — paint beach/sand/foam in the transition zone

Classes
-------
- :class:`CoastlineConfig` — tuneable noise + rendering parameters
- :class:`CoastlineMask` — mask array + biome metadata
- :class:`TileBiomeContext` — classification of a tile's biome neighbourhood
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# 19A.1 — CoastlineConfig
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CoastlineConfig:
    """Tuneable parameters for coastline mask generation and rendering.

    Parameters
    ----------
    noise_frequency : float
        Spatial frequency of the coastline warp noise.  Higher = more
        wiggly coastline with smaller-scale features.
    noise_octaves : int
        Number of fBm octaves for coastline displacement.
    noise_amplitude : float
        Maximum displacement of the coastline from the hex edge,
        as a fraction of tile size.  0.15 = up to 15% of tile width.
    noise_lacunarity : float
        Frequency multiplier between noise octaves.
    noise_persistence : float
        Amplitude decay between noise octaves.
    transition_width : float
        Width of the smooth gradient zone around the coastline, as
        fraction of tile size.  0.08 = 8% of tile width on each side.
    beach_width : float
        Width of the beach/sand strip on the land side, as fraction
        of tile size.
    foam_width : float
        Width of the foam/shallow-water strip on the ocean side.
    seed_offset : int
        Added to per-tile seed for coastline noise.
    beach_color : tuple of int
        RGB colour for the sandy beach strip.
    foam_color : tuple of int
        RGB colour for the coastal foam strip.
    beach_noise_amplitude : float
        Colour variation in the beach strip.
    foam_noise_amplitude : float
        Colour variation in the foam strip.
    """

    noise_frequency: float = 4.0
    noise_octaves: int = 4
    noise_amplitude: float = 0.18
    noise_lacunarity: float = 2.0
    noise_persistence: float = 0.5
    transition_width: float = 0.06
    beach_width: float = 0.04
    foam_width: float = 0.03
    seed_offset: int = 19000
    beach_color: Tuple[int, int, int] = (210, 195, 155)
    foam_color: Tuple[int, int, int] = (220, 230, 235)
    beach_noise_amplitude: float = 15.0
    foam_noise_amplitude: float = 10.0


# Presets
GENTLE_COAST = CoastlineConfig(
    noise_frequency=3.0,
    noise_octaves=3,
    noise_amplitude=0.12,
    transition_width=0.08,
    beach_width=0.05,
    foam_width=0.04,
)

RUGGED_COAST = CoastlineConfig(
    noise_frequency=6.0,
    noise_octaves=5,
    noise_amplitude=0.22,
    transition_width=0.04,
    beach_width=0.03,
    foam_width=0.02,
)

ARCHIPELAGO_COAST = CoastlineConfig(
    noise_frequency=8.0,
    noise_octaves=5,
    noise_amplitude=0.28,
    noise_persistence=0.55,
    transition_width=0.05,
    beach_width=0.03,
    foam_width=0.03,
)

COASTLINE_PRESETS: Dict[str, CoastlineConfig] = {
    "default": CoastlineConfig(),
    "gentle": GENTLE_COAST,
    "rugged": RUGGED_COAST,
    "archipelago": ARCHIPELAGO_COAST,
}


# ═══════════════════════════════════════════════════════════════════
# 19A.2 — Tile biome context classification
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TileBiomeContext:
    """Classification of a tile's biome neighbourhood.

    Attributes
    ----------
    face_id : str
        The tile's face ID.
    own_biome : str
        This tile's assigned biome (e.g. "forest", "ocean").
    is_interior : bool
        True if all neighbours share the same biome.
    is_edge : bool
        True if at least one neighbour has a different biome.
    neighbour_biomes : dict
        ``{neighbour_face_id: biome_type}`` for all neighbours.
    edge_neighbours : dict
        ``{neighbour_face_id: biome_type}`` for neighbours that
        differ from own_biome.  Empty if is_interior.
    """

    face_id: str
    own_biome: str
    is_interior: bool
    is_edge: bool
    neighbour_biomes: Dict[str, str]
    edge_neighbours: Dict[str, str]


def classify_tile_biome_context(
    face_id: str,
    biome_type_map: Dict[str, str],
    adjacency: Dict[str, List[str]],
    *,
    default_biome: str = "terrain",
) -> TileBiomeContext:
    """Classify a tile's biome neighbourhood.

    Parameters
    ----------
    face_id : str
        The tile to classify.
    biome_type_map : dict
        ``{face_id: biome_type}`` for every tile.
    adjacency : dict
        ``{face_id: [neighbour_ids]}`` from :func:`get_face_adjacency`.
    default_biome : str
        Biome type assigned to tiles not in *biome_type_map*.

    Returns
    -------
    TileBiomeContext
    """
    own_biome = biome_type_map.get(face_id, default_biome)
    neighbours = adjacency.get(face_id, [])

    neighbour_biomes: Dict[str, str] = {}
    edge_neighbours: Dict[str, str] = {}

    for nid in neighbours:
        nbr_biome = biome_type_map.get(nid, default_biome)
        neighbour_biomes[nid] = nbr_biome
        if nbr_biome != own_biome:
            edge_neighbours[nid] = nbr_biome

    is_interior = len(edge_neighbours) == 0
    is_edge = not is_interior

    return TileBiomeContext(
        face_id=face_id,
        own_biome=own_biome,
        is_interior=is_interior,
        is_edge=is_edge,
        neighbour_biomes=neighbour_biomes,
        edge_neighbours=edge_neighbours,
    )


def classify_all_tiles(
    biome_type_map: Dict[str, str],
    adjacency: Dict[str, List[str]],
    *,
    default_biome: str = "terrain",
) -> Dict[str, TileBiomeContext]:
    """Classify every tile in the biome map.

    Returns
    -------
    dict
        ``{face_id: TileBiomeContext}``
    """
    all_faces = set(biome_type_map.keys())
    # Also include faces from adjacency that might not be in biome_type_map
    for fid, nbrs in adjacency.items():
        all_faces.add(fid)
        all_faces.update(nbrs)

    return {
        fid: classify_tile_biome_context(
            fid, biome_type_map, adjacency,
            default_biome=default_biome,
        )
        for fid in all_faces
    }


# ═══════════════════════════════════════════════════════════════════
# 19A.3 — Coastline mask computation
# ═══════════════════════════════════════════════════════════════════

def compute_edge_direction(
    globe_grid,
    face_id: str,
    neighbour_id: str,
) -> Tuple[float, float]:
    """Compute the 2D direction from a tile's centre to a neighbour.

    Projects 3D face centres into the tile's local tangent plane,
    returning a normalised ``(dx, dy)`` direction.  For flat grids,
    uses the 2D face centres directly.

    Parameters
    ----------
    globe_grid : PolyGrid / GlobeGrid
    face_id : str
    neighbour_id : str

    Returns
    -------
    (dx, dy) : normalised 2D direction
    """
    from .geometry import face_center, face_center_3d

    face = globe_grid.faces.get(face_id)
    nbr_face = globe_grid.faces.get(neighbour_id)
    if face is None or nbr_face is None:
        return (0.0, -1.0)  # fallback: "north"

    # Try 3D first (globe grids)
    try:
        c1 = face_center_3d(globe_grid.vertices, face)
        c2 = face_center_3d(globe_grid.vertices, nbr_face)
        # Project to 2D using the first two components relative to c1
        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]
    except (AttributeError, TypeError):
        # Flat grid: use 2D centres
        c1 = face_center(globe_grid.vertices, face)
        c2 = face_center(globe_grid.vertices, nbr_face)
        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]

    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-12:
        return (0.0, -1.0)
    return (dx / length, dy / length)


def _noise_warp_1d(
    t: float,
    *,
    frequency: float = 4.0,
    octaves: int = 4,
    amplitude: float = 0.18,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
    seed: int = 0,
) -> float:
    """1D noise displacement for coastline warping.

    Uses the project's fBm noise sampled along a 1D parameter *t*
    (position along the coastline).

    Parameters
    ----------
    t : float
        Position along the coastline (0–1).
    frequency, octaves, amplitude, lacunarity, persistence, seed
        Noise parameters.

    Returns
    -------
    float
        Displacement value in approximately [-amplitude, +amplitude].
    """
    from .noise import fbm

    # Sample fBm at (t * freq, seed_offset) to get 1D noise
    # We use a 2D noise function with fixed y to create 1D noise.
    val = fbm(
        t * frequency * 3.0,
        seed * 0.1 + 0.5,
        octaves=octaves,
        lacunarity=lacunarity,
        persistence=persistence,
        frequency=1.0,
        seed=seed,
    )
    return val * amplitude


def compute_coastline_mask(
    tile_size: int,
    edge_neighbours: Dict[str, str],
    globe_grid,
    face_id: str,
    *,
    config: Optional[CoastlineConfig] = None,
    seed: int = 0,
) -> np.ndarray:
    """Generate a per-pixel coastline mask for a transition tile.

    For each pixel, computes the signed distance to the noise-warped
    coastline boundary, then maps it through a smooth step to produce
    a ``[0, 1]`` mask.

    - 0.0 = fully "own" biome
    - 1.0 = fully "other" biome

    When multiple neighbours have a different biome, their individual
    masks are combined (max), producing a coastline that wraps around
    from multiple directions.

    Parameters
    ----------
    tile_size : int
        Output mask size (square).
    edge_neighbours : dict
        ``{neighbour_face_id: biome_type}`` — neighbours that differ
        from the tile's own biome.
    globe_grid : PolyGrid / GlobeGrid
        For computing neighbour directions.
    face_id : str
        This tile's face ID.
    config : CoastlineConfig, optional
    seed : int
        Per-tile seed for reproducible noise.

    Returns
    -------
    np.ndarray
        ``(tile_size, tile_size)`` float32 in ``[0, 1]``.
    """
    if config is None:
        config = CoastlineConfig()

    mask = np.zeros((tile_size, tile_size), dtype=np.float32)

    if not edge_neighbours:
        return mask  # interior tile — all zeros

    # For each different-biome neighbour, compute a directional mask
    for nid in edge_neighbours:
        dir_x, dir_y = compute_edge_direction(globe_grid, face_id, nid)

        # Per-edge seed: combine tile and neighbour IDs for reproducibility
        # and cross-tile continuity (both tiles sharing this edge use
        # the same noise seed for the shared boundary)
        edge_seed = (
            config.seed_offset
            + seed
            + _stable_edge_hash(face_id, nid)
        )

        edge_mask = _compute_single_edge_mask(
            tile_size,
            dir_x, dir_y,
            config=config,
            seed=edge_seed,
        )

        # Combine via max — where any edge says "other biome", use it
        np.maximum(mask, edge_mask, out=mask)

    return mask


def _stable_edge_hash(fid_a: str, fid_b: str) -> int:
    """Compute a hash that is the same regardless of argument order.

    This ensures that both tiles sharing an edge use the same noise
    seed for that edge's coastline, producing a continuous coastline
    across the boundary.
    """
    # Sort to ensure order-independence
    pair = tuple(sorted([fid_a, fid_b]))
    return hash(pair) % 1_000_000


def _compute_single_edge_mask(
    tile_size: int,
    dir_x: float,
    dir_y: float,
    *,
    config: CoastlineConfig,
    seed: int,
) -> np.ndarray:
    """Compute a coastline mask for a single boundary direction.

    The boundary line is perpendicular to ``(dir_x, dir_y)`` and
    positioned at the tile edge in that direction.  Noise displaces
    the line to create organic coastline shapes.

    Parameters
    ----------
    tile_size : int
    dir_x, dir_y : float
        Normalised direction toward the differing-biome neighbour.
    config : CoastlineConfig
    seed : int

    Returns
    -------
    np.ndarray
        ``(tile_size, tile_size)`` float32 in ``[0, 1]``.
    """
    mask = np.zeros((tile_size, tile_size), dtype=np.float32)

    half = tile_size / 2.0
    tw = max(config.transition_width * tile_size, 1.0)

    # The base coastline sits at ~40% from centre toward the edge.
    # This places it inside the tile rather than exactly at the hex
    # boundary, so the transition zone is visible.
    base_offset = 0.35 * half

    # Pre-compute noise for each row of pixels along the coastline
    # to avoid recomputing per-pixel
    for py in range(tile_size):
        for px in range(tile_size):
            # Position relative to tile centre, normalised to [-0.5, 0.5]
            nx = (px - half) / tile_size
            ny = (py - half) / tile_size

            # Signed distance along the neighbour direction
            # Positive = toward the neighbour (i.e. toward the edge)
            dist_along = nx * dir_x + ny * dir_y

            # Position perpendicular to the boundary (for noise sampling)
            perp = -nx * dir_y + ny * dir_x

            # Noise displacement of the coastline
            # Use the perpendicular coordinate as the noise parameter
            # so the coastline wiggles along its length
            warp = _noise_warp_1d(
                perp + 0.5,  # remap from [-0.5, 0.5] to [0, 1]
                frequency=config.noise_frequency,
                octaves=config.noise_octaves,
                amplitude=config.noise_amplitude,
                lacunarity=config.noise_lacunarity,
                persistence=config.noise_persistence,
                seed=seed,
            )

            # Warped coastline position (as signed distance from centre)
            coastline_pos = base_offset / tile_size + warp

            # Signed distance from pixel to the coastline
            # (positive = on the "other biome" side)
            signed_dist = dist_along - coastline_pos

            # Smooth step: map signed distance through a smooth gradient
            # over the transition width
            t = (signed_dist / (tw / tile_size)) * 0.5 + 0.5
            t = max(0.0, min(1.0, t))

            # Smooth hermite interpolation for natural-looking transition
            t = t * t * (3.0 - 2.0 * t)

            mask[py, px] = t

    return mask


# ═══════════════════════════════════════════════════════════════════
# 19A.4 — CoastlineMask dataclass
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CoastlineMask:
    """Holds a per-pixel coastline mask and biome metadata.

    Attributes
    ----------
    mask : np.ndarray
        ``(tile_size, tile_size)`` float32 in ``[0, 1]``.
        0.0 = own_biome, 1.0 = other_biome.
    face_id : str
        The tile this mask was generated for.
    own_biome : str
        The tile's assigned biome type.
    other_biomes : set of str
        The biome type(s) of differing neighbours.
    config : CoastlineConfig
        The config used to generate this mask.
    seed : int
        The seed used for noise generation.
    """

    mask: np.ndarray
    face_id: str
    own_biome: str
    other_biomes: Set[str]
    config: CoastlineConfig
    seed: int

    @property
    def tile_size(self) -> int:
        """Tile size (width = height)."""
        return self.mask.shape[0]

    @property
    def has_transition(self) -> bool:
        """True if the mask contains any transition pixels."""
        return bool(np.any(self.mask > 0.01) and np.any(self.mask < 0.99))

    @property
    def transition_fraction(self) -> float:
        """Fraction of pixels in the transition zone (between 0.01 and 0.99)."""
        between = (self.mask > 0.01) & (self.mask < 0.99)
        return float(np.sum(between)) / max(1, self.mask.size)

    @property
    def coastline_pixels(self) -> np.ndarray:
        """Boolean mask of pixels near the coastline (mask ≈ 0.5)."""
        return (self.mask > 0.3) & (self.mask < 0.7)


def build_coastline_mask(
    face_id: str,
    context: TileBiomeContext,
    globe_grid,
    tile_size: int = 256,
    *,
    config: Optional[CoastlineConfig] = None,
    seed: int = 0,
) -> CoastlineMask:
    """Build a complete CoastlineMask for a tile.

    Convenience function that combines classification + mask computation.

    Parameters
    ----------
    face_id : str
    context : TileBiomeContext
        From :func:`classify_tile_biome_context`.
    globe_grid : PolyGrid / GlobeGrid
    tile_size : int
    config : CoastlineConfig, optional
    seed : int

    Returns
    -------
    CoastlineMask
    """
    if config is None:
        config = CoastlineConfig()

    if context.is_interior:
        # Interior tile — no transition needed
        mask = np.zeros((tile_size, tile_size), dtype=np.float32)
    else:
        mask = compute_coastline_mask(
            tile_size,
            context.edge_neighbours,
            globe_grid,
            face_id,
            config=config,
            seed=seed,
        )

    other_biomes = set(context.edge_neighbours.values())

    return CoastlineMask(
        mask=mask,
        face_id=face_id,
        own_biome=context.own_biome,
        other_biomes=other_biomes,
        config=config,
        seed=seed,
    )


# ═══════════════════════════════════════════════════════════════════
# 19B.1 — Blend biome images
# ═══════════════════════════════════════════════════════════════════

def blend_biome_images(
    img_own: "Image.Image",
    img_other: "Image.Image",
    mask: np.ndarray,
) -> "Image.Image":
    """Composite two biome images using a coastline mask.

    Parameters
    ----------
    img_own : PIL.Image
        The tile rendered as its own biome (e.g. forest).
    img_other : PIL.Image
        The tile rendered as the neighbour's biome (e.g. ocean).
    mask : np.ndarray
        ``(H, W)`` float32 in ``[0, 1]``.  0 = own, 1 = other.

    Returns
    -------
    PIL.Image (RGB)
    """
    from PIL import Image

    own_arr = np.array(img_own.convert("RGB"), dtype=np.float32)
    other_arr = np.array(img_other.convert("RGB"), dtype=np.float32)

    # Ensure mask matches image dimensions
    h, w = own_arr.shape[:2]
    m = mask[:h, :w, np.newaxis]  # broadcast to (H, W, 1)

    blended = own_arr * (1.0 - m) + other_arr * m
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended, "RGB")


# ═══════════════════════════════════════════════════════════════════
# 19B.3 — Coastal strip rendering
# ═══════════════════════════════════════════════════════════════════

def render_coastal_strip(
    image: "Image.Image",
    mask: np.ndarray,
    *,
    config: Optional[CoastlineConfig] = None,
    seed: int = 0,
) -> "Image.Image":
    """Paint beach/sand and foam detail in the coastline transition zone.

    Operates on the already-blended image to add coastal detail:
    - **Beach strip** on the land side (mask 0.25–0.50): sandy colour
    - **Foam strip** on the ocean side (mask 0.50–0.70): white foam/surf

    Parameters
    ----------
    image : PIL.Image (RGB)
        The blended biome image.
    mask : np.ndarray
        Coastline mask.
    config : CoastlineConfig, optional
    seed : int

    Returns
    -------
    PIL.Image (RGB)
    """
    from PIL import Image
    from .noise import fbm

    if config is None:
        config = CoastlineConfig()

    img = image.copy().convert("RGB")
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    m = mask[:h, :w]

    # Beach zone: mask ∈ [0.25, 0.50] → land side of coastline
    beach_lo, beach_hi = 0.25, 0.50
    # Foam zone: mask ∈ [0.50, 0.70] → ocean side of coastline
    foam_lo, foam_hi = 0.50, 0.70

    beach_r, beach_g, beach_b = config.beach_color
    foam_r, foam_g, foam_b = config.foam_color

    for py in range(h):
        for px in range(w):
            v = m[py, px]

            # Beach strip
            if beach_lo < v < beach_hi:
                # Strength: peaks at midpoint of beach zone
                mid = (beach_lo + beach_hi) / 2.0
                half_w = (beach_hi - beach_lo) / 2.0
                strength = 1.0 - abs(v - mid) / half_w
                strength = strength * strength  # quadratic falloff

                # Noise for variation
                noise_val = fbm(
                    px / max(w, 1) * 15.0,
                    py / max(h, 1) * 15.0,
                    octaves=2, frequency=1.0, seed=seed + 4444,
                )
                strength *= max(0.0, 0.6 + noise_val * 0.4)
                strength = min(1.0, strength * 0.6)  # subtle

                r, g, b = arr[py, px]
                arr[py, px, 0] = r + (beach_r - r) * strength
                arr[py, px, 1] = g + (beach_g - g) * strength
                arr[py, px, 2] = b + (beach_b - b) * strength

            # Foam strip
            elif foam_lo < v < foam_hi:
                mid = (foam_lo + foam_hi) / 2.0
                half_w = (foam_hi - foam_lo) / 2.0
                strength = 1.0 - abs(v - mid) / half_w
                strength = strength * strength

                noise_val = fbm(
                    px / max(w, 1) * 20.0,
                    py / max(h, 1) * 20.0,
                    octaves=3, frequency=1.0, seed=seed + 5555,
                )
                strength *= max(0.0, 0.4 + noise_val * 0.6)
                strength = min(1.0, strength * 0.5)

                r, g, b = arr[py, px]
                arr[py, px, 0] = r + (foam_r - r) * strength
                arr[py, px, 1] = g + (foam_g - g) * strength
                arr[py, px, 2] = b + (foam_b - b) * strength

    result = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(result, "RGB")
