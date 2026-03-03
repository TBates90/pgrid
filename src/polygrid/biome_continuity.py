"""Cross-tile biome continuity — seamless feature rendering across boundaries.

Ensures that biome features (forests, etc.) flow seamlessly across
Goldberg tile boundaries.  A forest that spans multiple tiles should
look like one continuous canopy, not separate patches.

Functions
---------
- :func:`build_biome_density_map` — globe-wide density map with neighbour falloff
- :func:`get_tile_margin_features` — collect margin features from neighbours
- :func:`compute_biome_transition_mask` — 2-D gradient mask for biome edges
- :func:`stitch_feature_boundary` — feather-blend boundary pixel strips
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from .biome_scatter import FeatureInstance, compute_density_field


# ═══════════════════════════════════════════════════════════════════
# 14C.1 — Globe-wide biome density map with neighbour transition
# ═══════════════════════════════════════════════════════════════════

def build_biome_density_map(
    globe_grid,
    face_ids: Sequence[str],
    *,
    biome_faces: Optional[set] = None,
    seed: int = 42,
    noise_frequency: float = 3.0,
    noise_octaves: int = 4,
    base_density: float = 0.85,
    edge_falloff: float = 0.35,
    neighbour_transition: float = 0.35,
) -> Dict[str, float]:
    """Globe-wide density map with smooth neighbour transitions.

    Extends :func:`compute_density_field` by adding a *transition zone*:
    tiles that are **not** biome members but are **adjacent** to a biome
    tile receive a reduced density (``neighbour_transition``), producing
    a gradual thinning at forest edges instead of a hard cutoff.

    Parameters
    ----------
    globe_grid : PolyGrid / GlobeGrid
        Globe topology with adjacency.
    face_ids : sequence of str
        All face IDs to evaluate.
    biome_faces : set of str, optional
        Faces classified as this biome.  If *None* all faces are biome.
    seed, noise_frequency, noise_octaves, base_density, edge_falloff
        Forwarded to :func:`compute_density_field`.
    neighbour_transition : float
        Density assigned to non-biome tiles adjacent to a biome tile.
        Represents the transition fringe.

    Returns
    -------
    dict
        ``{face_id: density}`` with values in ``[0.0, 1.0]``.
    """
    from .algorithms import get_face_adjacency

    # Core density from noise field
    core = compute_density_field(
        globe_grid, face_ids,
        biome_faces=biome_faces,
        seed=seed,
        noise_frequency=noise_frequency,
        noise_octaves=noise_octaves,
        base_density=base_density,
        edge_falloff=edge_falloff,
    )

    if biome_faces is None:
        # All faces are biome — no transition needed
        return core

    # Build adjacency for transition zone
    adj = get_face_adjacency(globe_grid)

    density = dict(core)
    for fid in face_ids:
        if density.get(fid, 0.0) > 0.01:
            continue  # already a biome tile
        # Check if any neighbour is a biome tile
        neighbours = adj.get(fid, [])
        for nid in neighbours:
            if nid in biome_faces:
                # Transition tile — assign reduced density
                density[fid] = max(
                    density.get(fid, 0.0),
                    neighbour_transition,
                )
                break  # one biome neighbour is enough

    return density


# ═══════════════════════════════════════════════════════════════════
# 14C.2 — Collect margin features from neighbouring tiles
# ═══════════════════════════════════════════════════════════════════

def get_tile_margin_features(
    tile_id: str,
    own_scatter: List[FeatureInstance],
    neighbour_scatters: Dict[str, List[FeatureInstance]],
    tile_size: int = 256,
    margin: float = 8.0,
) -> List[FeatureInstance]:
    """Collect features from neighbours that overlap into this tile.

    For each neighbour, features near the *shared boundary* are
    translated into this tile's coordinate space (offset by one
    tile width) and included so that canopies crossing the boundary
    are drawn in both tiles.

    In practice, this is a simplified model: we assume neighbours
    are laid out in a local 2-D arrangement and approximate the
    offset as ``±tile_size`` in x or y.  For globe rendering the
    atlas gutter already handles most seam issues, so the margin
    features are a refinement for the highest-quality renders.

    Parameters
    ----------
    tile_id : str
        Current tile's face ID (for identification only).
    own_scatter : list of FeatureInstance
        This tile's own scattered features (not modified).
    neighbour_scatters : dict
        ``{neighbour_face_id: [FeatureInstance, ...]}`` — scatter
        results for each adjacent tile.
    tile_size : int
        Tile image size in pixels.
    margin : float
        Pixel margin from edge to consider as overlap zone.

    Returns
    -------
    list of FeatureInstance
        Margin features from neighbours, translated into this tile's
        coordinate space.  Only features within ``margin`` of a boundary
        are included.
    """
    collected: List[FeatureInstance] = []

    for nid, n_instances in neighbour_scatters.items():
        for inst in n_instances:
            # Check if this feature is near *any* edge of the neighbour tile
            # Near right edge → could overlap into our left side
            # Near bottom edge → could overlap into our top side, etc.
            near_right = inst.px + inst.radius > tile_size - margin
            near_left = inst.px - inst.radius < margin
            near_bottom = inst.py + inst.radius > tile_size - margin
            near_top = inst.py - inst.radius < margin

            if not (near_right or near_left or near_bottom or near_top):
                continue

            # Translate into our coordinate space
            # This is a simplified model — for the general case we'd need
            # the actual spatial relationship between tiles.  Here we
            # mirror features across the boundary they're near.
            if near_right:
                new_px = inst.px - tile_size
            elif near_left:
                new_px = inst.px + tile_size
            else:
                new_px = inst.px

            if near_bottom:
                new_py = inst.py - tile_size
            elif near_top:
                new_py = inst.py + tile_size
            else:
                new_py = inst.py

            # Only include if the translated position is within
            # (or just beyond) this tile's bounds
            if (-inst.radius <= new_px <= tile_size + inst.radius and
                    -inst.radius <= new_py <= tile_size + inst.radius):
                collected.append(FeatureInstance(
                    px=new_px,
                    py=new_py,
                    radius=inst.radius,
                    color=inst.color,
                    shadow_color=inst.shadow_color,
                    species_id=inst.species_id,
                    depth=new_py,
                ))

    return collected


# ═══════════════════════════════════════════════════════════════════
# 14C.3 — Biome transition mask
# ═══════════════════════════════════════════════════════════════════

def compute_biome_transition_mask(
    tile_density: float,
    neighbour_densities: Dict[str, float],
    tile_size: int = 256,
    *,
    feather_width: float = 0.15,
) -> List[List[float]]:
    """Compute a 2-D transition mask for biome rendering.

    Returns a ``tile_size × tile_size`` float grid where 1.0 = full
    biome and 0.0 = no biome.  At tile edges adjacent to lower-density
    neighbours, a smooth gradient is applied over *feather_width*
    fraction of the tile.

    Parameters
    ----------
    tile_density : float
        This tile's biome density.
    neighbour_densities : dict
        ``{direction: density}`` where direction is one of
        ``"left", "right", "top", "bottom"`` mapping to adjacent
        tile densities.  Missing directions are treated as
        having the same density as this tile.
    tile_size : int
        Output mask size (square).
    feather_width : float
        Fraction of tile_size over which the gradient is applied
        at edges.  E.g. 0.15 = 15% of the tile width.

    Returns
    -------
    list of list of float
        ``mask[y][x]`` — values in ``[0.0, 1.0]``.
    """
    if tile_density <= 0.001:
        return [[0.0] * tile_size for _ in range(tile_size)]

    fw = max(1, int(feather_width * tile_size))

    # Per-direction edge density ratio
    d_left = neighbour_densities.get("left", tile_density)
    d_right = neighbour_densities.get("right", tile_density)
    d_top = neighbour_densities.get("top", tile_density)
    d_bottom = neighbour_densities.get("bottom", tile_density)

    mask = [[tile_density] * tile_size for _ in range(tile_size)]

    for y in range(tile_size):
        for x in range(tile_size):
            factor = 1.0

            # Left edge feather
            if x < fw and d_left < tile_density:
                t = x / fw  # 0 at edge → 1 at interior
                edge_val = d_left + t * (tile_density - d_left)
                factor = min(factor, edge_val / max(tile_density, 0.001))

            # Right edge feather
            if x >= tile_size - fw and d_right < tile_density:
                t = (tile_size - 1 - x) / fw
                edge_val = d_right + t * (tile_density - d_right)
                factor = min(factor, edge_val / max(tile_density, 0.001))

            # Top edge feather
            if y < fw and d_top < tile_density:
                t = y / fw
                edge_val = d_top + t * (tile_density - d_top)
                factor = min(factor, edge_val / max(tile_density, 0.001))

            # Bottom edge feather
            if y >= tile_size - fw and d_bottom < tile_density:
                t = (tile_size - 1 - y) / fw
                edge_val = d_bottom + t * (tile_density - d_bottom)
                factor = min(factor, edge_val / max(tile_density, 0.001))

            mask[y][x] = tile_density * factor

    return mask


# ═══════════════════════════════════════════════════════════════════
# 14C.4 — Boundary pixel-strip feather blending
# ═══════════════════════════════════════════════════════════════════

def stitch_feature_boundary(
    img_a: "Image.Image",
    img_b: "Image.Image",
    edge: str = "right",
    feather_pixels: int = 4,
) -> Tuple["Image.Image", "Image.Image"]:
    """Feather-blend the boundary strip between two adjacent tile images.

    Compares the pixel strips along the shared edge and applies a narrow
    linear blend to reduce visible colour discontinuities.

    Parameters
    ----------
    img_a : PIL.Image
        First tile image (left or top tile).
    img_b : PIL.Image
        Second tile image (right or bottom tile).
    edge : str
        ``"right"`` — *img_a*'s right edge meets *img_b*'s left edge.
        ``"bottom"`` — *img_a*'s bottom edge meets *img_b*'s top edge.
    feather_pixels : int
        Width of the blend zone on each side of the boundary.

    Returns
    -------
    (img_a, img_b) — modified copies with blended boundary strips.
    """
    if not _HAS_PIL:
        raise RuntimeError("PIL/Pillow required for stitch_feature_boundary")

    a = img_a.copy().convert("RGB")
    b = img_b.copy().convert("RGB")
    w, h = a.size
    fp = feather_pixels

    if edge == "right":
        # Blend the right strip of a with the left strip of b
        for dy in range(h):
            for dx in range(fp):
                # Position in a: (w - fp + dx, dy)
                # Position in b: (dx, dy)
                ax = w - fp + dx
                t = (dx + 0.5) / fp  # 0 at a's interior → 1 at boundary

                pa = a.getpixel((ax, dy))
                pb = b.getpixel((dx, dy))

                blended = tuple(
                    int(pa[c] * (1 - t) + pb[c] * t + 0.5)
                    for c in range(3)
                )
                a.putpixel((ax, dy), blended)
                b.putpixel((dx, dy), tuple(
                    int(pb[c] * (1 - t) + pa[c] * t + 0.5)
                    for c in range(3)
                ))

    elif edge == "bottom":
        for dx in range(w):
            for dy in range(fp):
                ay = h - fp + dy
                t = (dy + 0.5) / fp

                pa = a.getpixel((dx, ay))
                pb = b.getpixel((dx, dy))

                blended = tuple(
                    int(pa[c] * (1 - t) + pb[c] * t + 0.5)
                    for c in range(3)
                )
                a.putpixel((dx, ay), blended)
                b.putpixel((dx, dy), tuple(
                    int(pb[c] * (1 - t) + pa[c] * t + 0.5)
                    for c in range(3)
                ))

    return a, b
