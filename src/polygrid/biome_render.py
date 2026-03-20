# TODO REMOVE — Not used by any live script. Phase 14B forest rendering.
"""Biome feature rendering — pixel-level forest canopy, shadows, undergrowth.

Renders visual features onto PIL tile textures to produce a satellite-
style forest-from-above look.  Operates at the pixel level on 256×256
(or larger) images — much higher fidelity than per-sub-face colouring.

Functions
---------
- :func:`render_canopy` — draw one tree canopy onto an image
- :func:`render_undergrowth` — fill gaps with undergrowth texture
- :func:`render_forest_tile` — composit all features for one tile

Classes
-------
- :class:`ForestFeatureConfig` — tuneable parameters for forest rendering
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageDraw, ImageFilter
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from .biome_scatter import FeatureInstance


# ═══════════════════════════════════════════════════════════════════
# 14B.4 — ForestFeatureConfig
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ForestFeatureConfig:
    """All tuneable parameters for forest feature rendering.

    Attributes
    ----------
    canopy_radius_range : tuple
        ``(min_px, max_px)`` for canopy radius.
    canopy_colors : list of (r, g, b)
        Species palette for canopy tops.
    color_noise_amplitude : float
        Per-tree colour jitter (max channel offset).
    density_scale : float
        Global density multiplier (0–1).
    shadow_offset : tuple
        ``(dx, dy)`` pixel offset for drop shadows.
    shadow_opacity : float
        Shadow alpha (0–1).
    shadow_color : tuple
        ``(r, g, b)`` drop-shadow colour.
    highlight_strength : float
        Specular highlight intensity (0–1).
    highlight_offset : tuple
        ``(dx, dy)`` pixel offset for highlight (toward sun).
    undergrowth_color : tuple
        ``(r, g, b)`` base ground colour between trees.
    undergrowth_noise : float
        Colour variation amplitude for undergrowth.
    edge_thinning : float
        How aggressively density drops at biome edges (0–1).
    """

    canopy_radius_range: Tuple[float, float] = (2.5, 7.0)
    canopy_colors: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (34, 120, 30),
        (50, 135, 40),
        (65, 145, 35),
        (28, 100, 28),
        (45, 128, 50),
    ])
    color_noise_amplitude: float = 15.0
    density_scale: float = 0.85
    shadow_offset: Tuple[int, int] = (2, 3)
    shadow_opacity: float = 0.45
    shadow_color: Tuple[int, int, int] = (15, 35, 10)
    highlight_strength: float = 0.25
    highlight_offset: Tuple[int, int] = (-1, -2)
    undergrowth_color: Tuple[int, int, int] = (25, 55, 18)
    undergrowth_noise: float = 20.0
    edge_thinning: float = 0.35


# ── Presets ─────────────────────────────────────────────────────────

TEMPERATE_FOREST = ForestFeatureConfig()

TROPICAL_FOREST = ForestFeatureConfig(
    canopy_radius_range=(3.0, 9.0),
    canopy_colors=[
        (20, 105, 25),
        (30, 125, 20),
        (45, 140, 30),
        (15, 90, 20),
        (55, 130, 35),
        (25, 115, 40),
    ],
    density_scale=0.95,
    shadow_opacity=0.55,
    undergrowth_color=(18, 45, 12),
)

BOREAL_FOREST = ForestFeatureConfig(
    canopy_radius_range=(1.5, 5.0),
    canopy_colors=[
        (25, 80, 35),
        (35, 90, 40),
        (20, 70, 30),
        (30, 85, 45),
    ],
    density_scale=0.65,
    shadow_opacity=0.35,
    undergrowth_color=(45, 55, 30),
    undergrowth_noise=15.0,
)

SPARSE_WOODLAND = ForestFeatureConfig(
    canopy_radius_range=(3.0, 8.0),
    canopy_colors=[
        (55, 140, 45),
        (70, 155, 50),
        (50, 130, 35),
    ],
    density_scale=0.35,
    shadow_opacity=0.30,
    undergrowth_color=(80, 100, 50),
    undergrowth_noise=25.0,
)

FOREST_PRESETS: Dict[str, ForestFeatureConfig] = {
    "temperate": TEMPERATE_FOREST,
    "tropical": TROPICAL_FOREST,
    "boreal": BOREAL_FOREST,
    "sparse_woodland": SPARSE_WOODLAND,
}


# ═══════════════════════════════════════════════════════════════════
# 14B.1 — Render a single canopy
# ═══════════════════════════════════════════════════════════════════

def render_canopy(
    draw: "ImageDraw.ImageDraw",
    instance: FeatureInstance,
    config: ForestFeatureConfig,
    *,
    rng: Optional[random.Random] = None,
) -> None:
    """Draw a single tree canopy (shadow + crown + highlight) onto *draw*.

    Operates on an RGBA ``ImageDraw`` object.  The caller is
    responsible for creating and compositing the final image.

    Parameters
    ----------
    draw : ImageDraw.ImageDraw
        Draw context for an RGBA image.
    instance : FeatureInstance
        Position, radius, colour of this tree.
    config : ForestFeatureConfig
        Rendering parameters.
    rng : Random, optional
        For reproducible internal noise.  If *None*, a default is used.
    """
    if rng is None:
        rng = random.Random(int(instance.px * 1000 + instance.py))

    px, py = instance.px, instance.py
    r = instance.radius
    cr, cg, cb = instance.color

    # ── Shadow ──────────────────────────────────────────────────
    sdx, sdy = config.shadow_offset
    sx, sy = px + sdx, py + sdy
    sr, sg, sb = config.shadow_color
    shadow_alpha = int(config.shadow_opacity * 255)
    # Slightly larger than canopy for soft edge
    sr_size = r * 1.1
    draw.ellipse(
        [sx - sr_size, sy - sr_size, sx + sr_size, sy + sr_size],
        fill=(sr, sg, sb, shadow_alpha),
    )

    # ── Canopy base ─────────────────────────────────────────────
    # Slight irregularity: ±10% radius jitter
    jitter = 1.0 + rng.uniform(-0.1, 0.1)
    rx = r * jitter
    ry = r * (2.0 - jitter)  # inverse so ellipse stays roughly same area
    draw.ellipse(
        [px - rx, py - ry, px + rx, py + ry],
        fill=(cr, cg, cb, 255),
    )

    # ── Internal texture: darker inner ring for depth ───────────
    inner_r = r * 0.55
    shade = max(0, int(-15 + rng.randint(-5, 5)))
    inner_color = (
        max(0, cr + shade),
        max(0, cg + shade),
        max(0, cb + shade),
        200,
    )
    draw.ellipse(
        [px - inner_r, py - inner_r, px + inner_r, py + inner_r],
        fill=inner_color,
    )

    # ── Highlight (specular from waxy leaves) ───────────────────
    if config.highlight_strength > 0.05:
        hx = px + config.highlight_offset[0]
        hy = py + config.highlight_offset[1]
        hr = r * 0.3
        bright = int(config.highlight_strength * 80)
        highlight_color = (
            min(255, cr + bright),
            min(255, cg + bright),
            min(255, cb + bright),
            int(config.highlight_strength * 180),
        )
        draw.ellipse(
            [hx - hr, hy - hr, hx + hr, hy + hr],
            fill=highlight_color,
        )


# ═══════════════════════════════════════════════════════════════════
# 14B.3 — Render undergrowth
# ═══════════════════════════════════════════════════════════════════

def render_undergrowth(
    image: "Image.Image",
    density: float,
    config: ForestFeatureConfig,
    *,
    seed: int = 42,
) -> None:
    """Fill the image with undergrowth noise texture.

    Modifies *image* in place.  Best called before canopies so
    the ground is visible through gaps.

    Parameters
    ----------
    image : PIL.Image (RGBA)
        The image to paint onto.
    density : float
        Biome density (0–1).  Higher = darker, less ground visible.
    config : ForestFeatureConfig
        Rendering parameters.
    seed : int
        Noise seed.
    """
    rng = random.Random(seed)
    w, h = image.size
    draw = ImageDraw.Draw(image)

    ur, ug, ub = config.undergrowth_color
    noise_amp = config.undergrowth_noise

    # Paint small noise dots to create dappled ground
    step = max(2, int(4 - density * 2))  # denser = finer step
    for y in range(0, h, step):
        for x in range(0, w, step):
            dr = rng.randint(-int(noise_amp), int(noise_amp))
            dg = rng.randint(-int(noise_amp), int(noise_amp))
            db = rng.randint(-int(noise_amp), int(noise_amp))
            pixel_color = (
                max(0, min(255, ur + dr)),
                max(0, min(255, ug + dg)),
                max(0, min(255, ub + db)),
                int(density * 200),
            )
            # Small rectangle
            draw.rectangle(
                [x, y, x + step - 1, y + step - 1],
                fill=pixel_color,
            )

    # Scatter small shrub dots in gaps
    n_shrubs = int(density * w * h / 400)
    for _ in range(n_shrubs):
        sx = rng.randint(0, w - 1)
        sy = rng.randint(0, h - 1)
        sr = rng.uniform(0.5, 1.5)
        shade = rng.randint(-10, 10)
        shrub_color = (
            max(0, min(255, ur + 15 + shade)),
            max(0, min(255, ug + 20 + shade)),
            max(0, min(255, ub + 5 + shade)),
            int(density * 230),
        )
        draw.ellipse(
            [sx - sr, sy - sr, sx + sr, sy + sr],
            fill=shrub_color,
        )


# ═══════════════════════════════════════════════════════════════════
# 14B.2 — Render all forest features for one tile
# ═══════════════════════════════════════════════════════════════════

def render_forest_tile(
    ground_image: "Image.Image",
    instances: List[FeatureInstance],
    config: Optional[ForestFeatureConfig] = None,
    *,
    density: float = 0.8,
    seed: int = 42,
) -> "Image.Image":
    """Render all forest features for one tile on top of a ground texture.

    Parameters
    ----------
    ground_image : PIL.Image
        The existing ground/elevation texture (RGB or RGBA).
    instances : list of FeatureInstance
        Feature instances to render (already sorted by depth).
    config : ForestFeatureConfig, optional
        Uses TEMPERATE_FOREST defaults if not given.
    density : float
        Biome density (0–1) for undergrowth opacity.
    seed : int
        Random seed.

    Returns
    -------
    PIL.Image
        The composited image (RGBA).
    """
    if config is None:
        config = TEMPERATE_FOREST

    # Convert ground to RGBA for compositing
    base = ground_image.convert("RGBA")
    w, h = base.size

    # Layer 1: undergrowth
    undergrowth_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    render_undergrowth(undergrowth_layer, density, config, seed=seed)
    base = Image.alpha_composite(base, undergrowth_layer)

    # Layer 2: shadows + canopies (drawn onto a single overlay)
    canopy_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canopy_layer)

    # Sort by depth (back-to-front) — should already be sorted, but ensure
    sorted_instances = sorted(instances, key=lambda f: f.depth)

    rng = random.Random(seed + 1)
    for inst in sorted_instances:
        render_canopy(draw, inst, config, rng=rng)

    # Optional: slight blur for soft edges
    if len(sorted_instances) > 0:
        try:
            canopy_layer = canopy_layer.filter(
                ImageFilter.GaussianBlur(radius=0.5),
            )
        except Exception:
            pass  # Blur not critical

    base = Image.alpha_composite(base, canopy_layer)

    return base


# ═══════════════════════════════════════════════════════════════════
# Convenience: full forest render from density
# ═══════════════════════════════════════════════════════════════════

def render_forest_on_ground(
    ground_image: "Image.Image",
    tile_density: float,
    *,
    config: Optional[ForestFeatureConfig] = None,
    tile_size: int = 256,
    seed: int = 42,
    globe_3d_center: Optional[Tuple[float, float, float]] = None,
) -> "Image.Image":
    """End-to-end: scatter + render forest on a ground texture.

    Convenience function that combines :func:`scatter_features_on_tile`
    and :func:`render_forest_tile`.

    Parameters
    ----------
    ground_image : PIL.Image
        Ground/elevation texture.
    tile_density : float
        Biome density (0–1).
    config : ForestFeatureConfig, optional
    tile_size : int
    seed : int
    globe_3d_center : (x, y, z), optional

    Returns
    -------
    PIL.Image (RGBA)
    """
    from .biome_scatter import scatter_features_on_tile

    if config is None:
        config = TEMPERATE_FOREST

    instances = scatter_features_on_tile(
        tile_density,
        tile_size=tile_size,
        min_radius=config.canopy_radius_range[0],
        max_radius=config.canopy_radius_range[1],
        colors=config.canopy_colors,
        shadow_color=config.shadow_color,
        color_noise=config.color_noise_amplitude,
        seed=seed,
        globe_3d_center=globe_3d_center,
    )

    return render_forest_tile(
        ground_image, instances, config,
        density=tile_density, seed=seed,
    )


# ═══════════════════════════════════════════════════════════════════
# 16C.3 — Feature-level cross-fade using blend mask
# ═══════════════════════════════════════════════════════════════════

def render_forest_on_ground_fullslot(
    ground_image: "Image.Image",
    tile_density: float,
    *,
    config: Optional[ForestFeatureConfig] = None,
    tile_size: int = 256,
    seed: int = 42,
    globe_3d_center: Optional[Tuple[float, float, float]] = None,
    overscan: float = 0.15,
    blend_mask: Optional["np.ndarray"] = None,
    neighbour_densities: Optional[Dict[str, float]] = None,
    neighbour_seeds: Optional[Dict[str, int]] = None,
) -> "Image.Image":
    """Full-slot forest rendering with feature-level cross-fade.

    Uses :func:`scatter_features_fullslot` to place trees across the
    full square tile area (including margins), then optionally applies
    a per-pixel blend mask so that features near tile edges fade out
    gracefully.

    Parameters
    ----------
    ground_image : PIL.Image
        Ground/elevation texture (RGB).
    tile_density : float
        Biome density (0–1).
    config : ForestFeatureConfig, optional
    tile_size : int
    seed : int
    globe_3d_center : (x, y, z), optional
    overscan : float
        How far features extend beyond the hex tile boundary.
    blend_mask : np.ndarray, optional
        ``(tile_size, tile_size)`` float32 in ``[0, 1]`` from
        :func:`compute_tile_blend_mask`.  Features are alpha-scaled
        by this mask so canopy fades toward tile edges.  If *None*,
        no fade is applied.
    neighbour_densities : dict, optional
        ``{direction: density}`` for margin zone feature density.
    neighbour_seeds : dict, optional
        ``{direction: seed}`` for margin zone feature seeds.

    Returns
    -------
    PIL.Image (RGBA)
    """
    import numpy as np
    from .biome_scatter import scatter_features_fullslot

    if config is None:
        config = TEMPERATE_FOREST

    instances = scatter_features_fullslot(
        tile_density,
        tile_size=tile_size,
        overscan=overscan,
        min_radius=config.canopy_radius_range[0],
        max_radius=config.canopy_radius_range[1],
        colors=config.canopy_colors,
        shadow_color=config.shadow_color,
        color_noise=config.color_noise_amplitude,
        seed=seed,
        globe_3d_center=globe_3d_center,
        neighbour_densities=neighbour_densities,
        neighbour_seeds=neighbour_seeds,
    )

    # Render forest on the ground image
    result = render_forest_tile(
        ground_image, instances, config,
        density=tile_density, seed=seed,
    )

    # 16C.3 — Apply blend mask as alpha fade on the feature layer
    if blend_mask is not None:
        # Composite: blend between the original ground (no features)
        # and the featured image, using the mask as the mixing factor.
        # mask=1 → full features, mask=0 → ground only.
        ground_rgba = ground_image.convert("RGBA")
        result_rgba = result.convert("RGBA")

        ground_arr = np.array(ground_rgba, dtype=np.float64)
        result_arr = np.array(result_rgba, dtype=np.float64)

        # Expand mask to (H, W, 1) for broadcasting
        h, w = ground_arr.shape[:2]
        mask_resized = blend_mask[:h, :w]  # safety clamp
        mask_4d = mask_resized[:, :, np.newaxis]  # (H, W, 1)

        # Blend: output = ground * (1 - mask) + featured * mask
        blended = ground_arr * (1.0 - mask_4d) + result_arr * mask_4d
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        result = Image.fromarray(blended, "RGBA")

    return result
