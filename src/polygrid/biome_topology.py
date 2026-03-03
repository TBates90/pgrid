"""Topology-aware biome feature rendering — Phase 18C.

Places biome features (forest trees, ocean depth/coastal detail) at
sub-face positions in the detail grid rather than random pixel
positions.  This produces features that follow the grid topology and
integrate naturally with the terrain structure.

The module provides two new renderer classes that implement the
:class:`BiomeRenderer` protocol:

- :class:`TopologyForestRenderer` — trees placed at sub-face centroids
- :class:`TopologyOceanRenderer` — per-sub-face depth gradients + coastal features

And a hybrid rendering pipeline that layers:

1. Topology pass (sub-face polygon colouring)
2. Pixel noise overlay (existing 16D micro-detail)
3. Feature compositing (trees, waves, foam at polygon positions)

Functions
---------
- :func:`classify_subface_biome` — classify sub-faces for biome features
- :func:`scatter_trees_on_grid` — place trees at sub-face centroids
- :func:`render_topology_forest` — full forest render using grid topology
- :func:`compute_subface_ocean_depth` — per-sub-face depth from elevation
- :func:`identify_coastal_subfaces` — find sub-faces adjacent to land
- :func:`render_topology_ocean` — full ocean render using grid topology
- :func:`render_hybrid_biome` — combined topology + pixel-level rendering

Classes
-------
- :class:`TopologyForestRenderer` — BiomeRenderer for topology-aware forest
- :class:`TopologyOceanRenderer` — BiomeRenderer for topology-aware ocean
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

try:
    from PIL import Image, ImageDraw, ImageFilter
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ═══════════════════════════════════════════════════════════════════
# 18C.1 — Sub-face forest features
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SubfaceTree:
    """A tree placed at a sub-face centroid.

    Attributes
    ----------
    face_id : str
        Detail grid sub-face ID hosting this tree.
    px, py : float
        Pixel coordinates (tile-local) of the tree centre.
    radius : float
        Canopy radius in pixels (proportional to sub-face area).
    color : tuple of int
        ``(r, g, b)`` canopy colour.
    elevation : float
        Sub-face elevation (drives species variation).
    area : float
        Sub-face polygon area in pixel² (for size scaling).
    depth : float
        Drawing order (y-coordinate for painter's algorithm).
    """
    face_id: str
    px: float
    py: float
    radius: float
    color: Tuple[int, int, int] = (40, 120, 35)
    elevation: float = 0.0
    area: float = 100.0
    depth: float = 0.0


def _polygon_area_2d(vertices: List[Tuple[float, float]]) -> float:
    """Compute area of a 2D polygon using the shoelace formula."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


def _sub_face_pixel_centroid_and_area(
    grid,
    face,
    vertices,
    to_pixel,
) -> Tuple[float, float, float, List[Tuple[float, float]]]:
    """Compute centroid (px, py), area, and pixel vertices for a face."""
    pixel_verts = []
    for vid in face.vertex_ids:
        v = vertices.get(vid)
        if v is None or not v.has_position():
            return (0.0, 0.0, 0.0, [])
        pixel_verts.append(to_pixel(v.x, v.y))

    if len(pixel_verts) < 3:
        return (0.0, 0.0, 0.0, [])

    cx = sum(p[0] for p in pixel_verts) / len(pixel_verts)
    cy = sum(p[1] for p in pixel_verts) / len(pixel_verts)
    area = _polygon_area_2d(pixel_verts)
    return (cx, cy, area, pixel_verts)


def scatter_trees_on_grid(
    detail_grid,
    store,
    *,
    tile_size: int = 256,
    density: float = 0.8,
    canopy_colors: Optional[List[Tuple[int, int, int]]] = None,
    color_noise: float = 15.0,
    min_radius: float = 2.5,
    max_radius: float = 7.0,
    seed: int = 42,
    elevation_field: str = "elevation",
    elevation_range: Optional[Tuple[float, float]] = None,
    alpine_threshold: float = 0.7,
    face_ids: Optional[Set[str]] = None,
) -> List[SubfaceTree]:
    """Place trees at sub-face centroids of a detail grid.

    Each sub-face that passes the density check gets a tree at its
    centroid.  Tree radius is proportional to sub-face area.
    Elevation drives species colour and thinning at high altitudes.

    Parameters
    ----------
    detail_grid : PolyGrid
        The detail grid (or apron grid) to scatter on.
    store : TileDataStore
        Must contain elevation for each sub-face.
    tile_size : int
        Output image size for coordinate mapping.
    density : float
        Global density (0–1).  Higher = more trees.
    canopy_colors : list of (r, g, b), optional
        Species palette.
    color_noise : float
        Per-tree colour jitter amplitude.
    min_radius, max_radius : float
        Canopy radius range in pixels.
    seed : int
    elevation_field : str
    elevation_range : (min_elev, max_elev), optional
        If given, normalises elevation for alpine thinning.
    alpine_threshold : float
        Normalised elevation above which trees thin out.
    face_ids : set, optional
        If given, only scatter on these sub-faces.

    Returns
    -------
    list of SubfaceTree
    """
    from .geometry import face_center

    if canopy_colors is None:
        canopy_colors = [
            (34, 120, 30),
            (50, 135, 40),
            (65, 145, 35),
            (28, 100, 28),
            (45, 128, 50),
        ]

    rng = random.Random(seed)

    # Compute bounding box for pixel mapping
    xs, ys = [], []
    for v in detail_grid.vertices.values():
        if v.has_position():
            xs.append(v.x)
            ys.append(v.y)

    if not xs:
        return []

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0
    span = max(x_range, y_range)
    pad = span * 0.15  # overscan
    x_min -= pad
    x_max += pad
    y_min -= pad
    y_max += pad
    x_range = x_max - x_min
    y_range = y_max - y_min
    scale = tile_size / max(x_range, y_range)
    ox = (tile_size - x_range * scale) / 2.0
    oy = (tile_size - y_range * scale) / 2.0

    def to_pixel(vx: float, vy: float) -> Tuple[float, float]:
        px = (vx - x_min) * scale + ox
        py = tile_size - ((vy - y_min) * scale + oy)
        return (px, py)

    # Compute elevation range from store if not provided
    if elevation_range is None:
        elevations = []
        for fid in detail_grid.faces:
            e = store.get(fid, elevation_field)
            if e is not None:
                elevations.append(e)
        if elevations:
            elevation_range = (min(elevations), max(elevations))
        else:
            elevation_range = (0.0, 1.0)

    elev_span = elevation_range[1] - elevation_range[0]
    if elev_span <= 0:
        elev_span = 1.0

    # Reference area for radius scaling — median sub-face area
    areas = []
    target_faces = face_ids if face_ids else set(detail_grid.faces.keys())

    for fid in target_faces:
        face = detail_grid.faces.get(fid)
        if face is None:
            continue
        _, _, area, _ = _sub_face_pixel_centroid_and_area(
            detail_grid, face, detail_grid.vertices, to_pixel,
        )
        if area > 0:
            areas.append(area)

    if not areas:
        return []

    areas.sort()
    median_area = areas[len(areas) // 2]
    # Reference radius for median-area sub-face
    ref_radius = (min_radius + max_radius) / 2.0

    trees: List[SubfaceTree] = []

    for fid in target_faces:
        face = detail_grid.faces.get(fid)
        if face is None:
            continue

        cx, cy, area, pixel_verts = _sub_face_pixel_centroid_and_area(
            detail_grid, face, detail_grid.vertices, to_pixel,
        )
        if area <= 0:
            continue

        elev = store.get(fid, elevation_field)
        if elev is None:
            elev = 0.0

        # Normalised elevation (0 = lowest, 1 = highest)
        norm_elev = (elev - elevation_range[0]) / elev_span

        # Alpine thinning — reduce density at high elevations
        if norm_elev > alpine_threshold:
            thin_factor = 1.0 - (norm_elev - alpine_threshold) / (1.0 - alpine_threshold)
            thin_factor = max(0.0, thin_factor)
        else:
            thin_factor = 1.0

        # Density check — use face ID hash for deterministic per-face decision
        face_hash = (hash(fid) + seed) % 10000 / 10000.0
        effective_density = density * thin_factor
        if face_hash > effective_density:
            continue

        # Radius proportional to sqrt(area) relative to median
        area_ratio = math.sqrt(area / max(median_area, 1.0))
        base_radius = ref_radius * area_ratio
        # Add some randomness
        radius = base_radius * rng.uniform(0.8, 1.2)
        radius = max(min_radius, min(max_radius, radius))

        # Colour — select from palette, modulate by elevation
        species_idx = (hash(fid) + seed) % len(canopy_colors)
        cr, cg, cb = canopy_colors[species_idx]

        # Elevation-based colour shift: higher = darker/browner
        elev_shift = norm_elev * 20.0
        cr = max(0, min(255, int(cr - elev_shift + rng.uniform(-color_noise, color_noise))))
        cg = max(0, min(255, int(cg - elev_shift * 0.5 + rng.uniform(-color_noise, color_noise))))
        cb = max(0, min(255, int(cb - elev_shift * 0.3 + rng.uniform(-color_noise, color_noise))))

        # Small jitter from exact centroid for organic look
        jx = rng.uniform(-1.5, 1.5)
        jy = rng.uniform(-1.5, 1.5)

        trees.append(SubfaceTree(
            face_id=fid,
            px=cx + jx,
            py=cy + jy,
            radius=radius,
            color=(cr, cg, cb),
            elevation=elev,
            area=area,
            depth=cy + jy,  # painter's algorithm: higher y = further back
        ))

    # Sort by depth (back to front)
    trees.sort(key=lambda t: t.depth)

    return trees


def render_topology_forest(
    ground_image: "Image.Image",
    trees: List[SubfaceTree],
    *,
    density: float = 0.8,
    shadow_offset: Tuple[int, int] = (2, 3),
    shadow_opacity: float = 0.45,
    shadow_color: Tuple[int, int, int] = (15, 35, 10),
    highlight_strength: float = 0.25,
    highlight_offset: Tuple[int, int] = (-1, -2),
    undergrowth_color: Tuple[int, int, int] = (25, 55, 18),
    undergrowth_noise_amp: float = 20.0,
    seed: int = 42,
) -> "Image.Image":
    """Render forest features from topology-placed trees onto a ground image.

    Uses the same visual style as :func:`biome_render.render_forest_tile`
    but with :class:`SubfaceTree` instances placed at sub-face centroids
    instead of Poisson-disk-scattered positions.

    Parameters
    ----------
    ground_image : PIL.Image
        Ground/elevation texture (RGB).
    trees : list of SubfaceTree
        From :func:`scatter_trees_on_grid`.
    density : float
        For undergrowth opacity.
    shadow_offset, shadow_opacity, shadow_color
        Shadow rendering params.
    highlight_strength, highlight_offset
        Canopy highlight params.
    undergrowth_color, undergrowth_noise_amp
        Undergrowth fill params.
    seed : int

    Returns
    -------
    PIL.Image (RGBA)
    """
    base = ground_image.convert("RGBA")
    w, h = base.size
    rng = random.Random(seed)

    # Layer 1: undergrowth
    undergrowth = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    u_draw = ImageDraw.Draw(undergrowth)
    ur, ug, ub = undergrowth_color
    step = max(2, int(4 - density * 2))
    for y in range(0, h, step):
        for x in range(0, w, step):
            dr = rng.randint(-int(undergrowth_noise_amp), int(undergrowth_noise_amp))
            dg = rng.randint(-int(undergrowth_noise_amp), int(undergrowth_noise_amp))
            db = rng.randint(-int(undergrowth_noise_amp), int(undergrowth_noise_amp))
            pc = (
                max(0, min(255, ur + dr)),
                max(0, min(255, ug + dg)),
                max(0, min(255, ub + db)),
                int(density * 200),
            )
            u_draw.rectangle([x, y, x + step - 1, y + step - 1], fill=pc)
    base = Image.alpha_composite(base, undergrowth)

    # Layer 2: shadows + canopies
    canopy_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canopy_layer)

    for tree in trees:
        px, py = tree.px, tree.py
        r = tree.radius
        cr, cg, cb = tree.color
        tree_rng = random.Random(int(px * 1000 + py))

        # Shadow
        sdx, sdy = shadow_offset
        sx, sy = px + sdx, py + sdy
        sr, sg, sb = shadow_color
        s_alpha = int(shadow_opacity * 255)
        sr_size = r * 1.1
        draw.ellipse(
            [sx - sr_size, sy - sr_size, sx + sr_size, sy + sr_size],
            fill=(sr, sg, sb, s_alpha),
        )

        # Canopy base — slight shape irregularity
        jitter = 1.0 + tree_rng.uniform(-0.1, 0.1)
        rx = r * jitter
        ry = r * (2.0 - jitter)
        draw.ellipse(
            [px - rx, py - ry, px + rx, py + ry],
            fill=(cr, cg, cb, 255),
        )

        # Inner ring for depth
        inner_r = r * 0.55
        shade = max(0, int(-15 + tree_rng.randint(-5, 5)))
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

        # Highlight
        if highlight_strength > 0.05:
            hx = px + highlight_offset[0]
            hy = py + highlight_offset[1]
            hr = r * 0.3
            bright = int(highlight_strength * 80)
            hl_color = (
                min(255, cr + bright),
                min(255, cg + bright),
                min(255, cb + bright),
                int(highlight_strength * 180),
            )
            draw.ellipse(
                [hx - hr, hy - hr, hx + hr, hy + hr],
                fill=hl_color,
            )

    # Soft edge blur
    if trees:
        try:
            canopy_layer = canopy_layer.filter(ImageFilter.GaussianBlur(radius=0.5))
        except Exception:
            pass

    base = Image.alpha_composite(base, canopy_layer)
    return base


# ═══════════════════════════════════════════════════════════════════
# 18C.2 — Sub-face ocean features
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SubfaceOceanProps:
    """Ocean properties for a single sub-face.

    Attributes
    ----------
    face_id : str
    depth : float
        Normalised depth [0, 1] from elevation.
    is_coastal : bool
        Adjacent to at least one land (above-water) sub-face.
    pixel_verts : list of (float, float)
        Polygon vertices in pixel coords.
    centroid : tuple of (float, float)
        Centroid in pixel coords.
    area : float
        Polygon area in pixels².
    """
    face_id: str
    depth: float
    is_coastal: bool
    pixel_verts: List[Tuple[float, float]]
    centroid: Tuple[float, float]
    area: float


def compute_subface_ocean_depth(
    detail_grid,
    store,
    *,
    water_level: float = 0.12,
    elevation_field: str = "elevation",
    face_ids: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """Compute normalised depth [0, 1] for each sub-face from elevation.

    ``depth = (water_level - elevation) / water_level``, clamped to [0, 1].
    Sub-faces above water level get depth 0.

    Parameters
    ----------
    detail_grid : PolyGrid
    store : TileDataStore
    water_level : float
    elevation_field : str
    face_ids : set, optional

    Returns
    -------
    dict
        ``{face_id: depth}``
    """
    targets = face_ids if face_ids else set(detail_grid.faces.keys())
    depth_map: Dict[str, float] = {}

    for fid in targets:
        elev = store.get(fid, elevation_field)
        if elev is None:
            elev = 0.0
        d = (water_level - elev) / max(water_level, 0.001)
        depth_map[fid] = max(0.0, min(1.0, d))

    return depth_map


def identify_coastal_subfaces(
    detail_grid,
    depth_map: Dict[str, float],
    *,
    land_threshold: float = 0.05,
) -> Set[str]:
    """Find sub-faces that are adjacent to land (above-water) sub-faces.

    A sub-face is "coastal" if it is underwater (depth > land_threshold)
    and at least one of its topological neighbours is land
    (depth <= land_threshold).

    Parameters
    ----------
    detail_grid : PolyGrid
    depth_map : dict
        ``{face_id: depth}`` from :func:`compute_subface_ocean_depth`.
    land_threshold : float
        Depth below which a sub-face counts as land.

    Returns
    -------
    set of str
        Coastal sub-face IDs.
    """
    from .algorithms import get_face_adjacency

    adj = get_face_adjacency(detail_grid)
    coastal: Set[str] = set()

    for fid, depth in depth_map.items():
        if depth <= land_threshold:
            continue  # this is land, not coastal ocean
        for nid in adj.get(fid, []):
            nbr_depth = depth_map.get(nid, 0.0)
            if nbr_depth <= land_threshold:
                coastal.add(fid)
                break

    return coastal


def _ocean_depth_color(
    depth: float,
    shallow: Tuple[int, int, int] = (64, 164, 192),
    deep: Tuple[int, int, int] = (16, 48, 112),
    abyssal: Tuple[int, int, int] = (6, 16, 42),
    power: float = 1.8,
) -> Tuple[int, int, int]:
    """Map depth [0, 1] to RGB using three colour stops."""
    d = max(0.0, min(1.0, depth)) ** power
    if d < 0.5:
        t = d * 2.0
        return (
            int(shallow[0] + (deep[0] - shallow[0]) * t),
            int(shallow[1] + (deep[1] - shallow[1]) * t),
            int(shallow[2] + (deep[2] - shallow[2]) * t),
        )
    else:
        t = (d - 0.5) * 2.0
        return (
            int(deep[0] + (abyssal[0] - deep[0]) * t),
            int(deep[1] + (abyssal[1] - deep[1]) * t),
            int(deep[2] + (abyssal[2] - deep[2]) * t),
        )


def compute_ocean_subface_props(
    detail_grid,
    store,
    *,
    tile_size: int = 256,
    water_level: float = 0.12,
    elevation_field: str = "elevation",
    face_ids: Optional[Set[str]] = None,
) -> List[SubfaceOceanProps]:
    """Build ocean properties for each sub-face in a detail grid.

    Parameters
    ----------
    detail_grid : PolyGrid
    store : TileDataStore
    tile_size : int
    water_level : float
    elevation_field : str
    face_ids : set, optional

    Returns
    -------
    list of SubfaceOceanProps
    """
    from .geometry import face_center

    # Build pixel mapping
    xs, ys = [], []
    for v in detail_grid.vertices.values():
        if v.has_position():
            xs.append(v.x)
            ys.append(v.y)
    if not xs:
        return []

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span = max((x_max - x_min) or 1.0, (y_max - y_min) or 1.0)
    pad = span * 0.15
    x_min -= pad
    x_max += pad
    y_min -= pad
    y_max += pad
    x_range = x_max - x_min
    y_range = y_max - y_min
    scale = tile_size / max(x_range, y_range)
    ox = (tile_size - x_range * scale) / 2.0
    oy = (tile_size - y_range * scale) / 2.0

    def to_pixel(vx: float, vy: float) -> Tuple[float, float]:
        px = (vx - x_min) * scale + ox
        py = tile_size - ((vy - y_min) * scale + oy)
        return (px, py)

    depth_map = compute_subface_ocean_depth(
        detail_grid, store,
        water_level=water_level,
        elevation_field=elevation_field,
        face_ids=face_ids,
    )
    coastal = identify_coastal_subfaces(
        detail_grid, depth_map,
    )

    targets = face_ids if face_ids else set(detail_grid.faces.keys())
    props: List[SubfaceOceanProps] = []

    for fid in targets:
        face = detail_grid.faces.get(fid)
        if face is None:
            continue

        cx, cy, area, pixel_verts = _sub_face_pixel_centroid_and_area(
            detail_grid, face, detail_grid.vertices, to_pixel,
        )
        if area <= 0:
            continue

        props.append(SubfaceOceanProps(
            face_id=fid,
            depth=depth_map.get(fid, 0.5),
            is_coastal=fid in coastal,
            pixel_verts=pixel_verts,
            centroid=(cx, cy),
            area=area,
        ))

    return props


def render_topology_ocean(
    ground_image: "Image.Image",
    ocean_props: List[SubfaceOceanProps],
    *,
    tile_depth: float = 0.5,
    shallow_color: Tuple[int, int, int] = (64, 164, 192),
    deep_color: Tuple[int, int, int] = (16, 48, 112),
    abyssal_color: Tuple[int, int, int] = (6, 16, 42),
    foam_color: Tuple[int, int, int] = (220, 230, 235),
    depth_power: float = 1.8,
    wave_frequency: float = 6.0,
    wave_amplitude: float = 0.04,
    foam_strength: float = 0.6,
    seed: int = 42,
) -> "Image.Image":
    """Render ocean features using per-sub-face depth and coastal data.

    Instead of pixel-level depth gradient + wave overlay, this renders
    each sub-face polygon with its own depth colour, then composites
    coastal foam on identified coastal sub-faces, and adds subtle
    per-polygon wave modulation.

    Parameters
    ----------
    ground_image : PIL.Image
        Base ground texture.
    ocean_props : list of SubfaceOceanProps
    tile_depth : float
        Fallback depth for the tile.
    shallow_color, deep_color, abyssal_color : (r, g, b)
    foam_color : (r, g, b)
    depth_power : float
    wave_frequency, wave_amplitude : float
    foam_strength : float
    seed : int

    Returns
    -------
    PIL.Image (RGB)
    """
    img = ground_image.copy().convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)

    if not ocean_props:
        return img

    # ── Step 1: Polygon-fill each sub-face with depth colour ────
    for prop in ocean_props:
        if not prop.pixel_verts or len(prop.pixel_verts) < 3:
            continue

        base_color = _ocean_depth_color(
            prop.depth, shallow_color, deep_color, abyssal_color, depth_power,
        )

        # Per-polygon wave modulation — subtle brightness shift
        cx, cy = prop.centroid
        wave = math.sin(cy / max(h, 1) * wave_frequency * 2 * math.pi + cx / max(w, 1) * 1.5)
        brightness = 1.0 + wave * wave_amplitude
        # Depth-dependent wave damping — deeper = calmer
        brightness = 1.0 + (brightness - 1.0) * max(0.2, 1.0 - prop.depth * 0.7)

        r = max(0, min(255, int(base_color[0] * brightness)))
        g = max(0, min(255, int(base_color[1] * brightness)))
        b = max(0, min(255, int(base_color[2] * brightness)))

        # Slight per-face noise for variation
        noise_val = rng.uniform(-5, 5)
        r = max(0, min(255, r + int(noise_val)))
        g = max(0, min(255, g + int(noise_val * 0.8)))
        b = max(0, min(255, b + int(noise_val * 0.6)))

        poly = [(int(px), int(py)) for px, py in prop.pixel_verts]
        draw.polygon(poly, fill=(r, g, b))

    # ── Step 2: Coastal foam on coastal sub-faces ───────────────
    foam_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    foam_draw = ImageDraw.Draw(foam_layer)

    for prop in ocean_props:
        if not prop.is_coastal or not prop.pixel_verts:
            continue

        # Foam alpha depends on depth — shallower = more foam
        foam_alpha = int(foam_strength * (1.0 - min(prop.depth * 4, 1.0)) * 180)
        if foam_alpha < 5:
            continue

        # Add noise to foam
        fr, fg, fb = foam_color
        fn = rng.randint(-10, 10)
        fc = (
            max(0, min(255, fr + fn)),
            max(0, min(255, fg + fn)),
            max(0, min(255, fb + fn)),
            foam_alpha,
        )

        poly = [(int(px), int(py)) for px, py in prop.pixel_verts]
        foam_draw.polygon(poly, fill=fc)

    # Blur foam for soft edges
    try:
        foam_layer = foam_layer.filter(ImageFilter.GaussianBlur(radius=1.0))
    except Exception:
        pass

    img = Image.alpha_composite(img.convert("RGBA"), foam_layer).convert("RGB")

    # ── Step 3: Abyssal darkening for deep sub-faces ────────────
    for prop in ocean_props:
        if prop.depth <= 0.5 or not prop.pixel_verts:
            continue

        deep_factor = (prop.depth - 0.5) * 2.0
        darken = 1.0 - deep_factor * 0.15  # subtle

        poly = [(int(px), int(py)) for px, py in prop.pixel_verts]
        # Re-read and darken pixel values in the polygon
        # Use a simple overlay approach
        dark_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        dark_draw = ImageDraw.Draw(dark_layer)
        alpha = int(deep_factor * 40)
        dark_draw.polygon(poly, fill=(0, 0, 0, alpha))
        img = Image.alpha_composite(img.convert("RGBA"), dark_layer).convert("RGB")

    return img


# ═══════════════════════════════════════════════════════════════════
# 18C.3 — Hybrid rendering (topology + pixel-level)
# ═══════════════════════════════════════════════════════════════════

def render_hybrid_biome(
    ground_image: "Image.Image",
    detail_grid,
    store,
    *,
    biome_type: str = "forest",
    tile_size: int = 256,
    density: float = 0.8,
    tile_depth: float = 0.5,
    forest_config=None,
    ocean_config=None,
    seed: int = 42,
    noise_overlay: bool = True,
    noise_frequency: float = 0.05,
    noise_amplitude: float = 0.05,
    face_ids: Optional[Set[str]] = None,
) -> "Image.Image":
    """Hybrid rendering: topology-driven features + pixel-level noise.

    Layers:
    1. Ground texture (already rendered with terrain colours)
    2. Topology pass — sub-face features (trees or ocean depth)
    3. Pixel noise overlay for micro-detail

    Parameters
    ----------
    ground_image : PIL.Image
    detail_grid : PolyGrid
    store : TileDataStore
    biome_type : str
        ``"forest"`` or ``"ocean"``.
    tile_size : int
    density : float
    tile_depth : float
    forest_config, ocean_config : optional
    seed : int
    noise_overlay : bool
    noise_frequency, noise_amplitude : float
    face_ids : set, optional

    Returns
    -------
    PIL.Image (RGB or RGBA)
    """
    if biome_type == "forest":
        trees = scatter_trees_on_grid(
            detail_grid, store,
            tile_size=tile_size,
            density=density,
            seed=seed,
            face_ids=face_ids,
        )

        # Apply forest config if provided
        kwargs = {}
        if forest_config is not None:
            from .biome_render import ForestFeatureConfig
            if isinstance(forest_config, ForestFeatureConfig):
                kwargs.update(
                    shadow_offset=forest_config.shadow_offset,
                    shadow_opacity=forest_config.shadow_opacity,
                    shadow_color=forest_config.shadow_color,
                    highlight_strength=forest_config.highlight_strength,
                    highlight_offset=forest_config.highlight_offset,
                    undergrowth_color=forest_config.undergrowth_color,
                    undergrowth_noise_amp=forest_config.undergrowth_noise,
                )

        result = render_topology_forest(
            ground_image, trees,
            density=density,
            seed=seed,
            **kwargs,
        )

    elif biome_type == "ocean":
        ocean_props = compute_ocean_subface_props(
            detail_grid, store,
            tile_size=tile_size,
            face_ids=face_ids,
        )

        kwargs = {}
        if ocean_config is not None:
            from .ocean_render import OceanFeatureConfig
            if isinstance(ocean_config, OceanFeatureConfig):
                kwargs.update(
                    shallow_color=ocean_config.shallow_color,
                    deep_color=ocean_config.deep_color,
                    abyssal_color=ocean_config.abyssal_color,
                    depth_power=ocean_config.depth_gradient_power,
                    wave_frequency=ocean_config.wave_frequency,
                    wave_amplitude=ocean_config.wave_amplitude,
                    foam_color=ocean_config.coastal_foam_color,
                )

        result = render_topology_ocean(
            ground_image, ocean_props,
            tile_depth=tile_depth,
            seed=seed,
            **kwargs,
        )

    else:
        result = ground_image

    # Pixel noise overlay for micro-detail
    if noise_overlay:
        from .tile_texture import apply_noise_overlay
        arr = np.array(result.convert("RGB"))
        arr = apply_noise_overlay(
            arr,
            frequency=noise_frequency,
            amplitude=noise_amplitude,
            seed=seed + 7777,
        )
        result = Image.fromarray(arr, "RGB")

    return result


# ═══════════════════════════════════════════════════════════════════
# Renderer classes implementing BiomeRenderer protocol
# ═══════════════════════════════════════════════════════════════════

class TopologyForestRenderer:
    """Topology-aware forest renderer — implements ``BiomeRenderer``.

    Places trees at sub-face centroids instead of random Poisson-disk
    positions.  Requires detail grid and store to be passed via the
    ``set_grid_context()`` method before ``render()`` is called.

    Parameters
    ----------
    config : ForestFeatureConfig, optional
    """

    def __init__(self, config=None):
        from .biome_render import ForestFeatureConfig, TEMPERATE_FOREST
        self.config = config if config is not None else TEMPERATE_FOREST
        self._detail_grid = None
        self._store = None

    def set_grid_context(self, detail_grid, store) -> None:
        """Set the detail grid and store for topology-aware rendering.

        Must be called before ``render()`` for each tile.
        """
        self._detail_grid = detail_grid
        self._store = store

    def render(
        self,
        ground_image: "Image.Image",
        tile_id: str,
        density: float,
        *,
        seed: int = 42,
        globe_3d_center=None,
        blend_mask=None,
        neighbour_densities=None,
        neighbour_seeds=None,
    ) -> "Image.Image":
        """Render topology-aware forest features.

        If no grid context has been set, falls back to the standard
        pixel-based forest renderer.
        """
        tile_seed = seed + hash(tile_id) % 100_000

        if self._detail_grid is None or self._store is None:
            # Fallback to pixel-based
            from .biome_render import render_forest_on_ground
            return render_forest_on_ground(
                ground_image,
                density * self.config.density_scale,
                config=self.config,
                tile_size=ground_image.size[0],
                seed=tile_seed,
            )

        trees = scatter_trees_on_grid(
            self._detail_grid,
            self._store,
            tile_size=ground_image.size[0],
            density=density * self.config.density_scale,
            canopy_colors=self.config.canopy_colors,
            color_noise=self.config.color_noise_amplitude,
            min_radius=self.config.canopy_radius_range[0],
            max_radius=self.config.canopy_radius_range[1],
            seed=tile_seed,
        )

        result = render_topology_forest(
            ground_image,
            trees,
            density=density * self.config.density_scale,
            shadow_offset=self.config.shadow_offset,
            shadow_opacity=self.config.shadow_opacity,
            shadow_color=self.config.shadow_color,
            highlight_strength=self.config.highlight_strength,
            highlight_offset=self.config.highlight_offset,
            undergrowth_color=self.config.undergrowth_color,
            undergrowth_noise_amp=self.config.undergrowth_noise,
            seed=tile_seed,
        )

        # Apply blend mask if provided
        if blend_mask is not None:
            ground_rgba = ground_image.convert("RGBA")
            result_rgba = result.convert("RGBA")
            ground_arr = np.array(ground_rgba, dtype=np.float64)
            result_arr = np.array(result_rgba, dtype=np.float64)
            h, w = ground_arr.shape[:2]
            mask_resized = blend_mask[:h, :w]
            mask_4d = mask_resized[:, :, np.newaxis]
            blended = ground_arr * (1.0 - mask_4d) + result_arr * mask_4d
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            result = Image.fromarray(blended, "RGBA")

        return result


class TopologyOceanRenderer:
    """Topology-aware ocean renderer — implements ``BiomeRenderer``.

    Uses per-sub-face depth and coastal classification instead of
    per-tile depth.  Requires detail grid and store via
    ``set_grid_context()``.

    Parameters
    ----------
    config : OceanFeatureConfig, optional
    ocean_depth_map : dict, optional
        Globe-level ``{face_id: depth}`` for fallback.
    ocean_faces : set, optional
    globe_grid : PolyGrid, optional
    """

    def __init__(
        self,
        config=None,
        *,
        ocean_depth_map: Optional[Dict[str, float]] = None,
        ocean_faces: Optional[set] = None,
        globe_grid=None,
    ):
        from .ocean_render import TEMPERATE_OCEAN as _DEFAULT
        self.config = config if config is not None else _DEFAULT
        self.ocean_depth_map = ocean_depth_map or {}
        self.ocean_faces = ocean_faces or set()
        self.globe_grid = globe_grid
        self._detail_grid = None
        self._store = None

    def set_grid_context(self, detail_grid, store) -> None:
        """Set the detail grid and store for topology-aware rendering."""
        self._detail_grid = detail_grid
        self._store = store

    def render(
        self,
        ground_image: "Image.Image",
        tile_id: str,
        density: float,
        *,
        seed: int = 42,
        globe_3d_center=None,
        blend_mask=None,
        neighbour_densities=None,
        neighbour_seeds=None,
    ) -> "Image.Image":
        """Render topology-aware ocean features.

        If no grid context has been set, falls back to the standard
        pixel-based ocean renderer.
        """
        tile_seed = seed + hash(tile_id) % 100_000
        depth = self.ocean_depth_map.get(tile_id, 0.5)

        if self._detail_grid is None or self._store is None:
            # Fallback to pixel-based
            from .ocean_render import render_ocean_tile, compute_coast_direction
            coast_dir = None
            if self.globe_grid is not None and self.ocean_faces:
                coast_dir = compute_coast_direction(
                    self.globe_grid, tile_id, self.ocean_faces,
                )
            return render_ocean_tile(
                ground_image, depth,
                config=self.config,
                coast_direction=coast_dir,
                seed=tile_seed,
            )

        ocean_props = compute_ocean_subface_props(
            self._detail_grid,
            self._store,
            tile_size=ground_image.size[0],
        )

        result = render_topology_ocean(
            ground_image,
            ocean_props,
            tile_depth=depth,
            shallow_color=self.config.shallow_color,
            deep_color=self.config.deep_color,
            abyssal_color=self.config.abyssal_color,
            foam_color=self.config.coastal_foam_color,
            depth_power=self.config.depth_gradient_power,
            wave_frequency=self.config.wave_frequency,
            wave_amplitude=self.config.wave_amplitude,
            seed=tile_seed,
        )

        # Apply blend mask if provided
        if blend_mask is not None:
            ground_rgba = ground_image.convert("RGBA")
            result_rgba = result.convert("RGBA") if result.mode != "RGBA" else result
            ground_arr = np.array(ground_rgba, dtype=np.float64)
            result_arr = np.array(result_rgba, dtype=np.float64)
            h, w = ground_arr.shape[:2]
            mask_resized = blend_mask[:h, :w]
            mask_4d = mask_resized[:, :, np.newaxis]
            blended = ground_arr * (1.0 - mask_4d) + result_arr * mask_4d
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            result = Image.fromarray(blended, "RGBA")

        return result
