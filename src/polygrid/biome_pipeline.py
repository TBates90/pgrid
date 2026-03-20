# TODO REMOVE — Not used by any live script. Phase 14D biome atlas pipeline.
"""Biome feature pipeline — integrate feature rendering into the atlas.

Hooks the biome feature renderers into the existing texture atlas
pipeline so that ``build_feature_atlas()`` automatically overlays
forest (and future biome) features on top of the ground textures.

Functions
---------
- :func:`build_feature_atlas` — extended atlas with biome overlays
- :func:`identify_forest_tiles` — pick forest tiles from terrain patches

Classes / Protocols
-------------------
- :class:`BiomeRenderer` — protocol for all biome feature renderers
- :class:`ForestRenderer` — forest implementation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ═══════════════════════════════════════════════════════════════════
# 14D.1 — BiomeRenderer protocol
# ═══════════════════════════════════════════════════════════════════

class BiomeRenderer(Protocol):
    """Interface for all biome feature renderers.

    Any class that satisfies this protocol can be plugged into the
    atlas builder via ``build_feature_atlas()``.
    """

    def render(
        self,
        ground_image: "Image.Image",
        tile_id: str,
        density: float,
        *,
        seed: int = 42,
        globe_3d_center: Optional[Tuple[float, float, float]] = None,
    ) -> "Image.Image":
        """Overlay biome features on a ground texture.

        Parameters
        ----------
        ground_image : PIL.Image
            The existing ground texture (RGB).
        tile_id : str
            Globe-level face ID for this tile.
        density : float
            Biome density at this tile (0–1).
        seed : int
            Random seed.
        globe_3d_center : (x, y, z), optional
            3-D position of the tile centre on the globe.

        Returns
        -------
        PIL.Image
            Composited image with features on top.
        """
        ...


# ═══════════════════════════════════════════════════════════════════
# 14D.2 — ForestRenderer
# ═══════════════════════════════════════════════════════════════════

class ForestRenderer:
    """Forest biome renderer — implements :class:`BiomeRenderer`.

    Scatters tree canopies using Poisson disk sampling, draws
    undergrowth, shadows, and canopy highlights.
    """

    def __init__(self, config=None, *, fullslot: bool = False):
        from .biome_render import ForestFeatureConfig, TEMPERATE_FOREST
        self.config = config if config is not None else TEMPERATE_FOREST
        self.fullslot = fullslot

    def render(
        self,
        ground_image: "Image.Image",
        tile_id: str,
        density: float,
        *,
        seed: int = 42,
        globe_3d_center: Optional[Tuple[float, float, float]] = None,
        blend_mask=None,
        neighbour_densities: Optional[Dict[str, float]] = None,
        neighbour_seeds: Optional[Dict[str, int]] = None,
    ) -> "Image.Image":
        tile_seed = seed + hash(tile_id) % 100_000

        if self.fullslot:
            from .biome_render import render_forest_on_ground_fullslot
            return render_forest_on_ground_fullslot(
                ground_image,
                density * self.config.density_scale,
                config=self.config,
                tile_size=ground_image.size[0],
                seed=tile_seed,
                globe_3d_center=globe_3d_center,
                blend_mask=blend_mask,
                neighbour_densities=neighbour_densities,
                neighbour_seeds=neighbour_seeds,
            )

        from .biome_render import render_forest_on_ground
        return render_forest_on_ground(
            ground_image,
            density * self.config.density_scale,
            config=self.config,
            tile_size=ground_image.size[0],
            seed=tile_seed,
            globe_3d_center=globe_3d_center,
        )


# ═══════════════════════════════════════════════════════════════════
# 17C.1 — OceanRenderer
# ═══════════════════════════════════════════════════════════════════

class OceanRenderer:
    """Ocean biome renderer — implements :class:`BiomeRenderer`.

    Renders ocean tile textures with depth-based colour gradients,
    wave patterns, coastal detail, and deep-ocean effects.

    Parameters
    ----------
    config : OceanFeatureConfig, optional
        Ocean rendering parameters.  Defaults to ``TEMPERATE_OCEAN``.
    ocean_depth_map : dict, optional
        ``{face_id: float}`` normalised depth [0, 1] from
        :func:`compute_ocean_depth_map`.  If *None*, tiles default
        to depth 0.5.
    ocean_faces : set, optional
        Set of face IDs classified as ocean.  Needed for coast
        direction computation.
    globe_grid : PolyGrid, optional
        Required for coast direction computation.
    """

    def __init__(
        self,
        config=None,
        *,
        ocean_depth_map: Optional[Dict[str, float]] = None,
        ocean_faces: Optional[set] = None,
        globe_grid=None,
    ):
        from .ocean_render import TEMPERATE_OCEAN as _DEFAULT_OCEAN
        self.config = config if config is not None else _DEFAULT_OCEAN
        self.ocean_depth_map = ocean_depth_map or {}
        self.ocean_faces = ocean_faces or set()
        self.globe_grid = globe_grid

    def render(
        self,
        ground_image: "Image.Image",
        tile_id: str,
        density: float,
        *,
        seed: int = 42,
        globe_3d_center: Optional[Tuple[float, float, float]] = None,
        blend_mask=None,
        neighbour_densities: Optional[Dict[str, float]] = None,
        neighbour_seeds: Optional[Dict[str, int]] = None,
    ) -> "Image.Image":
        from .ocean_render import render_ocean_tile, compute_coast_direction

        depth = self.ocean_depth_map.get(tile_id, 0.5)

        # Compute coast direction if we have the info
        coast_dir = None
        if self.globe_grid is not None and self.ocean_faces:
            coast_dir = compute_coast_direction(
                self.globe_grid, tile_id, self.ocean_faces,
            )

        result = render_ocean_tile(
            ground_image,
            depth,
            config=self.config,
            coast_direction=coast_dir,
            seed=seed + hash(tile_id) % 100_000,
        )

        # Apply blend mask cross-fade if provided (16B integration)
        if blend_mask is not None:
            import numpy as np
            ground_arr = np.array(ground_image.convert("RGB")).astype(np.float32)
            result_arr = np.array(result).astype(np.float32)
            mask = blend_mask[:, :, np.newaxis]
            blended = ground_arr * (1.0 - mask) + result_arr * mask
            result = Image.fromarray(
                np.clip(blended, 0, 255).astype(np.uint8)
            )

        return result


# ═══════════════════════════════════════════════════════════════════
# 14D.3 — Identify forest tiles from terrain patches
# ═══════════════════════════════════════════════════════════════════

def identify_forest_tiles(
    patches: Sequence,
    *,
    terrain_type: str = "forest",
) -> set:
    """Return the set of face IDs belonging to the given terrain type.

    Parameters
    ----------
    patches : sequence of TerrainPatch
        The terrain patches from ``generate_terrain_patches()``.
    terrain_type : str
        The terrain type to match (default ``"forest"``).

    Returns
    -------
    set of str
        Face IDs classified as the given terrain type.
    """
    face_ids = set()
    for patch in patches:
        if patch.terrain_type == terrain_type:
            face_ids.update(patch.face_ids)
    return face_ids


# ═══════════════════════════════════════════════════════════════════
# 14D.3 — Feature atlas builder
# ═══════════════════════════════════════════════════════════════════

def build_feature_atlas(
    collection,
    globe_grid=None,
    *,
    biome_renderers: Optional[Dict[str, BiomeRenderer]] = None,
    density_map: Optional[Dict[str, float]] = None,
    biome_type_map: Optional[Dict[str, str]] = None,
    biome_config=None,
    output_dir: Path | str = Path("exports/detail_tiles"),
    tile_size: int = 256,
    columns: int = 0,
    noise_seed: int = 0,
    gutter: int = 4,
    fullslot: bool = False,
    soft_blend: bool = False,
    blend_fade_width: int = 16,
) -> Tuple[Path, Dict[str, Tuple[float, float, float, float]]]:
    """Build a texture atlas with biome feature overlays.

    First renders ground textures via the standard pipeline, then
    applies biome renderers to tiles that have a density > 0.

    Parameters
    ----------
    collection : DetailGridCollection
        Must have stores populated.
    globe_grid : PolyGrid / GlobeGrid, optional
        For 3-D centre lookups.  If *None*, features are placed
        without globe-coherent noise.
    biome_renderers : dict, optional
        ``{"forest": ForestRenderer(), ...}``.  If *None*, a default
        ``ForestRenderer`` is used for all tiles with density > 0.
    density_map : dict, optional
        ``{face_id: float}``.  If *None*, all tiles get density 0
        (no features — falls back to standard atlas).
    biome_type_map : dict, optional
        ``{face_id: "forest"|"ocean"|...}``.  Maps each tile to a
        renderer key in *biome_renderers*.  If *None*, the first
        renderer is used for all tiles with density > 0 (backward
        compatible).
    biome_config : BiomeConfig, optional
        Ground-texture rendering configuration.
    output_dir : Path or str
    tile_size : int
    columns : int
    noise_seed : int
    gutter : int
    fullslot : bool
        When *True*, use the Phase 16A full-slot renderer that fills
        every pixel with coherent terrain (no flat-fill background)
        and enables 16D hex-shape softening.
        Default *False* for backward compatibility.
    soft_blend : bool
        When *True*, enable the full Phase 16 pipeline: fullslot
        ground rendering (16A), soft tile-edge blending (16B),
        fullslot feature scatter (16C), and hex softening (16D).
        Implies ``fullslot=True``.  Default *False*.
    blend_fade_width : int
        Pixel width of the soft blend zone at tile edges (16B).
        Default 16.

    Returns
    -------
    (atlas_path, uv_layout)
        Same format as ``build_detail_atlas()``.
    """
    from .detail_render import BiomeConfig, render_detail_texture_enhanced
    from .geometry import face_center_3d

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if biome_config is None:
        biome_config = BiomeConfig()

    if density_map is None:
        density_map = {}

    # soft_blend implies fullslot
    if soft_blend:
        fullslot = True

    if biome_renderers is None:
        biome_renderers = {"forest": ForestRenderer(fullslot=soft_blend)}

    if biome_type_map is None:
        biome_type_map = {}

    # Default renderer is the first one (backward compat)
    default_renderer = next(iter(biome_renderers.values())) if biome_renderers else None

    def _get_renderer(fid: str) -> Optional[BiomeRenderer]:
        """Look up the correct renderer for a face ID."""
        biome_key = biome_type_map.get(fid)
        if biome_key is not None and biome_key in biome_renderers:
            return biome_renderers[biome_key]
        return default_renderer

    # Optionally use full-slot renderer (Phase 16A + 16D)
    if fullslot:
        from .tile_texture import render_detail_texture_fullslot
        _render_ground = render_detail_texture_fullslot
    else:
        _render_ground = render_detail_texture_enhanced

    # Pre-compute blend masks (16B) if soft_blend is enabled
    _blend_masks: Dict[str, Any] = {}
    if soft_blend:
        import numpy as np
        from .tile_texture import compute_tile_blend_mask, apply_blend_mask_to_atlas

    # Build face adjacency for neighbour info (16C scatter overflow)
    _adjacency: Dict[str, List[str]] = {}
    if soft_blend and globe_grid is not None:
        from .algorithms import get_face_adjacency
        _adjacency = get_face_adjacency(globe_grid)

    face_ids = collection.face_ids
    n = len(face_ids)
    if n == 0:
        raise ValueError("No detail grids in the collection")

    # Pre-compute blend masks for each tile (16B)
    if soft_blend:
        for fid in face_ids:
            grid, _store = collection.get(fid)
            _blend_masks[fid] = compute_tile_blend_mask(
                grid,
                tile_size=tile_size,
                fade_width=blend_fade_width,
            )

    # ── Render ground textures + apply biome overlays ───────────
    tile_paths: Dict[str, Path] = {}
    for fid in face_ids:
        grid, store = collection.get(fid)
        if store is None:
            raise ValueError(
                f"No terrain store for face '{fid}' — "
                "call generate_all_detail_terrain first"
            )

        ground_path = output_dir / f"tile_{fid}.png"
        _render_ground(
            grid, store, ground_path, biome_config,
            tile_size=tile_size,
            noise_seed=noise_seed + hash(fid) % 10000,
        )

        # Apply biome overlay if this tile has density
        tile_density = density_map.get(fid, 0.0)
        tile_renderer = _get_renderer(fid)
        if tile_density > 0.01 and tile_renderer is not None:
            ground_img = Image.open(str(ground_path)).convert("RGB")

            # Get 3D centre for globe-coherent placement
            center_3d = None
            if globe_grid is not None:
                face = globe_grid.faces.get(fid)
                if face is not None:
                    center_3d = face_center_3d(globe_grid.vertices, face)

            # Build neighbour density/seed maps for fullslot scatter (16C)
            nbr_densities: Optional[Dict[str, float]] = None
            nbr_seeds: Optional[Dict[str, int]] = None
            if soft_blend and fid in _adjacency:
                nbr_densities = {
                    nid: density_map.get(nid, 0.0)
                    for nid in _adjacency[fid]
                }
                nbr_seeds = {
                    nid: noise_seed + hash(nid) % 100_000
                    for nid in _adjacency[fid]
                }

            featured_img = tile_renderer.render(
                ground_img,
                fid,
                tile_density,
                seed=noise_seed + hash(fid) % 100_000,
                globe_3d_center=center_3d,
                blend_mask=_blend_masks.get(fid) if soft_blend else None,
                neighbour_densities=nbr_densities,
                neighbour_seeds=nbr_seeds,
            )
            # Save as RGB (atlas doesn't need alpha)
            featured_img.convert("RGB").save(str(ground_path))

        tile_paths[fid] = ground_path

    # ── Assemble atlas (reuse logic from texture_pipeline) ──────
    if columns <= 0:
        columns = max(1, math.isqrt(n))
        if columns * columns < n:
            columns += 1
    rows = math.ceil(n / columns)

    slot_size = tile_size + 2 * gutter
    atlas_w = columns * slot_size
    atlas_h = rows * slot_size
    atlas = Image.new("RGB", (atlas_w, atlas_h), (128, 128, 128))

    uv_layout: Dict[str, Tuple[float, float, float, float]] = {}

    for idx, fid in enumerate(face_ids):
        col = idx % columns
        row = idx // columns
        slot_x = col * slot_size
        slot_y = row * slot_size

        tile_img = Image.open(str(tile_paths[fid])).convert("RGB")
        tile_img = tile_img.resize((tile_size, tile_size), Image.LANCZOS)

        atlas.paste(tile_img, (slot_x + gutter, slot_y + gutter))

        # Fill gutter
        if gutter > 0:
            from .atlas_utils import fill_gutter
            fill_gutter(atlas, slot_x, slot_y, tile_size, gutter)

        inner_x = slot_x + gutter
        inner_y = slot_y + gutter
        u_min = inner_x / atlas_w
        u_max = (inner_x + tile_size) / atlas_w
        v_min = 1.0 - (inner_y + tile_size) / atlas_h
        v_max = 1.0 - inner_y / atlas_h
        uv_layout[fid] = (u_min, v_min, u_max, v_max)

    # ── Apply soft tile-edge blend to assembled atlas (16B) ─────
    if soft_blend and _blend_masks:
        atlas_arr = np.array(atlas)
        atlas_arr = apply_blend_mask_to_atlas(
            atlas_arr, _blend_masks, face_ids,
            tile_size, gutter, columns,
        )
        atlas = Image.fromarray(atlas_arr)

    atlas_path = output_dir / "detail_atlas.png"
    atlas.save(str(atlas_path))

    return atlas_path, uv_layout
