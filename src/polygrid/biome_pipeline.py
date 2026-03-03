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
    biome_config=None,
    output_dir: Path | str = Path("exports/detail_tiles"),
    tile_size: int = 256,
    columns: int = 0,
    noise_seed: int = 0,
    gutter: int = 4,
    fullslot: bool = False,
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
    biome_config : BiomeConfig, optional
        Ground-texture rendering configuration.
    output_dir : Path or str
    tile_size : int
    columns : int
    noise_seed : int
    gutter : int
    fullslot : bool
        When *True*, use the Phase 16A full-slot renderer that fills
        every pixel with coherent terrain (no flat-fill background).
        Default *False* for backward compatibility.

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

    if biome_renderers is None:
        biome_renderers = {"forest": ForestRenderer()}

    # Default renderer is the first one
    default_renderer = next(iter(biome_renderers.values())) if biome_renderers else None

    # Optionally use full-slot renderer (Phase 16A)
    if fullslot:
        from .tile_texture import render_detail_texture_fullslot
        _render_ground = render_detail_texture_fullslot
    else:
        _render_ground = render_detail_texture_enhanced

    face_ids = collection.face_ids
    n = len(face_ids)
    if n == 0:
        raise ValueError("No detail grids in the collection")

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
        if tile_density > 0.01 and default_renderer is not None:
            ground_img = Image.open(str(ground_path)).convert("RGB")

            # Get 3D centre for globe-coherent placement
            center_3d = None
            if globe_grid is not None:
                face = globe_grid.faces.get(fid)
                if face is not None:
                    center_3d = face_center_3d(globe_grid.vertices, face)

            featured_img = default_renderer.render(
                ground_img,
                fid,
                tile_density,
                seed=noise_seed + hash(fid) % 100_000,
                globe_3d_center=center_3d,
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
            from .texture_pipeline import _fill_gutter
            _fill_gutter(atlas, slot_x, slot_y, tile_size, gutter)

        inner_x = slot_x + gutter
        inner_y = slot_y + gutter
        u_min = inner_x / atlas_w
        u_max = (inner_x + tile_size) / atlas_w
        v_min = 1.0 - (inner_y + tile_size) / atlas_h
        v_max = 1.0 - inner_y / atlas_h
        uv_layout[fid] = (u_min, v_min, u_max, v_max)

    atlas_path = output_dir / "detail_atlas.png"
    atlas.save(str(atlas_path))

    return atlas_path, uv_layout
