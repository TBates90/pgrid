"""Texture atlas pipeline and UV mapping for Goldberg globe tiles.

Renders all detail grids, assembles them into a texture atlas, and
computes UV coordinates that map each globe tile's 3D vertices into
the correct atlas slot.

Functions
---------
- :func:`build_detail_atlas` — render + assemble texture atlas
- :func:`compute_tile_uvs` — per-tile UV mapping into atlas
- :func:`build_textured_tile_mesh` — per-tile mesh with atlas UVs
- :func:`build_textured_globe_meshes` — batch textured meshes
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .atlas_utils import fill_gutter, compute_atlas_layout
from .detail_render import BiomeConfig, render_detail_texture_enhanced
from .detail_terrain import generate_all_detail_terrain
from .tile_detail import TileDetailSpec, DetailGridCollection
from .polygrid import PolyGrid
from .tile_data import TileDataStore

try:
    from models.core import ShapeMesh, VertexAttribute
    from models.objects.goldberg import generate_goldberg_tiles
    from array import array as typed_array
    _HAS_MODELS = True
except ImportError:
    _HAS_MODELS = False


# ═══════════════════════════════════════════════════════════════════
# 10D.1 — Build detail atlas
# ═══════════════════════════════════════════════════════════════════

def build_detail_atlas(
    collection: DetailGridCollection,
    biome: Optional[BiomeConfig] = None,
    output_dir: Path | str = Path("exports/detail_tiles"),
    *,
    tile_size: int = 256,
    columns: int = 0,
    noise_seed: int = 0,
    gutter: int = 4,
) -> Tuple[Path, Dict[str, Tuple[float, float, float, float]]]:
    """Render every detail grid to a texture and assemble an atlas.

    Parameters
    ----------
    collection : DetailGridCollection
        Must already have stores populated (call
        ``generate_all_detail_terrain`` first).
    biome : BiomeConfig, optional
        Rendering configuration.
    output_dir : Path or str
        Directory for individual tile textures and the atlas.
    tile_size : int
        Side length of each tile texture in pixels.
    columns : int
        Atlas columns.  0 = auto (roughly square).
    noise_seed : int
        Seed for overlay noise.
    gutter : int
        Padding pixels around each tile slot.  Filled by clamping
        tile edge pixels outward, preventing bilinear bleed across
        slot boundaries.

    Returns
    -------
    tuple
        ``(atlas_path, uv_layout)`` where *uv_layout* maps
        ``face_id → (u_min, v_min, u_max, v_max)`` in atlas UV space.
    """
    from PIL import Image

    if biome is None:
        biome = BiomeConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    face_ids = collection.face_ids
    n = len(face_ids)
    if n == 0:
        raise ValueError("No detail grids in the collection")

    # Render individual tiles
    tile_paths: Dict[str, Path] = {}
    for fid in face_ids:
        grid, store = collection.get(fid)
        if store is None:
            raise ValueError(
                f"No terrain store for face '{fid}' — "
                "call generate_all_detail_terrain first"
            )
        path = output_dir / f"tile_{fid}.png"
        render_detail_texture_enhanced(
            grid, store, path, biome,
            tile_size=tile_size,
            noise_seed=noise_seed + hash(fid) % 10000,
        )
        tile_paths[fid] = path

    # Compute atlas layout — each slot is tile_size + 2*gutter
    columns, rows, atlas_w, atlas_h = compute_atlas_layout(
        n, tile_size, gutter, columns=columns,
    )

    slot_size = tile_size + 2 * gutter
    atlas = Image.new("RGB", (atlas_w, atlas_h), (128, 128, 128))

    uv_layout: Dict[str, Tuple[float, float, float, float]] = {}

    for idx, fid in enumerate(face_ids):
        col = idx % columns
        row = idx // columns
        # Pixel position of this slot (including gutter)
        slot_x = col * slot_size
        slot_y = row * slot_size

        tile_img = Image.open(tile_paths[fid]).convert("RGB")
        tile_img = tile_img.resize((tile_size, tile_size), Image.LANCZOS)

        # Paste the tile into the center of the slot
        atlas.paste(tile_img, (slot_x + gutter, slot_y + gutter))

        # Fill gutter by clamping edge pixels outward
        if gutter > 0:
            fill_gutter(atlas, slot_x, slot_y, tile_size, gutter)

        # UV coordinates map to the inner (non-gutter) tile region
        inner_x = slot_x + gutter
        inner_y = slot_y + gutter
        u_min = inner_x / atlas_w
        u_max = (inner_x + tile_size) / atlas_w
        # Image y increases downward, UV v increases upward
        v_min = 1.0 - (inner_y + tile_size) / atlas_h
        v_max = 1.0 - inner_y / atlas_h
        uv_layout[fid] = (u_min, v_min, u_max, v_max)

    atlas_path = output_dir / "detail_atlas.png"
    atlas.save(str(atlas_path))

    return atlas_path, uv_layout


# ═══════════════════════════════════════════════════════════════════
# 10D.2 — Compute tile UVs
# ═══════════════════════════════════════════════════════════════════

def compute_tile_uvs(
    tile_uv_vertices: List[Tuple[float, float]],
    atlas_slot: Tuple[float, float, float, float],
) -> List[Tuple[float, float]]:
    """Map a tile's normalised UV vertices into an atlas slot.

    Parameters
    ----------
    tile_uv_vertices : list of (u, v)
        Normalised 2D coordinates from the models library's
        ``GoldbergTile.uv_vertices`` (in [0, 1] space).
    atlas_slot : (u_min, v_min, u_max, v_max)
        The tile's rectangle in atlas UV space.

    Returns
    -------
    list of (u, v)
        Remapped UVs pointing into the atlas.
    """
    u_min, v_min, u_max, v_max = atlas_slot
    u_span = u_max - u_min
    v_span = v_max - v_min

    result = []
    for u, v in tile_uv_vertices:
        # Clamp to [0, 1] in case the tile UVs are slightly outside
        u_clamped = max(0.0, min(1.0, u))
        v_clamped = max(0.0, min(1.0, v))
        result.append((
            u_min + u_clamped * u_span,
            v_min + v_clamped * v_span,
        ))
    return result


# ═══════════════════════════════════════════════════════════════════
# 10D.3 — Build a textured per-tile mesh
# ═══════════════════════════════════════════════════════════════════

def build_textured_tile_mesh(
    tile,
    atlas_slot: Tuple[float, float, float, float],
) -> "ShapeMesh":
    """Build a per-tile triangle-fan mesh with atlas UVs.

    The mesh has stride=32: position(3) + color(3) + uv(2).
    Colour is set to white (1, 1, 1) so the texture provides all
    colour information via the fragment shader.

    Parameters
    ----------
    tile : GoldbergTile
        A tile from :func:`models.objects.goldberg.generate_goldberg_tiles`.
        Must have ``uv_vertices``, ``center``, and ``vertices`` attributes.
    atlas_slot : (u_min, v_min, u_max, v_max)
        This tile's slot in the atlas UV space.

    Returns
    -------
    ShapeMesh
    """
    if not _HAS_MODELS:
        raise ImportError("models library required for textured meshes")

    positions = [tile.center]
    positions.extend(tile.vertices)

    white = (1.0, 1.0, 1.0)
    colors = [white for _ in positions]

    # Map tile UVs into atlas slot
    mapped = compute_tile_uvs(list(tile.uv_vertices), atlas_slot)
    center_u = sum(uv[0] for uv in mapped) / len(mapped)
    center_v = sum(uv[1] for uv in mapped) / len(mapped)
    uvs = [(center_u, center_v), *mapped]

    vertex_data = typed_array("f")
    for pos, col, uv in zip(positions, colors, uvs):
        vertex_data.extend(pos)
        vertex_data.extend(col)
        vertex_data.extend(uv)

    indices = []
    n = len(positions)
    for i in range(1, n):
        nxt = 1 if i == n - 1 else i + 1
        indices.extend([0, i, nxt])
    index_data = typed_array("I", indices)

    return ShapeMesh(
        vertex_data=vertex_data,
        index_data=index_data,
        stride=8 * 4,
        attributes=(
            VertexAttribute(name="position", location=0, components=3, offset=0),
            VertexAttribute(name="color", location=1, components=3, offset=3 * 4),
            VertexAttribute(name="uv", location=2, components=2, offset=6 * 4),
        ),
    )


# ═══════════════════════════════════════════════════════════════════
# 10D.4 — Build textured meshes for all tiles
# ═══════════════════════════════════════════════════════════════════

def build_textured_globe_meshes(
    frequency: int,
    uv_layout: Dict[str, Tuple[float, float, float, float]],
    *,
    radius: float = 1.0,
) -> List["ShapeMesh"]:
    """Build textured per-tile meshes for every Goldberg tile.

    Parameters
    ----------
    frequency : int
        Goldberg subdivision frequency.
    uv_layout : dict
        ``{face_id: (u_min, v_min, u_max, v_max)}`` from
        :func:`build_detail_atlas`.
    radius : float
        Globe radius.

    Returns
    -------
    list of ShapeMesh
        One mesh per tile, with white vertex colour and atlas UVs.
    """
    if not _HAS_MODELS:
        raise ImportError("models library required for textured meshes")

    tiles = generate_goldberg_tiles(frequency=frequency, radius=radius)
    meshes = []
    for tile in tiles:
        fid = f"t{tile.index}"
        if fid not in uv_layout:
            continue
        mesh = build_textured_tile_mesh(tile, uv_layout[fid])
        meshes.append(mesh)
    return meshes
