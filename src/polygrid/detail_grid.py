"""Multi-resolution detail grids for globe tiles.

Each tile on a :class:`GlobeGrid` can be expanded into a local
``PolyGrid`` detail grid — either pentagon-centred (5-sided tile) or
hex (6-sided tile) — giving sub-tile terrain detail that can be
rendered as a texture.

Functions
---------
- :func:`build_detail_grid`  — build a detail grid for one globe face
- :func:`generate_detail_terrain` — terrain gen seeded by parent tile
- :func:`render_detail_texture` — render detail grid to a PNG tile texture
- :func:`build_texture_atlas` — combine per-tile PNGs into an atlas
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .builders import build_pure_hex_grid, build_pentagon_centered_grid, hex_face_count
from .goldberg_topology import goldberg_face_count
from .geometry import face_center
from .models import Face, Vertex
from .polygrid import PolyGrid
from .tile_data import FieldDef, TileDataStore, TileSchema


# ═══════════════════════════════════════════════════════════════════
# 9B.1 — Build detail grid
# ═══════════════════════════════════════════════════════════════════

def build_detail_grid(
    globe_grid: "PolyGrid",
    face_id: str,
    detail_rings: int = 2,
    *,
    size: float = 1.0,
) -> PolyGrid:
    """Build a detail grid for a single globe face.

    The detail grid is a local ``PolyGrid`` — pentagon-centred for
    pentagon tiles, pure-hex for hexagonal tiles — whose metadata
    records its parent face.

    Parameters
    ----------
    globe_grid : PolyGrid (typically GlobeGrid)
        The globe grid that owns the face.
    face_id : str
        The id of the globe face to expand.
    detail_rings : int
        Number of concentric rings around the centre cell.
    size : float
        Hex cell size for the detail grid.

    Returns
    -------
    PolyGrid
        A detail grid with ``metadata["parent_face_id"]``,
        ``metadata["parent_elevation"]`` (if a store is attached later),
        and ``metadata["detail_rings"]``.
    """
    face = globe_grid.faces.get(face_id)
    if face is None:
        raise KeyError(f"Face '{face_id}' not found in globe grid")

    if face.face_type == "pent":
        grid = build_pentagon_centered_grid(detail_rings, size=size)
    else:
        grid = build_pure_hex_grid(detail_rings, size=size)

    # ── Anchor metadata ─────────────────────────────────────────
    grid.metadata["parent_face_id"] = face_id
    grid.metadata["detail_rings"] = detail_rings

    # Copy parent face metadata if available
    if face.metadata:
        for key in ("center_3d", "normal_3d", "latitude_deg",
                     "longitude_deg", "tile_id"):
            if key in face.metadata:
                grid.metadata[f"parent_{key}"] = face.metadata[key]

    return grid


def detail_face_count(face_type: str, rings: int) -> int:
    """Expected face count for a detail grid.

    Parameters
    ----------
    face_type : ``"pent"`` or ``"hex"``
    rings : int

    Returns
    -------
    int
    """
    if face_type == "pent":
        return goldberg_face_count(rings)
    return hex_face_count(rings)


# ═══════════════════════════════════════════════════════════════════
# 9B.3 — Detail terrain generation
# ═══════════════════════════════════════════════════════════════════

def generate_detail_terrain(
    detail_grid: PolyGrid,
    parent_elevation: float,
    *,
    seed: int = 0,
    frequency: float = 4.0,
    octaves: int = 4,
    amplitude: float = 0.15,
    base_weight: float = 0.85,
) -> TileDataStore:
    """Generate terrain on a detail grid, seeded by parent tile elevation.

    The elevation is computed as::

        elevation = parent_elevation * base_weight
                  + noise(x, y) * amplitude * (1 - base_weight)

    So the detail grid inherits most of its elevation from the parent
    tile, with high-frequency local variation layered on top.

    Parameters
    ----------
    detail_grid : PolyGrid
        A detail grid (from :func:`build_detail_grid`).
    parent_elevation : float
        Elevation of the parent globe tile.
    seed : int
        Noise seed.
    frequency : float
        Noise spatial frequency (higher = more detail).
    octaves : int
        Noise octave count.
    amplitude : float
        Amplitude of local variation.
    base_weight : float
        How much the parent elevation dominates (0–1).

    Returns
    -------
    TileDataStore
        Store with ``"elevation"`` field populated.
    """
    from .noise import fbm, _init_noise

    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=detail_grid, schema=schema)

    noise_fn = _init_noise(seed)

    for fid in detail_grid.faces:
        face = detail_grid.faces[fid]
        c = face_center(detail_grid.vertices, face)
        if c is None:
            continue
        cx, cy = c
        # High-frequency local noise
        local_noise = fbm(
            cx, cy,
            octaves=octaves,
            frequency=frequency,
            seed=seed,
        )
        # Blend: parent base + local detail
        elevation = (
            parent_elevation * base_weight
            + local_noise * amplitude * (1.0 - base_weight)
        )
        store.set(fid, "elevation", elevation)

    return store


# ═══════════════════════════════════════════════════════════════════
# 9B.4 — Per-tile texture export
# ═══════════════════════════════════════════════════════════════════

def render_detail_texture(
    detail_grid: PolyGrid,
    store: TileDataStore,
    output_path: Union[str, Path],
    *,
    field_name: str = "elevation",
    ramp: str = "satellite",
    texture_size: int = 128,
    dpi: int = 100,
) -> Path:
    """Render a detail grid to a small PNG texture.

    Produces a square image of the grid coloured by elevation,
    suitable for UV-mapping onto a Goldberg tile surface.

    Parameters
    ----------
    detail_grid : PolyGrid
    store : TileDataStore
    output_path : str or Path
    field_name : str
    ramp : str
    texture_size : int
        Width and height in pixels.
    dpi : int

    Returns
    -------
    Path
        The output file path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    from .terrain_render import elevation_to_overlay

    overlay = elevation_to_overlay(
        detail_grid, store, field_name, ramp=ramp,
    )

    figsize = texture_size / dpi
    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    ax.set_aspect("equal")

    for region in overlay.regions:
        if len(region.points) < 3:
            continue
        fid = region.source_vertex_id
        color = overlay.metadata.get(f"color_{fid}", (0.5, 0.5, 0.5))
        poly = MplPolygon(
            region.points, closed=True, facecolor=color,
            edgecolor=color, linewidth=0.3,
        )
        ax.add_patch(poly)

    # Autofit
    xs = [v.x for v in detail_grid.vertices.values() if v.has_position()]
    ys = [v.y for v in detail_grid.vertices.values() if v.has_position()]
    if xs and ys:
        pad = 0.1 * max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════
# 9B.5 — Texture atlas
# ═══════════════════════════════════════════════════════════════════

def build_texture_atlas(
    texture_paths: Sequence[Path],
    output_path: Union[str, Path],
    *,
    tile_size: int = 128,
    columns: Optional[int] = None,
) -> Tuple[Path, Dict[str, Tuple[int, int, int, int]]]:
    """Combine per-tile PNG textures into a single atlas image.

    Parameters
    ----------
    texture_paths : sequence of Path
        Ordered list of tile texture PNGs.  The filename (stem) is used
        as the tile key in the returned layout dict.
    output_path : str or Path
        Output atlas PNG path.
    tile_size : int
        Width/height of each tile slot in the atlas (pixels).
    columns : int, optional
        Number of columns.  Defaults to ``ceil(sqrt(N))``.

    Returns
    -------
    (Path, dict)
        ``(atlas_path, layout)`` where *layout* maps tile key →
        ``(x, y, width, height)`` pixel rect in the atlas.
    """
    import numpy as np
    from PIL import Image

    n = len(texture_paths)
    if n == 0:
        raise ValueError("No texture paths provided")

    cols = columns or math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    atlas_w = cols * tile_size
    atlas_h = rows * tile_size
    atlas = Image.new("RGBA", (atlas_w, atlas_h), (0, 0, 0, 0))

    layout: Dict[str, Tuple[int, int, int, int]] = {}
    for idx, tex_path in enumerate(texture_paths):
        col = idx % cols
        row = idx // cols
        x = col * tile_size
        y = row * tile_size

        tile_img = Image.open(tex_path).convert("RGBA")
        tile_img = tile_img.resize((tile_size, tile_size), Image.LANCZOS)
        atlas.paste(tile_img, (x, y))

        key = tex_path.stem
        layout[key] = (x, y, tile_size, tile_size)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(out)
    return out, layout
