"""Shared atlas-packing utilities.

Small helpers used by multiple modules in the texture pipeline —
extracted here to eliminate duplication.

Functions
---------
- :func:`fill_gutter`          — clamp edge pixels outward into gutter region
- :func:`compute_atlas_layout` — compute grid layout for atlas packing
"""

from __future__ import annotations

import math
from typing import Tuple


def fill_gutter(
    atlas: "Image.Image",
    slot_x: int,
    slot_y: int,
    tile_size: int,
    gutter: int,
) -> None:
    """Fill gutter pixels around a tile slot by clamping edge pixels.

    Copies the outermost pixel rows/columns of the tile outward into
    the gutter region so that bilinear sampling at tile edges sees
    smooth colour instead of adjacent tile data or black.

    Parameters
    ----------
    atlas : PIL.Image.Image
        The atlas image (modified in-place).
    slot_x, slot_y : int
        Top-left pixel of the slot (including gutter).
    tile_size : int
        Inner tile size in pixels (excluding gutter).
    gutter : int
        Gutter width in pixels on each side.
    """
    inner_x = slot_x + gutter
    inner_y = slot_y + gutter

    # Top gutter — repeat top row
    top_strip = atlas.crop((inner_x, inner_y, inner_x + tile_size, inner_y + 1))
    for g in range(gutter):
        atlas.paste(top_strip, (inner_x, slot_y + g))

    # Bottom gutter — repeat bottom row
    bot_y = inner_y + tile_size - 1
    bot_strip = atlas.crop((inner_x, bot_y, inner_x + tile_size, bot_y + 1))
    for g in range(gutter):
        atlas.paste(bot_strip, (inner_x, inner_y + tile_size + g))

    # Left gutter — repeat left column (full height including gutter rows)
    full_top = slot_y
    full_bot = slot_y + tile_size + 2 * gutter
    left_strip = atlas.crop((inner_x, full_top, inner_x + 1, full_bot))
    for g in range(gutter):
        atlas.paste(left_strip, (slot_x + g, full_top))

    # Right gutter — repeat right column (full height including gutter rows)
    right_x = inner_x + tile_size - 1
    right_strip = atlas.crop((right_x, full_top, right_x + 1, full_bot))
    for g in range(gutter):
        atlas.paste(right_strip, (inner_x + tile_size + g, full_top))


def compute_atlas_layout(
    n_tiles: int,
    tile_size: int,
    gutter: int,
    columns: int = 0,
) -> Tuple[int, int, int, int]:
    """Compute atlas dimensions for *n_tiles* in a grid.

    Parameters
    ----------
    n_tiles : int
        Number of tiles to pack.
    tile_size : int
        Inner tile size in pixels.
    gutter : int
        Gutter pixels on each side of a slot.
    columns : int
        Forced column count.  0 = auto (roughly square).

    Returns
    -------
    (columns, rows, atlas_w, atlas_h)
    """
    if columns <= 0:
        columns = max(1, math.isqrt(n_tiles))
        if columns * columns < n_tiles:
            columns += 1
    rows = math.ceil(n_tiles / columns)
    slot_size = tile_size + 2 * gutter
    return columns, rows, columns * slot_size, rows * slot_size
