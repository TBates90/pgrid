"""Orientation-only seam policy regression tests.

These tests validate the public contract after removing seam post-processing
helpers from the production atlas path.
"""

from __future__ import annotations

import inspect

import pytest
from PIL import Image

from polygrid.tile_uv_align import build_polygon_cut_atlas, warp_tile_to_uv


def test_build_polygon_cut_atlas_keeps_seam_kwargs_for_api_compat() -> None:
    sig = inspect.signature(build_polygon_cut_atlas)
    assert "stitch_seams" not in sig.parameters
    assert "stitch_width" not in sig.parameters


def test_removed_stitch_seams_kwarg_raises_type_error() -> None:
    with pytest.raises(TypeError):
        build_polygon_cut_atlas(
            tile_images={},
            composites={},
            detail_grids={},
            globe_grid=object(),
            face_ids=[],
            stitch_seams=True,
        )


def test_warp_requires_piecewise_inputs() -> None:
    img = Image.new("RGB", (16, 16), (255, 0, 255))
    with pytest.raises(ValueError, match="requires grid_corners"):
        warp_tile_to_uv(
            img,
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            output_size=16,
            grid_corners=None,
            uv_corners=None,
            tile_size=None,
        )


def test_warp_replaces_all_invalid_fallback_samples() -> None:
    img = Image.new("RGB", (32, 32), (10, 20, 30))
    grid_corners = [
        (0.8, 0.5),
        (0.65, 0.75),
        (0.35, 0.75),
        (0.2, 0.5),
        (0.35, 0.25),
        (0.65, 0.25),
    ]
    uv_corners = [
        (0.82, 0.5),
        (0.66, 0.77),
        (0.34, 0.77),
        (0.18, 0.5),
        (0.34, 0.23),
        (0.66, 0.23),
    ]

    warped = warp_tile_to_uv(
        img,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        output_size=32,
        grid_corners=grid_corners,
        uv_corners=uv_corners,
        tile_size=24,
        gutter=4,
        dilate_cval=False,
    )

    arr = list(warped.getdata())
    assert (128, 128, 128) not in arr
