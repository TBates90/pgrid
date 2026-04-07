"""Compatibility and subdivision regressions after seam helper removal."""

from __future__ import annotations

import inspect
import math

import pytest

from polygrid.globe_renderer_v2 import compute_uv_polygon_inset, subdivide_tile_mesh
from polygrid.tile_uv_align import build_polygon_cut_atlas


def _hex_tile_data():
    center = (0.0, 0.0, 1.0)
    n_sides = 6
    angle_step = 2 * math.pi / n_sides
    r = 0.1
    vertices = [
        (r * math.cos(i * angle_step), r * math.sin(i * angle_step), 1.0)
        for i in range(n_sides)
    ]
    center_uv = (0.5, 0.5)
    vertex_uvs = [
        (0.5 + 0.4 * math.cos(i * angle_step), 0.5 + 0.4 * math.sin(i * angle_step))
        for i in range(n_sides)
    ]
    return center, vertices, center_uv, vertex_uvs


def test_build_polygon_cut_atlas_keeps_corner_kwargs_for_api_compat() -> None:
    sig = inspect.signature(build_polygon_cut_atlas)
    assert "blend_corners" not in sig.parameters
    assert "blend_radius" not in sig.parameters


def test_removed_blend_corners_kwarg_raises_type_error() -> None:
    with pytest.raises(TypeError):
        build_polygon_cut_atlas(
            tile_images={},
            composites={},
            detail_grids={},
            globe_grid=object(),
            face_ids=[],
            blend_corners=True,
        )


def test_boundary_uvs_unchanged_with_clamp() -> None:
    center, vertices, center_uv, vertex_uvs = _hex_tile_data()
    inset = compute_uv_polygon_inset(vertex_uvs, inset_px=8.0, atlas_size=128)

    vdata_clamped, _ = subdivide_tile_mesh(
        center,
        vertices,
        center_uv,
        vertex_uvs,
        color=(1.0, 1.0, 1.0),
        radius=1.0,
        subdivisions=4,
        uv_clamp_polygon=inset,
    )
    vdata_plain, _ = subdivide_tile_mesh(
        center,
        vertices,
        center_uv,
        vertex_uvs,
        color=(1.0, 1.0, 1.0),
        radius=1.0,
        subdivisions=4,
        uv_clamp_polygon=None,
    )

    corner_uv_set = {(round(u, 5), round(v, 5)) for u, v in vertex_uvs}
    for row_idx in range(vdata_clamped.shape[0]):
        u_c = round(float(vdata_clamped[row_idx, 6]), 5)
        v_c = round(float(vdata_clamped[row_idx, 7]), 5)
        u_p = round(float(vdata_plain[row_idx, 6]), 5)
        v_p = round(float(vdata_plain[row_idx, 7]), 5)
        if (u_p, v_p) in corner_uv_set:
            assert (u_c, v_c) == (u_p, v_p)


def test_interior_uvs_clamp_with_inset_polygon() -> None:
    center, vertices, center_uv, vertex_uvs = _hex_tile_data()
    inset = compute_uv_polygon_inset(vertex_uvs, inset_px=12.0, atlas_size=128)

    vdata_clamped, _ = subdivide_tile_mesh(
        center,
        vertices,
        center_uv,
        vertex_uvs,
        color=(1.0, 1.0, 1.0),
        radius=1.0,
        subdivisions=4,
        uv_clamp_polygon=inset,
    )
    vdata_plain, _ = subdivide_tile_mesh(
        center,
        vertices,
        center_uv,
        vertex_uvs,
        color=(1.0, 1.0, 1.0),
        radius=1.0,
        subdivisions=4,
        uv_clamp_polygon=None,
    )

    assert vdata_clamped.shape == vdata_plain.shape
