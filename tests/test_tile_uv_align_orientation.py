from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest

from polygrid.builders import build_pentagon_centered_grid
from polygrid.globe import _HAS_MODELS, build_globe_grid
from polygrid.rendering.uv_texture import get_tile_uv_vertices

from polygrid.rendering.tile_uv_align import (
    _apply_pentagon_uv_adjustments,
    _augment_ordered_fan_with_edge_controls,
    _blend_corner_sets,
    _compute_piecewise_warp_map,
    _compute_edge_profile_mismatch_metrics,
    _compute_pentagon_grid_scale,
    _equalise_sector_ratios,
    _is_hex_adjacent_to_pentagon,
    _PENTAGON_GRID_SCALE_MIN,
    _pentagon_smoothing_alpha_for_frequency,
    _should_use_pent_reflection,
    _stabilize_pentagon_scale_for_frequency,
    _normalize_ordered_pentagon_winding,
    _normalize_pentagon_winding_for_warp,
    _select_corner_match_indices,
    _winding_sign,
    build_polygon_cut_atlas,
    compute_gt_to_pg_corner_map,
    compute_pg_to_gt_edge_map,
    get_macro_edge_corners,
    match_grid_corners_to_uv,
)


def _regular_pentagon(radius: float = 1.0) -> list[tuple[float, float]]:
    return [
        (radius * math.cos(2.0 * math.pi * i / 5.0), radius * math.sin(2.0 * math.pi * i / 5.0))
        for i in range(5)
    ]


def test_normalize_ordered_pentagon_winding_flips_source_when_mismatched() -> None:
    dst = np.asarray(_regular_pentagon(), dtype=np.float64)
    src = dst[::-1].copy()

    fixed, changed = _normalize_ordered_pentagon_winding(src, dst)

    assert changed is True
    assert _winding_sign(fixed) == _winding_sign(dst)


def test_normalize_pentagon_winding_for_warp_preserves_anchor_corner() -> None:
    grid = _regular_pentagon(radius=0.6)
    uv = list(reversed(_regular_pentagon(radius=0.45)))

    fixed, changed = _normalize_pentagon_winding_for_warp(
        grid,
        uv,
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        img_w=128,
        img_h=128,
        tile_size=120,
        gutter=4,
        face_id="t0",
    )

    assert changed is True
    assert np.allclose(np.asarray(fixed[0]), np.asarray(grid[0]), atol=1e-9)


def test_select_corner_match_indices_can_force_rotation_only() -> None:
    n = 5
    macro = np.linspace(-math.pi, math.pi, num=n, endpoint=False)
    # Reflection is exact while rotation is imperfect.
    gt_reflected = macro[::-1].copy()

    indices, mode, rot_err, ref_err, _rot, _ref = _select_corner_match_indices(
        macro,
        gt_reflected,
        allow_reflection=False,
    )

    assert mode == "rotation"
    assert len(set(indices)) == n
    assert rot_err < float("inf")
    assert ref_err == float("inf")


def test_should_use_pent_reflection_requires_decisive_improvement() -> None:
    assert _should_use_pent_reflection(1.0, 0.9) is False
    assert _should_use_pent_reflection(1.0, 0.8) is True


def test_edge_profile_mismatch_metrics_reverse_pairing() -> None:
    profile_a = np.asarray(
        [
            (10.0, 20.0, 30.0),
            (40.0, 50.0, 60.0),
            (70.0, 80.0, 90.0),
        ],
        dtype=np.float64,
    )
    profile_b = profile_a[::-1].copy()

    metrics = _compute_edge_profile_mismatch_metrics(profile_a, profile_b)

    assert metrics["endpoint_0_mismatch"] == 0.0
    assert metrics["endpoint_1_mismatch"] == 0.0
    assert metrics["midpoint_mismatch"] == 0.0
    assert metrics["max_sampled_offset"] == 0.0


def test_piecewise_warp_map_rejects_degenerate_pentagon_sectors() -> None:
    # Force one zero-area fan sector by duplicating an adjacent source corner.
    grid = _regular_pentagon(radius=0.5)
    grid[1] = grid[0]
    uv = _regular_pentagon(radius=0.45)

    with np.testing.assert_raises_regex(ValueError, "Degenerate pentagon warp sector"):
        _compute_piecewise_warp_map(
            grid,
            uv,
            tile_size=120,
            gutter=4,
            img_w=128,
            img_h=128,
            xlim=(-1.0, 1.0),
            ylim=(-1.0, 1.0),
            output_size=128,
        )


def test_build_polygon_cut_atlas_pent_uv_rotation_default_is_neutral() -> None:
    sig = inspect.signature(build_polygon_cut_atlas)
    assert "pent_uv_rotation" in sig.parameters
    assert sig.parameters["pent_uv_rotation"].default == 0.0


def test_default_pentagon_uv_adjustments_are_noop() -> None:
    uv = _regular_pentagon(radius=0.42)
    adjusted = _apply_pentagon_uv_adjustments(
        uv,
        tile_size=256,
        pent_uv_scale=1.0,
        pent_uv_rotation=0.0,
        pent_uv_x=0.0,
        pent_uv_y=0.0,
    )

    assert np.allclose(np.asarray(adjusted), np.asarray(uv), atol=1e-12)


def test_pentagon_scale_floor_is_neutral() -> None:
    assert _PENTAGON_GRID_SCALE_MIN == 1.0


def test_compute_pentagon_grid_scale_without_hex_neighbours_is_neutral_floor() -> None:
    pent = _regular_pentagon(radius=0.5)
    uv = _regular_pentagon(radius=0.45)
    scale = _compute_pentagon_grid_scale(
        pent,
        uv,
        hex_neighbours_data=[],
        tile_size=128,
        gutter=4,
    )
    assert abs(scale - 1.0) < 1e-12


def test_stabilize_pentagon_scale_caps_freq_two_expansion() -> None:
    assert abs(_stabilize_pentagon_scale_for_frequency(1.12, frequency=2) - 1.0) < 1e-12


def test_stabilize_pentagon_scale_keeps_freq_three_value() -> None:
    assert abs(_stabilize_pentagon_scale_for_frequency(1.12, frequency=3) - 1.12) < 1e-12


def test_pentagon_smoothing_alpha_reduced_for_freq_two() -> None:
    assert abs(_pentagon_smoothing_alpha_for_frequency(2) - 0.0) < 1e-12


def test_pentagon_smoothing_alpha_full_for_freq_three() -> None:
    assert abs(_pentagon_smoothing_alpha_for_frequency(3) - 1.0) < 1e-12


def test_blend_corner_sets_interpolates_linearly() -> None:
    base = [(0.0, 0.0), (2.0, 2.0)]
    adj = [(2.0, 0.0), (2.0, 0.0)]
    blended = _blend_corner_sets(base, adj, 0.5)
    assert np.allclose(np.asarray(blended), np.asarray([(1.0, 0.0), (2.0, 1.0)]), atol=1e-12)


def test_augment_ordered_fan_with_edge_controls_noop_when_pull_zero() -> None:
    src = np.asarray(_regular_pentagon(radius=2.0), dtype=np.float64)
    dst = np.asarray(_regular_pentagon(radius=1.0), dtype=np.float64)
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)

    src_aug, dst_aug = _augment_ordered_fan_with_edge_controls(
        src,
        dst,
        src_c,
        dst_c,
        edge_interior_pull=0.0,
    )

    assert np.allclose(src_aug, src, atol=1e-12)
    assert np.allclose(dst_aug, dst, atol=1e-12)


def test_augment_ordered_fan_with_edge_controls_preserves_corner_anchors() -> None:
    src = np.asarray(_regular_pentagon(radius=2.0), dtype=np.float64)
    dst = np.asarray(_regular_pentagon(radius=1.0), dtype=np.float64)
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)

    src_aug, dst_aug = _augment_ordered_fan_with_edge_controls(
        src,
        dst,
        src_c,
        dst_c,
        edge_interior_pull=0.4,
    )

    assert src_aug.shape[0] == 10
    assert dst_aug.shape[0] == 10
    assert np.allclose(src_aug[::2], src, atol=1e-12)
    assert np.allclose(dst_aug[::2], dst, atol=1e-12)


def test_is_hex_adjacent_to_pentagon_true_for_mixed_neighbours() -> None:
    grid = SimpleNamespace(
        faces={
            "h0": SimpleNamespace(face_type="hex", neighbor_ids=["h1", "p0"]),
            "h1": SimpleNamespace(face_type="hex", neighbor_ids=["h0"]),
            "p0": SimpleNamespace(face_type="pent", neighbor_ids=["h0"]),
        }
    )
    assert _is_hex_adjacent_to_pentagon(grid, "h0") is True


def test_is_hex_adjacent_to_pentagon_false_for_non_hex_or_missing() -> None:
    grid = SimpleNamespace(
        faces={
            "h0": SimpleNamespace(face_type="hex", neighbor_ids=["h1"]),
            "h1": SimpleNamespace(face_type="hex", neighbor_ids=["h0"]),
            "p0": SimpleNamespace(face_type="pent", neighbor_ids=["h0"]),
        }
    )
    assert _is_hex_adjacent_to_pentagon(grid, "h0") is False
    assert _is_hex_adjacent_to_pentagon(grid, "p0") is False
    assert _is_hex_adjacent_to_pentagon(grid, "missing") is False


def test_equalised_sector_angles_follow_destination_with_y_flip() -> None:
    # Asymmetric pentagon to exercise non-uniform sector geometry.
    grid = [
        (0.0, 0.9),
        (0.85, 0.3),
        (0.5, -0.7),
        (-0.45, -0.75),
        (-0.95, 0.25),
    ]
    uv = [
        (0.52, 0.95),
        (0.93, 0.62),
        (0.76, 0.09),
        (0.25, 0.05),
        (0.06, 0.58),
    ]

    eq, _ = _equalise_sector_ratios(grid, uv, tile_size=256, gutter=4)

    gc = np.asarray(eq, dtype=np.float64)
    gc_c = gc.mean(axis=0)

    dst = np.asarray(uv, dtype=np.float64)
    dst_px = np.empty_like(dst)
    for i in range(len(uv)):
        dst_px[i, 0] = 4 + dst[i, 0] * 256
        dst_px[i, 1] = 4 + (1.0 - dst[i, 1]) * 256
    dst_c = dst_px.mean(axis=0)

    order = np.argsort(np.arctan2(dst_px[:, 1] - dst_c[1], dst_px[:, 0] - dst_c[0]))

    eq_sorted = gc[order]
    eq_angles = np.unwrap(np.arctan2(eq_sorted[:, 1] - gc_c[1], eq_sorted[:, 0] - gc_c[0]))

    dst_sorted = dst_px[order]
    dst_angles = np.unwrap(-np.arctan2(dst_sorted[:, 1] - dst_c[1], dst_sorted[:, 0] - dst_c[0]))

    # Angles should match up to a global 2pi-unwrapped shift.
    angle_delta = eq_angles - dst_angles
    assert np.max(np.abs(angle_delta - angle_delta[0])) < 1e-8


def test_equalise_sector_ratios_can_keep_uniform_radii() -> None:
    grid = [
        (0.0, 0.9),
        (0.85, 0.3),
        (0.5, -0.7),
        (-0.45, -0.75),
        (-0.95, 0.25),
    ]
    uv = [
        (0.52, 0.95),
        (0.93, 0.62),
        (0.76, 0.09),
        (0.25, 0.05),
        (0.06, 0.58),
    ]

    eq, _ = _equalise_sector_ratios(
        grid,
        uv,
        tile_size=256,
        gutter=4,
        match_radii=False,
    )

    arr = np.asarray(eq, dtype=np.float64)
    src_c = np.asarray(grid, dtype=np.float64).mean(axis=0)
    radii = np.linalg.norm(arr - src_c, axis=1)
    assert np.max(radii) - np.min(radii) < 1e-8


@pytest.mark.needs_models
@pytest.mark.skipif(not _HAS_MODELS, reason="models library not installed")
def test_all_pentagons_use_rotation_signature_and_valid_warp_map() -> None:
    globe = build_globe_grid(3)
    pent_ids = [fid for fid, face in globe.faces.items() if face.face_type == "pent"]
    assert len(pent_ids) == 12

    for fid in pent_ids:
        detail = build_pentagon_centered_grid(rings=2)
        detail.compute_macro_edges(
            n_sides=5,
            corner_ids=detail.metadata.get("corner_vertex_ids"),
        )

        grid_raw = get_macro_edge_corners(detail, 5)
        grid_matched = match_grid_corners_to_uv(grid_raw, globe, fid)
        uv_corners = get_tile_uv_vertices(globe, fid)

        # Assert the matched order is a pure cyclic rotation of raw corners.
        index_of = {corner: i for i, corner in enumerate(grid_raw)}
        indices = [index_of[corner] for corner in grid_matched]
        offsets = {(indices[k] - k) % 5 for k in range(5)}
        assert len(offsets) == 1, f"{fid}: non-rotation corner mapping {indices}"

        xs = [v.x for v in detail.vertices.values() if v.has_position()]
        ys = [v.y for v in detail.vertices.values() if v.has_position()]
        xlim = (min(xs), max(xs))
        ylim = (min(ys), max(ys))

        # This should never raise if corner pairing preserves pentagon orientation.
        _compute_piecewise_warp_map(
            grid_matched,
            uv_corners,
            tile_size=256,
            gutter=4,
            img_w=512,
            img_h=512,
            xlim=xlim,
            ylim=ylim,
            output_size=264,
        )


@pytest.mark.needs_models
@pytest.mark.skipif(not _HAS_MODELS, reason="models library not installed")
def test_all_pentagons_orientation_signature_report_is_stable() -> None:
    globe = build_globe_grid(3)
    pent_ids = sorted(
        fid for fid, face in globe.faces.items() if face.face_type == "pent"
    )
    assert len(pent_ids) == 12

    signatures: list[tuple[str, int, int, int]] = []

    for fid in pent_ids:
        detail = build_pentagon_centered_grid(rings=2)
        detail.compute_macro_edges(
            n_sides=5,
            corner_ids=detail.metadata.get("corner_vertex_ids"),
        )
        grid_raw = get_macro_edge_corners(detail, 5)
        grid_matched = match_grid_corners_to_uv(grid_raw, globe, fid)
        uv_corners = get_tile_uv_vertices(globe, fid)

        index_of = {corner: i for i, corner in enumerate(grid_raw)}
        indices = [index_of[corner] for corner in grid_matched]
        offsets = {(indices[k] - k) % 5 for k in range(5)}
        assert len(offsets) == 1, f"{fid}: non-rotation corner mapping {indices}"
        offset = next(iter(offsets))

        xs = [v.x for v in detail.vertices.values() if v.has_position()]
        ys = [v.y for v in detail.vertices.values() if v.has_position()]
        xlim = (min(xs), max(xs))
        ylim = (min(ys), max(ys))
        x_span = xlim[1] - xlim[0]
        y_span = ylim[1] - ylim[0]

        src = np.asarray(grid_matched, dtype=np.float64)
        src_px = np.empty_like(src)
        for i in range(5):
            src_px[i, 0] = (src[i, 0] - xlim[0]) / x_span * 512
            src_px[i, 1] = (1.0 - (src[i, 1] - ylim[0]) / y_span) * 512

        dst = np.asarray(uv_corners, dtype=np.float64)
        dst_px = np.empty_like(dst)
        for i in range(5):
            dst_px[i, 0] = 4 + dst[i, 0] * 256
            dst_px[i, 1] = 4 + (1.0 - dst[i, 1]) * 256

        src_sign = _winding_sign(src_px)
        dst_sign = _winding_sign(dst_px)
        assert src_sign != 0 and dst_sign != 0, f"{fid}: degenerate winding signature"
        assert src_sign == dst_sign, (
            f"{fid}: winding parity mismatch src_sign={src_sign} dst_sign={dst_sign}"
        )

        signatures.append((fid, offset, src_sign, dst_sign))

    # Compact deterministic signature artifact for regression checks.
    assert len(set(signatures)) == 12


@pytest.mark.needs_models
@pytest.mark.skipif(not _HAS_MODELS, reason="models library not installed")
def test_compute_gt_to_pg_corner_map_is_bijective_for_all_faces() -> None:
    globe = build_globe_grid(2)
    for fid, face in globe.faces.items():
        mapping = compute_gt_to_pg_corner_map(globe, fid)
        n = len(face.vertex_ids)
        assert set(mapping.keys()) == set(range(n))
        assert set(mapping.values()) == set(range(n))


@pytest.mark.needs_models
@pytest.mark.skipif(not _HAS_MODELS, reason="models library not installed")
def test_compute_pg_to_gt_edge_map_is_bijective_for_all_faces() -> None:
    globe = build_globe_grid(2)
    for fid, face in globe.faces.items():
        pg_to_gt, gt_to_pg = compute_pg_to_gt_edge_map(globe, fid)
        n = len(face.vertex_ids)
        assert set(pg_to_gt.keys()) == set(range(n))
        assert set(pg_to_gt.values()) == set(range(n))
        assert set(gt_to_pg.keys()) == set(range(n))
        assert set(gt_to_pg.values()) == set(range(n))
