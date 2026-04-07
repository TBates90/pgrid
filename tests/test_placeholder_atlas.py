"""Tests for placeholder_atlas — fast precomputed atlas generation.

These tests intentionally use a very small topology (frequency=2,
tile_size=32) so the first-time artifact build completes in a few seconds.

Tests that require the globe geometry (``models`` library) are guarded by
``needs_models`` and will be skipped in environments where that library
is not installed.
"""
from __future__ import annotations

import hashlib
import io
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    from polygrid.globe.globe import build_globe_grid as _probe
    _probe(2)
    _HAS_MODELS = True
except Exception:
    _HAS_MODELS = False

needs_models = pytest.mark.skipif(
    not _HAS_MODELS,
    reason="models library not installed",
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_spec(
    frequency: int = 2,
    detail_rings: int = 2,
    tile_size: int = 32,
    gutter: int = 2,
    base_color: tuple = (0.133, 0.400, 0.667),
    noise_amount: float = 0.05,
    seed: int = 0,
):
    from polygrid.integration import PlaceholderAtlasSpec

    return PlaceholderAtlasSpec(
        frequency=frequency,
        detail_rings=detail_rings,
        tile_size=tile_size,
        gutter=gutter,
        base_color=base_color,
        noise_amount=noise_amount,
        seed=seed,
    )


# ── PlaceholderAtlasSpec ─────────────────────────────────────────────────────


def test_placeholder_spec_defaults():
    from polygrid.integration import PlaceholderAtlasSpec

    spec = PlaceholderAtlasSpec()
    assert spec.frequency == 2
    assert spec.detail_rings == 2
    assert spec.tile_size == 128
    assert spec.gutter == 4
    assert len(spec.base_color) == 3
    assert 0.0 <= spec.noise_amount <= 1.0


def test_ocean_default_spec():
    from polygrid.integration import ocean_default_spec

    spec = ocean_default_spec(seed=7, frequency=2, tile_size=64)
    assert spec.frequency == 2
    assert spec.tile_size == 64
    assert spec.seed == 7
    # Default water colour is blue-ish
    assert spec.base_color[2] > spec.base_color[0]


def test_rocky_default_spec():
    from polygrid.integration import rocky_default_spec

    spec = rocky_default_spec(seed=3, rock_color=(0.6, 0.5, 0.4))
    assert spec.base_color == (0.6, 0.5, 0.4)
    assert spec.seed == 3


def test_spec_color_override():
    from polygrid.integration import ocean_default_spec

    spec = ocean_default_spec(water_color=(0.1, 0.2, 0.3))
    assert spec.base_color == (0.1, 0.2, 0.3)


@needs_models
def test_generate_placeholder_atlas_emits_seam_strips() -> None:
    from polygrid.placeholder_atlas import generate_placeholder_atlas

    spec = _make_spec(frequency=2, detail_rings=2, tile_size=32, gutter=2)
    result = generate_placeholder_atlas(spec)
    seam_strips = getattr(result, "seam_strips", {})

    assert isinstance(seam_strips, dict)
    metadata = seam_strips.get("metadata") or {}
    seams = seam_strips.get("seams")
    assert isinstance(metadata, dict)
    assert isinstance(seams, list)
    assert int(metadata.get("seam_count", 0) or 0) > 0
    assert len(seams) > 0


# ── _topology_key ────────────────────────────────────────────────────────────


def test_topology_key_deterministic():
    from polygrid.placeholder_atlas import _topology_key

    k1 = _topology_key(2, 2, 128, 4)
    k2 = _topology_key(2, 2, 128, 4)
    assert k1 == k2


def test_topology_key_changes_with_frequency():
    from polygrid.placeholder_atlas import _topology_key

    assert _topology_key(2, 2, 128, 4) != _topology_key(3, 2, 128, 4)


def test_topology_key_changes_with_detail_rings():
    from polygrid.placeholder_atlas import _topology_key

    assert _topology_key(2, 2, 128, 4) != _topology_key(2, 4, 128, 4)


def test_topology_key_changes_with_tile_size():
    from polygrid.placeholder_atlas import _topology_key

    assert _topology_key(2, 2, 128, 4) != _topology_key(2, 2, 256, 4)


def test_topology_key_changes_with_gutter():
    from polygrid.placeholder_atlas import _topology_key

    assert _topology_key(2, 2, 128, 4) != _topology_key(2, 2, 128, 8)


def test_topology_key_doesnt_change_with_color():
    """Color fields must not affect the topology cache key."""
    from polygrid.placeholder_atlas import _topology_key

    k = _topology_key(2, 2, 128, 4)
    # Keys are only (frequency, tile_size, gutter) — same regardless of color.
    assert len(k) == 16  # 16-char hex prefix of sha256


# ── _compute_uv_layout ───────────────────────────────────────────────────────


@needs_models
def test_uv_layout_all_faces_present():
    from polygrid.globe.globe import build_globe_grid
    from polygrid.placeholder_atlas import _compute_uv_layout

    grid = build_globe_grid(2)
    face_ids = sorted(grid.faces.keys())
    uv_layout, atlas_w, atlas_h = _compute_uv_layout(face_ids, tile_size=32, gutter=2)

    assert set(uv_layout.keys()) == set(face_ids)


@needs_models
def test_uv_layout_coords_in_range():
    from polygrid.globe.globe import build_globe_grid
    from polygrid.placeholder_atlas import _compute_uv_layout

    grid = build_globe_grid(2)
    face_ids = sorted(grid.faces.keys())
    uv_layout, atlas_w, atlas_h = _compute_uv_layout(face_ids, tile_size=32, gutter=2)

    for fid, (u_min, v_min, u_max, v_max) in uv_layout.items():
        assert 0.0 <= u_min < u_max <= 1.0, f"{fid}: u out of range"
        assert 0.0 <= v_min < v_max <= 1.0, f"{fid}: v out of range"


@needs_models
def test_uv_layout_atlas_dimensions():
    from polygrid.globe.globe import build_globe_grid
    from polygrid.placeholder_atlas import _compute_uv_layout

    grid = build_globe_grid(2)
    face_ids = sorted(grid.faces.keys())
    tile_size, gutter = 32, 2
    _uv, atlas_w, atlas_h = _compute_uv_layout(face_ids, tile_size=tile_size, gutter=gutter)

    slot_size = tile_size + 2 * gutter
    assert atlas_w % slot_size == 0
    assert atlas_h % slot_size == 0


# ── Artifact build (integration) ─────────────────────────────────────────────


@pytest.mark.slow
def test_build_artifact_shape(tmp_path):
    """Artifact build produces expected array shapes."""
    from polygrid.placeholder_atlas import _build_artifact

    artifact = _build_artifact(frequency=2, detail_rings=2, tile_size=32, gutter=2)

    assert artifact.tile_index_map.dtype == np.uint16
    assert artifact.tile_index_map.ndim == 2
    assert artifact.tile_index_map.shape == (artifact.atlas_height, artifact.atlas_width)
    assert len(artifact.face_ids) == len(artifact.uv_layout)
    assert artifact.vertex_data.ndim == 2
    assert artifact.index_data.ndim == 2


@pytest.mark.slow
def test_artifact_index_map_no_background_after_propagation():
    """After build, all pixels should be assigned to a tile (no background)."""
    from polygrid.placeholder_atlas import _BACKGROUND_IDX, _build_artifact

    artifact = _build_artifact(frequency=2, detail_rings=2, tile_size=32, gutter=2)
    n_bg = int(np.sum(artifact.tile_index_map == _BACKGROUND_IDX))
    assert n_bg == 0, f"{n_bg} background pixels remain after propagation"


@pytest.mark.slow
def test_artifact_save_load_roundtrip(tmp_path):
    """Artifact can be serialised and deserialised losslessly."""
    from polygrid.placeholder_atlas import _build_artifact, load_artifact, save_artifact

    artifact = _build_artifact(frequency=2, detail_rings=2, tile_size=32, gutter=2)
    path = tmp_path / artifact.topology_key
    save_artifact(artifact, path)

    loaded = load_artifact(path)
    assert loaded is not None
    assert loaded.face_ids == artifact.face_ids
    assert loaded.frequency == artifact.frequency
    np.testing.assert_array_equal(loaded.tile_index_map, artifact.tile_index_map)
    np.testing.assert_array_almost_equal(loaded.vertex_data, artifact.vertex_data, decimal=5)


# ── recolor_atlas ────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_recolor_is_deterministic():
    from polygrid.placeholder_atlas import _build_artifact, recolor_atlas
    from polygrid.integration import PlaceholderAtlasSpec

    artifact = _build_artifact(frequency=2, detail_rings=2, tile_size=32, gutter=2)
    spec = PlaceholderAtlasSpec(
        frequency=2, tile_size=32, gutter=2,
        base_color=(0.133, 0.400, 0.667), seed=42,
    )
    png1 = recolor_atlas(artifact, spec)
    png2 = recolor_atlas(artifact, spec)
    assert png1 == png2


@pytest.mark.slow
def test_recolor_respects_base_color():
    """Different base colours produce visually different atlases."""
    from polygrid.placeholder_atlas import _build_artifact, recolor_atlas
    from polygrid.integration import PlaceholderAtlasSpec

    artifact = _build_artifact(frequency=2, detail_rings=2, tile_size=32, gutter=2)

    spec_blue = PlaceholderAtlasSpec(
        frequency=2, tile_size=32, gutter=2,
        base_color=(0.0, 0.0, 1.0), noise_amount=0.0, seed=0,
    )
    spec_red = PlaceholderAtlasSpec(
        frequency=2, tile_size=32, gutter=2,
        base_color=(1.0, 0.0, 0.0), noise_amount=0.0, seed=0,
    )

    from PIL import Image

    img_blue = np.array(Image.open(io.BytesIO(recolor_atlas(artifact, spec_blue))))
    img_red = np.array(Image.open(io.BytesIO(recolor_atlas(artifact, spec_red))))

    # Blue atlas should have blue channel dominant.
    assert img_blue[:, :, 2].mean() > img_blue[:, :, 0].mean()
    # Red atlas should have red channel dominant.
    assert img_red[:, :, 0].mean() > img_red[:, :, 2].mean()
    # The two atlases must differ.
    assert not np.array_equal(img_blue, img_red)


@pytest.mark.slow
def test_recolor_returns_valid_png():
    from polygrid.placeholder_atlas import _build_artifact, recolor_atlas
    from polygrid.integration import PlaceholderAtlasSpec
    from PIL import Image

    artifact = _build_artifact(frequency=2, detail_rings=2, tile_size=32, gutter=2)
    spec = PlaceholderAtlasSpec(frequency=2, tile_size=32, gutter=2)
    png_bytes = recolor_atlas(artifact, spec)

    img = Image.open(io.BytesIO(png_bytes))
    assert img.mode == "RGB"
    assert img.size == (artifact.atlas_width, artifact.atlas_height)


# ── generate_placeholder_atlas (end-to-end) ───────────────────────────────────


@pytest.mark.slow
def test_generate_placeholder_atlas_result_shape():
    from polygrid.placeholder_atlas import generate_placeholder_atlas
    from polygrid.integration import PlaceholderAtlasSpec

    spec = PlaceholderAtlasSpec(frequency=2, tile_size=32, gutter=2)
    result = generate_placeholder_atlas(spec)

    assert isinstance(result.atlas_png, bytes) and len(result.atlas_png) > 0
    assert result.atlas_width > 0
    assert result.atlas_height > 0
    assert result.frequency == 2
    assert result.vertex_data.ndim == 2
    assert result.index_data.ndim == 2
    assert isinstance(result.uv_layout, dict) and len(result.uv_layout) > 0


@pytest.mark.slow
def test_generate_placeholder_atlas_memory_cache_hit():
    """Second call with same topology must return result without rebuilding."""
    from polygrid.placeholder_atlas import (
        _ARTIFACT_CACHE,
        _topology_key,
        generate_placeholder_atlas,
    )
    from polygrid.integration import PlaceholderAtlasSpec

    spec = PlaceholderAtlasSpec(frequency=2, tile_size=32, gutter=2)
    generate_placeholder_atlas(spec)

    key = _topology_key(
        spec.frequency,
        spec.detail_rings,
        spec.tile_size,
        spec.gutter,
    )
    assert key in _ARTIFACT_CACHE, "Artifact should be in memory cache after first call"

    # Second call (different colour, same topology) — must be fast.
    spec2 = PlaceholderAtlasSpec(
        frequency=2, tile_size=32, gutter=2, base_color=(0.9, 0.1, 0.1)
    )
    result = generate_placeholder_atlas(spec2)
    assert result.atlas_width > 0


@pytest.mark.slow
def test_generate_placeholder_atlas_disk_cache(tmp_path, monkeypatch):
    """Artifact is written to disk and loaded on a fresh call."""
    import polygrid.placeholder_atlas as _mod

    monkeypatch.setenv("PGRID_ARTIFACT_CACHE_DIR", str(tmp_path))
    # Clear memory cache to force disk lookup.
    _mod._ARTIFACT_CACHE.clear()

    from polygrid.integration import PlaceholderAtlasSpec

    spec = PlaceholderAtlasSpec(frequency=2, tile_size=32, gutter=2)
    _mod.generate_placeholder_atlas(spec)

    # Check a .npz file was written.
    npz_files = list(tmp_path.glob("*.npz"))
    assert len(npz_files) == 1, f"Expected 1 artifact file, found: {npz_files}"

    # Clear memory cache and call again — should load from disk.
    _mod._ARTIFACT_CACHE.clear()
    result = _mod.generate_placeholder_atlas(spec)
    assert result.atlas_width > 0


# ── integration_atlas public API ──────────────────────────────────────────────


@pytest.mark.slow
def test_integration_atlas_generate_placeholder_atlas():
    """generate_placeholder_atlas is accessible from integration_atlas."""
    from polygrid.integration_atlas import generate_placeholder_atlas
    from polygrid.integration import PlaceholderAtlasSpec

    spec = PlaceholderAtlasSpec(frequency=2, tile_size=32, gutter=2)
    result = generate_placeholder_atlas(spec)
    assert len(result.atlas_png) > 0
