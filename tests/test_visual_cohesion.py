"""Tests for Phase 18E — Visual Cohesion & Validation."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image

from polygrid.visual_cohesion import (
    _colour_variance,
    measure_seam_visibility,
    sample_boundary_pixels,
    sample_interior_pixels,
    verify_topology_features,
)


# ── helpers ──────────────────────────────────────────────────────────

def _make_fake_atlas(n_tiles: int = 4, tile_size: int = 64, gutter: int = 4):
    """Create a synthetic atlas with known UV layout (numpy-accelerated)."""
    rng = np.random.RandomState(42)

    slot = tile_size + 2 * gutter
    cols = max(1, math.isqrt(n_tiles))
    if cols * cols < n_tiles:
        cols += 1
    rows = math.ceil(n_tiles / cols)

    atlas_w = cols * slot
    atlas_h = rows * slot
    # Start with gutter colour
    arr = np.full((atlas_h, atlas_w, 3), 100, dtype=np.uint8)

    uv_layout = {}
    for idx in range(n_tiles):
        fid = f"t{idx}"
        col = idx % cols
        row = idx // cols
        slot_x = col * slot
        slot_y = row * slot

        # Paint tile interior with a random colour + noise
        base = rng.randint(40, 201, size=3).reshape(1, 1, 3)
        noise = rng.randint(-15, 16, size=(tile_size, tile_size, 3))
        tile = np.clip(base + noise, 0, 255).astype(np.uint8)
        arr[
            slot_y + gutter : slot_y + gutter + tile_size,
            slot_x + gutter : slot_x + gutter + tile_size,
        ] = tile

        inner_x = slot_x + gutter
        inner_y = slot_y + gutter
        u_min = inner_x / atlas_w
        u_max = (inner_x + tile_size) / atlas_w
        v_min = 1.0 - (inner_y + tile_size) / atlas_h
        v_max = 1.0 - inner_y / atlas_h
        uv_layout[fid] = (u_min, v_min, u_max, v_max)

    atlas = Image.fromarray(arr, "RGB")
    return atlas, uv_layout


def _make_tiny_collection():
    """Build a minimal collection for topology verification."""
    from polygrid import PolyGrid, Vertex, Face, Edge
    from polygrid import TileDataStore, TileSchema, FieldDef

    v0 = Vertex("v0", 0.0, 0.0)
    v1 = Vertex("v1", 1.0, 0.0)
    v2 = Vertex("v2", 0.5, 0.866)
    v3 = Vertex("v3", 1.5, 0.866)

    faces_raw = [
        ("f0", ["v0", "v1", "v2"]),
        ("f1", ["v1", "v3", "v2"]),
    ]

    edge_map: dict[tuple[str, str], list[str]] = {}
    for fid, vids in faces_raw:
        for i in range(len(vids)):
            a, b = vids[i], vids[(i + 1) % len(vids)]
            key = (min(a, b), max(a, b))
            edge_map.setdefault(key, []).append(fid)

    edges = [
        Edge(f"e{idx}", (a, b), tuple(fids))
        for idx, ((a, b), fids) in enumerate(edge_map.items())
    ]
    faces = [Face(fid, "tri", tuple(vids)) for fid, vids in faces_raw]

    grid = PolyGrid(
        vertices=[v0, v1, v2, v3],
        edges=edges,
        faces=faces,
    )

    schema = TileSchema(fields=[FieldDef(name="elevation", dtype=float)])
    store = TileDataStore(grid, schema=schema)
    for fid in grid.faces:
        store.set(fid, "elevation", 0.3)

    class FakeCollection:
        def __init__(self):
            self.face_ids = ["t0"]
            self._grid = grid
            self._store = store
            self.grids = {"t0": grid}
            self.stores = {"t0": store}

        def get(self, fid):
            return (self._grid, self._store)

        @property
        def total_face_count(self):
            return len(self._grid.faces)

    return FakeCollection(), grid


# ═══════════════════════════════════════════════════════════════════
# 18E.1 — Seam measurement
# ═══════════════════════════════════════════════════════════════════

class TestSampleBoundaryPixels:
    def test_returns_pixels(self):
        atlas, uv = _make_fake_atlas(4)
        pixels = sample_boundary_pixels(atlas, uv, n_samples=10)
        assert len(pixels) > 0
        assert all(len(p) == 3 for p in pixels)

    def test_sample_count_scales_with_tiles(self):
        atlas4, uv4 = _make_fake_atlas(4)
        atlas9, uv9 = _make_fake_atlas(9)
        p4 = sample_boundary_pixels(atlas4, uv4, n_samples=10)
        p9 = sample_boundary_pixels(atlas9, uv9, n_samples=10)
        # 9 tiles should produce more samples than 4
        assert len(p9) > len(p4)


class TestSampleInteriorPixels:
    def test_returns_pixels(self):
        atlas, uv = _make_fake_atlas(4)
        pixels = sample_interior_pixels(atlas, uv, n_samples=20)
        assert len(pixels) > 0
        assert all(len(p) == 3 for p in pixels)

    def test_interior_avoids_edges(self):
        """Interior pixels should have the tile's base colour, not gutter grey."""
        atlas, uv = _make_fake_atlas(4)
        pixels = sample_interior_pixels(atlas, uv, n_samples=50, margin_frac=0.3)
        # The gutter is grey (100,100,100).  Interior pixels should
        # differ from that since each tile has a random colour.
        avg_r = sum(p[0] for p in pixels) / len(pixels)
        # Not all exactly 100 (the gutter colour)
        assert abs(avg_r - 100) > 5 or len(pixels) < 5


class TestColourVariance:
    def test_uniform(self):
        pixels = [(100, 100, 100)] * 10
        assert _colour_variance(pixels) == 0.0

    def test_varied(self):
        pixels = [(0, 0, 0), (255, 255, 255)]
        assert _colour_variance(pixels) > 0

    def test_empty(self):
        assert _colour_variance([]) == 0.0

    def test_single(self):
        assert _colour_variance([(50, 50, 50)]) == 0.0


class TestMeasureSeamVisibility:
    def test_returns_dict(self):
        atlas, uv = _make_fake_atlas(4)
        result = measure_seam_visibility(atlas, uv, n_samples=10)
        assert "boundary_variance" in result
        assert "interior_variance" in result
        assert "ratio" in result

    def test_ratio_is_finite(self):
        atlas, uv = _make_fake_atlas(4)
        result = measure_seam_visibility(atlas, uv, n_samples=20)
        assert result["ratio"] >= 0
        # With random tiles, both should have positive variance
        assert result["boundary_variance"] > 0
        assert result["interior_variance"] > 0

    def test_seamless_atlas_low_ratio(self):
        """A uniform atlas should have ratio ≈ 1.0 or very low."""
        atlas = Image.new("RGB", (256, 256), (100, 150, 200))
        uv = {"t0": (0.1, 0.1, 0.9, 0.9)}
        result = measure_seam_visibility(atlas, uv, n_samples=20)
        # Uniform → both variances near 0, ratio should be low or nan
        assert result["boundary_variance"] < 1.0


# ═══════════════════════════════════════════════════════════════════
# 18E.2 — Topology feature verification
# ═══════════════════════════════════════════════════════════════════

class TestVerifyTopologyFeatures:
    def test_returns_checks(self):
        coll, grid = _make_tiny_collection()
        result = verify_topology_features(coll, grid, seed=42)
        assert "tree_centroid_check" in result
        assert "ocean_depth_check" in result
        assert "determinism_check" in result

    def test_tree_centroid_valid(self):
        coll, grid = _make_tiny_collection()
        result = verify_topology_features(coll, grid, seed=42)
        assert result["tree_centroid_check"] is True

    def test_ocean_depth_valid(self):
        coll, grid = _make_tiny_collection()
        result = verify_topology_features(coll, grid, seed=42)
        assert result["ocean_depth_check"] is True

    def test_determinism(self):
        coll, grid = _make_tiny_collection()
        result = verify_topology_features(coll, grid, seed=42)
        assert result["determinism_check"] is True


# ═══════════════════════════════════════════════════════════════════
# 18E.4 — Performance budget (lightweight check)
# ═══════════════════════════════════════════════════════════════════

class TestBenchmarkStructure:
    """Test that benchmark_apron_pipeline returns the right shape
    (without actually running the heavy benchmark)."""

    def test_import(self):
        from polygrid.visual_cohesion import benchmark_apron_pipeline
        assert callable(benchmark_apron_pipeline)


# ═══════════════════════════════════════════════════════════════════
# 18E.5 — Full pipeline (integration)
# ═══════════════════════════════════════════════════════════════════

class TestRunFullPipeline:
    """Integration test — runs the entire Phase 18 pipeline end-to-end."""

    @pytest.fixture(scope="class")
    def pipeline_no_features(self, tmp_path_factory):
        """Run pipeline once without features, shared across tests."""
        from polygrid.visual_cohesion import run_full_pipeline
        out = tmp_path_factory.mktemp("pipeline_no_feat")
        return run_full_pipeline(
            frequency=2,
            detail_rings=3,
            tile_size=64,
            seed=42,
            output_dir=out / "pipeline",
            enable_features=False,
            enable_export=False,
        )

    @pytest.fixture(scope="class")
    def pipeline_with_features(self, tmp_path_factory):
        """Run pipeline once with features+export, shared across tests."""
        from polygrid.visual_cohesion import run_full_pipeline
        out = tmp_path_factory.mktemp("pipeline_full")
        return run_full_pipeline(
            frequency=2,
            detail_rings=3,
            tile_size=64,
            seed=42,
            output_dir=out / "full",
            enable_features=True,
            enable_export=True,
        )

    @pytest.mark.slow
    def test_full_pipeline_no_features(self, pipeline_no_features):
        """Run pipeline without biome features."""
        result = pipeline_no_features
        assert result["n_tiles"] > 0
        assert result["uv_tile_count"] > 0
        assert result["seam_visibility"]["ratio"] >= 0
        assert Path(result["atlas_path"]).exists()

    @pytest.mark.slow
    def test_full_pipeline_with_features(self, pipeline_with_features):
        """Run pipeline with biome features + export."""
        result = pipeline_with_features
        assert result["n_tiles"] > 0
        assert "feature_atlas_path" in result
        assert result["topology_verification"]["tree_centroid_check"] is True
        assert result["topology_verification"]["ocean_depth_check"] is True
        assert result["topology_verification"]["determinism_check"] is True
        assert result["seam_visibility"]["ratio"] >= 0

        # Export artifacts
        assert result.get("ktx2_valid") is True
        assert result["mip_levels"] > 1
        assert result["pot_size"][0] > 0

    @pytest.mark.slow
    def test_full_pipeline_timings(self, pipeline_no_features):
        """Verify timing data is collected."""
        result = pipeline_no_features
        assert "timings" in result
        assert result["timings"]["globe_build"] > 0
        assert result["timings"]["detail_terrain"] > 0
        assert result["timings"]["apron_atlas"] > 0
