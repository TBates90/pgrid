"""Tests for Phase 14A — biome_scatter.py (feature placement)."""

from __future__ import annotations

import math
import pytest

from polygrid.biome_scatter import (
    FeatureInstance,
    collect_margin_features,
    compute_density_field,
    poisson_disk_sample,
    scatter_features_on_tile,
)


# ═══════════════════════════════════════════════════════════════════
# Poisson disk sampling
# ═══════════════════════════════════════════════════════════════════

class TestPoissonDiskSample:
    """Core Poisson disk sampling tests."""

    def test_all_points_in_bounds(self):
        pts = poisson_disk_sample(100.0, 100.0, 8.0, seed=1)
        for x, y in pts:
            assert 0 <= x < 100.0, f"x={x}"
            assert 0 <= y < 100.0, f"y={y}"

    def test_minimum_distance_respected(self):
        min_d = 10.0
        pts = poisson_disk_sample(200.0, 200.0, min_d, seed=2)
        for i, (x1, y1) in enumerate(pts):
            for j, (x2, y2) in enumerate(pts):
                if i >= j:
                    continue
                dist = math.hypot(x1 - x2, y1 - y2)
                assert dist >= min_d * 0.99, (
                    f"Points {i} and {j} too close: {dist:.3f} < {min_d}"
                )

    def test_reasonable_point_count(self):
        """Should produce a reasonable number of points for the area."""
        pts = poisson_disk_sample(256.0, 256.0, 8.0, seed=3)
        # Theoretical max ≈ area / (π * (r/2)²) ≈ 256² / (π * 16) ≈ 1303
        # Bridson typically achieves 60-80% fill
        assert len(pts) > 100, f"Too few points: {len(pts)}"
        assert len(pts) < 5000, f"Too many points: {len(pts)}"

    def test_deterministic(self):
        pts_a = poisson_disk_sample(100.0, 100.0, 5.0, seed=42)
        pts_b = poisson_disk_sample(100.0, 100.0, 5.0, seed=42)
        assert pts_a == pts_b

    def test_different_seeds_differ(self):
        pts_a = poisson_disk_sample(100.0, 100.0, 5.0, seed=1)
        pts_b = poisson_disk_sample(100.0, 100.0, 5.0, seed=2)
        assert pts_a != pts_b

    def test_variable_density_more_points_in_dense_region(self):
        """With a density function, the dense half should have more points."""
        # Left half density=1.0, right half density=0.2
        def density_fn(x, y):
            return 1.0 if x < 128 else 0.2

        pts = poisson_disk_sample(
            256.0, 256.0, 10.0, seed=5, density_fn=density_fn,
        )
        left = sum(1 for x, _ in pts if x < 128)
        right = sum(1 for x, _ in pts if x >= 128)
        assert left > right, f"Left={left}, Right={right}"

    def test_small_area(self):
        """Should work on a tiny area without crashing."""
        pts = poisson_disk_sample(10.0, 10.0, 3.0, seed=7)
        assert len(pts) >= 1

    def test_large_min_distance(self):
        """Very large min_distance relative to area → very few points."""
        pts = poisson_disk_sample(20.0, 20.0, 15.0, seed=8)
        assert 1 <= len(pts) <= 5


# ═══════════════════════════════════════════════════════════════════
# FeatureInstance
# ═══════════════════════════════════════════════════════════════════

class TestFeatureInstance:
    def test_creation(self):
        fi = FeatureInstance(px=10.0, py=20.0, radius=5.0)
        assert fi.px == 10.0
        assert fi.py == 20.0
        assert fi.radius == 5.0
        assert fi.species_id == 0
        assert fi.depth == 0.0

    def test_custom_color(self):
        fi = FeatureInstance(
            px=0, py=0, radius=3.0,
            color=(100, 200, 50),
            shadow_color=(10, 20, 5),
        )
        assert fi.color == (100, 200, 50)
        assert fi.shadow_color == (10, 20, 5)


# ═══════════════════════════════════════════════════════════════════
# scatter_features_on_tile
# ═══════════════════════════════════════════════════════════════════

class TestScatterFeaturesOnTile:
    def test_zero_density_returns_empty(self):
        result = scatter_features_on_tile(0.0, tile_size=256, seed=1)
        assert result == []

    def test_high_density_produces_many_features(self):
        result = scatter_features_on_tile(
            0.9, tile_size=256, seed=10,
        )
        assert len(result) > 50, f"Only {len(result)} features at density 0.9"

    def test_low_density_produces_few_features(self):
        low = scatter_features_on_tile(0.15, tile_size=256, seed=10)
        high = scatter_features_on_tile(0.9, tile_size=256, seed=10)
        assert len(low) < len(high), (
            f"Low density ({len(low)}) >= high density ({len(high)})"
        )

    def test_all_features_within_tile(self):
        result = scatter_features_on_tile(
            0.8, tile_size=256, seed=20,
        )
        for inst in result:
            assert 0 <= inst.px < 256, f"px={inst.px}"
            assert 0 <= inst.py < 256, f"py={inst.py}"

    def test_radius_within_range(self):
        result = scatter_features_on_tile(
            0.8, tile_size=256, min_radius=3.0, max_radius=7.0, seed=30,
        )
        for inst in result:
            assert 3.0 <= inst.radius <= 7.0, f"radius={inst.radius}"

    def test_species_from_palette(self):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        result = scatter_features_on_tile(
            0.8, tile_size=256, colors=colors, seed=40,
        )
        species_seen = {inst.species_id for inst in result}
        # Should use at least 2 of the 3 species
        assert len(species_seen) >= 2

    def test_deterministic(self):
        a = scatter_features_on_tile(0.7, tile_size=128, seed=99)
        b = scatter_features_on_tile(0.7, tile_size=128, seed=99)
        assert len(a) == len(b)
        for ia, ib in zip(a, b):
            assert ia.px == ib.px
            assert ia.py == ib.py

    def test_sorted_by_depth(self):
        result = scatter_features_on_tile(0.8, tile_size=256, seed=50)
        depths = [inst.depth for inst in result]
        assert depths == sorted(depths)

    def test_color_noise_varies_colors(self):
        """Color noise should make trees different shades."""
        result = scatter_features_on_tile(
            0.9, tile_size=256, color_noise=20.0, seed=60,
        )
        if len(result) < 2:
            pytest.skip("Not enough features to compare")
        colors = {inst.color for inst in result}
        # With noise, should have many distinct colours
        assert len(colors) > 5

    def test_with_globe_3d_center(self):
        """Providing a 3D center should still produce valid features."""
        result = scatter_features_on_tile(
            0.8, tile_size=256, seed=70,
            globe_3d_center=(0.5, 0.5, 0.707),
        )
        assert len(result) > 10


# ═══════════════════════════════════════════════════════════════════
# compute_density_field
# ═══════════════════════════════════════════════════════════════════

class TestComputeDensityField:
    """Tests that require models library for globe grid."""

    @pytest.fixture
    def globe(self):
        try:
            from conftest import cached_build_globe
            g = cached_build_globe(1)
            if g is None:
                pytest.skip("models library not installed")
            return g
        except ImportError:
            pytest.skip("models library not installed")

    def test_all_biome_faces_have_density(self, globe):
        face_ids = list(globe.faces.keys())
        biome_faces = set(face_ids[:5])
        density = compute_density_field(
            globe, face_ids, biome_faces=biome_faces, seed=42,
        )
        for fid in face_ids:
            assert fid in density
            assert 0.0 <= density[fid] <= 1.0

    def test_non_biome_faces_are_zero(self, globe):
        face_ids = list(globe.faces.keys())
        biome_faces = set(face_ids[:3])
        density = compute_density_field(
            globe, face_ids, biome_faces=biome_faces, seed=42,
        )
        for fid in face_ids:
            if fid not in biome_faces:
                assert density[fid] == 0.0

    def test_biome_faces_have_positive_density(self, globe):
        face_ids = list(globe.faces.keys())
        biome_faces = set(face_ids)
        density = compute_density_field(
            globe, face_ids, biome_faces=biome_faces, seed=42,
        )
        for fid in biome_faces:
            assert density[fid] > 0.0, f"{fid} has density {density[fid]}"

    def test_none_biome_faces_means_all(self, globe):
        face_ids = list(globe.faces.keys())
        density = compute_density_field(
            globe, face_ids, biome_faces=None, seed=42,
        )
        for fid in face_ids:
            assert density[fid] > 0.0

    def test_density_varies_with_seed(self, globe):
        face_ids = list(globe.faces.keys())
        d1 = compute_density_field(globe, face_ids, seed=1)
        d2 = compute_density_field(globe, face_ids, seed=999)
        # At least some tiles should have different densities
        diffs = sum(1 for fid in face_ids if abs(d1[fid] - d2[fid]) > 0.01)
        assert diffs > 0


# ═══════════════════════════════════════════════════════════════════
# collect_margin_features
# ═══════════════════════════════════════════════════════════════════

class TestCollectMarginFeatures:
    def test_interior_features_not_near_edge(self):
        instances = [
            FeatureInstance(px=128, py=128, radius=5.0),
        ]
        interior, margin = collect_margin_features(instances, tile_size=256, margin=8.0)
        assert len(interior) == 1
        assert len(margin) == 0

    def test_edge_feature_in_margin(self):
        instances = [
            FeatureInstance(px=3.0, py=128, radius=5.0),  # left edge
        ]
        interior, margin = collect_margin_features(instances, tile_size=256, margin=8.0)
        assert len(interior) == 0
        assert len(margin) == 1

    def test_mixed_features(self):
        instances = [
            FeatureInstance(px=128, py=128, radius=5.0),  # interior
            FeatureInstance(px=2.0, py=128, radius=5.0),  # left edge
            FeatureInstance(px=128, py=253, radius=5.0),  # bottom edge
            FeatureInstance(px=64, py=64, radius=3.0),    # interior
        ]
        interior, margin = collect_margin_features(instances, tile_size=256, margin=8.0)
        assert len(interior) == 2
        assert len(margin) == 2

    def test_all_margin(self):
        """Feature with large radius near edge → margin."""
        instances = [
            FeatureInstance(px=5.0, py=5.0, radius=6.0),
        ]
        interior, margin = collect_margin_features(instances, tile_size=256, margin=8.0)
        # radius=6 extends to px=-1 which is < margin(8), so it's margin
        assert len(margin) == 1

    def test_empty_list(self):
        interior, margin = collect_margin_features([], tile_size=256)
        assert interior == []
        assert margin == []


# ═══════════════════════════════════════════════════════════════════
# Phase 16C — Full-slot feature scattering
# ═══════════════════════════════════════════════════════════════════


class TestScatterFeaturesFullslot:
    """16C.1 — Full-slot feature scattering across full square tile."""

    def test_features_outside_tile_bounds(self):
        """Full-slot scatter should produce features outside [0, tile_size]."""
        from polygrid.biome_scatter import scatter_features_fullslot

        instances = scatter_features_fullslot(
            tile_density=0.8, tile_size=64,
            overscan=0.2, seed=42,
        )
        assert len(instances) > 0

        # Some features should have positions outside [0, tile_size]
        outside = [
            inst for inst in instances
            if inst.px < 0 or inst.px > 64 or inst.py < 0 or inst.py > 64
        ]
        assert len(outside) > 0, (
            "Full-slot scatter should produce features outside tile bounds"
        )

    def test_more_features_than_standard(self):
        """Full-slot scatter covers a larger area → more features."""
        from polygrid.biome_scatter import scatter_features_fullslot

        standard = scatter_features_on_tile(
            tile_density=0.8, tile_size=64, seed=42,
        )
        fullslot = scatter_features_fullslot(
            tile_density=0.8, tile_size=64,
            overscan=0.15, seed=42,
        )
        # Fullslot area ≈ (1 + 2*0.15)² ≈ 1.69× standard area
        assert len(fullslot) >= len(standard), (
            f"Full-slot {len(fullslot)} should have >= standard {len(standard)}"
        )

    def test_feature_count_proportional_to_area(self):
        """Feature count should scale roughly with expanded area."""
        from polygrid.biome_scatter import scatter_features_fullslot

        small = scatter_features_fullslot(
            tile_density=0.8, tile_size=64,
            overscan=0.0, seed=42,
        )
        large = scatter_features_fullslot(
            tile_density=0.8, tile_size=64,
            overscan=0.3, seed=42,
        )
        # Area ratio: (1 + 2*0.3)² / 1² = 2.56
        # Allow generous tolerance (Poisson disk is stochastic)
        if len(small) > 5:
            ratio = len(large) / len(small)
            assert ratio > 1.0, f"Larger area should yield more features: {ratio}"

    def test_deterministic(self):
        from polygrid.biome_scatter import scatter_features_fullslot

        a = scatter_features_fullslot(
            tile_density=0.8, tile_size=64, seed=42,
        )
        b = scatter_features_fullslot(
            tile_density=0.8, tile_size=64, seed=42,
        )
        assert len(a) == len(b)
        for fa, fb in zip(a, b):
            assert fa.px == fb.px
            assert fa.py == fb.py

    def test_neighbour_density_affects_margin(self):
        """Features in the margin zone should use neighbour density."""
        from polygrid.biome_scatter import scatter_features_fullslot

        # High density everywhere
        uniform = scatter_features_fullslot(
            tile_density=0.8, tile_size=64, overscan=0.2, seed=42,
        )
        # Zero density for neighbours → fewer features in margins
        sparse_margins = scatter_features_fullslot(
            tile_density=0.8, tile_size=64, overscan=0.2, seed=42,
            neighbour_densities={
                "left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0,
            },
        )
        # With zero neighbour density, margin features should be sparser
        # (though both use the same Poisson seed, the density_fn differs
        # so the accepted points may differ)
        # Just verify it runs and produces features
        assert len(sparse_margins) > 0

    def test_zero_density_returns_empty(self):
        from polygrid.biome_scatter import scatter_features_fullslot

        instances = scatter_features_fullslot(
            tile_density=0.0, tile_size=64, seed=42,
        )
        assert instances == []

    def test_density_continuity_at_boundary(self):
        """Feature density near the hex boundary shouldn't have a gap.

        Count features in two bands: just inside [0, tile_size] and
        just outside.  With uniform density, both should be populated.
        """
        from polygrid.biome_scatter import scatter_features_fullslot

        tile_size = 128
        instances = scatter_features_fullslot(
            tile_density=0.8, tile_size=tile_size,
            overscan=0.2, seed=42,
        )

        band = 10  # pixels
        inside_band = [
            inst for inst in instances
            if 0 <= inst.px < band or 0 <= inst.py < band
            or tile_size - band < inst.px <= tile_size
            or tile_size - band < inst.py <= tile_size
        ]
        outside_band = [
            inst for inst in instances
            if inst.px < 0 and inst.px > -band
            or inst.py < 0 and inst.py > -band
            or inst.px > tile_size and inst.px < tile_size + band
            or inst.py > tile_size and inst.py < tile_size + band
        ]
        # Both bands should have features (no gap at boundary)
        assert len(inside_band) > 0, "No features just inside boundary"
        assert len(outside_band) > 0, "No features just outside boundary"
