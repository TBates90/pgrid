"""Tests for noise.py — Phase 7A: Reusable noise primitives."""

from __future__ import annotations

import math

import pytest

from polygrid.noise import (
    fbm,
    ridged_noise,
    domain_warp,
    gradient_mask,
    terrace,
    normalize,
    remap,
)


# ═══════════════════════════════════════════════════════════════════
# fbm
# ═══════════════════════════════════════════════════════════════════


class TestFBM:
    """Fractal Brownian Motion tests."""

    def test_returns_float(self):
        assert isinstance(fbm(0.0, 0.0), float)

    def test_output_range(self):
        """FBM values should be in approximately [−1, 1]."""
        vals = [fbm(x * 0.1, y * 0.1) for x in range(-50, 51) for y in range(-50, 51)]
        assert all(-1.01 <= v <= 1.01 for v in vals), (
            f"min={min(vals):.4f}, max={max(vals):.4f}"
        )

    def test_determinism(self):
        """Same seed → same output."""
        a = fbm(1.23, 4.56, seed=99)
        b = fbm(1.23, 4.56, seed=99)
        assert a == b

    def test_different_seeds(self):
        """Different seeds should (almost certainly) produce different output."""
        a = fbm(1.0, 1.0, seed=1)
        b = fbm(1.0, 1.0, seed=2)
        assert a != b

    def test_octaves_affect_detail(self):
        """More octaves should produce a different (typically more varied) signal."""
        low = [fbm(x * 0.1, 0.0, octaves=1) for x in range(100)]
        high = [fbm(x * 0.1, 0.0, octaves=8) for x in range(100)]
        # They shouldn't be identical
        assert low != high

    def test_frequency_scales_features(self):
        """Higher frequency should change the spatial pattern."""
        a = fbm(1.0, 1.0, frequency=0.5)
        b = fbm(1.0, 1.0, frequency=5.0)
        # At the same point, different frequencies should give different values
        # (not guaranteed but extremely likely)
        assert a != b

    def test_zero_octaves_returns_zero(self):
        """Edge case: 0 octaves → 0."""
        assert fbm(1.0, 1.0, octaves=0) == 0.0


# ═══════════════════════════════════════════════════════════════════
# ridged_noise
# ═══════════════════════════════════════════════════════════════════


class TestRidgedNoise:
    """Ridged multifractal noise tests."""

    def test_returns_float(self):
        assert isinstance(ridged_noise(0.0, 0.0), float)

    def test_output_range(self):
        """Ridged noise should be in [0, 1]."""
        vals = [
            ridged_noise(x * 0.1, y * 0.1)
            for x in range(-50, 51)
            for y in range(-50, 51)
        ]
        assert all(-0.01 <= v <= 1.01 for v in vals), (
            f"min={min(vals):.4f}, max={max(vals):.4f}"
        )

    def test_determinism(self):
        a = ridged_noise(1.23, 4.56, seed=99)
        b = ridged_noise(1.23, 4.56, seed=99)
        assert a == b

    def test_has_high_values(self):
        """Should produce some values near 1 (ridge peaks)."""
        vals = [ridged_noise(x * 0.05, y * 0.05) for x in range(100) for y in range(100)]
        assert max(vals) > 0.5, f"max ridged value only {max(vals):.4f}"

    def test_ridge_offset(self):
        """Different ridge_offset should change the output."""
        a = ridged_noise(1.0, 1.0, ridge_offset=0.5)
        b = ridged_noise(1.0, 1.0, ridge_offset=2.0)
        assert a != b


# ═══════════════════════════════════════════════════════════════════
# domain_warp
# ═══════════════════════════════════════════════════════════════════


class TestDomainWarp:
    """Domain warping tests."""

    def test_returns_float(self):
        val = domain_warp(fbm, 1.0, 1.0)
        assert isinstance(val, float)

    def test_warp_changes_value(self):
        """Warped value should differ from unwarped (for non-zero strength)."""
        plain = fbm(1.0, 1.0, seed=42)
        warped = domain_warp(fbm, 1.0, 1.0, warp_strength=0.5, seed=42)
        assert plain != warped

    def test_zero_warp_matches_plain(self):
        """With warp_strength=0, warped value should match plain."""
        plain = fbm(1.0, 1.0, seed=42)
        warped = domain_warp(fbm, 1.0, 1.0, warp_strength=0.0, seed=42)
        assert abs(plain - warped) < 1e-10

    def test_composable_with_ridged(self):
        """domain_warp should work with ridged_noise too."""
        val = domain_warp(ridged_noise, 1.0, 1.0, seed=42)
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.01

    def test_determinism(self):
        a = domain_warp(fbm, 1.0, 1.0, seed=42, warp_seed_x=10, warp_seed_y=20)
        b = domain_warp(fbm, 1.0, 1.0, seed=42, warp_seed_x=10, warp_seed_y=20)
        assert a == b


# ═══════════════════════════════════════════════════════════════════
# gradient_mask
# ═══════════════════════════════════════════════════════════════════


class TestGradientMask:
    """Radial gradient mask tests."""

    def test_center_is_one(self):
        """At the center, mask value should be 1.0."""
        for falloff in ("linear", "smooth", "exponential"):
            assert gradient_mask(0.0, 0.0, falloff=falloff) == pytest.approx(1.0)

    def test_at_radius_is_zero(self):
        """At the radius distance, mask should be 0 (or near 0 for exponential)."""
        assert gradient_mask(1.0, 0.0, radius=1.0, falloff="linear") == pytest.approx(0.0)
        assert gradient_mask(1.0, 0.0, radius=1.0, falloff="smooth") == pytest.approx(0.0)
        # Exponential approaches 0 but doesn't hit it exactly
        assert gradient_mask(1.0, 0.0, radius=1.0, falloff="exponential") < 0.06

    def test_beyond_radius_clamped(self):
        """Beyond the radius, mask should be 0 (clamped)."""
        assert gradient_mask(5.0, 0.0, radius=1.0, falloff="linear") == pytest.approx(0.0)
        assert gradient_mask(5.0, 0.0, radius=1.0, falloff="smooth") == pytest.approx(0.0)

    def test_output_range(self):
        """All values should be in [0, 1]."""
        for falloff in ("linear", "smooth", "exponential"):
            vals = [
                gradient_mask(x * 0.1, y * 0.1, radius=3.0, falloff=falloff)
                for x in range(-40, 41)
                for y in range(-40, 41)
            ]
            assert all(0.0 <= v <= 1.0 + 1e-10 for v in vals)

    def test_custom_center(self):
        """Mask should be 1 at the custom center."""
        val = gradient_mask(3.0, 4.0, center=(3.0, 4.0))
        assert val == pytest.approx(1.0)

    def test_unknown_falloff_raises(self):
        with pytest.raises(ValueError, match="Unknown falloff"):
            gradient_mask(0.0, 0.0, falloff="cubic")


# ═══════════════════════════════════════════════════════════════════
# terrace
# ═══════════════════════════════════════════════════════════════════


class TestTerrace:
    """Stepped plateau remapping tests."""

    def test_basic_terracing(self):
        """Value of 0.3 with 4 steps → floor(0.3*4)/4 = floor(1.2)/4 = 0.25."""
        assert terrace(0.3, steps=4) == pytest.approx(0.25)

    def test_full_smoothing_returns_original(self):
        """smoothing=1.0 should return the original value."""
        assert terrace(0.3, steps=4, smoothing=1.0) == pytest.approx(0.3)

    def test_zero_smoothing_terraces(self):
        """smoothing=0 should give pure terrace."""
        assert terrace(0.3, steps=4, smoothing=0.0) == pytest.approx(0.25)

    def test_output_range(self):
        """Terraced values of [0, 1] input should stay in [0, 1]."""
        for v in [i * 0.01 for i in range(101)]:
            t = terrace(v, steps=5)
            assert 0.0 <= t <= 1.0 + 1e-10

    def test_steps_must_be_positive(self):
        with pytest.raises(ValueError, match="steps must be >= 1"):
            terrace(0.5, steps=0)

    def test_many_steps_approaches_identity(self):
        """With very many steps, terracing should approximate the identity."""
        assert terrace(0.3, steps=1000) == pytest.approx(0.3, abs=0.002)


# ═══════════════════════════════════════════════════════════════════
# normalize / remap
# ═══════════════════════════════════════════════════════════════════


class TestNormalize:
    """Range remapping tests."""

    def test_identity(self):
        """Default remap of 0 from [−1,1] to [0,1] → 0.5."""
        assert normalize(0.0) == pytest.approx(0.5)

    def test_min_maps_to_dst_min(self):
        assert normalize(-1.0) == pytest.approx(0.0)

    def test_max_maps_to_dst_max(self):
        assert normalize(1.0) == pytest.approx(1.0)

    def test_custom_range(self):
        assert normalize(50.0, src_min=0.0, src_max=100.0, dst_min=10.0, dst_max=20.0) == pytest.approx(15.0)

    def test_clamps_below(self):
        assert normalize(-5.0) == pytest.approx(0.0)

    def test_clamps_above(self):
        assert normalize(5.0) == pytest.approx(1.0)

    def test_degenerate_source_range(self):
        """If src_min == src_max, return midpoint of dest range."""
        assert normalize(1.0, src_min=1.0, src_max=1.0) == pytest.approx(0.5)

    def test_remap_is_alias(self):
        assert remap is normalize


# ═══════════════════════════════════════════════════════════════════
# Composability integration
# ═══════════════════════════════════════════════════════════════════


class TestComposability:
    """Verify that primitives can be composed freely."""

    def test_ridged_through_domain_warp(self):
        val = domain_warp(ridged_noise, 2.5, 3.5, warp_strength=0.4, seed=7)
        assert isinstance(val, float)

    def test_fbm_masked_and_terraced(self):
        raw = fbm(1.0, 1.0)
        masked = raw * gradient_mask(1.0, 1.0, center=(0.0, 0.0), radius=5.0)
        stepped = terrace(normalize(masked), steps=6)
        assert 0.0 <= stepped <= 1.0

    def test_warped_ridged_masked(self):
        """Full chain: warp → ridge → mask → terrace → normalize."""
        raw = domain_warp(ridged_noise, 1.0, 2.0, warp_strength=0.3, seed=42)
        masked = raw * gradient_mask(1.0, 2.0, center=(0.0, 0.0), radius=10.0)
        stepped = terrace(masked, steps=4)
        final = normalize(stepped, src_min=0.0, src_max=1.0, dst_min=0.0, dst_max=100.0)
        assert 0.0 <= final <= 100.0
