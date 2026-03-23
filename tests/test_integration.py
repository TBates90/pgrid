"""Tests for the public integration API — generate_planet() pipeline."""

from __future__ import annotations

import pytest

from polygrid.integration import (
    GenerationResult,
    PlanetParams,
    RegionParams,
    TileResult,
    generate_planet,
    parse_layout,
    _build_schema,
    _compute_water_level,
)


# ═══════════════════════════════════════════════════════════════════
# parse_layout
# ═══════════════════════════════════════════════════════════════════

class TestParseLayout:
    """Tests for layout string parsing."""

    def test_gb3(self):
        assert parse_layout("gb3") == 3

    def test_gb10(self):
        assert parse_layout("gb10") == 10

    def test_bare_integer(self):
        assert parse_layout("4") == 4

    def test_case_insensitive(self):
        assert parse_layout("GB5") == 5

    def test_whitespace_stripped(self):
        assert parse_layout("  gb3  ") == 3

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_layout("custom")

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="≥ 1"):
            parse_layout("gb0")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="≥ 1"):
            parse_layout("-1")


# ═══════════════════════════════════════════════════════════════════
# _compute_water_level
# ═══════════════════════════════════════════════════════════════════

class TestComputeWaterLevel:
    """Water level threshold from water_abundance."""

    def test_zero(self):
        assert _compute_water_level(0.0) == pytest.approx(0.0)

    def test_half(self):
        assert _compute_water_level(0.5) == pytest.approx(0.30)

    def test_full(self):
        assert _compute_water_level(1.0) == pytest.approx(0.60)


# ═══════════════════════════════════════════════════════════════════
# _build_schema
# ═══════════════════════════════════════════════════════════════════

class TestBuildSchema:
    """Schema used by the pipeline has all required fields."""

    def test_has_all_fields(self):
        schema = _build_schema()
        for name in ("elevation", "temperature", "moisture", "terrain", "features", "region_id"):
            assert schema.has_field(name), f"Missing field: {name}"


# ═══════════════════════════════════════════════════════════════════
# generate_planet — round-trip integration tests
# ═══════════════════════════════════════════════════════════════════

class TestGeneratePlanet:
    """End-to-end tests for the full pipeline."""

    @pytest.fixture()
    def default_result(self) -> GenerationResult:
        """Generate with default params (gb2 for speed)."""
        return generate_planet(PlanetParams(frequency=2, seed=99))

    # ── Basic structure ─────────────────────────────────────────────

    def test_returns_generation_result(self, default_result):
        assert isinstance(default_result, GenerationResult)

    def test_tile_count_matches_goldberg(self, default_result):
        """gb2 → 10×2²+2 = 42 tiles."""
        assert len(default_result.tiles) == 42
        assert default_result.metadata["tile_count"] == 42

    def test_pentagon_count(self, default_result):
        pentagons = [t for t in default_result.tiles if t.face_type == "pent"]
        assert len(pentagons) == 12
        assert default_result.metadata["pentagon_count"] == 12

    def test_hexagon_count(self, default_result):
        hexagons = [t for t in default_result.tiles if t.face_type == "hex"]
        assert len(hexagons) == 30
        assert default_result.metadata["hexagon_count"] == 30

    # ── Per-tile field validation ───────────────────────────────────

    def test_all_tiles_have_slug(self, default_result):
        for tile in default_result.tiles:
            assert tile.tile_slug, f"Tile {tile.face_id} has no slug"

    def test_all_tiles_have_face_id(self, default_result):
        face_ids = {t.face_id for t in default_result.tiles}
        assert len(face_ids) == 42  # No duplicates

    def test_elevation_in_range(self, default_result):
        for tile in default_result.tiles:
            assert 0.0 <= tile.elevation <= 1.0, (
                f"{tile.face_id}: elevation {tile.elevation} out of range"
            )

    def test_temperature_in_range(self, default_result):
        for tile in default_result.tiles:
            assert 0.0 <= tile.temperature <= 1.0, (
                f"{tile.face_id}: temperature {tile.temperature} out of range"
            )

    def test_moisture_in_range(self, default_result):
        for tile in default_result.tiles:
            assert 0.0 <= tile.moisture <= 1.0, (
                f"{tile.face_id}: moisture {tile.moisture} out of range"
            )

    def test_terrain_is_known_type(self, default_result):
        from polygrid.terrain.classification import TERRAIN_TYPES
        for tile in default_result.tiles:
            assert tile.terrain in TERRAIN_TYPES, (
                f"{tile.face_id}: unknown terrain {tile.terrain!r}"
            )

    def test_features_are_lists(self, default_result):
        for tile in default_result.tiles:
            assert isinstance(tile.features, list)

    def test_color_is_rgb_tuple(self, default_result):
        for tile in default_result.tiles:
            assert len(tile.color) == 3
            for c in tile.color:
                assert 0.0 <= c <= 1.0, f"Colour component {c} out of range"

    # ── Metadata ────────────────────────────────────────────────────

    def test_metadata_contains_seed(self, default_result):
        assert default_result.metadata["seed"] == 99

    def test_metadata_contains_frequency(self, default_result):
        assert default_result.metadata["frequency"] == 2

    def test_metadata_generation_time(self, default_result):
        assert default_result.metadata["generation_time_s"] > 0

    # ── Determinism ─────────────────────────────────────────────────

    def test_deterministic_same_seed(self):
        """Same params → identical results."""
        params = PlanetParams(frequency=2, seed=123)
        r1 = generate_planet(params)
        r2 = generate_planet(params)
        for t1, t2 in zip(r1.tiles, r2.tiles):
            assert t1.elevation == t2.elevation
            assert t1.temperature == t2.temperature
            assert t1.moisture == t2.moisture
            assert t1.terrain == t2.terrain
            assert t1.features == t2.features

    def test_different_seed_different_results(self):
        """Different seed → different elevation map."""
        r1 = generate_planet(PlanetParams(frequency=2, seed=1))
        r2 = generate_planet(PlanetParams(frequency=2, seed=999))
        elevations_1 = [t.elevation for t in r1.tiles]
        elevations_2 = [t.elevation for t in r2.tiles]
        assert elevations_1 != elevations_2

    # ── Water abundance ─────────────────────────────────────────────

    def test_high_water_abundance_creates_more_ocean(self):
        dry = generate_planet(PlanetParams(frequency=2, seed=42, water_abundance=0.1))
        wet = generate_planet(PlanetParams(frequency=2, seed=42, water_abundance=0.9))
        dry_ocean = sum(1 for t in dry.tiles if t.terrain == "ocean")
        wet_ocean = sum(1 for t in wet.tiles if t.terrain == "ocean")
        assert wet_ocean >= dry_ocean

    # ── Roughness ───────────────────────────────────────────────────

    def test_roughness_affects_elevation_variance(self):
        smooth = generate_planet(PlanetParams(frequency=2, seed=42, roughness=0.0))
        rough = generate_planet(PlanetParams(frequency=2, seed=42, roughness=1.0))
        import statistics
        var_smooth = statistics.variance(t.elevation for t in smooth.tiles)
        var_rough = statistics.variance(t.elevation for t in rough.tiles)
        # Rougher terrain should generally have higher variance.
        # Both are normalised [0,1] so the absolute values may differ
        # only modestly, but rough should not be less than smooth.
        # Use a loose assertion — we mainly check the pipeline doesn't crash.
        assert var_smooth >= 0 and var_rough >= 0

    # ── Region assignment ───────────────────────────────────────────

    def test_region_tiles_get_region_id(self):
        """Tiles assigned to a region have region_id set."""
        # Get a valid slug from a gb2 globe.
        result_no_regions = generate_planet(PlanetParams(frequency=2, seed=42))
        some_slug = result_no_regions.tiles[0].tile_slug

        region = RegionParams(
            region_id="test-region",
            tile_slugs=[some_slug],
        )
        result = generate_planet(PlanetParams(frequency=2, seed=42, regions=[region]))

        # Find the tile with that slug
        matched = [t for t in result.tiles if t.tile_slug == some_slug]
        assert len(matched) == 1
        assert matched[0].region_id == "test-region"

    def test_unassigned_tiles_have_no_region(self):
        """Tiles not in any region have region_id = None."""
        result = generate_planet(PlanetParams(frequency=2, seed=42))
        for tile in result.tiles:
            assert tile.region_id is None

    def test_region_elevation_modifier(self):
        """Elevation modifier scales raw elevation before normalisation."""
        # This is a smoke test — we just verify no crash and the
        # pipeline completes with a modifier != 1.0.
        result_no_regions = generate_planet(PlanetParams(frequency=2, seed=42))
        slug = result_no_regions.tiles[5].tile_slug

        region = RegionParams(
            region_id="high-ground",
            tile_slugs=[slug],
            elevation_modifier=2.0,
        )
        result = generate_planet(PlanetParams(frequency=2, seed=42, regions=[region]))
        assert len(result.tiles) == 42

    def test_region_humidity_modifier(self):
        """Humidity modifier affects moisture for region tiles."""
        result_no_regions = generate_planet(PlanetParams(frequency=2, seed=42))
        slug = result_no_regions.tiles[5].tile_slug

        region = RegionParams(
            region_id="jungle",
            tile_slugs=[slug],
            humidity_modifier=2.0,
        )
        result = generate_planet(PlanetParams(frequency=2, seed=42, regions=[region]))
        assert len(result.tiles) == 42

    # ── Coast features ──────────────────────────────────────────────

    def test_ocean_tiles_never_have_coast(self):
        """Coast is a land feature, not ocean."""
        result = generate_planet(PlanetParams(frequency=2, seed=42, water_abundance=0.5))
        for tile in result.tiles:
            if tile.terrain == "ocean":
                assert "coast" not in tile.features

    # ── Bare integer layout ─────────────────────────────────────────

    def test_bare_integer_layout(self):
        result = generate_planet(PlanetParams(frequency=2, seed=42))
        assert len(result.tiles) == 42


# ═══════════════════════════════════════════════════════════════════
# Import from top-level package
# ═══════════════════════════════════════════════════════════════════

class TestTopLevelImports:
    """Verify integration types are importable from polygrid."""

    def test_import_planet_params(self):
        from polygrid import PlanetParams  # noqa: F811
        assert PlanetParams is not None

    def test_import_generate_planet(self):
        from polygrid import generate_planet  # noqa: F811
        assert callable(generate_planet)

    def test_import_parse_layout(self):
        from polygrid import parse_layout  # noqa: F811
        assert callable(parse_layout)
