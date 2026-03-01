"""Tests for Phase 11A — 3-D coherent terrain generation.

Tests verify:
- Sub-face 3-D position computation (single + batch)
- Globe-coherent noise produces spatially continuous terrain
- Adjacent tiles' boundary sub-faces have similar elevations
- No patchwork artefact: cross-tile variance ≈ intra-tile variance
- Determinism: identical inputs → identical output
- Performance: not unreasonably slow
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import pytest

# ── Test helpers ─────────────────────────────────────────────────────

def _require_globe():
    """Skip if the models library isn't available."""
    try:
        from polygrid.globe import build_globe_grid
        return build_globe_grid
    except ImportError:
        pytest.skip("models library not installed")


def _build_globe_and_collection(frequency: int = 3, detail_rings: int = 3):
    """Build a small globe + detail grids + globe store for testing.

    Uses the session-level cache from conftest to avoid rebuilding the
    expensive Goldberg polyhedron on every single test method.
    """
    from conftest import cached_build_globe_and_collection

    result = cached_build_globe_and_collection(frequency, detail_rings)
    if result is None:
        pytest.skip("models library not installed")
    return result


# ═══════════════════════════════════════════════════════════════════
# Test: tangent basis is orthonormal
# ═══════════════════════════════════════════════════════════════════

class TestTangentBasis:
    """Tests for the internal _tangent_basis helper."""

    def test_orthonormal_z_axis(self):
        from polygrid.detail_terrain_3d import _tangent_basis
        normal = (0.0, 0.0, 1.0)
        u, v = _tangent_basis(normal)

        # u and v should be unit vectors
        assert abs(math.sqrt(u[0]**2 + u[1]**2 + u[2]**2) - 1.0) < 1e-10
        assert abs(math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) - 1.0) < 1e-10

        # u · v = 0
        dot_uv = u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
        assert abs(dot_uv) < 1e-10

        # u · n = 0, v · n = 0
        dot_un = u[0]*normal[0] + u[1]*normal[1] + u[2]*normal[2]
        dot_vn = v[0]*normal[0] + v[1]*normal[1] + v[2]*normal[2]
        assert abs(dot_un) < 1e-10
        assert abs(dot_vn) < 1e-10

    def test_orthonormal_arbitrary(self):
        from polygrid.detail_terrain_3d import _tangent_basis
        # Normalised arbitrary direction
        n = (1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3))
        u, v = _tangent_basis(n)

        u_len = math.sqrt(sum(c**2 for c in u))
        v_len = math.sqrt(sum(c**2 for c in v))
        assert abs(u_len - 1.0) < 1e-10
        assert abs(v_len - 1.0) < 1e-10

        dot_uv = sum(a*b for a, b in zip(u, v))
        assert abs(dot_uv) < 1e-10

    def test_x_aligned_normal(self):
        from polygrid.detail_terrain_3d import _tangent_basis
        normal = (1.0, 0.0, 0.0)
        u, v = _tangent_basis(normal)
        u_len = math.sqrt(sum(c**2 for c in u))
        assert abs(u_len - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════
# Test: normalize helper
# ═══════════════════════════════════════════════════════════════════

class TestNormalize:
    def test_unit_sphere(self):
        from polygrid.detail_terrain_3d import _normalize
        result = _normalize((3.0, 4.0, 0.0), radius=1.0)
        length = math.sqrt(sum(c**2 for c in result))
        assert abs(length - 1.0) < 1e-10

    def test_custom_radius(self):
        from polygrid.detail_terrain_3d import _normalize
        result = _normalize((1.0, 0.0, 0.0), radius=2.0)
        assert abs(result[0] - 2.0) < 1e-10
        assert abs(result[1]) < 1e-10
        assert abs(result[2]) < 1e-10

    def test_zero_vector(self):
        from polygrid.detail_terrain_3d import _normalize
        result = _normalize((0.0, 0.0, 0.0), radius=1.0)
        # Should handle gracefully
        assert result is not None


# ═══════════════════════════════════════════════════════════════════
# Test: single sub-face 3-D position
# ═══════════════════════════════════════════════════════════════════

class TestComputeSubface3DPosition:
    """Tests for compute_subface_3d_position."""

    def test_returns_on_sphere(self):
        """Computed positions should lie on the unit sphere."""
        globe, collection, _ = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import compute_subface_3d_position

        face_id = collection.face_ids[0]
        detail_grid = collection.grids[face_id]
        sub_fid = list(detail_grid.faces.keys())[0]

        pos = compute_subface_3d_position(globe, face_id, detail_grid, sub_fid)
        assert pos is not None
        length = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        assert abs(length - 1.0) < 0.01, f"Position not on unit sphere: length={length}"

    def test_center_subface_near_parent_center(self):
        """The centre sub-face should be near the parent tile's 3D centre."""
        globe, collection, _ = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import compute_subface_3d_position
        from polygrid.geometry import face_center

        face_id = collection.face_ids[0]
        detail_grid = collection.grids[face_id]
        parent_center = globe.faces[face_id].metadata["center_3d"]

        # Find the sub-face closest to local (0,0) — the centre
        best_fid = None
        best_dist = float("inf")
        for sf_id, sf in detail_grid.faces.items():
            c = face_center(detail_grid.vertices, sf)
            if c:
                d = math.sqrt(c[0]**2 + c[1]**2)
                if d < best_dist:
                    best_dist = d
                    best_fid = sf_id

        assert best_fid is not None
        pos = compute_subface_3d_position(globe, face_id, detail_grid, best_fid)
        assert pos is not None

        # Should be close to parent centre (within ~0.1 for a freq=3 tile)
        dist = math.sqrt(sum((a - b)**2 for a, b in zip(pos, parent_center)))
        assert dist < 0.15, f"Centre sub-face too far from parent: {dist}"

    def test_missing_face_returns_none(self):
        globe, collection, _ = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import compute_subface_3d_position

        face_id = collection.face_ids[0]
        detail_grid = collection.grids[face_id]
        result = compute_subface_3d_position(globe, face_id, detail_grid, "nonexistent")
        assert result is None

    def test_missing_globe_face_returns_none(self):
        globe, collection, _ = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import compute_subface_3d_position

        face_id = collection.face_ids[0]
        detail_grid = collection.grids[face_id]
        sub_fid = list(detail_grid.faces.keys())[0]
        result = compute_subface_3d_position(globe, "nonexistent", detail_grid, sub_fid)
        assert result is None


# ═══════════════════════════════════════════════════════════════════
# Test: batch 3-D position computation
# ═══════════════════════════════════════════════════════════════════

class TestPrecompute3DPositions:
    """Tests for precompute_3d_positions (single tile) and precompute_all_3d_positions."""

    def test_all_subfaces_have_positions(self):
        """Every sub-face should get a 3-D position."""
        globe, collection, _ = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import precompute_3d_positions

        face_id = collection.face_ids[0]
        detail_grid = collection.grids[face_id]
        positions = precompute_3d_positions(globe, face_id, detail_grid)

        assert len(positions) == len(detail_grid.faces)
        for sf_id, pos in positions.items():
            length = math.sqrt(sum(c**2 for c in pos))
            assert abs(length - 1.0) < 0.01

    def test_positions_are_distinct(self):
        """Different sub-faces should have different 3-D positions."""
        globe, collection, _ = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import precompute_3d_positions

        face_id = collection.face_ids[0]
        detail_grid = collection.grids[face_id]
        positions = precompute_3d_positions(globe, face_id, detail_grid)

        pos_list = list(positions.values())
        # Check a few pairs are not identical
        if len(pos_list) >= 2:
            d = math.sqrt(sum((a - b)**2 for a, b in zip(pos_list[0], pos_list[1])))
            assert d > 1e-6, "Two sub-faces have identical positions"

    def test_precompute_all_covers_every_tile(self):
        """precompute_all_3d_positions covers every face_id."""
        globe, collection, _ = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import precompute_all_3d_positions

        all_pos = precompute_all_3d_positions(collection, globe)
        assert set(all_pos.keys()) == set(collection.face_ids)

        # Every tile has positions for all its sub-faces
        for face_id, positions in all_pos.items():
            detail_grid = collection.grids[face_id]
            assert len(positions) == len(detail_grid.faces)

    def test_single_vs_batch_consistent(self):
        """Single-face and batch functions produce identical results."""
        globe, collection, _ = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import (
            compute_subface_3d_position,
            precompute_3d_positions,
        )

        face_id = collection.face_ids[0]
        detail_grid = collection.grids[face_id]
        batch = precompute_3d_positions(globe, face_id, detail_grid)

        for sf_id in list(detail_grid.faces.keys())[:5]:
            single = compute_subface_3d_position(globe, face_id, detail_grid, sf_id)
            assert single is not None
            b = batch[sf_id]
            for i in range(3):
                assert abs(single[i] - b[i]) < 1e-10, \
                    f"Mismatch for {sf_id}[{i}]: single={single[i]}, batch={b[i]}"


# ═══════════════════════════════════════════════════════════════════
# Test: terrain generation — single tile
# ═══════════════════════════════════════════════════════════════════

class TestGenerateDetailTerrain3D:
    """Tests for generate_detail_terrain_3d."""

    def test_produces_store_with_elevation(self):
        """Output store should have an elevation field for every sub-face."""
        globe, collection, globe_store = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import (
            Terrain3DSpec,
            precompute_3d_positions,
            generate_detail_terrain_3d,
        )

        face_id = collection.face_ids[0]
        detail_grid = collection.grids[face_id]
        positions = precompute_3d_positions(globe, face_id, detail_grid)
        parent_elev = globe_store.get(face_id, "elevation")

        spec = Terrain3DSpec()
        store = generate_detail_terrain_3d(detail_grid, positions, parent_elev, spec)

        assert store is not None
        for sf_id in detail_grid.faces:
            elev = store.get(sf_id, "elevation")
            assert isinstance(elev, float)

    def test_elevation_not_uniform(self):
        """Elevations should have variation (not all the same value)."""
        globe, collection, globe_store = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import (
            Terrain3DSpec,
            precompute_3d_positions,
            generate_detail_terrain_3d,
        )

        face_id = collection.face_ids[0]
        detail_grid = collection.grids[face_id]
        positions = precompute_3d_positions(globe, face_id, detail_grid)
        parent_elev = globe_store.get(face_id, "elevation")

        spec = Terrain3DSpec()
        store = generate_detail_terrain_3d(detail_grid, positions, parent_elev, spec)

        elevations = [store.get(sf_id, "elevation") for sf_id in detail_grid.faces]
        assert max(elevations) - min(elevations) > 1e-6, "No elevation variation"

    def test_deterministic(self):
        """Same inputs produce identical output."""
        globe, collection, globe_store = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import (
            Terrain3DSpec,
            precompute_3d_positions,
            generate_detail_terrain_3d,
        )

        face_id = collection.face_ids[0]
        detail_grid = collection.grids[face_id]
        positions = precompute_3d_positions(globe, face_id, detail_grid)
        parent_elev = globe_store.get(face_id, "elevation")
        spec = Terrain3DSpec()

        store_a = generate_detail_terrain_3d(detail_grid, positions, parent_elev, spec)
        store_b = generate_detail_terrain_3d(detail_grid, positions, parent_elev, spec)

        for sf_id in detail_grid.faces:
            assert store_a.get(sf_id, "elevation") == store_b.get(sf_id, "elevation")

    def test_base_weight_dominance(self):
        """With base_weight=1.0, all elevations should equal parent."""
        globe, collection, globe_store = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import (
            Terrain3DSpec,
            precompute_3d_positions,
            generate_detail_terrain_3d,
        )

        face_id = collection.face_ids[0]
        detail_grid = collection.grids[face_id]
        positions = precompute_3d_positions(globe, face_id, detail_grid)
        parent_elev = globe_store.get(face_id, "elevation")

        spec = Terrain3DSpec(base_weight=1.0, boundary_smoothing=0)
        store = generate_detail_terrain_3d(detail_grid, positions, parent_elev, spec)

        for sf_id in detail_grid.faces:
            assert abs(store.get(sf_id, "elevation") - parent_elev) < 1e-10

    def test_fbm_only(self):
        """With ridge_weight=0, only fbm is used."""
        globe, collection, globe_store = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import (
            Terrain3DSpec,
            precompute_3d_positions,
            generate_detail_terrain_3d,
        )

        face_id = collection.face_ids[0]
        detail_grid = collection.grids[face_id]
        positions = precompute_3d_positions(globe, face_id, detail_grid)
        parent_elev = globe_store.get(face_id, "elevation")

        spec = Terrain3DSpec(fbm_weight=1.0, ridge_weight=0.0, boundary_smoothing=0)
        store = generate_detail_terrain_3d(detail_grid, positions, parent_elev, spec)

        elevations = [store.get(sf_id, "elevation") for sf_id in detail_grid.faces]
        assert max(elevations) - min(elevations) > 1e-6

    def test_ridge_only(self):
        """With fbm_weight=0, only ridged noise is used."""
        globe, collection, globe_store = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import (
            Terrain3DSpec,
            precompute_3d_positions,
            generate_detail_terrain_3d,
        )

        face_id = collection.face_ids[0]
        detail_grid = collection.grids[face_id]
        positions = precompute_3d_positions(globe, face_id, detail_grid)
        parent_elev = globe_store.get(face_id, "elevation")

        spec = Terrain3DSpec(fbm_weight=0.0, ridge_weight=1.0, boundary_smoothing=0)
        store = generate_detail_terrain_3d(detail_grid, positions, parent_elev, spec)

        elevations = [store.get(sf_id, "elevation") for sf_id in detail_grid.faces]
        assert max(elevations) - min(elevations) > 1e-6


# ═══════════════════════════════════════════════════════════════════
# Test: cross-tile coherence (the KEY test)
# ═══════════════════════════════════════════════════════════════════

class TestCrossTileCoherence:
    """Tests that terrain is spatially continuous across tile boundaries.

    This is the critical test that validates the Phase 11A approach.
    Adjacent tiles should have similar elevation at their shared boundary
    because they're sampling the same global noise field.
    """

    def _get_adjacent_boundary_positions(
        self,
        globe,
        collection,
        face_a: str,
        face_b: str,
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        """Get 3-D positions of boundary sub-faces for two adjacent tiles.

        Returns two lists of 3-D positions: boundary faces from tile A
        and boundary faces from tile B that are near the shared edge.
        """
        from polygrid.detail_terrain_3d import precompute_3d_positions
        from polygrid.detail_terrain import _boundary_face_ids

        pos_a = precompute_3d_positions(globe, face_a, collection.grids[face_a])
        pos_b = precompute_3d_positions(globe, face_b, collection.grids[face_b])

        bnd_a = _boundary_face_ids(collection.grids[face_a])
        bnd_b = _boundary_face_ids(collection.grids[face_b])

        pts_a = [pos_a[fid] for fid in bnd_a if fid in pos_a]
        pts_b = [pos_b[fid] for fid in bnd_b if fid in pos_b]
        return pts_a, pts_b

    def test_adjacent_boundary_subfaces_have_nearby_positions(self):
        """Boundary sub-faces of adjacent tiles should be close in 3-D space."""
        globe, collection, _ = _build_globe_and_collection()

        # Pick first tile and one of its neighbours
        face_a = collection.face_ids[0]
        neighbor_ids = globe.faces[face_a].neighbor_ids
        assert len(neighbor_ids) > 0
        face_b = neighbor_ids[0]

        pts_a, pts_b = self._get_adjacent_boundary_positions(
            globe, collection, face_a, face_b,
        )
        assert len(pts_a) > 0 and len(pts_b) > 0

        # For each boundary point in A, find the closest point in B
        min_distances = []
        for pa in pts_a:
            dists = [math.sqrt(sum((a - b)**2 for a, b in zip(pa, pb))) for pb in pts_b]
            min_distances.append(min(dists))

        # At least some boundary faces should be close to each other
        assert min(min_distances) < 0.2, \
            f"No close boundary pairs found; min distance = {min(min_distances)}"

    def test_cross_tile_elevation_continuity(self):
        """Adjacent tiles' nearby boundary faces should have similar elevation.

        This is the key coherence test.  With 3-D noise, nearby points
        get similar noise values, so boundary sub-faces that are close
        in 3-D should have close elevations.
        """
        globe, collection, globe_store = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import (
            Terrain3DSpec,
            precompute_3d_positions,
            generate_detail_terrain_3d,
        )

        spec = Terrain3DSpec(
            noise_frequency=3.0,  # moderate frequency
            base_weight=0.5,  # give noise more influence for testability
            amplitude=0.3,
            boundary_smoothing=0,  # no smoothing — test raw continuity
        )

        # Pick an adjacent pair
        face_a = collection.face_ids[0]
        face_b = globe.faces[face_a].neighbor_ids[0]

        # Generate terrain for both tiles
        for fid in (face_a, face_b):
            dg = collection.grids[fid]
            pos = precompute_3d_positions(globe, fid, dg)
            parent_elev = globe_store.get(fid, "elevation")
            store = generate_detail_terrain_3d(dg, pos, parent_elev, spec)
            collection._stores[fid] = store

        # Collect boundary face positions and elevations
        from polygrid.detail_terrain import _boundary_face_ids

        bnd_a = _boundary_face_ids(collection.grids[face_a])
        bnd_b = _boundary_face_ids(collection.grids[face_b])
        pos_a = precompute_3d_positions(globe, face_a, collection.grids[face_a])
        pos_b = precompute_3d_positions(globe, face_b, collection.grids[face_b])
        store_a = collection._stores[face_a]
        store_b = collection._stores[face_b]

        # For each boundary face in A, find the nearest boundary face in B
        # and compare their elevations
        elevation_diffs = []
        for fa_id in bnd_a:
            if fa_id not in pos_a:
                continue
            pa = pos_a[fa_id]
            best_dist = float("inf")
            best_elev_b = None
            for fb_id in bnd_b:
                if fb_id not in pos_b:
                    continue
                pb = pos_b[fb_id]
                d = math.sqrt(sum((a - b)**2 for a, b in zip(pa, pb)))
                if d < best_dist:
                    best_dist = d
                    best_elev_b = store_b.get(fb_id, "elevation")

            # Only consider pairs that are actually close (near the shared edge)
            if best_dist < 0.12 and best_elev_b is not None:
                elev_a = store_a.get(fa_id, "elevation")
                elevation_diffs.append(abs(elev_a - best_elev_b))

        if elevation_diffs:
            mean_diff = sum(elevation_diffs) / len(elevation_diffs)
            # With coherent noise, nearby points should have small elevation diffs.
            # The parent elevation difference between adjacent tiles may cause an
            # offset, so allow for that.  The key test is that diffs are bounded.
            assert mean_diff < 0.3, \
                f"Mean cross-tile elevation diff = {mean_diff:.4f} (too large)"

    def test_no_patchwork_artefact(self):
        """Elevation variance ACROSS tile boundaries should be comparable
        to variance WITHIN tiles (no systematic discontinuity).

        The old Phase 10 approach would show high cross-tile variance
        because each tile had independent noise.  With 3-D coherent
        noise, the cross-tile variance should be of similar magnitude
        to the intra-tile variance.
        """
        globe, collection, globe_store = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import (
            Terrain3DSpec,
            generate_all_detail_terrain_3d,
            precompute_3d_positions,
        )

        spec = Terrain3DSpec(
            noise_frequency=3.0,
            base_weight=0.3,  # low base weight to amplify noise
            amplitude=0.5,
            boundary_smoothing=0,
        )
        generate_all_detail_terrain_3d(collection, globe, globe_store, spec)

        # Compute intra-tile variance (variance of elevations within each tile)
        import statistics
        intra_variances = []
        for face_id in collection.face_ids[:10]:  # sample a few tiles
            store = collection._stores[face_id]
            grid = collection.grids[face_id]
            elevs = [store.get(sf, "elevation") for sf in grid.faces]
            if len(elevs) >= 2:
                intra_variances.append(statistics.variance(elevs))

        mean_intra = sum(intra_variances) / len(intra_variances) if intra_variances else 0

        # Compute cross-tile variance at boundaries
        from polygrid.detail_terrain import _boundary_face_ids
        cross_diffs_sq = []
        checked = 0
        for face_a in collection.face_ids[:10]:
            for face_b in globe.faces[face_a].neighbor_ids:
                if face_b not in collection._stores:
                    continue
                pos_a = precompute_3d_positions(globe, face_a, collection.grids[face_a])
                pos_b = precompute_3d_positions(globe, face_b, collection.grids[face_b])
                bnd_a = _boundary_face_ids(collection.grids[face_a])
                store_a = collection._stores[face_a]
                store_b = collection._stores[face_b]

                for fa_id in bnd_a:
                    if fa_id not in pos_a:
                        continue
                    pa = pos_a[fa_id]
                    best_dist = float("inf")
                    best_elev = None
                    for fb_id in _boundary_face_ids(collection.grids[face_b]):
                        if fb_id not in pos_b:
                            continue
                        d = math.sqrt(sum((a-b)**2 for a, b in zip(pa, pos_b[fb_id])))
                        if d < best_dist:
                            best_dist = d
                            best_elev = store_b.get(fb_id, "elevation")
                    if best_dist < 0.12 and best_elev is not None:
                        ea = store_a.get(fa_id, "elevation")
                        cross_diffs_sq.append((ea - best_elev) ** 2)
                        checked += 1

        if cross_diffs_sq and mean_intra > 0:
            mean_cross_var = sum(cross_diffs_sq) / len(cross_diffs_sq)
            # Cross-tile variance should not be dramatically larger than intra-tile
            # Allow a generous factor since parent elevation differences contribute
            ratio = mean_cross_var / mean_intra if mean_intra > 1e-10 else 0
            # Even a ratio of 10 would indicate far less patchwork than the
            # old approach (which would be 100+).  Use a generous threshold.
            assert ratio < 50, \
                f"Cross/intra variance ratio = {ratio:.2f} (patchwork artefact?)"


# ═══════════════════════════════════════════════════════════════════
# Test: batch generation
# ═══════════════════════════════════════════════════════════════════

class TestGenerateAllDetailTerrain3D:
    """Tests for generate_all_detail_terrain_3d."""

    def test_populates_all_stores(self):
        """Every tile should get a terrain store."""
        globe, collection, globe_store = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import (
            Terrain3DSpec,
            generate_all_detail_terrain_3d,
        )

        generate_all_detail_terrain_3d(collection, globe, globe_store)

        for face_id in collection.face_ids:
            grid, store = collection.get(face_id)
            assert store is not None, f"No store for {face_id}"
            for sf_id in grid.faces:
                elev = store.get(sf_id, "elevation")
                assert isinstance(elev, float)

    def test_default_spec(self):
        """Works with default Terrain3DSpec."""
        globe, collection, globe_store = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import generate_all_detail_terrain_3d

        generate_all_detail_terrain_3d(collection, globe, globe_store)
        assert len(collection.stores) == len(collection.face_ids)

    def test_custom_spec(self):
        """Works with custom Terrain3DSpec."""
        globe, collection, globe_store = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import (
            Terrain3DSpec,
            generate_all_detail_terrain_3d,
        )

        spec = Terrain3DSpec(
            noise_frequency=6.0,
            ridge_frequency=4.0,
            fbm_weight=0.3,
            ridge_weight=0.7,
            base_weight=0.5,
            amplitude=0.2,
        )
        generate_all_detail_terrain_3d(collection, globe, globe_store, spec)
        assert len(collection.stores) == len(collection.face_ids)

    def test_deterministic_batch(self):
        """Two runs with same inputs produce identical results."""
        globe, coll_a, globe_store = _build_globe_and_collection()
        _, coll_b, _ = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import (
            Terrain3DSpec,
            generate_all_detail_terrain_3d,
        )

        spec = Terrain3DSpec(seed=123)
        generate_all_detail_terrain_3d(coll_a, globe, globe_store, spec)
        generate_all_detail_terrain_3d(coll_b, globe, globe_store, spec)

        for face_id in coll_a.face_ids:
            grid = coll_a.grids[face_id]
            for sf_id in grid.faces:
                a = coll_a._stores[face_id].get(sf_id, "elevation")
                b = coll_b._stores[face_id].get(sf_id, "elevation")
                assert a == b, f"Non-deterministic at {face_id}/{sf_id}"


# ═══════════════════════════════════════════════════════════════════
# Test: performance
# ═══════════════════════════════════════════════════════════════════

class TestPerformance:
    """Ensure the 3-D terrain gen is not unreasonably slow."""

    def test_single_tile_under_1_second(self):
        globe, collection, globe_store = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import (
            Terrain3DSpec,
            precompute_3d_positions,
            generate_detail_terrain_3d,
        )

        face_id = collection.face_ids[0]
        detail_grid = collection.grids[face_id]
        positions = precompute_3d_positions(globe, face_id, detail_grid)
        parent_elev = globe_store.get(face_id, "elevation")
        spec = Terrain3DSpec()

        start = time.perf_counter()
        for _ in range(10):
            generate_detail_terrain_3d(detail_grid, positions, parent_elev, spec)
        elapsed = time.perf_counter() - start

        per_tile = elapsed / 10
        assert per_tile < 1.0, f"Single tile took {per_tile:.3f}s (too slow)"

    def test_full_globe_under_30_seconds(self):
        """Full freq=3 globe terrain gen should complete in reasonable time."""
        globe, collection, globe_store = _build_globe_and_collection()
        from polygrid.detail_terrain_3d import (
            Terrain3DSpec,
            generate_all_detail_terrain_3d,
        )

        spec = Terrain3DSpec()
        start = time.perf_counter()
        generate_all_detail_terrain_3d(collection, globe, globe_store, spec)
        elapsed = time.perf_counter() - start

        assert elapsed < 30.0, f"Full globe took {elapsed:.1f}s (too slow)"
        print(f"  Full globe (freq=3, rings=3): {elapsed:.2f}s")
