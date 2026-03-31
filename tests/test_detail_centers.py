from __future__ import annotations

import math

import pytest

from polygrid.rendering.detail_centers import compute_detail_cell_centers_3d

try:
    from polygrid.globe import _HAS_MODELS
except Exception:
    _HAS_MODELS = False

needs_models = pytest.mark.skipif(not _HAS_MODELS, reason="models library not installed")


@needs_models
def test_detail_centers_include_canonical_and_ring_metadata():
    from conftest import cached_build_globe

    globe = cached_build_globe(2)
    if globe is None:
        pytest.skip("models library not installed")

    face_id = sorted(globe.faces.keys())[0]
    cells = compute_detail_cell_centers_3d(globe, face_id, detail_rings=2)

    assert cells
    center_count = 0
    detail_indices: list[int] = []
    for cell in cells:
        assert "canonical_center_3d" in cell
        assert "ring_index" in cell
        assert "position_in_ring" in cell
        assert "detail_index" in cell

        canonical = cell["canonical_center_3d"]
        assert isinstance(canonical, list)
        assert len(canonical) == 3
        mag = math.sqrt(sum(float(v) * float(v) for v in canonical))
        assert mag == pytest.approx(1.0, abs=1e-6)

        if int(cell["ring_index"]) == 0:
            center_count += 1
            assert int(cell["position_in_ring"]) == 0

        detail_indices.append(int(cell["detail_index"]))

    # Exactly one center cell in a connected detail grid.
    assert center_count == 1
    # Canonical detail indices are 1-based and contiguous.
    assert sorted(detail_indices) == list(range(1, len(cells) + 1))
