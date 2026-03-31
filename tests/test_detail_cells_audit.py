from __future__ import annotations

from polygrid.rendering.detail_cells_audit import audit_detail_cells_payload


def test_audit_payload_reports_full_coverage_for_canonical_cells():
    payload = {
        "metadata": {"frequency": 3, "detail_rings": 2},
        "tiles": {
            "3:f4:3-0-0": [
                {
                    "id": "f1",
                    "detail_index": 1,
                    "ring_index": 0,
                    "position_in_ring": 0,
                },
                {
                    "id": "f2",
                    "detail_index": 2,
                    "ring_index": 1,
                    "position_in_ring": 0,
                },
                {
                    "id": "f3",
                    "detail_index": 3,
                    "ring_index": 1,
                    "position_in_ring": 1,
                },
            ]
        },
    }

    report = audit_detail_cells_payload(payload)
    assert report["tiles"] == 1
    assert report["cells"] == 3
    assert report["fallback_tiles"] == 0
    assert report["fallback_cells"] == 0
    assert report["non_contiguous_tiles"] == 0
    assert report["canonical_index_coverage"] == 1.0


def test_audit_payload_flags_missing_and_non_contiguous_indices():
    payload = {
        "metadata": {"frequency": 3, "detail_rings": 2},
        "tiles": {
            "tile-a": [
                {"id": "f1", "detail_index": 1, "ring_index": 0, "position_in_ring": 0},
                {"id": "f2", "ring_index": 1, "position_in_ring": 0},
            ],
            "tile-b": [
                {"id": "f1", "detail_index": 1, "ring_index": 0, "position_in_ring": 0},
                {"id": "f2", "detail_index": 3, "ring_index": 1, "position_in_ring": 0},
            ],
        },
    }

    report = audit_detail_cells_payload(payload)
    assert report["tiles"] == 2
    assert report["cells"] == 4
    assert report["fallback_tiles"] == 2
    assert report["fallback_cells"] == 1
    assert report["non_contiguous_tiles"] == 1
    assert report["canonical_index_coverage"] == 0.75

    per_tile = {item["tile_id"]: item for item in report["tile_results"]}
    assert per_tile["tile-a"]["needs_fallback"] is True
    assert per_tile["tile-a"]["missing_detail_index"] == 1
    assert per_tile["tile-b"]["needs_fallback"] is True
    assert per_tile["tile-b"]["non_contiguous_detail_index"] is True
