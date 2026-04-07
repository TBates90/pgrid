from __future__ import annotations

import math

import pytest

from polygrid.rendering.detail_cell_contract import (
    normalize_detail_cells_tiles,
    normalize_detail_cells_tiles_with_report,
)


def test_normalize_detail_cells_tiles_filters_invalid_entries() -> None:
    result = normalize_detail_cells_tiles(
        {
            "F2:0-0-0": [
                {"id": "cell-a", "center_3d": [0.0, 0.0, 2.0]},
                {"id": "", "center_3d": [1.0, 0.0, 0.0]},
                {"id": "cell-b", "center_3d": [0.0, 0.0, 0.0]},
                {"id": "cell-c", "center_3d": [math.nan, 0.0, 1.0]},
                {"id": "cell-d"},
            ],
            "": [{"id": "ignored", "center_3d": [0.0, 1.0, 0.0]}],
            "f3:bad": "not-a-list",
        }
    )

    assert set(result.keys()) == {"f2:0-0-0"}
    cells = result["f2:0-0-0"]
    assert len(cells) == 1
    assert cells[0]["id"] == "cell-a"
    assert cells[0]["center_3d"] == [0.0, 0.0, 1.0]
    assert cells[0]["canonical_center_3d"] == [0.0, 0.0, 1.0]


def test_normalize_detail_cells_tiles_repairs_non_contiguous_indices() -> None:
    result = normalize_detail_cells_tiles(
        {
            "f4:0-0-0": [
                {"id": "c1", "center_3d": [1.0, 0.0, 0.0], "detail_index": 5},
                {"id": "c2", "center_3d": [0.0, 1.0, 0.0], "detail_index": 5},
                {"id": "c3", "center_3d": [0.0, 0.0, 1.0]},
            ]
        }
    )

    indices = [cell["detail_index"] for cell in result["f4:0-0-0"]]
    assert indices == [1, 2, 3]


def test_normalize_detail_cells_tiles_inferrs_sides_from_vertices() -> None:
    result = normalize_detail_cells_tiles(
        {
            "f7:0-0-0": [
                {
                    "id": "cell-1",
                    "canonical_center_3d": [0.0, 0.0, 2.0],
                    "vertices_3d": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                    ],
                }
            ]
        }
    )

    payload = result["f7:0-0-0"][0]
    assert payload["sides"] == 4
    assert len(payload["vertices_3d"]) == 4


def test_normalize_detail_cells_tiles_reports_repairs_and_drops() -> None:
    tiles, report = normalize_detail_cells_tiles_with_report(
        {
            "F3:0-0-0": [
                {"id": "c1", "center_3d": [1.0, 0.0, 0.0], "detail_index": 2},
                {"id": "c2", "center_3d": [0.0, 1.0, 0.0], "detail_index": 2},
                {"id": "", "center_3d": [0.0, 0.0, 1.0]},
            ],
            "tile-invalid": "bad-payload",
        }
    )

    report_payload = report.to_dict()
    assert set(tiles.keys()) == {"f3:0-0-0"}
    assert report_payload["tiles_seen"] == 2
    assert report_payload["tiles_emitted"] == 1
    assert report_payload["invalid_tile_entries"] == 1
    assert report_payload["cells_seen"] == 3
    assert report_payload["cells_dropped"] == 1
    assert report_payload["repaired_index_tiles"] == 1
    assert report_payload["repaired_index_cells"] == 1


def test_normalize_detail_cells_tiles_strict_mode_rejects_adjustments() -> None:
    with pytest.raises(ValueError, match="strict contract"):
        normalize_detail_cells_tiles_with_report(
            {
                "f3:0-0-0": [
                    {"id": "", "center_3d": [1.0, 0.0, 0.0]},
                    {"id": "c2", "center_3d": [0.0, 1.0, 0.0], "detail_index": 3},
                ]
            },
            strict=True,
        )
