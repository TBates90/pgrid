from __future__ import annotations

from polygrid.rendering.seam_strips import (
    build_seam_strip_manifest,
    build_seam_strip_payload,
    build_seam_strip_payload_from_globe_payload,
    canonical_seam_id,
)


def test_canonical_seam_id_is_order_independent() -> None:
    assert canonical_seam_id("F3:A", "f3:b") == "seam:f3:a|f3:b"
    assert canonical_seam_id("f3:b", "F3:A") == "seam:f3:a|f3:b"


def test_build_seam_strip_manifest_deduplicates_and_sorts_pairs() -> None:
    manifest = build_seam_strip_manifest(
        {
            "f2:0-0-0": ["f2:0-0-1", "f2:0-0-2"],
            "f2:0-0-1": ["f2:0-0-0"],
            "f2:0-0-2": ["f2:0-0-0"],
        }
    )

    seam_ids = [entry["seam_id"] for entry in manifest]
    assert seam_ids == sorted(seam_ids)
    assert seam_ids == [
        "seam:f2:0-0-0|f2:0-0-1",
        "seam:f2:0-0-0|f2:0-0-2",
    ]


def test_build_seam_strip_payload_envelope_contains_metadata() -> None:
    payload = build_seam_strip_payload(
        {
            "f2:0-0-0": ["f2:0-0-1"],
            "f2:0-0-1": ["f2:0-0-0"],
        },
        frequency=2,
        detail_rings=3,
        tile_centers={
            "f2:0-0-0": [1.0, 0.0, 0.0],
            "f2:0-0-1": [0.0, 1.0, 0.0],
        },
    )

    assert payload["metadata"]["frequency"] == 2
    assert payload["metadata"]["detail_rings"] == 3
    assert payload["metadata"]["seam_count"] == 1
    assert payload["metadata"]["geometry_count"] == 1
    assert payload["metadata"]["edge_geometry_count"] == 0
    assert payload["metadata"]["fallback_geometry_count"] == 1
    assert payload["metadata"]["schema"] == "seam-strips.v1"
    assert len(payload["seams"]) == 1
    seam = payload["seams"][0]
    assert seam["status"] == "geometry"
    assert len(seam["corners_3d"]) == 4


def test_build_seam_strip_payload_from_globe_payload_uses_slug_neighbors() -> None:
    payload = build_seam_strip_payload_from_globe_payload(
        {
            "tiles": [
                {
                    "id": "t0",
                    "tile_slug": "f2:0-0-0",
                    "center_3d": [1.0, 0.0, 0.0],
                    "normal_3d": [1.0, 0.0, 0.0],
                    "vertices_3d": [
                        [1.0, 0.0, 0.0],
                        [0.70710678, 0.70710678, 0.0],
                        [0.70710678, 0.0, 0.70710678],
                    ],
                    "neighbor_ids": ["t1"],
                },
                {
                    "id": "t1",
                    "tile_slug": "f2:0-0-1",
                    "center_3d": [0.0, 1.0, 0.0],
                    "normal_3d": [0.0, 1.0, 0.0],
                    "vertices_3d": [
                        [0.0, 1.0, 0.0],
                        [0.70710678, 0.70710678, 0.0],
                        [0.70710678, 0.0, 0.70710678],
                        [0.0, 0.70710678, 0.70710678],
                    ],
                    "neighbor_ids": ["t0"],
                },
            ],
            "adjacency": [["t0", "t1"]],
        },
        frequency=2,
        detail_rings=2,
    )

    assert payload["metadata"]["seam_count"] == 1
    assert payload["metadata"]["geometry_count"] == 1
    assert payload["metadata"]["edge_geometry_count"] == 1
    assert payload["metadata"]["fallback_geometry_count"] == 0
    seam = payload["seams"][0]
    assert seam["seam_id"] == "seam:f2:0-0-0|f2:0-0-1"
    assert seam["status"] == "edge-geometry"
    assert len(seam["edge_vertices_3d"]) == 2
