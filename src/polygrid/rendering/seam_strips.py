"""Seam-strip artifact scaffolding for child-grid seam integration.

Phase 2 kickoff: generate deterministic seam identifiers + metadata records
from tile adjacency, so downstream generators/renderers can attach geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SeamStripRecord:
    seam_id: str
    tile_a: str
    tile_b: str
    edge_key: str
    status: str = "placeholder"

    def to_dict(self) -> dict[str, str]:
        return {
            "seam_id": self.seam_id,
            "tile_a": self.tile_a,
            "tile_b": self.tile_b,
            "edge_key": self.edge_key,
            "status": self.status,
        }


def canonical_seam_id(tile_a: str, tile_b: str) -> str:
    """Return a deterministic seam ID for an unordered tile pair."""

    left, right = sorted((str(tile_a).strip().lower(), str(tile_b).strip().lower()))
    return f"seam:{left}|{right}"


def build_seam_strip_manifest(
    tile_neighbors: Mapping[str, Sequence[str]] | None,
) -> list[dict[str, str]]:
    """Build deterministic seam-strip metadata records from adjacency map."""

    if not isinstance(tile_neighbors, Mapping):
        return []

    seams: dict[str, SeamStripRecord] = {}
    for tile_id, raw_neighbors in tile_neighbors.items():
        base = str(tile_id).strip().lower()
        if not base:
            continue
        if not isinstance(raw_neighbors, Sequence) or isinstance(raw_neighbors, (str, bytes, bytearray)):
            continue
        for neighbor_id in raw_neighbors:
            neighbor = str(neighbor_id).strip().lower()
            if not neighbor or neighbor == base:
                continue
            seam_id = canonical_seam_id(base, neighbor)
            if seam_id in seams:
                continue
            left, right = sorted((base, neighbor))
            seams[seam_id] = SeamStripRecord(
                seam_id=seam_id,
                tile_a=left,
                tile_b=right,
                edge_key=f"{left}->{right}",
            )

    ordered = sorted(seams.values(), key=lambda item: item.seam_id)
    return [record.to_dict() for record in ordered]


def build_seam_strip_payload(
    tile_neighbors: Mapping[str, Sequence[str]] | None,
    *,
    frequency: int,
    detail_rings: int,
    tile_centers: Mapping[str, Sequence[float]] | None = None,
    tile_vertices: Mapping[str, Sequence[Sequence[float]]] | None = None,
    tile_normals: Mapping[str, Sequence[float]] | None = None,
    half_width: float = 0.008,
    half_length: float = 0.025,
) -> dict[str, Any]:
    """Build a seam-strip payload envelope with deterministic metadata."""

    seams = build_seam_strip_manifest(tile_neighbors)
    geometry = build_seam_strip_geometry(
        seams,
        tile_centers,
        tile_vertices,
        tile_normals,
        half_width=float(half_width),
        half_length=float(half_length),
    )
    edge_geometry_count = sum(1 for seam in geometry if seam.get("status") == "edge-geometry")
    fallback_geometry_count = sum(1 for seam in geometry if seam.get("status") == "geometry")
    return {
        "metadata": {
            "frequency": int(frequency),
            "detail_rings": int(detail_rings),
            "seam_count": int(len(seams)),
            "geometry_count": int(len(geometry)),
            "edge_geometry_count": int(edge_geometry_count),
            "fallback_geometry_count": int(fallback_geometry_count),
            "schema": "seam-strips.v1",
        },
        "seams": geometry,
    }


def build_seam_strip_payload_from_globe_payload(
    globe_payload: Mapping[str, Any] | None,
    *,
    frequency: int,
    detail_rings: int,
    half_width: float = 0.008,
    half_length: float = 0.025,
) -> dict[str, Any]:
    """Build seam-strip payload from exported globe payload tiles/adjacency."""

    if not isinstance(globe_payload, Mapping):
        return build_seam_strip_payload(
            {},
            frequency=frequency,
            detail_rings=detail_rings,
            half_width=half_width,
            half_length=half_length,
        )

    tile_entries = globe_payload.get("tiles")
    if not isinstance(tile_entries, Sequence) or isinstance(tile_entries, (str, bytes, bytearray)):
        tile_entries = []

    face_to_slug: dict[str, str] = {}
    centers: dict[str, Sequence[float]] = {}
    normals: dict[str, Sequence[float]] = {}
    vertices: dict[str, Sequence[Sequence[float]]] = {}
    neighbors: dict[str, list[str]] = {}

    for entry in tile_entries:
        if not isinstance(entry, Mapping):
            continue
        face_id = _token(entry.get("id"))
        slug = _token(entry.get("tile_slug")) or face_id
        if not face_id or not slug:
            continue
        face_to_slug[face_id] = slug
        center = entry.get("center_3d")
        if isinstance(center, Sequence) and not isinstance(center, (str, bytes, bytearray)) and len(center) >= 3:
            centers[slug] = center
        normal = entry.get("normal_3d")
        if isinstance(normal, Sequence) and not isinstance(normal, (str, bytes, bytearray)) and len(normal) >= 3:
            normals[slug] = normal
        verts = entry.get("vertices_3d")
        if isinstance(verts, Sequence) and not isinstance(verts, (str, bytes, bytearray)) and len(verts) >= 3:
            vertices[slug] = verts

    for entry in tile_entries:
        if not isinstance(entry, Mapping):
            continue
        face_id = _token(entry.get("id"))
        if not face_id:
            continue
        base_slug = face_to_slug.get(face_id)
        if not base_slug:
            continue
        raw_neighbors = entry.get("neighbor_ids")
        if not isinstance(raw_neighbors, Sequence) or isinstance(raw_neighbors, (str, bytes, bytearray)):
            continue
        mapped = []
        for neighbor in raw_neighbors:
            neighbor_face = _token(neighbor)
            if not neighbor_face:
                continue
            mapped_slug = face_to_slug.get(neighbor_face)
            if mapped_slug:
                mapped.append(mapped_slug)
        if mapped:
            neighbors[base_slug] = mapped

    return build_seam_strip_payload(
        neighbors,
        frequency=frequency,
        detail_rings=detail_rings,
        tile_centers=centers,
        tile_vertices=vertices,
        tile_normals=normals,
        half_width=half_width,
        half_length=half_length,
    )


def build_seam_strip_geometry(
    manifest: Sequence[Mapping[str, Any]] | None,
    tile_centers: Mapping[str, Sequence[float]] | None,
    tile_vertices: Mapping[str, Sequence[Sequence[float]]] | None,
    tile_normals: Mapping[str, Sequence[float]] | None,
    *,
    half_width: float,
    half_length: float,
) -> list[dict[str, Any]]:
    """Attach deterministic sphere-anchored strip geometry for seams."""

    if not isinstance(manifest, Sequence) or isinstance(manifest, (str, bytes, bytearray)):
        return []
    if not isinstance(tile_centers, Mapping):
        return [dict(item) for item in manifest if isinstance(item, Mapping)]

    out: list[dict[str, Any]] = []
    for item in manifest:
        if not isinstance(item, Mapping):
            continue
        tile_a = _token(item.get("tile_a"))
        tile_b = _token(item.get("tile_b"))
        if not tile_a or not tile_b:
            continue

        edge_geometry = _build_edge_aligned_geometry(
            tile_a,
            tile_b,
            tile_vertices,
            tile_normals,
            half_width=half_width,
        )
        if edge_geometry is not None:
            enriched = dict(item)
            enriched.update(edge_geometry)
            enriched["source"] = "shared-edge"
            enriched["status"] = "edge-geometry"
            out.append(enriched)
            continue

        a = _vec3(tile_centers.get(tile_a))
        b = _vec3(tile_centers.get(tile_b))
        if a is None or b is None:
            out.append(dict(item))
            continue

        midpoint = _normalize(a + b)
        tangent = b - a
        tangent = tangent - np.dot(tangent, midpoint) * midpoint
        tangent = _normalize(tangent)
        bitangent = _normalize(np.cross(midpoint, tangent))
        if tangent is None or bitangent is None:
            out.append(dict(item))
            continue

        c0 = _normalize(midpoint - tangent * half_length - bitangent * half_width)
        c1 = _normalize(midpoint + tangent * half_length - bitangent * half_width)
        c2 = _normalize(midpoint + tangent * half_length + bitangent * half_width)
        c3 = _normalize(midpoint - tangent * half_length + bitangent * half_width)
        if c0 is None or c1 is None or c2 is None or c3 is None:
            out.append(dict(item))
            continue

        enriched = dict(item)
        enriched["center_3d"] = _round_vec(midpoint)
        enriched["tangent_3d"] = _round_vec(tangent)
        enriched["bitangent_3d"] = _round_vec(bitangent)
        enriched["corners_3d"] = [_round_vec(c0), _round_vec(c1), _round_vec(c2), _round_vec(c3)]
        enriched["source"] = "center-pair"
        enriched["status"] = "geometry"
        out.append(enriched)

    return out


def _token(value: Any) -> str | None:
    token = str(value or "").strip().lower()
    return token or None


def _vec3(value: Any) -> np.ndarray | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) < 3:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
        z = float(value[2])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    return np.array((x, y, z), dtype=np.float64)


def _normalize(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    mag = float(np.linalg.norm(value))
    if mag <= 1e-9:
        return None
    return value / mag


def _round_vec(value: np.ndarray) -> list[float]:
    return [round(float(value[0]), 8), round(float(value[1]), 8), round(float(value[2]), 8)]


def _build_edge_aligned_geometry(
    tile_a: str,
    tile_b: str,
    tile_vertices: Mapping[str, Sequence[Sequence[float]]] | None,
    tile_normals: Mapping[str, Sequence[float]] | None,
    *,
    half_width: float,
) -> dict[str, Any] | None:
    if not isinstance(tile_vertices, Mapping):
        return None
    verts_a = _vertices3(tile_vertices.get(tile_a))
    verts_b = _vertices3(tile_vertices.get(tile_b))
    if len(verts_a) < 3 or len(verts_b) < 3:
        return None

    edge = _shared_edge_vertices(verts_a, verts_b, tolerance=1e-5)
    if edge is None:
        return None
    e0, e1 = edge
    tangent = _normalize(e1 - e0)
    if tangent is None:
        return None

    normal = _average_normal(tile_a, tile_b, tile_normals)
    if normal is None:
        normal = _normalize(e0 + e1)
    if normal is None:
        return None

    bitangent = _normalize(np.cross(normal, tangent))
    if bitangent is None:
        return None

    c0 = _normalize(e0 - bitangent * half_width)
    c1 = _normalize(e1 - bitangent * half_width)
    c2 = _normalize(e1 + bitangent * half_width)
    c3 = _normalize(e0 + bitangent * half_width)
    if c0 is None or c1 is None or c2 is None or c3 is None:
        return None

    midpoint = _normalize((e0 + e1) * 0.5)
    if midpoint is None:
        return None

    return {
        "center_3d": _round_vec(midpoint),
        "tangent_3d": _round_vec(tangent),
        "bitangent_3d": _round_vec(bitangent),
        "edge_vertices_3d": [_round_vec(e0), _round_vec(e1)],
        "corners_3d": [_round_vec(c0), _round_vec(c1), _round_vec(c2), _round_vec(c3)],
    }


def _vertices3(values: Any) -> list[np.ndarray]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return []
    out: list[np.ndarray] = []
    for value in values:
        vec = _vec3(value)
        nvec = _normalize(vec)
        if nvec is not None:
            out.append(nvec)
    return out


def _shared_edge_vertices(
    verts_a: Sequence[np.ndarray],
    verts_b: Sequence[np.ndarray],
    *,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    count_a = len(verts_a)
    for idx in range(count_a):
        a0 = verts_a[idx]
        a1 = verts_a[(idx + 1) % count_a]
        b0 = _match_vertex_index(verts_b, a0, tolerance=tolerance)
        b1 = _match_vertex_index(verts_b, a1, tolerance=tolerance)
        if b0 is None or b1 is None:
            continue
        if _is_adjacent_index_pair(len(verts_b), b0, b1):
            return a0, a1
    return None


def _match_vertex_index(
    candidates: Sequence[np.ndarray],
    target: np.ndarray,
    *,
    tolerance: float,
) -> int | None:
    best_idx: int | None = None
    best_dist = float("inf")
    for idx, candidate in enumerate(candidates):
        dist = float(np.linalg.norm(candidate - target))
        if dist <= tolerance and dist < best_dist:
            best_idx = idx
            best_dist = dist
    return best_idx


def _is_adjacent_index_pair(count: int, left: int, right: int) -> bool:
    if count <= 1:
        return False
    if left == right:
        return False
    diff = abs(left - right)
    return diff == 1 or diff == (count - 1)


def _average_normal(
    tile_a: str,
    tile_b: str,
    tile_normals: Mapping[str, Sequence[float]] | None,
) -> np.ndarray | None:
    if not isinstance(tile_normals, Mapping):
        return None
    na = _normalize(_vec3(tile_normals.get(tile_a)))
    nb = _normalize(_vec3(tile_normals.get(tile_b)))
    if na is None and nb is None:
        return None
    if na is None:
        return nb
    if nb is None:
        return na
    return _normalize(na + nb)
