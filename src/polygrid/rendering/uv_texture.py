"""UV-aligned texture rendering — Phase 20A+B.

Renders tile textures using the **same projection** that the 3D mesh
uses for UV mapping.  This eliminates the coordinate-system mismatch
between the apron grid's 2D Tutte-embedding positions and the
GoldbergTile's tangent-plane projection (which defines the atlas UVs).

The key idea: the 3D mesh builder calls ``generate_goldberg_tiles()``
from the models library to get per-tile ``uv_vertices`` (and the
``tangent`` / ``bitangent`` basis that produced them).  We use those
*exact same* GoldbergTile objects to derive the mapping from the
detail grid's 2D positions into the mesh's UV space.

Phase 20A — UV-aligned polygon warp (exact at corners, piecewise-linear).
Phase 20B — Shared-edge stitching: after rendering each tile independently,
            boundary pixels along shared Goldberg edges are averaged so that
            adjacent atlas slots have identical content at every seam.  This
            eliminates the visible tile disjointness on the 3D globe.

Functions
---------
- :func:`get_goldberg_tiles`             — cached access to GoldbergTile list
- :func:`compute_detail_to_uv_transform` — affine mapping detail-2D → tile-UV
- :func:`render_tile_uv_aligned`         — rasterise an apron grid in UV-aligned space
- :func:`build_uv_aligned_atlas`         — full atlas pipeline with UV alignment + edge stitching
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..detail.detail_render import BiomeConfig, detail_elevation_to_colour, _detail_hillshade
from ..core.geometry import face_center
from ..core.polygrid import PolyGrid
from ..data.tile_data import TileDataStore
from ..detail.tile_detail import DetailGridCollection

if TYPE_CHECKING:
    from PIL import Image

# Sentinel colour for background pixels.
_BG_SENTINEL = (255, 0, 255)

# ── Models library availability ─────────────────────────────────
try:
    from models.objects.goldberg import generate_goldberg_tiles as _gen_tiles
    _HAS_MODELS = True
except ImportError:
    _HAS_MODELS = False


# ═══════════════════════════════════════════════════════════════════
# 20A.1 — Authoritative GoldbergTile access
# ═══════════════════════════════════════════════════════════════════

def _normalize_vec(v: np.ndarray) -> np.ndarray:
    """Normalise a vector, returning zero-vector on degenerate input."""
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


@lru_cache(maxsize=8)
def get_goldberg_tiles(frequency: int, radius: float = 1.0):
    """Return the GoldbergTile tuple from the models library (cached).

    These are the **exact same** tile objects that the mesh builder
    (``build_batched_globe_mesh``) uses, so their ``uv_vertices``,
    ``tangent``, ``bitangent``, ``center``, and ``vertices`` are
    authoritative for UV alignment.
    """
    if not _HAS_MODELS:
        raise ImportError(
            "models library is required for UV-aligned rendering. "
            "Install it or ensure it is on PYTHONPATH."
        )
    return _gen_tiles(frequency=frequency, radius=radius)


def _match_tile_to_face(tiles, face_id: str):
    """Find the GoldbergTile whose index matches a face id like 't42'."""
    idx = int(face_id.replace("t", ""))
    return tiles[idx]


def compute_tile_basis(
    globe_grid: PolyGrid,
    face_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive tangent, bitangent, normal, and center for a globe tile.

    When the models library is available, returns the **authoritative**
    basis from the GoldbergTile (same as the mesh builder uses).
    Falls back to deriving from globe_grid vertices otherwise.

    Parameters
    ----------
    globe_grid : PolyGrid (GlobeGrid)
        Must have 3D vertex positions (x, y, z) and metadata with
        ``frequency`` and ``radius``.
    face_id : str
        E.g. ``"t0"``, ``"t1"``.

    Returns
    -------
    (center, normal, tangent, bitangent)
        All as numpy arrays of shape (3,).
    """
    if _HAS_MODELS:
        freq = globe_grid.metadata.get("frequency", 3)
        rad = globe_grid.metadata.get("radius", 1.0)
        tiles = get_goldberg_tiles(freq, rad)
        tile = _match_tile_to_face(tiles, face_id)
        return (
            np.array(tile.center, dtype=np.float64),
            np.array(tile.normal, dtype=np.float64),
            np.array(tile.tangent, dtype=np.float64),
            np.array(tile.bitangent, dtype=np.float64),
        )

    # Fallback: derive from globe_grid polygon vertices
    return _compute_tile_basis_from_grid(globe_grid, face_id)


def _compute_tile_basis_from_grid(
    globe_grid: PolyGrid,
    face_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fallback basis derivation from PolyGrid vertices."""
    face = globe_grid.faces[face_id]
    verts_3d = []
    for vid in face.vertex_ids:
        v = globe_grid.vertices[vid]
        verts_3d.append(np.array([v.x, v.y, v.z], dtype=np.float64))

    center = np.mean(verts_3d, axis=0)
    a = verts_3d[1] - verts_3d[0]
    b = verts_3d[2] - verts_3d[0]
    normal = _normalize_vec(np.cross(a, b))
    edge = verts_3d[1] - verts_3d[0]
    edge_proj = edge - normal * float(np.dot(edge, normal))
    if np.linalg.norm(edge_proj) < 1e-6:
        axis = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(axis, normal))) > 0.95:
            axis = np.array([0.0, 1.0, 0.0])
        edge_proj = axis - normal * float(np.dot(axis, normal))
    tangent = _normalize_vec(edge_proj)
    bitangent = _normalize_vec(np.cross(normal, tangent))
    return center, normal, tangent, bitangent


def get_tile_uv_vertices(
    globe_grid: PolyGrid,
    face_id: str,
) -> List[Tuple[float, float]]:
    """Return the authoritative normalised UV polygon for a tile.

    These are the exact ``GoldbergTile.uv_vertices`` that the mesh
    builder uses.  The texture must be arranged so that these UV
    coordinates sample the correct content.
    """
    if not _HAS_MODELS:
        raise ImportError("models library is required")
    freq = globe_grid.metadata.get("frequency", 3)
    rad = globe_grid.metadata.get("radius", 1.0)
    tiles = get_goldberg_tiles(freq, rad)
    tile = _match_tile_to_face(tiles, face_id)
    return list(tile.uv_vertices)


def project_point_to_tile_uv(
    point_3d: np.ndarray,
    center: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
) -> Tuple[float, float]:
    """Project a 3D point onto the tile's tangent plane.

    Returns raw (u, v) in tangent-plane coordinates (not yet normalised).
    """
    rel = point_3d - center
    return (float(np.dot(rel, tangent)), float(np.dot(rel, bitangent)))


def compute_tile_uv_bounds(
    globe_grid: PolyGrid,
    face_id: str,
    center: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute the min/max UV bounds for a tile's polygon vertices.

    When the models library is available, uses the **authoritative**
    GoldbergTile vertices (same polygon the mesh uses) rather than
    the globe_grid vertices (which may have different vertex ordering).
    """
    if _HAS_MODELS:
        freq = globe_grid.metadata.get("frequency", 3)
        rad = globe_grid.metadata.get("radius", 1.0)
        tiles = get_goldberg_tiles(freq, rad)
        tile = _match_tile_to_face(tiles, face_id)
        us, vs = [], []
        for vtx in tile.vertices:
            pt = np.array(vtx, dtype=np.float64)
            u, vv = project_point_to_tile_uv(pt, center, tangent, bitangent)
            us.append(u)
            vs.append(vv)
        return min(us), min(vs), max(us), max(vs)

    # Fallback
    face = globe_grid.faces[face_id]
    us, vs = [], []
    for vid in face.vertex_ids:
        v = globe_grid.vertices[vid]
        pt = np.array([v.x, v.y, v.z], dtype=np.float64)
        u, vv = project_point_to_tile_uv(pt, center, tangent, bitangent)
        us.append(u)
        vs.append(vv)
    return min(us), min(vs), max(us), max(vs)


def project_and_normalize(
    point_3d: np.ndarray,
    center: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
    uv_bounds: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """Project a 3D point to normalised [0,1] tile UV space.

    This replicates the models library's ``projected_vertices`` +
    ``normalize_uvs`` transform.  Uses **uniform scaling** (same span
    for both axes) with centering, matching the models library's
    aspect-ratio-preserving normalisation.
    """
    u_raw, v_raw = project_point_to_tile_uv(point_3d, center, tangent, bitangent)
    u_min, v_min, u_max, v_max = uv_bounds
    u_span = max(u_max - u_min, 1e-12)
    v_span = max(v_max - v_min, 1e-12)
    span = max(u_span, v_span)
    offset_u = (span - u_span) / (2.0 * span)
    offset_v = (span - v_span) / (2.0 * span)
    return ((u_raw - u_min) / span + offset_u, (v_raw - v_min) / span + offset_v)


# ═══════════════════════════════════════════════════════════════════
# 20A.2 — Affine transform from detail-2D to tile-UV
# ═══════════════════════════════════════════════════════════════════

class UVTransform:
    """Piecewise-linear polygon warp from detail-grid 2D to tile UV [0,1].

    The warp splits both the source polygon (detail-grid boundary) and
    destination polygon (GoldbergTile UV polygon) into triangle fans
    from their respective centroids.  For any interior point, we find
    which source triangle it belongs to, compute barycentric coords,
    and map to the corresponding destination triangle.

    This gives **exact** boundary-vertex alignment (zero error at
    polygon corners), unlike a global similarity transform.
    """

    def __init__(
        self,
        src_centroid: np.ndarray,
        src_corners: np.ndarray,
        dst_centroid: np.ndarray,
        dst_corners: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        src_centroid : (2,) — detail-grid centroid
        src_corners : (N, 2) — detail-grid boundary polygon vertices (ordered)
        dst_centroid : (2,) — UV polygon centroid
        dst_corners : (N, 2) — UV polygon vertices (same order, matched)
        """
        self.src_centroid = np.asarray(src_centroid, dtype=np.float64)
        self.src_corners = np.asarray(src_corners, dtype=np.float64)
        self.dst_centroid = np.asarray(dst_centroid, dtype=np.float64)
        self.dst_corners = np.asarray(dst_corners, dtype=np.float64)
        self.n = len(src_corners)
        # Pre-compute per-sector affine transforms for speed
        self._sector_A: List[np.ndarray] = []
        self._sector_t: List[np.ndarray] = []
        self._sector_angles: np.ndarray = np.zeros(self.n, dtype=np.float64)
        for i in range(self.n):
            j = (i + 1) % self.n
            # Source triangle: centroid, corner[i], corner[j]
            S = np.column_stack([
                self.src_corners[i] - self.src_centroid,
                self.src_corners[j] - self.src_centroid,
            ])  # (2, 2)
            # Destination triangle: centroid, corner[i], corner[j]
            D = np.column_stack([
                self.dst_corners[i] - self.dst_centroid,
                self.dst_corners[j] - self.dst_centroid,
            ])  # (2, 2)
            det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
            if abs(det) > 1e-20:
                S_inv = np.array([
                    [S[1, 1], -S[0, 1]],
                    [-S[1, 0], S[0, 0]],
                ], dtype=np.float64) / det
                A = D @ S_inv
            else:
                A = np.eye(2, dtype=np.float64)
            t = self.dst_centroid - A @ self.src_centroid
            self._sector_A.append(A)
            self._sector_t.append(t)
            # Angle of the bisector for sector classification
            mid = (self.src_corners[i] + self.src_corners[j]) * 0.5
            self._sector_angles[i] = math.atan2(
                mid[1] - self.src_centroid[1],
                mid[0] - self.src_centroid[0],
            )

    def _find_sector(self, x: float, y: float) -> int:
        """Find which triangle-fan sector a point belongs to."""
        angle = math.atan2(y - self.src_centroid[1], x - self.src_centroid[0])
        # Find sector by checking which pair of adjacent corners the
        # point's angle falls between.
        for i in range(self.n):
            j = (i + 1) % self.n
            a0 = math.atan2(
                self.src_corners[i][1] - self.src_centroid[1],
                self.src_corners[i][0] - self.src_centroid[0],
            )
            a1 = math.atan2(
                self.src_corners[j][1] - self.src_centroid[1],
                self.src_corners[j][0] - self.src_centroid[0],
            )
            if _angle_between(a0, a1, angle):
                return i
        # Fallback: nearest sector by angle difference
        best_i = 0
        best_diff = float("inf")
        for i in range(self.n):
            diff = abs(math.atan2(
                math.sin(angle - self._sector_angles[i]),
                math.cos(angle - self._sector_angles[i]),
            ))
            if diff < best_diff:
                best_diff = diff
                best_i = i
        return best_i

    def apply(self, x: float, y: float) -> Tuple[float, float]:
        """Transform a detail-grid 2D point to tile UV."""
        sector = self._find_sector(x, y)
        A = self._sector_A[sector]
        t = self._sector_t[sector]
        p = A @ np.array([x, y]) + t
        return (float(p[0]), float(p[1]))

    def apply_array(self, points: np.ndarray) -> np.ndarray:
        """Transform an array of points (N, 2) to UV coordinates (N, 2)."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        result = np.empty_like(pts)
        # Vectorised sector assignment by angle
        angles = np.arctan2(
            pts[:, 1] - self.src_centroid[1],
            pts[:, 0] - self.src_centroid[0],
        )
        # Pre-compute corner angles
        corner_angles = np.arctan2(
            self.src_corners[:, 1] - self.src_centroid[1],
            self.src_corners[:, 0] - self.src_centroid[0],
        )
        # Assign sectors
        sectors = np.zeros(len(pts), dtype=np.int32)
        for k in range(len(pts)):
            sectors[k] = self._find_sector(pts[k, 0], pts[k, 1])
        # Apply per-sector transforms
        for i in range(self.n):
            mask = sectors == i
            if not mask.any():
                continue
            A = self._sector_A[i]
            t = self._sector_t[i]
            sector_pts = pts[mask]
            result[mask] = (sector_pts @ A.T) + t
        return result


def _angle_between(a0: float, a1: float, angle: float) -> bool:
    """Check if *angle* lies in the arc from *a0* to *a1* (counter-clockwise)."""
    # Normalise everything to [0, 2π)
    def _norm(a: float) -> float:
        return a % (2.0 * math.pi)
    a0n = _norm(a0)
    a1n = _norm(a1)
    an = _norm(angle)
    if a0n <= a1n:
        return a0n <= an <= a1n
    else:
        # Wraps around 0
        return an >= a0n or an <= a1n


def _find_polygon_corners(detail_grid: PolyGrid) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the macro-polygon corner vertices from a detail grid.

    The detail grid is a regular hex or pentagon grid.  Its outermost
    vertices lie on a convex polygon.  We identify the polygon's
    corners by clustering the outermost vertices by angle.

    If the grid's metadata contains ``corner_vertex_ids`` (set by
    ``build_goldberg_grid`` for pentagon grids), those vertex positions
    are used directly — skipping the heuristic clustering entirely.

    Returns
    -------
    (centroid, corners)
        centroid : (2,) — mean of all vertices
        corners : (N, 2) — polygon corners, counter-clockwise ordered
    """
    all_verts = []
    for v in detail_grid.vertices.values():
        if v.has_position():
            all_verts.append(np.array([v.x, v.y], dtype=np.float64))

    if len(all_verts) < 3:
        return np.zeros(2), np.zeros((0, 2))

    arr = np.array(all_verts)
    centroid = arr.mean(axis=0)

    # ── Fast path: exact corner IDs from topology ───────────────
    corner_vids = detail_grid.metadata.get("corner_vertex_ids")
    if corner_vids:
        corner_positions = []
        for vid in corner_vids:
            v = detail_grid.vertices.get(vid)
            if v is not None and v.has_position():
                corner_positions.append(np.array([v.x, v.y], dtype=np.float64))
        if len(corner_positions) >= 3:
            corners = np.array(corner_positions, dtype=np.float64)
            # Sort counter-clockwise by angle from centroid
            angles = np.arctan2(
                corners[:, 1] - centroid[1],
                corners[:, 0] - centroid[0],
            )
            corners = corners[np.argsort(angles)]
            return centroid, corners

    # ── Clustering fallback ─────────────────────────────────────
    dists = np.linalg.norm(arr - centroid, axis=1)
    max_dist = dists.max()

    if max_dist < 1e-12:
        return centroid, np.zeros((0, 2))

    # Determine the expected number of corners from the grid type.
    face_type = detail_grid.metadata.get("parent_face_type")
    if face_type is None:
        # Infer from face count: pentagon grids have 5n²+5n+1 faces,
        # hex grids have 3n²+3n+1.  Pentagon grids are smaller.
        n_faces = len(detail_grid.faces)
        parent_fid = detail_grid.metadata.get("parent_face_id", "")
        parent_face = None
        # Try to detect pent vs hex from the detail grid's structure
        # A simpler heuristic: pentagon grids have 5-fold symmetry
        # For now, use a threshold: if the number of faces is closer
        # to the pentagon formula, it's pentagon.
        # With rings=r: hex_faces = 3r²+3r+1, pent_faces = 5r²+5r+1
        # Or just count outermost vertices and cluster.
        pass

    # Get the outermost ring of vertices (at max distance)
    outer_threshold = max_dist * 0.999
    outer_mask = dists >= outer_threshold
    outer_verts = arr[outer_mask]

    if len(outer_verts) < 3:
        # Relax threshold
        outer_threshold = max_dist * 0.98
        outer_mask = dists >= outer_threshold
        outer_verts = arr[outer_mask]

    if len(outer_verts) < 3:
        return centroid, np.zeros((0, 2))

    # Compute angles from centroid
    outer_angles = np.arctan2(
        outer_verts[:, 1] - centroid[1],
        outer_verts[:, 0] - centroid[0],
    )

    # Sort by angle
    order = np.argsort(outer_angles)
    outer_verts = outer_verts[order]
    outer_angles = outer_angles[order]

    # Cluster by angular proximity.  True polygon corners have
    # neighbouring vertices very close in angle, while there are
    # large angular gaps between different corners.
    # Find gap sizes between consecutive angles
    n_outer = len(outer_angles)
    gaps = np.empty(n_outer, dtype=np.float64)
    for i in range(n_outer):
        j = (i + 1) % n_outer
        gap = outer_angles[j] - outer_angles[i]
        if j == 0:
            gap += 2.0 * math.pi
        gaps[i] = gap

    # The large gaps separate clusters.  We expect 5 or 6 corners.
    # Try both and pick whichever gives cleaner clusters.
    for n_expected in (6, 5):
        if n_outer < n_expected:
            continue

        # Find the n_expected largest gaps
        gap_order = np.argsort(gaps)[::-1]
        split_indices = sorted(gap_order[:n_expected])

        # Build clusters: each cluster starts after a split gap
        clusters = []
        for c in range(n_expected):
            start = (split_indices[c] + 1) % n_outer
            end = split_indices[(c + 1) % n_expected]
            # Collect indices from start to end (inclusive), wrapping
            indices = []
            idx = start
            while True:
                indices.append(idx)
                if idx == end:
                    break
                idx = (idx + 1) % n_outer
            clusters.append(indices)

        # Verify: each cluster should have similar number of vertices
        cluster_sizes = [len(c) for c in clusters]
        if max(cluster_sizes) <= 3 * min(max(cluster_sizes), 1):
            # Compute corner as the mean of each cluster
            corners = np.empty((n_expected, 2), dtype=np.float64)
            for c_idx, cluster in enumerate(clusters):
                cluster_verts = outer_verts[cluster]
                corners[c_idx] = cluster_verts.mean(axis=0)

            # Sort by angle
            c_angles = np.arctan2(
                corners[:, 1] - centroid[1],
                corners[:, 0] - centroid[0],
            )
            corners = corners[np.argsort(c_angles)]
            return centroid, corners

    # Fallback: just use the outer vertices as-is
    return centroid, outer_verts


def compute_detail_to_uv_transform(
    globe_grid: PolyGrid,
    face_id: str,
    detail_grid: PolyGrid,
    center: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
    uv_bounds: Tuple[float, float, float, float],
) -> UVTransform:
    """Compute the piecewise-linear warp from detail-grid 2D to tile UV [0,1].

    Strategy
    --------
    The detail grid's boundary forms a regular polygon (hex or pent)
    in 2D Tutte-embedding space.  The GoldbergTile's ``uv_vertices``
    define the UV polygon that the 3D mesh samples.  We construct a
    triangle-fan warp that maps each sector of the source polygon to
    the corresponding sector of the UV polygon, giving **exact**
    boundary vertex alignment.

    Parameters
    ----------
    globe_grid : PolyGrid
    face_id : str
    detail_grid : PolyGrid
    center, tangent, bitangent : np.ndarray
    uv_bounds : (u_min, v_min, u_max, v_max)

    Returns
    -------
    UVTransform
    """
    # ── UV positions of the tile polygon vertices ───────────────
    if _HAS_MODELS:
        uv_verts = get_tile_uv_vertices(globe_grid, face_id)
        uv_poly = np.array(uv_verts, dtype=np.float64)
    else:
        face = globe_grid.faces[face_id]
        uv_list = []
        for vid in face.vertex_ids:
            v = globe_grid.vertices[vid]
            pt3 = np.array([v.x, v.y, v.z], dtype=np.float64)
            uv = project_and_normalize(pt3, center, tangent, bitangent, uv_bounds)
            uv_list.append(np.array(uv, dtype=np.float64))
        uv_poly = np.array(uv_list)

    uv_centroid = uv_poly.mean(axis=0)
    n_poly = len(uv_poly)

    # ── Find the detail grid's polygon corners ──────────────────
    detail_centroid, detail_corners = _find_polygon_corners(detail_grid)

    if len(detail_corners) < 3:
        # Degenerate — return identity-ish transform
        return UVTransform(
            src_centroid=np.zeros(2), src_corners=uv_poly,
            dst_centroid=uv_centroid, dst_corners=uv_poly,
        )

    # ── Match corners by angle ──────────────────────────────────
    # The detail grid and UV polygon may have the same or different
    # numbers of corners.  We need to match them 1:1.
    n_detail = len(detail_corners)

    if n_detail == n_poly:
        # Same number — match by angle from centroid
        uv_angles = np.arctan2(
            uv_poly[:, 1] - uv_centroid[1],
            uv_poly[:, 0] - uv_centroid[0],
        )
        detail_angles = np.arctan2(
            detail_corners[:, 1] - detail_centroid[1],
            detail_corners[:, 0] - detail_centroid[0],
        )
        # For each UV vertex, find the closest detail corner by angle
        matched_detail = np.empty_like(uv_poly)
        used = set()
        # Sort UV by angle for consistent matching
        uv_order = np.argsort(uv_angles)
        detail_order = np.argsort(detail_angles)
        # The detail corners and UV corners should have the same
        # angular ordering.  Find the best rotational offset.
        # Score each offset by angular difference, then validate the
        # winner using edge-length ratios (38C.1).
        offset_scores: List[Tuple[float, int]] = []
        for offset in range(n_poly):
            score = 0.0
            for k in range(n_poly):
                dk = detail_order[(k + offset) % n_poly]
                uk = uv_order[k]
                diff = math.atan2(
                    math.sin(detail_angles[dk] - uv_angles[uk]),
                    math.cos(detail_angles[dk] - uv_angles[uk]),
                )
                score += diff * diff
            offset_scores.append((score, offset))
        offset_scores.sort()

        def _edge_length_ratio_valid(offset: int, threshold: float = 0.5) -> bool:
            """Check that adjacent edge-length ratios are consistent.

            For each pair of adjacent corners, compute the edge length
            in both source (detail) and destination (UV) polygons.
            If the ratio between corresponding edges varies too much
            (i.e. one edge maps to a much longer/shorter one than its
            neighbour), the offset is likely wrong.
            """
            src_lens = []
            dst_lens = []
            for k in range(n_poly):
                k_next = (k + 1) % n_poly
                dk = detail_order[(k + offset) % n_poly]
                dk_next = detail_order[(k_next + offset) % n_poly]
                uk = uv_order[k]
                uk_next = uv_order[k_next]
                src_lens.append(float(np.linalg.norm(
                    detail_corners[dk] - detail_corners[dk_next]
                )))
                dst_lens.append(float(np.linalg.norm(
                    uv_poly[uk] - uv_poly[uk_next]
                )))
            # Compute per-edge ratios (src/dst), check consistency
            ratios = []
            for sl, dl in zip(src_lens, dst_lens):
                if dl < 1e-12 or sl < 1e-12:
                    continue
                ratios.append(sl / dl)
            if len(ratios) < 2:
                return True
            min_r, max_r = min(ratios), max(ratios)
            return (max_r - min_r) / max(max_r, 1e-12) < threshold

        # Pick the best offset that also passes edge-length validation
        best_offset = offset_scores[0][1]
        for score, offset in offset_scores:
            if _edge_length_ratio_valid(offset):
                best_offset = offset
                break
        else:
            # None passed validation — fall back to angular best and warn
            import logging
            logging.getLogger(__name__).warning(
                "compute_detail_to_uv_transform: no rotational offset "
                "passed edge-length validation for face %s; using "
                "angular best (offset=%d)", face_id, best_offset,
            )

        # Build matched arrays in UV polygon order
        src_corners_ordered = np.empty((n_poly, 2), dtype=np.float64)
        dst_corners_ordered = np.empty((n_poly, 2), dtype=np.float64)
        for k in range(n_poly):
            uk = uv_order[k]
            dk = detail_order[(k + best_offset) % n_poly]
            src_corners_ordered[k] = detail_corners[dk]
            dst_corners_ordered[k] = uv_poly[uk]

        # Re-sort both by the source angle for consistent sector ordering
        src_angles_final = np.arctan2(
            src_corners_ordered[:, 1] - detail_centroid[1],
            src_corners_ordered[:, 0] - detail_centroid[0],
        )
        final_order = np.argsort(src_angles_final)
        src_corners_ordered = src_corners_ordered[final_order]
        dst_corners_ordered = dst_corners_ordered[final_order]

    else:
        # Different numbers of corners — fall back to similarity
        # transform fitted to the available corners, then construct
        # a UVTransform from the UV polygon mapped back.
        # This shouldn't happen for well-formed globe grids.
        all_verts = []
        for v in detail_grid.vertices.values():
            if v.has_position():
                all_verts.append(np.array([v.x, v.y], dtype=np.float64))
        arr = np.array(all_verts)
        dc = arr.mean(axis=0)
        dists = np.linalg.norm(arr - dc, axis=1)
        boundary = arr[dists >= dists.max() * 0.85]
        boundary_angles = np.arctan2(boundary[:, 1] - dc[1], boundary[:, 0] - dc[0])
        uv_angles = np.arctan2(uv_poly[:, 1] - uv_centroid[1],
                               uv_poly[:, 0] - uv_centroid[0])
        src_pts, dst_pts = [], []
        for i in range(n_poly):
            diffs = np.abs(np.arctan2(
                np.sin(boundary_angles - uv_angles[i]),
                np.cos(boundary_angles - uv_angles[i]),
            ))
            src_pts.append(boundary[np.argmin(diffs)])
            dst_pts.append(uv_poly[i])
        src_pts = np.array(src_pts)
        dst_pts = np.array(dst_pts)
        # Use UV polygon as both src and dst corners, with a global
        # affine fitted to boundary points
        src_c = src_pts.mean(axis=0)
        dst_c = dst_pts.mean(axis=0)
        src_rel = src_pts - src_c
        dst_rel = dst_pts - dst_c
        ss = np.sum(src_rel[:, 0]**2 + src_rel[:, 1]**2)
        if ss > 1e-20:
            a = np.sum(src_rel[:, 0] * dst_rel[:, 0] + src_rel[:, 1] * dst_rel[:, 1]) / ss
            b = np.sum(src_rel[:, 0] * dst_rel[:, 1] - src_rel[:, 1] * dst_rel[:, 0]) / ss
            A_mat = np.array([[a, -b], [b, a]], dtype=np.float64)
        else:
            A_mat = np.eye(2)
        # Map detail corners to UV space via the affine
        mapped = (detail_corners @ A_mat.T) + (dst_c - A_mat @ src_c)
        src_angles_f = np.arctan2(detail_corners[:, 1] - detail_centroid[1],
                                  detail_corners[:, 0] - detail_centroid[0])
        order = np.argsort(src_angles_f)
        src_corners_ordered = detail_corners[order]
        dst_corners_ordered = mapped[order]

    return UVTransform(
        src_centroid=detail_centroid,
        src_corners=src_corners_ordered,
        dst_centroid=uv_centroid,
        dst_corners=dst_corners_ordered,
    )

# ═══════════════════════════════════════════════════════════════════
# 20B.1 — Shared-edge stitching
# ═══════════════════════════════════════════════════════════════════

def _find_shared_edges(
    globe_grid: PolyGrid,
    face_ids: List[str],
) -> List[Tuple[str, str, List[Tuple[int, int]]]]:
    """Identify shared Goldberg edges between adjacent tiles.

    Returns a list of ``(fid_a, fid_b, shared_vertex_pairs)`` where each
    pair gives the vertex indices (in the GoldbergTile) that coincide.

    Only edges where both tiles are in *face_ids* are returned.
    """
    if not _HAS_MODELS:
        return []
    from ..core.algorithms import get_face_adjacency

    freq = globe_grid.metadata.get("frequency", 3)
    rad = globe_grid.metadata.get("radius", 1.0)
    tiles_obj = get_goldberg_tiles(freq, rad)
    fid_set = set(face_ids)
    adj = get_face_adjacency(globe_grid)

    seen: set = set()
    edges: List[Tuple[str, str, List[Tuple[int, int]]]] = []

    for fid_a in face_ids:
        for fid_b in adj.get(fid_a, []):
            if fid_b not in fid_set:
                continue
            pair_key = tuple(sorted([fid_a, fid_b]))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            idx_a = int(fid_a.replace("t", ""))
            idx_b = int(fid_b.replace("t", ""))
            tile_a = tiles_obj[idx_a]
            tile_b = tiles_obj[idx_b]

            # Find shared 3D vertices
            verts_a = [tuple(round(float(c), 8) for c in v) for v in tile_a.vertices]
            verts_b = [tuple(round(float(c), 8) for c in v) for v in tile_b.vertices]

            shared = []
            for ia, va in enumerate(verts_a):
                for ib, vb in enumerate(verts_b):
                    if va == vb:
                        shared.append((ia, ib))
                        break

            if len(shared) == 2:
                edges.append((fid_a, fid_b, shared))

    return edges
