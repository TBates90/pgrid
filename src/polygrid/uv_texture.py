"""UV-aligned texture rendering — Phase 20A.

Renders tile textures using the **same projection** that the 3D mesh
uses for UV mapping.  This eliminates the coordinate-system mismatch
between the apron grid's 2D Tutte-embedding positions and the
GoldbergTile's tangent-plane projection (which defines the atlas UVs).

The key idea: the 3D mesh builder calls ``generate_goldberg_tiles()``
from the models library to get per-tile ``uv_vertices`` (and the
``tangent`` / ``bitangent`` basis that produced them).  We use those
*exact same* GoldbergTile objects to derive the mapping from the
detail grid's 2D positions into the mesh's UV space.

Functions
---------
- :func:`get_goldberg_tiles`             — cached access to GoldbergTile list
- :func:`compute_detail_to_uv_transform` — affine mapping detail-2D → tile-UV
- :func:`render_tile_uv_aligned`         — rasterise an apron grid in UV-aligned space
- :func:`build_uv_aligned_atlas`         — full atlas pipeline with UV alignment
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .detail_render import BiomeConfig, detail_elevation_to_colour, _detail_hillshade
from .geometry import face_center
from .polygrid import PolyGrid
from .tile_data import TileDataStore
from .tile_detail import DetailGridCollection

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
    ``normalize_uvs`` transform.
    """
    u_raw, v_raw = project_point_to_tile_uv(point_3d, center, tangent, bitangent)
    u_min, v_min, u_max, v_max = uv_bounds
    u_span = max(u_max - u_min, 1e-12)
    v_span = max(v_max - v_min, 1e-12)
    return ((u_raw - u_min) / u_span, (v_raw - v_min) / v_span)


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
        best_offset = 0
        best_score = float("inf")
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
            if score < best_score:
                best_score = score
                best_offset = offset

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
# 20A.3 — UV-aligned tile rasteriser
# ═══════════════════════════════════════════════════════════════════

def render_tile_uv_aligned(
    apron_grid: PolyGrid,
    store: TileDataStore,
    uv_transform: UVTransform,
    biome: Optional[BiomeConfig] = None,
    *,
    tile_size: int = 256,
    elevation_field: str = "elevation",
    noise_seed: int = 0,
    gutter_pixels: int = 4,
    vertex_jitter: float = 1.5,
    noise_overlay: bool = True,
    noise_frequency: float = 0.05,
    noise_amplitude: float = 0.05,
    colour_dither: bool = True,
    dither_radius: float = 6.0,
    k_neighbours: int = 4,
) -> "Image.Image":
    """Rasterise an apron grid using UV-aligned projection.

    Instead of using the grid's native 2D positions to compute the
    bounding box (which don't match the mesh UVs), this function
    transforms every vertex through *uv_transform* so that the
    resulting image is pixel-aligned with the atlas UV coordinates.

    The tile's polygon maps to the [0,1]×[0,1] square (the atlas slot),
    and apron sub-faces from neighbours naturally extend beyond into the
    gutter zone.

    Parameters
    ----------
    apron_grid : PolyGrid
        Extended grid (own + apron sub-faces).
    store : TileDataStore
        Elevation data for all sub-faces.
    uv_transform : UVTransform
        From :func:`compute_detail_to_uv_transform`.
    biome : BiomeConfig, optional
    tile_size : int
    elevation_field : str
    noise_seed : int
    gutter_pixels : int
        Gutter zone around the tile slot.  Total image width is
        ``tile_size + 2 * gutter_pixels`` during rasterisation, but
        the output is the central ``tile_size × tile_size`` region
        plus the gutter already filled.
    vertex_jitter : float
    noise_overlay : bool
    noise_frequency, noise_amplitude : float
    colour_dither : bool
    dither_radius : float
    k_neighbours : int

    Returns
    -------
    PIL.Image.Image
        RGB image of size (tile_size, tile_size).
        The gutter zone is incorporated into the rendering so that
        apron sub-faces fill it with real terrain data.
    """
    from PIL import Image, ImageDraw
    from scipy.spatial import KDTree
    from .tile_texture import (
        jitter_polygon_vertices,
        apply_noise_overlay,
        apply_colour_dithering,
    )

    if biome is None:
        biome = BiomeConfig()

    # ── Hillshade ───────────────────────────────────────────────
    hs_dict = _detail_hillshade(
        apron_grid, store, elevation_field,
        azimuth=biome.azimuth, altitude=biome.altitude,
    )

    # ── Render area: tile_size plus gutter on each side ─────────
    # In UV space, [0, 1] maps to the inner tile_size region.
    # The gutter extends to [-gutter_frac, 1+gutter_frac].
    gutter_frac = gutter_pixels / tile_size if tile_size > 0 else 0
    render_size = tile_size + 2 * gutter_pixels

    def _uv_to_pixel(u: float, v: float) -> Tuple[float, float]:
        """Map UV [0,1] coordinates to pixel coordinates.

        UV (0,0) → pixel (gutter, render_size - gutter)  (bottom-left)
        UV (1,1) → pixel (gutter + tile_size, gutter)     (top-right)
        """
        px = gutter_pixels + u * tile_size
        py = gutter_pixels + (1.0 - v) * tile_size  # flip Y
        return (px, py)

    # ── Step 1: Compute face colours + transform vertices ───────
    face_colours: Dict[str, Tuple[float, float, float]] = {}
    face_pixel_colours: Dict[str, Tuple[int, int, int]] = {}
    centroid_pixels: List[Tuple[float, float]] = []
    colour_array: List[Tuple[int, int, int]] = []

    for fid, face_obj in apron_grid.faces.items():
        # Check all vertices exist
        has_verts = True
        for vid in face_obj.vertex_ids:
            v = apron_grid.vertices.get(vid)
            if v is None or not v.has_position():
                has_verts = False
                break

        if has_verts and len(face_obj.vertex_ids) >= 3:
            elev = store.get(fid, elevation_field)
            c = face_center(apron_grid.vertices, face_obj)
            cx, cy = c if c else (0.0, 0.0)

            r, g, b = detail_elevation_to_colour(
                elev, biome,
                hillshade_val=hs_dict.get(fid, 0.5),
                noise_x=cx, noise_y=cy,
                noise_seed=noise_seed,
            )
            face_colours[fid] = (r, g, b)
            pc = (
                max(0, min(255, int(r * 255))),
                max(0, min(255, int(g * 255))),
                max(0, min(255, int(b * 255))),
            )
            face_pixel_colours[fid] = pc

            # Transform centroid through UV mapping
            uv_cx, uv_cy = uv_transform.apply(cx, cy)
            px, py = _uv_to_pixel(uv_cx, uv_cy)
            centroid_pixels.append((px, py))
            colour_array.append(pc)

    if not face_colours:
        return Image.new("RGB", (tile_size, tile_size), (0, 0, 0))

    # ── Draw polygon fills on sentinel background ───────────────
    img = Image.new("RGB", (render_size, render_size), _BG_SENTINEL)
    draw = ImageDraw.Draw(img)

    for fid, face_obj in apron_grid.faces.items():
        verts_uv = []
        for vid in face_obj.vertex_ids:
            v = apron_grid.vertices.get(vid)
            if v is None or not v.has_position():
                break
            # Transform vertex to UV, then to pixel
            uv_u, uv_v = uv_transform.apply(v.x, v.y)
            px, py = _uv_to_pixel(uv_u, uv_v)
            verts_uv.append((px, py))
        else:
            if len(verts_uv) >= 3 and fid in face_pixel_colours:
                if vertex_jitter > 0:
                    verts_uv = jitter_polygon_vertices(
                        verts_uv,
                        max_jitter=vertex_jitter,
                        seed=noise_seed + hash(fid) % 100_000,
                    )
                colour = face_pixel_colours[fid]
                draw.polygon(verts_uv, fill=colour, outline=colour)

    pixels = np.array(img)  # (H, W, 3) uint8

    # ── Step 1b: Colour dithering ───────────────────────────────
    if colour_dither and len(centroid_pixels) >= 2:
        centroid_arr_d = np.array(centroid_pixels, dtype=np.float64)
        colour_arr_d = np.array(colour_array, dtype=np.float64)
        pixels = apply_colour_dithering(
            pixels, centroid_arr_d, colour_arr_d,
            blend_radius=dither_radius,
            k_neighbours=min(k_neighbours, len(centroid_arr_d)),
        )

    # ── Step 2: IDW fill for remaining background pixels ────────
    bg_mask = (
        (pixels[:, :, 0] == _BG_SENTINEL[0]) &
        (pixels[:, :, 1] == _BG_SENTINEL[1]) &
        (pixels[:, :, 2] == _BG_SENTINEL[2])
    )
    bg_count = bg_mask.sum()
    if bg_count > 0 and len(centroid_pixels) > 0:
        centroid_arr = np.array(centroid_pixels, dtype=np.float64)
        colour_arr = np.array(colour_array, dtype=np.float64)
        tree = KDTree(centroid_arr)

        bg_rows, bg_cols = np.where(bg_mask)
        bg_points = np.column_stack([
            bg_cols.astype(np.float64),
            bg_rows.astype(np.float64),
        ])

        k_actual = min(k_neighbours, len(centroid_arr))
        dists, idxs = tree.query(bg_points, k=k_actual, workers=-1)

        if k_actual == 1:
            dists = dists[:, np.newaxis]
            idxs = idxs[:, np.newaxis]

        eps = 1e-12
        weights = 1.0 / np.maximum(dists, eps)
        w_sum = weights.sum(axis=1, keepdims=True)
        norm_w = weights / w_sum

        neighbour_colours = colour_arr[idxs]
        blended = (norm_w[:, :, np.newaxis] * neighbour_colours).sum(axis=1)

        blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
        pixels[bg_rows, bg_cols] = blended_uint8

    # ── Step 3: Noise overlay ───────────────────────────────────
    if noise_overlay:
        pixels = apply_noise_overlay(
            pixels,
            frequency=noise_frequency,
            amplitude=noise_amplitude,
            seed=noise_seed + 7777,
        )

    # ── Crop to tile_size (remove gutter from output if needed) ─
    # We keep the full render_size image — the caller (atlas builder)
    # will handle gutter placement.
    result = Image.fromarray(pixels, "RGB")

    # Return just the inner tile_size region for compatibility
    # with the existing atlas pipeline.  The gutter pixels in the
    # image naturally contain apron data.
    if gutter_pixels > 0:
        inner = result.crop((
            gutter_pixels, gutter_pixels,
            gutter_pixels + tile_size, gutter_pixels + tile_size,
        ))
        return inner

    return result


def render_tile_uv_aligned_full(
    apron_grid: PolyGrid,
    store: TileDataStore,
    uv_transform: UVTransform,
    biome: Optional[BiomeConfig] = None,
    *,
    tile_size: int = 256,
    gutter_pixels: int = 4,
    elevation_field: str = "elevation",
    noise_seed: int = 0,
    vertex_jitter: float = 1.5,
    noise_overlay: bool = True,
    noise_frequency: float = 0.05,
    noise_amplitude: float = 0.05,
    colour_dither: bool = True,
    dither_radius: float = 6.0,
    k_neighbours: int = 4,
) -> Tuple["Image.Image", "Image.Image"]:
    """Render both the inner tile and the full image with gutter.

    Returns
    -------
    (inner_image, full_image)
        inner_image : tile_size × tile_size (the atlas slot content)
        full_image  : (tile_size + 2*gutter) × (tile_size + 2*gutter)
    """
    from PIL import Image, ImageDraw
    from scipy.spatial import KDTree
    from .tile_texture import (
        jitter_polygon_vertices,
        apply_noise_overlay,
        apply_colour_dithering,
    )

    if biome is None:
        biome = BiomeConfig()

    hs_dict = _detail_hillshade(
        apron_grid, store, elevation_field,
        azimuth=biome.azimuth, altitude=biome.altitude,
    )

    render_size = tile_size + 2 * gutter_pixels

    def _uv_to_pixel(u: float, v: float) -> Tuple[float, float]:
        px = gutter_pixels + u * tile_size
        py = gutter_pixels + (1.0 - v) * tile_size
        return (px, py)

    face_colours: Dict[str, Tuple[float, float, float]] = {}
    face_pixel_colours: Dict[str, Tuple[int, int, int]] = {}
    centroid_pixels: List[Tuple[float, float]] = []
    colour_array: List[Tuple[int, int, int]] = []

    for fid, face_obj in apron_grid.faces.items():
        has_verts = True
        for vid in face_obj.vertex_ids:
            v = apron_grid.vertices.get(vid)
            if v is None or not v.has_position():
                has_verts = False
                break

        if has_verts and len(face_obj.vertex_ids) >= 3:
            elev = store.get(fid, elevation_field)
            c = face_center(apron_grid.vertices, face_obj)
            cx, cy = c if c else (0.0, 0.0)

            r, g, b = detail_elevation_to_colour(
                elev, biome,
                hillshade_val=hs_dict.get(fid, 0.5),
                noise_x=cx, noise_y=cy,
                noise_seed=noise_seed,
            )
            face_colours[fid] = (r, g, b)
            pc = (
                max(0, min(255, int(r * 255))),
                max(0, min(255, int(g * 255))),
                max(0, min(255, int(b * 255))),
            )
            face_pixel_colours[fid] = pc

            uv_cx, uv_cy = uv_transform.apply(cx, cy)
            px, py = _uv_to_pixel(uv_cx, uv_cy)
            centroid_pixels.append((px, py))
            colour_array.append(pc)

    if not face_colours:
        empty = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
        full_empty = Image.new("RGB", (render_size, render_size), (0, 0, 0))
        return empty, full_empty

    img = Image.new("RGB", (render_size, render_size), _BG_SENTINEL)
    draw = ImageDraw.Draw(img)

    for fid, face_obj in apron_grid.faces.items():
        verts_uv = []
        for vid in face_obj.vertex_ids:
            v = apron_grid.vertices.get(vid)
            if v is None or not v.has_position():
                break
            uv_u, uv_v = uv_transform.apply(v.x, v.y)
            px, py = _uv_to_pixel(uv_u, uv_v)
            verts_uv.append((px, py))
        else:
            if len(verts_uv) >= 3 and fid in face_pixel_colours:
                if vertex_jitter > 0:
                    verts_uv = jitter_polygon_vertices(
                        verts_uv,
                        max_jitter=vertex_jitter,
                        seed=noise_seed + hash(fid) % 100_000,
                    )
                colour = face_pixel_colours[fid]
                draw.polygon(verts_uv, fill=colour, outline=colour)

    pixels = np.array(img)

    if colour_dither and len(centroid_pixels) >= 2:
        centroid_arr_d = np.array(centroid_pixels, dtype=np.float64)
        colour_arr_d = np.array(colour_array, dtype=np.float64)
        pixels = apply_colour_dithering(
            pixels, centroid_arr_d, colour_arr_d,
            blend_radius=dither_radius,
            k_neighbours=min(k_neighbours, len(centroid_arr_d)),
        )

    bg_mask = (
        (pixels[:, :, 0] == _BG_SENTINEL[0]) &
        (pixels[:, :, 1] == _BG_SENTINEL[1]) &
        (pixels[:, :, 2] == _BG_SENTINEL[2])
    )
    bg_count = bg_mask.sum()
    if bg_count > 0 and len(centroid_pixels) > 0:
        centroid_arr = np.array(centroid_pixels, dtype=np.float64)
        colour_arr = np.array(colour_array, dtype=np.float64)
        tree = KDTree(centroid_arr)

        bg_rows, bg_cols = np.where(bg_mask)
        bg_points = np.column_stack([
            bg_cols.astype(np.float64),
            bg_rows.astype(np.float64),
        ])

        k_actual = min(k_neighbours, len(centroid_arr))
        dists, idxs = tree.query(bg_points, k=k_actual, workers=-1)

        if k_actual == 1:
            dists = dists[:, np.newaxis]
            idxs = idxs[:, np.newaxis]

        eps = 1e-12
        weights = 1.0 / np.maximum(dists, eps)
        w_sum = weights.sum(axis=1, keepdims=True)
        norm_w = weights / w_sum

        neighbour_colours = colour_arr[idxs]
        blended = (norm_w[:, :, np.newaxis] * neighbour_colours).sum(axis=1)

        blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
        pixels[bg_rows, bg_cols] = blended_uint8

    if noise_overlay:
        pixels = apply_noise_overlay(
            pixels,
            frequency=noise_frequency,
            amplitude=noise_amplitude,
            seed=noise_seed + 7777,
        )

    full_img = Image.fromarray(pixels, "RGB")
    inner = full_img.crop((
        gutter_pixels, gutter_pixels,
        gutter_pixels + tile_size, gutter_pixels + tile_size,
    ))

    return inner, full_img


# ═══════════════════════════════════════════════════════════════════
# 20B — UV-Aligned Atlas Builder
# ═══════════════════════════════════════════════════════════════════

def build_uv_aligned_atlas(
    collection: DetailGridCollection,
    globe_grid: PolyGrid,
    *,
    biome_renderers: Optional[Dict[str, Any]] = None,
    density_map: Optional[Dict[str, float]] = None,
    biome_type_map: Optional[Dict[str, str]] = None,
    biome_config: Optional[BiomeConfig] = None,
    output_dir: Path | str = Path("exports/detail_tiles"),
    tile_size: int = 256,
    columns: int = 0,
    noise_seed: int = 0,
    gutter: int = 4,
    smooth_iterations: int = 2,
    coastline_config: Optional[Any] = None,
    enable_coastlines: bool = True,
) -> Tuple[Path, Dict[str, Tuple[float, float, float, float]]]:
    """Build a texture atlas with UV-aligned tile rendering.

    This is the Phase 20 replacement for ``build_apron_feature_atlas``.
    Each tile is rendered using the same tangent-plane projection that
    the 3D mesh uses, so texture content aligns perfectly with the mesh
    UVs and adjacent tiles share matching boundary pixels.

    Parameters
    ----------
    collection : DetailGridCollection
    globe_grid : PolyGrid (GlobeGrid)
    biome_renderers : dict, optional
    density_map : dict, optional
    biome_type_map : dict, optional
    biome_config : BiomeConfig, optional
    output_dir : Path
    tile_size : int
    columns : int
    noise_seed : int
    gutter : int
    smooth_iterations : int
    coastline_config : CoastlineConfig, optional
    enable_coastlines : bool

    Returns
    -------
    (atlas_path, uv_layout)
    """
    from PIL import Image
    from .apron_grid import build_all_apron_grids
    from .geometry import face_center_3d

    if biome_config is None:
        biome_config = BiomeConfig()
    if density_map is None:
        density_map = {}
    if biome_renderers is None:
        biome_renderers = {}
    if biome_type_map is None:
        biome_type_map = {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    face_ids = collection.face_ids
    n = len(face_ids)
    if n == 0:
        raise ValueError("No detail grids in the collection")

    # Default renderer lookup
    default_renderer = (
        next(iter(biome_renderers.values())) if biome_renderers else None
    )

    def _get_renderer(fid: str):
        biome_key = biome_type_map.get(fid)
        if biome_key is not None and biome_key in biome_renderers:
            return biome_renderers[biome_key]
        return default_renderer

    def _get_renderer_for_biome(biome_key: str):
        return biome_renderers.get(biome_key)

    # ── Pre-compute UV transforms for every tile ────────────────
    uv_transforms: Dict[str, UVTransform] = {}
    tile_bases: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for fid in face_ids:
        detail_grid, _ = collection.get(fid)
        center, normal, tangent, bitangent = compute_tile_basis(globe_grid, fid)
        uv_bounds = compute_tile_uv_bounds(globe_grid, fid, center, tangent, bitangent)
        uv_xform = compute_detail_to_uv_transform(
            globe_grid, fid, detail_grid,
            center, tangent, bitangent, uv_bounds,
        )
        uv_transforms[fid] = uv_xform
        tile_bases[fid] = (center, normal, tangent, bitangent)

    # ── Coastline classification (Phase 19) ─────────────────────
    coastline_masks: Dict[str, Any] = {}
    if enable_coastlines and biome_type_map:
        from .coastline import (
            CoastlineConfig,
            classify_tile_biome_context,
            build_coastline_mask,
            blend_biome_images,
            render_coastal_strip,
        )
        from .algorithms import get_face_adjacency

        if coastline_config is None:
            coastline_config = CoastlineConfig()

        adjacency = get_face_adjacency(globe_grid)

        for fid in face_ids:
            ctx = classify_tile_biome_context(
                fid, biome_type_map, adjacency,
            )
            if ctx.is_edge:
                cm = build_coastline_mask(
                    fid, ctx, globe_grid,
                    tile_size=tile_size,
                    config=coastline_config,
                    seed=noise_seed,
                )
                if cm.has_transition:
                    coastline_masks[fid] = cm

    # ── Build apron grids ───────────────────────────────────────
    apron_results = build_all_apron_grids(
        globe_grid, collection,
        smooth_iterations=smooth_iterations,
    )

    # ── Render ground + apply biome overlays ────────────────────
    tile_images: Dict[str, Image.Image] = {}

    for fid in face_ids:
        ar = apron_results[fid]
        uv_xform = uv_transforms[fid]

        # UV-aligned ground texture
        ground_img = render_tile_uv_aligned(
            ar.grid, ar.store,
            uv_xform,
            biome_config,
            tile_size=tile_size,
            noise_seed=noise_seed + hash(fid) % 10000,
            gutter_pixels=gutter,
        )

        # ── Phase 19: Coastline dual-biome rendering ───────────
        if fid in coastline_masks:
            from .coastline import blend_biome_images, render_coastal_strip
            from .apron_texture import _pick_dominant_other_biome

            cm = coastline_masks[fid]
            tile_density = density_map.get(fid, 0.0)
            tile_renderer = _get_renderer(fid)
            img_own = ground_img.copy()

            if tile_density > 0.01 and tile_renderer is not None:
                center_3d = None
                face = globe_grid.faces.get(fid)
                if face is not None:
                    center_3d = face_center_3d(globe_grid.vertices, face)
                if hasattr(tile_renderer, "set_grid_context"):
                    tile_renderer.set_grid_context(ar.grid, ar.store)
                img_own = tile_renderer.render(
                    img_own, fid, tile_density,
                    seed=noise_seed + hash(fid) % 100_000,
                    globe_3d_center=center_3d,
                )

            other_biome = _pick_dominant_other_biome(cm.other_biomes)
            other_renderer = _get_renderer_for_biome(other_biome)
            img_other = ground_img.copy()

            if other_renderer is not None:
                other_density = density_map.get(fid, 0.8)
                if other_density < 0.5:
                    other_density = 0.8
                center_3d = None
                face = globe_grid.faces.get(fid)
                if face is not None:
                    center_3d = face_center_3d(globe_grid.vertices, face)
                if hasattr(other_renderer, "set_grid_context"):
                    other_renderer.set_grid_context(ar.grid, ar.store)
                img_other = other_renderer.render(
                    img_other, fid, other_density,
                    seed=noise_seed + hash(fid) % 100_000 + 7777,
                    globe_3d_center=center_3d,
                )

            ground_img = blend_biome_images(
                img_own.convert("RGB"),
                img_other.convert("RGB"),
                cm.mask,
            )

            ground_img = render_coastal_strip(
                ground_img, cm.mask,
                config=cm.config, seed=noise_seed + hash(fid) % 100_000,
            )
        else:
            # Standard single-biome rendering
            tile_density = density_map.get(fid, 0.0)
            tile_renderer = _get_renderer(fid)

            if tile_density > 0.01 and tile_renderer is not None:
                center_3d = None
                face = globe_grid.faces.get(fid)
                if face is not None:
                    center_3d = face_center_3d(globe_grid.vertices, face)

                if hasattr(tile_renderer, "set_grid_context"):
                    tile_renderer.set_grid_context(ar.grid, ar.store)

                ground_img = tile_renderer.render(
                    ground_img,
                    fid,
                    tile_density,
                    seed=noise_seed + hash(fid) % 100_000,
                    globe_3d_center=center_3d,
                )

        tile_images[fid] = ground_img

        # Save individual tile
        tile_path = output_dir / f"tile_{fid}.png"
        ground_img.convert("RGB").save(str(tile_path))

    # ── Assemble atlas ──────────────────────────────────────────
    if columns <= 0:
        columns = max(1, math.isqrt(n))
        if columns * columns < n:
            columns += 1
    rows = math.ceil(n / columns)

    slot_size = tile_size + 2 * gutter
    atlas_w = columns * slot_size
    atlas_h = rows * slot_size
    atlas = Image.new("RGB", (atlas_w, atlas_h), (128, 128, 128))

    uv_layout: Dict[str, Tuple[float, float, float, float]] = {}

    for idx, fid in enumerate(face_ids):
        col = idx % columns
        row = idx // columns
        slot_x = col * slot_size
        slot_y = row * slot_size

        tile_img = tile_images[fid].convert("RGB").resize(
            (tile_size, tile_size), Image.LANCZOS,
        )

        atlas.paste(tile_img, (slot_x + gutter, slot_y + gutter))

        # Fill gutter with apron data
        if gutter > 0:
            _fill_uv_gutter(atlas, slot_x, slot_y, tile_size, gutter)

        inner_x = slot_x + gutter
        inner_y = slot_y + gutter
        u_min = inner_x / atlas_w
        u_max = (inner_x + tile_size) / atlas_w
        v_min = 1.0 - (inner_y + tile_size) / atlas_h
        v_max = 1.0 - inner_y / atlas_h
        uv_layout[fid] = (u_min, v_min, u_max, v_max)

    atlas_path = output_dir / "detail_atlas.png"
    atlas.save(str(atlas_path))

    return atlas_path, uv_layout


def _fill_uv_gutter(
    atlas: "Image.Image",
    slot_x: int,
    slot_y: int,
    tile_size: int,
    gutter: int,
) -> None:
    """Fill gutter pixels around a tile slot by clamping edge pixels."""
    inner_x = slot_x + gutter
    inner_y = slot_y + gutter

    # Top gutter
    top_strip = atlas.crop((inner_x, inner_y, inner_x + tile_size, inner_y + 1))
    for g in range(gutter):
        atlas.paste(top_strip, (inner_x, slot_y + g))

    # Bottom gutter
    bot_y = inner_y + tile_size - 1
    bot_strip = atlas.crop((inner_x, bot_y, inner_x + tile_size, bot_y + 1))
    for g in range(gutter):
        atlas.paste(bot_strip, (inner_x, inner_y + tile_size + g))

    # Left gutter
    full_top = slot_y
    full_bot = slot_y + tile_size + 2 * gutter
    left_strip = atlas.crop((inner_x, full_top, inner_x + 1, full_bot))
    for g in range(gutter):
        atlas.paste(left_strip, (slot_x + g, full_top))

    # Right gutter
    right_x = inner_x + tile_size - 1
    right_strip = atlas.crop((right_x, full_top, right_x + 1, full_bot))
    for g in range(gutter):
        atlas.paste(right_strip, (inner_x + tile_size + g, full_top))
