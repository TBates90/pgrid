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
from dataclasses import dataclass
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

@dataclass(frozen=True)
class UVTransform:
    """Affine transform mapping detail-grid 2D positions to tile UV [0,1].

    The transform is: ``uv = A @ [x, y] + t``
    where A is 2×2 and t is 2×1.
    """
    A: np.ndarray   # shape (2, 2)
    t: np.ndarray   # shape (2,)

    def apply(self, x: float, y: float) -> Tuple[float, float]:
        """Transform a detail-grid 2D point to tile UV."""
        p = self.A @ np.array([x, y]) + self.t
        return (float(p[0]), float(p[1]))

    def apply_array(self, points: np.ndarray) -> np.ndarray:
        """Transform an array of points (N, 2) to UV coordinates (N, 2)."""
        return (points @ self.A.T) + self.t


def compute_detail_to_uv_transform(
    globe_grid: PolyGrid,
    face_id: str,
    detail_grid: PolyGrid,
    center: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
    uv_bounds: Tuple[float, float, float, float],
) -> UVTransform:
    """Compute the affine transform from detail-grid 2D to tile UV [0,1].

    Strategy
    --------
    When the models library is available, we use the **authoritative**
    ``GoldbergTile.uv_vertices`` — these are the exact UV polygon that
    the 3D mesh builder samples.  We match these UV vertices to the
    detail grid's outermost boundary vertices by angle from centroid,
    then solve a least-squares similarity transform (4 DOF: scale,
    rotation, translation).

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
    # Use the authoritative uv_vertices from the GoldbergTile when
    # available — these are what the mesh builder uses.
    if _HAS_MODELS:
        uv_verts = get_tile_uv_vertices(globe_grid, face_id)
        uv_poly = [np.array(uv, dtype=np.float64) for uv in uv_verts]
    else:
        # Fallback: derive from globe_grid vertices
        face = globe_grid.faces[face_id]
        uv_poly = []
        for vid in face.vertex_ids:
            v = globe_grid.vertices[vid]
            pt3 = np.array([v.x, v.y, v.z], dtype=np.float64)
            uv = project_and_normalize(pt3, center, tangent, bitangent, uv_bounds)
            uv_poly.append(np.array(uv, dtype=np.float64))

    uv_arr = np.array(uv_poly)
    uv_centroid = uv_arr.mean(axis=0)

    # ── Corresponding 2D positions from the detail grid ─────────
    # Collect all vertex positions
    all_detail_verts = []
    for v in detail_grid.vertices.values():
        if v.has_position():
            all_detail_verts.append(np.array([v.x, v.y], dtype=np.float64))

    if len(all_detail_verts) < 3:
        return UVTransform(A=np.eye(2), t=np.zeros(2))

    all_detail_verts_arr = np.array(all_detail_verts)
    detail_centroid = all_detail_verts_arr.mean(axis=0)
    dists = np.linalg.norm(all_detail_verts_arr - detail_centroid, axis=1)
    max_dist = dists.max()

    if max_dist < 1e-12:
        return UVTransform(A=np.eye(2), t=np.zeros(2))

    # The boundary vertices are those at (approximately) the maximum
    # distance.  For a regular hex/pent grid they sit on the outer ring.
    boundary_threshold = max_dist * 0.85
    boundary_mask = dists >= boundary_threshold
    boundary_verts = all_detail_verts_arr[boundary_mask]

    # ── Match polygon UV vertices to boundary detail vertices ───
    n_poly = len(uv_poly)
    uv_angles = np.arctan2(uv_arr[:, 1] - uv_centroid[1],
                           uv_arr[:, 0] - uv_centroid[0])
    boundary_angles = np.arctan2(boundary_verts[:, 1] - detail_centroid[1],
                                 boundary_verts[:, 0] - detail_centroid[0])

    src_points = []  # detail-2D
    dst_points = []  # UV [0,1]

    for i in range(n_poly):
        target_angle = uv_angles[i]
        angle_diffs = np.abs(np.arctan2(
            np.sin(boundary_angles - target_angle),
            np.cos(boundary_angles - target_angle),
        ))
        best_idx = np.argmin(angle_diffs)
        src_points.append(boundary_verts[best_idx])
        dst_points.append(uv_arr[i])

    src_pts = np.array(src_points)  # (N, 2)
    dst_pts = np.array(dst_points)  # (N, 2)

    # Solve for similarity transform: [u, v] = [a, -b; b, a] * [x, y] + [tx, ty]
    # This is a constrained affine (rotation + uniform scale + translation).
    # We centre both point sets first, then solve for rotation+scale.

    src_c = src_pts.mean(axis=0)
    dst_c = dst_pts.mean(axis=0)

    src_rel = src_pts - src_c
    dst_rel = dst_pts - dst_c

    # Least-squares similarity: minimise ||dst_rel - R*s * src_rel||²
    # where R*s = [[a, -b], [b, a]]
    # Normal equations:
    #   a = Σ(sx*dx + sy*dy) / Σ(sx² + sy²)
    #   b = Σ(sx*dy - sy*dx) / Σ(sx² + sy²)
    ss = np.sum(src_rel[:, 0]**2 + src_rel[:, 1]**2)
    if ss < 1e-20:
        # Degenerate
        return UVTransform(A=np.eye(2), t=dst_c - src_c)

    a = np.sum(src_rel[:, 0] * dst_rel[:, 0] + src_rel[:, 1] * dst_rel[:, 1]) / ss
    b = np.sum(src_rel[:, 0] * dst_rel[:, 1] - src_rel[:, 1] * dst_rel[:, 0]) / ss

    A = np.array([[a, -b], [b, a]], dtype=np.float64)
    t = dst_c - A @ src_c

    return UVTransform(A=A, t=t)


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
