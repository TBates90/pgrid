"""Fast placeholder atlas generation without terrain pipeline.

Precomputes a topology artifact (tile-index map, UV layout, mesh) for a
given ``(frequency, detail_rings, tile_size, gutter)`` combination and
caches it to disk.
Subsequent recolour calls are pure numpy array indexing — no scipy warp, no
PIL polygon rasterisation, no terrain generation.

Artifact cache
--------------
* Process-level memory cache — zero latency on repeat calls.
* Persistent disk cache under ``~/.cache/pgrid/placeholder/``
  (overridable via the ``PGRID_ARTIFACT_CACHE_DIR`` environment variable).
  Each artifact is stored as a pair of files: ``<key>.npz`` + ``<key>.json``.
* Increment :data:`ARTIFACT_VERSION` to invalidate all cached artifacts.

Debugging
---------
Set ``PGRID_PLACEHOLDER_DEBUG_DIR`` to dump per-run debug outputs:
atlas PNG, index-map visualisation PNG, and JSON metadata.

Public API
----------
- :class:`PlaceholderAtlasArtifact`
- :func:`get_or_build_artifact`
- :func:`recolor_atlas`
- :func:`generate_placeholder_atlas`
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from .rendering.detail_cell_contract import normalize_detail_cells_tiles_with_report

if TYPE_CHECKING:
    from .integration import PlaceholderAtlasSpec
    from .integration_atlas import PlanetAtlasResult

LOGGER = logging.getLogger(__name__)


def _detail_cells_strict_mode() -> bool:
    return os.environ.get("PGRID_DETAIL_CELLS_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

# Increment to invalidate all cached artifacts across all machines.
ARTIFACT_VERSION: int = 36

# Per-process in-memory cache: topology_key → PlaceholderAtlasArtifact
_ARTIFACT_CACHE: Dict[str, "PlaceholderAtlasArtifact"] = {}
_DETAIL_CELLS_CACHE: Dict[Tuple[int, int], Dict[str, Any]] = {}
_SEAM_STRIPS_CACHE: Dict[Tuple[int, int], Dict[str, Any]] = {}

# uint16 sentinel for atlas pixels that belong to no tile.
_BACKGROUND_IDX: int = 0xFFFF


# ═══════════════════════════════════════════════════════════════════
# Artifact dataclass
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PlaceholderAtlasArtifact:
    """Precomputed topology artifact for fast placeholder atlas generation.

    Attributes
    ----------
    tile_index_map : np.ndarray
        Shape ``(atlas_h, atlas_w)`` uint16.  Each pixel holds the index of
        its owning tile in *face_ids*, or ``0xFFFF`` for background.
    face_ids : tuple[str, ...]
        Ordered macro tile face ids matching the slot assignment.
    index_keys : tuple[str, ...]
        Ordered keys for label indices in ``tile_index_map``. For the
        detail-indexed artifact this is typically ``"<tile_slug>:<cell_id>"``.
    index_points : np.ndarray
        Shape ``(len(index_keys), 3)`` float32 anchor points on the unit
        sphere, aligned with *index_keys*. Used for smooth seam-safe noise.
    seam_mask : np.ndarray
        Shape ``(atlas_h, atlas_w)`` uint8 seam-band mask (0..255) used for
        seam-only post blend during recolor.
    seam_partner_map : np.ndarray
        Shape ``(atlas_h, atlas_w)`` uint16. For seam pixels, stores a nearby
        partner label index used for edge-aware cross-blending.
    seam_alpha : np.ndarray
        Shape ``(atlas_h, atlas_w)`` uint8 blend weight map for seam blending.
    uv_layout : dict
        ``{face_id: (u_min, v_min, u_max, v_max)}`` — atlas UV slot coords.
    vertex_data : np.ndarray
        Globe mesh vertex buffer (N, 8) float32.
    index_data : np.ndarray
        Globe mesh index buffer (M, 3) uint32.
    frequency : int
    detail_rings : int
        Goldberg polyhedron frequency.
    atlas_width : int
    atlas_height : int
    topology_key : str
        16-character hex key that identifies this artifact's topology.
    topology_mode : str
        Artifact generation mode (e.g. detail-cell-indexed-seam-aligned).
    seam_metrics_summary : dict
        Aggregated seam metrics summary from atlas alignment diagnostics.
    """

    tile_index_map: np.ndarray
    face_ids: Tuple[str, ...]
    index_keys: Tuple[str, ...]
    index_points: np.ndarray
    seam_mask: np.ndarray
    seam_partner_map: np.ndarray
    seam_alpha: np.ndarray
    uv_layout: Dict[str, Tuple[float, float, float, float]]
    vertex_data: np.ndarray
    index_data: np.ndarray
    frequency: int
    detail_rings: int
    atlas_width: int
    atlas_height: int
    topology_key: str
    topology_mode: str
    seam_metrics_summary: Dict[str, Any]


# ═══════════════════════════════════════════════════════════════════
# Key + cache-dir helpers
# ═══════════════════════════════════════════════════════════════════


def _topology_key(frequency: int, detail_rings: int, tile_size: int, gutter: int) -> str:
    """Stable 16-char hex cache key for a ``(frequency, detail_rings,
    tile_size, gutter)``
    topology, incorporating :data:`ARTIFACT_VERSION`."""
    raw = (
        f"placeholder:v{ARTIFACT_VERSION}:{frequency}:"
        f"{detail_rings}:{tile_size}:{gutter}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _artifact_cache_dir() -> Optional[Path]:
    """Return the on-disk artifact cache directory, creating it if needed.

    Returns *None* if the directory cannot be created (disk caching
    disabled gracefully in that case).
    """
    env = os.environ.get("PGRID_ARTIFACT_CACHE_DIR")
    base = Path(env) if env else Path.home() / ".cache" / "pgrid" / "placeholder"
    try:
        base.mkdir(parents=True, exist_ok=True)
        return base
    except OSError:
        return None


# ═══════════════════════════════════════════════════════════════════
# UV layout computation (no rendering)
# ═══════════════════════════════════════════════════════════════════


def _compute_uv_layout(
    face_ids: List[str],
    tile_size: int,
    gutter: int,
) -> Tuple[Dict[str, Tuple[float, float, float, float]], int, int]:
    """Compute atlas UV layout for *face_ids* without any rendering.

    Replicates the slot-assignment logic of ``build_polygon_cut_atlas``
    (using ``max_gutter`` = *gutter*, the behaviour when no per-type
    gutter overrides are active).

    Returns
    -------
    (uv_layout, atlas_w, atlas_h)
    """
    from .rendering.atlas_utils import compute_atlas_layout

    n = len(face_ids)
    cols, _rows, atlas_w, atlas_h = compute_atlas_layout(n, tile_size, gutter)
    slot_size = tile_size + 2 * gutter

    uv_layout: Dict[str, Tuple[float, float, float, float]] = {}
    for idx, fid in enumerate(face_ids):
        col = idx % cols
        row = idx // cols
        slot_x = col * slot_size
        slot_y = row * slot_size
        inner_x = slot_x + gutter
        inner_y = slot_y + gutter
        u_min = inner_x / atlas_w
        u_max = (inner_x + tile_size) / atlas_w
        v_min = 1.0 - (inner_y + tile_size) / atlas_h
        v_max = 1.0 - inner_y / atlas_h
        uv_layout[fid] = (u_min, v_min, u_max, v_max)

    return uv_layout, atlas_w, atlas_h


# ═══════════════════════════════════════════════════════════════════
# Tile-index map rasterisation
# ═══════════════════════════════════════════════════════════════════


def _build_tile_index_map(
    face_ids: List[str],
    uv_layout: Dict[str, Tuple[float, float, float, float]],
    globe_grid: object,
    tile_size: int,
    atlas_w: int,
    atlas_h: int,
) -> np.ndarray:
    """Rasterise each tile's UV polygon into an atlas-sized integer index map.

    Each pixel in the result holds the tile index (0-based position in
    *face_ids*) so a simple LUT lookup can produce any colour atlas.
    Background pixels (outside all polygons) are seeded with
    ``_BACKGROUND_IDX`` and then propagated from the nearest tile pixel to
    cover gutter regions.

    Returns
    -------
    np.ndarray
        Shape ``(atlas_h, atlas_w)`` uint16.
    """
    from PIL import Image, ImageDraw
    from scipy.ndimage import distance_transform_edt

    from .rendering.uv_texture import get_tile_uv_vertices

    # PIL "I" (int32) mode — supports the full uint16 range.
    img = Image.new("I", (atlas_w, atlas_h), _BACKGROUND_IDX)
    draw = ImageDraw.Draw(img)

    for tile_idx, fid in enumerate(face_ids):
        if fid not in uv_layout:
            continue
        u_min, v_min, u_max, v_max = uv_layout[fid]

        # Top-left pixel of the inner slot (PIL y=0 is top; v_max is the
        # uppermost v, which maps to the smallest y).
        inner_x_px = round(u_min * atlas_w)
        inner_y_px = round((1.0 - v_max) * atlas_h)

        try:
            uv_corners = get_tile_uv_vertices(globe_grid, fid)
        except Exception:
            LOGGER.debug("Cannot get UV corners for %s — filling slot", fid)
            # Fallback: fill the whole inner slot rectangle.
            draw.rectangle(
                [
                    inner_x_px,
                    inner_y_px,
                    inner_x_px + tile_size - 1,
                    inner_y_px + tile_size - 1,
                ],
                fill=tile_idx,
            )
            continue

        # Map tile-local UV [0, 1] → atlas pixel coords (Y-flipped).
        px_corners = [
            (
                round(inner_x_px + u * tile_size),
                round(inner_y_px + (1.0 - v) * tile_size),
            )
            for u, v in uv_corners
        ]
        draw.polygon(px_corners, fill=tile_idx)

    # PIL "I" → numpy int32 → clip to uint16 range.
    arr = np.array(img, dtype=np.int32)
    tile_index_map = arr.clip(0, 0xFFFF).astype(np.uint16)

    # Propagate nearest-tile index into background pixels (covers gutter).
    bg_mask = tile_index_map == _BACKGROUND_IDX
    if bg_mask.any():
        _, near_ij = distance_transform_edt(bg_mask, return_indices=True)
        tile_index_map[bg_mask] = tile_index_map[
            near_ij[0][bg_mask], near_ij[1][bg_mask]
        ]

    return tile_index_map


def _build_detail_index_map(
    globe_grid: object,
    face_ids: List[str],
    uv_layout: Dict[str, Tuple[float, float, float, float]],
    atlas_w: int,
    atlas_h: int,
    detail_rings: int,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Rasterise detail-cell polygons directly into atlas pixel space.

    This avoids bilinear interpolation artefacts from image-space warps,
    preserving exact integer labels in the artifact index map.
    """
    from PIL import Image, ImageDraw
    from scipy.ndimage import distance_transform_edt

    from .core.geometry import ordered_face_vertices
    from .detail.tile_detail import DetailGridCollection, TileDetailSpec
    from .rendering.uv_texture import (
        compute_detail_to_uv_transform,
        compute_tile_basis,
    )

    spec = TileDetailSpec(detail_rings=detail_rings)
    coll = DetailGridCollection.build(globe_grid, spec)

    slug_lookup: Dict[str, str] = {}
    try:
        slug_lookup = globe_grid.build_slug_lookup()  # type: ignore[attr-defined]
    except Exception:
        slug_lookup = {fid: fid for fid in face_ids}

    img = Image.new("I", (atlas_w, atlas_h), _BACKGROUND_IDX)
    draw = ImageDraw.Draw(img)

    index_lookup: Dict[str, int] = {}
    index_keys: List[str] = []

    for fid in face_ids:
        dg, _store = coll.get(fid)
        tile_slug = slug_lookup.get(fid, fid)
        u_min, v_min, u_max, v_max = uv_layout[fid]
        uv_bounds = (u_min, v_min, u_max, v_max)

        center, normal, tangent, bitangent = compute_tile_basis(globe_grid, fid)
        _ = normal
        xform = compute_detail_to_uv_transform(
            globe_grid,
            fid,
            dg,
            center,
            tangent,
            bitangent,
            uv_bounds,
        )

        for local_face_id in sorted(dg.faces.keys()):
            face = dg.faces[local_face_id]
            ordered_ids = ordered_face_vertices(dg.vertices, face)
            px_poly: List[Tuple[int, int]] = []
            for vid in ordered_ids:
                v = dg.vertices.get(vid)
                if v is None or not v.has_position():
                    continue
                u_local, v_local = xform.apply(float(v.x), float(v.y))
                u_abs = u_min + u_local * (u_max - u_min)
                v_abs = v_min + v_local * (v_max - v_min)
                px = int(round(u_abs * atlas_w))
                py = int(round((1.0 - v_abs) * atlas_h))
                px = min(max(px, 0), atlas_w - 1)
                py = min(max(py, 0), atlas_h - 1)
                px_poly.append((px, py))

            if len(px_poly) < 3:
                continue

            index_key = f"{tile_slug}:{local_face_id}"
            idx = index_lookup.get(index_key)
            if idx is None:
                idx = len(index_keys)
                index_lookup[index_key] = idx
                index_keys.append(index_key)

            draw.polygon(px_poly, fill=idx)

    arr = np.array(img, dtype=np.int32)
    index_map = arr.clip(0, 0xFFFF).astype(np.uint16)

    bg_mask = index_map == _BACKGROUND_IDX
    if bg_mask.any():
        _, near_ij = distance_transform_edt(bg_mask, return_indices=True)
        index_map[bg_mask] = index_map[near_ij[0][bg_mask], near_ij[1][bg_mask]]

    return index_map, tuple(index_keys)


def _build_index_points(
    globe_grid: object,
    detail_rings: int,
    index_keys: Tuple[str, ...],
) -> np.ndarray:
    """Build per-index unit-sphere anchor points aligned with *index_keys*."""
    from .rendering.detail_centers import build_slug_keyed_detail_centers

    detail_cells = build_slug_keyed_detail_centers(globe_grid, detail_rings=detail_rings)

    by_tile: Dict[str, Dict[str, List[float]]] = {}
    for tile_slug, cells in detail_cells.items():
        local_map: Dict[str, List[float]] = {}
        for cell in cells:
            cid = str(cell.get("id", ""))
            center = cell.get("canonical_center_3d") or cell.get("center_3d")
            if cid and isinstance(center, list) and len(center) >= 3:
                local_map[cid] = center
        by_tile[str(tile_slug)] = local_map

    points = np.zeros((len(index_keys), 3), dtype=np.float32)
    for i, key in enumerate(index_keys):
        if ":" not in key:
            continue
        tile_slug, local_id = key.split(":", 1)
        center = by_tile.get(tile_slug, {}).get(local_id)
        if center is None:
            continue
        points[i, 0] = float(center[0])
        points[i, 1] = float(center[1])
        points[i, 2] = float(center[2])
    return points


def _encode_index_rgb(idx: int) -> Tuple[int, int, int]:
    """Encode a 24-bit integer label index to RGB bytes."""
    val = idx + 1  # reserve 0 for background/miss
    return ((val >> 16) & 0xFF, (val >> 8) & 0xFF, val & 0xFF)


def _decode_index_map_from_rgb(
    atlas_rgb: np.ndarray,
    color_to_index: Dict[int, int],
) -> np.ndarray:
    """Decode atlas RGB into uint16 index map with nearest-valid fill."""
    from scipy.ndimage import distance_transform_edt

    packed = (
        (atlas_rgb[:, :, 0].astype(np.uint32) << 16)
        | (atlas_rgb[:, :, 1].astype(np.uint32) << 8)
        | atlas_rgb[:, :, 2].astype(np.uint32)
    )

    keys = np.array(sorted(color_to_index.keys()), dtype=np.uint32)
    vals = np.array([color_to_index[int(k)] for k in keys], dtype=np.uint32)

    flat = packed.ravel()
    pos = np.searchsorted(keys, flat)
    match = (pos < len(keys)) & (keys[np.clip(pos, 0, len(keys) - 1)] == flat)

    out = np.full(flat.shape, _BACKGROUND_IDX, dtype=np.uint16)
    if match.any():
        out[match] = vals[pos[match]].astype(np.uint16)

    out_map = out.reshape(packed.shape[0], packed.shape[1])

    bg_mask = out_map == _BACKGROUND_IDX
    if bg_mask.any():
        _, near_ij = distance_transform_edt(bg_mask, return_indices=True)
        out_map[bg_mask] = out_map[near_ij[0][bg_mask], near_ij[1][bg_mask]]

    return out_map


def _resolve_prefixed_label_key(
    prefixed_face_id: str,
    composite: object,
    slug_lookup: Dict[str, str],
    tile_face_id: str,
) -> str:
    """Resolve a merged-face id from CompositeGrid to canonical label key."""
    id_prefixes: Dict[str, str] = dict(getattr(composite, "id_prefixes", {}) or {})
    prefixes = sorted(
        ((comp_name, pref) for comp_name, pref in id_prefixes.items()),
        key=lambda x: len(x[1]),
        reverse=True,
    )

    for comp_name, pref in prefixes:
        if prefixed_face_id.startswith(pref):
            local_id = prefixed_face_id[len(pref):]
            tile_slug = slug_lookup.get(comp_name, comp_name)
            return f"{tile_slug}:{local_id}"

    tile_slug = slug_lookup.get(tile_face_id, tile_face_id)
    return f"{tile_slug}:{prefixed_face_id}"


def _build_detail_index_map_seam_aligned(
    globe_grid: object,
    face_ids: List[str],
    detail_rings: int,
    tile_size: int,
    gutter: int,
) -> Tuple[np.ndarray, Tuple[str, ...], Dict[str, Tuple[float, float, float, float]], int, int, Dict[str, Any]]:
    """Build seam-aligned detail index map using polygon-cut atlas pipeline.

    This path reuses CompositeGrid assembly and polygon-cut warp packing
    so macro-tile boundaries align like the full atlas path.
    """
    from PIL import Image

    from .detail.tile_detail import DetailGridCollection, TileDetailSpec, build_tile_with_neighbours
    from .integration_atlas import _render_tile_analytical
    from .rendering.tile_uv_align import (
        build_polygon_cut_atlas,
        compute_tile_view_limits,
        compute_uniform_half_span,
    )

    spec = TileDetailSpec(detail_rings=detail_rings)
    coll = DetailGridCollection.build(globe_grid, spec)

    slug_lookup: Dict[str, str] = {}
    try:
        slug_lookup = globe_grid.build_slug_lookup()  # type: ignore[attr-defined]
    except Exception:
        slug_lookup = {fid: fid for fid in face_ids}

    composites: Dict[str, object] = {}
    detail_grids: Dict[str, object] = {}
    for fid in face_ids:
        composites[fid] = build_tile_with_neighbours(
            coll,
            fid,
            globe_grid,
            skip_neighbour_closure=True,
        )
        detail_grids[fid] = coll.get(fid)[0]

    uniform_hs = compute_uniform_half_span(composites, face_ids)

    index_lookup: Dict[str, int] = {}
    index_keys: List[str] = []
    color_to_index: Dict[int, int] = {}
    tile_images: Dict[str, Image.Image] = {}

    def _index_for_key(key: str) -> int:
        idx = index_lookup.get(key)
        if idx is not None:
            return idx
        idx = len(index_keys)
        index_lookup[key] = idx
        index_keys.append(key)
        r, g, b = _encode_index_rgb(idx)
        color_to_index[(r << 16) | (g << 8) | b] = idx
        return idx

    for fid in face_ids:
        composite = composites[fid]
        merged_grid = composite.merged

        def _colour_fn(face_id_inner, _grid, _face):
            key = _resolve_prefixed_label_key(face_id_inner, composite, slug_lookup, fid)
            idx = _index_for_key(key)
            r, g, b = _encode_index_rgb(idx)
            return (r / 255.0, g / 255.0, b / 255.0)

        xlim, ylim = compute_tile_view_limits(
            composite,
            fid,
            uniform_half_span=uniform_hs,
        )
        tile_images[fid] = _render_tile_analytical(
            merged_grid,
            _colour_fn,
            tile_size,
            xlim,
            ylim,
            bg_colour=(1.0, 0.0, 1.0),
        )

    seam_metrics: Dict[str, Any] = {}
    atlas_img, uv_layout = build_polygon_cut_atlas(
        tile_images,
        composites,
        detail_grids,
        globe_grid,
        face_ids,
        tile_size=tile_size,
        gutter=gutter,
        uniform_half_span=uniform_hs,
        warp_sample_order=0,
        warp_dilate_cval=False,
        # Placeholder path: keep pentagon UV transform neutral so orientation
        # matches strict corner pairing from current atlas alignment logic.
        pent_uv_rotation=0.0,
        pentagon_allow_reflection=True,
        pent_edge_interior_pull=0.0,
        hex_pent_edge_interior_pull=0.0,
        seam_metrics_out=seam_metrics,
    )

    atlas_rgb = np.array(atlas_img.convert("RGB"), dtype=np.uint8)
    index_map = _decode_index_map_from_rgb(atlas_rgb, color_to_index)

    return index_map, tuple(index_keys), uv_layout, atlas_img.size[0], atlas_img.size[1], dict(seam_metrics.get("summary") or {})


def _build_macro_seam_mask(
    globe_grid: object,
    face_ids: List[str],
    uv_layout: Dict[str, Tuple[float, float, float, float]],
    atlas_w: int,
    atlas_h: int,
    *,
    band_px: int,
) -> np.ndarray:
    """Build an atlas-space mask covering macro tile edge seam bands."""
    from PIL import Image, ImageDraw
    from scipy.ndimage import gaussian_filter

    from .rendering.uv_texture import get_tile_uv_vertices

    band = max(1, int(band_px))
    mask_img = Image.new("L", (atlas_w, atlas_h), 0)
    draw = ImageDraw.Draw(mask_img)

    for fid in face_ids:
        if fid not in uv_layout:
            continue
        u_min, v_min, u_max, v_max = uv_layout[fid]
        inner_x = u_min * atlas_w
        inner_y = (1.0 - v_max) * atlas_h
        inner_w = max(1.0, (u_max - u_min) * atlas_w)
        inner_h = max(1.0, (v_max - v_min) * atlas_h)

        try:
            uv_corners = get_tile_uv_vertices(globe_grid, fid)
        except Exception:
            uv_corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

        pts = [
            (
                float(inner_x + u * inner_w),
                float(inner_y + (1.0 - v) * inner_h),
            )
            for u, v in uv_corners
        ]

        if len(pts) < 3:
            continue
        for i in range(len(pts)):
            p0 = pts[i]
            p1 = pts[(i + 1) % len(pts)]
            draw.line([p0, p1], fill=255, width=band)

    mask = np.array(mask_img, dtype=np.float32)
    if np.max(mask) <= 0.0:
        return np.zeros((atlas_h, atlas_w), dtype=np.uint8)

    sigma = max(0.5, band * 0.15)
    softened = gaussian_filter(mask, sigma=sigma)
    if float(np.max(softened)) > 1e-6:
        softened /= float(np.max(softened))
    softened = np.clip(softened * 255.0, 0.0, 255.0).astype(np.uint8)
    return softened


def _build_seam_partner_and_alpha(
    index_map: np.ndarray,
    seam_mask: np.ndarray,
    *,
    radius: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build per-pixel seam partner labels and seam alpha map.

    For each seam pixel, we keep its own label in ``index_map`` and find a
    nearby dominant *different* label within a small window. Runtime recolor
    can then cross-blend own vs partner colours deterministically.
    """
    h, w = index_map.shape
    partner = index_map.copy().astype(np.uint16)
    alpha = seam_mask.copy().astype(np.uint8)

    ys, xs = np.nonzero(seam_mask > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        own = int(index_map[y, x])
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        window = index_map[y0:y1, x0:x1].ravel()
        if window.size == 0:
            continue
        others = window[window != own]
        if others.size == 0:
            continue
        vals, counts = np.unique(others, return_counts=True)
        partner[y, x] = np.uint16(vals[int(np.argmax(counts))])

    return partner, alpha


# ═══════════════════════════════════════════════════════════════════
# Artifact build
# ═══════════════════════════════════════════════════════════════════


def _build_artifact(
    frequency: int,
    detail_rings: int,
    tile_size: int,
    gutter: int,
) -> PlaceholderAtlasArtifact:
    """Build a topology artifact without rendering or terrain generation.

    Steps
    -----
    1. ``build_globe_grid`` — pure geometry, no elevation.
    2. ``sorted(grid.faces.keys())`` — same ordering as
       ``DetailGridCollection.face_ids``.
    3. Slot-geometry UV layout computation (no PIL, no warp).
    4. PIL polygon rasterisation → tile-index map.
    5. ``build_batched_globe_mesh`` — mesh vertex/index buffers.
    """
    from .globe.globe import build_globe_grid
    from .rendering.globe_renderer_v2 import build_batched_globe_mesh

    t0 = time.monotonic()
    LOGGER.info(
        "Building placeholder atlas artifact: freq=%d detail_rings=%d tile_size=%d gutter=%d",
        frequency,
        detail_rings,
        tile_size,
        gutter,
    )

    grid = build_globe_grid(frequency)
    face_ids: List[str] = sorted(grid.faces.keys())

    uv_layout, atlas_w, atlas_h = _compute_uv_layout(face_ids, tile_size, gutter)

    seam_metrics_summary: Dict[str, Any] = {}
    try:
        (
            tile_index_map,
            index_keys,
            uv_layout,
            atlas_w,
            atlas_h,
            seam_metrics_summary,
        ) = _build_detail_index_map_seam_aligned(
            grid,
            face_ids,
            detail_rings,
            tile_size,
            gutter,
        )
        try:
            index_points = _build_index_points(grid, detail_rings, index_keys)
        except Exception:
            LOGGER.debug("Failed to build index points; falling back to hash jitter", exc_info=True)
            index_points = np.zeros((len(index_keys), 3), dtype=np.float32)
        topology_mode = "detail-cell-indexed-seam-aligned"
    except Exception:
        LOGGER.warning(
            "Seam-aligned detail index build failed; falling back to direct detail map",
            exc_info=True,
        )
        try:
            tile_index_map, index_keys = _build_detail_index_map(
                grid,
                face_ids,
                uv_layout,
                atlas_w,
                atlas_h,
                detail_rings,
            )
            try:
                index_points = _build_index_points(grid, detail_rings, index_keys)
            except Exception:
                LOGGER.debug("Failed to build index points; falling back to hash jitter", exc_info=True)
                index_points = np.zeros((len(index_keys), 3), dtype=np.float32)
            topology_mode = "detail-cell-indexed-direct"
        except Exception:
            LOGGER.warning(
                "Falling back to macro-face index map for placeholder artifact",
                exc_info=True,
            )
            tile_index_map = _build_tile_index_map(
                face_ids, uv_layout, grid, tile_size, atlas_w, atlas_h
            )
            index_keys = tuple(face_ids)
            index_points = np.zeros((len(index_keys), 3), dtype=np.float32)
            topology_mode = "macro-face-fast"

    vertex_data, index_data = build_batched_globe_mesh(
        frequency, uv_layout, subdivisions=3
    )

    try:
        seam_band_px = max(4, int(math.ceil(tile_size / 32.0)))
        seam_mask = _build_macro_seam_mask(
            grid,
            face_ids,
            uv_layout,
            atlas_w,
            atlas_h,
            band_px=seam_band_px,
        )
        seam_partner_map, seam_alpha = _build_seam_partner_and_alpha(
            tile_index_map,
            seam_mask,
            radius=max(3, seam_band_px // 2),
        )
    except Exception:
        LOGGER.debug("Failed to build seam mask; disabling seam blend", exc_info=True)
        seam_mask = np.zeros((atlas_h, atlas_w), dtype=np.uint8)
        seam_partner_map = tile_index_map.copy().astype(np.uint16)
        seam_alpha = np.zeros((atlas_h, atlas_w), dtype=np.uint8)

    key = _topology_key(frequency, detail_rings, tile_size, gutter)
    artifact = PlaceholderAtlasArtifact(
        tile_index_map=tile_index_map,
        face_ids=tuple(face_ids),
        index_keys=index_keys,
        index_points=index_points,
        seam_mask=seam_mask,
        seam_partner_map=seam_partner_map,
        seam_alpha=seam_alpha,
        uv_layout=uv_layout,
        vertex_data=vertex_data,
        index_data=index_data,
        frequency=frequency,
        detail_rings=detail_rings,
        atlas_width=atlas_w,
        atlas_height=atlas_h,
        topology_key=key,
        topology_mode=topology_mode,
        seam_metrics_summary=seam_metrics_summary,
    )
    LOGGER.info(
        "Artifact built in %.2fs — %d tiles, %d index keys, atlas %dx%d (%s)",
        time.monotonic() - t0,
        len(face_ids),
        len(index_keys),
        atlas_w,
        atlas_h,
        topology_mode,
    )
    return artifact


# ═══════════════════════════════════════════════════════════════════
# Disk serialisation
# ═══════════════════════════════════════════════════════════════════


def save_artifact(artifact: PlaceholderAtlasArtifact, path: Path) -> None:
    """Save *artifact* as ``<path>.npz`` + ``<path>.json``."""
    path = Path(path)
    npz_path = path.with_suffix(".npz")
    meta_path = path.with_suffix(".json")

    n = len(artifact.face_ids)
    uv_arr = np.empty((n, 4), dtype=np.float64)
    for i, fid in enumerate(artifact.face_ids):
        uv_arr[i] = artifact.uv_layout[fid]

    np.savez_compressed(
        npz_path,
        tile_index_map=artifact.tile_index_map,
        vertex_data=artifact.vertex_data,
        index_data=artifact.index_data,
        uv_data=uv_arr,
        index_points=artifact.index_points,
        seam_mask=artifact.seam_mask,
        seam_partner_map=artifact.seam_partner_map,
        seam_alpha=artifact.seam_alpha,
    )
    meta: Dict[str, Any] = {
        "face_ids": list(artifact.face_ids),
        "index_keys": list(artifact.index_keys),
        "frequency": artifact.frequency,
        "detail_rings": artifact.detail_rings,
        "atlas_width": artifact.atlas_width,
        "atlas_height": artifact.atlas_height,
        "topology_key": artifact.topology_key,
        "topology_mode": artifact.topology_mode,
        "seam_metrics_summary": dict(artifact.seam_metrics_summary),
    }
    meta_path.write_text(json.dumps(meta))
    LOGGER.debug("Saved placeholder artifact to %s", npz_path)


def load_artifact(path: Path) -> Optional[PlaceholderAtlasArtifact]:
    """Load a previously saved artifact; return *None* on any error."""
    path = Path(path)
    npz_path = path.with_suffix(".npz")
    meta_path = path.with_suffix(".json")
    try:
        meta = json.loads(meta_path.read_text())
        data = np.load(npz_path)
        face_ids: Tuple[str, ...] = tuple(meta["face_ids"])
        index_keys_meta = meta.get("index_keys")
        index_keys: Tuple[str, ...]
        if isinstance(index_keys_meta, list) and index_keys_meta:
            index_keys = tuple(str(k) for k in index_keys_meta)
        else:
            # Backward compatibility for artifacts written before index_keys.
            index_keys = face_ids
        uv_arr: np.ndarray = data["uv_data"]
        idx_pts: np.ndarray
        if "index_points" in data:
            idx_pts = data["index_points"].astype(np.float32)
        else:
            idx_pts = np.zeros((len(index_keys), 3), dtype=np.float32)
        if "seam_mask" in data:
            seam_mask = data["seam_mask"].astype(np.uint8)
        else:
            seam_mask = np.zeros((int(meta["atlas_height"]), int(meta["atlas_width"])), dtype=np.uint8)
        if "seam_partner_map" in data:
            seam_partner_map = data["seam_partner_map"].astype(np.uint16)
        else:
            seam_partner_map = data["tile_index_map"].astype(np.uint16)
        if "seam_alpha" in data:
            seam_alpha = data["seam_alpha"].astype(np.uint8)
        else:
            seam_alpha = seam_mask.astype(np.uint8)
        uv_layout: Dict[str, Tuple[float, float, float, float]] = {
            fid: (
                float(uv_arr[i, 0]),
                float(uv_arr[i, 1]),
                float(uv_arr[i, 2]),
                float(uv_arr[i, 3]),
            )
            for i, fid in enumerate(face_ids)
        }
        return PlaceholderAtlasArtifact(
            tile_index_map=data["tile_index_map"],
            face_ids=face_ids,
            index_keys=index_keys,
            index_points=idx_pts,
            seam_mask=seam_mask,
            seam_partner_map=seam_partner_map,
            seam_alpha=seam_alpha,
            uv_layout=uv_layout,
            vertex_data=data["vertex_data"],
            index_data=data["index_data"],
            frequency=int(meta["frequency"]),
            detail_rings=int(meta.get("detail_rings", 2)),
            atlas_width=int(meta["atlas_width"]),
            atlas_height=int(meta["atlas_height"]),
            topology_key=str(meta["topology_key"]),
            topology_mode=str(meta.get("topology_mode", "legacy")),
            seam_metrics_summary=dict(meta.get("seam_metrics_summary") or {}),
        )
    except Exception:
        LOGGER.debug(
            "Failed to load placeholder artifact from %s", path, exc_info=True
        )
        return None


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════


def get_or_build_artifact(spec: "PlaceholderAtlasSpec") -> PlaceholderAtlasArtifact:
    """Return a cached artifact for *spec*, building and caching if needed.

    Lookup order: process-level memory cache → on-disk cache → build + save.
    Color fields (*base_color*, *noise_amount*, *seed*) do not affect the
    artifact or its cache key.
    """
    key = _topology_key(
        spec.frequency,
        spec.detail_rings,
        spec.tile_size,
        spec.gutter,
    )

    # 1. Memory cache.
    if key in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[key]

    # 2. Disk cache.
    arc_dir = _artifact_cache_dir()
    disk_path: Optional[Path] = (arc_dir / key) if arc_dir is not None else None
    if disk_path is not None:
        artifact = load_artifact(disk_path)
        if artifact is not None:
            _ARTIFACT_CACHE[key] = artifact
            LOGGER.info("Loaded placeholder artifact from disk (key=%s)", key)
            return artifact

    # 3. Build from scratch.
    artifact = _build_artifact(
        spec.frequency,
        spec.detail_rings,
        spec.tile_size,
        spec.gutter,
    )
    _ARTIFACT_CACHE[key] = artifact

    if disk_path is not None:
        try:
            save_artifact(artifact, disk_path)
        except Exception:
            LOGGER.warning(
                "Failed to save placeholder artifact to disk", exc_info=True
            )

    return artifact


def recolor_atlas(
    artifact: PlaceholderAtlasArtifact,
    spec: "PlaceholderAtlasSpec",
) -> bytes:
    """Apply the *spec* colour palette to *artifact* and return PNG bytes.

    Per-tile brightness jitter is applied deterministically from
    ``spec.seed`` + face id.  The entire operation is pure numpy —
    no PIL drawing, no scipy, no terrain pipeline.

    Complexity is O(atlas pixels) and typically completes in < 50 ms
    for a 512 px tile atlas.
    """
    from PIL import Image

    r0, g0, b0 = spec.base_color

    # Build a 65536-entry (uint16 range) LUT: tile_index → (R, G, B).
    # Index 65535 (background) remains black.
    lut = np.zeros((0x10000, 3), dtype=np.uint8)
    use_point_noise = (
        artifact.index_points.shape[0] == len(artifact.index_keys)
        and artifact.index_points.size > 0
        and float(np.max(np.abs(artifact.index_points))) > 1e-9
    )

    for tile_idx, unit_key in enumerate(artifact.index_keys):
        if use_point_noise:
            p = artifact.index_points[tile_idx].astype(np.float64)
            phase = float(spec.seed) * 0.173
            n1 = math.sin(p[0] * 11.71 + p[1] * 23.57 + p[2] * 7.31 + phase)
            n2 = math.sin(p[0] * 29.11 + p[1] * 5.83 + p[2] * 17.19 + phase * 1.7)
            jitter = 0.5 * (n1 + n2) * spec.noise_amount
        else:
            # Deterministic fallback jitter keyed on (seed, index key).
            digest = hashlib.md5(f"{spec.seed}:{unit_key}".encode()).digest()
            jitter = (digest[0] / 255.0 - 0.5) * 2.0 * spec.noise_amount
        r = max(0.0, min(1.0, r0 + jitter))
        g = max(0.0, min(1.0, g0 + jitter))
        b = max(0.0, min(1.0, b0 + jitter))
        lut[tile_idx] = (round(r * 255), round(g * 255), round(b * 255))

    # Apply LUT: (H, W) uint16 index map → (H, W, 3) uint8 RGB atlas.
    atlas_arr: np.ndarray = lut[artifact.tile_index_map]

    # Seam-only label-aware blend: cross-fade own vs partner label colour.
    if (
        artifact.seam_alpha.shape == atlas_arr.shape[:2]
        and artifact.seam_partner_map.shape == artifact.tile_index_map.shape
        and np.max(artifact.seam_alpha) > 0
    ):
        src = atlas_arr.astype(np.float32)
        partner_rgb = lut[artifact.seam_partner_map].astype(np.float32)
        alpha = np.clip(artifact.seam_alpha.astype(np.float32) / 255.0, 0.0, 1.0)
        alpha = np.clip(alpha * 0.85, 0.0, 0.85)
        valid_partner = (artifact.seam_partner_map != artifact.tile_index_map).astype(np.float32)
        a = (alpha * valid_partner)[..., None]
        atlas_arr = np.clip(src * (1.0 - a) + partner_rgb * a, 0.0, 255.0).astype(np.uint8)

    img = Image.fromarray(atlas_arr.astype(np.uint8), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _maybe_export_debug_artifacts(
    spec: "PlaceholderAtlasSpec",
    artifact: PlaceholderAtlasArtifact,
    atlas_png: bytes,
    metadata: Dict[str, Any],
) -> None:
    """Optionally dump placeholder debug artifacts to disk.

    Controlled by ``PGRID_PLACEHOLDER_DEBUG_DIR`` env var.
    """
    debug_dir_raw = os.environ.get("PGRID_PLACEHOLDER_DEBUG_DIR", "").strip()
    if not debug_dir_raw:
        return

    from PIL import Image

    debug_dir = Path(debug_dir_raw)
    debug_dir.mkdir(parents=True, exist_ok=True)

    stem = (
        f"f{spec.frequency}_r{spec.detail_rings}_t{spec.tile_size}_"
        f"s{spec.seed}_{artifact.topology_key}"
    )

    (debug_dir / f"{stem}.atlas.png").write_bytes(atlas_png)

    idx = artifact.tile_index_map.astype(np.uint32)
    vis = np.zeros((idx.shape[0], idx.shape[1], 3), dtype=np.uint8)
    vis[..., 0] = (idx * 73) % 256
    vis[..., 1] = (idx * 151) % 256
    vis[..., 2] = (idx * 199) % 256
    Image.fromarray(vis, "RGB").save(debug_dir / f"{stem}.index_map.png")

    payload = {
        "spec": {
            "frequency": spec.frequency,
            "detail_rings": spec.detail_rings,
            "tile_size": spec.tile_size,
            "gutter": spec.gutter,
            "seed": spec.seed,
            "noise_amount": spec.noise_amount,
        },
        "artifact": {
            "topology_key": artifact.topology_key,
            "topology_mode": artifact.topology_mode,
            "face_count": len(artifact.face_ids),
            "index_key_count": len(artifact.index_keys),
            "seam_mask_coverage": float(np.mean(artifact.seam_mask > 0)),
            "seam_partner_coverage": float(np.mean(artifact.seam_partner_map != artifact.tile_index_map)),
            "atlas_width": artifact.atlas_width,
            "atlas_height": artifact.atlas_height,
        },
        "metadata": metadata,
    }
    (debug_dir / f"{stem}.meta.json").write_text(json.dumps(payload, indent=2))


def _detail_cells_cache_path(frequency: int, detail_rings: int) -> Optional[Path]:
    arc_dir = _artifact_cache_dir()
    if arc_dir is None:
        return None
    return arc_dir / f"detail_cells_f{frequency}_r{detail_rings}.json"


def _get_or_build_detail_cells(
    frequency: int,
    detail_rings: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    key = (frequency, detail_rings)
    cached = _DETAIL_CELLS_CACHE.get(key)
    if cached is not None:
        normalized_cached, report = normalize_detail_cells_tiles_with_report(
            cached,
            strict=_detail_cells_strict_mode(),
        )
        _DETAIL_CELLS_CACHE[key] = normalized_cached
        return normalized_cached, report.to_dict()

    path = _detail_cells_cache_path(frequency, detail_rings)
    if path is not None and path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict):
                normalized_loaded, report = normalize_detail_cells_tiles_with_report(
                    loaded,
                    strict=_detail_cells_strict_mode(),
                )
                _DETAIL_CELLS_CACHE[key] = normalized_loaded
                return normalized_loaded, report.to_dict()
        except Exception:
            LOGGER.debug("Failed loading detail-cell cache %s", path, exc_info=True)

    from .globe.globe import build_globe_grid
    from .rendering.detail_centers import build_slug_keyed_detail_centers

    grid = build_globe_grid(frequency)
    built, report = normalize_detail_cells_tiles_with_report(
        build_slug_keyed_detail_centers(grid, detail_rings=detail_rings),
        strict=_detail_cells_strict_mode(),
    )
    _DETAIL_CELLS_CACHE[key] = built
    if path is not None:
        try:
            path.write_text(json.dumps(built))
        except Exception:
            LOGGER.debug("Failed writing detail-cell cache %s", path, exc_info=True)
    return built, report.to_dict()


def _get_or_build_seam_strips(
    frequency: int,
    detail_rings: int,
) -> Dict[str, Any]:
    key = (frequency, detail_rings)
    cached = _SEAM_STRIPS_CACHE.get(key)
    if cached is not None:
        return dict(cached)

    from .data.tile_data import FieldDef, TileDataStore, TileSchema
    from .globe.globe import build_globe_grid
    from .globe.globe_export import export_globe_payload
    from .rendering.seam_strips import build_seam_strip_payload_from_globe_payload

    try:
        grid = build_globe_grid(frequency)
        schema = TileSchema([FieldDef("elevation", float, 0.0)])
        store = TileDataStore(grid=grid, schema=schema)
        globe_payload = export_globe_payload(grid, store, ramp="satellite")
        seam_strips = build_seam_strip_payload_from_globe_payload(
            globe_payload,
            frequency=frequency,
            detail_rings=detail_rings,
        )
    except Exception:
        LOGGER.warning("Failed to build placeholder seam-strip payload", exc_info=True)
        seam_strips = {
            "metadata": {
                "frequency": int(frequency),
                "detail_rings": int(detail_rings),
                "seam_count": 0,
                "geometry_count": 0,
                "schema": "seam-strips.v1",
            },
            "seams": [],
        }

    _SEAM_STRIPS_CACHE[key] = dict(seam_strips)
    return dict(seam_strips)


def bootstrap_placeholder_artifacts(
    *,
    frequency: int,
    detail_rings: int,
    tile_sizes: Tuple[int, ...] = (128, 512),
    gutter: int = 4,
) -> Dict[str, Any]:
    """Prewarm placeholder artifacts and detail-cell caches.

    Safe to call repeatedly. Returns summary metadata useful for logs.
    """
    from .integration import PlaceholderAtlasSpec

    built_keys: List[str] = []
    for ts in tile_sizes:
        spec = PlaceholderAtlasSpec(
            frequency=frequency,
            detail_rings=detail_rings,
            tile_size=int(ts),
            gutter=gutter,
        )
        artifact = get_or_build_artifact(spec)
        built_keys.append(artifact.topology_key)

    cells, detail_cells_report = _get_or_build_detail_cells(frequency, detail_rings)
    return {
        "frequency": frequency,
        "detail_rings": detail_rings,
        "tile_sizes": list(tile_sizes),
        "artifact_keys": built_keys,
        "detail_tile_count": len(cells),
        "detail_cells_normalization": detail_cells_report,
    }


def generate_placeholder_atlas(spec: "PlaceholderAtlasSpec") -> "PlanetAtlasResult":
    """Generate a placeholder atlas from *spec* using the fast precomputed path.

    Loads or builds a cached :class:`PlaceholderAtlasArtifact` for the
    topology, then applies the colour palette in pure numpy.  Returns a
    :class:`~polygrid.integration_atlas.PlanetAtlasResult` with the same
    shape as :func:`~polygrid.integration_atlas.generate_planet_atlas`.

    On the first call for a given ``(frequency, detail_rings, tile_size,
    gutter)``, the
    artifact build runs the lightweight geometry pipeline (~0.5–2 s) and
    saves to disk.  All subsequent calls (same process or new process) skip
    that step entirely and complete in < 100 ms.

    Parameters
    ----------
    spec : PlaceholderAtlasSpec
        Colour, topology, and seed parameters.
    """
    from .integration import GenerationResult
    from .integration_atlas import PlanetAtlasResult

    t0 = time.monotonic()
    artifact = get_or_build_artifact(spec)
    atlas_png = recolor_atlas(artifact, spec)

    try:
        detail_cells, detail_cells_report = _get_or_build_detail_cells(spec.frequency, spec.detail_rings)
    except Exception:
        detail_cells = {}
        detail_cells_report = {}

    seam_strips = _get_or_build_seam_strips(spec.frequency, spec.detail_rings)
    tile_layers = {
        fid: idx for idx, fid in enumerate(artifact.face_ids)
    }
    texture_array_layout = {
        "schema": "texture-array-layout.v1",
        "backend": "atlas",
        "compatibility_mode": True,
        "layer_count": int(len(artifact.face_ids)),
        "layer_width": int(spec.tile_size),
        "layer_height": int(spec.tile_size),
        "tile_layers": dict(tile_layers),
    }

    metadata: Dict[str, Any] = {
        "mode": "placeholder",
        "topology_mode": artifact.topology_mode,
        "seam_blend": bool(np.max(artifact.seam_mask) > 0),
        "frequency": spec.frequency,
        "detail_rings": spec.detail_rings,
        "seed": spec.seed,
        "tile_count": len(artifact.face_ids),
        "index_key_count": len(artifact.index_keys),
        "atlas_width": artifact.atlas_width,
        "atlas_height": artifact.atlas_height,
        "detail_cells_normalization": detail_cells_report,
        "seam_metrics": {
            "schema": "seam-metrics.v1",
            "summary": dict(artifact.seam_metrics_summary),
        },
        "texture_backend": "atlas",
        "texture_array_layout": dict(texture_array_layout),
        "generation_time_s": round(time.monotonic() - t0, 3),
    }

    _maybe_export_debug_artifacts(spec, artifact, atlas_png, metadata)

    return PlanetAtlasResult(
        generation=GenerationResult(tiles=[], metadata=metadata),
        atlas_png=atlas_png,
        uv_layout=artifact.uv_layout,
        globe_payload={},
        vertex_data=artifact.vertex_data,
        index_data=artifact.index_data,
        frequency=artifact.frequency,
        atlas_width=artifact.atlas_width,
        atlas_height=artifact.atlas_height,
        detail_cells=detail_cells,
        seam_strips=seam_strips,
        texture_backend="atlas",
        texture_array_layout=texture_array_layout,
    )
