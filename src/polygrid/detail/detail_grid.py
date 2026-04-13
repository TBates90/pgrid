"""Multi-resolution detail grids for globe tiles.

Each tile on a :class:`GlobeGrid` can be expanded into a local
``PolyGrid`` detail grid — either pentagon-centred (5-sided tile) or
hex (6-sided tile) — giving sub-tile terrain detail that can be
rendered as a texture.

Functions
---------
- :func:`build_detail_grid`  — build a detail grid for one globe face
- :func:`generate_detail_terrain` — terrain gen seeded by parent tile
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..building.builders import build_pure_hex_grid, build_pentagon_centered_grid, hex_face_count
from ..building.goldberg_topology import goldberg_face_count
from ..core.geometry import face_center
from ..core.models import Face, Vertex
from ..core.polygrid import PolyGrid
from ..data.tile_data import FieldDef, TileDataStore, TileSchema


# ═══════════════════════════════════════════════════════════════════
# 9B.1 — Build detail grid
# ═══════════════════════════════════════════════════════════════════

def build_detail_grid(
    globe_grid: "PolyGrid",
    face_id: str,
    detail_rings: int = 2,
    *,
    size: float = 1.0,
    uv_shape_match: bool = True,
) -> PolyGrid:
    """Build a detail grid for a single globe face.

    The detail grid is a local ``PolyGrid`` — pentagon-centred for
    pentagon tiles, pure-hex for hexagonal tiles — whose metadata
    records its parent face.

    After building, the grid is **uniformly scaled** so that its
    average macro-edge length matches the globe tile's average 3-D
    edge length.  This ensures that when neighbouring detail grids
    are stitched together the sub-face densities are compatible and
    no severe size mismatch appears at pentagon/hexagon boundaries.

    For hex tiles, when ``uv_shape_match`` is True (default) and the
    ``models`` library is available, the grid's macro-boundary is
    additionally **deformed** to match the proportions of the tile's
    UV polygon.  This eliminates the anisotropic distortion that
    otherwise occurs when a regular hex grid is piecewise-warped into
    an irregular UV polygon (most pronounced for pentagon-adjacent
    hex tiles where the UV edge ratio reaches ~1.27).

    Parameters
    ----------
    globe_grid : PolyGrid (typically GlobeGrid)
        The globe grid that owns the face.
    face_id : str
        The id of the globe face to expand.
    detail_rings : int
        Number of concentric rings around the centre cell.
    size : float
        Hex cell size for the detail grid.
    uv_shape_match : bool
        If True (default), deform hex detail grids so their boundary
        shape matches the UV polygon proportions.  Pentagon grids are
        unaffected (their UV is already regular).  Set to False to
        get the old behaviour (perfectly regular hex grid).

    Returns
    -------
    PolyGrid
        A detail grid with ``metadata["parent_face_id"]``,
        ``metadata["parent_elevation"]`` (if a store is attached later),
        and ``metadata["detail_rings"]``.
    """
    face = globe_grid.faces.get(face_id)
    if face is None:
        raise KeyError(f"Face '{face_id}' not found in globe grid")

    if face.face_type == "pent":
        grid = build_pentagon_centered_grid(detail_rings, size=size)
    else:
        grid = build_pure_hex_grid(detail_rings, size=size)

    # ── Scale to match globe edge length ────────────────────────
    # Compute the globe tile's average 3-D edge length.
    vids = face.vertex_ids
    n = len(vids)
    globe_edge_sum = 0.0
    for i in range(n):
        v0 = globe_grid.vertices[vids[i]]
        v1 = globe_grid.vertices[vids[(i + 1) % n]]
        dx = v1.x - v0.x
        dy = v1.y - v0.y
        dz = ((v1.z or 0.0) - (v0.z or 0.0))
        globe_edge_sum += math.sqrt(dx * dx + dy * dy + dz * dz)
    globe_avg_edge = globe_edge_sum / n if n else 1.0

    # Compute the detail grid's average macro-edge endpoint distance.
    grid.compute_macro_edges(
        n_sides=n,
        corner_ids=grid.metadata.get("corner_vertex_ids"),
    )
    if grid.macro_edges:
        detail_edge_sum = 0.0
        for me in grid.macro_edges:
            mv0 = grid.vertices[me.vertex_ids[0]]
            mv1 = grid.vertices[me.vertex_ids[-1]]
            detail_edge_sum += math.hypot(mv1.x - mv0.x, mv1.y - mv0.y)
        detail_avg_edge = detail_edge_sum / len(grid.macro_edges)
    else:
        detail_avg_edge = 1.0

    if detail_avg_edge > 1e-12:
        scale = globe_avg_edge / detail_avg_edge
        if abs(scale - 1.0) > 1e-6:
            from ..building.assembly import scale_grid
            grid = scale_grid(grid, scale, 0.0, 0.0)
            # Recompute macro edges at the new scale.
            grid.compute_macro_edges(
                n_sides=n,
                corner_ids=grid.metadata.get("corner_vertex_ids"),
            )

    # ── UV shape deformation (hex tiles only) ───────────────────
    # Deform the grid's macro-boundary so its proportions match the
    # UV polygon.  This makes the downstream piecewise warp nearly
    # isotropic, eliminating the visible anisotropic distortion that
    # appears when a regular hex grid is warped into an irregular UV
    # polygon (worst for pentagon-adjacent tiles: edge ratio ~1.27).
    if uv_shape_match and face.face_type == "hex":
        try:
            from ..uv_texture import get_tile_uv_vertices
            from ..tile_uv_align import get_macro_edge_corners, match_grid_corners_to_uv

            uv_corners = get_tile_uv_vertices(globe_grid, face_id)
            grid_corners_raw = get_macro_edge_corners(grid, n)
            grid_corners_matched = match_grid_corners_to_uv(
                grid_corners_raw, globe_grid, face_id,
            )
            grid = deform_grid_to_uv_shape(grid, grid_corners_matched, uv_corners)
            # Recompute macro edges after deformation.
            grid.compute_macro_edges(
                n_sides=n,
                corner_ids=grid.metadata.get("corner_vertex_ids"),
            )
            grid.metadata["uv_shape_matched"] = True
        except (ImportError, Exception):
            # models library not available or UV data unavailable —
            # fall back to regular grid (no deformation).
            grid.metadata["uv_shape_matched"] = False

    # ── Anchor metadata ─────────────────────────────────────────
    grid.metadata["parent_face_id"] = face_id
    grid.metadata["parent_face_type"] = face.face_type
    grid.metadata["detail_rings"] = detail_rings

    # Ensure corner_vertex_ids is always present.  Pentagon grids
    # already have it from build_goldberg_grid; for hex grids (and
    # as a safety net for pentagons) we extract the ordered corners
    # from the macro edges that were computed above.
    if "corner_vertex_ids" not in grid.metadata and grid.macro_edges:
        grid.metadata["corner_vertex_ids"] = [
            me.corner_start for me in grid.macro_edges
        ]

    # Copy parent face metadata if available
    if face.metadata:
        for key in ("center_3d", "normal_3d", "latitude_deg",
                     "longitude_deg", "tile_id"):
            if key in face.metadata:
                grid.metadata[f"parent_{key}"] = face.metadata[key]

    return grid


def detail_face_count(face_type: str, rings: int) -> int:
    """Expected face count for a detail grid.

    Parameters
    ----------
    face_type : ``"pent"`` or ``"hex"``
    rings : int

    Returns
    -------
    int
    """
    if face_type == "pent":
        return goldberg_face_count(rings)
    return hex_face_count(rings)


# ═══════════════════════════════════════════════════════════════════
# 9B.2 — Shape-matched grid deformation
# ═══════════════════════════════════════════════════════════════════

def deform_grid_to_uv_shape(
    grid: PolyGrid,
    grid_corners: List[Tuple[float, float]],
    uv_corners: List[Tuple[float, float]],
) -> PolyGrid:
    """Deform a regular hex detail grid so its boundary matches the UV polygon shape.

    A regular hex grid has all edges equal and all angles 120°.  When
    the UV polygon is irregular (e.g. pentagon-adjacent hex tiles with
    edge ratio ≈ 1.27), the piecewise-affine warp introduces anisotropic
    distortion — sub-faces near short UV edges are compressed while
    those near long UV edges are stretched.

    This function applies a smooth **barycentric interpolation** of
    corner displacements through the grid interior, so that:

    - The macro-boundary corners move to match the UV polygon's
      proportions (normalised to the same circumradius).
    - The centroid stays fixed (zero displacement).
    - Interior vertices transition smoothly via triangle-fan
      barycentric weights.

    After deformation the piecewise warp becomes close to a uniform
    scale, eliminating anisotropy.

    Parameters
    ----------
    grid : PolyGrid
        A hex detail grid with macro-edges already computed.
    grid_corners : list of (x, y)
        The grid's macro-edge corners in matched order (from
        ``match_grid_corners_to_uv``).
    uv_corners : list of (x, y)
        The UV polygon corners (from ``get_tile_uv_vertices``).

    Returns
    -------
    PolyGrid
        The same grid with vertex positions updated in-place.
    """
    gc = np.array(grid_corners, dtype=np.float64)
    uv = np.array(uv_corners, dtype=np.float64)
    n = len(gc)

    # ── Normalise UV corners to the same circumradius as grid corners ──
    gc_centroid = gc.mean(axis=0)
    uv_centroid = uv.mean(axis=0)

    gc_rel = gc - gc_centroid
    uv_rel = uv - uv_centroid

    gc_circumradius = np.max(np.linalg.norm(gc_rel, axis=1))
    uv_circumradius = np.max(np.linalg.norm(uv_rel, axis=1))

    if uv_circumradius < 1e-12 or gc_circumradius < 1e-12:
        return grid  # degenerate — skip

    # Scale UV shape to match grid's circumradius
    uv_normalised = uv_rel * (gc_circumradius / uv_circumradius) + gc_centroid

    # ── Compute per-corner displacement vectors ────────────────────
    displacements = uv_normalised - gc  # shape (n, 2)

    # ── Precompute sector geometry for barycentric lookup ──────────
    # Each sector is the triangle: (centroid, corner[i], corner[i+1])
    # We precompute the inverse of the 2×2 basis matrix for each sector.
    sector_inv: List[Optional[np.ndarray]] = []
    corner_angles = np.arctan2(gc_rel[:, 1], gc_rel[:, 0])

    for i in range(n):
        j = (i + 1) % n
        # Basis vectors from centroid to each corner
        b1 = gc[i] - gc_centroid
        b2 = gc[j] - gc_centroid
        # 2×2 matrix [b1 | b2]
        B = np.column_stack([b1, b2])
        det = B[0, 0] * B[1, 1] - B[0, 1] * B[1, 0]
        if abs(det) > 1e-20:
            inv = np.array([
                [B[1, 1], -B[0, 1]],
                [-B[1, 0], B[0, 0]],
            ], dtype=np.float64) / det
            sector_inv.append(inv)
        else:
            sector_inv.append(None)

    # ── Identify boundary vertices to pin ─────────────────────────
    # Vertices on macro-edges must keep their original positions so
    # that stitched composites have consistent boundaries regardless
    # of whether adjacent tiles were deformed differently.
    boundary_vids: set = set()
    for me in grid.macro_edges:
        boundary_vids.update(me.vertex_ids)

    # ── Assign each vertex to a sector and compute displacement ────
    for vid, vtx in grid.vertices.items():
        if vid in boundary_vids:
            continue  # pin boundary vertices at original positions

        rel = np.array([vtx.x - gc_centroid[0], vtx.y - gc_centroid[1]],
                       dtype=np.float64)
        dist = np.linalg.norm(rel)
        if dist < 1e-12:
            continue  # centroid vertex — no displacement

        # Find sector by angle
        angle = math.atan2(rel[1], rel[0])
        sector_idx = -1
        for i in range(n):
            j = (i + 1) % n
            a0 = corner_angles[i]
            a1 = corner_angles[j]
            # Normalise angles to [0, 2π)
            a0n = a0 % (2.0 * math.pi)
            a1n = a1 % (2.0 * math.pi)
            pn = angle % (2.0 * math.pi)
            if a0n <= a1n:
                if a0n <= pn <= a1n:
                    sector_idx = i
                    break
            else:
                # Arc wraps around 2π
                if pn >= a0n or pn <= a1n:
                    sector_idx = i
                    break

        if sector_idx < 0:
            # Fallback: find nearest sector by angular distance
            best_err = float("inf")
            for i in range(n):
                j = (i + 1) % n
                mid_angle = math.atan2(
                    (gc_rel[i, 1] + gc_rel[j, 1]) / 2,
                    (gc_rel[i, 0] + gc_rel[j, 0]) / 2,
                )
                err = abs((angle - mid_angle + math.pi) % (2 * math.pi) - math.pi)
                if err < best_err:
                    best_err = err
                    sector_idx = i

        si = sector_idx
        ji = (si + 1) % n

        # Compute barycentric coords within this sector triangle
        inv = sector_inv[si]
        if inv is not None:
            bary = inv @ rel  # (w_i, w_j) — weights for corner_i and corner_j
            w_i = float(bary[0])
            w_j = float(bary[1])
            # Clamp to avoid extrapolation for vertices slightly outside
            w_i = max(0.0, w_i)
            w_j = max(0.0, w_j)
            w_sum = w_i + w_j
            if w_sum > 1.0:
                w_i /= w_sum
                w_j /= w_sum
        else:
            # Degenerate sector — use equal weights
            w_i = 0.5
            w_j = 0.5

        # Interpolated displacement: centroid gets 0, corners get full displacement
        dx = w_i * displacements[si, 0] + w_j * displacements[ji, 0]
        dy = w_i * displacements[si, 1] + w_j * displacements[ji, 1]

        # Create a new Vertex with displaced position (Vertex is frozen).
        grid.vertices[vid] = Vertex(
            id=vtx.id,
            x=vtx.x + dx,
            y=vtx.y + dy,
            z=vtx.z,
        )

    return grid


# ═══════════════════════════════════════════════════════════════════
# 9B.3 — Detail terrain generation
# ═══════════════════════════════════════════════════════════════════

def generate_detail_terrain(
    detail_grid: PolyGrid,
    parent_elevation: float,
    *,
    seed: int = 0,
    frequency: float = 4.0,
    octaves: int = 4,
    amplitude: float = 0.15,
    base_weight: float = 0.85,
) -> TileDataStore:
    """Generate terrain on a detail grid, seeded by parent tile elevation.

    The elevation is computed as::

        elevation = parent_elevation * base_weight
                  + noise(x, y) * amplitude * (1 - base_weight)

    So the detail grid inherits most of its elevation from the parent
    tile, with high-frequency local variation layered on top.

    Parameters
    ----------
    detail_grid : PolyGrid
        A detail grid (from :func:`build_detail_grid`).
    parent_elevation : float
        Elevation of the parent globe tile.
    seed : int
        Noise seed.
    frequency : float
        Noise spatial frequency (higher = more detail).
    octaves : int
        Noise octave count.
    amplitude : float
        Amplitude of local variation.
    base_weight : float
        How much the parent elevation dominates (0–1).

    Returns
    -------
    TileDataStore
        Store with ``"elevation"`` field populated.
    """
    from ..terrain.noise import fbm, _init_noise

    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=detail_grid, schema=schema)

    noise_fn = _init_noise(seed)

    for fid in detail_grid.faces:
        face = detail_grid.faces[fid]
        c = face_center(detail_grid.vertices, face)
        if c is None:
            continue
        cx, cy = c
        # High-frequency local noise
        local_noise = fbm(
            cx, cy,
            octaves=octaves,
            frequency=frequency,
            seed=seed,
        )
        # Blend: parent base + local detail
        elevation = (
            parent_elevation * base_weight
            + local_noise * amplitude * (1.0 - base_weight)
        )
        store.set(fid, "elevation", elevation)

    return store
