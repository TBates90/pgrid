# TODO REMOVE — Not used by any live script. Phase 11A 3D-coherent terrain generation.
"""Globe-coherent 3-D terrain generation for sub-tile detail grids.

Phase 11A — replaces the per-tile noise sampling of Phase 10 with a
**globally continuous** noise field that is evaluated at each sub-face's
true 3-D position on the unit sphere.  This eliminates the "patchwork
quilt" artefact caused by each tile generating noise independently in
its own local coordinate space.

Key idea
--------
Every sub-face's 2-D local position is projected through the parent
tile's tangent-plane transform onto the globe sphere.  A single call
to ``fbm_3d(x, y, z)`` (or ``ridged_noise_3d``) at that position
produces noise that is spatially continuous across the entire globe —
no per-tile seeds, no boundary averaging.

Functions
---------
- :func:`compute_subface_3d_position` — single sub-face → 3-D point
- :func:`precompute_3d_positions` — batch for one detail grid
- :func:`precompute_all_3d_positions` — batch for entire collection
- :func:`generate_detail_terrain_3d` — terrain gen for one tile
- :func:`generate_all_detail_terrain_3d` — batch for entire collection
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .geometry import face_center
from .heightmap import smooth_field
from .noise import fbm_3d, ridged_noise_3d
from .polygrid import PolyGrid
from .tile_data import FieldDef, TileDataStore, TileSchema
from .tile_detail import TileDetailSpec, DetailGridCollection


# ═══════════════════════════════════════════════════════════════════
# Terrain3DSpec — configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Terrain3DSpec:
    """Configuration for 3-D globe-coherent terrain generation.

    Extends the concepts of :class:`TileDetailSpec` with parameters
    specific to 3-D noise sampling and ridge/fbm blending.

    Parameters
    ----------
    noise_frequency : float
        Spatial frequency of the globe-coherent fbm_3d field.
        Higher values → finer features.  Because this operates in
        3-D on a unit sphere, useful values are typically 2–12.
    noise_octaves : int
        Number of noise octaves for the fbm layer.
    ridge_frequency : float
        Spatial frequency for the ridged-noise mountain layer.
    ridge_octaves : int
        Octave count for ridged noise.
    fbm_weight : float
        Weight of the fbm (rolling terrain) layer (0–1).
    ridge_weight : float
        Weight of the ridged-noise (mountain ridge) layer (0–1).
        ``fbm_weight + ridge_weight`` should typically sum to ≤ 1.
    base_weight : float
        Weight of the parent tile elevation (0–1).  The noise
        contribution is scaled by ``(1 − base_weight)``.
    amplitude : float
        Amplitude of the combined noise relative to parent elevation.
    boundary_smoothing : int
        Number of smoothing passes on boundary faces (reduces any
        residual seam from base-weight blending).
    seed : int
        Global noise seed — the **same** seed is used for every tile,
        producing spatial continuity via the 3-D position.
    """

    noise_frequency: float = 6.0
    noise_octaves: int = 5
    ridge_frequency: float = 4.0
    ridge_octaves: int = 5
    fbm_weight: float = 0.6
    ridge_weight: float = 0.4
    base_weight: float = 0.55
    amplitude: float = 0.25
    boundary_smoothing: int = 2
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════
# 11A.1 — Compute sub-face 3-D position on the globe sphere
# ═══════════════════════════════════════════════════════════════════

def _normalize(v: Tuple[float, float, float], radius: float = 1.0) -> Tuple[float, float, float]:
    """Normalize a 3-D vector to lie on a sphere of given *radius*."""
    x, y, z = v
    length = math.sqrt(x * x + y * y + z * z)
    if length < 1e-15:
        return (0.0, 0.0, radius)
    scale = radius / length
    return (x * scale, y * scale, z * scale)


def _tangent_basis(
    normal: Tuple[float, float, float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Compute an orthonormal tangent basis for a plane with given *normal*.

    Returns ``(tangent_u, tangent_v)`` such that ``normal``, ``tangent_u``,
    ``tangent_v`` form a right-handed orthonormal frame.

    The basis is constructed by choosing a reference direction that is
    not parallel to *normal*, taking the cross product to get one
    tangent, then crossing again to get the second.
    """
    nx, ny, nz = normal
    # Choose reference axis least aligned with the normal
    if abs(nx) < 0.9:
        ref = (1.0, 0.0, 0.0)
    else:
        ref = (0.0, 1.0, 0.0)

    # tangent_u = ref × normal (normalised)
    ux = ref[1] * nz - ref[2] * ny
    uy = ref[2] * nx - ref[0] * nz
    uz = ref[0] * ny - ref[1] * nx
    u_len = math.sqrt(ux * ux + uy * uy + uz * uz)
    if u_len < 1e-15:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    ux /= u_len
    uy /= u_len
    uz /= u_len

    # tangent_v = normal × tangent_u
    vx = ny * uz - nz * uy
    vy = nz * ux - nx * uz
    vz = nx * uy - ny * ux

    return ((ux, uy, uz), (vx, vy, vz))


def compute_subface_3d_position(
    globe_grid: PolyGrid,
    face_id: str,
    detail_grid: PolyGrid,
    sub_face_id: str,
    *,
    radius: float = 1.0,
) -> Optional[Tuple[float, float, float]]:
    """Compute the approximate 3-D position of a sub-face on the globe sphere.

    Projects the sub-face's local 2-D centroid through the parent tile's
    tangent plane onto the sphere surface.

    Parameters
    ----------
    globe_grid : PolyGrid
        The globe grid (with ``center_3d`` and ``normal_3d`` metadata on faces).
    face_id : str
        The parent globe tile id.
    detail_grid : PolyGrid
        The detail grid for *face_id*.
    sub_face_id : str
        The sub-face within *detail_grid*.
    radius : float
        Sphere radius (default 1.0 for unit sphere).

    Returns
    -------
    tuple or None
        ``(x, y, z)`` on the sphere, or ``None`` if position data is missing.
    """
    globe_face = globe_grid.faces.get(face_id)
    if globe_face is None:
        return None

    center_3d = globe_face.metadata.get("center_3d")
    normal_3d = globe_face.metadata.get("normal_3d")
    if center_3d is None or normal_3d is None:
        return None

    # Get sub-face centroid in local 2-D coordinates
    sub_face = detail_grid.faces.get(sub_face_id)
    if sub_face is None:
        return None
    c = face_center(detail_grid.vertices, sub_face)
    if c is None:
        return None
    lx, ly = c

    # Get the detail grid's local extent for scaling
    # The detail grid is centred at (0, 0) with some extent.
    # We need to know how the local coordinate maps onto the globe tile.
    xs = [v.x for v in detail_grid.vertices.values() if v.x is not None]
    ys = [v.y for v in detail_grid.vertices.values() if v.y is not None]
    if not xs or not ys:
        return None
    local_extent = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    # Compute the angular size of the parent tile — approximate from
    # the globe tile's vertex positions.
    globe_verts = [globe_grid.vertices[vid] for vid in globe_face.vertex_ids
                   if vid in globe_grid.vertices]
    if not globe_verts:
        return None
    # Angular radius ≈ mean angle from tile centre to tile vertices
    cx3, cy3, cz3 = center_3d
    c_len = math.sqrt(cx3 * cx3 + cy3 * cy3 + cz3 * cz3)
    angles = []
    for v in globe_verts:
        if v.x is None or v.y is None or v.z is None:
            continue
        # Dot product between centre and vertex, normalised by magnitudes
        v_len = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        denom = c_len * v_len
        if denom < 1e-15:
            continue
        dot = (cx3 * v.x + cy3 * v.y + cz3 * v.z) / denom
        dot = max(-1.0, min(1.0, dot))  # clamp for numerical safety
        angles.append(math.acos(dot))
    if not angles:
        return None
    tile_angular_radius = sum(angles) / len(angles)

    # Scale: local coords → angular displacement on the tangent plane
    scale = tile_angular_radius / (local_extent / 2.0)

    # Build tangent frame at the tile centre
    tangent_u, tangent_v = _tangent_basis(normal_3d)

    # Offset in 3-D: move along tangent plane by scaled local coords
    dx = lx * scale
    dy = ly * scale
    px = cx3 + dx * tangent_u[0] + dy * tangent_v[0]
    py = cy3 + dx * tangent_u[1] + dy * tangent_v[1]
    pz = cz3 + dx * tangent_u[2] + dy * tangent_v[2]

    # Project back onto the sphere
    return _normalize((px, py, pz), radius)


# ═══════════════════════════════════════════════════════════════════
# 11A.2 — Batch: precompute all 3-D positions for one detail grid
# ═══════════════════════════════════════════════════════════════════

def precompute_3d_positions(
    globe_grid: PolyGrid,
    face_id: str,
    detail_grid: PolyGrid,
    *,
    radius: float = 1.0,
) -> Dict[str, Tuple[float, float, float]]:
    """Compute 3-D sphere positions for all sub-faces in a detail grid.

    This is the batched, cache-friendly version of
    :func:`compute_subface_3d_position`.  It pre-computes the tangent
    frame and scaling factor once and reuses them for every sub-face,
    which is significantly faster than calling the single-face version
    in a loop.

    Parameters
    ----------
    globe_grid : PolyGrid
        The globe grid.
    face_id : str
        Parent tile id.
    detail_grid : PolyGrid
        The detail grid for *face_id*.
    radius : float
        Sphere radius.

    Returns
    -------
    dict
        ``{sub_face_id: (x, y, z)}`` for every sub-face with valid positions.
    """
    globe_face = globe_grid.faces.get(face_id)
    if globe_face is None:
        return {}

    center_3d = globe_face.metadata.get("center_3d")
    normal_3d = globe_face.metadata.get("normal_3d")
    if center_3d is None or normal_3d is None:
        return {}

    # Local extent
    xs = [v.x for v in detail_grid.vertices.values() if v.x is not None]
    ys = [v.y for v in detail_grid.vertices.values() if v.y is not None]
    if not xs or not ys:
        return {}
    local_extent = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    # Angular size of the parent tile
    cx3, cy3, cz3 = center_3d
    c_len = math.sqrt(cx3 * cx3 + cy3 * cy3 + cz3 * cz3)
    globe_verts = [globe_grid.vertices[vid] for vid in globe_face.vertex_ids
                   if vid in globe_grid.vertices]
    angles = []
    for v in globe_verts:
        if v.x is None or v.y is None or v.z is None:
            continue
        v_len = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        denom = c_len * v_len
        if denom < 1e-15:
            continue
        dot = (cx3 * v.x + cy3 * v.y + cz3 * v.z) / denom
        dot = max(-1.0, min(1.0, dot))
        angles.append(math.acos(dot))
    if not angles:
        return {}
    tile_angular_radius = sum(angles) / len(angles)
    scale = tile_angular_radius / (local_extent / 2.0)

    # Tangent frame (computed once)
    tangent_u, tangent_v = _tangent_basis(normal_3d)

    # Map every sub-face
    positions: Dict[str, Tuple[float, float, float]] = {}
    for sub_fid, sub_face in detail_grid.faces.items():
        c = face_center(detail_grid.vertices, sub_face)
        if c is None:
            continue
        lx, ly = c
        dx = lx * scale
        dy = ly * scale
        px = cx3 + dx * tangent_u[0] + dy * tangent_v[0]
        py = cy3 + dx * tangent_u[1] + dy * tangent_v[1]
        pz = cz3 + dx * tangent_u[2] + dy * tangent_v[2]
        positions[sub_fid] = _normalize((px, py, pz), radius)

    return positions


def precompute_all_3d_positions(
    collection: DetailGridCollection,
    globe_grid: PolyGrid,
    *,
    radius: float = 1.0,
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """Precompute 3-D positions for every sub-face in every tile.

    Parameters
    ----------
    collection : DetailGridCollection
    globe_grid : PolyGrid
    radius : float

    Returns
    -------
    dict
        ``{face_id: {sub_face_id: (x, y, z)}}``
    """
    all_positions: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    for face_id, detail_grid in collection.grids.items():
        positions = precompute_3d_positions(
            globe_grid, face_id, detail_grid, radius=radius,
        )
        all_positions[face_id] = positions
    return all_positions


# ═══════════════════════════════════════════════════════════════════
# 11A.3 — 3-D coherent terrain generation for a single tile
# ═══════════════════════════════════════════════════════════════════

def generate_detail_terrain_3d(
    detail_grid: PolyGrid,
    positions_3d: Dict[str, Tuple[float, float, float]],
    parent_elevation: float,
    spec: Terrain3DSpec,
) -> TileDataStore:
    """Generate terrain for a single detail grid using 3-D globe-coherent noise.

    Unlike :func:`~detail_terrain.generate_detail_terrain_bounded`, this
    function does **not** use per-tile noise seeds.  Instead, every
    sub-face is sampled at its actual 3-D position on the globe sphere,
    producing a noise field that is spatially continuous across all tiles.

    Parameters
    ----------
    detail_grid : PolyGrid
        The detail grid for one globe tile.
    positions_3d : dict
        ``{sub_face_id: (x, y, z)}`` — precomputed 3-D positions
        from :func:`precompute_3d_positions`.
    parent_elevation : float
        Elevation of the parent globe tile (used as base).
    spec : Terrain3DSpec
        Noise and blending configuration.

    Returns
    -------
    TileDataStore
        Store with an ``"elevation"`` field for every sub-face.
    """
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=detail_grid, schema=schema)

    seed = spec.seed
    total_noise_weight = spec.fbm_weight + spec.ridge_weight
    if total_noise_weight < 1e-10:
        total_noise_weight = 1.0

    for sub_fid in detail_grid.faces:
        pos = positions_3d.get(sub_fid)
        if pos is None:
            store.set(sub_fid, "elevation", parent_elevation)
            continue

        x, y, z = pos

        # Layer 1: rolling terrain via fbm_3d
        fbm_val = 0.0
        if spec.fbm_weight > 0:
            fbm_val = fbm_3d(
                x, y, z,
                octaves=spec.noise_octaves,
                frequency=spec.noise_frequency,
                seed=seed,
            )

        # Layer 2: mountain ridges via ridged_noise_3d
        ridge_val = 0.0
        if spec.ridge_weight > 0:
            ridge_val = ridged_noise_3d(
                x, y, z,
                octaves=spec.ridge_octaves,
                frequency=spec.ridge_frequency,
                seed=seed + 7919,  # offset so ridges ≠ fbm
            )

        # Combine layers (normalised by their total weight)
        combined = (
            spec.fbm_weight * fbm_val
            + spec.ridge_weight * ridge_val
        ) / total_noise_weight

        # Blend with parent elevation
        elevation = (
            parent_elevation * spec.base_weight
            + combined * spec.amplitude * (1.0 - spec.base_weight)
        )
        store.set(sub_fid, "elevation", elevation)

    # Smooth to reduce any micro-scale noise jaggedness
    if spec.boundary_smoothing > 0:
        smooth_field(
            detail_grid, store, "elevation",
            iterations=spec.boundary_smoothing,
            self_weight=0.6,
        )

    return store


# ═══════════════════════════════════════════════════════════════════
# 11A.4 — Batch: generate 3-D coherent terrain for entire collection
# ═══════════════════════════════════════════════════════════════════

def generate_all_detail_terrain_3d(
    collection: DetailGridCollection,
    globe_grid: PolyGrid,
    globe_store: TileDataStore,
    spec: Optional[Terrain3DSpec] = None,
    *,
    elevation_field: str = "elevation",
    radius: float = 1.0,
) -> None:
    """Generate 3-D coherent terrain for every tile in a collection.

    This is the **drop-in replacement** for
    :func:`~detail_terrain.generate_all_detail_terrain`.  It precomputes
    3-D positions for every sub-face and then generates terrain using a
    single, globally-coherent noise field.

    Parameters
    ----------
    collection : DetailGridCollection
        Must already contain grids.
    globe_grid : PolyGrid
        The globe grid (with 3-D metadata on faces).
    globe_store : TileDataStore
        Globe-level elevation data.
    spec : Terrain3DSpec, optional
        Noise configuration.  Defaults to ``Terrain3DSpec()``.
    elevation_field : str
        Name of the elevation field in *globe_store*.
    radius : float
        Sphere radius for 3-D projection.
    """
    if spec is None:
        spec = Terrain3DSpec()

    # Pre-compute all 3-D positions in one pass
    all_positions = precompute_all_3d_positions(
        collection, globe_grid, radius=radius,
    )

    # Generate terrain for each tile
    for face_id, detail_grid in collection.grids.items():
        parent_elev = globe_store.get(face_id, elevation_field)
        positions = all_positions.get(face_id, {})

        store = generate_detail_terrain_3d(
            detail_grid, positions, parent_elev, spec,
        )
        # Write into collection's internal stores
        collection._stores[face_id] = store
