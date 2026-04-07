"""High-level API for generating a fully-rendered planet atlas.

This extends :mod:`polygrid.integration` by running the detail terrain,
tile rendering, and atlas-packing pipeline — producing a texture atlas
and UV layout that external consumers (e.g. playground) can use for
3D globe rendering.

Functions
---------
- :func:`generate_planet_atlas` — full pipeline from params → atlas.
"""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .integration import (
    GenerationResult,
    PlaceholderAtlasSpec,
    PlanetParams,
    generate_planet,
)

LOGGER = logging.getLogger(__name__)


def _detail_cells_strict_mode() -> bool:
    return os.environ.get("PGRID_DETAIL_CELLS_STRICT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

# ═══════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PlanetAtlasResult:
    """Full result from atlas generation.

    Attributes
    ----------
    generation : GenerationResult
        Per-tile terrain data (elevation, moisture, biome, etc.).
    atlas_png : bytes
        Atlas image as PNG bytes (loadable via ``PIL.Image.open``
        or directly uploadable as an OpenGL texture).
    uv_layout : dict[str, tuple[float, float, float, float]]
        ``{face_id: (u_min, v_min, u_max, v_max)}`` mapping each tile
        to its atlas slot in UV space.
    globe_payload : dict
        Globe export payload (tile geometry, colours, adjacency)
        matching the format from :func:`polygrid.globe_export.export_globe_payload`.
    vertex_data : np.ndarray
        Batched globe mesh vertices — shape ``(N, 8)`` with stride
        ``(x, y, z, r, g, b, u, v)``.  Ready for VBO upload.
    index_data : np.ndarray
        Triangle index array — shape ``(M, 3)`` uint32.
    frequency : int
        Goldberg polyhedron frequency.
    atlas_width : int
        Atlas image width in pixels.
    atlas_height : int
        Atlas image height in pixels.
    """

    generation: GenerationResult
    atlas_png: bytes
    uv_layout: Dict[str, Tuple[float, float, float, float]]
    globe_payload: Dict[str, Any]
    vertex_data: "np.ndarray"
    index_data: "np.ndarray"
    frequency: int
    atlas_width: int
    atlas_height: int
    detail_cells: Dict[str, Any]
    seam_strips: Dict[str, Any]



# ═══════════════════════════════════════════════════════════════════
# Pipeline helpers
# ═══════════════════════════════════════════════════════════════════


def _build_terrain(frequency: int, params: PlanetParams):
    """Build globe grid + elevation store (mirrors render_polygrids)."""
    from dataclasses import replace as _replace

    from .globe.globe import build_globe_grid
    from .mountains import (
        MOUNTAIN_RANGE,
        MountainConfig,
        generate_mountains,
    )
    from .data.tile_data import FieldDef, TileDataStore, TileSchema

    grid = build_globe_grid(frequency)
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)

    # Use roughness to adjust peak elevation.
    config = _replace(
        MOUNTAIN_RANGE,
        seed=params.seed,
        peak_elevation=0.5 + params.roughness * 0.5,
    )
    generate_mountains(grid, store, config)
    return grid, store


def _build_detail_pipeline(grid, store, frequency: int, seed: int,
                           detail_rings: int):
    """Build detail grids and generate boundary-aware terrain."""
    from .detail.tile_detail import TileDetailSpec, DetailGridCollection
    from .detail.detail_terrain import generate_all_detail_terrain

    spec = TileDetailSpec(detail_rings=detail_rings)
    coll = DetailGridCollection.build(grid, spec)
    generate_all_detail_terrain(coll, grid, store, spec, seed=seed)
    return coll, spec


def _build_stitched_store(composite, coll, face_id, globe_grid):
    """Build a TileDataStore for a stitched CompositeGrid."""
    from .data.tile_data import FieldDef, TileDataStore, TileSchema

    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=composite.merged, schema=schema)

    for comp_name, prefix in composite.id_prefixes.items():
        _, comp_store = coll.get(comp_name)
        if comp_store is None:
            continue
        for fid in composite.components[comp_name].faces:
            prefixed_fid = f"{prefix}{fid}"
            if prefixed_fid in composite.merged.faces:
                store.set(prefixed_fid, "elevation",
                          comp_store.get(fid, "elevation"))

    return store


def _render_tile_analytical(
    grid, colour_fn, tile_size: int,
    xlim: Tuple[float, float], ylim: Tuple[float, float],
    bg_colour: Tuple[float, float, float] = (1.0, 0.0, 1.0),
) -> "Image.Image":
    """Render grid faces to an in-memory PIL Image via point-in-polygon fill.

    Bypasses matplotlib entirely for speed and determinism.
    """
    import numpy as np
    from matplotlib.path import Path as MplPath
    from PIL import Image

    bg_rgb = (int(bg_colour[0] * 255), int(bg_colour[1] * 255),
              int(bg_colour[2] * 255))
    arr = np.full((tile_size, tile_size, 3),
                  [bg_rgb[0], bg_rgb[1], bg_rgb[2]], dtype=np.uint8)

    xs = np.linspace(xlim[0], xlim[1], tile_size)
    ys = np.linspace(ylim[1], ylim[0], tile_size)  # flip y
    px, py = np.meshgrid(xs, ys)

    for fid, face in grid.faces.items():
        colour = colour_fn(fid, grid, face)
        if colour is None:
            continue

        verts = []
        for vid in face.vertex_ids:
            v = grid.vertices.get(vid)
            if v is None or not v.has_position():
                break
            verts.append((v.x, v.y))
        else:
            if len(verts) < 3:
                continue
        if len(verts) < 3:
            continue

        verts_closed = verts + [verts[0]]
        path = MplPath(verts_closed)

        vx = [p[0] for p in verts]
        vy = [p[1] for p in verts]
        col_mask = (xs >= min(vx)) & (xs <= max(vx))
        row_mask = (ys >= min(vy)) & (ys <= max(vy))
        if not col_mask.any() or not row_mask.any():
            continue

        ci = np.where(col_mask)[0]
        ri = np.where(row_mask)[0]
        sub_px = px[np.ix_(ri, ci)].ravel()
        sub_py = py[np.ix_(ri, ci)].ravel()
        sub_pts = np.column_stack([sub_px, sub_py])
        mask = path.contains_points(sub_pts)

        rows_idx, cols_idx = np.meshgrid(ri, ci, indexing="ij")
        rows_flat = rows_idx.ravel()[mask]
        cols_flat = cols_idx.ravel()[mask]

        rgb = (int(colour[0] * 255), int(colour[1] * 255),
               int(colour[2] * 255))
        arr[rows_flat, cols_flat] = [rgb[0], rgb[1], rgb[2]]

    return Image.fromarray(arr)


def _terrain_colour_fn(grid, store, biome, noise_seed, hillshade=None):
    """Return a colour callback for terrain rendering."""
    from .detail.detail_render import detail_elevation_to_colour
    from .core.geometry import face_center as _face_center

    def _colour(fid, _grid, face):
        elev = store.get(fid, "elevation")
        c = _face_center(grid.vertices, face)
        cx, cy = c if c else (0.0, 0.0)
        hs_val = hillshade.get(fid, 0.5) if hillshade else 0.5
        return detail_elevation_to_colour(
            elev, biome,
            hillshade_val=hs_val,
            noise_x=cx, noise_y=cy,
            noise_seed=noise_seed,
        )

    return _colour


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════


def generate_planet_atlas(
    params: PlanetParams,
    *,
    detail_rings: int = 2,
    tile_size: int = 128,
    gutter: int = 4,
    ramp_fn: Optional[Callable[[float], Tuple[float, float, float]]] = None,
) -> PlanetAtlasResult:
    """Generate a complete planet with a texture atlas for 3D rendering.

    This is the single entry point for external consumers that want
    both terrain data AND a rendered atlas.  It runs:

    1. :func:`generate_planet` — macro-level terrain classification.
    2. Globe grid + elevation terrain (mountain preset).
    3. Detail grids at ``detail_rings`` resolution.
    4. Boundary-aware detail terrain interpolation.
    5. Per-tile analytical rendering (no matplotlib dependency for output).
    6. Polygon-cut atlas packing with UV layout.
    7. Globe payload export (geometry + colours).

    Parameters
    ----------
    params : PlanetParams
        Planet generation parameters.
    detail_rings : int
        Detail sub-grid ring count (2 = fast, 4 = high quality).
    tile_size : int
        Per-tile texture resolution in pixels (128 = fast, 256 = quality).
    gutter : int
        Atlas gutter pixels for bilinear bleed prevention.

    Returns
    -------
    PlanetAtlasResult
        Contains atlas PNG bytes, UV layout, globe payload, and the
        standard GenerationResult.
    """
    t_start = time.monotonic()

    # ── 1. Run the standard terrain generation pipeline ─────────────
    gen_result = generate_planet(params)
    frequency = params.frequency

    LOGGER.info(
        "Atlas gen: freq=%d tiles=%d detail_rings=%d tile_size=%d",
        frequency, len(gen_result.tiles), detail_rings, tile_size,
    )

    # ── 2. Build globe grid + elevation ─────────────────────────────
    grid, store = _build_terrain(frequency, params)

    # ── 3. Build detail grids + terrain ─────────────────────────────
    coll, spec = _build_detail_pipeline(
        grid, store, frequency, params.seed, detail_rings,
    )
    face_ids = coll.face_ids

    # ── 4. Build composites ─────────────────────────────────────────
    from .detail.tile_detail import build_tile_with_neighbours
    from .detail.detail_render import BiomeConfig, _detail_hillshade
    from .rendering.tile_uv_align import (
        build_polygon_cut_atlas,
        compute_tile_view_limits,
        compute_uniform_half_span,
    )

    composites = {}
    detail_grids = {}
    for fid in face_ids:
        composites[fid] = build_tile_with_neighbours(
            coll, fid, grid, skip_neighbour_closure=True,
        )
        detail_grids[fid] = coll.get(fid)[0]

    uniform_hs = compute_uniform_half_span(composites, face_ids)

    # ── 5. Render tiles ─────────────────────────────────────────────
    biome = BiomeConfig()

    tile_images = {}
    for fid in face_ids:
        composite = composites[fid]
        mg = composite.merged
        stitched_store = _build_stitched_store(composite, coll, fid, grid)

        # Compute hillshade for the composite
        hs = _detail_hillshade(
            mg, stitched_store, "elevation",
            azimuth=biome.azimuth, altitude=biome.altitude,
        )

        if ramp_fn is not None:
            # Use the caller-supplied ramp for solid-colour / custom rendering.
            def _custom_colour(fid_inner, _grid, _face,
                               _store=stitched_store, _fn=ramp_fn):
                elev = _store.get(fid_inner, "elevation")
                return _fn(max(0.0, min(1.0, float(elev))))
            colour_fn = _custom_colour
        else:
            colour_fn = _terrain_colour_fn(
                mg, stitched_store, biome, params.seed, hs,
            )
        xlim, ylim = compute_tile_view_limits(
            composite, fid, uniform_half_span=uniform_hs,
        )

        tile_images[fid] = _render_tile_analytical(
            mg, colour_fn, tile_size, xlim, ylim,
        )

    # ── 6. Build polygon-cut atlas ──────────────────────────────────
    atlas, uv_layout = build_polygon_cut_atlas(
        tile_images, composites, detail_grids, grid, face_ids,
        tile_size=tile_size,
        gutter=gutter,
        uniform_half_span=uniform_hs,
        pent_edge_interior_pull=(0.12 if int(frequency) <= 2 else 0.0),
        hex_pent_edge_interior_pull=(0.06 if int(frequency) <= 2 else 0.0),
    )

    # Encode atlas to PNG bytes
    buf = io.BytesIO()
    atlas.save(buf, format="PNG")
    atlas_png = buf.getvalue()

    # ── 7. Export globe payload ─────────────────────────────────────
    from .globe.globe_export import export_globe_payload

    payload = export_globe_payload(grid, store, ramp="satellite", ramp_fn=ramp_fn)

    # ── 8. Build batched globe mesh ─────────────────────────────────
    from .rendering.globe_renderer_v2 import build_batched_globe_mesh

    vertex_data, index_data = build_batched_globe_mesh(
        frequency,
        uv_layout,
        subdivisions=3,
    )

    # ── 9. Compute sub-tile detail cell 3D centres ──────────────────
    from .rendering.detail_centers import build_slug_keyed_detail_centers
    from .rendering.detail_cell_contract import normalize_detail_cells_tiles_with_report
    from .rendering.seam_strips import build_seam_strip_payload_from_globe_payload

    detail_cells_report: dict[str, int] = {}
    try:
        detail_cells, normalization_report = normalize_detail_cells_tiles_with_report(
            build_slug_keyed_detail_centers(grid, detail_rings=detail_rings),
            strict=_detail_cells_strict_mode(),
        )
        detail_cells_report = normalization_report.to_dict()
        gen_result.metadata["detail_cells_normalization"] = detail_cells_report
        if (
            detail_cells_report.get("cells_dropped", 0) > 0
            or detail_cells_report.get("repaired_index_tiles", 0) > 0
        ):
            LOGGER.warning("Detail-cell normalization adjustments: %s", detail_cells_report)
    except Exception:
        LOGGER.warning("Failed to compute detail cell centres", exc_info=True)
        detail_cells = {}

    try:
        seam_strips = build_seam_strip_payload_from_globe_payload(
            payload,
            frequency=frequency,
            detail_rings=detail_rings,
        )
    except Exception:
        LOGGER.warning("Failed to build seam-strip payload", exc_info=True)
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
    gen_result.metadata["seam_strips"] = dict(seam_strips.get("metadata") or {})

    t_elapsed = time.monotonic() - t_start
    LOGGER.info("Atlas generation complete in %.2fs", t_elapsed)

    return PlanetAtlasResult(
        generation=gen_result,
        atlas_png=atlas_png,
        uv_layout=uv_layout,
        globe_payload=payload,
        vertex_data=vertex_data,
        index_data=index_data,
        frequency=frequency,
        atlas_width=atlas.size[0],
        atlas_height=atlas.size[1],
        detail_cells=detail_cells,
        seam_strips=seam_strips,
    )


def generate_placeholder_atlas(
    spec: PlaceholderAtlasSpec,
) -> PlanetAtlasResult:
    """Fast placeholder atlas from *spec* — no terrain pipeline.

    Delegates to :mod:`polygrid.placeholder_atlas`, which maintains a
    per-process memory cache and an on-disk artifact cache keyed by
    ``(frequency, detail_rings, tile_size, gutter)``.  Subsequent calls with the same
    topology skip all geometry and warp computation.

    Parameters
    ----------
    spec : PlaceholderAtlasSpec
        Colour, topology, and seed parameters.

    Returns
    -------
    PlanetAtlasResult
        Same shape as :func:`generate_planet_atlas`.
    """
    from .placeholder_atlas import generate_placeholder_atlas as _gen

    return _gen(spec)


def bootstrap_placeholder_atlas(
    *,
    frequency: int,
    detail_rings: int,
    tile_sizes: tuple[int, ...] = (128, 512),
    gutter: int = 4,
) -> Dict[str, Any]:
    """Prewarm placeholder artifacts and detail-cell caches."""
    from .placeholder_atlas import bootstrap_placeholder_artifacts as _bootstrap

    return _bootstrap(
        frequency=frequency,
        detail_rings=detail_rings,
        tile_sizes=tile_sizes,
        gutter=gutter,
    )
