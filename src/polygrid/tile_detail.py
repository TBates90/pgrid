"""Sub-tile detail grid infrastructure for Goldberg globe tiles.

Each Goldberg tile on a :class:`GlobeGrid` is expanded into a local
``PolyGrid`` — a hex grid for hexagonal tiles and a pentagon-centred
grid for pentagonal tiles.  The resulting collection of detail grids
carries per-face terrain data that can be rendered as textures and
UV-mapped onto the 3-D tile surfaces.

This module provides:

- :class:`TileDetailSpec` — configuration dataclass
- :func:`build_all_detail_grids` — batch detail grid construction
- :class:`DetailGridCollection` — container for all detail grids + stores
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .detail_grid import build_detail_grid, detail_face_count
from .geometry import face_center
from .heightmap import smooth_field
from .noise import fbm, domain_warp
from .polygrid import PolyGrid
from .tile_data import FieldDef, TileDataStore, TileSchema


# ═══════════════════════════════════════════════════════════════════
# 10A.1 — TileDetailSpec
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TileDetailSpec:
    """Configuration for sub-tile detail grid generation.

    Controls the resolution and noise parameters used when expanding
    each Goldberg tile into a local PolyGrid with terrain data.

    Parameters
    ----------
    detail_rings : int
        Ring count for sub-tile grids.  A hex grid with 4 rings has
        61 sub-faces; 6 rings → 127 sub-faces.
    noise_frequency : float
        Spatial frequency of intra-tile noise (higher = finer detail).
    noise_octaves : int
        Number of noise octaves.
    amplitude : float
        How much local noise varies from the parent elevation (0–1).
    base_weight : float
        Parent elevation dominance (0–1).  Higher values mean the
        detail grid follows the parent more closely.
    boundary_smoothing : int
        Smoothing passes applied to boundary-band faces to reduce
        seam visibility between adjacent tiles.
    warp_strength : float
        Domain-warp strength for organic-looking detail variation.
    seed_offset : int
        Added to the parent seed to produce per-tile variation while
        keeping the global seed deterministic.
    """

    detail_rings: int = 4
    noise_frequency: float = 6.0
    noise_octaves: int = 5
    amplitude: float = 0.12
    base_weight: float = 0.80
    boundary_smoothing: int = 2
    warp_strength: float = 0.15
    seed_offset: int = 0


# ═══════════════════════════════════════════════════════════════════
# 10A.2 — Build all detail grids
# ═══════════════════════════════════════════════════════════════════

def build_all_detail_grids(
    globe_grid: PolyGrid,
    spec: TileDetailSpec,
    *,
    size: float = 1.0,
) -> Dict[str, PolyGrid]:
    """Build a detail grid for every face in a globe grid.

    Parameters
    ----------
    globe_grid : PolyGrid
        A :class:`GlobeGrid` (or any PolyGrid whose faces have
        ``face_type`` of ``"pent"`` or ``"hex"``).
    spec : TileDetailSpec
        Detail grid configuration.
    size : float
        Cell size passed to the grid builders.

    Returns
    -------
    dict
        ``{face_id: PolyGrid}`` — one detail grid per globe face.
    """
    grids: Dict[str, PolyGrid] = {}
    for face_id in globe_grid.faces:
        grid = build_detail_grid(
            globe_grid, face_id,
            detail_rings=spec.detail_rings,
            size=size,
        )
        grids[face_id] = grid
    return grids


# ═══════════════════════════════════════════════════════════════════
# 10A.3 — DetailGridCollection
# ═══════════════════════════════════════════════════════════════════

class DetailGridCollection:
    """Container managing detail grids and their tile-data stores.

    Holds one ``PolyGrid`` and one ``TileDataStore`` per globe face,
    with convenience methods for batch terrain generation and queries.

    Parameters
    ----------
    globe_grid : PolyGrid
        The globe grid that owns the faces.
    spec : TileDetailSpec
        Configuration used for grid construction and terrain gen.
    grids : dict
        ``{face_id: PolyGrid}`` — pre-built detail grids.
    """

    def __init__(
        self,
        globe_grid: PolyGrid,
        spec: TileDetailSpec,
        grids: Dict[str, PolyGrid],
    ) -> None:
        self._globe_grid = globe_grid
        self._spec = spec
        self._grids = dict(grids)
        self._stores: Dict[str, TileDataStore] = {}

    # ── Factory ─────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        globe_grid: PolyGrid,
        spec: Optional[TileDetailSpec] = None,
        *,
        size: float = 1.0,
    ) -> "DetailGridCollection":
        """Build a :class:`DetailGridCollection` for every globe face.

        Parameters
        ----------
        globe_grid : PolyGrid
        spec : TileDetailSpec, optional
            Uses defaults if not given.
        size : float
            Cell size for detail grids.

        Returns
        -------
        DetailGridCollection
        """
        if spec is None:
            spec = TileDetailSpec()
        grids = build_all_detail_grids(globe_grid, spec, size=size)
        return cls(globe_grid, spec, grids)

    # ── Properties ──────────────────────────────────────────────────

    @property
    def globe_grid(self) -> PolyGrid:
        """The parent globe grid."""
        return self._globe_grid

    @property
    def spec(self) -> TileDetailSpec:
        """The detail spec used for construction."""
        return self._spec

    @property
    def grids(self) -> Dict[str, PolyGrid]:
        """``{face_id: PolyGrid}`` — all detail grids."""
        return dict(self._grids)

    @property
    def stores(self) -> Dict[str, TileDataStore]:
        """``{face_id: TileDataStore}`` — all tile data stores."""
        return dict(self._stores)

    # ── Accessors ───────────────────────────────────────────────────

    def get(self, face_id: str) -> Tuple[PolyGrid, Optional[TileDataStore]]:
        """Return ``(detail_grid, store)`` for a face.

        The store may be ``None`` if terrain has not been generated yet.

        Raises
        ------
        KeyError
            If *face_id* is not in the collection.
        """
        if face_id not in self._grids:
            raise KeyError(f"No detail grid for face '{face_id}'")
        return self._grids[face_id], self._stores.get(face_id)

    @property
    def face_ids(self) -> List[str]:
        """Sorted list of face ids in the collection."""
        return sorted(self._grids.keys())

    @property
    def total_face_count(self) -> int:
        """Sum of sub-face counts across all detail grids."""
        return sum(len(g.faces) for g in self._grids.values())

    def detail_face_count_for(self, face_id: str) -> int:
        """Number of sub-faces in the detail grid for *face_id*."""
        if face_id not in self._grids:
            raise KeyError(f"No detail grid for face '{face_id}'")
        return len(self._grids[face_id].faces)

    # ── Terrain generation ──────────────────────────────────────────

    def generate_all_terrain(
        self,
        globe_store: TileDataStore,
        *,
        seed: int = 42,
        elevation_field: str = "elevation",
    ) -> None:
        """Generate terrain for every detail grid in the collection.

        This is the basic (non-boundary-aware) version.  For boundary-
        continuous terrain, use the ``detail_terrain`` module's
        :func:`generate_all_detail_terrain` function instead.

        Each detail grid receives elevation from its parent tile plus
        high-frequency noise variation.

        Parameters
        ----------
        globe_store : TileDataStore
            Globe-level tile data with an elevation field.
        seed : int
            Base noise seed.
        elevation_field : str
            Name of the elevation field in *globe_store*.
        """
        spec = self._spec

        for face_id, detail_grid in self._grids.items():
            parent_elev = globe_store.get(face_id, elevation_field)
            tile_seed = seed + spec.seed_offset + hash(face_id) % 10000

            schema = TileSchema([FieldDef("elevation", float, 0.0)])
            store = TileDataStore(grid=detail_grid, schema=schema)

            for fid in detail_grid.faces:
                face = detail_grid.faces[fid]
                c = face_center(detail_grid.vertices, face)
                if c is None:
                    continue
                cx, cy = c

                # Layer domain-warped fbm for organic variation
                if spec.warp_strength > 0:
                    noise_val = domain_warp(
                        fbm, cx, cy,
                        warp_strength=spec.warp_strength,
                        warp_frequency=spec.noise_frequency * 0.5,
                        warp_seed_x=tile_seed + 1000,
                        warp_seed_y=tile_seed + 2000,
                        octaves=spec.noise_octaves,
                        frequency=spec.noise_frequency,
                        seed=tile_seed,
                    )
                else:
                    noise_val = fbm(
                        cx, cy,
                        octaves=spec.noise_octaves,
                        frequency=spec.noise_frequency,
                        seed=tile_seed,
                    )

                elevation = (
                    parent_elev * spec.base_weight
                    + noise_val * spec.amplitude * (1.0 - spec.base_weight)
                )
                store.set(fid, "elevation", elevation)

            # Smooth to soften cell-to-cell jumps
            if spec.boundary_smoothing > 0:
                smooth_field(
                    detail_grid, store, "elevation",
                    iterations=spec.boundary_smoothing,
                    self_weight=0.6,
                )

            self._stores[face_id] = store

    # ── Summary ─────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable summary of the collection."""
        n_grids = len(self._grids)
        n_stores = len(self._stores)
        total = self.total_face_count
        n_pent = sum(
            1 for fid in self._grids
            if self._globe_grid.faces[fid].face_type == "pent"
        )
        n_hex = n_grids - n_pent

        pent_faces = detail_face_count("pent", self._spec.detail_rings)
        hex_faces = detail_face_count("hex", self._spec.detail_rings)

        lines = [
            f"DetailGridCollection: {n_grids} tiles, {total} total sub-faces",
            f"  Pentagon tiles: {n_pent} × {pent_faces} faces = {n_pent * pent_faces}",
            f"  Hexagon tiles:  {n_hex} × {hex_faces} faces = {n_hex * hex_faces}",
            f"  Detail rings:   {self._spec.detail_rings}",
            f"  Terrain stores: {n_stores} / {n_grids}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DetailGridCollection(tiles={len(self._grids)}, "
            f"sub_faces={self.total_face_count}, "
            f"rings={self._spec.detail_rings})"
        )
