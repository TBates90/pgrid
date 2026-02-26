"""Terrain partitioning — splitting grids into named regions.

This module provides the :class:`Region` and :class:`RegionMap` data
models plus a family of partitioning algorithms that assign every face
of a :class:`PolyGrid` to exactly one region.

Algorithms
----------
- :func:`partition_angular` — equal angular sectors around the grid
  centroid (adapts :func:`transforms.apply_partition` logic).
- :func:`partition_flood_fill` — competitive flood-fill from seed faces.
- :func:`partition_voronoi` — assign each face to its nearest seed by
  centroid distance.
- :func:`partition_noise` — Voronoi-based with simplex-noise boundary
  perturbation for organic shapes (requires ``opensimplex``; degrades
  gracefully to pure Voronoi if unavailable).

Integration helpers
-------------------
- :func:`assign_field` — bulk-set a tile-data field for all faces in a
  region.
- :func:`regions_to_overlay` — convert a :class:`RegionMap` into an
  :class:`Overlay` for visualisation.
- :func:`validate_region_map` — check coverage, no gaps, no overlaps,
  constraint satisfaction.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .algorithms import build_face_adjacency, get_face_adjacency
from .geometry import face_center, grid_center
from .models import Face, Vertex
from .polygrid import PolyGrid
from .transforms import Overlay, OverlayRegion


# ═══════════════════════════════════════════════════════════════════
# Data model
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Region:
    """A named collection of face ids with optional metadata.

    Parameters
    ----------
    name : str
        Human-readable name (e.g. ``"continent_1"``).
    face_ids : frozenset[str]
        Immutable set of face ids belonging to this region.
    metadata : dict
        Arbitrary key-value metadata (e.g. ``{"biome": "temperate"}``).
    """

    name: str
    face_ids: FrozenSet[str] = field(default_factory=frozenset)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """Number of faces in the region."""
        return len(self.face_ids)

    def __contains__(self, face_id: str) -> bool:
        return face_id in self.face_ids

    def __repr__(self) -> str:
        meta = f", metadata={self.metadata}" if self.metadata else ""
        return f"Region(name={self.name!r}, faces={self.size}{meta})"


@dataclass
class RegionMap:
    """Container holding all regions for a grid.

    Guarantees that every face belongs to exactly one region (validated
    on construction and via :func:`validate_region_map`).

    Parameters
    ----------
    regions : list[Region]
        The regions.  Face id sets must be disjoint.
    grid_face_ids : frozenset[str]
        All face ids in the grid (for coverage validation).
    """

    regions: List[Region] = field(default_factory=list)
    grid_face_ids: FrozenSet[str] = field(default_factory=frozenset)

    # ── convenience accessors ───────────────────────────────────────

    @property
    def region_names(self) -> List[str]:
        """Ordered list of region names."""
        return [r.name for r in self.regions]

    def get_region(self, name: str) -> Region:
        """Return the region with *name*, or raise ``KeyError``."""
        for r in self.regions:
            if r.name == name:
                return r
        raise KeyError(f"No region named {name!r}")

    def face_to_region(self) -> Dict[str, str]:
        """Return ``{face_id: region_name}`` lookup."""
        mapping: Dict[str, str] = {}
        for r in self.regions:
            for fid in r.face_ids:
                mapping[fid] = r.name
        return mapping

    def region_for_face(self, face_id: str) -> str:
        """Return the region name that owns *face_id*, or raise ``KeyError``."""
        for r in self.regions:
            if face_id in r.face_ids:
                return r.name
        raise KeyError(f"Face {face_id!r} not in any region")

    def region_adjacency(
        self, adjacency: Dict[str, List[str]]
    ) -> Dict[str, Set[str]]:
        """Return ``{region_name: {neighbouring region names}}``."""
        f2r = self.face_to_region()
        result: Dict[str, Set[str]] = {r.name: set() for r in self.regions}
        for r in self.regions:
            for fid in r.face_ids:
                for nid in adjacency.get(fid, []):
                    nr = f2r.get(nid)
                    if nr is not None and nr != r.name:
                        result[r.name].add(nr)
        return result

    def __len__(self) -> int:
        return len(self.regions)

    def __repr__(self) -> str:
        sizes = ", ".join(f"{r.name}={r.size}" for r in self.regions)
        return f"RegionMap([{sizes}])"


# ═══════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RegionValidation:
    """Result of :func:`validate_region_map`."""

    ok: bool
    errors: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def validate_region_map(
    region_map: RegionMap,
    *,
    min_region_size: int = 0,
    max_region_count: int = 0,
    adjacency: Optional[Dict[str, List[str]]] = None,
    required_adjacency: Optional[Dict[str, str]] = None,
) -> RegionValidation:
    """Check a :class:`RegionMap` for correctness and constraint satisfaction.

    Parameters
    ----------
    region_map : RegionMap
        The map to validate.
    min_region_size : int
        If > 0, every region must have at least this many faces.
    max_region_count : int
        If > 0, there must be at most this many regions.
    adjacency : dict, optional
        Face adjacency (needed for ``required_adjacency`` check).
    required_adjacency : dict, optional
        ``{region_name: must_touch_region_name}`` — e.g.
        ``{"continent_1": "ocean"}`` means continent_1 must be adjacent
        to the "ocean" region.

    Returns
    -------
    RegionValidation
    """
    errors: List[str] = []

    # 1. Full coverage — every grid face assigned
    assigned: Set[str] = set()
    for r in region_map.regions:
        assigned |= r.face_ids
    missing = region_map.grid_face_ids - assigned
    if missing:
        errors.append(
            f"Unassigned faces ({len(missing)}): "
            + ", ".join(sorted(missing)[:5])
            + ("…" if len(missing) > 5 else "")
        )

    # 2. No overlaps — face appears in at most one region
    seen: Dict[str, str] = {}
    for r in region_map.regions:
        for fid in r.face_ids:
            if fid in seen:
                errors.append(
                    f"Face {fid!r} in both {seen[fid]!r} and {r.name!r}"
                )
            else:
                seen[fid] = r.name

    # 3. No extra faces — all assigned faces must be grid faces
    extra = assigned - region_map.grid_face_ids
    if extra:
        errors.append(
            f"Extra faces not in grid ({len(extra)}): "
            + ", ".join(sorted(extra)[:5])
        )

    # 4. No duplicate region names
    names = [r.name for r in region_map.regions]
    if len(names) != len(set(names)):
        errors.append("Duplicate region names detected")

    # 5. Constraint: min_region_size
    if min_region_size > 0:
        for r in region_map.regions:
            if r.size < min_region_size:
                errors.append(
                    f"Region {r.name!r} has {r.size} faces "
                    f"(min required: {min_region_size})"
                )

    # 6. Constraint: max_region_count
    if max_region_count > 0 and len(region_map.regions) > max_region_count:
        errors.append(
            f"Too many regions: {len(region_map.regions)} "
            f"(max allowed: {max_region_count})"
        )

    # 7. Constraint: required adjacency
    if required_adjacency and adjacency:
        radj = region_map.region_adjacency(adjacency)
        for rname, must_touch in required_adjacency.items():
            if rname not in radj:
                errors.append(
                    f"Required adjacency: region {rname!r} not found"
                )
            elif must_touch not in radj[rname]:
                errors.append(
                    f"Region {rname!r} must be adjacent to {must_touch!r} "
                    f"but is not"
                )

    return RegionValidation(ok=len(errors) == 0, errors=errors)


# ═══════════════════════════════════════════════════════════════════
# Partitioning algorithms
# ═══════════════════════════════════════════════════════════════════

def _compute_centroids(
    grid: PolyGrid,
) -> Dict[str, Tuple[float, float]]:
    """Return ``{face_id: (cx, cy)}`` for all faces with positioned vertices."""
    centroids: Dict[str, Tuple[float, float]] = {}
    for face in grid.faces.values():
        c = face_center(grid.vertices, face)
        if c is not None:
            centroids[face.id] = c
    return centroids


# ── Angular sectors ─────────────────────────────────────────────────

def partition_angular(
    grid: PolyGrid,
    n_sections: int = 6,
    *,
    name_prefix: str = "sector",
) -> RegionMap:
    """Partition faces into *n_sections* equal angular sectors.

    Each sector is a wedge of angular space around the grid centroid.
    Adapts the logic from :func:`transforms.apply_partition` but
    produces :class:`Region` objects instead of an :class:`Overlay`.

    Parameters
    ----------
    grid : PolyGrid
    n_sections : int
        Number of sectors (default 6).
    name_prefix : str
        Region name prefix; regions are named ``"{prefix}_{i}"``.
    """
    if n_sections < 1:
        raise ValueError("n_sections must be >= 1")

    centroids = _compute_centroids(grid)

    # Grid centroid
    gcx, gcy = grid_center(grid.vertices)

    # Assign each face to a sector
    sector_width = 2.0 * math.pi / n_sections
    buckets: Dict[int, Set[str]] = {i: set() for i in range(n_sections)}

    for fid, (cx, cy) in centroids.items():
        angle = math.atan2(cy - gcy, cx - gcx)
        if angle < 0:
            angle += 2.0 * math.pi
        sector = int(angle / sector_width) % n_sections
        buckets[sector].add(fid)

    # Faces without centroids (shouldn't happen, but be safe)
    unassigned = set(grid.faces.keys()) - set(centroids.keys())
    if unassigned:
        buckets[0] |= unassigned

    regions = [
        Region(
            name=f"{name_prefix}_{i}",
            face_ids=frozenset(fids),
        )
        for i, fids in sorted(buckets.items())
        if fids  # skip empty sectors
    ]

    return RegionMap(
        regions=regions,
        grid_face_ids=frozenset(grid.faces.keys()),
    )


# ── Flood-fill from seeds ──────────────────────────────────────────

def partition_flood_fill(
    grid: PolyGrid,
    seed_face_ids: Sequence[str],
    *,
    names: Optional[Sequence[str]] = None,
    rng: Optional[random.Random] = None,
) -> RegionMap:
    """Competitive flood-fill from seed faces.

    Each seed expands outward one ring at a time; when two seeds
    contest the same face, the one that reaches it first wins.  Ties
    are broken randomly.

    Parameters
    ----------
    grid : PolyGrid
    seed_face_ids : sequence of str
        Starting face ids — one per region.
    names : sequence of str, optional
        Region names.  Defaults to ``"region_0"`` etc.
    rng : random.Random, optional
        Random number generator for tie-breaking. If *None* a
        deterministic default is used.
    """
    if not seed_face_ids:
        raise ValueError("At least one seed face required")
    for sid in seed_face_ids:
        if sid not in grid.faces:
            raise KeyError(f"Seed face {sid!r} not in grid")

    n = len(seed_face_ids)
    if names is None:
        names = [f"region_{i}" for i in range(n)]
    if len(names) != n:
        raise ValueError("len(names) must equal len(seed_face_ids)")

    if rng is None:
        rng = random.Random(42)

    adjacency = get_face_adjacency(grid)

    # owner[face_id] = region index
    owner: Dict[str, int] = {}
    # frontier per region
    frontiers: List[List[str]] = [[] for _ in range(n)]

    for i, sid in enumerate(seed_face_ids):
        owner[sid] = i
        frontiers[i].append(sid)

    # BFS rounds — all regions expand one layer per round
    while any(frontiers):
        # Collect candidates for this round
        candidates: Dict[str, List[int]] = {}  # face_id → list of region indices claiming it
        for i, frontier in enumerate(frontiers):
            for fid in frontier:
                for nid in adjacency.get(fid, []):
                    if nid not in owner:
                        candidates.setdefault(nid, []).append(i)

        if not candidates:
            break

        # Resolve: first-come wins; ties broken randomly
        new_frontiers: List[List[str]] = [[] for _ in range(n)]
        # Shuffle candidates so tie-breaking isn't biased by dict order
        clist = list(candidates.items())
        rng.shuffle(clist)

        for fid, claimants in clist:
            if fid in owner:
                continue  # already claimed in this round
            # Pick a winner — unique claimant wins outright,
            # otherwise random
            winner = claimants[0] if len(claimants) == 1 else rng.choice(claimants)
            owner[fid] = winner
            new_frontiers[winner].append(fid)

        frontiers = new_frontiers

    # Build regions
    buckets: Dict[int, Set[str]] = {i: set() for i in range(n)}
    for fid, idx in owner.items():
        buckets[idx].add(fid)

    # Assign any remaining unowned faces to nearest owned neighbour
    unowned = set(grid.faces.keys()) - set(owner.keys())
    for fid in unowned:
        # BFS from fid to find nearest owned face
        visited: Set[str] = {fid}
        queue = deque([fid])
        found = False
        while queue and not found:
            cur = queue.popleft()
            for nid in adjacency.get(cur, []):
                if nid in owner:
                    idx = owner[nid]
                    buckets[idx].add(fid)
                    owner[fid] = idx
                    found = True
                    break
                if nid not in visited:
                    visited.add(nid)
                    queue.append(nid)

    regions = [
        Region(name=names[i], face_ids=frozenset(buckets[i]))
        for i in range(n)
    ]

    return RegionMap(
        regions=regions,
        grid_face_ids=frozenset(grid.faces.keys()),
    )


# ── Voronoi-based partitioning ─────────────────────────────────────

def partition_voronoi(
    grid: PolyGrid,
    seed_face_ids: Sequence[str],
    *,
    names: Optional[Sequence[str]] = None,
) -> RegionMap:
    """Assign each face to the nearest seed by centroid distance.

    This produces cleaner, more regular boundaries than flood-fill
    because it's purely geometric rather than topological.

    Parameters
    ----------
    grid : PolyGrid
    seed_face_ids : sequence of str
        Seed face ids (one per region).
    names : sequence of str, optional
        Region names.
    """
    if not seed_face_ids:
        raise ValueError("At least one seed face required")
    for sid in seed_face_ids:
        if sid not in grid.faces:
            raise KeyError(f"Seed face {sid!r} not in grid")

    n = len(seed_face_ids)
    if names is None:
        names = [f"region_{i}" for i in range(n)]
    if len(names) != n:
        raise ValueError("len(names) must equal len(seed_face_ids)")

    centroids = _compute_centroids(grid)

    # Seed centroid positions
    seed_positions: List[Tuple[float, float]] = []
    for sid in seed_face_ids:
        if sid not in centroids:
            raise ValueError(f"Seed face {sid!r} has no centroid position")
        seed_positions.append(centroids[sid])

    # Assign each face to nearest seed
    buckets: Dict[int, Set[str]] = {i: set() for i in range(n)}

    for fid, (cx, cy) in centroids.items():
        best_idx = 0
        best_dist = float("inf")
        for i, (sx, sy) in enumerate(seed_positions):
            d = math.hypot(cx - sx, cy - sy)
            if d < best_dist:
                best_dist = d
                best_idx = i
        buckets[best_idx].add(fid)

    # Faces without centroids → assign to region 0
    unassigned = set(grid.faces.keys()) - set(centroids.keys())
    buckets[0] |= unassigned

    regions = [
        Region(name=names[i], face_ids=frozenset(buckets[i]))
        for i in range(n)
    ]

    return RegionMap(
        regions=regions,
        grid_face_ids=frozenset(grid.faces.keys()),
    )


# ── Noise-based boundary perturbation ──────────────────────────────

def partition_noise(
    grid: PolyGrid,
    seed_face_ids: Sequence[str],
    *,
    names: Optional[Sequence[str]] = None,
    noise_scale: float = 1.0,
    noise_weight: float = 0.3,
    seed: int = 42,
) -> RegionMap:
    """Voronoi-based partitioning with noise-perturbed distances.

    Each face's distance to a seed is perturbed by simplex noise,
    producing organic, irregular boundaries instead of straight Voronoi
    edges.

    If ``opensimplex`` is not installed, falls back to pure Voronoi
    partitioning (with a warning stored in metadata).

    Parameters
    ----------
    grid : PolyGrid
    seed_face_ids : sequence of str
        Seed face ids (one per region).
    names : sequence of str, optional
        Region names.
    noise_scale : float
        Spatial scale of the noise — larger values produce broader
        perturbations.
    noise_weight : float
        How much the noise affects distance (0 = pure Voronoi, 1 =
        heavily perturbed).  Values in [0.1, 0.5] work well.
    seed : int
        Random seed for the noise generator.
    """
    if not seed_face_ids:
        raise ValueError("At least one seed face required")
    for sid in seed_face_ids:
        if sid not in grid.faces:
            raise KeyError(f"Seed face {sid!r} not in grid")

    n = len(seed_face_ids)
    if names is None:
        names = [f"region_{i}" for i in range(n)]
    if len(names) != n:
        raise ValueError("len(names) must equal len(seed_face_ids)")

    centroids = _compute_centroids(grid)

    seed_positions: List[Tuple[float, float]] = []
    for sid in seed_face_ids:
        if sid not in centroids:
            raise ValueError(f"Seed face {sid!r} has no centroid position")
        seed_positions.append(centroids[sid])

    # Try to import opensimplex
    noise_fn: Optional[Callable[[float, float], float]] = None
    fallback = False
    try:
        import opensimplex

        opensimplex.seed(seed)
        noise_fn = opensimplex.noise2
    except ImportError:
        fallback = True
        # Use a simple deterministic hash-based pseudo-noise fallback
        _rng = random.Random(seed)

        def _pseudo_noise(x: float, y: float) -> float:
            # Deterministic pseudo-noise from coordinates
            h = hash((round(x * 1000), round(y * 1000), seed))
            return (h % 10000) / 10000.0 * 2.0 - 1.0  # [-1, 1]

        noise_fn = _pseudo_noise

    # Assign each face to nearest seed (distance perturbed by noise)
    buckets: Dict[int, Set[str]] = {i: set() for i in range(n)}

    for fid, (cx, cy) in centroids.items():
        noise_val = noise_fn(cx / max(noise_scale, 1e-6), cy / max(noise_scale, 1e-6))
        best_idx = 0
        best_dist = float("inf")
        for i, (sx, sy) in enumerate(seed_positions):
            d = math.hypot(cx - sx, cy - sy)
            # Perturb distance — noise_val in roughly [-1, 1]
            d_perturbed = d * (1.0 + noise_weight * noise_val)
            if d_perturbed < best_dist:
                best_dist = d_perturbed
                best_idx = i
        buckets[best_idx].add(fid)

    # Faces without centroids → assign to region 0
    unassigned = set(grid.faces.keys()) - set(centroids.keys())
    buckets[0] |= unassigned

    regions = [
        Region(name=names[i], face_ids=frozenset(buckets[i]))
        for i in range(n)
    ]

    rm = RegionMap(
        regions=regions,
        grid_face_ids=frozenset(grid.faces.keys()),
    )

    # Stash noise info in first region's metadata
    if regions:
        regions[0].metadata["noise_fallback"] = fallback
        regions[0].metadata["noise_scale"] = noise_scale
        regions[0].metadata["noise_weight"] = noise_weight

    return rm


# ═══════════════════════════════════════════════════════════════════
# TileData integration
# ═══════════════════════════════════════════════════════════════════

def assign_field(
    region: Region,
    tile_data: Any,  # TileData or TileDataStore
    key: str,
    value: Any,
) -> None:
    """Set *key* = *value* on all faces in *region*.

    Works with both :class:`TileData` and :class:`TileDataStore`.

    >>> assign_field(region, store, "biome", "temperate")
    """
    tile_data.bulk_set(region.face_ids, key, value)


def assign_biome(
    region: Region,
    tile_data: Any,  # TileData or TileDataStore
    biome_type: str,
    biome_key: str = "biome",
) -> None:
    """Convenience: set biome type for all faces in a region.

    Also stores the biome in the region's metadata.

    >>> assign_biome(region, store, "temperate")
    """
    assign_field(region, tile_data, biome_key, biome_type)
    region.metadata["biome"] = biome_type


# ═══════════════════════════════════════════════════════════════════
# Overlay conversion (for visualisation)
# ═══════════════════════════════════════════════════════════════════

def regions_to_overlay(
    region_map: RegionMap,
    grid: PolyGrid,
) -> Overlay:
    """Convert a :class:`RegionMap` into an :class:`Overlay` for rendering.

    Each face becomes an :class:`OverlayRegion` whose
    ``source_vertex_id`` stores the region index (as a string),
    matching the convention used by :func:`transforms.apply_partition`
    so that :func:`visualize._draw_overlay` colours them correctly.
    """
    overlay = Overlay(kind="partition")

    # Build face → region index mapping
    f2r: Dict[str, int] = {}
    for i, region in enumerate(region_map.regions):
        for fid in region.face_ids:
            f2r[fid] = i

    for face in grid.faces.values():
        if face.id not in f2r:
            continue
        pts: List[Tuple[float, float]] = []
        for vid in face.vertex_ids:
            v = grid.vertices[vid]
            if v.has_position():
                pts.append((v.x, v.y))
        if len(pts) < 3:
            continue
        overlay.regions.append(OverlayRegion(
            id=f"reg_{face.id}",
            points=pts,
            source_vertex_id=str(f2r[face.id]),
        ))

    overlay.metadata["n_sections"] = len(region_map.regions)
    overlay.metadata["region_names"] = region_map.region_names

    return overlay
