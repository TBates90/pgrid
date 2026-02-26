"""River generation — Phase 7E.

Rivers are linear features that flow *through* existing terrain following
the elevation gradient downhill.  They modify the existing elevation
(carving valleys) and add ``"river"`` / ``"river_width"`` tile-data
fields rather than creating new region partitions.

Architecture
------------
- **Primitives** (7E.1): steepest_descent_path, find_drainage_basins,
  fill_depressions, flow_accumulation — generic hydrology building blocks.
- **Network** (7E.2): RiverSegment, RiverNetwork, generate_rivers —
  constructs the actual river tree from primitives.
- **Integration** (7E.3): carve_river_valleys, assign_river_data,
  river_to_overlay — connect rivers back to terrain and rendering.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
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
from .geometry import face_center
from .polygrid import PolyGrid
from .tile_data import TileDataStore
from .transforms import Overlay, OverlayRegion


# ═══════════════════════════════════════════════════════════════════
# 7E.1 — River primitives
# ═══════════════════════════════════════════════════════════════════


def steepest_descent_path(
    adjacency: Dict[str, List[str]],
    store: TileDataStore,
    start: str,
    field_name: str = "elevation",
    *,
    max_steps: int = 10000,
) -> List[str]:
    """Follow the steepest downhill neighbour from *start* until stuck.

    Returns an ordered list of face ids from *start* to the terminus
    (a local minimum or grid boundary).  Handles plateaus by BFS to
    find the nearest strictly-lower face.

    Parameters
    ----------
    adjacency : dict
        Face adjacency map.
    store : TileDataStore
    start : str
        Starting face id.
    field_name : str
        Elevation field.
    max_steps : int
        Safety limit.

    Returns
    -------
    list of str
        Ordered face ids forming the descent path.
    """
    path = [start]
    visited: Set[str] = {start}

    for _ in range(max_steps):
        current = path[-1]
        cur_elev = store.get(current, field_name)

        # Find steepest downhill neighbour
        best_nid: Optional[str] = None
        best_elev = cur_elev

        for nid in adjacency.get(current, []):
            if nid in visited:
                continue
            ne = store.get(nid, field_name)
            if ne < best_elev:
                best_elev = ne
                best_nid = nid

        if best_nid is not None:
            visited.add(best_nid)
            path.append(best_nid)
            continue

        # Plateau handling: BFS to find nearest strictly-lower face
        plateau_target = _bfs_find_lower(adjacency, store, current, field_name, visited)
        if plateau_target is not None:
            visited.add(plateau_target)
            path.append(plateau_target)
            continue

        # Stuck at a local minimum — stop
        break

    return path


def _bfs_find_lower(
    adjacency: Dict[str, List[str]],
    store: TileDataStore,
    start: str,
    field_name: str,
    already_visited: Set[str],
) -> Optional[str]:
    """BFS from *start* to find the nearest face with strictly lower elevation."""
    cur_elev = store.get(start, field_name)
    queue: deque[str] = deque()
    seen: Set[str] = {start} | already_visited

    for nid in adjacency.get(start, []):
        if nid not in seen:
            seen.add(nid)
            queue.append(nid)

    while queue:
        fid = queue.popleft()
        if store.get(fid, field_name) < cur_elev:
            return fid
        for nid in adjacency.get(fid, []):
            if nid not in seen:
                seen.add(nid)
                queue.append(nid)

    return None


def find_drainage_basins(
    adjacency: Dict[str, List[str]],
    store: TileDataStore,
    field_name: str = "elevation",
) -> Dict[str, str]:
    """For every face, find which local minimum it drains to.

    Returns ``{face_id: basin_id}`` where *basin_id* is the face id
    of the local minimum.
    """
    # Cache steepest-descent terminus for each face
    result: Dict[str, str] = {}

    for fid in adjacency:
        if fid in result:
            continue
        path = steepest_descent_path(adjacency, store, fid, field_name, max_steps=len(adjacency))
        terminus = path[-1]
        # All faces on this path drain to the same terminus
        for pid in path:
            if pid not in result:
                result[pid] = terminus

    return result


def fill_depressions(
    grid: PolyGrid,
    store: TileDataStore,
    adjacency: Dict[str, List[str]],
    field_name: str = "elevation",
    *,
    boundary_face_ids: Optional[Set[str]] = None,
) -> int:
    """Fill interior depressions (sinks) so all water can drain outward.

    Raises the elevation of interior local minima so that water can
    flow to the grid boundary.  This is a simplified priority-flood
    approach.

    Parameters
    ----------
    grid : PolyGrid
    store : TileDataStore
    adjacency : dict
    field_name : str
    boundary_face_ids : set, optional
        If not given, boundary faces are detected automatically
        (faces with fewer than the maximum number of neighbours).

    Returns
    -------
    int
        Number of faces whose elevation was raised.
    """
    if boundary_face_ids is None:
        # Heuristic: boundary faces have fewer neighbours
        max_nbrs = max(len(v) for v in adjacency.values()) if adjacency else 0
        boundary_face_ids = {
            fid for fid, nbrs in adjacency.items()
            if len(nbrs) < max_nbrs
        }

    # Priority flood from boundary inward
    import heapq

    filled: Dict[str, float] = {}
    visited: Set[str] = set()
    heap: List[Tuple[float, str]] = []

    for fid in boundary_face_ids:
        elev = store.get(fid, field_name)
        heapq.heappush(heap, (elev, fid))
        visited.add(fid)
        filled[fid] = elev

    changes = 0

    while heap:
        w_elev, fid = heapq.heappop(heap)
        for nid in adjacency.get(fid, []):
            if nid in visited:
                continue
            visited.add(nid)
            ne = store.get(nid, field_name)
            # If neighbour is lower than the water surface, raise it
            new_elev = max(ne, w_elev)
            filled[nid] = new_elev
            if new_elev > ne:
                store.set(nid, field_name, new_elev)
                changes += 1
            heapq.heappush(heap, (new_elev, nid))

    return changes


def flow_accumulation(
    adjacency: Dict[str, List[str]],
    store: TileDataStore,
    field_name: str = "elevation",
) -> Dict[str, int]:
    """Compute flow accumulation — how many upstream faces drain through each face.

    Each face starts with 1 (itself).  Then faces are processed from
    highest to lowest; each face distributes its accumulated count to
    its steepest downhill neighbour.

    Returns ``{face_id: accumulation}``.
    """
    # Sort faces by descending elevation
    face_elevs: List[Tuple[float, str]] = []
    for fid in adjacency:
        face_elevs.append((store.get(fid, field_name), fid))
    face_elevs.sort(reverse=True)

    acc: Dict[str, int] = {fid: 1 for fid in adjacency}

    for _, fid in face_elevs:
        cur_elev = store.get(fid, field_name)
        # Find steepest downhill neighbour
        best_nid: Optional[str] = None
        best_elev = cur_elev
        for nid in adjacency.get(fid, []):
            ne = store.get(nid, field_name)
            if ne < best_elev:
                best_elev = ne
                best_nid = nid
        if best_nid is not None:
            acc[best_nid] += acc[fid]

    return acc


# ═══════════════════════════════════════════════════════════════════
# 7E.2 — River network construction
# ═══════════════════════════════════════════════════════════════════


@dataclass
class RiverSegment:
    """An ordered sequence of face ids forming one river stretch.

    Attributes
    ----------
    name : str
        Human-readable name (e.g. ``"river_0"``).
    face_ids : list of str
        Ordered face ids from source to mouth.
    order : int
        Strahler stream order (1 = headwater, higher = larger).
    width : float
        Representative width derived from flow accumulation.
    """

    name: str
    face_ids: List[str] = field(default_factory=list)
    order: int = 1
    width: float = 1.0


@dataclass
class RiverNetwork:
    """Collection of :class:`RiverSegment` s forming a river system.

    Attributes
    ----------
    segments : list of RiverSegment
    """

    segments: List[RiverSegment] = field(default_factory=list)

    def all_river_face_ids(self) -> Set[str]:
        """Return the set of all face ids that are part of any river."""
        result: Set[str] = set()
        for seg in self.segments:
            result.update(seg.face_ids)
        return result

    def segments_through(self, face_id: str) -> List[RiverSegment]:
        """Return all segments that pass through *face_id*."""
        return [s for s in self.segments if face_id in s.face_ids]

    def main_stem(self) -> Optional[RiverSegment]:
        """Return the segment with the highest order (longest river)."""
        if not self.segments:
            return None
        return max(self.segments, key=lambda s: (s.order, len(s.face_ids)))

    def __len__(self) -> int:
        return len(self.segments)

    def __repr__(self) -> str:
        n = len(self.segments)
        total = sum(len(s.face_ids) for s in self.segments)
        return f"RiverNetwork(segments={n}, total_faces={total})"


@dataclass
class RiverConfig:
    """Configuration for river generation.

    Attributes
    ----------
    min_accumulation : int
        Minimum flow accumulation for a face to be a river face.
    min_length : int
        Minimum number of faces in a valid river segment.
    carve_depth : float
        How much to lower river-face elevation.
    valley_width : float
        How much to lower neighbours of river faces (fraction of
        carve_depth).
    seed : int
        Random seed (for tie-breaking).
    """

    min_accumulation: int = 5
    min_length: int = 3
    carve_depth: float = 0.05
    valley_width: float = 0.3
    seed: int = 42


def generate_rivers(
    grid: PolyGrid,
    store: TileDataStore,
    config: RiverConfig,
    *,
    field_name: str = "elevation",
) -> RiverNetwork:
    """Generate a river network from existing elevation data.

    Steps:
    1. Fill depressions so drainage is continuous
    2. Compute flow accumulation
    3. Threshold: faces above ``min_accumulation`` are river faces
    4. Trace river paths from each river-head downhill
    5. Merge converging paths at confluences
    6. Assign Strahler stream order
    7. Return a :class:`RiverNetwork`

    Parameters
    ----------
    grid : PolyGrid
    store : TileDataStore
        Must have an elevation field.
    config : RiverConfig
    field_name : str
        Name of the elevation field.

    Returns
    -------
    RiverNetwork
    """
    adj = get_face_adjacency(grid)

    # 1. Fill depressions
    fill_depressions(grid, store, adj, field_name)

    # 2. Flow accumulation
    acc = flow_accumulation(adj, store, field_name)

    # 3. Identify river faces
    river_faces: Set[str] = {
        fid for fid, a in acc.items() if a >= config.min_accumulation
    }

    if not river_faces:
        return RiverNetwork()

    # 4. Find river heads — river faces whose upstream neighbours are NOT river faces
    heads: List[Tuple[int, str]] = []  # (accumulation, face_id) for sorting
    for fid in river_faces:
        # A head is a river face where no upstream neighbour is also a river face
        upstream_rivers = [
            nid for nid in adj.get(fid, [])
            if nid in river_faces and store.get(nid, field_name) > store.get(fid, field_name)
        ]
        if not upstream_rivers:
            heads.append((acc[fid], fid))

    # Also consider: high-accumulation faces at the top
    if not heads:
        # Fallback: pick the highest-elevation river faces
        heads = sorted(
            [(acc[fid], fid) for fid in river_faces],
            key=lambda x: store.get(x[1], field_name),
            reverse=True,
        )[:5]

    # 5. Trace paths downhill from each head
    used_faces: Set[str] = set()
    raw_paths: List[List[str]] = []

    # Sort heads by accumulation (highest first → main stems first)
    heads.sort(reverse=True)

    for _, head_fid in heads:
        path = _trace_river_path(adj, store, head_fid, river_faces, used_faces, field_name)
        if len(path) >= config.min_length:
            raw_paths.append(path)
            used_faces.update(path)

    # 6. Build segments with width from accumulation
    max_acc = max(acc.values()) if acc else 1
    segments: List[RiverSegment] = []
    for i, path in enumerate(raw_paths):
        max_path_acc = max(acc.get(fid, 1) for fid in path)
        width = max(0.5, 3.0 * (max_path_acc / max_acc))
        segments.append(RiverSegment(
            name=f"river_{i}",
            face_ids=path,
            order=1,
            width=width,
        ))

    # 7. Assign Strahler order (simplified: longer/higher-acc paths get higher order)
    _assign_strahler_order(segments, acc)

    return RiverNetwork(segments=segments)


def _trace_river_path(
    adj: Dict[str, List[str]],
    store: TileDataStore,
    start: str,
    river_faces: Set[str],
    used: Set[str],
    field_name: str,
) -> List[str]:
    """Trace a single river path from *start* downhill through river faces."""
    path = [start]
    visited: Set[str] = {start}

    for _ in range(10000):
        current = path[-1]
        cur_elev = store.get(current, field_name)

        # Find best downhill neighbour that is a river face (or allow exit to non-river)
        candidates: List[Tuple[float, str]] = []
        for nid in adj.get(current, []):
            if nid in visited:
                continue
            ne = store.get(nid, field_name)
            if ne <= cur_elev:
                # Prefer river faces, but allow non-river for the last step
                priority = 0 if nid in river_faces else 1
                candidates.append((priority, ne, nid))  # type: ignore[arg-type]

        if not candidates:
            break

        # Sort: prefer river faces, then lowest elevation
        candidates.sort(key=lambda c: (c[0], c[1]))
        _, _, best = candidates[0]
        visited.add(best)
        path.append(best)

        # Stop if we've left river faces
        if best not in river_faces:
            break

    return path


def _assign_strahler_order(segments: List[RiverSegment], acc: Dict[str, int]) -> None:
    """Simplified Strahler ordering: based on max accumulation in each segment."""
    if not segments:
        return
    max_acc = max(max(acc.get(fid, 1) for fid in seg.face_ids) for seg in segments)
    if max_acc <= 0:
        return
    for seg in segments:
        seg_acc = max(acc.get(fid, 1) for fid in seg.face_ids)
        # Map accumulation to order 1-5
        ratio = seg_acc / max_acc
        seg.order = max(1, min(5, int(ratio * 5) + 1))


# ═══════════════════════════════════════════════════════════════════
# 7E.3 — River ↔ terrain integration
# ═══════════════════════════════════════════════════════════════════


def carve_river_valleys(
    grid: PolyGrid,
    store: TileDataStore,
    network: RiverNetwork,
    *,
    carve_depth: float = 0.05,
    valley_width: float = 0.3,
    field_name: str = "elevation",
) -> None:
    """Lower elevation along river paths, carving valleys.

    River faces are lowered by ``carve_depth × width_factor``.
    Immediate neighbours are lowered by ``valley_width × carve_depth``.

    Parameters
    ----------
    grid : PolyGrid
    store : TileDataStore
    network : RiverNetwork
    carve_depth : float
        Base depth to carve.
    valley_width : float
        Fraction of carve_depth applied to neighbouring faces.
    field_name : str
        Elevation field name.
    """
    adj = get_face_adjacency(grid)
    river_faces = network.all_river_face_ids()

    # Build face→max segment width lookup
    face_width: Dict[str, float] = {}
    for seg in network.segments:
        for fid in seg.face_ids:
            face_width[fid] = max(face_width.get(fid, 0.0), seg.width)

    max_width = max(face_width.values()) if face_width else 1.0

    # Carve river faces
    for fid in river_faces:
        w = face_width.get(fid, 1.0)
        width_factor = w / max_width
        elev = store.get(fid, field_name)
        store.set(fid, field_name, elev - carve_depth * width_factor)

    # Carve valley walls (neighbours of river faces, not themselves rivers)
    valley_lowered: Set[str] = set()
    for fid in river_faces:
        for nid in adj.get(fid, []):
            if nid not in river_faces and nid not in valley_lowered:
                ne = store.get(nid, field_name)
                store.set(nid, field_name, ne - carve_depth * valley_width)
                valley_lowered.add(nid)


def assign_river_data(
    store: TileDataStore,
    network: RiverNetwork,
) -> None:
    """Set ``"river"`` (bool) and ``"river_width"`` (float) for river faces.

    The store's schema must include these fields.
    """
    river_faces = network.all_river_face_ids()

    # Build face→width
    face_width: Dict[str, float] = {}
    for seg in network.segments:
        for fid in seg.face_ids:
            face_width[fid] = max(face_width.get(fid, 0.0), seg.width)

    # Set all non-river faces to defaults first
    for fid in store.grid.faces:
        if fid not in river_faces:
            store.set(fid, "river", False)
            store.set(fid, "river_width", 0.0)

    # Set river faces
    for fid in river_faces:
        store.set(fid, "river", True)
        store.set(fid, "river_width", face_width.get(fid, 1.0))


def river_to_overlay(
    grid: PolyGrid,
    network: RiverNetwork,
) -> Overlay:
    """Convert a :class:`RiverNetwork` into an :class:`Overlay` for rendering.

    River faces become :class:`OverlayRegion` s coloured by stream order
    (thin tributaries = light blue, main stems = dark blue).

    Returns
    -------
    Overlay
        An overlay with ``kind="river"`` and per-face coloured regions.
    """
    overlay = Overlay(kind="river")

    # Blue colour gradient by order (1 = lightest, 5 = darkest)
    order_colors = {
        1: (0.6, 0.8, 1.0),   # light blue
        2: (0.4, 0.65, 0.9),
        3: (0.25, 0.5, 0.85),
        4: (0.15, 0.35, 0.75),
        5: (0.08, 0.2, 0.6),   # dark blue
    }

    for seg in network.segments:
        color = order_colors.get(seg.order, (0.3, 0.5, 0.9))
        for fid in seg.face_ids:
            face = grid.faces.get(fid)
            if face is None:
                continue
            pts: List[Tuple[float, float]] = []
            for vid in face.vertex_ids:
                v = grid.vertices[vid]
                if v.has_position():
                    pts.append((v.x, v.y))
            if len(pts) < 3:
                continue
            region = OverlayRegion(
                id=f"riv_{fid}",
                points=pts,
                source_vertex_id=fid,
            )
            overlay.metadata[f"color_{fid}"] = color
            overlay.regions.append(region)

    return overlay
