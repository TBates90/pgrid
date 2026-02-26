"""Composite grid assembly — stitching polygrids along macro-edges."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

from .models import Edge, Face, MacroEdge, Vertex
from .polygrid import PolyGrid


@dataclass(frozen=True)
class StitchSpec:
    """Specifies that *grid_a*'s macro-edge *edge_a* joins to
    *grid_b*'s macro-edge *edge_b*.

    The boundary vertices of the two macro-edges are merged pair-wise.
    By default edge_b is traversed in reverse (the natural orientation
    when two polygons share a side).  Set *flip* to ``False`` to keep
    both edges in the same direction.
    """

    grid_a: str
    edge_a: int
    grid_b: str
    edge_b: int
    flip: bool = True


@dataclass(frozen=True)
class CompositeGrid:
    """Composite container: merged PolyGrid + component book-keeping."""

    merged: PolyGrid
    components: Dict[str, PolyGrid]
    id_prefixes: Dict[str, str]


def stitch_grids(
    grids: Dict[str, PolyGrid],
    stitches: List[StitchSpec],
    id_prefixes: Dict[str, str] | None = None,
) -> CompositeGrid:
    """Merge several polygrids and stitch them along macro-edges.

    Each grid must have its macro-edges computed (call
    ``grid.compute_macro_edges()`` first).

    The stitching process:
    1. Prefix all ids so they don't collide.
    2. For each :class:`StitchSpec`, identify the matching boundary
       vertices on both sides and unify them (vertex merging).
    3. Remove duplicate edges / update face references accordingly.

    Returns a :class:`CompositeGrid` whose ``.merged`` is the unified grid.
    """
    prefixes = id_prefixes or {name: f"{name}_" for name in grids}

    # ── 1. Collect prefixed primitives ──────────────────────────────
    merged_vertices: Dict[str, Vertex] = {}
    merged_edges: Dict[str, Edge] = {}
    merged_faces: Dict[str, Face] = {}

    # Keep track of original→prefixed vertex mapping per grid
    v_maps: Dict[str, Dict[str, str]] = {}

    for name, grid in grids.items():
        prefix = prefixes[name]
        v_map = {vid: f"{prefix}{vid}" for vid in grid.vertices}
        v_maps[name] = v_map

        for v in grid.vertices.values():
            new_id = v_map[v.id]
            merged_vertices[new_id] = Vertex(new_id, v.x, v.y)

        for edge in grid.edges.values():
            new_id = f"{prefix}{edge.id}"
            merged_edges[new_id] = Edge(
                id=new_id,
                vertex_ids=(v_map[edge.vertex_ids[0]], v_map[edge.vertex_ids[1]]),
                face_ids=tuple(f"{prefix}{fid}" for fid in edge.face_ids),
            )

        for face in grid.faces.values():
            new_id = f"{prefix}{face.id}"
            merged_faces[new_id] = Face(
                id=new_id,
                face_type=face.face_type,
                vertex_ids=tuple(v_map[vid] for vid in face.vertex_ids),
                edge_ids=tuple(f"{prefix}{eid}" for eid in face.edge_ids),
                neighbor_ids=tuple(f"{prefix}{nid}" for nid in face.neighbor_ids),
            )

    # ── 2. Build vertex merge map from stitch specs ─────────────────
    # merge_map: old_vid → canonical_vid  (many-to-one)
    merge_map: Dict[str, str] = {}

    for spec in stitches:
        me_a = _get_macro_edge(grids[spec.grid_a], spec.edge_a)
        me_b = _get_macro_edge(grids[spec.grid_b], spec.edge_b)

        vids_a = [v_maps[spec.grid_a][vid] for vid in me_a.vertex_ids]
        vids_b = [v_maps[spec.grid_b][vid] for vid in me_b.vertex_ids]

        if spec.flip:
            vids_b = list(reversed(vids_b))

        if len(vids_a) != len(vids_b):
            raise ValueError(
                f"Macro-edge length mismatch: {spec.grid_a}.edge{spec.edge_a} "
                f"has {len(vids_a)} vertices but {spec.grid_b}.edge{spec.edge_b} "
                f"has {len(vids_b)} vertices"
            )

        for va, vb in zip(vids_a, vids_b):
            # Both map to the 'a' vertex as canonical
            canonical = _canonical(merge_map, va)
            target = _canonical(merge_map, vb)
            if canonical != target:
                merge_map[target] = canonical

    # ── 3. Apply merge map to all primitives ────────────────────────
    def remap(vid: str) -> str:
        return _canonical(merge_map, vid)

    # Rebuild vertices (drop merged duplicates)
    final_vertices: Dict[str, Vertex] = {}
    for vid, v in merged_vertices.items():
        canon = remap(vid)
        if canon not in final_vertices:
            final_vertices[canon] = Vertex(canon, v.x, v.y)

    # Rebuild edges, deduplicating by vertex-pair
    edge_by_pair: Dict[Tuple[str, str], Edge] = {}
    for edge in merged_edges.values():
        a = remap(edge.vertex_ids[0])
        b = remap(edge.vertex_ids[1])
        if a == b:
            continue  # degenerate
        key = (min(a, b), max(a, b))
        fids = tuple(edge.face_ids)
        if key in edge_by_pair:
            existing = edge_by_pair[key]
            # Merge face references
            all_fids = existing.face_ids + fids
            seen: set[str] = set()
            unique_fids: list[str] = []
            for f in all_fids:
                if f not in seen:
                    seen.add(f)
                    unique_fids.append(f)
            edge_by_pair[key] = Edge(existing.id, existing.vertex_ids, tuple(unique_fids))
        else:
            edge_by_pair[key] = Edge(edge.id, key, fids)

    final_edges = {e.id: e for e in edge_by_pair.values()}

    # Build edge-id remap (old edge id → new edge id after dedup)
    old_edge_to_new: Dict[str, str] = {}
    for edge in merged_edges.values():
        a = remap(edge.vertex_ids[0])
        b = remap(edge.vertex_ids[1])
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        new_edge = edge_by_pair.get(key)
        if new_edge:
            old_edge_to_new[edge.id] = new_edge.id

    # Rebuild faces with remapped vertex and edge ids
    final_faces: Dict[str, Face] = {}
    for face in merged_faces.values():
        new_vids = tuple(remap(vid) for vid in face.vertex_ids)
        new_eids = tuple(
            old_edge_to_new.get(eid, eid)
            for eid in face.edge_ids
            if eid in old_edge_to_new or eid in final_edges
        )
        final_faces[face.id] = Face(
            id=face.id,
            face_type=face.face_type,
            vertex_ids=new_vids,
            edge_ids=new_eids,
            neighbor_ids=face.neighbor_ids,
        )

    merged = PolyGrid(
        final_vertices.values(),
        final_edges.values(),
        final_faces.values(),
        metadata={"generator": "composite", "components": list(grids.keys())},
    )

    return CompositeGrid(merged=merged, components=grids, id_prefixes=prefixes)


def join_grids(
    grids: Dict[str, PolyGrid],
    id_prefixes: Dict[str, str] | None = None,
) -> CompositeGrid:
    """Join grids without stitching (simple prefix merge)."""
    return stitch_grids(grids, stitches=[], id_prefixes=id_prefixes)


def split_composite(composite: CompositeGrid) -> Dict[str, PolyGrid]:
    """Return original component grids."""
    return composite.components


# ═══════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════

def _get_macro_edge(grid: PolyGrid, edge_id: int) -> MacroEdge:
    for me in grid.macro_edges:
        if me.id == edge_id:
            return me
    raise ValueError(
        f"Grid has no macro-edge {edge_id}. "
        f"Available: {[me.id for me in grid.macro_edges]}. "
        f"Call grid.compute_macro_edges() first."
    )


def _canonical(merge_map: Dict[str, str], vid: str) -> str:
    """Follow the merge chain to find the canonical vertex id."""
    visited: set[str] = set()
    while vid in merge_map:
        if vid in visited:
            break
        visited.add(vid)
        vid = merge_map[vid]
    return vid
