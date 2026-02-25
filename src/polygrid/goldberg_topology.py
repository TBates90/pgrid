"""Correct combinatorial topology for a Goldberg‑polyhedron face.

One central pentagon surrounded by *R* rings of hexagons, forming an
overall pentagonal shape.  Structural invariants:

    Ring 0:  1 pentagon  (5 vertices, 5 edges)
    Ring k:  5·k hexagons  (k = 1 … R)
    Total faces:  1 + 5·R·(R+1)/2
    Boundary vertices:  5·(2R+1)
    All interior vertices have degree 3
    Boundary has exactly 5 "corners"

Strategy: build a triangular grid on a cone (5-fold symmetry, apex
at the centre), then dualize.  The apex becomes the pentagon; every
other interior vertex becomes a hexagon.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .models import Edge, Face, Vertex
from .polygrid import PolyGrid


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════

def goldberg_face_count(rings: int) -> int:
    """Total number of faces (1 pentagon + hexes) for *rings* rings."""
    if rings < 0:
        raise ValueError("rings must be >= 0")
    return 1 + 5 * rings * (rings + 1) // 2


def goldberg_topology(rings: int) -> Tuple[
    Dict[str, Vertex],
    List[Edge],
    List[Face],
    List[str],
]:
    """Build the pure combinatorial graph for a Goldberg‑face grid.

    Returns ``(dual_vertices, edges, faces, corner_ids)``.
    Vertices have no positions.

    *corner_ids* is a list of 5 vertex id strings identifying the
    sector‑corner vertices on the boundary — these are the "tips"
    of the pentagonal outline.
    """
    if rings < 0:
        raise ValueError("rings must be >= 0")

    # ── 1. Build triangulation vertices by layer ────────────────────
    # Layer 0: 1 vertex (apex, becomes pentagon in dual)
    # Layer k ≥ 1: 5k vertices in a ring with 5-fold symmetry
    # We need layers 0 … rings+1.

    max_layer = rings + 1
    tri_verts: Dict[int, List[str]] = {}  # layer → list of vertex ids
    _vcnt = [0]

    def _new_tv() -> str:
        _vcnt[0] += 1
        return f"_tv{_vcnt[0]}"

    tri_verts[0] = [_new_tv()]  # apex
    for k in range(1, max_layer + 1):
        tri_verts[k] = [_new_tv() for _ in range(5 * k)]

    def tv(layer: int, idx: int) -> str:
        """Triangulation vertex with automatic index wrapping."""
        if layer == 0:
            return tri_verts[0][0]
        return tri_verts[layer][idx % (5 * layer)]

    # ── 2. Enumerate triangles ──────────────────────────────────────
    #
    # Between layer 0 (apex) and layer 1 (5 verts): 5 triangles.
    #
    # Between layer k and k+1 (k ≥ 1):
    #   Layer k has 5k verts, layer k+1 has 5(k+1) verts.
    #   Within sector s (0…4):
    #     inner positions in flat ring: s*k … s*k + (k-1)   [k verts]
    #     outer positions in flat ring: s*(k+1) … s*(k+1)+k  [k+1 verts]
    #   Per sector we produce:
    #     1 "leading down" connecting inner start to previous outer end
    #     k "down" triangles (inner vertex, 2 outer)
    #     k-1 "up" triangles (2 inner, 1 outer) between consecutive inner
    #     1 "bridging up" at sector end
    #   = 2k+1 per sector, ×5 = 10k+5 per layer pair.

    triangles: List[Tuple[str, str, str]] = []
    # Track the 5 sector-corner triangle indices (leading-down at outermost layer)
    _corner_tri_indices: List[int] = []

    # Apex → layer 1
    for i in range(5):
        triangles.append((tv(0, 0), tv(1, i), tv(1, i + 1)))
        # For rings==0 these ARE the boundary; for rings==1 they happen to be
        # the same as the leading-down triangles at k=1 only if max_layer==1,
        # but we handle that below.

    # Layer k → k+1
    for k in range(1, max_layer):
        for s in range(5):
            i0 = s * k          # first inner index for sector s
            o0 = s * (k + 1)    # first outer index for sector s

            # Leading "down" triangle at sector start
            # Connects inner[i0] to outer[o0-1] (last of prev sector) and outer[o0]
            tidx_leading = len(triangles)
            triangles.append((tv(k, i0), tv(k + 1, o0 - 1), tv(k + 1, o0)))

            # Tag the 5 leading-down triangles of the outermost layer pair
            if k == max_layer - 1:
                _corner_tri_indices.append(tidx_leading)

            # Zigzag within sector
            for j in range(k):
                # "Down" triangle: 1 inner + 2 outer
                triangles.append((
                    tv(k, i0 + j),
                    tv(k + 1, o0 + j),
                    tv(k + 1, o0 + j + 1),
                ))
                # "Up" triangle: 2 inner + 1 outer (skip after last inner)
                if j < k - 1:
                    triangles.append((
                        tv(k, i0 + j),
                        tv(k, i0 + j + 1),
                        tv(k + 1, o0 + j + 1),
                    ))

            # Bridging "up" triangle at sector end
            triangles.append((
                tv(k, i0 + k - 1),
                tv(k, i0 + k),  # first inner of next sector (wraps)
                tv(k + 1, o0 + k),
            ))

    # Sanity check
    if rings == 0:
        assert len(triangles) == 5
    else:
        expected = 5 + sum(10 * k + 5 for k in range(1, max_layer))
        assert len(triangles) == expected, f"{len(triangles)} != {expected}"

    # ── 3. Dualize ──────────────────────────────────────────────────
    # Each triangle → one dual vertex.
    # Each interior triangulation vertex → one dual face.
    #   apex → pentagon (5 incident triangles)
    #   layer 1…rings verts → hexagon (6 incident triangles each)
    # Layer rings+1 = boundary, not dualized.

    # Build incidence: tri-vertex → [triangle indices]
    all_tri_vids: set[str] = set()
    for layer_vids in tri_verts.values():
        all_tri_vids.update(layer_vids)
    incident: Dict[str, List[int]] = {v: [] for v in all_tri_vids}
    for tidx, tri in enumerate(triangles):
        for v in tri:
            incident[v].append(tidx)

    # Order fan of triangles around a vertex into a proper cycle
    def _order_fan(v: str, fan: List[int]) -> List[int]:
        if len(fan) <= 2:
            return fan
        adj: Dict[int, List[int]] = {t: [] for t in fan}
        for i in range(len(fan)):
            si = set(triangles[fan[i]]) - {v}
            for j in range(i + 1, len(fan)):
                sj = set(triangles[fan[j]]) - {v}
                if si & sj:
                    adj[fan[i]].append(fan[j])
                    adj[fan[j]].append(fan[i])
        ordered = [fan[0]]
        visited = {fan[0]}
        for _ in range(len(fan) - 1):
            cur = ordered[-1]
            for nb in adj[cur]:
                if nb not in visited:
                    ordered.append(nb)
                    visited.add(nb)
                    break
        return ordered

    # Dual vertices
    dual_vertices: Dict[str, Vertex] = {}
    tri_to_dvid: Dict[int, str] = {}
    _dcnt = [0]
    for tidx in range(len(triangles)):
        _dcnt[0] += 1
        dvid = f"v{_dcnt[0]}"
        tri_to_dvid[tidx] = dvid
        dual_vertices[dvid] = Vertex(dvid)

    # Dual faces
    dual_faces: List[Face] = []
    _fcnt = [0]
    apex_vid = tri_verts[0][0]

    for layer_idx in range(0, rings + 1):
        for vid in tri_verts[layer_idx]:
            fan = incident[vid]
            expected = 5 if vid == apex_vid else 6
            if len(fan) != expected:
                continue
            ordered = _order_fan(vid, fan)
            if len(ordered) != expected:
                continue
            _fcnt[0] += 1
            dual_faces.append(Face(
                id=f"f{_fcnt[0]}",
                face_type="pent" if vid == apex_vid else "hex",
                vertex_ids=tuple(tri_to_dvid[t] for t in ordered),
            ))

    # ── 4. Build edges from faces ───────────────────────────────────
    edges = _edges_from_faces(dual_faces)

    # ── 5. Record the 5 sector‑corner vertex ids ────────────────────
    # For rings >= 1 these are the dual vertices of the leading-down
    # triangles at the outermost layer pair.  For rings == 0 there is
    # no boundary (single pentagon), so corners = pentagon vertices.
    if rings == 0:
        corner_vids = list(dual_faces[0].vertex_ids)
    else:
        corner_vids = [tri_to_dvid[ti] for ti in _corner_tri_indices]

    return dual_vertices, edges, dual_faces, corner_vids


def fix_face_winding(
    vertices: Dict[str, Vertex],
    faces: List[Face],
) -> List[Face]:
    """Ensure all faces have CCW winding (positive signed area).

    Call this after embedding so that vertex positions are available.
    Returns a new face list with vertex_ids reversed where needed.
    """
    fixed: List[Face] = []
    for face in faces:
        vids = face.vertex_ids
        n = len(vids)
        area = 0.0
        for i in range(n):
            v1 = vertices[vids[i]]
            v2 = vertices[vids[(i + 1) % n]]
            area += v1.x * v2.y - v2.x * v1.y
        area /= 2.0
        if area < 0:
            # Reverse winding to make CCW
            fixed.append(Face(
                id=face.id,
                face_type=face.face_type,
                vertex_ids=tuple(reversed(vids)),
                edge_ids=face.edge_ids,
                neighbor_ids=face.neighbor_ids,
            ))
        else:
            fixed.append(face)
    return fixed


# ═══════════════════════════════════════════════════════════════════
# Embedding: Tutte + optimise
# ═══════════════════════════════════════════════════════════════════

def goldberg_embed_tutte(
    vertices: Dict[str, Vertex],
    edges: List[Edge],
    faces: List[Face],
    rings: int,
    size: float = 1.0,
) -> Dict[str, Vertex]:
    """Tutte embedding with pentagonal outer boundary.

    Pins boundary vertices to a convex pentagon, then solves the
    Laplacian system for interior positions.  Guaranteed crossing-free
    for 3-connected planar graphs with convex boundary.
    """
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    # Boundary = vertices incident to edges with only 1 face
    boundary_nbrs: Dict[str, List[str]] = {}
    for e in edges:
        if len(e.face_ids) < 2:
            a, b = e.vertex_ids
            boundary_nbrs.setdefault(a, []).append(b)
            boundary_nbrs.setdefault(b, []).append(a)

    if not boundary_nbrs:
        return vertices

    # Walk boundary cycle
    cycle: List[str] = [sorted(boundary_nbrs.keys())[0]]
    visited = {cycle[0]}
    while True:
        cur = cycle[-1]
        advanced = False
        for nb in boundary_nbrs.get(cur, []):
            if nb not in visited:
                cycle.append(nb)
                visited.add(nb)
                advanced = True
                break
        if not advanced:
            break

    n_boundary = len(cycle)

    # Pin boundary to regular pentagon with subdivided edges
    outer_radius = size * (rings + 1) * 0.95
    verts_per_edge = n_boundary // 5 if n_boundary >= 5 else max(n_boundary, 1)

    corners = []
    for i in range(5):
        angle = math.pi / 2 + 2 * math.pi * i / 5
        corners.append((outer_radius * math.cos(angle), outer_radius * math.sin(angle)))

    boundary_pos: Dict[str, Tuple[float, float]] = {}
    for i, vid in enumerate(cycle):
        edge_idx = min(i // verts_per_edge, 4)
        local_t = (i - edge_idx * verts_per_edge) / max(verts_per_edge, 1)
        c0 = corners[edge_idx]
        c1 = corners[(edge_idx + 1) % 5]
        x = c0[0] + local_t * (c1[0] - c0[0])
        y = c0[1] + local_t * (c1[1] - c0[1])
        boundary_pos[vid] = (x, y)

    # Tutte solve
    all_vids = list(vertices.keys())
    idx = {vid: i for i, vid in enumerate(all_vids)}
    n = len(all_vids)
    fixed = set(boundary_pos.keys())
    interior = [vid for vid in all_vids if vid not in fixed]

    xy = np.zeros((n, 2))
    for vid, (bx, by) in boundary_pos.items():
        xy[idx[vid]] = (bx, by)

    rows, cols, data = [], [], []
    deg = np.zeros(n)
    for e in edges:
        a, b = e.vertex_ids
        ia, ib = idx[a], idx[b]
        deg[ia] += 1
        deg[ib] += 1
        rows += [ia, ib]
        cols += [ib, ia]
        data += [-1.0, -1.0]
    rows += list(range(n))
    cols += list(range(n))
    data += list(deg)

    L = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    if not interior:
        return {vid: Vertex(vid, float(xy[idx[vid], 0]), float(xy[idx[vid], 1])) for vid in all_vids}

    I_idx = np.array([idx[vid] for vid in interior])
    B_idx = np.array([idx[vid] for vid in boundary_pos])

    L_II = L[I_idx][:, I_idx]
    L_IB = L[I_idx][:, B_idx]

    x_int = spla.spsolve(L_II, -L_IB @ xy[B_idx, 0])
    y_int = spla.spsolve(L_II, -L_IB @ xy[B_idx, 1])
    xy[I_idx, 0] = x_int
    xy[I_idx, 1] = y_int

    return {vid: Vertex(vid, float(xy[idx[vid], 0]), float(xy[idx[vid], 1])) for vid in all_vids}


def goldberg_optimise(
    vertices: Dict[str, Vertex],
    edges: List[Edge],
    faces: List[Face],
    rings: int,
    corner_ids: List[str] | None = None,
    iterations: int = 600,
) -> Dict[str, Vertex]:
    """Optimise vertex positions for edge-length and angle regularity.

    Uses scipy least_squares with fully vectorised numpy residuals.
    Penalises edge-length deviation, angle deviation, and area inversion.

    Only the 5 sector-corner vertices (from *corner_ids*) and the
    pentagon vertices are held fixed, so that boundary hexagons can
    relax outward into their natural shape.
    """
    from scipy.optimize import least_squares
    import numpy as np

    # Identify fixed vertices: 5 corners + pentagon
    fixed_vids: set[str] = set()
    if corner_ids:
        fixed_vids.update(corner_ids)
    pent = next((f for f in faces if f.face_type == "pent"), None)
    if pent:
        fixed_vids.update(pent.vertex_ids)

    all_vids = sorted(vertices.keys())
    vid_to_i = {vid: i for i, vid in enumerate(all_vids)}
    n_all = len(all_vids)

    movable = [vid for vid in all_vids if vid not in fixed_vids]
    if not movable:
        return vertices
    mov_set = set(movable)
    mov_to_j = {vid: j for j, vid in enumerate(movable)}

    # All positions as flat array: [x0, y0, x1, y1, ...]
    all_xy = np.zeros(n_all * 2)
    for vid in all_vids:
        i = vid_to_i[vid]
        all_xy[2 * i] = vertices[vid].x
        all_xy[2 * i + 1] = vertices[vid].y

    # Movable subset indices into all_xy
    n_mov = len(movable)
    x0 = np.zeros(n_mov * 2)
    for j, vid in enumerate(movable):
        i = vid_to_i[vid]
        x0[2 * j] = all_xy[2 * i]
        x0[2 * j + 1] = all_xy[2 * i + 1]

    # Pre-build scatter index: for each movable vertex, where it sits in all_xy
    scatter_x = np.array([2 * vid_to_i[vid] for vid in movable])
    scatter_y = scatter_x + 1

    # Pre-build edge index arrays
    edge_a = np.array([vid_to_i[e.vertex_ids[0]] for e in edges])
    edge_b = np.array([vid_to_i[e.vertex_ids[1]] for e in edges])
    n_edges = len(edges)

    # Target edge length
    dx0 = all_xy[2 * edge_b] - all_xy[2 * edge_a]
    dy0 = all_xy[2 * edge_b + 1] - all_xy[2 * edge_a + 1]
    target_len = float(np.mean(np.hypot(dx0, dy0))) or 1.0

    # Pre-build angle index arrays: (prev, center, next) per face vertex
    pent_target = math.radians(108.0)
    hex_target = math.radians(120.0)
    angle_prev = []
    angle_cent = []
    angle_next = []
    angle_target = []
    for face in faces:
        vids = face.vertex_ids
        nv = len(vids)
        tgt = pent_target if face.face_type == "pent" else hex_target
        for k in range(nv):
            angle_prev.append(vid_to_i[vids[(k - 1) % nv]])
            angle_cent.append(vid_to_i[vids[k]])
            angle_next.append(vid_to_i[vids[(k + 1) % nv]])
            angle_target.append(tgt)
    angle_prev = np.array(angle_prev)
    angle_cent = np.array(angle_cent)
    angle_next = np.array(angle_next)
    angle_target = np.array(angle_target)
    n_angles = len(angle_target)

    # Pre-build area arrays: for each face, list vertex indices in order
    face_starts = []  # start index into flat area arrays
    face_lengths = []
    area_vid_i = []
    for face in faces:
        face_starts.append(len(area_vid_i))
        face_lengths.append(len(face.vertex_ids))
        for vid in face.vertex_ids:
            area_vid_i.append(vid_to_i[vid])
    face_starts = np.array(face_starts)
    face_lengths = np.array(face_lengths)
    area_vid_i = np.array(area_vid_i)
    n_faces = len(faces)

    aw = 0.5  # angle weight
    bw = 10.0  # barrier weight

    # Pre-build padded face vertex arrays for vectorised area computation
    max_nv = max(face_lengths)
    # Pad each face's vertex indices to max_nv using the first vertex (creates zero-area extra triangles)
    face_vi_padded = np.zeros((n_faces, max_nv), dtype=np.intp)
    for fi in range(n_faces):
        s = face_starts[fi]
        nv = face_lengths[fi]
        face_vi_padded[fi, :nv] = area_vid_i[s:s + nv]
        # Pad remaining with first vertex (produces zero contribution)
        if nv < max_nv:
            face_vi_padded[fi, nv:] = area_vid_i[s]

    def _residuals(x):
        # Scatter movable back into full array
        xy = all_xy.copy()
        xy[scatter_x] = x[0::2]
        xy[scatter_y] = x[1::2]

        # --- Edge length residuals ---
        dxe = xy[2 * edge_b] - xy[2 * edge_a]
        dye = xy[2 * edge_b + 1] - xy[2 * edge_a + 1]
        dist = np.hypot(dxe, dye)
        dist = np.maximum(dist, 1e-12)
        edge_res = (dist - target_len) / target_len  # (n_edges,)

        # --- Angle residuals ---
        upx = xy[2 * angle_prev] - xy[2 * angle_cent]
        upy = xy[2 * angle_prev + 1] - xy[2 * angle_cent + 1]
        vpx = xy[2 * angle_next] - xy[2 * angle_cent]
        vpy = xy[2 * angle_next + 1] - xy[2 * angle_cent + 1]
        u_len = np.hypot(upx, upy)
        v_len = np.hypot(vpx, vpy)
        denom = np.maximum(u_len * v_len, 1e-12)
        dot = np.clip((upx * vpx + upy * vpy) / denom, -1.0, 1.0)
        angles = np.arccos(dot)
        angle_res = (angles - angle_target) * aw  # (n_angles,)

        # --- Area barrier residuals (fully vectorised) ---
        # face_vi_padded: (n_faces, max_nv)  indices into all_vids
        fx = xy[2 * face_vi_padded]          # (n_faces, max_nv)
        fy = xy[2 * face_vi_padded + 1]      # (n_faces, max_nv)
        fx2 = np.roll(fx, -1, axis=1)
        fy2 = np.roll(fy, -1, axis=1)
        cross_sum = np.sum(fx * fy2 - fx2 * fy, axis=1) / 2.0  # (n_faces,)
        area_res = np.where(cross_sum < 1e-4, (1e-4 - cross_sum) * bw, 0.0)

        return np.concatenate([edge_res, angle_res, area_res])

    result = least_squares(_residuals, x0, max_nfev=iterations, method="trf")

    # Scatter final positions back
    xy = all_xy.copy()
    xy[scatter_x] = result.x[0::2]
    xy[scatter_y] = result.x[1::2]

    return {
        vid: Vertex(vid, float(xy[2 * vid_to_i[vid]]), float(xy[2 * vid_to_i[vid] + 1]))
        for vid in all_vids
    }


# ═══════════════════════════════════════════════════════════════════
# High-level builder
# ═══════════════════════════════════════════════════════════════════

def build_goldberg_grid(
    rings: int,
    size: float = 1.0,
    optimise: bool = True,
) -> PolyGrid:
    """Build a complete pentagon-centred Goldberg grid.

    Returns a fully embedded, validated PolyGrid.
    """
    verts, edges, faces, corner_ids = goldberg_topology(rings)

    if rings == 0:
        # Simple regular pentagon
        radius = size / (2 * math.sin(math.pi / 5))
        pent = faces[0]
        positioned = {}
        for i, vid in enumerate(pent.vertex_ids):
            angle = math.pi / 2 + 2 * math.pi * i / 5
            positioned[vid] = Vertex(vid, radius * math.cos(angle), radius * math.sin(angle))
        faces = fix_face_winding(positioned, faces)
        edges = _edges_from_faces(faces)
        return PolyGrid(
            positioned.values(), edges, faces,
            metadata={"generator": "goldberg", "rings": rings},
        )

    # Tutte embed → fix winding → optimise → fix winding again
    positioned = goldberg_embed_tutte(verts, edges, faces, rings, size)
    faces = fix_face_winding(positioned, faces)
    edges = _edges_from_faces(faces)

    if optimise:
        positioned = goldberg_optimise(
            positioned, edges, faces, rings,
            corner_ids=corner_ids,
        )
        faces = fix_face_winding(positioned, faces)
        edges = _edges_from_faces(faces)

    return PolyGrid(
        positioned.values(), edges, faces,
        metadata={"generator": "goldberg", "rings": rings},
    )


# ═══════════════════════════════════════════════════════════════════
# Edge helpers
# ═══════════════════════════════════════════════════════════════════

def _edges_from_faces(faces: List[Face]) -> List[Edge]:
    """Build edge list from face vertex cycles, with face back-references."""
    edge_map: Dict[Tuple[str, str], List[str]] = {}
    for face in faces:
        vids = face.vertex_ids
        n = len(vids)
        for i in range(n):
            a, b = vids[i], vids[(i + 1) % n]
            key = (min(a, b), max(a, b))
            if key not in edge_map:
                edge_map[key] = []
            edge_map[key].append(face.id)

    edges: List[Edge] = []
    for eidx, ((a, b), fids) in enumerate(sorted(edge_map.items()), 1):
        edges.append(Edge(id=f"e{eidx}", vertex_ids=(a, b), face_ids=tuple(fids)))
    return edges
