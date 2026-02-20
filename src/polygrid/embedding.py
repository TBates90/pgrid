from __future__ import annotations

from typing import Dict, Iterable, List

from .models import Edge, Vertex


def tutte_embedding(
    vertices: Dict[str, Vertex],
    edges: Iterable[Edge],
    fixed_positions: Dict[str, tuple[float, float]],
) -> Dict[str, Vertex]:
    """Compute a Tutte embedding with fixed vertex positions."""
    try:
        import numpy as np
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Tutte embedding requires numpy and scipy. Install with `pip install numpy scipy`."
        ) from exc

    vertex_ids = list(vertices.keys())
    index = {vid: i for i, vid in enumerate(vertex_ids)}
    n = len(vertex_ids)

    fixed_set = set(fixed_positions.keys())
    interior_ids = [vid for vid in vertex_ids if vid not in fixed_set]

    xy = np.zeros((n, 2), dtype=float)
    for vid, (x, y) in fixed_positions.items():
        xy[index[vid]] = (x, y)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    deg = np.zeros(n, dtype=int)

    for edge in edges:
        a, b = edge.vertex_ids
        ia = index[a]
        ib = index[b]
        deg[ia] += 1
        deg[ib] += 1
        rows += [ia, ib]
        cols += [ib, ia]
        data += [-1.0, -1.0]

    rows += list(range(n))
    cols += list(range(n))
    data += list(deg.astype(float))

    L = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    if not interior_ids:
        return {vid: vertices[vid] for vid in vertex_ids}

    I = np.array([index[vid] for vid in interior_ids], dtype=int)
    B = np.array([index[vid] for vid in fixed_positions.keys()], dtype=int)

    L_II = L[I][:, I]
    L_IB = L[I][:, B]

    rhs_x = -L_IB @ xy[B, 0]
    rhs_y = -L_IB @ xy[B, 1]

    x_i = spla.spsolve(L_II, rhs_x)
    y_i = spla.spsolve(L_II, rhs_y)

    xy[I, 0] = x_i
    xy[I, 1] = y_i

    embedded: Dict[str, Vertex] = {}
    for vid in vertex_ids:
        i = index[vid]
        embedded[vid] = Vertex(vid, float(xy[i, 0]), float(xy[i, 1]))

    return embedded
