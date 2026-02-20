Recommended practices for pentagon–hex grids in Python
General approach that stays stable across ring counts

A practical workflow that aligns with the strongest guarantees in the literature is:

    Define the combinatorics first (cells/edges/vertices, ring count, boundary). The one‑pentagonal nanocone graph literature is a good reference model for how layering is defined and indexed.

    Compute an initial planar embedding using a barycentric (Tutte) solve (outer boundary fixed convex). Use this as an initialisation, not the final geometry.

    Improve regularity with constrained optimisation:
        penalise edge-length variance,
        penalise angle variance in faces,
        add inversion barriers.
        If the optimisation is performed on a triangulation, periodically enforce intrinsic Delaunay structure before building cotangent‑type operators.

    Choose operators/solvers based on what you compute:
        for diffusion/spectral: use well‑behaved polygonal Laplacians or intrinsic‑Delaunay cotangent Laplacians, informed by polygonal Laplacian surveys,
        for PDEs: prefer VEM on polygons (if available) or triangulate carefully with quality control; heed classic FEM distortion caveats,
        for routing/distances: always use weighted shortest paths consistent with the embedding metric.

Python libraries and “pointers that actually map to the research”

The items below are chosen because they connect directly to the primary literature cited earlier (operators, embeddings, DEC, meshing).

Graph algorithms (routing, shortest paths, spectral graph primitives).
Use SciPy sparse CSGraph routines for shortest paths and Laplacians; they explicitly support shortest‑path computations on sparse weighted graphs.

Planar graph plumbing / combinatorial embeddings.
NetworkX provides planar embedding data structures and standard layouts (useful for initialisation, diagnostics, and experimentation).

Polygonisation (build faces from edges).
Shapely can polygonise planar linework into polygons via polygonize / polygonize_full, which is useful if you generate the edge set first and want faces as polygons.

Intrinsic Delaunay cotangent Laplacian and geometry processing.
libigl exposes intrinsic Delaunay triangulation and intrinsic‑Delaunay cotangent matrices; this maps directly onto the intrinsic Delaunay line of work and is a practical way to avoid negative weight pathologies.

Discrete exterior calculus.
PyDEC is a DEC/FEEC library with an accompanying paper describing algorithms and features, and it is a natural candidate if you want DEC-style operators on your complexes.

Mesh I/O and interchange.
meshio is a Python package for reading/writing many mesh formats; it’s often the simplest way to move between your custom grid generator and external tools (Paraview, FEM codes, etc.).

Triangulation with quality control (if you must triangulate).
The Python wrapper triangle wraps Shewchuk’s Triangle, which generates Delaunay/constrained Delaunay triangulations and quality meshes, aligning with the Delaunay refinement literature.

Visualisation and inspection.
polyscope provides C++/Python visualisation for meshes/graphs; trimesh provides a Python mesh object model (triangular meshes).
Minimal code sketch: a Tutte-style barycentric embedding solve

The code below illustrates the pattern used in barycentric embeddings: solve a Laplacian system for interior vertex coordinates with the boundary pinned. This matches the “linear system” underpinning Tutte-style embeddings.

python
'''
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def tutte_barycentric_embedding(n, edges, boundary, boundary_xy):
    """
    Compute a barycentric (Tutte-style) embedding for an undirected graph.

    Parameters
    ----------
    n : int
        Number of vertices, labelled 0..n-1.
    edges : list[tuple[int,int]]
        Undirected edges.
    boundary : list[int]
        Vertex indices on the outer face, in cyclic order.
    boundary_xy : np.ndarray shape (len(boundary), 2)
        Fixed positions for boundary vertices (strictly convex polygon recommended).

    Returns
    -------
    xy : np.ndarray shape (n, 2)
        Embedded coordinates.
    """
    boundary = list(boundary)
    bset = set(boundary)
    interior = [v for v in range(n) if v not in bset]
    if len(interior) == 0:
        xy = np.zeros((n, 2), dtype=float)
        xy[boundary] = boundary_xy
        return xy

    # Build combinatorial Laplacian L = D - A
    row = []
    col = []
    data = []
    deg = np.zeros(n, dtype=int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
        # adjacency entries
        row += [u, v]
        col += [v, u]
        data += [-1.0, -1.0]
    # diagonal
    row += list(range(n))
    col += list(range(n))
    data += list(deg.astype(float))

    L = sp.csr_matrix((data, (row, col)), shape=(n, n))

    # Partition into interior (I) and boundary (B):
    # L_II x_I = - L_IB x_B
    I = np.array(interior, dtype=int)
    B = np.array(boundary, dtype=int)

    L_II = L[I][:, I]
    L_IB = L[I][:, B]

    xy = np.zeros((n, 2), dtype=float)
    xy[B] = boundary_xy

    rhs_x = -L_IB @ xy[B, 0]
    rhs_y = -L_IB @ xy[B, 1]

    # Solve sparse linear systems
    x_i = spla.spsolve(L_II, rhs_x)
    y_i = spla.spsolve(L_II, rhs_y)

    xy[I, 0] = x_i
    xy[I, 1] = y_i
    return xy
'''

Practical notes specific to “one defect + increasing rings”

If your goal is to compare algorithm behaviour as ring count grows, two pitfalls recur in the convergence literature:

    If your embedding/relaxation process changes character with ring count (e.g., produces progressively worse minimum angles), you can inadvertently create a family of meshes that violates regularity assumptions, leading to misleading scaling and, in extreme cases, loss of convergence (the FEM counterexample literature is the cautionary tale here).

    If you swap between Laplacian/operator constructions as you tweak regularity, you may be changing the mathematical object; discrete Laplacian work stresses that different operators satisfy different subsets of desired properties, so results may move because the operator changed, not because the defect/rings changed.

A robust experimental protocol is to report (and, when possible, constrain) a small set of mesh quality metrics across rings (e.g., minimum angle, max aspect ratio, distribution of dual cell areas) and to fix one operator family (polygonal Laplacian, intrinsic‑Delaunay cotan Laplacian, or a VEM/DEC discretisation with stated assumptions).