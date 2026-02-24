Pentagon‑centred hex grid in TBates90/pgrid: diagnosis and task plan to stabilise the geometry
Problem framing and non‑negotiable constraints

Your target object is a hexagonal lattice with a single 5‑valent topological defect (a centred pentagon) and an increasing number of surrounding rings. The repository already describes this correctly as a “topology‑first” approach, and explicitly supports “pentagon‑centered grids” and “pure hex grids”.

Two constraints dominate everything that follows:

A perfect Euclidean (flat) hex lattice cannot contain a regular pentagon without distortion: the pentagon introduces a positive angular defect (curvature). When a regular pentagon (108° interior angle) meets two regular hexagons (120° each) at a vertex, the angle sum is 348°, leaving a 12° defect.
So “irregularity near the centre, more regular far away” is not a bug; it is the mathematically expected behaviour.

If you want a stable planar drawing across ring counts, you need a pipeline that (a) fixes combinatorics/topology first, then (b) computes a globally consistent embedding (vertex positions) under constraints. This exact pipeline is already recommended in the repo’s research notes: define combinatorics → compute a Tutte/barycentric embedding as an initialisation → improve regularity with constrained optimisation and barriers.
What the repository already has that is worth preserving

The codebase contains a strong “topology-first” skeleton and multiple embedding/diagnostic building blocks:

The CLI explicitly supports:

    building pure hex grids (build-hex)
    building pentagon-centred grids (build-pent)
    choosing embedding modes: tutte, angle, or no embedding.

The pentagon-centred builder is based on a well-known defect construction:

    build a triangular lattice
    remove a 60° wedge (disclination)
    merge boundary rays
    dualise the triangulation into pent+hex faces.
    This matches the internal “requirements” narrative and is the right starting point for a Goldberg-like “one pentagon” patch in 2D.

There is already a Tutte embedding implementation (SciPy sparse solve) and a “fixed positions” strategy (boundary + pent vertices) that can act as the stable baseline.

There is already an angle/length spec model for ring propagation (RingAngleSpec, ring_angle_spec) and a solver that tries to find symmetric hex edge lengths that close given fixed angles via least squares.

Diagnostics exist (ring-by-ring metrics for protruding lengths and angles), and there is at least one script for angle diagnostics.

The repo even has a “Task List” that matches what you need: robust topology, Tutte baseline, and quality validation across ring counts.
What the screenshots imply, and the most likely failure modes in the current implementation

From the three example renders you provided (increasing ring sizes), the dominant symptom is self-intersection / overlap near the centre that gets worse as rings increase. That typically comes from one (or a combination) of these technical issues:

The embedding step is not producing a globally consistent planar layout. The “angle-first” approach is explicitly implemented as an “exact layout” that avoids later relaxation (“avoid post-relaxation that can distort the target ring angles”).
That design choice is risky: for multi-face meshes, local exact constructions almost always accumulate error unless you do a global consistency solve (shared vertices must satisfy constraints from multiple incident faces simultaneously).

Your topology may be planar, but a Tutte embedding is only guaranteed crossing-free (and faces convex) under sufficient connectivity assumptions (notably, simple 3-vertex-connected planar graphs) and a convex fixed outer face. Tutte’s spring theorem is the standard reference for this guarantee.
If the graph is not 3-connected (quite plausible with wedge merge artefacts, boundary structure, and dualisation), a pure Tutte solve can collapse parts of the graph or produce degeneracies; that’s a known limitation of barycentric embeddings in lower connectivity.

Even if topology is fine, polygons will render as self-intersecting shapes if face vertex order is not a proper boundary cycle. Rendering uses face.vertex_ids “as is” to draw a polygon.
So you need a hard guarantee that Face.vertex_ids is a correct cyclic order around the face boundary after all topology operations (wedge removal, merging, dualisation, edge rebuild).

Finally, the current “angle-first” construction is doing several “heuristic” ordering steps (ring face ordering, boundary edge ordering, selecting CW vs CCW candidate polygons, etc.).
Those heuristics can work for ring=1 and then fail catastrophically for larger rings because the boundary becomes less “circle-like” due to the pentagonal defect.
Recommended technical direction that will scale with ring count
Make Tutte embedding the stable baseline, and treat angles/regularity as a refinement objective

The repo’s own research notes recommend: combinatorics first → Tutte embedding (outer boundary convex and pinned) → constrained optimisation that penalises edge-length and angle variance plus inversion barriers.
This is also the most robust engineering approach.

Use the “angle-first” logic as targets (soft constraints), not as the constructor of final coordinates.

That means: keep the existing ring_angle_spec() logic and the ring hex length solver as generators of desired local geometry, but move the actual placement to a global optimisation that can reconcile conflicts across shared vertices.
Ground rules for the embedding solver

Lock the following invariants before you optimise anything:

Every face has a valid cycle order (for rendering and for angle computation).

Shared edges must have consistent length targets, not “per-face” targets. The solver in angle_solver.py is currently a per-ring/per-face symmetric closure tool; you need to translate that into per-edge targets based on ring membership and adjacency.

Prevent inverted faces (negative signed area) and self-intersections. The repo’s research notes explicitly mention inversion barriers; treat that as mandatory for anything beyond ring=1.

Once you have those, you can safely “dial up” regularity objectives.
Task plan to fix the grid generator and make outputs stable
Stabilise and validate topology before touching geometry

Create a dedicated “topology validation” function for the pentagon-centred grid builder that is run by tests and optionally by CLI.

Tasks:

    Add validation checks promised in the repo’s own task list: exactly one pentagon face, all other faces are hex, interior vertex degree 3 (boundary degree 2), and expected ring face counts per ring.
    Extend Face.validate_polygon() beyond “pent must have 5 vertices, hex must have 6” to also verify:
        no repeated vertex ids in a face
        edge_ids count matches vertex_ids count when edges are present
        each face edge actually references the face id in its Edge.face_ids.
    Add a --strict or --topology-checks mode for build-pent so you can fail fast before rendering. The CLI already supports build-pent; extend it rather than adding a new entry point.

Enforce correct face boundary cycles

Right now, the renderer will happily draw a self-intersecting polygon if the vertex ordering is wrong.
So: treat correct cyclic ordering as a data model invariant.

Tasks:

    After _rebuild_edges_from_faces(...), compute a canonical cycle ordering for each face by walking its incident edges (there is already a face cycle walker pattern in the codebase). Ensure face.vertex_ids is rewritten to that cycle.
    Add an assertion: the cycle length equals the original number of unique vertices in the face, otherwise mark the face invalid and surface it in diagnostics.
    Update diagnostics to always compute angles from the cycle ordering (not from arbitrary vertex_ids order). Diagnostics already contain both a cycle method and a fallback ordering-by-angle method; make the cycle method authoritative.

Replace “angle-first exact placement” with a global constrained solve

The ring-based angle specification is logically consistent (it matches the 108° pentagon + two equal hex angles = 126° reasoning, and the 720° interior-angle sum of hexagons).
The issue is not the angles; it’s the global consistency of placement.

Make the pipeline:

    Build topology (current wedge + dualisation).
    Compute baseline embedding via Tutte with fixed boundary + fixed pent vertices.
    Run a constrained least-squares optimisation whose variables are vertex positions of non-fixed vertices. Penalise:
        edge length error (target per edge comes from ring category, not per face)
        angle error (target per face vertex comes from ring_angle_spec and whether that vertex is “inner boundary” vs “outer/pointy” in that ring)
        inversion barrier for each face (signed area should stay positive and above an epsilon)
        optionally, boundary regularisation so the outer boundary stays “pentagon-like” (fivefold symmetry constraint, if desired).

Tasks:

    Implement an “edge classification” pass to label each edge as one of:
        pent-inner (edges of central pent)
        ring radial / “protrude” (connects ring k–1 to ring k)
        ring tangential (between faces in same ring)
        outer boundary edge
        This classification should be based on face adjacency (rings) that already exist (ring_faces) plus edge face_ids.
    Turn solve_ring_hex_lengths(...) into a provider of initial target ratios that seed per-edge targets, but do not enforce them as hard equalities. (Use them to initialise weights and expected lengths.)
    Add a new embed mode tutte+optimise (or similar) and make it the default for build-pent. The README currently presents Tutte as the stable route; align the default with that.
    Keep the existing “angle-first” code path, but reclassify it as experimental/diagnostic rather than the default production embedder.

Why this direction is robust:

    Tutte gives you a stable non-degenerate starting point when the prerequisites are met; when they are not, it still often provides a useful initial state, and the optimisation stage can resolve remaining distortions.
    A single global solve ensures shared vertices satisfy all incident face constraints simultaneously (the core issue that local edge-walk constructions tend to violate as rings grow).

Convert diagnostics into “quality gates” so regressions stop immediately

You already have ring diagnostics in-library. Turn them into CI blockers.

Tasks:

    Add “no inverted faces” metric (min signed area across faces must be > epsilon).
    Add “no edge crossings” check for embedded edges (segment intersection test excluding shared endpoints). Fail the build if any crossings exist.
    Add ring-level thresholds:
        inner angle range should be near the ring spec targets within tolerance
        protruding edge lengths should have bounded variance (and bound should shrink as ring count increases).
    Wire --diagnose cleanly in CLI. The CLI currently references args.diagnose in build-hex without defining the flag; either add the flag to build-hex or remove that branch.

Codebase cleanup tasks to remove failed prototypes and stop future mess

The repo currently includes build artefacts that should not be versioned, plus duplicated “compiled” sources:

There is a build/lib/polygrid/... tree duplicating your real src/polygrid/... modules.
There is also src/polygrid.egg-info/..., another packaging artefact that should be excluded from git.

Tasks:

    Delete from the repository:
        build/ (entire directory)
        src/polygrid.egg-info/ (entire directory)
    Add/expand .gitignore to include at minimum: build/, dist/, *.egg-info/, __pycache__/, .pytest_cache/.
    Consolidate “experiments”:
        Move any non-production embedding prototypes (angle-first experiments, ad-hoc snap/relax steps) out of builders.py into src/polygrid/experiments/ or scripts/, so the core builder only contains the supported pipeline(s).
    Make the public API explicit:
        build_pure_hex_grid and build_pentagon_centered_grid stay, but geometry strategies live in a dedicated module (e.g., embedding_strategies.py) and are selected by the CLI.
    Align documentation with reality:
        the pentagon builder docstring and README describe Tutte as a baseline; if angle is not stable, do not keep it as default.
    Add a single “golden command” for reproducible outputs:
        e.g., polygrid build-pent-all --embed tutte+optimise --dir exports --diagnose and make sure it generates PNGs + metrics JSON deterministically.

Finally, keep the repository’s existing intent visible: the in-repo “task list” already describes the correct end state (robust topology, Tutte embedding, and validation across ring counts). Use it as the canonical roadmap, but update it to reflect the new “global optimisation” step as the actual mechanism of regularisation. 