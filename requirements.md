my core requirement is to create grids that have pentagons at the centre, with rings of hexagons around it, with everything as close to regular as possible. Needs to work for varying numbers of rings. kind of like the face of a 1st class goldberg polyhedron, but 2d. will require a small amount of irregularity in the hexs to work I know, especially closer to the pentagon. any info about this online? What's my best way forwards? if you can link me to any resources that would be helpful.

Yes — what you’re trying to build is not a generic tessellation problem. It’s a very specific object:

a planar extraction of a Goldberg–Coxeter construction (a geodesic / fullerene-style hex grid with a single topological defect).

That phrasing is important because it points you to the right mathematics and avoids dead-ends in Voronoi / generic tiling literature.

🔎 The Key Idea: You Are Introducing a “Topological Defect” Into a Hexagonal Lattice

A perfect Euclidean hex grid cannot contain a pentagon without distortion.
The reason is curvature: inserting a pentagon removes 
60∘
60
∘
 of angular sum, creating a positive curvature singularity.

Goldberg structures are exactly hexagonal meshes with a fixed number of these defects.
They consist only of pentagons and hexagons with three faces meeting at each vertex.

On a sphere, Euler’s formula forces there to be exactly 12 pentagons, surrounded otherwise by hexagons.

When you flatten to 2D and keep just one of those pentagons, you are effectively working with a locally curved hex lattice embedded in the plane.

A useful interpretation from the geometry literature:

The pentagons act like curvature sources; they “repel” each other because each creates an angle deficiency.

Away from the defect, faces approach regular hexagons.

That is exactly the behaviour you described:
hexes near the centre must distort slightly; outer rings become nearly regular.

🔷 The Construction You Want Has a Name: Goldberg–Coxeter Subdivision

Goldberg polyhedra can be parameterized by two integers 
(m,n)
(m,n) describing how you walk a hex lattice:

move 
m
m steps, rotate 
60∘
60
∘
, move 
n
n steps.

This 
(m,n)
(m,n) pair controls how many “rings” of hexagons appear around each pentagon and defines the combinatorial layout (often written GP
(m,n)
(m,n)).

This same construction underlies geodesic domes, virus capsids, and fullerenes.

So your “varying number of rings” corresponds directly to choosing different Goldberg parameters.

🧠 Why a Purely Planar Regular Solution Does Not Exist

There is no Euclidean tiling with regular pentagons surrounded by perfectly regular hexagons.
Any planar realization must introduce irregularity — typically by varying hexagon edge lengths or angles — because a hexagonal grid projected onto curved topology necessarily distorts.

This is not a numerical issue; it’s a geometric constraint.

🧩 Related Planar Tilings (Why They’re Similar but Not What You Want)
Cairo pentagonal tiling

This tiling overlays two hexagonal tessellations and yields pentagon-based patterns.
However it produces periodic pentagons everywhere, not an isolated defect, so it doesn’t match the Goldberg topology.

Rotational pentagon–hexagon tilings

Research shows families of tilings with rotational symmetry built from convex pentagons (and similarly for hexagons).
These are closer conceptually but still lack the “single curvature source” structure.

🛠️ Best Practical Way Forward (Algorithmically)

You do not want to construct polygons directly.

Instead, replicate the standard Goldberg workflow but stop before projecting to the sphere.

Recommended pipeline
1️⃣ Start from a triangular lattice (not hexes)

All Goldberg constructions begin with an equilateral triangle mesh that is then dualized.

triangular lattice
→ apply (m,n) subdivision
→ take dual graph
→ get pentagon + hexagon mesh

This is the Caspar–Klug / Goldberg–Coxeter approach described in structural biology and geodesic design.

2️⃣ Work combinatorially first (pure topology)

Build adjacency using axial / barycentric coordinates in a hex lattice, then insert a 5-valent vertex at the origin (remove one 60° wedge).

This is equivalent to modelling a disclination in crystallography — a known technique for hex grids.

3️⃣ Embed geometrically using relaxation

Once topology is fixed:

Place vertices using a spring / Tutte embedding / energy minimization.

Constrain outer ring to stay roughly circular.

Let inner hexes deform automatically.

This reproduces the “almost regular away from centre” behaviour predicted mathematically.

🧮 Mathematical Model You Can Implement

The cleanest coordinate system is the Eisenstein integer lattice:

Hexagonal grids can be expressed using numbers 
a+bω
a+bω (with 
ω
ω a cube root of unity), giving a natural algebraic indexing of hex positions.

This is extremely convenient for:

ring indexing,

axial coordinates,

implementing the 
(m,n)
(m,n) walk.

🧑‍💻 Existing Code You Can Study

A reference implementation exists that generates Goldberg polyhedra meshes programmatically (MATLAB example).
Even though it targets spheres, the combinatorics are reusable for 2D.

✔️ What This Means for Your Python Design

Treat this as:

Generate a hex lattice with a single 5-fold disclination, then relax.

Not:

tile pentagons + hexagons explicitly.

If you try the latter, you will fight constraints forever.

If You Want, Next Step

I can outline a concrete Python structure (data model + equations + relaxation method) for:

ring-parameterized grid generation,

deterministic reproducibility,

controllable distortion profile.

That’s where this gets interesting from an implementation standpoint.
