"""High-level assembly recipes for building composite polygrids.

An *assembly* is a collection of named component polygrids with stitch
specifications that describe how their macro-edges join together.

The primary recipe today is ``pent_hex_assembly`` — one pentagon-centred
grid surrounded by five hex grids — but the helpers are generic enough
for arbitrary layouts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .builders import build_pentagon_centered_grid, build_pure_hex_grid
from .composite import CompositeGrid, StitchSpec, stitch_grids
from .models import Vertex
from .polygrid import PolyGrid


# ═══════════════════════════════════════════════════════════════════
# Assembly descriptor
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AssemblyPlan:
    """Describes a set of named grids + stitch instructions.

    *components* maps component name → PolyGrid (flush-positioned for
    stitching — boundary edges of joined pairs overlap exactly).

    *stitches* is the list of macro-edge join specs.
    """

    components: Dict[str, PolyGrid] = field(default_factory=dict)
    stitches: List[StitchSpec] = field(default_factory=list)

    def build(self) -> CompositeGrid:
        """Stitch all components into a single CompositeGrid."""
        return stitch_grids(self.components, self.stitches)

    def exploded(self, gap: float | None = None) -> "AssemblyPlan":
        """Return a copy with non-centre components pushed outward.

        Each hex component is translated radially away from the centre
        grid by *gap* world units.

        If *gap* is ``None`` an automatic gap is chosen so that
        components are clearly separated.
        """
        if not self.stitches:
            return self  # nothing to explode

        centre_name = self.stitches[0].grid_a
        centre_grid = self.components[centre_name]

        if gap is None:
            outer_diags = [
                _grid_bbox_diagonal(self.components[s.grid_b])
                for s in self.stitches
                if s.grid_b in self.components
            ]
            centre_diag = _grid_bbox_diagonal(centre_grid)
            max_outer = max(outer_diags) if outer_diags else 0.0
            gap = centre_diag * 0.5 + max_outer * 0.5 + centre_diag * 0.15

        new_components: Dict[str, PolyGrid] = {centre_name: centre_grid}

        for spec in self.stitches:
            if spec.grid_a != centre_name:
                continue  # only explode pent→hex stitches
            other_name = spec.grid_b
            other_grid = self.components[other_name]

            normal = _macro_edge_outward_normal(centre_grid, spec.edge_a)
            dx = normal[0] * gap
            dy = normal[1] * gap
            new_components[other_name] = translate_grid(other_grid, dx, dy)

        return AssemblyPlan(components=new_components, stitches=self.stitches)


# ═══════════════════════════════════════════════════════════════════
# Grid transforms
# ═══════════════════════════════════════════════════════════════════

def translate_grid(grid: PolyGrid, dx: float, dy: float) -> PolyGrid:
    """Return a new PolyGrid with all vertex positions shifted."""
    new_verts = [
        Vertex(v.id, v.x + dx, v.y + dy) if v.has_position() else v
        for v in grid.vertices.values()
    ]
    return PolyGrid(
        new_verts, grid.edges.values(), grid.faces.values(),
        grid.metadata, grid.macro_edges,
    )


def rotate_grid(grid: PolyGrid, angle_rad: float,
                cx: float = 0.0, cy: float = 0.0) -> PolyGrid:
    """Return a new PolyGrid rotated *angle_rad* around (cx, cy)."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    new_verts = []
    for v in grid.vertices.values():
        if v.has_position():
            rx = v.x - cx
            ry = v.y - cy
            new_verts.append(Vertex(v.id, cx + rx * cos_a - ry * sin_a,
                                          cy + rx * sin_a + ry * cos_a))
        else:
            new_verts.append(v)
    return PolyGrid(
        new_verts, grid.edges.values(), grid.faces.values(),
        grid.metadata, grid.macro_edges,
    )


def scale_grid(grid: PolyGrid, factor: float,
               cx: float = 0.0, cy: float = 0.0) -> PolyGrid:
    """Return a new PolyGrid scaled by *factor* around (cx, cy)."""
    new_verts = []
    for v in grid.vertices.values():
        if v.has_position():
            new_verts.append(Vertex(v.id, cx + (v.x - cx) * factor,
                                          cy + (v.y - cy) * factor))
        else:
            new_verts.append(v)
    return PolyGrid(
        new_verts, grid.edges.values(), grid.faces.values(),
        grid.metadata, grid.macro_edges,
    )


# ═══════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════

def _grid_bbox_center(grid: PolyGrid) -> Tuple[float, float]:
    """Bounding-box centre of positioned vertices."""
    xs = [v.x for v in grid.vertices.values() if v.has_position()]
    ys = [v.y for v in grid.vertices.values() if v.has_position()]
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)


def _grid_bbox_diagonal(grid: PolyGrid) -> float:
    """Bounding-box diagonal length."""
    xs = [v.x for v in grid.vertices.values() if v.has_position()]
    ys = [v.y for v in grid.vertices.values() if v.has_position()]
    if not xs or not ys:
        return 1.0
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _macro_edge_midpoint(grid: PolyGrid,
                         edge_id: int) -> Tuple[float, float]:
    """Midpoint of a macro-edge (average of vertex positions)."""
    me = next(m for m in grid.macro_edges if m.id == edge_id)
    xs = [grid.vertices[vid].x for vid in me.vertex_ids]
    ys = [grid.vertices[vid].y for vid in me.vertex_ids]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _macro_edge_outward_normal(grid: PolyGrid,
                               edge_id: int) -> Tuple[float, float]:
    """Unit outward-pointing normal of a macro-edge."""
    me = next(m for m in grid.macro_edges if m.id == edge_id)
    v0 = grid.vertices[me.vertex_ids[0]]
    v1 = grid.vertices[me.vertex_ids[-1]]
    dx = v1.x - v0.x
    dy = v1.y - v0.y
    length = math.hypot(dx, dy) or 1.0
    cx, cy = _grid_bbox_center(grid)
    mx, my = _macro_edge_midpoint(grid, edge_id)
    n1 = (-dy / length, dx / length)
    n2 = (dy / length, -dx / length)
    outward_x = mx - cx
    outward_y = my - cy
    dot1 = n1[0] * outward_x + n1[1] * outward_y
    return n1 if dot1 > 0 else n2


def _hex_corner_positions(grid: PolyGrid) -> List[Tuple[float, float]]:
    """Return the 6 corner positions of a hex grid (in macro-edge order).

    Corner i is the first vertex of macro-edge i.
    """
    return [
        (grid.vertices[me.vertex_ids[0]].x,
         grid.vertices[me.vertex_ids[0]].y)
        for me in sorted(grid.macro_edges, key=lambda m: m.id)
    ]


def _pent_corner_positions(pent: PolyGrid) -> List[Tuple[float, float]]:
    """Return the 5 corner positions of the pentagon grid (in order)."""
    corners = []
    for me in sorted(pent.macro_edges, key=lambda m: m.id):
        v = pent.vertices[me.vertex_ids[0]]
        corners.append((v.x, v.y))
    return corners


# ═══════════════════════════════════════════════════════════════════
# Pentagon + 5 hex assembly
# ═══════════════════════════════════════════════════════════════════

def pent_hex_assembly(
    rings: int,
    hex_size: float = 1.0,
    pent_size: float = 1.0,
) -> AssemblyPlan:
    """Build one pentagon-centred grid with 5 hex grids around it.

    The hex grids are regular hexagonal grids, positioned flush against
    each pent macro-edge.  Adjacent hex grids meet at the exterior
    angle bisectors of the pentagon corners with a small gap (~6° per
    side, from the 120° hex corner vs 108° pent corner mismatch).

    Boundary vertices on hex-hex edges are snapped to their averaged
    positions so that stitching produces a clean merge with only
    minimal local distortion in the outermost boundary cells.

    Returns an :class:`AssemblyPlan` with components named
    ``"pent"`` and ``"hex0"`` … ``"hex4"``.
    """
    pent = build_pentagon_centered_grid(rings, size=pent_size)
    pent.compute_macro_edges(n_sides=5)

    plan = AssemblyPlan()
    plan.components["pent"] = pent

    for i in range(5):
        hex_grid = build_pure_hex_grid(rings, size=hex_size)
        hex_grid.compute_macro_edges(n_sides=6)

        # Position hex flush to pent edge i (edge 3 ↔ pent edge i)
        positioned = _position_hex_for_stitch(pent, i, hex_grid, 3)
        positioned.compute_macro_edges(n_sides=6)

        name = f"hex{i}"
        plan.components[name] = positioned

        # Pent ↔ hex stitch
        plan.stitches.append(
            StitchSpec(grid_a="pent", edge_a=i, grid_b=name, edge_b=3)
        )

    # Hex ↔ hex stitches: hex{i}.e2 meets hex{(i+1)%5}.e4 at
    # the shared pent corner (i+1)%5.
    for i in range(5):
        j = (i + 1) % 5
        plan.stitches.append(
            StitchSpec(grid_a=f"hex{i}", edge_a=2,
                       grid_b=f"hex{j}", edge_b=4)
        )

    # Snap hex-hex boundary vertices to averaged positions so the
    # stitch merge can find matching vertices.
    _snap_hex_hex_boundaries(plan)

    return plan


def _snap_hex_hex_boundaries(plan: AssemblyPlan) -> None:
    """Snap vertices on hex-hex boundary edges to shared positions.

    For each hex-hex stitch, sets both corresponding vertex pairs to
    their averaged position so they overlap exactly.
    """
    for spec in plan.stitches:
        if not spec.grid_a.startswith("hex"):
            continue  # only hex-hex
        ga = plan.components[spec.grid_a]
        gb = plan.components[spec.grid_b]

        me_a = next(m for m in ga.macro_edges if m.id == spec.edge_a)
        me_b = next(m for m in gb.macro_edges if m.id == spec.edge_b)

        # Forward of edge_a aligns with reversed edge_b (flip=True)
        vids_a = list(me_a.vertex_ids)
        vids_b = list(me_b.vertex_ids)[::-1]

        for va_id, vb_id in zip(vids_a, vids_b):
            va = ga.vertices[va_id]
            vb = gb.vertices[vb_id]
            if va.has_position() and vb.has_position():
                mx = (va.x + vb.x) / 2
                my = (va.y + vb.y) / 2
                ga.vertices[va_id] = Vertex(va_id, mx, my)
                gb.vertices[vb_id] = Vertex(vb_id, mx, my)


def _position_hex_for_stitch(
    target_grid: PolyGrid,
    target_edge: int,
    source_grid: PolyGrid,
    source_edge: int,
) -> PolyGrid:
    """Translate, rotate, and scale *source_grid* so that its
    *source_edge* aligns vertex-to-vertex with *target_grid*'s
    *target_edge*.

    The source edge is reversed (flipped) to match the natural
    stitching orientation.  The source body is guaranteed to be on
    the outward side of the target edge (away from the target
    grid's centre).
    """
    tme = next(m for m in target_grid.macro_edges if m.id == target_edge)
    t0 = target_grid.vertices[tme.vertex_ids[0]]
    t1 = target_grid.vertices[tme.vertex_ids[-1]]
    t_len = math.hypot(t1.x - t0.x, t1.y - t0.y)
    t_angle = math.atan2(t1.y - t0.y, t1.x - t0.x)

    sme = next(m for m in source_grid.macro_edges if m.id == source_edge)
    s0 = source_grid.vertices[sme.vertex_ids[-1]]   # reversed start
    s1 = source_grid.vertices[sme.vertex_ids[0]]     # reversed end
    s_len = math.hypot(s1.x - s0.x, s1.y - s0.y)

    # 1) Scale
    scale = t_len / s_len if s_len > 1e-12 else 1.0
    scx, scy = (s0.x + s1.x) / 2, (s0.y + s1.y) / 2
    grid = scale_grid(source_grid, scale, scx, scy)

    # Recompute after scale
    sme2 = next(m for m in grid.macro_edges if m.id == source_edge)
    s0b = grid.vertices[sme2.vertex_ids[-1]]
    s1b = grid.vertices[sme2.vertex_ids[0]]
    s_angle2 = math.atan2(s1b.y - s0b.y, s1b.x - s0b.x)

    # 2) Rotate
    rotation = t_angle - s_angle2
    rcx, rcy = (s0b.x + s1b.x) / 2, (s0b.y + s1b.y) / 2
    grid = rotate_grid(grid, rotation, rcx, rcy)

    # 3) Translate
    sme3 = next(m for m in grid.macro_edges if m.id == source_edge)
    s0c = grid.vertices[sme3.vertex_ids[-1]]
    dx = t0.x - s0c.x
    dy = t0.y - s0c.y
    grid = translate_grid(grid, dx, dy)

    # 4) Ensure source body is on the OUTSIDE of the target edge.
    #    Compare source centroid vs target centroid relative to
    #    the outward normal of the target edge.  If source is on
    #    the same side as target centre (dot < 0), reflect.
    normal = _macro_edge_outward_normal(target_grid, target_edge)
    emx, emy = _macro_edge_midpoint(target_grid, target_edge)

    src_xs = [v.x for v in grid.vertices.values() if v.has_position()]
    src_ys = [v.y for v in grid.vertices.values() if v.has_position()]
    src_cx = sum(src_xs) / len(src_xs)
    src_cy = sum(src_ys) / len(src_ys)

    dot = (src_cx - emx) * normal[0] + (src_cy - emy) * normal[1]
    if dot < 0:
        grid = _reflect_across_edge(grid, source_edge)

    return grid


def _reflect_across_edge(grid: PolyGrid, edge_id: int) -> PolyGrid:
    """Reflect all vertices across the line defined by a macro-edge."""
    me = next(m for m in grid.macro_edges if m.id == edge_id)
    v0 = grid.vertices[me.vertex_ids[0]]
    v1 = grid.vertices[me.vertex_ids[-1]]

    ldx = v1.x - v0.x
    ldy = v1.y - v0.y
    l2 = ldx * ldx + ldy * ldy
    if l2 < 1e-15:
        return grid

    new_verts = []
    for v in grid.vertices.values():
        if v.has_position():
            dx = v.x - v0.x
            dy = v.y - v0.y
            t = (dx * ldx + dy * ldy) / l2
            px = v0.x + t * ldx
            py = v0.y + t * ldy
            new_verts.append(Vertex(v.id, 2 * px - v.x, 2 * py - v.y))
        else:
            new_verts.append(v)

    return PolyGrid(
        new_verts, grid.edges.values(), grid.faces.values(),
        grid.metadata, grid.macro_edges,
    )
