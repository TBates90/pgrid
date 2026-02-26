"""Multi-panel composite visualisation.

Produces PNG images that show:
- **Exploded** components with macro-edge IDs and stitch arrows
- **Stitched** composite grid
- **Overlay** (e.g. Voronoi) on the stitched grid
- **Unstitched** components with the overlay preserved

All functions accept an optional *ax* so they can be composed into
multi-panel figures externally, but also work standalone.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .assembly import (
    AssemblyPlan,
    translate_grid,
    _grid_bbox_center,
    _macro_edge_midpoint,
    _macro_edge_outward_normal,
)
from .composite import CompositeGrid, StitchSpec
from .models import Edge, Face, MacroEdge, Vertex
from .polygrid import PolyGrid
from .transforms import (
    Overlay,
    OverlayPoint,
    OverlaySegment,
    OverlayRegion,
)


# ═══════════════════════════════════════════════════════════════════
# Colour palettes
# ═══════════════════════════════════════════════════════════════════

# Distinct colours for up to 12 component grids
_COMPONENT_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324",
]

_VORONOI_EDGE_COLOR = "#e63946"
_VORONOI_SITE_COLOR = "#e63946"
_VORONOI_REGION_COLOR = "#e6394620"
_STITCH_ARROW_COLOR = "#ff6600"
_MACRO_EDGE_LABEL_COLOR = "#0066cc"


# ═══════════════════════════════════════════════════════════════════
# Low-level drawing helpers
# ═══════════════════════════════════════════════════════════════════

def _ensure_mpl():
    """Lazy-import matplotlib; raise helpful error if missing."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, FancyArrowPatch
        return plt, Polygon, FancyArrowPatch
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for visualisation. "
            "Install with `pip install matplotlib`."
        ) from exc


def _draw_grid(
    ax,
    grid: PolyGrid,
    face_color: str = "#5aa9e6",
    edge_color: str = "#2b2b2b",
    face_alpha: float = 0.15,
    linewidth: float = 0.8,
    vertex_size: float = 4.0,
    vertex_color: str = "#2b2b2b",
    show_face_ids: bool = False,
    face_id_fontsize: float = 5.0,
) -> None:
    """Draw all faces, edges, and vertices of a grid onto *ax*."""
    _, Polygon, _ = _ensure_mpl()
    for face in grid.faces.values():
        pts = _face_points(grid, face)
        if len(pts) < 3:
            continue
        poly = Polygon(pts, closed=True, facecolor=face_color,
                       edgecolor=edge_color, alpha=face_alpha,
                       linewidth=linewidth)
        ax.add_patch(poly)
        if show_face_ids:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            ax.text(cx, cy, face.id, fontsize=face_id_fontsize,
                    ha="center", va="center", color="#333333", alpha=0.7)

    for v in grid.vertices.values():
        if v.has_position():
            ax.plot(v.x, v.y, ".", ms=vertex_size, color=vertex_color, zorder=3)


def _draw_macro_edge_labels(
    ax,
    grid: PolyGrid,
    component_name: str = "",
    fontsize: float = 8.0,
    color: str = _MACRO_EDGE_LABEL_COLOR,
) -> None:
    """Label each macro-edge with its ID near the edge midpoint."""
    for me in grid.macro_edges:
        mid = _me_midpoint(grid, me)
        cx, cy = _grid_bbox_center(grid)
        # Offset the label slightly outward from the grid centre
        dx, dy = mid[0] - cx, mid[1] - cy
        norm = math.hypot(dx, dy) or 1.0
        off = 0.4
        lx = mid[0] + dx / norm * off
        ly = mid[1] + dy / norm * off
        label = f"{component_name}e{me.id}" if component_name else f"e{me.id}"
        ax.text(lx, ly, label, fontsize=fontsize, ha="center", va="center",
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.85))


def _draw_stitch_arrows(
    ax,
    plan: AssemblyPlan,
    fontsize: float = 6.0,
) -> None:
    """Draw arrows between matching macro-edge midpoints."""
    _, _, FancyArrowPatch = _ensure_mpl()
    for spec in plan.stitches:
        grid_a = plan.components[spec.grid_a]
        grid_b = plan.components[spec.grid_b]
        me_a = next(m for m in grid_a.macro_edges if m.id == spec.edge_a)
        me_b = next(m for m in grid_b.macro_edges if m.id == spec.edge_b)
        mid_a = _me_midpoint(grid_a, me_a)
        mid_b = _me_midpoint(grid_b, me_b)

        arrow = FancyArrowPatch(
            mid_a, mid_b,
            arrowstyle="<->",
            color=_STITCH_ARROW_COLOR,
            linewidth=1.5,
            connectionstyle="arc3,rad=0.15",
            zorder=5,
        )
        ax.add_patch(arrow)
        mx = (mid_a[0] + mid_b[0]) / 2
        my = (mid_a[1] + mid_b[1]) / 2
        ax.text(mx, my,
                f"{spec.grid_a}.e{spec.edge_a}↔{spec.grid_b}.e{spec.edge_b}",
                fontsize=fontsize, ha="center", va="center",
                color=_STITCH_ARROW_COLOR, alpha=0.85)


_PARTITION_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#ffe119", "#aaffc3", "#808000",
    "#000075",
]


def _draw_overlay(
    ax,
    overlay: Overlay,
    site_size: float = 4.0,
    site_color: str = _VORONOI_SITE_COLOR,
    edge_color: str = _VORONOI_EDGE_COLOR,
    edge_width: float = 0.7,
    region_color: str = _VORONOI_REGION_COLOR,
    draw_regions: bool = True,
    draw_sites: bool = True,
    draw_segments: bool = True,
) -> None:
    """Draw an overlay (e.g. Voronoi or partition) onto an axes."""
    _, Polygon, _ = _ensure_mpl()

    is_partition = overlay.kind == "partition"
    n_sections = overlay.metadata.get("n_sections", 8) if is_partition else 0

    if draw_regions:
        for region in overlay.regions:
            if len(region.points) >= 3:
                if is_partition:
                    # source_vertex_id stores the section index
                    try:
                        sect = int(region.source_vertex_id)
                    except (ValueError, TypeError):
                        sect = 0
                    fc = _PARTITION_COLORS[sect % len(_PARTITION_COLORS)]
                    poly = Polygon(region.points, closed=True,
                                   facecolor=fc, edgecolor="#444444",
                                   alpha=0.45, linewidth=0.3, zorder=1)
                else:
                    poly = Polygon(region.points, closed=True,
                                   facecolor=region_color, edgecolor="none",
                                   alpha=0.15, zorder=1)
                ax.add_patch(poly)

    if draw_segments and not is_partition:
        for seg in overlay.segments:
            ax.plot([seg.start[0], seg.end[0]], [seg.start[1], seg.end[1]],
                    color=edge_color, linewidth=edge_width, zorder=4, alpha=0.7)

    if draw_sites and not is_partition:
        for pt in overlay.points:
            ax.plot(pt.x, pt.y, "o", ms=site_size, color=site_color,
                    zorder=5, alpha=0.6, markeredgewidth=0)


def _translate_overlay(overlay: Overlay, dx: float, dy: float) -> Overlay:
    """Return a copy of *overlay* with all coordinates shifted."""
    new_points = [
        OverlayPoint(id=p.id, x=p.x + dx, y=p.y + dy,
                     label=p.label, source_face_id=p.source_face_id)
        for p in overlay.points
    ]
    new_segs = [
        OverlaySegment(
            id=s.id,
            start=(s.start[0] + dx, s.start[1] + dy),
            end=(s.end[0] + dx, s.end[1] + dy),
            source_edge_id=s.source_edge_id,
        )
        for s in overlay.segments
    ]
    new_regions = [
        OverlayRegion(
            id=r.id,
            points=[(p[0] + dx, p[1] + dy) for p in r.points],
            source_vertex_id=r.source_vertex_id,
        )
        for r in overlay.regions
    ]
    return Overlay(
        kind=overlay.kind,
        points=new_points,
        segments=new_segs,
        regions=new_regions,
        metadata=overlay.metadata,
    )


# ═══════════════════════════════════════════════════════════════════
# Panel renderers
# ═══════════════════════════════════════════════════════════════════

def render_exploded(
    ax,
    plan: AssemblyPlan,
    show_edge_ids: bool = True,
    show_stitch_arrows: bool = True,
) -> None:
    """Render components in exploded positions with macro-edge labels
    and stitch arrows.

    *plan* should already be in exploded layout
    (i.e. ``plan.exploded()``).
    """
    for idx, (name, grid) in enumerate(plan.components.items()):
        color = _COMPONENT_COLORS[idx % len(_COMPONENT_COLORS)]
        _draw_grid(ax, grid, face_color=color, face_alpha=0.20)
        if show_edge_ids:
            _draw_macro_edge_labels(ax, grid, component_name=f"{name}.")
    if show_stitch_arrows:
        _draw_stitch_arrows(ax, plan)
    _autofit_plan(ax, plan)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.set_title("Components (exploded)", fontsize=10)


def render_stitched(
    ax,
    composite: CompositeGrid,
) -> None:
    """Render the merged (stitched) composite grid."""
    _draw_grid(ax, composite.merged, face_color="#5aa9e6", face_alpha=0.15)
    _autofit(ax, composite.merged)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.set_title("Stitched composite", fontsize=10)


def render_stitched_with_overlay(
    ax,
    composite: CompositeGrid,
    overlay: Overlay,
    draw_regions: bool = None,
) -> None:
    """Render stitched grid plus an overlay (e.g. Voronoi or partition)."""
    if draw_regions is None:
        draw_regions = overlay.kind == "partition"
    _draw_grid(ax, composite.merged, face_color="#5aa9e6", face_alpha=0.10,
               edge_color="#999999")
    _draw_overlay(ax, overlay, draw_regions=draw_regions)
    _autofit(ax, composite.merged)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    title = f"Stitched + {overlay.kind}"
    ax.set_title(title, fontsize=10)


def render_unstitched_with_overlay(
    ax,
    flush_plan: AssemblyPlan,
    exploded_plan: AssemblyPlan,
    overlay: Overlay,
    draw_regions: bool = None,
) -> None:
    """Render **pristine** (undistorted) components with overlay colours.

    The flush/exploded plans contain grids that have been scaled,
    rotated, reflected, and boundary-snapped for stitching.  This
    function rebuilds each component from its original builder so
    that hex grids appear as clean, regular hexagons — the
    positioning distortion is only visible on the stitched panels.

    Overlay region assignments (which face belongs to which colour
    group) are transferred from the merged overlay to the pristine
    grid by face-id mapping.
    """
    if draw_regions is None:
        draw_regions = overlay.kind == "partition"

    # Build a face-id → region-index lookup from the merged overlay.
    # Overlay region ids look like "reg_{prefix}{face_id}" or
    # "part_{prefix}{face_id}".  The source_vertex_id holds the
    # region (colour) index as a string.
    merged_face_to_section: Dict[str, str] = {}
    for oreg in overlay.regions:
        # Strip the "reg_" or "part_" prefix to get the merged face id
        for tag in ("reg_", "part_"):
            if oreg.id.startswith(tag):
                merged_fid = oreg.id[len(tag):]
                merged_face_to_section[merged_fid] = oreg.source_vertex_id
                break

    # Rebuild pristine grids and lay them out
    pristine_grids: Dict[str, PolyGrid] = {}
    for name, grid in flush_plan.components.items():
        pristine_grids[name] = _build_pristine_grid(grid)

    # Arrange pristine grids in an exploded layout matching the
    # original exploded plan's arrangement (same translation offsets).
    for name in exploded_plan.components:
        flush_grid = flush_plan.components[name]
        expl_grid = exploded_plan.components[name]
        pristine = pristine_grids[name]

        # Translation: centre of pristine → centre of exploded position
        pc = _grid_bbox_center(pristine)
        ec = _grid_bbox_center(expl_grid)
        dx = ec[0] - pc[0]
        dy = ec[1] - pc[1]
        pristine_grids[name] = translate_grid(pristine, dx, dy)

    # Draw the pristine grids
    for idx, (name, grid) in enumerate(pristine_grids.items()):
        color = _COMPONENT_COLORS[idx % len(_COMPONENT_COLORS)]
        _draw_grid(ax, grid, face_color=color, face_alpha=0.12,
                   edge_color="#aaaaaa")

    # Build and draw overlay per component from pristine geometry
    for name, grid in pristine_grids.items():
        prefix = f"{name}_"
        comp_overlay = _build_pristine_overlay(
            grid, prefix, merged_face_to_section, overlay.kind, overlay.metadata,
        )
        _draw_overlay(ax, comp_overlay, draw_regions=draw_regions,
                      edge_width=0.6, site_size=3.0)

    _autofit_multi(ax, list(pristine_grids.values()))
    ax.set_aspect("equal", "box")
    ax.axis("off")
    ax.set_title(f"Unstitched + {overlay.kind}", fontsize=10)


# ═══════════════════════════════════════════════════════════════════
# High-level multi-panel output
# ═══════════════════════════════════════════════════════════════════

def render_assembly_panels(
    plan: AssemblyPlan,
    output_path: str | Path,
    overlay: Optional[Overlay] = None,
    dpi: int = 150,
    figsize: Tuple[float, float] = (24, 6),
) -> None:
    """Produce a 4-panel PNG:

    1. Exploded components + edge IDs + stitch arrows
    2. Stitched composite
    3. Stitched + overlay (e.g. Voronoi)
    4. Unstitched with overlay preserved

    *plan* should be flush-positioned (for stitching).  The exploded
    layout is computed automatically.

    If *overlay* is ``None``, panels 3 & 4 show the stitched grid
    without an overlay.
    """
    plt, _, _ = _ensure_mpl()

    composite = plan.build()
    exploded = plan.exploded()

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    render_exploded(axes[0], exploded)
    render_stitched(axes[1], composite)

    if overlay is not None:
        render_stitched_with_overlay(axes[2], composite, overlay)
        render_unstitched_with_overlay(axes[3], plan, exploded, overlay)
    else:
        render_stitched(axes[2], composite)
        axes[2].set_title("Stitched (no overlay)", fontsize=10)
        render_exploded(axes[3], exploded, show_stitch_arrows=False)
        axes[3].set_title("Unstitched (no overlay)", fontsize=10)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def render_single_panel(
    grid: PolyGrid,
    output_path: str | Path,
    overlay: Optional[Overlay] = None,
    title: str = "",
    dpi: int = 150,
    figsize: Tuple[float, float] = (8, 8),
) -> None:
    """Render a single grid (with optional overlay) to a PNG."""
    plt, _, _ = _ensure_mpl()
    fig, ax = plt.subplots(figsize=figsize)
    _draw_grid(ax, grid, face_color="#5aa9e6", face_alpha=0.15)
    if overlay is not None:
        _draw_overlay(ax, overlay)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    _autofit(ax, grid)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def render_png(
    grid: PolyGrid,
    output_path: str | Path,
    face_alpha: float = 0.15,
    edge_color: str = "#2b2b2b",
    face_color: str = "#5aa9e6",
    vertex_color: str = "#2b2b2b",
    vertex_size: float = 8.0,
    padding: float = 0.5,
    dpi: int = 150,
    show_pent_axes: bool = False,
) -> None:
    """Render a single grid to PNG with optional pentagon symmetry axes.

    This is the primary single-grid rendering entry point, supporting
    all style parameters and the pentagon-axes diagnostic overlay.
    """
    plt, Polygon, _ = _ensure_mpl()

    fig, ax = plt.subplots()
    _draw_grid(
        ax, grid,
        face_color=face_color,
        edge_color=edge_color,
        face_alpha=face_alpha,
        vertex_size=vertex_size,
        vertex_color=vertex_color,
    )

    if show_pent_axes:
        _draw_pent_axes(ax, grid)

    _autofit(ax, grid, padding=padding)
    ax.set_aspect("equal", "box")
    ax.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _draw_pent_axes(ax, grid: PolyGrid) -> None:
    """Draw symmetry axes through pentagon edge midpoints."""
    pent = next(
        (f for f in grid.faces.values() if f.face_type == "pent"), None
    )
    if pent is None:
        return
    verts = [grid.vertices[vid] for vid in pent.vertex_ids]
    if not all(v.has_position() for v in verts):
        return
    cx = sum(v.x for v in verts if v.x is not None) / len(verts)
    cy = sum(v.y for v in verts if v.y is not None) / len(verts)

    xs = [v.x for v in grid.vertices.values() if v.x is not None]
    ys = [v.y for v in grid.vertices.values() if v.y is not None]
    if not xs or not ys:
        return
    length = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.2

    for i in range(len(verts)):
        v1 = verts[i]
        v2 = verts[(i + 1) % len(verts)]
        mx = (v1.x + v2.x) / 2
        my = (v1.y + v2.y) / 2
        dx, dy = mx - cx, my - cy
        norm = math.hypot(dx, dy) or 1.0
        dx /= norm
        dy /= norm
        ax.plot(
            [cx - dx * length, cx + dx * length],
            [cy - dy * length, cy + dy * length],
            color="#d1495b", linewidth=1.0, linestyle=(0, (3, 3)),
        )


# ═══════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════

def _face_points(grid: PolyGrid, face: Face) -> List[Tuple[float, float]]:
    pts = []
    for vid in face.vertex_ids:
        v = grid.vertices[vid]
        if v.has_position():
            pts.append((v.x, v.y))
    return pts


def _me_midpoint(grid: PolyGrid, me: MacroEdge) -> Tuple[float, float]:
    xs = [grid.vertices[vid].x for vid in me.vertex_ids if grid.vertices[vid].has_position()]
    ys = [grid.vertices[vid].y for vid in me.vertex_ids if grid.vertices[vid].has_position()]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _autofit(ax, grid: PolyGrid, padding: float = 0.5) -> None:
    """Set axis limits to fit a single grid."""
    xs = [v.x for v in grid.vertices.values() if v.has_position()]
    ys = [v.y for v in grid.vertices.values() if v.has_position()]
    if xs and ys:
        ax.set_xlim(min(xs) - padding, max(xs) + padding)
        ax.set_ylim(min(ys) - padding, max(ys) + padding)


def _autofit_plan(ax, plan: AssemblyPlan, padding: float = 1.0) -> None:
    """Set axis limits to fit all components of a plan."""
    all_x: List[float] = []
    all_y: List[float] = []
    for grid in plan.components.values():
        for v in grid.vertices.values():
            if v.has_position():
                all_x.append(v.x)
                all_y.append(v.y)
    if all_x and all_y:
        ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
        ax.set_ylim(min(all_y) - padding, max(all_y) + padding)


def _grid_bbox(
    grid: PolyGrid,
    margin: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) for a grid's positioned verts."""
    xs = [v.x for v in grid.vertices.values() if v.has_position()]
    ys = [v.y for v in grid.vertices.values() if v.has_position()]
    return (min(xs) - margin, min(ys) - margin,
            max(xs) + margin, max(ys) + margin)


def _clip_overlay(
    overlay: Overlay,
    bbox: Tuple[float, float, float, float],
) -> Overlay:
    """Return a copy of *overlay* with elements outside *bbox* removed."""
    xmin, ymin, xmax, ymax = bbox

    def _in(x: float, y: float) -> bool:
        return xmin <= x <= xmax and ymin <= y <= ymax

    pts = [p for p in overlay.points if _in(p.x, p.y)]
    segs = [s for s in overlay.segments
            if _in(*s.start) or _in(*s.end)]
    regions = [r for r in overlay.regions
               if any(_in(p[0], p[1]) for p in r.points)]

    return Overlay(
        kind=overlay.kind,
        points=pts,
        segments=segs,
        regions=regions,
        metadata=overlay.metadata,
    )


def _filter_overlay_by_prefix(
    overlay: Overlay,
    prefix: str,
) -> Overlay:
    """Return a copy of *overlay* keeping only elements whose ids contain *prefix*.

    For partition/region overlays the region id embeds the face id
    (e.g. ``reg_hex0_f3`` or ``part_hex0_f3``).  This filter keeps
    only the regions (and points/segments) that belong to a specific
    component, avoiding cross-component bleed in exploded views.
    """
    pts = [p for p in overlay.points
           if prefix in p.id or prefix in p.source_face_id]
    segs = [s for s in overlay.segments
            if prefix in s.id or prefix in s.source_edge_id]
    regions = [r for r in overlay.regions if prefix in r.id]

    return Overlay(
        kind=overlay.kind,
        points=pts,
        segments=segs,
        regions=regions,
        metadata=overlay.metadata,
    )


def _build_pristine_grid(grid: PolyGrid) -> PolyGrid:
    """Rebuild a component grid from its metadata to get an undistorted copy.

    Uses the ``generator`` and ``rings`` keys stored in
    :attr:`PolyGrid.metadata` by the builder functions.  If the
    metadata is missing, returns the grid unchanged.
    """
    from .builders import build_pure_hex_grid, build_pentagon_centered_grid

    gen = grid.metadata.get("generator")
    rings = grid.metadata.get("rings")
    if rings is None:
        return grid

    if gen == "hex":
        pristine = build_pure_hex_grid(rings)
    elif gen == "goldberg":
        pristine = build_pentagon_centered_grid(rings)
    else:
        return grid

    pristine.compute_macro_edges(n_sides=grid.metadata.get("sides", 6 if gen == "hex" else 5))
    return pristine


def _build_pristine_overlay(
    pristine_grid: PolyGrid,
    merged_prefix: str,
    merged_face_to_section: Dict[str, str],
    kind: str,
    metadata: Dict,
) -> Overlay:
    """Build an overlay for a pristine grid using region assignments
    from the merged overlay.

    *merged_prefix* is e.g. ``"hex0_"``.  For each face ``f3`` in the
    pristine grid, we look up ``"hex0_f3"`` in *merged_face_to_section*
    to find which colour group it belongs to, then build an
    :class:`OverlayRegion` using the pristine grid's vertex positions.
    """
    overlay = Overlay(kind=kind, metadata=metadata)

    for face in pristine_grid.faces.values():
        merged_fid = f"{merged_prefix}{face.id}"
        section = merged_face_to_section.get(merged_fid)
        if section is None:
            continue
        pts: List[Tuple[float, float]] = []
        for vid in face.vertex_ids:
            v = pristine_grid.vertices[vid]
            if v.has_position():
                pts.append((v.x, v.y))
        if len(pts) < 3:
            continue
        overlay.regions.append(OverlayRegion(
            id=f"reg_{merged_fid}",
            points=pts,
            source_vertex_id=section,
        ))

    return overlay


def _autofit_multi(ax, grids: Iterable[PolyGrid], padding: float = 1.0) -> None:
    """Set axis limits to fit multiple grids."""
    all_x: List[float] = []
    all_y: List[float] = []
    for grid in grids:
        for v in grid.vertices.values():
            if v.has_position():
                all_x.append(v.x)
                all_y.append(v.y)
    if all_x and all_y:
        ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
        ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
