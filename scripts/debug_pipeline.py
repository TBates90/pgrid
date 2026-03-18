#!/usr/bin/env python3
"""Debug visualisation script for the polygrid → atlas → globe pipeline.

Produces a series of annotated diagnostic images that trace the full
rendering pipeline for a configurable set of tiles.  Each stage is
visualised so you can see exactly what data is flowing through and
where misalignments originate.

Stages
------
1. **Globe topology** — Goldberg polyhedron with face IDs, pentagon
   highlight, and neighbour adjacency.
2. **Detail grid** — the Tutte-embedded polygrid with boundary
   vertices, detected corners, and macro-edge labels.
3. **Stitched composite** — centre tile + neighbour aprons with
   component boundaries and the view-limits box.
4. **Corner matching** — grid corners (macro-edge order) overlaid
   on the UV polygon, showing the angular alignment and any
   reflection / rotation detected.
5. **Sector equalisation** — before/after grid corners for tiles
   adjacent to pentagons, with sector angle annotations.
6. **Piecewise warp** — the triangle-fan sectors colour-coded,
   with source and destination centroids and per-sector affine
   anisotropy values.
7. **Warped tile** — the final warped slot image with the UV polygon
   boundary overlaid.
8. **Atlas slot** — the tile placed in its atlas position with
   gutter visualisation.

Usage
-----
::

    # Debug all tiles for a frequency-3 globe:
    python scripts/debug_pipeline.py

    # Debug specific tiles:
    python scripts/debug_pipeline.py --tiles t0 t5 t11

    # Higher detail:
    python scripts/debug_pipeline.py -f 3 --detail-rings 4

Output goes to ``exports/debug_pipeline/``.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon as MplPolygon
from matplotlib.collections import PatchCollection


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _angle_deg(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx))


def _centroid(pts: List[Tuple[float, float]]) -> Tuple[float, float]:
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    return cx, cy


def _save_fig(fig, path: Path, dpi: int = 150):
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"    → {path.name}")


# ═══════════════════════════════════════════════════════════════════
# Stage 1 — Globe topology overview
# ═══════════════════════════════════════════════════════════════════

def stage1_globe_topology(
    globe_grid,
    face_ids: List[str],
    output_dir: Path,
):
    """Draw an unfolded view of globe faces with adjacency info."""
    from polygrid.geometry import face_center as _face_center

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title("Stage 1: Globe Topology", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")

    # Assign colours by golden-ratio hue
    golden = 0.618033988749895
    hues = {fid: (0.08 + i * golden) % 1.0 for i, fid in enumerate(face_ids)}

    patches = []
    colours = []
    for fid in face_ids:
        face = globe_grid.faces[fid]
        verts_3d = []
        for vid in face.vertex_ids:
            v = globe_grid.vertices[vid]
            verts_3d.append((v.x, v.y, v.z))

        # Project to 2D using azimuthal equidistant from +Z
        verts_2d = []
        for x, y, z in verts_3d:
            r3 = math.sqrt(x * x + y * y + z * z)
            if r3 < 1e-12:
                verts_2d.append((0.0, 0.0))
                continue
            theta = math.acos(max(-1, min(1, z / r3)))
            phi = math.atan2(y, x)
            r2 = theta  # azimuthal equidistant
            verts_2d.append((r2 * math.cos(phi), r2 * math.sin(phi)))

        if len(verts_2d) >= 3:
            n_sides = len(face.vertex_ids)
            hue = hues[fid]
            lightness = 0.55 if n_sides == 5 else 0.65
            r, g, b = colorsys.hls_to_rgb(hue, lightness, 0.7)
            patches.append(MplPolygon(verts_2d, closed=True))
            colours.append((r, g, b))

            # Label
            cx, cy = _centroid(verts_2d)
            label = fid
            if n_sides == 5:
                label += " ★"
            ax.annotate(
                label, (cx, cy), fontsize=5, ha="center", va="center",
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6),
            )

    if patches:
        pc = PatchCollection(
            patches, facecolors=colours,
            edgecolors="white", linewidths=0.5,
        )
        ax.add_collection(pc)
    ax.autoscale_view()

    _save_fig(fig, output_dir / "stage1_globe_topology.png")


# ═══════════════════════════════════════════════════════════════════
# Stage 2 — Detail grid with boundary + corners + macro edges
# ═══════════════════════════════════════════════════════════════════

def stage2_detail_grid(
    face_id: str,
    detail_grid,
    globe_grid,
    output_dir: Path,
):
    """Draw the detail grid with boundary, corners, and macro edges."""
    n_sides = len(globe_grid.faces[face_id].vertex_ids)

    # Compute macro edges
    corner_ids = detail_grid.metadata.get("corner_vertex_ids")
    detail_grid.compute_macro_edges(n_sides=n_sides, corner_ids=corner_ids)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(
        f"Stage 2: Detail Grid — {face_id} "
        f"({'pentagon' if n_sides == 5 else 'hexagon'})",
        fontsize=13, fontweight="bold",
    )
    ax.set_aspect("equal")

    # Draw all faces
    for fid, face in detail_grid.faces.items():
        verts = []
        for vid in face.vertex_ids:
            v = detail_grid.vertices.get(vid)
            if v and v.has_position():
                verts.append((v.x, v.y))
        if len(verts) >= 3:
            poly = MplPolygon(verts, closed=True, fc="#e0e0e0", ec="#888888", lw=0.3)
            ax.add_patch(poly)

    # Highlight boundary edges
    for edge in detail_grid.edges.values():
        if len(edge.face_ids) < 2:
            va = detail_grid.vertices[edge.vertex_ids[0]]
            vb = detail_grid.vertices[edge.vertex_ids[1]]
            ax.plot([va.x, vb.x], [va.y, vb.y], "r-", lw=1.5, alpha=0.7)

    # Highlight macro edges with distinct colours
    me_colours = plt.cm.tab10(np.linspace(0, 1, max(n_sides, 1)))

    # Compute centroid for outward label nudging
    all_bverts = []
    for me in detail_grid.macro_edges:
        for vid in me.vertex_ids:
            v = detail_grid.vertices.get(vid)
            if v and v.has_position():
                all_bverts.append((v.x, v.y))
    if all_bverts:
        bcx = sum(x for x, _ in all_bverts) / len(all_bverts)
        bcy = sum(y for _, y in all_bverts) / len(all_bverts)
    else:
        bcx, bcy = 0.0, 0.0

    for me in detail_grid.macro_edges:
        colour = me_colours[me.id % len(me_colours)]
        for j in range(len(me.vertex_ids) - 1):
            va = detail_grid.vertices[me.vertex_ids[j]]
            vb = detail_grid.vertices[me.vertex_ids[j + 1]]
            ax.plot([va.x, vb.x], [va.y, vb.y], "-", color=colour, lw=2.5)

        # Label macro edge at midpoint
        mid_idx = len(me.vertex_ids) // 2
        vm = detail_grid.vertices[me.vertex_ids[mid_idx]]
        ax.annotate(
            f"ME{me.id}", (vm.x, vm.y), fontsize=8, fontweight="bold",
            color=colour, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8),
        )

        # Label each boundary vertex along this macro edge
        corner_set_local = {me.vertex_ids[0], me.vertex_ids[-1]}
        for vid in me.vertex_ids:
            v = detail_grid.vertices.get(vid)
            if not v or not v.has_position():
                continue
            # Skip corners — they already have C0..Cn labels
            if vid in corner_set_local:
                continue
            # Nudge label outward from centroid
            dx, dy = v.x - bcx, v.y - bcy
            norm = math.hypot(dx, dy) or 1.0
            ax.plot(v.x, v.y, ".", color=colour, ms=3, zorder=4)
            ax.annotate(
                vid, (v.x, v.y), fontsize=5, color=colour, alpha=0.85,
                xytext=(dx / norm * 12, dy / norm * 12),
                textcoords="offset points", ha="center", va="center",
            )

    # Draw corner vertices
    from polygrid.tile_uv_align import get_macro_edge_corners
    corners = get_macro_edge_corners(detail_grid, n_sides)
    for k, (cx, cy) in enumerate(corners):
        ax.plot(cx, cy, "ko", ms=8, zorder=5)
        ax.plot(cx, cy, "o", color=me_colours[k], ms=6, zorder=6)
        ax.annotate(
            f"C{k}", (cx, cy), fontsize=9, fontweight="bold",
            xytext=(8, 8), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.9),
        )

    # Show grid centroid
    gcx = sum(x for x, _ in corners) / len(corners)
    gcy = sum(y for _, y in corners) / len(corners)
    ax.plot(gcx, gcy, "r+", ms=15, mew=2, zorder=5)
    ax.annotate(
        "centroid", (gcx, gcy), fontsize=7,
        xytext=(10, -10), textcoords="offset points",
        color="red", fontstyle="italic",
    )

    # Show angles from centroid to each corner
    for k, (cx, cy) in enumerate(corners):
        angle = math.degrees(math.atan2(cy - gcy, cx - gcx))
        ax.annotate(
            f"{angle:.1f}°", (cx, cy), fontsize=6, color="gray",
            xytext=(-5, -12), textcoords="offset points",
        )

    ax.autoscale_view()
    ax.margins(0.05)
    _save_fig(fig, output_dir / f"stage2_detail_{face_id}.png")

    return corners


# ═══════════════════════════════════════════════════════════════════
# Stage 3 — Stitched composite with view-limits box
# ═══════════════════════════════════════════════════════════════════

def stage3_stitched_composite(
    face_id: str,
    composite,
    globe_grid,
    output_dir: Path,
):
    """Draw the stitched composite showing centre + neighbour regions."""
    from polygrid.tile_uv_align import compute_tile_view_limits

    mg = composite.merged
    center_prefix = composite.id_prefixes[face_id]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(
        f"Stage 3: Stitched Composite — {face_id}",
        fontsize=13, fontweight="bold",
    )
    ax.set_aspect("equal")

    # Assign a hue per component
    comp_names = list(composite.id_prefixes.keys())
    comp_hues = {}
    golden = 0.618033988749895
    for i, name in enumerate(comp_names):
        comp_hues[name] = (0.08 + i * golden) % 1.0

    patches = []
    colours = []
    for fid, face in mg.faces.items():
        verts = []
        for vid in face.vertex_ids:
            v = mg.vertices.get(vid)
            if v and v.has_position():
                verts.append((v.x, v.y))
        if len(verts) < 3:
            continue

        comp_name = None
        for name, prefix in composite.id_prefixes.items():
            if fid.startswith(prefix):
                comp_name = name
                break
        if comp_name is None:
            continue

        hue = comp_hues[comp_name]
        is_center = (comp_name == face_id)
        lightness = 0.6 if is_center else 0.4
        saturation = 0.8 if is_center else 0.5
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        patches.append(MplPolygon(verts, closed=True))
        colours.append((r, g, b))

    if patches:
        pc = PatchCollection(
            patches, facecolors=colours,
            edgecolors="#00000030", linewidths=0.3,
        )
        ax.add_collection(pc)

    # Draw view-limits box
    xlim, ylim = compute_tile_view_limits(composite, face_id)
    rect_x = [xlim[0], xlim[1], xlim[1], xlim[0], xlim[0]]
    rect_y = [ylim[0], ylim[0], ylim[1], ylim[1], ylim[0]]
    ax.plot(rect_x, rect_y, "r--", lw=2, label="view limits")

    # Label components
    for name, prefix in composite.id_prefixes.items():
        xs, ys = [], []
        for fid, face in mg.faces.items():
            if not fid.startswith(prefix):
                continue
            for vid in face.vertex_ids:
                v = mg.vertices.get(vid)
                if v and v.has_position():
                    xs.append(v.x)
                    ys.append(v.y)
        if xs:
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            label = name
            if name == face_id:
                label += " (centre)"
            ax.annotate(
                label, (cx, cy), fontsize=7, ha="center", va="center",
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7),
            )

    ax.legend(fontsize=8)
    ax.autoscale_view()
    ax.margins(0.05)
    _save_fig(fig, output_dir / f"stage3_composite_{face_id}.png")


# ═══════════════════════════════════════════════════════════════════
# Stage 4 — Corner matching: grid corners ↔ UV corners
# ═══════════════════════════════════════════════════════════════════

def stage4_corner_matching(
    face_id: str,
    grid_corners_raw: List[Tuple[float, float]],
    grid_corners_matched: List[Tuple[float, float]],
    uv_corners: List[Tuple[float, float]],
    globe_grid,
    output_dir: Path,
):
    """Show grid corners and UV corners side-by-side with match arrows."""
    n = len(uv_corners)
    n_sides = len(globe_grid.faces[face_id].vertex_ids)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Stage 4: Corner Matching — {face_id} "
        f"({'pent' if n_sides == 5 else 'hex'})",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: Raw grid corners (macro-edge order)
    ax = axes[0]
    ax.set_title("Grid Corners (macro-edge order)", fontsize=10)
    ax.set_aspect("equal")
    gc = np.array(grid_corners_raw)
    gc_c = gc.mean(axis=0)
    for k in range(n):
        j = (k + 1) % n
        ax.plot([gc[k, 0], gc[j, 0]], [gc[k, 1], gc[j, 1]], "b-", lw=1.5)
        ax.plot(gc[k, 0], gc[k, 1], "bo", ms=8)
        ax.annotate(f"C{k}", gc[k], fontsize=9, fontweight="bold",
                    xytext=(6, 6), textcoords="offset points",
                    color="blue")
        # Angle from centroid
        angle = math.degrees(math.atan2(gc[k, 1] - gc_c[1], gc[k, 0] - gc_c[0]))
        ax.annotate(f"{angle:.1f}°", gc[k], fontsize=6, color="gray",
                    xytext=(-5, -12), textcoords="offset points")
    ax.plot(*gc_c, "r+", ms=12, mew=2)

    # Panel 2: UV corners
    ax = axes[1]
    ax.set_title("UV Corners (GoldbergTile order)", fontsize=10)
    ax.set_aspect("equal")
    uv = np.array(uv_corners)
    uv_c = uv.mean(axis=0)
    for k in range(n):
        j = (k + 1) % n
        ax.plot([uv[k, 0], uv[j, 0]], [uv[k, 1], uv[j, 1]], "g-", lw=1.5)
        ax.plot(uv[k, 0], uv[k, 1], "go", ms=8)
        ax.annotate(f"UV{k}", uv[k], fontsize=9, fontweight="bold",
                    xytext=(6, 6), textcoords="offset points",
                    color="green")
        angle = math.degrees(math.atan2(uv[k, 1] - uv_c[1], uv[k, 0] - uv_c[0]))
        ax.annotate(f"{angle:.1f}°", uv[k], fontsize=6, color="gray",
                    xytext=(-5, -12), textcoords="offset points")
    ax.plot(*uv_c, "r+", ms=12, mew=2)

    # Panel 3: Matched grid corners with arrows to UV corners
    ax = axes[2]
    ax.set_title("Matched Pairs (grid[k] → UV[k])", fontsize=10)
    ax.set_aspect("equal")

    # Normalise both to [0,1] for overlay comparison
    gc_m = np.array(grid_corners_matched)
    gc_m_min = gc_m.min(axis=0)
    gc_m_range = gc_m.max(axis=0) - gc_m_min
    gc_m_range[gc_m_range < 1e-9] = 1.0
    gc_norm = (gc_m - gc_m_min) / gc_m_range

    uv_min = uv.min(axis=0)
    uv_range = uv.max(axis=0) - uv_min
    uv_range[uv_range < 1e-9] = 1.0
    uv_norm = (uv - uv_min) / uv_range

    for k in range(n):
        # Grid corner (blue)
        ax.plot(gc_norm[k, 0], gc_norm[k, 1], "bs", ms=8)
        ax.annotate(f"G{k}", gc_norm[k], fontsize=8, color="blue",
                    xytext=(-12, 6), textcoords="offset points")
        # UV corner (green)
        ax.plot(uv_norm[k, 0], uv_norm[k, 1], "g^", ms=8)
        ax.annotate(f"U{k}", uv_norm[k], fontsize=8, color="green",
                    xytext=(6, -10), textcoords="offset points")
        # Arrow from grid to UV
        ax.annotate(
            "", xy=uv_norm[k], xytext=gc_norm[k],
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2, alpha=0.7),
        )

    fig.tight_layout()
    _save_fig(fig, output_dir / f"stage4_matching_{face_id}.png")


# ═══════════════════════════════════════════════════════════════════
# Stage 5 — Sector equalisation
# ═══════════════════════════════════════════════════════════════════

def stage5_sector_equalisation(
    face_id: str,
    grid_corners_before: List[Tuple[float, float]],
    grid_corners_after: List[Tuple[float, float]],
    uv_corners: List[Tuple[float, float]],
    src_centroid,
    output_dir: Path,
):
    """Show before/after sector equalisation with angle annotations."""
    n = len(uv_corners)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        f"Stage 5: Sector Equalisation — {face_id}",
        fontsize=13, fontweight="bold",
    )

    for panel, (corners, title) in enumerate([
        (grid_corners_before, "Before Equalisation"),
        (grid_corners_after, "After Equalisation"),
    ]):
        ax = axes[panel]
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")

        pts = np.array(corners)
        c = pts.mean(axis=0)

        # Draw polygon
        for k in range(n):
            j = (k + 1) % n
            ax.plot([pts[k, 0], pts[j, 0]], [pts[k, 1], pts[j, 1]], "b-", lw=1.5)

        # Draw sector lines from centroid
        for k in range(n):
            ax.plot([c[0], pts[k, 0]], [c[1], pts[k, 1]], "r--", lw=0.8, alpha=0.5)

        # Annotate sector angles
        for k in range(n):
            j = (k + 1) % n
            a0 = math.atan2(pts[k, 1] - c[1], pts[k, 0] - c[0])
            a1 = math.atan2(pts[j, 1] - c[1], pts[j, 0] - c[0])
            span = (a1 - a0) % (2 * math.pi)
            mid_a = a0 + span / 2
            r = np.linalg.norm(pts[k] - c) * 0.4
            mx = c[0] + r * math.cos(mid_a)
            my = c[1] + r * math.sin(mid_a)
            ax.annotate(
                f"{math.degrees(span):.1f}°", (mx, my),
                fontsize=7, ha="center", va="center",
                color="darkred", fontweight="bold",
            )

        # Corner labels
        for k in range(n):
            ax.plot(pts[k, 0], pts[k, 1], "bo", ms=7)
            ax.annotate(f"C{k}", pts[k], fontsize=8, fontweight="bold",
                        xytext=(6, 6), textcoords="offset points")

        ax.plot(*c, "r+", ms=12, mew=2)
        if src_centroid is not None and panel == 1:
            ax.plot(src_centroid[0], src_centroid[1], "gx", ms=10, mew=2)
            ax.annotate(
                "fixed centroid", (src_centroid[0], src_centroid[1]),
                fontsize=7, color="green", xytext=(8, -10),
                textcoords="offset points",
            )

        ax.margins(0.1)

    fig.tight_layout()
    _save_fig(fig, output_dir / f"stage5_equalise_{face_id}.png")


# ═══════════════════════════════════════════════════════════════════
# Stage 6 — Piecewise warp sectors
# ═══════════════════════════════════════════════════════════════════

def stage6_warp_sectors(
    face_id: str,
    grid_corners: List[Tuple[float, float]],
    uv_corners: List[Tuple[float, float]],
    tile_size: int,
    gutter: int,
    output_dir: Path,
):
    """Visualise the triangle-fan sectors in both source and dest space."""
    from polygrid.tile_uv_align import _build_sector_affines

    n = len(grid_corners)
    src = np.array(grid_corners, dtype=np.float64)
    dst_uv = np.array(uv_corners, dtype=np.float64)

    src_c = src.mean(axis=0)
    dst_px = np.empty_like(dst_uv)
    for i in range(n):
        u, v = dst_uv[i]
        dst_px[i, 0] = gutter + u * tile_size
        dst_px[i, 1] = gutter + (1.0 - v) * tile_size
    dst_px_c = dst_px.mean(axis=0)

    # Sort by destination angle (same as _compute_piecewise_warp_map)
    dst_angles = np.arctan2(
        dst_px[:, 1] - dst_px_c[1],
        dst_px[:, 0] - dst_px_c[0],
    )
    order = np.argsort(dst_angles)
    src_sorted = src[order]
    dst_sorted = dst_px[order]

    # Build sector affines
    fwd_sectors = _build_sector_affines(src_sorted, src_c, dst_sorted, dst_px_c)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        f"Stage 6: Piecewise Warp Sectors — {face_id}",
        fontsize=13, fontweight="bold",
    )

    sector_colours = plt.cm.Set3(np.linspace(0, 1, n))

    for panel, (pts, centroid, title) in enumerate([
        (src_sorted, src_c, "Source (Grid Space)"),
        (dst_sorted, dst_px_c, "Destination (Slot Pixels)"),
    ]):
        ax = axes[panel]
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")

        # Draw each sector triangle
        for k in range(n):
            j = (k + 1) % n
            tri = MplPolygon(
                [centroid, pts[k], pts[j]], closed=True,
                fc=sector_colours[k], ec="black", lw=1, alpha=0.5,
            )
            ax.add_patch(tri)

            # Sector label at centroid of triangle
            tcx = (centroid[0] + pts[k, 0] + pts[j, 0]) / 3
            tcy = (centroid[1] + pts[k, 1] + pts[j, 1]) / 3

            # Compute anisotropy of the sector affine
            A = fwd_sectors[k][0]
            svs = np.linalg.svd(A, compute_uv=False)
            aniso = svs[0] / svs[1] if svs[1] > 1e-12 else float("inf")

            ax.annotate(
                f"S{k}\n{aniso:.2f}×", (tcx, tcy),
                fontsize=7, ha="center", va="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8),
            )

        # Corner labels
        for k in range(n):
            ax.plot(pts[k, 0], pts[k, 1], "ko", ms=6, zorder=5)
            ax.annotate(
                f"{order[k]}", (pts[k, 0], pts[k, 1]),
                fontsize=7, xytext=(5, 5), textcoords="offset points",
            )

        ax.plot(*centroid, "r+", ms=12, mew=2, zorder=5)
        ax.autoscale_view()
        ax.margins(0.1)

    fig.tight_layout()
    _save_fig(fig, output_dir / f"stage6_sectors_{face_id}.png")


# ═══════════════════════════════════════════════════════════════════
# Stage 7 — Warped tile with UV polygon overlay
# ═══════════════════════════════════════════════════════════════════

def stage7_warped_tile(
    face_id: str,
    warped_img,
    uv_corners: List[Tuple[float, float]],
    tile_size: int,
    gutter: int,
    output_dir: Path,
):
    """Show the warped image with the UV polygon boundary overlaid."""
    from polygrid.tile_uv_align import uv_polygon_px

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(
        f"Stage 7: Warped Tile — {face_id}",
        fontsize=13, fontweight="bold",
    )

    ax.imshow(np.array(warped_img))

    # UV polygon in pixel coords
    uv_px = uv_polygon_px(uv_corners, tile_size, gutter)
    uv_px_closed = list(uv_px) + [uv_px[0]]
    xs = [p[0] for p in uv_px_closed]
    ys = [p[1] for p in uv_px_closed]
    ax.plot(xs, ys, "r-", lw=2, label="UV polygon")

    # Corner labels
    for k, (px, py) in enumerate(uv_px):
        ax.plot(px, py, "ro", ms=6)
        ax.annotate(
            f"UV{k}", (px, py), fontsize=8, color="red",
            fontweight="bold", xytext=(5, -10), textcoords="offset points",
        )

    # Gutter boundary
    slot_size = tile_size + 2 * gutter
    rect_x = [gutter, tile_size + gutter, tile_size + gutter, gutter, gutter]
    rect_y = [gutter, gutter, tile_size + gutter, tile_size + gutter, gutter]
    ax.plot(rect_x, rect_y, "y--", lw=1, alpha=0.7, label="inner tile area")

    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, slot_size)
    ax.set_ylim(slot_size, 0)

    _save_fig(fig, output_dir / f"stage7_warped_{face_id}.png")


# ═══════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════

def debug_tile(
    face_id: str,
    globe_grid,
    coll,
    output_dir: Path,
    *,
    tile_size: int = 512,
    gutter: int = 4,
    tile_hues: dict,
):
    """Run the full debug pipeline for a single tile."""
    from polygrid.tile_detail import build_tile_with_neighbours
    from polygrid.tile_uv_align import (
        get_macro_edge_corners,
        match_grid_corners_to_uv,
        compute_tile_view_limits,
        compute_grid_to_px_affine,
        warp_tile_to_uv,
        _equalise_sector_ratios,
        _scale_corners_from_centroid,
        _PENTAGON_GRID_SCALE,
    )
    from polygrid.uv_texture import get_tile_uv_vertices

    tile_dir = output_dir / face_id
    tile_dir.mkdir(parents=True, exist_ok=True)

    dg, _ = coll.get(face_id)
    n_sides = len(globe_grid.faces[face_id].vertex_ids)

    print(f"  [{face_id}] {'pentagon' if n_sides == 5 else 'hexagon'} ({n_sides} sides)")

    # ── Stage 2: Detail grid ──
    grid_corners_raw = stage2_detail_grid(face_id, dg, globe_grid, tile_dir)

    # ── Stage 3: Stitched composite ──
    composite = build_tile_with_neighbours(coll, face_id, globe_grid)
    stage3_stitched_composite(face_id, composite, globe_grid, tile_dir)

    # ── Stage 4: Corner matching ──
    uv_corners = get_tile_uv_vertices(globe_grid, face_id)
    grid_corners_matched = match_grid_corners_to_uv(
        grid_corners_raw, globe_grid, face_id,
    )
    stage4_corner_matching(
        face_id, grid_corners_raw, grid_corners_matched,
        uv_corners, globe_grid, tile_dir,
    )

    # ── Stage 5: Sector equalisation ──
    grid_corners_eq, src_centroid = _equalise_sector_ratios(
        grid_corners_matched, uv_corners,
        tile_size=tile_size, gutter=gutter,
    )
    if n_sides == 5:
        grid_corners_eq = _scale_corners_from_centroid(
            grid_corners_eq, _PENTAGON_GRID_SCALE,
        )
    stage5_sector_equalisation(
        face_id, grid_corners_matched, grid_corners_eq,
        uv_corners, src_centroid, tile_dir,
    )

    # ── Stage 6: Warp sectors ──
    stage6_warp_sectors(
        face_id, grid_corners_eq, uv_corners,
        tile_size, gutter, tile_dir,
    )

    # ── Render a colour-debug stitched image for warping ──
    from PIL import Image

    # Render using the same colour-debug approach as render_polygrids
    _render_colour_debug_for_warp(
        face_id, composite, tile_hues,
        tile_size, tile_dir / "stitched.png",
    )
    tile_img = Image.open(str(tile_dir / "stitched.png")).convert("RGB")

    # ── Stage 7: Warped tile ──
    xlim, ylim = compute_tile_view_limits(composite, face_id)
    affine = compute_grid_to_px_affine(
        grid_corners_eq, uv_corners,
        tile_size=tile_size, gutter=gutter,
    )
    slot_size = tile_size + 2 * gutter
    warped = warp_tile_to_uv(
        tile_img, xlim, ylim, affine, slot_size,
        grid_corners=grid_corners_eq,
        uv_corners=uv_corners,
        tile_size=tile_size,
        gutter=gutter,
        src_centroid_override=src_centroid,
    )
    warped.save(str(tile_dir / "warped_raw.png"))

    stage7_warped_tile(
        face_id, warped, uv_corners,
        tile_size, gutter, tile_dir,
    )


def _render_colour_debug_for_warp(
    face_id: str,
    composite,
    tile_hues: dict,
    tile_size: int,
    output_path: Path,
):
    """Render a minimal colour-debug stitched image for warp debugging."""
    from polygrid.geometry import face_center as _face_center
    from polygrid.tile_uv_align import compute_tile_view_limits

    mg = composite.merged
    center_prefix = composite.id_prefixes[face_id]

    comp_centroids = {}
    comp_max_dist = {}
    for name, prefix in composite.id_prefixes.items():
        xs, ys = [], []
        for fid, face in mg.faces.items():
            if not fid.startswith(prefix):
                continue
            c = _face_center(mg.vertices, face)
            if c is not None:
                xs.append(c[0])
                ys.append(c[1])
        if xs:
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            comp_centroids[name] = (cx, cy)
            max_d = max(
                ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                for x, y in zip(xs, ys)
            )
            comp_max_dist[name] = max_d if max_d > 1e-9 else 1.0
        else:
            comp_centroids[name] = (0.0, 0.0)
            comp_max_dist[name] = 1.0

    patches = []
    colours = []
    for fid, face in mg.faces.items():
        verts = []
        for vid in face.vertex_ids:
            v = mg.vertices.get(vid)
            if v is None or not v.has_position():
                break
            verts.append((v.x, v.y))
        else:
            if len(verts) < 3:
                continue
            comp_name = None
            for name, prefix in composite.id_prefixes.items():
                if fid.startswith(prefix):
                    comp_name = name
                    break
            if comp_name is None:
                continue

            hue = tile_hues.get(comp_name, 0.0)
            cx, cy = comp_centroids[comp_name]
            c = _face_center(mg.vertices, face)
            fx, fy = c if c else (cx, cy)
            dist = ((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5
            t = min(dist / comp_max_dist[comp_name], 1.0)
            lightness = 0.72 - 0.20 * t
            saturation = 0.65 + 0.15 * t
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            patches.append(MplPolygon(verts, closed=True))
            colours.append((r, g, b))

    if not patches:
        return

    xlim, ylim = compute_tile_view_limits(composite, face_id)

    dpi = 100
    fig_size = tile_size / dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    pc = PatchCollection(
        patches, facecolors=colours,
        edgecolors="#00000030", linewidths=0.4,
    )
    ax.add_collection(pc)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    fig.savefig(
        str(output_path), dpi=dpi, bbox_inches="tight", pad_inches=0,
        facecolor=(0.12, 0.12, 0.12),
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Debug visualisation for the polygrid → atlas pipeline.",
    )
    parser.add_argument(
        "-f", "--frequency", type=int, default=3,
        help="Goldberg polyhedron frequency (default: 3)",
    )
    parser.add_argument(
        "--detail-rings", type=int, default=4,
        help="Detail grid ring count (default: 4)",
    )
    parser.add_argument(
        "--tile-size", type=int, default=512,
        help="Output tile size in pixels (default: 512)",
    )
    parser.add_argument(
        "--gutter", type=int, default=4,
        help="Atlas gutter size in pixels (default: 4)",
    )
    parser.add_argument(
        "--tiles", nargs="*", default=None,
        help="Specific tile IDs to debug (default: all, or a sample "
             "of 1 pentagon + 3 hexagons if > 20 tiles)",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="Output directory (default: exports/debug_pipeline/)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parent.parent / "exports" / "debug_pipeline"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    from polygrid.globe import build_globe_grid
    from polygrid.tile_detail import TileDetailSpec, DetailGridCollection

    print(f"Building globe (freq={args.frequency})...")
    grid = build_globe_grid(args.frequency)

    spec = TileDetailSpec(detail_rings=args.detail_rings)
    print(f"Building detail grids (rings={args.detail_rings})...")
    t0 = time.perf_counter()
    coll = DetailGridCollection.build(grid, spec)
    print(f"  → {coll.total_face_count} sub-faces in "
          f"{time.perf_counter() - t0:.2f}s")

    face_ids = coll.face_ids

    # Select tiles to debug
    if args.tiles:
        debug_ids = [t for t in args.tiles if t in face_ids]
        if not debug_ids:
            print(f"ERROR: None of {args.tiles} found in {face_ids[:5]}...")
            sys.exit(1)
    elif len(face_ids) > 20:
        # Auto-select: 1 pentagon + 3 hexagons
        pents = [f for f in face_ids if len(grid.faces[f].vertex_ids) == 5]
        hexes = [f for f in face_ids if len(grid.faces[f].vertex_ids) == 6]
        debug_ids = pents[:1] + hexes[:3]
        print(f"Auto-selecting {len(debug_ids)} tiles: {debug_ids}")
    else:
        debug_ids = face_ids

    # Tile hues for colour-debug rendering
    golden = 0.618033988749895
    tile_hues = {fid: (0.08 + i * golden) % 1.0 for i, fid in enumerate(face_ids)}

    # ── Stage 1: Globe topology ──
    print("Stage 1: Globe topology...")
    stage1_globe_topology(grid, face_ids, output_dir)

    # ── Per-tile stages ──
    for fid in debug_ids:
        print(f"\nProcessing {fid}...")
        debug_tile(
            fid, grid, coll, output_dir,
            tile_size=args.tile_size,
            gutter=args.gutter,
            tile_hues=tile_hues,
        )

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Debug output: {output_dir}/")
    total = sum(1 for _ in output_dir.rglob("*.png"))
    print(f"Total images: {total}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
