#!/usr/bin/env python3
"""Render a diagnostic globe with tile IDs, type labels, and polygon outlines
burned directly into the atlas texture.

Usage
-----
::
    # First generate tiles:
    python scripts/render_polygrids.py -f 3 --detail-rings 3 -o exports/f3

    # Then render debug globe:
    python scripts/debug_globe_labels.py exports/f3

    # Or with freq-4:
    python scripts/render_polygrids.py -f 4 --detail-rings 3 -o exports/f4
    python scripts/debug_globe_labels.py exports/f4
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image, ImageDraw, ImageFont
import numpy as np


def _get_font(size: int):
    """Try to load a monospace TTF; fall back to PIL default."""
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def burn_debug_atlas(
    atlas_path: Path,
    uv_layout: dict[str, tuple[float, float, float, float]],
    globe_grid,
    goldberg_tiles,
    frequency: int,
    output_path: Path,
):
    """Burn tile IDs, type labels, edge labels, and polygon outlines
    into a copy of the atlas."""
    from polygrid.uv_texture import (
        get_tile_uv_vertices,
        _match_tile_to_face,
    )
    from polygrid.tile_uv_align import compute_uv_to_polygrid_offset

    atlas = Image.open(str(atlas_path)).convert("RGB")
    draw = ImageDraw.Draw(atlas)
    atlas_w, atlas_h = atlas.size

    font_big = _get_font(14)
    font_sm = _get_font(10)

    for fid, (u_min, v_min, u_max, v_max) in uv_layout.items():
        face = globe_grid.faces.get(fid)
        if face is None:
            continue

        tile = _match_tile_to_face(goldberg_tiles, fid)
        n_sides = len(tile.vertices)
        is_pent = (n_sides == 5)
        face_type = "P" if is_pent else "H"

        # Atlas pixel coordinates of the slot
        px_left = int(u_min * atlas_w)
        px_right = int(u_max * atlas_w)
        px_top = int((1.0 - v_max) * atlas_h)
        px_bot = int((1.0 - v_min) * atlas_h)
        slot_w = px_right - px_left
        slot_h = px_bot - px_top

        # Get UV polygon vertices (normalised [0,1])
        uv_verts = list(tile.uv_vertices)

        # Map UV vertices to atlas pixel coordinates
        polygon_px = []
        for u, v in uv_verts:
            px_x = px_left + u * slot_w
            px_y = px_top + (1.0 - v) * slot_h
            polygon_px.append((px_x, px_y))

        # Draw polygon outline
        outline_colour = (255, 0, 0) if is_pent else (0, 200, 255)
        for i in range(n_sides):
            x0, y0 = polygon_px[i]
            x1, y1 = polygon_px[(i + 1) % n_sides]
            draw.line([(x0, y0), (x1, y1)], fill=outline_colour, width=2)

        # Compute polygon centroid in pixel coords
        cx = sum(p[0] for p in polygon_px) / n_sides
        cy = sum(p[1] for p in polygon_px) / n_sides

        # Draw tile ID at centre
        label = f"{fid}\n{face_type}{n_sides}"
        bbox = draw.textbbox((0, 0), label, font=font_big)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # Background rectangle for readability
        pad = 2
        draw.rectangle(
            [cx - tw/2 - pad, cy - th/2 - pad,
             cx + tw/2 + pad, cy + th/2 + pad],
            fill=(0, 0, 0, 180),
        )
        draw.text(
            (cx - tw/2, cy - th/2), label,
            fill=(255, 255, 0) if is_pent else (255, 255, 255),
            font=font_big,
        )

        # Edge labels with neighbour IDs
        # Use the GoldbergTile neighbour ordering
        offset = compute_uv_to_polygrid_offset(globe_grid, fid)
        face_vids = face.vertex_ids
        neighbours = {}
        for edge in globe_grid.edges.values():
            if fid in edge.face_ids and len(edge.face_ids) == 2:
                nid = [f for f in edge.face_ids if f != fid][0]
                evids = set(edge.vertex_ids)
                for k in range(len(face_vids)):
                    e_pair = {face_vids[k], face_vids[(k + 1) % len(face_vids)]}
                    if evids == e_pair:
                        gt_k = (k + offset) % n_sides
                        neighbours[gt_k] = nid
                        break

        for gt_k, nid in neighbours.items():
            # Midpoint of GT edge k
            p0 = polygon_px[gt_k]
            p1 = polygon_px[(gt_k + 1) % n_sides]
            mx = (p0[0] + p1[0]) / 2
            my = (p0[1] + p1[1]) / 2

            # Push label slightly inward toward centre
            dx = cx - mx
            dy = cy - my
            d = math.sqrt(dx*dx + dy*dy) + 1e-6
            mx += dx / d * 8
            my += dy / d * 8

            nface = globe_grid.faces.get(nid)
            ntype = "P" if nface and len(nface.vertex_ids) == 5 else "H"
            edge_label = f"{nid}"

            bbox_e = draw.textbbox((0, 0), edge_label, font=font_sm)
            ew = bbox_e[2] - bbox_e[0]
            eh = bbox_e[3] - bbox_e[1]
            draw.rectangle(
                [mx - ew/2 - 1, my - eh/2 - 1,
                 mx + ew/2 + 1, my + eh/2 + 1],
                fill=(0, 0, 0),
            )
            draw.text(
                (mx - ew/2, my - eh/2), edge_label,
                fill=(200, 200, 200),
                font=font_sm,
            )

    atlas.save(str(output_path))
    print(f"Debug atlas saved: {output_path}")
    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Render debug globe with tile IDs and outlines.",
    )
    parser.add_argument("export_dir", type=Path,
                        help="Directory from render_polygrids.py output")
    parser.add_argument("--no-view", action="store_true",
                        help="Save debug atlas only, don't launch 3D viewer")
    args = parser.parse_args()

    export_dir = args.export_dir
    atlas_path = export_dir / "atlas.png"
    uv_layout_path = export_dir / "uv_layout.json"
    metadata_path = export_dir / "metadata.json"
    payload_path = export_dir / "globe_payload.json"

    if not atlas_path.exists():
        print(f"Error: {atlas_path} not found. Run render_polygrids.py first.")
        sys.exit(1)

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)
    frequency = metadata.get("frequency", 3)
    seed = metadata.get("seed", 42)

    with open(uv_layout_path) as f:
        uv_layout = json.load(f)

    with open(payload_path) as f:
        payload = json.load(f)

    # Rebuild globe grid + GoldbergTiles for the debug overlay
    from polygrid.globe import build_globe_grid
    from polygrid.uv_texture import get_goldberg_tiles

    print(f"Building globe grid (freq={frequency})...")
    globe_grid = build_globe_grid(frequency)
    goldberg_tiles = get_goldberg_tiles(frequency)

    # Burn debug info into atlas
    debug_atlas_path = export_dir / "atlas_debug.png"
    burn_debug_atlas(
        atlas_path, uv_layout, globe_grid, goldberg_tiles,
        frequency, debug_atlas_path,
    )

    if args.no_view:
        print("Done (--no-view).")
        return

    # Launch the 3D viewer with the debug atlas
    from polygrid.globe_renderer_v2 import render_globe_v2

    print("Launching debug 3D viewer...")
    render_globe_v2(
        payload,
        debug_atlas_path,
        uv_layout,
        title=f"DEBUG Globe — freq={frequency}",
    )


if __name__ == "__main__":
    main()
