#!/usr/bin/env python3
"""Debug script: render detail polygrids for adjacent tiles.

Outputs a matplotlib figure showing the actual sub-face polygons
coloured by elevation, so boundary continuity can be inspected
directly without any UV mapping or texture pipeline.

Usage
-----
    python scripts/debug_polygrids.py [-f FREQ] [--rings RINGS] [--tiles T1 T2 ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _elevation_to_rgb(elev: float) -> tuple:
    """Simple green-brown ramp for elevation visualisation."""
    # Clamp to [0, 1]
    e = max(0.0, min(1.0, elev))
    if e < 0.3:
        # Deep water → shallow water
        r = int(30 + 60 * (e / 0.3))
        g = int(60 + 80 * (e / 0.3))
        b = int(120 + 60 * (e / 0.3))
    elif e < 0.5:
        # Lowland green
        t = (e - 0.3) / 0.2
        r = int(90 + 40 * t)
        g = int(140 + 30 * t)
        b = int(60 - 20 * t)
    elif e < 0.7:
        # Highland brown
        t = (e - 0.5) / 0.2
        r = int(130 + 50 * t)
        g = int(120 - 20 * t)
        b = int(40 + 20 * t)
    else:
        # Mountain grey/white
        t = (e - 0.7) / 0.3
        r = int(180 + 60 * t)
        g = int(170 + 70 * t)
        b = int(160 + 80 * t)
    return (min(255, r), min(255, g), min(255, b))


def main():
    parser = argparse.ArgumentParser(description="Render detail polygrids")
    parser.add_argument("-f", "--frequency", type=int, default=3)
    parser.add_argument("--rings", type=int, default=3)
    parser.add_argument("--tiles", nargs="*", default=None,
                        help="Tile IDs to render (e.g. t5 t3 t4). "
                             "Default: t5 and its neighbours.")
    parser.add_argument("-o", "--output", default="exports/debug_polygrids.png")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import numpy as np

    from polygrid.globe import build_globe_grid
    from polygrid.mountains import MountainConfig, generate_mountains
    from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
    from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
    from polygrid.detail_terrain import generate_all_detail_terrain
    from polygrid.algorithms import get_face_adjacency

    # ── Build globe + terrain ──────────────────────────────────────
    print(f"Building globe (freq={args.frequency})...")
    grid = build_globe_grid(args.frequency)
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)
    config = MountainConfig(
        seed=42, ridge_frequency=2.0, ridge_octaves=4,
        peak_elevation=1.0, base_elevation=0.0,
    )
    generate_mountains(grid, store, config)

    # ── Build detail grids + terrain ───────────────────────────────
    print(f"Building detail grids (rings={args.rings})...")
    spec = TileDetailSpec(detail_rings=args.rings)
    coll = DetailGridCollection.build(grid, spec)
    generate_all_detail_terrain(coll, grid, store, spec, seed=42)

    # ── Select tiles to render ─────────────────────────────────────
    if args.tiles:
        tile_ids = args.tiles
    else:
        # Default: t5 and its immediate neighbours
        adj = get_face_adjacency(grid)
        center_tile = "t5"
        tile_ids = [center_tile] + list(adj.get(center_tile, []))

    print(f"Rendering {len(tile_ids)} tiles: {tile_ids}")

    # ── Render each detail grid ────────────────────────────────────
    n = len(tile_ids)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, fid in enumerate(tile_ids):
        row, col = divmod(idx, cols)
        ax = axes[row, col]

        detail_grid, detail_store = coll.get(fid)
        parent_elev = store.get(fid, "elevation")

        patches = []
        colors = []
        for sub_fid, face in detail_grid.faces.items():
            pts = []
            for vid in face.vertex_ids:
                v = detail_grid.vertices.get(vid)
                if v is None or v.x is None or v.y is None:
                    break
                pts.append((v.x, v.y))
            if len(pts) < 3:
                continue

            elev = detail_store.get(sub_fid, "elevation")
            r, g, b = _elevation_to_rgb(elev)
            patches.append(Polygon(pts, closed=True))
            colors.append((r / 255, g / 255, b / 255))

        pc = PatchCollection(patches, facecolors=colors,
                             edgecolors="black", linewidths=0.3)
        ax.add_collection(pc)

        # Auto-fit
        all_x = [v.x for v in detail_grid.vertices.values()
                 if v.x is not None]
        all_y = [v.y for v in detail_grid.vertices.values()
                 if v.y is not None]
        if all_x and all_y:
            pad = 0.5
            ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
            ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

        ax.set_aspect("equal")
        ax.set_title(f"{fid} (elev={parent_elev:.3f})", fontsize=10)
        ax.axis("off")

    # Hide unused axes
    for idx in range(n, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis("off")

    fig.suptitle(
        f"Detail Polygrids — freq={args.frequency}, rings={args.rings}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
