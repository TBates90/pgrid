#!/usr/bin/env python3
"""Demo: generate terrain on a Goldberg polyhedron with 3D mesh output.

This script builds terrain on a globe grid and produces:
- A matplotlib 3D render (PNG)
- A models-compatible mesh payload (JSON metadata)
- Optionally, a region-coloured view

Usage:
    python scripts/demo_globe_3d.py [--frequency N] [--preset PRESET] [--out DIR] [--seed S]

Presets:
    mountain_range   — continental ridges + valleys (default)
    alpine_peaks     — sharp summits, deep valleys
    rolling_hills    — gentle undulation
    mesa_plateau     — flat tablelands
    regions          — Voronoi region partitioning (coloured by region, not elevation)
    rivers           — mountains + river network
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

# Ensure src/ is on the path when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polygrid.globe import build_globe_grid
from polygrid.mountains import (
    MountainConfig,
    generate_mountains,
    MOUNTAIN_RANGE,
    ALPINE_PEAKS,
    ROLLING_HILLS,
    MESA_PLATEAU,
)
from polygrid.tile_data import FieldDef, TileSchema, TileDataStore
from polygrid.globe_render import (
    render_globe_flat,
    render_globe_3d,
    globe_to_colour_map,
    globe_to_tile_colours,
)
from polygrid.globe_mesh import (
    build_terrain_layout_mesh,
    build_terrain_face_meshes,
    build_terrain_edge_mesh,
)

MOUNTAIN_PRESETS = {
    "mountain_range": MOUNTAIN_RANGE,
    "alpine_peaks": ALPINE_PEAKS,
    "rolling_hills": ROLLING_HILLS,
    "mesa_plateau": MESA_PLATEAU,
}


def _build_mountains(grid, store, preset: str, seed: int):
    """Generate mountain terrain and return colour map."""
    config = MOUNTAIN_PRESETS[preset]
    config = replace(config, seed=seed)
    generate_mountains(grid, store, config)
    return globe_to_colour_map(grid, store)


def _build_regions(grid, seed: int):
    """Partition globe into Voronoi regions, return colour map."""
    import random
    from polygrid.regions import partition_voronoi

    rng = random.Random(seed)
    face_ids = list(grid.faces.keys())
    n_regions = max(4, len(face_ids) // 15)
    seeds = rng.sample(face_ids, n_regions)
    region_map = partition_voronoi(grid, seeds)

    # Assign distinct colours per region
    palette = [
        (0.906, 0.298, 0.235),  # red
        (0.180, 0.800, 0.443),  # green
        (0.204, 0.596, 0.859),  # blue
        (0.608, 0.349, 0.714),  # purple
        (0.945, 0.769, 0.059),  # yellow
        (0.902, 0.494, 0.133),  # orange
        (0.098, 0.737, 0.612),  # teal
        (0.584, 0.647, 0.651),  # grey
        (0.827, 0.329, 0.510),  # pink
        (0.353, 0.282, 0.765),  # indigo
    ]
    colour_map = {}
    for i, region in enumerate(region_map.regions):
        c = palette[i % len(palette)]
        for fid in region.face_ids:
            colour_map[fid] = c
    return colour_map


def _build_rivers(grid, store, seed: int):
    """Generate mountains + rivers and return colour map with rivers marked blue."""
    from polygrid.rivers import RiverConfig, generate_rivers, carve_river_valleys

    config = replace(MOUNTAIN_RANGE, seed=seed)
    generate_mountains(grid, store, config)

    river_config = RiverConfig(
        min_accumulation=3, min_length=2, carve_depth=0.03, seed=seed,
    )
    network = generate_rivers(grid, store, river_config)
    if len(network) > 0:
        carve_river_valleys(grid, store, network, carve_depth=0.03)

    # Get base terrain colours, then overlay river faces in blue
    colour_map = globe_to_colour_map(grid, store)
    river_faces = network.all_river_face_ids()
    for fid in river_faces:
        colour_map[fid] = (0.15, 0.35, 0.75)

    return colour_map, network


def main() -> None:
    all_presets = list(MOUNTAIN_PRESETS.keys()) + ["regions", "rivers"]
    parser = argparse.ArgumentParser(description="Globe 3D terrain demo")
    parser.add_argument("--frequency", type=int, default=3, help="Goldberg frequency (default: 3)")
    parser.add_argument("--preset", choices=all_presets, default="mountain_range")
    parser.add_argument("--out", type=str, default="exports", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ramp", choices=["satellite", "topo"], default="satellite")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    freq = args.frequency
    print(f"Building globe grid (frequency={freq})...")
    grid = build_globe_grid(freq)
    pent = grid.metadata["pentagon_count"]
    hexa = grid.metadata["hexagon_count"]
    print(f"  → {len(grid.faces)} tiles ({pent} pent, {hexa} hex)")

    # Build schema — includes river fields if using rivers preset
    fields = [FieldDef("elevation", float, 0.0)]
    if args.preset == "rivers":
        fields.extend([
            FieldDef("river", bool, False),
            FieldDef("river_width", float, 0.0),
        ])
    schema = TileSchema(fields)
    store = TileDataStore(grid=grid, schema=schema)

    # Generate terrain / regions
    network = None
    if args.preset == "regions":
        print(f"Partitioning into Voronoi regions (seed={args.seed})...")
        colour_map = _build_regions(grid, args.seed)
    elif args.preset == "rivers":
        print(f"Generating mountains + rivers (seed={args.seed})...")
        colour_map, network = _build_rivers(grid, store, args.seed)
    else:
        print(f"Generating terrain (preset={args.preset}, seed={args.seed})...")
        colour_map = _build_mountains(grid, store, args.preset, args.seed)

    # Elevation stats (if applicable)
    if args.preset != "regions":
        elevations = [store.get(fid, "elevation") for fid in grid.faces]
        print(f"  → elevation range: [{min(elevations):.3f}, {max(elevations):.3f}]")
    if network is not None:
        river_faces = network.all_river_face_ids()
        print(f"  → river segments: {len(network)}, river faces: {len(river_faces)}")

    # ── Render 3D polyhedron (matplotlib) ──────────────────────────
    d3_path = out_dir / f"globe_f{freq}_{args.preset}_3d.png"
    print(f"Rendering 3D polyhedron → {d3_path}")
    # For region/river modes we supply colours directly, for mountain modes
    # we let render_globe_3d derive from the store
    if args.preset in ("regions", "rivers"):
        # Custom colour map — render_globe_3d accepts store, so we'll
        # write a quick wrapper: render with colour_map directly via
        # the low-level matplotlib path
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        polys = []
        facecolors = []
        for fid, face in grid.faces.items():
            verts_3d = []
            for vid in face.vertex_ids:
                v = grid.vertices[vid]
                verts_3d.append([v.x, v.y, v.z])
            polys.append(verts_3d)
            facecolors.append(colour_map.get(fid, (0.5, 0.5, 0.5)))
        collection = Poly3DCollection(polys, facecolors=facecolors,
                                       edgecolors="black", linewidths=0.3)
        ax.add_collection3d(collection)
        r = grid.radius * 1.05
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_zlim(-r, r)
        ax.set_box_aspect([1, 1, 1])
        ax.axis("off")
        ax.set_title(f"Globe f{freq} — {args.preset}")
        fig.savefig(str(d3_path), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    else:
        render_globe_3d(grid, store, d3_path, ramp=args.ramp)

    # ── Render flat map ────────────────────────────────────────────
    flat_path = out_dir / f"globe_f{freq}_{args.preset}_flat.png"
    print(f"Rendering flat projection → {flat_path}")
    if args.preset in ("regions", "rivers"):
        # Quick equirectangular with custom colours
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import RegularPolygon
        import math

        fig, ax = plt.subplots(figsize=(14, 7))
        for fid, face in grid.faces.items():
            lat_lon = grid.tile_lat_lon(fid)
            if lat_lon is None:
                continue
            lat, lon = lat_lon
            n_sides = len(face.vertex_ids)
            radius_patch = 360.0 / (len(grid.faces) ** 0.5) * 0.7
            patch = RegularPolygon(
                (lon, lat), numVertices=n_sides, radius=radius_patch,
                facecolor=colour_map.get(fid, (0.5, 0.5, 0.5)),
                edgecolor="black", linewidth=0.3,
            )
            ax.add_patch(patch)
        ax.set_xlim(-190, 190)
        ax.set_ylim(-100, 100)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.set_title(f"Globe f{freq} — {args.preset} (equirectangular)")
        fig.savefig(str(flat_path), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    else:
        render_globe_flat(grid, store, flat_path, ramp=args.ramp)

    # ── Export mesh metadata ───────────────────────────────────────
    mesh_meta_path = out_dir / f"globe_f{freq}_{args.preset}_mesh.json"
    print(f"Exporting mesh metadata → {mesh_meta_path}")

    # Build models-compatible mesh and export metadata
    layout_mesh = build_terrain_layout_mesh(grid, colour_map, radius=grid.radius)
    face_meshes = build_terrain_face_meshes(grid, colour_map, radius=grid.radius)
    edge_mesh = build_terrain_edge_mesh(grid, radius=grid.radius)

    meta = {
        "frequency": freq,
        "preset": args.preset,
        "seed": args.seed,
        "tile_count": len(grid.faces),
        "pentagon_count": pent,
        "hexagon_count": hexa,
        "layout_mesh": {
            "vertex_count": len(layout_mesh.vertex_data) // (layout_mesh.stride // 4),
            "index_count": len(layout_mesh.index_data),
            "stride": layout_mesh.stride,
        },
        "face_meshes_count": len(face_meshes),
        "edge_mesh": {
            "vertex_count": len(edge_mesh.vertex_data) // (edge_mesh.stride // 4),
            "index_count": len(edge_mesh.index_data),
            "stride": edge_mesh.stride,
        },
    }
    mesh_meta_path.write_text(json.dumps(meta, indent=2))

    # ── Export tile colours JSON ───────────────────────────────────
    colours_path = out_dir / f"globe_f{freq}_{args.preset}_colours.json"
    print(f"Exporting tile colours → {colours_path}")
    tile_colour_payload = {}
    for fid in grid.faces:
        rgb = colour_map.get(fid, (0.5, 0.5, 0.5))
        entry = {
            "color": [round(c, 4) for c in rgb],
        }
        lat_lon = grid.tile_lat_lon(fid)
        if lat_lon:
            entry["latitude_deg"] = round(lat_lon[0], 2)
            entry["longitude_deg"] = round(lat_lon[1], 2)
        if args.preset != "regions":
            entry["elevation"] = round(store.get(fid, "elevation"), 4)
        tile_colour_payload[fid] = entry
    colours_path.write_text(json.dumps(tile_colour_payload, indent=2))

    print("Done!")


if __name__ == "__main__":
    main()
