#!/usr/bin/env python3
"""Render a 3D globe from a directory of pre-rendered tile PNGs.

Skips all terrain/detail generation and goes straight from tile images
to the interactive 3D viewer.

Usage
-----
::

    # Generate tiles first (any of these):
    python scripts/render_polygrids.py -f 3 --detail-rings 3 --stitched -o exports/my_tiles
    python scripts/render_polygrids.py -f 3 --detail-rings 3 -o exports/my_tiles

    # Then view as a 3D globe:
    python scripts/render_globe_from_tiles.py exports/my_tiles -f 3

    # With v2 renderer (newer, better quality):
    python scripts/render_globe_from_tiles.py exports/my_tiles -f 3 --v2

    # Save a flat atlas without launching the viewer:
    python scripts/render_globe_from_tiles.py exports/my_tiles -f 3 --no-view

    # Custom tile/atlas size:
    python scripts/render_globe_from_tiles.py exports/my_tiles -f 3 --tile-size 256

The tile directory must contain PNGs named ``t0.png``, ``t1.png``, etc.
— one per globe face.  The ``--frequency`` flag must match the globe
frequency used to generate the tiles.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════
# Atlas packing (from tile PNGs → atlas + UV layout)
# ═══════════════════════════════════════════════════════════════════

def _pack_atlas(
    tile_dir: Path,
    face_ids: list[str],
    *,
    tile_size: int = 256,
    gutter: int = 4,
) -> tuple[Path, dict[str, tuple[float, float, float, float]]]:
    """Pack tile PNGs into a texture atlas with gutter padding.

    Returns ``(atlas_path, uv_layout)`` where *uv_layout* maps
    ``face_id → (u_min, v_min, u_max, v_max)`` in atlas UV space.
    """
    from PIL import Image

    # Verify all tile PNGs exist
    tile_paths: dict[str, Path] = {}
    missing = []
    for fid in face_ids:
        p = tile_dir / f"{fid}.png"
        if p.exists():
            tile_paths[fid] = p
        else:
            missing.append(fid)

    if missing:
        print(f"  ⚠ Missing {len(missing)} tile PNGs: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    if not tile_paths:
        raise FileNotFoundError(f"No tile PNGs found in {tile_dir}")

    ordered_ids = [fid for fid in face_ids if fid in tile_paths]
    n = len(ordered_ids)

    # Compute grid layout
    columns = max(1, math.isqrt(n))
    if columns * columns < n:
        columns += 1
    rows = math.ceil(n / columns)

    slot_size = tile_size + 2 * gutter
    atlas_w = columns * slot_size
    atlas_h = rows * slot_size
    atlas = Image.new("RGB", (atlas_w, atlas_h), (128, 128, 128))

    uv_layout: dict[str, tuple[float, float, float, float]] = {}

    for idx, fid in enumerate(ordered_ids):
        col = idx % columns
        row = idx // columns
        slot_x = col * slot_size
        slot_y = row * slot_size

        tile_img = Image.open(tile_paths[fid]).convert("RGB")
        tile_img = tile_img.resize((tile_size, tile_size), Image.LANCZOS)

        # Paste tile into centre of slot
        atlas.paste(tile_img, (slot_x + gutter, slot_y + gutter))

        # Fill gutter by clamping edge pixels outward
        if gutter > 0:
            _fill_gutter(atlas, slot_x, slot_y, tile_size, gutter)

        # UV coordinates (inner tile region)
        inner_x = slot_x + gutter
        inner_y = slot_y + gutter
        u_min = inner_x / atlas_w
        u_max = (inner_x + tile_size) / atlas_w
        v_min = 1.0 - (inner_y + tile_size) / atlas_h
        v_max = 1.0 - inner_y / atlas_h
        uv_layout[fid] = (u_min, v_min, u_max, v_max)

    atlas_path = tile_dir / "atlas.png"
    atlas.save(str(atlas_path))
    return atlas_path, uv_layout


def _fill_gutter(atlas, slot_x: int, slot_y: int,
                 tile_size: int, gutter: int) -> None:
    """Fill gutter pixels by clamping edge pixels outward."""
    inner_x = slot_x + gutter
    inner_y = slot_y + gutter

    # Top gutter
    top_strip = atlas.crop((inner_x, inner_y, inner_x + tile_size, inner_y + 1))
    for g in range(gutter):
        atlas.paste(top_strip, (inner_x, slot_y + g))

    # Bottom gutter
    bot_y = inner_y + tile_size - 1
    bot_strip = atlas.crop((inner_x, bot_y, inner_x + tile_size, bot_y + 1))
    for g in range(gutter):
        atlas.paste(bot_strip, (inner_x, inner_y + tile_size + g))

    # Left gutter (full height including top/bottom gutter)
    full_top = slot_y
    full_bot = slot_y + tile_size + 2 * gutter
    left_strip = atlas.crop((inner_x, full_top, inner_x + 1, full_bot))
    for g in range(gutter):
        atlas.paste(left_strip, (slot_x + g, full_top))

    # Right gutter
    right_x = inner_x + tile_size - 1
    right_strip = atlas.crop((right_x, full_top, right_x + 1, full_bot))
    for g in range(gutter):
        atlas.paste(right_strip, (inner_x + tile_size + g, full_top))


# ═══════════════════════════════════════════════════════════════════
# Globe payload generation (minimal — geometry only)
# ═══════════════════════════════════════════════════════════════════

def _build_payload(frequency: int, seed: int, preset: str):
    """Build the globe grid + payload needed for 3D rendering."""
    from polygrid.globe import build_globe_grid
    from polygrid.mountains import MountainConfig, generate_mountains
    from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
    from polygrid.globe_export import export_globe_payload

    print(f"Building globe geometry (freq={frequency})...")
    grid = build_globe_grid(frequency)
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)

    presets = {
        "mountain_range": MountainConfig(
            seed=seed, ridge_frequency=2.0, ridge_octaves=4,
            peak_elevation=1.0, base_elevation=0.0,
        ),
        "alpine_peaks": MountainConfig(
            seed=seed, ridge_frequency=3.0, ridge_octaves=5,
            peak_elevation=1.0, base_elevation=0.1,
        ),
        "rolling_hills": MountainConfig(
            seed=seed, ridge_frequency=1.5, ridge_octaves=3,
            peak_elevation=0.5, base_elevation=0.2,
        ),
    }
    config = presets.get(preset, presets["mountain_range"])
    generate_mountains(grid, store, config)

    payload = export_globe_payload(grid, store, ramp="satellite")
    print(f"  → {len(grid.faces)} tiles")
    return grid, payload


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Render a 3D globe from pre-rendered tile PNGs.",
    )
    parser.add_argument(
        "tile_dir", type=str,
        help="Directory containing tile PNGs (t0.png, t1.png, ...)",
    )
    parser.add_argument(
        "-f", "--frequency", type=int, default=3,
        help="Goldberg polyhedron frequency (must match the tiles)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (for globe terrain geometry, default: 42)",
    )
    parser.add_argument(
        "--preset", default="mountain_range",
        choices=["mountain_range", "alpine_peaks", "rolling_hills"],
        help="Terrain preset (for globe geometry, default: mountain_range)",
    )
    parser.add_argument(
        "--tile-size", type=int, default=256,
        help="Tile size in atlas pixels (default: 256)",
    )
    parser.add_argument(
        "--gutter", type=int, default=4,
        help="Atlas gutter pixels (default: 4)",
    )
    parser.add_argument(
        "--subdivisions", type=int, default=3,
        help="Triangle subdivision level for v2 renderer (default: 3)",
    )
    parser.add_argument(
        "--width", type=int, default=900,
        help="Window width (default: 900)",
    )
    parser.add_argument(
        "--height", type=int, default=700,
        help="Window height (default: 700)",
    )
    parser.add_argument(
        "--v2", action="store_true",
        help="Use v2 renderer (newer, better sphere projection)",
    )
    parser.add_argument(
        "--no-view", action="store_true",
        help="Build atlas only, don't launch 3D viewer",
    )
    parser.add_argument(
        "--polygon-cut", action="store_true",
        help="Use polygon-cut atlas (atlas.png + uv_layout.json already "
             "present in tile_dir, generated by render_polygrids.py --polygon-cut)",
    )
    parser.add_argument(
        "--payload", type=str, default=None,
        help="Path to an existing globe_payload.json (skips globe generation)",
    )
    args = parser.parse_args()

    tile_dir = Path(args.tile_dir)
    if not tile_dir.is_dir():
        print(f"Error: {tile_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    t_total = time.perf_counter()

    # 1. Load or generate globe payload
    if args.payload:
        payload_path = Path(args.payload)
        if not payload_path.exists():
            print(f"Error: {payload_path} not found", file=sys.stderr)
            sys.exit(1)
        print(f"Loading payload from {payload_path}...")
        payload = json.loads(payload_path.read_text())
        grid = None
    else:
        grid, payload = _build_payload(args.frequency, args.seed, args.preset)

    # Extract face IDs from payload
    face_ids = [t["id"] for t in payload["tiles"]]
    print(f"Globe has {len(face_ids)} tiles")

    # 2. Discover tile PNGs
    found = [fid for fid in face_ids if (tile_dir / f"{fid}.png").exists()]
    print(f"Found {len(found)}/{len(face_ids)} tile PNGs in {tile_dir}")

    if not found and not args.polygon_cut:
        print("Error: no tile PNGs found!", file=sys.stderr)
        sys.exit(1)

    # 3. Pack atlas (or use pre-built polygon-cut atlas)
    if args.polygon_cut:
        # Use the pre-built polygon-cut atlas from render_polygrids.py --polygon-cut
        atlas_path = tile_dir / "atlas.png"
        uv_path = tile_dir / "uv_layout.json"
        if not atlas_path.exists() or not uv_path.exists():
            print("Error: --polygon-cut requires atlas.png + uv_layout.json",
                  file=sys.stderr)
            print("Run: render_polygrids.py --polygon-cut first", file=sys.stderr)
            sys.exit(1)
        uv_layout = json.loads(uv_path.read_text())
        print(f"Using polygon-cut atlas: {atlas_path}")
    else:
        print(f"Packing atlas (tile_size={args.tile_size}, gutter={args.gutter})...")
        t0 = time.perf_counter()
        atlas_path, uv_layout = _pack_atlas(
            tile_dir, face_ids,
            tile_size=args.tile_size,
            gutter=args.gutter,
        )
        print(f"  → {atlas_path} in {time.perf_counter() - t0:.2f}s")

        # Save UV layout alongside atlas
        uv_path = tile_dir / "uv_layout.json"
        uv_path.write_text(json.dumps(uv_layout, indent=2))
        print(f"  → UV layout: {uv_path}")

    # Save payload if we generated it
    if not args.payload:
        payload_out = tile_dir / "globe_payload.json"
        payload_out.write_text(json.dumps(payload, indent=2))
        print(f"  → Payload: {payload_out}")

    elapsed = time.perf_counter() - t_total
    print(f"\nAtlas ready in {elapsed:.2f}s")

    if args.no_view:
        print(f"Done. Atlas: {atlas_path}")
        return

    # 4. Launch 3D viewer
    freq = payload["metadata"]["frequency"]
    tile_count = payload["metadata"]["tile_count"]
    title = f"Polygrid Globe — freq={freq}, {tile_count} tiles"

    if args.v2:
        print("Launching v2 3D viewer...")
        from polygrid.globe_renderer_v2 import render_globe_v2
        render_globe_v2(
            payload, atlas_path, uv_layout,
            title=title,
            subdivisions=args.subdivisions,
            width=args.width,
            height=args.height,
        )
    else:
        print("Launching 3D viewer...")
        from polygrid.globe_renderer import render_textured_globe_opengl
        render_textured_globe_opengl(
            payload, atlas_path, uv_layout,
            title=title,
            width=args.width,
            height=args.height,
        )


if __name__ == "__main__":
    main()
