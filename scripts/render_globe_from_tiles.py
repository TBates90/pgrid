#!/usr/bin/env python3
"""Render a 3D globe from a directory of pre-rendered tile PNGs.

Skips all terrain/detail generation and goes straight from tile images
to the interactive 3D viewer.

The tile directory is resolved relative to ``EXPORT_DIR`` from
``.env``, so you only need to pass the sub-directory name.

Usage
-----
::

    # Generate tiles first:
    python scripts/render_polygrids.py -f 3 --detail-rings 4

    # Then view as a 3D globe (reads metadata from the export directory):
    python scripts/render_globe_from_tiles.py f3-d4

    # Save a flat atlas without launching the viewer:
    python scripts/render_globe_from_tiles.py f3-d4 --no-view

    # Custom tile/atlas size:
    python scripts/render_globe_from_tiles.py f3-d4 --tile-size 256

The tile directory must contain the polygon-cut atlas (``atlas.png``,
``uv_layout.json``) and metadata (``metadata.json``,
``globe_payload.json``) — all produced automatically by
``render_polygrids.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _script_utils import load_env as _load_env, _PROJECT_ROOT


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
    from polygrid.atlas_utils import fill_gutter, compute_atlas_layout

    columns, rows, atlas_w, atlas_h = compute_atlas_layout(
        n, tile_size, gutter,
    )
    slot_size = tile_size + 2 * gutter
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
            fill_gutter(atlas, slot_x, slot_y, tile_size, gutter)

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


# ═══════════════════════════════════════════════════════════════════
# Globe payload generation (minimal — geometry only)
# ═══════════════════════════════════════════════════════════════════

def _build_payload(frequency: int, seed: int, preset: str):
    """Build the globe grid + payload needed for 3D rendering."""
    from dataclasses import replace

    from polygrid.globe import build_globe_grid
    from polygrid.mountains import (
        ALPINE_PEAKS, MOUNTAIN_RANGE, ROLLING_HILLS, generate_mountains,
    )
    from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
    from polygrid.globe_export import export_globe_payload

    print(f"Building globe geometry (freq={frequency})...")
    grid = build_globe_grid(frequency)
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)

    presets = {
        "mountain_range": MOUNTAIN_RANGE,
        "alpine_peaks": ALPINE_PEAKS,
        "rolling_hills": ROLLING_HILLS,
    }
    config = replace(presets.get(preset, MOUNTAIN_RANGE), seed=seed)
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
        help="Sub-directory name under EXPORT_DIR (e.g. 'f3-d4'), "
             "or a full path to the tile directory.",
    )
    parser.add_argument(
        "-f", "--frequency", type=int, default=None,
        help="Goldberg polyhedron frequency (read from metadata.json if omitted)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (read from metadata.json if omitted)",
    )
    parser.add_argument(
        "--preset", default=None,
        choices=["mountain_range", "alpine_peaks", "rolling_hills"],
        help="Terrain preset (read from metadata.json if omitted)",
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
        "--subdivisions", type=int, default=5,
        help="Triangle subdivision level for sphere projection (default: 5)",
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
        "--uv-inset-px", type=float, default=0.0,
        help="UV polygon inset in atlas pixels before optional clamping "
             "(default: 0.0 for polygon-cut atlases which already have "
             "gutter + seam stitching; use 1.5 for raw atlases)",
    )
    parser.add_argument(
        "--no-view", action="store_true",
        help="Build atlas only, don't launch 3D viewer",
    )
    parser.add_argument(
        "--no-polygon-cut", action="store_true",
        help="Disable polygon-cut mode (pack a new atlas from raw tile PNGs "
             "instead of using the pre-built atlas.png + uv_layout.json)",
    )
    parser.add_argument(
        "--payload", type=str, default=None,
        help="Path to an existing globe_payload.json (skips globe generation)",
    )
    args = parser.parse_args()

    # Resolve tile directory: if the argument is a bare name (no path
    # separators), treat it as a sub-directory of EXPORT_DIR from .env.
    raw = args.tile_dir
    if os.sep not in raw and "/" not in raw:
        export_root = Path(_load_env("EXPORT_DIR", "./exports"))
        if not export_root.is_absolute():
            export_root = _PROJECT_ROOT / export_root
        tile_dir = export_root / raw
    else:
        tile_dir = Path(raw)

    if not tile_dir.is_dir():
        print(f"Error: {tile_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Derive effective polygon_cut flag (default on, unless --no-polygon-cut)
    polygon_cut = not args.no_polygon_cut

    t_total = time.perf_counter()

    # 0. Load metadata from tile directory (if present)
    metadata_path = tile_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        print(f"Loaded metadata from {metadata_path}")

    # Resolve frequency / seed / preset: CLI overrides > metadata > defaults
    frequency = args.frequency or metadata.get("frequency", 3)
    seed = args.seed if args.seed is not None else metadata.get("seed", 42)
    preset = args.preset or metadata.get("preset", "mountain_range")

    # 1. Load or generate globe payload
    payload_in = tile_dir / "globe_payload.json"
    if args.payload:
        payload_path = Path(args.payload)
        if not payload_path.exists():
            print(f"Error: {payload_path} not found", file=sys.stderr)
            sys.exit(1)
        print(f"Loading payload from {payload_path}...")
        payload = json.loads(payload_path.read_text())
        grid = None
    elif payload_in.exists():
        print(f"Loading payload from {payload_in}...")
        payload = json.loads(payload_in.read_text())
        grid = None
    else:
        grid, payload = _build_payload(frequency, seed, preset)

    # Extract face IDs from payload
    face_ids = [t["id"] for t in payload["tiles"]]
    print(f"Globe has {len(face_ids)} tiles")

    # 2. Discover tile PNGs
    found = [fid for fid in face_ids if (tile_dir / f"{fid}.png").exists()]
    print(f"Found {len(found)}/{len(face_ids)} tile PNGs in {tile_dir}")

    if not found and not polygon_cut:
        print("Error: no tile PNGs found!", file=sys.stderr)
        sys.exit(1)

    # 3. Pack atlas (or use pre-built polygon-cut atlas)
    if polygon_cut:
        # Use the pre-built polygon-cut atlas from render_polygrids.py
        atlas_path = tile_dir / "atlas.png"
        uv_path = tile_dir / "uv_layout.json"
        if not atlas_path.exists() or not uv_path.exists():
            print("Error: polygon-cut mode requires atlas.png + uv_layout.json",
                  file=sys.stderr)
            print("Run: render_polygrids.py first", file=sys.stderr)
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

    # Save payload if we generated it fresh
    if not args.payload and not payload_in.exists():
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

    print("Launching 3D viewer...")
    from polygrid.globe_renderer_v2 import render_globe_v2
    render_globe_v2(
        payload, atlas_path, uv_layout,
        title=title,
        subdivisions=args.subdivisions,
        width=args.width,
        height=args.height,
        uv_inset_px=args.uv_inset_px,
    )


if __name__ == "__main__":
    main()
