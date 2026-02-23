from __future__ import annotations

import argparse
from pathlib import Path

from .io import load_json, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PolyGrid CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Validate and optionally round-trip a grid")
    validate.add_argument("--in", dest="input_path", required=True)
    validate.add_argument("--out", dest="output_path")
    validate.add_argument("--strict", action="store_true")

    render = sub.add_parser("render", help="Render a grid to PNG")
    render.add_argument("--in", dest="input_path", required=True)
    render.add_argument("--out", dest="output_path", required=True)
    render.add_argument("--pent-axes", action="store_true")

    build_hex = sub.add_parser("build-hex", help="Build a pure hex grid")
    build_hex.add_argument("--rings", type=int, required=True)
    build_hex.add_argument("--out", dest="output_path", required=True)
    build_hex.add_argument("--render-out", dest="render_path")
    build_hex.add_argument("--pent-axes", action="store_true")

    build_pent = sub.add_parser("build-pent", help="Build a pentagon-centered grid")
    build_pent.add_argument("--rings", type=int, required=True)
    build_pent.add_argument("--out", dest="output_path", required=True)
    build_pent.add_argument("--render-out", dest="render_path")
    build_pent.add_argument("--embed", choices=["tutte", "none", "angle"], default="angle")
    build_pent.add_argument("--pent-axes", action="store_true")
    build_pent.add_argument("--diagnose", action="store_true")

    build_pent_all = sub.add_parser("build-pent-all", help="Build pentagon-centered grids for rings 0-3")
    build_pent_all.add_argument("--dir", dest="output_dir", default="exports")
    build_pent_all.add_argument("--embed", choices=["tutte", "none", "angle"], default="angle")
    build_pent_all.add_argument("--pent-axes", action="store_true")
    build_pent_all.add_argument("--diagnose", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "validate":
        grid = load_json(args.input_path)
        errors = grid.validate(strict=args.strict)
        if errors:
            for error in errors:
                print(error)
            raise SystemExit(1)
        if args.output_path:
            save_json(grid, args.output_path)
        print("OK")
    elif args.command == "render":
        from .render import render_png

        grid = load_json(args.input_path)
        render_png(grid, args.output_path, show_pent_axes=args.pent_axes)
        print(f"Saved {args.output_path}")
    elif args.command == "build-hex":
        from .builders import build_pure_hex_grid

        grid = build_pure_hex_grid(args.rings)
        save_json(grid, args.output_path)
        if args.render_path:
            from .render import render_png

            render_png(grid, args.render_path, show_pent_axes=args.pent_axes)
        if args.diagnose:
            from .diagnostics import ring_diagnostics, summarize_ring_stats

            stats = ring_diagnostics(grid, max_ring=args.rings)
            for ring, ring_stats in sorted(stats.items()):
                summary = summarize_ring_stats(ring_stats)
                print(f"ring {ring} diagnostics:")
                for key, value in summary.items():
                    print(f"  {key}: {value:.4f}")
        print(f"Saved {args.output_path}")
    elif args.command == "build-pent":
        from .builders import build_pentagon_centered_grid

        try:
            grid = build_pentagon_centered_grid(
                args.rings,
                embed=args.embed != "none",
                embed_mode=args.embed,
            )
        except RuntimeError as exc:
            print(exc)
            raise SystemExit(1)
        save_json(grid, args.output_path)
        if args.render_path:
            from .render import render_png

            render_png(grid, args.render_path, show_pent_axes=args.pent_axes)
        print(f"Saved {args.output_path}")
    elif args.command == "build-pent-all":
        from .builders import build_pentagon_centered_grid

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for rings in range(4):
            json_path = output_dir / f"pent_r{rings}.json"
            png_path = output_dir / f"pent_r{rings}.png"
            grid = build_pentagon_centered_grid(
                rings,
                embed=args.embed != "none",
                embed_mode=args.embed,
            )
            save_json(grid, json_path)
            from .render import render_png

            render_png(grid, png_path, show_pent_axes=args.pent_axes)
            if args.diagnose:
                from .diagnostics import ring_diagnostics, summarize_ring_stats

                stats = ring_diagnostics(grid, max_ring=rings)
                for ring, ring_stats in sorted(stats.items()):
                    summary = summarize_ring_stats(ring_stats)
                    print(f"ring {ring} diagnostics:")
                    for key, value in summary.items():
                        print(f"  {key}: {value:.4f}")
            print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
