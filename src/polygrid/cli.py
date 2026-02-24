from __future__ import annotations

import argparse
import json
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
    build_hex.add_argument("--diagnose", action="store_true")
    build_hex.add_argument("--diagnose-json", dest="diagnose_json")

    build_pent = sub.add_parser("build-pent", help="Build a pentagon-centered grid")
    build_pent.add_argument("--rings", type=int, required=True)
    build_pent.add_argument("--out", dest="output_path", required=True)
    build_pent.add_argument("--render-out", dest="render_path")
    build_pent.add_argument(
        "--embed",
        choices=["tutte", "tutte+optimise", "none", "angle"],
        default="tutte+optimise",
    )
    build_pent.add_argument("--pent-axes", action="store_true")
    build_pent.add_argument("--diagnose", action="store_true")
    build_pent.add_argument("--strict", action="store_true")
    build_pent.add_argument("--diagnose-json", dest="diagnose_json")

    build_pent_all = sub.add_parser("build-pent-all", help="Build pentagon-centered grids for rings 0-3")
    build_pent_all.add_argument("--dir", dest="output_dir", default="exports")
    build_pent_all.add_argument(
        "--embed",
        choices=["tutte", "tutte+optimise", "none", "angle"],
        default="tutte+optimise",
    )
    build_pent_all.add_argument("--pent-axes", action="store_true")
    build_pent_all.add_argument("--diagnose", action="store_true")
    build_pent_all.add_argument("--strict", action="store_true")
    build_pent_all.add_argument("--diagnose-json", dest="diagnose_json")

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
        from .diagnostics import diagnostics_report

        grid = build_pure_hex_grid(args.rings)
        save_json(grid, args.output_path)
        if args.render_path:
            from .render import render_png

            render_png(grid, args.render_path, show_pent_axes=args.pent_axes)
        if args.diagnose:
            from .diagnostics import (
                ring_diagnostics,
                summarize_ring_stats,
                min_face_signed_area,
                has_edge_crossings,
                ring_quality_gates,
            )

            stats = ring_diagnostics(grid, max_ring=args.rings)
            for ring, ring_stats in sorted(stats.items()):
                summary = summarize_ring_stats(ring_stats)
                print(f"ring {ring} diagnostics:")
                for key, value in summary.items():
                    print(f"  {key}: {value:.4f}")
                quality = ring_quality_gates(ring_stats)
                print("  quality gates:")
                print(
                    f"    inner_angle_ok: {quality['inner_angle_ok']} "
                    f"(mean {quality['inner_angle_mean']:.2f} vs target {quality['inner_angle_target']:.2f})"
                )
                print(
                    f"    pointy_angle_ok: {quality['pointy_angle_ok']} "
                    f"(mean {quality['pointy_angle_mean']:.2f} vs target {quality['pointy_angle_target']:.2f})"
                )
                print(
                    f"    protrude_ok: {quality['protrude_ok']} "
                    f"(rel_range {quality['protrude_rel_range']:.3f})"
                )
            print("quality gates:")
            print(f"  min_face_signed_area: {min_face_signed_area(grid):.4f}")
            print(f"  edge_crossings: {has_edge_crossings(grid)}")
        if args.diagnose_json:
            report = diagnostics_report(grid, max_ring=args.rings)
            Path(args.diagnose_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved {args.output_path}")
    elif args.command == "build-pent":
        from .builders import build_pentagon_centered_grid
        from .diagnostics import diagnostics_report

        if args.embed == "angle":
            print("Warning: --embed angle is experimental and may self-intersect for larger rings.")

        try:
            grid = build_pentagon_centered_grid(
                args.rings,
                embed=args.embed != "none",
                embed_mode=args.embed,
                validate_topology=args.strict,
            )
        except RuntimeError as exc:
            print(exc)
            raise SystemExit(1)
        save_json(grid, args.output_path)
        if args.render_path:
            from .render import render_png

            render_png(grid, args.render_path, show_pent_axes=args.pent_axes)
        if args.diagnose:
            from .diagnostics import (
                ring_diagnostics,
                summarize_ring_stats,
                min_face_signed_area,
                has_edge_crossings,
                ring_quality_gates,
            )

            stats = ring_diagnostics(grid, max_ring=args.rings)
            for ring, ring_stats in sorted(stats.items()):
                summary = summarize_ring_stats(ring_stats)
                print(f"ring {ring} diagnostics:")
                for key, value in summary.items():
                    print(f"  {key}: {value:.4f}")
                quality = ring_quality_gates(ring_stats)
                print("  quality gates:")
                print(
                    f"    inner_angle_ok: {quality['inner_angle_ok']} "
                    f"(mean {quality['inner_angle_mean']:.2f} vs target {quality['inner_angle_target']:.2f})"
                )
                print(
                    f"    pointy_angle_ok: {quality['pointy_angle_ok']} "
                    f"(mean {quality['pointy_angle_mean']:.2f} vs target {quality['pointy_angle_target']:.2f})"
                )
                print(
                    f"    protrude_ok: {quality['protrude_ok']} "
                    f"(rel_range {quality['protrude_rel_range']:.3f})"
                )
            print("quality gates:")
            print(f"  min_face_signed_area: {min_face_signed_area(grid):.4f}")
            print(f"  edge_crossings: {has_edge_crossings(grid)}")
        if args.diagnose_json:
            report = diagnostics_report(grid, max_ring=args.rings)
            Path(args.diagnose_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved {args.output_path}")
    elif args.command == "build-pent-all":
        from .builders import build_pentagon_centered_grid
        from .diagnostics import diagnostics_report

        if args.embed == "angle":
            print("Warning: --embed angle is experimental and may self-intersect for larger rings.")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for rings in range(4):
            json_path = output_dir / f"pent_r{rings}.json"
            png_path = output_dir / f"pent_r{rings}.png"
            grid = build_pentagon_centered_grid(
                rings,
                embed=args.embed != "none",
                embed_mode=args.embed,
                validate_topology=args.strict,
            )
            save_json(grid, json_path)
            from .render import render_png

            render_png(grid, png_path, show_pent_axes=args.pent_axes)
            if args.diagnose:
                from .diagnostics import (
                    ring_diagnostics,
                    summarize_ring_stats,
                    min_face_signed_area,
                    has_edge_crossings,
                    ring_quality_gates,
                )

                stats = ring_diagnostics(grid, max_ring=rings)
                for ring, ring_stats in sorted(stats.items()):
                    summary = summarize_ring_stats(ring_stats)
                    print(f"ring {ring} diagnostics:")
                    for key, value in summary.items():
                        print(f"  {key}: {value:.4f}")
                    quality = ring_quality_gates(ring_stats)
                    print("  quality gates:")
                    print(
                        f"    inner_angle_ok: {quality['inner_angle_ok']} "
                        f"(mean {quality['inner_angle_mean']:.2f} vs target {quality['inner_angle_target']:.2f})"
                    )
                    print(
                        f"    pointy_angle_ok: {quality['pointy_angle_ok']} "
                        f"(mean {quality['pointy_angle_mean']:.2f} vs target {quality['pointy_angle_target']:.2f})"
                    )
                    print(
                        f"    protrude_ok: {quality['protrude_ok']} "
                        f"(rel_range {quality['protrude_rel_range']:.3f})"
                    )
                print("quality gates:")
                print(f"  min_face_signed_area: {min_face_signed_area(grid):.4f}")
                print(f"  edge_crossings: {has_edge_crossings(grid)}")
            if args.diagnose_json:
                report = diagnostics_report(grid, max_ring=rings)
                diag_path = output_dir / f"pent_r{rings}_diagnostics.json"
                diag_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
