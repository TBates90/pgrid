"""PolyGrid command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from .io import load_json, save_json

if TYPE_CHECKING:
    from .polygrid import PolyGrid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PolyGrid CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Validate a grid")
    validate.add_argument("--in", dest="input_path", required=True)
    validate.add_argument("--out", dest="output_path")
    validate.add_argument("--strict", action="store_true")

    render = sub.add_parser("render", help="Render a grid to PNG")
    render.add_argument("--in", dest="input_path", required=True)
    render.add_argument("--out", dest="output_path", required=True)
    render.add_argument("--pent-axes", action="store_true")

    build_hex = sub.add_parser("build-hex", help="Build a pure hex grid (6-sided)")
    build_hex.add_argument("--rings", type=int, required=True)
    build_hex.add_argument("--out", dest="output_path", required=True)
    build_hex.add_argument("--render-out", dest="render_path")
    build_hex.add_argument("--pent-axes", action="store_true")
    build_hex.add_argument("--diagnose", action="store_true")
    build_hex.add_argument("--diagnose-json", dest="diagnose_json")

    build_pent = sub.add_parser("build-pent", help="Build a pentagon-centred grid (5-sided)")
    build_pent.add_argument("--rings", type=int, required=True)
    build_pent.add_argument("--out", dest="output_path", required=True)
    build_pent.add_argument("--render-out", dest="render_path")
    build_pent.add_argument(
        "--embed",
        choices=["tutte", "tutte+optimise", "none"],
        default="tutte+optimise",
    )
    build_pent.add_argument("--pent-axes", action="store_true")
    build_pent.add_argument("--diagnose", action="store_true")
    build_pent.add_argument("--strict", action="store_true")
    build_pent.add_argument("--diagnose-json", dest="diagnose_json")

    build_pent_all = sub.add_parser("build-pent-all", help="Build pentagon grids for rings 0-N")
    build_pent_all.add_argument("--dir", dest="output_dir", default="exports")
    build_pent_all.add_argument("--max-rings", type=int, default=3)
    build_pent_all.add_argument(
        "--embed",
        choices=["tutte", "tutte+optimise", "none"],
        default="tutte+optimise",
    )
    build_pent_all.add_argument("--pent-axes", action="store_true")
    build_pent_all.add_argument("--diagnose", action="store_true")
    build_pent_all.add_argument("--strict", action="store_true")
    build_pent_all.add_argument("--diagnose-json", dest="diagnose_json")

    assembly = sub.add_parser("assembly", help="Build pent+hex assembly with visualisation")
    assembly.add_argument("--rings", type=int, default=3, help="Rings per component grid")
    assembly.add_argument("--out", dest="output_path", default="exports/assembly_demo.png",
                          help="Output path for 4-panel PNG")
    assembly.add_argument("--dpi", type=int, default=150)

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
        from .visualize import render_png
        grid = load_json(args.input_path)
        render_png(grid, args.output_path, show_pent_axes=args.pent_axes)
        print(f"Saved {args.output_path}")

    elif args.command == "build-hex":
        _cmd_build_hex(args)

    elif args.command == "build-pent":
        _cmd_build_pent(args)

    elif args.command == "build-pent-all":
        _cmd_build_pent_all(args)

    elif args.command == "assembly":
        _cmd_assembly(args)


def _cmd_build_hex(args) -> None:
    from .builders import build_pure_hex_grid
    from .diagnostics import diagnostics_report

    grid = build_pure_hex_grid(args.rings)
    save_json(grid, args.output_path)
    if args.render_path:
        from .visualize import render_png
        render_png(grid, args.render_path, show_pent_axes=args.pent_axes)
    if args.diagnose:
        _print_diagnostics(grid, args.rings)
    if args.diagnose_json:
        report = diagnostics_report(grid, max_ring=args.rings)
        Path(args.diagnose_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output_path}")


def _cmd_build_pent(args) -> None:
    from .builders import build_pentagon_centered_grid
    from .diagnostics import diagnostics_report

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
        from .visualize import render_png
        render_png(grid, args.render_path, show_pent_axes=args.pent_axes)
    if args.diagnose:
        _print_diagnostics(grid, args.rings)
    if args.diagnose_json:
        report = diagnostics_report(grid, max_ring=args.rings)
        Path(args.diagnose_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output_path}")


def _cmd_build_pent_all(args) -> None:
    from .builders import build_pentagon_centered_grid
    from .diagnostics import diagnostics_report

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for rings in range(args.max_rings + 1):
        json_path = output_dir / f"pent_r{rings}.json"
        png_path = output_dir / f"pent_r{rings}.png"
        grid = build_pentagon_centered_grid(
            rings,
            embed=args.embed != "none",
            embed_mode=args.embed,
            validate_topology=args.strict,
        )
        save_json(grid, json_path)

        from .visualize import render_png
        render_png(grid, png_path, show_pent_axes=args.pent_axes)

        if args.diagnose:
            diag_lines = _diagnostics_lines(grid, rings)
            diag_path = output_dir / f"pent_r{rings}_diagnostics.txt"
            diag_path.write_text("\n".join(diag_lines) + "\n", encoding="utf-8")

        if args.diagnose_json:
            report = diagnostics_report(grid, max_ring=rings)
            diag_path = output_dir / f"pent_r{rings}_diagnostics.json"
            diag_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(f"Saved {json_path}")


def _print_diagnostics(grid: "PolyGrid", max_ring: int) -> None:
    for line in _diagnostics_lines(grid, max_ring):
        print(line)


def _diagnostics_lines(grid: "PolyGrid", max_ring: int) -> list[str]:
    from .diagnostics import (
        ring_diagnostics,
        summarize_ring_stats,
        min_face_signed_area,
        has_edge_crossings,
        ring_quality_gates,
    )

    lines: list[str] = []
    stats = ring_diagnostics(grid, max_ring=max_ring)
    for ring, ring_stats in sorted(stats.items()):
        summary = summarize_ring_stats(ring_stats)
        lines.append(f"ring {ring} diagnostics:")
        for key, value in summary.items():
            lines.append(f"  {key}: {value:.4f}")
        quality = ring_quality_gates(ring_stats)
        lines.append("  quality gates:")
        lines.append(
            f"    inner_angle_ok: {quality['inner_angle_ok']} "
            f"(mean {quality['inner_angle_mean']:.2f} vs target {quality['inner_angle_target']:.2f})"
        )
        lines.append(
            f"    pointy_angle_ok: {quality['pointy_angle_ok']} "
            f"(mean {quality['pointy_angle_mean']:.2f} vs target {quality['pointy_angle_target']:.2f})"
        )
        lines.append(
            f"    protrude_ok: {quality['protrude_ok']} "
            f"(rel_range {quality['protrude_rel_range']:.3f})"
        )
    lines.append("quality gates:")
    lines.append(f"  min_face_signed_area: {min_face_signed_area(grid):.4f}")
    lines.append(f"  edge_crossings: {has_edge_crossings(grid)}")
    return lines


def _cmd_assembly(args) -> None:
    from .assembly import pent_hex_assembly
    from .transforms import apply_voronoi
    from .visualize import render_assembly_panels

    print(f"Building pent+hex assembly with {args.rings} rings …")
    plan = pent_hex_assembly(rings=args.rings)
    composite = plan.build()

    print(f"Merged: {len(composite.merged.faces)} faces, "
          f"{len(composite.merged.vertices)} vertices")

    print("Computing Voronoi overlay …")
    overlay = apply_voronoi(composite.merged)

    print(f"Rendering to {args.output_path} …")
    render_assembly_panels(plan, args.output_path, overlay=overlay,
                           dpi=args.dpi, figsize=(28, 7))
    print("Done ✓")


if __name__ == "__main__":
    main()
