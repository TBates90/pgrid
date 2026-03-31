#!/usr/bin/env python3
"""Audit canonical detail-cell metadata coverage in pgrid exports.

This script reports where runtime decode still needs legacy local-ID fallback
because canonical ``detail_index`` metadata is missing or invalid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from polygrid.rendering.detail_cells_audit import audit_detail_cells_payload


def _discover_targets(raw_dirs: list[str]) -> list[Path]:
    targets: list[Path] = []
    for raw in raw_dirs:
        path = Path(raw)
        if not path.exists():
            print(f"WARNING: {path} does not exist - skipping")
            continue
        if (path / "metadata.json").exists():
            targets.append(path)
            continue
        children = sorted(
            child
            for child in path.iterdir()
            if child.is_dir() and (child / "metadata.json").exists()
        )
        if children:
            targets.extend(children)
        else:
            print(
                f"WARNING: {path} has no metadata.json and no children with it - skipping"
            )
    return targets


def _status_for(report: dict[str, object]) -> str:
    if bool(report.get("missing_detail_cells_file", False)):
        return "FAIL"
    fallback_cells = int(report.get("fallback_cells", 0) or 0)
    non_contiguous_tiles = int(report.get("non_contiguous_tiles", 0) or 0)
    if fallback_cells > 0 or non_contiguous_tiles > 0:
        return "WARN"
    return "OK"


def _audit_export_dir(export_dir: Path) -> dict[str, object]:
    detail_cells_path = export_dir / "detail_cells.json"
    if not detail_cells_path.exists():
        return {
            "export": export_dir.name,
            "path": str(export_dir),
            "missing_detail_cells_file": True,
            "tiles": 0,
            "cells": 0,
            "fallback_tiles": 0,
            "fallback_cells": 0,
            "non_contiguous_tiles": 0,
            "canonical_index_coverage": 0.0,
        }

    try:
        payload = json.loads(detail_cells_path.read_text())
    except json.JSONDecodeError as exc:
        return {
            "export": export_dir.name,
            "path": str(export_dir),
            "missing_detail_cells_file": True,
            "error": f"Invalid JSON: {exc}",
            "tiles": 0,
            "cells": 0,
            "fallback_tiles": 0,
            "fallback_cells": 0,
            "non_contiguous_tiles": 0,
            "canonical_index_coverage": 0.0,
        }

    summary = audit_detail_cells_payload(payload)
    summary["export"] = export_dir.name
    summary["path"] = str(export_dir)
    summary["missing_detail_cells_file"] = False
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit detail_cells canonical index coverage for one or more exports."
    )
    parser.add_argument(
        "dirs",
        nargs="+",
        metavar="DIR",
        help=(
            "Export directory (or parent directory containing multiple exports). "
            "Children with metadata.json are auto-discovered."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON report instead of text table.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when WARN or FAIL rows are found.",
    )
    args = parser.parse_args()

    targets = _discover_targets(args.dirs)
    if not targets:
        print("No valid export directories found.")
        sys.exit(1)

    reports = [_audit_export_dir(target) for target in targets]

    total_exports = len(reports)
    fail_exports = 0
    warn_exports = 0
    total_tiles = 0
    total_cells = 0
    total_fallback_cells = 0

    for report in reports:
        status = _status_for(report)
        if status == "FAIL":
            fail_exports += 1
        elif status == "WARN":
            warn_exports += 1
        total_tiles += int(report.get("tiles", 0) or 0)
        total_cells += int(report.get("cells", 0) or 0)
        total_fallback_cells += int(report.get("fallback_cells", 0) or 0)

    if args.json:
        print(
            json.dumps(
                {
                    "summary": {
                        "exports": total_exports,
                        "ok": total_exports - fail_exports - warn_exports,
                        "warn": warn_exports,
                        "fail": fail_exports,
                        "tiles": total_tiles,
                        "cells": total_cells,
                        "fallback_cells": total_fallback_cells,
                        "canonical_index_coverage": (
                            1.0
                            if total_cells == 0
                            else max(
                                0.0,
                                float(total_cells - total_fallback_cells)
                                / float(total_cells),
                            )
                        ),
                    },
                    "exports": reports,
                },
                indent=2,
            )
        )
    else:
        for report in reports:
            status = _status_for(report)
            export_name = str(report.get("export", "<unknown>"))
            coverage = float(report.get("canonical_index_coverage", 0.0) or 0.0)
            fallback_cells = int(report.get("fallback_cells", 0) or 0)
            non_contiguous = int(report.get("non_contiguous_tiles", 0) or 0)
            print(
                f"{status:<5} {export_name:<12} "
                f"coverage={coverage:.1%} fallback_cells={fallback_cells} "
                f"non_contiguous_tiles={non_contiguous}"
            )
        total_coverage = (
            1.0
            if total_cells == 0
            else max(0.0, float(total_cells - total_fallback_cells) / float(total_cells))
        )
        print(
            "\n"
            f"Summary: exports={total_exports} ok={total_exports - fail_exports - warn_exports} "
            f"warn={warn_exports} fail={fail_exports} "
            f"cells={total_cells} fallback_cells={total_fallback_cells} "
            f"coverage={total_coverage:.1%}"
        )

    if fail_exports > 0:
        sys.exit(2)
    if args.strict and warn_exports > 0:
        sys.exit(3)


if __name__ == "__main__":
    main()
