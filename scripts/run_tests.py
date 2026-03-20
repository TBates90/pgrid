#!/usr/bin/env python3
# TODO REVIEW — Custom test runner.  May be useful, but references dead test
#   files in tier/phase lists.  Prune to match live test set or remove if
#   plain pytest is sufficient.
"""
PolyGrid Test Runner — grouped progress output with timing.

Usage:
    python scripts/run_tests.py              # run all tests
    python scripts/run_tests.py --fast       # only fast-tier files (<3s)
    python scripts/run_tests.py --phase 13   # only Phase 13 tests
    python scripts/run_tests.py --file noise # files matching pattern
    python scripts/run_tests.py --summary    # just the final table
    python scripts/run_tests.py -v           # verbose (show test names)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

# ── Colours ──────────────────────────────────────────────────────
USE_COLOUR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOUR else text

def green(t: str) -> str: return _c("32", t)
def red(t: str) -> str: return _c("31", t)
def yellow(t: str) -> str: return _c("33", t)
def cyan(t: str) -> str: return _c("36", t)
def bold(t: str) -> str: return _c("1", t)
def dim(t: str) -> str: return _c("2", t)


# ── Test file → group mapping ───────────────────────────────────
# Each group: (display_name, [test_files], tier)
# tier: "fast" (<3s), "medium" (3-20s), "slow" (>20s)

GROUPS: list[tuple[str, list[str], str]] = [
    ("Phase 1-4 — Core Topology & Transforms", [
        "test_core_topology.py",
        "test_stitching.py",
        "test_assembly.py",
        "test_macro_edges.py",
        "test_pentagon_centered.py",
        "test_transforms.py",
        "test_diagnostics.py",
        "test_visualize.py",
    ], "fast"),

    ("Phase 2 — Goldberg Topology", [
        "test_goldberg.py",
    ], "medium"),

    ("Phase 5-7 — Tile Data & Terrain", [
        "test_tile_data.py",
        "test_regions.py",
        "test_noise.py",
        "test_heightmap.py",
        "test_mountains.py",
        "test_rivers.py",
        "test_pipeline.py",
        "test_terrain_render.py",
        "test_determinism.py",
    ], "fast"),

    ("Phase 8-9 — Globe & Export", [
        "test_globe.py",
    ], "slow"),

    ("Phase 10 — Sub-Tile Detail", [
        "test_tile_detail.py",
        "test_detail_render.py",
        "test_detail_perf.py",
    ], "medium"),

    ("Phase 11 — Cohesive Terrain", [
        "test_detail_terrain.py",
        "test_detail_terrain_3d.py",
        "test_terrain_patches.py",
        "test_globe_terrain.py",
        "test_region_stitch.py",
        "test_render_enhanced.py",
        "test_texture_pipeline.py",
    ], "medium"),

    ("Phase 12-13 — Rendering & PBR", [
        "test_globe_renderer_v2.py",
        "test_phase13_rendering.py",
    ], "medium"),

    ("Phase 14 — Biome Features", [
        "test_biome_scatter.py",
        "test_biome_render.py",
        "test_biome_pipeline.py",
        "test_biome_continuity.py",
    ], "medium"),
]

# Build reverse lookup: filename → (group_name, tier)
_FILE_GROUP: dict[str, tuple[str, str]] = {}
for _gname, _files, _tier in GROUPS:
    for _f in _files:
        _FILE_GROUP[_f] = (_gname, _tier)


# ── Phase number mapping (for --phase filter) ───────────────────
PHASE_MAP: dict[int, list[str]] = {}
for _gname, _files, _ in GROUPS:
    # Extract phase numbers from group name, e.g. "Phase 1-4" → [1,2,3,4]
    _match = re.search(r"Phase (\d+)(?:-(\d+))?", _gname)
    if _match:
        _lo = int(_match.group(1))
        _hi = int(_match.group(2)) if _match.group(2) else _lo
        for _p in range(_lo, _hi + 1):
            PHASE_MAP.setdefault(_p, []).extend(_files)


# ── Data classes ─────────────────────────────────────────────────
@dataclass
class FileResult:
    filename: str
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    elapsed: float = 0.0
    returncode: int = 0

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass
class GroupResult:
    name: str
    tier: str
    file_results: list[FileResult] = field(default_factory=list)

    @property
    def total_tests(self) -> int:
        return sum(r.total for r in self.file_results)

    @property
    def total_passed(self) -> int:
        return sum(r.passed for r in self.file_results)

    @property
    def total_failed(self) -> int:
        return sum(r.failed + r.errors for r in self.file_results)

    @property
    def elapsed(self) -> float:
        return sum(r.elapsed for r in self.file_results)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.file_results)


# ── Test execution ───────────────────────────────────────────────
def _parse_pytest_line(line: str) -> tuple[int, int, int, int]:
    """Parse pytest summary line like '52 passed, 1 failed in 1.23s'."""
    passed = failed = errors = skipped = 0
    m = re.search(r"(\d+) passed", line)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+) failed", line)
    if m:
        failed = int(m.group(1))
    m = re.search(r"(\d+) error", line)
    if m:
        errors = int(m.group(1))
    m = re.search(r"(\d+) skipped", line)
    if m:
        skipped = int(m.group(1))
    return passed, failed, errors, skipped


def run_file(filename: str, verbose: bool = False) -> FileResult:
    """Run a single test file and return results."""
    filepath = TESTS_DIR / filename
    if not filepath.exists():
        return FileResult(filename=filename, errors=1, returncode=1)

    cmd = [PYTHON, "-m", "pytest", str(filepath), "-q", "--tb=short", "--no-header"]
    if verbose:
        cmd.append("-v")

    t0 = time.monotonic()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    elapsed = time.monotonic() - t0

    # Parse output for counts
    output = result.stdout + result.stderr
    lines = output.strip().splitlines()

    passed = failed = errors = skipped = 0
    for line in reversed(lines):
        if "passed" in line or "failed" in line or "error" in line:
            passed, failed, errors, skipped = _parse_pytest_line(line)
            break

    return FileResult(
        filename=filename,
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        elapsed=elapsed,
        returncode=result.returncode,
    )


def run_group_inline(
    group_name: str,
    tier: str,
    files: list[str],
    verbose: bool = False,
    summary_only: bool = False,
) -> GroupResult:
    """Run all files in a group as a single pytest invocation for cache sharing."""
    filepaths = [str(TESTS_DIR / f) for f in files if (TESTS_DIR / f).exists()]
    if not filepaths:
        return GroupResult(name=group_name, tier=tier)

    if not summary_only:
        print()
        print(f"  {bold(group_name)}")

    gr = GroupResult(name=group_name, tier=tier)

    # Run files individually for per-file timing and progress
    for filename in files:
        filepath = TESTS_DIR / filename
        if not filepath.exists():
            continue

        t0 = time.monotonic()
        cmd = [PYTHON, "-m", "pytest", str(filepath), "--tb=short", "--no-header"]
        if verbose:
            cmd.append("-v")

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        elapsed = time.monotonic() - t0

        output = result.stdout + result.stderr
        lines = output.strip().splitlines()

        passed = failed = errors = skipped = 0
        for line in reversed(lines):
            if "passed" in line or "failed" in line or "error" in line:
                passed, failed, errors, skipped = _parse_pytest_line(line)
                break

        fr = FileResult(
            filename=filename,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            elapsed=elapsed,
            returncode=result.returncode,
        )
        gr.file_results.append(fr)

        if not summary_only:
            _print_file_result(fr)

        # If verbose, also show any failure output
        if verbose and not fr.ok:
            for line in lines:
                if line.strip() and not line.startswith("="):
                    print(f"        {line}")

    if not summary_only and gr.file_results:
        total = gr.total_tests
        elapsed = gr.elapsed
        print(f"    {'':38s} {'─' * 24}")
        print(f"    {'':38s} {total:>4} tests  {_fmt_time(elapsed):>7s}")

    return gr


def _print_file_result(fr: FileResult) -> None:
    """Print a single file result line."""
    icon = green("✅") if fr.ok else red("❌")
    name = fr.filename
    dots = "." * max(1, 38 - len(name))
    count_str = f"{fr.total:>4} {'test' if fr.total == 1 else 'tests'}"
    time_str = _fmt_time(fr.elapsed)

    extra = ""
    if fr.failed > 0:
        extra = red(f"  ({fr.failed} failed)")
    elif fr.errors > 0:
        extra = red(f"  ({fr.errors} errors)")

    print(f"    {icon} {name} {dim(dots)} {count_str}  {time_str}{extra}")


def _fmt_time(seconds: float) -> str:
    """Format seconds as a human-friendly string."""
    if seconds < 60:
        return f"{seconds:>5.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:04.1f}s"


# ── Banner & summary ─────────────────────────────────────────────
def _print_banner(total_tests: int, total_files: int) -> None:
    width = 58
    print()
    print(bold("═" * width))
    print(bold(f" PolyGrid Test Suite — {total_tests:,} tests across {total_files} files"))
    print(bold("═" * width))


def _print_summary(groups: list[GroupResult]) -> None:
    width = 58
    print()
    print(bold("═" * width))
    print(bold(" SUMMARY"))
    print(bold("═" * width))

    total_tests = 0
    total_failed = 0
    total_time = 0.0

    for gr in groups:
        if not gr.file_results:
            continue
        icon = green("✅") if gr.ok else red("❌")
        tests = gr.total_tests
        elapsed = gr.elapsed
        total_tests += tests
        total_failed += gr.total_failed
        total_time += elapsed

        name = gr.name
        if len(name) > 36:
            name = name[:33] + "..."
        print(f"  {name:<36s} {tests:>4} tests  {_fmt_time(elapsed):>7s}  {icon}")

    print("─" * width)
    status = green("ALL PASSED") if total_failed == 0 else red(f"{total_failed} FAILED")
    print(f"  {'TOTAL':<36s} {total_tests:>4} tests  {_fmt_time(total_time):>7s}  {bold(status)}")
    print()


# ── Discover test files not in any group ─────────────────────────
def _discover_ungrouped() -> list[str]:
    """Find test files that aren't assigned to any group."""
    known = set()
    for _, files, _ in GROUPS:
        known.update(files)

    ungrouped = []
    for p in sorted(TESTS_DIR.glob("test_*.py")):
        if p.name not in known:
            ungrouped.append(p.name)
    return ungrouped


# ── Main ─────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="PolyGrid Test Runner — grouped progress output",
    )
    parser.add_argument("--fast", action="store_true", help="Only fast-tier files (<3s)")
    parser.add_argument("--medium", action="store_true", help="Fast + medium tier files")
    parser.add_argument("--phase", type=int, help="Only tests for a specific phase (1-13)")
    parser.add_argument("--file", type=str, help="Only files matching this pattern")
    parser.add_argument("--summary", action="store_true", help="Only show summary table")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show individual test names")
    args = parser.parse_args()

    # Determine which groups to run
    groups_to_run: list[tuple[str, list[str], str]] = []

    if args.phase is not None:
        phase_files = PHASE_MAP.get(args.phase, [])
        if not phase_files:
            print(red(f"No tests mapped to Phase {args.phase}"))
            return 1
        groups_to_run.append((f"Phase {args.phase}", phase_files, "all"))

    elif args.file:
        pattern = args.file.lower()
        matched = []
        for _, files, _ in GROUPS:
            for f in files:
                if pattern in f.lower():
                    matched.append(f)
        if not matched:
            print(red(f"No test files matching '{args.file}'"))
            return 1
        groups_to_run.append((f"Files matching '{args.file}'", matched, "all"))

    elif args.fast:
        for name, files, tier in GROUPS:
            if tier == "fast":
                groups_to_run.append((name, files, tier))

    elif args.medium:
        for name, files, tier in GROUPS:
            if tier in ("fast", "medium"):
                groups_to_run.append((name, files, tier))

    else:
        groups_to_run = list(GROUPS)

    # Check for ungrouped files
    ungrouped = _discover_ungrouped()
    if ungrouped and not (args.phase or args.file or args.fast or args.medium):
        groups_to_run.append(("Ungrouped", ungrouped, "unknown"))

    # Count total tests (approximate from known data)
    total_files = sum(len(files) for _, files, _ in groups_to_run)

    # Collect test file paths
    all_file_paths = []
    for _, files, _ in groups_to_run:
        for f in files:
            fp = TESTS_DIR / f
            if fp.exists():
                all_file_paths.append(str(fp))

    # Quick collection count
    cmd = [PYTHON, "-m", "pytest", "--collect-only", "-q", "--no-header"] + all_file_paths
    collect = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    total_tests = 0
    for line in collect.stdout.strip().splitlines():
        # Format: "tests/test_foo.py: 42"
        m = re.search(r":\s*(\d+)\s*$", line)
        if m:
            total_tests += int(m.group(1))

    _print_banner(total_tests, total_files)

    # Run groups
    results: list[GroupResult] = []
    overall_start = time.monotonic()

    for group_name, files, tier in groups_to_run:
        gr = run_group_inline(group_name, tier, files, args.verbose, args.summary)
        results.append(gr)

    overall_elapsed = time.monotonic() - overall_start

    # Print summary
    _print_summary(results)

    # Return non-zero if any failures
    all_ok = all(gr.ok for gr in results)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
