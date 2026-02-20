import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polygrid.builders import build_pentagon_centered_grid, build_pure_hex_grid
from polygrid.render import render_png


def main() -> None:
    output_dir = ROOT / "validation_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    hex_grid = build_pure_hex_grid(2)
    render_png(hex_grid, output_dir / "hex_r2.png")

    pent_grid = build_pentagon_centered_grid(2)
    render_png(pent_grid, output_dir / "pent_r2.png")

    print("Saved validation PNGs to", output_dir)


if __name__ == "__main__":
    main()
