import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polygrid.io import load_json


def main() -> None:
    grid = load_json(ROOT / "examples" / "minimal_grid.json")
    errors = grid.validate()
    if errors:
        raise SystemExit("\n".join(errors))

    print("Vertices:", len(grid.vertices))
    print("Edges:", len(grid.edges))
    print("Faces:", len(grid.faces))
    print("Face neighbors:", grid.compute_face_neighbors())


if __name__ == "__main__":
    main()
