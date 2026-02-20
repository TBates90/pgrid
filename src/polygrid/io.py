from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from .polygrid import PolyGrid


PathLike = Union[str, Path]


def load_json(path: PathLike) -> PolyGrid:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return PolyGrid.from_dict(data)


def save_json(grid: PolyGrid, path: PathLike, include_neighbors: bool = True) -> None:
    Path(path).write_text(grid.to_json(include_neighbors=include_neighbors), encoding="utf-8")
