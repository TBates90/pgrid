from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from .models import Face, Vertex
from .polygrid import PolyGrid


def render_png(
    grid: PolyGrid,
    output_path: str | Path,
    face_alpha: float = 0.15,
    edge_color: str = "#2b2b2b",
    face_color: str = "#5aa9e6",
    vertex_color: str = "#2b2b2b",
    vertex_size: float = 8.0,
    padding: float = 0.5,
    dpi: int = 150,
    show_pent_axes: bool = False,
) -> None:
    """Render a grid to PNG using vertex positions.

    Requires matplotlib; imported lazily to keep core package lightweight.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError as exc:  # pragma: no cover - requires optional dep
        raise RuntimeError(
            "matplotlib is required for rendering. Install with `pip install matplotlib`."
        ) from exc

    vertices = sorted(grid.vertices.values(), key=lambda v: v.id)
    if not _all_vertices_positioned(vertices):
        raise ValueError("All vertices must have positions for rendering.")

    fig, ax = plt.subplots()

    for face in sorted(grid.faces.values(), key=lambda f: f.id):
        _draw_face(
            ax,
            face,
            grid.vertices,
            Polygon,
            edge_color,
            face_color,
            face_alpha,
        )

    if show_pent_axes:
        _draw_pent_axes(ax, grid)

    for vertex in vertices:
        ax.scatter(vertex.x, vertex.y, s=vertex_size, c=vertex_color, zorder=3)

    xs = [v.x for v in vertices if v.x is not None]
    ys = [v.y for v in vertices if v.y is not None]
    min_x, max_x = min(xs) - padding, max(xs) + padding
    min_y, max_y = min(ys) - padding, max(ys) + padding

    ax.set_aspect("equal", "box")
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _all_vertices_positioned(vertices: Iterable[Vertex]) -> bool:
    return all(vertex.has_position() for vertex in vertices)


def _draw_face(
    ax,
    face: Face,
    vertices: dict[str, Vertex],
    polygon_cls,
    edge_color: str,
    face_color: str,
    face_alpha: float,
) -> None:
    points: list[tuple[float, float]] = []
    for vertex_id in face.vertex_ids:
        vertex = vertices[vertex_id]
        if vertex.x is None or vertex.y is None:
            return
        points.append((vertex.x, vertex.y))

    if len(points) < 3:
        return

    polygon = polygon_cls(points, closed=True, facecolor=face_color, alpha=face_alpha)
    ax.add_patch(polygon)
    xs, ys = zip(*(points + [points[0]]))
    ax.plot(xs, ys, color=edge_color, linewidth=1.0)


def _draw_pent_axes(ax, grid: PolyGrid) -> None:
    pent = next((face for face in grid.faces.values() if face.face_type == "pent"), None)
    if pent is None:
        return
    verts = [grid.vertices[vid] for vid in pent.vertex_ids]
    if not all(v.has_position() for v in verts):
        return
    cx = sum(v.x for v in verts if v.x is not None) / len(verts)
    cy = sum(v.y for v in verts if v.y is not None) / len(verts)

    xs = [v.x for v in grid.vertices.values() if v.x is not None]
    ys = [v.y for v in grid.vertices.values() if v.y is not None]
    if not xs or not ys:
        return
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    length = max(max_x - min_x, max_y - min_y) * 1.2

    for i in range(len(verts)):
        v1 = verts[i]
        v2 = verts[(i + 1) % len(verts)]
        mx = (v1.x + v2.x) / 2
        my = (v1.y + v2.y) / 2
        dx = mx - cx
        dy = my - cy
        norm = (dx**2 + dy**2) ** 0.5 or 1.0
        dx /= norm
        dy /= norm
        x0 = cx - dx * length
        y0 = cy - dy * length
        x1 = cx + dx * length
        y1 = cy + dy * length
        ax.plot([x0, x1], [y0, y1], color="#d1495b", linewidth=1.0, linestyle=(0, (3, 3)))
