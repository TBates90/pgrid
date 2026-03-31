"""Notebook-friendly debug rendering for small 3D polygrid columns."""

from __future__ import annotations

from collections.abc import Iterable

from ..core.models import Face
from ..core.polygrid import PolyGrid


_ROLE_COLOURS = {
    "top": "#8fb6d9",
    "side": "#405160",
    "base": "#20262e",
}


def _face_colour(face: Face) -> str:
    role = face.metadata.get("surface_role")
    return _ROLE_COLOURS.get(role, "#708090")


def plot_polygrid_column(
    grid: PolyGrid,
    *,
    ax=None,
    elev: float = 24,
    azim: float = -58,
    annotate: bool = False,
):
    """Plot a small 3D polygrid using matplotlib's 3D collections."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt

    created_ax = ax is None
    if created_ax:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

    polygons: list[list[tuple[float, float, float]]] = []
    colours: list[str] = []

    for face in grid.faces.values():
        points: list[tuple[float, float, float]] = []
        for vid in face.vertex_ids:
            vertex = grid.vertices[vid]
            if not vertex.has_position():
                points = []
                break
            points.append((vertex.x, vertex.y, vertex.z or 0.0))
        if len(points) < 3:
            continue
        polygons.append(points)
        colours.append(_face_colour(face))

    if polygons:
        collection = Poly3DCollection(
            polygons,
            facecolors=colours,
            edgecolors="#d8dee6",
            linewidths=0.8,
            alpha=0.95,
        )
        ax.add_collection3d(collection)

    xs = [vertex.x for vertex in grid.vertices.values() if vertex.x is not None]
    ys = [vertex.y for vertex in grid.vertices.values() if vertex.y is not None]
    zs = [vertex.z or 0.0 for vertex in grid.vertices.values() if vertex.x is not None]
    if xs and ys:
        pad = max(max(xs) - min(xs), max(ys) - min(ys), 1.0) * 0.2
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
        ax.set_zlim(min(zs) - pad * 0.2, max(zs) + pad)

    if annotate:
        for face in grid.faces.values():
            coords = [grid.vertices[vid] for vid in face.vertex_ids]
            if not all(vertex.has_position() for vertex in coords):
                continue
            cx = sum(vertex.x for vertex in coords) / len(coords)
            cy = sum(vertex.y for vertex in coords) / len(coords)
            cz = sum((vertex.z or 0.0) for vertex in coords) / len(coords)
            ax.text(cx, cy, cz, face.id, fontsize=7, color="#f3f5f8")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.5))
    ax.set_title(grid.metadata.get("generator", "polygrid-column"))
    return ax