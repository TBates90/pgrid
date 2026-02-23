"""PolyGrid package."""

from .models import Vertex, Edge, Face
from .polygrid import PolyGrid
from .algorithms import build_face_adjacency, ring_faces
from .io import load_json, save_json
from .render import render_png
from .builders import build_pure_hex_grid, build_pentagon_centered_grid, hex_face_count
from .embedding import tutte_embedding
from .composite import CompositeGrid, join_grids, split_composite
from .angle_solver import ring_angle_spec, solve_ring_hex_lengths, solve_ring_hex_outer_length
from .diagnostics import ring_diagnostics, summarize_ring_stats

__all__ = [
    "Vertex",
    "Edge",
    "Face",
    "PolyGrid",
    "build_face_adjacency",
    "ring_faces",
    "load_json",
    "save_json",
    "render_png",
    "build_pure_hex_grid",
    "build_pentagon_centered_grid",
    "hex_face_count",
    "tutte_embedding",
    "CompositeGrid",
    "join_grids",
    "split_composite",
    "ring_angle_spec",
    "solve_ring_hex_lengths",
    "solve_ring_hex_outer_length",
    "ring_diagnostics",
    "summarize_ring_stats",
]
