"""Core types, container, and low-level algorithms."""

from .models import Vertex, Edge, Face, MacroEdge, Region
from .polygrid import PolyGrid
from .algorithms import build_face_adjacency, get_face_adjacency, ring_faces

__all__ = [
    "Vertex", "Edge", "Face", "MacroEdge", "Region",
    "PolyGrid",
    "build_face_adjacency", "get_face_adjacency", "ring_faces",
]
