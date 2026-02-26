"""Models renderer integration — terrain-coloured OpenGL rendering.

Bridges :mod:`globe_mesh` terrain meshes into the ``models`` library's
OpenGL rendering pipeline.  These helpers prepare everything needed to
display a terrain-coloured Goldberg polyhedron in a pyglet window.

Requires: ``models``, ``pyglet``, and a working OpenGL 3.3+ context.

Functions
---------
- :func:`prepare_terrain_scene` — build meshes + metadata for OpenGL render
- :func:`render_terrain_globe_opengl` — full interactive pyglet render
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

try:
    from models.core import Color, ShapeMesh
    from models.objects.goldberg.layout import layout_for_frequency
    from models.objects.goldberg.mesh import (
        build_layout_mesh,
        build_layout_edge_mesh,
        build_layout_tile_meshes,
    )

    _HAS_MODELS = True
except ImportError:  # pragma: no cover
    _HAS_MODELS = False


def _require_models() -> None:
    if not _HAS_MODELS:
        raise ImportError(
            "The 'models' library is required for globe rendering.  "
            "Install it with: pip install models"
        )


# ═══════════════════════════════════════════════════════════════════
# 9C.1 — Coloured mesh builder (from JSON export)
# ═══════════════════════════════════════════════════════════════════

def build_coloured_globe_mesh(
    frequency: int,
    tile_colours: Dict[str, Tuple[float, float, float]],
    *,
    radius: float = 1.0,
) -> "ShapeMesh":
    """Build a single Goldberg ``ShapeMesh`` with per-tile terrain colours.

    This is a standalone function that doesn't require a ``GlobeGrid``
    — just a frequency and a mapping of face IDs to RGB colours.  It's
    ideal for loading a globe export JSON and rendering it.

    Parameters
    ----------
    frequency : int
        Goldberg subdivision frequency.
    tile_colours : dict
        ``{face_id: (r, g, b)}`` mapping (face IDs like ``"t0"``,
        ``"t1"``, etc.).
    radius : float
        Mesh radius.

    Returns
    -------
    ShapeMesh
    """
    _require_models()

    layout = layout_for_frequency(frequency)
    colors: list["Color"] = []
    for polygon in layout.polygons:
        face_id = f"t{polygon.index}"
        rgb = tile_colours.get(face_id, (0.5, 0.5, 0.5))
        colors.append(Color(rgb[0], rgb[1], rgb[2], 1.0))
    return build_layout_mesh(layout, radius=radius, colors=tuple(colors))


def build_coloured_globe_mesh_from_export(
    payload: Dict[str, Any],
    *,
    radius: Optional[float] = None,
) -> "ShapeMesh":
    """Build a ``ShapeMesh`` from a globe export payload dict.

    Parameters
    ----------
    payload : dict
        A globe export payload (as from :func:`export_globe_payload`
        or loaded from JSON).
    radius : float, optional
        Override the payload's radius.

    Returns
    -------
    ShapeMesh
    """
    _require_models()

    meta = payload["metadata"]
    freq = meta["frequency"]
    r = radius if radius is not None else meta.get("radius", 1.0)

    tile_colours = {}
    for tile in payload["tiles"]:
        fid = tile["id"]
        tile_colours[fid] = tuple(tile["color"])

    return build_coloured_globe_mesh(freq, tile_colours, radius=r)


def build_edge_mesh_for_frequency(
    frequency: int,
    *,
    radius: float = 1.0,
    color: Optional["Color"] = None,
    segments: int = 1,
) -> "ShapeMesh":
    """Build a wireframe edge mesh for the given frequency.

    Parameters
    ----------
    frequency : int
    radius : float
    color : Color, optional
    segments : int

    Returns
    -------
    ShapeMesh
    """
    _require_models()

    layout = layout_for_frequency(frequency)
    return build_layout_edge_mesh(
        layout, radius=radius, color=color, segments=segments,
    )


# ═══════════════════════════════════════════════════════════════════
# 9C.2 — Scene preparation (mesh + metadata, pre-OpenGL)
# ═══════════════════════════════════════════════════════════════════

def prepare_terrain_scene(
    payload: Dict[str, Any],
    *,
    radius: Optional[float] = None,
    include_edges: bool = True,
) -> Dict[str, Any]:
    """Prepare all meshes for an OpenGL terrain globe scene.

    Returns a dict with:
    - ``"mesh"`` — the main ``ShapeMesh``
    - ``"edge_mesh"`` — wireframe ``ShapeMesh`` (if *include_edges*)
    - ``"frequency"`` — Goldberg frequency
    - ``"radius"`` — mesh radius
    - ``"tile_count"`` — number of tiles

    This does not require an OpenGL context — it produces CPU-side
    mesh data ready for ``SimpleMeshRenderer.upload_mesh()``.

    Parameters
    ----------
    payload : dict
        Globe export payload.
    radius : float, optional
    include_edges : bool

    Returns
    -------
    dict
    """
    _require_models()

    meta = payload["metadata"]
    freq = meta["frequency"]
    r = radius if radius is not None else meta.get("radius", 1.0)

    scene: Dict[str, Any] = {
        "mesh": build_coloured_globe_mesh_from_export(payload, radius=r),
        "frequency": freq,
        "radius": r,
        "tile_count": meta["tile_count"],
    }

    if include_edges:
        scene["edge_mesh"] = build_edge_mesh_for_frequency(
            freq, radius=r,
            color=Color(0.2, 0.2, 0.2, 0.5) if _HAS_MODELS else None,
        )
    else:
        scene["edge_mesh"] = None

    return scene


# ═══════════════════════════════════════════════════════════════════
# 9C.2 — Full OpenGL render (requires pyglet)
# ═══════════════════════════════════════════════════════════════════

def render_terrain_globe_opengl(
    payload: Dict[str, Any],
    *,
    radius: float = 1.0,
    width: int = 800,
    height: int = 600,
    title: str = "Polygrid Terrain Globe",
) -> None:
    """Launch a pyglet window rendering the terrain-coloured globe.

    Requires ``pyglet`` and an OpenGL 3.3+ capable display.

    Parameters
    ----------
    payload : dict
        Globe export payload (from :func:`export_globe_payload` or JSON).
    radius : float
    width, height : int
        Window dimensions.
    title : str
        Window title.
    """
    _require_models()

    try:
        import pyglet
        from pyglet import gl
    except ImportError as exc:
        raise ImportError(
            "pyglet is required for interactive rendering. "
            "Install with: pip install pyglet"
        ) from exc

    scene = prepare_terrain_scene(payload, radius=radius)
    mesh = scene["mesh"]
    edge_mesh = scene["edge_mesh"]

    # Defer all OpenGL calls to the window's on_draw
    from models.rendering.opengl import SimpleMeshRenderer
    import numpy as np
    import math
    import ctypes

    _VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 uv;

uniform mat4 u_mvp;

out vec3 v_color;
out vec3 v_normal;

void main() {
    v_color = color;
    v_normal = normalize(position);
    gl_Position = u_mvp * vec4(position, 1.0);
}
"""

    _FRAGMENT_SHADER = """
#version 330 core
in vec3 v_color;
in vec3 v_normal;

out vec4 frag_color;

void main() {
    vec3 light_dir = normalize(vec3(0.3, 0.8, 0.5));
    float light = max(dot(normalize(v_normal), light_dir), 0.25);
    frag_color = vec4(v_color * light, 1.0);
}
"""

    def _perspective(fovy, aspect, near, far):
        f = 1.0 / math.tan(fovy / 2.0)
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0],
        ], dtype=np.float32)

    def _look_at(eye, center, up):
        f = np.array(center) - np.array(eye)
        f = f / np.linalg.norm(f)
        s = np.cross(f, np.array(up))
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)
        m = np.eye(4, dtype=np.float32)
        m[0, :3] = s
        m[1, :3] = u
        m[2, :3] = -f
        m[0, 3] = -np.dot(s, eye)
        m[1, 3] = -np.dot(u, eye)
        m[2, 3] = np.dot(f, eye)
        return m

    def _compile_shader(source, shader_type):
        shader = gl.glCreateShader(shader_type)
        source_bytes = source.encode("utf-8")
        length = ctypes.c_int(len(source_bytes))
        src_buffer = ctypes.create_string_buffer(source_bytes)
        src_ptr = ctypes.cast(src_buffer, ctypes.POINTER(ctypes.c_char))
        src_array = (ctypes.POINTER(ctypes.c_char) * 1)(src_ptr)
        gl.glShaderSource(shader, 1, src_array, ctypes.byref(length))
        gl.glCompileShader(shader)
        return shader

    def _create_program():
        vs = _compile_shader(_VERTEX_SHADER, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(_FRAGMENT_SHADER, gl.GL_FRAGMENT_SHADER)
        prog = gl.glCreateProgram()
        gl.glAttachShader(prog, vs)
        gl.glAttachShader(prog, fs)
        gl.glLinkProgram(prog)
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        return prog

    config = pyglet.gl.Config(
        major_version=3, minor_version=3,
        forward_compatible=True,
        sample_buffers=1, samples=4,
    )
    window = pyglet.window.Window(
        width=width, height=height,
        caption=title, config=config,
    )

    renderer = SimpleMeshRenderer()
    mesh_handle = renderer.upload_mesh(mesh)
    edge_handle = renderer.upload_mesh(edge_mesh) if edge_mesh else None

    program = _create_program()
    mvp_loc = gl.glGetUniformLocation(program, b"u_mvp")

    rotation = [0.0, 0.0]
    zoom = [3.0]

    @window.event
    def on_draw():
        window.clear()
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glUseProgram(program)

        proj = _perspective(math.radians(45), width / height, 0.1, 100.0)
        eye = np.array([
            zoom[0] * math.sin(rotation[0]) * math.cos(rotation[1]),
            zoom[0] * math.sin(rotation[1]),
            zoom[0] * math.cos(rotation[0]) * math.cos(rotation[1]),
        ])
        view = _look_at(eye, [0, 0, 0], [0, 1, 0])
        mvp = (proj @ view).astype(np.float32)
        gl.glUniformMatrix4fv(mvp_loc, 1, gl.GL_TRUE, mvp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

        renderer.draw(mesh_handle)
        if edge_handle:
            renderer.draw(edge_handle)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        rotation[0] += dx * 0.01
        rotation[1] += dy * 0.01
        rotation[1] = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, rotation[1]))

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        zoom[0] = max(1.5, min(10.0, zoom[0] - scroll_y * 0.3))

    pyglet.app.run()
