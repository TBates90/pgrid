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
    from models.objects.goldberg import generate_goldberg_tiles

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

    Uses a per-tile mesh approach identical to the ``models`` library's
    ``goldberg_demo.py`` — each tile is a separate triangle-fan mesh
    uploaded individually and drawn with its own model matrix.

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

    from array import array as typed_array
    from models.core import VertexAttribute
    from models.objects.goldberg import generate_goldberg_tiles
    from models.rendering.opengl import SimpleMeshRenderer
    import numpy as np
    import math
    import ctypes

    # ── Build per-tile colour map from payload ──────────────────────

    tile_colour_map: Dict[int, Tuple[float, float, float]] = {}
    for tile in payload["tiles"]:
        # tile["id"] is like "t0", "t1", …
        idx = int(tile["id"][1:])
        tile_colour_map[idx] = tuple(tile["color"][:3])

    meta = payload["metadata"]
    frequency = meta["frequency"]

    # ── Shaders (matching goldberg_demo pattern) ────────────────────

    _VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 uv;

uniform mat4 u_mvp;
uniform mat4 u_model;

out vec3 v_color;
out vec3 v_normal;

void main() {
    vec3 world_pos = (u_model * vec4(position, 1.0)).xyz;
    v_color = color;
    v_normal = normalize(world_pos);
    gl_Position = u_mvp * vec4(world_pos, 1.0);
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

    def _look_at(eye, target):
        eye_v = np.array(eye, dtype=np.float32)
        target_v = np.array(target, dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        forward = target_v - eye_v
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        view = np.identity(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -eye_v @ view[:3, :3]
        return view

    def _compile_shader(source, shader_type):
        shader = gl.glCreateShader(shader_type)
        source_bytes = source.encode("utf-8")
        length = ctypes.c_int(len(source_bytes))
        src_buffer = ctypes.create_string_buffer(source_bytes)
        src_ptr = ctypes.cast(src_buffer, ctypes.POINTER(ctypes.c_char))
        src_array = (ctypes.POINTER(ctypes.c_char) * 1)(src_ptr)
        gl.glShaderSource(shader, 1, src_array, ctypes.byref(length))
        gl.glCompileShader(shader)
        # Check compilation status
        status = gl.GLint()
        gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS, ctypes.byref(status))
        if not status.value:
            info_log = ctypes.create_string_buffer(1024)
            gl.glGetShaderInfoLog(shader, 1024, None, info_log)
            raise RuntimeError(
                f"Shader compilation failed: {info_log.value.decode('utf-8')}"
            )
        return shader

    def _create_program():
        vs = _compile_shader(_VERTEX_SHADER, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(_FRAGMENT_SHADER, gl.GL_FRAGMENT_SHADER)
        prog = gl.glCreateProgram()
        gl.glAttachShader(prog, vs)
        gl.glAttachShader(prog, fs)
        gl.glLinkProgram(prog)
        # Check link status
        status = gl.GLint()
        gl.glGetProgramiv(prog, gl.GL_LINK_STATUS, ctypes.byref(status))
        if not status.value:
            info_log = ctypes.create_string_buffer(1024)
            gl.glGetProgramInfoLog(prog, 1024, None, info_log)
            raise RuntimeError(
                f"Shader link failed: {info_log.value.decode('utf-8')}"
            )
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        return prog

    def _tile_world_mesh(tile, color):
        """Build a per-tile triangle-fan mesh in world space (stride=32).

        Matches the ``goldberg_demo._tile_world_mesh`` pattern exactly:
        position(3) + color(3) + uv(2) = 8 floats × 4 bytes = 32.
        """
        positions = [tile.center]
        positions.extend(tile.vertices)

        colors = [color for _ in positions]
        if tile.uv_vertices:
            center_uv = (
                sum(uv[0] for uv in tile.uv_vertices) / len(tile.uv_vertices),
                sum(uv[1] for uv in tile.uv_vertices) / len(tile.uv_vertices),
            )
            uvs = [center_uv, *tile.uv_vertices]
        else:
            uvs = [(0.5, 0.5) for _ in positions]

        vertex_data = typed_array("f")
        for pos, col, uv in zip(positions, colors, uvs):
            vertex_data.extend(pos)
            vertex_data.extend(col)
            vertex_data.extend(uv)

        indices = []
        vertex_count = len(positions)
        for i in range(1, vertex_count):
            next_idx = 1 if i == vertex_count - 1 else i + 1
            indices.extend([0, i, next_idx])
        index_data = typed_array("I", indices)

        position_attr = VertexAttribute(
            name="position", location=0, components=3, offset=0,
        )
        color_attr = VertexAttribute(
            name="color", location=1, components=3, offset=3 * 4,
        )
        uv_attr = VertexAttribute(
            name="uv", location=2, components=2, offset=6 * 4,
        )
        return ShapeMesh(
            vertex_data=vertex_data,
            index_data=index_data,
            stride=8 * 4,
            attributes=(position_attr, color_attr, uv_attr),
        )

    # ── Create window and OpenGL resources ──────────────────────────

    config = pyglet.gl.Config(
        double_buffer=True, depth_size=24,
        major_version=3, minor_version=3,
    )
    window = pyglet.window.Window(
        width=width, height=height,
        caption=title, resizable=True, config=config,
    )

    # Generate models tiles and build per-tile meshes
    tiles = generate_goldberg_tiles(frequency=frequency, radius=radius)
    renderer = SimpleMeshRenderer()

    tile_handles = []
    for tile in tiles:
        color = tile_colour_map.get(tile.index, (0.5, 0.5, 0.5))
        mesh = _tile_world_mesh(tile, color)
        handle = renderer.upload_mesh(mesh, draw_mode=gl.GL_TRIANGLES)
        tile_handles.append(handle)

    program = _create_program()
    mvp_loc = gl.glGetUniformLocation(program, b"u_mvp")
    model_loc = gl.glGetUniformLocation(program, b"u_model")

    # Camera stays fixed on the Z axis looking at the origin.
    # Mouse drag rotates the *model* (globe) in place via u_model.
    yaw = [0.0]    # rotation around Y axis (left/right drag)
    pitch = [0.0]  # rotation around X axis (up/down drag)
    zoom = [3.5]   # camera distance from origin

    def _rotation_y(angle):
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [ c, 0, s, 0],
            [ 0, 1, 0, 0],
            [-s, 0, c, 0],
            [ 0, 0, 0, 1],
        ], dtype=np.float32)

    def _rotation_x(angle):
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [1,  0, 0, 0],
            [0,  c, -s, 0],
            [0,  s,  c, 0],
            [0,  0,  0, 1],
        ], dtype=np.float32)

    @window.event
    def on_draw():
        window.clear()
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.05, 0.05, 0.08, 1.0)
        gl.glClear(int(gl.GL_COLOR_BUFFER_BIT) | int(gl.GL_DEPTH_BUFFER_BIT))

        aspect = window.width / max(1, window.height)
        proj = _perspective(math.radians(45), aspect, 0.1, 100.0)
        # Fixed camera on +Z axis, looking at origin
        eye = np.array([0.0, 0.0, zoom[0]], dtype=np.float32)
        view = _look_at(eye, (0.0, 0.0, 0.0))
        mvp = (proj @ view).astype(np.float32)

        # Globe rotation: pitch (X) then yaw (Y)
        model = (_rotation_y(yaw[0]) @ _rotation_x(pitch[0])).astype(np.float32)

        gl.glUseProgram(program)
        gl.glUniformMatrix4fv(
            mvp_loc, 1, gl.GL_TRUE,
            mvp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        gl.glUniformMatrix4fv(
            model_loc, 1, gl.GL_TRUE,
            model.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

        for handle in tile_handles:
            renderer.draw(handle)

        gl.glUseProgram(0)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        yaw[0] += dx * 0.01
        pitch[0] += dy * 0.01
        # Clamp pitch to avoid flipping
        pitch[0] = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, pitch[0]))

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        zoom[0] = max(1.5, min(10.0, zoom[0] - scroll_y * 0.3))

    pyglet.app.run()
