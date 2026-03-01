"""Improved globe renderer — gap-free, subdivided, batched.

Phase 12 renderer that addresses three problems with the v1 renderer:

1. **Texture bleeding** — tile textures are flood-filled to remove
   black borders, so bilinear sampling never sees black pixels.

2. **Sphere subdivision** — each tile's triangle fan is subdivided
   and projected onto the sphere surface, producing smooth curvature
   instead of flat facets.

3. **Batched draw** — all tiles are merged into a single VBO with
   one draw call, eliminating per-tile overhead.

Public API
----------
- :func:`render_globe_v2` — launch an interactive pyglet 3D viewer
- :func:`flood_fill_tile_texture` — remove black borders from a tile PNG
- :func:`flood_fill_atlas` — apply flood-fill to a whole atlas
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from models.objects.goldberg import generate_goldberg_tiles
    _HAS_MODELS = True
except ImportError:
    _HAS_MODELS = False


# ═══════════════════════════════════════════════════════════════════
# 12A — Texture bleeding (flood-fill black borders)
# ═══════════════════════════════════════════════════════════════════

def flood_fill_tile_texture(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    iterations: int = 8,
) -> Path:
    """Remove black borders from a tile texture by flood-filling outward.

    Black pixels (R+G+B < threshold) adjacent to coloured pixels are
    replaced with the average of their coloured neighbours.  Repeated
    for *iterations* to fill the full border region.

    Parameters
    ----------
    image_path : Path
        Input tile texture PNG.
    output_path : Path, optional
        Output path.  Defaults to overwriting input.
    iterations : int
        Number of dilation passes.

    Returns
    -------
    Path
    """
    from PIL import Image

    image_path = Path(image_path)
    if output_path is None:
        output_path = image_path
    output_path = Path(output_path)

    img = Image.open(str(image_path)).convert("RGB")
    arr = np.array(img, dtype=np.float32)  # (H, W, 3)
    h, w = arr.shape[:2]

    # Mask: True where pixel has meaningful colour (not black)
    threshold = 10.0  # sum of RGB channels
    filled = arr[:, :, 0] + arr[:, :, 1] + arr[:, :, 2] > threshold

    for _ in range(iterations):
        # Find unfilled pixels that have at least one filled neighbour
        new_arr = arr.copy()
        new_filled = filled.copy()

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            # Shifted filled mask
            shifted = np.zeros_like(filled)
            shifted_col = np.zeros_like(arr)

            sy = slice(max(0, -dy), h + min(0, -dy))
            sx = slice(max(0, -dx), w + min(0, -dx))
            ty = slice(max(0, dy), h + min(0, dy))
            tx = slice(max(0, dx), w + min(0, dx))

            shifted[ty, tx] = filled[sy, sx]
            shifted_col[ty, tx] = arr[sy, sx]

            # Unfilled pixels that can be filled from this direction
            candidates = (~filled) & shifted
            if candidates.any():
                new_arr[candidates] += shifted_col[candidates]
                new_filled[candidates] = True

        # Average the accumulated colours
        # Count how many neighbours contributed
        count = np.zeros((h, w), dtype=np.float32)
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            shifted = np.zeros_like(filled)
            sy = slice(max(0, -dy), h + min(0, -dy))
            sx = slice(max(0, -dx), w + min(0, -dx))
            ty = slice(max(0, dy), h + min(0, dy))
            tx = slice(max(0, dx), w + min(0, dx))
            shifted[ty, tx] = filled[sy, sx]
            count += (~filled) & shifted

        newly_filled = (~filled) & new_filled
        count_safe = np.maximum(count, 1.0)
        for c in range(3):
            # new_arr already has sum of neighbour colours for newly_filled
            # But we also added the original (which was ~0 for black pixels)
            # So subtract the original black value and divide by count
            new_arr[newly_filled, c] = (
                (new_arr[newly_filled, c] - arr[newly_filled, c])
                / count_safe[newly_filled]
            )

        arr = new_arr
        filled = new_filled

    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    out.save(str(output_path))
    return output_path


def flood_fill_atlas(
    atlas_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    iterations: int = 8,
) -> Path:
    """Apply flood-fill border removal to an entire atlas image.

    The atlas is a grid of tile textures.  Each tile may have black
    corners/borders.  This flood-fills every black pixel that borders
    a coloured pixel, extending tile colours to fill the full atlas.

    Parameters
    ----------
    atlas_path : Path
    output_path : Path, optional
    iterations : int

    Returns
    -------
    Path
    """
    return flood_fill_tile_texture(atlas_path, output_path, iterations=iterations)


# ═══════════════════════════════════════════════════════════════════
# 12B — Sphere subdivision
# ═══════════════════════════════════════════════════════════════════

def _normalize_vec3(v: np.ndarray) -> np.ndarray:
    """Normalize a 3-vector to unit length."""
    length = np.linalg.norm(v)
    if length < 1e-12:
        return v
    return v / length


def _project_to_sphere(point: np.ndarray, radius: float) -> np.ndarray:
    """Project a point onto the sphere of given radius."""
    return _normalize_vec3(point) * radius


def subdivide_tile_mesh(
    center: Tuple[float, float, float],
    vertices: List[Tuple[float, float, float]],
    center_uv: Tuple[float, float],
    vertex_uvs: List[Tuple[float, float]],
    color: Tuple[float, float, float],
    radius: float = 1.0,
    subdivisions: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subdivide a tile's triangle fan and project onto sphere surface.

    Each triangle (center → v[i] → v[i+1]) is subdivided into a grid
    of ``subdivisions²`` smaller triangles using barycentric interpolation,
    then each vertex is projected onto the sphere.

    Parameters
    ----------
    center, vertices
        3D positions (center of tile and polygon boundary vertices).
    center_uv, vertex_uvs
        Corresponding UV coordinates.
    color
        RGB vertex colour.
    radius
        Sphere radius.
    subdivisions
        Number of subdivision levels per triangle edge.

    Returns
    -------
    (vertex_data, index_data)
        vertex_data : np.ndarray, shape (N, 8) — pos(3) + col(3) + uv(2)
        index_data : np.ndarray, shape (M, 3) — triangle indices
    """
    center_pos = np.array(center, dtype=np.float64)
    n = len(vertices)
    all_verts = []  # list of (pos, uv) tuples
    all_tris = []   # list of (i0, i1, i2) index tuples
    vert_map = {}   # (tri_idx, row, col) → global vertex index

    s = subdivisions

    for tri_idx in range(n):
        i_next = (tri_idx + 1) % n

        p0 = center_pos
        p1 = np.array(vertices[tri_idx], dtype=np.float64)
        p2 = np.array(vertices[i_next], dtype=np.float64)

        uv0 = np.array(center_uv, dtype=np.float64)
        uv1 = np.array(vertex_uvs[tri_idx], dtype=np.float64)
        uv2 = np.array(vertex_uvs[i_next], dtype=np.float64)

        # Generate subdivided vertices using barycentric coords
        # For each row i (0..s) and col j (0..s-i):
        #   bary = ((s-i-j)/s, i/s, j/s) for (p0, p1, p2)
        for i in range(s + 1):
            for j in range(s - i + 1):
                k = s - i - j
                b0 = k / s
                b1 = i / s
                b2 = j / s

                pos = b0 * p0 + b1 * p1 + b2 * p2
                pos = _project_to_sphere(pos, radius)
                uv = b0 * uv0 + b1 * uv1 + b2 * uv2

                key = (tri_idx, i, j)
                # Check if this vertex can be shared with a previous triangle
                # Shared vertices: (tri_idx, 0, j) → center column, same as
                # (prev_tri, j, 0) rotated. We'll use deduplication by position.
                vert_map[key] = len(all_verts)
                all_verts.append((pos, uv))

        # Generate triangles
        for i in range(s):
            for j in range(s - i):
                # Upper triangle
                v00 = vert_map[(tri_idx, i, j)]
                v10 = vert_map[(tri_idx, i + 1, j)]
                v01 = vert_map[(tri_idx, i, j + 1)]
                all_tris.append((v00, v10, v01))

                # Lower triangle (if it exists)
                if i + j + 1 < s:
                    v11 = vert_map[(tri_idx, i + 1, j + 1)]
                    all_tris.append((v10, v11, v01))

    # Deduplicate vertices by position (merge shared boundary vertices)
    MERGE_EPS = 1e-8
    final_verts = []
    final_map = {}  # old_index → new_index
    pos_hash = {}   # rounded position tuple → new_index

    for old_idx, (pos, uv) in enumerate(all_verts):
        key = (round(pos[0], 7), round(pos[1], 7), round(pos[2], 7))
        if key in pos_hash:
            final_map[old_idx] = pos_hash[key]
        else:
            new_idx = len(final_verts)
            pos_hash[key] = new_idx
            final_map[old_idx] = new_idx
            final_verts.append((pos, uv))

    # Build output arrays
    vertex_data = np.zeros((len(final_verts), 8), dtype=np.float32)
    for i, (pos, uv) in enumerate(final_verts):
        vertex_data[i, 0:3] = pos
        vertex_data[i, 3:6] = color
        vertex_data[i, 6:8] = uv

    index_data = np.array(
        [(final_map[a], final_map[b], final_map[c]) for a, b, c in all_tris],
        dtype=np.uint32,
    )

    return vertex_data, index_data


# ═══════════════════════════════════════════════════════════════════
# 12C — Batched globe mesh builder
# ═══════════════════════════════════════════════════════════════════

def build_batched_globe_mesh(
    frequency: int,
    uv_layout: Dict[str, Tuple[float, float, float, float]],
    tile_colour_map: Optional[Dict[int, Tuple[float, float, float]]] = None,
    *,
    radius: float = 1.0,
    subdivisions: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a single merged mesh for the entire globe.

    Each tile is subdivided and sphere-projected, then all vertices
    and indices are concatenated into one VBO + IBO.

    Parameters
    ----------
    frequency : int
        Goldberg subdivision frequency.
    uv_layout : dict
        ``{face_id: (u_min, v_min, u_max, v_max)}`` from atlas.
    tile_colour_map : dict, optional
        ``{tile_index: (r, g, b)}``.  White if absent.
    radius : float
    subdivisions : int
        Per-triangle subdivision level.

    Returns
    -------
    (vertex_data, index_data)
        Concatenated arrays for a single draw call.
    """
    if not _HAS_MODELS:
        raise ImportError("models library required")

    from .texture_pipeline import compute_tile_uvs

    tiles = generate_goldberg_tiles(frequency=frequency, radius=radius)

    all_vertex_chunks = []
    all_index_chunks = []
    vertex_offset = 0

    for tile in tiles:
        fid = f"t{tile.index}"
        slot = uv_layout.get(fid)
        if slot is None:
            continue

        color = (1.0, 1.0, 1.0)
        if tile_colour_map:
            color = tile_colour_map.get(tile.index, color)

        # Compute atlas-mapped UVs
        mapped_uvs = compute_tile_uvs(list(tile.uv_vertices), slot)
        center_u = sum(uv[0] for uv in mapped_uvs) / len(mapped_uvs)
        center_v = sum(uv[1] for uv in mapped_uvs) / len(mapped_uvs)

        vdata, idata = subdivide_tile_mesh(
            center=tuple(tile.center),
            vertices=[tuple(v) for v in tile.vertices],
            center_uv=(center_u, center_v),
            vertex_uvs=mapped_uvs,
            color=color,
            radius=radius,
            subdivisions=subdivisions,
        )

        # Offset indices
        idata_offset = idata + vertex_offset
        vertex_offset += len(vdata)

        all_vertex_chunks.append(vdata)
        all_index_chunks.append(idata_offset)

    if not all_vertex_chunks:
        return np.zeros((0, 8), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)

    vertex_data = np.concatenate(all_vertex_chunks, axis=0)
    index_data = np.concatenate(all_index_chunks, axis=0)

    return vertex_data, index_data


# ═══════════════════════════════════════════════════════════════════
# 12D — Interactive OpenGL renderer (v2)
# ═══════════════════════════════════════════════════════════════════

_V2_VERTEX_SHADER = """\
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 uv;

uniform mat4 u_mvp;
uniform mat4 u_model;

out vec3 v_color;
out vec3 v_normal;
out vec2 v_uv;

void main() {
    vec3 world_pos = (u_model * vec4(position, 1.0)).xyz;
    v_color  = color;
    v_normal = normalize(world_pos);   // sphere normal = normalised position
    v_uv     = uv;
    gl_Position = u_mvp * vec4(world_pos, 1.0);
}
"""

_V2_FRAGMENT_SHADER = """\
#version 330 core
in vec3 v_color;
in vec3 v_normal;
in vec2 v_uv;

uniform sampler2D u_atlas;
uniform int       u_use_texture;
uniform vec3      u_light_dir;

out vec4 frag_color;

void main() {
    vec3 base;
    if (u_use_texture == 1) {
        base = texture(u_atlas, v_uv).rgb;
    } else {
        base = v_color;
    }

    // Hemisphere lighting: key light + ambient
    vec3 n = normalize(v_normal);
    float ndotl = dot(n, u_light_dir);
    float light = clamp(ndotl * 0.6 + 0.4, 0.2, 1.0);

    frag_color = vec4(base * light, 1.0);
}
"""


def render_globe_v2(
    payload: Dict[str, Any],
    atlas_path: Union[str, Path],
    uv_layout: Dict[str, Tuple[float, float, float, float]],
    *,
    radius: float = 1.0,
    subdivisions: int = 3,
    width: int = 900,
    height: int = 700,
    title: str = "Polygrid Globe v2",
    flood_fill: bool = False,
    flood_fill_iterations: int = 8,
) -> None:
    """Launch an interactive pyglet window with the improved globe renderer.

    Improvements over v1:
    - Tile textures have terrain-coloured backgrounds (no black corners)
    - Atlas uses gutter pixels to prevent bilinear bleed at slot edges
    - Tiles are subdivided and sphere-projected for smooth curvature
    - All geometry in a single VBO + draw call

    Parameters
    ----------
    payload : dict
        Globe export payload.
    atlas_path : Path
        Detail texture atlas.
    uv_layout : dict
        ``{face_id: (u_min, v_min, u_max, v_max)}``.
    radius : float
    subdivisions : int
        Per-triangle subdivision level (3 = good quality, 5 = high).
    width, height : int
    title : str
    flood_fill : bool
        Whether to additionally flood-fill the atlas.  Usually not
        needed with the 13A background-colour fix.
    flood_fill_iterations : int
        Number of dilation passes for flood-fill.
    """
    if not _HAS_MODELS:
        raise ImportError("models library required for globe rendering")

    try:
        import pyglet
        from pyglet import gl
    except ImportError as exc:
        raise ImportError(
            "pyglet required for interactive rendering. "
            "Install with: pip install pyglet"
        ) from exc

    from PIL import Image
    import tempfile

    atlas_path = Path(atlas_path)

    # ── Flood-fill atlas to remove black borders ────────────────────
    if flood_fill:
        filled_path = atlas_path.parent / f"{atlas_path.stem}_filled{atlas_path.suffix}"
        flood_fill_atlas(atlas_path, filled_path, iterations=flood_fill_iterations)
        atlas_path = filled_path

    # ── Build tile colour map from payload ──────────────────────────
    tile_colour_map: Dict[int, Tuple[float, float, float]] = {}
    for tile in payload["tiles"]:
        idx = int(tile["id"][1:])
        tile_colour_map[idx] = tuple(tile["color"][:3])

    meta = payload["metadata"]
    frequency = meta["frequency"]

    # ── Build batched mesh ──────────────────────────────────────────
    print(f"  Building subdivided globe mesh (subdivisions={subdivisions})...")
    vertex_data, index_data = build_batched_globe_mesh(
        frequency, uv_layout,
        tile_colour_map=tile_colour_map,
        radius=radius,
        subdivisions=subdivisions,
    )
    n_verts = len(vertex_data)
    n_tris = len(index_data)
    print(f"  → {n_verts:,} vertices, {n_tris:,} triangles")

    # ── Create window ───────────────────────────────────────────────
    config = pyglet.gl.Config(
        double_buffer=True, depth_size=24,
        major_version=3, minor_version=3,
        sample_buffers=1, samples=4,  # MSAA
    )
    try:
        window = pyglet.window.Window(
            width=width, height=height,
            caption=title, resizable=True, config=config,
        )
    except pyglet.window.NoSuchConfigException:
        # Fall back without MSAA
        config = pyglet.gl.Config(
            double_buffer=True, depth_size=24,
            major_version=3, minor_version=3,
        )
        window = pyglet.window.Window(
            width=width, height=height,
            caption=title, resizable=True, config=config,
        )

    # ── Compile shaders ─────────────────────────────────────────────
    def _compile_shader(source, shader_type):
        shader = gl.glCreateShader(shader_type)
        source_bytes = source.encode("utf-8")
        length = ctypes.c_int(len(source_bytes))
        src_buffer = ctypes.create_string_buffer(source_bytes)
        src_ptr = ctypes.cast(src_buffer, ctypes.POINTER(ctypes.c_char))
        src_array = (ctypes.POINTER(ctypes.c_char) * 1)(src_ptr)
        gl.glShaderSource(shader, 1, src_array, ctypes.byref(length))
        gl.glCompileShader(shader)
        status = gl.GLint()
        gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS, ctypes.byref(status))
        if not status.value:
            info_log = ctypes.create_string_buffer(1024)
            gl.glGetShaderInfoLog(shader, 1024, None, info_log)
            raise RuntimeError(f"Shader error: {info_log.value.decode()}")
        return shader

    vs = _compile_shader(_V2_VERTEX_SHADER, gl.GL_VERTEX_SHADER)
    fs = _compile_shader(_V2_FRAGMENT_SHADER, gl.GL_FRAGMENT_SHADER)
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vs)
    gl.glAttachShader(program, fs)
    gl.glLinkProgram(program)
    status = gl.GLint()
    gl.glGetProgramiv(program, gl.GL_LINK_STATUS, ctypes.byref(status))
    if not status.value:
        info_log = ctypes.create_string_buffer(1024)
        gl.glGetProgramInfoLog(program, 1024, None, info_log)
        raise RuntimeError(f"Link error: {info_log.value.decode()}")
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)

    mvp_loc = gl.glGetUniformLocation(program, b"u_mvp")
    model_loc = gl.glGetUniformLocation(program, b"u_model")
    atlas_loc = gl.glGetUniformLocation(program, b"u_atlas")
    use_tex_loc = gl.glGetUniformLocation(program, b"u_use_texture")
    light_loc = gl.glGetUniformLocation(program, b"u_light_dir")

    # ── Upload VBO + IBO ────────────────────────────────────────────
    vao = gl.GLuint()
    gl.glGenVertexArrays(1, ctypes.byref(vao))
    gl.glBindVertexArray(vao)

    vbo = gl.GLuint()
    gl.glGenBuffers(1, ctypes.byref(vbo))
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    vbo_data = vertex_data.astype(np.float32).tobytes()
    gl.glBufferData(gl.GL_ARRAY_BUFFER, len(vbo_data), vbo_data, gl.GL_STATIC_DRAW)

    stride = 8 * 4  # 8 floats × 4 bytes
    # position: location 0, 3 floats, offset 0
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
    # color: location 1, 3 floats, offset 12
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(12))
    # uv: location 2, 2 floats, offset 24
    gl.glEnableVertexAttribArray(2)
    gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(24))

    ibo = gl.GLuint()
    gl.glGenBuffers(1, ctypes.byref(ibo))
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ibo)
    ibo_data = index_data.astype(np.uint32).tobytes()
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, len(ibo_data), ibo_data, gl.GL_STATIC_DRAW)
    n_indices = index_data.size

    gl.glBindVertexArray(0)

    # ── Load atlas texture with mipmaps ─────────────────────────────
    atlas_img = Image.open(str(atlas_path)).convert("RGBA").transpose(
        Image.FLIP_TOP_BOTTOM,
    )
    tex_w, tex_h = atlas_img.size
    atlas_bytes = atlas_img.tobytes()

    tex_id = gl.GLuint()
    gl.glGenTextures(1, ctypes.byref(tex_id))
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, tex_w, tex_h, 0,
        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, atlas_bytes,
    )
    gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    # ── Camera state ────────────────────────────────────────────────
    yaw = [0.0]
    pitch = [0.0]
    zoom = [3.0]

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
            [1,  0,  0, 0],
            [0,  c, -s, 0],
            [0,  s,  c, 0],
            [0,  0,  0, 1],
        ], dtype=np.float32)

    # Normalised light direction
    light_dir = _normalize_vec3(np.array([0.3, 0.8, 0.5], dtype=np.float32))

    @window.event
    def on_draw():
        window.clear()
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.05, 0.05, 0.08, 1.0)
        gl.glClear(int(gl.GL_COLOR_BUFFER_BIT) | int(gl.GL_DEPTH_BUFFER_BIT))

        aspect = window.width / max(1, window.height)
        proj = _perspective(math.radians(45), aspect, 0.1, 100.0)
        eye = np.array([0.0, 0.0, zoom[0]], dtype=np.float32)
        view = _look_at(eye, (0.0, 0.0, 0.0))
        mvp = (proj @ view).astype(np.float32)
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
        gl.glUniform3f(light_loc, light_dir[0], light_dir[1], light_dir[2])

        # Bind atlas
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
        gl.glUniform1i(atlas_loc, 0)
        gl.glUniform1i(use_tex_loc, 1)

        # Single draw call for entire globe
        gl.glBindVertexArray(vao)
        gl.glDrawElements(gl.GL_TRIANGLES, n_indices, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glUseProgram(0)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        yaw[0] += dx * 0.01
        pitch[0] += dy * 0.01
        pitch[0] = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, pitch[0]))

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        zoom[0] = max(1.5, min(10.0, zoom[0] - scroll_y * 0.3))

    pyglet.app.run()
