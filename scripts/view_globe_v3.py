#!/usr/bin/env python3
"""Phase 13 integrated globe viewer — PBR + water + atmosphere + bloom + LOD.

Wires together all Phase 12-13 rendering features into a single
interactive viewer:

- **PBR lighting** (13E) — normal-mapped, warm key + cool fill, Fresnel rim
- **Water rendering** (13H) — depth-based colour, animated waves, coastline foam
- **Atmosphere** (13G) — Fresnel limb haze shell
- **Bloom** (13G) — 3-pass luminance extraction + Gaussian blur + composite
- **Background gradient** (13G) — radial dark-blue-to-black
- **Adaptive LOD** (13F) — per-tile subdivision by screen fraction, backface cull

Usage
-----
::

    # Full pipeline: terrain → atlas → normal maps → PBR viewer
    python scripts/view_globe_v3.py

    # Options
    python scripts/view_globe_v3.py -f 3 --detail-rings 4 --preset earthlike
    python scripts/view_globe_v3.py -f 3 --seed 99 --subdivisions 3
    python scripts/view_globe_v3.py --no-bloom          # disable bloom
    python scripts/view_globe_v3.py --no-atmosphere      # disable atmosphere
    python scripts/view_globe_v3.py --no-lod             # fixed subdivision, no LOD
    python scripts/view_globe_v3.py --no-water           # disable water detection

Requires: pyglet, Pillow, numpy, opensimplex, models (editable install).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════
# Terrain pipeline (same as demo_cohesive_globe.py)
# ═══════════════════════════════════════════════════════════════════

def _build_terrain(frequency: int, detail_rings: int, seed: int,
                   preset: str, tile_size: int, soft_blend: bool = False):
    """Run the full Phase 11 terrain pipeline and return everything
    the viewer needs: atlas path, UV layout, normal-map atlas, colour
    map, water tile map.
    """
    from polygrid.globe import build_globe_grid
    from polygrid.mountains import MountainConfig, generate_mountains
    from polygrid.tile_data import FieldDef, TileDataStore, TileSchema
    from polygrid.tile_detail import TileDetailSpec, DetailGridCollection
    from polygrid.detail_terrain_3d import Terrain3DSpec, generate_all_detail_terrain_3d
    from polygrid.terrain_patches import (
        TERRAIN_PRESETS, generate_terrain_patches, apply_terrain_patches,
    )
    from polygrid.texture_pipeline import build_detail_atlas
    from polygrid.biome_pipeline import build_feature_atlas
    from polygrid.detail_render import BiomeConfig
    from polygrid.render_enhanced import compute_all_normal_maps
    from polygrid.globe_renderer_v2 import (
        build_normal_map_atlas, classify_water_tiles,
    )
    from polygrid.globe_export import export_globe_payload

    print(f"Building globe (freq={frequency}, seed={seed})...")
    grid = build_globe_grid(frequency)
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    store = TileDataStore(grid=grid, schema=schema)

    config = MountainConfig(
        seed=seed, ridge_frequency=2.0, ridge_octaves=4,
        peak_elevation=1.0, base_elevation=0.0,
    )
    generate_mountains(grid, store, config)
    print(f"  → {len(grid.faces)} tiles")

    # ── Detail grids + 3D terrain ───────────────────────────────────
    print(f"  Building detail grids (rings={detail_rings})...")
    spec = TileDetailSpec(detail_rings=detail_rings)
    coll = DetailGridCollection.build(grid, spec)

    spec_3d = Terrain3DSpec(
        noise_frequency=4.0, ridge_frequency=3.0,
        fbm_weight=0.6, ridge_weight=0.4,
        base_weight=0.70, amplitude=0.15, seed=seed,
    )
    generate_all_detail_terrain_3d(coll, grid, store, spec_3d)

    # ── Terrain patches ─────────────────────────────────────────────
    print(f"  Applying terrain patches (preset={preset})...")
    terrain_presets = TERRAIN_PRESETS
    preset_config = terrain_presets.get(preset, terrain_presets["earthlike"])
    patches = generate_terrain_patches(grid, distribution=preset_config, seed=seed)
    apply_terrain_patches(coll, grid, store, patches, seed=seed)
    print(f"    → {len(patches)} patches")

    # ── Colour atlas ────────────────────────────────────────────────
    out_dir = Path("exports/v3_viewer")
    out_dir.mkdir(parents=True, exist_ok=True)

    biome = BiomeConfig()
    print("  Rendering colour atlas...")
    if soft_blend:
        print("    (soft-blend mode: fullslot + 16B blend + 16C scatter + 16D softening)")
        atlas_path, uv_layout = build_feature_atlas(
            coll, grid,
            biome_config=biome,
            output_dir=out_dir / "tiles",
            tile_size=tile_size, noise_seed=seed,
            soft_blend=True,
        )
    else:
        atlas_path, uv_layout = build_detail_atlas(
            coll, biome, out_dir / "tiles",
            tile_size=tile_size, noise_seed=seed,
        )
    print(f"    → {atlas_path}")

    # ── Normal-map atlas ────────────────────────────────────────────
    print("  Computing normal maps...")
    normals = compute_all_normal_maps(coll, scale=2.0)
    normal_atlas_img, normal_uv = build_normal_map_atlas(
        normals, coll, tile_size=tile_size, gutter=4,
    )
    normal_atlas_path = out_dir / "normal_atlas.png"
    normal_atlas_img.save(str(normal_atlas_path))
    print(f"    → {normal_atlas_path}")

    # ── Tile colour map + water classification ──────────────────────
    payload = export_globe_payload(grid, store, ramp="satellite")
    tile_colour_map = {}
    for tile in payload["tiles"]:
        idx = int(tile["id"][1:])
        tile_colour_map[idx] = tuple(tile["color"][:3])

    water_tiles = classify_water_tiles(tile_colour_map)
    n_water = sum(1 for v in water_tiles.values() if v)
    print(f"  Water tiles: {n_water}/{len(water_tiles)}")

    # Read atlas dimensions for UV inset clamping
    from PIL import Image as _PILImage
    _atlas_img = _PILImage.open(str(atlas_path))
    atlas_size = _atlas_img.width  # atlas is square
    _atlas_img.close()

    return {
        "payload": payload,
        "atlas_path": atlas_path,
        "atlas_size": atlas_size,
        "uv_layout": uv_layout,
        "normal_atlas_path": normal_atlas_path,
        "normal_uv": normal_uv,
        "tile_colour_map": tile_colour_map,
        "water_tiles": water_tiles,
        "frequency": frequency,
        "coll": coll,
    }


# ═══════════════════════════════════════════════════════════════════
# OpenGL viewer
# ═══════════════════════════════════════════════════════════════════

def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def launch_viewer(data, *, subdivisions=3, width=1100, height=850,
                  enable_bloom=True, enable_atmosphere=True,
                  enable_lod=True, enable_water=True):
    """Launch the Phase 13 integrated globe viewer."""
    import pyglet
    from pyglet import gl
    from PIL import Image

    from polygrid.globe_renderer_v2 import (
        build_batched_globe_mesh,
        build_atmosphere_shell,
        build_background_quad,
        get_pbr_shader_sources,
        get_atmosphere_shader_sources,
        get_background_shader_sources,
        get_bloom_shader_sources,
        ATMOSPHERE_SCALE, ATMOSPHERE_COLOR,
        BLOOM_THRESHOLD, BLOOM_INTENSITY,
        BG_CENTER_COLOR, BG_EDGE_COLOR,
    )

    frequency = data["frequency"]
    uv_layout = data["uv_layout"]
    tile_colour_map = data["tile_colour_map"]
    water_tiles = data["water_tiles"] if enable_water else None
    atlas_path = Path(data["atlas_path"])
    normal_atlas_path = Path(data["normal_atlas_path"])
    atlas_size = data["atlas_size"]

    # ── Build globe mesh (PBR stride 15: pos+col+uv+T+B+water) ─────
    print(f"\n  Building PBR globe mesh (subdivisions={subdivisions})...")
    vertex_data, index_data = build_batched_globe_mesh(
        frequency, uv_layout,
        tile_colour_map=tile_colour_map,
        radius=1.0,
        subdivisions=subdivisions,
        normal_mapped=True,
        water_tiles=water_tiles,
        edge_blend=0.4,
        uv_inset_px=1.5,
        atlas_size=atlas_size,
    )
    n_verts = len(vertex_data)
    n_tris = len(index_data)

    # Determine stride: 15 if water tiles exist, 14 otherwise
    stride_floats = vertex_data.shape[1] if vertex_data.ndim == 2 else 15
    has_water_channel = stride_floats >= 15
    print(f"  → {n_verts:,} vertices, {n_tris:,} triangles, stride={stride_floats}")

    # ── Build atmosphere shell ──────────────────────────────────────
    atmo_verts, atmo_indices = None, None
    if enable_atmosphere:
        atmo_verts, atmo_indices = build_atmosphere_shell(
            radius=1.0, scale=ATMOSPHERE_SCALE, color=ATMOSPHERE_COLOR,
        )
        print(f"  Atmosphere: {len(atmo_verts)} verts, {len(atmo_indices)} tris")

    # ── Build background quad ───────────────────────────────────────
    bg_quad = build_background_quad()

    # ── Create window ───────────────────────────────────────────────
    config = pyglet.gl.Config(
        double_buffer=True, depth_size=24,
        major_version=3, minor_version=3,
        sample_buffers=1, samples=4,
    )
    try:
        window = pyglet.window.Window(
            width=width, height=height,
            caption="PolyGrid Globe — Phase 13 (PBR + Water + Atmosphere + Bloom)",
            resizable=True, config=config,
        )
    except pyglet.window.NoSuchConfigException:
        config = pyglet.gl.Config(
            double_buffer=True, depth_size=24,
            major_version=3, minor_version=3,
        )
        window = pyglet.window.Window(
            width=width, height=height,
            caption="PolyGrid Globe — Phase 13 (PBR + Water + Atmosphere + Bloom)",
            resizable=True, config=config,
        )

    # ── Shader compilation helper ───────────────────────────────────
    def _compile_shader(source, shader_type):
        shader = gl.glCreateShader(shader_type)
        source_bytes = source.encode("utf-8")
        length = ctypes.c_int(len(source_bytes))
        buf = ctypes.create_string_buffer(source_bytes)
        ptr = ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))
        arr = (ctypes.POINTER(ctypes.c_char) * 1)(ptr)
        gl.glShaderSource(shader, 1, arr, ctypes.byref(length))
        gl.glCompileShader(shader)
        status = gl.GLint()
        gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS, ctypes.byref(status))
        if not status.value:
            log = ctypes.create_string_buffer(2048)
            gl.glGetShaderInfoLog(shader, 2048, None, log)
            raise RuntimeError(f"Shader compile error:\n{log.value.decode()}")
        return shader

    def _link_program(vs_src, fs_src):
        vs = _compile_shader(vs_src, gl.GL_VERTEX_SHADER)
        fs = _compile_shader(fs_src, gl.GL_FRAGMENT_SHADER)
        prog = gl.glCreateProgram()
        gl.glAttachShader(prog, vs)
        gl.glAttachShader(prog, fs)
        gl.glLinkProgram(prog)
        status = gl.GLint()
        gl.glGetProgramiv(prog, gl.GL_LINK_STATUS, ctypes.byref(status))
        if not status.value:
            log = ctypes.create_string_buffer(2048)
            gl.glGetProgramInfoLog(prog, 2048, None, log)
            raise RuntimeError(f"Shader link error:\n{log.value.decode()}")
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        return prog

    # ── Compile all shader programs ─────────────────────────────────
    pbr_vs, pbr_fs = get_pbr_shader_sources()
    globe_prog = _link_program(pbr_vs, pbr_fs)

    atmo_prog = None
    if enable_atmosphere:
        atmo_vs, atmo_fs = get_atmosphere_shader_sources()
        atmo_prog = _link_program(atmo_vs, atmo_fs)

    bg_vs, bg_fs = get_background_shader_sources()
    bg_prog = _link_program(bg_vs, bg_fs)

    bloom_extract_src, bloom_blur_src, bloom_composite_src = get_bloom_shader_sources()
    bloom_extract_prog = bloom_blur_prog = bloom_composite_prog = None
    if enable_bloom:
        # All bloom passes share the background vertex shader (fullscreen quad)
        bloom_extract_prog = _link_program(bg_vs, bloom_extract_src)
        bloom_blur_prog = _link_program(bg_vs, bloom_blur_src)
        bloom_composite_prog = _link_program(bg_vs, bloom_composite_src)

    # ── Uniform locations: globe PBR ────────────────────────────────
    def _uloc(prog, name):
        return gl.glGetUniformLocation(prog, name.encode("utf-8"))

    g_mvp = _uloc(globe_prog, "u_mvp")
    g_model = _uloc(globe_prog, "u_model")
    g_normal_mat = _uloc(globe_prog, "u_normal_matrix")
    g_atlas = _uloc(globe_prog, "u_atlas")
    g_normal_map = _uloc(globe_prog, "u_normal_map")
    g_use_tex = _uloc(globe_prog, "u_use_texture")
    g_use_nm = _uloc(globe_prog, "u_use_normal_map")
    g_light = _uloc(globe_prog, "u_light_dir")
    g_fill = _uloc(globe_prog, "u_fill_dir")
    g_eye = _uloc(globe_prog, "u_eye_pos")
    g_time = _uloc(globe_prog, "u_time")

    # ── Uniform locations: atmosphere ───────────────────────────────
    a_mvp = a_model = a_eye = None
    if atmo_prog:
        a_mvp = _uloc(atmo_prog, "u_mvp")
        a_model = _uloc(atmo_prog, "u_model")
        a_eye = _uloc(atmo_prog, "u_eye_pos")

    # ── Uniform locations: background ───────────────────────────────
    bg_center = _uloc(bg_prog, "u_center_color")
    bg_edge = _uloc(bg_prog, "u_edge_color")

    # ── Uniform locations: bloom ────────────────────────────────────
    be_scene = be_thresh = None
    bb_source = bb_dir = None
    bc_scene = bc_bloom = bc_intensity = None
    if enable_bloom:
        be_scene = _uloc(bloom_extract_prog, "u_scene")
        be_thresh = _uloc(bloom_extract_prog, "u_threshold")
        bb_source = _uloc(bloom_blur_prog, "u_source")
        bb_dir = _uloc(bloom_blur_prog, "u_direction")
        bc_scene = _uloc(bloom_composite_prog, "u_scene")
        bc_bloom = _uloc(bloom_composite_prog, "u_bloom")
        bc_intensity = _uloc(bloom_composite_prog, "u_bloom_intensity")

    # ── Upload globe VBO + IBO ──────────────────────────────────────
    globe_vao = gl.GLuint()
    gl.glGenVertexArrays(1, ctypes.byref(globe_vao))
    gl.glBindVertexArray(globe_vao)

    globe_vbo = gl.GLuint()
    gl.glGenBuffers(1, ctypes.byref(globe_vbo))
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, globe_vbo)
    vbo_bytes = vertex_data.astype(np.float32).tobytes()
    gl.glBufferData(gl.GL_ARRAY_BUFFER, len(vbo_bytes), vbo_bytes, gl.GL_STATIC_DRAW)

    stride_bytes = stride_floats * 4
    # location 0: position (3 floats, offset 0)
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE,
                             stride_bytes, ctypes.c_void_p(0))
    # location 1: color (3 floats, offset 12)
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE,
                             stride_bytes, ctypes.c_void_p(12))
    # location 2: uv (2 floats, offset 24)
    gl.glEnableVertexAttribArray(2)
    gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE,
                             stride_bytes, ctypes.c_void_p(24))
    # location 3: tangent (3 floats, offset 32)
    gl.glEnableVertexAttribArray(3)
    gl.glVertexAttribPointer(3, 3, gl.GL_FLOAT, gl.GL_FALSE,
                             stride_bytes, ctypes.c_void_p(32))
    # location 4: bitangent (3 floats, offset 44)
    gl.glEnableVertexAttribArray(4)
    gl.glVertexAttribPointer(4, 3, gl.GL_FLOAT, gl.GL_FALSE,
                             stride_bytes, ctypes.c_void_p(44))
    # location 5: water_flag (1 float, offset 56) — only if stride >= 15
    if has_water_channel:
        gl.glEnableVertexAttribArray(5)
        gl.glVertexAttribPointer(5, 1, gl.GL_FLOAT, gl.GL_FALSE,
                                 stride_bytes, ctypes.c_void_p(56))
    else:
        # Provide a constant 0.0 for water_flag
        gl.glDisableVertexAttribArray(5)
        gl.glVertexAttrib1f(5, 0.0)

    globe_ibo = gl.GLuint()
    gl.glGenBuffers(1, ctypes.byref(globe_ibo))
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, globe_ibo)
    ibo_bytes = index_data.astype(np.uint32).tobytes()
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, len(ibo_bytes), ibo_bytes, gl.GL_STATIC_DRAW)
    n_indices = index_data.size

    gl.glBindVertexArray(0)

    # ── Upload atmosphere VBO + IBO ─────────────────────────────────
    atmo_vao = gl.GLuint()
    atmo_n_indices = 0
    if enable_atmosphere and atmo_verts is not None:
        gl.glGenVertexArrays(1, ctypes.byref(atmo_vao))
        gl.glBindVertexArray(atmo_vao)

        abuf = gl.GLuint()
        gl.glGenBuffers(1, ctypes.byref(abuf))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, abuf)
        adata = atmo_verts.astype(np.float32).tobytes()
        gl.glBufferData(gl.GL_ARRAY_BUFFER, len(adata), adata, gl.GL_STATIC_DRAW)

        atmo_stride = 7 * 4  # pos(3) + rgba(4)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE,
                                 atmo_stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE,
                                 atmo_stride, ctypes.c_void_p(12))

        aibo = gl.GLuint()
        gl.glGenBuffers(1, ctypes.byref(aibo))
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, aibo)
        ai_data = atmo_indices.astype(np.uint32).tobytes()
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, len(ai_data), ai_data, gl.GL_STATIC_DRAW)
        atmo_n_indices = atmo_indices.size

        gl.glBindVertexArray(0)

    # ── Upload background quad ──────────────────────────────────────
    bg_vao = gl.GLuint()
    gl.glGenVertexArrays(1, ctypes.byref(bg_vao))
    gl.glBindVertexArray(bg_vao)

    bg_buf = gl.GLuint()
    gl.glGenBuffers(1, ctypes.byref(bg_buf))
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, bg_buf)
    bg_bytes = bg_quad.astype(np.float32).tobytes()
    gl.glBufferData(gl.GL_ARRAY_BUFFER, len(bg_bytes), bg_bytes, gl.GL_STATIC_DRAW)

    bg_stride = 4 * 4  # x, y, u, v
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE,
                             bg_stride, ctypes.c_void_p(0))
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE,
                             bg_stride, ctypes.c_void_p(8))
    gl.glBindVertexArray(0)

    # ── Load colour atlas texture ───────────────────────────────────
    def _load_texture(path, *, flip=True):
        img = Image.open(str(path)).convert("RGBA")
        if flip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        w, h = img.size
        raw = img.tobytes()
        tex = gl.GLuint()
        gl.glGenTextures(1, ctypes.byref(tex))
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, raw)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return tex

    colour_tex = _load_texture(atlas_path)
    normal_tex = _load_texture(normal_atlas_path)

    # ── Bloom FBOs (ping-pong) ──────────────────────────────────────
    scene_fbo = gl.GLuint()
    scene_tex = gl.GLuint()
    scene_depth = gl.GLuint()
    bloom_fbo_a = gl.GLuint()
    bloom_tex_a = gl.GLuint()
    bloom_fbo_b = gl.GLuint()
    bloom_tex_b = gl.GLuint()

    def _create_fbo(w, h):
        """Create an FBO with a colour texture + depth renderbuffer."""
        fbo = gl.GLuint()
        gl.glGenFramebuffers(1, ctypes.byref(fbo))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)

        tex = gl.GLuint()
        gl.glGenTextures(1, ctypes.byref(tex))
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA16F, w, h, 0,
                        gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                  gl.GL_TEXTURE_2D, tex, 0)

        depth = gl.GLuint()
        gl.glGenRenderbuffers(1, ctypes.byref(depth))
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, w, h)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
                                     gl.GL_RENDERBUFFER, depth)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        return fbo, tex, depth

    def _create_color_fbo(w, h):
        """Create an FBO with just a colour texture (no depth)."""
        fbo = gl.GLuint()
        gl.glGenFramebuffers(1, ctypes.byref(fbo))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)

        tex = gl.GLuint()
        gl.glGenTextures(1, ctypes.byref(tex))
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA16F, w, h, 0,
                        gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                  gl.GL_TEXTURE_2D, tex, 0)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        return fbo, tex

    fbo_w, fbo_h = [width], [height]

    def _rebuild_fbos():
        nonlocal scene_fbo, scene_tex, scene_depth
        nonlocal bloom_fbo_a, bloom_tex_a, bloom_fbo_b, bloom_tex_b
        w, h = fbo_w[0], fbo_h[0]
        scene_fbo, scene_tex, scene_depth = _create_fbo(w, h)
        if enable_bloom:
            bloom_fbo_a, bloom_tex_a = _create_color_fbo(w // 2, h // 2)
            bloom_fbo_b, bloom_tex_b = _create_color_fbo(w // 2, h // 2)

    if enable_bloom:
        _rebuild_fbos()

    # ── Camera state ────────────────────────────────────────────────
    yaw = [0.0]
    pitch = [0.0]
    zoom = [3.0]
    start_time = [time.monotonic()]

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
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            right = np.cross(forward, up)
            rn = np.linalg.norm(right)
        right /= rn
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

    # Lighting directions
    light_dir = _normalize(np.array([0.3, 0.8, 0.5], dtype=np.float32))
    fill_dir = _normalize(np.array([-0.5, -0.3, 0.4], dtype=np.float32))

    def _set_mat4(loc, mat):
        gl.glUniformMatrix4fv(
            loc, 1, gl.GL_TRUE,
            mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

    def _set_mat3(loc, mat):
        gl.glUniformMatrix3fv(
            loc, 1, gl.GL_TRUE,
            mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

    def _draw_fullscreen_quad():
        """Draw the background quad VAO as a triangle strip."""
        gl.glBindVertexArray(bg_vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        gl.glBindVertexArray(0)

    # ── Draw function ───────────────────────────────────────────────
    @window.event
    def on_draw():
        elapsed = time.monotonic() - start_time[0]
        w, h = window.width, max(1, window.height)
        aspect = w / h

        # Camera
        proj = _perspective(math.radians(45), aspect, 0.1, 100.0)
        eye = np.array([0.0, 0.0, zoom[0]], dtype=np.float32)
        view = _look_at(eye, (0.0, 0.0, 0.0))
        model = (_rotation_y(yaw[0]) @ _rotation_x(pitch[0])).astype(np.float32)
        mvp = (proj @ view).astype(np.float32)
        normal_matrix = np.linalg.inv(model[:3, :3]).T.astype(np.float32)

        # ── Render to FBO (if bloom) or directly to screen ──────────
        if enable_bloom:
            if fbo_w[0] != w or fbo_h[0] != h:
                fbo_w[0], fbo_h[0] = w, h
                _rebuild_fbos()
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, scene_fbo)
            gl.glViewport(0, 0, w, h)
        else:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(int(gl.GL_COLOR_BUFFER_BIT) | int(gl.GL_DEPTH_BUFFER_BIT))

        # ── Pass 0: Background gradient ─────────────────────────────
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glUseProgram(bg_prog)
        gl.glUniform3f(bg_center, *BG_CENTER_COLOR)
        gl.glUniform3f(bg_edge, *BG_EDGE_COLOR)
        _draw_fullscreen_quad()

        # ── Pass 1: Globe (PBR) ─────────────────────────────────────
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glUseProgram(globe_prog)

        _set_mat4(g_mvp, mvp)
        _set_mat4(g_model, model)
        _set_mat3(g_normal_mat, normal_matrix)
        gl.glUniform3f(g_light, *light_dir)
        gl.glUniform3f(g_fill, *fill_dir)
        gl.glUniform3f(g_eye, *eye)
        gl.glUniform1f(g_time, elapsed)
        gl.glUniform1i(g_use_tex, 1)
        gl.glUniform1i(g_use_nm, 1)

        # Bind atlas on unit 0
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, colour_tex)
        gl.glUniform1i(g_atlas, 0)

        # Bind normal map on unit 1
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, normal_tex)
        gl.glUniform1i(g_normal_map, 1)

        gl.glBindVertexArray(globe_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, n_indices, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        # ── Pass 2: Atmosphere shell ────────────────────────────────
        if enable_atmosphere and atmo_prog and atmo_n_indices > 0:
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glDepthMask(gl.GL_FALSE)  # don't write depth for translucent
            gl.glDisable(gl.GL_CULL_FACE)

            gl.glUseProgram(atmo_prog)
            _set_mat4(a_mvp, mvp)
            _set_mat4(a_model, model)
            gl.glUniform3f(a_eye, *eye)

            gl.glBindVertexArray(atmo_vao)
            gl.glDrawElements(gl.GL_TRIANGLES, atmo_n_indices, gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)

            gl.glDepthMask(gl.GL_TRUE)
            gl.glDisable(gl.GL_BLEND)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glUseProgram(0)

        # ── Pass 3: Bloom post-processing ───────────────────────────
        if enable_bloom:
            gl.glDisable(gl.GL_DEPTH_TEST)
            bw, bh = w // 2, h // 2

            # 3a: Extract bright pixels
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, bloom_fbo_a)
            gl.glViewport(0, 0, bw, bh)
            gl.glClear(int(gl.GL_COLOR_BUFFER_BIT))
            gl.glUseProgram(bloom_extract_prog)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, scene_tex)
            gl.glUniform1i(be_scene, 0)
            gl.glUniform1f(be_thresh, BLOOM_THRESHOLD)
            _draw_fullscreen_quad()

            # 3b: Horizontal blur
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, bloom_fbo_b)
            gl.glClear(int(gl.GL_COLOR_BUFFER_BIT))
            gl.glUseProgram(bloom_blur_prog)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, bloom_tex_a)
            gl.glUniform1i(bb_source, 0)
            gl.glUniform2f(bb_dir, 1.0 / max(1, bw), 0.0)
            _draw_fullscreen_quad()

            # 3c: Vertical blur
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, bloom_fbo_a)
            gl.glClear(int(gl.GL_COLOR_BUFFER_BIT))
            gl.glBindTexture(gl.GL_TEXTURE_2D, bloom_tex_b)
            gl.glUniform2f(bb_dir, 0.0, 1.0 / max(1, bh))
            _draw_fullscreen_quad()

            # 3d: Composite bloom + scene to screen
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
            gl.glViewport(0, 0, w, h)
            gl.glClear(int(gl.GL_COLOR_BUFFER_BIT))
            gl.glUseProgram(bloom_composite_prog)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, scene_tex)
            gl.glUniform1i(bc_scene, 0)
            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, bloom_tex_a)
            gl.glUniform1i(bc_bloom, 1)
            gl.glUniform1f(bc_intensity, BLOOM_INTENSITY)
            _draw_fullscreen_quad()

            # Cleanup
            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glUseProgram(0)
            gl.glEnable(gl.GL_DEPTH_TEST)

    # ── Input handlers ──────────────────────────────────────────────
    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        yaw[0] += dx * 0.01
        pitch[0] += dy * 0.01
        pitch[0] = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, pitch[0]))

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        zoom[0] = max(1.5, min(10.0, zoom[0] - scroll_y * 0.3))

    @window.event
    def on_key_press(symbol, modifiers):
        """Toggle features with keyboard shortcuts."""
        nonlocal enable_bloom, enable_atmosphere
        if symbol == pyglet.window.key.B:
            enable_bloom = not enable_bloom
            print(f"  Bloom: {'ON' if enable_bloom else 'OFF'}")
            if enable_bloom:
                _rebuild_fbos()
        elif symbol == pyglet.window.key.A:
            enable_atmosphere = not enable_atmosphere
            print(f"  Atmosphere: {'ON' if enable_atmosphere else 'OFF'}")

    print("\n  ╔══════════════════════════════════════════════════╗")
    print("  ║  Phase 13 Globe Viewer                          ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print("  ║  Mouse drag  — rotate globe                     ║")
    print("  ║  Scroll      — zoom in/out                      ║")
    print("  ║  B key       — toggle bloom                     ║")
    print("  ║  A key       — toggle atmosphere                ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()

    pyglet.app.run()


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 13 integrated globe viewer (PBR + water + atmosphere + bloom)",
    )
    parser.add_argument("-f", "--frequency", type=int, default=3,
                        help="Goldberg polyhedron frequency (default: 3)")
    parser.add_argument("--detail-rings", type=int, default=4,
                        help="Detail grid ring count (default: 4)")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--tile-size", type=int, default=256,
                        help="Atlas tile size in pixels")
    parser.add_argument("--subdivisions", type=int, default=3,
                        help="Mesh subdivision level (1-5)")
    parser.add_argument("--preset", type=str, default="earthlike",
                        choices=["earthlike", "mountainous", "archipelago", "pangaea"],
                        help="Terrain preset")
    parser.add_argument("--no-bloom", action="store_true",
                        help="Disable bloom post-processing")
    parser.add_argument("--no-atmosphere", action="store_true",
                        help="Disable atmosphere shell")
    parser.add_argument("--no-water", action="store_true",
                        help="Disable water detection")
    parser.add_argument("--soft-blend", action="store_true", default=False,
                        help="Enable Phase 16 soft tile-edge blending (fullslot + 16B-D)")
    parser.add_argument("-W", "--width", type=int, default=1100)
    parser.add_argument("-H", "--height", type=int, default=850)
    args = parser.parse_args()

    data = _build_terrain(
        frequency=args.frequency,
        detail_rings=args.detail_rings,
        seed=args.seed,
        preset=args.preset,
        tile_size=args.tile_size,
        soft_blend=args.soft_blend,
    )

    launch_viewer(
        data,
        subdivisions=args.subdivisions,
        width=args.width,
        height=args.height,
        enable_bloom=not args.no_bloom,
        enable_atmosphere=not args.no_atmosphere,
        enable_water=not args.no_water,
    )


if __name__ == "__main__":
    main()
