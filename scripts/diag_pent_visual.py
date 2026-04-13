#!/usr/bin/env python3
"""Generate a visual direction-test atlas for pentagon orientation debugging.

Renders each pentagon's detail cells with directional arrows/gradients
that make rotation/reflection immediately visible on the 3D globe.
Produces:
  1. The raw placeholder atlas (index map visualised)
  2. A directional-coded atlas (each cell has a gradient pointing away from center)
  3. A 3D globe render using a software rasterizer

Output directory: /tmp/pent_orientation_test/
"""
import sys
import math
import os
import io
from pathlib import Path

# Activate pgrid venv
venv_path = Path(__file__).resolve().parent.parent / ".venv" / "lib"
for p in sorted(venv_path.glob("python3.*")):
    sp = str(p / "site-packages")
    if sp not in sys.path:
        sys.path.insert(0, sp)

src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
from PIL import Image, ImageDraw

OUT_DIR = Path("/tmp/pent_orientation_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FREQ = 2
DETAIL_RINGS = 4
TILE_SIZE = 256
GUTTER = 4

# ─── Build placeholder artifact ────────────────────────────────
os.environ["PGRID_PLACEHOLDER_DEBUG_DIR"] = str(OUT_DIR)
from polygrid.placeholder_atlas import get_or_build_artifact, _build_artifact
from polygrid.integration import PlaceholderAtlasSpec

spec = PlaceholderAtlasSpec(
    frequency=FREQ,
    detail_rings=DETAIL_RINGS,
    tile_size=TILE_SIZE,
    gutter=GUTTER,
    base_color=(0.5, 0.5, 0.5),
    noise_amount=0.0,
    seed=42,
)

# Force rebuild (bypass cache) to ensure consistent state
artifact = _build_artifact(FREQ, DETAIL_RINGS, TILE_SIZE, GUTTER)

print(f"Artifact built: {artifact.atlas_width}x{artifact.atlas_height}")
print(f"Index keys: {len(artifact.index_keys)}")
print(f"Topology mode: {artifact.topology_mode}")

# ─── Save index map visualization ──────────────────────────────
idx_map = artifact.tile_index_map.astype(np.uint32)
vis = np.zeros((idx_map.shape[0], idx_map.shape[1], 3), dtype=np.uint8)
vis[..., 0] = (idx_map * 73) % 256
vis[..., 1] = (idx_map * 151) % 256
vis[..., 2] = (idx_map * 199) % 256
# Background: index 0xFFFF
bg_mask = artifact.tile_index_map == 0xFFFF
vis[bg_mask] = [32, 32, 32]
Image.fromarray(vis, "RGB").save(OUT_DIR / "index_map.png")
print(f"Saved index map: {OUT_DIR / 'index_map.png'}")

# ─── Create a directional-coded atlas ───────────────────────────
# For each patch in the atlas, create a gradient that points RADIALLY
# outward from the tile center so rotation/reflection will be obvious
from polygrid.rendering.uv_texture import get_tile_uv_vertices, get_goldberg_tiles
from polygrid.globe.globe import build_globe_grid
from polygrid.rendering.globe_renderer_v2 import build_batched_globe_mesh

globe_grid = build_globe_grid(FREQ)
face_ids = sorted(globe_grid.faces.keys(), key=lambda f: int(f[1:]))
tiles = get_goldberg_tiles(FREQ, 1.0)
tile_by_fid = {f"t{t.index}": t for t in tiles}

# Build a per-pixel atlas with angle-coded colours
atlas_h, atlas_w = artifact.tile_index_map.shape
dir_atlas = np.full((atlas_h, atlas_w, 3), 32, dtype=np.uint8)

# For each tile slot, draw a radial gradient with edge-specific colours
from polygrid.rendering.atlas_utils import compute_atlas_layout

n_tiles = len(face_ids)
cols, rows, aw, ah = compute_atlas_layout(n_tiles, TILE_SIZE, GUTTER)
slot_size = TILE_SIZE + 2 * GUTTER

# Assign 5 distinct edge colours for pentagons, 6 for hexagons
PENT_EDGE_COLORS = [
    (255, 0, 0),    # Edge 0: Red
    (0, 255, 0),    # Edge 1: Green
    (0, 0, 255),    # Edge 2: Blue
    (255, 255, 0),  # Edge 3: Yellow
    (255, 0, 255),  # Edge 4: Magenta
]
HEX_EDGE_COLORS = [
    (255, 0, 0),    # Edge 0: Red
    (0, 255, 0),    # Edge 1: Green
    (0, 0, 255),    # Edge 2: Blue
    (255, 255, 0),  # Edge 3: Yellow
    (255, 0, 255),  # Edge 4: Magenta
    (0, 255, 255),  # Edge 5: Cyan
]

for idx, fid in enumerate(face_ids):
    slot = artifact.uv_layout.get(fid)
    if not slot:
        continue

    u_min, v_min, u_max, v_max = slot
    tile = tile_by_fid[fid]
    uv_verts = get_tile_uv_vertices(globe_grid, fid)
    n_sides = len(uv_verts)
    is_pent = n_sides == 5

    # Tile slot location in atlas
    col = idx % cols
    row = idx // cols
    sx = col * slot_size
    sy = row * slot_size

    # UV polygon center in atlas pixel coords
    uv_cx = sum(u for u, _ in uv_verts) / n_sides
    uv_cy = sum(v for _, v in uv_verts) / n_sides

    # Convert UV corners to atlas pixel coords
    inner_x = u_min * atlas_w
    inner_y = (1.0 - v_max) * atlas_h
    inner_w = (u_max - u_min) * atlas_w
    inner_h = (v_max - v_min) * atlas_h

    uv_px_corners = []
    for u, v in uv_verts:
        px_x = inner_x + u * inner_w
        px_y = inner_y + (1.0 - v) * inner_h
        uv_px_corners.append((px_x, px_y))

    center_px = (
        inner_x + uv_cx * inner_w,
        inner_y + (1.0 - uv_cy) * inner_h,
    )

    # For each pixel in the slot, determine which edge sector it's closest to
    # and color accordingly
    for py in range(max(0, sy), min(atlas_h, sy + slot_size)):
        for px in range(max(0, sx), min(atlas_w, sx + slot_size)):
            # Vector from center to this pixel
            dx = px - center_px[0]
            dy = py - center_px[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 0.5:
                dir_atlas[py, px] = [128, 128, 128]
                continue

            # Find which UV polygon sector this point falls in
            angle = math.atan2(dy, dx)
            best_edge = 0
            best_angle_diff = 999.0
            for eidx in range(n_sides):
                j = (eidx + 1) % n_sides
                mid_px = (
                    (uv_px_corners[eidx][0] + uv_px_corners[j][0]) / 2,
                    (uv_px_corners[eidx][1] + uv_px_corners[j][1]) / 2,
                )
                edge_dx = mid_px[0] - center_px[0]
                edge_dy = mid_px[1] - center_px[1]
                edge_angle = math.atan2(edge_dy, edge_dx)
                diff = abs(angle - edge_angle)
                if diff > math.pi:
                    diff = 2 * math.pi - diff
                if diff < best_angle_diff:
                    best_angle_diff = diff
                    best_edge = eidx

            colors = PENT_EDGE_COLORS if is_pent else HEX_EDGE_COLORS
            color = colors[best_edge % len(colors)]

            # Fade with distance from center for gradient effect
            fade = min(1.0, dist / (slot_size * 0.4))
            r = int(color[0] * fade + 128 * (1 - fade))
            g = int(color[1] * fade + 128 * (1 - fade))
            b = int(color[2] * fade + 128 * (1 - fade))
            dir_atlas[py, px] = [r, g, b]

Image.fromarray(dir_atlas, "RGB").save(OUT_DIR / "directional_atlas.png")
print(f"Saved directional atlas: {OUT_DIR / 'directional_atlas.png'}")

# ─── Build 3D globe mesh and render ────────────────────────────
print("Building 3D globe mesh...")
vdata, idata = build_batched_globe_mesh(
    frequency=FREQ,
    uv_layout=artifact.uv_layout,
    subdivisions=5,
)

# Software rasterizer with the directional atlas
from PIL import Image as PILImage

RENDER_SIZE = 1024
render_img = np.full((RENDER_SIZE, RENDER_SIZE, 3), 16, dtype=np.uint8)
depth_buf = np.full((RENDER_SIZE, RENDER_SIZE), np.inf, dtype=np.float64)

atlas_tex = dir_atlas.astype(np.float64)

# Camera: look at the north pole region (where t0 pentagon is)
# For f=2, t0 is at the "north pole" of the icosahedron
t0_center = np.array(tile_by_fid["t0"].center, dtype=float)
t0_center_norm = t0_center / np.linalg.norm(t0_center)

# Camera facing t0's center
cam_dist = 2.5
cam_pos = t0_center_norm * cam_dist
cam_target = np.array([0.0, 0.0, 0.0])
cam_up = np.array([0.0, 1.0, 0.0])

# Build view matrix
cam_fwd = cam_target - cam_pos
cam_fwd = cam_fwd / np.linalg.norm(cam_fwd)
cam_right = np.cross(cam_fwd, cam_up)
cam_right = cam_right / np.linalg.norm(cam_right)
cam_up = np.cross(cam_right, cam_fwd)

fov = 45.0
aspect = 1.0
near, far = 0.1, 10.0
f_val = 1.0 / math.tan(math.radians(fov) / 2)

# Projection matrix
proj = np.zeros((4, 4))
proj[0, 0] = f_val / aspect
proj[1, 1] = f_val
proj[2, 2] = (far + near) / (near - far)
proj[2, 3] = 2 * far * near / (near - far)
proj[3, 2] = -1.0

# View matrix
view = np.eye(4)
view[0, :3] = cam_right
view[1, :3] = cam_up
view[2, :3] = -cam_fwd
view[0, 3] = -np.dot(cam_right, cam_pos)
view[1, 3] = -np.dot(cam_up, cam_pos)
view[2, 3] = np.dot(cam_fwd, cam_pos)

mvp = proj @ view

def project_vertex(pos):
    p = np.array([pos[0], pos[1], pos[2], 1.0])
    clip = mvp @ p
    if abs(clip[3]) < 1e-12:
        return None
    ndc = clip[:3] / clip[3]
    sx = (ndc[0] * 0.5 + 0.5) * RENDER_SIZE
    sy = (1.0 - (ndc[1] * 0.5 + 0.5)) * RENDER_SIZE
    return (sx, sy, ndc[2])


def sample_atlas(u, v):
    """Sample the directional atlas at UV coordinates."""
    px = u * atlas_w
    py = (1.0 - v) * atlas_h
    ix = int(np.clip(px, 0, atlas_w - 1))
    iy = int(np.clip(py, 0, atlas_h - 1))
    return atlas_tex[iy, ix]


def rasterize_triangle(v0, v1, v2, uv0, uv1, uv2):
    """Rasterize a single triangle with UV-mapped texture."""
    sx0, sy0, sz0 = v0
    sx1, sy1, sz1 = v1
    sx2, sy2, sz2 = v2

    min_x = max(0, int(min(sx0, sx1, sx2)))
    max_x = min(RENDER_SIZE - 1, int(max(sx0, sx1, sx2)) + 1)
    min_y = max(0, int(min(sy0, sy1, sy2)))
    max_y = min(RENDER_SIZE - 1, int(max(sy0, sy1, sy2)) + 1)

    denom = (sy1 - sy2) * (sx0 - sx2) + (sx2 - sx1) * (sy0 - sy2)
    if abs(denom) < 1e-10:
        return

    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            w0 = ((sy1 - sy2) * (px - sx2) + (sx2 - sx1) * (py - sy2)) / denom
            w1 = ((sy2 - sy0) * (px - sx2) + (sx0 - sx2) * (py - sy2)) / denom
            w2 = 1.0 - w0 - w1

            if w0 < -0.001 or w1 < -0.001 or w2 < -0.001:
                continue

            z = w0 * sz0 + w1 * sz1 + w2 * sz2
            if z >= depth_buf[py, px]:
                continue

            u = w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0]
            v = w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1]

            color = sample_atlas(u, v)
            depth_buf[py, px] = z
            render_img[py, px] = color.astype(np.uint8)


print(f"Rasterizing {len(idata)} triangles...")
stride = vdata.shape[1]
total = len(idata)
for tri_idx, tri in enumerate(idata):
    if tri_idx % 500 == 0:
        print(f"  Triangle {tri_idx}/{total}...")

    i0, i1, i2 = tri
    pos0 = vdata[i0, :3]
    pos1 = vdata[i1, :3]
    pos2 = vdata[i2, :3]
    uv_0 = vdata[i0, 6:8]
    uv_1 = vdata[i1, 6:8]
    uv_2 = vdata[i2, 6:8]

    p0 = project_vertex(pos0)
    p1 = project_vertex(pos1)
    p2 = project_vertex(pos2)

    if p0 is None or p1 is None or p2 is None:
        continue

    # Backface cull
    edge1 = np.array([p1[0] - p0[0], p1[1] - p0[1]])
    edge2 = np.array([p2[0] - p0[0], p2[1] - p0[1]])
    cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
    if cross < 0:
        continue

    rasterize_triangle(p0, p1, p2, uv_0, uv_1, uv_2)

result = Image.fromarray(render_img, "RGB")
result.save(OUT_DIR / "globe_directional.png")
print(f"Saved globe render: {OUT_DIR / 'globe_directional.png'}")

# ─── Also draw labels on the globe render ───────────────────────
result_labeled = result.copy()
draw = ImageDraw.Draw(result_labeled)

# Project each tile center to screen and label it
for fid in face_ids:
    tile = tile_by_fid[fid]
    center = np.array(tile.center, dtype=float)
    # Only label if facing camera
    to_cam = cam_pos - center
    normal = center / np.linalg.norm(center)
    if np.dot(normal, to_cam) < 0:
        continue

    p = project_vertex(center)
    if p is None:
        continue
    sx, sy, _ = p
    if 0 <= sx < RENDER_SIZE and 0 <= sy < RENDER_SIZE:
        n_sides = len(globe_grid.faces[fid].vertex_ids)
        label = f"{fid}{'*' if n_sides == 5 else ''}"
        draw.text((sx - 10, sy - 5), label, fill=(255, 255, 255))

result_labeled.save(OUT_DIR / "globe_directional_labeled.png")
print(f"Saved labeled render: {OUT_DIR / 'globe_directional_labeled.png'}")

print("\nDone! Check /tmp/pent_orientation_test/ for outputs.")
