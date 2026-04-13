#!/usr/bin/env python3
"""Directional atlas test: render arrows showing cell orientation.

Each atlas slot gets arrows that point "up" in the tile's tangent plane.
Pentagon slots and hex slots should show consistent arrow direction if
the warp/orientation is correct. Any rotation or reflection will be
immediately visible.
"""
import sys, math, io
from pathlib import Path
from typing import Dict, Tuple

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
from polygrid.placeholder_atlas import _build_artifact, recolor_atlas
from polygrid.integration import PlaceholderAtlasSpec
from polygrid.globe.globe import build_globe_grid
from polygrid.rendering.uv_texture import get_goldberg_tiles, compute_tile_basis
from polygrid.rendering.globe_renderer_v2 import build_batched_globe_mesh

FREQ = 2
DETAIL_RINGS = 4
TILE_SIZE = 256
GUTTER = 4
OUT_DIR = Path("/tmp/pent_directional")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Build artifact
artifact = _build_artifact(FREQ, DETAIL_RINGS, TILE_SIZE, GUTTER)
globe = build_globe_grid(FREQ)
tiles_gb = get_goldberg_tiles(FREQ, 1.0)

# Create an atlas with directional arrows in each slot
# Each tile gets its tangent direction projected to 2D
# Pentagon slots are colored RED, hex slots GREEN

atlas_w = artifact.atlas_width
atlas_h = artifact.atlas_height
atlas_img = Image.new("RGB", (atlas_w, atlas_h), (40, 40, 40))
draw = ImageDraw.Draw(atlas_img)

for tile in tiles_gb:
    fid = f"t{tile.index}"
    slot = artifact.uv_layout.get(fid)
    if slot is None:
        continue
    
    u_min, v_min, u_max, v_max = slot
    # Convert to pixel coordinates
    px_left = int(u_min * atlas_w)
    px_right = int(u_max * atlas_w)
    # OpenGL texture: v=0 at bottom, PIL image: y=0 at top
    # So v_min (bottom) maps to higher y, v_max (top) to lower y
    px_top = int((1.0 - v_max) * atlas_h)
    px_bottom = int((1.0 - v_min) * atlas_h)
    
    ns = len(tile.vertices)
    is_pent = ns == 5
    
    # Fill slot with base color
    base_color = (180, 80, 80) if is_pent else (80, 150, 80)
    draw.rectangle([px_left, px_top, px_right, px_bottom], fill=base_color)
    
    # Draw tile label
    cx = (px_left + px_right) // 2
    cy = (px_top + px_bottom) // 2
    draw.text((cx - 10, cy - 5), fid, fill=(255, 255, 255))
    
    # Draw a directional arrow from center pointing "up" in UV space
    # In OpenGL UV: "up" is +v direction = -y in PIL
    arrow_len = (px_bottom - px_top) * 0.3
    arrow_start = (cx, cy + int(arrow_len * 0.4))
    arrow_end = (cx, cy - int(arrow_len * 0.6))
    draw.line([arrow_start, arrow_end], fill=(255, 255, 0), width=3)
    # Arrowhead
    head_size = 8
    draw.polygon([
        arrow_end,
        (arrow_end[0] - head_size, arrow_end[1] + head_size),
        (arrow_end[0] + head_size, arrow_end[1] + head_size),
    ], fill=(255, 255, 0))
    
    # Also draw a small circle at UV corner 0 to show orientation
    uv0_px_x = px_left + int((tile.uv_vertices[0][0]) * (px_right - px_left))
    uv0_px_y = px_top + int((1.0 - tile.uv_vertices[0][1]) * (px_bottom - px_top))
    draw.ellipse([uv0_px_x - 5, uv0_px_y - 5, uv0_px_x + 5, uv0_px_y + 5], fill=(255, 0, 255))

atlas_img.save(OUT_DIR / "directional_atlas.png")
print(f"Saved directional atlas: {atlas_w}x{atlas_h}")

# Now render the globe using a software rasterizer
print("Building globe mesh...")
vdata, idata = build_batched_globe_mesh(
    frequency=FREQ,
    uv_layout=artifact.uv_layout,
    radius=1.0,
    subdivisions=4,
)

# Flip atlas for OpenGL convention (PIL is Y-flipped from OpenGL)
atlas_arr = np.array(atlas_img.convert("RGB"), dtype=np.uint8)
atlas_gl = atlas_arr[::-1]  # flip Y for OpenGL sampling simulation

# Software rasterizer
def render_globe(view_dir, img_size=800, filename="globe.png"):
    """Render the globe from a given view direction using simple software raster."""
    # Camera setup
    view_dir = np.array(view_dir, dtype=np.float64)
    view_dir /= np.linalg.norm(view_dir)
    
    # Build orthonormal basis
    up = np.array([0, 1, 0], dtype=np.float64)
    if abs(np.dot(view_dir, up)) > 0.99:
        up = np.array([0, 0, 1], dtype=np.float64)
    right = np.cross(view_dir, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, view_dir)
    up /= np.linalg.norm(up)
    
    out_img = np.full((img_size, img_size, 3), 20, dtype=np.uint8)
    zbuf = np.full((img_size, img_size), -1e9, dtype=np.float64)
    
    # Extract triangles
    pos = vdata[:, :3]
    colors = vdata[:, 3:6]
    uvs = vdata[:, 6:8]
    
    scale = img_size * 0.45
    cx_screen = img_size / 2
    cy_screen = img_size / 2
    
    for tri_idx in range(len(idata)):
        i0, i1, i2 = idata[tri_idx]
        
        # Project vertices
        p0 = pos[i0]; p1 = pos[i1]; p2 = pos[i2]
        
        # Simple depth-based visibility: check if facing camera
        tri_center = (p0 + p1 + p2) / 3.0
        tri_normal = np.cross(p1 - p0, p2 - p0)
        tn_len = np.linalg.norm(tri_normal)
        if tn_len < 1e-10:
            continue
        tri_normal /= tn_len
        if np.dot(tri_normal, view_dir) >= 0:
            continue
        
        # Project to screen (orthographic for simplicity)
        def proj(p):
            x = np.dot(p, right) * scale + cx_screen
            y = -np.dot(p, up) * scale + cy_screen  # flip Y for screen
            z = np.dot(p, view_dir)
            return x, y, z
        
        x0, y0, z0 = proj(p0)
        x1, y1, z1 = proj(p1)
        x2, y2, z2 = proj(p2)
        
        # Bounding box
        min_x = max(0, int(min(x0, x1, x2)))
        max_x = min(img_size - 1, int(max(x0, x1, x2)) + 1)
        min_y = max(0, int(min(y0, y1, y2)))
        max_y = min(img_size - 1, int(max(y0, y1, y2)) + 1)
        
        # Barycentric rasterization
        def edge(ax, ay, bx, by, px, py):
            return (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        
        area = edge(x0, y0, x1, y1, x2, y2)
        if abs(area) < 1e-6:
            continue
        
        uv0 = uvs[i0]; uv1 = uvs[i1]; uv2 = uvs[i2]
        
        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                w0 = edge(x1, y1, x2, y2, px + 0.5, py + 0.5) / area
                w1 = edge(x2, y2, x0, y0, px + 0.5, py + 0.5) / area
                w2 = 1.0 - w0 - w1
                
                if w0 < -0.001 or w1 < -0.001 or w2 < -0.001:
                    continue
                
                z = w0 * z0 + w1 * z1 + w2 * z2
                if z <= zbuf[py, px]:
                    continue
                zbuf[py, px] = z
                
                # Interpolate UV
                u = w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0]
                v = w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1]
                
                # Sample atlas (OpenGL convention: v=0 at bottom)
                # atlas_gl is already Y-flipped for this
                tx = int(np.clip(u * atlas_w, 0, atlas_w - 1))
                ty = int(np.clip((1.0 - v) * atlas_h, 0, atlas_h - 1))
                
                out_img[py, px] = atlas_gl[ty, tx]
    
    # Add simple hemisphere lighting
    # Skip - just use the raw texture colors for directional clarity
    
    Image.fromarray(out_img).save(OUT_DIR / filename)
    print(f"  Saved {filename}")


# Render from multiple angles, focusing on areas where pentagons are visible
views = [
    ([-1, 0.5, 0], "view_pent0.png"),  # Should show t0 (north pole pent)
    ([0, 1, 0.1], "view_top.png"),      # Top view
    ([0, -1, 0.1], "view_bottom.png"),  # Bottom view  
    ([1, 0.3, 0], "view_side1.png"),    # Side view
    ([0, 0.3, 1], "view_side2.png"),    # Another side
]

print("Rendering globe views...")
for view_dir, fname in views:
    render_globe(view_dir, img_size=800, filename=fname)

print(f"\nDone! Check {OUT_DIR}")
