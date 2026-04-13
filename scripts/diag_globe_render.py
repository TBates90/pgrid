#!/usr/bin/env python3
"""Render the recolored atlas on a globe to check pentagon visibility."""
import sys, math
from pathlib import Path

venv_path = Path(__file__).resolve().parent.parent / ".venv" / "lib"
for p in sorted(venv_path.glob("python3.*")):
    sp = str(p / "site-packages")
    if sp not in sys.path:
        sys.path.insert(0, sp)
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
from PIL import Image
from polygrid.placeholder_atlas import _build_artifact, recolor_atlas
from polygrid.integration import PlaceholderAtlasSpec
from polygrid.rendering.globe_renderer_v2 import build_batched_globe_mesh

FREQ = 2
DETAIL_RINGS = 4
TILE_SIZE = 256
GUTTER = 4
OUT = Path("/tmp/pent_globe_render")
OUT.mkdir(parents=True, exist_ok=True)

artifact = _build_artifact(FREQ, DETAIL_RINGS, TILE_SIZE, GUTTER)

# Use same spec the playground would use
spec = PlaceholderAtlasSpec(
    frequency=FREQ,
    detail_rings=DETAIL_RINGS,
    tile_size=TILE_SIZE,
    gutter=GUTTER,
    base_color=(0.35, 0.55, 0.3),   # Green-ish land
    noise_amount=0.08,
    seed=42,
)
atlas_png = recolor_atlas(artifact, spec)
atlas_img = Image.open(__import__("io").BytesIO(atlas_png)).convert("RGB")
atlas_arr = np.array(atlas_img, dtype=np.uint8)
# Flip for OpenGL (v=0 at bottom)
atlas_gl = atlas_arr[::-1].copy()
atlas_h, atlas_w = atlas_gl.shape[:2]

print(f"Atlas: {atlas_w}x{atlas_h}")
atlas_img.save(OUT / "atlas.png")

# Build mesh
vdata, idata = build_batched_globe_mesh(
    frequency=FREQ,
    uv_layout=artifact.uv_layout,
    radius=1.0,
    subdivisions=5,  # Higher for better quality
)
print(f"Mesh: {len(vdata)} verts, {len(idata)} tris")

def render(view_dir, img_size=1000, filename="globe.png"):
    view_dir = np.array(view_dir, dtype=np.float64)
    view_dir /= np.linalg.norm(view_dir)
    up = np.array([0, 1, 0], dtype=np.float64)
    if abs(np.dot(view_dir, up)) > 0.99:
        up = np.array([0, 0, 1], dtype=np.float64)
    right = np.cross(view_dir, up); right /= np.linalg.norm(right)
    up = np.cross(right, view_dir); up /= np.linalg.norm(up)
    
    out = np.full((img_size, img_size, 3), 15, dtype=np.uint8)
    zbuf = np.full((img_size, img_size), -1e9, dtype=np.float64)
    
    pos = vdata[:, :3]
    uvs = vdata[:, 6:8]
    scale = img_size * 0.45
    cx = cy = img_size / 2.0
    
    # Light direction (sun from upper-right)
    light = np.array([0.5, 0.7, -0.5], dtype=np.float64)
    light /= np.linalg.norm(light)
    
    for tri in idata:
        i0, i1, i2 = tri
        p0, p1, p2 = pos[i0], pos[i1], pos[i2]
        n = np.cross(p1 - p0, p2 - p0)
        nl = np.linalg.norm(n)
        if nl < 1e-10: continue
        n /= nl
        if np.dot(n, view_dir) >= 0: continue
        
        def proj(p):
            return (np.dot(p, right)*scale+cx, -np.dot(p, up)*scale+cy, np.dot(p, view_dir))
        
        sx0,sy0,sz0 = proj(p0); sx1,sy1,sz1 = proj(p1); sx2,sy2,sz2 = proj(p2)
        
        bx0 = max(0, int(min(sx0,sx1,sx2)))
        bx1 = min(img_size-1, int(max(sx0,sx1,sx2))+1)
        by0 = max(0, int(min(sy0,sy1,sy2)))
        by1 = min(img_size-1, int(max(sy0,sy1,sy2))+1)
        
        area = (sx1-sx0)*(sy2-sy0) - (sx2-sx0)*(sy1-sy0)
        if abs(area) < 0.5: continue
        
        uv0, uv1, uv2 = uvs[i0], uvs[i1], uvs[i2]
        
        # hemisphere lighting
        ndl = max(0.15, float(np.dot(n, light)))
        
        for py in range(by0, by1+1):
            for px in range(bx0, bx1+1):
                ppx, ppy = px+0.5, py+0.5
                w0 = ((sx1-ppx)*(sy2-ppy)-(sx2-ppx)*(sy1-ppy))/area
                w1 = ((sx2-ppx)*(sy0-ppy)-(sx0-ppx)*(sy2-ppy))/area
                w2 = 1.0-w0-w1
                if w0<-0.001 or w1<-0.001 or w2<-0.001: continue
                z = w0*sz0+w1*sz1+w2*sz2
                if z<=zbuf[py,px]: continue
                zbuf[py,px]=z
                u = w0*uv0[0]+w1*uv1[0]+w2*uv2[0]
                v = w0*uv0[1]+w1*uv1[1]+w2*uv2[1]
                tx = int(np.clip(u*atlas_w, 0, atlas_w-1))
                ty = int(np.clip((1.0-v)*atlas_h, 0, atlas_h-1))
                c = atlas_gl[ty, tx].astype(np.float64)*ndl
                out[py,px] = np.clip(c, 0, 255).astype(np.uint8)
    
    Image.fromarray(out).save(OUT / filename)
    print(f"  {filename}")

print("Rendering...")
render([-1, 0.3, 0.2], filename="view1.png")
render([0, 0, -1], filename="view2.png")  
render([0.5, 0.8, 0.3], filename="view3.png")
render([0, -1, 0.1], filename="view_south.png")
print(f"Done → {OUT}")
