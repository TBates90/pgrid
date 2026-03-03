"""Texture export pipeline — Phase 18D.

Output proper texture assets suitable for game engines: power-of-two
atlas dimensions, mipmap chains, KTX2 containers, channel-packed
material textures (ORM), and glTF 2.0 export.

Functions
---------
- :func:`next_power_of_two` — round up to nearest PoT
- :func:`resize_atlas_pot` — resize an atlas to power-of-two dimensions
- :func:`generate_atlas_mipmaps` — generate a full mipmap chain
- :func:`export_atlas_ktx2` — export atlas as KTX2 container (if toolchain available)
- :func:`build_orm_atlas` — build Occlusion/Roughness/Metallic channel-packed atlas
- :func:`build_material_set` — generate albedo + normal + ORM atlas set
- :func:`export_globe_gltf` — export textured globe as glTF 2.0
"""

from __future__ import annotations

import io
import json
import logging
import math
import struct
import warnings
from base64 import b64encode
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ═══════════════════════════════════════════════════════════════════
# 18D.1 — Power-of-two atlas dimensions
# ═══════════════════════════════════════════════════════════════════

def next_power_of_two(n: int) -> int:
    """Return the smallest power of two ≥ *n*.

    >>> next_power_of_two(1000)
    1024
    >>> next_power_of_two(2048)
    2048
    """
    if n <= 0:
        return 1
    # Bit-twiddling: fill all bits below the highest set bit, then add 1
    p = 1
    while p < n:
        p <<= 1
    return p


def atlas_pot_size(
    n_tiles: int,
    tile_size: int = 256,
    gutter: int = 4,
    *,
    max_size: int = 8192,
) -> Tuple[int, int, int]:
    """Compute power-of-two atlas dimensions for *n_tiles*.

    Finds the smallest square PoT atlas that fits all tiles at the
    given *tile_size* and *gutter*.

    Parameters
    ----------
    n_tiles : int
    tile_size : int
    gutter : int
    max_size : int
        GPU maximum texture size limit.

    Returns
    -------
    (atlas_size, columns, rows)
        *atlas_size* is the PoT side length.  *columns* and *rows*
        are the tile grid dimensions.

    Raises
    ------
    ValueError
        If tiles cannot fit within *max_size*.
    """
    slot_size = tile_size + 2 * gutter

    # Try increasing column counts until we find a PoT that fits
    best_size = max_size + 1
    best_cols = 1
    best_rows = n_tiles

    for cols in range(1, n_tiles + 1):
        rows = math.ceil(n_tiles / cols)
        raw_w = cols * slot_size
        raw_h = rows * slot_size
        pot_side = next_power_of_two(max(raw_w, raw_h))
        if pot_side <= max_size and pot_side < best_size:
            best_size = pot_side
            best_cols = cols
            best_rows = rows
        # Once columns exceed what can fit, stop
        if cols * slot_size > max_size:
            break

    if best_size > max_size:
        raise ValueError(
            f"Cannot fit {n_tiles} tiles at size {tile_size}+{gutter}×2 "
            f"within max atlas size {max_size}"
        )

    return best_size, best_cols, best_rows


def resize_atlas_pot(
    atlas: "Image.Image",
    *,
    max_size: int = 8192,
) -> "Image.Image":
    """Resize an atlas image to the nearest power-of-two dimensions.

    Uses Lanczos resampling for high quality.  If the atlas is already
    PoT, returns a copy unchanged.

    Parameters
    ----------
    atlas : PIL.Image
    max_size : int

    Returns
    -------
    PIL.Image
        PoT-dimensioned image.
    """
    w, h = atlas.size
    pot_w = min(next_power_of_two(w), max_size)
    pot_h = min(next_power_of_two(h), max_size)

    if pot_w == w and pot_h == h:
        return atlas.copy()

    # Use square PoT (common for GPU textures)
    pot = max(pot_w, pot_h)
    return atlas.resize((pot, pot), Image.LANCZOS)


# ═══════════════════════════════════════════════════════════════════
# 18D.2 — Mipmap generation
# ═══════════════════════════════════════════════════════════════════

def compute_mip_levels(width: int, height: int) -> int:
    """Compute the number of mipmap levels for a texture.

    ``levels = floor(log2(max(width, height))) + 1``

    Parameters
    ----------
    width, height : int

    Returns
    -------
    int
    """
    return int(math.log2(max(width, height))) + 1


def generate_atlas_mipmaps(
    atlas_path: Path | str,
    *,
    levels: Optional[int] = None,
    output_dir: Optional[Path | str] = None,
    resample: int = 1,  # Image.LANCZOS
) -> List[Path]:
    """Generate a full mipmap chain for an atlas image.

    Parameters
    ----------
    atlas_path : Path
        Path to the base (level 0) atlas image.
    levels : int, optional
        Number of mip levels.  *None* = full chain down to 1×1.
    output_dir : Path, optional
        Where to save mip images.  Defaults to same directory as atlas.
    resample : int
        PIL resampling filter.  Default ``Image.LANCZOS``.

    Returns
    -------
    list of Path
        Paths to each mip level image, starting with level 0 (the
        PoT-resized original).
    """
    atlas_path = Path(atlas_path)
    if output_dir is None:
        output_dir = atlas_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = Image.open(str(atlas_path)).convert("RGB")

    # Ensure PoT
    base = resize_atlas_pot(base)
    w, h = base.size

    if levels is None:
        levels = compute_mip_levels(w, h)
    else:
        levels = min(levels, compute_mip_levels(w, h))

    stem = atlas_path.stem
    mip_paths: List[Path] = []

    for level in range(levels):
        mip_w = max(1, w >> level)
        mip_h = max(1, h >> level)

        if level == 0:
            mip_img = base
        else:
            mip_img = base.resize((mip_w, mip_h), Image.LANCZOS)

        mip_path = output_dir / f"{stem}_mip{level}.png"
        mip_img.save(str(mip_path))
        mip_paths.append(mip_path)

    return mip_paths


# ═══════════════════════════════════════════════════════════════════
# 18D.3 — KTX2 export
# ═══════════════════════════════════════════════════════════════════

# KTX2 header constants
_KTX2_IDENTIFIER = bytes([
    0xAB, 0x4B, 0x54, 0x58,  # «KTX
    0x20, 0x32, 0x30, 0xBB,  #  20»
    0x0D, 0x0A, 0x1A, 0x0A,  # \r\n\x1a\n
])
# VK_FORMAT_R8G8B8_UNORM = 23
_VK_FORMAT_R8G8B8_UNORM = 23
# VK_FORMAT_R8G8B8A8_UNORM = 37
_VK_FORMAT_R8G8B8A8_UNORM = 37


def export_atlas_ktx2(
    atlas_path: Path | str,
    output_path: Path | str,
    *,
    include_mipmaps: bool = True,
    max_levels: Optional[int] = None,
) -> Path:
    """Export an atlas image as a KTX2 container with optional mipmaps.

    This produces a minimal valid KTX2 file with uncompressed
    R8G8B8A8 pixel data.  For GPU-compressed formats (BC7, ETC2, ASTC),
    use an external tool like ``toktx`` from the KTX-Software SDK.

    Parameters
    ----------
    atlas_path : Path
        Input atlas PNG.
    output_path : Path
        Output .ktx2 file.
    include_mipmaps : bool
        Whether to include mipmap levels.
    max_levels : int, optional
        Maximum mip levels to include.

    Returns
    -------
    Path
        The written KTX2 file path.
    """
    atlas_path = Path(atlas_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = Image.open(str(atlas_path)).convert("RGBA")
    base = resize_atlas_pot(base)
    w, h = base.size

    total_levels = compute_mip_levels(w, h)
    if not include_mipmaps:
        total_levels = 1
    elif max_levels is not None:
        total_levels = min(total_levels, max_levels)

    # Build mip images
    mip_images = []
    for level in range(total_levels):
        mip_w = max(1, w >> level)
        mip_h = max(1, h >> level)
        if level == 0:
            mip_images.append(base)
        else:
            mip_images.append(base.resize((mip_w, mip_h), Image.LANCZOS))

    # Compute level data sizes and offsets
    level_data = []
    for img in mip_images:
        raw = img.tobytes()  # RGBA bytes
        level_data.append(raw)

    # Build KTX2 file
    # Header: 80 bytes
    # Level index: total_levels * 24 bytes (offset, length, uncompressed_length)
    # DFD (Data Format Descriptor): minimal block
    # Pixel data: concatenated levels

    header_size = 80
    level_index_size = total_levels * 24

    # Minimal DFD for R8G8B8A8_UNORM
    # DFD total size (uint32) + basic descriptor block
    dfd_block = _build_minimal_dfd_rgba()
    dfd_total_size = 4 + len(dfd_block)  # 4 bytes for dfdTotalSize field

    # SGD (Supercompression Global Data) — empty
    sgd_size = 0

    # Align data start
    data_start = header_size + level_index_size + dfd_total_size + sgd_size
    # Align to 16 bytes for GL compatibility
    data_start = (data_start + 15) & ~15

    # Compute level offsets (from file start, levels stored largest-first)
    level_offsets = []
    offset = data_start
    for ld in level_data:
        level_offsets.append(offset)
        # Align each level to 16 bytes
        aligned_size = (len(ld) + 15) & ~15
        offset += aligned_size

    # Write header
    buf = io.BytesIO()

    # KTX2 identifier (12 bytes)
    buf.write(_KTX2_IDENTIFIER)

    # vkFormat (uint32) — R8G8B8A8_UNORM
    buf.write(struct.pack("<I", _VK_FORMAT_R8G8B8A8_UNORM))

    # typeSize (uint32) — 1 for byte formats
    buf.write(struct.pack("<I", 1))

    # pixelWidth, pixelHeight, pixelDepth (uint32 each)
    buf.write(struct.pack("<III", w, h, 0))

    # layerCount, faceCount (uint32 each)
    buf.write(struct.pack("<II", 0, 1))

    # levelCount (uint32)
    buf.write(struct.pack("<I", total_levels))

    # supercompressionScheme (uint32) — 0 = none
    buf.write(struct.pack("<I", 0))

    # DFD byte offset, byte length (uint32 each)
    dfd_offset = header_size + level_index_size
    buf.write(struct.pack("<II", dfd_offset, dfd_total_size))

    # KVD byte offset, byte length (uint32 each) — 0 = none
    buf.write(struct.pack("<II", 0, 0))

    # SGD byte offset, byte length (uint64 each)
    buf.write(struct.pack("<QQ", 0, 0))

    assert buf.tell() == header_size

    # Level index
    for i in range(total_levels):
        byte_offset = level_offsets[i]
        byte_length = len(level_data[i])
        uncompressed_byte_length = byte_length  # no compression
        buf.write(struct.pack("<QQQ", byte_offset, byte_length, uncompressed_byte_length))

    # DFD
    buf.write(struct.pack("<I", dfd_total_size))  # dfdTotalSize
    buf.write(dfd_block)

    # Pad to data_start
    current = buf.tell()
    if current < data_start:
        buf.write(b"\x00" * (data_start - current))

    # Level data (largest first)
    for i, ld in enumerate(level_data):
        assert buf.tell() == level_offsets[i]
        buf.write(ld)
        # Pad to 16-byte alignment
        remainder = len(ld) % 16
        if remainder:
            buf.write(b"\x00" * (16 - remainder))

    output_path.write_bytes(buf.getvalue())
    return output_path


def _build_minimal_dfd_rgba() -> bytes:
    """Build a minimal KTX2 Data Format Descriptor for R8G8B8A8_UNORM.

    Returns the basic descriptor block (without the 4-byte totalSize prefix).
    """
    # Basic descriptor block header (24 bytes)
    # vendorId=0 (Khronos), descriptorType=0 (BASICFORMAT)
    # versionNumber=2, descriptorBlockSize = 24 + 4*16 = 88
    n_samples = 4  # RGBA
    block_size = 24 + n_samples * 16

    buf = io.BytesIO()
    # vendorId (uint16) + descriptorType (uint16)
    buf.write(struct.pack("<HH", 0, 0))
    # versionNumber (uint16) + descriptorBlockSize (uint16)
    buf.write(struct.pack("<HH", 2, block_size))
    # colorModel (uint8) = 1 (KHR_DF_MODEL_RGBSDA)
    buf.write(struct.pack("<B", 1))
    # colorPrimaries (uint8) = 1 (sRGB)
    buf.write(struct.pack("<B", 1))
    # transferFunction (uint8) = 2 (sRGB)
    buf.write(struct.pack("<B", 2))
    # flags (uint8) = 0
    buf.write(struct.pack("<B", 0))
    # texelBlockDimension[0-3] (4 bytes) — 0,0,0,0 for 1×1×1×1
    buf.write(struct.pack("<BBBB", 0, 0, 0, 0))
    # bytesPlane[0-7] (8 bytes) — 4 bytes per texel in plane 0
    buf.write(struct.pack("<BBBBBBBB", 4, 0, 0, 0, 0, 0, 0, 0))

    # Samples: R, G, B, A (16 bytes each)
    for channel_id, bit_offset in enumerate([0, 8, 16, 24]):
        # bitOffset (uint16)
        buf.write(struct.pack("<H", bit_offset))
        # bitLength (uint8) — 7 means 8 bits (bitLength+1)
        buf.write(struct.pack("<B", 7))
        # channelType (uint8) — channel_id (0=R, 1=G, 2=B, 15=A)
        ch = channel_id if channel_id < 3 else 15
        buf.write(struct.pack("<B", ch))
        # samplePosition[0-3] (4 bytes)
        buf.write(struct.pack("<BBBB", 0, 0, 0, 0))
        # sampleLower (uint32)
        buf.write(struct.pack("<I", 0))
        # sampleUpper (uint32)
        buf.write(struct.pack("<I", 255))

    return buf.getvalue()


def validate_ktx2_header(path: Path | str) -> bool:
    """Check if a file has a valid KTX2 identifier.

    Parameters
    ----------
    path : Path

    Returns
    -------
    bool
    """
    path = Path(path)
    if not path.exists():
        return False
    data = path.read_bytes()
    if len(data) < 80:
        return False
    return data[:12] == _KTX2_IDENTIFIER


# ═══════════════════════════════════════════════════════════════════
# 18D.4 — Channel-packed ORM atlas
# ═══════════════════════════════════════════════════════════════════

def build_orm_atlas(
    collection,
    *,
    biome_type_map: Optional[Dict[str, str]] = None,
    tile_size: int = 256,
    columns: int = 0,
    gutter: int = 4,
    hillshade_scale: float = 1.0,
    water_roughness: float = 0.15,
    rock_roughness: float = 0.55,
    foliage_roughness: float = 0.85,
    default_roughness: float = 0.65,
    metallic: float = 0.0,
    elevation_field: str = "elevation",
) -> Tuple["Image.Image", Dict[str, Tuple[float, float, float, float]]]:
    """Build an ORM (Occlusion-Roughness-Metallic) channel-packed atlas.

    Channels:
    - **R** = Ambient Occlusion (from hillshade — darker in crevices)
    - **G** = Roughness (biome-dependent: low for water, medium for rock, high for foliage)
    - **B** = Metallic (0.0 for all natural terrain)

    Parameters
    ----------
    collection : DetailGridCollection
        Must have stores populated.
    biome_type_map : dict, optional
        ``{face_id: "ocean"|"forest"|"terrain"}``.
    tile_size : int
    columns : int
    gutter : int
    hillshade_scale : float
        Vertical exaggeration for hillshade-derived AO.
    water_roughness : float
    rock_roughness : float
    foliage_roughness : float
    default_roughness : float
    metallic : float
    elevation_field : str

    Returns
    -------
    (orm_image, uv_layout)
    """
    from .detail_render import _detail_hillshade, BiomeConfig
    from .geometry import face_center

    if biome_type_map is None:
        biome_type_map = {}

    face_ids = collection.face_ids
    n = len(face_ids)
    if n == 0:
        raise ValueError("No detail grids in the collection")

    if columns <= 0:
        columns = max(1, math.isqrt(n))
        if columns * columns < n:
            columns += 1
    rows = math.ceil(n / columns)

    slot_size = tile_size + 2 * gutter
    atlas_w = columns * slot_size
    atlas_h = rows * slot_size

    # Default: mid-grey ORM (AO=1.0, roughness=0.5, metallic=0.0)
    atlas = Image.new("RGB", (atlas_w, atlas_h), (255, 128, 0))
    uv_layout: Dict[str, Tuple[float, float, float, float]] = {}

    for idx, fid in enumerate(face_ids):
        col = idx % columns
        row = idx // columns
        slot_x = col * slot_size
        slot_y = row * slot_size

        grid, store = collection.get(fid)
        if store is None:
            continue

        # Compute hillshade for AO
        hs = _detail_hillshade(
            grid, store, elevation_field,
            azimuth=315.0, altitude=45.0,
        )

        # Determine roughness for this tile
        biome = biome_type_map.get(fid, "terrain")
        if biome == "ocean":
            roughness_val = water_roughness
        elif biome == "forest":
            roughness_val = foliage_roughness
        else:
            roughness_val = default_roughness

        # Render ORM tile
        tile_img = _render_orm_tile(
            grid, hs, roughness_val, metallic,
            tile_size=tile_size,
        )

        atlas.paste(tile_img, (slot_x + gutter, slot_y + gutter))

        # Fill gutter
        if gutter > 0:
            _fill_orm_gutter(atlas, slot_x, slot_y, tile_size, gutter)

        inner_x = slot_x + gutter
        inner_y = slot_y + gutter
        u_min = inner_x / atlas_w
        u_max = (inner_x + tile_size) / atlas_w
        v_min = 1.0 - (inner_y + tile_size) / atlas_h
        v_max = 1.0 - inner_y / atlas_h
        uv_layout[fid] = (u_min, v_min, u_max, v_max)

    return atlas, uv_layout


def _render_orm_tile(
    grid,
    hillshade: Dict[str, float],
    roughness: float,
    metallic: float,
    *,
    tile_size: int = 256,
) -> "Image.Image":
    """Render ORM channels for a single tile."""
    from PIL import ImageDraw
    from .geometry import face_center

    img = Image.new("RGB", (tile_size, tile_size), (255, int(roughness * 255), int(metallic * 255)))
    draw = ImageDraw.Draw(img)

    # Bounding box
    xs, ys = [], []
    for v in grid.vertices.values():
        if v.has_position():
            xs.append(v.x)
            ys.append(v.y)
    if not xs:
        return img

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span = max((x_max - x_min) or 1.0, (y_max - y_min) or 1.0)
    pad = span * 0.15
    x_min -= pad
    x_max += pad
    y_min -= pad
    y_max += pad
    x_range = x_max - x_min
    y_range = y_max - y_min
    scale = tile_size / max(x_range, y_range)
    ox = (tile_size - x_range * scale) / 2.0
    oy = (tile_size - y_range * scale) / 2.0

    def to_pixel(vx, vy):
        px = (vx - x_min) * scale + ox
        py = tile_size - ((vy - y_min) * scale + oy)
        return (px, py)

    for fid, face in grid.faces.items():
        verts = []
        for vid in face.vertex_ids:
            v = grid.vertices.get(vid)
            if v is None or not v.has_position():
                break
            verts.append(to_pixel(v.x, v.y))
        else:
            if len(verts) >= 3:
                hs = hillshade.get(fid, 0.5)
                # AO: invert hillshade so crevices (low hs) are darker
                # Map hs [0,1] to AO where 0.5 = neutral
                ao = int(max(0, min(255, hs * 255)))
                r_val = int(max(0, min(255, roughness * 255)))
                m_val = int(max(0, min(255, metallic * 255)))
                draw.polygon(verts, fill=(ao, r_val, m_val))

    return img


def _fill_orm_gutter(atlas, slot_x, slot_y, tile_size, gutter):
    """Fill ORM atlas gutter by clamping edge pixels."""
    inner_x = slot_x + gutter
    inner_y = slot_y + gutter

    top_strip = atlas.crop((inner_x, inner_y, inner_x + tile_size, inner_y + 1))
    for g in range(gutter):
        atlas.paste(top_strip, (inner_x, slot_y + g))

    bot_y = inner_y + tile_size - 1
    bot_strip = atlas.crop((inner_x, bot_y, inner_x + tile_size, bot_y + 1))
    for g in range(gutter):
        atlas.paste(bot_strip, (inner_x, inner_y + tile_size + g))

    full_top = slot_y
    full_bot = slot_y + tile_size + 2 * gutter
    left_strip = atlas.crop((inner_x, full_top, inner_x + 1, full_bot))
    for g in range(gutter):
        atlas.paste(left_strip, (slot_x + g, full_top))

    right_x = inner_x + tile_size - 1
    right_strip = atlas.crop((right_x, full_top, right_x + 1, full_bot))
    for g in range(gutter):
        atlas.paste(right_strip, (inner_x + tile_size + g, full_top))


# ═══════════════════════════════════════════════════════════════════
# 18D.4b — Material set builder
# ═══════════════════════════════════════════════════════════════════

def build_material_set(
    collection,
    *,
    albedo_atlas: "Image.Image",
    normal_maps: Optional[Dict[str, Dict[str, Tuple[float, float, float]]]] = None,
    biome_type_map: Optional[Dict[str, str]] = None,
    output_dir: Path | str = Path("exports/materials"),
    tile_size: int = 256,
    columns: int = 0,
    gutter: int = 4,
    pot: bool = True,
) -> Dict[str, Path]:
    """Generate a complete PBR material set: albedo + normal + ORM.

    Parameters
    ----------
    collection : DetailGridCollection
    albedo_atlas : PIL.Image
        The colour atlas (from any atlas builder).
    normal_maps : dict, optional
        ``{face_id: {sub_face_id: (nx, ny, nz)}}``.
        If *None*, a flat normal map is generated.
    biome_type_map : dict, optional
    output_dir : Path
    tile_size : int
    columns : int
    gutter : int
    pot : bool
        Resize all textures to power-of-two.

    Returns
    -------
    dict
        ``{"albedo": Path, "normal": Path, "orm": Path}``
    """
    from .globe_renderer_v2 import build_normal_map_atlas
    from .render_enhanced import compute_all_normal_maps

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Albedo
    albedo = albedo_atlas.convert("RGB")
    if pot:
        albedo = resize_atlas_pot(albedo)
    albedo_path = output_dir / "albedo.png"
    albedo.save(str(albedo_path))

    # Normal map
    if normal_maps is None:
        normal_maps = compute_all_normal_maps(collection)
    normal_img, _ = build_normal_map_atlas(
        normal_maps, collection,
        tile_size=tile_size, columns=columns, gutter=gutter,
    )
    if pot:
        normal_img = resize_atlas_pot(normal_img)
    normal_path = output_dir / "normal.png"
    normal_img.save(str(normal_path))

    # ORM
    orm_img, _ = build_orm_atlas(
        collection,
        biome_type_map=biome_type_map,
        tile_size=tile_size, columns=columns, gutter=gutter,
    )
    if pot:
        orm_img = resize_atlas_pot(orm_img)
    orm_path = output_dir / "orm.png"
    orm_img.save(str(orm_path))

    return {"albedo": albedo_path, "normal": normal_path, "orm": orm_path}


# ═══════════════════════════════════════════════════════════════════
# 18D.5 — glTF 2.0 export
# ═══════════════════════════════════════════════════════════════════

def _encode_buffer_as_data_uri(data: bytes, mime: str = "application/octet-stream") -> str:
    """Encode binary data as a base64 data URI for embedded glTF."""
    return f"data:{mime};base64,{b64encode(data).decode('ascii')}"


def _encode_image_as_data_uri(img_path: Path) -> str:
    """Encode a PNG image as a base64 data URI."""
    data = img_path.read_bytes()
    return f"data:image/png;base64,{b64encode(data).decode('ascii')}"


def export_globe_gltf(
    frequency: int,
    uv_layout: Dict[str, Tuple[float, float, float, float]],
    *,
    albedo_path: Optional[Path | str] = None,
    normal_path: Optional[Path | str] = None,
    orm_path: Optional[Path | str] = None,
    output_path: Path | str = Path("exports/globe.gltf"),
    radius: float = 1.0,
    embed_textures: bool = True,
) -> Path:
    """Export the textured globe as a glTF 2.0 asset.

    Produces a single .gltf JSON file with embedded binary buffers
    and textures (when *embed_textures=True*).

    Parameters
    ----------
    frequency : int
        Goldberg subdivision frequency.
    uv_layout : dict
        ``{face_id: (u_min, v_min, u_max, v_max)}``.
    albedo_path : Path, optional
    normal_path : Path, optional
    orm_path : Path, optional
    output_path : Path
    radius : float
    embed_textures : bool

    Returns
    -------
    Path
    """
    try:
        from models.objects.goldberg import generate_goldberg_tiles
    except ImportError:
        raise ImportError("models library required for glTF export")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tiles = generate_goldberg_tiles(frequency=frequency, radius=radius)

    # Build combined vertex buffer + index buffer
    all_positions = []  # (x, y, z)
    all_normals = []    # (nx, ny, nz)
    all_uvs = []        # (u, v)
    all_indices = []
    vertex_offset = 0

    for tile in tiles:
        fid = f"t{tile.index}"
        if fid not in uv_layout:
            continue

        u_min, v_min, u_max, v_max = uv_layout[fid]
        u_span = u_max - u_min
        v_span = v_max - v_min

        center = tile.center
        verts = tile.vertices
        uv_verts = list(tile.uv_vertices)

        # Centre UV
        center_u = sum(uv[0] for uv in uv_verts) / len(uv_verts)
        center_v = sum(uv[1] for uv in uv_verts) / len(uv_verts)

        # Positions: center + ring vertices
        positions = [center] + list(verts)

        # Normals: for a sphere, normal = normalize(position)
        norms = []
        for pos in positions:
            length = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
            if length > 1e-10:
                norms.append((pos[0]/length, pos[1]/length, pos[2]/length))
            else:
                norms.append((0.0, 0.0, 1.0))

        # UVs: center + ring mapped into atlas
        uvs = [(u_min + center_u * u_span, v_min + center_v * v_span)]
        for uv in uv_verts:
            u_clamped = max(0.0, min(1.0, uv[0]))
            v_clamped = max(0.0, min(1.0, uv[1]))
            uvs.append((u_min + u_clamped * u_span, v_min + v_clamped * v_span))

        all_positions.extend(positions)
        all_normals.extend(norms)
        all_uvs.extend(uvs)

        # Triangle fan indices
        n = len(positions)
        for i in range(1, n):
            nxt = 1 if i == n - 1 else i + 1
            all_indices.extend([
                vertex_offset + 0,
                vertex_offset + i,
                vertex_offset + nxt,
            ])
        vertex_offset += n

    if not all_positions:
        raise ValueError("No tiles matched the UV layout")

    # Pack binary data
    pos_data = struct.pack(f"<{len(all_positions) * 3}f",
        *[c for p in all_positions for c in p])
    norm_data = struct.pack(f"<{len(all_normals) * 3}f",
        *[c for n in all_normals for c in n])
    uv_data = struct.pack(f"<{len(all_uvs) * 2}f",
        *[c for uv in all_uvs for c in uv])
    idx_data = struct.pack(f"<{len(all_indices)}I", *all_indices)

    # Compute bounding box
    min_pos = [min(p[i] for p in all_positions) for i in range(3)]
    max_pos = [max(p[i] for p in all_positions) for i in range(3)]

    # Combine into single buffer
    # Align each accessor to 4 bytes (already float/uint32, so OK)
    buffer_data = pos_data + norm_data + uv_data + idx_data

    # Build glTF JSON
    gltf: Dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": "pgrid texture_export Phase 18D",
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
    }

    # Buffer
    if embed_textures:
        buffer_uri = _encode_buffer_as_data_uri(buffer_data)
    else:
        bin_path = output_path.with_suffix(".bin")
        bin_path.write_bytes(buffer_data)
        buffer_uri = bin_path.name

    gltf["buffers"] = [{
        "uri": buffer_uri,
        "byteLength": len(buffer_data),
    }]

    # Buffer views
    pos_offset = 0
    norm_offset = len(pos_data)
    uv_offset = norm_offset + len(norm_data)
    idx_offset = uv_offset + len(uv_data)

    gltf["bufferViews"] = [
        {"buffer": 0, "byteOffset": pos_offset, "byteLength": len(pos_data), "target": 34962},
        {"buffer": 0, "byteOffset": norm_offset, "byteLength": len(norm_data), "target": 34962},
        {"buffer": 0, "byteOffset": uv_offset, "byteLength": len(uv_data), "target": 34962},
        {"buffer": 0, "byteOffset": idx_offset, "byteLength": len(idx_data), "target": 34963},
    ]

    # Accessors
    n_verts = len(all_positions)
    n_indices = len(all_indices)

    gltf["accessors"] = [
        {
            "bufferView": 0, "componentType": 5126, "count": n_verts,
            "type": "VEC3", "min": min_pos, "max": max_pos,
        },
        {
            "bufferView": 1, "componentType": 5126, "count": n_verts,
            "type": "VEC3",
        },
        {
            "bufferView": 2, "componentType": 5126, "count": n_verts,
            "type": "VEC2",
        },
        {
            "bufferView": 3, "componentType": 5125, "count": n_indices,
            "type": "SCALAR",
        },
    ]

    # Mesh
    gltf["meshes"] = [{
        "primitives": [{
            "attributes": {
                "POSITION": 0,
                "NORMAL": 1,
                "TEXCOORD_0": 2,
            },
            "indices": 3,
            "material": 0,
        }],
    }]

    # Material
    material: Dict[str, Any] = {
        "pbrMetallicRoughness": {},
    }

    # Textures + images + samplers
    textures = []
    images = []
    samplers: List[Dict[str, int]] = []

    if not samplers:
        # Default sampler: linear filtering, repeat wrap
        samplers.append({
            "magFilter": 9729,  # LINEAR
            "minFilter": 9987,  # LINEAR_MIPMAP_LINEAR
            "wrapS": 10497,     # REPEAT
            "wrapT": 10497,
        })

    tex_idx = 0

    if albedo_path is not None:
        albedo_path = Path(albedo_path)
        img_entry: Dict[str, Any] = {"mimeType": "image/png"}
        if embed_textures:
            img_entry["uri"] = _encode_image_as_data_uri(albedo_path)
        else:
            img_entry["uri"] = albedo_path.name
        images.append(img_entry)
        textures.append({"source": tex_idx, "sampler": 0})
        material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": tex_idx}
        tex_idx += 1

    if normal_path is not None:
        normal_path = Path(normal_path)
        img_entry = {"mimeType": "image/png"}
        if embed_textures:
            img_entry["uri"] = _encode_image_as_data_uri(normal_path)
        else:
            img_entry["uri"] = normal_path.name
        images.append(img_entry)
        textures.append({"source": tex_idx, "sampler": 0})
        material["normalTexture"] = {"index": tex_idx}
        tex_idx += 1

    if orm_path is not None:
        orm_path = Path(orm_path)
        img_entry = {"mimeType": "image/png"}
        if embed_textures:
            img_entry["uri"] = _encode_image_as_data_uri(orm_path)
        else:
            img_entry["uri"] = orm_path.name
        images.append(img_entry)
        textures.append({"source": tex_idx, "sampler": 0})
        material["occlusionTexture"] = {"index": tex_idx}
        material["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": tex_idx}
        tex_idx += 1

    if textures:
        gltf["textures"] = textures
        gltf["images"] = images
        gltf["samplers"] = samplers

    gltf["materials"] = [material]

    # Write
    with open(str(output_path), "w") as f:
        json.dump(gltf, f, indent=2)

    return output_path
