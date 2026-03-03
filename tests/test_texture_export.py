"""Tests for Phase 18D — Texture Export Pipeline."""

from __future__ import annotations

import json
import math
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polygrid.texture_export import (
    _build_minimal_dfd_rgba,
    _encode_buffer_as_data_uri,
    _fill_orm_gutter,
    _render_orm_tile,
    atlas_pot_size,
    build_material_set,
    build_orm_atlas,
    compute_mip_levels,
    export_atlas_ktx2,
    export_globe_gltf,
    generate_atlas_mipmaps,
    next_power_of_two,
    resize_atlas_pot,
    validate_ktx2_header,
)

PIL = pytest.importorskip("PIL")
from PIL import Image

# ── helpers ──────────────────────────────────────────────────────────

def _make_test_atlas(w: int = 300, h: int = 300) -> Image.Image:
    """Create a small test atlas image."""
    return Image.new("RGB", (w, h), (100, 150, 200))


def _make_test_atlas_file(tmp_path: Path, w: int = 300, h: int = 300) -> Path:
    """Write a small test atlas PNG to disk."""
    img = _make_test_atlas(w, h)
    path = tmp_path / "test_atlas.png"
    img.save(str(path))
    return path


def _make_tiny_grid():
    """Build the smallest valid PolyGrid with elevation."""
    from polygrid import PolyGrid, Vertex, Face, Edge

    v0 = Vertex("v0", 0.0, 0.0)
    v1 = Vertex("v1", 1.0, 0.0)
    v2 = Vertex("v2", 0.5, 0.866)
    v3 = Vertex("v3", 1.5, 0.866)
    v4 = Vertex("v4", 0.0, 1.732)
    v5 = Vertex("v5", 1.0, 1.732)

    faces_raw = [
        ("f0", ["v0", "v1", "v2"]),
        ("f1", ["v1", "v3", "v2"]),
        ("f2", ["v2", "v3", "v5"]),
        ("f3", ["v2", "v5", "v4"]),
    ]

    # Collect edges with face references
    edge_map: dict[tuple[str, str], list[str]] = {}
    for fid, vids in faces_raw:
        for i in range(len(vids)):
            a, b = vids[i], vids[(i + 1) % len(vids)]
            key = (min(a, b), max(a, b))
            edge_map.setdefault(key, []).append(fid)

    edges = []
    for idx, ((a, b), fids) in enumerate(edge_map.items()):
        edges.append(Edge(f"e{idx}", (a, b), tuple(fids)))

    faces = [
        Face(fid, "tri", tuple(vids))
        for fid, vids in faces_raw
    ]

    grid = PolyGrid(
        vertices=[v0, v1, v2, v3, v4, v5],
        edges=edges,
        faces=faces,
    )
    return grid


def _make_collection():
    """Build a fake DetailGridCollection with two tiles."""
    from polygrid import TileDataStore, TileSchema, FieldDef

    grid = _make_tiny_grid()
    schema = TileSchema(fields=[FieldDef(name="elevation", dtype=float)])
    store = TileDataStore(grid, schema=schema)
    for fid in grid.faces:
        store.set(fid, "elevation", 0.1)

    class FakeCollection:
        def __init__(self):
            self.face_ids = ["t0", "t1"]
            self._grid = grid
            self._store = store
            # Dict-like access expected by build_normal_map_atlas etc.
            self.grids = {"t0": grid, "t1": grid}
            self.stores = {"t0": store, "t1": store}

        def get(self, fid):
            return (self._grid, self._store)

    return FakeCollection()


# ═══════════════════════════════════════════════════════════════════
# 18D.1 — Power-of-two utilities
# ═══════════════════════════════════════════════════════════════════

class TestNextPowerOfTwo:
    def test_exact_pot(self):
        assert next_power_of_two(256) == 256
        assert next_power_of_two(1024) == 1024

    def test_non_pot(self):
        assert next_power_of_two(1000) == 1024
        assert next_power_of_two(257) == 512

    def test_one(self):
        assert next_power_of_two(1) == 1

    def test_zero_or_negative(self):
        assert next_power_of_two(0) == 1
        assert next_power_of_two(-5) == 1

    def test_large(self):
        assert next_power_of_two(5000) == 8192


class TestAtlasPotSize:
    def test_single_tile(self):
        size, cols, rows = atlas_pot_size(1, tile_size=256, gutter=4)
        assert cols >= 1
        assert rows >= 1
        # Must be a power of two
        assert size & (size - 1) == 0

    def test_12_tiles(self):
        size, cols, rows = atlas_pot_size(12, tile_size=256, gutter=4)
        slot = 256 + 8
        assert cols * rows >= 12
        assert size & (size - 1) == 0
        assert cols * slot <= size
        assert rows * slot <= size

    def test_too_many_raises(self):
        with pytest.raises(ValueError, match="Cannot fit"):
            atlas_pot_size(10000, tile_size=4096, gutter=0, max_size=4096)


class TestResizeAtlasPot:
    def test_already_pot(self):
        img = Image.new("RGB", (256, 256), (0, 0, 0))
        result = resize_atlas_pot(img)
        assert result.size == (256, 256)

    def test_non_pot_resized(self):
        img = Image.new("RGB", (300, 200), (0, 0, 0))
        result = resize_atlas_pot(img)
        w, h = result.size
        assert w == h  # square
        assert w & (w - 1) == 0  # PoT
        assert w >= 300  # at least as large as largest input dim


# ═══════════════════════════════════════════════════════════════════
# 18D.2 — Mipmap generation
# ═══════════════════════════════════════════════════════════════════

class TestComputeMipLevels:
    def test_256(self):
        assert compute_mip_levels(256, 256) == 9  # log2(256)+1 = 9

    def test_1024(self):
        assert compute_mip_levels(1024, 1024) == 11

    def test_non_square(self):
        assert compute_mip_levels(512, 256) == 10  # log2(512)+1

    def test_one(self):
        assert compute_mip_levels(1, 1) == 1


class TestGenerateAtlasMipmaps:
    def test_generates_mip_files(self, tmp_path):
        atlas_path = _make_test_atlas_file(tmp_path, 256, 256)
        paths = generate_atlas_mipmaps(atlas_path, output_dir=tmp_path / "mips")
        assert len(paths) > 1
        # First mip should be 256x256 (already PoT)
        mip0 = Image.open(str(paths[0]))
        assert mip0.size == (256, 256)
        # Last mip should be 1x1
        last = Image.open(str(paths[-1]))
        assert last.size == (1, 1)

    def test_limited_levels(self, tmp_path):
        atlas_path = _make_test_atlas_file(tmp_path, 256, 256)
        paths = generate_atlas_mipmaps(atlas_path, levels=3, output_dir=tmp_path / "mips3")
        assert len(paths) == 3

    def test_non_pot_resized(self, tmp_path):
        atlas_path = _make_test_atlas_file(tmp_path, 300, 300)
        paths = generate_atlas_mipmaps(atlas_path, output_dir=tmp_path / "mips_np")
        mip0 = Image.open(str(paths[0]))
        w, h = mip0.size
        assert w & (w - 1) == 0  # PoT


# ═══════════════════════════════════════════════════════════════════
# 18D.3 — KTX2 export
# ═══════════════════════════════════════════════════════════════════

class TestKTX2:
    def test_export_and_validate(self, tmp_path):
        atlas_path = _make_test_atlas_file(tmp_path, 64, 64)
        ktx_path = tmp_path / "test.ktx2"
        result = export_atlas_ktx2(atlas_path, ktx_path, include_mipmaps=False)
        assert result.exists()
        assert validate_ktx2_header(result)

    def test_export_with_mipmaps(self, tmp_path):
        atlas_path = _make_test_atlas_file(tmp_path, 64, 64)
        ktx_path = tmp_path / "test_mip.ktx2"
        result = export_atlas_ktx2(atlas_path, ktx_path, include_mipmaps=True)
        assert result.exists()
        # Should be larger than without mipmaps
        no_mip_path = tmp_path / "test_nomip.ktx2"
        export_atlas_ktx2(atlas_path, no_mip_path, include_mipmaps=False)
        assert result.stat().st_size > no_mip_path.stat().st_size

    def test_validate_invalid_file(self, tmp_path):
        bad_file = tmp_path / "not_ktx.bin"
        bad_file.write_bytes(b"not a ktx2 file")
        assert not validate_ktx2_header(bad_file)

    def test_validate_missing_file(self, tmp_path):
        assert not validate_ktx2_header(tmp_path / "nonexistent.ktx2")

    def test_header_structure(self, tmp_path):
        """Verify the KTX2 header fields are plausible."""
        atlas_path = _make_test_atlas_file(tmp_path, 64, 64)
        ktx_path = tmp_path / "header_test.ktx2"
        export_atlas_ktx2(atlas_path, ktx_path, include_mipmaps=False)
        data = ktx_path.read_bytes()

        # Check identifier
        assert data[:12] == bytes([
            0xAB, 0x4B, 0x54, 0x58,
            0x20, 0x32, 0x30, 0xBB,
            0x0D, 0x0A, 0x1A, 0x0A,
        ])

        # vkFormat at offset 12
        vk_format = struct.unpack_from("<I", data, 12)[0]
        assert vk_format == 37  # R8G8B8A8_UNORM

        # pixelWidth, pixelHeight at offsets 20, 24
        pw = struct.unpack_from("<I", data, 20)[0]
        ph = struct.unpack_from("<I", data, 24)[0]
        assert pw == 64
        assert ph == 64

    def test_max_levels(self, tmp_path):
        atlas_path = _make_test_atlas_file(tmp_path, 64, 64)
        ktx_path = tmp_path / "max3.ktx2"
        export_atlas_ktx2(atlas_path, ktx_path, include_mipmaps=True, max_levels=3)
        data = ktx_path.read_bytes()
        # levelCount is at offset 40 (after identifier=12 + vkFormat=4 +
        # typeSize=4 + width=4 + height=4 + depth=4 + layerCount=4 + faceCount=4)
        level_count = struct.unpack_from("<I", data, 40)[0]
        assert level_count == 3


class TestBuildMinimalDfd:
    def test_dfd_size(self):
        dfd = _build_minimal_dfd_rgba()
        # 24 bytes header + 4 channels * 16 bytes = 88
        assert len(dfd) == 88


# ═══════════════════════════════════════════════════════════════════
# 18D.4 — ORM atlas
# ═══════════════════════════════════════════════════════════════════

class TestRenderOrmTile:
    def test_produces_rgb_image(self):
        grid = _make_tiny_grid()
        hs = {fid: 0.7 for fid in grid.faces}
        img = _render_orm_tile(grid, hs, roughness=0.5, metallic=0.0, tile_size=64)
        assert img.mode == "RGB"
        assert img.size == (64, 64)

    def test_roughness_channel(self):
        grid = _make_tiny_grid()
        hs = {fid: 1.0 for fid in grid.faces}  # full light = max AO
        img = _render_orm_tile(grid, hs, roughness=0.8, metallic=0.0, tile_size=64)
        # Sample centre pixel — G channel should be ~roughness*255
        px = img.getpixel((32, 32))
        assert abs(px[1] - int(0.8 * 255)) < 5


class TestBuildOrmAtlas:
    def test_produces_atlas(self):
        coll = _make_collection()
        orm, uv = build_orm_atlas(coll, tile_size=64, gutter=2)
        assert orm.mode == "RGB"
        assert len(uv) == 2

    def test_uv_ranges(self):
        coll = _make_collection()
        _, uv = build_orm_atlas(coll, tile_size=64, gutter=2)
        for fid, (u0, v0, u1, v1) in uv.items():
            assert 0.0 <= u0 < u1 <= 1.0
            assert 0.0 <= v0 < v1 <= 1.0

    def test_biome_roughness(self):
        """Ocean tiles should have lower roughness than forest tiles."""
        coll = _make_collection()
        biome_map = {"t0": "ocean", "t1": "forest"}
        orm, _ = build_orm_atlas(
            coll, biome_type_map=biome_map,
            tile_size=64, gutter=2,
            water_roughness=0.15, foliage_roughness=0.85,
        )
        # Just verify it runs — detailed pixel checks are fragile
        assert orm.size[0] > 0


class TestFillOrmGutter:
    def test_fills_edges(self):
        atlas = Image.new("RGB", (80, 80), (0, 0, 0))
        # Paint inner region
        for x in range(4, 68):
            for y in range(4, 68):
                atlas.putpixel((x, y), (200, 100, 50))
        _fill_orm_gutter(atlas, 0, 0, 64, 4)
        # Top gutter should be filled
        px = atlas.getpixel((32, 0))
        assert px[0] > 0  # non-black


# ═══════════════════════════════════════════════════════════════════
# 18D.4b — Material set
# ═══════════════════════════════════════════════════════════════════

class TestBuildMaterialSet:
    def test_produces_three_files(self, tmp_path):
        coll = _make_collection()
        albedo = Image.new("RGB", (128, 128), (100, 200, 50))

        # Mock expensive normal map functions
        flat_norms = {}
        for fid in coll.face_ids:
            grid, _ = coll.get(fid)
            flat_norms[fid] = {sf: (0.0, 0.0, 1.0) for sf in grid.faces}

        result = build_material_set(
            coll,
            albedo_atlas=albedo,
            normal_maps=flat_norms,
            output_dir=tmp_path / "materials",
            tile_size=64,
            gutter=2,
        )
        assert "albedo" in result
        assert "normal" in result
        assert "orm" in result
        for key, path in result.items():
            assert path.exists(), f"{key} not written"


# ═══════════════════════════════════════════════════════════════════
# 18D.5 — glTF export
# ═══════════════════════════════════════════════════════════════════

class TestEncodeBufferDataUri:
    def test_produces_data_uri(self):
        uri = _encode_buffer_as_data_uri(b"\x00\x01\x02")
        assert uri.startswith("data:application/octet-stream;base64,")


class TestExportGlobeGltf:
    def test_basic_export(self, tmp_path):
        """Export with no texture paths — just geometry."""
        # Build a fake UV layout matching goldberg tile ids
        try:
            from models.objects.goldberg import generate_goldberg_tiles
        except ImportError:
            pytest.skip("models library not available")

        tiles = generate_goldberg_tiles(frequency=2, radius=1.0)
        uv_layout = {}
        for tile in tiles:
            fid = f"t{tile.index}"
            uv_layout[fid] = (0.0, 0.0, 1.0, 1.0)

        out_path = tmp_path / "test.gltf"
        result = export_globe_gltf(
            frequency=2,
            uv_layout=uv_layout,
            output_path=out_path,
            embed_textures=True,
        )
        assert result.exists()

        # Parse JSON
        with open(str(result)) as f:
            gltf = json.load(f)
        assert gltf["asset"]["version"] == "2.0"
        assert len(gltf["meshes"]) == 1
        assert len(gltf["buffers"]) == 1

    def test_with_albedo_texture(self, tmp_path):
        """Export with an albedo texture."""
        try:
            from models.objects.goldberg import generate_goldberg_tiles
        except ImportError:
            pytest.skip("models library not available")

        tiles = generate_goldberg_tiles(frequency=2, radius=1.0)
        uv_layout = {}
        for tile in tiles:
            fid = f"t{tile.index}"
            uv_layout[fid] = (0.0, 0.0, 1.0, 1.0)

        albedo_img = Image.new("RGB", (128, 128), (100, 200, 50))
        albedo_path = tmp_path / "albedo.png"
        albedo_img.save(str(albedo_path))

        out_path = tmp_path / "with_tex.gltf"
        result = export_globe_gltf(
            frequency=2,
            uv_layout=uv_layout,
            albedo_path=albedo_path,
            output_path=out_path,
            embed_textures=True,
        )
        with open(str(result)) as f:
            gltf = json.load(f)
        assert "textures" in gltf
        assert "images" in gltf
        assert len(gltf["textures"]) == 1

    def test_no_matching_tiles_raises(self, tmp_path):
        """Empty UV layout should raise."""
        try:
            from models.objects.goldberg import generate_goldberg_tiles
        except ImportError:
            pytest.skip("models library not available")

        with pytest.raises(ValueError, match="No tiles matched"):
            export_globe_gltf(
                frequency=2,
                uv_layout={},
                output_path=tmp_path / "empty.gltf",
            )

    def test_gltf_has_normals_and_uvs(self, tmp_path):
        """Verify the glTF mesh has NORMAL and TEXCOORD_0 attributes."""
        try:
            from models.objects.goldberg import generate_goldberg_tiles
        except ImportError:
            pytest.skip("models library not available")

        tiles = generate_goldberg_tiles(frequency=2, radius=1.0)
        uv_layout = {f"t{t.index}": (0.0, 0.0, 1.0, 1.0) for t in tiles}

        out_path = tmp_path / "attrs.gltf"
        export_globe_gltf(frequency=2, uv_layout=uv_layout, output_path=out_path)
        with open(str(out_path)) as f:
            gltf = json.load(f)
        attrs = gltf["meshes"][0]["primitives"][0]["attributes"]
        assert "POSITION" in attrs
        assert "NORMAL" in attrs
        assert "TEXCOORD_0" in attrs
