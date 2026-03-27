"""Globe data export — comprehensive JSON payload for 3D renderers.

Exports a complete description of a globe grid with terrain data,
suitable for consumption by the ``models`` library renderer, web-based
3D viewers, or any downstream tool.

Functions
---------
- :func:`export_globe_payload` — build the full export dict
- :func:`export_globe_json` — write payload to a JSON file
- :func:`validate_globe_payload` — validate against the schema
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .globe import GlobeGrid
from ..data.tile_data import TileDataStore


# ═══════════════════════════════════════════════════════════════════
# Colour ramps (inlined from former globe_render.py)
# ═══════════════════════════════════════════════════════════════════

def _lerp_colour(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    t: float,
) -> Tuple[float, float, float]:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def _ramp_satellite(elevation: float) -> Tuple[float, float, float]:
    """Earth-like colour ramp: deep blue → green → brown → snow."""
    if elevation < 0.15:
        return _lerp_colour((0.05, 0.10, 0.40), (0.10, 0.25, 0.55), elevation / 0.15)
    elif elevation < 0.35:
        t = (elevation - 0.15) / 0.20
        return _lerp_colour((0.10, 0.40, 0.15), (0.30, 0.55, 0.20), t)
    elif elevation < 0.60:
        t = (elevation - 0.35) / 0.25
        return _lerp_colour((0.45, 0.40, 0.25), (0.55, 0.45, 0.30), t)
    elif elevation < 0.85:
        t = (elevation - 0.60) / 0.25
        return _lerp_colour((0.55, 0.45, 0.30), (0.70, 0.65, 0.60), t)
    else:
        t = (elevation - 0.85) / 0.15
        return _lerp_colour((0.80, 0.80, 0.80), (1.0, 1.0, 1.0), min(t, 1.0))


def _ramp_topo(elevation: float) -> Tuple[float, float, float]:
    """Topographic colour ramp: blue → cyan → green → yellow → red → white."""
    stops = [
        (0.00, (0.0, 0.0, 0.5)),
        (0.20, (0.0, 0.4, 0.8)),
        (0.35, (0.1, 0.6, 0.3)),
        (0.50, (0.5, 0.7, 0.2)),
        (0.65, (0.8, 0.7, 0.1)),
        (0.80, (0.8, 0.3, 0.1)),
        (1.00, (1.0, 1.0, 1.0)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if elevation <= t1:
            t = (elevation - t0) / (t1 - t0) if t1 > t0 else 0.0
            return _lerp_colour(c0, c1, t)
    return stops[-1][1]


_RAMPS: Dict[str, Callable[[float], Tuple[float, float, float]]] = {
    "satellite": _ramp_satellite,
    "topo": _ramp_topo,
}


def make_solid_ramp(
    base_color: Tuple[float, float, float],
    *,
    noise_amount: float = 0.08,
    seed: int = 0,
) -> Callable[[float], Tuple[float, float, float]]:
    """Create a ramp that returns *base_color* with slight brightness noise.

    The ``elevation`` parameter is used as a hash input (combined with
    *seed*) to produce deterministic per-tile jitter of ±\ *noise_amount*.
    """

    def _ramp(elevation: float) -> Tuple[float, float, float]:
        # Deterministic jitter derived from elevation value + seed.
        h = hashlib.md5(f"{seed}:{elevation:.8f}".encode()).digest()
        jitter = (h[0] / 255.0 - 0.5) * 2.0 * noise_amount
        r = max(0.0, min(1.0, base_color[0] + jitter))
        g = max(0.0, min(1.0, base_color[1] + jitter))
        b = max(0.0, min(1.0, base_color[2] + jitter))
        return (r, g, b)

    return _ramp


def globe_to_colour_map(
    globe_grid: GlobeGrid,
    store: TileDataStore,
    *,
    field_name: str = "elevation",
    ramp: str = "satellite",
    ramp_fn: Optional[Callable[[float], Tuple[float, float, float]]] = None,
) -> Dict[str, Tuple[float, float, float]]:
    """Map per-tile data to RGB colours.

    Returns ``{face_id: (r, g, b)}`` where each component is in ``[0, 1]``.
    If *ramp_fn* is provided it takes precedence over *ramp*.
    """
    if ramp_fn is None:
        ramp_fn = _RAMPS.get(ramp, _ramp_satellite)
    colours: Dict[str, Tuple[float, float, float]] = {}
    for fid in globe_grid.faces:
        val = store.get(fid, field_name)
        val = max(0.0, min(1.0, float(val)))
        colours[fid] = ramp_fn(val)
    return colours

_EXPORT_VERSION = "1.0"


def export_globe_payload(
    globe_grid: GlobeGrid,
    store: TileDataStore,
    *,
    field_name: str = "elevation",
    ramp: str = "satellite",
    ramp_fn: Optional[Callable[[float], Tuple[float, float, float]]] = None,
    extra_fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build a comprehensive JSON-serialisable export of a globe grid.

    The returned dict has three top-level keys:

    ``metadata``
        Frequency, radius, tile counts, generator info.
    ``tiles``
        List of per-tile dicts with id, face_type, tile_slug, centre,
        normal, lat/lon, vertices, elevation, colour, and any extra
        fields.
    ``adjacency``
        Edge-list representation of the full tile graph.

    Parameters
    ----------
    globe_grid : GlobeGrid
        The globe grid to export.
    store : TileDataStore
        Tile data store (must have the *field_name* field populated).
    field_name : str
        Elevation field name in the store.
    ramp : str
        Colour ramp for terrain colours (``"satellite"`` or ``"topo"``).
    extra_fields : sequence of str, optional
        Additional store field names to include per tile (e.g.
        ``["river", "river_width", "biome"]``).

    Returns
    -------
    dict
        JSON-serialisable payload.
    """
    colour_map = globe_to_colour_map(globe_grid, store, field_name=field_name, ramp=ramp, ramp_fn=ramp_fn)
    slug_lookup = globe_grid.build_slug_lookup()
    extra = list(extra_fields or [])

    # ── Metadata ────────────────────────────────────────────────────
    metadata = {
        "version": _EXPORT_VERSION,
        "generator": "polygrid.globe_export",
        "frequency": globe_grid.frequency,
        "radius": globe_grid.radius,
        "tile_count": len(globe_grid.faces),
        "pentagon_count": sum(1 for f in globe_grid.faces.values() if f.face_type == "pent"),
        "hexagon_count": sum(1 for f in globe_grid.faces.values() if f.face_type == "hex"),
        "field_name": field_name,
        "colour_ramp": ramp,
        "extra_fields": extra,
    }

    # ── Tiles ───────────────────────────────────────────────────────
    tiles: List[Dict[str, Any]] = []
    for fid, face in sorted(globe_grid.faces.items()):
        # 3D vertex positions
        verts_3d = []
        for vid in face.vertex_ids:
            v = globe_grid.vertices[vid]
            verts_3d.append([round(v.x, 8), round(v.y, 8), round(v.z, 8)] if v.z is not None else [round(v.x, 8), round(v.y, 8)])

        # Colour
        rgb = colour_map.get(fid, (0.5, 0.5, 0.5))

        # Elevation
        elevation = float(store.get(fid, field_name))

        # Centre and normal from metadata
        center_3d = face.metadata.get("center_3d")
        normal_3d = face.metadata.get("normal_3d")
        lat = face.metadata.get("latitude_deg")
        lon = face.metadata.get("longitude_deg")
        tile_id = face.metadata.get("tile_id")

        entry: Dict[str, Any] = {
            "id": fid,
            "face_type": face.face_type,
            "models_tile_id": tile_id,
            "tile_slug": slug_lookup.get(fid),
            "vertices_3d": verts_3d,
            "center_3d": [round(c, 8) for c in center_3d] if center_3d else None,
            "normal_3d": [round(n, 8) for n in normal_3d] if normal_3d else None,
            "latitude_deg": round(lat, 6) if lat is not None else None,
            "longitude_deg": round(lon, 6) if lon is not None else None,
            "elevation": round(elevation, 6),
            "color": [round(c, 4) for c in rgb],
            "neighbor_ids": list(face.neighbor_ids),
        }

        # Well-known biome pipeline fields — included when present in the store.
        for biome_key in ("temperature", "moisture", "terrain", "region_id"):
            if store.schema.has_field(biome_key):
                try:
                    val = store.get(fid, biome_key)
                    if isinstance(val, float):
                        entry[biome_key] = round(val, 6)
                    else:
                        entry[biome_key] = val
                except (KeyError, ValueError):
                    entry[biome_key] = None
            else:
                entry[biome_key] = None

        # Features list — stored as comma-separated string in TileDataStore
        if store.schema.has_field("features"):
            try:
                raw = store.get(fid, "features")
                if isinstance(raw, str) and raw:
                    entry["features"] = [f.strip() for f in raw.split(",") if f.strip()]
                else:
                    entry["features"] = []
            except (KeyError, ValueError):
                entry["features"] = []
        else:
            entry["features"] = None

        # Extra fields from the store
        for key in extra:
            try:
                val = store.get(fid, key)
                # Make JSON-safe
                if isinstance(val, bool):
                    entry[key] = val
                elif isinstance(val, (int, float)):
                    entry[key] = round(float(val), 6) if isinstance(val, float) else val
                elif isinstance(val, str):
                    entry[key] = val
                else:
                    entry[key] = str(val)
            except (KeyError, ValueError):
                entry[key] = None

        tiles.append(entry)

    # ── Adjacency edge list ─────────────────────────────────────────
    adjacency_edges: List[List[str]] = []
    seen: set = set()
    for fid, face in globe_grid.faces.items():
        for nid in face.neighbor_ids:
            edge_key = tuple(sorted([fid, nid]))
            if edge_key not in seen:
                seen.add(edge_key)
                adjacency_edges.append([fid, nid])

    return {
        "metadata": metadata,
        "tiles": tiles,
        "adjacency": adjacency_edges,
    }


def export_globe_json(
    globe_grid: GlobeGrid,
    store: TileDataStore,
    path: Union[str, Path],
    *,
    field_name: str = "elevation",
    ramp: str = "satellite",
    extra_fields: Optional[Sequence[str]] = None,
    indent: int = 2,
) -> Path:
    """Export globe data to a JSON file.

    Parameters
    ----------
    globe_grid : GlobeGrid
    store : TileDataStore
    path : str or Path
        Output file path.
    field_name, ramp, extra_fields
        Forwarded to :func:`export_globe_payload`.
    indent : int
        JSON indentation.

    Returns
    -------
    Path
        The output path.
    """
    payload = export_globe_payload(
        globe_grid, store,
        field_name=field_name, ramp=ramp, extra_fields=extra_fields,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=indent))
    return out


def validate_globe_payload(payload: Dict[str, Any]) -> List[str]:
    """Validate a globe export payload against expected structure.

    Returns a list of error messages (empty = valid).

    This is a lightweight structural validator — not a full JSON Schema
    check.  Use the ``schemas/globe.schema.json`` file for formal
    validation with ``jsonschema``.
    """
    errors: List[str] = []

    # Top-level keys
    for key in ("metadata", "tiles", "adjacency"):
        if key not in payload:
            errors.append(f"Missing top-level key: {key}")

    meta = payload.get("metadata", {})
    for key in ("version", "frequency", "radius", "tile_count"):
        if key not in meta:
            errors.append(f"Missing metadata key: {key}")

    tiles = payload.get("tiles", [])
    if not isinstance(tiles, list):
        errors.append("'tiles' must be a list")
    else:
        expected_count = meta.get("tile_count", 0)
        if len(tiles) != expected_count:
            errors.append(f"tile_count mismatch: metadata says {expected_count}, got {len(tiles)}")

        for i, tile in enumerate(tiles):
            if "id" not in tile:
                errors.append(f"Tile {i}: missing 'id'")
            if "tile_slug" not in tile:
                errors.append(f"Tile {i}: missing 'tile_slug'")
            if "elevation" not in tile:
                errors.append(f"Tile {i}: missing 'elevation'")
            if "color" not in tile:
                errors.append(f"Tile {i}: missing 'color'")
            elif not isinstance(tile.get("color"), list) or len(tile.get("color", [])) != 3:
                errors.append(f"Tile {i}: 'color' must be [r, g, b]")
            if "vertices_3d" not in tile:
                errors.append(f"Tile {i}: missing 'vertices_3d'")
            if "center_3d" not in tile:
                errors.append(f"Tile {i}: missing 'center_3d'")
            if "neighbor_ids" not in tile:
                errors.append(f"Tile {i}: missing 'neighbor_ids'")

    adj = payload.get("adjacency", [])
    if not isinstance(adj, list):
        errors.append("'adjacency' must be a list")

    return errors
