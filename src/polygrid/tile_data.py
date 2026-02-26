"""Tile data layer — per-face key-value storage for terrain generation.

This module provides a way to attach arbitrary game data (elevation,
biome, moisture, etc.) to individual faces of a :class:`PolyGrid` or
:class:`CompositeGrid`, while keeping topology and data strictly
separated (SoC).

Architecture
------------
- A :class:`TileSchema` declares which keys exist and their types.
- A :class:`TileData` is the raw data container — a mapping of face ids
  to field values, validated against its schema.
- A :class:`TileDataStore` binds a ``TileData`` to a grid, providing
  convenience accessors, neighbour-aware queries, ring queries, and
  bulk operations.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

from .algorithms import build_face_adjacency, ring_faces
from .polygrid import PolyGrid


# ═══════════════════════════════════════════════════════════════════
# Schema
# ═══════════════════════════════════════════════════════════════════

# Types that can appear in a schema.  Mapped by name so that schemas
# round-trip through JSON without pickling Python type objects.
_TYPE_MAP: Dict[str, Type] = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
}

_TYPE_NAME_MAP: Dict[Type, str] = {v: k for k, v in _TYPE_MAP.items()}


@dataclass(frozen=True)
class FieldDef:
    """Definition of a single tile-data field.

    *name* — the key used when reading/writing.
    *dtype* — one of ``int``, ``float``, ``str``, ``bool``.
    *default* — the value assigned to faces that haven't been
    explicitly set.  Must match *dtype*.  Defaults to ``None`` which
    means "no default — field is required".
    """

    name: str
    dtype: Type
    default: Any = None

    def __post_init__(self) -> None:
        if self.dtype not in _TYPE_NAME_MAP:
            raise TypeError(
                f"Unsupported dtype {self.dtype!r} for field '{self.name}'. "
                f"Allowed: {list(_TYPE_NAME_MAP.keys())}"
            )
        if self.default is not None and not isinstance(self.default, self.dtype):
            raise TypeError(
                f"Default for '{self.name}' must be {self.dtype.__name__}, "
                f"got {type(self.default).__name__}"
            )


class TileSchema:
    """Declares the set of typed fields that can be stored per face.

    >>> schema = TileSchema([
    ...     FieldDef("elevation", float, 0.0),
    ...     FieldDef("biome", str, "none"),
    ... ])
    """

    def __init__(self, fields: Iterable[FieldDef]) -> None:
        self._fields: Dict[str, FieldDef] = {}
        for fd in fields:
            if fd.name in self._fields:
                raise ValueError(f"Duplicate field name: '{fd.name}'")
            self._fields[fd.name] = fd

    # ── public API ──────────────────────────────────────────────────

    @property
    def field_names(self) -> List[str]:
        """Ordered list of field names."""
        return list(self._fields.keys())

    def get_field(self, name: str) -> FieldDef:
        """Return the :class:`FieldDef` for *name*, or raise ``KeyError``."""
        return self._fields[name]

    def has_field(self, name: str) -> bool:
        return name in self._fields

    def validate_value(self, name: str, value: Any) -> None:
        """Raise ``TypeError`` if *value* is wrong for field *name*.

        Raise ``KeyError`` if *name* is not in the schema.
        """
        fd = self._fields[name]  # KeyError if missing
        if not isinstance(value, fd.dtype):
            raise TypeError(
                f"Field '{name}' expects {fd.dtype.__name__}, "
                f"got {type(value).__name__}"
            )

    def default_record(self) -> Dict[str, Any]:
        """Return a dict with default values for all fields that have one.

        Fields without a default are omitted — they must be set
        explicitly.
        """
        return {
            fd.name: fd.default
            for fd in self._fields.values()
            if fd.default is not None
        }

    # ── serialisation ───────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fields": [
                {
                    "name": fd.name,
                    "dtype": _TYPE_NAME_MAP[fd.dtype],
                    **({"default": fd.default} if fd.default is not None else {}),
                }
                for fd in self._fields.values()
            ]
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> TileSchema:
        fields: List[FieldDef] = []
        for entry in payload.get("fields", []):
            dtype = _TYPE_MAP[entry["dtype"]]
            default = entry.get("default")
            # Coerce JSON ints to float for float fields
            if dtype is float and isinstance(default, int):
                default = float(default)
            fields.append(FieldDef(name=entry["name"], dtype=dtype, default=default))
        return cls(fields)

    def __repr__(self) -> str:
        names = ", ".join(self.field_names)
        return f"TileSchema([{names}])"


# ═══════════════════════════════════════════════════════════════════
# Tile data container
# ═══════════════════════════════════════════════════════════════════

class TileData:
    """Raw per-face data container, validated against a :class:`TileSchema`.

    Internally stores ``{face_id: {field_name: value}}``.
    """

    def __init__(self, schema: TileSchema) -> None:
        self._schema = schema
        self._data: Dict[str, Dict[str, Any]] = {}

    # ── properties ──────────────────────────────────────────────────

    @property
    def schema(self) -> TileSchema:
        return self._schema

    @property
    def face_ids(self) -> Set[str]:
        """Face ids that have at least one field set."""
        return set(self._data.keys())

    # ── read / write ────────────────────────────────────────────────

    def get(self, face_id: str, key: str) -> Any:
        """Return the value for *key* on *face_id*.

        Falls back to the schema default.  Raises ``KeyError`` if the
        face has no value and the field has no default.
        """
        record = self._data.get(face_id)
        if record is not None and key in record:
            return record[key]
        fd = self._schema.get_field(key)  # KeyError if bad key
        if fd.default is not None:
            return fd.default
        raise KeyError(
            f"Face '{face_id}' has no value for '{key}' and field has no default"
        )

    def set(self, face_id: str, key: str, value: Any) -> None:
        """Set a single field, validating against the schema."""
        self._schema.validate_value(key, value)
        if face_id not in self._data:
            self._data[face_id] = {}
        self._data[face_id][key] = value

    def bulk_set(self, face_ids: Iterable[str], key: str, value: Any) -> None:
        """Set *key* = *value* for every face in *face_ids*."""
        self._schema.validate_value(key, value)
        for fid in face_ids:
            if fid not in self._data:
                self._data[fid] = {}
            self._data[fid][key] = value

    def has(self, face_id: str, key: str) -> bool:
        """Return True if *face_id* has an explicit value for *key*."""
        record = self._data.get(face_id)
        return record is not None and key in record

    def clear(self, face_id: str, key: Optional[str] = None) -> None:
        """Remove data for *face_id*.

        If *key* is given, remove just that field.  Otherwise remove
        all fields for the face.
        """
        if face_id not in self._data:
            return
        if key is None:
            del self._data[face_id]
        else:
            self._data[face_id].pop(key, None)
            if not self._data[face_id]:
                del self._data[face_id]

    # ── serialisation ───────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (schema + data)."""
        return {
            "schema": self._schema.to_dict(),
            "tiles": deepcopy(self._data),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> TileData:
        """Deserialise from a plain dict."""
        schema = TileSchema.from_dict(payload["schema"])
        td = cls(schema)
        for face_id, record in payload.get("tiles", {}).items():
            for key, value in record.items():
                fd = schema.get_field(key)
                # Coerce JSON ints → float for float fields
                if fd.dtype is float and isinstance(value, int):
                    value = float(value)
                td.set(face_id, key, value)
        return td

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> TileData:
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        return (
            f"TileData(schema={self._schema!r}, "
            f"faces={len(self._data)})"
        )


# ═══════════════════════════════════════════════════════════════════
# Tile data store — grid-aware wrapper
# ═══════════════════════════════════════════════════════════════════

class TileDataStore:
    """Binds a :class:`TileData` to a :class:`PolyGrid`, adding
    grid-aware queries and bulk operations.

    The store does **not** own the grid — it holds a reference for
    adjacency lookups.  Topology and data remain separate objects.

    Parameters
    ----------
    grid : PolyGrid
        The grid whose face ids this store maps data to.
    tile_data : TileData, optional
        Existing data to wrap.  If *None* a fresh :class:`TileData` is
        created from *schema*.
    schema : TileSchema, optional
        Required if *tile_data* is not given.
    """

    def __init__(
        self,
        grid: PolyGrid,
        tile_data: Optional[TileData] = None,
        schema: Optional[TileSchema] = None,
    ) -> None:
        if tile_data is None and schema is None:
            raise ValueError("Either tile_data or schema must be provided")
        self._grid = grid
        self._data = tile_data if tile_data is not None else TileData(schema)  # type: ignore[arg-type]
        self._adjacency: Optional[Dict[str, List[str]]] = None

    # ── properties ──────────────────────────────────────────────────

    @property
    def grid(self) -> PolyGrid:
        return self._grid

    @property
    def tile_data(self) -> TileData:
        return self._data

    @property
    def schema(self) -> TileSchema:
        return self._data.schema

    # ── adjacency (lazy-built) ──────────────────────────────────────

    def _ensure_adjacency(self) -> Dict[str, List[str]]:
        if self._adjacency is None:
            self._adjacency = build_face_adjacency(
                self._grid.faces.values(), self._grid.edges.values()
            )
        return self._adjacency

    # ── basic CRUD (delegate to TileData) ───────────────────────────

    def get(self, face_id: str, key: str) -> Any:
        """Get a field value for a face."""
        return self._data.get(face_id, key)

    def set(self, face_id: str, key: str, value: Any) -> None:
        """Set a field value for a face."""
        self._data.set(face_id, key, value)

    def bulk_set(self, face_ids: Iterable[str], key: str, value: Any) -> None:
        """Set *key* = *value* for all given face ids."""
        self._data.bulk_set(face_ids, key, value)

    # ── neighbour-aware queries ─────────────────────────────────────

    def get_neighbors_data(
        self, face_id: str, key: str
    ) -> List[Tuple[str, Any]]:
        """Return ``[(neighbor_id, value)]`` for all neighbours of *face_id*.

        Each neighbour's value is looked up via :meth:`TileData.get`,
        so schema defaults apply.
        """
        adj = self._ensure_adjacency()
        result: List[Tuple[str, Any]] = []
        for nid in adj.get(face_id, []):
            result.append((nid, self._data.get(nid, key)))
        return result

    def get_ring_data(
        self, face_id: str, radius: int, key: str
    ) -> Dict[int, List[Tuple[str, Any]]]:
        """Return data for faces within *radius* hops of *face_id*.

        Returns ``{ring_index: [(face_id, value), ...]}``.
        Ring 0 is the centre face itself.
        """
        adj = self._ensure_adjacency()
        rings = ring_faces(adj, face_id, max_depth=radius)
        result: Dict[int, List[Tuple[str, Any]]] = {}
        for depth, fids in rings.items():
            result[depth] = [
                (fid, self._data.get(fid, key)) for fid in fids
            ]
        return result

    # ── bulk operations ─────────────────────────────────────────────

    def apply_to_all(self, key: str, fn: Callable[[str, Any], Any]) -> None:
        """Apply *fn(face_id, current_value)* to every face in the grid.

        The return value of *fn* is written back as the new value for
        *key*.  Schema validation applies.
        """
        for face_id in self._grid.faces:
            current = self._data.get(face_id, key)
            new_val = fn(face_id, current)
            self._data.set(face_id, key, new_val)

    def apply_to_ring(
        self,
        center: str,
        radius: int,
        key: str,
        fn: Callable[[str, Any], Any],
    ) -> None:
        """Apply *fn* to every face within *radius* of *center*."""
        adj = self._ensure_adjacency()
        rings = ring_faces(adj, center, max_depth=radius)
        for fids in rings.values():
            for fid in fids:
                current = self._data.get(fid, key)
                self._data.set(fid, key, fn(fid, current))

    def apply_to_faces(
        self,
        face_ids: Iterable[str],
        key: str,
        fn: Callable[[str, Any], Any],
    ) -> None:
        """Apply *fn* to an explicit set of faces."""
        for fid in face_ids:
            current = self._data.get(fid, key)
            self._data.set(fid, key, fn(fid, current))

    # ── initialisation helpers ──────────────────────────────────────

    def initialise_all(self) -> None:
        """Set all fields to their schema defaults for every grid face.

        Useful after creating a store to ensure every face has values.
        Raises ``ValueError`` if any field lacks a default.
        """
        defaults = self._data.schema.default_record()
        missing = set(self._data.schema.field_names) - set(defaults.keys())
        if missing:
            raise ValueError(
                f"Cannot initialise: fields {missing} have no default"
            )
        for face_id in self._grid.faces:
            for key, value in defaults.items():
                if not self._data.has(face_id, key):
                    self._data.set(face_id, key, value)

    # ── serialisation ───────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return self._data.to_dict()

    def to_json(self, indent: int = 2) -> str:
        return self._data.to_json(indent=indent)

    @classmethod
    def from_dict(
        cls, grid: PolyGrid, payload: Dict[str, Any]
    ) -> TileDataStore:
        td = TileData.from_dict(payload)
        return cls(grid, tile_data=td)

    @classmethod
    def from_json(
        cls, grid: PolyGrid, json_str: str
    ) -> TileDataStore:
        td = TileData.from_json(json_str)
        return cls(grid, tile_data=td)

    def __repr__(self) -> str:
        return (
            f"TileDataStore(grid_faces={len(self._grid.faces)}, "
            f"data_faces={len(self._data.face_ids)}, "
            f"schema={self._data.schema!r})"
        )


# ═══════════════════════════════════════════════════════════════════
# File I/O helpers
# ═══════════════════════════════════════════════════════════════════

PathLike = Union[str, Path]


def save_tile_data(tile_data: TileData, path: PathLike) -> None:
    """Write tile data to a JSON file."""
    Path(path).write_text(tile_data.to_json(), encoding="utf-8")


def load_tile_data(path: PathLike) -> TileData:
    """Read tile data from a JSON file."""
    return TileData.from_json(Path(path).read_text(encoding="utf-8"))
