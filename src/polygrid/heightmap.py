"""Grid ↔ noise bridge — apply noise primitives to PolyGrid tile data.

This is the thin adapter layer between :mod:`noise` (pure math, no grid
dependency) and :class:`~tile_data.TileDataStore` (grid-aware per-face
data).  It provides helpers to sample any noise function at face
centroids and write the results into a TileData field, plus smoothing,
blending, and normalization utilities that operate on existing fields.

Functions
---------
- :func:`sample_noise_field` — evaluate a noise fn at every face centroid
- :func:`sample_noise_field_region` — same, restricted to a :class:`Region`
- :func:`smooth_field` — neighbour-averaging pass
- :func:`blend_fields` — combine two fields via a blend function
- :func:`clamp_field` — clamp a field to ``[lo, hi]``
- :func:`normalize_field` — rescale a field across all faces to ``[lo, hi]``
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from .algorithms import build_face_adjacency, get_face_adjacency
from .geometry import face_center, face_center_3d
from .polygrid import PolyGrid
from .tile_data import TileDataStore


# ═══════════════════════════════════════════════════════════════════
# Sample noise → TileData
# ═══════════════════════════════════════════════════════════════════

def sample_noise_field(
    grid: PolyGrid,
    store: TileDataStore,
    field_name: str,
    noise_fn: Callable[[float, float], float],
    *,
    face_ids: Optional[Iterable[str]] = None,
) -> None:
    """Evaluate *noise_fn* at every face centroid and write to *field_name*.

    Parameters
    ----------
    grid : PolyGrid
        The grid whose face centroids provide sample coordinates.
    store : TileDataStore
        Tile data store to write into.
    field_name : str
        Name of the TileData field to write (must exist in schema).
    noise_fn : callable
        A function ``(x, y) → float``.  Typically a closure over one
        of the :mod:`noise` primitives with parameters baked in.
    face_ids : iterable of str, optional
        If given, only these faces are sampled.  Others are untouched.
    """
    targets = face_ids if face_ids is not None else grid.faces.keys()
    for fid in targets:
        face = grid.faces.get(fid)
        if face is None:
            continue
        c = face_center(grid.vertices, face)
        if c is None:
            continue
        cx, cy = c
        store.set(fid, field_name, noise_fn(cx, cy))


def sample_noise_field_region(
    grid: PolyGrid,
    store: TileDataStore,
    field_name: str,
    noise_fn: Callable[[float, float], float],
    region_face_ids: Iterable[str],
) -> None:
    """Like :func:`sample_noise_field` but restricted to a region.

    A convenience wrapper — simply forwards *region_face_ids* as the
    ``face_ids`` parameter.
    """
    sample_noise_field(grid, store, field_name, noise_fn, face_ids=region_face_ids)


def sample_noise_field_3d(
    grid: PolyGrid,
    store: TileDataStore,
    field_name: str,
    noise_fn: Callable[[float, float, float], float],
    *,
    face_ids: Optional[Iterable[str]] = None,
) -> None:
    """Evaluate a 3-D *noise_fn* at every face centroid and write to *field_name*.

    This is the 3-D analogue of :func:`sample_noise_field`.  It computes
    face centroids using ``(x, y, z)`` coordinates and passes all three
    to *noise_fn*.  This avoids seam and polar artefacts that occur when
    mapping spherical coordinates to a 2-D noise domain.

    Requires every face vertex to have a ``z`` coordinate; faces with
    missing 3-D positions are skipped.

    Parameters
    ----------
    grid : PolyGrid
    store : TileDataStore
    field_name : str
    noise_fn : callable
        A function ``(x, y, z) → float``.
    face_ids : iterable of str, optional
    """
    targets = face_ids if face_ids is not None else grid.faces.keys()
    for fid in targets:
        face = grid.faces.get(fid)
        if face is None:
            continue
        c = face_center_3d(grid.vertices, face)
        if c is None:
            continue
        cx, cy, cz = c
        store.set(fid, field_name, noise_fn(cx, cy, cz))


# ═══════════════════════════════════════════════════════════════════
# Smoothing
# ═══════════════════════════════════════════════════════════════════

def smooth_field(
    grid: PolyGrid,
    store: TileDataStore,
    field_name: str,
    *,
    iterations: int = 1,
    self_weight: float = 0.5,
    face_ids: Optional[Iterable[str]] = None,
) -> None:
    """Neighbour-averaging pass on a TileData field.

    For each target face, the new value is:

        ``self_weight * self + (1 − self_weight) * mean(neighbours)``

    Repeated *iterations* times.  Operates in-place (each iteration
    reads from the values written by the previous iteration).

    Parameters
    ----------
    grid : PolyGrid
    store : TileDataStore
    field_name : str
        The field to smooth.
    iterations : int
        Number of smoothing passes.
    self_weight : float
        Weight of the face's own value vs. its neighbours' mean.
        Must be in ``[0, 1]``.
    face_ids : iterable of str, optional
        If given, only these faces are smoothed.  Neighbour values
        outside the set are still read (boundary smoothing).
    """
    adj = get_face_adjacency(grid)

    target_set: Optional[Set[str]] = None
    if face_ids is not None:
        target_set = set(face_ids)

    targets = target_set if target_set is not None else set(grid.faces.keys())

    for _ in range(iterations):
        new_vals: Dict[str, float] = {}
        for fid in targets:
            own = store.get(fid, field_name)
            neighbours = adj.get(fid, [])
            if not neighbours:
                new_vals[fid] = own
                continue
            n_sum = sum(store.get(nid, field_name) for nid in neighbours)
            n_mean = n_sum / len(neighbours)
            new_vals[fid] = self_weight * own + (1.0 - self_weight) * n_mean

        for fid, val in new_vals.items():
            store.set(fid, field_name, val)


# ═══════════════════════════════════════════════════════════════════
# Blend
# ═══════════════════════════════════════════════════════════════════

def blend_fields(
    store: TileDataStore,
    field_a: str,
    field_b: str,
    field_out: str,
    blend_fn: Callable[[float, float], float],
    *,
    face_ids: Optional[Iterable[str]] = None,
) -> None:
    """Combine two TileData fields into a third via a blend function.

    For every face: ``out = blend_fn(a, b)``.

    Parameters
    ----------
    store : TileDataStore
    field_a, field_b : str
        Source fields.
    field_out : str
        Destination field (may be the same as *field_a* or *field_b*).
    blend_fn : callable
        ``(a_value, b_value) → output_value``.
    face_ids : iterable of str, optional
        Restrict to these faces.
    """
    targets = face_ids if face_ids is not None else store.grid.faces.keys()
    for fid in targets:
        a = store.get(fid, field_a)
        b = store.get(fid, field_b)
        store.set(fid, field_out, blend_fn(a, b))


# ═══════════════════════════════════════════════════════════════════
# Clamp / normalize
# ═══════════════════════════════════════════════════════════════════

def clamp_field(
    store: TileDataStore,
    field_name: str,
    lo: float = 0.0,
    hi: float = 1.0,
    *,
    face_ids: Optional[Iterable[str]] = None,
) -> None:
    """Clamp a TileData field to ``[lo, hi]``."""
    targets = face_ids if face_ids is not None else store.grid.faces.keys()
    for fid in targets:
        v = store.get(fid, field_name)
        store.set(fid, field_name, max(lo, min(hi, v)))


def normalize_field(
    store: TileDataStore,
    field_name: str,
    lo: float = 0.0,
    hi: float = 1.0,
    *,
    face_ids: Optional[Iterable[str]] = None,
) -> None:
    """Rescale a TileData field so its actual min→*lo* and actual max→*hi*.

    If all values are identical, every face is set to ``(lo + hi) / 2``.
    """
    targets = list(face_ids) if face_ids is not None else list(store.grid.faces.keys())
    if not targets:
        return

    vals = [store.get(fid, field_name) for fid in targets]
    vmin, vmax = min(vals), max(vals)

    if vmin == vmax:
        mid = (lo + hi) / 2.0
        for fid in targets:
            store.set(fid, field_name, mid)
        return

    for fid in targets:
        v = store.get(fid, field_name)
        t = (v - vmin) / (vmax - vmin)
        store.set(fid, field_name, lo + t * (hi - lo))
