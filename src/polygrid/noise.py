"""Reusable noise primitives for terrain generation.

Every function in this module operates on plain ``(x, y)`` coordinates
and returns a ``float``.  There is **no** dependency on grids, tile data,
or any other PolyGrid module — these are pure-math building blocks that
higher-level modules (:mod:`heightmap`, :mod:`mountains`, :mod:`rivers`)
compose into terrain features.

Functions
---------
- :func:`fbm` — Fractal Brownian Motion (multi-octave noise)
- :func:`ridged_noise` — inverted-abs noise that forms sharp ridges
- :func:`domain_warp` — warp coordinates through a secondary noise field
- :func:`gradient_mask` — radial / directional falloff mask
- :func:`terrace` — remap continuous values into stepped plateaus
- :func:`normalize` — rescale ``[a, b] → [c, d]``
- :func:`remap` — alias for :func:`normalize`
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════
# Base noise source — opensimplex if available, hash fallback
# ═══════════════════════════════════════════════════════════════════

_HAS_OPENSIMPLEX = False

try:
    import opensimplex as _osx

    _HAS_OPENSIMPLEX = True
except ImportError:
    _osx = None  # type: ignore[assignment]


def _init_noise(seed: int) -> Callable[[float, float], float]:
    """Return a 2-D noise function seeded with *seed*.

    Uses ``opensimplex.noise2`` when available.  Falls back to a
    deterministic hash-based pseudo-noise that is smooth-ish but not
    as pretty — good enough for tests and CI where opensimplex may
    not be installed.
    """
    if _HAS_OPENSIMPLEX:
        _osx.seed(seed)
        return _osx.noise2  # type: ignore[return-value]

    # ── deterministic hash fallback ─────────────────────────────────
    def _hash_noise(x: float, y: float) -> float:
        # Smooth-ish hash noise via a grid of hashed values + bilinear
        # interpolation.  Not great quality, but deterministic and has
        # no dependencies.
        ix, iy = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - ix, y - iy

        # Smoothstep
        fx = fx * fx * (3.0 - 2.0 * fx)
        fy = fy * fy * (3.0 - 2.0 * fy)

        def _h(px: int, py: int) -> float:
            h = hash((px, py, seed)) & 0xFFFFFFFF
            return (h / 0xFFFFFFFF) * 2.0 - 1.0  # → [−1, 1]

        v00 = _h(ix, iy)
        v10 = _h(ix + 1, iy)
        v01 = _h(ix, iy + 1)
        v11 = _h(ix + 1, iy + 1)

        top = v00 + (v10 - v00) * fx
        bot = v01 + (v11 - v01) * fx
        return top + (bot - top) * fy

    return _hash_noise


# ═══════════════════════════════════════════════════════════════════
# Fractal Brownian Motion
# ═══════════════════════════════════════════════════════════════════

def fbm(
    x: float,
    y: float,
    *,
    octaves: int = 6,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
    frequency: float = 1.0,
    seed: int = 42,
) -> float:
    """Fractal Brownian Motion — layered multi-octave noise.

    Sums several octaves of a base noise function, each at higher
    frequency and lower amplitude, producing natural-looking terrain
    variation.

    Parameters
    ----------
    x, y : float
        Sample coordinates.
    octaves : int
        Number of noise layers (more = finer detail).
    lacunarity : float
        Frequency multiplier between octaves (typically ~2.0).
    persistence : float
        Amplitude multiplier between octaves (typically ~0.5).
    frequency : float
        Base spatial frequency (larger = more zoomed-in features).
    seed : int
        Random seed for the noise source.

    Returns
    -------
    float
        A value in approximately ``[−1, 1]`` (exact bounds depend on
        parameters; see :func:`normalize` if you need a strict range).
    """
    noise2 = _init_noise(seed)
    value = 0.0
    amplitude = 1.0
    freq = frequency
    max_amp = 0.0  # for normalisation

    for _ in range(octaves):
        value += amplitude * noise2(x * freq, y * freq)
        max_amp += amplitude
        amplitude *= persistence
        freq *= lacunarity

    # Normalize to roughly [−1, 1]
    return value / max_amp if max_amp > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# Ridged noise
# ═══════════════════════════════════════════════════════════════════

def ridged_noise(
    x: float,
    y: float,
    *,
    octaves: int = 6,
    lacunarity: float = 2.2,
    persistence: float = 0.5,
    frequency: float = 1.0,
    ridge_offset: float = 1.0,
    seed: int = 42,
) -> float:
    """Ridged multifractal noise — sharp ridges at zero-crossings.

    Each octave's noise is ``offset − |noise|``, so values near zero
    in the base noise become peaks.  Subsequent octaves are weighted
    by the previous octave's signal, concentrating detail on the
    ridges themselves.

    Parameters
    ----------
    x, y : float
        Sample coordinates.
    octaves : int
        Number of noise layers.
    lacunarity : float
        Frequency multiplier between octaves.
    persistence : float
        Base amplitude decay.  Actual per-octave weight is also
        modulated by the previous octave's signal.
    frequency : float
        Base spatial frequency.
    ridge_offset : float
        Controls ridge height — higher values push ridges taller.
    seed : int
        Random seed.

    Returns
    -------
    float
        A value in approximately ``[0, 1]``.
    """
    noise2 = _init_noise(seed)
    value = 0.0
    weight = 1.0
    freq = frequency

    for i in range(octaves):
        signal = noise2(x * freq, y * freq)
        signal = ridge_offset - abs(signal)
        signal *= signal  # sharpen ridges
        signal *= weight
        weight = max(0.0, min(1.0, signal * persistence))
        value += signal * (persistence ** i)
        freq *= lacunarity

    # Normalize: empirical max ≈ sum of geometric series
    max_val = sum(persistence ** i for i in range(octaves))
    return max(0.0, min(1.0, value / max_val)) if max_val > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# Domain warp
# ═══════════════════════════════════════════════════════════════════

def domain_warp(
    noise_fn: Callable[..., float],
    x: float,
    y: float,
    *,
    warp_strength: float = 0.3,
    warp_frequency: float = 1.0,
    warp_seed_x: int = 137,
    warp_seed_y: int = 251,
    **noise_kwargs,
) -> float:
    """Apply domain warping to *noise_fn*.

    Offsets ``(x, y)`` by secondary noise fields before evaluating
    *noise_fn*, producing organic, swirly distortions.

    Parameters
    ----------
    noise_fn : callable
        The noise function to evaluate (e.g. :func:`fbm`,
        :func:`ridged_noise`).  Must accept ``(x, y, **kwargs)``.
    x, y : float
        Original sample coordinates.
    warp_strength : float
        Amplitude of the warp offset (larger = more distortion).
    warp_frequency : float
        Spatial frequency of the warp noise.
    warp_seed_x, warp_seed_y : int
        Seeds for the two warp noise channels (should differ from
        each other and from the main noise seed).
    **noise_kwargs
        Forwarded to *noise_fn* (e.g. ``octaves``, ``seed``).

    Returns
    -------
    float
        The warped noise value.
    """
    warp_nx = _init_noise(warp_seed_x)
    warp_ny = _init_noise(warp_seed_y)

    dx = warp_nx(x * warp_frequency, y * warp_frequency) * warp_strength
    dy = warp_ny(x * warp_frequency, y * warp_frequency) * warp_strength

    return noise_fn(x + dx, y + dy, **noise_kwargs)


# ═══════════════════════════════════════════════════════════════════
# Gradient mask
# ═══════════════════════════════════════════════════════════════════

def gradient_mask(
    x: float,
    y: float,
    *,
    center: Tuple[float, float] = (0.0, 0.0),
    radius: float = 1.0,
    falloff: str = "smooth",
) -> float:
    """Radial gradient mask — 1.0 at *center*, falling to 0.0 at *radius*.

    Useful for fading elevation toward region edges or coastlines.

    Parameters
    ----------
    x, y : float
        Sample coordinates.
    center : tuple of float
        Centre of the mask.
    radius : float
        Distance at which the mask reaches 0.
    falloff : str
        Falloff curve: ``"linear"``, ``"smooth"`` (Hermite/smoothstep),
        or ``"exponential"``.

    Returns
    -------
    float
        A value in ``[0, 1]``.
    """
    d = math.hypot(x - center[0], y - center[1])
    t = min(1.0, d / radius) if radius > 0 else 1.0

    if falloff == "linear":
        return 1.0 - t
    elif falloff == "smooth":
        # Smoothstep: 3t² − 2t³ inverted
        s = t * t * (3.0 - 2.0 * t)
        return 1.0 - s
    elif falloff == "exponential":
        return math.exp(-3.0 * t)
    else:
        raise ValueError(f"Unknown falloff type: {falloff!r}")


# ═══════════════════════════════════════════════════════════════════
# Terrace
# ═══════════════════════════════════════════════════════════════════

def terrace(
    value: float,
    *,
    steps: int = 4,
    smoothing: float = 0.0,
) -> float:
    """Remap a continuous value into stepped plateaus.

    Parameters
    ----------
    value : float
        Input value (typically in ``[0, 1]``).
    steps : int
        Number of discrete levels (must be ≥ 1).
    smoothing : float
        Blend factor between terraced and original value.
        0.0 = pure terrace, 1.0 = no terracing (original value).

    Returns
    -------
    float
        The terraced value.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    terraced = math.floor(value * steps) / steps
    smoothing = max(0.0, min(1.0, smoothing))
    return terraced * (1.0 - smoothing) + value * smoothing


# ═══════════════════════════════════════════════════════════════════
# Normalize / remap
# ═══════════════════════════════════════════════════════════════════

def normalize(
    value: float,
    *,
    src_min: float = -1.0,
    src_max: float = 1.0,
    dst_min: float = 0.0,
    dst_max: float = 1.0,
) -> float:
    """Linearly remap *value* from ``[src_min, src_max]`` to ``[dst_min, dst_max]``.

    Values outside the source range are clamped.
    """
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    t = (value - src_min) / (src_max - src_min)
    t = max(0.0, min(1.0, t))
    return dst_min + t * (dst_max - dst_min)


# Alias
remap = normalize


# ═══════════════════════════════════════════════════════════════════
# 3-D noise variants (for seamless globe terrain)
# ═══════════════════════════════════════════════════════════════════

def _init_noise3(seed: int) -> Callable[[float, float, float], float]:
    """Return a 3-D noise function seeded with *seed*.

    Uses ``opensimplex.noise3`` when available.  Falls back to a
    deterministic hash-based pseudo-noise.
    """
    if _HAS_OPENSIMPLEX:
        _osx.seed(seed)
        return _osx.noise3  # type: ignore[return-value]

    # ── deterministic hash fallback ─────────────────────────────────
    def _hash_noise3(x: float, y: float, z: float) -> float:
        ix, iy, iz = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
        fx = x - ix
        fy = y - iy
        fz = z - iz
        fx = fx * fx * (3.0 - 2.0 * fx)
        fy = fy * fy * (3.0 - 2.0 * fy)
        fz = fz * fz * (3.0 - 2.0 * fz)

        def _h(px: int, py: int, pz: int) -> float:
            h = hash((px, py, pz, seed)) & 0xFFFFFFFF
            return (h / 0xFFFFFFFF) * 2.0 - 1.0

        # Trilinear interpolation
        v000 = _h(ix, iy, iz)
        v100 = _h(ix + 1, iy, iz)
        v010 = _h(ix, iy + 1, iz)
        v110 = _h(ix + 1, iy + 1, iz)
        v001 = _h(ix, iy, iz + 1)
        v101 = _h(ix + 1, iy, iz + 1)
        v011 = _h(ix, iy + 1, iz + 1)
        v111 = _h(ix + 1, iy + 1, iz + 1)

        x00 = v000 + (v100 - v000) * fx
        x10 = v010 + (v110 - v010) * fx
        x01 = v001 + (v101 - v001) * fx
        x11 = v011 + (v111 - v011) * fx
        y0 = x00 + (x10 - x00) * fy
        y1 = x01 + (x11 - x01) * fy
        return y0 + (y1 - y0) * fz

    return _hash_noise3


def fbm_3d(
    x: float,
    y: float,
    z: float,
    *,
    octaves: int = 6,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
    frequency: float = 1.0,
    seed: int = 42,
) -> float:
    """3-D Fractal Brownian Motion — seamless noise for spherical surfaces.

    Identical algorithm to :func:`fbm` but samples in 3-D, making it
    suitable for evaluating noise on the surface of a unit sphere
    ``(x, y, z)`` with no seam or polar artefacts.

    Returns a value in approximately ``[-1, 1]``.
    """
    noise3 = _init_noise3(seed)
    value = 0.0
    amplitude = 1.0
    freq = frequency
    max_amp = 0.0

    for _ in range(octaves):
        value += amplitude * noise3(x * freq, y * freq, z * freq)
        max_amp += amplitude
        amplitude *= persistence
        freq *= lacunarity

    return value / max_amp if max_amp > 0 else 0.0


def ridged_noise_3d(
    x: float,
    y: float,
    z: float,
    *,
    octaves: int = 6,
    lacunarity: float = 2.2,
    persistence: float = 0.5,
    frequency: float = 1.0,
    ridge_offset: float = 1.0,
    seed: int = 42,
) -> float:
    """3-D ridged multifractal noise — sharp ridges on a sphere.

    Same algorithm as :func:`ridged_noise` but in 3-D.
    Returns a value in approximately ``[0, 1]``.
    """
    noise3 = _init_noise3(seed)
    value = 0.0
    weight = 1.0
    freq = frequency

    for i in range(octaves):
        signal = noise3(x * freq, y * freq, z * freq)
        signal = ridge_offset - abs(signal)
        signal *= signal
        signal *= weight
        weight = max(0.0, min(1.0, signal * persistence))
        value += signal * (persistence ** i)
        freq *= lacunarity

    max_val = sum(persistence ** i for i in range(octaves))
    return max(0.0, min(1.0, value / max_val)) if max_val > 0 else 0.0
