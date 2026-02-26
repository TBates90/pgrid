"""Terrain pipeline — composable step-based terrain generation framework.

Phase 7F — provides :class:`TerrainStep` (protocol) and
:class:`TerrainPipeline` (sequencer) so that terrain-generation stages
(mountains, rivers, biome assignment, etc.) can be declared in order and
run as a single pipeline.

Usage
-----
>>> from polygrid.pipeline import TerrainPipeline, MountainStep, RiverStep
>>> from polygrid.mountains import MOUNTAIN_RANGE
>>> from polygrid.rivers import RiverConfig
>>>
>>> pipe = TerrainPipeline([
...     MountainStep(config=MOUNTAIN_RANGE),
...     RiverStep(config=RiverConfig()),
... ])
>>> result = pipe.run(grid, store)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

from .mountains import MountainConfig, generate_mountains
from .polygrid import PolyGrid
from .regions import RegionMap
from .rivers import (
    RiverConfig,
    RiverNetwork,
    carve_river_valleys,
    generate_rivers,
)
from .tile_data import TileDataStore


# ═══════════════════════════════════════════════════════════════════
# 7F.1 — TerrainStep protocol
# ═══════════════════════════════════════════════════════════════════


@dataclass
class StepResult:
    """Optional return value from a terrain step, carrying artefacts.

    Steps may attach arbitrary artefacts (e.g. a :class:`RiverNetwork`)
    so that downstream steps or the caller can inspect them.
    """

    artefacts: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TerrainStep(Protocol):
    """Protocol for a terrain-generation step.

    Any object that satisfies this protocol can be added to a
    :class:`TerrainPipeline`.  The step mutates *store* in place and
    may optionally return a :class:`StepResult` containing artefacts.
    """

    @property
    def name(self) -> str:
        """Human-readable name for logging."""
        ...

    def __call__(
        self,
        grid: PolyGrid,
        store: TileDataStore,
        region_map: Optional[RegionMap] = None,
    ) -> Optional[StepResult]:
        """Execute the step, mutating *store* in place."""
        ...


# ═══════════════════════════════════════════════════════════════════
# 7F.2 — TerrainPipeline
# ═══════════════════════════════════════════════════════════════════

Hook = Callable[[str, int, int], None]
"""Signature for before/after hooks: ``(step_name, step_index, total_steps)``."""


@dataclass
class PipelineResult:
    """Aggregate result of running a full pipeline.

    Attributes
    ----------
    step_results : dict[str, StepResult]
        Mapping of ``step.name → StepResult`` for every step that
        returned one.
    elapsed : dict[str, float]
        Mapping of ``step.name → seconds`` wall-clock time per step.
    """

    step_results: Dict[str, StepResult] = field(default_factory=dict)
    elapsed: Dict[str, float] = field(default_factory=dict)

    def artefact(self, step_name: str, key: str) -> Any:
        """Convenience accessor for a specific artefact.

        Raises ``KeyError`` if the step or key is not present.
        """
        return self.step_results[step_name].artefacts[key]


class TerrainPipeline:
    """Ordered sequence of :class:`TerrainStep` instances.

    Parameters
    ----------
    steps : list[TerrainStep]
        Steps to execute in order.
    before : Hook | None
        Called *before* each step.
    after : Hook | None
        Called *after* each step.

    Usage::

        pipe = TerrainPipeline([step_a, step_b])
        result = pipe.run(grid, store)
    """

    def __init__(
        self,
        steps: Optional[List[TerrainStep]] = None,
        *,
        before: Optional[Hook] = None,
        after: Optional[Hook] = None,
    ) -> None:
        self._steps: List[TerrainStep] = list(steps or [])
        self._before = before
        self._after = after

    # ── mutation ────────────────────────────────────────────────────

    def add(self, step: TerrainStep) -> "TerrainPipeline":
        """Append a step and return *self* for chaining."""
        self._steps.append(step)
        return self

    def insert(self, index: int, step: TerrainStep) -> "TerrainPipeline":
        """Insert a step at *index* and return *self* for chaining."""
        self._steps.insert(index, step)
        return self

    # ── execution ───────────────────────────────────────────────────

    def run(
        self,
        grid: PolyGrid,
        store: TileDataStore,
        region_map: Optional[RegionMap] = None,
    ) -> PipelineResult:
        """Execute all steps in order, returning aggregate results."""
        result = PipelineResult()
        total = len(self._steps)

        for idx, step in enumerate(self._steps):
            sname = step.name
            if self._before:
                self._before(sname, idx, total)

            t0 = time.perf_counter()
            step_result = step(grid, store, region_map)
            dt = time.perf_counter() - t0

            result.elapsed[sname] = dt
            if step_result is not None:
                result.step_results[sname] = step_result

            if self._after:
                self._after(sname, idx, total)

        return result

    # ── introspection ───────────────────────────────────────────────

    @property
    def step_names(self) -> List[str]:
        """Ordered list of step names."""
        return [s.name for s in self._steps]

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        names = ", ".join(self.step_names)
        return f"TerrainPipeline([{names}])"


# ═══════════════════════════════════════════════════════════════════
# 7F.3 — Built-in steps
# ═══════════════════════════════════════════════════════════════════


@dataclass
class MountainStep:
    """Terrain step wrapping :func:`mountains.generate_mountains`.

    Parameters
    ----------
    config : MountainConfig
        Mountain generation configuration.
    region_key : str | None
        If set, restrict mountains to the named region from the pipeline's
        *region_map*.  If ``None``, mountains cover the whole grid.
    """

    config: MountainConfig
    region_key: Optional[str] = None

    @property
    def name(self) -> str:
        return "mountains"

    def __call__(
        self,
        grid: PolyGrid,
        store: TileDataStore,
        region_map: Optional[RegionMap] = None,
    ) -> Optional[StepResult]:
        region = None
        if self.region_key and region_map is not None:
            region = region_map.get_region(self.region_key)
        generate_mountains(grid, store, self.config, region=region)
        return None


@dataclass
class RiverStep:
    """Terrain step wrapping :func:`rivers.generate_rivers`.

    Generates the river network *and* carves valleys into the elevation
    field.  The resulting :class:`RiverNetwork` is returned as an
    artefact named ``"network"`` so downstream code can render it.

    Parameters
    ----------
    config : RiverConfig
        River generation configuration.
    carve : bool
        Whether to carve river valleys into the elevation field.
    """

    config: RiverConfig = field(default_factory=RiverConfig)
    carve: bool = True

    @property
    def name(self) -> str:
        return "rivers"

    def __call__(
        self,
        grid: PolyGrid,
        store: TileDataStore,
        region_map: Optional[RegionMap] = None,
    ) -> Optional[StepResult]:
        network = generate_rivers(grid, store, self.config)
        if self.carve and len(network) > 0:
            carve_river_valleys(
                grid,
                store,
                network,
                carve_depth=self.config.carve_depth,
            )
        return StepResult(artefacts={"network": network})


@dataclass
class CustomStep:
    """Inline terrain step from an arbitrary callable.

    Convenience wrapper for one-off steps that don't warrant a full
    class.

    Usage::

        step = CustomStep("flatten", lambda g, s, rm: s.set_many(
            {fid: {"elevation": 0.0} for fid in g.faces}
        ))
        pipe = TerrainPipeline([step])
    """

    _name: str
    fn: Callable[
        [PolyGrid, TileDataStore, Optional[RegionMap]],
        Optional[StepResult],
    ]

    @property
    def name(self) -> str:
        return self._name

    def __call__(
        self,
        grid: PolyGrid,
        store: TileDataStore,
        region_map: Optional[RegionMap] = None,
    ) -> Optional[StepResult]:
        return self.fn(grid, store, region_map)
