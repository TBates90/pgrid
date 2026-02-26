"""Tests for ``polygrid.pipeline`` — terrain step/pipeline framework (7F)."""

from __future__ import annotations

import pytest

from polygrid import build_pentagon_centered_grid
from polygrid.mountains import MOUNTAIN_RANGE, MountainConfig
from polygrid.pipeline import (
    CustomStep,
    MountainStep,
    PipelineResult,
    RiverStep,
    StepResult,
    TerrainPipeline,
    TerrainStep,
)
from polygrid.rivers import RiverConfig
from polygrid.tile_data import FieldDef, TileSchema, TileDataStore


# ── helpers ─────────────────────────────────────────────────────────

@pytest.fixture()
def small_grid():
    """Build a small pent-centred grid for fast pipeline tests."""
    return build_pentagon_centered_grid(1)


@pytest.fixture()
def store(small_grid):
    """TileDataStore with an ``elevation`` field, initialised to 0."""
    schema = TileSchema([FieldDef("elevation", float, 0.0)])
    s = TileDataStore(small_grid, schema=schema)
    s.initialise_all()
    return s


# ═══════════════════════════════════════════════════════════════════
# 7F.1 — TerrainStep protocol
# ═══════════════════════════════════════════════════════════════════


def test_custom_step_satisfies_protocol():
    step = CustomStep("noop", lambda g, s, rm: None)
    assert isinstance(step, TerrainStep)
    assert step.name == "noop"


def test_mountain_step_satisfies_protocol():
    step = MountainStep(config=MOUNTAIN_RANGE)
    assert isinstance(step, TerrainStep)
    assert step.name == "mountains"


def test_river_step_satisfies_protocol():
    step = RiverStep()
    assert isinstance(step, TerrainStep)
    assert step.name == "rivers"


# ═══════════════════════════════════════════════════════════════════
# 7F.2 — TerrainPipeline basics
# ═══════════════════════════════════════════════════════════════════


def test_empty_pipeline_is_noop(small_grid, store):
    """An empty pipeline should run without error and change nothing."""
    pipe = TerrainPipeline()
    result = pipe.run(small_grid, store)
    assert len(result.elapsed) == 0
    assert len(result.step_results) == 0
    # Store unchanged — all elevations still 0
    for fid in small_grid.faces:
        assert store.get(fid, "elevation") == 0.0


def test_pipeline_len():
    step_a = CustomStep("a", lambda g, s, rm: None)
    step_b = CustomStep("b", lambda g, s, rm: None)
    pipe = TerrainPipeline([step_a, step_b])
    assert len(pipe) == 2
    assert pipe.step_names == ["a", "b"]


def test_pipeline_add_and_insert():
    pipe = TerrainPipeline()
    pipe.add(CustomStep("a", lambda g, s, rm: None))
    pipe.add(CustomStep("c", lambda g, s, rm: None))
    pipe.insert(1, CustomStep("b", lambda g, s, rm: None))
    assert pipe.step_names == ["a", "b", "c"]


def test_pipeline_repr():
    pipe = TerrainPipeline([CustomStep("x", lambda g, s, rm: None)])
    assert "x" in repr(pipe)


# ═══════════════════════════════════════════════════════════════════
# 7F.2 — Pipeline runs steps in declared order
# ═══════════════════════════════════════════════════════════════════


def test_pipeline_runs_in_order(small_grid, store):
    """Steps execute in declaration order and each sees prior mutations."""
    order: list[str] = []

    def step_a(g, s, rm):
        order.append("a")
        # Set elevation to 1.0 on first face
        first_fid = list(g.faces.keys())[0]
        s.set(first_fid, "elevation", 1.0)
        return None

    def step_b(g, s, rm):
        order.append("b")
        # Verify step_a's write is visible
        first_fid = list(g.faces.keys())[0]
        assert s.get(first_fid, "elevation") == 1.0
        s.set(first_fid, "elevation", 2.0)
        return None

    pipe = TerrainPipeline([
        CustomStep("a", step_a),
        CustomStep("b", step_b),
    ])
    pipe.run(small_grid, store)
    assert order == ["a", "b"]
    first_fid = list(small_grid.faces.keys())[0]
    assert store.get(first_fid, "elevation") == 2.0


def test_step_reads_previous_step_data(small_grid, store):
    """A later step can read data written by an earlier step."""
    captured: dict = {}

    def writer(g, s, rm):
        for fid in g.faces:
            s.set(fid, "elevation", 0.5)
        return None

    def reader(g, s, rm):
        vals = [s.get(fid, "elevation") for fid in g.faces]
        captured["mean"] = sum(vals) / len(vals)
        return None

    pipe = TerrainPipeline([
        CustomStep("writer", writer),
        CustomStep("reader", reader),
    ])
    pipe.run(small_grid, store)
    assert captured["mean"] == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════
# 7F.2 — Hooks
# ═══════════════════════════════════════════════════════════════════


def test_before_after_hooks(small_grid, store):
    """Before and after hooks fire with correct arguments."""
    events: list[tuple] = []

    def on_before(name, idx, total):
        events.append(("before", name, idx, total))

    def on_after(name, idx, total):
        events.append(("after", name, idx, total))

    pipe = TerrainPipeline(
        [CustomStep("s1", lambda g, s, rm: None),
         CustomStep("s2", lambda g, s, rm: None)],
        before=on_before,
        after=on_after,
    )
    pipe.run(small_grid, store)
    assert events == [
        ("before", "s1", 0, 2),
        ("after", "s1", 0, 2),
        ("before", "s2", 1, 2),
        ("after", "s2", 1, 2),
    ]


# ═══════════════════════════════════════════════════════════════════
# 7F.2 — PipelineResult
# ═══════════════════════════════════════════════════════════════════


def test_pipeline_result_elapsed(small_grid, store):
    """Elapsed times are recorded for every step."""
    pipe = TerrainPipeline([
        CustomStep("fast", lambda g, s, rm: None),
    ])
    result = pipe.run(small_grid, store)
    assert "fast" in result.elapsed
    assert result.elapsed["fast"] >= 0.0


def test_pipeline_result_artefacts(small_grid, store):
    """Steps that return StepResult have artefacts accessible."""
    def produces(g, s, rm):
        return StepResult(artefacts={"answer": 42})

    pipe = TerrainPipeline([CustomStep("producer", produces)])
    result = pipe.run(small_grid, store)
    assert result.artefact("producer", "answer") == 42


def test_pipeline_result_artefact_missing():
    """Accessing a missing artefact raises KeyError."""
    result = PipelineResult()
    with pytest.raises(KeyError):
        result.artefact("no_such_step", "key")


# ═══════════════════════════════════════════════════════════════════
# 7F.3 — Built-in MountainStep
# ═══════════════════════════════════════════════════════════════════


def test_mountain_step_sets_elevation(small_grid, store):
    """MountainStep should populate elevation on all faces."""
    from dataclasses import replace
    config = replace(MOUNTAIN_RANGE, seed=99)
    step = MountainStep(config=config)
    step(small_grid, store)
    elevations = [store.get(fid, "elevation") for fid in small_grid.faces]
    assert any(e > 0.0 for e in elevations)


def test_mountain_step_in_pipeline(small_grid, store):
    """MountainStep works when executed via a pipeline."""
    from dataclasses import replace
    config = replace(MOUNTAIN_RANGE, seed=42)
    pipe = TerrainPipeline([MountainStep(config=config)])
    result = pipe.run(small_grid, store)
    assert "mountains" in result.elapsed
    elevations = [store.get(fid, "elevation") for fid in small_grid.faces]
    assert any(e > 0.0 for e in elevations)


# ═══════════════════════════════════════════════════════════════════
# 7F.3 — Built-in RiverStep
# ═══════════════════════════════════════════════════════════════════


def test_river_step_returns_network(small_grid, store):
    """RiverStep should return a RiverNetwork artefact."""
    from dataclasses import replace
    # First give terrain some elevation so rivers have something to flow on
    config = replace(MOUNTAIN_RANGE, seed=99)
    MountainStep(config=config)(small_grid, store)

    river_cfg = RiverConfig(min_accumulation=2, min_length=2, seed=99)
    step = RiverStep(config=river_cfg)
    result = step(small_grid, store)
    assert result is not None
    assert "network" in result.artefacts


def test_river_step_in_pipeline(small_grid, store):
    """Mountains → Rivers pipeline runs end-to-end."""
    from dataclasses import replace
    m_cfg = replace(MOUNTAIN_RANGE, seed=42)
    r_cfg = RiverConfig(min_accumulation=2, min_length=2, seed=42)
    pipe = TerrainPipeline([
        MountainStep(config=m_cfg),
        RiverStep(config=r_cfg),
    ])
    result = pipe.run(small_grid, store)
    assert "mountains" in result.elapsed
    assert "rivers" in result.elapsed


# ═══════════════════════════════════════════════════════════════════
# 7F.3 — CustomStep
# ═══════════════════════════════════════════════════════════════════


def test_custom_step_runs(small_grid, store):
    """CustomStep executes its function."""
    called = {"yes": False}

    def fn(g, s, rm):
        called["yes"] = True
        return None

    step = CustomStep("my_step", fn)
    step(small_grid, store)
    assert called["yes"]


def test_custom_step_with_return(small_grid, store):
    """CustomStep can return a StepResult."""
    def fn(g, s, rm):
        return StepResult(artefacts={"val": 7})

    step = CustomStep("ret", fn)
    result = step(small_grid, store)
    assert result.artefacts["val"] == 7


# ═══════════════════════════════════════════════════════════════════
# Integration — full pipeline
# ═══════════════════════════════════════════════════════════════════


def test_full_mountain_river_pipeline(small_grid, store):
    """Full mountains → rivers pipeline modifies store correctly."""
    from dataclasses import replace

    m_cfg = replace(MOUNTAIN_RANGE, seed=7)
    r_cfg = RiverConfig(min_accumulation=2, min_length=2, seed=7)

    pipe = TerrainPipeline([
        MountainStep(config=m_cfg),
        RiverStep(config=r_cfg, carve=True),
    ])
    result = pipe.run(small_grid, store)

    # Mountains ran
    elevations = [store.get(fid, "elevation") for fid in small_grid.faces]
    assert max(elevations) > min(elevations), "Some elevation variation expected"

    # River network is accessible
    network = result.artefact("rivers", "network")
    assert network is not None
