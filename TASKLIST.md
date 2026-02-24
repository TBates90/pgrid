# TASKLIST — Pentagon‑centered hex grids

This task list consolidates the research notes in `pentagon-hex grid research.md` and `pentagon-hex grid research followup.md` into an actionable roadmap for stabilising pentagon‑centered grids across ring counts.

## ✅ High-level goals

- **Topology is correct and validated** before any embedding steps.
- **Embedding is globally consistent** (Tutte baseline + optimisation), not “angle‑first” local construction.
- **Quality gates prevent regressions** (no inverted faces, no crossings, angle/edge variance within tolerances).
- **CLI + docs reflect the stable pipeline** and expose diagnostics.

## ✅ Completed work (archived)

- Phases 0–7 completed: repo hygiene, topology validation, face cycle invariants, embedding pipeline, optimisation, diagnostics, CLI/docs, and tests are all in place.
- Deterministic exports and golden command workflow are implemented.
- Experimental angle-first code is isolated under `src/polygrid/experiments/`.

## Phase 8 — Cleanup & refactor (new)

- [ ] **Refactor `builders.py` into smaller modules** (topology, embedding helpers, optimisation, snapping utilities).
  - [ ] Move ring snap/relax helpers into `embedding_utils.py` (or similar) and re-export only public helpers.
  - [ ] Keep builder surface area to `build_pure_hex_grid`, `build_pentagon_centered_grid`, and validation helpers.
- [ ] **Remove legacy/duplicated implementations** that are no longer part of the research-aligned pipeline.
  - [ ] Delete unused “angle-first” legacy helpers still referenced in `builders.py` (keep only experiments version).
  - [ ] Remove any deprecated snapping paths no longer used by `embedding_strategies.py`.
- [ ] **Consolidate embedding logic** so ring-1 special handling is contained in one place.
  - [ ] Move ring-1 scaling/symmetry logic into `embedding_strategies.py` with a documented “ring-1 exception” section.
  - [ ] Ensure `tutte+optimise` for rings >1 remains the only production path.
- [ ] **Simplify diagnostics helpers** (remove duplicated cycle/angle utilities across modules).
  - [ ] Keep a single authoritative cycle/angle implementation; re-use it in diagnostics and builders.
- [ ] **Delete legacy build artefacts** (if any remain) and keep `.gitignore` aligned.
  - [ ] Remove `build/lib/` copies if present in repo.
  - [ ] Verify no `src/polygrid.egg-info/` artifacts remain.

## Phase 9 — Research-aligned follow-ups (new)

- [ ] **Document operator choices** (polygonal Laplacian vs cotan vs VEM/DEC) and ensure a single operator family is used per experiment.
- [ ] **Add mesh quality metrics** for ring scaling experiments (min angle, aspect ratio, dual cell area stats).
- [ ] **Guard Tutte preconditions** by checking boundary convexity and connectivity; log a warning when guarantees don’t hold.
- [ ] **Add optional export adapters** (`meshio` or simple OBJ/PLY) for downstream analysis.

## Completion criteria

- [ ] Pent grids are stable visually up to at least 4–5 rings without self‑intersection.
- [ ] All diagnostics pass in CI for hex + pent builds.
- [ ] Cleanup/refactor complete: `builders.py` slimmed, legacy paths removed.
- [ ] Experimental paths remain isolated and documented as such.
