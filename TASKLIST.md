# Task List — Polygrid (Updated)

> Goal: Produce a **robust pentagon‑centered hex grid** using **combinatorics‑first topology**, a **Tutte embedding**, and **quality validation** so PNGs are trustworthy across ring counts.

## Assumptions
- Implementation language is **Python**.
- We prefer correctness over speed and will use additional packages when they improve results.

## Phase 0 — Scope & Contract
- [ ] Confirm ring counts, expected PNG sizes, and symmetry expectations.
- [ ] Define accepted quality metrics (min angle, aspect ratio, edge variance).
- [ ] Lock JSON contract fields for topology + embedding metadata.

## Phase 1 — Topology: true 5‑valent disclination
- [x] Build **triangular lattice** for a given ring count.
- [x] Remove a **60° wedge** (disclination) and **merge boundary rays**.
- [x] Construct **dual graph** to obtain pentagon + hex faces.
- [ ] Validate combinatorics:
  - exactly **one pentagon** (center), rest hexes,
  - vertex degree = 3,
  - expected face counts per ring.

## Phase 2 — Embedding (Tutte baseline + relaxation)
- [x] Implement **Tutte embedding** (SciPy sparse solve):
  - fix outer boundary to convex polygon,
  - solve for interior vertices.
- [ ] Implement **spring/energy relaxation** stage (optional):
  - edge length variance penalty,
  - face angle variance penalty,
  - inversion barrier.
- [ ] Add **embedding metadata** (iterations, residuals) to JSON export.

## Phase 3 — Rendering & Visual Validation
- [ ] Render PNGs from embedded grids:
  - per‑ring output set (e.g., rings 0–5),
  - highlight center pentagon,
  - optional ring labels.
- [ ] Add **PNG snapshot tests** (hash or metadata checks).

## Phase 4 — Validation Metrics
- [ ] Compute quality metrics per ring:
  - min angle, max aspect ratio,
  - edge length variance by ring,
  - dual cell area distribution.
- [ ] Export metrics to JSON/CSV for comparison across rings.

## Phase 5 — Algorithms (Topology‑Only)
- [ ] Implement shortest path (weighted by edge length when embedded).
- [ ] Implement region selection (face flood fill / ring windowing).
- [ ] Keep algorithms compatible with **pent + hex** grids.

## Phase 6 — Dependencies (if needed)
- [ ] Add SciPy (sparse solves for Tutte).
- [ ] Add NetworkX (planar diagnostics, optional checks).
- [ ] Add Shapely (polygonization diagnostics, optional).

## Phase 7 — CLI + Reproducibility
- [ ] CLI: `build-pent --rings N --embed tutte --render-out ...`.
- [ ] CLI: `validate --metrics --out metrics.json`.
- [ ] Versioned exports for experiment reproducibility.

---

## Quality Gates
- [ ] Unit tests for topology (disclination + dualization).
- [ ] Embedding tests (Tutte residual, no inverted faces).
- [ ] PNG render smoke tests.
- [ ] Lint/format checks.
