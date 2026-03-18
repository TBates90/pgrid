# Testing quickguide

This repository has a large test-suite. Use these quick commands to avoid long runs while iterating.

Prerequisites
- Install the sibling `models` package in editable mode (required for globe tests):

```bash
# from the pgrid repo root, with the pgrid venv active
pip install -e ../models
```

Run only fast tests (recommended during development)

```bash
# from repo root
./scripts/run_fast.sh
# or directly
python3 -m pytest -m fast
```

Run the full test-suite with grouped output and timing (slower):

```bash
python3 scripts/run_tests.py
```

Run only the medium+fast but not slow tests:

```bash
python3 -m pytest -m "not slow"
```

How tests are grouped
- Fast: quick unit tests (< ~3s per file)
- Medium: moderately expensive integration tests (3-60s per file)
- Slow: long-running or performance tests (> 1m per file)

If a particular test file is taking too long for your iteration flow, run it explicitly or mark it `slow` in `tests/conftest.py` so it is omitted from routine runs.
