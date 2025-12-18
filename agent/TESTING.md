# Testing Strategy

## Test tiers
1) **Unit tests (fast, required)**
- lineup parsing (string → slots)
- salary/projection computation
- stack pattern classification
- archetype assignment
- hashing stability
- distribution fitting (binning correctness)

2) **Golden file tests (recommended)**
- For a small fixture slate + contest:
  - enriched_entries output matches expected (row count, key columns)
  - distribution JSON matches expected structure

3) **Integration tests (optional early, required later)**
- training pipeline end-to-end on small fixtures
- contest pipeline end-to-end with tiny candidate pool (e.g., 200 lineups)

## Fixtures
- Put sanitized sample CSVs in `tests/fixtures/`.
- Provide a `conftest.py` with helper loaders.

## Determinism
- Tests must set a fixed seed.
- Hashing must be stable across platforms:
  - canonicalize lineup ordering
  - explicit encoding (utf-8)
  - use sha256 of canonical string

## What to assert
- Parse success rate
- No illegal lineups (salary cap, duplicates)
- Joins resolve player names (or raise with clear error)
- Distribution sums to ~1.0 where appropriate

## CI expectations
- `pytest -q`
- `ruff`/`flake8`/`black --check`
- `mypy`/`pyright` (choose one)
