# Style Guide (Python)

## Code style
- Use **Black** formatting + **isort**.
- Prefer **dataclasses** for internal immutable structures; use **pydantic** for external config/schemas where validation matters.
- Every function has:
  - type hints
  - docstring describing inputs/outputs and invariants
  - explicit error behavior (exceptions)

## Functional purity (for pipeline steps)
- Core transforms should be **pure functions**:
  - input DataFrame(s)/objects in
  - output DataFrame(s)/objects out
  - no global state
  - no reading/writing inside transform functions

## Naming conventions
- Modules: `snake_case.py`
- Classes: `PascalCase`
- Functions/vars: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

## Logging (required)
- Use `structlog` or the standard `logging` module with JSON formatter.
- Every log line includes:
  - `run_id`
  - `pipeline` (training/contest)
  - `step`
  - key counts (rows in/out)
- Never log entire DataFrames.

## Errors & validation
- Validate early with explicit exceptions:
  - missing columns
  - unexpected nulls
  - duplicate players
  - lineup parse failures
- Convert “soft problems” into warnings only if they are safe and measurable.

## Configuration
- All CLI configs load from:
  - a YAML/JSON config file
  - plus CLI overrides
- Config must be serializable and saved into the run manifest.

## Performance guidelines
- Prefer vectorized pandas where possible.
- For combinatorial optimization, isolate heavy compute behind interfaces
  (`ILineupOptimizer`, `ISimulator`), so you can swap implementations later.

## Git hygiene / PR checklist
- No notebooks checked in unless specifically requested.
- No “quick” scripts containing business logic.
- If adding a dependency, justify it in the PR description.


## Artifact + metadata emission (mandatory)
Every pipeline step must emit:
- `preview.csv` (first N rows, N<=200, UTF-8)
- `schema.json` (columns, dtypes, null counts/rates, example values if cheap)
- `step_manifest.json` (inputs/outputs checksums + row counts + timings + metrics)

Implementation rules:
- Core transforms stay pure (no I/O). They return `(data, StepStats)` where `StepStats` holds metrics.
- Orchestration code (pipeline runner) is responsible for writing artifacts via a shared utility
  (e.g. `io/artifacts.py` with `ArtifactWriter.write_step(...)`).
- Never sprinkle ad-hoc `to_csv()` throughout the codebase. **One place** owns artifact writing.
- Always include `config_fingerprint` + `code_version` (git SHA if available) in manifests.
