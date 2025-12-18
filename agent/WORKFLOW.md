# Workflow for Cursor / AI Agents

## 1) Work in small vertical slices
Do not build everything at once. Each slice should:
- add one module or one pipeline step
- add tests for that step
- update docs (contracts/pipelines) if needed

Suggested slice order:
1. Parsers (`parsing/`) + tests
2. Feature engineering (`features/`) + tests
3. Training pipeline orchestration (`pipelines/training.py`)
4. Distribution fitting (`distributions/`)
5. Candidate pool generation (`lineup_pool/`)
6. Reweighting (`reweighting/`)
7. Simulation (`simulation/`)
8. DKEntries writer (`io/`)

## 2) Prompt template for Cursor tasks
When asking Cursor to implement something, include:
- Relevant doc links: ARCHITECTURE + DATA_CONTRACTS + STYLE_GUIDE
- Concrete acceptance criteria:
  - new modules/functions/classes
  - expected inputs/outputs
  - unit tests to add
  - any new config keys

Example:
- “Implement DK lineup parser in `dfs_opt/parsing/dk_lineup.py` producing `ParsedLineup`.
  Add tests for 5 known lineup strings, including edge cases.”

## 3) Guardrails you should enforce in review
Reject PRs that:
- introduce business logic in `scripts/`
- write files from inside `features/` or `simulation/`
- add new columns without updating DATA_CONTRACTS
- add parameters without wiring through config + manifest
- skip tests for core logic

## 4) Logging + manifests
Every CLI run must:
- emit `run_manifest.json`
- log row counts per step
- record warnings with enough context to reproduce

## 5) Documentation updates
- If a file format changes: update DATA_CONTRACTS.
- If a pipeline step changes: update PIPELINES.
- If architecture changes: update ARCHITECTURE.


## 6) Inspecting artifacts (required dev loop)
After each slice, the agent must demonstrate debuggability by pointing to artifacts:
- open `artifacts/<pipeline>/<run_id>/steps/<NN_step>/preview.csv`
- check `schema.json` for dtype drift
- check `step_manifest.json` for row counts and metrics

If a bug is reported, first reproduce it by re-running with the same config + seed and comparing
`schema_fingerprint` and `data_fingerprint` across runs.
