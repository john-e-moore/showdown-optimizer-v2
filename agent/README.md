# Cursor Agent Instructions (dfs-optimizer reboot)

This folder is the **single source of truth** for how the codebase must be built.
Any AI coding agent must follow these documents as hard requirements.

## Non‑negotiables
- **Artifacts + metadata for every step:** each transformation writes a step artifact folder with previews + a `step_manifest.json` you can inspect.
- **Idempotent pipelines:** same inputs + same config ⇒ same outputs (byte-for-byte where feasible).
- **Data contracts:** every transformation produces documented schema + metadata.
- **DRY, but readable:** prefer small, named functions over clever reuse.
- **Typed Python:** type hints everywhere; run `mypy` (or `pyright`) in CI.
- **Tests first for logic:** parsing, feature engineering, and optimization constraints must have unit tests.
- **Structured logging:** every CLI run emits a machine-readable `run_manifest.json`.
- **No silent failures:** validate inputs, raise explicit exceptions, and log context.

## How to use these docs with Cursor
1. Paste the relevant doc(s) into the agent prompt when starting a new task.
2. Tell the agent: **“You must not deviate from the architecture and data contracts.”**
3. For each change request, include:
   - the *exact* function/class/module names to modify,
   - expected inputs/outputs,
   - and the tests to add/update.

## Definition of Done (DoD) for every PR/change
- [ ] Code follows `STYLE_GUIDE.md`
- [ ] Input/output schemas updated in `DATA_CONTRACTS.md` (if touched)
- [ ] Affected pipeline steps documented in `PIPELINES.md` (if touched)
- [ ] Unit tests added/updated; `pytest` passes
- [ ] Logs include run_id and key counts; manifest emitted
- [ ] No new “misc” scripts outside `src/` and `scripts/` with purpose noted

## Repo layout (required)
See `ARCHITECTURE.md` for the authoritative structure.
