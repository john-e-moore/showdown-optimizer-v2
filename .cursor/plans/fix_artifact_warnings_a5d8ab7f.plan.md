---
name: Fix artifact warnings
overview: Fix the end-of-pipeline artifact writing by standardizing warning serialization (so StepManifest validation can’t fail) and making the data fingerprint computation safe for duplicate DataFrame columns (so pandas doesn’t warn or drop data).
todos:
  - id: warn-schema
    content: Update StructuredWarning to accept legacy `type` alias and add `details` field while always serializing `code`.
    status: completed
  - id: writer-coerce-warnings
    content: In ArtifactWriter.write_step, normalize/validate warnings into StructuredWarning objects; fix the parquet-skip warning to emit `code` + `details`.
    status: completed
  - id: writer-stable-fingerprint
    content: Replace preview.to_dict-based fingerprinting with a duplicate-column-safe, positional preview serializer (e.g., keys with `@index`).
    status: completed
  - id: test-dup-cols
    content: Add a regression test ensuring write_step succeeds with duplicate columns when persist_parquet=True and writes a valid manifest warning.
    status: completed
---

# Clean fix for end-of-pipeline artifacts

## What’s failing (root cause)

- `ArtifactWriter.write_step()` appends a warning dict shaped like `{"type": ..., "message": ..., ...}` when `persist_parquet=True` and `df_out` has duplicate columns (like duplicated `UTIL`).
- But `StepManifest.warnings` is typed as `List[StructuredWarning]`, and `StructuredWarning` currently requires a **`code`** field.
- Result: `StepManifest(... warnings=[{"type": ...}]) `triggers Pydantic error `warnings.0.code Field required`, killing the pipeline at the very end.

## Goals

- Make artifact writing **never crash** just because we skipped parquet (or any other warning).
- Make preview/data fingerprinting **safe with duplicate columns** (no pandas “columns are not unique” warning; no silent column dropping).
- Keep this change **backward compatible** with any existing call sites that pass `type` instead of `code`.

## Implementation plan

### 1) Make warning schema robust + backward compatible

- Update [`src/dfs_opt/models/manifests.py`](/home/john/showdown-optimizer-v2/src/dfs_opt/models/manifests.py):
- Extend `StructuredWarning` to accept either `code` or legacy `type` as input (alias), but always serialize as `code`.
- Add an optional `details: Dict[str, Any] = Field(default_factory=dict) `so we can retain structured info like `duplicate_columns` without cramming it into the message.

### 2) Normalize warnings inside the artifact writer

- Update [`src/dfs_opt/io/artifacts.py`](/home/john/showdown-optimizer-v2/src/dfs_opt/io/artifacts.py):
- Replace the internal parquet-skip warning payload from `{"type": ...}` to `{"code": ...}` and move `duplicate_columns` into `details`.
- Add a small coercion step in `write_step()`:
    - Accept the existing `warnings: Sequence[Dict[str, Any]]` input.
    - Convert every warning dict into a `StructuredWarning` via `StructuredWarning.model_validate(...)` so `StepManifest` always receives `List[StructuredWarning]`.

### 3) Fix data fingerprinting for duplicate columns (remove pandas warning)

- Update [`src/dfs_opt/io/artifacts.py`](/home/john/showdown-optimizer-v2/src/dfs_opt/io/artifacts.py):
- Replace `preview.to_dict(orient="records")` (which warns and drops data when columns are duplicated) with a stable preview serializer that:
    - iterates columns **by position**
    - uses keys like `"{col_name}@{i}"` to disambiguate duplicates
    - builds `List[Dict[str, object]]` for hashing
- This keeps `data_fingerprint` deterministic and removes the noisy `DataFrame columns are not unique` warning.

### 4) Add a targeted regression test

- Update/add tests in [`tests/test_training_pipeline_artifacts.py`](/home/john/showdown-optimizer-v2/tests/test_training_pipeline_artifacts.py) (or a small new test file if clearer):
- Unit test `ArtifactWriter.write_step()` with a DataFrame that has duplicate columns and `persist_parquet=True`.
- Assert:
    - step folder is written
    - `step_manifest.json` exists
    - manifest `warnings[0].code == "parquet_skipped_duplicate_columns"`
    - pipeline does not raise

### 5) Verify by rerunning the contest pipeline

- Rerun `python scripts/run_pipeline_b.py` with `persist_step_outputs=True` and confirm:
- step `07_assign_best_lineups_to_entries` completes
- run finishes and writes `run_manifest.json`
- no end-of-pipeline crash

## Files expected to change

- [`src/dfs_opt/models/manifests.py`](/home/john/showdown-optimizer-v2/src/dfs_opt/models/manifests.py)
- [`src/dfs_opt/io/artifacts.py`](/home/john/showdown-optimizer-v2/src/dfs_opt/io/artifacts.py)
- [`tests/test_training_pipeline_artifacts.py`](/home/john/showdown-optimizer-v2/tests/test_training_pipeline_artifacts.py) (or a new focused test)

## Implementation todos

- **warn-schema**: Update `StructuredWarning` to accept `type` alias and add `details`
- **writer-coerce-warnings**: Coerce warning dicts to `StructuredWarning` in `ArtifactWriter.write_step`