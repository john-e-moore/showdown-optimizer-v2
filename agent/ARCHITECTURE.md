# Architecture

## Goal
Two clear, composable pipelines with strict boundaries:

1) **Training/Data pipeline**  
Raw projections + contest standings/exports → enriched entry table → learned target distributions.

2) **Contest execution pipeline**  
Raw projections + correlations + DKEntries CSV → enumerate slate lineup universe → compute softmax
utilities/shares from `theta.json` → simulate contest fields → prune → grade (ROI/top-% rates) →
assign unique top-X lineups per contest → filled DKEntries.

## Required directory structure
```

.
├─ src/
│  └─ dfs_opt/
│     ├─ __init__.py
│     ├─ config/              # pydantic settings, defaults, schema
│     ├─ models/              # dataclasses / pydantic models (schemas)
│     ├─ io/                  # readers/writers (csv/xlsx), path utilities
│     ├─ parsing/             # DK lineup parsing, Sabersim parsing
│     ├─ features/            # feature engineering (salary, stacks, archetypes, gaps)
│     ├─ distributions/       # fit/validate target distributions
│     ├─ lineup_pool/         # (optional) candidate generation helpers (not required for theta-based field sim)
│     ├─ reweighting/         # (optional) max-entropy / raking utilities (not used in Pipeline B for now)
│     ├─ simulation/          # score sims, contest sims, ROI, duplication-aware EV
│     ├─ share_model/         # softmax lineup-share model (train + apply theta)
│     ├─ cli/                 # Typer/argparse entrypoints
│     └─ utils/               # small shared helpers (pure functions only)
├─ tests/
│  ├─ fixtures/
│  └─ test_*.py
├─ scripts/                   # thin wrappers calling src (no business logic)
├─ agent/                     # these docs
└─ data/                      # ignored; local only (raw + outputs)
```

## Layering rules (imports)
- `models` and `utils` must not import from higher layers.
- `io` and `parsing` may import `models`, `utils`.
- `features` may import `models`, `utils`, `parsing` (but not `simulation`).
- `simulation` may import `features` but not `io` (simulation stays pure on in-memory data).
- `cli` is the *only* layer allowed to orchestrate reading/writing + calling pipeline functions.

## “One place” rules
- **All schemas** live in `models/` and are referenced everywhere else.
- **All file formats** live in `io/` (column mapping, encodings, etc).
- **All parsing** lives in `parsing/` (string → structured).
- **All feature engineering** lives in `features/` (structured → enriched).
- **All randomness** is controlled via `config` and seeded at pipeline start.

## Public API (what downstream code uses)
- `dfs_opt.pipelines.training.run_training_pipeline(config) -> TrainingArtifacts`
- `dfs_opt.pipelines.contest.run_contest_pipeline(config) -> ContestArtifacts`
- `dfs_opt.cli.*` for command-line.

Agents must not invent alternate “side pipelines” without updating this doc.


## Artifacts layout (required)
All pipeline runs must write debuggable artifacts under a stable path:

```
artifacts/
  <pipeline_name>/                 # training | contest
    <run_id>/                      # e.g. 2025-12-18T104455Z_ab12cd
      run_manifest.json
      logs/
        run.log
      steps/
        00_ingest/
          step_manifest.json
          preview.csv              # small human-readable sample (<=200 rows)
          schema.json              # column names + dtypes + null rates
          outputs.parquet          # optional: step output persisted when enabled
        01_parse/
        ...
```

**Rule:** Step folders are append-only for that run. Never overwrite files in-place; write new files and update manifests.
