# Data Contracts

This file defines canonical schemas. Any new column must be documented here.

## 1) Raw Sabersim Projections (input)
**Name:** `sabersim_projections_raw.csv` (example)
- Required columns (canonical names):
  - `player_name` (string)
  - `team` (string)
  - `opponent` (string)
  - `position` (string, optional)
  - `salary` (int)
  - `proj_points` (float)
  - `minutes` (float, optional)
  - `is_captain_row` (bool) — derived during parse
- Parsing rules:
  - Keep **one canonical FLEX row** per player (lowest salary row).
  - Drop CPT duplicates; derive CPT values as needed as 1.5x.

## 2) Raw DraftKings Contest Standings/Export (input)
**Name:** `contest-standings-<id>.csv`
- Required columns (canonical names):
  - `entry_id` (string or int, if present)
  - `contest_id` (string or int, if present)
  - `user` (string)
  - `lineup_str` (string) — the DK lineup field
  - `points` (float)
  - `rank` (int)
  - `winnings` (float)

## 2b) Raw Sabersim Correlation Matrix (input)
**Name:** `*_corr_matrix.csv` (example: `NBA_slate1_corr_matrix.csv`)

This file is expected to be a square correlation matrix over the slate player pool.

- Required columns:
  - `Column1` (string): player name for the row
  - One column per player (string column name matches player name)
- Required invariants:
  - The set of column names (excluding `Column1`) matches the set of `Column1` values
  - Diagonal entries are 1.0 (or very close)
  - Values are finite floats in [-1, 1]

## 2c) DraftKings DKEntries CSV (input)
**Name:** `DKEntries.csv`

This is the user’s entry template used by Pipeline B. It may contain one or many contests; Pipeline B
groups rows by `contest_id` and fills each contest independently.

- Required columns (canonical names):
  - `entry_id` (string or int) — from DK (`Entry ID` / `EntryId`)
  - `contest_id` (string or int) — from DK (`Contest ID` / `ContestId`)
  - Showdown roster slots (exact column names vary by export; Pipeline B normalizes):
    - `cpt` (string) — captain slot (`CPT` / `Captain`)
    - `util1`..`util5` (string) — utility slots (commonly `UTIL`, `UTIL.1`… or `UTIL1`..`UTIL5`)
- Optional passthrough columns (preserved as-is):
  - `contest_name`, `entry_fee`, `draft_group_id`, `draft_group`, etc.

Notes:
- Slot values are typically player-identifying strings in DK’s upload format. Pipeline B must be
  consistent with the chosen write format (names vs `Name (ID)`), and name matching must be
  compatible with the projections/correlation inputs for the slate.

## 3) Parsed Lineup (intermediate)
**Name:** `ParsedLineup`
- `cpt_name` (string)
- `util_names` (list[string], length 5)
- `players_all` (list[string], length 6)

Invariants:
- No duplicates in `players_all`
- Exactly 6 slots (1 CPT + 5 UTIL)

## 4) Enriched Contest Entries (output of feature pipeline)
**Name:** `entries_enriched.parquet` (preferred) or `.csv`
Required columns:
- identifiers
  - `slate_id` (string)
  - `contest_id` (string)
  - `entry_id` (string)
  - `user` (string)
- lineup
  - `cpt_name`
  - `util1_name` ... `util5_name`
  - `lineup_hash` (string; stable canonical hash)
- computed
  - `salary_used` (int)
  - `salary_left` (int)
  - `salary_left_bin` (string; bins: `0_200`, `200_500`, `500_1000`, `1000_2000`, `2000_4000`, `4000_8000`, `8000_plus`)
  - `proj_points` (float) — computed from projections
  - `optimal_proj_points` (float)
  - `proj_gap_to_optimal` (float)
  - `pct_proj_gap_to_optimal` (float)
  - `pct_proj_gap_to_optimal_bin` (string; bins: `0_0.01`, `0.01_0.02`, `0.02_0.04`, `0.04_0.07`, `0.07_0.15`, `0.15_0.30`, `0.30_plus`)
  - `stack_pattern` (string like `4-2`)
  - `heavy_team` (string)
  - `cpt_team` (string)
  - `cpt_archetype` (string: `stud_1_2`, `stud_3_5`, `mid_6_10`, `value_11_plus`)
  - `dup_count` (int) — count of identical `lineup_hash` in contest
  - `pct_contest_lineups` (float) — `dup_count / contest_size`
  - `own_score_logprod` (float) — sum of `log(own)` across all 6 rostered players (CPT treated same as UTIL)
  - `own_max_log` (float) — max `log(own)` across the 6 players
  - `own_min_log` (float) — min `log(own)` across the 6 players
  - `avg_corr` (float) — average pairwise Pearson correlation across the 6 players (denominator 15)

## 5) Target Distributions (output of training pipeline)
Store as versioned JSON files:
- `salary_left_bins.json`
- `proj_gap_bins.json`
- `stack_pattern_rates.json`
- `cpt_archetype_rates.json`
- `duplication_histogram.json`

Each distribution file includes:
- `schema_version`
- `generated_at`
- `source_contests` (list)
- `counts` and/or `bin_edges`
- validation stats (mean, p50, p90)

## 5b) Lineup Universe (output of contest pipeline, optional early step)
**Name:** `lineups.parquet` + `lineups_enriched.parquet` + `players.parquet` + `metadata.json`

This artifact represents the full set of legal showdown lineups for a slate.
Lineups are stored as **indices** into the corresponding `players.parquet` row order.

### 5b.1 Players table (index basis)
**Name:** `players.parquet`

Minimum expected columns:
- `name_norm` (string)
- `team` (string)
- `salary` (int)
- `proj_points` (float)

Additional columns may be present (e.g., `own`, `minutes`) and should be treated as optional.

### 5b.2 Lineups table
**Name:** `lineups.parquet`

Columns:
- slots (uint16 indices into `players.parquet`):
  - `cpt`, `u1`, `u2`, `u3`, `u4`, `u5`
- computed:
  - `salary_used` (int32)
  - `salary_left` (int32)
  - `proj_points` (float32) — CPT treated as 1.5x
  - `stack_code` (uint8):
    - `0` = `3-3`
    - `1` = `4-2`
    - `2` = `5-1`

Invariants:
- No duplicates within a lineup (`cpt` not in `u1..u5`, and `u1..u5` all distinct).
- Salary cap respected: `salary_used <= 50000` (unless config overrides salary cap).
- NBA showdown stack legality: no `6-0` team stacks.

### 5b.2b Enriched lineups table (dup-model features)
**Name:** `lineups_enriched.parquet`

This is the same row set as `lineups.parquet` with additional per-lineup features used by the
duplication/share model.

Columns:
- base (same meaning as `lineups.parquet`):
  - `cpt`, `u1`, `u2`, `u3`, `u4`, `u5`
  - `salary_used`, `salary_left`, `proj_points`, `stack_code`
- added:
  - `own_score_logprod` (float) — sum of `log(own)` across all 6 players; CPT treated same as UTIL
  - `own_max_log` (float) — max `log(own)` across the 6 players
  - `own_min_log` (float) — min `log(own)` across the 6 players
  - `avg_corr` (float) — average pairwise Pearson correlation across the 6 players (denominator 15)
  - `cpt_archetype` (string; see `SegmentDefinitions.captain_tiers`)
  - `salary_left_bin` (string; bins: `0_200`, `200_500`, `500_1000`, `1000_2000`, `2000_4000`, `4000_8000`, `8000_plus`)
  - `pct_proj_gap_to_optimal` (float) — \((optimal - proj_points) / optimal\)
  - `pct_proj_gap_to_optimal_bin` (string; bins: `0_0.01`, `0.01_0.02`, `0.02_0.04`, `0.04_0.07`, `0.07_0.15`, `0.15_0.30`, `0.30_plus`)

Note on storage: in the parquet written by Pipeline B, these categorical columns are typically **dictionary-encoded**
for size (indices are signed ints for pandas compatibility).

## 5c) Softmax lineup share model artifacts (output of Pipeline A, optional)

### 5c.1 Model parameters
**Name:** `share_models/<gpp_category>/theta.json`

Minimum keys:
- `schema_version` (int)
- `gpp_category` (string)
- `sport` (string)
- `feature_schema` (object):
  - `intercept` (bool)
  - `continuous` (list[string])
  - `cpt_archetype_levels` (list[string])
  - `stack_pattern_levels` (list[string])
  - `salary_left_bin_levels` (list[string])
  - `pct_gap_bin_levels` (list[string])
  - `param_names` (list[string])
- `theta` (object): mapping from `param_name` → coefficient (float)

### 5c.2 Fit metrics
**Name:** `share_models/<gpp_category>/fit_metrics.json`

Minimum keys:
- `schema_version` (int)
- `gpp_category` (string)
- `sport` (string)
- `optimizer` (object): method, success, status, message, iterations
- `fit_config` (object): lambda, max_iter, val split
- `data` (object): train/val slate ids, missing-universe slates
- `metrics` (object): train/val NLL and per-entry versions

### 5c.3 Diagnostics (optional)
**Name:** `share_models/<gpp_category>/diagnostics/val_marginals.json`

Contains predicted vs actual marginal distributions on held-out validation slates for:
- `salary_left_bin`
- `pct_proj_gap_to_optimal_bin`
- `stack_pattern`
- `cpt_archetype`

### 5b.3 Metadata
**Name:** `metadata.json`

Minimum keys:
- `slate_id`, `sport`, `created_at_utc`
- `salary_cap`
- `num_players`, `num_lineups`
- `team_mapping` (stable mapping of team string → 0/1 used for legality checks)
- `schema` (column → dtype)
- `stack_code_map`
- `timings` (kernel timing metrics)

## 6) Run manifests (required for every CLI run)
**Name:** `run_manifest.json`
Minimum keys:
- `run_id`
- `pipeline`
- `started_at`, `finished_at`
- `git_sha` (if available)
- `config` (full resolved config)
- `inputs` (paths + checksums)
- `outputs` (paths + checksums)
- `row_counts_by_step`
- `warnings` (structured list)


## 7) Step manifests (required for every transformation)
**Name:** `step_manifest.json` (one per step folder)

Minimum keys:
- `run_id`
- `pipeline`
- `step_name`
- `started_at`, `finished_at`, `duration_s`
- `inputs`: list of `{path, checksum_sha256, logical_name}`
- `outputs`: list of `{path, checksum_sha256, logical_name}`
- `row_count_in`, `row_count_out`
- `schema_fingerprint` (hash of canonicalized column names + dtypes)
- `data_fingerprint` (hash of canonicalized preview rows; used for quick regression checks)
- `metrics`: dict (step-specific; e.g., parse_success_rate, unmatched_names, dup_rate)
- `warnings`: list of structured warnings `{code, message, sample_rows?}`
- `errors`: list (only if step handled/continued)

Sidecar files expected in the same folder:
- `preview.csv`
- `schema.json`

## 8) DKEntries filled output (Pipeline B output)
**Name:** `DKEntries_filled.csv`

This is the DKEntries CSV after assignment of contest-specific lineups.

- Output rules:
  - Row count is identical to the input DKEntries CSV.
  - All non-slot columns are preserved (passthrough).
  - Slot columns (`cpt`, `util1`..`util5`) are filled for every row.
  - Within each `contest_id`, assigned lineups are **unique** up to the number of entries for that
    contest (top-X selection where \(X=\#\) entries for that contest).
