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
  - `salary_left_bin` (string; bins: `0_200`, `200_500`, `500_1000`, `1000_2000`, `2000_plus`)
  - `proj_points` (float) — computed from projections
  - `optimal_proj_points` (float)
  - `proj_gap_to_optimal` (float)
  - `pct_proj_gap_to_optimal` (float)
  - `pct_proj_gap_to_optimal_bin` (string; bins: `0_0.01`, `0.01_0.02`, `0.02_0.04`, `0.04_0.07`, `0.07_plus`)
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
