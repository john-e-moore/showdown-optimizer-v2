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

## 5b) Lineup Universe (output of contest pipeline, optional early step)
**Name:** `lineups.parquet` + `players.parquet` + `metadata.json`

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
