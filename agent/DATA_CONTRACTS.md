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
  - `proj_points` (float) — computed from projections
  - `optimal_proj_points` (float)
  - `proj_gap_to_optimal` (float)
  - `stack_pattern` (string like `4-2`)
  - `heavy_team` (string)
  - `cpt_team` (string)
  - `cpt_archetype` (string: `stud_1_2`, `stud_3_5`, `mid_6_10`, `value_11_plus`)
  - `dup_count` (int) — count of identical `lineup_hash` in contest

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
