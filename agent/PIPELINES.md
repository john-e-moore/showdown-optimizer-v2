# Pipelines

This doc defines the *canonical* steps and the required artifacts written at each step.
Step names and ordering must remain stable; if you change them, update this document and `DATA_CONTRACTS.md`.

## Shared rules (all pipelines)
- Each step writes a `steps/<NN_step_name>/` folder with:
  - `step_manifest.json`
  - `preview.csv` (<=200 rows)
  - `schema.json`
  - optional persisted outputs (`.parquet`) when `persist_step_outputs=true`
- All randomness is seeded from config and recorded in manifests.
- Row counts and key metrics must be recorded per step.

---

## Pipeline A: Training / Target Distributions

**Purpose:** Learn what real fields look like for a contest segment
(e.g., NBA Showdown, MME 3k–10k entrants).

### Inputs
- Sabersim projections file(s) (raw)
- DK contest standings/export CSV(s) (raw)

### Steps
00. **ingest**
- Read raw files, normalize headers, store input checksums.
- Artifacts: input inventory table (files, sizes, checksums).

01. **parse_projections**
- Convert Sabersim raw to canonical `players_flex` table (drop CPT rows, normalize names/teams).
- Metrics: num_players, dropped_zero_proj, name_normalization_changes.

02. **parse_contest_entries**
- Parse DK `Lineup` strings into slots + canonical `lineup_hash`.
- Metrics: parse_success_rate, invalid_lineup_rows.

03. **join_and_enrich**
- Join entries ↔ players (salary/proj/team), compute:
  `salary_used`, `salary_left`, `proj_points`, `stack_pattern`, `heavy_team`, `cpt_archetype`, `dup_count`.
- Metrics: unmatched_player_names, join_coverage.

04. **optimal_and_gap**
- Compute `optimal_proj_points` for the slate (under DK rules) and `proj_gap_to_optimal` per entry.
- Metrics: optimal_value, compute_time_s.

05. **fit_target_distributions**
- Fit and write versioned distributions:
  - salary_left bins
  - proj_gap bins
  - stack pattern frequencies
  - captain archetype frequencies
  - duplication histogram (dup_count distribution)
- Artifacts: `target_distributions.json` + validation summary (mean/p50/p90, totals sum checks).

06. **fit_softmax_lineup_share** (optional)
- Fit the multinomial softmax lineup share model (discrete-choice) per `gpp_category`, using:
  - `entries_enriched.parquet` (observed contest lineup counts + lineup features)
  - precomputed `lineups_enriched.parquet` universes under `TrainingConfig.universe_root`
- Outputs under the run dir:
  - `share_models/<gpp_category>/theta.json`
  - `share_models/<gpp_category>/fit_metrics.json`
  - optional diagnostics: `share_models/<gpp_category>/diagnostics/val_marginals.json`

### Outputs
- `entries_enriched.parquet`
- `target_distributions/*.json` (versioned by segment and date range)

---

## Pipeline B: Contest Execution (Build + Fill DKEntries)

**Purpose:** Given projections + correlations + a DKEntries CSV, generate the slate lineup universe,
compute lineup shares from a trained softmax lineup-share model (`theta.json`), simulate each contest
field (one contest or all contests in DKEntries), prune to a reasonable subset, grade lineups (ROI +
top-% finish rates), and fill DKEntries with the best contest-specific lineups.

### Inputs
- Sabersim projections CSV (raw)
- Player correlation matrix (for `avg_corr` features and correlated outcome simulation)
- DKEntries CSV (your entries to fill; may contain one or multiple contests)
- Softmax lineup-share model output:
  - `share_models/<gpp_category>/theta.json`

**Note:** Pipeline B does **not** use target distributions to construct fields (for now). Fields are
sampled directly from softmax-implied lineup shares.

### Steps
00. **ingest**
- Read inputs, normalize headers, record checksums.

01. **parse_projections**
- Build canonical `players_flex`.

02. **enumerate_lineup_universe**
- Enumerate all legal showdown lineups for the slate (CPT + 5 UTIL) under DK rules.
- Outputs:
  - `players.parquet` (the indexed player table used for enumeration)
  - `lineups.parquet` (the full lineup universe, stored as slot indices)
  - `metadata.json` (schema, team mapping, counts, timings)
- Metrics: num_players, num_lineups, stack distribution, kernel runtimes.

03. **enrich_lineup_universe_features**
- Compute per-lineup features required by the share model and simulations, e.g.:
  `own_score_logprod`, `own_max_log`, `own_min_log`, `avg_corr`, `cpt_archetype`,
  `salary_left_bin`, `pct_proj_gap_to_optimal`, `pct_proj_gap_to_optimal_bin`.
- Outputs:
  - `lineups_enriched.parquet`
- Metrics: `optimal_proj_points`, correlation matrix coverage.

04. **compute_softmax_shares**
- Load `theta.json`; compute per-lineup utility \(u(L)\) and softmax shares \(p(L)\) over the
  slate universe (stable log-sum-exp).
- Outputs:
  - `lineup_utilities.parquet` (u, logp, p; keyed by lineup id/hash)
- Metrics: share concentration (entropy), top-k cumulative mass.

05. **simulate_contests_from_dkentries**
- For one contest (if specified) or for **each contest present in DKEntries**:
  - Read contest parameters (N entries, entry fee, payout table)
  - **Sample the contest field** by drawing N lineups with replacement from \(p(L)\)
  - **Prune** the lineup universe for scoring (e.g., mass threshold; always force-include user
    candidate lineups if scoring a provided set)
- Outputs (per contest):
  - `field_sample.parquet` (entry-level or `{lineup_id -> dup_count}`)
  - `pruned_universe.parquet`
- Metrics: implied duplication histogram, prune size, cumulative mass retained.

06. **grade_lineups**
- Run correlated outcome simulations; compute contest-aware winnings with payout splitting across
  duplicates; produce:
  - ROI
  - top-% finish rates (e.g., 0.1%, 1%, 5%, 20%)
- Outputs (per contest):
  - `lineup_grades.parquet`
- Metrics: runtime, sim count, stability diagnostics.

07. **assign_best_lineups_to_entries**
- For each contest, select **unique** top-X ROI lineups, where X is the number of DKEntries for
  that contest, and assign them to the corresponding entries.
- Outputs:
  - `DKEntries_filled.csv`
  - exposure/summary tables

08. **write_outputs**
- Write run + step manifests and artifacts for reproducibility.

### Outputs
- `DKEntries_filled.csv`
- `lineups.parquet`, `lineups_enriched.parquet`, `lineup_utilities.parquet`
- per-contest artifacts (one folder per contest): `field_sample.parquet`, `pruned_universe.parquet`,
  `lineup_grades.parquet`
