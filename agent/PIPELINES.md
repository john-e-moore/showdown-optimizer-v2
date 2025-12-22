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

### Outputs
- `entries_enriched.parquet`
- `target_distributions/*.json` (versioned by segment and date range)

---

## Pipeline B: Contest Execution (Build + Fill DKEntries)

**Purpose:** Given projections + correlations + a DKEntries template, generate diversified lineups,
simulate outcomes, and fill entries with lineups.

### Inputs
- Sabersim projections (raw)
- Player correlation matrix (required if computing lineup-universe features like `avg_corr`; also required for sims)
- DKEntries template CSV (your entries to fill)
- (Optional) learned target distributions from Pipeline A

### Steps
00. **ingest**
- Read inputs, normalize headers, record checksums.

01. **parse_projections**
- Build canonical `players_flex`.

02. **enumerate_lineup_universe** (optional early step)
- Enumerate all legal showdown lineups for the slate (CPT + 5 UTIL) under DK rules.
- Outputs:
  - `players.parquet` (the indexed player table used for enumeration)
  - `lineups.parquet` (the full lineup universe, stored as slot indices)
  - `lineups_enriched.parquet` (optional; if the run computes dup-model features)
  - `metadata.json` (schema, team mapping, counts, timings)
- Metrics: num_players, num_lineups, stack distribution, kernel runtimes.

03. **enrich_lineup_universe_features** (optional early step)
- Compute per-lineup features used by the duplication/share model:
  `own_score_logprod`, `own_max_log`, `own_min_log`, `avg_corr`, `cpt_archetype`,
  `salary_left_bin`, `pct_proj_gap_to_optimal`, `pct_proj_gap_to_optimal_bin`.
- Outputs:
  - `lineups_enriched.parquet`
- Metrics: `optimal_proj_points`, correlation matrix coverage.

04. **build_candidate_pool**
- Generate 50k–500k plausible lineups using optimizer “brains” (temperature/noise + constraints).
- Artifacts:
  - summary stats (salary_left histogram, stack pattern counts, captain archetype counts)
  - small sample of lineups (preview)

05. **base_weighting**
- Assign base weight `q(L)` (e.g., exp(tau*proj - lambda*salary_left - penalties)).
- Metrics: weight concentration (entropy), top-100 mass.

06. **reweight_to_targets**
- Rake weights to match:
  - CPT/FLEX ownership targets
  - salary_left bins, stack patterns, archetypes, gap bins (if provided)
- Metrics: constraint error per target, number of iterations, convergence status.

07. **sample_field**
- Sample N entries from reweighted distribution **with replacement** (duplicates allowed).
- Metrics: implied duplication histogram.

08. **simulate_and_score**
- Run correlated simulations → ROI/EV/top1% rates with payout splitting.
- Metrics: runtime, sim count, stability diagnostics.

09. **select_and_diversify_for_user_entries**
- Select lineups for user DKEntries with diversification constraints (exposure caps, etc.).
- Metrics: exposure summary table.

10. **write_outputs**
- Write `DKEntries_filled.csv` plus run manifest.

### Outputs
- `DKEntries_filled.csv`
- optional: `candidate_pool.parquet`, `field_sample.parquet`, `sim_results.parquet`
