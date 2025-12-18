# Pipelines

## Pipeline A: Training / Target Distributions
**Purpose:** Learn what real fields look like for a contest “segment”
(e.g., NBA Showdown, MME 3k–10k entrants).

### Inputs
- Sabersim projections file(s)
- DK contest standings/export CSV(s)

### Steps
1. **Ingest**
   - Load raw files
   - Normalize column names
   - Save input checksums to manifest
2. **Parse**
   - Parse DK `lineup_str` into `ParsedLineup`
   - Parse Sabersim → canonical player table (FLEX row per player)
3. **Feature engineering**
   - salary_used/left
   - projection (lineup)
   - stack pattern + heavy team
   - captain archetype
   - optimal projection + gap-to-optimal
   - dup_count per unique lineup
4. **Quality checks**
   - parse success rate >= threshold
   - salary_used <= 50000
   - missing player joins flagged
5. **Fit distributions**
   - choose stable bins and store edges
   - compute rates and histograms
6. **Persist artifacts**
   - `entries_enriched.parquet`
   - distribution JSONs
   - summary report (optional)

### Idempotency requirements
- All outputs written to `data/outputs/training/<run_id>/`
- If `--rerun` is false, refuse to overwrite existing run_id
- Deterministic ordering and stable hashing

---

## Pipeline B: Contest Execution (Optimizer + Simulation → DKEntries)
**Purpose:** For a given upcoming slate, generate a diversified set of lineups
that accounts for field realism and duplication.

### Inputs
- Sabersim projections (upcoming slate)
- Player correlation matrix (optional but recommended)
- DKEntries CSV (template)
- Target distributions (from Pipeline A) for the matching segment
- Ownership projections (if used)

### Steps
1. **Ingest & validate**
2. **Generate candidate pool**
   - 50k–500k plausible lineups
   - multiple “brains” (projection max, stack rules, temperature)
3. **Base weighting**
   - q(L) ∝ exp(τ * proj(L) - λ * salary_left - penalties)
4. **Reweight to targets**
   - ownership CPT/FLEX (optional)
   - salary_left bins
   - stack patterns
   - cpt archetypes
   - proj-gap bins
5. **Diversify selection**
   - sample with replacement to allow duplicates
   - then enforce *your* entry diversification constraints (max exposure, etc.)
6. **Simulation**
   - simulate player outcomes using correlation
   - compute lineup EV / ROI with payout splitting
7. **Fill DKEntries**
   - produce final `DKEntries_filled.csv`
   - include metadata columns (run_id, version, timestamp)

### Idempotency requirements
- Seeded randomness (seed in config + logged)
- Candidate pool persisted (optional flag) for replay/debug
- All derived outputs include checksums in manifest
