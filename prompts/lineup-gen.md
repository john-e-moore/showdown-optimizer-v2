# Fast Enumeration of DK Showdown Lineups (NBA) — Numba + NumPy + Parquet

This doc outlines the **fastest practical** method (Python) to enumerate *all valid* DraftKings **Showdown Captain Mode** lineups for an NBA slate on commodity hardware (e.g., 16GB RAM, ~20 cores).

We will:
1. Build compact **NumPy arrays** for player attributes.
2. Use **Numba JIT** to enumerate lineups with compiled nested loops.
3. Enforce DK rules including **stack legality** (NBA showdown must be **3–3, 4–2, or 5–1**, i.e., **no 6–0**).
4. Persist the enumerated universe to a **Parquet** file.
5. Write a **metadata JSON** describing the dataset and its schema.

---

## 1) Inputs

### 1.1 Players table (per slate)
You need a “FLEX-level player table” with, at minimum:

- `player_id` (or stable integer index)
- `name`
- `team` (for NBA showdown, exactly **two teams** in the slate)
- `salary_flex` (int)
- `proj_flex` (float)
- `own_flex` (float in [0,1], optional for enumeration but likely needed downstream)
- `own_cpt` (float in [0,1], optional)
- `active` flag (e.g., proj > 0 and/or own threshold)

**Note:** If your projection source includes CPT rows, drop them; use FLEX rows only. CPT salary/proj is handled as 1.5× at runtime.

### 1.2 Correlation matrix (optional at enumeration time)
If you plan to compute `corr_avg` during enumeration, provide:
- `corr[i, j]` Pearson correlation between FLEX players `i` and `j` (float32)

If you only need to enumerate slots + salary/proj, you can compute correlation features later.

---

## 2) Key constraints (DK NBA Showdown Captain Mode)

For each lineup:
- Slots: `CPT + 5 UTIL`
- No duplicate players across slots
- Salary cap: `salary_used <= 50000`
- Salary computation:
  - `salary_used = 1.5 * salary_flex[CPT] + sum(salary_flex[UTILs])`
- Projection computation (for later):
  - `proj = 1.5 * proj_flex[CPT] + sum(proj_flex[UTILs])`

### 2.1 Stack legality (NBA showdown)
For NBA showdown (two teams), a lineup’s team counts across the **6 rostered players** must be one of:
- `3-3`
- `4-2`
- `5-1`

**6-0 is NOT legal and must be rejected.**

Implementation: represent team as `0/1` per player index; let:
- `t = sum(team[player] for all 6 players)` where team is in {0,1}
Then `t` must be in `{1,2,3,4,5}` but **not** `0` or `6`.
Equivalently reject if `t == 0 or t == 6`.

---

## 3) Performance approach

### 3.1 Why Numba + nested loops
Python-level `itertools.combinations` is too slow for millions of lineups.
Numba compiles loops to machine code; nested loops are extremely fast and predictable.

### 3.2 Parallelization strategy
Parallelize over the Captain slot:
- Outer loop over `cpt_idx` uses `numba.prange`
- Each CPT independently enumerates `C(n-1, 5)` UTIL combinations
- This maps well to multi-core CPUs

### 3.3 Memory strategy (critical)
Avoid Python list appends and per-row DataFrame writes.

Use one of:
- **Two-pass strategy** (recommended):  
  Pass 1 counts valid lineups per CPT → prefix sums → allocate exact arrays → Pass 2 fills arrays.
- **Chunked buffers** (acceptable):  
  Allocate chunks (e.g., 1–5 million rows), fill, flush to Parquet, repeat.

Two-pass is fastest and simplest for correctness with parallel CPT counting.

---

## 4) Core enumeration algorithm (high level)

Let `n` be the number of eligible players (FLEX-level), indexed `0..n-1`.

For each CPT index `c`:
1. Iterate all 5-combinations of UTIL indices among remaining players:
   - `i < j < k < l < m`, all distinct and not equal to `c`
2. Compute `salary_used` with 1.5× CPT salary.
3. If `salary_used > 50000`, skip.
4. Compute team sum across 6 players; if `t==0 or t==6`, skip (**reject 6-0**).
5. Emit lineup record:
   - slot indices: `c, i, j, k, l, m`
   - optional: `salary_used`, `salary_left`, `proj`, etc.

### 4.1 Suggested emitted columns (v1)
At minimum (to support downstream modeling/sampling):
- `cpt` (uint16)
- `u1..u5` (uint16)
- `salary_used` (int32 or float32)
- `salary_left` (int16/int32)
- `team_count_team1` (uint8) or `stack_pattern` code (uint8)
- `proj` (float32) (optional but usually worth doing here)

You can also compute:
- ownership features (`own_score_logprod`, `own_max_log`, `own_min_log`)
- correlation (`corr_avg`)
…but if you want absolute maximum throughput, do slots + salary/proj first, then enrich later in vectorized passes.

---

## 5) Implementation details for the agent

### 5.1 Data preparation (Python)
- Convert player attributes to contiguous NumPy arrays:
  - `salary = np.asarray(..., dtype=np.int32)`
  - `proj = np.asarray(..., dtype=np.float32)`
  - `team = np.asarray(..., dtype=np.uint8)`  # 0/1
  - optional: `own_flex`, `own_cpt` as float32
- For team encoding:
  - pick one team as 0, the other as 1
  - store mapping in metadata

### 5.2 Numba kernels

#### Kernel A: count valid lineups per CPT
Signature idea:
- inputs: `salary, team, n`
- output: `counts_per_cpt` (int64 length n)
- loops: for each `c` compute count of (i,j,k,l,m) passing salary + stack rules

#### Kernel B: fill output arrays
Use prefix sums to compute offsets:
- `offsets[c]` tells where CPT `c` should write its lineups.

Then fill:
- `cpt_arr[offset + r] = c`
- `u1_arr[...] = i`, etc.
- `salary_used_arr[...] = computed salary_used`
- etc.

Use `parallel=True` and `prange` over CPT in both kernels.

### 5.3 Stack legality check (fast)
Compute team sum:
- `t = team[c] + team[i] + team[j] + team[k] + team[l] + team[m]`
Reject if:
- `t == 0 or t == 6`

If you want to encode stack pattern:
- `t` gives number of team==1 players
- stack pattern is:
  - if `t in {1,5}` => `5-1`
  - if `t in {2,4}` => `4-2`
  - if `t == 3` => `3-3`

### 5.4 Types to minimize memory
- Slot indices: `uint16` (safe for n < 65535)
- Salary/proj: `int32`, `float32`
- Stack code / team counts: `uint8`

---

## 6) Output: Parquet + metadata JSON

### 6.1 Parquet file
Write the full lineup universe to:
- `artifacts/lineup_universe/{slate_id}/lineups.parquet`

Recommended writer:
- `pyarrow.parquet` (fast, standard)
- optionally use `polars.DataFrame.write_parquet`

Compression:
- `zstd` or `snappy` (zstd compresses better; snappy is faster)

### 6.2 Metadata JSON
Write:
- `artifacts/lineup_universe/{slate_id}/metadata.json`

Include at minimum:

- `slate_id`, `site`, `sport`, `format`
- `created_at_utc`
- `num_players` (n)
- `num_lineups`
- `salary_cap` (50000)
- `stack_rules`:
  - `allowed`: ["3-3", "4-2", "5-1"]
  - `disallowed`: ["6-0"]
- `team_mapping`:
  - `{ "TEAM_A": 0, "TEAM_B": 1 }`
- `columns` with names and dtypes
- `feature_definitions` (salary_used, salary_left, stack_code mapping)
- `filters_applied` for player eligibility (proj>0, ownership threshold, etc.)
- `input_sources` (paths + hashes if available)

Example schema section:

```json
{
  "columns": {
    "cpt": "uint16",
    "u1": "uint16",
    "u2": "uint16",
    "u3": "uint16",
    "u4": "uint16",
    "u5": "uint16",
    "salary_used": "int32",
    "salary_left": "int32",
    "proj": "float32",
    "stack_code": "uint8"
  },
  "stack_code_map": {
    "0": "3-3",
    "1": "4-2",
    "2": "5-1"
  }
}
```

---

## 7) Validation checks (required)

After generation, assert:
1. `salary_used <= 50000` for all rows
2. No duplicates within a lineup (CPT not in UTILs; UTILs unique)
3. Stack legality:
   - no lineup has all 6 players from the same team
   - stack_pattern in {3-3,4-2,5-1}
4. Row count equals `sum(counts_per_cpt)` from the count kernel
5. Spot-check a few CPTs: compute expected combinatorics and ensure filtering makes sense.

Write a small `validation_report.json` artifact summarizing:
- min/max salary_used
- distribution of stack patterns
- num_lineups
- elapsed time per kernel

---

## 8) Suggested file/module layout

- `src/lineups/enumerate_showdown.py`
  - public API: `enumerate_showdown_universe(players_df, out_dir, slate_id, ...)`
- `src/lineups/numba_kernels.py`
  - `count_valid_per_cpt(...)`
  - `fill_lineups(...)`
- `src/io/parquet_writer.py`
- `src/io/metadata_writer.py`
- `tests/test_enumeration_small_slate.py`
  - tiny slate (n=8–10) with known expected counts

---

## 9) Notes / future upgrades (optional)
- Add chunked writing if n is large (n>32) and you emit many features during enumeration.
- Compute only slots+salary in enumeration, then enrich features (ownership/corr) in a second pass for modularity.
- Add `corr_avg` using the 15 pair indices if correlation matrix is available (still cheap).
