---
name: Vectorize lineup grader
overview: Speed up Pipeline B’s `grade_lineups` by removing Python loops and reusing precomputations; optionally add an approximate rank method to avoid per-sim sorting when candidate pools are huge.
todos:
  - id: precompute-dup-and-payout
    content: In `contest_sim.grade_lineups_for_contest`, precompute candidate `block_size` (dup+1) once and build `payout_cumsum` for vectorized payout range sums.
    status: completed
  - id: vectorize-winnings-and-toprates
    content: Replace the per-sim Python candidate loop and `payout_for_block` calls with fully vectorized winnings + top-rate updates.
    status: completed
    dependencies:
      - precompute-dup-and-payout
  - id: batch-sims
    content: Add sim batching to avoid allocating all draws at once; keep API stable and ensure results remain statistically consistent.
    status: completed
    dependencies:
      - vectorize-winnings-and-toprates
  - id: optional-hist-rank
    content: If needed after benchmarking, add `rank_method` with a fast histogram-based approximate rank to remove per-sim sorting costs.
    status: completed
    dependencies:
      - batch-sims
  - id: regression-benchmark
    content: Add a small regression/benchmark harness comparing old vs new grader outputs (sanity) and runtime (timing) on a representative contest.
    status: completed
    dependencies:
      - vectorize-winnings-and-toprates
---

# Speed up `grade_lineups` (vectorized grader)

## Goals

- Make Pipeline B’s `06_grade_lineups` substantially faster for large candidate pools (100k+), enabling 20k–100k+ sims.
- Preserve **statistical equivalence** (same general ordering/ROI behavior), not exact per-seed identity.

## Key observation (current bottlenecks)

In [`/home/john/showdown-optimizer-v2/src/dfs_opt/simulation/contest_sim.py`](/home/john/showdown-optimizer-v2/src/dfs_opt/simulation/contest_sim.py), `grade_lineups_for_contest()` currently:

- Sorts field unique scores **every sim**.
- Builds a `lookup` dict **every sim** (unnecessary; dup counts don’t change by sim).
- Runs a Python `for i in range(len(cand_ids))` loop **every sim** calling [`payout_for_block()`](/home/john/showdown-optimizer-v2/src/dfs_opt/simulation/payouts.py).

That Python loop is the dominant cost once `num_sims` and `n_candidates` get large.

## Implementation approach

### 1) Precompute everything that does not depend on the sim draw

Update [`/home/john/showdown-optimizer-v2/src/dfs_opt/simulation/contest_sim.py`](/home/john/showdown-optimizer-v2/src/dfs_opt/simulation/contest_sim.py):

- **Candidate duplication block sizes**: compute `block_size_per_candidate = dup_in_field + 1` once.
- `dup_in_field` is determined solely by `field_counts`.
- Replace the per-sim `lookup = {lineup_id: dup_count}` dict.
- **Payout prefix sums**: in `grade_lineups_for_contest()`, build `payout_cumsum` once:
- `payout_cumsum[r] = sum(payout_table[0:r])` for fast range sums.
- This enables a fully vectorized payout calculation.

### 2) Vectorize payout + win accumulation (remove Python-per-candidate loop)

Still in `grade_lineups_for_contest()`:

- Keep the current (exact-ish) rank computation pattern:
- sort `field_scores` (with tie-break if desired)
- compute `cum_dups`
- compute `rank_start` via `searchsorted`
- Replace:
- `for i in range(len(cand_ids)):`
- `payout_for_block(...)`

With a vectorized computation:

- For each candidate, winnings is:
- `sum(payout_table[rank_start .. rank_start+block_size-1]) / block_size`
- Using prefix sums, for vectors `rank_start` and `block_size`:
- `lo = rank_start - 1`
- `hi = min(lo + block_size, len(payout_table))`
- `sum = payout_cumsum[hi] - payout_cumsum[lo]`
- `winnings = sum / block_size`
- Accumulate `win_sum += winnings`.
- Vectorize top-rate counters as boolean adds:
- `top_rates[pct] += (rank_start <= cutoff[pct])`.

This should be the biggest win and keeps semantics close to current.

### 3) Reduce memory + improve throughput via sim batching

Update `grade_lineups_for_contest()` to simulate in chunks instead of `draws = simulate_correlated_normals(..., num_sims)` allocating all at once:

- Add a `batch_sims` parameter (default like 512 or 1024).
- In a loop over batches:
- generate draws for the batch using existing outcome spec in [`/home/john/showdown-optimizer-v2/src/dfs_opt/simulation/outcomes.py`](/home/john/showdown-optimizer-v2/src/dfs_opt/simulation/outcomes.py)
- process each sim within the batch.

This won’t remove per-sim sorting, but it improves cache behavior and prevents huge allocations for large `num_sims`.

### 4) If still too slow: add an optional approximate “fast rank” method (no per-sim sort)

If profiling shows per-sim `argsort/lexsort` dominates after vectorizing payouts:

- Add an optional `rank_method` argument to `grade_lineups_for_contest()`:
- `rank_method="sort"` (default; current behavior)
- `rank_method="hist"` (fast, approximate)
- `hist` method idea (statistical equivalence target):
- Choose fixed score bins (e.g. width 0.05 or 0.1 points; configurable).
- Per sim:
  - bin `field_scores` into histogram weighted by `field_dups` via `np.bincount`
  - compute cumulative “entries above score” from histogram prefix (descending)
  - map each `cand_score` to its bin and read `above_entries`
  - compute `rank_start = above_entries + 1`
- This replaces `sort + searchsorted` with `O(m + k + nbins)` operations.

We’ll keep `rank_method="sort"` as default to avoid surprises; you can flip it on only when you need 100k sims with very large candidate pools.

### 5) Add a small correctness/regression harness

Add a lightweight test or script (depending on your current test setup) to validate:

- Vectorized payouts match `payout_for_block()` for random `rank_start`/`block_size` samples.
- New grader outputs are numerically close to old grader on a small slate for a small `num_sims` (e.g. 200–1000) with a fixed seed.

Files likely involved:

- [`/home/john/showdown-optimizer-v2/src/dfs_opt/simulation/contest_sim.py`](/home/john/showdown-optimizer-v2/src/dfs_opt/simulation/contest_sim.py)
- [`/home/john/showdown-optimizer-v2/src/dfs_opt/simulation/payouts.py`](/home/john/showdown-optimizer-v2/src/dfs_opt/simulation/payouts.py) (may add a vectorized helper, optional)
- (Optional) a new `scripts/benchmark_grade_lineups.py` for timing comparisons.

## Rollout / measurement

- First land steps (1)–(3) and benchmark on a representative contest.
- If `num_sims=100k` is still too slow, implement step (4) and benchmark `rank_method=hist`.