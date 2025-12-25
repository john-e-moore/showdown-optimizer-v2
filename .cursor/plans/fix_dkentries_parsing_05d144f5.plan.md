---
name: Fix DKEntries parsing
overview: Make `read_dkentries()` ignore DK’s appended player-list section (mixed-width rows / columns beyond A–J) without truncating entry rows, so Pipeline B counts and fills the correct number of entries.
todos:
  - id: dkentries-prescan-window
    content: Make the ParserError fallback robust for large MME files by removing the “prescan must see a mismatch within first ~200 lines” gating (fallback should use header width from line 1 and skip mismatched-width rows anywhere in the file).
    status: completed
  - id: dkentries-fallback-skip-mismatches
    content: Modify `read_dkentries()` ParserError fallback to keep scanning the whole file and write only rows that match the header width (skip mismatched-width player-list rows), rather than breaking at the first mismatch.
    status: completed
  - id: dkentries-regression-test
    content: Add a unit test that builds a multi-section DKEntries CSV (entries + mismatched-width player-list rows) and asserts `read_dkentries()` returns all entry rows (including a variant where the appendix begins after >200 lines).
    status: completed
  - id: optional-step05-logging
    content: (Optional) Add a small debug log/metric in step 05 of `run_contest_pipeline` showing detected entry-row counts by contest_id.
    status: completed
---

# Fix DKEntries multi-section parsing

## Summary

Update the DKEntries reader to **retain all entry rows** while **skipping the appended player-list section** that begins later in the CSV and causes mixed-width rows (often starting around column L). This ensures Pipeline B’s `num_user_entries` reflects the true number of DK entry rows.

## What’s happening today

- Pipeline B counts “how many entries to fill” as `len(dkentries.entries)` in step 05 (`run_contest_pipeline`).
- `read_dkentries()` currently falls back on `ParserError` by **copying rows until the first width mismatch and then `break`**; this can truncate valid entry rows in real DK exports.
- DKEntries exports can be **multi-section**:
- An entries upload template section with a fixed header width.
- A trailing “player list” appendix with a different width (extra columns).
- Important edge case: for **large MME files**, the appendix might start **after a few hundred entry rows**. If the code only “detects mismatches” in an early window, the ParserError fallback may not activate even though it is needed.

## Implementation steps

- **Change fallback parsing in** [`src/dfs_opt/io/dkentries.py`](/home/john/showdown-optimizer-v2/src/dfs_opt/io/dkentries.py)
- In the `ParserError` fallback loop, replace the current “stop at first mismatch” behavior with:
- Keep writing the header row.
- For subsequent rows:
- If `len(row) == header_fields`, write it.
- Otherwise, **skip it and continue** (do not break).
- Keep the rest of the function intact so we still:
- Preserve the full entry-section columns (including `Instructions`).
- Identify entry rows via non-empty `Entry ID`.
- Maintain `_row_idx` for stable write-back.
- **Follow-up (robustness for large MME)**:
- Ensure the fallback does **not** depend on a prescan window having already observed a mismatch.
- The fallback can safely derive `header_fields` from line 1 and then skip mismatched-width rows anywhere in the file.

- **Add a regression test** in `tests/` (new file, e.g. `tests/test_dkentries_reader.py`)
- Build a small synthetic CSV string that mimics DK’s structure:
- Header with repeated `UTIL` columns.
- 28 entry rows with valid `Entry ID`.
- A player-list header row with a different width (extra columns) + a couple player rows.
- Assert:
- `len(read_dkentries(...).entries) == 28`
- All returned `contest_id` values match expected.
- No exception is thrown.
- Add a second test case (or parameterize) where:
- There are **>200** entry rows (e.g. 250), and the player-list appendix begins after that,
- and `read_dkentries()` still returns **all** entry rows.

- **(Optional but recommended) Add a quick debug metric** in [`src/dfs_opt/pipelines/contest.py`](/home/john/showdown-optimizer-v2/src/dfs_opt/pipelines/contest.py)
- Log `len(dkentries.raw)` / `len(dkentries.entries)` and the detected `contest_id` counts at step 05 start. This helps immediately spot future DK export format drift.

## Files to touch

- [`src/dfs_opt/io/dkentries.py`](/home/john/showdown-optimizer-v2/src/dfs_opt/io/dkentries.py)
- `tests/test_dkentries_reader.py` (new)
- (Optional) [`src/dfs_opt/pipelines/contest.py`](/home/john/showdown-optimizer-v2/src/dfs_opt/pipelines/contest.py)

## Rollout / validation

- Run unit tests:
- `pytest -q tests/test_dkentries_reader.py`
- Re-run Pipeline B on the same `data/inputs/DKEntries.csv`:
- `/home/john/showdown-optimizer-v2/.venv/bin/python scripts/run_pipeline_b.py`
- Confirm:
- `run_manifest.json` shows step 05 `row_count_in` equals the true number of DK entry rows.
- The step 05 log line prints the expected `num_entry_rows` and `contest_id_counts`.
- `DKEntries_filled.csv` contains the same number of entry rows as the input DKEntries template section.

## Implementation todos

- `dkentries-prescan-window`: Ensure ParserError fallback doesn’t depend on a small prescan window; must work when appendix begins after >200 rows.
- `dkentries-fallback-skip-mismatches`: Update `read_dkentries()` fallback to skip mismatched-width rows instead of breaking.
- `dkentries-regression-test`: Add/extend tests covering “entries + player-list appendix” including a large-MME (>200 rows) variant.
- `optional-step05-logging`: Add a small log/metric in step 05 to print detected entry counts by contest.