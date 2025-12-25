---
name: Fill all DKEntries rows
overview: Fix DKEntries parsing so entry rows that have extra appended player-list columns are preserved (by truncating to header width) instead of being dropped, restoring correct entry counts and filling every entry in Pipeline B.
todos:
  - id: dkentries-fallback-truncate-extra-cols
    content: Change read_dkentries() ParserError fallback to truncate rows with extra columns (and pad short rows) to header width, instead of skipping mismatched rows.
    status: completed
  - id: dkentries-test-appended-cols
    content: Add regression test covering DKEntries where entry rows contain appended player-list columns; assert all entries are retained.
    status: completed
    dependencies:
      - dkentries-fallback-truncate-extra-cols
  - id: validate-pipelineb-35-entries
    content: Rerun Pipeline B and verify step 05 row_count_in and DKEntries_filled.csv row count match all entries.
    status: completed
    dependencies:
      - dkentries-test-appended-cols
---

# Fix DKEntries parsing when entry rows contain appended player-list columns

## Goal

Ensure Pipeline B fills **every entry row** in `data/inputs/DKEntries.csv` (35 in your case) even when DraftKings appends player-list columns **onto the same CSV rows** as the entries section.

## Root cause (confirmed)

In your `data/inputs/DKEntries.csv`, rows 8–36 still have valid `Entry ID` / `Contest ID`, but also include extra columns (e.g. `Position,Name + ID,...`) appended after the blank column. This makes the row width larger than the header, triggers `pandas` `ParserError`, and our current fallback **skips** these mismatched-width rows entirely—dropping most real entries and leaving only 6.

## Implementation changes

### 1) Make ParserError fallback **truncate**, not skip

Update [`src/dfs_opt/io/dkentries.py`](/home/john/showdown-optimizer-v2/src/dfs_opt/io/dkentries.py) in `read_dkentries()`:

- In the `ParserError` fallback loop, instead of:
- `if i > 1 and len(row) != header_fields: continue`
- Do:
- If `len(row) > header_fields`: **truncate**: `row = row[:header_fields]` (preserves entry rows that have extra appended columns)
- If `len(row) < header_fields`: **pad** with empty strings to `header_fields` (handles occasional short rows)
- Always write the normalized row.

This keeps entry rows intact while ensuring the trailing player-list-only rows (with blank `Entry ID`) still get filtered out by the existing `entry_mask`.

### 2) Add/extend regression tests for the “appended columns on entry rows” format

Update [`tests/test_dkentries_reader.py`](/home/john/showdown-optimizer-v2/tests/test_dkentries_reader.py):

- Add a test that builds a DKEntries CSV where:
- First K rows are normal-width entries
- Next rows are **entries with extra trailing player-list columns appended**
- Final rows are **player-list-only** rows with empty `Entry ID`
- Assert `read_dkentries()` returns **all entry rows** (e.g. 35) and correct `contest_id` distribution.

(Keep the existing >200-line appendix test; this new test covers the real-world structure in your attached file.)

### 3) Validate end-to-end

- Run unit tests (via venv):
- `.venv/bin/python -m pytest -q tests/test_dkentries_reader.py`
- Re-run Pipeline B:
- `.venv/bin/python scripts/run_pipeline_b.py`
- Confirm:
- Step 05 log shows `num_entry_rows=35`
- `run_manifest.json` step `05_simulate_contests_from_dkentries.row_count_in == 35`
- `artifacts/.../DKEntries_filled.csv` has 35 filled entry rows

## Files touched

- [`src/dfs_opt/io/dkentries.py`](/home/john/showdown-optimizer-v2/src/dfs_opt/io/dkentries.py)
- [`tests/test_dkentries_reader.py`](/home/john/showdown-optimizer-v2/tests/test_dkentries_reader.py)