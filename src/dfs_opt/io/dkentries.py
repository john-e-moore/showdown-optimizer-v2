from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import csv

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dfs_opt.parsing.names import norm_name


@dataclass(frozen=True)
class DkEntriesFile:
    """
    Parsed DKEntries upload template.

    We keep:
    - `raw`: the entries section as read from the CSV (rows with entry_id present)
    - `entries`: a normalized view with canonical columns + an `_row_idx` for stable updates
    - `slot_cols`: the concrete CSV *column indices* used for CPT/UTIL slots (so we can write back even
      when DK repeats column names like UTIL 5x)
    """

    raw: pd.DataFrame
    entries: pd.DataFrame
    slot_cols: Dict[str, int]  # canonical slot -> raw column index


def _normalize_columns(cols: Sequence[str]) -> List[str]:
    return [str(c).strip().lower().replace("\ufeff", "") for c in cols]


def _pick_first(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def read_dkentries(path: Path) -> DkEntriesFile:
    """
    Read a DKEntries CSV and return only the entry rows (not the trailing player list section).

    DK uses repeated column names (e.g., FLEX 5 times); pandas disambiguates to:
      FLEX, FLEX.1, FLEX.2, FLEX.3, FLEX.4
    """
    # DK often repeats column names (e.g., UTIL 5x). pandas.read_csv will mangle duplicates
    # into UTIL, UTIL.1, ... which then propagates into DKEntries_filled.csv.
    #
    # We instead parse with csv.reader and build a DataFrame while preserving the original
    # header fields exactly (including duplicates). We also normalize each row to the header
    # width to safely handle the trailing player-list appendix or appended columns.
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        rdr = csv.reader(f, delimiter=",", quotechar='"')
        header = next(rdr, None)
        if header is None:
            raise ValueError(f"{path}: empty file")
        header_fields = len(header)
        rows: List[List[str]] = []
        for row in rdr:
            if len(row) > header_fields:
                row = row[:header_fields]
            elif len(row) < header_fields:
                row = row + ([""] * (header_fields - len(row)))
            rows.append(row)

    raw = pd.DataFrame(rows, columns=header)

    # IMPORTANT: when column names contain duplicates (e.g. UTIL repeated 5x), selecting
    # columns by label can "explode" duplicates (each 'UTIL' label selects all UTIL cols).
    # Filter by *position* instead.
    keep_idxs = [
        i
        for i, c in enumerate(list(raw.columns))
        if str(c).strip() != "" and not str(c).startswith("Unnamed:")
    ]
    raw = raw.iloc[:, keep_idxs].copy()

    cols_norm = _normalize_columns(list(raw.columns))
    # Keep duplicates: assign a normalized column list (not a dict rename which would collapse dupes).
    df = raw.copy()
    df.columns = cols_norm

    entry_id_col = _pick_first(df, ["entry id", "entryid", "entry_id"])
    contest_id_col = _pick_first(df, ["contest id", "contestid", "contest_id"])
    if entry_id_col is None or contest_id_col is None:
        raise ValueError(f"{path}: missing required Entry ID / Contest ID columns; got {list(raw.columns)}")

    # Keep only entry rows (DK appends a player list section with empty Entry ID)
    entry_mask = df[entry_id_col].astype(str).str.strip() != ""
    entries_raw = raw.loc[entry_mask].copy().reset_index(drop=True)
    entries_norm = df.loc[entry_mask].copy().reset_index(drop=True)
    entries_norm["_row_idx"] = np.arange(len(entries_norm), dtype=np.int64)

    # Slot columns: CPT + 5 flex/util columns (varies by export).
    # Use *positions* so we can handle duplicate column names (e.g. UTIL repeated 5x).
    cols_norm_entries = list(entries_norm.columns)
    try:
        cpt_idx = next(i for i, c in enumerate(cols_norm_entries) if c in ("cpt", "captain"))
    except StopIteration as e:
        raise ValueError(f"{path}: missing captain slot column (CPT/Captain); got {list(raw.columns)}") from e

    flex_idxs = [i for i, c in enumerate(cols_norm_entries) if isinstance(c, str) and c.startswith("flex")]
    util_idxs = [i for i, c in enumerate(cols_norm_entries) if isinstance(c, str) and c.startswith("util")]
    slot_flex_idxs = flex_idxs if flex_idxs else util_idxs
    slot_flex_idxs = [i for i in slot_flex_idxs if cols_norm_entries[i] != "_row_idx"]
    slot_flex_idxs = slot_flex_idxs[:5]

    if len(slot_flex_idxs) < 5:
        got = [cols_norm_entries[i] for i in (flex_idxs if flex_idxs else util_idxs)]
        raise ValueError(f"{path}: expected 5 FLEX/UTIL columns; got {got}")

    # Build canonical view
    entries = pd.DataFrame()
    entries["_row_idx"] = entries_norm["_row_idx"].astype(int)
    entries["entry_id"] = entries_norm[entry_id_col].astype(str)
    entries["contest_id"] = entries_norm[contest_id_col].astype(str)
    entries["cpt"] = entries_norm.iloc[:, cpt_idx].astype(str)
    for i, idx in enumerate(slot_flex_idxs, start=1):
        entries[f"util{i}"] = entries_norm.iloc[:, idx].astype(str)

    slot_cols_raw: Dict[str, int] = {
        "cpt": int(cpt_idx),
        **{f"util{i}": int(slot_flex_idxs[i - 1]) for i in range(1, 6)},
    }

    # Preserve raw entries section for passthrough write-back.
    return DkEntriesFile(raw=entries_raw, entries=entries, slot_cols=slot_cols_raw)


def _find_raw_col(raw_cols: Sequence[str], colmap: Dict[str, str], normalized: str) -> str:
    # Reverse lookup normalized -> raw
    for raw in raw_cols:
        if colmap.get(raw) == normalized:
            return str(raw)
    # should never happen
    return normalized


def _parse_name_id_cell(s: str) -> Tuple[str, Optional[str]]:
    """
    Parse DK-style \"Name (12345)\" cell. Returns (name, id_str).
    """
    txt = (s or "").strip()
    if not txt:
        return "", None
    if txt.endswith(")") and "(" in txt:
        name = txt[: txt.rfind("(")].strip()
        pid = txt[txt.rfind("(") + 1 : -1].strip()
        return name, (pid if pid.isdigit() else pid)
    return txt, None


def _build_nameid_map(players_df: pd.DataFrame) -> Dict[str, str]:
    """
    Map normalized name -> \"Name (DFS ID)\" using projections columns.
    """
    # Accept a few possible column names
    name_col = None
    for c in ["player_name", "Name", "name", "player", "display_name"]:
        if c in players_df.columns:
            name_col = c
            break
    if name_col is None:
        # fallback to name_norm (not ideal but deterministic)
        name_col = "name_norm"
    id_col = None
    for c in ["DFS ID", "dfs_id", "dk_id", "id"]:
        if c in players_df.columns:
            id_col = c
            break

    out: Dict[str, str] = {}
    for _, r in players_df.iterrows():
        name = str(r.get(name_col, "")).strip()
        if not name:
            continue
        nm = norm_name(name)
        pid = None if id_col is None else str(r.get(id_col, "")).strip()
        if pid and pid != "nan":
            out[nm] = f"{name} ({pid})"
        else:
            out[nm] = name
    return out


def fill_dkentries_with_assignments(
    *,
    dkentries: DkEntriesFile,
    players_df: pd.DataFrame,
    lineups_parquet: Path,
    assignments: Dict[str, List[int]],
    output_format: str = "name_id",
) -> pd.DataFrame:
    """
    Fill a DKEntries template with chosen lineup ids per contest.

    `assignments`: contest_id -> list of lineup_id (one per entry row, in row order).
    """
    out = dkentries.raw.copy()

    # materialize lineups slots for requested lineup_ids
    all_ids: List[int] = []
    for cid, ids in assignments.items():
        all_ids.extend([int(x) for x in ids])
    unique_ids = sorted(set(all_ids))
    if not unique_ids:
        raise ValueError("No assignments provided")

    tbl = pq.read_table(lineups_parquet, columns=["cpt", "u1", "u2", "u3", "u4", "u5"], memory_map=True)
    lineups = tbl.to_pandas()
    if max(unique_ids) >= len(lineups):
        raise ValueError(f"Assignment lineup_id out of range: max={max(unique_ids)} n_lineups={len(lineups)}")

    # name mapping for output
    nameid_map = _build_nameid_map(players_df) if output_format == "name_id" else {}
    names_norm = players_df["name_norm"].astype(str).tolist() if "name_norm" in players_df.columns else []

    def fmt_player(idx: int) -> str:
        if not names_norm:
            return str(idx)
        nm = str(names_norm[int(idx)])
        return nameid_map.get(nm, nm)

    # Fill entries in the same order as they appear in dkentries.entries
    for contest_id, chosen in assignments.items():
        # Use normalized entries order to align.
        entry_rows = dkentries.entries[dkentries.entries["contest_id"].astype(str) == str(contest_id)]
        if len(entry_rows) != len(chosen):
            raise ValueError(f"Contest {contest_id}: assignments {len(chosen)} != dkentries rows {len(entry_rows)}")
        # Determine which raw rows correspond (same order as read)
        raw_idx = entry_rows["_row_idx"].astype(int).tolist()
        for i, lid in enumerate(chosen):
            row_pos = raw_idx[i]
            slots = lineups.iloc[int(lid)]
            out.iloc[row_pos, dkentries.slot_cols["cpt"]] = fmt_player(int(slots["cpt"]))
            out.iloc[row_pos, dkentries.slot_cols["util1"]] = fmt_player(int(slots["u1"]))
            out.iloc[row_pos, dkentries.slot_cols["util2"]] = fmt_player(int(slots["u2"]))
            out.iloc[row_pos, dkentries.slot_cols["util3"]] = fmt_player(int(slots["u3"]))
            out.iloc[row_pos, dkentries.slot_cols["util4"]] = fmt_player(int(slots["u4"]))
            out.iloc[row_pos, dkentries.slot_cols["util5"]] = fmt_player(int(slots["u5"]))

    return out


