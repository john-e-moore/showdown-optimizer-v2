from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import csv
import io

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
    - `slot_cols`: the concrete CSV column names used for CPT/UTIL slots (so we can write back)
    """

    raw: pd.DataFrame
    entries: pd.DataFrame
    slot_cols: Dict[str, str]  # canonical slot -> raw column name


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
    # Prescan the first chunk to detect multi-section DK exports:
    # an "entries" section (header width N) followed by a "player list" section (header width M).
    header_fields = None
    mismatches: List[Dict[str, int]] = []
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
            rdr = csv.reader(f, delimiter=",", quotechar='"')
            for i, row in enumerate(rdr, start=1):
                if i == 1:
                    header_fields = len(row)
                # only scan the first ~200 lines; enough to catch the common "player list" tail section
                if i <= 200:
                    if header_fields is not None and len(row) != header_fields and len(mismatches) < 20:
                        mismatches.append({"line": i, "fields": len(row)})
                else:
                    break
    except Exception:
        # If prescan fails, we still attempt pandas; fallback may not be available.
        header_fields = None
        mismatches = []

    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:
        # If this is a multi-section DKEntries export (entries + player list section),
        # parse only the first section which matches the header width.
        if type(e).__name__ == "ParserError" and header_fields and mismatches:
            buf = io.StringIO()
            with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
                rdr = csv.reader(f, delimiter=",", quotechar='"')
                w = csv.writer(buf, lineterminator="\n")
                for i, row in enumerate(rdr, start=1):
                    if header_fields is not None and i > 1 and len(row) != header_fields:
                        break
                    w.writerow(row)
            buf.seek(0)
            raw = pd.read_csv(buf, dtype=str, keep_default_na=False)
        else:
            raise

    raw = raw.loc[:, [c for c in raw.columns if str(c).strip() != "" and not str(c).startswith("Unnamed:")]].copy()

    cols_norm = _normalize_columns(raw.columns)
    colmap = dict(zip(raw.columns, cols_norm))
    df = raw.rename(columns=colmap).copy()

    entry_id_col = _pick_first(df, ["entry id", "entryid", "entry_id"])
    contest_id_col = _pick_first(df, ["contest id", "contestid", "contest_id"])
    if entry_id_col is None or contest_id_col is None:
        raise ValueError(f"{path}: missing required Entry ID / Contest ID columns; got {list(raw.columns)}")

    # Keep only entry rows (DK appends a player list section with empty Entry ID)
    entry_mask = df[entry_id_col].astype(str).str.strip() != ""
    entries_raw = raw.loc[entry_mask].copy()
    entries_norm = df.loc[entry_mask].copy()
    entries_norm["_row_idx"] = np.arange(len(entries_norm), dtype=np.int64)

    # Slot columns: CPT + 5 flex/util columns (varies by export)
    cpt_col = _pick_first(entries_norm, ["cpt", "captain"])
    # Common showdown exports: FLEX, FLEX.1.. or UTIL, UTIL.1..
    flex_cols = [c for c in entries_norm.columns if c.startswith("flex")]
    util_cols = [c for c in entries_norm.columns if c.startswith("util")]
    slot_flex = flex_cols if flex_cols else util_cols

    if cpt_col is None:
        raise ValueError(f"{path}: missing captain slot column (CPT/Captain); got {list(raw.columns)}")
    if len(slot_flex) < 5:
        raise ValueError(f"{path}: expected 5 FLEX/UTIL columns; got {slot_flex}")

    slot_flex = sorted(slot_flex, key=lambda s: (len(s), s))[:5]

    # Build canonical view
    entries = pd.DataFrame()
    entries["_row_idx"] = entries_norm["_row_idx"].astype(int)
    entries["entry_id"] = entries_norm[entry_id_col].astype(str)
    entries["contest_id"] = entries_norm[contest_id_col].astype(str)
    entries["cpt"] = entries_norm[cpt_col].astype(str)
    for i, c in enumerate(slot_flex, start=1):
        entries[f"util{i}"] = entries_norm[c].astype(str)

    slot_cols_raw: Dict[str, str] = {
        "cpt": _find_raw_col(raw.columns, colmap, cpt_col),
        **{
            f"util{i}": _find_raw_col(raw.columns, colmap, slot_flex[i - 1])
            for i in range(1, 6)
        },
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
        mask = out["Contest ID"].astype(str) == str(contest_id) if "Contest ID" in out.columns else out[dkentries.slot_cols.get("contest_id","Contest ID")].astype(str) == str(contest_id)
        # Use normalized entries order to align.
        entry_rows = dkentries.entries[dkentries.entries["contest_id"].astype(str) == str(contest_id)]
        if len(entry_rows) != len(chosen):
            raise ValueError(f"Contest {contest_id}: assignments {len(chosen)} != dkentries rows {len(entry_rows)}")
        # Determine which raw rows correspond (same order as read)
        raw_idx = entry_rows["_row_idx"].astype(int).tolist()
        for i, lid in enumerate(chosen):
            row_pos = raw_idx[i]
            slots = lineups.iloc[int(lid)]
            out.at[out.index[row_pos], dkentries.slot_cols["cpt"]] = fmt_player(int(slots["cpt"]))
            out.at[out.index[row_pos], dkentries.slot_cols["util1"]] = fmt_player(int(slots["u1"]))
            out.at[out.index[row_pos], dkentries.slot_cols["util2"]] = fmt_player(int(slots["u2"]))
            out.at[out.index[row_pos], dkentries.slot_cols["util3"]] = fmt_player(int(slots["u3"]))
            out.at[out.index[row_pos], dkentries.slot_cols["util4"]] = fmt_player(int(slots["u4"]))
            out.at[out.index[row_pos], dkentries.slot_cols["util5"]] = fmt_player(int(slots["u5"]))

    return out


