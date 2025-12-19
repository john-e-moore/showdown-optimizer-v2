from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from dfs_opt.parsing.dk_lineup import parse_dk_showdown_lineup
from dfs_opt.parsing.names import norm_name
from dfs_opt.utils.hashing import lineup_hash


@dataclass(frozen=True)
class ContestSegment:
    sport: str
    slate_id: str
    entry_type: str  # "single-entry" | "mme"
    size_bin: str

    @property
    def gpp_category(self) -> str:
        return f"{self.sport.lower()}-showdown-{self.entry_type}-{self.size_bin}"


_ENTRYNAME_MAX_RE = re.compile(r"\((\d+)\s*/\s*(\d+)\)\s*$")


def infer_max_entries_per_user(entry_name: object) -> Optional[int]:
    """
    Parse EntryName suffix like 'foo (60/150)' -> 150.
    Returns None if missing/unparseable.
    """
    if entry_name is None:
        return None
    s = str(entry_name)
    m = _ENTRYNAME_MAX_RE.search(s)
    if not m:
        return None
    return int(m.group(2))


def guess_contest_id_from_path(path: Path) -> str:
    m = re.search(r"(\d{6,})", path.stem)
    return m.group(1) if m else path.stem


def read_dk_standings_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Some DK exports include unnamed blank columns, normalize those away.
    df = df.loc[:, [c for c in df.columns if str(c).strip() != "" and not str(c).startswith("Unnamed:")]].copy()
    return df


def parse_dk_showdown_entries(
    standings_csv: Path,
    *,
    sport: str,
    slate_id: str,
    size_bin: str,
) -> Tuple[pd.DataFrame, dict]:
    """
    Parse a DK standings/export CSV to a canonical entries table with slots + lineup_hash.
    """
    raw = read_dk_standings_csv(standings_csv)

    if "Lineup" not in raw.columns:
        raise ValueError(f"{standings_csv}: missing Lineup column; got {list(raw.columns)}")

    df = pd.DataFrame()
    df["lineup_str"] = raw["Lineup"]

    if "EntryId" in raw.columns:
        df["entry_id"] = raw["EntryId"].astype(str)
    else:
        df["entry_id"] = pd.Series(range(len(raw))).astype(str)

    if "EntryName" in raw.columns:
        df["user"] = raw["EntryName"].astype(str)
        max_entries = df["user"].map(infer_max_entries_per_user)
        df["max_entries_per_user"] = max_entries
    else:
        df["user"] = ""
        df["max_entries_per_user"] = None

    if "Points" in raw.columns:
        df["points"] = pd.to_numeric(raw["Points"], errors="coerce")
    else:
        df["points"] = pd.NA

    if "Rank" in raw.columns:
        df["rank"] = pd.to_numeric(raw["Rank"], errors="coerce")
    else:
        df["rank"] = pd.NA

    df["contest_id"] = guess_contest_id_from_path(standings_csv)
    df["slate_id"] = slate_id
    df["sport"] = sport.lower()

    parsed = df["lineup_str"].map(parse_dk_showdown_lineup)
    df["cpt_name"] = parsed.map(lambda x: x.cpt_name if x else None)
    df["util_names"] = parsed.map(lambda x: x.util_names if x else None)

    for i in range(5):
        df[f"util{i+1}_name"] = df["util_names"].map(
            lambda u, idx=i: (u[idx] if isinstance(u, list) and len(u) == 5 else None)
        )

    # drop unparseable
    before = len(df)
    df = df[df["cpt_name"].notna()].copy()
    after = len(df)

    # normalized slot names + stable lineup hash
    df["cpt_name_norm"] = df["cpt_name"].astype(str).map(norm_name)
    util_norm_cols = []
    for i in range(5):
        col = f"util{i+1}_name_norm"
        util_norm_cols.append(col)
        df[col] = df[f"util{i+1}_name"].astype(str).map(norm_name)

    df["lineup_hash"] = [
        lineup_hash(cpt, [u1, u2, u3, u4, u5])
        for cpt, u1, u2, u3, u4, u5 in zip(
            df["cpt_name_norm"],
            df["util1_name_norm"],
            df["util2_name_norm"],
            df["util3_name_norm"],
            df["util4_name_norm"],
            df["util5_name_norm"],
            strict=True,
        )
    ]

    parse_success_rate = 0.0 if before == 0 else after / before

    # segment inference
    contest_size = int(df["entry_id"].nunique())
    max_entries_observed = df["max_entries_per_user"].dropna()
    max_entries_per_user = int(max_entries_observed.max()) if len(max_entries_observed) else None
    entry_type = "single-entry" if (max_entries_per_user is not None and 1 <= max_entries_per_user <= 5) else "mme"

    segment = ContestSegment(sport=sport, slate_id=slate_id, entry_type=entry_type, size_bin=size_bin)
    df["gpp_category"] = segment.gpp_category

    metrics = {
        "parse_success_rate": parse_success_rate,
        "invalid_lineup_rows": int(before - after),
        "contest_size": contest_size,
        "max_entries_per_user": max_entries_per_user,
        "gpp_category": segment.gpp_category,
    }

    return df, metrics


