from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from dfs_opt.parsing.names import norm_name


@dataclass(frozen=True)
class SabersimColumns:
    # Common Sabersim exports (as seen in scripts/build_showdown_training_dataset.py)
    name_col: str = "Name"
    team_col: str = "Team"
    opp_col: Optional[str] = None  # some exports include "Opp"
    pos_col: Optional[str] = "Pos"
    salary_col: str = "Salary"
    proj_col: str = "SS Proj"
    own_col: str = "My Own"
    minutes_col: Optional[str] = "Min"
    dfs_id_col: Optional[str] = "DFS ID"
    dk_std_col: Optional[str] = "dk_std"


def parse_sabersim_showdown_csv(path: Path, *, cols: SabersimColumns = SabersimColumns()) -> Tuple[pd.DataFrame, Dict]:
    """
    Parse Sabersim projections to canonical FLEX-only table.

    Rule: Sabersim may include CPT and FLEX rows for the same player name.
    We keep the row with the **lowest salary** per normalized name as canonical FLEX.
    """
    df = pd.read_csv(path)
    required = [cols.name_col, cols.team_col, cols.salary_col, cols.proj_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}; got {list(df.columns)}")
    if cols.own_col not in df.columns:
        raise ValueError(f"{path}: missing required ownership column '{cols.own_col}'; got {list(df.columns)}")

    out = pd.DataFrame()
    out["player_name"] = df[cols.name_col].astype(str)
    out["name_norm"] = out["player_name"].map(norm_name)
    out["team"] = df[cols.team_col].astype(str)

    if cols.opp_col and cols.opp_col in df.columns:
        out["opponent"] = df[cols.opp_col].astype(str)
    else:
        out["opponent"] = pd.NA

    if cols.pos_col and cols.pos_col in df.columns:
        out["position"] = df[cols.pos_col].astype(str)
    else:
        out["position"] = pd.NA

    out["salary"] = pd.to_numeric(df[cols.salary_col], errors="coerce").astype("Int64")
    out["proj_points"] = pd.to_numeric(df[cols.proj_col], errors="coerce")
    # Sabersim exports ownership as percent; canonicalize to probability in [0,1].
    own_raw = pd.to_numeric(df[cols.own_col], errors="coerce")
    out["own"] = (own_raw / 100.0).clip(lower=0.0, upper=1.0)

    if cols.minutes_col and cols.minutes_col in df.columns:
        out["minutes"] = pd.to_numeric(df[cols.minutes_col], errors="coerce")
    else:
        out["minutes"] = pd.NA

    # Optional columns used by Pipeline B grading + DKEntries formatting
    if cols.dfs_id_col and cols.dfs_id_col in df.columns:
        out["dfs_id"] = pd.to_numeric(df[cols.dfs_id_col], errors="coerce").astype("Int64")
    else:
        out["dfs_id"] = pd.NA

    if cols.dk_std_col and cols.dk_std_col in df.columns:
        out["dk_std"] = pd.to_numeric(df[cols.dk_std_col], errors="coerce")
    else:
        out["dk_std"] = pd.NA

    before = len(out)
    out = out.dropna(subset=["name_norm", "salary", "proj_points", "own"]).copy()

    # choose min salary row per player as FLEX
    out = out.sort_values(["name_norm", "salary"], ascending=[True, True])
    out = out.groupby("name_norm", as_index=False).first()

    dropped_zero_proj = int((out["proj_points"].fillna(0.0) <= 0).sum())

    metrics = {
        "num_rows_raw": int(before),
        "num_players": int(len(out)),
        "dropped_zero_proj": dropped_zero_proj,
    }
    return out, metrics


