from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EnrichMetrics:
    unmatched_player_names: int
    join_coverage: float


def _archetype_from_salary_rank(rank: int, tiers: Sequence[Tuple[int, str]]) -> str:
    for max_rank, label in tiers:
        if rank <= max_rank:
            return label
    return tiers[-1][1]


def enrich_showdown_entries(
    entries: pd.DataFrame,
    players_flex: pd.DataFrame,
    *,
    captain_tiers: Sequence[Tuple[int, str]],
) -> Tuple[pd.DataFrame, Dict]:
    """
    Join parsed DK entries to Sabersim FLEX projections and compute lineup-level features.

    Required input columns on entries:
      - contest_id, slate_id
      - cpt_name, util1_name..util5_name
      - cpt_name_norm, util1_name_norm..util5_name_norm
      - lineup_hash

    Required input columns on players_flex:
      - name_norm, team, salary, proj_points

    Returns:
      (enriched_df, metrics)
    """
    required_entries = [
        "contest_id",
        "slate_id",
        "lineup_hash",
        "cpt_name",
        "cpt_name_norm",
        "util1_name",
        "util2_name",
        "util3_name",
        "util4_name",
        "util5_name",
        "util1_name_norm",
        "util2_name_norm",
        "util3_name_norm",
        "util4_name_norm",
        "util5_name_norm",
    ]
    missing_e = [c for c in required_entries if c not in entries.columns]
    if missing_e:
        raise ValueError(f"entries missing required columns: {missing_e}")

    required_players = ["name_norm", "team", "salary", "proj_points"]
    missing_p = [c for c in required_players if c not in players_flex.columns]
    if missing_p:
        raise ValueError(f"players_flex missing required columns: {missing_p}")

    df = entries.copy()
    p = players_flex.copy()
    p = p.dropna(subset=["name_norm"]).copy()
    p["name_norm"] = p["name_norm"].astype(str)

    # salary rank among active players (proj > 0)
    active = p[p["proj_points"].fillna(0.0) > 0].copy()
    active["salary_rank"] = active["salary"].rank(ascending=False, method="min").astype(int)
    salary_rank_map = dict(zip(active["name_norm"], active["salary_rank"]))

    # join player features for each slot
    p_small = p[["name_norm", "team", "salary", "proj_points"]].rename(
        columns={"team": "slot_team", "salary": "slot_salary", "proj_points": "slot_proj"}
    )

    def join_slot(slot_prefix: str) -> None:
        nonlocal df
        slot_norm = f"{slot_prefix}_name_norm"
        joined = df[[slot_norm]].merge(
            p_small, how="left", left_on=slot_norm, right_on="name_norm", validate="many_to_one"
        )
        df[f"{slot_prefix}_team"] = joined["slot_team"]
        df[f"{slot_prefix}_salary"] = joined["slot_salary"]
        df[f"{slot_prefix}_proj"] = joined["slot_proj"]

    join_slot("cpt")
    for i in range(1, 6):
        join_slot(f"util{i}")

    # identify missing joins
    join_cols = ["cpt_salary"] + [f"util{i}_salary" for i in range(1, 6)]
    missing_mask = df[join_cols].isna().any(axis=1)
    df["missing_players"] = ""
    if missing_mask.any():
        missing_lists: List[str] = []
        for _, r in df.loc[missing_mask].iterrows():
            missing_names = []
            if pd.isna(r["cpt_salary"]):
                missing_names.append(str(r["cpt_name"]))
            for i in range(1, 6):
                if pd.isna(r[f"util{i}_salary"]):
                    missing_names.append(str(r[f"util{i}_name"]))
            missing_lists.append(";".join(missing_names))
        df.loc[missing_mask, "missing_players"] = missing_lists

    unmatched_player_names = int(missing_mask.sum())
    join_coverage = 0.0 if len(df) == 0 else float(1.0 - (unmatched_player_names / len(df)))

    # drop rows we can't compute features for (but keep counts in metrics)
    df_ok = df.loc[~missing_mask].copy()

    # dup_count within contest
    df_ok["dup_count"] = (
        df_ok.groupby(["contest_id", "lineup_hash"])["lineup_hash"].transform("count").astype(int)
    )

    cpt_salary = df_ok["cpt_salary"].astype(float).to_numpy()
    util_salary = np.vstack([df_ok[f"util{i}_salary"].astype(float).to_numpy() for i in range(1, 6)]).T
    cpt_proj = df_ok["cpt_proj"].astype(float).to_numpy()
    util_proj = np.vstack([df_ok[f"util{i}_proj"].astype(float).to_numpy() for i in range(1, 6)]).T

    df_ok["salary_used"] = (1.5 * cpt_salary + util_salary.sum(axis=1)).round(0).astype(int)
    df_ok["salary_left"] = 50000 - df_ok["salary_used"]
    df_ok["proj_points"] = 1.5 * cpt_proj + util_proj.sum(axis=1)

    # teams and stack patterns
    def stack_info(row: pd.Series) -> Tuple[str | None, str | None]:
        teams = [
            row["cpt_team"],
            row["util1_team"],
            row["util2_team"],
            row["util3_team"],
            row["util4_team"],
            row["util5_team"],
        ]
        counts: Dict[str, int] = {}
        for t in teams:
            counts[str(t)] = counts.get(str(t), 0) + 1
        if len(counts) != 2:
            return None, None
        items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        heavy_team, heavy_n = items[0]
        light_n = items[1][1]
        return heavy_team, f"{heavy_n}-{light_n}"

    heavy_team_vals: List[str | None] = []
    stack_pattern_vals: List[str | None] = []
    for _, r in df_ok.iterrows():
        heavy_t, sp = stack_info(r)
        heavy_team_vals.append(heavy_t)
        stack_pattern_vals.append(sp)

    df_ok["heavy_team"] = heavy_team_vals
    df_ok["stack_pattern"] = stack_pattern_vals
    df_ok["cpt_team"] = df_ok["cpt_team"]

    # captain archetype from salary rank
    def cpt_arch(name_norm: str) -> str:
        rank = int(salary_rank_map.get(name_norm, 9999))
        return _archetype_from_salary_rank(rank, captain_tiers)

    df_ok["cpt_archetype"] = df_ok["cpt_name_norm"].astype(str).map(cpt_arch)

    metrics = EnrichMetrics(unmatched_player_names=unmatched_player_names, join_coverage=join_coverage)
    return df_ok, {"unmatched_player_names": metrics.unmatched_player_names, "join_coverage": metrics.join_coverage}


