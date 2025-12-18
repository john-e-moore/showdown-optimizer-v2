
"""
build_showdown_training_dataset.py

Purpose:
- Read Sabersim projection CSVs (NBA DK Showdown format)
- Read DraftKings contest standings/export CSVs
- Parse each entry's lineup into CPT + 5 UTIL
- Join to projection data (FLEX rows) to compute lineup-level features:
    * salary_used / salary_left
    * projected_points
    * projection_gap_to_optimal
    * stack_pattern (4-2, 3-3, 5-1, 6-0) + heavy team
    * captain archetype (configurable tiers)
    * dup_count (how many entrants played the exact same lineup)
- Write an enriched per-entry dataset you can aggregate across contests.

Run (single pair):
  python build_showdown_training_dataset.py \
    --proj_csv "/path/to/Sabersim.csv" \
    --standings_csv "/path/to/contest-standings.csv" \
    --out_csv "/path/to/enriched_entries.csv"

Run (batch directories):
  python build_showdown_training_dataset.py \
    --proj_dir "/path/to/projections/" \
    --standings_dir "/path/to/standings/" \
    --out_dir "/path/to/out/"

Assumptions:
- Sabersim CSV contains BOTH CPT and FLEX rows per player (same Name). We keep the FLEX row:
    FLEX row = the row with the smaller Salary for the same player name.
- DraftKings standings CSV has a 'Lineup' column with format:
    "CPT <name> UTIL <name> UTIL <name> UTIL <name> UTIL <name> UTIL <name>"
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Helpers / normalization
# -----------------------------

_NAME_CLEAN_RE = re.compile(r"[\.\'\-]")
_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b", re.IGNORECASE)

def norm_name(s: str) -> str:
    s = (s or "").lower()
    s = _NAME_CLEAN_RE.sub("", s)
    s = _SUFFIX_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_lineup_str(lineup: str) -> Optional[Tuple[str, List[str]]]:
    """
    Returns (captain_name, [util1..util5]) or None if parse fails.
    """
    if lineup is None or (isinstance(lineup, float) and math.isnan(lineup)):
        return None
    lineup = str(lineup).strip()
    if not lineup.startswith("CPT "):
        return None

    util_sep = " UTIL "
    first_util_pos = lineup.find(util_sep)
    if first_util_pos == -1:
        return None

    cpt = lineup[4:first_util_pos].strip()
    rest = lineup[first_util_pos + len(util_sep):]
    utils = [x.strip() for x in rest.split(util_sep)]
    if len(utils) != 5:
        # Some exports include fewer/more due to bad rows—skip those.
        return None
    return cpt, utils


# -----------------------------
# Projection ingestion
# -----------------------------

@dataclass
class ProjectionSpec:
    name_col: str = "Name"
    team_col: str = "Team"
    salary_col: str = "Salary"       # FLEX salary
    proj_col: str = "SS Proj"        # Sabersim projection (pre-contest)
    minutes_col: str = "Min"         # optional
    pos_col: str = "Pos"             # optional


def read_sabersim_projection_csv(path: str | Path, spec: ProjectionSpec) -> pd.DataFrame:
    """
    Reads Sabersim CSV and returns one row per player (FLEX row), with:
      name_norm, Name, Team, flex_salary, flex_proj, Min, Pos
    """
    df = pd.read_csv(path)
    required = [spec.name_col, spec.team_col, spec.salary_col, spec.proj_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required projection columns: {missing}. Found: {list(df.columns)}")

    df = df.copy()
    df["name_norm"] = df[spec.name_col].astype(str).map(norm_name)

    # Sabersim exports both CPT and FLEX rows per player (same Name).
    # FLEX row has the smaller Salary → keep min salary per name_norm.
    df = df.sort_values(spec.salary_col, ascending=True).groupby("name_norm", as_index=False).first()

    out = pd.DataFrame({
        "name_norm": df["name_norm"],
        "Name": df[spec.name_col].astype(str),
        "Team": df[spec.team_col].astype(str),
        "flex_salary": df[spec.salary_col].astype(float),
        "flex_proj": df[spec.proj_col].astype(float),
    })

    if spec.minutes_col in df.columns:
        out["min_proj"] = pd.to_numeric(df[spec.minutes_col], errors="coerce")
    else:
        out["min_proj"] = np.nan

    if spec.pos_col in df.columns:
        out["pos"] = df[spec.pos_col].astype(str)
    else:
        out["pos"] = ""

    return out


# -----------------------------
# Standings ingestion
# -----------------------------

def read_dk_standings_csv(path: str | Path) -> pd.DataFrame:
    """
    Reads a DraftKings contest standings/export CSV.
    Keeps only the fields we care about if they exist.
    """
    df = pd.read_csv(path)

    # Keep canonical columns if present
    cols = []
    for c in ["Rank", "EntryId", "EntryName", "Points", "Lineup"]:
        if c in df.columns:
            cols.append(c)

    if "Lineup" not in cols:
        raise ValueError(f"{path}: could not find a 'Lineup' column. Found: {list(df.columns)}")

    df = df[cols].copy()

    # Drop rows with missing lineup (some contests include empty/zero-point rows)
    df = df[df["Lineup"].notna()].copy()

    return df


# -----------------------------
# Optimal projection (bruteforce)
# -----------------------------

def compute_optimal_showdown_proj(players: pd.DataFrame) -> float:
    """
    Bruteforce best projected showdown lineup:
      1 CPT (1.5x salary and proj) + 5 UTIL, salary cap 50,000
    Uses players with flex_proj > 0 by default.
    """
    ply = players[players["flex_proj"] > 0].reset_index(drop=True)
    if len(ply) < 6:
        return float("nan")

    sal = ply["flex_salary"].to_numpy(dtype=float)
    prj = ply["flex_proj"].to_numpy(dtype=float)
    n = len(ply)

    best_proj = -1.0

    idx_all = list(range(n))
    for cpt_idx in idx_all:
        cpt_cost = sal[cpt_idx] * 1.5
        cpt_pts = prj[cpt_idx] * 1.5

        remaining = [i for i in idx_all if i != cpt_idx]
        # combinations of 5 UTIL
        for comb in itertools.combinations(remaining, 5):
            cost = cpt_cost + sal[list(comb)].sum()
            if cost > 50000:
                continue
            pts = cpt_pts + prj[list(comb)].sum()
            if pts > best_proj:
                best_proj = pts

    return float(best_proj)


# -----------------------------
# Feature engineering
# -----------------------------

def captain_archetype_from_salary_rank(
    cpt_name_norm: str,
    salary_rank_map: Dict[str, int],
    tiers: List[Tuple[int, str]],
) -> str:
    """
    tiers: list like [(2,"stud1-2"), (5,"stud3-5"), (10,"mid6-10"), (999,"value11+")]
    """
    r = salary_rank_map.get(cpt_name_norm, 9999)
    for max_rank, label in tiers:
        if r <= max_rank:
            return label
    return tiers[-1][1]


def enrich_entries(
    standings: pd.DataFrame,
    players: pd.DataFrame,
    *,
    slate_id: str = "",
    contest_id: str = "",
    captain_tiers: Optional[List[Tuple[int, str]]] = None,
) -> pd.DataFrame:

    if captain_tiers is None:
        captain_tiers = [(2, "stud1-2"), (5, "stud3-5"), (10, "mid6-10"), (999, "value11+")]

    players = players.copy()
    players_map = players.set_index("name_norm", drop=False)

    # Salary ranks among active players (flex_proj > 0)
    active = players[players["flex_proj"] > 0].copy()
    active["salary_rank"] = active["flex_salary"].rank(ascending=False, method="min").astype(int)
    salary_rank_map = dict(zip(active["name_norm"], active["salary_rank"]))

    optimal_proj = compute_optimal_showdown_proj(players)

    df = standings.copy()
    parsed = df["Lineup"].apply(parse_lineup_str)
    df["cpt_name"] = parsed.apply(lambda x: x[0] if x else None)
    df["utils"] = parsed.apply(lambda x: x[1] if x else None)

    for i in range(5):
        df[f"util{i+1}_name"] = df["utils"].apply(lambda u: u[i] if isinstance(u, list) and len(u) > i else None)

    # Drop unparseable rows
    df = df[df["cpt_name"].notna()].copy()

    # Duplication: how many entries used the same exact lineup string
    df["dup_count"] = df.groupby("Lineup")["Lineup"].transform("count").astype(int)

    # Join player data and compute lineup features
    def get_player(name: str) -> Optional[pd.Series]:
        if not name:
            return None
        key = norm_name(name)
        if key in players_map.index:
            return players_map.loc[key]
        return None

    salary_used = []
    proj_points = []
    stack_pattern = []
    heavy_team = []
    cpt_team = []
    cpt_flex_salary = []
    cpt_flex_proj = []
    cpt_min = []
    cpt_arch = []
    missing = []

    for _, r in df.iterrows():
        names = [r["cpt_name"]] + [r[f"util{i}_name"] for i in range(1, 6)]
        rows = [get_player(n) for n in names]

        if any(x is None for x in rows):
            bad = [names[i] for i, x in enumerate(rows) if x is None]
            missing.append(";".join(bad))
            salary_used.append(np.nan)
            proj_points.append(np.nan)
            stack_pattern.append(None)
            heavy_team.append(None)
            cpt_team.append(None)
            cpt_flex_salary.append(np.nan)
            cpt_flex_proj.append(np.nan)
            cpt_min.append(np.nan)
            cpt_arch.append(None)
            continue

        missing.append("")
        flex_sals = np.array([x["flex_salary"] for x in rows], dtype=float)
        flex_prjs = np.array([x["flex_proj"] for x in rows], dtype=float)
        teams = [x["Team"] for x in rows]

        su = flex_sals[0] * 1.5 + flex_sals[1:].sum()
        pp = flex_prjs[0] * 1.5 + flex_prjs[1:].sum()

        # stack pattern for two-team showdown
        counts: Dict[str, int] = {}
        for t in teams:
            counts[t] = counts.get(t, 0) + 1

        if len(counts) == 2:
            items = sorted(counts.items(), key=lambda kv: -kv[1])
            heavy_t, heavy_n = items[0]
            light_n = items[1][1]
            sp = f"{heavy_n}-{light_n}"
        else:
            heavy_t, sp = None, None

        cpt_key = norm_name(r["cpt_name"])
        arch = captain_archetype_from_salary_rank(cpt_key, salary_rank_map, captain_tiers)

        salary_used.append(float(su))
        proj_points.append(float(pp))
        stack_pattern.append(sp)
        heavy_team.append(heavy_t)
        cpt_team.append(teams[0])
        cpt_flex_salary.append(float(flex_sals[0]))
        cpt_flex_proj.append(float(flex_prjs[0]))
        cpt_min.append(float(rows[0].get("min_proj", np.nan)))
        cpt_arch.append(arch)

    df["salary_used"] = salary_used
    df["salary_left"] = 50000.0 - df["salary_used"]
    df["proj_points"] = proj_points
    df["proj_gap_to_optimal"] = float(optimal_proj) - df["proj_points"]
    df["stack_pattern"] = stack_pattern
    df["heavy_team"] = heavy_team
    df["cpt_team"] = cpt_team
    df["cpt_flex_salary"] = cpt_flex_salary
    df["cpt_flex_proj"] = cpt_flex_proj
    df["cpt_min_proj"] = cpt_min
    df["cpt_archetype"] = cpt_arch
    df["missing_players"] = missing

    # Attach IDs for easier multi-contest aggregation
    df["slate_id"] = slate_id
    df["contest_id"] = contest_id
    df["optimal_proj_points"] = float(optimal_proj)

    # Filter out obvious garbage rows (optional; tune thresholds)
    df = df[df["salary_used"].notna()].copy()
    df = df[df["salary_used"] >= 45000].copy()

    return df


# -----------------------------
# CLI / IO
# -----------------------------

def guess_id_from_filename(path: Path) -> str:
    s = path.stem
    # crude, but useful: contest-standings-186170418 -> 186170418
    m = re.search(r"(\d{6,})", s)
    return m.group(1) if m else s


def run_single(proj_csv: Path, standings_csv: Path, out_csv: Path) -> None:
    spec = ProjectionSpec()
    players = read_sabersim_projection_csv(proj_csv, spec)
    standings = read_dk_standings_csv(standings_csv)

    slate_id = proj_csv.stem
    contest_id = guess_id_from_filename(standings_csv)

    enriched = enrich_entries(standings, players, slate_id=slate_id, contest_id=contest_id)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_csv, index=False)


def run_batch(proj_dir: Path, standings_dir: Path, out_dir: Path) -> None:
    spec = ProjectionSpec()
    out_dir.mkdir(parents=True, exist_ok=True)

    proj_files = sorted(proj_dir.glob("*.csv"))
    if not proj_files:
        raise ValueError(f"No projection CSVs found in {proj_dir}")

    stand_files = sorted(standings_dir.glob("*.csv"))
    if not stand_files:
        raise ValueError(f"No standings CSVs found in {standings_dir}")

    # Default strategy: match files by date substring if possible, else process cartesian pairs manually.
    # You can replace this with a proper mapping file (recommended once you have many contests).
    for proj_csv in proj_files:
        # naive: look for a standings file that contains the same date token YYYY-MM-DD
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", proj_csv.name)
        candidates = stand_files
        if date_match:
            token = date_match.group(0)
            candidates = [f for f in stand_files if token in f.name] or stand_files

        # if multiple candidates, just take the first; in practice you’ll want a mapping.
        standings_csv = Path(candidates[0])
        players = read_sabersim_projection_csv(proj_csv, spec)
        standings = read_dk_standings_csv(standings_csv)

        slate_id = proj_csv.stem
        contest_id = guess_id_from_filename(standings_csv)
        enriched = enrich_entries(standings, players, slate_id=slate_id, contest_id=contest_id)

        out_csv = out_dir / f"enriched_{slate_id}__contest_{contest_id}.csv"
        enriched.to_csv(out_csv, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--proj_csv", type=str, default="")
    p.add_argument("--standings_csv", type=str, default="")
    p.add_argument("--out_csv", type=str, default="")

    p.add_argument("--proj_dir", type=str, default="")
    p.add_argument("--standings_dir", type=str, default="")
    p.add_argument("--out_dir", type=str, default="")

    args = p.parse_args()

    if args.proj_csv and args.standings_csv:
        out_csv = Path(args.out_csv) if args.out_csv else Path("enriched_entries.csv")
        run_single(Path(args.proj_csv), Path(args.standings_csv), out_csv)
        print(f"Wrote {out_csv}")
        return

    if args.proj_dir and args.standings_dir and args.out_dir:
        run_batch(Path(args.proj_dir), Path(args.standings_dir), Path(args.out_dir))
        print(f"Wrote batch outputs to {args.out_dir}")
        return

    raise SystemExit("Provide either --proj_csv + --standings_csv (single) OR --proj_dir + --standings_dir + --out_dir (batch).")


if __name__ == "__main__":
    main()
