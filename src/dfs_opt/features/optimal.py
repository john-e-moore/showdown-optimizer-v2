from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from dfs_opt.utils.hashing import sha256_hex


@dataclass
class OptimalResult:
    optimal_proj_points: float
    compute_time_s: float
    num_players_considered: int


class OptimalShowdownCache:
    def __init__(self) -> None:
        self._cache: Dict[str, OptimalResult] = {}

    def get(self, key: str) -> Optional[OptimalResult]:
        return self._cache.get(key)

    def set(self, key: str, value: OptimalResult) -> None:
        self._cache[key] = value


def players_fingerprint(players_flex: pd.DataFrame) -> str:
    """
    Stable fingerprint for a slate's player table.
    Only depends on (name_norm, salary, proj_points) sorted by name_norm.
    """
    df = players_flex[["name_norm", "salary", "proj_points"]].copy()
    df["name_norm"] = df["name_norm"].astype(str)
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df["proj_points"] = pd.to_numeric(df["proj_points"], errors="coerce")
    df = df.dropna().sort_values("name_norm")
    payload = "|".join(f"{r.name_norm}:{int(r.salary)}:{float(r.proj_points):.6f}" for r in df.itertuples())
    return sha256_hex(payload.encode("utf-8"))


def compute_optimal_showdown_proj(
    players_flex: pd.DataFrame, *, salary_cap: int = 50000, min_proj_points: float = 0.0
) -> OptimalResult:
    """
    Brute-force best projected showdown lineup:
      - 1 CPT (1.5x salary and proj)
      - 5 UTIL
      - salary cap (default 50,000)

    Expected columns on players_flex: salary, proj_points
    """
    t0 = time.perf_counter()

    df = players_flex.copy()
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df["proj_points"] = pd.to_numeric(df["proj_points"], errors="coerce")
    df = df.dropna(subset=["salary", "proj_points"]).copy()
    df = df[df["proj_points"] > min_proj_points].reset_index(drop=True)

    n = int(len(df))
    if n < 6:
        return OptimalResult(optimal_proj_points=float("nan"), compute_time_s=time.perf_counter() - t0, num_players_considered=n)

    sal = df["salary"].to_numpy(dtype=float)
    prj = df["proj_points"].to_numpy(dtype=float)

    best = -math.inf
    idx_all = list(range(n))

    for cpt_idx in idx_all:
        cpt_cost = 1.5 * sal[cpt_idx]
        cpt_pts = 1.5 * prj[cpt_idx]

        if cpt_cost > salary_cap:
            continue

        remaining = [i for i in idx_all if i != cpt_idx]
        for comb in itertools.combinations(remaining, 5):
            cost = cpt_cost + float(sal[list(comb)].sum())
            if cost > salary_cap:
                continue
            pts = cpt_pts + float(prj[list(comb)].sum())
            if pts > best:
                best = pts

    if best == -math.inf:
        best = float("nan")

    return OptimalResult(optimal_proj_points=float(best), compute_time_s=time.perf_counter() - t0, num_players_considered=n)


def add_optimal_and_gap(
    enriched_entries: pd.DataFrame, *, optimal_proj_points: float
) -> pd.DataFrame:
    out = enriched_entries.copy()
    out["optimal_proj_points"] = float(optimal_proj_points)
    out["proj_gap_to_optimal"] = float(optimal_proj_points) - out["proj_points"].astype(float)
    return out


