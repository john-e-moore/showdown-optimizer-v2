from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from dfs_opt.lineup_pool.numba_kernels_showdown import count_valid_per_cpt, fill_lineups


@dataclass(frozen=True)
class PlayerArrays:
    salary: np.ndarray  # int32[n]
    proj_points: np.ndarray  # float32[n]
    team01: np.ndarray  # uint8[n] in {0,1}
    team_mapping: Dict[str, int]  # stable mapping (sorted team name -> 0/1)


@dataclass(frozen=True)
class LineupUniverseResult:
    num_players: int
    num_lineups: int
    counts_per_cpt: np.ndarray  # int64[n]
    offsets: np.ndarray  # int64[n]

    cpt: np.ndarray  # uint16[num_lineups]
    u1: np.ndarray
    u2: np.ndarray
    u3: np.ndarray
    u4: np.ndarray
    u5: np.ndarray

    salary_used: np.ndarray  # int32[num_lineups]
    salary_left: np.ndarray  # int32[num_lineups]
    proj_points: np.ndarray  # float32[num_lineups]
    stack_code: np.ndarray  # uint8[num_lineups]

    metrics: Dict[str, Any]


def prepare_player_arrays(
    players_flex: pd.DataFrame,
    *,
    min_proj_points: float = 0.0,
    max_players: Optional[int] = None,
) -> Tuple[pd.DataFrame, PlayerArrays, Dict[str, Any]]:
    """
    Prepare a deterministic, filtered players table + compact NumPy arrays for enumeration.

    Expected columns on players_flex: name_norm, team, salary, proj_points
    """
    required = ["name_norm", "team", "salary", "proj_points"]
    missing = [c for c in required if c not in players_flex.columns]
    if missing:
        raise ValueError(f"players_flex missing required columns: {missing}")

    df = players_flex.copy()
    df["name_norm"] = df["name_norm"].astype(str)
    df["team"] = df["team"].astype(str)
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df["proj_points"] = pd.to_numeric(df["proj_points"], errors="coerce")
    df = df.dropna(subset=["name_norm", "team", "salary", "proj_points"]).copy()

    df = df[df["proj_points"].astype(float) > float(min_proj_points)].copy()
    if len(df) < 6:
        raise ValueError(f"Need at least 6 eligible players after filtering; got {len(df)}")

    # Deterministic player ordering: by name_norm, then salary, then proj_points.
    df = df.sort_values(["name_norm", "salary", "proj_points"], ascending=[True, True, False]).reset_index(drop=True)

    if max_players is not None:
        df = df.head(int(max_players)).reset_index(drop=True)
        if len(df) < 6:
            raise ValueError(f"Need at least 6 players after max_players cap; got {len(df)}")

    teams = sorted(set(df["team"].astype(str).tolist()))
    if len(teams) != 2:
        raise ValueError(f"NBA showdown enumeration requires exactly 2 teams; got {len(teams)}: {teams}")

    team_mapping = {teams[0]: 0, teams[1]: 1}
    team01 = df["team"].map(team_mapping).astype(np.uint8).to_numpy()

    salary = df["salary"].astype(int).to_numpy(dtype=np.int32)
    proj = df["proj_points"].astype(float).to_numpy(dtype=np.float32)

    arrays = PlayerArrays(salary=salary, proj_points=proj, team01=team01, team_mapping=team_mapping)
    prep_meta = {
        "num_players": int(len(df)),
        "teams": teams,
        "team_mapping": team_mapping,
        "min_proj_points": float(min_proj_points),
        "max_players": (None if max_players is None else int(max_players)),
    }
    return df, arrays, prep_meta


def enumerate_showdown_universe(
    arrays: PlayerArrays,
    *,
    salary_cap: int = 50000,
) -> LineupUniverseResult:
    """
    Enumerate all legal showdown lineups using the Numba two-pass approach.
    """
    t0 = time.perf_counter()
    counts_t0 = time.perf_counter()
    counts = count_valid_per_cpt(arrays.salary, arrays.team01, salary_cap=int(salary_cap))
    counts_s = time.perf_counter() - counts_t0

    offsets = np.zeros_like(counts, dtype=np.int64)
    running = np.int64(0)
    for i in range(int(len(counts))):
        offsets[i] = running
        running += np.int64(counts[i])
    total = int(running)

    # Preallocate outputs.
    cpt = np.empty(total, dtype=np.uint16)
    u1 = np.empty(total, dtype=np.uint16)
    u2 = np.empty(total, dtype=np.uint16)
    u3 = np.empty(total, dtype=np.uint16)
    u4 = np.empty(total, dtype=np.uint16)
    u5 = np.empty(total, dtype=np.uint16)
    salary_used = np.empty(total, dtype=np.int32)
    salary_left = np.empty(total, dtype=np.int32)
    proj_points = np.empty(total, dtype=np.float32)
    stack_code = np.empty(total, dtype=np.uint8)
    written_per_cpt = np.zeros(int(len(arrays.salary)), dtype=np.int64)

    fill_t0 = time.perf_counter()
    fill_lineups(
        arrays.salary,
        arrays.proj_points,
        arrays.team01,
        offsets,
        counts,
        salary_cap=int(salary_cap),
        cpt_out=cpt,
        u1_out=u1,
        u2_out=u2,
        u3_out=u3,
        u4_out=u4,
        u5_out=u5,
        salary_used_out=salary_used,
        salary_left_out=salary_left,
        proj_points_out=proj_points,
        stack_code_out=stack_code,
        written_per_cpt_out=written_per_cpt,
    )
    fill_s = time.perf_counter() - fill_t0

    # Validate count kernel matches fill results.
    if not np.array_equal(written_per_cpt, counts):
        # Find the first mismatch for a helpful error.
        diffs = np.nonzero(written_per_cpt != counts)[0]
        i = int(diffs[0]) if diffs.size else -1
        raise ValueError(
            f"fill_lineups mismatch: cpt_idx={i} counted={int(counts[i])} written={int(written_per_cpt[i])}"
        )

    metrics: Dict[str, Any] = {
        "num_players": int(len(arrays.salary)),
        "num_lineups": int(total),
        "salary_cap": int(salary_cap),
        "counts_kernel_s": float(counts_s),
        "fill_kernel_s": float(fill_s),
        "total_s": float(time.perf_counter() - t0),
    }

    return LineupUniverseResult(
        num_players=int(len(arrays.salary)),
        num_lineups=int(total),
        counts_per_cpt=counts,
        offsets=offsets,
        cpt=cpt,
        u1=u1,
        u2=u2,
        u3=u3,
        u4=u4,
        u5=u5,
        salary_used=salary_used,
        salary_left=salary_left,
        proj_points=proj_points,
        stack_code=stack_code,
        metrics=metrics,
    )


def sample_lineups_df(res: LineupUniverseResult, *, n: int = 200) -> pd.DataFrame:
    """
    Return a small DataFrame sample of the lineup universe suitable for ArtifactWriter preview/schema.
    """
    k = min(int(n), int(res.num_lineups))
    return pd.DataFrame(
        {
            "cpt": res.cpt[:k],
            "u1": res.u1[:k],
            "u2": res.u2[:k],
            "u3": res.u3[:k],
            "u4": res.u4[:k],
            "u5": res.u5[:k],
            "salary_used": res.salary_used[:k],
            "salary_left": res.salary_left[:k],
            "proj_points": res.proj_points[:k],
            "stack_code": res.stack_code[:k],
        }
    )


