from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from dfs_opt.features.correlation import load_corr_matrix_csv
from dfs_opt.features.optimal import compute_optimal_showdown_proj
from dfs_opt.lineup_pool.numba_kernels_features import avg_corr_for_lineups


@dataclass(frozen=True)
class EnrichUniverseMetrics:
    optimal_proj_points: float
    own_log_eps: float
    num_players: int
    num_lineups: int
    corr_missing_pairs_rate: float


def _salary_left_bin_codes(salary_left: np.ndarray) -> np.ndarray:
    # bins: 0-200, 200-500, 500-1000, 1000-2000, 2000-4000, 4000-8000, 8000+
    # labels: ["0_200", "200_500", "500_1000", "1000_2000", "2000_4000", "4000_8000", "8000_plus"]
    x = salary_left.astype(np.int32, copy=False)
    # codes in 0..6
    codes = np.empty(x.shape[0], dtype=np.uint8)
    # vectorized thresholds
    codes[x < 200] = 0
    codes[(x >= 200) & (x < 500)] = 1
    codes[(x >= 500) & (x < 1000)] = 2
    codes[(x >= 1000) & (x < 2000)] = 3
    codes[(x >= 2000) & (x < 4000)] = 4
    codes[(x >= 4000) & (x < 8000)] = 5
    codes[x >= 8000] = 6
    return codes


def _pct_gap_bin_codes(pct_gap: np.ndarray) -> np.ndarray:
    # bins: 0-1%, 1-2%, 2-4%, 4-7%, 7-15%, 15-30%, 30%+
    # labels: ["0_0.01", "0.01_0.02", "0.02_0.04", "0.04_0.07", "0.07_0.15", "0.15_0.30", "0.30_plus"]
    x = pct_gap.astype(np.float32, copy=False)
    codes = np.empty(x.shape[0], dtype=np.uint8)
    codes[x < 0.01] = 0
    codes[(x >= 0.01) & (x < 0.02)] = 1
    codes[(x >= 0.02) & (x < 0.04)] = 2
    codes[(x >= 0.04) & (x < 0.07)] = 3
    codes[(x >= 0.07) & (x < 0.15)] = 4
    codes[(x >= 0.15) & (x < 0.30)] = 5
    codes[x >= 0.30] = 6
    return codes


def _archetype_codes_from_salary_rank(
    salary_rank: np.ndarray,
    captain_tiers: Sequence[Tuple[int, str]],
) -> Tuple[np.ndarray, List[str]]:
    """
    Return (codes, labels) where codes are uint8 indices into labels.
    labels are the tier labels in order of first appearance in captain_tiers.
    """
    # Ensure tiers are in ascending max_rank order
    tiers = list(captain_tiers)
    tiers = sorted(tiers, key=lambda t: int(t[0]))
    labels = [str(lbl) for _, lbl in tiers]

    r = salary_rank.astype(np.int32, copy=False)
    codes = np.empty(r.shape[0], dtype=np.uint8)
    for i in range(r.shape[0]):
        ri = int(r[i])
        code = len(tiers) - 1
        for j, (mx, _) in enumerate(tiers):
            if ri <= int(mx):
                code = j
                break
        codes[i] = np.uint8(code)
    return codes, labels


def build_dense_corr_matrix(players_enum_df: pd.DataFrame, *, corr_matrix_csv: Path) -> Tuple[np.ndarray, float]:
    """
    Build a dense float32[n,n] correlation matrix aligned to players_enum_df row order.
    Returns (corr_mat, missing_pairs_rate) where missing_pairs_rate is fraction of off-diagonal
    pairs that were missing from the source lookup (and thus defaulted to 0.0).
    """
    required = ["name_norm"]
    missing = [c for c in required if c not in players_enum_df.columns]
    if missing:
        raise ValueError(f"players_enum_df missing required columns: {missing}")

    names = players_enum_df["name_norm"].astype(str).tolist()
    n = int(len(names))
    lookup = load_corr_matrix_csv(Path(corr_matrix_csv))

    corr = np.zeros((n, n), dtype=np.float32)
    missing_pairs = 0
    total_pairs = 0
    for i in range(n):
        corr[i, i] = np.float32(1.0)
    for i in range(n):
        a = names[i]
        for j in range(i + 1, n):
            b = names[j]
            total_pairs += 1
            v = lookup.get_pair(a, b)
            # CorrLookup returns 0.0 for missing pairs; we can't perfectly distinguish
            # between true 0.0 and missing. We approximate missing by checking presence
            # in either direction in the underlying dict.
            if (a, b) not in lookup.corr and (b, a) not in lookup.corr:
                missing_pairs += 1
            corr[i, j] = np.float32(v)
            corr[j, i] = np.float32(v)

    miss_rate = 0.0 if total_pairs == 0 else float(missing_pairs / total_pairs)
    return corr, miss_rate


def enrich_lineup_universe_showdown(
    *,
    players_enum_df: pd.DataFrame,
    cpt: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    u3: np.ndarray,
    u4: np.ndarray,
    u5: np.ndarray,
    salary_left: np.ndarray,
    proj_points: np.ndarray,
    corr_matrix_csv: Path,
    captain_tiers: Sequence[Tuple[int, str]],
    optimal_proj_points: float | None = None,
    own_log_eps: float = 1e-6,
    salary_cap: int = 50000,
    min_proj_points: float = 0.0,
    own_chunk_size: int = 500_000,
) -> Tuple[Dict[str, Any], EnrichUniverseMetrics]:
    """
    Compute dup-model features for an enumerated showdown lineup universe.

    Returns (columns, metrics) where columns is a dict of column_name -> array-like,
    aligned to the lineup arrays.
    """
    req = ["salary", "proj_points", "own", "name_norm"]
    missing = [c for c in req if c not in players_enum_df.columns]
    if missing:
        raise ValueError(f"players_enum_df missing required columns for enrichment: {missing}")

    n_lineups = int(cpt.shape[0])
    if n_lineups == 0:
        raise ValueError("Cannot enrich empty lineup universe")

    # Optimal projection (once per slate)
    if optimal_proj_points is None:
        opt_res = compute_optimal_showdown_proj(
            players_enum_df.rename(columns={"salary": "salary", "proj_points": "proj_points"}),
            salary_cap=int(salary_cap),
            min_proj_points=float(min_proj_points),
        )
        optimal_proj = float(opt_res.optimal_proj_points)
    else:
        optimal_proj = float(optimal_proj_points)
    if (not math.isfinite(optimal_proj)) or (not optimal_proj > 0.0):
        raise ValueError(f"Invalid optimal_proj_points computed: {optimal_proj}")

    pct_gap = (optimal_proj - proj_points.astype(np.float32)) / np.float32(optimal_proj)
    pct_gap = np.clip(pct_gap, 0.0, np.float32(1e9)).astype(np.float32)
    pct_gap_bin_code = _pct_gap_bin_codes(pct_gap)

    # salary_left bins
    salary_left_bin_code = _salary_left_bin_codes(salary_left)

    # captain archetype from FLEX salary rank (among active players proj>0)
    p = players_enum_df.copy()
    p["proj_points"] = pd.to_numeric(p["proj_points"], errors="coerce")
    p["salary"] = pd.to_numeric(p["salary"], errors="coerce")
    active = p[p["proj_points"].fillna(0.0) > 0].copy()
    active["salary_rank"] = active["salary"].rank(ascending=False, method="min").astype(int)
    salary_rank = active.set_index("name_norm")["salary_rank"].to_dict()
    cpt_names = players_enum_df["name_norm"].astype(str).to_numpy()
    cpt_salary_rank = np.array([int(salary_rank.get(str(cpt_names[i]), 9999)) for i in range(len(cpt_names))], dtype=np.int32)
    cpt_arch_code_per_player, cpt_arch_labels = _archetype_codes_from_salary_rank(cpt_salary_rank, captain_tiers)
    cpt_arch_code = cpt_arch_code_per_player[cpt.astype(np.int64, copy=False)]

    # ownership features (chunked)
    own = players_enum_df["own"].astype(float).to_numpy(dtype=np.float32)
    own_score_logprod = np.empty(n_lineups, dtype=np.float32)
    own_max_log = np.empty(n_lineups, dtype=np.float32)
    own_min_log = np.empty(n_lineups, dtype=np.float32)
    eps = float(own_log_eps)

    for start in range(0, n_lineups, int(own_chunk_size)):
        end = min(n_lineups, start + int(own_chunk_size))
        idx0 = cpt[start:end].astype(np.int64, copy=False)
        idx1 = u1[start:end].astype(np.int64, copy=False)
        idx2 = u2[start:end].astype(np.int64, copy=False)
        idx3 = u3[start:end].astype(np.int64, copy=False)
        idx4 = u4[start:end].astype(np.int64, copy=False)
        idx5 = u5[start:end].astype(np.int64, copy=False)

        mat = np.column_stack(
            [
                own[idx0],
                own[idx1],
                own[idx2],
                own[idx3],
                own[idx4],
                own[idx5],
            ]
        )
        mat = np.clip(mat, eps, 1.0)
        logm = np.log(mat, dtype=np.float32)
        own_score_logprod[start:end] = logm.sum(axis=1, dtype=np.float32)
        own_max_log[start:end] = logm.max(axis=1)
        own_min_log[start:end] = logm.min(axis=1)

    # correlation (dense matrix + Numba kernel over all lineups)
    corr_mat, miss_rate = build_dense_corr_matrix(players_enum_df, corr_matrix_csv=Path(corr_matrix_csv))
    avg_corr = avg_corr_for_lineups(cpt, u1, u2, u3, u4, u5, corr_mat)

    cols: Dict[str, Any] = {
        "own_score_logprod": own_score_logprod,
        "own_max_log": own_max_log,
        "own_min_log": own_min_log,
        "avg_corr": avg_corr,
        "cpt_archetype_code": cpt_arch_code.astype(np.uint8),
        "cpt_archetype_labels": cpt_arch_labels,
        "salary_left_bin_code": salary_left_bin_code.astype(np.uint8),
        "pct_proj_gap_to_optimal": pct_gap,
        "pct_proj_gap_to_optimal_bin_code": pct_gap_bin_code.astype(np.uint8),
        "optimal_proj_points": optimal_proj,
    }
    metrics = EnrichUniverseMetrics(
        optimal_proj_points=optimal_proj,
        own_log_eps=float(own_log_eps),
        num_players=int(len(players_enum_df)),
        num_lineups=int(n_lineups),
        corr_missing_pairs_rate=float(miss_rate),
    )
    return cols, metrics


