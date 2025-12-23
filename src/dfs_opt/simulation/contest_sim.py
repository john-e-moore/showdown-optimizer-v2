from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dfs_opt.simulation.outcomes import build_outcome_spec, simulate_correlated_normals
from dfs_opt.simulation.payouts import payout_for_block


@dataclass(frozen=True)
class PrunedUniverse:
    lineup_ids: np.ndarray  # int64[k]
    p_renorm: np.ndarray  # float64[k]
    cum_mass: float


def build_pruned_universe(p: np.ndarray, *, mass_threshold: float) -> PrunedUniverse:
    """
    Probability-mass pruning: keep smallest prefix of lineups (sorted by p desc)
    whose cumulative mass >= mass_threshold; renormalize p over kept set.
    """
    if p.ndim != 1:
        raise ValueError("p must be 1D")
    if not (0.0 < float(mass_threshold) <= 1.0):
        raise ValueError(f"mass_threshold must be in (0,1], got {mass_threshold}")

    n = int(p.shape[0])
    if n == 0:
        raise ValueError("Empty probability vector")

    order = np.argsort(p.astype(np.float64, copy=False))[::-1]
    p_sorted = p[order].astype(np.float64, copy=False)
    csum = np.cumsum(p_sorted)
    k = int(np.searchsorted(csum, float(mass_threshold), side="left") + 1)
    k = max(1, min(n, k))

    keep = order[:k].astype(np.int64, copy=False)
    kept_p = p[keep].astype(np.float64, copy=False)
    tot = float(np.sum(kept_p))
    if tot <= 0:
        raise ValueError("Pruned mass sum <= 0")
    return PrunedUniverse(lineup_ids=keep, p_renorm=kept_p / tot, cum_mass=float(csum[k - 1]))


def sample_field_counts(
    *,
    p: np.ndarray,
    lineup_ids: np.ndarray,
    n_entries: int,
    seed: int,
    dirichlet_alpha: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Sample a field of size n_entries with replacement from lineup_ids with probs p.
    Returns compact counts as arrays {lineup_id, dup_count}.
    """
    if int(n_entries) <= 0:
        raise ValueError(f"n_entries must be > 0, got {n_entries}")
    if len(p) != len(lineup_ids):
        raise ValueError("p and lineup_ids length mismatch")
    rng = np.random.default_rng(int(seed))

    probs = p.astype(np.float64, copy=False)
    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / float(np.sum(probs))

    if dirichlet_alpha is not None:
        a = float(dirichlet_alpha)
        if a <= 0:
            raise ValueError(f"dirichlet_alpha must be > 0, got {a}")
        probs = rng.dirichlet(a * probs)

    draws = rng.choice(lineup_ids.astype(np.int64, copy=False), size=int(n_entries), replace=True, p=probs)
    uniq, counts = np.unique(draws, return_counts=True)
    return {"lineup_id": uniq.astype(np.int64), "dup_count": counts.astype(np.int32)}


@dataclass(frozen=True)
class GradeResult:
    table: pa.Table
    metrics: Dict[str, Any]


def grade_lineups_for_contest(
    *,
    players_parquet: Path,
    lineups_parquet: Path,
    field_counts: Dict[str, np.ndarray],
    candidate_lineup_ids: np.ndarray,
    payout_table: List[float],
    entry_fee: float,
    num_sims: int,
    seed: int,
    corr_matrix_csv: Path,
    std_mode: str,
    std_scale: float,
    tie_break: str = "lineup_id",
) -> GradeResult:
    """
    Grade candidate lineups as if each were a single entry against the sampled field.

    Implementation notes:
    - Deterministic tie-break: sort by (score desc, lineup_id asc).
    - Duplicate payout splitting is applied within identical-lineup blocks only.
    """
    players = pq.read_table(players_parquet, memory_map=True).to_pandas()
    lineups_tbl = pq.read_table(lineups_parquet, columns=["cpt", "u1", "u2", "u3", "u4", "u5"], memory_map=True)
    lineups = lineups_tbl.to_pandas()

    field_ids = field_counts["lineup_id"].astype(np.int64, copy=False)
    field_dups = field_counts["dup_count"].astype(np.int32, copy=False)
    if int(field_ids.shape[0]) != int(field_dups.shape[0]):
        raise ValueError("field_counts arrays length mismatch")

    cand_ids = candidate_lineup_ids.astype(np.int64, copy=False)
    # Precompute slot indices for field unique lineups
    field_slots = lineups.iloc[field_ids][["cpt", "u1", "u2", "u3", "u4", "u5"]].to_numpy(dtype=np.int64)

    # Candidate slot indices
    cand_slots = lineups.iloc[cand_ids][["cpt", "u1", "u2", "u3", "u4", "u5"]].to_numpy(dtype=np.int64)

    spec, spec_metrics = build_outcome_spec(
        players_df=players,
        corr_matrix_csv=Path(corr_matrix_csv),
        std_mode=str(std_mode),
        std_scale=float(std_scale),
    )
    draws = simulate_correlated_normals(spec, num_sims=int(num_sims), seed=int(seed))

    # Accumulators
    win_sum = np.zeros(len(cand_ids), dtype=np.float64)
    top_rates = {0.001: np.zeros(len(cand_ids), dtype=np.int32), 0.01: np.zeros(len(cand_ids), dtype=np.int32), 0.05: np.zeros(len(cand_ids), dtype=np.int32), 0.20: np.zeros(len(cand_ids), dtype=np.int32)}

    n_field_entries = int(field_dups.sum())
    contest_size = n_field_entries + 1  # candidate added
    if len(payout_table) < contest_size:
        # pad payout table with zeros if short
        payout_table = list(payout_table) + [0.0] * (contest_size - len(payout_table))

    for t in range(int(num_sims)):
        y = draws[t]
        field_scores = (
            1.5 * y[field_slots[:, 0]] + y[field_slots[:, 1]] + y[field_slots[:, 2]] + y[field_slots[:, 3]] + y[field_slots[:, 4]] + y[field_slots[:, 5]]
        )
        cand_scores = (
            1.5 * y[cand_slots[:, 0]] + y[cand_slots[:, 1]] + y[cand_slots[:, 2]] + y[cand_slots[:, 3]] + y[cand_slots[:, 4]] + y[cand_slots[:, 5]]
        )

        # Sort field unique lineups for cumulative rank ranges
        order = np.lexsort((field_ids, -field_scores)) if tie_break == "lineup_id" else np.argsort(-field_scores)
        field_scores_s = field_scores[order]
        field_ids_s = field_ids[order]
        field_dups_s = field_dups[order].astype(np.int64)
        cum = np.cumsum(field_dups_s)

        # For each candidate, compute rank_start = 1 + (# field entries with score > cand_score).
        # This is a simplification (ties by score are broken by lineup_id implicitly via lexsort above).
        # Compute #entries with score > s by finding first index where field_scores_s <= s
        # using search on descending scores.
        # Vectorized search:
        idx = np.searchsorted(-field_scores_s, -cand_scores, side="left")
        above_entries = np.where(idx <= 0, 0, cum[idx - 1])
        rank_start = above_entries + 1  # 1-indexed rank

        # Candidate duplicates: if candidate lineup_id exists in field_ids_s, block_size = dup+1 else 1.
        # Build lookup map once per sim (field unique is typically small).
        lookup = {int(lid): int(dup) for lid, dup in zip(field_ids_s.tolist(), field_dups_s.tolist(), strict=True)}
        for i in range(len(cand_ids)):
            lid = int(cand_ids[i])
            dup_field = int(lookup.get(lid, 0))
            block = dup_field + 1
            winnings = payout_for_block(payout_table, rank_start=int(rank_start[i]), block_size=block)
            win_sum[i] += float(winnings)

            # finish rates thresholds (based on contest_size)
            for pct, arr in top_rates.items():
                cutoff = int(np.ceil(float(pct) * float(contest_size)))
                if int(rank_start[i]) <= cutoff:
                    arr[i] += 1

    exp_win = win_sum / float(num_sims)
    roi = (exp_win - float(entry_fee)) / float(entry_fee) if float(entry_fee) > 0 else np.zeros_like(exp_win)

    data = {
        "lineup_id": pa.array(cand_ids, type=pa.int64()),
        "exp_winnings": pa.array(exp_win.astype(np.float64), type=pa.float64()),
        "roi": pa.array(roi.astype(np.float64), type=pa.float64()),
        "top_0_1_pct": pa.array((top_rates[0.001] / float(num_sims)).astype(np.float64), type=pa.float64()),
        "top_1_pct": pa.array((top_rates[0.01] / float(num_sims)).astype(np.float64), type=pa.float64()),
        "top_5_pct": pa.array((top_rates[0.05] / float(num_sims)).astype(np.float64), type=pa.float64()),
        "top_20_pct": pa.array((top_rates[0.20] / float(num_sims)).astype(np.float64), type=pa.float64()),
    }
    tbl = pa.table(data)

    metrics: Dict[str, Any] = {
        "num_sims": int(num_sims),
        "n_candidates": int(len(cand_ids)),
        "n_field_unique": int(len(field_ids)),
        "n_field_entries": int(n_field_entries),
        "contest_size_simulated": int(contest_size),
        "tie_break": str(tie_break),
        **spec_metrics,
    }
    return GradeResult(table=tbl, metrics=metrics)


