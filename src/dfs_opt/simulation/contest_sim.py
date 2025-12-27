from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dfs_opt.simulation.outcomes import build_outcome_spec, simulate_correlated_normals_batches


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
    batch_sims: int = 1024,
    rank_method: str = "sort",
    rank_hist_bin_width: float = 0.1,
    rank_hist_min: float = -100.0,
    rank_hist_max: float = 300.0,
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

    # Precompute candidate block sizes (dup_in_field + 1). Duplicates are determined by lineup_id only.
    # This avoids building a dict lookup every sim.
    field_sort_idx = np.argsort(field_ids)
    field_ids_sorted = field_ids[field_sort_idx]
    field_dups_sorted = field_dups[field_sort_idx].astype(np.int64, copy=False)
    pos = np.searchsorted(field_ids_sorted, cand_ids)
    # NOTE: `pos` can be == len(field_ids_sorted); clip for safe indexing then mask with `hit`.
    pos_clip = np.clip(pos, 0, max(0, int(field_ids_sorted.shape[0]) - 1))
    hit = (pos < field_ids_sorted.shape[0]) & (field_ids_sorted[pos_clip] == cand_ids)
    dup_in_field = np.where(hit, field_dups_sorted[pos_clip], 0).astype(np.int64, copy=False)
    block_size = dup_in_field + 1  # int64[n_candidates]

    # Payout prefix sums for fast range sum queries.
    n_field_entries = int(field_dups.sum())
    contest_size = n_field_entries + 1  # candidate added
    if len(payout_table) < contest_size:
        payout_table = list(payout_table) + [0.0] * (contest_size - len(payout_table))
    payout_arr = np.asarray(payout_table, dtype=np.float64)
    payout_cumsum = np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(payout_arr, dtype=np.float64)])
    payout_len = int(payout_arr.shape[0])

    # Precompute top-% cutoffs (1-indexed ranks).
    top_pcts = (0.001, 0.01, 0.05, 0.20)
    cutoffs = {pct: int(np.ceil(float(pct) * float(contest_size))) for pct in top_pcts}

    # Accumulators
    win_sum = np.zeros(len(cand_ids), dtype=np.float64)
    top_rates = {pct: np.zeros(len(cand_ids), dtype=np.int32) for pct in top_pcts}

    if int(batch_sims) <= 0:
        raise ValueError(f"batch_sims must be > 0, got {batch_sims}")

    # Local views for speed in the sim loop.
    f0, f1, f2, f3, f4, f5 = (
        field_slots[:, 0],
        field_slots[:, 1],
        field_slots[:, 2],
        field_slots[:, 3],
        field_slots[:, 4],
        field_slots[:, 5],
    )
    c0, c1, c2, c3, c4, c5 = (
        cand_slots[:, 0],
        cand_slots[:, 1],
        cand_slots[:, 2],
        cand_slots[:, 3],
        cand_slots[:, 4],
        cand_slots[:, 5],
    )

    method = str(rank_method).lower().strip()
    if method not in {"sort", "hist"}:
        raise ValueError(f"rank_method must be one of ['sort','hist'], got {rank_method!r}")
    if method == "hist":
        bw = float(rank_hist_bin_width)
        if bw <= 0:
            raise ValueError(f"rank_hist_bin_width must be > 0, got {rank_hist_bin_width}")
        hmin = float(rank_hist_min)
        hmax = float(rank_hist_max)
        if not (hmax > hmin):
            raise ValueError(f"rank_hist_max must be > rank_hist_min, got {rank_hist_min}..{rank_hist_max}")
        nbins = int(np.ceil((hmax - hmin) / bw))
        nbins = max(1, nbins)

    sims_done = 0
    for batch in simulate_correlated_normals_batches(spec, num_sims=int(num_sims), seed=int(seed), batch_sims=int(batch_sims)):
        for y in batch:
            sims_done += 1
            field_scores = 1.5 * y[f0] + y[f1] + y[f2] + y[f3] + y[f4] + y[f5]
            cand_scores = 1.5 * y[c0] + y[c1] + y[c2] + y[c3] + y[c4] + y[c5]

            if method == "sort":
                # Sort field unique lineups for cumulative rank ranges.
                order = np.lexsort((field_ids, -field_scores)) if tie_break == "lineup_id" else np.argsort(-field_scores)
                field_scores_s = field_scores[order]
                field_dups_s = field_dups[order].astype(np.int64, copy=False)
                cum = np.cumsum(field_dups_s)

                # For each candidate, compute rank_start = 1 + (# field entries with score > cand_score).
                # Ties are broken by lineup_id via lexsort above.
                idx = np.searchsorted(-field_scores_s, -cand_scores, side="left")
                above_entries = np.where(idx <= 0, 0, cum[idx - 1])
                rank_start = above_entries + 1  # int64[n_candidates], 1-indexed ranks
            else:
                # Approximate rank via weighted histogram over field scores (no per-sim sort).
                # Note: this ignores tie-break and within-bin ordering for speed.
                fs = np.clip(field_scores, hmin, np.nextafter(hmax, hmin))
                bins = ((fs - hmin) / bw).astype(np.int32)
                counts = np.bincount(bins, weights=field_dups.astype(np.int64, copy=False), minlength=nbins).astype(np.int64)
                cum = np.cumsum(counts)  # ascending by score bin
                cs = np.clip(cand_scores, hmin, np.nextafter(hmax, hmin))
                cbins = ((cs - hmin) / bw).astype(np.int32)
                above_entries = n_field_entries - cum[cbins]
                rank_start = above_entries + 1

            # Vectorized payout splitting within a candidate duplicate block:
            # winnings = sum(payout[rank_start .. rank_start+block_size-1]) / block_size
            lo = rank_start.astype(np.int64, copy=False) - 1
            hi = lo + block_size
            lo_clip = np.clip(lo, 0, payout_len)
            hi_clip = np.clip(hi, 0, payout_len)
            sums = payout_cumsum[hi_clip] - payout_cumsum[lo_clip]
            winnings = sums / block_size.astype(np.float64, copy=False)
            winnings = np.where(lo >= payout_len, 0.0, winnings)
            win_sum += winnings

            for pct, arr in top_rates.items():
                arr += (rank_start <= int(cutoffs[pct])).astype(np.int32, copy=False)

    if sims_done != int(num_sims):
        raise RuntimeError(f"simulate_correlated_normals_batches produced {sims_done} sims, expected {num_sims}")

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


