from __future__ import annotations

"""
Quick benchmark / sanity checks for `grade_lineups_for_contest`.

Usage examples:
  /home/john/showdown-optimizer-v2/.venv/bin/python scripts/benchmark_grade_lineups.py \
    --run-dir artifacts/contest/<RUN_ID> \
    --contest-id 186465273 \
    --num-sims 20000 \
    --batch-sims 1024 \
    --rank-method sort
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from dfs_opt.simulation.contest_sim import grade_lineups_for_contest
from dfs_opt.simulation.payouts import payout_for_block


def _vectorized_payout(
    payout_table: list[float], *, rank_start: np.ndarray, block_size: np.ndarray
) -> np.ndarray:
    payout_arr = np.asarray(payout_table, dtype=np.float64)
    csum = np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(payout_arr, dtype=np.float64)])
    n = int(payout_arr.shape[0])
    lo = rank_start.astype(np.int64, copy=False) - 1
    hi = lo + block_size.astype(np.int64, copy=False)
    lo_clip = np.clip(lo, 0, n)
    hi_clip = np.clip(hi, 0, n)
    sums = csum[hi_clip] - csum[lo_clip]
    out = sums / block_size.astype(np.float64, copy=False)
    return np.where(lo >= n, 0.0, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--contest-id", type=str, required=True)
    ap.add_argument("--corr-matrix-csv", type=Path, default=None)
    ap.add_argument("--num-sims", type=int, default=20_000)
    ap.add_argument("--batch-sims", type=int, default=1024)
    ap.add_argument("--rank-method", type=str, default="sort", choices=["sort", "hist"])
    ap.add_argument(
        "--compare-sort-vs-hist",
        action="store_true",
        help="Run both rank methods and report ROI correlation + top-k overlap (sanity check).",
    )
    ap.add_argument("--candidate-topk", type=int, default=0, help="0 => use pruned_universe as-is")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--std-mode", type=str, default="dk_std_or_fallback")
    ap.add_argument("--std-scale", type=float, default=1.0)
    ap.add_argument("--tie-break", type=str, default="lineup_id")
    ap.add_argument("--rank-hist-bin-width", type=float, default=0.1)
    ap.add_argument("--rank-hist-min", type=float, default=-100.0)
    ap.add_argument("--rank-hist-max", type=float, default=300.0)
    args = ap.parse_args()

    run_dir = args.run_dir
    contest_dir = run_dir / "contests" / str(args.contest_id)
    players_parquet = run_dir / "players.parquet"
    lineups_parquet = run_dir / "lineups.parquet"
    if args.corr_matrix_csv is not None:
        corr_matrix_csv = args.corr_matrix_csv
    else:
        corr_matrix_csv = run_dir / "inputs" / "corr_matrix.csv"
        if not corr_matrix_csv.exists():
            # fallback: pipeline runners sometimes pass a corr csv path outside run_dir
            corr_matrix_csv = run_dir / "corr_matrix.csv"

    pruned_tbl = pq.read_table(contest_dir / "pruned_universe.parquet", columns=["lineup_id"])
    cand_ids = pruned_tbl["lineup_id"].to_numpy(zero_copy_only=False).astype(np.int64)
    if int(args.candidate_topk) > 0:
        cand_ids = cand_ids[: int(args.candidate_topk)]

    field_tbl = pq.read_table(contest_dir / "field_sample.parquet", columns=["lineup_id", "dup_count"])
    field_counts = {
        "lineup_id": field_tbl["lineup_id"].to_numpy(zero_copy_only=False).astype(np.int64),
        "dup_count": field_tbl["dup_count"].to_numpy(zero_copy_only=False).astype(np.int32),
    }

    # --- Sanity: vectorized payout matches payout_for_block for random samples
    rng = np.random.default_rng(0)
    payout_table = [100.0, 50.0, 25.0] + [0.0] * 100  # small synthetic payout table
    rs = rng.integers(1, 50, size=500, dtype=np.int64)
    bs = rng.integers(1, 30, size=500, dtype=np.int64)
    v = _vectorized_payout(payout_table, rank_start=rs, block_size=bs)
    s = np.array([payout_for_block(payout_table, rank_start=int(r), block_size=int(b)) for r, b in zip(rs, bs)], dtype=np.float64)
    max_abs = float(np.max(np.abs(v - s)))
    print(f"payout_vectorized_max_abs_diff={max_abs:.3e}")

    # --- Benchmark grading
    print(
        "grading:",
        {
            "contest_dir": str(contest_dir),
            "n_candidates": int(len(cand_ids)),
            "n_field_unique": int(len(field_counts["lineup_id"])),
            "num_sims": int(args.num_sims),
            "batch_sims": int(args.batch_sims),
            "rank_method": str(args.rank_method),
        },
    )

    # Payout table / entry fee: read from run manifest if present, otherwise require user to edit.
    # Here we use a simple heuristic: attempt to read from dk_api_cache is too coupled, so we just warn.
    print("NOTE: This script does not fetch DK payout table; using placeholder payout table will invalidate ROI.")
    print("      If you want real ROI timing, run via the pipeline or adapt this script to load DK meta.")

    # Placeholder payout table to exercise the code path (timing only).
    payout_table = [1000.0] + [0.0] * 200_000
    entry_fee = 1.0

    t0 = time.perf_counter()
    res_sort = grade_lineups_for_contest(
        players_parquet=players_parquet,
        lineups_parquet=lineups_parquet,
        field_counts=field_counts,
        candidate_lineup_ids=cand_ids,
        payout_table=payout_table,
        entry_fee=float(entry_fee),
        num_sims=int(args.num_sims),
        seed=int(args.seed),
        corr_matrix_csv=corr_matrix_csv,
        std_mode=str(args.std_mode),
        std_scale=float(args.std_scale),
        tie_break=str(args.tie_break),
        batch_sims=int(args.batch_sims),
        rank_method=str(args.rank_method),
        rank_hist_bin_width=float(args.rank_hist_bin_width),
        rank_hist_min=float(args.rank_hist_min),
        rank_hist_max=float(args.rank_hist_max),
    )
    dt = time.perf_counter() - t0
    print(f"grade_duration_s={dt:.3f}")
    print(res_sort.metrics)

    if args.compare_sort_vs_hist:
        # Compare sort vs hist on the same draws (same seed) to sanity-check ordering similarity.
        res_hist = grade_lineups_for_contest(
            players_parquet=players_parquet,
            lineups_parquet=lineups_parquet,
            field_counts=field_counts,
            candidate_lineup_ids=cand_ids,
            payout_table=payout_table,
            entry_fee=float(entry_fee),
            num_sims=int(args.num_sims),
            seed=int(args.seed),
            corr_matrix_csv=corr_matrix_csv,
            std_mode=str(args.std_mode),
            std_scale=float(args.std_scale),
            tie_break=str(args.tie_break),
            batch_sims=int(args.batch_sims),
            rank_method="hist",
            rank_hist_bin_width=float(args.rank_hist_bin_width),
            rank_hist_min=float(args.rank_hist_min),
            rank_hist_max=float(args.rank_hist_max),
        )
        roi_sort = res_sort.table["roi"].to_numpy(zero_copy_only=False).astype(np.float64)
        roi_hist = res_hist.table["roi"].to_numpy(zero_copy_only=False).astype(np.float64)
        corr = float(np.corrcoef(roi_sort, roi_hist)[0, 1]) if len(roi_sort) > 1 else float("nan")
        k = min(200, int(len(roi_sort)))
        top_sort = set(np.argsort(-roi_sort)[:k].tolist())
        top_hist = set(np.argsort(-roi_hist)[:k].tolist())
        overlap = 0.0 if k == 0 else float(len(top_sort & top_hist) / k)
        print(f"compare_sort_vs_hist_roi_corr={corr:.4f} top{int(k)}_overlap={overlap:.3f}")


if __name__ == "__main__":
    main()


