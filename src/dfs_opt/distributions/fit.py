from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from dfs_opt.distributions.schemas import CategoricalRates, Histogram, TargetDistributions


def _histogram(values: np.ndarray, *, bin_width: float, min_edge: float = 0.0) -> Tuple[List[float], List[int], List[float]]:
    if len(values) == 0:
        return [min_edge, min_edge + bin_width], [0], [0.0]

    vmax = float(np.nanmax(values))
    vmax = max(vmax, min_edge)
    num_bins = int(np.ceil((vmax - min_edge) / bin_width)) + 1
    edges = [min_edge + i * bin_width for i in range(num_bins + 1)]

    counts, _ = np.histogram(values, bins=np.array(edges, dtype=float))
    total = int(counts.sum())
    rates = (counts / total).tolist() if total > 0 else [0.0 for _ in counts]
    return edges, counts.astype(int).tolist(), [float(r) for r in rates]


def _stats(values: pd.Series) -> Dict[str, float]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {"mean": float(v.mean()), "p50": float(v.quantile(0.5)), "p90": float(v.quantile(0.9))}


def _categorical_rates(values: pd.Series) -> CategoricalRates:
    s = values.fillna("UNKNOWN").astype(str)
    counts = s.value_counts(dropna=False).to_dict()
    total = float(sum(counts.values())) or 1.0
    rates = {k: float(v / total) for k, v in counts.items()}
    return CategoricalRates(counts={str(k): int(v) for k, v in counts.items()}, rates=rates)


def fit_target_distributions(enriched: pd.DataFrame, *, gpp_category: str) -> TargetDistributions:
    """
    Fit target distributions for a single segment bucket.
    """
    required = [
        "contest_id",
        "salary_left",
        "proj_gap_to_optimal",
        "stack_pattern",
        "cpt_archetype",
        "dup_count",
    ]
    missing = [c for c in required if c not in enriched.columns]
    if missing:
        raise ValueError(f"enriched missing required columns for distributions: {missing}")

    df = enriched.copy()
    source_contests = sorted(set(df["contest_id"].astype(str).tolist()))

    salary_left = pd.to_numeric(df["salary_left"], errors="coerce").dropna().to_numpy(dtype=float)
    gap = pd.to_numeric(df["proj_gap_to_optimal"], errors="coerce").dropna().to_numpy(dtype=float)
    dup = pd.to_numeric(df["dup_count"], errors="coerce").dropna().to_numpy(dtype=float)

    sal_edges, sal_counts, sal_rates = _histogram(salary_left, bin_width=500.0, min_edge=0.0)
    gap_edges, gap_counts, gap_rates = _histogram(gap, bin_width=1.0, min_edge=0.0)

    # dup_count is discrete; make integer bin edges [1..max+1]
    if len(dup) == 0:
        dup_edges, dup_counts, dup_rates = [1.0, 2.0], [0], [0.0]
    else:
        dmax = int(np.nanmax(dup))
        dup_edges = [float(i) for i in range(1, dmax + 2)]
        dup_counts_arr, _ = np.histogram(dup, bins=np.array(dup_edges, dtype=float))
        total = int(dup_counts_arr.sum()) or 1
        dup_counts = dup_counts_arr.astype(int).tolist()
        dup_rates = [float(c / total) for c in dup_counts]

    dist = TargetDistributions(
        generated_at=datetime.now(timezone.utc),
        gpp_category=gpp_category,
        source_contests=source_contests,
        salary_left=Histogram(
            bin_edges=sal_edges,
            counts=sal_counts,
            rates=sal_rates,
            stats=_stats(df["salary_left"]),
        ),
        proj_gap_to_optimal=Histogram(
            bin_edges=gap_edges,
            counts=gap_counts,
            rates=gap_rates,
            stats=_stats(df["proj_gap_to_optimal"]),
        ),
        stack_pattern=_categorical_rates(df["stack_pattern"]),
        cpt_archetype=_categorical_rates(df["cpt_archetype"]),
        dup_count=Histogram(bin_edges=dup_edges, counts=dup_counts, rates=dup_rates, stats=_stats(df["dup_count"])),
        validation={
            "n_rows": int(len(df)),
            "salary_left_rate_sum": float(sum(sal_rates)),
            "gap_rate_sum": float(sum(gap_rates)),
            "dup_rate_sum": float(sum(dup_rates)),
        },
    )
    return dist


