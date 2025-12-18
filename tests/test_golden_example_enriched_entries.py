from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dfs_opt.distributions.fit import fit_target_distributions


def test_golden_enriched_entries_example_can_fit_distributions() -> None:
    path = Path("data/historical/enriched/enriched_entries_example.csv")
    if not path.exists():
        pytest.skip("golden example data not present (data/ is often local-only)")

    df = pd.read_csv(path)
    # minimal sanity: required fields exist in the provided example file
    required = ["contest_id", "salary_left", "proj_gap_to_optimal", "stack_pattern", "cpt_archetype", "dup_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        pytest.skip(f"golden example missing required columns for distribution fitting: {missing}")

    dist = fit_target_distributions(df, gpp_category="golden-example")
    assert dist.validation["n_rows"] == len(df)
    assert abs(dist.validation["salary_left_rate_sum"] - 1.0) < 1e-9
    assert abs(dist.validation["gap_rate_sum"] - 1.0) < 1e-9
    assert abs(dist.validation["dup_rate_sum"] - 1.0) < 1e-9


