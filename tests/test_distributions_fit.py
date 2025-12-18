from __future__ import annotations

import pandas as pd

from dfs_opt.distributions.fit import fit_target_distributions


def test_fit_target_distributions_basic_validation_sums() -> None:
    df = pd.DataFrame(
        [
            {
                "contest_id": "c1",
                "salary_left": 5000,
                "proj_gap_to_optimal": 3.2,
                "stack_pattern": "4-2",
                "cpt_archetype": "stud_1_2",
                "dup_count": 1,
            },
            {
                "contest_id": "c1",
                "salary_left": 4500,
                "proj_gap_to_optimal": 7.9,
                "stack_pattern": "3-3",
                "cpt_archetype": "value_11_plus",
                "dup_count": 2,
            },
        ]
    )

    dist = fit_target_distributions(df, gpp_category="nba-showdown-mme-0-1k")
    assert dist.gpp_category == "nba-showdown-mme-0-1k"
    assert dist.validation["n_rows"] == 2
    assert abs(dist.validation["salary_left_rate_sum"] - 1.0) < 1e-9
    assert abs(dist.validation["gap_rate_sum"] - 1.0) < 1e-9
    assert abs(dist.validation["dup_rate_sum"] - 1.0) < 1e-9
    assert sum(dist.stack_pattern.counts.values()) == 2
    assert sum(dist.cpt_archetype.counts.values()) == 2


