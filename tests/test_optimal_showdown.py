from __future__ import annotations

import pandas as pd

from dfs_opt.features.optimal import compute_optimal_showdown_proj


def test_compute_optimal_showdown_proj_bruteforce() -> None:
    # Construct players where best CPT is 'a' due to high projection.
    # Keep salaries low so cap never binds.
    df = pd.DataFrame(
        [
            {"name_norm": "a", "salary": 10000, "proj_points": 20.0},
            {"name_norm": "b", "salary": 9000, "proj_points": 19.0},
            {"name_norm": "c", "salary": 8000, "proj_points": 18.0},
            {"name_norm": "d", "salary": 7000, "proj_points": 17.0},
            {"name_norm": "e", "salary": 6000, "proj_points": 16.0},
            {"name_norm": "f", "salary": 5000, "proj_points": 15.0},
            {"name_norm": "g", "salary": 4000, "proj_points": 14.0},
        ]
    )

    res = compute_optimal_showdown_proj(df)
    # optimal lineup: CPT a (30) + top 5 utils excluding a: b+c+d+e+f = 85 -> total 115
    assert res.optimal_proj_points == 115.0


