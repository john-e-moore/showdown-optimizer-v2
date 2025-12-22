from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from dfs_opt.config.settings import ContestConfig
from dfs_opt.pipelines.contest import run_contest_lineup_gen


def test_lineups_enriched_has_expected_columns_and_sane_values(tmp_path: Path) -> None:
    # Minimal Sabersim-like projections fixture using the repo's parser expectations.
    proj_path = tmp_path / "NBA_slate1.csv"
    pd.DataFrame(
        [
            {"Name": "A", "Team": "AAA", "Salary": 10000, "SS Proj": 20.0, "My Own": 10.0},
            {"Name": "B", "Team": "AAA", "Salary": 8000, "SS Proj": 16.0, "My Own": 12.0},
            {"Name": "C", "Team": "AAA", "Salary": 6000, "SS Proj": 12.0, "My Own": 8.0},
            {"Name": "D", "Team": "AAA", "Salary": 4000, "SS Proj": 8.0, "My Own": 6.0},
            {"Name": "E", "Team": "BBB", "Salary": 7000, "SS Proj": 14.0, "My Own": 9.0},
            {"Name": "F", "Team": "BBB", "Salary": 5000, "SS Proj": 10.0, "My Own": 7.0},
            {"Name": "G", "Team": "BBB", "Salary": 3000, "SS Proj": 6.0, "My Own": 5.0},
        ]
    ).to_csv(proj_path, index=False)

    # Correlation matrix fixture (identity).
    corr_path = tmp_path / "_corr_matrix.csv"
    names = ["A", "B", "C", "D", "E", "F", "G"]
    corr_df = {"Column1": names}
    for n in names:
        corr_df[n] = [1.0 if m == n else 0.0 for m in names]
    pd.DataFrame(corr_df).to_csv(corr_path, index=False)

    artifacts_root = tmp_path / "artifacts"
    cfg = ContestConfig(
        projection_csv=proj_path,
        corr_matrix_csv=corr_path,
        slate_id="slate1",
        sport="nba",
        artifacts_root=artifacts_root,
        persist_step_outputs=False,
        salary_cap=50000,
        min_proj_points=0.0,
    )
    res = run_contest_lineup_gen(cfg)
    run_dir = Path(res["artifacts_dir"])

    tbl = pq.read_table(run_dir / "lineups_enriched.parquet")
    cols = set(tbl.column_names)
    expected = {
        "cpt",
        "u1",
        "u2",
        "u3",
        "u4",
        "u5",
        "salary_used",
        "salary_left",
        "proj_points",
        "stack_code",
        "own_score_logprod",
        "own_max_log",
        "own_min_log",
        "avg_corr",
        "cpt_archetype",
        "salary_left_bin",
        "pct_proj_gap_to_optimal",
        "pct_proj_gap_to_optimal_bin",
    }
    assert expected.issubset(cols)

    df = tbl.to_pandas()
    assert len(df) > 0

    # Basic invariants / sanity checks
    assert (df["pct_proj_gap_to_optimal"].astype(float) >= 0.0).all()
    assert (df["own_min_log"].astype(float) <= df["own_max_log"].astype(float)).all()
    assert all(math.isfinite(float(x)) for x in df["avg_corr"].astype(float).tolist())

    salary_bins = {"0_200", "200_500", "500_1000", "1000_2000", "2000_4000", "4000_8000", "8000_plus"}
    gap_bins = {"0_0.01", "0.01_0.02", "0.02_0.04", "0.04_0.07", "0.07_0.15", "0.15_0.30", "0.30_plus"}
    assert set(map(str, df["salary_left_bin"].tolist())).issubset(salary_bins)
    assert set(map(str, df["pct_proj_gap_to_optimal_bin"].tolist())).issubset(gap_bins)


