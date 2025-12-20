from __future__ import annotations

from pathlib import Path

import pandas as pd

from dfs_opt.parsing.sabersim import parse_sabersim_showdown_csv


def test_parse_sabersim_keeps_min_salary_row(tmp_path: Path) -> None:
    # two rows for same name: CPT (higher salary) + FLEX (lower salary)
    p = tmp_path / "proj.csv"
    df = pd.DataFrame(
        [
            {"Name": "John Doe", "Team": "AAA", "Salary": 12000, "SS Proj": 30.0, "My Own": 10.0},
            {"Name": "John Doe", "Team": "AAA", "Salary": 8000, "SS Proj": 30.0, "My Own": 12.0},
            {"Name": "Jane Roe", "Team": "BBB", "Salary": 9000, "SS Proj": 28.0, "My Own": 5.0},
        ]
    )
    df.to_csv(p, index=False)

    out, metrics = parse_sabersim_showdown_csv(p)
    assert metrics["num_players"] == 2
    john = out[out["name_norm"] == "john doe"].iloc[0]
    assert int(john["salary"]) == 8000
    assert float(john["own"]) == 0.12


