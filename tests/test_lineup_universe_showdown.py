from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dfs_opt.config.settings import ContestConfig
from dfs_opt.lineup_pool.enumerate_universe_showdown import (
    enumerate_showdown_universe,
    prepare_player_arrays,
)
from dfs_opt.pipelines.contest import run_contest_lineup_gen


def _toy_players_df() -> pd.DataFrame:
    # 7 players, 2 teams; enough to form legal showdown lineups.
    return pd.DataFrame(
        [
            {"name_norm": "a", "team": "AAA", "salary": 10000, "proj_points": 20.0},
            {"name_norm": "b", "team": "AAA", "salary": 8000, "proj_points": 16.0},
            {"name_norm": "c", "team": "AAA", "salary": 6000, "proj_points": 12.0},
            {"name_norm": "d", "team": "AAA", "salary": 4000, "proj_points": 8.0},
            {"name_norm": "e", "team": "BBB", "salary": 7000, "proj_points": 14.0},
            {"name_norm": "f", "team": "BBB", "salary": 5000, "proj_points": 10.0},
            {"name_norm": "g", "team": "BBB", "salary": 3000, "proj_points": 6.0},
        ]
    )


def test_enumerate_showdown_universe_legality_and_determinism() -> None:
    players = _toy_players_df()
    players2, arrays, _ = prepare_player_arrays(players, min_proj_points=0.0, max_players=None)
    res1 = enumerate_showdown_universe(arrays, salary_cap=50000)

    # Repeat to ensure determinism (ordering and counts).
    players3, arrays2, _ = prepare_player_arrays(players, min_proj_points=0.0, max_players=None)
    res2 = enumerate_showdown_universe(arrays2, salary_cap=50000)
    assert players2.equals(players3)
    assert res1.num_lineups == res2.num_lineups
    assert (res1.cpt == res2.cpt).all()
    assert (res1.u1 == res2.u1).all()
    assert (res1.salary_used == res2.salary_used).all()

    # Salary cap.
    assert int(res1.salary_used.max()) <= 50000
    assert int(res1.salary_left.min()) >= 0

    # No duplicates within lineups.
    # (Use a small sample, since this is a unit test; the kernel already constructs distinct indices.)
    n = min(5000, res1.num_lineups)
    for i in range(n):
        slots = [int(res1.cpt[i]), int(res1.u1[i]), int(res1.u2[i]), int(res1.u3[i]), int(res1.u4[i]), int(res1.u5[i])]
        assert len(set(slots)) == 6

    # Stack legality: no 6-0; since there are only 2 teams, stack_code must be in {0,1,2}
    assert set(map(int, pd.unique(pd.Series(res1.stack_code[:n])))).issubset({0, 1, 2})


def test_prepare_player_arrays_requires_two_teams() -> None:
    df = pd.DataFrame(
        [
            {"name_norm": "a", "team": "AAA", "salary": 10000, "proj_points": 20.0},
            {"name_norm": "b", "team": "AAA", "salary": 8000, "proj_points": 16.0},
            {"name_norm": "c", "team": "AAA", "salary": 6000, "proj_points": 12.0},
            {"name_norm": "d", "team": "AAA", "salary": 4000, "proj_points": 8.0},
            {"name_norm": "e", "team": "AAA", "salary": 7000, "proj_points": 14.0},
            {"name_norm": "f", "team": "AAA", "salary": 5000, "proj_points": 10.0},
        ]
    )
    with pytest.raises(ValueError, match=r"exactly 2 teams"):
        prepare_player_arrays(df)


def test_contest_lineup_gen_writes_required_artifacts(tmp_path: Path) -> None:
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

    artifacts_root = tmp_path / "artifacts"
    cfg = ContestConfig(
        projection_csv=proj_path,
        slate_id="slate1",
        sport="nba",
        artifacts_root=artifacts_root,
        persist_step_outputs=False,
        salary_cap=50000,
        min_proj_points=0.0,
    )
    res = run_contest_lineup_gen(cfg)

    run_dir = Path(res["artifacts_dir"])
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "logs" / "run.log").exists()
    assert (run_dir / "players.parquet").exists()
    assert (run_dir / "lineups.parquet").exists()
    assert (run_dir / "metadata.json").exists()

    steps = run_dir / "steps"
    for step_dir_name in [
        "00_ingest",
        "01_parse_projections",
        "02_enumerate_lineup_universe",
    ]:
        step_dir = steps / step_dir_name
        assert step_dir.exists(), f"missing step folder {step_dir_name}"
        assert (step_dir / "step_manifest.json").exists()
        assert (step_dir / "preview.csv").exists()
        assert (step_dir / "schema.json").exists()


