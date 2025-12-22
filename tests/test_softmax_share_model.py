from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from dfs_opt.config.settings import ContestConfig, TrainingConfig
from dfs_opt.pipelines.contest import run_contest_lineup_gen
from dfs_opt.pipelines.training import run_training_pipeline
from dfs_opt.share_model.softmax_share import dense_logZ_and_expX


def test_dense_logZ_gradient_matches_expected_X() -> None:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(25, 7)).astype(np.float64)
    theta = rng.normal(size=(7,)).astype(np.float64)

    logZ, expX = dense_logZ_and_expX(X, theta)
    assert np.isfinite(logZ)
    assert expX.shape == theta.shape

    eps = 1e-6
    fd = np.zeros_like(theta)
    for j in range(theta.shape[0]):
        e = np.zeros_like(theta)
        e[j] = 1.0
        lp, _ = dense_logZ_and_expX(X, theta + eps * e)
        lm, _ = dense_logZ_and_expX(X, theta - eps * e)
        fd[j] = (lp - lm) / (2 * eps)

    # gradient of logZ wrt theta is E_p[X]
    assert np.allclose(fd, expX, rtol=5e-5, atol=5e-5)


def _write_minimal_showdown_training_fixture(data_root: Path, *, sport: str = "nba") -> str:
    slate_dir = data_root / "dk-results" / "showdown" / sport / "slate1"
    slate_dir.mkdir(parents=True, exist_ok=True)

    # projections
    proj_path = slate_dir / f"{sport.upper()}_slate1.csv"
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

    # correlation matrix (identity)
    corr_path = slate_dir / f"{sport.upper()}_slate1_corr_matrix.csv"
    pd.DataFrame(
        {
            "Column1": ["A", "B", "C", "D", "E", "F", "G"],
            "A": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "B": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "C": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "D": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "E": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "F": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "G": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    ).to_csv(corr_path, index=False)

    contests_dir = slate_dir / "contests" / "contest-standings-111111"
    contests_dir.mkdir(parents=True, exist_ok=True)
    standings_path = contests_dir / "contest-standings-111111.csv"
    pd.DataFrame(
        [
            {
                "Rank": 1,
                "EntryId": 1,
                "EntryName": "user1 (1/150)",
                "Points": 100.0,
                "Lineup": "CPT A UTIL B UTIL C UTIL D UTIL E UTIL F",
            },
            {
                "Rank": 2,
                "EntryId": 2,
                "EntryName": "user2 (2/150)",
                "Points": 90.0,
                "Lineup": "CPT B UTIL A UTIL C UTIL D UTIL E UTIL F",
            },
        ]
    ).to_csv(standings_path, index=False)

    return f"{sport}-showdown-mme-0-1k"


def test_training_pipeline_share_model_writes_theta_and_metrics(tmp_path: Path) -> None:
    # Raw data for Pipeline A
    data_root = tmp_path / "data_root"
    expected_cat = _write_minimal_showdown_training_fixture(data_root, sport="nba")

    # Build a matching universe using the contest pipeline, then copy it into the expected universe_root layout.
    slate_dir = data_root / "dk-results" / "showdown" / "nba" / "slate1"
    proj_path = slate_dir / "NBA_slate1.csv"
    corr_path = slate_dir / "NBA_slate1_corr_matrix.csv"

    tmp_universe_artifacts = tmp_path / "tmp_universe_artifacts"
    res = run_contest_lineup_gen(
        ContestConfig(
            projection_csv=proj_path,
            corr_matrix_csv=corr_path,
            slate_id="slate1",
            sport="nba",
            artifacts_root=tmp_universe_artifacts,
            persist_step_outputs=False,
        )
    )
    run_dir = Path(res["artifacts_dir"])

    universe_root = tmp_path / "universe_root"
    target = universe_root / "dk-results" / "showdown" / "nba" / "slate1"
    target.mkdir(parents=True, exist_ok=True)
    for name in ["players.parquet", "lineups.parquet", "lineups_enriched.parquet", "metadata.json"]:
        shutil.copy(run_dir / name, target / name)

    artifacts_root = tmp_path / "artifacts"
    cfg = TrainingConfig(
        data_root=data_root,
        artifacts_root=artifacts_root,
        persist_step_outputs=True,
        gpp_category=expected_cat,
        universe_root=universe_root,
        share_model_enabled=True,
        share_model_lambda=1e-3,
        share_model_max_iter=20,
        share_model_val_slate_frac=0.0,
        share_model_seed=123,
    )
    out = run_training_pipeline(cfg)
    run_dir = Path(out["artifacts_dir"])

    theta_path = run_dir / "share_models" / expected_cat / "theta.json"
    metrics_path = run_dir / "share_models" / expected_cat / "fit_metrics.json"
    assert theta_path.exists()
    assert metrics_path.exists()

    # Step folder should exist when enabled
    assert (run_dir / "steps" / "06_fit_softmax_lineup_share").exists()


