from __future__ import annotations

from pathlib import Path

import pandas as pd

from dfs_opt.config.settings import TrainingConfig
from dfs_opt.pipelines.training import run_training_pipeline


def _write_minimal_showdown_fixture(data_root: Path) -> str:
    slate_dir = data_root / "dk-results" / "showdown" / "nba" / "slate1"
    slate_dir.mkdir(parents=True, exist_ok=True)

    # projections
    proj_path = slate_dir / "NBA_slate1.csv"
    pd.DataFrame(
        [
            {"Name": "A", "Team": "AAA", "Salary": 10000, "SS Proj": 20.0},
            {"Name": "B", "Team": "AAA", "Salary": 8000, "SS Proj": 16.0},
            {"Name": "C", "Team": "AAA", "Salary": 6000, "SS Proj": 12.0},
            {"Name": "D", "Team": "AAA", "Salary": 4000, "SS Proj": 8.0},
            {"Name": "E", "Team": "BBB", "Salary": 7000, "SS Proj": 14.0},
            {"Name": "F", "Team": "BBB", "Salary": 5000, "SS Proj": 10.0},
            {"Name": "G", "Team": "BBB", "Salary": 3000, "SS Proj": 6.0},
        ]
    ).to_csv(proj_path, index=False)

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

    # expected gpp category: nba-showdown-mme-0-1k (size=2, max entries=150)
    return "nba-showdown-mme-0-1k"


def test_training_pipeline_writes_required_artifacts(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    expected_cat = _write_minimal_showdown_fixture(data_root)

    artifacts_root = tmp_path / "artifacts"
    cfg = TrainingConfig(data_root=data_root, artifacts_root=artifacts_root, persist_step_outputs=True)
    res = run_training_pipeline(cfg)

    run_dir = Path(res["artifacts_dir"])
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "entries_enriched.parquet").exists()
    assert (run_dir / "target_distributions" / f"{expected_cat}.json").exists()

    # step folders + mandatory sidecars
    steps = run_dir / "steps"
    for step_dir_name in [
        "00_ingest",
        "01_parse_projections",
        "02_parse_contest_entries",
        "03_join_and_enrich",
        "04_optimal_and_gap",
        "05_fit_target_distributions",
    ]:
        step_dir = steps / step_dir_name
        assert step_dir.exists(), f"missing step folder {step_dir_name}"
        assert (step_dir / "step_manifest.json").exists()
        assert (step_dir / "preview.csv").exists()
        assert (step_dir / "schema.json").exists()


def test_training_pipeline_gpp_category_filter(tmp_path: Path) -> None:
    data_root = tmp_path / "data_root"
    expected_cat = _write_minimal_showdown_fixture(data_root)

    artifacts_root = tmp_path / "artifacts"
    cfg = TrainingConfig(
        data_root=data_root,
        artifacts_root=artifacts_root,
        persist_step_outputs=False,
        gpp_category=expected_cat,
    )
    res = run_training_pipeline(cfg)
    run_dir = Path(res["artifacts_dir"])
    assert (run_dir / "target_distributions" / f"{expected_cat}.json").exists()


