from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from dfs_opt.config.load import apply_contest_cli_overrides, load_contest_config
from dfs_opt.config.settings import ContestConfig
from dfs_opt.pipelines.contest import run_contest_lineup_gen

app = typer.Typer(add_completion=False, help="Pipeline B: contest execution (lineup-gen slice)")


@app.command("lineup-gen")
def lineup_gen(
    config: Optional[Path] = typer.Option(
        None, help="Optional YAML config file; CLI options override values from config"
    ),
    projection_csv: Path = typer.Option(..., help="Sabersim projections CSV (raw)"),
    corr_matrix_csv: Path = typer.Option(..., help="Sabersim correlation matrix CSV for the slate"),
    slate_id: str = typer.Option(..., help="Slate identifier (recorded in manifests)"),
    sport: str = typer.Option("nba", help="Sport label (recorded in manifests)"),
    artifacts_root: Path = typer.Option(Path("artifacts/lineup-gen"), help="Where to write run artifacts"),
    seed: int = typer.Option(1337, help="Global seed recorded in manifests"),
    persist_step_outputs: bool = typer.Option(
        False, help="If set, write step outputs as parquet alongside preview/schema/manifests"
    ),
    log_level: str = typer.Option("INFO", help="Python logging level (INFO, DEBUG, WARNING, ...)"),
    salary_cap: int = typer.Option(50000, help="Salary cap"),
    min_proj_points: float = typer.Option(0.0, help="Filter players by proj_points > min_proj_points"),
    max_players: Optional[int] = typer.Option(None, help="Debug: cap eligible players after filtering"),
) -> None:
    if config is None:
        cfg = ContestConfig(
            projection_csv=projection_csv,
            corr_matrix_csv=corr_matrix_csv,
            slate_id=slate_id,
            sport=sport,
            artifacts_root=artifacts_root,
            seed=seed,
            persist_step_outputs=persist_step_outputs,
            log_level=log_level,
            salary_cap=salary_cap,
            min_proj_points=min_proj_points,
            max_players=max_players,
        )
    else:
        cfg = load_contest_config(config)
        cfg = apply_contest_cli_overrides(
            cfg,
            projection_csv=projection_csv,
            corr_matrix_csv=corr_matrix_csv,
            slate_id=slate_id,
            sport=sport,
            artifacts_root=artifacts_root,
            seed=seed,
            persist_step_outputs=persist_step_outputs,
            log_level=log_level,
            salary_cap=salary_cap,
            min_proj_points=min_proj_points,
            max_players=max_players,
        )

    run_contest_lineup_gen(cfg)


